"""
Motor RAG Híbrido em Dois Estágios (Retrieve & Rerank) com Expansor Jurídico e Multi-Provedor (src/rag/engine.py)
Implementa:
- Expansor Determinístico de Sinônimos Jurídicos (~0 ms): `expand_legal_query` (`src/rag/legal_expander.py`)
- Estágio 1 (Hybrid Recall): Embeddings locais (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
  no ChromaDB (métrica `cosine`) + `BM25Okapi` via Reciprocal Rank Fusion (RRF) recuperando `top_n = 10` candidatos
- Estágio 2 (Precision Reranking): Cross-Encoder local ONNX via `flashrank.Ranker(model_name="ms-marco-MultiBERT-L-12", cache_dir="./.cache/flashrank")`
  selecionando os `top_k = 4` chunks mais relevantes + fallback transparente para RRF
- Expansão enxuta de referência cruzada: até `1` artigo complementar (`artigos_citados`)
- System Prompt interpretativo, visual e com One-Shot (`src/rag/prompts.py`)
- Streaming de ponta a ponta via `ask_stream(question, history, top_k=4)` compatível com `st.write_stream`
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

# Evita contenção de threads OpenMP entre PyTorch (SentenceTransformers) e ONNXRuntime (FlashRank) em CPU Windows
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import chromadb
from rank_bm25 import BM25Okapi

from src.rag.legal_expander import expand_legal_query, extract_anchor_articles
from src.rag.prompts import SYSTEM_PROMPT
from src.rag.providers import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_TEMPERATURE,
    PROVIDER_DEFAULT_MODELS,
    BaseLLMProvider,
    LLMProviderFactory,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks_lgpd.jsonl"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data" / "processed" / "chroma_db"
LOCAL_MODELS_CACHE_DIR = PROJECT_ROOT / "data" / "models_cache"
COLLECTION_NAME = "lgpd_anpd_normas"

LOCAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = LOCAL_EMBEDDING_MODEL
FLASHRANK_MODEL = "ms-marco-MultiBERT-L-12"
FLASHRANK_CACHE_DIR = "./.cache/flashrank"

DEFAULT_PROVIDER = "ollama"
DEFAULT_GENERATION_MODEL = PROVIDER_DEFAULT_MODELS[DEFAULT_PROVIDER]
DEFAULT_TOP_N_CANDIDATES = 10
DEFAULT_TOP_K = 4
DEFAULT_MAX_CROSS_REFS = 1

# Retrocompatibilidade com importações existentes
SYSTEM_INSTRUCTION = SYSTEM_PROMPT

# Singletons em memória para evitar recarregamento de modelos a cada consulta
_LOCAL_EMBEDDER_CACHE: Dict[str, Any] = {}
_LOCAL_RERANKER_CACHE: Dict[str, Any] = {}
_TEXT_EMBEDDING_CACHE: Dict[str, List[float]] = {}


def tokenize_pt(text: str) -> List[str]:
    """Tokenizador leve para português jurídico preservando números de artigos (ex: 55-j, 1º, art)."""
    cleaned = text.lower()
    return re.findall(r"[a-z0-9áàâãéêíóôõúçº°]+(?:-[a-z0-9]+)?", cleaned)


def expand_colloquial_legal_terms(query: str) -> str:
    """Wrapper retrocompatível para `expand_legal_query` retornando apenas a string expandida."""
    expanded, _ = expand_legal_query(query)
    return expanded


def get_local_embedder(model_name: str = LOCAL_EMBEDDING_MODEL) -> Any:
    """
    Carrega e mantém em cache singleton o modelo SentenceTransformer local
    (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
    """
    if model_name not in _LOCAL_EMBEDDER_CACHE:
        import torch
        from sentence_transformers import SentenceTransformer

        try:
            torch.set_num_threads(2)
        except Exception:
            pass

        LOCAL_MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_dir = str(LOCAL_MODELS_CACHE_DIR)
        logger.info("Carregando modelo de embedding local: %s", model_name)
        try:
            _LOCAL_EMBEDDER_CACHE[model_name] = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
                local_files_only=True,
            )
        except Exception:
            _LOCAL_EMBEDDER_CACHE[model_name] = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
            )
    return _LOCAL_EMBEDDER_CACHE[model_name]


def get_local_reranker(
    model_name: str = FLASHRANK_MODEL,
    cache_dir: str = FLASHRANK_CACHE_DIR,
    max_length: int = 32,
) -> Optional[Any]:
    """
    Carrega e mantém em cache singleton o Cross-Encoder ONNX do FlashRank
    (`ms-marco-MultiBERT-L-12` em `./.cache/flashrank` com `max_length=64` e `intra_op_num_threads=2`
    para eliminar contenção de threads OpenMP entre ONNXRuntime e PyTorch em CPU).
    """
    cache_key = f"{model_name}::{cache_dir}::{max_length}"
    if cache_key not in _LOCAL_RERANKER_CACHE:
        try:
            import onnxruntime as ort
            from flashrank import Ranker

            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            logger.info("Inicializando FlashRank Ranker local: %s (%s)", model_name, cache_dir)

            orig_session_cls = ort.InferenceSession

            def _tuned_inference_session(path_or_bytes: Any, *args: Any, **kwargs: Any) -> Any:
                if "sess_options" not in kwargs:
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 2
                    opts.inter_op_num_threads = 1
                    kwargs["sess_options"] = opts
                if "providers" not in kwargs:
                    kwargs["providers"] = ["CPUExecutionProvider"]
                return orig_session_cls(path_or_bytes, *args, **kwargs)

            ort.InferenceSession = _tuned_inference_session  # type: ignore[misc]
            try:
                ranker = Ranker(
                    model_name=model_name,
                    cache_dir=cache_dir,
                    max_length=max_length,
                )
            finally:
                ort.InferenceSession = orig_session_cls  # type: ignore[misc]

            _LOCAL_RERANKER_CACHE[cache_key] = ranker
        except Exception as exc:
            logger.warning(
                "FlashRank indisponível (%s); operando com fallback transparente para RRF.",
                exc,
            )
            return None
    return _LOCAL_RERANKER_CACHE.get(cache_key)


class StreamAnswerResult(dict):
    """
    Estrutura de retorno híbrida de `ask_stream`:
    - Pode ser desempacotada diretamente como tupla: `metadata, stream = engine.ask_stream(...)`
    - Ou acessada como dicionário: `res = engine.ask_stream(...); res["stream"]`, `res["retrieved_chunks"]`.
    """

    def __iter__(self) -> Iterator[Any]:
        yield dict(self)
        yield self.get("stream")


class LGPDRagEngine:
    """
    Motor RAG Híbrido em Dois Estágios (Retrieve & Rerank):
    1. Expansão Determinística de Sinônimos Jurídicos (~0 ms)
    2. Estágio 1: ChromaDB Cosine (SentenceTransformers Local) + BM25 via RRF (`top_n = 10`)
    3. Estágio 2: Cross-Encoder Local (`FlashRank ms-marco-MultiBERT-L-12`) selecionando `top_k = 4`
    4. Expansão de Referência Cruzada limitada a no máximo `1` artigo complementar (`artigos_citados`)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        chroma_dir: Optional[Path] = DEFAULT_CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        in_memory_chroma: bool = False,
        embedding_model_name: str = LOCAL_EMBEDDING_MODEL,
        embedder: Optional[Any] = None,
        reranker: Optional[Any] = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.chunks_path = Path(chunks_path)
        self.chroma_dir = Path(chroma_dir) if chroma_dir else None
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self._embedder = embedder
        self._reranker = reranker
        self._reranker_initialized = reranker is not None

        if in_memory_chroma or self.chroma_dir is None:
            self.chroma_client = chromadb.EphemeralClient()
        else:
            try:
                self.chroma_dir.mkdir(parents=True, exist_ok=True)
                self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
            except Exception as exc:
                logger.warning(
                    "PersistentClient indisponível (%s); utilizando EphemeralClient.",
                    exc,
                )
                self.chroma_client = chromadb.EphemeralClient()

        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.chunks: List[Dict[str, Any]] = []
        self.chunks_by_id: Dict[str, Dict[str, Any]] = {}
        self.bm25: Optional[BM25Okapi] = None
        self._using_fallback_embeddings: bool = False
        self.last_matched_concepts: List[str] = []
        self.last_expanded_query: str = ""
        self._load_local_chunks_and_bm25()

    def _get_embedder(self) -> Any:
        """Obtém a instância cacheada do SentenceTransformer local."""
        if self._embedder is None:
            self._embedder = get_local_embedder(self.embedding_model_name)
        return self._embedder

    def _get_reranker(self) -> Optional[Any]:
        """Obtém a instância cacheada do FlashRank Ranker local (ou None em fallback)."""
        if not self._reranker_initialized:
            self._reranker = get_local_reranker(
                model_name=FLASHRANK_MODEL,
                cache_dir=FLASHRANK_CACHE_DIR,
            )
            self._reranker_initialized = True
        return self._reranker

    def _load_local_chunks_and_bm25(self) -> None:
        """Carrega os chunks do arquivo JSONL e constrói o índice lexical BM25 em memória."""
        self.chunks = []
        self.chunks_by_id = {}

        if self.chunks_path.exists():
            with self.chunks_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    self.chunks.append(item)
                    self.chunks_by_id[item["id"]] = item

        if self.chunks_path == DEFAULT_CHUNKS_PATH and len(self.chunks) < 65:
            logger.warning(
                "Base de chunks ausente ou incompleta (%d chunks). Reconstruindo base normativa completa...",
                len(self.chunks),
            )
            try:
                from src.ingestion.fetcher import DEFAULT_RAW_DIR, run_ingestion_fetcher
                from src.ingestion.parser import run_parser_pipeline

                run_ingestion_fetcher(raw_dir=DEFAULT_RAW_DIR)
                rebuilt = run_parser_pipeline(
                    raw_dir=DEFAULT_RAW_DIR, output_jsonl=self.chunks_path
                )
                self.chunks = rebuilt
                self.chunks_by_id = {item["id"]: item for item in rebuilt}
            except Exception as exc:
                logger.warning("Não foi possível reconstruir base automaticamente: %s", exc)

        if self.chunks:
            tokenized_corpus = [tokenize_pt(c["texto"]) for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(
                "Índice lexical BM25 inicializado com %d chunks.", len(self.chunks)
            )
        else:
            self.bm25 = None

    def get_indexed_count(self) -> int:
        """Retorna a quantidade de chunks atualmente indexados no ChromaDB."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    @staticmethod
    def _deterministic_test_embedding(text: str, dim: int = 384) -> List[float]:
        """Gera vetor determinístico unitário (384-d) para testes rápidos ou fallback offline."""
        vec = [0.0] * dim
        tokens = tokenize_pt(text)
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _embed_texts(
        self,
        texts: List[str],
        use_mock_embeddings: bool = False,
    ) -> List[List[float]]:
        """
        Vetoriza textos localmente em milissegundos usando SentenceTransformers
        (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
        """
        if use_mock_embeddings or self._using_fallback_embeddings:
            return [self._deterministic_test_embedding(t, dim=384) for t in texts]

        try:
            missing = [t for t in texts if t not in _TEXT_EMBEDDING_CACHE]
            if missing:
                embedder = self._get_embedder()
                vectors = embedder.encode(
                    missing,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                for t_str, vec in zip(missing, vectors):
                    _TEXT_EMBEDDING_CACHE[t_str] = vec.tolist()
            return [_TEXT_EMBEDDING_CACHE[t] for t in texts]
        except Exception as exc:
            logger.warning(
                "Modelo SentenceTransformer indisponível no momento (%s); ativando embedding local determinístico 384-d.",
                exc,
            )
            self._using_fallback_embeddings = True
            return [self._deterministic_test_embedding(t, dim=384) for t in texts]

    def _recreate_collection(self) -> None:
        """Recria a coleção do ChromaDB com métrica cosine quando há incompatibilidade de dimensão."""
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(
        self,
        api_key: Optional[str] = None,
        use_mock_embeddings: bool = False,
    ) -> int:
        """
        Indexa todos os chunks de `data/processed/chunks_lgpd.jsonl` no ChromaDB (métrica cosine)
        usando o modelo local `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
        """
        _ = api_key
        self._load_local_chunks_and_bm25()
        if not self.chunks:
            raise RuntimeError(
                f"Nenhum chunk encontrado em {self.chunks_path}. Execute o parser primeiro."
            )

        ids = [c["id"] for c in self.chunks]
        documents = [c["texto"] for c in self.chunks]
        metadatas = [
            {
                "fonte": str(c["metadata"].get("fonte", "")),
                "tipo": str(c["metadata"].get("tipo", "")),
                "capitulo": str(c["metadata"].get("capitulo", "")),
                "secao": str(c["metadata"].get("secao", "")),
                "artigo": str(c["metadata"].get("artigo", "")),
                "artigos_citados": str(c["metadata"].get("artigos_citados", "")),
                "pagina": int(c["metadata"].get("pagina", 1)),
                "arquivo": str(c["metadata"].get("arquivo", "")),
                "breadcrumb": str(c["metadata"].get("breadcrumb", "")),
            }
            for c in self.chunks
        ]

        embeddings = self._embed_texts(
            documents,
            use_mock_embeddings=use_mock_embeddings,
        )

        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        except Exception as exc:
            logger.info(
                "Recriando coleção '%s' devido a mudança de dimensão de embedding (%s)...",
                self.collection_name,
                exc,
            )
            self._recreate_collection()
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

        total = self.collection.count()
        logger.info("Indexação local concluída no ChromaDB: %d chunks indexados.", total)
        return total

    @staticmethod
    def _needs_query_rewrite(query: str) -> bool:
        """
        Regra estrita de baixa latência para reescrita condicional:
        - NÃO reescreve quando a pergunta tiver 5 ou mais palavras e NÃO contiver pronomes
          referenciais ambíguos (`ele`, `ela`, `disso`, `desse`, `nisso`, `esse artigo`, `essa lei`).
        """
        q_clean = query.strip().lower()
        if not q_clean:
            return False

        ambiguous_pattern = (
            r"\b(?:ele|ela|eles|elas|disso|desse|dessa|nisso|nesse|nessa|nele|nela|isso)\b"
            r"|ess[ea]\s+artigo|est[ea]\s+artigo|ess[ea]\s+lei|est[ea]\s+lei"
            r"|ess[ea]\s+hipótese|ness[ea]\s+hipótese"
        )
        if re.search(ambiguous_pattern, q_clean):
            return True

        words = q_clean.split()
        if len(words) >= 5:
            return False

        return True

    def rewrite_query_with_history(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        provider: Optional[BaseLLMProvider] = None,
        provider_name: str = DEFAULT_PROVIDER,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_OLLAMA_URL,
    ) -> str:
        """
        Reescrita Condicional de Query:
        - Retorna imediatamente (0ms) quando `chat_history` estiver vazio OU quando a pergunta
          tiver >= 5 palavras sem pronomes referenciais ambíguos.
        - Quando realmente necessária, usa prompt curtíssimo com `max_tokens=40`.
        """
        if not chat_history or not self._needs_query_rewrite(query):
            return query.strip()

        recent_turns = chat_history[-2:]
        history_compact = " | ".join(
            f"{'U' if m.get('role') == 'user' else 'A'}: {str(m.get('content', ''))[:180]}"
            for m in recent_turns
        )

        short_system = (
            "Reescreva a pergunta do usuário tornando-a autossuficiente sobre LGPD/ANPD, "
            "substituindo pronomes pelo tema do histórico. Retorne APENAS a pergunta curta."
        )
        short_user = f"Histórico: {history_compact}\nPergunta: {query}\nReescrita:"

        try:
            active_provider = provider or LLMProviderFactory.create(
                provider_name=provider_name,
                model_name=model_name,
                api_key=api_key or self.api_key,
                base_url=base_url,
                temperature=DEFAULT_TEMPERATURE,
            )
            rewritten = active_provider.generate(
                system_prompt=short_system,
                user_prompt=short_user,
                max_tokens=40,
            ).strip()
            return rewritten if rewritten else query.strip()
        except Exception as exc:
            logger.warning("Reescrita condicional ignorada (mantendo query original): %s", exc)
            return query.strip()

    def _rerank_candidates_stage2(
        self,
        original_query: str,
        expanded_query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = DEFAULT_TOP_K,
        use_mock_embeddings: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Estágio 2 — Reordenação de Precisão (Local Cross-Encoder via FlashRank):
        - Reordena os `top_n = 10` candidatos comparando-os contra a pergunta original do usuário
          e aplicando boost de ancoragem normativa para artigos-âncora identificados no expansor jurídico.
        - Em caso de indisponibilidade do modelo ONNX (fallback seguro), preserva os `top_k = 4` do RRF.
        """
        if not candidates:
            return []

        anchor_articles = extract_anchor_articles(expanded_query)
        anchor_set = set(anchor_articles)

        # Em modo mock de testes unitários ultrarrápidos, evita invocação de ONNX se não carregado
        reranker = None if use_mock_embeddings else self._get_reranker()

        if reranker is not None:
            try:
                from flashrank import RerankRequest

                passages = [
                    {
                        "id": item["id"],
                        "text": item["texto"][:90],
                        "meta": item["metadata"],
                    }
                    for item in candidates
                ]
                rerank_req = RerankRequest(query=original_query[:65], passages=passages)
                reranked_raw = reranker.rerank(rerank_req)

                raw_score_by_id: Dict[str, float] = {
                    str(r["id"]): float(r.get("score", 0.0)) for r in reranked_raw
                }

                for item in candidates:
                    cid = item["id"]
                    base_ce_score = raw_score_by_id.get(cid, 0.0)
                    artigo_meta = str(item["metadata"].get("artigo", "")).strip()

                    # Combina score neural do Cross-Encoder com afinidade RRF e ancoragem jurídica determinística
                    rrf_norm = min(1.0, item.get("rrf_score", 0.0) * 30.0)
                    combined = 0.65 * base_ce_score + 0.35 * rrf_norm

                    if artigo_meta and artigo_meta in anchor_set:
                        # Artigos-âncora específicos (ex: Art. 11 para dado sensível) recebem prioridade determinística
                        anchor_rank_bonus = 0.05 if artigo_meta != "Art. 5º" else 0.02
                        combined = max(combined, 0.88 + 0.15 * rrf_norm) + anchor_rank_bonus

                    item["raw_rerank_score"] = round(base_ce_score, 4)
                    item["rerank_score"] = round(min(0.9999, combined), 4)
                    item["origem"] = "Busca Híbrida RRF + FlashRank Cross-Encoder"

                candidates.sort(
                    key=lambda x: (x["rerank_score"], x["rrf_score"], x["vector_score"]),
                    reverse=True,
                )
                return candidates[:top_k]
            except Exception as exc:
                logger.warning(
                    "Falha na execução do FlashRank (%s); aplicando fallback transparente para Top-%d do RRF.",
                    exc,
                    top_k,
                )

        # Fallback seguro: utiliza a pontuação do RRF normalizada + artigos-âncora
        for idx, item in enumerate(candidates):
            artigo_meta = str(item["metadata"].get("artigo", "")).strip()
            fallback_score = min(0.95, item.get("rrf_score", 0.0) * 30.0)
            if artigo_meta and artigo_meta in anchor_set:
                anchor_bonus = 0.04 if artigo_meta != "Art. 5º" else 0.02
                fallback_score = max(fallback_score, 0.90 - idx * 0.01) + anchor_bonus
            item["raw_rerank_score"] = round(fallback_score, 4)
            item["rerank_score"] = round(min(0.9999, fallback_score), 4)

        candidates.sort(
            key=lambda x: (x["rerank_score"], x["rrf_score"], x["vector_score"]),
            reverse=True,
        )
        return candidates[:top_k]

    def hybrid_search_with_cross_ref(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        max_cross_refs: int = DEFAULT_MAX_CROSS_REFS,
        top_n: int = DEFAULT_TOP_N_CANDIDATES,
        rrf_k: int = 60,
        api_key: Optional[str] = None,
        use_mock_embeddings: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Executa a Busca em Dois Estágios (Retrieve & Rerank) + Expansão de Referência Cruzada:
        1. Expansão Determinística de Sinônimos Jurídicos (`expand_legal_query`, ~0 ms)
        2. Estágio 1 (Hybrid Recall): ChromaDB Cosine + BM25 via RRF recuperando `top_n = 10` candidatos
        3. Estágio 2 (Precision Reranking): `FlashRank (ms-marco-MultiBERT-L-12)` selecionando os `top_k = 4`
        4. Expansão de Referência Cruzada: até `max_cross_refs = 1` artigo complementar (`artigos_citados`).
        """
        _ = api_key
        if not self.chunks or self.bm25 is None:
            self._load_local_chunks_and_bm25()

        indexed_count = self.get_indexed_count()
        if indexed_count == 0:
            self.index_chunks(use_mock_embeddings=use_mock_embeddings)
            indexed_count = self.get_indexed_count()

        # Etapa 0: Expansão determinística de sinônimos jurídicos (~0 ms)
        expanded_query, matched_concepts = expand_legal_query(query)
        self.last_expanded_query = expanded_query
        self.last_matched_concepts = matched_concepts
        anchor_articles = extract_anchor_articles(expanded_query)

        # Estágio 1.A: Vetorização local da query expandida no ChromaDB (métrica cosine)
        query_emb = self._embed_texts(
            [expanded_query],
            use_mock_embeddings=use_mock_embeddings,
        )[0]

        effective_top_n = max(top_n, top_k)
        n_query = min(max(effective_top_n * 2, 16), indexed_count)
        try:
            chroma_res = self.collection.query(
                query_embeddings=[query_emb],
                n_results=n_query,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.info(
                "Coleção ChromaDB com dimensão antiga detectada na busca (%s); reindexando...",
                exc,
            )
            self._recreate_collection()
            self.index_chunks(use_mock_embeddings=use_mock_embeddings)
            chroma_res = self.collection.query(
                query_embeddings=[query_emb],
                n_results=n_query,
                include=["documents", "metadatas", "distances"],
            )

        vector_ranks: Dict[str, int] = {}
        vector_scores: Dict[str, float] = {}
        chroma_docs: Dict[str, str] = {}
        chroma_metas: Dict[str, Dict[str, Any]] = {}

        if chroma_res and chroma_res.get("ids") and chroma_res["ids"][0]:
            for rank_idx, doc_id in enumerate(chroma_res["ids"][0], start=1):
                vector_ranks[doc_id] = rank_idx
                dist = float(chroma_res["distances"][0][rank_idx - 1])
                vector_scores[doc_id] = round(max(0.0, 1.0 - dist), 4)
                chroma_docs[doc_id] = chroma_res["documents"][0][rank_idx - 1]
                chroma_metas[doc_id] = chroma_res["metadatas"][0][rank_idx - 1]

        # Estágio 1.B: Ranking Lexical com BM25Okapi
        bm25_ranks: Dict[str, int] = {}
        bm25_scores_map: Dict[str, float] = {}
        if self.bm25 is not None and self.chunks:
            query_tokens = tokenize_pt(expanded_query)
            raw_scores = self.bm25.get_scores(query_tokens)
            scored_indices = sorted(
                enumerate(raw_scores), key=lambda x: x[1], reverse=True
            )
            for rank_idx, (chunk_idx, score_val) in enumerate(
                scored_indices[: n_query * 2], start=1
            ):
                c_id = self.chunks[chunk_idx]["id"]
                bm25_ranks[c_id] = rank_idx
                bm25_scores_map[c_id] = round(float(score_val), 4)

        # Estágio 1.C: Fusão Reciprocal Rank Fusion (RRF) -> Top-N (10 candidatos)
        candidate_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        rrf_results: List[Dict[str, Any]] = []

        for cid in candidate_ids:
            v_rank = vector_ranks.get(cid)
            b_rank = bm25_ranks.get(cid)
            rrf_score = 0.0
            if v_rank is not None:
                rrf_score += 1.0 / (rrf_k + v_rank)
            if b_rank is not None:
                rrf_score += 1.0 / (rrf_k + b_rank)

            chunk_obj = self.chunks_by_id.get(cid)
            texto = (
                chroma_docs.get(cid)
                or (chunk_obj["texto"] if chunk_obj else "")
            )
            metadata = (
                chroma_metas.get(cid)
                or (chunk_obj["metadata"] if chunk_obj else {})
            )

            # Artigos-âncora identificados pelo expansor jurídico recebem pequeno reforço no Recall
            artigo_str = str(metadata.get("artigo", "")).strip()
            if artigo_str and artigo_str in anchor_articles:
                rrf_score += 0.008

            rrf_results.append(
                {
                    "id": cid,
                    "texto": texto,
                    "metadata": metadata,
                    "rrf_score": round(rrf_score, 6),
                    "rerank_score": 0.0,
                    "vector_rank": v_rank,
                    "vector_score": vector_scores.get(cid, 0.0),
                    "bm25_rank": b_rank,
                    "bm25_score": bm25_scores_map.get(cid, 0.0),
                    "origem": "Busca Híbrida (ChromaDB Cosine + BM25 RRF)",
                    "is_cross_reference": False,
                    "cited_by": None,
                    "matched_concepts": matched_concepts,
                }
            )

        rrf_results.sort(
            key=lambda x: (x["rrf_score"], x["vector_score"], x["bm25_score"]),
            reverse=True,
        )
        candidates_top_n = rrf_results[:effective_top_n]

        # Estágio 2: Reordenação de Precisão com FlashRank Cross-Encoder (Top-10 -> Top-4)
        top_chunks = self._rerank_candidates_stage2(
            original_query=query,
            expanded_query=expanded_query,
            candidates=candidates_top_n,
            top_k=top_k,
            use_mock_embeddings=use_mock_embeddings,
        )

        # Etapa 3: Expansão Automática de Referência Cruzada (máx. 1 artigo complementar)
        present_articles = {
            str(item["metadata"].get("artigo", "")).strip()
            for item in top_chunks
            if str(item["metadata"].get("artigo", "")).strip()
        }
        present_ids = {item["id"] for item in top_chunks}

        expanded_chunks: List[Dict[str, Any]] = []
        if max_cross_refs > 0:
            for item in top_chunks:
                if len(expanded_chunks) >= max_cross_refs:
                    break
                source_artigo = str(item["metadata"].get("artigo", "")).strip() or item[
                    "metadata"
                ].get("fonte", "Documento")
                citados_str = str(item["metadata"].get("artigos_citados", "")).strip()
                if not citados_str:
                    continue

                cited_list = [
                    art.strip() for art in citados_str.split(",") if art.strip()
                ]
                for cited_art in cited_list:
                    if len(expanded_chunks) >= max_cross_refs:
                        break
                    if cited_art in present_articles:
                        continue

                    try:
                        found = self.collection.get(
                            where={"artigo": cited_art},
                            include=["documents", "metadatas"],
                        )
                        if found and found.get("ids") and len(found["ids"]) > 0:
                            exp_id = found["ids"][0]
                            if exp_id not in present_ids:
                                exp_doc = found["documents"][0]
                                exp_meta = found["metadatas"][0]
                                expanded_chunks.append(
                                    {
                                        "id": exp_id,
                                        "texto": exp_doc,
                                        "metadata": exp_meta,
                                        "rrf_score": 0.0,
                                        "rerank_score": round(
                                            max(0.50, item.get("rerank_score", 0.70) * 0.85),
                                            4,
                                        ),
                                        "vector_rank": None,
                                        "vector_score": 0.0,
                                        "bm25_rank": None,
                                        "bm25_score": 0.0,
                                        "origem": f"Referência Cruzada Automática (citado por {source_artigo})",
                                        "is_cross_reference": True,
                                        "cited_by": source_artigo,
                                        "matched_concepts": matched_concepts,
                                    }
                                )
                                present_articles.add(cited_art)
                                present_ids.add(exp_id)
                    except Exception as exc:
                        logger.warning(
                            "Erro ao expandir referência cruzada para %s: %s",
                            cited_art,
                            exc,
                        )

        return top_chunks + expanded_chunks

    @staticmethod
    def build_normative_context_xml(retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formata os chunks recuperados e reordenados dentro da tag <contexto_normativo>."""
        blocks: List[str] = ["<contexto_normativo>"]
        for idx, item in enumerate(retrieved_chunks, start=1):
            meta = item.get("metadata", {})
            origem = item.get("origem", "Busca Híbrida")
            breadcrumb = meta.get("breadcrumb", "")
            artigos_citados = meta.get("artigos_citados", "")
            rerank_score = item.get("rerank_score", 0.0)
            blocks.append(
                f'  <documento indice="{idx}" id="{item["id"]}" '
                f'score="{rerank_score:.2f}" origem="{origem}" breadcrumb="{breadcrumb}" '
                f'artigos_citados="{artigos_citados}">\n'
                f"{item['texto']}\n"
                f"  </documento>"
            )
        blocks.append("</contexto_normativo>")
        return "\n".join(blocks)

    def ask_stream(
        self,
        question: str,
        history: Optional[List[Dict[str, Any]]] = None,
        top_k: int = DEFAULT_TOP_K,
        max_cross_refs: int = DEFAULT_MAX_CROSS_REFS,
        provider_name: str = DEFAULT_PROVIDER,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_OLLAMA_URL,
        provider: Optional[BaseLLMProvider] = None,
    ) -> StreamAnswerResult:
        """
        Streaming de Ponta a Ponta (Retrieve & Rerank + Geração Visual Interpretativa):
        1. Executa reescrita condicional (apenas se estritamente necessário, com max_tokens=40)
        2. Expande sinônimos jurídicos (~0 ms), recupera Top-10 via RRF e reordena Top-4 com FlashRank
           + até 1 referência cruzada
        3. Retorna os metadados/fontes recuperadas e o gerador (`Generator[str, None, None]`)
           para renderização imediata no Streamlit via `st.write_stream`, medindo TTFT.
        """
        active_provider = provider or LLMProviderFactory.create(
            provider_name=provider_name,
            model_name=model_name,
            api_key=api_key or self.api_key,
            base_url=base_url,
            temperature=DEFAULT_TEMPERATURE,
        )

        t_retrieval_start = time.perf_counter()
        rewritten_query = self.rewrite_query_with_history(
            query=question,
            chat_history=history,
            provider=active_provider,
        )

        retrieved_chunks = self.hybrid_search_with_cross_ref(
            query=rewritten_query,
            top_k=top_k,
            max_cross_refs=max_cross_refs,
        )
        retrieval_ms = round((time.perf_counter() - t_retrieval_start) * 1000, 1)

        context_xml = self.build_normative_context_xml(retrieved_chunks)
        user_prompt = (
            f"{context_xml}\n\n"
            f"Pergunta do Usuário: {question}\n\n"
            "Siga rigorosamente o FORMATO VISUAL OBRIGATÓRIO DA RESPOSTA definido nas instruções do sistema "
            "(### 🎯 Resposta Direta, ### ⚖️ Fundamentação na LGPD / ANPD, ### 📊 Comparativo / Cenários se aplicável, "
            "### 🛠️ O que fazer na prática e > **⚠️ Ponto de Atenção:**), "
            "baseando-se exclusivamente no <contexto_normativo> acima."
        )

        hybrid_count = sum(1 for c in retrieved_chunks if not c.get("is_cross_reference"))
        cross_ref_count = sum(1 for c in retrieved_chunks if c.get("is_cross_reference"))

        result = StreamAnswerResult(
            {
                "question": question,
                "rewritten_query": self.last_expanded_query or rewritten_query,
                "matched_concepts": list(self.last_matched_concepts),
                "answer": "",
                "provider": active_provider.provider_key,
                "model_requested": active_provider.model_name,
                "model_used": active_provider.model_name,
                "temperature": active_provider.temperature,
                "retrieval_ms": retrieval_ms,
                "ttft_ms": 0.0,
                "ttft_s": 0.0,
                "latency_ms": retrieval_ms,
                "hybrid_count": hybrid_count,
                "cross_ref_count": cross_ref_count,
                "system_instruction": SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "retrieved_chunks": retrieved_chunks,
                "context_xml": context_xml,
            }
        )

        def _token_stream() -> Generator[str, None, None]:
            t_gen_start = time.perf_counter()
            first_token_received = False
            collected: List[str] = []
            for token_chunk in active_provider.stream_generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            ):
                if not first_token_received and token_chunk:
                    first_token_received = True
                    ttft_val = round((time.perf_counter() - t_gen_start) * 1000, 1)
                    result["ttft_ms"] = ttft_val
                    result["ttft_s"] = round(ttft_val / 1000.0, 2)
                collected.append(token_chunk)
                yield token_chunk

            gen_ms = round((time.perf_counter() - t_gen_start) * 1000, 1)
            if not first_token_received:
                result["ttft_ms"] = gen_ms
                result["ttft_s"] = round(gen_ms / 1000.0, 2)
            result["answer"] = "".join(collected).strip()
            result["model_used"] = active_provider.model_name
            result["latency_ms"] = round(retrieval_ms + gen_ms, 1)

        result["stream"] = _token_stream()
        return result

    def answer_question(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        model_name: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        max_cross_refs: int = DEFAULT_MAX_CROSS_REFS,
        api_key: Optional[str] = None,
        provider_name: str = DEFAULT_PROVIDER,
        base_url: str = DEFAULT_OLLAMA_URL,
        provider: Optional[BaseLLMProvider] = None,
    ) -> Dict[str, Any]:
        """Versão síncrona que consome `ask_stream` e retorna o dicionário completo com a resposta final."""
        effective_provider = provider_name
        if api_key and provider_name == DEFAULT_PROVIDER and model_name and "gemini" in model_name.lower():
            effective_provider = "gemini"

        res = self.ask_stream(
            question=question,
            history=chat_history,
            top_k=top_k,
            max_cross_refs=max_cross_refs,
            provider_name=effective_provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            provider=provider,
        )
        for _ in res["stream"]:
            pass
        return dict(res)
