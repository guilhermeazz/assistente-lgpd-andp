"""
Motor RAG Híbrido Otimizado com Fast-Path por Regex, Token Budget por Tier e FlashRank (src/rag/engine.py)
Implementa:
- Expansor Determinístico de Sinônimos Jurídicos (~0 ms): `expand_legal_query` (`src/rag/legal_expander.py`)
- Fast-Path por Regex: identificação instantânea O(1) de consultas diretas a dispositivos (ex: "Art. 18", "artigo 7º")
- Estágio 1 (Hybrid Recall): Top-10 Vetorial (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
  + Top-10 Lexical (`BM25Okapi`) via Reciprocal Rank Fusion (RRF) -> Top-8/10 candidatos
- Estágio 2 (Precision Reranking): Cross-Encoder local ONNX via `FlashRank (ms-marco-MultiBERT-L-12)`
- Orçamento Rígido de Contexto (Token Budget): respeita estritamente `max_context_tokens` do Tier do modelo
  (`recommended`: 2200 tokens / top_k=5; `economic`: 1200 tokens / top_k=3) e remove metadados ruidosos do payload XML
- Janela deslizante de histórico (`max_history_turns`) e suporte aos 3 níveis de linguagem (`Simples`, `Equilibrado`, `Técnico`)
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

# Evita contenção de threads OpenMP entre PyTorch (SentenceTransformers) e ONNXRuntime (FlashRank) em CPU
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import chromadb
from rank_bm25 import BM25Okapi

from src.rag.legal_expander import (
    expand_legal_query,
    extract_anchor_articles,
    extract_direct_article_references,
)
from src.rag.prompts import (
    SYSTEM_PROMPT,
    build_system_prompt,
    normalize_language_level,
)
from src.rag.providers import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_TEMPERATURE,
    PROVIDER_DEFAULT_MODELS,
    BaseLLMProvider,
    get_llm_provider,
    get_model_metadata,
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

DEFAULT_PROVIDER = "gemini"
DEFAULT_GENERATION_MODEL = PROVIDER_DEFAULT_MODELS[DEFAULT_PROVIDER]
DEFAULT_TOP_N_CANDIDATES = 8
DEFAULT_TOP_K = 4
DEFAULT_MAX_CROSS_REFS = 1

# Retrocompatibilidade
SYSTEM_INSTRUCTION = SYSTEM_PROMPT

# Singletons em memória para evitar recarregamento entre interações
_LOCAL_EMBEDDER_CACHE: Dict[str, Any] = {}
_LOCAL_RERANKER_CACHE: Dict[str, Any] = {}
_TEXT_EMBEDDING_CACHE: Dict[str, List[float]] = {}


def tokenize_pt(text: str) -> List[str]:
    """Tokenizador leve para português jurídico preservando números de artigos (ex: 55-j, 1º, art)."""
    cleaned = text.lower()
    return re.findall(r"[a-z0-9áàâãéêíóôõúçº°]+(?:-[a-z0-9]+)?", cleaned)


def estimate_tokens(text: str) -> int:
    """
    Estimativa rápida e calibrada de tokens para português jurídico (~3.8 caracteres por token
    com margem de segurança), sem custo de latência.
    """
    if not text:
        return 0
    return max(1, int(math.ceil(len(text) / 3.8)))


def expand_colloquial_legal_terms(query: str) -> str:
    """Wrapper retrocompatível para `expand_legal_query` retornando apenas a string expandida."""
    expanded, _ = expand_legal_query(query)
    return expanded


def trim_chat_history(
    chat_history: Optional[List[Dict[str, Any]]],
    max_history_turns: int = 3,
    max_turn_chars: int = 340,
) -> List[Dict[str, str]]:
    """
    Janela deslizante de histórico (Fase 5.4):
    Limita o histórico enviado à LLM a `max_history_turns` turnos (pares user/assistant)
    e trunca respostas anteriores longas apenas no payload do prompt (preservando a UI intacta).
    """
    if not chat_history or max_history_turns <= 0:
        return []

    max_messages = max_history_turns * 2
    recent = chat_history[-max_messages:]
    trimmed: List[Dict[str, str]] = []
    for msg in recent:
        role = str(msg.get("role", "user")).strip().lower()
        if role not in ("user", "assistant"):
            continue
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if len(content) > max_turn_chars:
            content = content[:max_turn_chars].rstrip() + "..."
        trimmed.append({"role": role, "content": content})
    return trimmed


def get_local_embedder(model_name: str = LOCAL_EMBEDDING_MODEL) -> Any:
    """Carrega e mantém em cache singleton o modelo SentenceTransformer local."""
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
    (`ms-marco-MultiBERT-L-12` em `./.cache/flashrank` com `max_length=32` e `intra_op_num_threads=2`).
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
    - Pode ser desempacotada como tupla: `metadata, stream = engine.ask_stream(...)`
    - Ou acessada como dicionário: `res = engine.ask_stream(...); res["stream"]`, `res["retrieved_chunks"]`.
    """

    def __iter__(self) -> Iterator[Any]:
        yield dict(self)
        yield self.get("stream")


class LGPDRagEngine:
    """
    Motor RAG Híbrido em Dois Estágios (Retrieve & Rerank) com Fast-Path por Regex e Token Budget:
    1. Fast-Path por Regex para consultas diretas a artigos (ex: "Art. 18", "artigo 7º")
    2. Expansão Determinística de Sinônimos Jurídicos (~0 ms, custo zero de tokens)
    3. Estágio 1: Top-10 ChromaDB Cosine + Top-10 BM25 -> Fusão RRF -> Top-8 candidatos
    4. Estágio 2: Cross-Encoder Local (`FlashRank ms-marco-MultiBERT-L-12`) -> `top_k` final
    5. Orçamento Rígido de Contexto (`max_context_tokens`) sem metadados ruidosos no XML.
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
        self.chunks_by_article: Dict[str, Dict[str, Any]] = {}
        self.bm25: Optional[BM25Okapi] = None
        self._using_fallback_embeddings: bool = False
        self.last_matched_concepts: List[str] = []
        self.last_expanded_query: str = ""
        self.last_fast_path_articles: List[str] = []
        self._load_local_chunks_and_bm25()

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = get_local_embedder(self.embedding_model_name)
        return self._embedder

    def _get_reranker(self) -> Optional[Any]:
        if not self._reranker_initialized:
            self._reranker = get_local_reranker(
                model_name=FLASHRANK_MODEL,
                cache_dir=FLASHRANK_CACHE_DIR,
            )
            self._reranker_initialized = True
        return self._reranker

    def _load_local_chunks_and_bm25(self) -> None:
        """Carrega os chunks do JSONL, cria índice O(1) por artigo e constrói o índice BM25."""
        self.chunks = []
        self.chunks_by_id = {}
        self.chunks_by_article = {}

        if self.chunks_path.exists():
            with self.chunks_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    self.chunks.append(item)
                    self.chunks_by_id[item["id"]] = item
                    art = str(item.get("metadata", {}).get("artigo", "")).strip()
                    if art:
                        self.chunks_by_article[art] = item

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
                self.chunks_by_article = {
                    str(item.get("metadata", {}).get("artigo", "")).strip(): item
                    for item in rebuilt
                    if str(item.get("metadata", {}).get("artigo", "")).strip()
                }
            except Exception as exc:
                logger.warning("Não foi possível reconstruir base automaticamente: %s", exc)

        if self.chunks:
            tokenized_corpus = [tokenize_pt(c["texto"]) for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(
                "Índice lexical BM25 inicializado com %d chunks (%d artigos indexados por chave direta).",
                len(self.chunks),
                len(self.chunks_by_article),
            )
        else:
            self.bm25 = None

    def get_indexed_count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    @staticmethod
    def _deterministic_test_embedding(text: str, dim: int = 384) -> List[float]:
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
                "Modelo SentenceTransformer indisponível (%s); ativando embedding local determinístico 384-d.",
                exc,
            )
            self._using_fallback_embeddings = True
            return [self._deterministic_test_embedding(t, dim=384) for t in texts]

    def _recreate_collection(self) -> None:
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
        """Reescrita condicional de query (apenas se houver histórico e pronomes referenciais)."""
        if not chat_history or not self._needs_query_rewrite(query):
            return query.strip()

        recent_turns = trim_chat_history(chat_history, max_history_turns=1, max_turn_chars=180)
        if not recent_turns:
            return query.strip()

        history_compact = " | ".join(
            f"{'U' if m['role'] == 'user' else 'A'}: {m['content']}"
            for m in recent_turns
        )

        short_system = (
            "Reescreva a pergunta do usuário tornando-a autossuficiente sobre LGPD/ANPD, "
            "substituindo pronomes pelo tema do histórico. Retorne APENAS a pergunta curta."
        )
        short_user = f"Histórico: {history_compact}\nPergunta: {query}\nReescrita:"

        try:
            active_provider = provider or get_llm_provider(
                provider_id=provider_name,
                model_id=model_name,
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
        direct_articles: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Estágio 2 — Reordenação de Precisão (Local Cross-Encoder via FlashRank):
        - Prioriza dispositivos citados diretamente via Fast-Path por Regex (`direct_articles`).
        - Reordena os candidatos via `FlashRank (ms-marco-MultiBERT-L-12)` + artigos-âncora.
        """
        if not candidates:
            return []

        direct_set = set(direct_articles or [])
        anchor_articles = extract_anchor_articles(expanded_query)
        anchor_set = set(anchor_articles) | direct_set

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

                    rrf_norm = min(1.0, item.get("rrf_score", 0.0) * 30.0)
                    combined = 0.65 * base_ce_score + 0.35 * rrf_norm

                    if artigo_meta and artigo_meta in direct_set:
                        combined = 0.9999
                        item["origem"] = "Fast-Path Regex (Dispositivo Direto) + FlashRank"
                    elif artigo_meta and artigo_meta in anchor_set:
                        anchor_rank_bonus = 0.05 if artigo_meta != "Art. 5º" else 0.02
                        combined = max(combined, 0.88 + 0.15 * rrf_norm) + anchor_rank_bonus
                        item["origem"] = "Busca Híbrida RRF + FlashRank Cross-Encoder"
                    else:
                        item["origem"] = "Busca Híbrida RRF + FlashRank Cross-Encoder"

                    item["raw_rerank_score"] = round(base_ce_score, 4)
                    item["rerank_score"] = round(min(0.9999, combined), 4)

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

        for idx, item in enumerate(candidates):
            artigo_meta = str(item["metadata"].get("artigo", "")).strip()
            fallback_score = min(0.95, item.get("rrf_score", 0.0) * 30.0)
            if artigo_meta and artigo_meta in direct_set:
                fallback_score = 0.9999
            elif artigo_meta and artigo_meta in anchor_set:
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
        Pipeline de Busca Otimizado (Fase 5.5):
        1. Fast-Path por Regex: identifica menções diretas a artigos (ex: "Art. 18", "artigo 7º") em O(1).
        2. Expansão determinística de sinônimos jurídicos (`expand_legal_query`, ~0 ms).
        3. Funil enxuto para CPU: Top-10 Vetorial + Top-10 BM25 -> Fusão RRF -> Top-8/10 no Reranker -> `top_k`.
        4. Expansão automática de até `max_cross_refs` (1) artigo complementar citado.
        """
        _ = api_key
        if not self.chunks or self.bm25 is None:
            self._load_local_chunks_and_bm25()

        indexed_count = self.get_indexed_count()
        if indexed_count == 0:
            self.index_chunks(use_mock_embeddings=use_mock_embeddings)
            indexed_count = self.get_indexed_count()

        # Etapa 0.A: Fast-Path por Regex para consultas diretas a dispositivos
        direct_articles = extract_direct_article_references(query)
        self.last_fast_path_articles = direct_articles

        # Etapa 0.B: Expansão determinística de sinônimos jurídicos (~0 ms)
        expanded_query, matched_concepts = expand_legal_query(query)
        if direct_articles:
            matched_concepts = [
                f"Fast-Path Direto por Regex → {', '.join(direct_articles)}"
            ] + matched_concepts
        self.last_expanded_query = expanded_query
        self.last_matched_concepts = matched_concepts
        anchor_articles = list(dict.fromkeys(direct_articles + extract_anchor_articles(expanded_query)))

        # Estágio 1.A: Top-10 Vetorial no ChromaDB (métrica cosine)
        query_emb = self._embed_texts(
            [expanded_query],
            use_mock_embeddings=use_mock_embeddings,
        )[0]

        effective_top_n = max(top_n, top_k)
        n_query = min(max(effective_top_n, 10), indexed_count)
        try:
            chroma_res = self.collection.query(
                query_embeddings=[query_emb],
                n_results=n_query,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.info(
                "Coleção ChromaDB com dimensão antiga detectada (%s); reindexando...",
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

        # Estágio 1.B: Top-10 Lexical com BM25Okapi
        bm25_ranks: Dict[str, int] = {}
        bm25_scores_map: Dict[str, float] = {}
        if self.bm25 is not None and self.chunks:
            query_tokens = tokenize_pt(expanded_query)
            raw_scores = self.bm25.get_scores(query_tokens)
            scored_indices = sorted(
                enumerate(raw_scores), key=lambda x: x[1], reverse=True
            )
            for rank_idx, (chunk_idx, score_val) in enumerate(
                scored_indices[:n_query], start=1
            ):
                c_id = self.chunks[chunk_idx]["id"]
                bm25_ranks[c_id] = rank_idx
                bm25_scores_map[c_id] = round(float(score_val), 4)

        # Garante inclusão O(1) de quaisquer artigos capturados pelo Fast-Path por Regex
        candidate_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        for d_art in anchor_articles:
            direct_chunk = self.chunks_by_article.get(d_art)
            if direct_chunk:
                candidate_ids.add(direct_chunk["id"])

        # Estágio 1.C: Fusão Reciprocal Rank Fusion (RRF) -> Top-N (8 a 10 candidatos)
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

            artigo_str = str(metadata.get("artigo", "")).strip()
            if artigo_str and artigo_str in direct_articles:
                rrf_score += 0.05
            elif artigo_str and artigo_str in anchor_articles:
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
                    "origem": (
                        "Fast-Path Regex (Dispositivo Exato)"
                        if artigo_str in direct_articles
                        else "Busca Híbrida (ChromaDB Cosine + BM25 RRF)"
                    ),
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

        # Estágio 2: Reordenação de Precisão com FlashRank Cross-Encoder
        top_chunks = self._rerank_candidates_stage2(
            original_query=query,
            expanded_query=expanded_query,
            candidates=candidates_top_n,
            top_k=top_k,
            use_mock_embeddings=use_mock_embeddings,
            direct_articles=direct_articles,
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

                    # Fast lookup em memória (0 ms) com fallback para ChromaDB metadata filter
                    direct_ref_chunk = self.chunks_by_article.get(cited_art)
                    if direct_ref_chunk and direct_ref_chunk["id"] not in present_ids:
                        exp_id = direct_ref_chunk["id"]
                        expanded_chunks.append(
                            {
                                "id": exp_id,
                                "texto": direct_ref_chunk["texto"],
                                "metadata": direct_ref_chunk["metadata"],
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
    def select_chunks_within_token_budget(
        retrieved_chunks: List[Dict[str, Any]],
        max_context_tokens: int = 2200,
    ) -> Tuple[List[Dict[str, Any]], str, int]:
        """
        Orçamento Rígido de Contexto (Fase 5.3):
        - Remove metadados ruidosos do payload XML enviado à LLM (caminhos de arquivo, hashes, scores internos),
          mantendo apenas o identificador limpo da fonte/artigo e o texto normativo.
        - Para de adicionar chunks assim que atingir `max_context_tokens` do Tier do modelo selecionado.
        """
        if max_context_tokens <= 0 or not retrieved_chunks:
            return [], "<contexto_normativo>\n</contexto_normativo>", 0

        wrapper_overhead = estimate_tokens("<contexto_normativo>\n</contexto_normativo>")
        used_tokens = wrapper_overhead
        selected_chunks: List[Dict[str, Any]] = []
        blocks: List[str] = ["<contexto_normativo>"]

        for item in retrieved_chunks:
            meta = item.get("metadata", {})
            fonte = str(meta.get("fonte", "LGPD/ANPD")).strip()
            artigo = str(meta.get("artigo", "")).strip()
            secao = str(meta.get("secao", "")).strip()
            clean_source = f"{fonte} | {artigo}" if artigo else f"{fonte} | {secao}"

            raw_text = str(item.get("texto", "")).strip()
            block_str = f'  <fonte ref="{clean_source}">\n{raw_text}\n  </fonte>'
            block_tokens = estimate_tokens(block_str)

            if used_tokens + block_tokens > max_context_tokens:
                # Se ainda não adicionou nenhum chunk, trunca o primeiro para caber estritamente no orçamento
                if not selected_chunks:
                    remaining_budget = max(30, max_context_tokens - used_tokens - 15)
                    max_chars = int(remaining_budget * 3.6)
                    truncated_text = raw_text[:max_chars].rstrip() + "..."
                    block_str = f'  <fonte ref="{clean_source}">\n{truncated_text}\n  </fonte>'
                    t_tokens = estimate_tokens(block_str)
                    if used_tokens + t_tokens <= max_context_tokens:
                        blocks.append(block_str)
                        used_tokens += t_tokens
                        selected_chunks.append(item)
                break

            blocks.append(block_str)
            used_tokens += block_tokens
            selected_chunks.append(item)

        blocks.append("</contexto_normativo>")
        xml_str = "\n".join(blocks)
        final_tokens = min(max_context_tokens, estimate_tokens(xml_str))
        return selected_chunks, xml_str, final_tokens

    @staticmethod
    def build_normative_context_xml(
        retrieved_chunks: List[Dict[str, Any]],
        max_context_tokens: int = 2200,
    ) -> str:
        """Formata os chunks respeitando o orçamento de tokens e sem metadados ruidosos."""
        _, xml_str, _ = LGPDRagEngine.select_chunks_within_token_budget(
            retrieved_chunks=retrieved_chunks,
            max_context_tokens=max_context_tokens,
        )
        return xml_str

    def ask_stream(
        self,
        question: str,
        history: Optional[List[Dict[str, Any]]] = None,
        top_k: Optional[int] = None,
        max_cross_refs: int = DEFAULT_MAX_CROSS_REFS,
        provider_name: str = DEFAULT_PROVIDER,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_OLLAMA_URL,
        provider: Optional[BaseLLMProvider] = None,
        language_level: str = "equilibrado",
        max_context_tokens: Optional[int] = None,
        max_history_turns: Optional[int] = None,
    ) -> StreamAnswerResult:
        """
        Streaming de Ponta a Ponta com Política de Tiers e Nível de Linguagem (Fases 3, 4, 5 e 6):
        1. Valida Tier do modelo (`recommended`, `economic`, `unsupported`).
        2. Aplica limites automáticos do Tier (`top_k`, `max_context_tokens`, `max_history_turns`).
        3. Recupera chunks via Fast-Path + RRF + FlashRank Reranker e aplica Token Budget rígido.
        4. Faz streaming real via `generate_stream(messages)`, cronometrando `retrieval_ms`, `ttft_ms` e `total_ms`.
        """
        active_provider = provider or get_llm_provider(
            provider_id=provider_name,
            model_id=model_name,
            api_key=api_key or self.api_key,
            base_url=base_url,
            temperature=DEFAULT_TEMPERATURE,
        )

        model_meta = get_model_metadata(
            active_provider.provider_key if active_provider.provider_key != "base" else provider_name,
            active_provider.model_name,
        )
        tier = str(model_meta.get("tier", "recommended"))
        if tier == "unsupported":
            raise ValueError(
                f"O modelo '{active_provider.model_name}' possui tier 'unsupported' (janela de contexto "
                "insuficiente para RAG jurídico). Selecione um modelo Recomendado ou Econômico."
            )

        effective_top_k = int(top_k if top_k is not None else model_meta.get("top_k", DEFAULT_TOP_K))
        effective_max_ctx_tokens = int(
            max_context_tokens
            if max_context_tokens is not None
            else model_meta.get("max_context_tokens", 2200)
        )
        effective_max_history = int(
            max_history_turns
            if max_history_turns is not None
            else model_meta.get("max_history_turns", 3)
        )
        norm_lang = normalize_language_level(language_level)
        system_prompt_str = build_system_prompt(norm_lang)

        trimmed_history = trim_chat_history(
            history,
            max_history_turns=effective_max_history,
        )

        t_retrieval_start = time.perf_counter()
        rewritten_query = self.rewrite_query_with_history(
            query=question,
            chat_history=trimmed_history,
            provider=active_provider,
        )

        raw_retrieved_chunks = self.hybrid_search_with_cross_ref(
            query=rewritten_query,
            top_k=effective_top_k,
            max_cross_refs=max_cross_refs,
        )

        # Aplica o orçamento rígido de contexto (Token Budget) do Tier
        budgeted_chunks, context_xml, context_tokens = self.select_chunks_within_token_budget(
            retrieved_chunks=raw_retrieved_chunks,
            max_context_tokens=effective_max_ctx_tokens,
        )
        retrieval_ms = round((time.perf_counter() - t_retrieval_start) * 1000, 1)

        user_prompt = (
            f"{context_xml}\n\n"
            f"Pergunta: {question}\n\n"
            "Responda seguindo estritamente o FORMATO VISUAL OBRIGATÓRIO (Pirâmide Invertida) e o nível de linguagem "
            "configurado, baseando-se exclusivamente em <contexto_normativo>."
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt_str}]
        messages.extend(trimmed_history)
        messages.append({"role": "user", "content": user_prompt})

        prompt_tokens_est = estimate_tokens(system_prompt_str) + estimate_tokens(user_prompt) + sum(
            estimate_tokens(m["content"]) for m in trimmed_history
        )
        hybrid_count = sum(1 for c in budgeted_chunks if not c.get("is_cross_reference"))
        cross_ref_count = sum(1 for c in budgeted_chunks if c.get("is_cross_reference"))

        result = StreamAnswerResult(
            {
                "question": question,
                "rewritten_query": self.last_expanded_query or rewritten_query,
                "matched_concepts": list(self.last_matched_concepts),
                "fast_path_articles": list(self.last_fast_path_articles),
                "answer": "",
                "provider": active_provider.provider_key,
                "model_requested": active_provider.model_name,
                "model_used": active_provider.model_name,
                "model_tier": tier,
                "language_level": norm_lang,
                "max_context_tokens": effective_max_ctx_tokens,
                "context_tokens": context_tokens,
                "estimated_tokens": prompt_tokens_est,
                "temperature": active_provider.temperature,
                "retrieval_ms": retrieval_ms,
                "ttft_ms": 0.0,
                "ttft_s": 0.0,
                "total_ms": retrieval_ms,
                "latency_ms": retrieval_ms,
                "hybrid_count": hybrid_count,
                "cross_ref_count": cross_ref_count,
                "system_instruction": system_prompt_str,
                "user_prompt": user_prompt,
                "retrieved_chunks": budgeted_chunks,
                "context_xml": context_xml,
            }
        )

        def _token_stream() -> Generator[str, None, None]:
            t_gen_start = time.perf_counter()
            first_token_received = False
            collected: List[str] = []
            for token_chunk in active_provider.generate_stream(messages):
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
            full_answer = "".join(collected).strip()
            result["answer"] = full_answer
            result["model_used"] = active_provider.model_name
            total_val = round(retrieval_ms + gen_ms, 1)
            result["total_ms"] = total_val
            result["latency_ms"] = total_val
            result["estimated_tokens"] = prompt_tokens_est + estimate_tokens(full_answer)

        result["stream"] = _token_stream()
        return result

    def answer_question(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        max_cross_refs: int = DEFAULT_MAX_CROSS_REFS,
        api_key: Optional[str] = None,
        provider_name: str = DEFAULT_PROVIDER,
        base_url: str = DEFAULT_OLLAMA_URL,
        provider: Optional[BaseLLMProvider] = None,
        language_level: str = "equilibrado",
    ) -> Dict[str, Any]:
        """Versão síncrona que consome `ask_stream` e retorna o dicionário completo."""
        res = self.ask_stream(
            question=question,
            history=chat_history,
            top_k=top_k,
            max_cross_refs=max_cross_refs,
            provider_name=provider_name,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            provider=provider,
            language_level=language_level,
        )
        for _ in res["stream"]:
            pass
        return dict(res)
