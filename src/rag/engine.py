"""
Motor RAG Híbrido com Expansão Automática de Referências Cruzadas (src/rag/engine.py)
Implementa:
- Uso exclusivo do SDK oficial `google-genai` (`from google import genai` e `from google.genai import types`)
- Autenticação BYOK dinâmica (`genai.Client(api_key=user_key)`)
- Embeddings com `text-embedding-004` (`RETRIEVAL_DOCUMENT` na indexação e `RETRIEVAL_QUERY` na busca)
- Indexação vetorial no ChromaDB com métrica `cosine`
- Busca lexical com `rank_bm25` (`BM25Okapi`)
- Fusão de rankings via Reciprocal Rank Fusion (RRF)
- Expansão automática de até 2 artigos citados (`artigos_citados`) via filtro de metadado no ChromaDB
- Reescrita de query contextualizada com histórico do chat
- Geração estruturada com `temperature=0.1` e System Instruction restritivo ao `<contexto_normativo>`
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks_lgpd.jsonl"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data" / "processed" / "chroma_db"
COLLECTION_NAME = "lgpd_anpd_normas"

EMBEDDING_MODEL = "text-embedding-004"
DEFAULT_GENERATION_MODEL = "gemini-3.8-flash"
SUPPORTED_GENERATION_MODELS = ["gemini-3.8-flash"]

SYSTEM_INSTRUCTION = """Você é o Assistente Inteligente de Conformidade LGPD & ANPD, um especialista jurídico-regulatório em Proteção de Dados Pessoais no Brasil.

REGRAS ESTRITAS DE GROUNDING:
1. Responda EXCLUSIVAMENTE com base nas informações contidas dentro da tag <contexto_normativo> fornecida.
2. NÃO invente artigos, incisos, parágrafos, prazos, alíquotas ou interpretações que não estejam amparados no <contexto_normativo>.
3. Se a informação solicitada não constar no <contexto_normativo>, declare expressamente que não há base suficiente nos trechos recuperados.
4. Formate OBRIGATORIAMENTE toda resposta nas três seções numeradas abaixo:

### 1. Resumo Direto
Síntese clara, objetiva e executiva respondendo diretamente à dúvida do usuário.

### 2. Fundamentação Normativa Exata
Citação precisa dos dispositivos legais e regulatórios presentes no <contexto_normativo>, indicando expressamente o Artigo, Inciso, Alínea, Parágrafo da Lei nº 13.709/2018 (LGPD) e/ou o Nome e a Página da Resolução/Guia da ANPD, incluindo as referências cruzadas aplicáveis.

### 3. Implicação Prática / Checklist de Conformidade
Lista acionável (checklist em bullet points) com as medidas práticas de governança, segurança ou adequação que o Controlador, Operador ou Encarregado (DPO) deve adotar."""


def tokenize_pt(text: str) -> List[str]:
    """Tokenizador leve para português jurídico preservando números de artigos (ex: 55-j, 1º, art)."""
    cleaned = text.lower()
    return re.findall(r"[a-z0-9áàâãéêíóôõúçº°]+(?:-[a-z0-9]+)?", cleaned)


class LGPDRagEngine:
    """
    Motor RAG Híbrido (ChromaDB Cosine + BM25 via RRF) com Expansão Automática de Referências Cruzadas
    e Geração Controlada via SDK oficial `google-genai`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        chroma_dir: Optional[Path] = DEFAULT_CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        in_memory_chroma: bool = False,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.chunks_path = Path(chunks_path)
        self.chroma_dir = Path(chroma_dir) if chroma_dir else None
        self.collection_name = collection_name

        # Inicializa ChromaDB com métrica cosine obrigatória
        if in_memory_chroma or self.chroma_dir is None:
            self.chroma_client = chromadb.EphemeralClient()
        else:
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))

        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.chunks: List[Dict[str, Any]] = []
        self.chunks_by_id: Dict[str, Dict[str, Any]] = {}
        self.bm25: Optional[BM25Okapi] = None
        self._load_local_chunks_and_bm25()

    def _get_genai_client(self, api_key: Optional[str] = None) -> genai.Client:
        """Instancia dinamicamente o cliente oficial google.genai.Client(api_key=user_key)."""
        key = (api_key or self.api_key or "").strip()
        if not key:
            raise ValueError(
                "Gemini API Key não fornecida. Insira sua chave na barra lateral para autenticar a sessão (BYOK)."
            )
        return genai.Client(api_key=key)

    def _load_local_chunks_and_bm25(self) -> None:
        """Carrega os chunks do arquivo JSONL e constrói o índice lexical BM25 em memória."""
        self.chunks = []
        self.chunks_by_id = {}
        if not self.chunks_path.exists():
            logger.info("Arquivo de chunks ainda não existe em %s", self.chunks_path)
            self.bm25 = None
            return

        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.chunks.append(item)
                self.chunks_by_id[item["id"]] = item

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

    def _embed_texts(
        self,
        texts: List[str],
        task_type: str,
        api_key: Optional[str] = None,
        batch_size: int = 50,
    ) -> List[List[float]]:
        """
        Gera embeddings usando o modelo `text-embedding-004` via SDK `google-genai`,
        passando explicitamente `config=types.EmbedContentConfig(task_type=task_type)`.
        """
        client = self._get_genai_client(api_key=api_key)
        all_embeddings: List[List[float]] = []

        embed_config = types.EmbedContentConfig(task_type=task_type)

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = None
            last_exc: Optional[Exception] = None
            for candidate_emb_model in (EMBEDDING_MODEL, "gemini-embedding-001"):
                for attempt in range(3):
                    try:
                        response = client.models.embed_content(
                            model=candidate_emb_model,
                            contents=batch,
                            config=embed_config,
                        )
                        break
                    except Exception as exc:
                        last_exc = exc
                        err_lower = str(exc).lower()
                        if "not found" in err_lower or "404" in err_lower:
                            logger.warning(
                                "Modelo %s indisponível no endpoint da chave; tentando próximo modelo de embedding.",
                                candidate_emb_model,
                            )
                            break
                        if any(
                            transient in err_lower
                            for transient in ("503", "unavailable", "429", "high demand", "overloaded")
                        ) and attempt < 2:
                            wait_s = 1.5 * (attempt + 1)
                            logger.warning(
                                "Pico de demanda (503/429) em %s. Nova tentativa em %.1fs...",
                                candidate_emb_model,
                                wait_s,
                            )
                            time.sleep(wait_s)
                            continue
                        raise
                if response is not None:
                    break

            if response is None and last_exc is not None:
                raise last_exc

            for emb in response.embeddings:
                all_embeddings.append(list(emb.values))

        return all_embeddings

    @staticmethod
    def _deterministic_test_embedding(text: str, dim: int = 128) -> List[float]:
        """
        Gera vetor determinístico unitário baseado em termos apenas para testes locais offline
        quando explicitamente invocado com use_mock_embeddings=True.
        """
        vec = [0.0] * dim
        tokens = tokenize_pt(text)
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def index_chunks(
        self,
        api_key: Optional[str] = None,
        use_mock_embeddings: bool = False,
    ) -> int:
        """
        Indexa todos os chunks de `data/processed/chunks_lgpd.jsonl` no ChromaDB (métrica cosine)
        utilizando `text-embedding-004` com `task_type="RETRIEVAL_DOCUMENT"`.
        """
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

        if use_mock_embeddings:
            embeddings = [self._deterministic_test_embedding(doc) for doc in documents]
        else:
            embeddings = self._embed_texts(
                documents,
                task_type="RETRIEVAL_DOCUMENT",
                api_key=api_key,
            )

        # Recria a coleção caso a dimensionalidade do embedding mude
        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        except Exception:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

        total = self.collection.count()
        logger.info("Indexação concluída no ChromaDB: %d chunks indexados.", total)
        return total

    def _discover_available_models(self, client: genai.Client) -> List[str]:
        """Consulta dinamicamente a API para listar modelos Gemini ativos na chave do usuário."""
        discovered: List[str] = []
        try:
            for m in client.models.list():
                name = str(getattr(m, "name", "") or "")
                if name.startswith("models/"):
                    name = name.split("models/", 1)[1]
                lower = name.lower()
                if (
                    "gemini" in lower
                    and ("flash" in lower or "pro" in lower)
                    and not any(
                        skip in lower
                        for skip in ("embedding", "vision", "image", "tts", "audio", "live", "1.5", "2.0", "2.5-pro")
                    )
                ):
                    discovered.append(name)
        except Exception as exc:
            logger.debug("Não foi possível listar modelos dinamicamente: %s", exc)
        return discovered

    def _generate_content_resilient(
        self,
        client: genai.Client,
        preferred_model: str,
        contents: str,
        config: types.GenerateContentConfig,
    ) -> Tuple[Any, str]:
        """
        Executa client.models.generate_content priorizando a geração Gemini 3.x / Flash
        com retry exponencial e failover automático contra erros 404 e 503 UNAVAILABLE / 429.
        """
        fallback_chain = [
            preferred_model,
            "gemini-3-flash-preview",
            "gemini-3.1-flash-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-3.1-pro-preview",
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
        ]
        candidates: List[str] = []
        for m in fallback_chain:
            if m and m not in candidates:
                candidates.append(m)

        last_transient_exc: Optional[Exception] = None
        last_any_exc: Optional[Exception] = None

        for pass_idx in range(2):
            if pass_idx == 1:
                # Na segunda passagem, adiciona modelos descobertos dinamicamente na conta do usuário
                for dm in self._discover_available_models(client):
                    if dm not in candidates:
                        candidates.append(dm)

            for candidate_model in candidates:
                for attempt in range(3):
                    try:
                        resp = client.models.generate_content(
                            model=candidate_model,
                            contents=contents,
                            config=config,
                        )
                        return resp, candidate_model
                    except Exception as exc:
                        last_any_exc = exc
                        err_lower = str(exc).lower()

                        # Modelo inexistente ou descontinuado (404): avança imediatamente sem sobrescrever erro 503
                        if (
                            "not found" in err_lower
                            or "404" in err_lower
                            or "no longer available" in err_lower
                        ):
                            logger.info(
                                "Modelo %s indisponível no endpoint (404); tentando próximo da família Gemini 3/Flash.",
                                candidate_model,
                            )
                            break

                        # Pico de demanda temporária (503 UNAVAILABLE / 429)
                        if any(
                            code in err_lower
                            for code in (
                                "503",
                                "unavailable",
                                "high demand",
                                "overloaded",
                                "429",
                                "resource_exhausted",
                                "500",
                                "502",
                                "504",
                            )
                        ):
                            last_transient_exc = exc
                            if attempt < 2:
                                wait_s = 1.5 * (attempt + 1)
                                logger.warning(
                                    "Modelo %s com alta demanda (503/429). Tentativa %d/3 em %.1fs...",
                                    candidate_model,
                                    attempt + 2,
                                    wait_s,
                                )
                                time.sleep(wait_s)
                                continue
                            logger.warning(
                                "Modelo %s permaneceu ocupado; realizando failover para próximo modelo disponível.",
                                candidate_model,
                            )
                            break

                        # Erros de autenticação/permissão (400/401/403) são propagados imediatamente
                        raise

        if last_transient_exc is not None:
            raise last_transient_exc
        if last_any_exc is not None:
            raise last_any_exc
        raise RuntimeError("Não foi possível obter resposta dos modelos Gemini.")

    @staticmethod
    def _needs_query_rewrite(query: str) -> bool:
        """Verifica se a pergunta depende do contexto anterior (pronomes ou pergunta curta ambígua)."""
        q_clean = query.strip().lower()
        words = q_clean.split()
        if len(words) <= 5:
            return True
        ambiguous_markers = (
            " isso",
            " disso",
            " nele",
            " nela",
            " nesse caso",
            " nesta hipótese",
            " nessa hipótese",
            " e qual a multa",
            " e se ",
            " sobre ele",
            " sobre ela",
            " esse artigo",
            " este artigo",
        )
        return any(marker in f" {q_clean} " for marker in ambiguous_markers)

    def rewrite_query_with_history(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        model_name: str = DEFAULT_GENERATION_MODEL,
        api_key: Optional[str] = None,
    ) -> str:
        """
        Se houver histórico no chat e a pergunta for ambígua/referencial, reescreve-a
        em uma query autossuficiente antes da busca híbrida.
        """
        if not chat_history or not self._needs_query_rewrite(query):
            return query.strip()

        recent_turns = chat_history[-4:]
        history_lines: List[str] = []
        for msg in recent_turns:
            role = "Usuário" if msg.get("role") == "user" else "Assistente"
            content = str(msg.get("content", ""))[:400]
            history_lines.append(f"{role}: {content}")

        history_str = "\n".join(history_lines)
        rewrite_prompt = (
            "Dado o histórico recente de conversa sobre a LGPD (Lei 13.709/2018) e normas da ANPD, "
            "reescreva a 'Pergunta Atual' em uma única consulta de busca autossuficiente, clara e explícita, "
            "substituindo pronomes como 'isso', 'ele', 'essa hipótese' pelos termos jurídicos/técnicos do histórico. "
            "Retorne APENAS a consulta reescrita, sem explicações.\n\n"
            f"Histórico:\n{history_str}\n\n"
            f"Pergunta Atual: {query}\n\n"
            "Consulta Autossuficiente:"
        )

        try:
            client = self._get_genai_client(api_key=api_key)
            resp, _ = self._generate_content_resilient(
                client=client,
                preferred_model=model_name or DEFAULT_GENERATION_MODEL,
                contents=rewrite_prompt,
                config=types.GenerateContentConfig(temperature=0.1),
            )
            rewritten = (resp.text or "").strip()
            return rewritten if rewritten else query.strip()
        except Exception as exc:
            logger.warning("Falha na reescrita de query, mantendo original: %s", exc)
            return query.strip()

    def hybrid_search_with_cross_ref(
        self,
        query: str,
        top_k: int = 5,
        max_cross_refs: int = 2,
        rrf_k: int = 60,
        api_key: Optional[str] = None,
        use_mock_embeddings: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Executa a Busca Híbrida (Reciprocal Rank Fusion - RRF) combinando:
        1. Busca Vetorial no ChromaDB (`RETRIEVAL_QUERY` com métrica `cosine`)
        2. Busca Lexical via `rank_bm25` (`BM25Okapi`)
        3. Expansão Automática de Referência Cruzada: busca até `max_cross_refs` (2) artigos
           mencionados em `artigos_citados` dos chunks Top-K que ainda não estejam na lista.
        """
        if not self.chunks or self.bm25 is None:
            self._load_local_chunks_and_bm25()

        indexed_count = self.get_indexed_count()
        if indexed_count == 0:
            raise RuntimeError(
                "A coleção do ChromaDB está vazia. Sincronize e indexe a base primeiro."
            )

        # 1. Ranking Vetorial no ChromaDB (task_type="RETRIEVAL_QUERY")
        if use_mock_embeddings:
            query_emb = self._deterministic_test_embedding(query)
        else:
            query_emb = self._embed_texts(
                [query],
                task_type="RETRIEVAL_QUERY",
                api_key=api_key,
            )[0]

        n_Query = min(max(top_k * 3, 15), indexed_count)
        chroma_res = self.collection.query(
            query_embeddings=[query_emb],
            n_results=n_Query,
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

        # 2. Ranking Lexical com BM25Okapi
        bm25_ranks: Dict[str, int] = {}
        bm25_scores_map: Dict[str, float] = {}
        if self.bm25 is not None and self.chunks:
            query_tokens = tokenize_pt(query)
            raw_scores = self.bm25.get_scores(query_tokens)
            scored_indices = sorted(
                enumerate(raw_scores), key=lambda x: x[1], reverse=True
            )
            for rank_idx, (chunk_idx, score_val) in enumerate(
                scored_indices[: n_Query * 2], start=1
            ):
                c_id = self.chunks[chunk_idx]["id"]
                bm25_ranks[c_id] = rank_idx
                bm25_scores_map[c_id] = round(float(score_val), 4)

        # 3. Fusão Reciprocal Rank Fusion (RRF)
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

            rrf_results.append(
                {
                    "id": cid,
                    "texto": texto,
                    "metadata": metadata,
                    "rrf_score": round(rrf_score, 6),
                    "vector_rank": v_rank,
                    "vector_score": vector_scores.get(cid, 0.0),
                    "bm25_rank": b_rank,
                    "bm25_score": bm25_scores_map.get(cid, 0.0),
                    "origem": "Busca Híbrida (ChromaDB Cosine + BM25 RRF)",
                    "is_cross_reference": False,
                    "cited_by": None,
                }
            )

        rrf_results.sort(
            key=lambda x: (x["rrf_score"], x["vector_score"], x["bm25_score"]),
            reverse=True,
        )
        top_chunks = rrf_results[:top_k]

        # 4. Expansão Automática de Referência Cruzada (até max_cross_refs = 2 artigos complementares)
        present_articles = {
            str(item["metadata"].get("artigo", "")).strip()
            for item in top_chunks
            if str(item["metadata"].get("artigo", "")).strip()
        }
        present_ids = {item["id"] for item in top_chunks}

        expanded_chunks: List[Dict[str, Any]] = []
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

                # Busca no ChromaDB por filtro de metadado {"artigo": cited_art}
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
                                    "vector_rank": None,
                                    "vector_score": 0.0,
                                    "bm25_rank": None,
                                    "bm25_score": 0.0,
                                    "origem": f"Referência Cruzada Automática (citado por {source_artigo})",
                                    "is_cross_reference": True,
                                    "cited_by": source_artigo,
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
        """Formata os chunks recuperados e expandidos dentro da tag <contexto_normativo>."""
        blocks: List[str] = ["<contexto_normativo>"]
        for idx, item in enumerate(retrieved_chunks, start=1):
            meta = item.get("metadata", {})
            origem = item.get("origem", "Busca Híbrida")
            breadcrumb = meta.get("breadcrumb", "")
            artigos_citados = meta.get("artigos_citados", "")
            blocks.append(
                f'  <documento indice="{idx}" id="{item["id"]}" '
                f'origem="{origem}" breadcrumb="{breadcrumb}" '
                f'artigos_citados="{artigos_citados}">\n'
                f"{item['texto']}\n"
                f"  </documento>"
            )
        blocks.append("</contexto_normativo>")
        return "\n".join(blocks)

    def answer_question(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        model_name: str = "gemini-3.8-flash",
        top_k: int = 5,
        max_cross_refs: int = 2,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executa o pipeline RAG completo:
        1. Reescrita de Query com Histórico (se aplicável)
        2. Busca Híbrida (ChromaDB + BM25 via RRF) + Expansão Automática de Referências Cruzadas (até 2 artigos)
        3. Geração da resposta fundamentada no <contexto_normativo> com temperature=0.1
        """
        rewritten_query = self.rewrite_query_with_history(
            query=question,
            chat_history=chat_history,
            model_name=model_name,
            api_key=api_key,
        )

        retrieved_chunks = self.hybrid_search_with_cross_ref(
            query=rewritten_query,
            top_k=top_k,
            max_cross_refs=max_cross_refs,
            api_key=api_key,
        )

        context_xml = self.build_normative_context_xml(retrieved_chunks)
        user_prompt = (
            f"{context_xml}\n\n"
            f"Pergunta Original do Usuário: {question}\n"
            f"Consulta Técnica Utilizada na Busca: {rewritten_query}\n\n"
            "Elabore a resposta seguindo estritamente as 3 seções obrigatórias "
            "(1. Resumo Direto, 2. Fundamentação Normativa Exata, 3. Implicação Prática / Checklist de Conformidade) "
            "baseando-se exclusivamente no <contexto_normativo> acima."
        )

        client = self._get_genai_client(api_key=api_key)
        gen_config = types.GenerateContentConfig(
            temperature=0.1,
            system_instruction=SYSTEM_INSTRUCTION,
        )

        selected_model = DEFAULT_GENERATION_MODEL
        t0 = time.perf_counter()
        response, used_model = self._generate_content_resilient(
            client=client,
            preferred_model=selected_model,
            contents=user_prompt,
            config=gen_config,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        answer_text = (response.text or "").strip()
        hybrid_count = sum(1 for c in retrieved_chunks if not c.get("is_cross_reference"))
        cross_ref_count = sum(1 for c in retrieved_chunks if c.get("is_cross_reference"))

        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": answer_text,
            "model_requested": selected_model,
            "model_used": used_model,
            "temperature": 0.1,
            "latency_ms": latency_ms,
            "hybrid_count": hybrid_count,
            "cross_ref_count": cross_ref_count,
            "system_instruction": SYSTEM_INSTRUCTION,
            "user_prompt": user_prompt,
            "retrieved_chunks": retrieved_chunks,
            "context_xml": context_xml,
        }
