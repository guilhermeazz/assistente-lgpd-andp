"""
Interface Conversacional Auditável Multi-Provedor e Documentação Arquitetural (src/app.py)
Implementa:
- Cache em Memória (@st.cache_resource) para SentenceTransformer local, ChromaDB e BM25
- Seletor Multi-Provedor na Sidebar:
  1. Qwen (Local - Ollama) [Padrão] — 100% offline/local com diagnóstico `check_ollama_status()`
  2. Google Gemini (API Key - BYOK)
  3. OpenAI / ChatGPT (API Key - BYOK)
  4. Anthropic / Claude (API Key - BYOK)
- Renderização de respostas em tempo real via `st.write_stream`
- Telemetria discreta abaixo de cada resposta: Latência de Recuperação (ms) e TTFT (Tempo até o 1º Token)
- Design Dark Obsidian & Teal Glassmorphism com ondas SVG responsivas e barra de digitação fixa
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VENV_SITE_PACKAGES = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from src.ingestion.fetcher import DEFAULT_RAW_DIR, run_ingestion_fetcher
from src.ingestion.parser import DEFAULT_OUTPUT_JSONL, run_parser_pipeline
from src.rag.engine import (
    DEFAULT_MAX_CROSS_REFS,
    DEFAULT_TOP_K,
    FLASHRANK_CACHE_DIR,
    FLASHRANK_MODEL,
    LOCAL_EMBEDDING_MODEL,
    LGPDRagEngine,
    get_local_embedder,
    get_local_reranker,
)
from src.rag.prompts import SYSTEM_PROMPT
from src.rag.providers import (
    ANTHROPIC_MODELS,
    DEFAULT_OLLAMA_URL,
    GEMINI_MODELS,
    HF_QWEN_MODELS,
    OLLAMA_MODELS,
    OPENAI_MODELS,
    PROVIDER_DEFAULT_MODELS,
    PROVIDER_LABELS,
    check_ollama_status,
    get_configured_hf_token,
    normalize_provider_key,
)

FEATURED_CARDS: List[Dict[str, str]] = [
    {
        "icon": "💬",
        "title": "Bases Legais & Art. 7º",
        "desc": "Analise as hipóteses autorizativas da LGPD e regras de revogação do consentimento.",
        "prompt": "Quais são as hipóteses legais do Art. 7º da LGPD e os requisitos do consentimento?",
    },
    {
        "icon": "⚖️",
        "title": "Sanções & Dosimetria",
        "desc": "Consulte multas do Art. 52 e a tabela de alíquotas da Resolução CD/ANPD nº 4/2023.",
        "prompt": "Quais são as sanções do Art. 52 da LGPD e as alíquotas de dosimetria da ANPD?",
    },
    {
        "icon": "📄",
        "title": "ANPD & Pequeno Porte",
        "desc": "Verifique competências do Art. 55-J e dispensas da Resolução CD/ANPD nº 2/2022.",
        "prompt": "Quais dispensas e prazos diferenciados a Resolução CD/ANPD nº 2/2022 concede a agentes de pequeno porte?",
    },
]

EXTRA_SUGGESTIONS: List[str] = [
    "Quais são as competências da ANPD previstas no Art. 55-J e o papel sancionatório do Art. 55-K?",
    "Qual é o prazo para comunicar incidentes de segurança à ANPD segundo o Art. 48 e a Resolução nº 15/2024?",
    "Quais são os direitos do titular no Art. 18 da LGPD e o prazo de atendimento do Art. 19?",
    "Quais são as regras sobre conflito de interesses e indicação do Encarregado (DPO) na Resolução nº 18/2024?",
]


@st.cache_resource(show_spinner=False)
def get_cached_embedder() -> Any:
    """Cacheia o modelo SentenceTransformer local em memória para toda a sessão do servidor."""
    return get_local_embedder(LOCAL_EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_cached_reranker() -> Any:
    """Cacheia o objeto Ranker do FlashRank (`ms-marco-MultiBERT-L-12`) para manter o reranking < 100ms."""
    return get_local_reranker(
        model_name=FLASHRANK_MODEL,
        cache_dir=FLASHRANK_CACHE_DIR,
    )


@st.cache_resource(show_spinner=False)
def get_cached_rag_engine() -> LGPDRagEngine:
    """
    Cacheia a instância do LGPDRagEngine (incluindo cliente ChromaDB, SentenceTransformer local,
    FlashRank Cross-Encoder e índice lexical BM25) para evitar recarregamento de disco a cada pergunta.
    """
    embedder = get_cached_embedder()
    reranker = get_cached_reranker()
    engine = LGPDRagEngine(embedder=embedder, reranker=reranker)
    if engine.chunks and engine.get_indexed_count() != len(engine.chunks):
        try:
            engine.index_chunks()
        except Exception:
            pass
    return engine


def _inject_lix_inspired_css() -> None:
    """Injeta o tema visual Dark Obsidian & Teal Glassmorphism com ondas responsivas e métricas discretas."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(ellipse 75% 42% at 50% 102%, rgba(14, 165, 233, 0.22) 0%, rgba(13, 148, 136, 0.10) 38%, rgba(9, 13, 20, 0.0) 75%),
                linear-gradient(180deg, #070a10 0%, #090d14 55%, #0b131e 100%) !important;
            color: #f8fafc !important;
        }

        section[data-testid="stSidebar"] {
            background: #06090f !important;
            border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
        }

        .block-container {
            padding-top: 2.0rem !important;
            padding-bottom: 3.5rem !important;
            max-width: 920px !important;
        }

        .brand-logo {
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            color: #f8fafc;
            margin-bottom: 0.25rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .brand-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #0ea5e9;
            box-shadow: 0 0 12px #0ea5e9;
            display: inline-block;
        }

        .wave-hero-container {
            position: relative;
            width: 100%;
            height: clamp(58px, 8.5vw, 96px);
            overflow: visible;
            margin-top: 0.1rem;
            margin-bottom: 0.65rem;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        .wave-svg {
            width: 100%;
            height: 100%;
            display: block;
            background: transparent !important;
            overflow: visible;
            filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.28));
        }
        .wave-path-1 {
            animation: wavePulse1 7s ease-in-out infinite alternate;
            transform-origin: center;
        }
        .wave-path-2 {
            animation: wavePulse2 9s ease-in-out infinite alternate;
            transform-origin: center;
        }
        .wave-path-3 {
            animation: wavePulse1 11s ease-in-out infinite alternate-reverse;
            transform-origin: center;
        }
        @keyframes wavePulse1 {
            0% { transform: translateY(-3px) scaleY(0.94); opacity: 0.75; }
            100% { transform: translateY(4px) scaleY(1.06); opacity: 1.0; }
        }
        @keyframes wavePulse2 {
            0% { transform: translateY(4px) scaleY(1.05); opacity: 0.65; }
            100% { transform: translateY(-4px) scaleY(0.92); opacity: 0.95; }
        }

        .lix-greeting-wrap {
            text-align: center;
            margin-bottom: 1.55rem;
        }
        .lix-hi {
            font-size: 1.25rem;
            font-weight: 600;
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 52%, #2dd4bf 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 0.01em;
            margin-bottom: 0.2rem;
        }
        .lix-headline {
            font-size: 2.25rem;
            font-weight: 700;
            background: linear-gradient(180deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.025em;
            margin-bottom: 0.55rem;
        }
        .lix-subtext {
            font-size: 0.90rem;
            color: #64748b;
            max-width: 560px;
            margin: 0 auto;
            line-height: 1.45;
        }

        .pro-shell {
            width: 100%;
            box-sizing: border-box;
            background: linear-gradient(180deg, rgba(30, 58, 82, 0.42) 0%, rgba(17, 29, 44, 0.36) 100%);
            border: 1px solid rgba(56, 189, 248, 0.22);
            border-radius: 20px;
            padding: 0.55rem 0.65rem 0.65rem 0.65rem;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.48);
            margin-bottom: 0.75rem;
        }

        div[data-testid="stBottom"] {
            background: linear-gradient(180deg, rgba(9, 13, 20, 0.0) 0%, rgba(9, 13, 20, 0.92) 38%, #090d14 100%) !important;
        }
        div[data-testid="stBottom"] > div,
        div[data-testid="stBottomBlockContainer"] {
            max-width: 920px !important;
            width: 100% !important;
            margin: 0 auto !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-bottom: 1.1rem !important;
            box-sizing: border-box !important;
            background: transparent !important;
        }
        div[data-testid="stChatInput"] {
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 auto !important;
            box-sizing: border-box !important;
        }
        div[data-testid="stChatInput"] > div {
            width: 100% !important;
            max-width: 100% !important;
            background: #0b0f17 !important;
            border: 1px solid rgba(56, 189, 248, 0.28) !important;
            border-radius: 16px !important;
            padding: 0.25rem 0.45rem !important;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.55) !important;
        }
        .pro-header-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.25rem 0.75rem 0.50rem 0.75rem;
            font-size: 0.80rem;
            font-weight: 600;
            color: #e2e8f0;
        }
        .pro-header-left {
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }
        .pro-spark {
            color: #38bdf8;
            font-size: 0.95rem;
        }
        .pro-inner-box {
            background: #0b0f17;
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 15px;
            padding: 0.85rem 1.05rem;
        }
        .tool-pills-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.35rem;
        }
        .tool-pill {
            font-size: 0.73rem;
            color: #94a3b8;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.09);
            border-radius: 999px;
            padding: 0.22rem 0.68rem;
            display: inline-flex;
            align-items: center;
            gap: 0.32rem;
        }

        .latency-bar {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.65rem;
            font-size: 0.75rem;
            color: #94a3b8;
            background: rgba(11, 17, 27, 0.72);
            border: 1px solid rgba(56, 189, 248, 0.16);
            border-radius: 10px;
            padding: 0.38rem 0.75rem;
            margin-top: 0.45rem;
            margin-bottom: 0.45rem;
        }
        .latency-badge {
            color: #38bdf8;
            font-weight: 600;
        }

        .feature-card {
            background: rgba(15, 25, 38, 0.52);
            border: 1px solid rgba(56, 189, 248, 0.12);
            border-radius: 14px;
            padding: 0.95rem 1.05rem 0.65rem 1.05rem;
            min-height: 96px;
            transition: all 0.2s ease;
        }
        .feature-card:hover {
            border-color: rgba(56, 189, 248, 0.35);
            background: rgba(20, 35, 54, 0.68);
        }
        .feature-card-title {
            font-size: 0.88rem;
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 0.3rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .feature-card-desc {
            font-size: 0.77rem;
            color: #64748b;
            line-height: 1.38;
        }

        .stButton > button {
            border-radius: 11px !important;
            border: 1px solid rgba(56, 189, 248, 0.18) !important;
            background: rgba(15, 23, 36, 0.72) !important;
            color: #e2e8f0 !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
            transition: all 0.18s ease !important;
        }
        .stButton > button:hover {
            border-color: #0ea5e9 !important;
            background: rgba(14, 165, 233, 0.16) !important;
            color: #ffffff !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 100%) !important;
            border: none !important;
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        div[data-testid="stChatMessage"] {
            background: rgba(15, 23, 36, 0.58) !important;
            border: 1px solid rgba(255, 255, 255, 0.07) !important;
            border-radius: 16px !important;
            padding: 1rem 1.15rem !important;
            margin-bottom: 0.85rem !important;
        }
        div[data-testid="stExpander"] {
            background: rgba(11, 17, 27, 0.65) !important;
            border: 1px solid rgba(56, 189, 248, 0.14) !important;
            border-radius: 12px !important;
        }

        .arch-card {
            background: rgba(15, 25, 38, 0.58);
            border: 1px solid rgba(56, 189, 248, 0.16);
            border-radius: 16px;
            padding: 1.2rem 1.3rem;
            margin-bottom: 1rem;
        }
        .arch-card h4 {
            color: #f8fafc;
            font-size: 1.02rem;
            margin-bottom: 0.6rem;
        }
        .arch-card li {
            color: #94a3b8;
            font-size: 0.86rem;
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    """Inicializa variáveis de sessão para arquitetura Multi-Provedor (Qwen Local Padrão + Qwen HF GPU Sem Chave + Cloud BYOK)."""
    if "provider_key" not in st.session_state:
        st.session_state["provider_key"] = "ollama"
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = PROVIDER_DEFAULT_MODELS["ollama"]
    if "ollama_base_url" not in st.session_state:
        st.session_state["ollama_base_url"] = DEFAULT_OLLAMA_URL
    if "custom_ollama_tag" not in st.session_state:
        st.session_state["custom_ollama_tag"] = ""
    if "api_keys" not in st.session_state:
        st.session_state["api_keys"] = {
            "gemini": "",
            "openai": "",
            "anthropic": "",
        }
    if "gemini_api_key" not in st.session_state:
        st.session_state["gemini_api_key"] = ""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "pending_suggestion" not in st.session_state:
        st.session_state["pending_suggestion"] = None


def _render_latency_telemetry_bar(trace: Dict[str, Any]) -> None:
    """Exibe resumo de latência detalhado abaixo da resposta: Busca + Rerank (ms) e TTFT (s)."""
    if not trace:
        return
    retrieval_ms = trace.get("retrieval_ms", 0.0)
    ttft_ms = trace.get("ttft_ms", 0.0)
    ttft_s = trace.get("ttft_s", round(float(ttft_ms) / 1000.0, 2))
    provider = trace.get("provider", "ollama").upper()
    model_used = trace.get("model_used", "")
    hybrid_cnt = trace.get("hybrid_count", 0)
    cross_cnt = trace.get("cross_ref_count", 0)

    st.markdown(
        f"""
        <div class="latency-bar">
            <span>⚡ Busca + Rerank: <span class="latency-badge">{retrieval_ms} ms</span></span>
            <span>|</span>
            <span>⏱️ Primeiro Token (TTFT): <span class="latency-badge">{ttft_s:.2f} s</span> ({ttft_ms} ms)</span>
            <span>|</span>
            <span>🤖 <code>{provider} / {model_used}</code> ({hybrid_cnt} Reranked + {cross_cnt} Cross-Ref)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_grounding_expander(
    retrieved_chunks: List[Dict[str, Any]],
    trace: Dict[str, Any] | None = None,
) -> None:
    """
    Renderiza o expander auditável '🔍 Fontes Oficiais Recuperadas e Auditoria (Grounding)' com:
    1. Conceitos jurídicos expandidos automaticamente (se ativados no dicionário).
    2. Nota de relevância do Reranker (Score: 0.XX) ao lado da referência normativa e texto exato.
    3. Resumo de latência detalhado: Busca + Rerank: X ms | Primeiro Token (TTFT): Y s.
    """
    with st.expander(
        "🔍 Fontes Oficiais Recuperadas e Auditoria (Grounding)",
        expanded=False,
    ):
        trace_data = trace or {}
        retrieval_ms = trace_data.get("retrieval_ms", 0.0)
        ttft_ms = trace_data.get("ttft_ms", 0.0)
        ttft_s = trace_data.get("ttft_s", round(float(ttft_ms) / 1000.0, 2))

        st.caption(
            f"⏱️ **Telemetria do Pipeline:** `Busca + Rerank: {retrieval_ms} ms | Primeiro Token (TTFT): {ttft_s:.2f} s`"
        )

        # 1. Conceitos jurídicos expandidos automaticamente
        matched_concepts = trace_data.get("matched_concepts") or (
            retrieved_chunks[0].get("matched_concepts", []) if retrieved_chunks else []
        )
        if matched_concepts:
            st.markdown("##### 🧠 Conceitos Jurídicos Expandidos Automaticamente (`~0 ms`)")
            for concept in matched_concepts:
                st.markdown(f"- 🔹 `{concept}`")
            st.divider()

        if not retrieved_chunks:
            st.info("Nenhum chunk normativo associado a esta resposta.")
            return

        # 2. Chunks recuperados com Score do Reranker (Score: 0.XX)
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            meta = chunk.get("metadata", {})
            origem = chunk.get("origem", "Busca Híbrida")
            is_cross = chunk.get("is_cross_reference", False)
            rerank_score = float(chunk.get("rerank_score", 0.0))
            badge = (
                "🔗 **[EXPANSÃO DE REFERÊNCIA CRUZADA]**"
                if is_cross
                else "🎯 **[TOP-4 RERANKER / RRF]**"
            )

            st.markdown(
                f"##### #{idx} {badge} `{meta.get('breadcrumb', chunk.get('id'))}` — **Score: `{rerank_score:.2f}`**"
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"🧭 **Origem:** {origem}")
            with col2:
                if is_cross:
                    st.caption(
                        f"↪️ **Citado por:** `{chunk.get('cited_by', '-')}` | **Score:** `{rerank_score:.2f}`"
                    )
                else:
                    st.caption(
                        f"🏅 **Rerank Score:** `{rerank_score:.2f}` | "
                        f"**RRF:** `{chunk.get('rrf_score', 0.0):.5f}` | "
                        f"**Cosine:** `{chunk.get('vector_score', 0.0):.4f}`"
                    )
            with col3:
                citados = meta.get("artigos_citados") or "Nenhuma"
                st.caption(f"🔗 **Artigos Citados:** `{citados}`")

            with st.container(border=True):
                st.markdown(chunk.get("texto", ""))
                with st.expander("📋 Metadados Estruturais (JSON)", expanded=False):
                    st.json(meta)


def _render_request_response_inspector(msg: Dict[str, Any]) -> None:
    """Renderiza o inspetor completo do contexto de Requisição e Resposta de cada mensagem."""
    trace = msg.get("trace")
    if not trace:
        return

    with st.expander(
        "🧾 Contexto Completo de Requisição e Resposta (Auditoria Técnica)",
        expanded=False,
    ):
        tab_req, tab_resp, tab_sys = st.tabs(
            [
                "📤 Contexto da Requisição (Prompt + XML)",
                "📥 Telemetria & Latência",
                "⚙️ System Prompt & Reranker",
            ]
        )

        with tab_req:
            c1, c2 = st.columns(2)
            c1.markdown(
                f"**💬 Pergunta do Usuário:**\n> {trace.get('question', '-')}"
            )
            c2.markdown(
                f"**🔎 Query Expandida (Expansor Jurídico):**\n> `{trace.get('rewritten_query', '-')}`"
            )
            st.markdown("**📦 Tag `<contexto_normativo>` Enviada ao Modelo:**")
            st.code(trace.get("context_xml", ""), language="xml")
            st.markdown("**📝 Prompt Completo de Entrada:**")
            st.code(trace.get("user_prompt", ""), language="markdown")

        with tab_resp:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🤖 Modelo", trace.get("model_used", "-"))
            m2.metric("⚡ Busca + Rerank", f"{trace.get('retrieval_ms', 0)} ms")
            ttft_s = trace.get("ttft_s", round(float(trace.get("ttft_ms", 0.0)) / 1000.0, 2))
            m3.metric("⏱️ Primeiro Token (TTFT)", f"{ttft_s:.2f} s")
            m4.metric("🏁 Tempo Total", f"{trace.get('latency_ms', 0)} ms")

            st.caption(
                f"Provedor: `{trace.get('provider', 'ollama')}` | "
                f"Chunks Reranked (Top-K={DEFAULT_TOP_K}): `{trace.get('hybrid_count', 0)}` | "
                f"Cross-Reference (Máx={DEFAULT_MAX_CROSS_REFS}): `{trace.get('cross_ref_count', 0)}` | "
                f"Temperatura: `{trace.get('temperature', 0.1)}`"
            )
            st.markdown("**📄 Resposta Bruta Retornada (Markdown):**")
            st.code(trace.get("answer", ""), language="markdown")

        with tab_sys:
            st.markdown(
                f"- **Provedor Ativo:** `{trace.get('provider', 'ollama')}` (`{trace.get('model_used', '-')}`)\n"
                f"- **Embeddings 100% Locais:** `{LOCAL_EMBEDDING_MODEL}` (384-d Cosine)\n"
                f"- **Reranker Cross-Encoder Local:** `FlashRank ({FLASHRANK_MODEL})`\n"
                f"- **Busca em 2 Estágios:** `Top-10 RRF (Chroma + BM25)` $\\rightarrow$ `Top-{DEFAULT_TOP_K} FlashRank` + `{DEFAULT_MAX_CROSS_REFS} Cross-Ref`"
            )
            st.markdown("**📜 System Prompt Visual & Interpretativo (`src/rag/prompts.py`):**")
            st.code(
                trace.get("system_instruction", SYSTEM_PROMPT), language="markdown"
            )


def _render_chat_page(engine: LGPDRagEngine) -> None:
    """Renderiza a página principal de Chat com streaming nativo e telemetria TTFT."""
    indexed_count = engine.get_indexed_count()
    jsonl_count = len(engine.chunks)
    provider_key = st.session_state["provider_key"]
    active_model = st.session_state["selected_model"]
    active_api_key = st.session_state["api_keys"].get(provider_key, "")
    hf_system_token = get_configured_hf_token()

    if provider_key == "ollama":
        ollama_info = check_ollama_status(
            base_url=st.session_state["ollama_base_url"],
            model_name=active_model,
        )
        if ollama_info["online"] and ollama_info["model_ready"]:
            status_badge = f"🟢 Ollama Local Online ({active_model})"
        elif ollama_info["online"]:
            status_badge = f"🟡 Ollama Ativo (falta pull {active_model})"
        elif hf_system_token:
            status_badge = "🟢 Qwen Cloud GPU (Failover HF Ativo)"
        else:
            status_badge = "🟠 Ollama Local Offline"
    elif provider_key == "huggingface":
        status_badge = (
            f"🟢 Qwen GPU Hugging Face ({active_model.split('/')[-1]})"
            if hf_system_token
            else "🟠 Aguardando HF_TOKEN nos Secrets"
        )
    else:
        status_badge = (
            f"🟢 API Key ({provider_key.upper()})"
            if active_api_key
            else f"🟠 Aguardando API Key ({provider_key.upper()})"
        )

    st.markdown(
        """
        <div class="wave-hero-container">
            <svg class="wave-svg" viewBox="0 0 900 120" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="waveGrad1" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#0ea5e9" stop-opacity="0.05"/>
                        <stop offset="28%" stop-color="#38bdf8" stop-opacity="0.85"/>
                        <stop offset="55%" stop-color="#818cf8" stop-opacity="0.90"/>
                        <stop offset="80%" stop-color="#2dd4bf" stop-opacity="0.75"/>
                        <stop offset="100%" stop-color="#0ea5e9" stop-opacity="0.05"/>
                    </linearGradient>
                    <linearGradient id="waveGrad2" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#6366f1" stop-opacity="0.0"/>
                        <stop offset="35%" stop-color="#06b6d4" stop-opacity="0.65"/>
                        <stop offset="65%" stop-color="#3b82f6" stop-opacity="0.70"/>
                        <stop offset="100%" stop-color="#14b8a6" stop-opacity="0.0"/>
                    </linearGradient>
                    <linearGradient id="waveGrad3" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#2dd4bf" stop-opacity="0.0"/>
                        <stop offset="50%" stop-color="#38bdf8" stop-opacity="0.45"/>
                        <stop offset="100%" stop-color="#818cf8" stop-opacity="0.0"/>
                    </linearGradient>
                </defs>
                <path class="wave-path-1" d="M 0,62 C 180,18 320,104 480,58 C 640,12 760,94 900,52" fill="none" stroke="url(#waveGrad1)" stroke-width="2.4"/>
                <path class="wave-path-2" d="M 0,76 C 150,108 330,24 510,68 C 680,110 790,30 900,66" fill="none" stroke="url(#waveGrad2)" stroke-width="1.8"/>
                <path class="wave-path-3" d="M 0,46 C 210,88 370,16 540,52 C 700,88 810,28 900,44" fill="none" stroke="url(#waveGrad3)" stroke-width="1.3" stroke-dasharray="6 4"/>
                <path class="wave-path-2" d="M 0,88 C 220,42 410,98 600,48 C 740,14 830,78 900,60" fill="none" stroke="url(#waveGrad1)" stroke-width="1.0" opacity="0.55"/>
            </svg>
        </div>
        <div class="lix-greeting-wrap">
            <div class="lix-hi">Assistente Inteligente LGPD & ANPD</div>
            <div class="lix-headline">Como posso ajudar hoje?</div>
            <div class="lix-subtext">
                Arquitetura RAG Híbrida de Baixa Latência — Embeddings 100% locais com SentenceTransformers
                e inferência em Streaming via Qwen Local (Ollama), Gemini, ChatGPT ou Claude.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="pro-shell">
            <div class="pro-header-bar">
                <div class="pro-header-left">
                    <span class="pro-spark">⚡</span>
                    <span>Motor RAG Multi-Provedor Ativo com <strong>{active_model}</strong></span>
                </div>
                <span style="font-size: 0.74rem; color: #38bdf8;">{status_badge} • {indexed_count}/{jsonl_count} Chunks</span>
            </div>
            <div class="pro-inner-box">
                <div style="font-size: 0.83rem; color: #94a3b8; margin-bottom: 0.45rem;">
                    Pipeline de Baixa Latência (Embeddings Locais + Streaming Token-a-Token):
                </div>
                <div class="tool-pills-row">
                    <span class="tool-pill">🧠 MiniLM-L12-v2 Local</span>
                    <span class="tool-pill">⚖️ Chroma Cosine + BM25 RRF</span>
                    <span class="tool-pill">📦 Context Budget (Top-4 + 1 Cross-Ref)</span>
                    <span class="tool-pill">⚡ Streaming Nativo (TTFT)</span>
                    <span class="tool-pill">🤖 {PROVIDER_LABELS[provider_key]}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    for idx, card in enumerate(FEATURED_CARDS):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-card-title">{card['icon']} {card['title']}</div>
                    <div class="feature-card-desc">{card['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                f"Consultar {card['title']}",
                key=f"feat_card_{idx}",
                use_container_width=True,
            ):
                st.session_state["pending_suggestion"] = card["prompt"]

    with st.expander("💡 Mais Sugestões de Consulta (minimizadas)", expanded=False):
        s_cols = st.columns(2)
        for idx, sug in enumerate(EXTRA_SUGGESTIONS):
            if s_cols[idx % 2].button(
                f"↗ {sug}", key=f"extra_sug_{idx}", use_container_width=True
            ):
                st.session_state["pending_suggestion"] = sug

    if st.session_state["messages"]:
        st.divider()

    for msg in st.session_state["messages"]:
        avatar = "🧑‍💼" if msg["role"] == "user" else "🛡️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("trace") is not None:
                    _render_latency_telemetry_bar(msg["trace"])
                if msg.get("retrieved_chunks") is not None:
                    _render_grounding_expander(
                        msg["retrieved_chunks"],
                        trace=msg.get("trace"),
                    )
                if msg.get("trace") is not None:
                    _render_request_response_inspector(msg)

    user_input = st.chat_input(
        "Pergunte qualquer coisa sobre a LGPD ou Resoluções da ANPD..."
    )
    active_question = user_input or st.session_state.get("pending_suggestion")
    if st.session_state.get("pending_suggestion"):
        st.session_state["pending_suggestion"] = None

    if active_question:
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(active_question)

        if provider_key in ("gemini", "openai", "anthropic") and not active_api_key:
            with st.chat_message("assistant", avatar="🛡️"):
                st.warning(
                    f"🔑 **Autenticação BYOK Necessária:** Insira sua API Key para **{PROVIDER_LABELS[provider_key]}** na barra lateral esquerda, "
                    "ou selecione **Qwen (Hugging Face GPU - Sem Chave)** / **Qwen (Local - Ollama) [Padrão]** para usar sem chave."
                )
            return

        if provider_key == "huggingface" and not hf_system_token:
            with st.chat_message("assistant", avatar="🛡️"):
                st.warning(
                    "☁️ **Configuração do Deploy Necessária:** Adicione `HF_TOKEN = \"hf_...\"` nos **Secrets** do Streamlit Cloud "
                    "(ou `.streamlit/secrets.toml` local) para habilitar o Qwen nas GPUs do Hugging Face para todos os usuários."
                )
            return

        if provider_key == "ollama":
            st_check = check_ollama_status(
                base_url=st.session_state["ollama_base_url"],
                model_name=active_model,
            )
            if not st_check["online"] and not hf_system_token:
                with st.chat_message("assistant", avatar="🛡️"):
                    st.warning(f"⚠️ **Ollama Local Offline:** {st_check['message']}")
                return
            if st_check["online"] and not st_check["model_ready"] and not hf_system_token:
                with st.chat_message("assistant", avatar="🛡️"):
                    st.warning(f"📥 **Modelo Qwen Ausente no Ollama:** {st_check['message']}")
                return

        with st.chat_message("assistant", avatar="🛡️"):
            try:
                prior_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["messages"]
                ]

                stream_result = engine.ask_stream(
                    question=active_question,
                    history=prior_history,
                    top_k=DEFAULT_TOP_K,
                    max_cross_refs=DEFAULT_MAX_CROSS_REFS,
                    provider_name=provider_key,
                    model_name=active_model,
                    api_key=active_api_key,
                    base_url=st.session_state["ollama_base_url"],
                )

                answer_md = st.write_stream(stream_result["stream"])
                if not isinstance(answer_md, str):
                    answer_md = stream_result.get("answer", "")

                trace_dict = dict(stream_result)
                trace_dict.pop("stream", None)
                trace_dict["answer"] = answer_md

                _render_latency_telemetry_bar(trace_dict)
                _render_grounding_expander(
                    trace_dict["retrieved_chunks"],
                    trace=trace_dict,
                )

                msg_record = {
                    "role": "assistant",
                    "content": answer_md,
                    "retrieved_chunks": trace_dict["retrieved_chunks"],
                    "trace": trace_dict,
                }
                _render_request_response_inspector(msg_record)

                st.session_state["messages"].append(
                    {"role": "user", "content": active_question}
                )
                st.session_state["messages"].append(msg_record)

            except Exception as exc:
                st.error(f"Não foi possível concluir a consulta: {exc}")


def _render_project_details_page(engine: LGPDRagEngine) -> None:
    """Renderiza a página dedicada ao detalhamento arquitetural e engenharia do projeto."""
    st.markdown(
        """
        <div class="lix-greeting-wrap" style="margin-bottom: 1.2rem;">
            <div class="lix-hi">Engenharia de Dados & IA Generativa de Baixa Latência</div>
            <div class="lix-headline">Arquitetura & Construção do Projeto</div>
            <div class="lix-subtext">
                Conheça como o motor RAG Híbrido Multi-Provedor combina Embeddings Locais (SentenceTransformers)
                e Streaming Token-a-Token via Qwen (Ollama), Gemini, OpenAI e Claude.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lgpd_chunks = [
        c for c in engine.chunks if c.get("metadata", {}).get("tipo") == "lgpd_artigo"
    ]
    anpd_chunks = [
        c for c in engine.chunks if c.get("metadata", {}).get("tipo") == "anpd_pdf"
    ]
    cross_ref_chunks = [
        c
        for c in engine.chunks
        if str(c.get("metadata", {}).get("artigos_citados", "")).strip()
    ]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📚 Total de Chunks", len(engine.chunks))
    k2.metric("⚖️ Artigos da LGPD", len(lgpd_chunks))
    k3.metric("📄 Chunks ANPD (PDF)", len(anpd_chunks))
    k4.metric("🔗 Com Cross-Reference", len(cross_ref_chunks))

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div class="arch-card">
                <h4>🧠 1. Embeddings 100% Locais (<code>SentenceTransformers</code>)</h4>
                <ul>
                    <li><strong>Desacoplamento Completo:</strong> Usa <code>paraphrase-multilingual-MiniLM-L12-v2</code> (384-d Cosine) rodando localmente em CPU/GPU.</li>
                    <li><strong>Indexação Única:</strong> O ChromaDB é indexado uma única vez sem gastar cota de API e atende Qwen, Gemini, ChatGPT e Claude.</li>
                    <li><strong>Latência em Milissegundos:</strong> Vetorização de queries e fusão RRF com BM25 ocorrem em menos de 100 ms.</li>
                </ul>
            </div>
            <div class="arch-card">
                <h4>⚖️ 3. Context Budget & Cross-Reference (<code>src/rag/engine.py</code>)</h4>
                <ul>
                    <li><strong>Orçamento Enxuto:</strong> Recupera <code>top_k=4</code> chunks via RRF + até <code>1</code> artigo complementar de referência cruzada.</li>
                    <li><strong>Reescrita Condicional:</strong> Evita chamadas duplas de LLM quando a pergunta tem 5+ palavras sem pronomes ambíguos.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="arch-card">
                <h4>🤖 2. Padrão Strategy/Factory Multi-Provedor (<code>src/rag/providers.py</code>)</h4>
                <ul>
                    <li><strong>OllamaProvider (Padrão):</strong> Conecta ao servidor local Ollama (<code>qwen2.5:7b</code>, <code>qwen2.5:3b</code>, <code>qwen2.5:14b</code>) 100% offline.</li>
                    <li><strong>Cloud BYOK:</strong> Suporte nativo em streaming para <code>google-genai</code> (Gemini), <code>openai</code> (ChatGPT) e <code>anthropic</code> (Claude).</li>
                    <li><strong>Temperatura Padronizada:</strong> Todos os provedores operam com <code>temperature=0.1</code> para rigor normativo.</li>
                </ul>
            </div>
            <div class="arch-card">
                <h4>⚡ 4. Streaming Nativo & Telemetria TTFT (<code>src/app.py</code>)</h4>
                <ul>
                    <li><strong>Cache de Recursos:</strong> <code>@st.cache_resource</code> mantém SentenceTransformer, ChromaDB e BM25 em RAM.</li>
                    <li><strong>Renderização Imediata:</strong> Usa <code>st.write_stream</code> com medição em tempo real de Recuperação (ms) e TTFT (Time To First Token).</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("🗂️ Explorador Interativo dos 82 Chunks (`data/processed/chunks_lgpd.jsonl`)")
    if not engine.chunks:
        st.info("Nenhum chunk carregado.")
        return

    chunk_options = {
        f"{idx:02d}. {c['metadata'].get('breadcrumb', c['id'])}": c
        for idx, c in enumerate(engine.chunks, start=1)
    }
    selected_label = st.selectbox(
        "Inspecione qualquer artigo da LGPD ou página de Resolução da ANPD:",
        options=list(chunk_options.keys()),
    )
    if selected_label:
        selected_chunk = chunk_options[selected_label]
        c_meta, c_text = st.columns([1, 2])
        with c_meta:
            st.markdown("**📋 Metadados Extraídos:**")
            st.json(selected_chunk["metadata"])
        with c_text:
            st.markdown("**📄 Texto Integral do Chunk:**")
            st.code(selected_chunk["texto"], language="markdown")


def _render_sidebar_provider_controls() -> None:
    """Renderiza o seletor Multi-Provedor e controles condicionais de Ollama ou BYOK na Sidebar."""
    st.markdown("##### 🤖 Provedor de IA")

    provider_options = list(PROVIDER_LABELS.keys())
    current_idx = (
        provider_options.index(st.session_state["provider_key"])
        if st.session_state["provider_key"] in provider_options
        else 0
    )

    selected_provider_label = st.selectbox(
        "Provedor de IA",
        options=[PROVIDER_LABELS[k] for k in provider_options],
        index=current_idx,
        label_visibility="collapsed",
    )
    provider_key = normalize_provider_key(selected_provider_label)
    st.session_state["provider_key"] = provider_key

    if provider_key == "ollama":
        preset_choice = st.selectbox(
            "Modelo Qwen (Ollama Local)",
            options=OLLAMA_MODELS + ["Outra tag local..."],
            index=0,
            help="qwen2.5:7b (padrão equilibrado), qwen2.5:3b (modo ultra-rápido), qwen2.5:14b ou digite qualquer tag.",
        )
        if preset_choice == "Outra tag local...":
            custom_tag = st.text_input(
                "Tag do modelo no Ollama",
                value=st.session_state["custom_ollama_tag"] or "qwen3:8b",
                placeholder="Ex: qwen3:8b",
            ).strip()
            st.session_state["custom_ollama_tag"] = custom_tag
            st.session_state["selected_model"] = custom_tag or "qwen2.5:7b"
        else:
            st.session_state["selected_model"] = preset_choice

        with st.expander("⚙️ Endpoint Ollama", expanded=False):
            st.session_state["ollama_base_url"] = st.text_input(
                "URL do Servidor Ollama",
                value=st.session_state["ollama_base_url"],
                placeholder="http://localhost:11434",
            ).strip() or DEFAULT_OLLAMA_URL

        status = check_ollama_status(
            base_url=st.session_state["ollama_base_url"],
            model_name=st.session_state["selected_model"],
        )
        if status["online"] and status["model_ready"]:
            st.success(status["message"])
        elif status["online"]:
            st.warning(status["message"])
        elif get_configured_hf_token():
            st.info(
                "☁️ Ollama local não detectado neste servidor. "
                "Failover automático para **Qwen (Hugging Face GPU)** ativado via `HF_TOKEN` do Streamlit!"
            )
        else:
            st.info(status["message"])

    elif provider_key == "huggingface":
        st.session_state["selected_model"] = st.selectbox(
            "Modelo Qwen (GPUs Hugging Face)",
            options=HF_QWEN_MODELS,
            index=0,
            help="Executado nas GPUs da Hugging Face Inference API usando a chave configurada no Streamlit Secrets.",
        )
        if get_configured_hf_token():
            st.success(
                "✅ **Acesso Liberado (Sem Chave):** Autenticado automaticamente via `HF_TOKEN` do sistema no Streamlit."
            )
        else:
            st.warning(
                "⚙️ Configure `HF_TOKEN = \"hf_...\"` nos **Secrets do Streamlit** (ou `.streamlit/secrets.toml`) "
                "para liberar o uso gratuito sem pedir chave ao usuário."
            )

    elif provider_key == "gemini":
        st.session_state["selected_model"] = st.selectbox(
            "Modelo Google Gemini",
            options=GEMINI_MODELS,
            index=0,
        )
        key_val = st.text_input(
            "🔑 Gemini API Key (BYOK)",
            value=st.session_state["api_keys"].get("gemini", ""),
            type="password",
            placeholder="AIzaSy...",
            help="Mantida exclusivamente em st.session_state durante sua sessão.",
        ).strip()
        st.session_state["api_keys"]["gemini"] = key_val
        st.session_state["gemini_api_key"] = key_val

    elif provider_key == "openai":
        st.session_state["selected_model"] = st.selectbox(
            "Modelo OpenAI (ChatGPT)",
            options=OPENAI_MODELS,
            index=0,
        )
        key_val = st.text_input(
            "🔑 OpenAI API Key (BYOK)",
            value=st.session_state["api_keys"].get("openai", ""),
            type="password",
            placeholder="sk-...",
            help="Mantida exclusivamente em st.session_state durante sua sessão.",
        ).strip()
        st.session_state["api_keys"]["openai"] = key_val

    elif provider_key == "anthropic":
        st.session_state["selected_model"] = st.selectbox(
            "Modelo Anthropic (Claude)",
            options=ANTHROPIC_MODELS,
            index=0,
        )
        key_val = st.text_input(
            "🔑 Anthropic API Key (BYOK)",
            value=st.session_state["api_keys"].get("anthropic", ""),
            type="password",
            placeholder="sk-ant-...",
            help="Mantida exclusivamente em st.session_state durante sua sessão.",
        ).strip()
        st.session_state["api_keys"]["anthropic"] = key_val

    st.caption(
        f"🧠 **Embeddings Locais:** `MiniLM-L12-v2`\n\n"
        f"🎯 **Context Budget:** `Top-{DEFAULT_TOP_K}` + `{DEFAULT_MAX_CROSS_REFS}` Cross-Ref (`temp=0.1`)"
    )


def main() -> None:
    st.set_page_config(
        page_title="LGPD.AI — Assistente Inteligente LGPD & ANPD",
        page_icon="🛡️",
        layout="wide",
    )
    _inject_lix_inspired_css()
    _init_session_state()
    engine = get_cached_rag_engine()

    with st.sidebar:
        st.markdown(
            '<div class="brand-logo"><span class="brand-dot"></span>LGPD.AI</div>',
            unsafe_allow_html=True,
        )
        st.caption("Conformidade LGPD & ANPD • Multi-LLM")

        page = st.radio(
            "Navegação",
            options=[
                "🏠 Início & Chat Consultivo",
                "📐 Arquitetura & Projeto",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()
        _render_sidebar_provider_controls()

        st.divider()
        st.markdown("##### 🗄️ Base Normativa Local")
        indexed_count = engine.get_indexed_count()
        jsonl_count = len(engine.chunks)
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("ChromaDB", indexed_count)
        col_m2.metric("Chunks", jsonl_count)

        if st.button(
            "🔄 Sincronizar e Indexar Base Oficial",
            use_container_width=True,
            type="primary",
        ):
            with st.status(
                "Sincronizando LGPD + Resoluções ANPD (100% Local)...", expanded=True
            ) as status_box:
                st.write("🌐 1/3 Coleta com SHA-256 e fallback local...")
                fetch_report = run_ingestion_fetcher(raw_dir=DEFAULT_RAW_DIR)
                if fetch_report["overall_status"] != "ok":
                    st.warning("Contingência local utilizada para arquivos gov.br.")

                st.write("🧩 2/3 Executando Parser Semântico-Estrutural...")
                chunks = run_parser_pipeline(
                    raw_dir=DEFAULT_RAW_DIR, output_jsonl=DEFAULT_OUTPUT_JSONL
                )
                st.write(f"Extraídos {len(chunks)} chunks normativos.")

                st.write(f"🧠 3/3 Indexando localmente com `{LOCAL_EMBEDDING_MODEL}`...")
                try:
                    total_indexed = engine.index_chunks()
                    status_box.update(
                        label=f"✅ {total_indexed} chunks indexados localmente!",
                        state="complete",
                    )
                    st.rerun()
                except Exception as exc:
                    status_box.update(
                        label="❌ Erro na indexação", state="error"
                    )
                    st.error(f"Falha ao indexar base: {exc}")

        with st.expander("📎 Importar PDF da ANPD", expanded=False):
            uploaded_files = st.file_uploader(
                "Novas Resoluções ou Guias (.pdf)",
                type=["pdf"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                DEFAULT_RAW_DIR.mkdir(parents=True, exist_ok=True)
                saved_names: List[str] = []
                for uf in uploaded_files:
                    dest = DEFAULT_RAW_DIR / uf.name
                    dest.write_bytes(uf.getvalue())
                    saved_names.append(uf.name)
                st.success(
                    f"Salvo em `data/raw/`: {', '.join(saved_names)}. "
                    "Clique em 'Sincronizar e Indexar Base Oficial'."
                )

        if st.button("🗑️ Limpar Conversa", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["pending_suggestion"] = None
            st.rerun()

    if page == "🏠 Início & Chat Consultivo":
        _render_chat_page(engine)
    else:
        _render_project_details_page(engine)


if __name__ == "__main__":
    if get_script_run_ctx() is not None:
        main()
    else:
        venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        python_exe = str(venv_python) if venv_python.exists() else sys.executable
        sys.exit(
            subprocess.call(
                [python_exe, "-m", "streamlit", "run", str(Path(__file__).resolve())]
            )
        )
