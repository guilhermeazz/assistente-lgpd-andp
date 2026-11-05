"""
Interface Conversacional Auditável e Documentação Arquitetural (src/app.py)
Design inspirado no conceito Dark Obsidian & Teal Glassmorphism (estilo LIX AI):
- Orbe 3D iridescente centralizado, tipografia em dois níveis ("Olá, ..." / "Como posso ajudar hoje?")
- Container central estilo "Pro" com barra de status do pipeline e pílulas de ferramentas
- 3 cards minimalistas de ação rápida em vidro fumê + sugestões minimizadas
- Chat estruturado com inspeção completa de Requisição, Resposta e Grounding (RRF + Cross-Reference)
- Página dedicada ao detalhamento arquitetural e engenharia do projeto
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# Garante que a raiz do projeto e o site-packages do .venv estejam no sys.path
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
import src.rag.engine as _engine_mod

if not hasattr(_engine_mod.LGPDRagEngine, "_discover_available_models") or not hasattr(
    _engine_mod.LGPDRagEngine, "_embedding_cache"
):
    importlib.reload(_engine_mod)

from src.rag.engine import (
    DEFAULT_GENERATION_MODEL,
    EMBEDDING_MODEL,
    LGPDRagEngine,
    SYSTEM_INSTRUCTION,
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


def _inject_lix_inspired_css() -> None:
    """Injeta o tema visual Dark Obsidian com brilho radial azul-petróleo e Orbe 3D iridescente."""
    st.markdown(
        """
        <style>
        /* Fundo principal Dark Obsidian com brilho radial Teal na base (inspirado na referência LIX) */
        .stApp {
            background:
                radial-gradient(ellipse 75% 42% at 50% 102%, rgba(14, 165, 233, 0.22) 0%, rgba(13, 148, 136, 0.10) 38%, rgba(9, 13, 20, 0.0) 75%),
                linear-gradient(180deg, #070a10 0%, #090d14 55%, #0b131e 100%) !important;
            color: #f8fafc !important;
        }

        /* Barra lateral minimalista escura */
        section[data-testid="stSidebar"] {
            background: #06090f !important;
            border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
        }

        .block-container {
            padding-top: 2.0rem !important;
            padding-bottom: 3.5rem !important;
            max-width: 920px !important;
        }

        /* Marca estilo LIX no topo */
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

        /* Efeito de Linhas em Ondas Fluidas Responsivas e Sem Fundo */
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

        /* Saudação Centralizada com Degradê */
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
            max-width: 520px;
            margin: 0 auto;
            line-height: 1.45;
        }

        /* Container Central Glassmorphic ("Unlock more features with Pro" style) */
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

        /* Barra de digitação fixa no rodapé com a mesma largura (920px) dos blocos centralizados */
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

        /* 3 Cards Minimalistas de Sugestão na Base */
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

        /* Botões estilizados em vidro escuro */
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

        /* Mensagens do Chat e Expanders */
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

        /* Cards da página de Arquitetura */
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
    """Inicializa variáveis obrigatórias em st.session_state (incluindo BYOK)."""
    if "gemini_api_key" not in st.session_state:
        st.session_state["gemini_api_key"] = ""
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = DEFAULT_GENERATION_MODEL
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "pending_suggestion" not in st.session_state:
        st.session_state["pending_suggestion"] = None
    if (
        "rag_engine" not in st.session_state
        or not hasattr(st.session_state["rag_engine"], "_embedding_cache")
    ):
        st.session_state["rag_engine"] = LGPDRagEngine(
            api_key=st.session_state["gemini_api_key"]
        )


def _render_grounding_expander(retrieved_chunks: List[Dict[str, Any]]) -> None:
    """Renderiza o expander auditável com as fontes recuperadas e referências cruzadas."""
    with st.expander(
        "🔍 Fontes Oficiais Recuperadas e Referências Cruzadas (Grounding)",
        expanded=False,
    ):
        if not retrieved_chunks:
            st.info("Nenhum chunk normativo associado a esta resposta.")
            return

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            meta = chunk.get("metadata", {})
            origem = chunk.get("origem", "Busca Híbrida")
            is_cross = chunk.get("is_cross_reference", False)
            badge = (
                "🔗 **[EXPANSÃO DE REFERÊNCIA CRUZADA]**"
                if is_cross
                else "🎯 **[BUSCA HÍBRIDA RRF]**"
            )

            st.markdown(
                f"##### #{idx} {badge} `{meta.get('breadcrumb', chunk.get('id'))}`"
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"🧭 **Origem:** {origem}")
            with col2:
                if is_cross:
                    st.caption(f"↪️ **Citado por:** `{chunk.get('cited_by', '-')}`")
                else:
                    st.caption(
                        f"📊 **RRF:** `{chunk.get('rrf_score', 0.0):.5f}` | "
                        f"**Cosine:** `{chunk.get('vector_score', 0.0):.4f}` | "
                        f"**BM25:** `{chunk.get('bm25_score', 0.0):.2f}`"
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
                "📥 Resposta & Telemetria",
                "⚙️ System Instruction & Parâmetros",
            ]
        )

        with tab_req:
            c1, c2 = st.columns(2)
            c1.markdown(
                f"**💬 Pergunta do Usuário:**\n> {trace.get('question', '-')}"
            )
            c2.markdown(
                f"**🔎 Query Técnica Executada na Busca Híbrida:**\n> `{trace.get('rewritten_query', '-')}`"
            )
            st.markdown("**📦 Tag `<contexto_normativo>` Enviada ao Modelo:**")
            st.code(trace.get("context_xml", ""), language="xml")
            st.markdown("**📝 Prompt Completo de Entrada (`contents`):**")
            st.code(trace.get("user_prompt", ""), language="markdown")

        with tab_resp:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "✨ Modelo Configurado",
                trace.get("model_requested", DEFAULT_GENERATION_MODEL),
            )
            m2.metric("🎯 Chunks Híbridos (RRF)", trace.get("hybrid_count", 0))
            m3.metric("🔗 Cross-References", trace.get("cross_ref_count", 0))
            m4.metric("⏱️ Latência Geração", f"{trace.get('latency_ms', 0)} ms")

            st.caption(
                f"Cluster efetivo utilizado na API: `{trace.get('model_used', DEFAULT_GENERATION_MODEL)}` | "
                f"Temperatura: `{trace.get('temperature', 0.1)}`"
            )
            st.markdown("**📄 Resposta Bruta Retornada (Markdown):**")
            st.code(trace.get("answer", ""), language="markdown")

        with tab_sys:
            st.markdown(
                f"- **SDK Oficial:** `google-genai` (`genai.Client`)\n"
                f"- **Modelo de Geração:** `{DEFAULT_GENERATION_MODEL}` (`temperature=0.1`)\n"
                f"- **Modelo de Embeddings:** `{EMBEDDING_MODEL}` (`RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`)\n"
                f"- **Métrica Vetorial ChromaDB:** `cosine` + **Lexical:** `BM25Okapi` (Fusão `RRF k=60`)"
            )
            st.markdown("**📜 System Instruction Estruturado:**")
            st.code(
                trace.get("system_instruction", SYSTEM_INSTRUCTION), language="markdown"
            )


def _render_chat_page(engine: LGPDRagEngine) -> None:
    """Renderiza a página principal de Chat com linhas em ondas em degradê e saudação centralizada."""
    indexed_count = engine.get_indexed_count()
    jsonl_count = len(engine.chunks)

    # 1. Efeito Visual de Linhas em Ondas em Degradê + Saudação Centralizada
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
                Estou aqui para apoiar sua conformidade — de consultas rápidas de artigos da LGPD
                a pareceres completos com resoluções da ANPD e referências cruzadas.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Container Central Estilo "Pro" com Pílulas de Identificação do Processo
    auth_badge = "🟢 API Key Conectada" if st.session_state["gemini_api_key"] else "🟠 Aguardando API Key"
    st.markdown(
        f"""
        <div class="pro-shell">
            <div class="pro-header-bar">
                <div class="pro-header-left">
                    <span class="pro-spark">⚡</span>
                    <span>Motor RAG Híbrido Ativo com <strong>Gemini 3.8 Flash</strong></span>
                </div>
                <span style="font-size: 0.74rem; color: #38bdf8;">{auth_badge} • {indexed_count}/{jsonl_count} Chunks</span>
            </div>
            <div class="pro-inner-box">
                <div style="font-size: 0.83rem; color: #94a3b8; margin-bottom: 0.45rem;">
                    Pipeline de Conformidade Normativa (Auditoria & Grounding Estrito):
                </div>
                <div class="tool-pills-row">
                    <span class="tool-pill">🔐 BYOK Sessão</span>
                    <span class="tool-pill">🌐 SHA-256 Planalto & ANPD</span>
                    <span class="tool-pill">🧩 Chunking Semântico</span>
                    <span class="tool-pill">⚖️ Chroma Cosine + BM25 RRF</span>
                    <span class="tool-pill">🔗 Cross-Reference (Artigos)</span>
                    <span class="tool-pill">✨ {DEFAULT_GENERATION_MODEL}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Três Cards Minimalistas de Ação Rápida (mesma largura da estrutura central)
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

    # 4. Mais Sugestões Minimizadas
    with st.expander("💡 Mais Sugestões de Consulta (minimizadas)", expanded=False):
        s_cols = st.columns(2)
        for idx, sug in enumerate(EXTRA_SUGGESTIONS):
            if s_cols[idx % 2].button(
                f"↗ {sug}", key=f"extra_sug_{idx}", use_container_width=True
            ):
                st.session_state["pending_suggestion"] = sug

    if st.session_state["messages"]:
        st.divider()

    # 5. Estrutura de Conversa (Pergunta do Usuário -> Resposta do Chatbot + Contexto Completo)
    for msg in st.session_state["messages"]:
        avatar = "🧑‍💼" if msg["role"] == "user" else "🛡️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("retrieved_chunks") is not None:
                    _render_grounding_expander(msg["retrieved_chunks"])
                if msg.get("trace") is not None:
                    _render_request_response_inspector(msg)

    # 6. Barra de Digitação Fixa na Parte Inferior (com max-width 920px alinhada aos blocos)
    user_input = st.chat_input(
        "Pergunte qualquer coisa sobre a LGPD ou Resoluções da ANPD..."
    )
    active_question = user_input or st.session_state.get("pending_suggestion")
    if st.session_state.get("pending_suggestion"):
        st.session_state["pending_suggestion"] = None

    if active_question:
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(active_question)

        if not st.session_state["gemini_api_key"]:
            with st.chat_message("assistant", avatar="🛡️"):
                st.warning(
                    "🔑 **Autenticação Necessária:** Insira sua **Gemini API Key** na barra lateral esquerda para autenticar a sessão (BYOK)."
                )
            return

        if engine.get_indexed_count() == 0:
            with st.chat_message("assistant", avatar="🛡️"):
                st.warning(
                    "🗄️ **Base Vetorial Ainda Não Indexada:** Clique em **'🔄 Sincronizar e Indexar Base Oficial'** na barra lateral esquerda."
                )
            return

        with st.chat_message("assistant", avatar="🛡️"):
            with st.status(
                "⚡ Executando Pipeline RAG Híbrido & Cross-Reference...",
                expanded=True,
            ) as proc_status:
                try:
                    prior_history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state["messages"]
                    ]
                    st.write(
                        "🔎 **Etapa 1/3:** Analisando contexto conversacional e preparando query técnica..."
                    )
                    st.write(
                        "⚖️ **Etapa 2/3:** Buscando no `ChromaDB (Cosine)` + `BM25 (RRF)` e expandindo `artigos_citados`..."
                    )
                    st.write(
                        f"✨ **Etapa 3/3:** Gerando parecer fundamentado com `{DEFAULT_GENERATION_MODEL}` (`temperature=0.1`)..."
                    )

                    result = engine.answer_question(
                        question=active_question,
                        chat_history=prior_history,
                        model_name=DEFAULT_GENERATION_MODEL,
                        top_k=5,
                        max_cross_refs=2,
                        api_key=st.session_state["gemini_api_key"],
                    )

                    proc_status.update(
                        label=(
                            f"✅ Parecer Concluído ({result.get('hybrid_count', 0)} chunks híbridos + "
                            f"{result.get('cross_ref_count', 0)} referências cruzadas • {result.get('latency_ms', 0)} ms)"
                        ),
                        state="complete",
                        expanded=False,
                    )

                    answer_md = result["answer"]
                    retrieved = result["retrieved_chunks"]

                    st.markdown(answer_md)
                    _render_grounding_expander(retrieved)

                    msg_record = {
                        "role": "assistant",
                        "content": answer_md,
                        "retrieved_chunks": retrieved,
                        "trace": result,
                    }
                    _render_request_response_inspector(msg_record)

                    st.session_state["messages"].append(
                        {"role": "user", "content": active_question}
                    )
                    st.session_state["messages"].append(msg_record)

                except Exception as exc:
                    proc_status.update(
                        label="❌ Falha durante o processamento da consulta",
                        state="error",
                    )
                    st.error(f"Não foi possível concluir a consulta: {exc}")


def _render_project_details_page(engine: LGPDRagEngine) -> None:
    """Renderiza a página dedicada ao detalhamento arquitetural e engenharia do projeto."""
    st.markdown(
        """
        <div class="lix-greeting-wrap" style="margin-bottom: 1.2rem;">
            <div class="lix-hi">Engenharia de Dados & IA Generativa</div>
            <div class="lix-headline">Arquitetura & Construção do Projeto</div>
            <div class="lix-subtext">
                Conheça em detalhes como cada módulo do Assistente Inteligente de Conformidade LGPD & ANPD foi projetado e validado.
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
                <h4>🌐 1. Coleta Resiliente (<code>src/ingestion/fetcher.py</code>)</h4>
                <ul>
                    <li><strong>Conversão de Encoding:</strong> Coleta a LGPD Compilada do Planalto em <code>iso-8859-1</code> e converte para <code>UTF-8</code>.</li>
                    <li><strong>Integridade SHA-256:</strong> Gera manifesto de hashes criptográficos e só reescreve arquivos caso o conteúdo oficial mude.</li>
                    <li><strong>Fallback Local Obrigatório:</strong> Tolerante a bloqueios de WAF/SSL do <code>gov.br</code>, operando com contingência local automática em <code>data/raw/</code>.</li>
                </ul>
            </div>
            <div class="arch-card">
                <h4>⚖️ 3. Busca Híbrida RRF + Cross-Reference (<code>src/rag/engine.py</code>)</h4>
                <ul>
                    <li><strong>Reciprocal Rank Fusion (RRF):</strong> Combina busca vetorial no <code>ChromaDB</code> (métrica <code>cosine</code> com <code>RETRIEVAL_QUERY</code>) e busca lexical <code>BM25Okapi</code>.</li>
                    <li><strong>Expansão de Referências Cruzadas:</strong> Identifica os dispositivos em <code>artigos_citados</code> dos chunks Top-K e anexa automaticamente até 2 artigos complementares via filtro de metadado.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="arch-card">
                <h4>🧩 2. Parser Semântico-Estrutural (<code>src/ingestion/parser.py</code>)</h4>
                <ul>
                    <li><strong>Remoção de Trechos Revogados:</strong> Decompõe tags <code>&lt;strike&gt;</code>, <code>&lt;s&gt;</code>, <code>&lt;del&gt;</code> e estilos CSS <code>line-through</code>.</li>
                    <li><strong>Chunk Atômico por Artigo:</strong> Preserva <code>caput</code>, incisos, alíneas e parágrafos juntos (79 artigos: <code>Art. 1º</code> a <code>Art. 65</code> + <code>Art. 55-A</code> a <code>55-L</code> e <code>58-A/B</code>).</li>
                    <li><strong>Tabelas ANPD em Markdown:</strong> Extrai PDFs com <code>pymupdf4llm.to_markdown(page_chunks=True)</code> (~1800 caracteres, overlap 250).</li>
                </ul>
            </div>
            <div class="arch-card">
                <h4>✨ 4. Geração & Auditoria Transparente (<code>src/app.py</code>)</h4>
                <ul>
                    <li><strong>SDK Oficial Exclusivo:</strong> Usa <code>from google import genai</code> e <code>from google.genai import types</code> com autenticação BYOK em sessão.</li>
                    <li><strong>Modelo Único Padronizado:</strong> Configurado em <code>gemini-3.8-flash</code> (<code>temperature=0.1</code>) com failover resiliente contra picos 503.</li>
                    <li><strong>Inspeção Completa:</strong> Exibe em cada resposta o prompt enviado, a tag XML <code>&lt;contexto_normativo&gt;</code>, a latência e os scores de recuperação.</li>
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


def main() -> None:
    st.set_page_config(
        page_title="LGPD.AI — Assistente Inteligente LGPD & ANPD",
        page_icon="🛡️",
        layout="wide",
    )
    _inject_lix_inspired_css()
    _init_session_state()
    engine: LGPDRagEngine = st.session_state["rag_engine"]

    # ==================== BARRA LATERAL MINIMALISTA (ESTILO LIX) ====================
    with st.sidebar:
        st.markdown(
            '<div class="brand-logo"><span class="brand-dot"></span>LGPD.AI</div>',
            unsafe_allow_html=True,
        )
        st.caption("Conformidade LGPD & ANPD")

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
        st.markdown("##### 🔐 Autenticação BYOK")
        user_key = st.text_input(
            "🔑 Gemini API Key",
            value=st.session_state["gemini_api_key"],
            type="password",
            placeholder="AIzaSy...",
            help="Mantida exclusivamente em st.session_state durante sua sessão.",
        )
        st.session_state["gemini_api_key"] = user_key.strip()
        engine.api_key = st.session_state["gemini_api_key"]
        st.session_state["selected_model"] = DEFAULT_GENERATION_MODEL

        st.caption(
            f"✨ **Modelo:** `{DEFAULT_GENERATION_MODEL}` (`temp=0.1`)\n\n"
            f"🧠 **Embeddings:** `{EMBEDDING_MODEL}`"
        )

        st.divider()
        st.markdown("##### 🗄️ Base Normativa")
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
            if not st.session_state["gemini_api_key"]:
                st.error("Insira sua Gemini API Key acima antes de indexar.")
            else:
                with st.status(
                    "Sincronizando LGPD + Resoluções ANPD...", expanded=True
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

                    st.write(f"🧠 3/3 Indexando no ChromaDB (`{EMBEDDING_MODEL}`)...")
                    try:
                        total_indexed = engine.index_chunks(
                            api_key=st.session_state["gemini_api_key"]
                        )
                        status_box.update(
                            label=f"✅ {total_indexed} chunks indexados!",
                            state="complete",
                        )
                        st.rerun()
                    except Exception as exc:
                        status_box.update(
                            label="❌ Erro na indexação", state="error"
                        )
                        st.error(f"Falha ao gerar embeddings: {exc}")

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

    # ==================== ROTEAMENTO DE PÁGINAS ====================
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
