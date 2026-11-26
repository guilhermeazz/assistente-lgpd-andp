"""
Ponto de Entrada Principal do Assistente Inteligente de Conformidade LGPD & ANPD (src/app.py)
Arquitetura modularizada em `src/ui/`:
- `src/ui/state.py` — inicialização, detecção Cloud vs. Local e sincronização de `st.session_state`
- `src/ui/sidebar.py` — seleção de provedor/modelo, API key, nível de linguagem e controle da base
- `src/ui/chat.py` — histórico, onboarding, indicador em 2 fases, streaming e telemetria
- `src/ui/components.py` — estilos CSS leves, badges de Tier e proteção Privacy by Design (PII)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VENV_SITE_PACKAGES = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from src.rag.engine import (
    FLASHRANK_CACHE_DIR,
    FLASHRANK_MODEL,
    LOCAL_EMBEDDING_MODEL,
    LGPDRagEngine,
    get_local_embedder,
    get_local_reranker,
)
from src.ui.architecture import render_project_details_page
from src.ui.chat import render_chat_page
from src.ui.components import inject_custom_css
from src.ui.sidebar import render_top_navbar
from src.ui.state import init_session_state


@st.cache_resource(show_spinner=False)
def get_cached_embedder() -> Any:
    """Cacheia o modelo SentenceTransformer local em memória para toda a sessão do servidor."""
    return get_local_embedder(LOCAL_EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_cached_reranker() -> Any:
    """Cacheia o objeto Ranker ONNX do FlashRank (`ms-marco-MultiBERT-L-12`) em memória."""
    return get_local_reranker(
        model_name=FLASHRANK_MODEL,
        cache_dir=FLASHRANK_CACHE_DIR,
    )


@st.cache_resource(show_spinner=False)
def get_cached_rag_engine() -> LGPDRagEngine:
    """
    Cacheia a instância do LGPDRagEngine (ChromaDB, SentenceTransformer, FlashRank e BM25)
    isolada do provedor de LLM selecionado na sessão.
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


def main() -> None:
    st.set_page_config(
        page_title="LGPD.AI — Assistente Inteligente LGPD & ANPD",
        page_icon="🛡️",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_custom_css()
    runtime_env = init_session_state()
    engine = get_cached_rag_engine()

    selected_page = render_top_navbar(engine=engine, runtime_env=runtime_env)
    if selected_page == "🏠 Início & Chat Consultivo":
        render_chat_page(engine=engine, runtime_env=runtime_env)
    else:
        render_project_details_page(engine=engine)


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
