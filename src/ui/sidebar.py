"""
Barra Lateral de Configuração Multi-Provedor, Nível de Linguagem e Base Normativa (src/ui/sidebar.py)
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from src.ingestion.fetcher import DEFAULT_RAW_DIR, run_ingestion_fetcher
from src.ingestion.parser import DEFAULT_OUTPUT_JSONL, run_parser_pipeline
from src.rag.engine import LOCAL_EMBEDDING_MODEL, LGPDRagEngine
from src.rag.prompts import LANGUAGE_LEVEL_LABELS, normalize_language_level
from src.rag.providers import (
    DEFAULT_OLLAMA_URL,
    PROVIDER_LABELS,
    get_available_providers,
    get_model_metadata,
    get_provider_api_key,
    get_selectable_models,
    normalize_provider_key,
)
from src.ui.components import format_tier_badge
from src.ui.state import (
    on_model_change,
    on_provider_change,
    sync_provider_and_model_state,
)


def render_sidebar(engine: LGPDRagEngine, runtime_env: Dict[str, Any]) -> str:
    """Renderiza a barra lateral completa e retorna a página selecionada."""
    with st.sidebar:
        st.markdown(
            '<div class="brand-logo"><span class="brand-dot"></span>LGPD.AI</div>',
            unsafe_allow_html=True,
        )
        env_mode = runtime_env.get("mode", "cloud")
        ollama_online = bool(runtime_env.get("ollama_online", False))
        env_badge = (
            "🖥️ Ambiente: Local (Ollama Ativo)"
            if env_mode == "local"
            else "☁️ Ambiente: Cloud / Web"
        )
        st.caption(f"Conformidade LGPD & ANPD • {env_badge}")

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
        _render_provider_and_model_controls(ollama_online=ollama_online)

        st.divider()
        _render_language_and_privacy_controls()

        st.divider()
        _render_knowledge_base_controls(engine)

        if st.button("🗑️ Limpar Conversa", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["pending_suggestion"] = None
            st.rerun()

    return page


def _render_provider_and_model_controls(ollama_online: bool) -> None:
    """Renderiza seletores sincronizados de Provedor, Modelo (com badge de Tier) e API Key."""
    st.markdown("##### 🤖 Provedor e Modelo de IA")

    available_keys = get_available_providers(ollama_online=ollama_online)
    sync_provider_and_model_state()
    current_prov = st.session_state["provider_key"]
    if current_prov not in available_keys:
        current_prov = available_keys[0]
        st.session_state["provider_key"] = current_prov

    prov_labels = [PROVIDER_LABELS[k] for k in available_keys]
    curr_prov_idx = available_keys.index(current_prov)

    selected_prov_label = st.selectbox(
        "Provedor de IA",
        options=prov_labels,
        index=curr_prov_idx,
        key="sidebar_provider_select",
        on_change=on_provider_change,
    )
    provider_key = normalize_provider_key(selected_prov_label)
    if provider_key != st.session_state["provider_key"]:
        on_provider_change(new_provider=provider_key)

    if not ollama_online:
        st.caption("ℹ️ *Qwen Local (Ollama): Disponível apenas em execução local.*")

    selectable_models = get_selectable_models(provider_key, include_unsupported=False)
    current_model = st.session_state.get("selected_model", "")
    if current_model not in selectable_models and selectable_models:
        current_model = selectable_models[0]
        st.session_state["selected_model"] = current_model

    model_idx = selectable_models.index(current_model) if current_model in selectable_models else 0
    selected_model = st.selectbox(
        "Modelo",
        options=selectable_models,
        index=model_idx,
        key="sidebar_model_select",
        on_change=on_model_change,
    )
    st.session_state["selected_model"] = selected_model

    model_meta = get_model_metadata(provider_key, selected_model)
    tier = str(model_meta.get("tier", "recommended"))
    st.caption(
        f"{format_tier_badge(tier)} • "
        f"Top-{model_meta.get('top_k', 4)} trechos • "
        f"Máx {model_meta.get('max_context_tokens', 2200)} tokens"
    )
    if tier == "economic":
        st.info(
            "⚡ **Modo Econômico ativo:** O contexto RAG foi otimizado automaticamente "
            f"(`top_k={model_meta.get('top_k', 3)}`, teto de `{model_meta.get('max_context_tokens', 1200)}` tokens) "
            "para evitar truncamento ou estouro de limite neste modelo."
        )

    # Controles específicos de autenticação / endpoint
    if provider_key == "ollama":
        with st.expander("⚙️ Endpoint Ollama Local", expanded=False):
            st.session_state["ollama_base_url"] = (
                st.text_input(
                    "URL do Servidor Ollama",
                    value=st.session_state.get("ollama_base_url", DEFAULT_OLLAMA_URL),
                ).strip()
                or DEFAULT_OLLAMA_URL
            )
    else:
        placeholder_map = {
            "gemini": "AIzaSy...",
            "openai": "sk-...",
            "anthropic": "sk-ant-...",
        }
        user_saved_key = st.session_state["api_keys"].get(provider_key, "")
        system_key = get_provider_api_key(provider_key, user_api_key=None)

        if system_key and not user_saved_key:
            st.success(
                f"✅ Chave de demonstração (`st.secrets`) ativa para **{PROVIDER_LABELS[provider_key]}**."
            )

        typed_key = st.text_input(
            f"🔑 API Key — {PROVIDER_LABELS[provider_key]} (BYOK)",
            value=user_saved_key,
            type="password",
            placeholder=placeholder_map.get(provider_key, "Insira sua API Key..."),
            help="Sua chave fica apenas em st.session_state durante a sessão e nunca é gravada em logs.",
        ).strip()
        st.session_state["api_keys"][provider_key] = typed_key
        if provider_key == "gemini":
            st.session_state["gemini_api_key"] = typed_key


def _render_language_and_privacy_controls() -> None:
    """Renderiza o seletor de Nível de Linguagem (Fase 6.2) e Privacy by Design / PII (Fase 7.2)."""
    st.markdown("##### 🗣️ Nível de Linguagem")
    level_keys = list(LANGUAGE_LEVEL_LABELS.keys())
    level_labels = [LANGUAGE_LEVEL_LABELS[k] for k in level_keys]
    curr_level = normalize_language_level(st.session_state.get("language_level", "equilibrado"))
    curr_idx = level_keys.index(curr_level) if curr_level in level_keys else 1

    chosen_label = st.radio(
        "Estilo da resposta",
        options=level_labels,
        index=curr_idx,
        help=(
            "Simples (Cidadão): linguagem cotidiana sem juridiquês. "
            "Equilibrado: foco em produto/TI/negócios. "
            "Técnico / Jurídico: vocabulário normativo e dosimetria."
        ),
    )
    st.session_state["language_level"] = normalize_language_level(chosen_label)

    st.markdown("##### 🛡️ Privacy by Design (PII)")
    st.session_state["mask_pii_enabled"] = st.toggle(
        "Mascarar CPF, CNPJ e E-mail automaticamente",
        value=bool(st.session_state.get("mask_pii_enabled", True)),
        help="Substitui CPF, CNPJ e E-mail por tags seguras antes de enviar a pergunta ao provedor de IA.",
    )
    st.caption("🔒 *Evite inserir dados pessoais reais de titulares no chat.*")


def _render_knowledge_base_controls(engine: LGPDRagEngine) -> None:
    """Renderiza métricas e botão de sincronização/indexação local da base normativa."""
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
            st.write("🌐 1/3 Coleta com validação `%PDF-`, SHA-256 e fallback local...")
            fetch_report = run_ingestion_fetcher(raw_dir=DEFAULT_RAW_DIR)
            if fetch_report["overall_status"] != "ok":
                st.warning("Contingência local validada utilizada para arquivos gov.br.")

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
                status_box.update(label="❌ Erro na indexação", state="error")
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
