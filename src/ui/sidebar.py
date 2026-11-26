"""
Barra Superior Minimalista e Modal Central de Configurações em Split-View (src/ui/sidebar.py)
Implementa:
- Cabeçalho superior limpo com marca `LGPD.AI`, pílula discreta de convergência e 2 ícones minimalistas
  brancos no canto superior direito:
  1. `ⓘ` (Informações) -> alterna para a página `📐 Arquitetura & Projeto` (e volta ao Chat).
  2. `⚙` (Configurações) -> abre o modal centralizado (`@st.dialog`) em formato Split-View:
     - Lado esquerdo: retângulos clicáveis para cada setor de configuração (com o setor ativo em destaque, sem caixas de seleção/radio).
     - Lado direito: retângulos em destaque para o Modo de Operação e Nível de Linguagem, diagnóstico de convergência
       Modo <-> Capacidade do Modelo, chaves de API (BYOK), Privacy by Design (PII) e sincronização da base legal.
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
    OPERATING_MODE_PROFILES,
    PROVIDER_LABELS,
    get_available_providers,
    get_model_metadata,
    get_model_performance_context,
    get_models_for_operating_mode,
    get_provider_api_key,
    normalize_provider_key,
)
from src.ui.components import format_tier_badge
from src.ui.state import (
    on_model_change,
    on_operating_mode_change,
    on_provider_change,
    sync_provider_and_model_state,
)

SETTINGS_CATEGORIES: List[Dict[str, str]] = [
    {"id": "mode_model", "title": "Modo & Modelo de IA"},
    {"id": "api_keys", "title": "Chaves & Endpoints"},
    {"id": "lang_privacy", "title": "Linguagem & Privacidade"},
    {"id": "knowledge_base", "title": "Base Normativa & Sessão"},
]


def _clean_mode_label(raw_label: str) -> str:
    """Remove emojis iniciais dos rótulos de modo para manter a estética minimalista."""
    for prefix in ("⚡ ", "🎯 ", "🖥️ "):
        if raw_label.startswith(prefix):
            return raw_label[len(prefix) :]
    return raw_label


def _render_convergence_panel(perf_ctx: Dict[str, Any]) -> None:
    """Renderiza o card minimalista de convergência entre a capacidade do modelo e o contexto de performance."""
    ctx_win_k = int(perf_ctx.get("context_window", 128000) / 1000)
    clean_short_mode = _clean_mode_label(str(perf_ctx.get("mode_short_label", "")))
    html = (
        '<div class="convergence-card">'
        f'<div class="convergence-title">Convergência Ativa: {clean_short_mode} ↔ {perf_ctx["model_id"]}</div>'
        f'<div class="convergence-desc">{perf_ctx["fit_reason"]}</div>'
        '<div class="convergence-metrics">'
        f'<div class="convergence-metric-item"><span class="convergence-metric-label">Perfil de Latência</span><span class="convergence-metric-val">{perf_ctx["speed_class"]}</span></div>'
        f'<div class="convergence-metric-item"><span class="convergence-metric-label">Orçamento RAG</span><span class="convergence-metric-val">Top-{perf_ctx["top_k"]} · {perf_ctx["max_context_tokens"]} tok</span></div>'
        f'<div class="convergence-metric-item"><span class="convergence-metric-label">Histórico · Janela</span><span class="convergence-metric-val">{perf_ctx["max_history_turns"]} turnos · {ctx_win_k}k</span></div>'
        "</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_mode_and_model_details(ollama_online: bool) -> None:
    """Painel direito do setor 1: Convergência Modo de Operação <-> Provedor & Capacidade do Modelo."""
    st.markdown(
        '<div class="settings-right-header">Contexto de Performance &amp; Convergência de Modelo</div>',
        unsafe_allow_html=True,
    )

    sync_provider_and_model_state()
    mode_keys = ["fast_eco", "high_precision"] + (["local_offline"] if ollama_online else [])
    curr_mode = str(st.session_state.get("preset_mode") or "fast_eco")
    if curr_mode not in mode_keys:
        curr_mode = mode_keys[0]

    st.caption("1. Selecione o Contexto de Operação (o retângulo em destaque indica o modo ativo):")
    mode_cols = st.columns(len(mode_keys))
    for idx, m_key in enumerate(mode_keys):
        prof = OPERATING_MODE_PROFILES[m_key]
        clean_short = _clean_mode_label(prof["short_label"])
        is_selected_mode = m_key == curr_mode
        with mode_cols[idx]:
            if st.button(
                clean_short,
                key=f"mode_rect_btn_{m_key}",
                type="primary" if is_selected_mode else "secondary",
                use_container_width=True,
                help=prof["description"],
            ):
                if m_key != curr_mode:
                    on_operating_mode_change(new_mode=m_key, ollama_online=ollama_online)
                    st.rerun()

    active_mode_key = str(st.session_state.get("preset_mode") or "fast_eco")
    if active_mode_key in OPERATING_MODE_PROFILES:
        st.caption(OPERATING_MODE_PROFILES[active_mode_key]["description"])

    available_keys = get_available_providers(ollama_online=ollama_online)
    current_prov = st.session_state["provider_key"]
    if current_prov not in available_keys:
        current_prov = available_keys[0]
        st.session_state["provider_key"] = current_prov

    prov_labels = [PROVIDER_LABELS[k] for k in available_keys]
    curr_prov_idx = available_keys.index(current_prov)

    col_p, col_m = st.columns(2)
    with col_p:
        selected_prov_label = st.selectbox(
            "2. Provedor de IA",
            options=prov_labels,
            index=curr_prov_idx,
        )
        provider_key = normalize_provider_key(selected_prov_label)
        if provider_key != st.session_state["provider_key"]:
            on_provider_change(new_provider=provider_key)
            st.rerun()

    provider_key = st.session_state["provider_key"]
    converged_models = get_models_for_operating_mode(
        provider_key,
        operating_mode=st.session_state.get("preset_mode", "fast_eco"),
    )
    current_model = st.session_state.get("selected_model", "")
    if current_model not in converged_models and converged_models:
        current_model = converged_models[0]
        st.session_state["selected_model"] = current_model

    model_idx = converged_models.index(current_model) if current_model in converged_models else 0
    target_tier = OPERATING_MODE_PROFILES.get(
        st.session_state.get("preset_mode", "fast_eco"), {}
    ).get("target_tier")

    with col_m:
        selected_model = st.selectbox(
            "3. Modelo Convergente",
            options=converged_models,
            index=model_idx,
            format_func=lambda m: (
                f"{m} · Recomendado no modo"
                if get_model_metadata(provider_key, m).get("tier") == target_tier
                else m
            ),
        )
        if selected_model != st.session_state.get("selected_model"):
            on_model_change(new_model=selected_model)
            st.rerun()

    if not ollama_online:
        st.caption("Nota: O modo Local / Ollama é habilitado automaticamente em execução local.")

    perf_ctx = get_model_performance_context(
        provider_id=st.session_state["provider_key"],
        model_id=st.session_state["selected_model"],
        operating_mode=st.session_state.get("preset_mode"),
    )
    st.session_state["performance_context"] = perf_ctx
    _render_convergence_panel(perf_ctx)


def _render_api_keys_and_endpoints_details() -> None:
    """Painel direito do setor 2: Chaves de API (BYOK) e Endpoints."""
    st.markdown(
        '<div class="settings-right-header">Autenticação BYOK &amp; Endpoints</div>',
        unsafe_allow_html=True,
    )
    provider_key = st.session_state.get("provider_key", "gemini")

    if provider_key == "ollama":
        st.info("Provedor Local Ativo (Qwen via Ollama): Nenhuma chave de API em nuvem é necessária.")
        st.session_state["ollama_base_url"] = (
            st.text_input(
                "URL do Servidor Ollama Local",
                value=st.session_state.get("ollama_base_url", DEFAULT_OLLAMA_URL),
                help="Endereço padrão do daemon local do Ollama.",
            ).strip()
            or DEFAULT_OLLAMA_URL
        )
    else:
        placeholder_map = {
            "gemini": "AIzaSy...",
            "groq": "gsk_...",
            "openrouter": "sk-or-...",
            "openai": "sk-...",
            "anthropic": "sk-ant-...",
        }
        user_saved_key = st.session_state["api_keys"].get(provider_key, "")
        system_key = get_provider_api_key(provider_key, user_api_key=None)

        if system_key and not user_saved_key:
            st.success(
                f"Chave do ambiente (st.secrets) ativa para {PROVIDER_LABELS[provider_key]}."
            )

        typed_key = st.text_input(
            f"API Key — {PROVIDER_LABELS[provider_key]} (BYOK)",
            value=user_saved_key,
            type="password",
            placeholder=placeholder_map.get(provider_key, "Insira sua API Key..."),
            help="Sua chave permanece exclusivamente em memória durante a sessão e passa por validação ASCII automática.",
        ).strip()
        st.session_state["api_keys"][provider_key] = typed_key
        if provider_key == "gemini":
            st.session_state["gemini_api_key"] = typed_key

        effective_key = get_provider_api_key(provider_key, user_api_key=typed_key)
        if effective_key:
            st.caption("Status: Chave configurada e validada para consultas.")
        else:
            st.warning(
                f"Insira sua chave para {PROVIDER_LABELS[provider_key]} acima para liberar o envio de perguntas."
            )

    with st.expander("Gerenciar chaves de outros provedores", expanded=False):
        for pk, plabel in PROVIDER_LABELS.items():
            if pk in ("ollama", provider_key):
                continue
            val = st.text_input(
                f"API Key — {plabel}",
                value=st.session_state["api_keys"].get(pk, ""),
                type="password",
                key=f"modal_extra_key_{pk}",
            ).strip()
            st.session_state["api_keys"][pk] = val
            if pk == "gemini":
                st.session_state["gemini_api_key"] = val


def _render_language_and_privacy_details() -> None:
    """Painel direito do setor 3: Nível de Linguagem (em retângulos destacados) e Privacy by Design (PII)."""
    st.markdown(
        '<div class="settings-right-header">Nível de Linguagem &amp; Privacy by Design</div>',
        unsafe_allow_html=True,
    )

    level_keys = list(LANGUAGE_LEVEL_LABELS.keys())
    curr_level = normalize_language_level(st.session_state.get("language_level", "equilibrado"))
    if curr_level not in level_keys:
        curr_level = "equilibrado"

    st.caption("Estilo e Profundidade das Respostas (retângulo em destaque indica o estilo ativo):")
    lang_cols = st.columns(len(level_keys))
    for idx, l_key in enumerate(level_keys):
        l_label = LANGUAGE_LEVEL_LABELS[l_key]
        is_active_lang = l_key == curr_level
        with lang_cols[idx]:
            if st.button(
                l_label,
                key=f"lang_rect_btn_{l_key}",
                type="primary" if is_active_lang else "secondary",
                use_container_width=True,
            ):
                if l_key != curr_level:
                    st.session_state["language_level"] = l_key
                    st.rerun()

    level_descriptions = {
        "simples": "Simples (Cidadão): linguagem cotidiana acessível, explicando conceitos sem jargão técnico.",
        "equilibrado": "Equilibrado (Padrão): linguagem executiva e direta para equipes de tecnologia, produto e negócios.",
        "tecnico": "Técnico / Jurídico: vocabulário normativo rigoroso com citação de incisos, parágrafos e dosimetria da ANPD.",
    }
    st.caption(level_descriptions.get(curr_level, ""))

    st.divider()
    st.markdown("**Proteção de Dados Pessoais no Prompt (Privacy by Design — Art. 46 LGPD)**")
    st.session_state["mask_pii_enabled"] = st.toggle(
        "Mascarar CPF, CNPJ e E-mail automaticamente antes do envio",
        value=bool(st.session_state.get("mask_pii_enabled", True)),
        help="Intercepta localmente números de CPF, CNPJ e endereços de e-mail e substitui por tags neutras.",
    )
    st.caption(
        "Quando ativo, 123.456.789-00 → [CPF_MASCARADO], 12.345.678/0001-99 → [CNPJ_MASCARADO] "
        "e nome@empresa.com → [EMAIL_MASCARADO] antes de qualquer chamada externa."
    )


def _render_knowledge_base_and_session_details(engine: LGPDRagEngine) -> None:
    """Painel direito do setor 4: Base Normativa Local e Controle da Sessão."""
    st.markdown(
        '<div class="settings-right-header">Base Normativa Local &amp; Sessão</div>',
        unsafe_allow_html=True,
    )

    indexed_count = engine.get_indexed_count()
    jsonl_count = len(engine.chunks)
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("ChromaDB (Vetores)", indexed_count)
    col_m2.metric("Chunks Hierárquicos", jsonl_count)
    col_m3.metric("Artigos LGPD", len(engine.chunks_by_article))

    if st.button(
        "Sincronizar e Reindexar Base Oficial (LGPD + ANPD)",
        use_container_width=True,
        type="primary",
    ):
        with st.status(
            "Sincronizando LGPD + Resoluções ANPD (100% Local)...", expanded=True
        ) as status_box:
            st.write("1/3 Coleta com validação %PDF-, SHA-256 e fallback local...")
            fetch_report = run_ingestion_fetcher(raw_dir=DEFAULT_RAW_DIR)
            if fetch_report["overall_status"] != "ok":
                st.warning("Contingência local validada utilizada para arquivos gov.br.")

            st.write("2/3 Executando Parser Semântico-Estrutural Hierárquico...")
            chunks = run_parser_pipeline(
                raw_dir=DEFAULT_RAW_DIR, output_jsonl=DEFAULT_OUTPUT_JSONL
            )
            st.write(f"Extraídos {len(chunks)} chunks normativos.")

            st.write(f"3/3 Indexando localmente com {LOCAL_EMBEDDING_MODEL} + SQLite FTS5...")
            try:
                total_indexed = engine.index_chunks()
                status_box.update(
                    label=f"Concluído: {total_indexed} chunks indexados localmente.",
                    state="complete",
                )
                st.rerun()
            except Exception as exc:
                status_box.update(label="Erro na indexação", state="error")
                st.error(f"Falha ao indexar base: {exc}")

    uploaded_files = st.file_uploader(
        "Importar novos PDFs de Resoluções ou Guias da ANPD (.pdf)",
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
        st.success(f"Salvos em data/raw/: {', '.join(saved_names)}. Clique em Sincronizar acima.")

    st.divider()
    if st.button("Limpar Histórico da Conversa Atual", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["pending_suggestion"] = None
        st.rerun()


@st.dialog("Configurações", width="large")
def open_settings_dialog(engine: LGPDRagEngine, runtime_env: Dict[str, Any]) -> None:
    """
    Modal centralizado em formato Split-View:
    - Coluna esquerda: lista de retângulos clicáveis para cada setor de configuração
      (o retângulo do setor ativo fica em destaque, sem caixas de seleção/radio).
    - Coluna direita: controles detalhados do setor selecionado.
    """
    ollama_online = bool(runtime_env.get("ollama_online", False))
    sync_provider_and_model_state()

    valid_ids = [c["id"] for c in SETTINGS_CATEGORIES]
    curr_cat_id = str(st.session_state.get("settings_category_id") or "mode_model")
    if curr_cat_id not in valid_ids:
        curr_cat_id = "mode_model"
        st.session_state["settings_category_id"] = curr_cat_id

    col_left, col_right = st.columns([1.05, 2.3], gap="large")

    with col_left:
        st.markdown('<div class="settings-left-title">Setores</div>', unsafe_allow_html=True)
        for cat in SETTINGS_CATEGORIES:
            cid = cat["id"]
            is_active = cid == curr_cat_id
            if st.button(
                cat["title"],
                key=f"settings_nav_rect_{cid}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                if cid != curr_cat_id:
                    st.session_state["settings_category_id"] = cid
                    st.rerun()

        st.divider()
        perf_ctx = get_model_performance_context(
            provider_id=st.session_state["provider_key"],
            model_id=st.session_state["selected_model"],
            operating_mode=st.session_state.get("preset_mode"),
        )
        clean_mode = _clean_mode_label(str(perf_ctx.get("mode_short_label", "")))
        st.caption(
            f"**Perfil Ativo**\n\n"
            f"Modo: {clean_mode}\n\n"
            f"Modelo: `{perf_ctx['model_id']}`\n\n"
            f"Contexto: Top-{perf_ctx['top_k']} · {perf_ctx['max_context_tokens']} tok"
        )

    with col_right:
        if curr_cat_id == "mode_model":
            _render_mode_and_model_details(ollama_online=ollama_online)
        elif curr_cat_id == "api_keys":
            _render_api_keys_and_endpoints_details()
        elif curr_cat_id == "lang_privacy":
            _render_language_and_privacy_details()
        else:
            _render_knowledge_base_and_session_details(engine=engine)


def render_top_navbar(engine: LGPDRagEngine, runtime_env: Dict[str, Any]) -> str:
    """
    Renderiza a barra superior minimalista:
    - Esquerda: logo `LGPD.AI` + pílula sóbria do modelo/modo convergente.
    - Canto superior direito: 2 ícones minimalistas brancos lado a lado:
      1. `ⓘ` (Informações de Arquitetura & Projeto — alterna entre Chat e Arquitetura)
      2. `⚙` (Configurações — abre o modal centralizado Split-View)
    """
    sync_provider_and_model_state()
    perf_ctx = get_model_performance_context(
        provider_id=st.session_state["provider_key"],
        model_id=st.session_state["selected_model"],
        operating_mode=st.session_state.get("preset_mode"),
    )
    st.session_state["performance_context"] = perf_ctx

    active_page = st.session_state.get("active_page", "🏠 Início & Chat Consultivo")
    is_arch_page = active_page == "📐 Arquitetura & Projeto"
    clean_mode = _clean_mode_label(str(perf_ctx.get("mode_short_label", "")))

    col_brand, col_info, col_settings = st.columns([9.6, 0.7, 0.7], vertical_alignment="center")

    with col_brand:
        st.markdown(
            f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:0.35rem;">'
            f'<div class="brand-logo"><span class="brand-dot"></span>LGPD.AI</div>'
            f'<span class="top-status-pill">'
            f'<span class="pill-dot"></span>'
            f'<span class="pill-mode">{clean_mode}</span> · '
            f'<code>{perf_ctx["model_id"]}</code> · '
            f'Top-{perf_ctx["top_k"]}'
            f'</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_info:
        info_help = (
            "Voltar ao Chat Consultivo"
            if is_arch_page
            else "Informações da Arquitetura & Detalhes do Projeto"
        )
        if st.button(
            "ⓘ",
            key="top_nav_info_btn",
            help=info_help,
        ):
            st.session_state["active_page"] = (
                "🏠 Início & Chat Consultivo"
                if is_arch_page
                else "📐 Arquitetura & Projeto"
            )
            st.rerun()

    with col_settings:
        if st.button(
            "⚙",
            key="top_nav_settings_btn",
            help="Abrir Configurações do Sistema",
        ):
            open_settings_dialog(engine=engine, runtime_env=runtime_env)

    return st.session_state.get("active_page", "🏠 Início & Chat Consultivo")


# Alias retrocompatível
render_sidebar = render_top_navbar
