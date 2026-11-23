"""
Gerenciamento Blindado de Estado e Sincronização de Provedor/Modelo (src/ui/state.py)
Implementa:
- Detecção de ambiente Cloud vs. Local cacheada (`@st.cache_data(ttl=60)`).
- Inicialização e validação de `st.session_state` garantindo que em modo Cloud o Ollama nunca
  seja selecionado como padrão.
- Callbacks `on_provider_change` que redefinem automaticamente o modelo para o padrão válido
  do novo provedor ao alternar provedores, eliminando dessincronização entre `provider` e `model`.
"""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional

import streamlit as st

from src.rag.prompts import normalize_language_level
from src.rag.providers import (
    DEFAULT_OLLAMA_URL,
    MODEL_CATALOG,
    PROVIDER_DEFAULT_MODELS,
    PROVIDER_LABELS,
    detect_runtime_environment,
    get_available_providers,
    get_model_metadata,
    get_selectable_models,
    normalize_provider_key,
)


@st.cache_data(ttl=60, show_spinner=False)
def get_cached_runtime_environment(base_url: str = DEFAULT_OLLAMA_URL) -> Dict[str, Any]:
    """Detecta e cacheia por 60s o ambiente de execução (Cloud vs. Local e status do Ollama)."""
    return detect_runtime_environment(base_url=base_url)


def is_model_valid_for_provider(provider_key: str, model_id: str) -> bool:
    """
    Verifica se `model_id` pertence ao catálogo suportado de `provider_key`
    e não possui tier `unsupported`.
    """
    if not model_id:
        return False
    prov = normalize_provider_key(provider_key)
    selectable = get_selectable_models(prov, include_unsupported=False)
    if model_id in selectable:
        return True

    # Verifica se o modelo pertence explicitamente a outro provedor
    for other_prov, other_cat in MODEL_CATALOG.items():
        if other_prov != prov and model_id in other_cat:
            return False

    # No Ollama local, permite tags customizadas desde que não sejam do tier unsupported
    if prov == "ollama":
        meta = get_model_metadata("ollama", model_id)
        return meta.get("tier") != "unsupported"

    return False


def sync_provider_and_model_state(
    state: Optional[MutableMapping[str, Any]] = None,
    env_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Valida e sincroniza de forma blindada `provider_key` e `selected_model` no estado:
    1. Se estiver em modo Cloud (`ollama_online=False`) e `provider_key == 'ollama'`,
       faz fallback automático para o provedor em nuvem padrão do ambiente.
    2. Se `selected_model` não pertencer ao `provider_key` atual ou for `unsupported`,
       redefine para `PROVIDER_DEFAULT_MODELS[provider_key]`.
    """
    target_state: MutableMapping[str, Any] = state if state is not None else st.session_state
    base_url = str(target_state.get("ollama_base_url") or DEFAULT_OLLAMA_URL)
    runtime_env = env_info if env_info is not None else get_cached_runtime_environment(base_url)

    ollama_online = bool(runtime_env.get("ollama_online", False))
    available_providers = get_available_providers(ollama_online=ollama_online)
    default_provider = str(runtime_env.get("default_provider", "gemini"))
    if default_provider not in available_providers:
        default_provider = available_providers[0]

    raw_provider = str(target_state.get("provider_key") or default_provider)
    try:
        provider_key = normalize_provider_key(raw_provider)
    except ValueError:
        provider_key = default_provider

    # Em ambiente Cloud (sem Ollama local), nunca mantém 'ollama' ativo
    if provider_key not in available_providers:
        provider_key = default_provider

    target_state["provider_key"] = provider_key

    current_model = str(target_state.get("selected_model") or "").strip()
    if not is_model_valid_for_provider(provider_key, current_model):
        target_state["selected_model"] = PROVIDER_DEFAULT_MODELS[provider_key]

    target_state["language_level"] = normalize_language_level(
        str(target_state.get("language_level") or "equilibrado")
    )
    return runtime_env


def on_provider_change(
    state: Optional[MutableMapping[str, Any]] = None,
    new_provider: Optional[str] = None,
) -> None:
    """
    Callback acionado ao trocar o seletor de Provedor na Sidebar (`on_change`):
    Atualiza `provider_key` e redefine imediatamente `selected_model` para o modelo padrão
    válido do novo provedor, evitando qualquer erro de modelo incompatível.
    """
    target_state: MutableMapping[str, Any] = state if state is not None else st.session_state
    label_or_key = (
        new_provider
        or target_state.get("sidebar_provider_select")
        or target_state.get("provider_key")
        or "gemini"
    )
    try:
        prov_key = normalize_provider_key(str(label_or_key))
    except ValueError:
        prov_key = "gemini"

    target_state["provider_key"] = prov_key
    target_state["selected_model"] = PROVIDER_DEFAULT_MODELS[prov_key]
    target_state["sidebar_model_select"] = PROVIDER_DEFAULT_MODELS[prov_key]


def on_model_change(
    state: Optional[MutableMapping[str, Any]] = None,
    new_model: Optional[str] = None,
) -> None:
    """Callback acionado ao trocar o modelo selecionado na Sidebar."""
    target_state: MutableMapping[str, Any] = state if state is not None else st.session_state
    prov_key = str(target_state.get("provider_key") or "gemini")
    candidate = str(new_model or target_state.get("sidebar_model_select") or "").strip()
    if is_model_valid_for_provider(prov_key, candidate):
        target_state["selected_model"] = candidate
    else:
        target_state["selected_model"] = PROVIDER_DEFAULT_MODELS[prov_key]


def init_session_state(
    state: Optional[MutableMapping[str, Any]] = None,
    env_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Inicializa todas as chaves obrigatórias em `st.session_state` e sincroniza provedor/modelo."""
    target_state: MutableMapping[str, Any] = state if state is not None else st.session_state

    if "ollama_base_url" not in target_state:
        target_state["ollama_base_url"] = DEFAULT_OLLAMA_URL

    runtime_env = env_info if env_info is not None else get_cached_runtime_environment(
        str(target_state["ollama_base_url"])
    )

    if "provider_key" not in target_state:
        target_state["provider_key"] = runtime_env["default_provider"]
    if "selected_model" not in target_state:
        target_state["selected_model"] = PROVIDER_DEFAULT_MODELS[target_state["provider_key"]]
    if "language_level" not in target_state:
        target_state["language_level"] = "equilibrado"
    if "mask_pii_enabled" not in target_state:
        target_state["mask_pii_enabled"] = True
    if "custom_ollama_tag" not in target_state:
        target_state["custom_ollama_tag"] = ""
    if "api_keys" not in target_state:
        target_state["api_keys"] = {
            "gemini": "",
            "openai": "",
            "anthropic": "",
        }
    if "gemini_api_key" not in target_state:
        target_state["gemini_api_key"] = ""
    if "messages" not in target_state:
        target_state["messages"] = []
    if "pending_suggestion" not in target_state:
        target_state["pending_suggestion"] = None
    if "demo_message_count" not in target_state:
        target_state["demo_message_count"] = 0

    return sync_provider_and_model_state(state=target_state, env_info=runtime_env)
