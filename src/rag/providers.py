"""
Arquitetura Multi-Provedor LLM com Política de Tiers, Detecção Cloud/Local e Streaming (src/rag/providers.py)
Implementa:
- Catálogo rico de modelos com metadados de Tier (`recommended`, `economic`, `unsupported`),
  `context_window`, `max_context_tokens`, `top_k`, `max_history_turns` e `supports_streaming`.
- Detecção automática e rápida de ambiente (Cloud vs. Local) com timeout <= 0.6s.
- Validação prévia (Preflight Check) e tradução amigável de erros de API sem vazamento de chaves.
- Streaming real (`generate_stream(messages, **kwargs)` e `stream_generate(system_prompt, user_prompt)`)
  em todos os provedores: Google Gemini (`google-genai`), OpenAI, Anthropic Claude e Ollama Local.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import requests
from google import genai
from google.genai import types

try:
    import ollama as ollama_sdk
except ImportError:
    ollama_sdk = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.1

PROVIDER_LABELS: Dict[str, str] = {
    "gemini": "Google Gemini (Flash / Cloud)",
    "openai": "OpenAI (ChatGPT)",
    "anthropic": "Anthropic (Claude)",
    "ollama": "Qwen Local (Ollama)",
}

# Políticas padrão por Tier (Fase 3)
TIER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "recommended": {
        "tier": "recommended",
        "context_window": 128000,
        "max_context_tokens": 2200,
        "top_k": 5,
        "max_history_turns": 3,
        "supports_streaming": True,
    },
    "economic": {
        "tier": "economic",
        "context_window": 32000,
        "max_context_tokens": 1200,
        "top_k": 3,
        "max_history_turns": 1,
        "supports_streaming": True,
    },
    "unsupported": {
        "tier": "unsupported",
        "context_window": 2048,
        "max_context_tokens": 0,
        "top_k": 0,
        "max_history_turns": 0,
        "supports_streaming": False,
    },
}

# Catálogo completo de modelos por provedor com metadados explícitos (Fase 3)
MODEL_CATALOG: Dict[str, Dict[str, Dict[str, Any]]] = {
    "gemini": {
        "gemini-3.8-flash": {
            **TIER_DEFAULTS["recommended"],
            "label": "Gemini 3.8 Flash (Recomendado / Baixa Latência)",
            "context_window": 1000000,
            "selectable": True,
        },
        # Mantidos no catálogo apenas para retrocompatibilidade de metadados/testes (ocultos na UI pois retornam 404/503 na API atual)
        "gemini-2.5-flash": {
            **TIER_DEFAULTS["recommended"],
            "label": "Gemini 2.5 Flash (Legado -> Redirecionado para 3.8 Flash)",
            "context_window": 1000000,
            "selectable": False,
            "legacy_alias_for": "gemini-3.8-flash",
        },
        "gemini-2.5-flash-lite": {
            **TIER_DEFAULTS["economic"],
            "label": "Gemini 2.5 Flash-Lite (Legado -> Redirecionado para 3.8 Flash)",
            "context_window": 1000000,
            "selectable": False,
            "legacy_alias_for": "gemini-3.8-flash",
        },
        "gemini-nano-legacy": {
            **TIER_DEFAULTS["unsupported"],
            "label": "Gemini Nano Legacy (Não Suportado)",
            "context_window": 2048,
            "selectable": False,
        },
    },
    "openai": {
        "gpt-4o-mini": {
            **TIER_DEFAULTS["economic"],
            "label": "GPT-4o Mini (Econômico / Rápido)",
            "context_window": 128000,
        },
        "gpt-4o": {
            **TIER_DEFAULTS["recommended"],
            "label": "GPT-4o (Recomendado)",
            "context_window": 128000,
        },
        "o3-mini": {
            **TIER_DEFAULTS["economic"],
            "label": "o3-mini (Econômico)",
            "context_window": 128000,
        },
        "gpt-3.5-turbo-instruct": {
            **TIER_DEFAULTS["unsupported"],
            "label": "GPT-3.5 Turbo Instruct (Não Suportado)",
            "context_window": 4096,
        },
    },
    "anthropic": {
        "claude-3-5-haiku-latest": {
            **TIER_DEFAULTS["economic"],
            "label": "Claude 3.5 Haiku (Econômico / Rápido)",
            "context_window": 200000,
        },
        "claude-sonnet-4-20250514": {
            **TIER_DEFAULTS["recommended"],
            "label": "Claude Sonnet 4 (Recomendado)",
            "context_window": 200000,
        },
        "claude-instant-1.2": {
            **TIER_DEFAULTS["unsupported"],
            "label": "Claude Instant 1.2 (Não Suportado)",
            "context_window": 4096,
        },
    },
    "ollama": {
        "qwen2.5:7b": {
            **TIER_DEFAULTS["recommended"],
            "label": "Qwen 2.5 7B Local (Recomendado)",
            "context_window": 32768,
        },
        "qwen2.5:14b": {
            **TIER_DEFAULTS["recommended"],
            "label": "Qwen 2.5 14B Local (Recomendado)",
            "context_window": 32768,
        },
        "qwen2.5:3b": {
            **TIER_DEFAULTS["economic"],
            "label": "Qwen 2.5 3B Local (Econômico / Ultra-Rápido)",
            "context_window": 16384,
        },
        "qwen2.5:0.5b": {
            **TIER_DEFAULTS["unsupported"],
            "label": "Qwen 2.5 0.5B Local (Não Suportado)",
            "context_window": 2048,
        },
        "tinyllama:1.1b": {
            **TIER_DEFAULTS["unsupported"],
            "label": "TinyLlama 1.1B (Não Suportado)",
            "context_window": 2048,
        },
    },
}


def normalize_provider_key(provider_name: str) -> str:
    """Normaliza o identificador ou rótulo da UI para a chave canônica do provedor."""
    cleaned = (provider_name or "gemini").strip().lower()
    if any(k in cleaned for k in ("ollama", "local", "qwen")):
        return "ollama"
    if any(k in cleaned for k in ("gemini", "google")):
        return "gemini"
    if any(k in cleaned for k in ("openai", "chatgpt", "gpt")):
        return "openai"
    if any(k in cleaned for k in ("anthropic", "claude")):
        return "anthropic"
    raise ValueError(
        f"Provedor desconhecido: '{provider_name}'. Opções válidas: gemini, openai, anthropic, ollama."
    )


def get_selectable_models(provider_id: str, include_unsupported: bool = False) -> List[str]:
    """Retorna apenas os modelos ativos e selecionáveis (`recommended` e `economic`) de um provedor."""
    key = normalize_provider_key(provider_id)
    catalog = MODEL_CATALOG.get(key, {})
    return [
        m_id
        for m_id, meta in catalog.items()
        if (include_unsupported or (meta.get("tier") != "unsupported" and meta.get("selectable", True)))
    ]


GEMINI_MODELS: List[str] = get_selectable_models("gemini")
OPENAI_MODELS: List[str] = get_selectable_models("openai")
ANTHROPIC_MODELS: List[str] = get_selectable_models("anthropic")
OLLAMA_MODELS: List[str] = get_selectable_models("ollama")

PROVIDER_DEFAULT_MODELS: Dict[str, str] = {
    "gemini": "gemini-3.8-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "ollama": "qwen2.5:7b",
}


def get_model_metadata(provider_id: str, model_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Retorna os metadados de Tier e limites de tokens (`tier`, `context_window`,
    `max_context_tokens`, `top_k`, `max_history_turns`, `supports_streaming`) do modelo.
    """
    try:
        key = normalize_provider_key(provider_id)
    except ValueError:
        key = (provider_id or "base").strip().lower()
    resolved_model = (model_id or PROVIDER_DEFAULT_MODELS.get(key, "")).strip()
    provider_cat = MODEL_CATALOG.get(key, {})

    if resolved_model in provider_cat:
        return dict(provider_cat[resolved_model])

    # Heurística para tags customizadas (ex: "qwen3:8b", "qwen2.5:0.5b")
    lower_m = resolved_model.lower()
    if any(u in lower_m for u in ("0.5b", "1b", "1.1b", "nano", "tiny", "instruct-legacy", "unsupported")):
        return {**TIER_DEFAULTS["unsupported"], "label": resolved_model}
    if any(e in lower_m for e in ("3b", "mini", "haiku", "lite", "7b-instruct", "2.0-flash")):
        return {**TIER_DEFAULTS["economic"], "label": resolved_model}

    return {**TIER_DEFAULTS["recommended"], "label": resolved_model}


def _read_secret_or_env(*keys: str) -> Optional[str]:
    """Lê uma chave primeiro de variáveis de ambiente e depois de `st.secrets` sem lançar erro."""
    for k in keys:
        val = os.environ.get(k, "").strip()
        if val:
            return val
    try:
        import streamlit as st

        for k in keys:
            if k in st.secrets:
                s_val = str(st.secrets[k]).strip()
                if s_val:
                    return s_val
    except Exception:
        pass
    return None


def sanitize_api_key(raw_key: Optional[str]) -> str:
    """
    Higieniza o campo de API Key:
    - Remove espaços invisíveis Unicode (zero-width, non-breaking space) e aspas ao redor.
    - Caso o usuário tenha colado uma linha inteira (ex.: `GEMINI_API_KEY="AIzaSy..."` ou texto ao redor da chave),
      extrai automaticamente apenas o token ASCII da chave (`AIza...` ou `sk-...`).
    """
    if not raw_key:
        return ""
    cleaned = re.sub(r"[\u200b\u200c\u200d\ufeff\xa0]+", " ", str(raw_key)).strip().strip("\"'`")
    token_match = re.search(r"(AIza[0-9A-Za-z\-_]{20,}|sk-[0-9A-Za-z\-_]{15,})", cleaned)
    if token_match:
        return token_match.group(1)
    return cleaned.strip()


def is_valid_ascii_api_key(key: Optional[str]) -> bool:
    """
    Verifica se a string é uma chave de API ASCII válida para cabeçalhos HTTP (`httpx`),
    rejeitando textos com acentos (`ç`, `ã`, etc.) ou espaços colados por engano no input de senha.
    """
    if not key:
        return False
    return key.isascii() and not any(ch.isspace() for ch in key) and len(key) >= 8


def get_provider_api_key(provider_id: str, user_api_key: Optional[str] = None) -> Optional[str]:
    """
    Resolve a API key de um provedor priorizando a chave BYOK informada pelo usuário na sessão
    (após higienização) e fazendo fallback para `st.secrets` / variáveis de ambiente.
    """
    cleaned_user = sanitize_api_key(user_api_key)
    if cleaned_user:
        return cleaned_user

    key = normalize_provider_key(provider_id)
    raw_env_key: Optional[str] = None
    if key == "gemini":
        raw_env_key = _read_secret_or_env("GEMINI_API_KEY", "GOOGLE_API_KEY")
    elif key == "openai":
        raw_env_key = _read_secret_or_env("OPENAI_API_KEY")
    elif key == "anthropic":
        raw_env_key = _read_secret_or_env("ANTHROPIC_API_KEY")

    cleaned_env = sanitize_api_key(raw_env_key)
    return cleaned_env or None


def is_cloud_environment() -> bool:
    """
    Detecta se a aplicação está rodando em ambiente Cloud headless (como Streamlit Community Cloud).
    Pode ser forçada em testes via variável de ambiente `ASS_LGPD_ENV_MODE=cloud|local`.
    """
    forced = os.environ.get("ASS_LGPD_ENV_MODE", "").strip().lower()
    if forced == "cloud":
        return True
    if forced == "local":
        return False

    cloud_env_markers = (
        "STREAMLIT_SHARING_MODE",
        "IS_STREAMLIT_CLOUD",
        "STREAMLIT_SERVER_HEADLESS",
    )
    if any(os.environ.get(k) for k in cloud_env_markers):
        if Path("/mount/src").exists() or os.environ.get("USER") == "appuser":
            return True

    if Path("/mount/src").exists():
        return True

    return False


def check_ollama_status(
    base_url: str = DEFAULT_OLLAMA_URL,
    model_name: str = "qwen2.5:7b",
    timeout: float = 0.5,
    skip_if_cloud: bool = True,
) -> Dict[str, Any]:
    """
    Verifica rapidamente (timeout <= 0.6s) se o servidor local Ollama está ativo.
    Pula automaticamente a chamada HTTP caso detecte ambiente Cloud.
    """
    clean_url = (base_url or DEFAULT_OLLAMA_URL).strip().rstrip("/")
    target_model = (model_name or "qwen2.5:7b").strip()
    pull_cmd = f"ollama pull {target_model}"

    if skip_if_cloud and is_cloud_environment():
        return {
            "online": False,
            "model_ready": False,
            "installed_models": [],
            "requested_model": target_model,
            "base_url": clean_url,
            "is_cloud": True,
            "message": "Execução em nuvem detectada (Ollama local desabilitado).",
            "pull_command": pull_cmd,
        }

    try:
        eff_timeout = min(float(timeout), 0.6)
        resp = requests.get(f"{clean_url}/api/tags", timeout=eff_timeout)
        resp.raise_for_status()
        payload = resp.json() if resp.content else {}
        models_raw = payload.get("models", []) or []
        installed_models: List[str] = []
        for item in models_raw:
            name = str(item.get("name") or item.get("model") or "").strip()
            if name:
                installed_models.append(name)

        target_lower = target_model.lower()
        target_base = target_lower.split(":")[0]
        model_ready = any(
            m.lower() == target_lower
            or m.lower().startswith(f"{target_lower}:")
            or (":" not in target_lower and m.lower().startswith(f"{target_base}:"))
            for m in installed_models
        )

        if model_ready:
            msg = f"Ollama online em {clean_url} • Modelo `{target_model}` pronto."
        else:
            installed_str = ", ".join(installed_models[:5]) if installed_models else "nenhum modelo"
            msg = (
                f"Ollama ativo em {clean_url}, mas `{target_model}` não foi baixado "
                f"(instalados: {installed_str}). Execute: `{pull_cmd}`"
            )

        return {
            "online": True,
            "model_ready": model_ready,
            "installed_models": installed_models,
            "requested_model": target_model,
            "base_url": clean_url,
            "is_cloud": False,
            "message": msg,
            "pull_command": pull_cmd,
        }
    except Exception as exc:
        return {
            "online": False,
            "model_ready": False,
            "installed_models": [],
            "requested_model": target_model,
            "base_url": clean_url,
            "is_cloud": is_cloud_environment(),
            "message": f"Ollama local indisponível em `{clean_url}`.",
            "pull_command": pull_cmd,
            "error": str(exc),
        }


def detect_runtime_environment(base_url: str = DEFAULT_OLLAMA_URL) -> Dict[str, Any]:
    """
    Detecta o modo de operação (Cloud vs. Local) e define o provedor padrão seguro:
    - Em modo Cloud (ou quando Ollama local está offline): NUNCA escolhe Ollama como padrão;
      prioriza `gemini` (ou `openai`/`anthropic` caso apenas outra chave em nuvem esteja configurada).
    - Em modo Local (Ollama online e com modelo instalado): habilita todos os provedores.
    """
    status = check_ollama_status(base_url=base_url, timeout=0.5, skip_if_cloud=True)
    ollama_online = bool(status.get("online"))
    mode = "local" if ollama_online and not status.get("is_cloud") else "cloud"

    # Seleção inteligente de provedor padrão em nuvem
    if get_provider_api_key("gemini"):
        cloud_default = "gemini"
    elif get_provider_api_key("openai"):
        cloud_default = "openai"
    elif get_provider_api_key("anthropic"):
        cloud_default = "anthropic"
    else:
        cloud_default = "gemini"

    default_provider = cloud_default if mode == "cloud" else (
        "ollama" if status.get("model_ready") else cloud_default
    )

    return {
        "mode": mode,
        "is_cloud": mode == "cloud",
        "ollama_online": ollama_online,
        "ollama_model_ready": bool(status.get("model_ready")),
        "installed_ollama_models": status.get("installed_models", []),
        "default_provider": default_provider,
        "default_model": PROVIDER_DEFAULT_MODELS[default_provider],
        "ollama_status": status,
    }


def get_available_providers(ollama_online: bool = False) -> List[str]:
    """
    Retorna a lista de provedores habilitados para o ambiente atual:
    - No modo Cloud (`ollama_online=False`), oculta `ollama` do seletor principal para evitar erro.
    - No modo Local (`ollama_online=True`), inclui todos os provedores (nuvem + local).
    """
    cloud_providers = ["gemini", "openai", "anthropic"]
    if ollama_online:
        return cloud_providers + ["ollama"]
    return cloud_providers


def scrub_sensitive_keys(text: str) -> str:
    """Remove padrões de chaves de API de mensagens de erro/logs para evitar vazamento."""
    if not text:
        return ""
    scrubbed = re.sub(r"\b(AIza[0-9A-Za-z\-_]{20,}|sk-[0-9A-Za-z\-_]{16,}|hf_[0-9A-Za-z]{16,})\b", "[REDACTED_KEY]", text)
    return scrubbed


PROVIDER_USAGE_URLS: Dict[str, str] = {
    "gemini": "https://aistudio.google.com/usage",
    "openai": "https://platform.openai.com/usage",
    "anthropic": "https://console.anthropic.com/settings/limits",
}

QUOTA_ERROR_PATTERNS: Tuple[str, ...] = (
    "429",
    "resource_exhausted",
    "rate limit",
    "rate_limit",
    "too many requests",
    "quota",
    "insufficient_quota",
    "exceeded your current quota",
    "credit balance is too low",
    "insufficient credits",
    "billing_hard_limit",
    "spending limit",
    "usage limit",
    "requests per minute",
    "requests per day",
    "tokens per minute",
    "limit: 0",
)

HARD_QUOTA_PATTERNS: Tuple[str, ...] = (
    "insufficient_quota",
    "exceeded your current quota",
    "credit balance is too low",
    "insufficient credits",
    "billing_hard_limit",
    "spending limit",
    "requests per day",
    "limit: 0",
)


def is_quota_limit_error(exc: Union[Exception, str]) -> bool:
    """Detecta se a exceção corresponde a estouro de cota, créditos ou limite de taxa (429/Quota)."""
    lower = str(exc).lower()
    return any(pattern in lower for pattern in QUOTA_ERROR_PATTERNS)


def classify_api_error(
    exc: Exception,
    provider_id: str = "",
    using_user_key: bool = False,
) -> Dict[str, str]:
    """
    Classifica exceções técnicas de APIs de LLM (`quota`, `auth`, `model_not_found`, `context`,
    `timeout`, `overloaded`, `unknown`) e produz mensagens claras em português sem expor chaves de API.
    """
    raw = scrub_sensitive_keys(str(exc))
    lower = raw.lower()
    norm_prov = ""
    if provider_id:
        try:
            norm_prov = normalize_provider_key(provider_id)
        except ValueError:
            norm_prov = provider_id.strip().lower()
    prov_label = PROVIDER_LABELS.get(norm_prov, provider_id or "Provedor de IA")
    usage_url = PROVIDER_USAGE_URLS.get(norm_prov, "")
    key_origin = "da sua chave de API" if using_user_key else "da API configurada"

    if any(
        k in lower
        for k in (
            "codec can't encode character",
            "ordinal not in range(128)",
            "unicodeencodeerror",
            "caracteres inválidos",
            "acentos ou espaços",
        )
    ):
        return {
            "category": "auth",
            "message": (
                f"🔑 **Texto inválido no campo de API Key ({prov_label}):** "
                "O campo **🔑 API Key** na barra lateral contém acentos (como `ç`) ou frases em português "
                "coladas por engano no lugar da chave."
            ),
            "action_hint": (
                "Apague completamente o conteúdo do campo **🔑 API Key** na barra lateral esquerda e "
                "cole apenas sua chave real (ex.: `AIzaSy...`, `sk-...` ou `sk-ant-...`)."
            ),
        }

    if any(
        k in lower
        for k in (
            "401",
            "unauthorized",
            "invalid api key",
            "invalid_api_key",
            "api_key_invalid",
            "invalid token",
            "authentication_error",
        )
    ):
        return {
            "category": "auth",
            "message": (
                f"🔑 **Chave de API inválida ou sem permissão ({prov_label} — Erro 401):** "
                "Verifique se a chave informada na barra lateral está correta e ativa."
            ),
            "action_hint": "Revise ou substitua sua API Key na barra lateral esquerda.",
        }

    if is_quota_limit_error(raw):
        is_hard_quota = any(k in lower for k in HARD_QUOTA_PATTERNS)
        if is_hard_quota:
            msg = (
                f"🚫 **Limite da sua API atingido ({prov_label} — Cota/Créditos Esgotados):** "
                f"A cota de uso ou saldo disponível {key_origin} foi esgotado."
            )
        else:
            msg = (
                f"⏳ **Limite da sua API atingido ({prov_label} — Erro 429 / Cota Excedida):** "
                f"O limite de requisições ou tokens {key_origin} foi atingido neste momento."
            )
        usage_hint = f" Consulte seu consumo e limites em {usage_url}." if usage_url else ""
        return {
            "category": "quota",
            "message": msg,
            "action_hint": (
                "💡 **O que fazer agora:** Aguarde alguns instantes (se for limite por minuto), "
                "ou troque o provedor/chave na barra lateral (ex.: `gemini-3.8-flash` ou `gpt-4o-mini`)."
                f"{usage_hint}"
            ),
        }

    if any(
        k in lower
        for k in ("404", "not_found", "not found", "no longer available", "model_not_found", "does not exist")
    ):
        return {
            "category": "model_not_found",
            "message": (
                f"🔎 **Modelo indisponível ou descontinuado ({prov_label} — Erro 404):** "
                "O modelo selecionado não está disponível para esta conta ou versão da API."
            ),
            "action_hint": (
                "Selecione o modelo ativo na barra lateral (`gemini-3.8-flash`, "
                "`gpt-4o-mini` ou `claude-3-5-haiku-latest`)."
            ),
        }

    if any(k in lower for k in ("413", "context_length", "too many tokens", "maximum context", "token limit")):
        return {
            "category": "context",
            "message": (
                f"📏 **Limite de contexto excedido ({prov_label} — Erro 413):** "
                "A consulta excedeu a janela deste modelo. Selecione um modelo Recomendado ou clique em 'Limpar Conversa'."
            ),
            "action_hint": "",
        }

    if any(k in lower for k in ("timeout", "timed out", "504", "deadline")):
        return {
            "category": "timeout",
            "message": (
                f"⌛ **Tempo limite esgotado ({prov_label} — Timeout):** "
                "O provedor demorou muito para responder. Tente novamente ou escolha um modelo rápido (ex: Gemini Flash / GPT-4o-mini)."
            ),
            "action_hint": "",
        }

    if any(k in lower for k in ("503", "529", "unavailable", "overloaded", "high demand", "502")):
        return {
            "category": "overloaded",
            "message": (
                f"🚧 **Serviço temporariamente sobrecarregado ({prov_label} — Erro 503):** "
                "O provedor está com alta demanda no momento. Tente novamente em instantes ou alterne o modelo."
            ),
            "action_hint": "",
        }

    return {
        "category": "unknown",
        "message": f"⚠️ **Não foi possível concluir a geração ({prov_label}):** {raw}",
        "action_hint": "",
    }


def format_friendly_api_error(
    exc: Exception,
    provider_id: str = "",
    using_user_key: bool = False,
) -> str:
    """
    Converte exceções técnicas de APIs de LLM (401, 429/cota, 404, 413, timeout, 503) em mensagens
    claras em português com orientação prática e sem expor chaves de API.
    """
    info = classify_api_error(exc, provider_id=provider_id, using_user_key=using_user_key)
    if info["action_hint"]:
        return f"{info['message']}\n\n{info['action_hint']}"
    return info["message"]


def preflight_check_provider(
    provider_id: str,
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    ollama_online: Optional[bool] = None,
) -> Tuple[bool, str]:
    """
    Validação Prévia (Preflight Check) antes de executar o pipeline RAG ou chamar a LLM:
    1. Verifica se o provedor e o modelo são compatíveis.
    2. Bloqueia modelos do tier `unsupported`.
    3. Verifica disponibilidade do Ollama local ou presença de API Key (BYOK ou st.secrets).
    """
    try:
        key = normalize_provider_key(provider_id)
    except ValueError as exc:
        return False, str(exc)

    resolved_model = (model_id or PROVIDER_DEFAULT_MODELS[key]).strip()
    meta = get_model_metadata(key, resolved_model)

    if meta.get("tier") == "unsupported":
        return (
            False,
            f"🚫 O modelo `{resolved_model}` não é suportado para RAG jurídico (janela de contexto insuficiente). "
            "Selecione um modelo Recomendado ou Econômico na barra lateral.",
        )

    # Verifica se o modelo pertence a outro provedor por dessincronização
    for other_prov, other_cat in MODEL_CATALOG.items():
        if other_prov != key and resolved_model in other_cat:
            return (
                False,
                f"⚠️ O modelo `{resolved_model}` pertence ao provedor `{other_prov}` e não a `{key}`. "
                "Selecione um modelo compatível na barra lateral.",
            )

    if key == "ollama":
        if ollama_online is None:
            st_info = check_ollama_status(base_url=base_url, model_name=resolved_model, timeout=0.5)
            is_up = bool(st_info.get("online"))
        else:
            is_up = bool(ollama_online)

        if not is_up:
            return (
                False,
                "🖥️ **Qwen Local (Ollama) disponível apenas em execução local.** "
                "Para consultar na nuvem, selecione **Google Gemini**, **OpenAI** ou **Anthropic** na barra lateral.",
            )
        return True, "ok"

    resolved_key = get_provider_api_key(key, user_api_key=api_key)
    if not resolved_key:
        return (
            False,
            f"🔑 **Chave de API necessária:** Insira sua API Key para **{PROVIDER_LABELS[key]}** na barra lateral esquerda para habilitar o chat.",
        )

    if not is_valid_ascii_api_key(resolved_key):
        return (
            False,
            f"🔑 **Chave de API inválida para {PROVIDER_LABELS[key]}:** O campo **🔑 API Key** na barra lateral "
            "contém acentos, espaços ou texto colado por engano (ex.: mensagem em português). "
            "Limpe o campo de senha e cole apenas sua chave de API (`AIzaSy...` / `sk-...`).",
        )

    return True, "ok"


def _messages_to_system_and_user(
    messages: List[Dict[str, str]],
) -> Tuple[str, str, List[Dict[str, str]]]:
    """Extrai o system_prompt, o último user_prompt e a lista normalizada de mensagens."""
    system_parts: List[str] = []
    chat_turns: List[Dict[str, str]] = []
    for m in messages:
        role = str(m.get("role", "user")).strip().lower()
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            chat_turns.append({"role": role, "content": content})

    system_prompt = "\n\n".join(system_parts).strip()
    user_prompt = chat_turns[-1]["content"] if chat_turns else ""
    return system_prompt, user_prompt, chat_turns


class BaseLLMProvider(ABC):
    """Classe base abstrata (Strategy) para provedores de LLM com suporte a Streaming em Tempo Real."""

    provider_key: str = "base"

    def __init__(
        self,
        model_name: str,
        temperature: float = DEFAULT_TEMPERATURE,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model_name = model_name.strip()
        self.temperature = float(temperature)
        self.api_key = (api_key or "").strip() or None
        self.base_url = (base_url or "").strip() or None
        self.metadata = get_model_metadata(self.provider_key, self.model_name)

    @abstractmethod
    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Gera a resposta em streaming (token a token) a partir de system_prompt e user_prompt."""

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Interface padrão de Streaming (Fase 4): recebe `messages: list[dict]` (com roles
        `system`, `user`, `assistant`) e produz pedaços de texto em tempo real.
        """
        max_tokens = kwargs.get("max_tokens")
        system_prompt, _, chat_turns = _messages_to_system_and_user(messages)
        if len(chat_turns) <= 1:
            user_payload = chat_turns[0]["content"] if chat_turns else ""
        else:
            history_lines = [
                f"{'Usuário' if t['role'] == 'user' else 'Assistente'}: {t['content']}"
                for t in chat_turns[:-1]
            ]
            user_payload = (
                "Histórico recente da conversa:\n"
                + "\n".join(history_lines)
                + f"\n\nMensagem atual:\n{chat_turns[-1]['content']}"
            )

        if not self.metadata.get("supports_streaming", True):
            full_text = self.generate(
                system_prompt=system_prompt,
                user_prompt=user_payload,
                max_tokens=max_tokens,
            )
            if full_text:
                yield full_text
            return

        yield from self.stream_generate(
            system_prompt=system_prompt,
            user_prompt=user_payload,
            max_tokens=max_tokens,
        )

    def generate(
        self,
        system_prompt: Union[str, List[Dict[str, str]]] = "",
        user_prompt: str = "",
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Consome o gerador de streaming e retorna a string completa (aceita messages ou system+user)."""
        if isinstance(system_prompt, list):
            return "".join(self.generate_stream(system_prompt, max_tokens=max_tokens, **kwargs)).strip()
        return "".join(
            self.stream_generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
        ).strip()


class OllamaProvider(BaseLLMProvider):
    """
    Provedor Local conectando ao servidor Ollama (`http://localhost:11434`).
    """

    provider_key = "ollama"

    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        base_url: str = DEFAULT_OLLAMA_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_name=model_name or "qwen2.5:7b",
            temperature=temperature,
            api_key=api_key,
            base_url=base_url or DEFAULT_OLLAMA_URL,
        )

    def check_status(self) -> Dict[str, Any]:
        return check_ollama_status(
            base_url=self.base_url or DEFAULT_OLLAMA_URL,
            model_name=self.model_name,
        )

    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        host = (self.base_url or DEFAULT_OLLAMA_URL).rstrip("/")
        options: Dict[str, Any] = {
            "temperature": self.temperature,
            "num_ctx": 4096,
        }
        if max_tokens is not None:
            options["num_predict"] = int(max_tokens)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if ollama_sdk is not None:
            try:
                client = ollama_sdk.Client(host=host)
                chat_kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": True,
                    "options": options,
                    "keep_alive": "15m",
                }
                if "qwen3" in self.model_name.lower():
                    chat_kwargs["think"] = False
                stream = client.chat(**chat_kwargs)
                for chunk in stream:
                    msg = chunk.get("message", {}) if isinstance(chunk, dict) else getattr(chunk, "message", None)
                    content = ""
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                    elif msg is not None:
                        content = getattr(msg, "content", "") or ""
                    if content:
                        yield content
                return
            except Exception as exc:
                err_msg = str(exc)
                if "not found" in err_msg.lower() or "404" in err_msg.lower():
                    raise RuntimeError(
                        f"Modelo `{self.model_name}` não encontrado no Ollama local. "
                        f"Execute: `ollama pull {self.model_name}`"
                    ) from exc
                if any(k in err_msg.lower() for k in ("connection", "refused", "failed to connect", "10061")):
                    raise RuntimeError(
                        f"Serviço Ollama local não detectado em `{host}`."
                    ) from exc

        if OpenAI is not None:
            client_oai = OpenAI(base_url=f"{host}/v1", api_key="ollama")
            create_kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "stream": True,
            }
            if max_tokens is not None:
                create_kwargs["max_tokens"] = int(max_tokens)
            stream = client_oai.chat.completions.create(**create_kwargs)
            for event in stream:
                if event.choices and event.choices[0].delta and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
            return

        raise RuntimeError("Biblioteca `ollama` ou `openai` não instalada.")


_GEMINI_CLIENT_CACHE: Dict[str, genai.Client] = {}


class GeminiProvider(BaseLLMProvider):
    """
    Provedor Google Gemini (BYOK / Secrets) utilizando EXCLUSIVAMENTE o SDK oficial `google-genai`
    e `client.models.generate_content_stream`, otimizado para baixa latência com `gemini-3.8-flash`
    (`thinking_level="MINIMAL"`, reutilização de conexão HTTP e redirecionamento automático de modelos 2.x legados).
    """

    provider_key = "gemini"

    def __init__(
        self,
        model_name: str = "gemini-3.8-flash",
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        resolved_key = get_provider_api_key("gemini", user_api_key=api_key)
        super().__init__(
            model_name=model_name or "gemini-3.8-flash",
            temperature=temperature,
            api_key=resolved_key,
        )

    def _get_client(self) -> genai.Client:
        key = get_provider_api_key("gemini", user_api_key=self.api_key)
        if not key:
            raise ValueError(
                "API Key do Google Gemini não informada. Insira sua chave na barra lateral (BYOK)."
            )
        if not is_valid_ascii_api_key(key):
            raise ValueError(
                "A API Key informada contém caracteres inválidos (acentos ou espaços colados por engano). "
                "Limpe o campo de API Key na barra lateral e cole apenas sua chave AIzaSy..."
            )
        cached = _GEMINI_CLIENT_CACHE.get(key)
        if cached is None:
            cached = genai.Client(api_key=key)
            _GEMINI_CLIENT_CACHE[key] = cached
        return cached

    @staticmethod
    def _build_thinking_configs(candidate_model: str) -> List[Optional[types.ThinkingConfig]]:
        """
        Retorna a sequência de ThinkingConfig para minimizar drasticamente a latência (TTFT):
        - Para `gemini-3*` / `gemini-4*` (ex.: `gemini-3.8-flash`): usa `thinking_level="MINIMAL"`
          (evitando o modo padrão HIGH que leva 8-15s pensando antes do 1º token), com fallback para `"LOW"` e `None`.
        - Para outros modelos: `None`.
        """
        lower_m = candidate_model.lower()
        configs: List[Optional[types.ThinkingConfig]] = []
        if lower_m.startswith(("gemini-3", "gemini-4")):
            for level in ("MINIMAL", "LOW"):
                try:
                    configs.append(
                        types.ThinkingConfig(
                            thinking_level=level,  # type: ignore[arg-type]
                            include_thoughts=False,
                        )
                    )
                except Exception:
                    pass
        configs.append(None)
        return configs

    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        import time as _time

        client = self._get_client()

        # Redireciona automaticamente qualquer modelo 2.x legado/descontinuado para gemini-3.8-flash
        requested = (self.model_name or "gemini-3.8-flash").strip()
        if requested.lower().startswith("gemini-2."):
            active_target = "gemini-3.8-flash"
        else:
            active_target = requested

        fallback_models: List[str] = []
        for candidate in (active_target, "gemini-3.8-flash"):
            if candidate and candidate not in fallback_models:
                fallback_models.append(candidate)

        primary_exc: Optional[Exception] = None
        last_exc: Optional[Exception] = None

        for idx, candidate_model in enumerate(fallback_models):
            thinking_configs = self._build_thinking_configs(candidate_model)
            for t_cfg in thinking_configs:
                # Até 2 tentativas rápidas no mesmo modelo caso ocorra pico momentâneo 503 (overloaded)
                for attempt_503 in range(2):
                    try:
                        cfg_kwargs: Dict[str, Any] = {
                            "temperature": self.temperature,
                            "system_instruction": system_prompt,
                        }
                        if max_tokens is not None:
                            cfg_kwargs["max_output_tokens"] = int(max_tokens)
                        if t_cfg is not None:
                            cfg_kwargs["thinking_config"] = t_cfg

                        config = types.GenerateContentConfig(**cfg_kwargs)
                        stream = client.models.generate_content_stream(
                            model=candidate_model,
                            contents=user_prompt,
                            config=config,
                        )
                        yielded_any = False
                        for chunk in stream:
                            text = getattr(chunk, "text", None)
                            if text:
                                yielded_any = True
                                yield text
                        if yielded_any:
                            self.model_name = candidate_model
                            return
                        break
                    except Exception as exc:
                        if idx == 0 and primary_exc is None:
                            primary_exc = exc
                        last_exc = exc
                        err_lower = str(exc).lower()

                        # Se a cota/limite da API estourou, interrompe imediatamente para avisar o usuário
                        if is_quota_limit_error(exc):
                            raise

                        # Retry rápido (0.35s) se for 503 temporário (alta demanda momentânea no servidor Google)
                        if attempt_503 == 0 and any(
                            code in err_lower for code in ("503", "unavailable", "overloaded")
                        ):
                            _time.sleep(0.35)
                            continue

                        # Se thinking_level não for aceito por algum endpoint específico, tenta o próximo t_cfg
                        if t_cfg is not None and any(
                            k in err_lower
                            for k in ("thinking", "level", "budget", "invalid_argument", "400", "not supported")
                        ):
                            break

                        # Faz fallback para gemini-3.8-flash se um modelo customizado retornar 404/503
                        if any(
                            code in err_lower
                            for code in (
                                "404",
                                "not found",
                                "not_found",
                                "no longer available",
                                "503",
                                "unavailable",
                                "overloaded",
                            )
                        ):
                            break
                        raise
                else:
                    continue
                # Se saiu do loop de 503 por 404/503 definitivo, avança para o próximo candidate_model
                if last_exc is not None and any(
                    code in str(last_exc).lower()
                    for code in ("404", "not found", "not_found", "no longer available", "503", "unavailable", "overloaded")
                ):
                    break

        if primary_exc is not None:
            raise primary_exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Não foi possível obter stream de resposta do Google Gemini.")


class OpenAIProvider(BaseLLMProvider):
    """Provedor OpenAI / ChatGPT (BYOK / Secrets) utilizando o SDK oficial `openai`."""

    provider_key = "openai"

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        base_url: Optional[str] = None,
    ) -> None:
        resolved_key = get_provider_api_key("openai", user_api_key=api_key)
        super().__init__(
            model_name=model_name or "gpt-4o-mini",
            temperature=temperature,
            api_key=resolved_key,
            base_url=base_url,
        )

    def _get_client(self) -> Any:
        if OpenAI is None:
            raise RuntimeError("SDK `openai` não está instalado no ambiente.")
        key = get_provider_api_key("openai", user_api_key=self.api_key)
        if not key:
            raise ValueError(
                "API Key da OpenAI não informada. Insira sua chave na barra lateral (BYOK)."
            )
        kwargs: Dict[str, Any] = {"api_key": key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        client = self._get_client()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        create_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
        }
        if not self.model_name.lower().startswith(("o1", "o3")):
            create_kwargs["temperature"] = self.temperature
        if max_tokens is not None:
            if self.model_name.lower().startswith(("o1", "o3")):
                create_kwargs["max_completion_tokens"] = int(max_tokens)
            else:
                create_kwargs["max_tokens"] = int(max_tokens)

        stream = client.chat.completions.create(**create_kwargs)
        for event in stream:
            if event.choices and event.choices[0].delta and event.choices[0].delta.content:
                yield event.choices[0].delta.content


class AnthropicProvider(BaseLLMProvider):
    """Provedor Anthropic / Claude (BYOK / Secrets) utilizando o SDK oficial `anthropic`."""

    provider_key = "anthropic"

    def __init__(
        self,
        model_name: str = "claude-3-5-haiku-latest",
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        resolved_key = get_provider_api_key("anthropic", user_api_key=api_key)
        super().__init__(
            model_name=model_name or "claude-3-5-haiku-latest",
            temperature=temperature,
            api_key=resolved_key,
        )

    def _get_client(self) -> Any:
        if Anthropic is None:
            raise RuntimeError("SDK `anthropic` não está instalado no ambiente.")
        key = get_provider_api_key("anthropic", user_api_key=self.api_key)
        if not key:
            raise ValueError(
                "API Key da Anthropic não informada. Insira sua chave na barra lateral (BYOK)."
            )
        return Anthropic(api_key=key)

    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        client = self._get_client()
        budget_tokens = int(max_tokens) if max_tokens is not None else 1200
        with client.messages.stream(
            model=self.model_name,
            max_tokens=budget_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text


class LLMProviderFactory:
    """Factory leve e desacoplada para instanciar o provedor de LLM selecionado."""

    @staticmethod
    def create(
        provider_name: str = "gemini",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_OLLAMA_URL,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> BaseLLMProvider:
        key = normalize_provider_key(provider_name)
        resolved_model = (model_name or PROVIDER_DEFAULT_MODELS[key]).strip()

        # Se o modelo recebido pertencer a outro provedor (dessincronização), corrige para o padrão do provedor
        for other_prov, other_cat in MODEL_CATALOG.items():
            if other_prov != key and resolved_model in other_cat:
                resolved_model = PROVIDER_DEFAULT_MODELS[key]
                break

        if key == "ollama":
            return OllamaProvider(
                model_name=resolved_model,
                base_url=base_url or DEFAULT_OLLAMA_URL,
                temperature=temperature,
                api_key=api_key,
            )
        if key == "gemini":
            return GeminiProvider(
                model_name=resolved_model,
                api_key=api_key,
                temperature=temperature,
            )
        if key == "openai":
            return OpenAIProvider(
                model_name=resolved_model,
                api_key=api_key,
                temperature=temperature,
            )
        if key == "anthropic":
            return AnthropicProvider(
                model_name=resolved_model,
                api_key=api_key,
                temperature=temperature,
            )
        raise ValueError(f"Provedor não suportado: {provider_name}")


def get_llm_provider(
    provider_id: str = "gemini",
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> BaseLLMProvider:
    """Factory principal desacoplada para instanciar provedores de LLM (Fase 2)."""
    return LLMProviderFactory.create(
        provider_name=provider_id,
        model_name=model_id,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def create_llm_provider(
    provider_name: str = "gemini",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> BaseLLMProvider:
    """Alias retrocompatível para `get_llm_provider`."""
    return get_llm_provider(
        provider_id=provider_name,
        model_id=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )
