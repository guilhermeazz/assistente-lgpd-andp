"""
Arquitetura Multi-Provedor LLM (src/rag/providers.py)
Implementa o padrão Strategy/Factory com streaming nativo de tokens e temperature=0.1:
1. OllamaProvider (100% Local/Offline com Qwen via Ollama, com failover transparente para
   Qwen no Hugging Face GPU quando rodando no Streamlit Cloud com HF_TOKEN configurado)
2. HuggingFaceQwenProvider (Qwen hospedado nas GPUs do Hugging Face via st.secrets["HF_TOKEN"],
   sem exigir que o usuário final digite chave de API na interface)
3. GeminiProvider (BYOK - SDK oficial `google-genai` via `generate_content_stream`)
4. OpenAIProvider (BYOK - SDK oficial `openai` via `chat.completions.create(..., stream=True)`)
5. AnthropicProvider (BYOK - SDK oficial `anthropic` via `messages.stream(...)`)
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional

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
DEFAULT_HF_ROUTER_URL = "https://router.huggingface.co/v1"
DEFAULT_TEMPERATURE = 0.1

PROVIDER_LABELS: Dict[str, str] = {
    "ollama": "Qwen (Local - Ollama)",
    "gemini": "Google Gemini (API Key)",
    "openai": "OpenAI / ChatGPT (API Key)",
    "anthropic": "Anthropic / Claude (API Key)",
}

OLLAMA_MODELS: List[str] = [
    "qwen2.5:7b",
    "qwen2.5:3b",
    "qwen2.5:14b",
]

HF_QWEN_MODELS: List[str] = [
    "Qwen/Qwen2.5-7B-Instruct:fastest",
    "Qwen/Qwen2.5-72B-Instruct:fastest",
    "Qwen/Qwen2.5-32B-Instruct:fastest",
    "Qwen/Qwen3-8B:fastest",
]

GEMINI_MODELS: List[str] = [
    "gemini-3.8-flash",
]

OPENAI_MODELS: List[str] = [
    "gpt-4o-mini",
    "gpt-4o",
    "o3-mini",
]

ANTHROPIC_MODELS: List[str] = [
    "claude-3-5-haiku-latest",
    "claude-sonnet-4-20250514",
]

PROVIDER_DEFAULT_MODELS: Dict[str, str] = {
    "ollama": "qwen2.5:7b",
    "huggingface": "Qwen/Qwen2.5-7B-Instruct:fastest",
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
}


def get_configured_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """
    Obtém o token do Hugging Face configurado no servidor/deploy (`st.secrets["HF_TOKEN"]`
    ou variável de ambiente `HF_TOKEN`), permitindo que o usuário final utilize o Qwen nas
    GPUs do Hugging Face sem precisar digitar nenhuma chave na interface.
    """
    if explicit_token and explicit_token.strip():
        return explicit_token.strip()

    for env_var in ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGINGFACEHUB_API_TOKEN"):
        val = os.environ.get(env_var, "").strip()
        if val:
            return val

    try:
        import streamlit as st

        for secret_key in ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGINGFACEHUB_API_TOKEN"):
            if secret_key in st.secrets:
                secret_val = str(st.secrets[secret_key]).strip()
                if secret_val:
                    return secret_val
    except Exception:
        pass

    return None


def _discover_hf_qwen_models(token: str, base_url: str = DEFAULT_HF_ROUTER_URL) -> List[str]:
    """Consulta `/v1/models` no Hugging Face Router para localizar modelos Qwen habilitados na conta."""
    try:
        resp = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {token}"},
            timeout=4.0,
        )
        if resp.status_code != 200:
            return []
        data = resp.json().get("data", []) or []
        qwen_ids = [
            str(item.get("id", "")).strip()
            for item in data
            if "qwen" in str(item.get("id", "")).lower()
            and "embed" not in str(item.get("id", "")).lower()
            and "vl" not in str(item.get("id", "")).lower()
        ]
        return qwen_ids[:4]
    except Exception:
        return []


def normalize_provider_key(provider_name: str) -> str:
    """Normaliza o identificador ou rótulo da UI para a chave canônica do provedor."""
    cleaned = (provider_name or "ollama").strip().lower()
    if any(k in cleaned for k in ("huggingface", "hugging face", "hf", "gpu")):
        return "huggingface"
    if any(k in cleaned for k in ("ollama", "qwen", "local")):
        return "ollama"
    if any(k in cleaned for k in ("gemini", "google")):
        return "gemini"
    if any(k in cleaned for k in ("openai", "chatgpt", "gpt")):
        return "openai"
    if any(k in cleaned for k in ("anthropic", "claude")):
        return "anthropic"
    raise ValueError(
        f"Provedor desconhecido: '{provider_name}'. Opções válidas: ollama, huggingface, gemini, openai, anthropic."
    )


def check_ollama_status(
    base_url: str = DEFAULT_OLLAMA_URL,
    model_name: str = "qwen2.5:7b",
    timeout: float = 1.5,
) -> Dict[str, Any]:
    """
    Verifica rapidamente se o servidor local Ollama está ativo e se o modelo Qwen
    solicitado já foi baixado localmente.
    """
    clean_url = (base_url or DEFAULT_OLLAMA_URL).strip().rstrip("/")
    target_model = (model_name or "qwen2.5:7b").strip()
    pull_cmd = f"ollama pull {target_model}"

    try:
        resp = requests.get(f"{clean_url}/api/tags", timeout=timeout)
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
            msg = f"Ollama online em {clean_url} • Modelo `{target_model}` pronto para inferência local."
        else:
            installed_str = ", ".join(installed_models[:5]) if installed_models else "nenhum modelo"
            msg = (
                f"Serviço Ollama ativo em {clean_url}, mas o modelo `{target_model}` ainda não foi baixado "
                f"(instalados: {installed_str}). Execute no terminal: `{pull_cmd}`"
            )

        return {
            "online": True,
            "model_ready": model_ready,
            "installed_models": installed_models,
            "requested_model": target_model,
            "base_url": clean_url,
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
            "message": (
                f"Servidor Ollama não detectado em `{clean_url}`. "
                f"Inicie o aplicativo Ollama (`ollama serve`) e baixe o modelo com `{pull_cmd}`."
            ),
            "pull_command": pull_cmd,
            "error": str(exc),
        }


class BaseLLMProvider(ABC):
    """Classe base abstrata (Strategy) para provedores de LLM com suporte a Streaming nativo."""

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

    @abstractmethod
    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Gera a resposta em streaming (token a token) com temperature=0.1."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Consome o gerador de streaming e retorna a string completa."""
        return "".join(
            self.stream_generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
        ).strip()


class HuggingFaceQwenProvider(BaseLLMProvider):
    """
    Provedor Qwen hospedado nas GPUs do Hugging Face (`https://router.huggingface.co/v1`).
    Utiliza a chave `HF_TOKEN` pré-configurada nos Secrets do Streamlit (`st.secrets["HF_TOKEN"]`
    ou variável de ambiente `HF_TOKEN`), dispensando o usuário final de inserir API Key.
    """

    provider_key = "huggingface"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct:fastest",
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        base_url: str = DEFAULT_HF_ROUTER_URL,
    ) -> None:
        resolved_token = get_configured_hf_token(api_key)
        super().__init__(
            model_name=model_name or "Qwen/Qwen2.5-7B-Instruct:fastest",
            temperature=temperature,
            api_key=resolved_token,
            base_url=base_url or DEFAULT_HF_ROUTER_URL,
        )

    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        if OpenAI is None:
            raise RuntimeError("SDK `openai` necessário para o roteador do Hugging Face.")

        token = get_configured_hf_token(self.api_key)
        if not token:
            raise ValueError(
                "Token `HF_TOKEN` não encontrado nos Secrets do Streamlit (`st.secrets['HF_TOKEN']`) "
                "nem nas variáveis de ambiente. Configure `HF_TOKEN = \"hf_...\"` nos Secrets do projeto."
            )

        endpoint = (self.base_url or DEFAULT_HF_ROUTER_URL).rstrip("/")
        client = OpenAI(base_url=endpoint, api_key=token)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Garante variantes com sufixo `:fastest` e sem sufixo para evitar HTTP 400 (model_not_supported)
        raw_candidates: List[str] = [self.model_name]
        if ":" not in self.model_name:
            raw_candidates.insert(0, f"{self.model_name}:fastest")
        raw_candidates.extend(
            [
                "Qwen/Qwen2.5-7B-Instruct:fastest",
                "Qwen/Qwen2.5-72B-Instruct:fastest",
                "Qwen/Qwen2.5-32B-Instruct:fastest",
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-72B-Instruct",
            ]
        )

        candidate_models: List[str] = []
        for m in raw_candidates:
            if m and m not in candidate_models:
                candidate_models.append(m)

        # Orçamento seguro de saída (768 tokens) para não ultrapassar janelas de 4.096 tokens
        primary_budget = int(max_tokens) if max_tokens is not None else 768
        token_budgets = [primary_budget] if max_tokens is not None else [768, 450]

        last_exc: Optional[Exception] = None

        def _try_models(models_to_try: List[str]) -> Generator[str, None, bool]:
            nonlocal last_exc
            for candidate_model in models_to_try:
                for budget in token_budgets:
                    try:
                        stream = client.chat.completions.create(
                            model=candidate_model,
                            messages=messages,
                            temperature=self.temperature,
                            max_tokens=budget,
                            stream=True,
                        )
                        yielded_any = False
                        for event in stream:
                            if (
                                event.choices
                                and event.choices[0].delta
                                and event.choices[0].delta.content
                            ):
                                yielded_any = True
                                yield event.choices[0].delta.content
                        if yielded_any:
                            self.model_name = candidate_model
                            return True
                    except Exception as exc:
                        last_exc = exc
                        err_lower = str(exc).lower()
                        if any(k in err_lower for k in ("401", "unauthorized", "invalid token")):
                            raise ValueError(
                                "Token `HF_TOKEN` inválido ou sem permissão 'Make calls to Inference Providers' no Hugging Face."
                            ) from exc
                        # Se o erro 400 for estouro de contexto/tokens, tenta o próximo budget menor no mesmo modelo
                        if any(k in err_lower for k in ("max_new_tokens", "max_tokens", "context", "4096")):
                            logger.warning(
                                "Ajustando max_tokens para %s devido a limite de janela (%s).",
                                candidate_model,
                                exc,
                            )
                            continue
                        logger.warning(
                            "Modelo Qwen indisponível no Hugging Face Router (%s): %s",
                            candidate_model,
                            exc,
                        )
                        break
            return False

        # 1. Tenta a lista prioritária de modelos Qwen
        gen = _try_models(candidate_models)
        try:
            while True:
                yield next(gen)
        except StopIteration as stop:
            if stop.value:
                return

        # 2. Descoberta dinâmica via `/v1/models` caso os modelos fixos não estejam habilitados na conta
        discovered = [
            m for m in _discover_hf_qwen_models(token, endpoint) if m not in candidate_models
        ]
        if discovered:
            gen_disc = _try_models(discovered)
            try:
                while True:
                    yield next(gen_disc)
            except StopIteration as stop:
                if stop.value:
                    return

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Não foi possível obter resposta das GPUs do Hugging Face.")


class OllamaProvider(BaseLLMProvider):
    """
    Provedor Local conectando ao servidor Ollama (`http://localhost:11434`)
    para execução 100% offline/local da família Qwen (`qwen2.5:7b`, `qwen2.5:3b`, `qwen2.5:14b`, etc.).
    Se o Ollama local estiver indisponível (ex: deploy no Streamlit Cloud) e `HF_TOKEN` estiver
    configurado nos Secrets, faz failover transparente para o Qwen nas GPUs do Hugging Face.
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
        """Retorna o diagnóstico rápido do servidor Ollama e do modelo configurado."""
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

        # 1. Tenta via SDK oficial `ollama` (com keep_alive="15m")
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
                hf_token = get_configured_hf_token(self.api_key)
                if hf_token and any(
                    k in err_msg.lower()
                    for k in ("connection", "refused", "failed to connect", "10061", "10013", "not found", "404")
                ):
                    logger.info(
                        "Ollama local indisponível (%s); redirecionando automaticamente para Qwen nas GPUs do Hugging Face.",
                        err_msg,
                    )
                    hf_provider = HuggingFaceQwenProvider(
                        model_name="Qwen/Qwen2.5-7B-Instruct:fastest",
                        api_key=hf_token,
                        temperature=self.temperature,
                    )
                    yield from hf_provider.stream_generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=max_tokens,
                    )
                    self.model_name = hf_provider.model_name
                    return

                if "not found" in err_msg.lower() or "404" in err_msg.lower():
                    raise RuntimeError(
                        f"Modelo `{self.model_name}` não encontrado no Ollama local. "
                        f"Execute no terminal: `ollama pull {self.model_name}`"
                    ) from exc
                if any(k in err_msg.lower() for k in ("connection", "refused", "failed to connect", "10061")):
                    raise RuntimeError(
                        f"Não foi possível conectar ao Ollama em `{host}`. "
                        f"Verifique se o serviço está ativo (`ollama serve`) ou utilize o provedor 'Qwen (Hugging Face GPU)'."
                    ) from exc
                logger.warning("Fallback do SDK ollama para OpenAI-compatible /v1: %s", exc)

        # 2. Fallback via SDK `openai` apontando para `{host}/v1`
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

        raise RuntimeError(
            "Biblioteca `ollama` ou `openai` não instalada no ambiente virtual."
        )


class GeminiProvider(BaseLLMProvider):
    """
    Provedor Google Gemini (BYOK) utilizando EXCLUSIVAMENTE o SDK oficial `google-genai`
    e o método `client.models.generate_content_stream`.
    """

    provider_key = "gemini"

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        super().__init__(
            model_name=model_name or "gemini-2.5-flash",
            temperature=temperature,
            api_key=api_key,
        )

    def _get_client(self) -> genai.Client:
        if not self.api_key:
            raise ValueError(
                "API Key do Google Gemini não informada. Insira sua chave na barra lateral (BYOK)."
            )
        return genai.Client(api_key=self.api_key)

    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        client = self._get_client()

        fallback_models: List[str] = []
        for candidate in (
            self.model_name,
            "gemini-2.5-flash",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-preview",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
        ):
            if candidate and candidate not in fallback_models:
                fallback_models.append(candidate)

        last_exc: Optional[Exception] = None
        for candidate_model in fallback_models:
            for use_thinking_zero in (True, False):
                try:
                    cfg_kwargs: Dict[str, Any] = {
                        "temperature": self.temperature,
                        "system_instruction": system_prompt,
                    }
                    if max_tokens is not None:
                        cfg_kwargs["max_output_tokens"] = int(max_tokens)
                    if use_thinking_zero:
                        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

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
                except Exception as exc:
                    last_exc = exc
                    err_lower = str(exc).lower()
                    if use_thinking_zero and "thinking" in err_lower:
                        continue
                    if any(
                        code in err_lower
                        for code in ("404", "not found", "no longer available", "503", "unavailable", "overloaded")
                    ):
                        logger.warning(
                            "Modelo Gemini %s indisponível (%s); tentando próximo da cadeia Flash.",
                            candidate_model,
                            exc,
                        )
                        break
                    raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Não foi possível obter stream de resposta do Google Gemini.")


class OpenAIProvider(BaseLLMProvider):
    """
    Provedor OpenAI / ChatGPT (BYOK) utilizando o SDK oficial `openai`
    (`client.chat.completions.create(..., stream=True)`).
    """

    provider_key = "openai"

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_name=model_name or "gpt-4o-mini",
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )

    def _get_client(self) -> Any:
        if OpenAI is None:
            raise RuntimeError("SDK `openai` não está instalado no ambiente.")
        if not self.api_key:
            raise ValueError(
                "API Key da OpenAI não informada. Insira sua chave na barra lateral (BYOK)."
            )
        kwargs: Dict[str, Any] = {"api_key": self.api_key}
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
    """
    Provedor Anthropic / Claude (BYOK) utilizando o SDK oficial `anthropic`
    (`client.messages.stream(...)`).
    """

    provider_key = "anthropic"

    def __init__(
        self,
        model_name: str = "claude-3-5-haiku-latest",
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        super().__init__(
            model_name=model_name or "claude-3-5-haiku-latest",
            temperature=temperature,
            api_key=api_key,
        )

    def _get_client(self) -> Any:
        if Anthropic is None:
            raise RuntimeError("SDK `anthropic` não está instalado no ambiente.")
        if not self.api_key:
            raise ValueError(
                "API Key da Anthropic não informada. Insira sua chave na barra lateral (BYOK)."
            )
        return Anthropic(api_key=self.api_key)

    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        client = self._get_client()
        budget_tokens = int(max_tokens) if max_tokens is not None else 1536
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
    """Factory para instanciar o provedor de LLM selecionado pelo usuário."""

    @staticmethod
    def create(
        provider_name: str = "huggingface",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_OLLAMA_URL,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> BaseLLMProvider:
        key = normalize_provider_key(provider_name)
        resolved_model = (model_name or PROVIDER_DEFAULT_MODELS[key]).strip()

        if key == "ollama":
            return OllamaProvider(
                model_name=resolved_model,
                base_url=base_url or DEFAULT_OLLAMA_URL,
                temperature=temperature,
                api_key=api_key,
            )
        if key == "huggingface":
            return HuggingFaceQwenProvider(
                model_name=resolved_model,
                api_key=api_key,
                temperature=temperature,
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


def create_llm_provider(
    provider_name: str = "huggingface",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> BaseLLMProvider:
    """Atalho funcional para LLMProviderFactory.create(...)."""
    return LLMProviderFactory.create(
        provider_name=provider_name,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )
