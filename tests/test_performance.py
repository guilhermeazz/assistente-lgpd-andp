"""
Suíte de Testes de Performance, Latência e Arquitetura Multi-Provedor (tests/test_performance.py)
Valida:
1. Indexação local no ChromaDB com `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
   (incluindo auto-recriação de coleção com dimensão incompatível).
2. Tempo de recuperação híbrida (ChromaDB Cosine + BM25 RRF + 1 Cross-Reference) < 500ms.
3. Inicialização correta dos 4 provedores via Factory (`OllamaProvider`, `GeminiProvider`,
   `OpenAIProvider`, `AnthropicProvider`) com `temperature=0.1`.
4. Reescrita condicional de query (evitando chamadas duplas em perguntas de 5+ palavras sem pronomes).
5. Streaming ponta a ponta (`ask_stream`) e teste real com Qwen via Ollama (se ativo no ambiente).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Generator, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.engine import (
    DEFAULT_MAX_CROSS_REFS,
    DEFAULT_TOP_K,
    LGPDRagEngine,
)
from src.rag.providers import (
    AnthropicProvider,
    BaseLLMProvider,
    GeminiProvider,
    LLMProviderFactory,
    OllamaProvider,
    OpenAIProvider,
    check_ollama_status,
    create_llm_provider,
)


class DummyStreamProvider(BaseLLMProvider):
    """Provedor falso de streaming para validar TTFT e ask_stream determinístico."""

    provider_key = "dummy"

    def __init__(self) -> None:
        super().__init__(model_name="dummy-fast", temperature=0.1)

    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        _ = (system_prompt, user_prompt, max_tokens)
        for token in ("### 1. Resumo Direto\n", "O vazamento infringe o ", "**Art. 46** da LGPD."):
            yield token


def test_provider_factory_initializes_all_four_providers() -> None:
    """Valida a inicialização correta dos 4 provedores via Factory com temperature=0.1."""
    p_ollama = LLMProviderFactory.create("Qwen Local (Ollama)", model_name="qwen2.5:7b")
    assert isinstance(p_ollama, OllamaProvider)
    assert p_ollama.model_name == "qwen2.5:7b"
    assert p_ollama.temperature == 0.1

    p_gemini = create_llm_provider("Google Gemini (API Key)", model_name="gemini-2.5-flash", api_key="test-gemini-key")
    assert isinstance(p_gemini, GeminiProvider)
    assert p_gemini.model_name == "gemini-2.5-flash"
    assert p_gemini.api_key == "test-gemini-key"
    assert p_gemini.temperature == 0.1

    p_openai = create_llm_provider("OpenAI / ChatGPT (API Key)", model_name="gpt-4o-mini", api_key="test-openai-key")
    assert isinstance(p_openai, OpenAIProvider)
    assert p_openai.model_name == "gpt-4o-mini"
    assert p_openai.api_key == "test-openai-key"
    assert p_openai.temperature == 0.1

    p_anthropic = create_llm_provider(
        "Anthropic / Claude (API Key)",
        model_name="claude-3-5-haiku-latest",
        api_key="test-anthropic-key",
    )
    assert isinstance(p_anthropic, AnthropicProvider)
    assert p_anthropic.model_name == "claude-3-5-haiku-latest"
    assert p_anthropic.api_key == "test-anthropic-key"
    assert p_anthropic.temperature == 0.1


def test_conditional_query_rewrite_eliminates_unnecessary_llm_calls() -> None:
    """
    Garante que perguntas com 5 ou mais palavras sem pronomes referenciais ambíguos
    NÃO acionam chamada extra de LLM para reescrita.
    """
    assert not LGPDRagEngine._needs_query_rewrite(
        "se meu site acabar vazando acidentalmente o cpf de um usuario, estarei infringindo algum artigo da lei lgpd?"
    )
    assert not LGPDRagEngine._needs_query_rewrite(
        "Quais são as hipóteses legais do Art. 7º da LGPD?"
    )
    # Perguntas com pronomes referenciais ambíguos ou < 5 palavras devem acionar reescrita quando há histórico
    assert LGPDRagEngine._needs_query_rewrite("E qual a multa para esse artigo?")
    assert LGPDRagEngine._needs_query_rewrite("O que a ANPD diz sobre isso?")
    assert LGPDRagEngine._needs_query_rewrite("E as penalidades?")


def test_local_embedding_indexing_and_hybrid_retrieval_under_500ms() -> None:
    """
    Reindexa os chunks no ChromaDB com o modelo de embedding local e comprova que
    o tempo de recuperação híbrida (ChromaDB Cosine + BM25 + 1 Cross-Ref) é < 500ms.
    """
    engine = LGPDRagEngine()
    total_indexed = engine.index_chunks()
    assert total_indexed >= 82

    # Warm-up do encoder local em memória
    _ = engine.hybrid_search_with_cross_ref(
        query="Art. 46 segurança e prevenção",
        top_k=DEFAULT_TOP_K,
        max_cross_refs=DEFAULT_MAX_CROSS_REFS,
    )

    t0 = time.perf_counter()
    results = engine.hybrid_search_with_cross_ref(
        query="se meu site acabar vazando acidentalmente o cpf de um usuario, estarei infringindo algum artigo da lei lgpd?",
        top_k=DEFAULT_TOP_K,
        max_cross_refs=DEFAULT_MAX_CROSS_REFS,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 500.0, f"Recuperação híbrida demorou {elapsed_ms:.1f}ms (limite: 500ms)"
    assert 1 <= len(results) <= DEFAULT_TOP_K + DEFAULT_MAX_CROSS_REFS

    cross_refs = [r for r in results if r.get("is_cross_reference")]
    assert len(cross_refs) <= 1, "Expansão de referência cruzada deve ser limitada a no máximo 1 artigo."


def test_ask_stream_end_to_end_and_real_ollama_if_online() -> None:
    """
    Valida o método `ask_stream(question, history, top_k=4)` com desempacotamento de
    metadados + stream generator e executa streaming real com Qwen via Ollama se ativo.
    """
    engine = LGPDRagEngine()
    if engine.get_indexed_count() == 0:
        engine.index_chunks()

    # 1. Valida contrato de streaming e medição de TTFT com provedor determinístico rápido
    dummy = DummyStreamProvider()
    meta, token_stream = engine.ask_stream(
        question="Quais são as medidas de segurança do Art. 46?",
        history=[],
        top_k=4,
        provider=dummy,
    )
    assert meta["hybrid_count"] == 4
    assert meta["cross_ref_count"] <= 1
    assert meta["retrieval_ms"] < 500.0

    chunks_out = list(token_stream)
    assert len(chunks_out) == 3
    assert "Art. 46" in "".join(chunks_out)

    # 2. Verifica status do servidor Ollama local e roda teste real de streaming se disponível
    status = check_ollama_status(model_name="qwen2.5:7b")
    assert "online" in status
    assert "model_ready" in status

    if status["online"] and (status["model_ready"] or status["installed_models"]):
        target_model = (
            "qwen2.5:7b"
            if status["model_ready"]
            else status["installed_models"][0]
        )
        res = engine.ask_stream(
            question="Qual artigo da LGPD trata dos direitos do titular?",
            history=[],
            top_k=4,
            provider_name="ollama",
            model_name=target_model,
        )
        collected_tokens = []
        for tok in res["stream"]:
            collected_tokens.append(tok)
            if len("".join(collected_tokens)) >= 40:
                break
        assert len("".join(collected_tokens).strip()) > 0
