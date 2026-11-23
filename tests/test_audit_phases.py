"""
Suíte Completa de Testes de Auditoria (Fases 1 a 8) — tests/test_audit_phases.py
Cobre:
1. Fase 1: Validação rigorosa de PDFs (`%PDF-`, tamanho mínimo, bloqueio de HTML 403/404 salvo como .pdf)
   e presença dos artigos da Resolução CD/ANPD nº 4/2023 em `chunks_lgpd.jsonl`.
2. Fase 2: Sincronização blindada de `session_state` na troca de provedor/modelo (`on_provider_change`),
   detecção de ambiente Cloud vs. Local (garantindo que Ollama nunca seja padrão em modo Cloud)
   e Preflight Check / higienização de chaves em mensagens de erro.
3. Fase 3: Política de Tiers (`recommended`, `economic`, `unsupported`) e respeito estrito ao teto `max_context_tokens`.
4. Fase 4: Contrato de streaming `generate_stream(messages, **kwargs)` e telemetria (`retrieval_ms`, `ttft_ms`, `total_ms`).
5. Fase 5: Redução de tokens do System Prompt (>= 35%), Fast-Path por Regex para dispositivos diretos e janela de histórico.
6. Fase 6: Injeção dos 3 níveis de linguagem (`Simples`, `Equilibrado`, `Técnico`) preservando o núcleo de Grounding.
7. Fase 7: Detecção e mascaramento leve de PII (CPF, CNPJ e E-mail) e importação modular de `src/app.py` e `src/ui/`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.fetcher import (
    DEFAULT_RAW_DIR,
    validate_html_content,
    validate_pdf_content,
)
from src.ingestion.parser import DEFAULT_OUTPUT_JSONL
from src.rag.engine import (
    LGPDRagEngine,
    estimate_tokens,
    trim_chat_history,
)
from src.rag.legal_expander import extract_direct_article_references
from src.rag.prompts import (
    CORE_SYSTEM_PROMPT,
    build_system_prompt,
    normalize_language_level,
)
from src.rag.providers import (
    BaseLLMProvider,
    detect_runtime_environment,
    format_friendly_api_error,
    get_available_providers,
    get_llm_provider,
    get_model_metadata,
    get_selectable_models,
    preflight_check_provider,
    scrub_sensitive_keys,
)
from src.ui.architecture import (
    build_c4_level3_svg,
    build_pipeline_flow_html,
    get_c4_level3_mermaid,
)
from src.ui.components import detect_pii, mask_pii
from src.ui.state import (
    init_session_state,
    is_model_valid_for_provider,
    on_provider_change,
    sync_provider_and_model_state,
)


class MockStreamingProvider(BaseLLMProvider):
    """Mock de provedor de LLM para validar `generate_stream(messages)` e tiers."""

    provider_key = "gemini"

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        super().__init__(model_name=model_name, temperature=0.1, api_key="mock-key")

    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        _ = (system_prompt, user_prompt, max_tokens)
        yield "### 🎯 Resposta Direta\n"
        yield "**Sim.** O titular tem direito à confirmação e acesso (**Art. 18 da LGPD**).\n"
        yield "### 📜 Base Legal\n* `LGPD — Art. 18`"


def test_phase1_corrupted_pdf_validation_and_anpd_resolution_04_2023_chunks() -> None:
    """
    Valida que `validate_pdf_content` rejeita páginas HTML de erro 403/404 salvas como .pdf,
    arquivos truncados ou sem magic bytes `%PDF-`, e comprova que `resolucao_anpd_04_2023_dosimetria.pdf`
    e `chunks_lgpd.jsonl` contêm os dispositivos da Resolução CD/ANPD nº 4/2023.
    """
    # 1. HTML 404/403 salvo com extensão .pdf deve ser rejeitado
    fake_html_pdf = b"<!DOCTYPE html><html><body><h1>403 Forbidden - Access Denied</h1></body></html>"
    ok_html, msg_html = validate_pdf_content(fake_html_pdf, source_label="fake.pdf")
    assert not ok_html
    assert "HTML" in msg_html or "403" in msg_html

    # 2. Bytes sem magic header %PDF- devem ser rejeitados
    ok_magic, msg_magic = validate_pdf_content(b"NOT_A_PDF_HEADER" * 200, source_label="bad_magic.pdf")
    assert not ok_magic
    assert "Magic bytes" in msg_magic

    # 3. PDF truncado (< min_size_bytes) deve ser rejeitado
    ok_trunc, msg_trunc = validate_pdf_content(b"%PDF-1.7\nshort", min_size_bytes=2048, source_label="short.pdf")
    assert not ok_trunc
    assert "abaixo do tamanho" in msg_trunc

    # 4. Verifica o PDF real em data/raw/resolucao_anpd_04_2023_dosimetria.pdf
    real_pdf = DEFAULT_RAW_DIR / "resolucao_anpd_04_2023_dosimetria.pdf"
    assert real_pdf.exists()
    pdf_bytes = real_pdf.read_bytes()
    ok_real, _ = validate_pdf_content(pdf_bytes, min_size_bytes=2048, source_label=real_pdf.name)
    assert ok_real, "O arquivo resolucao_anpd_04_2023_dosimetria.pdf deve ser um PDF válido."

    # 5. Verifica que chunks_lgpd.jsonl contém os artigos de dosimetria da Resolução CD/ANPD nº 4/2023
    chunks = [
        json.loads(line)
        for line in DEFAULT_OUTPUT_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    anpd_texts = "\n".join(
        c["texto"] for c in chunks if c.get("metadata", {}).get("tipo") == "anpd_pdf"
    )
    assert "RESOLUÇÃO CD/ANPD Nº 4" in anpd_texts
    assert "Art. 8º" in anpd_texts
    assert "Art. 12" in anpd_texts
    assert "Art. 13" in anpd_texts


def test_phase2_cloud_vs_local_detection_and_session_state_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Garante que:
    - Em modo Cloud, Ollama/Qwen local NUNCA é escolhido como provedor padrão e não aparece ativo.
    - A troca de provedor (`on_provider_change`) redefine automaticamente o modelo para o padrão do novo provedor,
      eliminando erro de modelo incompatível.
    """
    monkeypatch.setenv("ASS_LGPD_ENV_MODE", "cloud")
    env_cloud = detect_runtime_environment()
    assert env_cloud["mode"] == "cloud"
    assert env_cloud["default_provider"] != "ollama"
    assert "ollama" not in get_available_providers(ollama_online=False)

    # Simula estado dessincronizado (provedor gemini com modelo da openai ou ollama em modo cloud)
    fake_state: Dict[str, Any] = {
        "provider_key": "ollama",
        "selected_model": "qwen2.5:7b",
    }
    sync_provider_and_model_state(state=fake_state, env_info=env_cloud)
    assert fake_state["provider_key"] != "ollama"
    assert is_model_valid_for_provider(fake_state["provider_key"], fake_state["selected_model"])

    # Simula troca de provedor para OpenAI e verifica se o modelo é atualizado automaticamente
    on_provider_change(state=fake_state, new_provider="OpenAI (ChatGPT)")
    assert fake_state["provider_key"] == "openai"
    assert fake_state["selected_model"] in get_selectable_models("openai")

    # Simula troca para Anthropic Claude
    on_provider_change(state=fake_state, new_provider="Anthropic (Claude)")
    assert fake_state["provider_key"] == "anthropic"
    assert fake_state["selected_model"] in get_selectable_models("anthropic")

    # Verifica que chaves de API são higienizadas em mensagens de erro
    err_str = format_friendly_api_error(
         RuntimeError("401 Unauthorized: invalid key sk-1234567890abcdefghijklmnop"),
        provider_id="openai",
    )
    assert "sk-1234567890" not in err_str
    assert "Erro 401" in err_str


def test_phase3_model_tiers_and_strict_max_context_tokens() -> None:
    """
    Valida a política de Tiers (`recommended`, `economic`, `unsupported`) e o respeito estrito
    ao orçamento `max_context_tokens`.
    """
    rec_meta = get_model_metadata("gemini", "gemini-2.5-flash")
    assert rec_meta["tier"] == "recommended"
    assert rec_meta["top_k"] == 5
    assert rec_meta["max_context_tokens"] == 2200

    econ_meta = get_model_metadata("openai", "gpt-4o-mini")
    assert econ_meta["tier"] == "economic"
    assert econ_meta["top_k"] == 3
    assert econ_meta["max_context_tokens"] == 1200

    unsup_meta = get_model_metadata("ollama", "qwen2.5:0.5b")
    assert unsup_meta["tier"] == "unsupported"
    assert "qwen2.5:0.5b" not in get_selectable_models("ollama", include_unsupported=False)

    # Preflight check deve bloquear modelo unsupported
    ok_unsup, msg_unsup = preflight_check_provider("ollama", "qwen2.5:0.5b", ollama_online=True)
    assert not ok_unsup
    assert "não é suportado" in msg_unsup

    # Verifica respeito estrito ao teto max_context_tokens
    sample_chunks = [
        {
            "id": f"chunk_{i}",
            "texto": ("Art. 46. Medidas de segurança técnicas e administrativas. " * 25),
            "metadata": {"fonte": "Lei nº 13.709/2018 (LGPD)", "artigo": f"Art. {i}"},
        }
        for i in range(1, 7)
    ]
    selected, xml_payload, used_tokens = LGPDRagEngine.select_chunks_within_token_budget(
        retrieved_chunks=sample_chunks,
        max_context_tokens=600,
    )
    assert used_tokens <= 600
    assert estimate_tokens(xml_payload) <= 600
    assert len(selected) < len(sample_chunks)


def test_phase4_and_phase5_generate_stream_contract_fast_path_and_prompt_compression() -> None:
    """
    Valida:
    - Enxugamento do System Prompt (~45-50% menor que os ~2350 chars originais).
    - Fast-Path por Regex para dispositivos diretos (`Art. 18`, `artigo 7º`, `Art. 48 da LGPD`).
    - Contrato `generate_stream(messages)` e telemetria `retrieval_ms`, `ttft_ms`, `total_ms`.
    """
    # 1. Verifica compressão do prompt base (antes: ~2350 caracteres; meta: redução >= 35%, ou seja <= 1520 chars)
    assert len(CORE_SYSTEM_PROMPT) <= 1500, f"Prompt base muito longo: {len(CORE_SYSTEM_PROMPT)} chars"

    # 2. Verifica Fast-Path por Regex
    assert extract_direct_article_references("Quais os direitos no Art. 18 e artigo 7º?") == [
        "Art. 18",
        "Art. 7º",
    ]
    assert extract_direct_article_references("Como funciona o Art. 55-J da LGPD?") == ["Art. 55-J"]

    # 3. Verifica streaming via generate_stream(messages) e telemetria no ask_stream
    mock_prov = MockStreamingProvider(model_name="gemini-2.0-flash")  # economic tier
    chunks_streamed = list(
        mock_prov.generate_stream(
            [
                {"role": "system", "content": "Responda curto."},
                {"role": "user", "content": "O que diz o Art. 18?"},
            ]
        )
    )
    assert len(chunks_streamed) == 3
    assert "Art. 18" in "".join(chunks_streamed)

    engine = LGPDRagEngine()
    res = engine.ask_stream(
        question="Quais são os direitos do titular no Art. 18 da LGPD?",
        history=[{"role": "user", "content": "Olá"}, {"role": "assistant", "content": "Olá! Como posso ajudar?"}],
        provider=mock_prov,
        language_level="simples",
    )
    output_text = "".join(res["stream"])
    assert "Art. 18" in output_text
    assert res["model_tier"] == "economic"
    assert res["max_context_tokens"] == 1200
    assert res["context_tokens"] <= 1200
    assert "Art. 18" in res["fast_path_articles"]
    assert res["retrieval_ms"] >= 0.0
    assert res["ttft_ms"] >= 0.0
    assert res["total_ms"] >= res["retrieval_ms"]


def test_phase6_and_phase7_language_levels_and_pii_masking() -> None:
    """
    Valida a injeção dos 3 níveis de linguagem mantendo as regras de Grounding
    e o detector/mascarador de PII (CPF, CNPJ e E-mail).
    """
    for level_input, expected_tag in (
        ("Simples (Cidadão)", "ESTILO [Simples / Cidadão]"),
        ("Equilibrado (Padrão)", "ESTILO [Equilibrado / Padrão]"),
        ("Técnico / Jurídico", "ESTILO [Técnico / Jurídico]"),
    ):
        prompt = build_system_prompt(level_input)
        assert expected_tag in prompt
        assert "REGRAS ESTRITAS DE ANCORAGEM (GROUNDING):" in prompt
        assert "VERIFICAÇÃO DE DADO SENSÍVEL:" in prompt

    # Validação de detecção e mascaramento de PII
    raw_query = (
        "O cliente de CPF 123.456.789-01 e email joao.silva@empresa.com.br "
        "da empresa CNPJ 12.345.678/0001-90 pediu exclusão."
    )
    detected = detect_pii(raw_query)
    assert "CPF" in detected
    assert "CNPJ" in detected
    assert "E-mail" in detected

    masked_query, masked_types = mask_pii(raw_query)
    assert "123.456.789-01" not in masked_query
    assert "joao.silva@empresa.com.br" not in masked_query
    assert "12.345.678/0001-90" not in masked_query
    assert "[CPF_MASCARADO]" in masked_query
    assert "[EMAIL_MASCARADO]" in masked_query
    assert "[CNPJ_MASCARADO]" in masked_query
    assert set(masked_types) == {"CPF", "CNPJ", "E-mail"}


def test_phase7_architecture_pipeline_and_c4_level3_diagram() -> None:
    """
    Valida a visualização do Pipeline Completo (6 etapas) e o diagrama C4 Model Nível 3
    (SVG inline puro + código Mermaid) na página de Arquitetura.
    """
    pipeline_html = build_pipeline_flow_html()
    assert "\n\n" not in pipeline_html
    for stage_kw in ("Coleta", "Parser", "PII", "Busca Híbrida", "Rerank", "Streaming"):
        assert stage_kw in pipeline_html

    c4_svg = build_c4_level3_svg()
    assert c4_svg.startswith("<div") or c4_svg.startswith("<svg")
    assert "</svg>" in c4_svg
    assert "<foreignObject" not in c4_svg
    assert "\n\n" not in c4_svg

    expected_components = [
        "UI &amp; Chat Controller",
        "Session &amp; Env Manager",
        "PII Privacy Guard",
        "Legal Query Expander",
        "Hybrid RAG &amp; Rerank Engine",
        "Prompt &amp; Language Builder",
        "LLM Provider Factory",
        "Normative Fetcher",
        "Semantic-Structural Parser",
        "ChromaDB",
        "Planalto",
        "Cloud LLM APIs",
        "Ollama",
    ]
    for comp in expected_components:
        assert comp in c4_svg, f"Componente ausente no SVG C4 Nível 3: {comp}"

    c4_mermaid = get_c4_level3_mermaid()
    assert c4_mermaid.startswith("flowchart TB")
    for m_comp in (
        "UI & Chat Controller",
        "Session & Env Manager",
        "PII Privacy Guard",
        "Legal Query Expander",
        "Hybrid RAG & Rerank Engine",
        "Prompt & Language Builder",
        "LLM Provider Factory",
        "Normative Fetcher",
        "Semantic-Structural Parser",
        "ChromaDB",
    ):
        assert m_comp in c4_mermaid, f"Componente ausente no Mermaid C4 Nível 3: {m_comp}"
