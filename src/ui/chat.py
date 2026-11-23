"""
Fluxo Conversacional, Onboarding, Indicador Animado em Duas Fases e Telemetria (src/ui/chat.py)
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from src.rag.engine import LGPDRagEngine
from src.rag.prompts import LANGUAGE_LEVEL_LABELS, SYSTEM_PROMPT
from src.rag.providers import (
    format_friendly_api_error,
    get_provider_api_key,
    preflight_check_provider,
)
from src.ui.architecture import render_project_details_page
from src.ui.components import detect_pii, mask_pii, render_hero_header

ONBOARDING_SUGGESTIONS: List[Dict[str, str]] = [
    {
        "label": "📋 Direitos do Titular (Art. 18)",
        "prompt": "Quais são os direitos do titular previstos no Art. 18 da LGPD e o prazo de atendimento do Art. 19?",
    },
    {
        "label": "⚖️ Base Legal e Biometria (Art. 7º e 11)",
        "prompt": "Posso usar legítimo interesse para tratar biometria facial ou preciso seguir o Art. 11 da LGPD?",
    },
    {
        "label": "🚨 Incidente de Segurança (Art. 48)",
        "prompt": "Qual é o prazo e os critérios para comunicar um incidente de segurança ou vazamento à ANPD (Art. 48)?",
    },
    {
        "label": "💰 Multas e Dosimetria (Res. ANPD nº 4/2023)",
        "prompt": "Quais são as sanções do Art. 52 da LGPD e os critérios de dosimetria e alíquotas da Resolução CD/ANPD nº 4/2023?",
    },
]


def _render_response_telemetry_caption(trace: Dict[str, Any]) -> None:
    """
    Exibe telemetria discreta abaixo da resposta (Fase 7.2):
    `Modelo · Nível · Busca (ms) · 1º token (ms) · Total (s) · ~Tokens`
    """
    if not trace:
        return
    model_used = trace.get("model_used", "-")
    lang_key = trace.get("language_level", "equilibrado")
    lang_label = LANGUAGE_LEVEL_LABELS.get(lang_key, lang_key)
    retrieval_ms = trace.get("retrieval_ms", 0.0)
    ttft_ms = trace.get("ttft_ms", 0.0)
    total_ms = trace.get("total_ms", trace.get("latency_ms", 0.0))
    total_s = round(float(total_ms) / 1000.0, 2)
    est_tokens = trace.get("estimated_tokens", 0)

    st.caption(
        f"🤖 `{model_used}` · 🗣️ `{lang_label}` · ⚡ Busca: `{retrieval_ms} ms` · "
        f"⏱️ 1º token: `{ttft_ms} ms` · 🏁 Total: `{total_s:.2f} s` · 🪙 `~{est_tokens} tokens`"
    )


def _render_sources_expander(
    retrieved_chunks: List[Dict[str, Any]],
    trace: Dict[str, Any] | None = None,
) -> None:
    """Renderiza `st.expander(f'Ver fontes utilizadas ({len(chunks)})')` com trechos e scores."""
    count = len(retrieved_chunks or [])
    with st.expander(f"📚 Ver fontes utilizadas ({count})", expanded=False):
        trace_data = trace or {}
        matched_concepts = trace_data.get("matched_concepts") or (
            retrieved_chunks[0].get("matched_concepts", []) if retrieved_chunks else []
        )
        if matched_concepts:
            st.markdown("**🧠 Expansão Jurídica & Fast-Path Ativados (`~0 ms`):**")
            for concept in matched_concepts:
                st.markdown(f"- 🔹 `{concept}`")
            st.divider()

        if not retrieved_chunks:
            st.info("Nenhum trecho normativo associado a esta resposta.")
            return

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            meta = chunk.get("metadata", {})
            origem = chunk.get("origem", "Busca Híbrida")
            is_cross = chunk.get("is_cross_reference", False)
            rerank_score = float(chunk.get("rerank_score", 0.0))
            badge = "🔗 [Referência Cruzada]" if is_cross else "🎯 [Busca + Reranker]"

            st.markdown(
                f"**#{idx} {badge}** `{meta.get('breadcrumb', chunk.get('id'))}` — **Score: `{rerank_score:.2f}`**"
            )
            st.caption(
                f"Origem: {origem} | Artigos citados: `{meta.get('artigos_citados') or 'Nenhuma'}`"
            )
            with st.container(border=True):
                st.markdown(chunk.get("texto", ""))


def _render_audit_inspector(msg: Dict[str, Any]) -> None:
    """Renderiza inspetor técnico opcional com payload XML enxuto e prompt enviado."""
    trace = msg.get("trace")
    if not trace:
        return
    with st.expander("🧾 Auditoria Técnica (Payload XML & Prompt)", expanded=False):
        st.caption(
            f"Tier do Modelo: `{trace.get('model_tier', 'recommended')}` | "
            f"Orçamento de Contexto: `{trace.get('context_tokens', 0)} / {trace.get('max_context_tokens', 2200)} tokens`"
        )
        st.code(trace.get("context_xml", ""), language="xml")


def render_chat_page(engine: LGPDRagEngine, runtime_env: Dict[str, Any]) -> None:
    """Renderiza a página principal de Chat com Onboarding, Preflight Check e Streaming em 2 fases."""
    render_hero_header()

    provider_key = st.session_state["provider_key"]
    selected_model = st.session_state["selected_model"]
    user_api_key = st.session_state["api_keys"].get(provider_key, "")
    effective_api_key = get_provider_api_key(provider_key, user_api_key=user_api_key)
    ollama_online = bool(runtime_env.get("ollama_online", False))

    # Validação Prévia (Preflight Check — Fase 2.4)
    is_ready, preflight_msg = preflight_check_provider(
        provider_id=provider_key,
        model_id=selected_model,
        api_key=effective_api_key,
        base_url=st.session_state.get("ollama_base_url", "http://localhost:11434"),
        ollama_online=ollama_online,
    )

    # Onboarding quando não houver mensagens no chat (Fase 7.2)
    if not st.session_state["messages"]:
        st.markdown("##### 💡 Perguntas Prontas para Iniciar:")
        cols = st.columns(2)
        for idx, item in enumerate(ONBOARDING_SUGGESTIONS):
            if cols[idx % 2].button(
                item["label"],
                key=f"onboard_btn_{idx}",
                use_container_width=True,
                help=item["prompt"],
            ):
                st.session_state["pending_suggestion"] = item["prompt"]

    if not is_ready:
        st.info(preflight_msg)

    # Renderiza histórico de mensagens
    for msg in st.session_state["messages"]:
        avatar = "🧑‍💼" if msg["role"] == "user" else "🛡️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("trace") is not None:
                    _render_response_telemetry_caption(msg["trace"])
                if msg.get("retrieved_chunks") is not None:
                    _render_sources_expander(
                        msg["retrieved_chunks"],
                        trace=msg.get("trace"),
                    )
                if msg.get("trace") is not None:
                    _render_audit_inspector(msg)

    user_input = st.chat_input(
        "Pergunte sobre artigos da LGPD, bases legais, incidentes ou multas da ANPD...",
        disabled=not is_ready,
    )
    active_question = user_input or st.session_state.get("pending_suggestion")
    if st.session_state.get("pending_suggestion"):
        st.session_state["pending_suggestion"] = None

    if not active_question:
        return

    if not is_ready:
        st.warning(preflight_msg)
        return

    # Privacy by Design (PII): detecta e mascara CPF, CNPJ e E-mail antes de enviar à LLM
    detected_pii = detect_pii(active_question)
    sanitized_question = active_question
    pii_masked_types: List[str] = []
    if detected_pii:
        if st.session_state.get("mask_pii_enabled", True):
            sanitized_question, pii_masked_types = mask_pii(active_question)
        else:
            st.warning(
                f"⚠️ **Aviso de Privacidade (PII):** Detectamos `{', '.join(detected_pii.keys())}` na sua pergunta. "
                "Recomendamos ativar o mascaramento automático na barra lateral."
            )

    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(sanitized_question)
        if pii_masked_types:
            st.caption(
                f"🛡️ *Privacy by Design: Dados pessoais ({', '.join(pii_masked_types)}) foram mascarados automaticamente antes do envio.*"
            )

    with st.chat_message("assistant", avatar="🛡️"):
        try:
            prior_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["messages"]
            ]

            # Fase 1 do Indicador Animado: Busca RAG com st.status recolhendo automaticamente
            with st.status(
                "Consultando base legal da LGPD/ANPD...",
                expanded=True,
            ) as status_box:
                st.write("🔎 Buscando dispositivos relevantes (Fast-Path + ChromaDB + BM25 + FlashRank)...")
                stream_result = engine.ask_stream(
                    question=sanitized_question,
                    history=prior_history,
                    provider_name=provider_key,
                    model_name=selected_model,
                    api_key=effective_api_key,
                    base_url=st.session_state.get("ollama_base_url", "http://localhost:11434"),
                    language_level=st.session_state.get("language_level", "equilibrado"),
                )
                chunks_found = stream_result.get("retrieved_chunks", [])
                ret_ms = stream_result.get("retrieval_ms", 0.0)
                st.write(f"✅ {len(chunks_found)} trechos selecionados em {ret_ms} ms.")
                status_box.update(
                    label=f"Base legal consultada: {len(chunks_found)} trechos selecionados em {ret_ms} ms",
                    state="complete",
                    expanded=False,
                )

            # Fase 2 do Indicador Animado: Geração em tempo real com st.write_stream
            answer_md = st.write_stream(stream_result["stream"])
            if not isinstance(answer_md, str):
                answer_md = stream_result.get("answer", "")

            trace_dict = dict(stream_result)
            trace_dict.pop("stream", None)
            trace_dict["answer"] = answer_md

            _render_response_telemetry_caption(trace_dict)
            _render_sources_expander(trace_dict["retrieved_chunks"], trace=trace_dict)

            msg_record = {
                "role": "assistant",
                "content": answer_md,
                "retrieved_chunks": trace_dict["retrieved_chunks"],
                "trace": trace_dict,
            }
            _render_audit_inspector(msg_record)

            st.session_state["messages"].append(
                {"role": "user", "content": sanitized_question}
            )
            st.session_state["messages"].append(msg_record)

        except Exception as exc:
            friendly_err = format_friendly_api_error(exc, provider_id=provider_key)
            st.error(friendly_err)
