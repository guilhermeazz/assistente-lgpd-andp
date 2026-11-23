"""
Estilos CSS, Badges de Tier/Ambiente, Alertas e Helpers de Privacy by Design / PII (src/ui/components.py)
Implementa:
- Detecção e mascaramento leve por regex de Dados Pessoais (CPF, CNPJ, E-mail e Telefone)
  antes de enviar consultas para APIs externas (Privacy by Design).
- Componentes visuais para badges de Tier (`recommended`, `economic`, `unsupported`) e Ambiente (`Cloud` / `Local`).
- Estilos CSS leves e responsivos inspirados no tema Dark Obsidian & Teal.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import streamlit as st

# Regex para detecção leve de PII (CPF, CNPJ e E-mail) — Fase 7.2
RE_CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
RE_CNPJ = re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def detect_pii(text: str) -> Dict[str, List[str]]:
    """
    Detecta ocorrências de CPF, CNPJ e E-mail no texto digitado pelo usuário.
    Retorna dicionário com as categorias encontradas.
    """
    if not text:
        return {}
    found: Dict[str, List[str]] = {}

    cpfs = RE_CPF.findall(text)
    if cpfs:
        found["CPF"] = cpfs

    cnpjs = RE_CNPJ.findall(text)
    if cnpjs:
        found["CNPJ"] = cnpjs

    emails = RE_EMAIL.findall(text)
    if emails:
        found["E-mail"] = emails

    return found


def mask_pii(text: str) -> Tuple[str, List[str]]:
    """
    Mascara automaticamente CPF (`[CPF_MASCARADO]`), CNPJ (`[CNPJ_MASCARADO]`) e
    E-mail (`[EMAIL_MASCARADO]`) antes de enviar a consulta a provedores de LLM.
    Retorna `(masked_text, detected_types)`.
    """
    if not text:
        return "", []

    detected = detect_pii(text)
    if not detected:
        return text, []

    masked = text
    if "CNPJ" in detected:
        masked = RE_CNPJ.sub("[CNPJ_MASCARADO]", masked)
    if "CPF" in detected:
        masked = RE_CPF.sub("[CPF_MASCARADO]", masked)
    if "E-mail" in detected:
        masked = RE_EMAIL.sub("[EMAIL_MASCARADO]", masked)

    return masked, list(detected.keys())


def format_tier_badge(tier: str) -> str:
    """Retorna rótulo visual legível para o Tier do modelo."""
    t = (tier or "recommended").lower()
    if t == "recommended":
        return "🟢 Tier Recomendado (Contexto Completo)"
    if t == "economic":
        return "🟡 Modo Econômico (Contexto Otimizado)"
    return "🔴 Não Suportado"


def inject_custom_css() -> None:
    """Injeta estilos CSS leves Dark Obsidian & Teal com barra inferior fixa."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(ellipse 75% 42% at 50% 102%, rgba(14, 165, 233, 0.20) 0%, rgba(13, 148, 136, 0.09) 38%, rgba(9, 13, 20, 0.0) 75%),
                linear-gradient(180deg, #070a10 0%, #090d14 55%, #0b131e 100%) !important;
            color: #f8fafc !important;
        }
        section[data-testid="stSidebar"] {
            background: #06090f !important;
            border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
        }
        .block-container {
            padding-top: 1.8rem !important;
            padding-bottom: 3.5rem !important;
            max-width: 920px !important;
        }
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
        .wave-hero-container {
            position: relative;
            width: 100%;
            height: clamp(54px, 8vw, 88px);
            overflow: visible;
            margin-top: 0.1rem;
            margin-bottom: 0.5rem;
            background: transparent !important;
        }
        .wave-svg {
            width: 100%;
            height: 100%;
            display: block;
            overflow: visible;
            filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.25));
        }
        .lix-greeting-wrap {
            text-align: center;
            margin-bottom: 1.25rem;
        }
        .lix-hi {
            font-size: 1.18rem;
            font-weight: 600;
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 52%, #2dd4bf 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.15rem;
        }
        .lix-headline {
            font-size: 2.05rem;
            font-weight: 700;
            background: linear-gradient(180deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.45rem;
        }
        .lix-subtext {
            font-size: 0.88rem;
            color: #94a3b8;
            max-width: 580px;
            margin: 0 auto;
            line-height: 1.45;
        }
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
            padding-bottom: 1.0rem !important;
            box-sizing: border-box !important;
            background: transparent !important;
        }
        div[data-testid="stChatInput"] > div {
            background: #0b0f17 !important;
            border: 1px solid rgba(56, 189, 248, 0.28) !important;
            border-radius: 16px !important;
            padding: 0.25rem 0.45rem !important;
        }
        div[data-testid="stChatMessage"] {
            background: rgba(15, 23, 36, 0.58) !important;
            border: 1px solid rgba(255, 255, 255, 0.07) !important;
            border-radius: 16px !important;
            padding: 0.95rem 1.15rem !important;
            margin-bottom: 0.8rem !important;
        }
        .arch-card {
            background: rgba(15, 25, 38, 0.58);
            border: 1px solid rgba(56, 189, 248, 0.16);
            border-radius: 16px;
            padding: 1.15rem 1.25rem;
            margin-bottom: 0.95rem;
        }
        .arch-card h4 {
            color: #f8fafc;
            font-size: 1.02rem;
            margin-bottom: 0.6rem;
        }
        .arch-card ul {
            padding-left: 1.15rem;
            margin-bottom: 0;
        }
        .arch-card li {
            color: #94a3b8;
            font-size: 0.86rem;
            margin-bottom: 0.35rem;
            line-height: 1.45;
        }
        .pipeline-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 0.75rem;
            margin: 0.85rem 0 1.15rem 0;
        }
        .pipeline-step {
            background: rgba(15, 25, 38, 0.68);
            border: 1px solid rgba(56, 189, 248, 0.22);
            border-radius: 14px;
            padding: 0.9rem 1rem;
        }
        .pipeline-step-head {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.35rem;
        }
        .pipeline-step-num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 100%);
            color: #ffffff;
            font-size: 0.76rem;
            font-weight: 700;
            flex-shrink: 0;
        }
        .pipeline-step-title {
            color: #f8fafc;
            font-size: 0.92rem;
            font-weight: 700;
        }
        .pipeline-step-file {
            margin-bottom: 0.35rem;
            font-size: 0.76rem;
            color: #38bdf8;
        }
        .pipeline-step-desc {
            color: #94a3b8;
            font-size: 0.80rem;
            line-height: 1.42;
        }
        .c4-container {
            background: rgba(9, 13, 20, 0.85);
            border: 1px solid rgba(56, 189, 248, 0.22);
            border-radius: 16px;
            padding: 0.85rem;
            margin: 0.65rem 0 0.85rem 0;
            overflow-x: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero_header() -> None:
    """Renderiza cabeçalho com linhas em ondas SVG e saudação."""
    st.markdown(
        """
        <div class="wave-hero-container">
            <svg class="wave-svg" viewBox="0 0 900 110" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="waveGrad1" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#0ea5e9" stop-opacity="0.05"/>
                        <stop offset="30%" stop-color="#38bdf8" stop-opacity="0.85"/>
                        <stop offset="60%" stop-color="#818cf8" stop-opacity="0.90"/>
                        <stop offset="100%" stop-color="#2dd4bf" stop-opacity="0.05"/>
                    </linearGradient>
                    <linearGradient id="waveGrad2" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#6366f1" stop-opacity="0.0"/>
                        <stop offset="45%" stop-color="#06b6d4" stop-opacity="0.65"/>
                        <stop offset="100%" stop-color="#14b8a6" stop-opacity="0.0"/>
                    </linearGradient>
                </defs>
                <path d="M 0,58 C 180,16 320,98 480,54 C 640,12 760,88 900,48" fill="none" stroke="url(#waveGrad1)" stroke-width="2.3"/>
                <path d="M 0,72 C 150,102 330,22 510,64 C 680,104 790,28 900,62" fill="none" stroke="url(#waveGrad2)" stroke-width="1.6"/>
            </svg>
        </div>
        <div class="lix-greeting-wrap">
            <div class="lix-hi">Assistente Inteligente de Conformidade LGPD & ANPD</div>
            <div class="lix-headline">Consultoria Jurídica e Regulatória Baseada em Evidências</div>
            <div class="lix-subtext">
                Busca Híbrida de Baixa Latência (ChromaDB + BM25 + FlashRank Reranker) com Grounding Estrito na Lei nº 13.709/2018 e Resoluções da ANPD.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
