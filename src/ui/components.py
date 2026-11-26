"""
Estilos CSS Minimalistas, Badges de Tier e Helpers de Privacy by Design / PII (src/ui/components.py)
Implementa:
- Detecção e mascaramento leve por regex de Dados Pessoais (CPF, CNPJ e E-mail)
  antes de enviar consultas para APIs externas (Privacy by Design).
- Tema visual minimalista Dark Obsidian com tipografia sóbria, sem poluição de emojis coloridos.
- Estilização exclusiva para os 2 ícones brancos do topo direito (`.st-key-top_nav_info_btn`,
  `.st-key-top_nav_settings_btn`), cards de sugestões iniciais sem sobreposição e retângulos
  em destaque no modal central de configurações (`@st.dialog`).
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

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
    """Retorna rótulo visual sóbrio e minimalista para o Tier do modelo."""
    t = (tier or "recommended").lower()
    if t == "recommended":
        return "Tier Recomendado · Contexto Completo"
    if t == "economic":
        return "Modo Econômico · Contexto Otimizado"
    return "Não Suportado"


def inject_custom_css() -> None:
    """Injeta estilos CSS minimalistas Dark Obsidian sem conflito de largura entre botões."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(ellipse 68% 36% at 18% 12%, rgba(56, 189, 248, 0.11) 0%, rgba(99, 102, 241, 0.05) 42%, rgba(7, 10, 15, 0.0) 75%),
                radial-gradient(ellipse 65% 38% at 82% 24%, rgba(129, 140, 248, 0.10) 0%, rgba(45, 212, 191, 0.05) 45%, rgba(7, 10, 15, 0.0) 78%),
                radial-gradient(ellipse 80% 45% at 50% 102%, rgba(14, 165, 233, 0.18) 0%, rgba(20, 184, 166, 0.09) 40%, rgba(8, 11, 17, 0.0) 76%),
                linear-gradient(180deg, #06090f 0%, #080d15 52%, #0a121e 100%) !important;
            color: #f1f5f9 !important;
        }
        /* Ondulações coloridas e suavemente desfocadas (ambient aurora waves) no fundo */
        .ambient-bg-waves {
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            overflow: hidden;
        }
        .ambient-wave-blob-1 {
            position: absolute;
            top: 6%;
            left: 50%;
            transform: translateX(-52%) rotate(-4deg);
            width: min(1100px, 96vw);
            height: 210px;
            background: linear-gradient(90deg, rgba(14, 165, 233, 0.18) 0%, rgba(129, 140, 248, 0.20) 48%, rgba(45, 212, 191, 0.16) 100%);
            filter: blur(58px);
            border-radius: 50%;
            opacity: 0.68;
        }
        .ambient-wave-blob-2 {
            position: absolute;
            bottom: 8%;
            left: 50%;
            transform: translateX(-48%) rotate(3deg);
            width: min(980px, 92vw);
            height: 180px;
            background: linear-gradient(90deg, rgba(45, 212, 191, 0.14) 0%, rgba(56, 189, 248, 0.18) 52%, rgba(99, 102, 241, 0.14) 100%);
            filter: blur(64px);
            border-radius: 50%;
            opacity: 0.62;
        }
        .block-container {
            position: relative;
            z-index: 1;
            padding-top: 1.25rem !important;
            padding-bottom: 4.2rem !important;
            max-width: 840px !important;
            margin: 0 auto !important;
        }
        section[data-testid="stSidebar"],
        div[data-testid="collapsedControl"],
        div[data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }
        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 0rem !important;
            min-height: 0rem !important;
        }
        .brand-logo {
            font-size: 1.08rem;
            font-weight: 800;
            letter-spacing: 0.18em;
            color: #ffffff;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
        .brand-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: linear-gradient(135deg, #38bdf8 0%, #2dd4bf 100%);
            box-shadow: 0 0 12px rgba(56, 189, 248, 0.85);
            display: inline-block;
        }
        /* Pílula superior em destaque colorido sofisticado para o modelo e modo ativos */
        .top-status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.24rem 0.78rem;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(14, 165, 233, 0.14) 0%, rgba(129, 140, 248, 0.10) 55%, rgba(45, 212, 191, 0.12) 100%);
            border: 1px solid rgba(56, 189, 248, 0.38);
            box-shadow: 0 0 18px rgba(14, 165, 233, 0.14);
            color: #e2e8f0;
            font-size: 0.75rem;
            font-weight: 500;
            letter-spacing: 0.01em;
            margin-left: 0.65rem;
        }
        .top-status-pill .pill-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #2dd4bf;
            box-shadow: 0 0 8px #2dd4bf;
            display: inline-block;
        }
        .top-status-pill .pill-mode {
            color: #bae6fd;
            font-weight: 600;
        }
        .top-status-pill code {
            color: #38bdf8 !important;
            background: rgba(14, 165, 233, 0.14) !important;
            border: 1px solid rgba(56, 189, 248, 0.25) !important;
            border-radius: 6px !important;
            padding: 0.08rem 0.42rem !important;
            font-size: 0.74rem !important;
            font-weight: 600 !important;
        }
        /* EXCLUSIVO para os 2 ícones minimalistas brancos no canto superior direito */
        .st-key-top_nav_info_btn button,
        .st-key-top_nav_settings_btn button {
            background: rgba(255, 255, 255, 0.03) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.16) !important;
            border-radius: 9px !important;
            width: 36px !important;
            min-width: 36px !important;
            max-width: 36px !important;
            height: 36px !important;
            min-height: 36px !important;
            padding: 0 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: all 0.16s ease !important;
            box-shadow: none !important;
            float: right !important;
        }
        .st-key-top_nav_info_btn button:hover,
        .st-key-top_nav_settings_btn button:hover {
            background: rgba(255, 255, 255, 0.11) !important;
            border-color: rgba(255, 255, 255, 0.65) !important;
            color: #ffffff !important;
        }
        .st-key-top_nav_info_btn button p,
        .st-key-top_nav_settings_btn button p {
            color: #ffffff !important;
            font-size: 1.05rem !important;
            line-height: 1 !important;
            margin: 0 !important;
        }
        /* Cards de perguntas iniciais na Home (sem sobreposição e alinhados à esquerda) */
        .st-key-onboard_btn_0 button,
        .st-key-onboard_btn_1 button,
        .st-key-onboard_btn_2 button,
        .st-key-onboard_btn_3 button {
            width: 100% !important;
            min-height: 52px !important;
            height: auto !important;
            padding: 0.75rem 1.0rem !important;
            background: rgba(15, 22, 34, 0.55) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 11px !important;
            color: #cbd5e1 !important;
            text-align: left !important;
            justify-content: flex-start !important;
            white-space: normal !important;
            transition: all 0.16s ease !important;
        }
        .st-key-onboard_btn_0 button:hover,
        .st-key-onboard_btn_1 button:hover,
        .st-key-onboard_btn_2 button:hover,
        .st-key-onboard_btn_3 button:hover {
            background: linear-gradient(90deg, rgba(14, 165, 233, 0.12) 0%, rgba(45, 212, 191, 0.06) 100%) !important;
            border-color: rgba(56, 189, 248, 0.45) !important;
            box-shadow: 0 4px 18px rgba(14, 165, 233, 0.12) !important;
            color: #ffffff !important;
        }
        .st-key-onboard_btn_0 button p,
        .st-key-onboard_btn_1 button p,
        .st-key-onboard_btn_2 button p,
        .st-key-onboard_btn_3 button p {
            font-size: 0.84rem !important;
            font-weight: 500 !important;
            line-height: 1.38 !important;
            text-align: left !important;
            white-space: normal !important;
        }
        /* Retângulos de seleção no Modal de Configurações (destaque colorido para o setor/modo ativo) */
        div[data-testid="stDialog"] button[kind="primary"] {
            background: linear-gradient(90deg, rgba(14, 165, 233, 0.20) 0%, rgba(129, 140, 248, 0.10) 100%) !important;
            border: 1px solid rgba(56, 189, 248, 0.62) !important;
            box-shadow: inset 3px 0 0 #38bdf8, 0 0 16px rgba(14, 165, 233, 0.16) !important;
            color: #ffffff !important;
            border-radius: 10px !important;
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 0.62rem 0.9rem !important;
            font-weight: 600 !important;
        }
        div[data-testid="stDialog"] button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.025) !important;
            border: 1px solid rgba(255, 255, 255, 0.07) !important;
            color: #94a3b8 !important;
            border-radius: 10px !important;
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 0.62rem 0.9rem !important;
            font-weight: 400 !important;
        }
        div[data-testid="stDialog"] button[kind="secondary"]:hover {
            background: rgba(56, 189, 248, 0.07) !important;
            border-color: rgba(56, 189, 248, 0.28) !important;
            color: #e2e8f0 !important;
        }
        /* Modal Central de Configurações ("Quadrado no Centro" em Split-View) */
        div[data-testid="stDialog"] > div > div,
        div[role="dialog"] {
            background: #0a0f18 !important;
            border: 1px solid rgba(56, 189, 248, 0.24) !important;
            border-radius: 16px !important;
            box-shadow: 0 28px 72px rgba(0, 0, 0, 0.85), 0 0 32px rgba(14, 165, 233, 0.08) !important;
            color: #f8fafc !important;
        }
        .settings-left-title {
            font-size: 0.70rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #38bdf8;
            margin-bottom: 0.5rem;
        }
        .settings-right-header {
            font-size: 0.98rem;
            font-weight: 600;
            color: #f8fafc;
            padding-bottom: 0.5rem;
            margin-bottom: 0.8rem;
            border-bottom: 1px solid rgba(56, 189, 248, 0.18);
        }
        .convergence-card {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.09) 0%, rgba(15, 23, 36, 0.85) 100%);
            border: 1px solid rgba(56, 189, 248, 0.28);
            border-radius: 12px;
            padding: 0.85rem 1.0rem;
            margin-top: 0.65rem;
            margin-bottom: 0.5rem;
        }
        .convergence-title {
            font-size: 0.83rem;
            font-weight: 600;
            color: #38bdf8;
            margin-bottom: 0.25rem;
        }
        .convergence-desc {
            font-size: 0.78rem;
            color: #cbd5e1;
            line-height: 1.42;
            margin-bottom: 0.6rem;
        }
        .convergence-metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.45rem;
        }
        .convergence-metric-item {
            background: rgba(7, 10, 16, 0.75);
            border: 1px solid rgba(56, 189, 248, 0.14);
            border-radius: 8px;
            padding: 0.38rem 0.5rem;
            text-align: center;
        }
        .convergence-metric-label {
            font-size: 0.66rem;
            color: #64748b;
            display: block;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .convergence-metric-val {
            font-size: 0.79rem;
            font-weight: 600;
            color: #38bdf8;
        }
        .wave-hero-container {
            position: relative;
            width: 100%;
            height: 82px;
            overflow: visible;
            margin-top: 0.15rem;
            margin-bottom: 0.45rem;
            background: transparent !important;
        }
        .wave-svg-blur {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            display: block;
            overflow: visible;
            filter: blur(14px);
            opacity: 0.82;
        }
        .wave-svg {
            position: relative;
            width: 100%;
            height: 100%;
            display: block;
            overflow: visible;
            filter: drop-shadow(0 0 12px rgba(56, 189, 248, 0.32));
        }
        .lix-greeting-wrap {
            text-align: center;
            margin-bottom: 1.6rem;
        }
        .lix-hi {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 52%, #2dd4bf 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.35rem;
        }
        .lix-headline {
            font-size: 1.82rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(180deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.45rem;
        }
        .lix-subtext {
            font-size: 0.85rem;
            color: #94a3b8;
            max-width: 560px;
            margin: 0 auto;
            line-height: 1.5;
        }
        .onboard-section-label {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            color: #38bdf8;
            margin-bottom: 0.55rem;
        }
        div[data-testid="stBottom"] {
            background: linear-gradient(180deg, rgba(9, 13, 20, 0.0) 0%, rgba(9, 13, 20, 0.94) 40%, #090d14 100%) !important;
        }
        div[data-testid="stBottom"] > div,
        div[data-testid="stBottomBlockContainer"] {
            max-width: 840px !important;
            width: 100% !important;
            margin: 0 auto !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-bottom: 1.0rem !important;
            box-sizing: border-box !important;
            background: transparent !important;
        }
        div[data-testid="stChatInput"] > div {
            background: #0b101a !important;
            border: 1px solid rgba(56, 189, 248, 0.30) !important;
            box-shadow: 0 0 20px rgba(14, 165, 233, 0.08) !important;
            border-radius: 14px !important;
            padding: 0.22rem 0.45rem !important;
        }
        div[data-testid="stChatMessage"] {
            background: rgba(15, 22, 34, 0.48) !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            border-radius: 14px !important;
            padding: 0.9rem 1.15rem !important;
            margin-bottom: 0.75rem !important;
        }
        .arch-card {
            background: rgba(15, 23, 35, 0.52);
            border: 1px solid rgba(255, 255, 255, 0.09);
            border-radius: 14px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 0.9rem;
        }
        .arch-card h4 {
            color: #f8fafc;
            font-size: 0.96rem;
            margin-bottom: 0.55rem;
        }
        .arch-card ul {
            padding-left: 1.1rem;
            margin-bottom: 0;
        }
        .arch-card li {
            color: #94a3b8;
            font-size: 0.84rem;
            margin-bottom: 0.32rem;
            line-height: 1.45;
        }
        .pipeline-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 0.75rem;
            margin: 0.85rem 0 1.15rem 0;
        }
        .pipeline-step {
            background: rgba(15, 23, 35, 0.62);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 0.85rem 0.95rem;
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
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.12);
            color: #ffffff;
            font-size: 0.72rem;
            font-weight: 600;
            flex-shrink: 0;
        }
        .pipeline-step-title {
            color: #f8fafc;
            font-size: 0.88rem;
            font-weight: 600;
        }
        .pipeline-step-file {
            margin-bottom: 0.32rem;
            font-size: 0.74rem;
            color: #38bdf8;
        }
        .pipeline-step-desc {
            color: #94a3b8;
            font-size: 0.78rem;
            line-height: 1.42;
        }
        .c4-container {
            background: rgba(9, 13, 20, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 14px;
            padding: 0.85rem;
            margin: 0.65rem 0 0.85rem 0;
            overflow-x: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero_header() -> None:
    """Renderiza cabeçalho minimalista com ondulações coloridas e suavemente embaçadas de fundo."""
    st.markdown(
        """
        <div class="ambient-bg-waves">
            <div class="ambient-wave-blob-1"></div>
            <div class="ambient-wave-blob-2"></div>
        </div>
        <div class="wave-hero-container">
            <svg class="wave-svg-blur" viewBox="0 0 900 105" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="waveBlurGrad1" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#0ea5e9" stop-opacity="0.0"/>
                        <stop offset="28%" stop-color="#38bdf8" stop-opacity="0.72"/>
                        <stop offset="55%" stop-color="#818cf8" stop-opacity="0.78"/>
                        <stop offset="82%" stop-color="#2dd4bf" stop-opacity="0.68"/>
                        <stop offset="100%" stop-color="#14b8a6" stop-opacity="0.0"/>
                    </linearGradient>
                    <linearGradient id="waveBlurGrad2" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#6366f1" stop-opacity="0.0"/>
                        <stop offset="45%" stop-color="#06b6d4" stop-opacity="0.65"/>
                        <stop offset="100%" stop-color="#2dd4bf" stop-opacity="0.0"/>
                    </linearGradient>
                </defs>
                <path d="M 0,56 C 175,14 325,96 485,52 C 645,10 765,86 900,46" fill="none" stroke="url(#waveBlurGrad1)" stroke-width="11"/>
                <path d="M 0,70 C 155,100 335,20 515,62 C 685,102 795,26 900,60" fill="none" stroke="url(#waveBlurGrad2)" stroke-width="8"/>
            </svg>
            <svg class="wave-svg" viewBox="0 0 900 105" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="waveGrad1" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#0ea5e9" stop-opacity="0.04"/>
                        <stop offset="28%" stop-color="#38bdf8" stop-opacity="0.88"/>
                        <stop offset="58%" stop-color="#818cf8" stop-opacity="0.92"/>
                        <stop offset="84%" stop-color="#2dd4bf" stop-opacity="0.82"/>
                        <stop offset="100%" stop-color="#2dd4bf" stop-opacity="0.04"/>
                    </linearGradient>
                    <linearGradient id="waveGrad2" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#6366f1" stop-opacity="0.0"/>
                        <stop offset="46%" stop-color="#06b6d4" stop-opacity="0.65"/>
                        <stop offset="100%" stop-color="#14b8a6" stop-opacity="0.0"/>
                    </linearGradient>
                </defs>
                <path d="M 0,56 C 175,14 325,96 485,52 C 645,10 765,86 900,46" fill="none" stroke="url(#waveGrad1)" stroke-width="2.1"/>
                <path d="M 0,70 C 155,100 335,20 515,62 C 685,102 795,26 900,60" fill="none" stroke="url(#waveGrad2)" stroke-width="1.4"/>
            </svg>
        </div>
        <div class="lix-greeting-wrap">
            <div class="lix-hi">Conformidade LGPD &amp; ANPD</div>
            <div class="lix-headline">Consultoria Normativa Baseada em Evidências</div>
            <div class="lix-subtext">
                Recuperação híbrida de baixa latência com ancoragem estrita na Lei nº 13.709/2018 e Resoluções da ANPD.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
