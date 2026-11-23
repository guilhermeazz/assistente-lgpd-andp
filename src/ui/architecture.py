"""
Página de Arquitetura, Pipeline Visual e C4 Model Nível 3 (src/ui/architecture.py)
Implementa:
- Fluxo visual do Pipeline Completo (6 etapas: Coleta & Validação -> Parser Estrutural ->
  Filtro PII & Fast-Path -> Busca Híbrida RRF -> FlashRank Rerank & Token Budget -> Streaming Multi-LLM).
- Diagrama C4 Model — Nível 3 (Diagrama de Componentes) em SVG inline puro responsivo
  + código Mermaid copiável em expander.
- Explorador interativo de todos os chunks de `data/processed/chunks_lgpd.jsonl`.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from src.rag.engine import LGPDRagEngine


def build_pipeline_flow_html() -> str:
    """
    Constrói o HTML compacto (sem linhas em branco internas para preservar o parsing CommonMark
    do Streamlit) representando as 6 etapas ponta a ponta do Pipeline RAG + Ingestão.
    """
    steps = [
        (
            "1",
            "Coleta & Validação",
            "src/ingestion/fetcher.py",
            "Download do Planalto (iso-8859-1 → UTF-8) e PDFs ANPD com magic bytes %PDF-, SHA-256 e contingência anti-WAF.",
        ),
        (
            "2",
            "Parser Estrutural",
            "src/ingestion/parser.py",
            "Remove &lt;strike&gt;/&lt;del&gt;/line-through; gera 79 arts. LGPD + 8 chunks ANPD via pymupdf4llm e extrai artigos_citados.",
        ),
        (
            "3",
            "Filtro PII & Fast-Path",
            "ui/components.py + legal_expander.py",
            "Mascara CPF, CNPJ e E-mail (Privacy by Design), expande sinônimos LGPD (~0ms) e detecta artigos diretos por Regex.",
        ),
        (
            "4",
            "Busca Híbrida (Estágio 1)",
            "src/rag/engine.py",
            "Vetorização local MiniLM-L12-v2 (384-d) no ChromaDB Cosine + BM25Okapi fundidos via Reciprocal Rank Fusion (RRF).",
        ),
        (
            "5",
            "Rerank & Token Budget (Estágio 2)",
            "FlashRank + engine.py",
            "Reordenação Cross-Encoder ONNX (ms-marco-MultiBERT-L-12) + até 1 Cross-Ref + corte rígido max_context_tokens por Tier.",
        ),
        (
            "6",
            "Streaming Multi-LLM",
            "providers.py + prompts.py",
            "Pirâmide Invertida em 3 Níveis de Linguagem com st.write_stream para Gemini, OpenAI, Anthropic ou Ollama Local.",
        ),
    ]
    cards_html = "".join(
        f'<div class="pipeline-step">'
        f'<div class="pipeline-step-head"><span class="pipeline-step-num">{num}</span>'
        f'<span class="pipeline-step-title">{title}</span></div>'
        f'<div class="pipeline-step-file"><code>{mod}</code></div>'
        f'<div class="pipeline-step-desc">{desc}</div>'
        f"</div>"
        for num, title, mod, desc in steps
    )
    return f'<div class="pipeline-grid">{cards_html}</div>'


def build_c4_level3_svg() -> str:
    """
    Gera o diagrama C4 Model — Nível 3 (Diagrama de Componentes) em SVG inline puro
    (sem <foreignObject> e sem linhas em branco internas para compatibilidade total com Streamlit/DOMPurify).
    """
    lines: List[str] = [
        '<div class="c4-container">',
        '<svg viewBox="0 0 1080 820" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="C4 Model Nível 3 - Diagrama de Componentes do Assistente LGPD & ANPD">',
        '<defs>',
        '<marker id="c4Arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">',
        '<path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#38bdf8"/>',
        '</marker>',
        '<marker id="c4ArrowMuted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '<path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#94a3b8"/>',
        '</marker>',
        '<marker id="c4ArrowTeal" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '<path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#2dd4bf"/>',
        '</marker>',
        '</defs>',
        '<rect x="0" y="0" width="1080" height="820" rx="14" fill="#090d14"/>',
        # Pessoa no topo
        '<rect x="370" y="14" width="340" height="66" rx="10" fill="#08427b" stroke="#38bdf8" stroke-width="1.6"/>',
        '<text x="540" y="34" text-anchor="middle" fill="#f8fafc" font-size="13" font-weight="700" font-family="sans-serif">Usuário (Cidadão / DPO / Dev)</text>',
        '<text x="540" y="49" text-anchor="middle" fill="#bae6fd" font-size="10.5" font-family="sans-serif">[Person]</text>',
        '<text x="540" y="66" text-anchor="middle" fill="#e2e8f0" font-size="10.5" font-family="sans-serif">Faz perguntas jurídicas e audita fontes LGPD/ANPD</text>',
        # Container Boundary (Streamlit App)
        '<rect x="20" y="108" width="1040" height="442" rx="14" fill="#0d1522" fill-opacity="0.72" stroke="#38bdf8" stroke-width="1.6" stroke-dasharray="7 5"/>',
        '<text x="38" y="129" fill="#38bdf8" font-size="12" font-weight="700" font-family="sans-serif">Assistente LGPD &amp; ANPD  [Container: Python 3.12 / Streamlit App]</text>',
        # Faixa 1 - Camada UI (src/app.py, src/ui/)
        '<text x="38" y="150" fill="#94a3b8" font-size="10" font-weight="700" font-family="sans-serif">CAMADA DE INTERFACE (src/app.py · src/ui/)</text>',
        '<rect x="38" y="158" width="320" height="78" rx="9" fill="#1e3a5f" stroke="#38bdf8" stroke-width="1.3"/>',
        '<text x="198" y="178" text-anchor="middle" fill="#f8fafc" font-size="12.5" font-weight="700" font-family="sans-serif">UI &amp; Chat Controller</text>',
        '<text x="198" y="194" text-anchor="middle" fill="#7dd3fc" font-size="10" font-family="sans-serif">[Component: app.py, ui/chat.py, ui/sidebar.py]</text>',
        '<text x="198" y="211" text-anchor="middle" fill="#cbd5e1" font-size="10" font-family="sans-serif">Onboarding, st.status, st.write_stream, telemetria</text>',
        '<text x="198" y="226" text-anchor="middle" fill="#94a3b8" font-size="9.5" font-family="sans-serif">Seletor de Provedor, Nível de Linguagem e Auditoria</text>',
        '<rect x="380" y="158" width="320" height="78" rx="9" fill="#1e3a5f" stroke="#38bdf8" stroke-width="1.3"/>',
        '<text x="540" y="178" text-anchor="middle" fill="#f8fafc" font-size="12.5" font-weight="700" font-family="sans-serif">Session &amp; Env Manager</text>',
        '<text x="540" y="194" text-anchor="middle" fill="#7dd3fc" font-size="10" font-family="sans-serif">[Component: ui/state.py]</text>',
        '<text x="540" y="211" text-anchor="middle" fill="#cbd5e1" font-size="10" font-family="sans-serif">@st.cache_data, detecção Cloud vs. Local,</text>',
        '<text x="540" y="226" text-anchor="middle" fill="#94a3b8" font-size="9.5" font-family="sans-serif">callbacks on_provider_change / sync session_state</text>',
        '<rect x="722" y="158" width="320" height="78" rx="9" fill="#1e3a5f" stroke="#38bdf8" stroke-width="1.3"/>',
        '<text x="882" y="178" text-anchor="middle" fill="#f8fafc" font-size="12.5" font-weight="700" font-family="sans-serif">PII Privacy Guard</text>',
        '<text x="882" y="194" text-anchor="middle" fill="#7dd3fc" font-size="10" font-family="sans-serif">[Component: ui/components.py]</text>',
        '<text x="882" y="211" text-anchor="middle" fill="#cbd5e1" font-size="10" font-family="sans-serif">Regex detect_pii / mask_pii (CPF, CNPJ, E-mail)</text>',
        '<text x="882" y="226" text-anchor="middle" fill="#94a3b8" font-size="9.5" font-family="sans-serif">Mascaramento Privacy by Design antes de APIs externas</text>',
        # Faixa 2 - Camada Core RAG (src/rag/)
        '<text x="38" y="264" fill="#94a3b8" font-size="10" font-weight="700" font-family="sans-serif">CAMADA CORE RAG &amp; MULTI-PROVEDOR (src/rag/)</text>',
        '<rect x="38" y="272" width="232" height="92" rx="9" fill="#1168bd" fill-opacity="0.55" stroke="#38bdf8" stroke-width="1.3"/>',
        '<text x="154" y="293" text-anchor="middle" fill="#f8fafc" font-size="12" font-weight="700" font-family="sans-serif">Legal Query Expander</text>',
        '<text x="154" y="309" text-anchor="middle" fill="#bae6fd" font-size="9.5" font-family="sans-serif">[Component: rag/legal_expander.py]</text>',
        '<text x="154" y="328" text-anchor="middle" fill="#e2e8f0" font-size="10" font-family="sans-serif">Sinônimos LGPD (~0ms, 0 tokens)</text>',
        '<text x="154" y="344" text-anchor="middle" fill="#cbd5e1" font-size="9.5" font-family="sans-serif">+ Fast-Path Regex por Artigo</text>',
        '<rect x="292" y="272" width="268" height="92" rx="9" fill="#1168bd" fill-opacity="0.72" stroke="#38bdf8" stroke-width="1.5"/>',
        '<text x="426" y="293" text-anchor="middle" fill="#f8fafc" font-size="12" font-weight="700" font-family="sans-serif">Hybrid RAG &amp; Rerank Engine</text>',
        '<text x="426" y="309" text-anchor="middle" fill="#bae6fd" font-size="9.5" font-family="sans-serif">[Component: rag/engine.py]</text>',
        '<text x="426" y="328" text-anchor="middle" fill="#e2e8f0" font-size="10" font-family="sans-serif">MiniLM + ChromaDB + BM25 RRF</text>',
        '<text x="426" y="344" text-anchor="middle" fill="#cbd5e1" font-size="9.5" font-family="sans-serif">FlashRank ONNX + Cross-Ref + Token Budget</text>',
        '<rect x="582" y="272" width="222" height="92" rx="9" fill="#1168bd" fill-opacity="0.55" stroke="#38bdf8" stroke-width="1.3"/>',
        '<text x="693" y="293" text-anchor="middle" fill="#f8fafc" font-size="12" font-weight="700" font-family="sans-serif">Prompt &amp; Language Builder</text>',
        '<text x="693" y="309" text-anchor="middle" fill="#bae6fd" font-size="9.5" font-family="sans-serif">[Component: rag/prompts.py]</text>',
        '<text x="693" y="328" text-anchor="middle" fill="#e2e8f0" font-size="10" font-family="sans-serif">Grounding + Pirâmide Invertida</text>',
        '<text x="693" y="344" text-anchor="middle" fill="#cbd5e1" font-size="9.5" font-family="sans-serif">+ 3 Níveis de Linguagem (~325 tok)</text>',
        '<rect x="826" y="272" width="216" height="92" rx="9" fill="#1168bd" fill-opacity="0.55" stroke="#38bdf8" stroke-width="1.3"/>',
        '<text x="934" y="293" text-anchor="middle" fill="#f8fafc" font-size="12" font-weight="700" font-family="sans-serif">LLM Provider Factory</text>',
        '<text x="934" y="309" text-anchor="middle" fill="#bae6fd" font-size="9.5" font-family="sans-serif">[Component: rag/providers.py]</text>',
        '<text x="934" y="328" text-anchor="middle" fill="#e2e8f0" font-size="10" font-family="sans-serif">Política de Tiers, Preflight Check,</text>',
        '<text x="934" y="344" text-anchor="middle" fill="#cbd5e1" font-size="9.5" font-family="sans-serif">Streaming generate_stream</text>',
        # Faixa 3 - Camada de Ingestão (src/ingestion/)
        '<text x="38" y="395" fill="#94a3b8" font-size="10" font-weight="700" font-family="sans-serif">CAMADA DE INGESTÃO NORMATIVA (src/ingestion/)</text>',
        '<rect x="590" y="404" width="380" height="82" rx="9" fill="#1e3a5f" stroke="#38bdf8" stroke-width="1.3"/>',
        '<text x="780" y="425" text-anchor="middle" fill="#f8fafc" font-size="12" font-weight="700" font-family="sans-serif">Normative Fetcher</text>',
        '<text x="780" y="441" text-anchor="middle" fill="#7dd3fc" font-size="9.5" font-family="sans-serif">[Component: ingestion/fetcher.py]</text>',
        '<text x="780" y="458" text-anchor="middle" fill="#cbd5e1" font-size="10" font-family="sans-serif">Coleta Planalto/ANPD, validação %PDF-,</text>',
        '<text x="780" y="473" text-anchor="middle" fill="#94a3b8" font-size="9.5" font-family="sans-serif">manifesto SHA-256 e contingência anti-WAF</text>',
        '<rect x="110" y="404" width="390" height="82" rx="9" fill="#1e3a5f" stroke="#38bdf8" stroke-width="1.3"/>',
        '<text x="305" y="425" text-anchor="middle" fill="#f8fafc" font-size="12" font-weight="700" font-family="sans-serif">Semantic-Structural Parser</text>',
        '<text x="305" y="441" text-anchor="middle" fill="#7dd3fc" font-size="9.5" font-family="sans-serif">[Component: ingestion/parser.py]</text>',
        '<text x="305" y="458" text-anchor="middle" fill="#cbd5e1" font-size="10" font-family="sans-serif">BeautifulSoup (remove &lt;strike&gt;/revogados)</text>',
        '<text x="305" y="473" text-anchor="middle" fill="#94a3b8" font-size="9.5" font-family="sans-serif">+ pymupdf4llm (87 chunks + artigos_citados)</text>',
        # Rótulo seção inferior: Armazenamento Local e Sistemas Externos
        '<text x="24" y="576" fill="#2dd4bf" font-size="10.5" font-weight="700" font-family="sans-serif">ARMAZENAMENTO &amp; CACHE LOCAL (Disco / RAM)</text>',
        '<text x="580" y="576" fill="#cbd5e1" font-size="10.5" font-weight="700" font-family="sans-serif">SISTEMAS EXTERNOS &amp; RUNTIMES DE LLM</text>',
        # Armazenamento Local (3 bancos/stores em teal escuro)
        '<rect x="24" y="588" width="168" height="82" rx="10" fill="#134e4a" stroke="#2dd4bf" stroke-width="1.4"/>',
        '<ellipse cx="108" cy="596" rx="72" ry="6" fill="#0f766e" stroke="#2dd4bf" stroke-width="1.1"/>',
        '<text x="108" y="619" text-anchor="middle" fill="#f8fafc" font-size="11.5" font-weight="700" font-family="sans-serif">ChromaDB Vector Store</text>',
        '<text x="108" y="634" text-anchor="middle" fill="#99f6e4" font-size="9" font-family="sans-serif">[Database: SQLite / HNSW]</text>',
        '<text x="108" y="651" text-anchor="middle" fill="#ccfbf1" font-size="9.5" font-family="sans-serif">Métrica Cosine · 384-d</text>',
        '<rect x="204" y="588" width="168" height="82" rx="10" fill="#134e4a" stroke="#2dd4bf" stroke-width="1.4"/>',
        '<text x="288" y="614" text-anchor="middle" fill="#f8fafc" font-size="11.5" font-weight="700" font-family="sans-serif">JSONL &amp; BM25 Corpus</text>',
        '<text x="288" y="630" text-anchor="middle" fill="#99f6e4" font-size="9" font-family="sans-serif">[Store: chunks_lgpd.jsonl]</text>',
        '<text x="288" y="648" text-anchor="middle" fill="#ccfbf1" font-size="9.5" font-family="sans-serif">87 chunks + BM25Okapi</text>',
        '<rect x="384" y="588" width="168" height="82" rx="10" fill="#134e4a" stroke="#2dd4bf" stroke-width="1.4"/>',
        '<text x="468" y="614" text-anchor="middle" fill="#f8fafc" font-size="11.5" font-weight="700" font-family="sans-serif">Local ONNX / Models</text>',
        '<text x="468" y="630" text-anchor="middle" fill="#99f6e4" font-size="9" font-family="sans-serif">[Filesystem: Cache]</text>',
        '<text x="468" y="648" text-anchor="middle" fill="#ccfbf1" font-size="9.5" font-family="sans-serif">MiniLM-L12-v2 + FlashRank</text>',
        # Sistemas Externos (3 sistemas em slate cinza)
        '<rect x="576" y="588" width="152" height="82" rx="10" fill="#334155" stroke="#94a3b8" stroke-width="1.4"/>',
        '<text x="652" y="614" text-anchor="middle" fill="#f8fafc" font-size="11.5" font-weight="700" font-family="sans-serif">Planalto &amp; ANPD</text>',
        '<text x="652" y="630" text-anchor="middle" fill="#cbd5e1" font-size="9" font-family="sans-serif">[External System: gov.br]</text>',
        '<text x="652" y="648" text-anchor="middle" fill="#e2e8f0" font-size="9.5" font-family="sans-serif">HTML LGPD &amp; PDFs Oficiais</text>',
        '<rect x="740" y="588" width="160" height="82" rx="10" fill="#334155" stroke="#94a3b8" stroke-width="1.4"/>',
        '<text x="820" y="614" text-anchor="middle" fill="#f8fafc" font-size="11.5" font-weight="700" font-family="sans-serif">Cloud LLM APIs</text>',
        '<text x="820" y="630" text-anchor="middle" fill="#cbd5e1" font-size="9" font-family="sans-serif">[External System: BYOK]</text>',
        '<text x="820" y="648" text-anchor="middle" fill="#e2e8f0" font-size="9.5" font-family="sans-serif">Gemini / OpenAI / Claude</text>',
        '<rect x="912" y="588" width="148" height="82" rx="10" fill="#334155" stroke="#94a3b8" stroke-width="1.4"/>',
        '<text x="986" y="614" text-anchor="middle" fill="#f8fafc" font-size="11.5" font-weight="700" font-family="sans-serif">Ollama Local Runtime</text>',
        '<text x="986" y="630" text-anchor="middle" fill="#cbd5e1" font-size="9" font-family="sans-serif">[External: Localhost]</text>',
        '<text x="986" y="648" text-anchor="middle" fill="#e2e8f0" font-size="9.5" font-family="sans-serif">Qwen 2.5 (11434)</text>',
        # Conectores direcionais entre Pessoa, UI, RAG, Ingestão, Storage e Externos
        # Pessoa -> UI & Chat Controller
        '<line x1="460" y1="80" x2="240" y2="156" stroke="#38bdf8" stroke-width="1.5" marker-end="url(#c4Arrow)"/>',
        '<rect x="285" y="96" width="84" height="16" rx="4" fill="#090d14"/>',
        '<text x="327" y="108" text-anchor="middle" fill="#bae6fd" font-size="9.5" font-family="sans-serif">HTTPS / UI</text>',
        # UI -> Session Manager & PII Guard
        '<line x1="358" y1="197" x2="378" y2="197" stroke="#38bdf8" stroke-width="1.4" marker-end="url(#c4Arrow)"/>',
        '<line x1="700" y1="197" x2="720" y2="197" stroke="#38bdf8" stroke-width="1.4" marker-end="url(#c4Arrow)"/>',
        # UI & PII Guard -> Hybrid RAG Engine
        '<line x1="250" y1="236" x2="370" y2="270" stroke="#38bdf8" stroke-width="1.4" marker-end="url(#c4Arrow)"/>',
        '<rect x="242" y="242" width="86" height="15" rx="4" fill="#0d1522"/>',
        '<text x="285" y="253" text-anchor="middle" fill="#bae6fd" font-size="9" font-family="sans-serif">PII Mascarado</text>',
        # Core RAG horizontal links
        '<line x1="270" y1="318" x2="290" y2="318" stroke="#38bdf8" stroke-width="1.4" marker-end="url(#c4Arrow)"/>',
        '<line x1="560" y1="318" x2="580" y2="318" stroke="#38bdf8" stroke-width="1.4" marker-end="url(#c4Arrow)"/>',
        '<line x1="804" y1="318" x2="824" y2="318" stroke="#38bdf8" stroke-width="1.4" marker-end="url(#c4Arrow)"/>',
        # Fetcher -> Parser
        '<line x1="590" y1="445" x2="502" y2="445" stroke="#38bdf8" stroke-width="1.4" marker-end="url(#c4Arrow)"/>',
        '<rect x="518" y="427" width="60" height="14" rx="4" fill="#0d1522"/>',
        '<text x="548" y="438" text-anchor="middle" fill="#bae6fd" font-size="8.5" font-family="sans-serif">HTML/PDF</text>',
        # RAG Engine -> ChromaDB, JSONL & Local Models
        '<line x1="360" y1="364" x2="120" y2="586" stroke="#2dd4bf" stroke-width="1.3" stroke-dasharray="4 3" marker-end="url(#c4ArrowTeal)"/>',
        '<rect x="148" y="502" width="76" height="15" rx="4" fill="#090d14"/>',
        '<text x="186" y="513" text-anchor="middle" fill="#99f6e4" font-size="9" font-family="sans-serif">RRF + Cosine</text>',
        '<line x1="426" y1="364" x2="465" y2="586" stroke="#2dd4bf" stroke-width="1.3" stroke-dasharray="4 3" marker-end="url(#c4ArrowTeal)"/>',
        '<rect x="405" y="502" width="84" height="15" rx="4" fill="#090d14"/>',
        '<text x="447" y="513" text-anchor="middle" fill="#99f6e4" font-size="9" font-family="sans-serif">MiniLM / ONNX</text>',
        # Parser -> JSONL Corpus
        '<line x1="295" y1="486" x2="290" y2="586" stroke="#2dd4bf" stroke-width="1.4" marker-end="url(#c4ArrowTeal)"/>',
        '<rect x="262" y="528" width="64" height="15" rx="4" fill="#090d14"/>',
        '<text x="294" y="539" text-anchor="middle" fill="#99f6e4" font-size="9" font-family="sans-serif">87 Chunks</text>',
        # Fetcher -> Planalto & ANPD
        '<line x1="740" y1="486" x2="660" y2="586" stroke="#94a3b8" stroke-width="1.4" marker-end="url(#c4ArrowMuted)"/>',
        '<rect x="652" y="528" width="84" height="15" rx="4" fill="#090d14"/>',
        '<text x="694" y="539" text-anchor="middle" fill="#cbd5e1" font-size="9" font-family="sans-serif">HTTP / SHA-256</text>',
        # Provider Factory -> Cloud LLM APIs & Ollama Local
        '<line x1="910" y1="364" x2="830" y2="586" stroke="#94a3b8" stroke-width="1.4" marker-end="url(#c4ArrowMuted)"/>',
        '<rect x="815" y="502" width="82" height="15" rx="4" fill="#090d14"/>',
        '<text x="856" y="513" text-anchor="middle" fill="#cbd5e1" font-size="9" font-family="sans-serif">Streaming SSE</text>',
        '<line x1="975" y1="364" x2="990" y2="586" stroke="#94a3b8" stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#c4ArrowMuted)"/>',
        '<rect x="945" y="528" width="76" height="15" rx="4" fill="#090d14"/>',
        '<text x="983" y="539" text-anchor="middle" fill="#cbd5e1" font-size="9" font-family="sans-serif">Local Stream</text>',
        # Legenda C4 Model na base
        '<rect x="24" y="704" width="1036" height="96" rx="10" fill="#0d1522" stroke="#1e293b" stroke-width="1.2"/>',
        '<text x="44" y="728" fill="#f8fafc" font-size="11.5" font-weight="700" font-family="sans-serif">Legenda — C4 Model (Nível 3: Diagrama de Componentes)</text>',
        '<rect x="44" y="746" width="18" height="18" rx="4" fill="#08427b" stroke="#38bdf8" stroke-width="1.2"/>',
        '<text x="70" y="759" fill="#cbd5e1" font-size="10.5" font-family="sans-serif">Pessoa (Ator Humano)</text>',
        '<rect x="235" y="746" width="18" height="18" rx="4" fill="#1e3a5f" stroke="#38bdf8" stroke-width="1.2"/>',
        '<text x="261" y="759" fill="#cbd5e1" font-size="10.5" font-family="sans-serif">Componente de UI / Ingestão</text>',
        '<rect x="465" y="746" width="18" height="18" rx="4" fill="#1168bd" fill-opacity="0.72" stroke="#38bdf8" stroke-width="1.2"/>',
        '<text x="491" y="759" fill="#cbd5e1" font-size="10.5" font-family="sans-serif">Componente Core RAG / Factory</text>',
        '<rect x="705" y="746" width="18" height="18" rx="4" fill="#134e4a" stroke="#2dd4bf" stroke-width="1.2"/>',
        '<text x="731" y="759" fill="#cbd5e1" font-size="10.5" font-family="sans-serif">Armazenamento Local</text>',
        '<rect x="895" y="746" width="18" height="18" rx="4" fill="#334155" stroke="#94a3b8" stroke-width="1.2"/>',
        '<text x="921" y="759" fill="#cbd5e1" font-size="10.5" font-family="sans-serif">Sistema Externo</text>',
        '<text x="44" y="784" fill="#64748b" font-size="9.5" font-family="sans-serif">Nota: Embeddings locais (MiniLM-L12-v2) e FlashRank ONNX rodam 100% em CPU/RAM dentro do Container Streamlit via @st.cache_resource.</text>',
        '</svg>',
        '</div>',
    ]
    return "\n".join(lines)


def get_c4_level3_mermaid() -> str:
    """Retorna o diagrama C4 Model Nível 3 equivalente em sintaxe Mermaid."""
    return """flowchart TB
    User["Usuário - Cidadão / DPO / Dev [Person]"]

    subgraph StreamlitApp["Assistente LGPD & ANPD [Container: Python 3.12 / Streamlit App]"]
        direction TB
        UI["UI & Chat Controller [app.py, ui/chat.py, ui/sidebar.py]"]
        State["Session & Env Manager [ui/state.py]"]
        PII["PII Privacy Guard [ui/components.py]"]

        Expander["Legal Query Expander [rag/legal_expander.py]"]
        Engine["Hybrid RAG & Rerank Engine [rag/engine.py]"]
        Prompts["Prompt & Language Builder [rag/prompts.py]"]
        Factory["LLM Provider Factory [rag/providers.py]"]

        Fetcher["Normative Fetcher [ingestion/fetcher.py]"]
        Parser["Semantic-Structural Parser [ingestion/parser.py]"]
    end

    Chroma[("ChromaDB Vector Store [HNSW Cosine 384-d]")]
    Corpus[("JSONL & BM25 Corpus [87 chunks]")]
    Models[("Local ONNX / Model Cache [MiniLM + FlashRank]")]

    GovBr["Planalto & ANPD gov.br [External System]"]
    CloudLLMs["Cloud LLM APIs - Gemini / OpenAI / Anthropic [External System]"]
    Ollama["Ollama Local Runtime - Qwen 2.5 [External Local]"]

    User -->|"HTTPS / UI"| UI
    UI -->|"Sincroniza estado"| State
    UI -->|"Mascara CPF/CNPJ/Email"| PII
    PII -->|"Consulta sanitizada"| Engine
    Engine -->|"Sinônimos ~0ms + Fast-Path"| Expander
    Engine -->|"Busca vetorial"| Chroma
    Engine -->|"Busca lexical BM25"| Corpus
    Engine -->|"Embeddings + Rerank"| Models
    Engine -->|"Monta prompt"| Prompts
    Engine -->|"Streaming generate_stream"| Factory
    Factory -->|"SSE Cloud BYOK/Secrets"| CloudLLMs
    Factory -->|"Stream local"| Ollama
    Fetcher -->|"HTTP + SHA-256 + %PDF-"| GovBr
    Fetcher -->|"HTML/PDF validados"| Parser
    Parser -->|"Grava chunks"| Corpus
"""


def _render_architecture_detail_cards() -> None:
    """Renderiza os 4 cards técnicos detalhando os módulos do sistema."""
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            '<div class="arch-card">'
            '<h4>🌐 1. Coleta Resiliente (<code>src/ingestion/fetcher.py</code>)</h4>'
            '<ul>'
            '<li><strong>Validação Rigorosa:</strong> Verifica magic bytes <code>%PDF-</code>, rejeita páginas HTML de erro 403/404 e garante tamanho mínimo plausível.</li>'
            '<li><strong>Conversão & Integridade:</strong> Baixa a LGPD em <code>iso-8859-1 → UTF-8</code> com manifesto <code>SHA-256</code> e contingência anti-WAF automática.</li>'
            '</ul>'
            '</div>'
            '<div class="arch-card">'
            '<h4>⚖️ 3. Busca Híbrida em 2 Estágios + Fast-Path (<code>src/rag/engine.py</code>)</h4>'
            '<ul>'
            '<li><strong>Fast-Path + Expansor (~0 ms):</strong> Identifica artigos citados por Regex (<code>Art. 18</code>, <code>Art. 55-J</code>) e mapeia gírias corporativas sem custo de tokens.</li>'
            '<li><strong>Retrieve &amp; Rerank:</strong> Top-10 <code>ChromaDB Cosine</code> (<code>MiniLM-L12-v2</code>) + <code>BM25Okapi</code> → RRF → <code>FlashRank ONNX</code> → corte rígido <code>Token Budget</code>.</li>'
            '</ul>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            '<div class="arch-card">'
            '<h4>🧩 2. Parser Semântico-Estrutural (<code>src/ingestion/parser.py</code>)</h4>'
            '<ul>'
            '<li><strong>Limpeza de Revogados:</strong> Decompõe tags <code>&lt;strike&gt;</code>, <code>&lt;s&gt;</code>, <code>&lt;del&gt;</code> e estilos CSS <code>line-through</code>.</li>'
            '<li><strong>Chunking Atômico + PDFs:</strong> Preserva 79 artigos da LGPD inteiros + extrai tabelas das Resoluções ANPD (<code>4/2023</code>, <code>2/2022</code>, <code>15/2024</code>, <code>18/2024</code>) via <code>pymupdf4llm</code>.</li>'
            '</ul>'
            '</div>'
            '<div class="arch-card">'
            '<h4>🤖 4. Multi-LLM Tiers &amp; Privacy Guard (<code>providers.py</code> / <code>src/ui/</code>)</h4>'
            '<ul>'
            '<li><strong>Detecção Cloud vs. Local:</strong> Seleciona provedores Cloud no deploy e habilita <code>Ollama</code> localmente, com callbacks sincronizados de estado.</li>'
            '<li><strong>Streaming &amp; PII:</strong> Políticas <code>recommended</code>/<code>economic</code>, streaming via <code>st.write_stream</code> e mascaramento local de CPF/CNPJ/E-mail antes de APIs externas.</li>'
            '</ul>'
            '</div>',
            unsafe_allow_html=True,
        )


def render_project_details_page(engine: LGPDRagEngine) -> None:
    """Renderiza a página completa de Arquitetura: KPIs, Pipeline, C4 Model Nível 3 e Explorador."""
    st.markdown(
        """
        <div class="lix-greeting-wrap" style="margin-bottom: 1.1rem;">
            <div class="lix-hi">Engenharia de Dados &amp; Sistemas RAG de Baixa Latência</div>
            <div class="lix-headline">Arquitetura &amp; Auditoria da Base Legal</div>
            <div class="lix-subtext">
                Visualize o pipeline completo (Ingestão + RAG Híbrido em 2 Estágios + Streaming),
                o diagrama C4 Model Nível 3 (Componentes) e todos os chunks normativos indexados.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lgpd_chunks = [
        c for c in engine.chunks if c.get("metadata", {}).get("tipo") == "lgpd_artigo"
    ]
    anpd_chunks = [
        c for c in engine.chunks if c.get("metadata", {}).get("tipo") == "anpd_pdf"
    ]
    cross_ref_chunks = [
        c
        for c in engine.chunks
        if str(c.get("metadata", {}).get("artigos_citados", "")).strip()
    ]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📚 Total de Chunks", len(engine.chunks))
    k2.metric("⚖️ Artigos da LGPD", len(lgpd_chunks))
    k3.metric("📄 Chunks ANPD (PDF)", len(anpd_chunks))
    k4.metric("🔗 Com Cross-Reference", len(cross_ref_chunks))

    st.divider()
    st.subheader("⚙️ Pipeline Completo (Ingestão + RAG Híbrido em 2 Estágios + Streaming)")
    st.markdown(build_pipeline_flow_html(), unsafe_allow_html=True)
    _render_architecture_detail_cards()

    st.divider()
    st.subheader("📐 C4 Model — Nível 3 (Diagrama de Componentes)")
    st.caption(
        "Visão estrutural dos componentes internos do container Streamlit (UI, Core RAG Multi-Provedor e Ingestão), "
        "armazenamento local e integrações externas."
    )
    st.markdown(build_c4_level3_svg(), unsafe_allow_html=True)
    with st.expander("📝 Código Mermaid do C4 Model (Nível 3)", expanded=False):
        st.code(get_c4_level3_mermaid(), language="mermaid")

    st.divider()
    st.subheader(
        f"🗂️ Explorador Interativo dos {len(engine.chunks)} Chunks (`data/processed/chunks_lgpd.jsonl`)"
    )
    if not engine.chunks:
        st.info("Nenhum chunk carregado.")
        return

    chunk_options = {
        f"{idx:02d}. {c['metadata'].get('breadcrumb', c['id'])}": c
        for idx, c in enumerate(engine.chunks, start=1)
    }
    selected_label = st.selectbox(
        "Inspecione qualquer artigo da LGPD ou página de Resolução da ANPD:",
        options=list(chunk_options.keys()),
    )
    if selected_label:
        selected_chunk = chunk_options[selected_label]
        c_meta, c_text = st.columns([1, 2])
        with c_meta:
            st.markdown("**📋 Metadados Extraídos:**")
            st.json(selected_chunk["metadata"])
        with c_text:
            st.markdown("**📄 Texto Integral do Chunk:**")
            st.code(selected_chunk["texto"], language="markdown")
