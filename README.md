# 🛡️ Assistente Inteligente de Conformidade LGPD & ANPD

Sistema completo de **RAG (Retrieval-Augmented Generation) Híbrido** com **Expansão Automática de Referências Cruzadas** para consulta jurídica e regulatória sobre a **Lei Geral de Proteção de Dados Pessoais (Lei nº 13.709/2018 - LGPD)** e **Resoluções/Guias da ANPD**.

---

## 🏗️ Arquitetura e Especificações Técnicas

- **SDK Oficial do Gemini:** Utiliza exclusivamente `google-genai` (`from google import genai` e `from google.genai import types`).
- **Autenticação BYOK (Bring Your Own Key):** Nenhuma chave hardcoded. A `Gemini API Key` é solicitada na barra lateral do Streamlit, armazenada apenas em `st.session_state` e injetada dinamicamente em `genai.Client(api_key=user_key)`.
- **Modelos de Geração e Embeddings:**
  - Geração: Seleção em tempo real entre `gemini-3.8-flash` e `gemini-2.5-flash` com `temperature=0.1`.
  - Embeddings: `text-embedding-004` com `types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")` na indexação e `task_type="RETRIEVAL_QUERY"` na busca.
- **Coleta Resiliente (`src/ingestion/fetcher.py`):**
  - Download da LGPD Compilada direto do Planalto (`iso-8859-1` convertido para `UTF-8`) e de Resoluções/Guias da ANPD em PDF.
  - Verificação de integridade via hash `SHA-256` (só reescreve arquivos alterados).
  - **Fallback Local Obrigatório:** Tolerante a bloqueios de WAF/SSL do `gov.br`.
- **Parser Semântico-Estrutural (`src/ingestion/parser.py`):**
  - **LGPD (HTML):** Remove tags `<strike>`, `<s>`, `<del>` e elementos CSS `line-through` (trechos revogados/vetados). Preserva cada artigo (`Art. 1º` a `Art. 65`, incluindo `Art. 55-A` a `Art. 55-L` e `Art. 58-A`/`58-B`) como um chunk atômico com Breadcrumb `[Lei nº 13.709/2018 (LGPD) | {capitulo} | {secao} | {artigo}]` e metadado `artigos_citados`.
  - **ANPD (PDFs):** Extração em Markdown por página via `pymupdf4llm.to_markdown(pdf_path, page_chunks=True)` preservando tabelas regulatórias, segmentado em chunks de ~1800 caracteres (overlap de 250) com Breadcrumb `[Nome do Documento ANPD | Página X]`.
- **Motor RAG Híbrido + Cross-Reference (`src/rag/engine.py`):**
  - Reescrita de query ambígua com base no histórico de chat.
  - **Reciprocal Rank Fusion (RRF):** Combina busca vetorial no `ChromaDB` (métrica `cosine`) com busca lexical `rank-bm25` (`BM25Okapi`).
  - **Expansão Automática de Referência Cruzada:** Busca no ChromaDB por filtro de metadado até 2 artigos complementares citados no campo `artigos_citados` dos chunks Top-K.

---

## 📂 Estrutura de Diretórios

```text
assistente_lgpd/
├── data/
│   ├── raw/                  # HTML compilado da LGPD e PDFs das Resoluções/Guias da ANPD
│   └── processed/            # Arquivo chunks_lgpd.jsonl gerado pelo parser + banco ChromaDB
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── fetcher.py        # Coleta automatizada com SHA-256, headers anti-bloqueio e fallback local
│   │   └── parser.py         # Chunking Semântico-Estrutural (HTML Planalto + PDFs em Markdown)
│   ├── rag/
│   │   ├── __init__.py
│   │   └── engine.py         # Indexação, Busca Híbrida (Chroma + BM25), Expansor de Artigos e Geração
│   └── app.py                # Interface Streamlit com Chat, Upload de PDFs e Auditoria de Fontes
├── tests/
│   └── test_pipeline.py      # Testes automatizados do Parser (incluindo Art. 55-A a 55-L) e Retrieval
├── requirements.txt          # google-genai, chromadb, pymupdf, pymupdf4llm, beautifulsoup4, rank-bm25, requests, streamlit, pytest
└── README.md
```

---

## 🚀 Como Executar

### 1. Ativar o Ambiente Virtual e Instalar Dependências
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Executar Coleta e Parsing
```powershell
.\.venv\Scripts\python.exe src/ingestion/fetcher.py
.\.venv\Scripts\python.exe src/ingestion/parser.py
```

### 3. Executar os Testes Automatizados
```powershell
.\.venv\Scripts\pytest.exe tests/test_pipeline.py -v
```

### 4. Iniciar a Interface Conversacional (Streamlit)
```powershell
.\.venv\Scripts\streamlit.exe run src/app.py
```
