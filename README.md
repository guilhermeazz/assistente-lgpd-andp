# 🛡️ Assistente Inteligente de Conformidade LGPD & ANPD

Sistema completo de **RAG (Retrieval-Augmented Generation) Híbrido de Baixa Latência** com **Arquitetura Multi-Provedor (Local + Cloud BYOK)** e **Expansão Automática de Referências Cruzadas** para consulta jurídica e regulatória sobre a **Lei Geral de Proteção de Dados Pessoais (Lei nº 13.709/2018 - LGPD)** e **Resoluções/Guias da ANPD**.

---

## 🏗️ Arquitetura e Especificações Técnicas

- **Embeddings 100% Locais (`SentenceTransformers`):**
  - Utiliza `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d Cosine) rodando localmente.
  - **Indexação Única e Independente de Provedor:** O ChromaDB é indexado uma única vez localmente e funciona com qualquer LLM selecionada (Qwen Local, Gemini, ChatGPT ou Claude), reduzindo o tempo de vetorização de queries para milissegundos.
- **Arquitetura Multi-Provedor (`src/rag/providers.py` — Strategy/Factory):**
  1. **`OllamaProvider` (Padrão - 100% Local/Offline):** Conecta ao servidor local Ollama (`http://localhost:11434`) com suporte pré-configurado para `qwen2.5:7b` (padrão), `qwen2.5:3b` (modo ultra-rápido), `qwen2.5:14b` ou qualquer tag local customizada, além de diagnóstico automático (`check_ollama_status()`).
  2. **`GeminiProvider` (BYOK):** Utiliza exclusivamente o SDK oficial `google-genai` (`client.models.generate_content_stream`) com `gemini-2.5-flash`, `gemini-3.8-flash` e `gemini-2.0-flash`.
  3. **`OpenAIProvider` (BYOK):** Utiliza o SDK oficial `openai` (`client.chat.completions.create(..., stream=True)`) com `gpt-4o-mini`, `gpt-4o` e `o3-mini`.
  4. **`AnthropicProvider` (BYOK):** Utiliza o SDK oficial `anthropic` (`client.messages.stream(...)`) com `claude-3-5-haiku-latest` e `claude-sonnet-4-20250514`.
- **Otimizações de Latência no Motor RAG (`src/rag/engine.py`):**
  - **Streaming de Ponta a Ponta (`ask_stream`):** Retorna imediatamente os metadados/fontes recuperadas e um gerador (`Generator`) de tokens para renderização em tempo real via `st.write_stream`, medindo **Recuperação (ms)** e **TTFT (Time To First Token)**.
  - **Reescrita Condicional de Query:** Ignora chamadas extras de LLM quando o histórico está vazio ou quando a pergunta possui 5+ palavras sem pronomes referenciais ambíguos (`ele`, `ela`, `disso`, `desse`, `nisso`, `esse artigo`, `essa lei`). Quando necessária, utiliza `max_tokens=40`.
  - **Orçamento Enxuto de Contexto (*Context Budget*):** `top_k=4` chunks via RRF (`ChromaDB Cosine` + `BM25Okapi`) + até `1` artigo complementar por referência cruzada (`artigos_citados`).

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
│   │   ├── providers.py      # Strategy/Factory Multi-Provedor (Ollama/Qwen, Gemini, OpenAI, Anthropic)
│   │   └── engine.py         # Embeddings locais, Busca Híbrida RRF, Cross-Ref e Streaming (ask_stream)
│   └── app.py                # Interface Streamlit com Cache (@st.cache_resource), Streaming e Telemetria TTFT
├── tests/
│   ├── test_pipeline.py      # Testes automatizados do Parser (incluindo Art. 55-A a 55-L) e Retrieval
│   └── test_performance.py   # Testes de performance (< 500ms), Factory Multi-Provedor e Streaming
├── requirements.txt
└── README.md
```

---

## 🚀 Como Executar

### 1. Ativar o Ambiente Virtual e Instalar Dependências
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Executar os Testes Automatizados (Pipeline + Performance)
```powershell
.\.venv\Scripts\pytest.exe tests/test_pipeline.py tests/test_performance.py -v
```

### 3. Iniciar a Interface Conversacional (Streamlit)
```powershell
.\.venv\Scripts\streamlit.exe run src/app.py
```
