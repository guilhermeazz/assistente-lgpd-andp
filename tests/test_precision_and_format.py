"""
Suíte de Validação de Assertividade Jurídica, FlashRank Reranker e Formato Visual (tests/test_precision_and_format.py)
Valida:
1. Expansor determinístico de sinônimos jurídicos (`expand_legal_query` em `src/rag/legal_expander.py`),
   incluindo ativação de dado pessoal sensível (Art. 11) para "Posso usar legítimo interesse para tratar biometria facial?".
2. Pipeline de Busca em 2 Estágios (ChromaDB + BM25 Top-10 RRF -> FlashRank Cross-Encoder Top-4)
   retornando o `Art. 11` no Top-2 em menos de 600 ms.
3. Presença de todas as seções visuais obrigatórias, regra de dado sensível e exemplo One-Shot no `SYSTEM_PROMPT` (`src/rag/prompts.py`).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.engine import LGPDRagEngine
from src.rag.legal_expander import LEGAL_SYNONYMS_MAP, expand_legal_query
from src.rag.prompts import SYSTEM_PROMPT


def test_expand_legal_query_activates_sensitive_data_art_11() -> None:
    """
    Valida que `expand_legal_query("Posso usar legítimo interesse para tratar biometria facial?")`
    ativa corretamente os termos de dado pessoal sensível e o Art. 11 em ~0 ms.
    """
    query = "Posso usar legítimo interesse para tratar biometria facial?"
    t0 = time.perf_counter()
    expanded_query, matched_concepts = expand_legal_query(query)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 15.0, f"Expansor regex deve rodar em ~0ms, levou {elapsed_ms:.2f}ms"
    assert "dado pessoal sensível" in expanded_query.lower()
    assert "Art. 11" in expanded_query
    assert "Art. 5º II" in expanded_query
    assert len(matched_concepts) >= 1
    assert any("Art. 11" in c for c in matched_concepts)


def test_expand_legal_query_covers_all_ten_mandatory_mappings() -> None:
    """Garante que os 10 mapeamentos obrigatórios do dicionário regex disparam corretamente."""
    assert len(LEGAL_SYNONYMS_MAP) >= 10

    cases = [
        ("Quem é o DPO da empresa?", "Art. 41"),
        ("Tivemos um vazamento e ataque hacker", "Art. 48"),
        ("Cadastro de criança e adolescente no app", "Art. 14"),
        ("Cliente pediu para apagar e excluir a conta", "Art. 18"),
        ("Coleta de biometria facial e dados de saúde", "Art. 11"),
        ("Banner de cookies e pixel de analytics", "Guia Orientativo de Cookies ANPD"),
        ("Qual o valor da multa e sanção administrativa?", "Art. 52"),
        ("Somos uma startup e microempresa (MEI)", "Resolução CD/ANPD nº 2/2022"),
        ("Servidor na AWS no exterior fora do Brasil", "Art. 33"),
        ("Como anonimizar ou aplicar hash na tabela?", "Art. 12"),
    ]
    for sample_q, expected_anchor in cases:
        expanded, concepts = expand_legal_query(sample_q)
        assert expected_anchor in expanded, f"Falha ao expandir '{sample_q}' para '{expected_anchor}'"
        assert len(concepts) >= 1


def test_two_stage_hybrid_plus_flashrank_returns_art_11_in_top2_under_600ms() -> None:
    """
    Valida que o pipeline completo de Busca Híbrida (Top-10 RRF) + FlashRank Reranker (Top-4)
    retorna o `Art. 11` no Top-2 em menos de 600 ms para consulta sobre biometria facial e legítimo interesse.
    """
    engine = LGPDRagEngine()
    if engine.get_indexed_count() == 0:
        engine.index_chunks()

    query = "Posso usar legítimo interesse para tratar biometria facial?"

    # Warm-up do SentenceTransformer e do FlashRank ONNX em memória
    _ = engine.hybrid_search_with_cross_ref(query="Art. 46 segurança", top_k=4, max_cross_refs=1)

    t0 = time.perf_counter()
    results = engine.hybrid_search_with_cross_ref(
        query=query,
        top_k=4,
        max_cross_refs=1,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 600.0, f"Busca Híbrida + Reranker levou {elapsed_ms:.1f} ms (limite: 600 ms)"
    assert len(results) >= 4

    top2_articles = [r["metadata"].get("artigo") for r in results[:2]]
    assert "Art. 11" in top2_articles, (
        f"Esperado 'Art. 11' no Top-2 após Reranker, mas obteve Top-2={top2_articles} "
        f"(todos: {[r['metadata'].get('artigo') for r in results]})"
    )

    # Verifica que todos os chunks principais possuem rerank_score atribuído (> 0)
    for item in results[:4]:
        assert "rerank_score" in item
        assert float(item["rerank_score"]) > 0.0


def test_system_prompt_contains_all_mandatory_visual_sections() -> None:
    """Valida que o SYSTEM_PROMPT em `src/rag/prompts.py` contém todas as seções visuais e regras obrigatórias."""
    required_elements = [
        "REGRAS ESTRITAS DE ANCORAGEM (GROUNDING):",
        "VERIFICAÇÃO DE DADO SENSÍVEL:",
        "Art. 11 da LGPD",
        "### 🎯 Resposta Direta",
        "### ⚖️ Fundamentação na LGPD / ANPD",
        "### 📊 Comparativo / Cenários",
        "### 🛠️ O que fazer na prática",
        "> **⚠️ Ponto de Atenção:**",
        "EXEMPLO DE REFERÊNCIA (SIGA ESTE ESTILO E TOM):",
    ]
    for elem in required_elements:
        assert elem in SYSTEM_PROMPT, f"Elemento obrigatório ausente do SYSTEM_PROMPT: {elem}"
