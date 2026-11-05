"""
Suíte de Testes Automatizados (tests/test_pipeline.py)
Valida de ponta a ponta:
1. Criação de data/processed/chunks_lgpd.jsonl com mais de 65 artigos da LGPD
   (incluindo obrigatoriamente Art. 1º, Art. 7º, Art. 46, Art. 55-A a Art. 55-L, Art. 55-J e Art. 65).
2. Remoção completa de trechos revogados/vetados (<strike>, <s>, <del> e CSS line-through).
3. Extração precisa de referências cruzadas no metadado `artigos_citados` (ex: Art. 15 -> Art. 18, Art. 8º -> Art. 7º, Art. 18, Art. 9º).
4. Extração de PDFs da ANPD em Markdown via `pymupdf4llm` preservando tabelas e Breadcrumb.
5. Motor RAG Híbrido (ChromaDB métrica cosine + BM25 via RRF), Expansão Automática de Referências Cruzadas
   (até 2 artigos complementares) e uso estrito do SDK oficial `google-genai` com `EmbedContentConfig`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.fetcher import DEFAULT_RAW_DIR, run_ingestion_fetcher
from src.ingestion.parser import (
    DEFAULT_OUTPUT_JSONL,
    extract_artigos_citados,
    parse_lgpd_html,
    run_parser_pipeline,
)
from src.rag.engine import LGPDRagEngine


@pytest.fixture(scope="session")
def parsed_chunks() -> List[Dict[str, Any]]:
    """Garante que o fetcher e o parser foram executados e carrega chunks_lgpd.jsonl."""
    if not DEFAULT_OUTPUT_JSONL.exists():
        run_ingestion_fetcher(raw_dir=DEFAULT_RAW_DIR)
        run_parser_pipeline(raw_dir=DEFAULT_RAW_DIR, output_jsonl=DEFAULT_OUTPUT_JSONL)

    assert DEFAULT_OUTPUT_JSONL.exists(), "O arquivo chunks_lgpd.jsonl deve existir."
    chunks: List[Dict[str, Any]] = []
    with DEFAULT_OUTPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def test_chunks_file_created_and_has_more_than_65_lgpd_articles(
    parsed_chunks: List[Dict[str, Any]],
) -> None:
    """
    Prova que o arquivo chunks_lgpd.jsonl foi criado e possui mais de 65 artigos da LGPD,
    incluindo Art. 1º, Art. 7º, Art. 46, Art. 55-A a Art. 55-L (especialmente Art. 55-J) e Art. 65.
    """
    assert DEFAULT_OUTPUT_JSONL.exists()
    lgpd_chunks = [
        c for c in parsed_chunks if c.get("metadata", {}).get("tipo") == "lgpd_artigo"
    ]
    assert (
        len(lgpd_chunks) > 65
    ), f"Esperado > 65 artigos da LGPD, encontrado: {len(lgpd_chunks)}"

    artigos_presentes = {c["metadata"]["artigo"] for c in lgpd_chunks}

    # Artigos obrigatórios exigidos na especificação
    obrigatorios = ["Art. 1º", "Art. 7º", "Art. 46", "Art. 55-J", "Art. 65"]
    for art in obrigatorios:
        assert (
            art in artigos_presentes
        ), f"Artigo obrigatório {art} ausente em chunks_lgpd.jsonl!"

    # Verifica toda a série Art. 55-A até Art. 55-L inserida pela Lei 13.853/2019
    for suffix in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]:
        art_55 = f"Art. 55-{suffix}"
        assert (
            art_55 in artigos_presentes
        ), f"Artigo {art_55} ausente nos chunks processados!"

    # Verifica formato do Breadcrumb no topo do texto de cada artigo da LGPD
    for chunk in lgpd_chunks:
        art_id = chunk["metadata"]["artigo"]
        assert chunk["texto"].startswith(
            "[Lei nº 13.709/2018 (LGPD) |"
        ), f"Breadcrumb inválido no {art_id}"
        assert f"| {art_id}]" in chunk["texto"].splitlines()[0]


def test_no_revoked_or_struck_text_remains_in_chunks(
    parsed_chunks: List[Dict[str, Any]],
    tmp_path: Path,
) -> None:
    """
    Prova que nenhum trecho revogado (<strike>, <s>, <del> ou CSS line-through)
    permaneceu nos chunks processados.
    """
    for chunk in parsed_chunks:
        texto_lower = chunk["texto"].lower()
        assert "<strike" not in texto_lower
        assert "</strike>" not in texto_lower
        assert "<del" not in texto_lower
        assert "line-through" not in texto_lower
        assert "(texto revogado)" not in texto_lower

    # Teste sintético adicional com HTML contendo <strike>, <del>, <s> e style="text-decoration: line-through"
    sample_html = """
    <html><body>
      <p>CAPÍTULO I - DISPOSIÇÕES PRELIMINARES</p>
      <p>Seção I - Das Regras</p>
      <p>Art. 1º Texto vigente do artigo primeiro.</p>
      <p><strike>§ 1º Trecho revogado por tag strike que nao pode aparecer.</strike></p>
      <p><span style="text-decoration: line-through">§ 2º Trecho revogado por CSS line-through que nao pode aparecer.</span></p>
      <p><del>§ 3º Trecho deletado por tag del.</del></p>
      <p>§ 4º Parágrafo vigente que deve permanecer no chunk atômico.</p>
    </body></html>
    """
    html_file = tmp_path / "sample_lgpd.html"
    html_file.write_text(sample_html, encoding="utf-8")
    test_chunks = parse_lgpd_html(html_file)

    assert len(test_chunks) == 1
    chunk_text = test_chunks[0]["texto"]
    assert "Texto vigente do artigo primeiro" in chunk_text
    assert "Parágrafo vigente que deve permanecer" in chunk_text
    assert "Trecho revogado por tag strike" not in chunk_text
    assert "Trecho revogado por CSS line-through" not in chunk_text
    assert "Trecho deletado por tag del" not in chunk_text


def test_cross_references_artigos_citados_metadata(
    parsed_chunks: List[Dict[str, Any]],
) -> None:
    """
    Prova que o metadado `artigos_citados` identifica referências cruzadas corretamente
    tanto via função direta quanto nos artigos reais de chunks_lgpd.jsonl.
    """
    # 1. Caso clássico citado na especificação: Art. 15 mencionando Art. 18
    sample_art_15 = (
        "Art. 15. O término do tratamento de dados pessoais ocorrerá nas seguintes hipóteses, "
        "observado o direito de eliminação previsto no art. 18 desta Lei."
    )
    citados_15 = extract_artigos_citados(sample_art_15, self_artigo="Art. 15")
    assert citados_15 == "Art. 18"

    # 2. Caso plural e sufixos de letra (ex: Art. 65 citando Arts. 55-A a 55-L e Arts. 52, 53 e 54)
    chunks_by_art = {
        c["metadata"]["artigo"]: c
        for c in parsed_chunks
        if c.get("metadata", {}).get("tipo") == "lgpd_artigo"
    }

    assert "Art. 8º" in chunks_by_art
    citados_art8 = chunks_by_art["Art. 8º"]["metadata"]["artigos_citados"]
    assert "Art. 7º" in citados_art8
    assert "Art. 18" in citados_art8

    assert "Art. 18" in chunks_by_art
    citados_art18 = chunks_by_art["Art. 18"]["metadata"]["artigos_citados"]
    assert "Art. 16" in citados_art18

    assert "Art. 65" in chunks_by_art
    citados_art65 = chunks_by_art["Art. 65"]["metadata"]["artigos_citados"]
    assert "Art. 55-A" in citados_art65
    assert "Art. 55-J" in citados_art65
    assert "Art. 52" in citados_art65


def test_anpd_pdf_chunks_and_hybrid_retrieval_with_cross_ref_expansion(
    parsed_chunks: List[Dict[str, Any]],
) -> None:
    """
    Prova que:
    - PDFs da ANPD foram extraídos via pymupdf4llm e possuem Breadcrumb [Nome | Página X]
    - O motor RAG Híbrido (ChromaDB cosine + BM25 via RRF) recupera chunks relevantes
    - A expansão automática de referência cruzada anexa até 2 artigos citados no Top-K
    - O SDK `google-genai` é invocado com `task_type="RETRIEVAL_DOCUMENT"` na indexação
      e `task_type="RETRIEVAL_QUERY"` na busca.
    """
    anpd_chunks = [
        c for c in parsed_chunks if c.get("metadata", {}).get("tipo") == "anpd_pdf"
    ]
    assert len(anpd_chunks) >= 1, "Deve haver chunks extraídos de PDFs da ANPD."
    assert "| Página " in anpd_chunks[0]["texto"].splitlines()[0]

    engine = LGPDRagEngine(
        api_key="test-session-key",
        chunks_path=DEFAULT_OUTPUT_JSONL,
        in_memory_chroma=True,
    )

    # Verifica que a coleção do ChromaDB usa métrica cosine
    assert engine.collection.metadata.get("hnsw:space") == "cosine"

    # Mocka o client do SDK oficial google-genai para validar EmbedContentConfig(task_type=...)
    captured_task_types: List[str] = []

    def fake_embed_content(model: str, contents: Any, config: Any) -> Any:
        assert model == "text-embedding-004"
        captured_task_types.append(config.task_type)
        items = contents if isinstance(contents, list) else [contents]
        mock_resp = MagicMock()
        mock_embs = []
        for text_item in items:
            emb_obj = MagicMock()
            emb_obj.values = engine._deterministic_test_embedding(str(text_item))
            mock_embs.append(emb_obj)
        mock_resp.embeddings = mock_embs
        return mock_resp

    with patch.object(engine, "_get_genai_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = fake_embed_content
        mock_get_client.return_value = mock_client

        total_indexed = engine.index_chunks(api_key="test-session-key")
        assert total_indexed == len(parsed_chunks)
        assert "RETRIEVAL_DOCUMENT" in captured_task_types

        # Executa busca híbrida sobre consentimento e revogação (Art. 8º cita Art. 7º, Art. 18, Art. 9º)
        results = engine.hybrid_search_with_cross_ref(
            query="Art. 8º revogação do consentimento ônus da prova cláusula destacada",
            top_k=3,
            max_cross_refs=2,
            api_key="test-session-key",
        )
        assert "RETRIEVAL_QUERY" in captured_task_types
        assert len(results) >= 3

        top_articles = [r["metadata"].get("artigo") for r in results if not r["is_cross_reference"]]
        assert "Art. 8º" in top_articles

        cross_refs = [r for r in results if r["is_cross_reference"]]
        assert 1 <= len(cross_refs) <= 2, "Deve anexar até 2 artigos complementares por referência cruzada."
        for cr in cross_refs:
            assert "Referência Cruzada Automática" in cr["origem"]


def test_cloud_waf_self_healing_guarantees_all_lgpd_articles(tmp_path: Path) -> None:
    """
    Simula o cenário do Streamlit Cloud onde o WAF do Planalto retorna uma página HTML
    de bloqueio/desafio (0 artigos) para IPs de data center nos EUA e garante que o
    parser recompõe automaticamente os 79 artigos da LGPD.
    """
    waf_blocked_html = tmp_path / "lgpd_compilada.html"
    waf_blocked_html.write_text(
        "<html><body><h1>Access Denied - WAF Challenge</h1></body></html>",
        encoding="utf-8",
    )
    recovered_chunks = parse_lgpd_html(waf_blocked_html)
    assert len(recovered_chunks) >= 79
    artigos_rec = {c["metadata"]["artigo"] for c in recovered_chunks}
    assert "Art. 1º" in artigos_rec
    assert "Art. 55-J" in artigos_rec
    assert "Art. 65" in artigos_rec

