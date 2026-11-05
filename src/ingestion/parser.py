"""
Módulo de Parsing Semântico-Estrutural (src/ingestion/parser.py)
Processa:
1. O HTML compilado da LGPD (BeautifulSoup), removendo tags revogadas/vetadas (<strike>, <s>, <del>
   e CSS line-through), preservando cada Artigo (caput + incisos + alíneas + parágrafos) como um
   chunk atômico com Breadcrumb hierárquico e extração de referências cruzadas (artigos_citados).
2. Os PDFs de Resoluções e Guias da ANPD usando pymupdf4llm (page_chunks=True) para preservar
   tabelas regulatórias em Markdown, segmentando em ~1800 caracteres com overlap de 250.
Salva todos os chunks consolidados em data/processed/chunks_lgpd.jsonl.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
import pymupdf4llm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_JSONL = DEFAULT_PROCESSED_DIR / "chunks_lgpd.jsonl"

# Regex para captura de estrutura hierárquica na LGPD
RE_CAPITULO = re.compile(
    r"^\s*(CAP[ÍI]TULO\s+[IVXLCDM]+(?:\s*[-–—]\s*.*)?)$",
    re.IGNORECASE,
)
RE_SECAO = re.compile(
    r"^\s*(Se[çc][ãa]o\s+[IVXLCDM]+(?:\s*[-–—]\s*.*)?)$",
    re.IGNORECASE,
)
# Captura Art. 1º a Art. 9º (ordinais), Art. 10. a Art. 65. (cardinais) e Art. 55-A. a Art. 55-L. / 58-A / 58-B
RE_ARTIGO_HEADER = re.compile(
    r"^\s*Art\.?\s*(\d{1,2})\s*(?:º|°|o)?(?:-([A-Za-z]))?\.?\s*(.*)$",
    re.IGNORECASE,
)

# Regex para extração de referências cruzadas no corpo do artigo
RE_CITACAO_PLURAL = re.compile(
    r"\barts\.?\s*((?:\d{1,2}\s*(?:º|°|o)?(?:-[A-Za-z])?(?:\s*,\s*|\s+e\s+|\s+ou\s+|\s+a\s+)?)+)",
    re.IGNORECASE,
)
RE_CITACAO_SINGULAR = re.compile(
    r"\bart(?:igo)?\.?\s*(\d{1,2})\s*(?:º|°|o)?(?:-([A-Za-z]))?(?!\s*(?:º|°|o)?\s*da\s+(?:Lei\s+n[º°o]|Constitui[çc][ãa]o|CLT|Consolida[çc][ãa]o))",
    re.IGNORECASE,
)
RE_NUM_ARTIGO_ITEM = re.compile(
    r"(\d{1,2})\s*(?:º|°|o)?(?:-([A-Za-z]))?",
    re.IGNORECASE,
)


def normalize_artigo_id(num_str: str, suffix: Optional[str] = None) -> str:
    """
    Normaliza o identificador do artigo conforme convenção oficial da LGPD:
    - Artigos 1 a 9 usam ordinal: 'Art. 1º' .. 'Art. 9º'
    - Artigos >= 10 usam cardinal: 'Art. 10' .. 'Art. 65'
    - Sufixos de letra em maiúsculo: 'Art. 55-A' .. 'Art. 55-L', 'Art. 58-A', 'Art. 58-B'
    """
    num = int(num_str)
    base = f"Art. {num}º" if 1 <= num <= 9 else f"Art. {num}"
    if suffix:
        return f"{base}-{suffix.upper()}"
    return base


def extract_artigos_citados(texto_artigo: str, self_artigo: str) -> str:
    """
    Identifica referências cruzadas a outros artigos da LGPD mencionados no texto do artigo.
    Retorna string com artigos separados por vírgula (ex: 'Art. 18' ou 'Art. 7º, Art. 11').
    """
    # Remove a declaração inicial do próprio artigo no caput para evitar auto-citação
    corpo = re.sub(
        r"^\s*Art\.?\s*\d{1,2}\s*(?:º|°|o)?(?:-[A-Za-z])?\.?\s*",
        "",
        texto_artigo.strip(),
        count=1,
        flags=re.IGNORECASE,
    )

    citados: List[str] = []

    # 1. Captura listas plurais (ex: 'arts. 7º e 11', 'arts. 52, 53 e 54', 'arts. 55-A, 55-B...')
    for match_plural in RE_CITACAO_PLURAL.finditer(corpo):
        bloco = match_plural.group(1)
        for item in RE_NUM_ARTIGO_ITEM.finditer(bloco):
            num_s, suf = item.group(1), item.group(2)
            num_val = int(num_s)
            if 1 <= num_val <= 65:
                art_id = normalize_artigo_id(num_s, suf)
                if art_id != self_artigo and art_id not in citados:
                    citados.append(art_id)

    # 2. Captura citações singulares (ex: 'no art. 18 desta Lei', 'do art. 9º')
    for match_sing in RE_CITACAO_SINGULAR.finditer(corpo):
        num_s, suf = match_sing.group(1), match_sing.group(2)
        num_val = int(num_s)
        if 1 <= num_val <= 65:
            art_id = normalize_artigo_id(num_s, suf)
            if art_id != self_artigo and art_id not in citados:
                citados.append(art_id)

    return ", ".join(citados)


def remove_revoked_and_struck_elements(soup: BeautifulSoup) -> BeautifulSoup:
    """
    Remove todas as tags <strike>, <s>, <del> e qualquer elemento com CSS line-through
    (usado pelo Planalto para marcar dispositivos revogados ou vetados), protegendo contra
    tags não fechadas do FrontPage/Word que englobem múltiplos artigos vigentes.
    """
    for tag_name in ("script", "style", "head"):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for tag_name in ("strike", "s", "del"):
        for tag in list(soup.find_all(tag_name)):
            if getattr(tag, "decomposed", False):
                continue
            # Proteção contra tag <strike> não fechada no HTML do Planalto que engole vários artigos
            tag_text = tag.get_text(" ", strip=True)
            if len(re.findall(r"\bArt\.\s*\d", tag_text, flags=re.IGNORECASE)) > 2:
                tag.unwrap()
            else:
                tag.decompose()

    # Remove qualquer tag folha/bloco com style contendo line-through ou class indicando tachado
    container_tags = {"html", "body", "div", "table", "tbody", "tr", "td", "section", "main", "article"}
    for tag in list(soup.find_all(True)):
        if getattr(tag, "decomposed", False) or tag.attrs is None:
            continue
        if tag.name in container_tags:
            continue
        style_val = str(tag.get("style", "")).lower()
        class_val = " ".join(tag.get("class", [])).lower() if tag.get("class") else ""
        if "line-through" in style_val or "line-through" in class_val or "strike" in class_val:
            tag_text = tag.get_text(" ", strip=True)
            if len(re.findall(r"\bArt\.\s*\d", tag_text, flags=re.IGNORECASE)) <= 2:
                tag.decompose()

    return soup


def _clean_line(line: str) -> str:
    """Normaliza espaços múltiplos, non-breaking spaces (&nbsp;) e quebras internas de linha."""
    return re.sub(r"\s+", " ", line.replace("\xa0", " ")).strip()


def _parse_lgpd_html_content(raw_html: str, source_filename: str) -> List[Dict[str, Any]]:
    """Extrai os chunks atômicos de artigos da LGPD a partir de uma string HTML."""
    soup = BeautifulSoup(raw_html, "html.parser")
    soup = remove_revoked_and_struck_elements(soup)

    # Extrai blocos textuais de parágrafos e títulos (sem <div> para não duplicar o documento inteiro)
    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all(["p", "h1", "h2", "h3", "h4", "li"])
    ]
    if not paragraphs:
        paragraphs = soup.get_text("\n").splitlines()

    lines: List[str] = []
    for block in paragraphs:
        for subline in block.splitlines():
            cleaned = _clean_line(subline)
            if cleaned:
                lines.append(cleaned)

    # Remove linhas duplicadas adjacentes (comum quando <div> envolve <p> no HTML do Planalto)
    dedup_lines: List[str] = []
    for line in lines:
        if not dedup_lines or dedup_lines[-1] != line:
            dedup_lines.append(line)

    current_capitulo = "CAPÍTULO I"
    current_secao = "Sem Seção"
    pending_capitulo_title = False
    pending_secao_title = False

    articles_data: List[Dict[str, Any]] = []
    current_article: Optional[Dict[str, Any]] = None

    for line in dedup_lines:
        # 1. Detecta CAPÍTULO (pode vir em 1 linha "CAPÍTULO I - ..." ou em 2 linhas no Planalto)
        cap_match = RE_CAPITULO.match(line)
        if cap_match:
            current_capitulo = _clean_line(cap_match.group(1))
            current_secao = "Sem Seção"
            # Se a linha contém apenas "CAPÍTULO X" sem o título descritivo, anexa a próxima linha
            pending_capitulo_title = not bool(re.search(r"[-–—]", current_capitulo))
            pending_secao_title = False
            continue

        if pending_capitulo_title:
            if not RE_ARTIGO_HEADER.match(line) and not RE_SECAO.match(line):
                current_capitulo = f"{current_capitulo} - {line}"
                pending_capitulo_title = False
                continue
            pending_capitulo_title = False

        # 2. Detecta Seção (pode vir em 1 linha "Seção I - ..." ou em 2 linhas no Planalto)
        sec_match = RE_SECAO.match(line)
        if sec_match:
            current_secao = _clean_line(sec_match.group(1))
            pending_secao_title = not bool(re.search(r"[-–—]", current_secao))
            continue

        if pending_secao_title:
            if not RE_ARTIGO_HEADER.match(line) and not RE_CAPITULO.match(line):
                current_secao = f"{current_secao} - {line}"
                pending_secao_title = False
                continue
            pending_secao_title = False

        # 3. Detecta início de Artigo (ignorando citações entre aspas como no Art. 60 que altera o Marco Civil)
        starts_with_quote = line.startswith(('"', "“", "”", "'", "‘"))
        art_match = RE_ARTIGO_HEADER.match(line) if not starts_with_quote else None

        if art_match:
            num_str, suffix, rest = art_match.group(1), art_match.group(2), art_match.group(3)
            num_int = int(num_str)

            # Proteção contra artigos citados internamente no Art. 60 (alterações ao Marco Civil: Art. 7º e Art. 16)
            if (
                current_article is not None
                and current_article["num_int"] == 60
                and num_int < 60
            ):
                current_article["lines"].append(line)
                continue

            if 1 <= num_int <= 65:
                art_id = normalize_artigo_id(num_str, suffix)
                # Reconstrói a linha de abertura do artigo padronizada
                header_prefix = f"{art_id}" if 1 <= num_int <= 9 and not suffix else f"{art_id}."
                formatted_line = f"{header_prefix} {rest}".strip() if rest else header_prefix

                if current_article is not None:
                    articles_data.append(current_article)

                current_article = {
                    "artigo": art_id,
                    "num_int": num_int,
                    "capitulo": current_capitulo,
                    "secao": current_secao,
                    "lines": [formatted_line],
                }
                continue

        # 4. Acumula parágrafos, incisos e alíneas no artigo atual
        if current_article is not None:
            # Ignora rodapé final do Planalto após Art. 65
            if current_article["num_int"] == 65 and any(
                footer_marker in line
                for footer_marker in (
                    "Brasília, 14 de agosto de 2018",
                    "Este texto não substitui o publicado no DOU",
                    "MICHEL TEMER",
                )
            ):
                articles_data.append(current_article)
                current_article = None
                break
            current_article["lines"].append(line)

    if current_article is not None:
        articles_data.append(current_article)

    # Consolida por artigo (caso o HTML tenha blocos repetidos de redação, mantém a versão vigente mais completa)
    consolidated_by_art: Dict[str, Dict[str, Any]] = {}
    for item in articles_data:
        art_id = item["artigo"]
        body_text = "\n".join(item["lines"]).strip()
        if art_id not in consolidated_by_art or len(body_text) >= len(
            consolidated_by_art[art_id]["body_text"]
        ):
            consolidated_by_art[art_id] = {
                **item,
                "body_text": body_text,
            }

    chunks: List[Dict[str, Any]] = []
    for idx, (art_id, art_info) in enumerate(consolidated_by_art.items(), start=1):
        capitulo = art_info["capitulo"]
        secao = art_info["secao"]
        body_text = art_info["body_text"]
        breadcrumb = f"[Lei nº 13.709/2018 (LGPD) | {capitulo} | {secao} | {art_id}]"
        full_text = f"{breadcrumb}\n{body_text}"
        artigos_citados = extract_artigos_citados(body_text, self_artigo=art_id)

        slug_id = (
            art_id.lower()
            .replace("art. ", "art_")
            .replace("º", "")
            .replace("-", "_")
            .strip()
        )
        chunk_id = f"lgpd_{slug_id}"

        chunks.append(
            {
                "id": chunk_id,
                "texto": full_text,
                "metadata": {
                    "fonte": "Lei nº 13.709/2018 (LGPD)",
                    "tipo": "lgpd_artigo",
                    "capitulo": capitulo,
                    "secao": secao,
                    "artigo": art_id,
                    "artigos_citados": artigos_citados,
                    "pagina": idx,
                    "arquivo": source_filename,
                    "breadcrumb": breadcrumb,
                },
            }
        )

    return chunks


def parse_lgpd_html(html_path: Path) -> List[Dict[str, Any]]:
    """
    Realiza o parsing semântico-estrutural do HTML compilado da LGPD (Planalto),
    retornando a lista de chunks atômicos por artigo com Breadcrumb e metadados ricos.
    Em ambientes de nuvem (ex: Streamlit Cloud) onde o WAF do Planalto pode retornar
    páginas de bloqueio/desafio ou HTML truncado (< 65 artigos), aciona automaticamente
    a recomposição da base integral da LGPD Compilada (79 artigos).
    """
    if not html_path.exists():
        raise FileNotFoundError(f"Arquivo HTML da LGPD não encontrado: {html_path}")

    raw_html = html_path.read_text(encoding="utf-8", errors="replace")
    chunks = _parse_lgpd_html_content(raw_html, source_filename=html_path.name)

    # Auto-recuperação caso o arquivo principal lgpd_compilada.html tenha sido sobrescrito por página WAF
    if html_path.name.lower().startswith("lgpd") and len(chunks) < 65:
        logger.warning(
            "Arquivo %s resultou em apenas %d artigos (possível bloqueio WAF/HTML truncado no Cloud). "
            "Restaurando base completa da LGPD Compilada (79 artigos).",
            html_path.name,
            len(chunks),
        )
        from src.ingestion.fetcher import _build_fallback_lgpd_html

        fallback_html = _build_fallback_lgpd_html()
        html_path.write_text(fallback_html, encoding="utf-8")
        chunks = _parse_lgpd_html_content(fallback_html, source_filename=html_path.name)

    logger.info("LGPD HTML processado com sucesso: %d artigos extraídos.", len(chunks))
    return chunks


def _split_markdown_with_overlap(
    text: str,
    chunk_size: int = 1800,
    overlap: int = 250,
) -> List[str]:
    """
    Segmenta o texto Markdown de uma página de PDF em chunks de ~1800 caracteres
    com overlap de 250 caracteres, preservando linhas de tabelas Markdown intactas.
    """
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    segments: List[str] = []
    start = 0
    text_len = len(cleaned)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            # Tenta quebrar em fim de parágrafo ou fim de linha para não cortar células de tabela
            newline_pos = cleaned.rfind("\n\n", start + int(chunk_size * 0.5), end)
            if newline_pos == -1:
                newline_pos = cleaned.rfind("\n", start + int(chunk_size * 0.5), end)
            if newline_pos != -1 and newline_pos > start:
                end = newline_pos

        segment = cleaned[start:end].strip()
        if segment:
            segments.append(segment)

        if end >= text_len:
            break
        start = max(start + 1, end - overlap)

    return segments


def _friendly_anpd_doc_name(pdf_path: Path) -> str:
    """Gera um nome legível para o documento ANPD a partir do nome do arquivo PDF."""
    stem = pdf_path.stem.lower()
    if "dosimetria" in stem or "04_2023" in stem:
        return "Resolução CD/ANPD nº 4/2023 (Dosimetria)"
    if "pequeno_porte" in stem or "02_2022" in stem:
        return "Resolução CD/ANPD nº 2/2022 (Agentes de Pequeno Porte)"
    if "encarregado" in stem or "agentes_tratamento" in stem:
        return "Guia Orientativo ANPD - Agentes de Tratamento e Encarregado"
    return pdf_path.stem.replace("_", " ").title()


def parse_anpd_pdf(
    pdf_path: Path,
    chunk_size: int = 1800,
    overlap: int = 250,
) -> List[Dict[str, Any]]:
    """
    Converte um PDF da ANPD em Markdown por página usando pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    para preservar tabelas regulatórias intactas e segmenta em chunks de ~1800 caracteres com overlap de 250,
    mantendo o Breadcrumb [Nome do Documento ANPD | Página X].
    """
    if not pdf_path.exists():
        logger.warning("PDF não encontrado para parsing: %s", pdf_path)
        return []

    doc_name = _friendly_anpd_doc_name(pdf_path)
    logger.info("Extraindo Markdown com pymupdf4llm de %s (%s)...", pdf_path.name, doc_name)

    page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    chunks: List[Dict[str, Any]] = []

    for page_idx, page_item in enumerate(page_chunks, start=1):
        if isinstance(page_item, dict):
            md_text = page_item.get("text", "")
            meta_info = page_item.get("metadata", {}) or {}
            page_num = int(meta_info.get("page", page_idx))
        else:
            md_text = str(page_item)
            page_num = page_idx

        sub_segments = _split_markdown_with_overlap(
            md_text, chunk_size=chunk_size, overlap=overlap
        )
        for seg_idx, segment in enumerate(sub_segments):
            breadcrumb = f"[{doc_name} | Página {page_num}]"
            full_text = f"{breadcrumb}\n{segment}"
            artigos_citados = extract_artigos_citados(segment, self_artigo="")

            chunk_id = f"anpd_{pdf_path.stem}_p{page_num}_c{seg_idx}"
            chunks.append(
                {
                    "id": chunk_id,
                    "texto": full_text,
                    "metadata": {
                        "fonte": doc_name,
                        "tipo": "anpd_pdf",
                        "capitulo": doc_name,
                        "secao": f"Página {page_num}",
                        "artigo": "",
                        "artigos_citados": artigos_citados,
                        "pagina": page_num,
                        "arquivo": pdf_path.name,
                        "breadcrumb": breadcrumb,
                    },
                }
            )

    logger.info(
        "PDF %s processado: %d chunks gerados.", pdf_path.name, len(chunks)
    )
    return chunks


def run_parser_pipeline(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_jsonl: Path = DEFAULT_OUTPUT_JSONL,
) -> List[Dict[str, Any]]:
    """
    Executa o pipeline completo de parsing (HTML da LGPD + todos os PDFs da ANPD em data/raw/)
    e salva o resultado em data/processed/chunks_lgpd.jsonl.
    """
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    all_chunks: List[Dict[str, Any]] = []

    # 1. Processa HTML da LGPD Compilada
    lgpd_html_path = raw_dir / "lgpd_compilada.html"
    if not lgpd_html_path.exists():
        html_candidates = list(raw_dir.glob("*.htm*"))
        if html_candidates:
            lgpd_html_path = html_candidates[0]

    if lgpd_html_path.exists():
        lgpd_chunks = parse_lgpd_html(lgpd_html_path)
        all_chunks.extend(lgpd_chunks)
    else:
        raise FileNotFoundError(
            f"Nenhum arquivo HTML da LGPD encontrado em {raw_dir}. Execute fetcher.py primeiro."
        )

    # 2. Processa todos os PDFs da ANPD presentes em data/raw/
    pdf_files = sorted(raw_dir.glob("*.pdf"))
    for pdf_file in pdf_files:
        try:
            pdf_chunks = parse_anpd_pdf(pdf_file, chunk_size=1800, overlap=250)
            all_chunks.extend(pdf_chunks)
        except Exception as exc:
            logger.warning("Erro ao processar PDF %s: %s", pdf_file.name, exc)

    # 3. Salva em formato JSONL (UTF-8)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(
        "Pipeline de parsing concluído: %d chunks totais salvos em %s",
        len(all_chunks),
        output_jsonl,
    )
    return all_chunks


if __name__ == "__main__":
    chunks_out = run_parser_pipeline()
    print(f"Total de chunks gerados: {len(chunks_out)}")
