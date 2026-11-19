"""
Expansor Determinístico de Sinônimos Jurídicos (src/rag/legal_expander.py)
Mapeia em ~0 ms (sem chamadas de LLM) expressões cotidianas, termos de TI e gírias corporativas
para a terminologia normativa exata da LGPD (Lei nº 13.709/2018) e Resoluções/Guias da ANPD,
incluindo seus artigos-âncora.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Dicionário regex determinístico: padrão -> expansão normativa com artigos-âncora
LEGAL_SYNONYMS_MAP: Dict[str, str] = {
    r"\b(dpo|encarregado de dados|responsável lgpd)\b": (
        "encarregado pelo tratamento de dados pessoais comunicação titular ANPD Art. 41"
    ),
    r"\b(vazamento|vazando|vazou|vazar|hackeado|hacker|ataque|exposição de dados|breach)\b": (
        "incidente de segurança comunicação à ANPD e ao titular risco ou dano relevante Art. 46 Art. 48"
    ),
    r"\b(menor|menores|criança|crianças|adolescente|adolescentes|idade)\b": (
        "tratamento de dados pessoais de crianças e de adolescentes melhor interesse consentimento específico Art. 14"
    ),
    r"\b(apagar|excluir|deletar|esquecimento|remover cadastro|cancelar conta)\b": (
        "eliminação dos dados pessoais término do tratamento direito do titular revogação do consentimento Art. 15 Art. 16 Art. 18"
    ),
    r"\b(sensível|sensíveis|saúde|médico|biometria|biométrico|facial|digital|religião|política|sexual|raça|etnia|genético)\b": (
        "dado pessoal sensível hipóteses de tratamento vedação legítimo interesse Art. 5º II Art. 11"
    ),
    r"\b(cookie|cookies|rastreador|pixel|analytics|banner)\b": (
        "cookies estritamente necessários consentimento legítimo interesse Guia Orientativo de Cookies ANPD"
    ),
    r"\b(multa|multas|penalidade|punição|sanção|processo administrativo|infringindo|infringir)\b": (
        "sanções administrativas advertência multa simples 2% faturamento dosimetria Art. 52 Resolução CD/ANPD nº 4/2023"
    ),
    r"\b(startup|mei|pequena empresa|microempresa|pequeno porte|epp)\b": (
        "agentes de tratamento de pequeno porte flexibilização registro simplificado Resolução CD/ANPD nº 2/2022"
    ),
    r"\b(exterior|fora do brasil|nuvem internacional|aws|servidor estrangeiro|transferência internacional)\b": (
        "transferência internacional de dados pessoais país nível de proteção adequado cláusulas-padrão Art. 33"
    ),
    r"\b(anonimizar|anonimização|pseudonimização|mascarar|hash)\b": (
        "dado anonimizado técnica razoável reversão Art. 5º III Art. 12"
    ),
    r"\b(cpf|rg|cnh|endereço|telefone|celular|email|e-mail)\b": (
        "dado pessoal informação relacionada a pessoa natural identificada ou identificável Art. 5º I"
    ),
}

# Rótulos legíveis para exibição na auditoria de Grounding da interface
CONCEPT_LABELS_MAP: Dict[str, str] = {
    r"\b(dpo|encarregado de dados|responsável lgpd)\b": "Encarregado / DPO (Art. 41)",
    r"\b(vazamento|vazando|vazou|vazar|hackeado|hacker|ataque|exposição de dados|breach)\b": "Incidente de Segurança / Vazamento (Art. 46 e Art. 48)",
    r"\b(menor|menores|criança|crianças|adolescente|adolescentes|idade)\b": "Dados de Crianças e Adolescentes (Art. 14)",
    r"\b(apagar|excluir|deletar|esquecimento|remover cadastro|cancelar conta)\b": "Eliminação e Direitos do Titular (Arts. 15, 16 e 18)",
    r"\b(sensível|sensíveis|saúde|médico|biometria|biométrico|facial|digital|religião|política|sexual|raça|etnia|genético)\b": "Dado Pessoal Sensível (Art. 5º, II e Art. 11)",
    r"\b(cookie|cookies|rastreador|pixel|analytics|banner)\b": "Cookies e Rastreamento (Guia ANPD)",
    r"\b(multa|multas|penalidade|punição|sanção|processo administrativo|infringindo|infringir)\b": "Sanções e Dosimetria (Art. 52 e Res. ANPD nº 4/2023)",
    r"\b(startup|mei|pequena empresa|microempresa|pequeno porte|epp)\b": "Agentes de Pequeno Porte (Res. ANPD nº 2/2022)",
    r"\b(exterior|fora do brasil|nuvem internacional|aws|servidor estrangeiro|transferência internacional)\b": "Transferência Internacional (Art. 33)",
    r"\b(anonimizar|anonimização|pseudonimização|mascarar|hash)\b": "Anonimização e Pseudonimização (Art. 5º, III e Art. 12)",
    r"\b(cpf|rg|cnh|endereço|telefone|celular|email|e-mail)\b": "Dado Pessoal Identificável (Art. 5º, I)",
}

_ANCHOR_ARTICLE_REGEX = re.compile(r"\bArt\.\s*(\d+(?:-[A-Z])?(?:º)?)", re.IGNORECASE)


def expand_legal_query(query: str) -> Tuple[str, List[str]]:
    """
    Expande deterministicamente (~0 ms) uma consulta do usuário mapeando termos cotidianos
    para termos técnicos da LGPD/ANPD e seus artigos-âncora.

    Retorna:
        (expanded_query, matched_concepts)
        - expanded_query: pergunta original concatenada aos termos jurídicos correspondentes.
        - matched_concepts: lista com os conceitos/termos identificados para auditoria na interface.
    """
    clean_query = (query or "").strip()
    if not clean_query:
        return "", []

    expansions: List[str] = []
    matched_concepts: List[str] = []

    for pattern, expansion_str in LEGAL_SYNONYMS_MAP.items():
        matches = re.findall(pattern, clean_query, flags=re.IGNORECASE)
        if matches:
            expansions.append(expansion_str)
            label = CONCEPT_LABELS_MAP.get(pattern, expansion_str)
            matched_terms = ", ".join(sorted({m if isinstance(m, str) else m[0] for m in matches}))
            concept_entry = f"{label} [gatilho: '{matched_terms}'] → {expansion_str}"
            matched_concepts.append(concept_entry)

    if not expansions:
        return clean_query, []

    expanded_query = f"{clean_query} {' '.join(expansions)}"
    return expanded_query, matched_concepts


def extract_anchor_articles(text: str) -> List[str]:
    """Extrai identificadores canônicos de artigos (ex: 'Art. 11', 'Art. 5º') presentes na query expandida."""
    found: List[str] = []
    for match in _ANCHOR_ARTICLE_REGEX.finditer(text or ""):
        raw_num = match.group(1).strip()
        try:
            base_num = int(re.match(r"^\d+", raw_num).group(0))  # type: ignore[union-attr]
            if 1 <= base_num <= 9 and "-" not in raw_num and not raw_num.endswith("º"):
                raw_num = f"{raw_num}º"
        except Exception:
            pass
        canonical = f"Art. {raw_num}"
        if canonical not in found:
            found.append(canonical)
    return found
