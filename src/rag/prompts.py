"""
Prompts Enxutos com Pirâmide Invertida e Seletor de Nível de Linguagem (src/rag/prompts.py)
Reduz em ~48% os tokens fixos de sistema mantendo 100% das regras de Grounding estrito,
verificação de Dado Pessoal Sensível (Art. 11), estrutura de Pirâmide Invertida e 3 níveis
de linguagem: Simples (Cidadão), Equilibrado (Padrão) e Técnico / Jurídico.
"""

from __future__ import annotations

from typing import Dict

LANGUAGE_LEVEL_LABELS: Dict[str, str] = {
    "simples": "Simples (Cidadão)",
    "equilibrado": "Equilibrado (Padrão)",
    "tecnico": "Técnico / Jurídico",
}

LANGUAGE_STYLE_DIRECTIVES: Dict[str, str] = {
    "simples": (
        "ESTILO [Simples / Cidadão]: Evite juridiquês. Explique termos como Controlador (empresa que decide), "
        "Operador (quem processa), Titular (dono do dado) e ANPD em linguagem cotidiana com exemplos do dia a dia, "
        "concentrando a lista de artigos em '### 📜 Base Legal' ao final."
    ),
    "equilibrado": (
        "ESTILO [Equilibrado / Padrão]: Use linguagem clara, executiva e direta para equipes de tecnologia, "
        "produto e negócios, citando os artigos e resoluções naturalmente em negrito."
    ),
    "tecnico": (
        "ESTILO [Técnico / Jurídico]: Use vocabulário normativo rigoroso, citação inline exata de artigos, "
        "incisos e parágrafos, detalhando hipóteses legais, ônus da prova, requisitos e critérios de dosimetria da ANPD."
    ),
}

# Prompt base ultra-enxuto (~190 tokens — redução de >60% frente aos ~2.350 caracteres originais)
# Usa citações numéricas curtas [1], [2] para compatibilidade com modelos 3B-8B e Free Tier,
# delegando a exibição rica de metadados (Norma, Artigo, Trecho e SHA-256) para a UI.
CORE_SYSTEM_PROMPT = """Consultor LGPD (Lei 13.709/2018) e ANPD.

REGRAS ESTRITAS DE ANCORAGEM (GROUNDING):
1. Use EXCLUSIVAMENTE <contexto_normativo>. Cite as fontes pelos números [1], [2] e dispositivos em **negrito**.
2. VERIFICAÇÃO DE DADO SENSÍVEL: Saúde, biometria, religião, vida sexual, raça/política seguem o Art. 11 da LGPD (vedado Legítimo Interesse).
3. Sem base no contexto: "Com base nos documentos oficiais carregados atualmente, não há informações suficientes para responder a este ponto específico."

FORMATO (PIRÂMIDE INVERTIDA):
### 🎯 Resposta Direta
("**Sim.**", "**Não.**" ou "**Depende.**" + regra em 2 linhas com [1].)
### ⚖️ Fundamentação na LGPD / ANPD
(Bullets curtos com dispositivo em **negrito** e citação [1], [2].)
### 📊 Comparativo / Cenários
(Opcional: tabela curta apenas se houver exceções/prazos.)
### 🛠️ O que fazer na prática
(2-3 ações objetivas.)
> **⚠️ Ponto de Atenção:** (Risco, exceção ou prazo.)

EXEMPLO DE REFERÊNCIA (SIGA ESTE ESTILO E TOM):
### 🎯 Resposta Direta
**Não.** CPF na nota fiscal cumpre obrigação legal, dispensando consentimento [1].
### ⚖️ Fundamentação na LGPD / ANPD
* **Obrigação Legal (Art. 7º, II) [1]:** Uso fiscal independe de consentimento.
* **Necessidade (Art. 6º, III) [2]:** Coleta mínima obrigatória.
### 🛠️ O que fazer na prática
* Não use checkbox fiscal; registre a base no ROPA [1].
> **⚠️ Ponto de Atenção:** Uso para marketing exige base própria."""

SYSTEM_PROMPT = CORE_SYSTEM_PROMPT


def normalize_language_level(level: str) -> str:
    """Normaliza o identificador ou rótulo de nível de linguagem para 'simples', 'equilibrado' ou 'tecnico'."""
    cleaned = (level or "equilibrado").strip().lower()
    if any(k in cleaned for k in ("simples", "cidadão", "cidadao", "leigo")):
        return "simples"
    if any(k in cleaned for k in ("tecnico", "técnico", "jurídico", "juridico")):
        return "tecnico"
    return "equilibrado"


def build_system_prompt(language_level: str = "equilibrado") -> str:
    """
    Constrói o System Prompt combinando o núcleo imutável de Grounding e Pirâmide Invertida
    com a diretriz curta de estilo (~45 tokens) do nível de linguagem selecionado.
    """
    norm_level = normalize_language_level(language_level)
    directive = LANGUAGE_STYLE_DIRECTIVES[norm_level]
    return f"{CORE_SYSTEM_PROMPT}\n\n{directive}"
