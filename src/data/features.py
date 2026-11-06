"""Derivação de features não-sentinela: idade (ponto médio), agrupamento de cirurgia e
radiação (alta cardinalidade -> famílias estáveis) e estadiamento ordinal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    AGE_MIDPOINTS,
    RADIATION_GROUP_LEVELS,
    RADIATION_GROUP_MAP,
    STAGE_ORDER,
    SURGERY_CONSERVATIVE,
    SURGERY_GROUP_LEVELS,
    SURGERY_MASTECTOMY,
    SURGERY_NONE,
    SURGERY_OTHER_UNKNOWN,
)


def decode_age_mid(age: pd.Series) -> pd.Series:
    """`age_mid`: ponto médio numérico da faixa etária de 5 anos (a idade é praticamente
    contínua e ordenada; um one-hot perderia essa ordem). Faixa não mapeada vira NaN
    (isso não acontece na base, e é conferido no critério de aceite).
    """
    return age.map(AGE_MIDPOINTS).astype(float).rename("age_mid")


def unmapped_age(age: pd.Series) -> list[str]:
    """Faixas etárias presentes na base que não estão em AGE_MIDPOINTS (deve ser vazio)."""
    return sorted(set(age.unique()) - set(AGE_MIDPOINTS))


def group_surgery(surgery_code: pd.Series) -> pd.Series:
    """Cirurgia (48 códigos) -> 4 famílias, pela codificação do SEER: 20-24 conservadora,
    40-59 mastectomia, 00 nenhuma; o resto (10-19, 30-39, 60-90, 99, ausente) vira
    outra/desconhecida.
    """
    n = pd.to_numeric(surgery_code, errors="coerce")
    conditions = [n == 0, n.between(20, 24), n.between(40, 59)]
    choices = [SURGERY_NONE, SURGERY_CONSERVATIVE, SURGERY_MASTECTOMY]
    grouped = np.select(conditions, choices, default=SURGERY_OTHER_UNKNOWN)
    return pd.Series(
        pd.Categorical(grouped, categories=SURGERY_GROUP_LEVELS),
        index=surgery_code.index,
        name="surgery_group",
    )


def group_radiation(radiation: pd.Series) -> pd.Series:
    """Radiação (8 categorias) -> 4 níveis. Valor fora do mapa vira NaN; o critério de
    aceite exige que todos os valores estejam mapeados, então um valor inesperado faz o
    script parar.
    """
    grouped = radiation.map(RADIATION_GROUP_MAP)
    return pd.Series(
        pd.Categorical(grouped, categories=RADIATION_GROUP_LEVELS),
        index=radiation.index,
        name="radiation_group",
    )


def clean_stage(stage: pd.Series) -> pd.Series:
    """Estadiamento -> ordinal {in_situ < localized < regional < distant}. As 3 variantes
    "Regional ..." viram `regional`; `Blank(s)` (2000-2003) e
    `Unknown/unstaged/unspecified/DCO` viram NaN. A busca é por trecho do texto, para não
    depender da escrita exata das categorias longas.
    """
    s = stage.astype("string")
    conditions = [
        s.str.contains("In situ", na=False),
        s.str.contains("Localized", na=False),
        s.str.contains("Regional", na=False),
        s.str.contains("Distant", na=False),
    ]
    cleaned = np.select(conditions, STAGE_ORDER, default=None)
    return pd.Series(
        pd.Categorical(cleaned, categories=STAGE_ORDER, ordered=True),
        index=stage.index,
        name="stage",
    )
