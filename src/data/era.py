"""Reconstrução da era de diagnóstico a partir do padrão de `Blank(s)`.

O export não traz `Year of diagnosis`, mas metade das colunas só existe a partir de
certo ano, então o período de diagnóstico fica registrado no padrão de vazios. Por isso
este passo vem antes de qualquer limpeza: se os `Blank(s)` fossem limpos primeiro, não
daria mais para reconstruir a era.

A era é um confundidor estrutural: é usada como estrato do split e como restrição de
coorte, nunca como variável preditiva. A mortalidade bruta cai de 59,3% (2000-03) para
9,6% (2018-22) quase só por diferença de tempo de acompanhamento. Um modelo que
enxergasse a era aprenderia o calendário, e não a biologia do tumor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    BLANK_TOKEN,
    BREAST_SUBTYPE_NA_TOKEN,
    COL_BREAST_SUBTYPE,
    COL_GRADE_2018,
    COL_STAGE,
    COL_TUMOR_SIZE_SUMMARY,
    ERA_2000_2003,
    ERA_2004_2009,
    ERA_2010_2015,
    ERA_2016_2017,
    ERA_2018_2022,
    ERA_LABELS,
)

# Nome da coluna derivada.
ERA_COLUMN = "era_diagnostico"


def reconstruct_era(df: pd.DataFrame) -> pd.Series:
    """Devolve a era de diagnóstico (categórica ordenada, 5 níveis) para cada linha.

    Regra de decisão, na ordem de prioridade:

        se   Combined Summary Stage == "Blank(s)"         -> 2000-2003
        senão se Derived Summary Grade 2018 != "Blank(s)" -> 2018-2022
        senão se Tumor Size Summary        != "Blank(s)"  -> 2016-2017
        senão se Breast Subtype != "Recode not available" -> 2010-2015
        senão                                             -> 2004-2009

    `np.select` aplica a primeira condição verdadeira por linha, o que reproduz a
    cadeia de `elif` acima (a prioridade resolve sobreposições — ex.: uma linha
    2000–2003 é capturada pela 1ª condição antes de qualquer outra ser avaliada).

    Precisa das 4 colunas específicas de era ainda brutas (com `Blank(s)` intacto).
    """
    for col in (COL_STAGE, COL_GRADE_2018, COL_TUMOR_SIZE_SUMMARY, COL_BREAST_SUBTYPE):
        if col not in df.columns:
            raise KeyError(
                f"reconstruct_era exige a coluna bruta {col!r}, ausente do DataFrame. "
                "A era precisa ser reconstruída ANTES de qualquer limpeza/renomeação."
            )

    conditions = [
        df[COL_STAGE] == BLANK_TOKEN,
        df[COL_GRADE_2018] != BLANK_TOKEN,
        df[COL_TUMOR_SIZE_SUMMARY] != BLANK_TOKEN,
        df[COL_BREAST_SUBTYPE] != BREAST_SUBTYPE_NA_TOKEN,
    ]
    choices = [ERA_2000_2003, ERA_2018_2022, ERA_2016_2017, ERA_2010_2015]
    era = np.select(conditions, choices, default=ERA_2004_2009)

    return pd.Series(
        pd.Categorical(era, categories=ERA_LABELS, ordered=True),
        index=df.index,
        name=ERA_COLUMN,
    )
