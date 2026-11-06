"""Construção do alvo de sobrevivência (duration, event).

Desfecho = sobrevivência global: `event=1` significa "óbito por qualquer causa", e não
"óbito por câncer de mama". `event` é guardado como inteiro (e não booleano) para que,
se um dia houver a causa do óbito, dê para passar a riscos competitivos trocando
{0,1} por {0,1,2} sem reescrever o resto.

Não há `dropna` aqui. As linhas sem tempo já foram separadas em `cohort.apply_cohort`;
este módulo recebe só casos com tempo conhecido e cria os rótulos.
"""

from __future__ import annotations

import pandas as pd

from src.config import COL_SURVIVAL_MONTHS, COL_VITAL_STATUS, EVENT_MAP

DURATION_COL = "duration"
EVENT_COL = "event"


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva `duration` (meses, float) e `event` (0/1, int) e remove as colunas brutas
    de desfecho. Assume tempo conhecido em todas as linhas (coorte já filtrada).
    """
    df = df.copy()

    duration = pd.to_numeric(df[COL_SURVIVAL_MONTHS], errors="coerce")
    if duration.isna().any():
        n = int(duration.isna().sum())
        raise ValueError(
            f"{n} linhas com `{COL_SURVIVAL_MONTHS}` não numérico chegaram a build_target. "
            "O tempo desconhecido deveria ter sido separado em apply_cohort."
        )

    event = df[COL_VITAL_STATUS].map(EVENT_MAP)
    if event.isna().any():
        bad = df.loc[event.isna(), COL_VITAL_STATUS].unique().tolist()
        raise ValueError(
            f"`{COL_VITAL_STATUS}` com valores fora de {EVENT_MAP}: {bad}"
        )

    df[DURATION_COL] = duration.astype(float)
    df[EVENT_COL] = event.astype(int)
    return df.drop(columns=[COL_SURVIVAL_MONTHS, COL_VITAL_STATUS])
