"""Calibração por decil de risco.

Boa discriminação com calibração ruim tem pouca utilidade clínica. Aqui os pacientes
são agrupados em decis pela S(t) prevista e, em cada decil, a sobrevivência prevista
média é comparada com a observada (Kaplan-Meier). Num modelo bem calibrado, previsto e
observado ficam próximos (na diagonal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


def survival_at(surv_df: pd.DataFrame, t: float) -> np.ndarray:
    """S(t) por indivíduo: usa o maior tempo da grade <= t (função escada)."""
    idx = surv_df.index.to_numpy()
    pos = max(int(np.searchsorted(idx, t, side="right")) - 1, 0)
    return surv_df.iloc[pos].to_numpy()


def calibration_by_decile(surv_df: pd.DataFrame, duration, event, horizon: float, n_bins: int = 10) -> pd.DataFrame:
    """DataFrame (decil, n, previsto, observado) da sobrevivência em `horizon`.

    `previsto` = média de S(horizon) no decil. `observado` = S(horizon) por Kaplan-Meier
    dentro do decil (respeita a censura). Decis pela S(horizon) prevista.
    """
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event)
    pred = survival_at(surv_df, horizon)

    bins = pd.qcut(pred, n_bins, labels=False, duplicates="drop")
    rows = []
    for b in sorted(pd.unique(bins)):
        mask = bins == b
        kmf = KaplanMeierFitter().fit(duration[mask], event[mask])
        observed = float(kmf.predict(horizon))
        rows.append({
            "decil": int(b),
            "n": int(mask.sum()),
            "previsto": float(pred[mask].mean()),
            "observado": observed,
        })
    return pd.DataFrame(rows)
