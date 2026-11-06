"""Intervalos de confiança por bootstrap (200 reamostragens na avaliação final).

Reamostra os índices do conjunto de avaliação com reposição e recalcula a métrica em
cada reamostra; o intervalo vem dos percentis. A métrica é passada como uma função dos
índices, para funcionar igual com Harrell, Uno (que usa o y de treino fixo) e Antolini.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    metric_on_indices: Callable[[np.ndarray], float],
    n: int,
    n_boot: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Devolve (lo, hi) do IC de (1-alpha) para a métrica, via bootstrap dos índices."""
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            stats.append(metric_on_indices(idx))
        except Exception:
            continue
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
