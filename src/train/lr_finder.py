"""LR finder: teste de faixa da taxa de aprendizado antes do treino.

Faz um treino curto aumentando a taxa aos poucos e observa onde a perda começa a
divergir; sugere uma taxa na região estável, em vez de escolher no chute.
"""

from __future__ import annotations

import numpy as np
import torchtuples as tt

from src.config import TrainConfig


def find_lr(
    model: tt.Model, x_train: np.ndarray, y_train: tuple, config: TrainConfig = TrainConfig()
) -> float:
    """Devolve a taxa sugerida pelo LR finder do torchtuples. Se ele falhar (por exemplo,
    numa amostra muito pequena), usa o valor padrão do config.
    """
    try:
        finder = model.lr_finder(x_train, y_train, batch_size=config.batch_size, tolerance=10)
        best = finder.get_best_lr()
        # limita a uma faixa razoável (o get_best_lr às vezes devolve valores extremos).
        return float(np.clip(best, 1e-4, 1e-1))
    except Exception:
        return config.learning_rate
