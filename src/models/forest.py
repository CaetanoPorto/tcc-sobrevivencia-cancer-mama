"""Random Survival Forest (scikit-survival): modelo não linear sem rede neural.

Costuma competir de igual para igual com deep learning em dados tabulares, por isso é
uma das comparações mais importantes para saber se as redes trazem algum ganho.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest

from src.config import RANDOM_SEED, RSFConfig


def fit_rsf(X_train: np.ndarray, y_train: np.ndarray, config: RSFConfig = RSFConfig()) -> RandomSurvivalForest:
    model = RandomSurvivalForest(
        n_estimators=config.n_estimators,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


def rsf_risk(model: RandomSurvivalForest, X: np.ndarray, batch: int = 2_000) -> np.ndarray:
    """Escore de risco (soma do hazard cumulativo): maior = maior risco.

    A predição é feita em lotes e em um único processo: o `predict` do scikit-survival
    monta a curva de risco de todas as linhas de uma vez e, em paralelo, cada processo
    aloca a sua cópia, o que estoura a memória com n grande. Lotes pequenos e n_jobs=1
    limitam o uso de memória sem mudar o resultado. (O `fit` continua em paralelo.)
    """
    model.n_jobs = 1
    X = np.asarray(X)
    out = np.empty(len(X), dtype=float)
    for i in range(0, len(X), batch):
        out[i:i + batch] = model.predict(X[i:i + batch])
    return out


def rsf_survival(model: RandomSurvivalForest, X: np.ndarray, batch: int = 2_000) -> pd.DataFrame:
    """Curva S(t) por indivíduo (índice = tempos do modelo, colunas = indivíduos). Feita em
    lotes e em um único processo, pelo mesmo motivo de memória de `rsf_risk`.
    """
    model.n_jobs = 1
    X = np.asarray(X)
    times = getattr(model, "unique_times_", None)
    if times is None:
        times = model.event_times_
    parts = [model.predict_survival_function(X[i:i + batch], return_array=True) for i in range(0, len(X), batch)]
    return pd.DataFrame(np.vstack(parts).T, index=times)
