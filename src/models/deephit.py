"""DeepHit: MLP + tempo discreto, via `pycox.models.DeepHitSingle`.

Prevê uma distribuição de probabilidade sobre uma grade discreta de tempo e não
assume riscos proporcionais, o que faz dele o contraponto à violação de
proporcionalidade vista no teste de Schoenfeld (Etapa 6). Os cortes de tempo são
aprendidos só no treino e depois aplicados em validação e teste. Os quantis incluem um
intervalo para t=0, porque há muitos eventos e censuras no mês 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torchtuples as tt
from pycox.models import DeepHitSingle
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

from src.config import DeepHitConfig, MLPConfig


def build_label_transform(config: DeepHitConfig = DeepHitConfig()) -> LabTransDiscreteTime:
    return DeepHitSingle.label_transform(config.num_time_bins, scheme=config.scheme)


def fit_label_transform(labtrans: LabTransDiscreteTime, duration_train, event_train) -> LabTransDiscreteTime:
    """Aprende os cortes de tempo (`cuts`) usando apenas o treino."""
    labtrans.fit(np.asarray(duration_train, dtype="float32"), np.asarray(event_train))
    return labtrans


def make_target(labtrans: LabTransDiscreteTime, duration, event) -> tuple[np.ndarray, np.ndarray]:
    """Discretiza (duration, event) com os cortes já aprendidos no treino."""
    return labtrans.transform(np.asarray(duration, dtype="float32"), np.asarray(event))


def build_net(in_features: int, labtrans: LabTransDiscreteTime, mlp_config: MLPConfig = MLPConfig()) -> tt.practical.MLPVanilla:
    num_nodes = [mlp_config.hidden_units] * mlp_config.num_layers
    return tt.practical.MLPVanilla(
        in_features, num_nodes, out_features=labtrans.out_features,
        batch_norm=mlp_config.batch_norm, dropout=mlp_config.dropout,
    )


def build_model(net: tt.practical.MLPVanilla, labtrans: LabTransDiscreteTime, config: DeepHitConfig = DeepHitConfig()) -> DeepHitSingle:
    return DeepHitSingle(net, tt.optim.Adam, alpha=config.alpha, sigma=config.sigma, duration_index=labtrans.cuts)


def predict_survival_function(model: DeepHitSingle, x: np.ndarray) -> pd.DataFrame:
    return model.predict_surv_df(x)


def predict_risk(model: DeepHitSingle, x: np.ndarray) -> np.ndarray:
    """Escore de risco = menos a soma de S(t) (a área sob a curva aproxima o tempo
    médio de sobrevida restrito; quanto menor, maior o risco).

    `1 - S(t_max)` não funciona aqui: no último intervalo a curva vai a quase 0 para
    quase todo mundo (efeito do tempo discreto) e o ordenamento sai invertido. A soma
    usa a curva inteira e dá um C-index coerente com o Cox e o DeepSurv. Esse escore
    serve para o C-index de Harrell e de Uno; a métrica mais adequada para o DeepHit é
    o C-index de Antolini, que usa a curva completa.
    """
    surv = model.predict_surv_df(x)
    return -surv.sum(axis=0).to_numpy()
