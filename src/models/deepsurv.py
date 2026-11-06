"""DeepSurv: MLP + verossimilhança parcial de Cox, via `pycox.models.CoxPH`.

Usa o `MLPVanilla` do torchtuples (a rede padrão do pycox), para não mudar detalhes de
implementação (inicialização dos pesos, ordem de batch norm e dropout) em relação ao
que a literatura usa. Mantém riscos proporcionais, com log-risco não linear.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torchtuples as tt
from pycox.models import CoxPH

from src.config import MLPConfig


def build_net(in_features: int, mlp_config: MLPConfig = MLPConfig()) -> tt.practical.MLPVanilla:
    num_nodes = [mlp_config.hidden_units] * mlp_config.num_layers
    return tt.practical.MLPVanilla(
        in_features, num_nodes, out_features=1,
        batch_norm=mlp_config.batch_norm, dropout=mlp_config.dropout,
        output_bias=False,  # como no DeepSurv original: só a ordem entre escores importa
    )


def build_model(net: tt.practical.MLPVanilla) -> CoxPH:
    return CoxPH(net, tt.optim.Adam)


def make_target(duration, event) -> tuple[np.ndarray, np.ndarray]:
    """pycox espera (durations, events) como float32."""
    return np.asarray(duration, dtype="float32"), np.asarray(event, dtype="float32")


def predict_risk(model: CoxPH, x: np.ndarray) -> np.ndarray:
    """Log do risco relativo: maior valor = maior risco (a convenção de evaluate.metrics)."""
    return model.predict(x).flatten()


def compute_baseline_hazards(model: CoxPH, x_train: np.ndarray, y_train: tuple) -> None:
    """Estima o risco basal (Breslow) com os dados de treino. É necessário antes de
    `predict_survival_function` e nunca é feito com validação ou teste, senão a curva de
    um paciente do teste seria influenciada pelos outros pacientes do teste.
    """
    model.compute_baseline_hazards(input=x_train, target=y_train)


def predict_survival_function(model: CoxPH, x: np.ndarray) -> pd.DataFrame:
    """Curva S(t) por indivíduo (precisa de compute_baseline_hazards antes)."""
    return model.predict_surv_df(x)
