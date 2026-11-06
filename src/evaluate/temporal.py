"""Validação temporal: o teste principal para saber se a era está influenciando o resultado.

Treina em 2010-2015 e avalia em 2016-2017. Se o C-index não cair muito em relação ao
split aleatório, a era não estava sustentando o desempenho. (O teste de previsão da era
na Etapa 5 é só um diagnóstico, sem limiar de aprovação.)

Usa o Cox, que é rápido e transparente. O pré-processamento é ajustado só na janela de
treino (2010-2015).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.era import ERA_COLUMN
from src.evaluate.metrics import harrell_cindex
from src.models.cox import build_cox_frame, cox_risk, fit_cox
from src.preprocessing.transform import INPUT_COLUMNS, build_preprocessor


def temporal_validation_cox(analytic_df: pd.DataFrame, train_eras, eval_eras) -> dict:
    """Ajusta o Cox nas eras de treino e avalia nas de teste. Devolve o C-index temporal
    e os tamanhos das janelas.
    """
    tr = analytic_df[analytic_df[ERA_COLUMN].isin(train_eras)]
    ev_ = analytic_df[analytic_df[ERA_COLUMN].isin(eval_eras)]

    ct = build_preprocessor()
    x_tr = ct.fit_transform(tr[INPUT_COLUMNS]).astype(np.float32)
    x_ev = ct.transform(ev_[INPUT_COLUMNS]).astype(np.float32)
    names = list(ct.get_feature_names_out())

    cox = fit_cox(build_cox_frame(x_tr, names, tr["duration"], tr["event"]))
    risk = cox_risk(cox, build_cox_frame(x_ev, names, ev_["duration"], ev_["event"]))

    return {
        "n_train": int(len(tr)),
        "n_eval": int(len(ev_)),
        "harrell_temporal": harrell_cindex(ev_["duration"].to_numpy(), ev_["event"].to_numpy(), risk),
    }
