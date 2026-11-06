"""Monta a base analítica, sempre na mesma ordem.

Todos os scripts montam a coorte por aqui, para a sequência de passos ser sempre a mesma:

    leitura (texto) -> reconstruir era -> inclusão/exclusão -> limpeza SEER
    -> alvo -> (split, fora daqui) -> (fit só no treino, fora daqui)

Nenhum `fit` e nenhum split acontecem neste módulo; isso fica em `preprocessing/`,
depois do split.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.data.cleaning import clean_features
from src.data.cohort import CohortResult, apply_cohort, restrict_to_eras
from src.data.era import ERA_COLUMN, reconstruct_era
from src.data.loading import load_raw_full
from src.data.target import build_target

_UNSET = object()  # sentinela: distingue "não passado" de "None" (= base completa)


def add_era(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a coluna `era_diagnostico`, reconstruída a partir das colunas brutas.
    Precisa acontecer antes de qualquer limpeza de `Blank(s)`.
    """
    df = df.copy()
    df[ERA_COLUMN] = reconstruct_era(df)
    return df


def build_cohort(raw: pd.DataFrame | None = None) -> CohortResult:
    """Etapa 1: carrega o CSV (se não for passado), reconstrói a era, aplica os
    critérios de coorte e cria `duration` e `event`.

    Devolve um `CohortResult` com:
      - `analytic`: coorte feminina com tempo conhecido, com `era_diagnostico`,
        `duration`, `event` e as variáveis ainda brutas (a limpeza é a Etapa 2);
      - `unknown_time`: mulheres com tempo desconhecido, guardadas à parte só para a
        análise de sensibilidade;
      - `flow`: as contagens do fluxo de coorte.
    """
    if raw is None:
        raw = load_raw_full()

    result = apply_cohort(add_era(raw))
    result.analytic = build_target(result.analytic)
    return result


def build_analytic_frame(raw: pd.DataFrame | None = None, era_restriction=_UNSET) -> CohortResult:
    """Etapas 1, 2 e 3: coorte, limpeza SEER e restrição de era.

    Devolve um `CohortResult` com `analytic` já limpo e, se `era_restriction` for uma
    lista de eras, filtrado a elas. `era_restriction=None` usa a base completa; se o
    argumento for omitido, vale `config.ERA_RESTRICTION`. `unknown_time` continua bruto
    (é usado só na sensibilidade). Aqui ainda não há split nem `fit`.
    """
    if era_restriction is _UNSET:
        era_restriction = config.ERA_RESTRICTION

    result = build_cohort(raw)
    result.analytic = clean_features(result.analytic)
    result.analytic = restrict_to_eras(result.analytic, era_restriction)
    return result
