"""Etapa 2 — limpeza segundo as regras de codificação do SEER.

Ordem dos passos:
  1. igualar o vocabulário e unir os pares de colunas antiga/nova (isso resolve o
     `Blank(s)` estrutural, o que só pode ser feito porque a era já foi reconstruída);
  2. decodificar os códigos especiais de linfonodos e tamanho (sentinels.py);
  3. criar as variáveis agrupadas: idade, cirurgia, radiação, estadiamento (features.py);
  4. trocar por NaN os textos que indicam informação não obtida (Unknown,
     Borderline/Unknown, ...).

`Blank(s)` não é tratado como nulo global, senão a informação de era se perderia. E os
códigos 95/97 de linfonodos (positivo confirmado) e as faixas 991-995 de tamanho são
recuperados, em vez de virarem faltante.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    BLANK_TOKEN,
    CATEGORICAL_FEATURES,
    COL_AGE,
    COL_BREAST_SUBTYPE,
    COL_CHEMOTHERAPY,
    COL_NODES_EXAMINED,
    COL_NODES_POSITIVE,
    COL_RACE,
    COL_RADIATION,
    COL_STAGE,
    COL_SURGERY,
    GRADE_NULL_TOKENS,
    KEPT_NOT_MODELED,
    NUMERIC_FEATURES,
    REAL_NULL_TOKENS,
    REDUNDANT_PAIRS,
)
from src.data.era import ERA_COLUMN
from src.data.features import clean_stage, decode_age_mid, group_radiation, group_surgery
from src.data.sentinels import (
    decode_node_count,
    decode_node_status,
    decode_nodes_examined,
    decode_tumor_size,
    tumor_size_diffuse_flag,
)
from src.data.target import DURATION_COL, EVENT_COL

pd.set_option("future.no_silent_downcasting", True)

LABEL_STRUCTURAL_COLS = [ERA_COLUMN, DURATION_COL, EVENT_COL]


def _mask(series: pd.Series, tokens: set[str]) -> pd.Series:
    """Substitui `tokens` por NaN sem downcast silencioso (isin + where)."""
    return series.where(~series.isin(tokens), other=np.nan)


def harmonize_and_combine_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Une cada par de colunas antiga/nova em uma só, com prioridade para a mais nova.

    `Blank(s)` vira NaN nos dois lados antes de unir (a era já foi salva), assim fica o
    valor real do outro lado. O vocabulário é igualado antes, senão "Positive" e
    "ER positive" virariam categorias diferentes.
    """
    df = df.copy()
    for pair in REDUNDANT_PAIRS:
        old = _mask(df[pair.old_col], {BLANK_TOKEN})
        new = _mask(df[pair.new_col], {BLANK_TOKEN})
        if pair.old_value_map:
            old = old.replace(pair.old_value_map)
        if pair.new_value_map:
            new = new.replace(pair.new_value_map)
        df[pair.output_col] = new.where(new.notna(), old)
        df = df.drop(columns=[pair.old_col, pair.new_col])
    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recebe a coorte analítica (variáveis brutas + era + duration + event) e devolve a
    tabela com todas as variáveis limpas e derivadas, pronta para o split.
    """
    df = harmonize_and_combine_pairs(df)

    # --- Códigos especiais de linfonodos e do tamanho do tumor ---
    df["node_count"] = decode_node_count(df[COL_NODES_POSITIVE])
    df["node_status"] = decode_node_status(df[COL_NODES_POSITIVE])
    df["nodes_examined_n"] = decode_nodes_examined(df[COL_NODES_EXAMINED])
    df["tumor_size_diffuse"] = tumor_size_diffuse_flag(df["tumor_size_mm"])
    df["tumor_size_mm"] = decode_tumor_size(df["tumor_size_mm"])
    df = df.drop(columns=[COL_NODES_POSITIVE, COL_NODES_EXAMINED])

    # --- Features agrupadas ---
    df["age_mid"] = decode_age_mid(df[COL_AGE])
    df["surgery_group"] = group_surgery(df[COL_SURGERY])
    df["radiation_group"] = group_radiation(df[COL_RADIATION])
    df["stage"] = clean_stage(df[COL_STAGE])
    df = df.rename(columns={COL_RACE: "race", COL_BREAST_SUBTYPE: "breast_subtype",
                            COL_CHEMOTHERAPY: "chemotherapy"})
    df = df.drop(columns=[COL_AGE, COL_SURGERY, COL_RADIATION, COL_STAGE])

    # --- Textos de informação não obtida viram NaN nas categóricas ---
    for col in ("race", "er_status", "pr_status", "her2_status", "breast_subtype"):
        df[col] = _mask(df[col], REAL_NULL_TOKENS)
    df["tumor_grade"] = _mask(df["tumor_grade"], REAL_NULL_TOKENS | GRADE_NULL_TOKENS)

    keep = LABEL_STRUCTURAL_COLS + NUMERIC_FEATURES + CATEGORICAL_FEATURES + KEPT_NOT_MODELED + ["tumor_size_diffuse"]
    return df[keep]
