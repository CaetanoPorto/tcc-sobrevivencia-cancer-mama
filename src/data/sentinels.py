"""Decodificação dos códigos especiais ("sentinelas") do SEER.

Números altos no SEER são códigos, não quantidades. Passar essas colunas por um scaler
sem decodificar criaria pacientes com 98 linfonodos positivos e tumores de 99,9 cm. Por
outro lado, apagar todos esses códigos jogaria fora informação clínica real (95/97 =
linfonodo positivo confirmado; faixas 991-995 = faixa de tamanho conhecida). Então o
que dá para decodificar é decodificado, e o resto vira faltante.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    CS_RANGE_MIDPOINTS_MM,
    NODE_COUNT_MAX,
    NODE_STATUS_LEVELS,
    NODE_STATUS_NEGATIVE,
    NODE_STATUS_NOT_ASSESSED,
    NODE_STATUS_POSITIVE,
    NODE_STATUS_UNKNOWN,
    TUMOR_SIZE_CEILING_MM,
    TUMOR_SIZE_DIFFUSE,
)


def decode_node_count(nodes_positive: pd.Series) -> pd.Series:
    """`node_count`: contagem numérica de linfonodos positivos. 00-90 -> valor;
    95/97/98/99 -> NaN (são códigos, não contagens).
    """
    n = pd.to_numeric(nodes_positive, errors="coerce")
    return n.where(n.between(0, NODE_COUNT_MAX), other=np.nan).astype(float)


def decode_nodes_examined(nodes_examined: pd.Series) -> pd.Series:
    """`nodes_examined_n`: nº de linfonodos examinados. 00-90 -> valor; 95/96/97/98/99 -> NaN."""
    n = pd.to_numeric(nodes_examined, errors="coerce")
    return n.where(n.between(0, NODE_COUNT_MAX), other=np.nan).astype(float)


def decode_node_status(nodes_positive: pd.Series) -> pd.Series:
    """`node_status` (4 níveis): recupera os casos com linfonodo positivo que estão nos códigos.

    00           -> negativo
    01-90, 95, 97 -> positivo   (95 = aspiração positiva; 97 = positivos, nº não especificado)
    98           -> nao_avaliado (nenhum linfonodo examinado; não é um faltante aleatório)
    99 / outros  -> desconhecido
    """
    n = pd.to_numeric(nodes_positive, errors="coerce")
    conditions = [
        n == 0,
        n.between(1, NODE_COUNT_MAX) | n.isin([95, 97]),
        n == 98,
    ]
    choices = [NODE_STATUS_NEGATIVE, NODE_STATUS_POSITIVE, NODE_STATUS_NOT_ASSESSED]
    status = np.select(conditions, choices, default=NODE_STATUS_UNKNOWN)
    return pd.Series(
        pd.Categorical(status, categories=NODE_STATUS_LEVELS),
        index=nodes_positive.index,
        name="node_status",
    )


def decode_tumor_size(size: pd.Series) -> pd.Series:
    """`tumor_size_mm`: tamanho em mm a partir do código combinado (CS ∪ Summary).

    000            -> 0 mm (sem tumor primário)
    001-200        -> valor em mm
    991-995        -> ponto médio da faixa (esquema CS)
    990, 996-999 e outros > 200  -> NaN (código especial ou valor implausível)
    """
    n = pd.to_numeric(size, errors="coerce")
    out = pd.Series(np.nan, index=size.index, dtype=float, name="tumor_size_mm")

    # faixas 991-995 -> ponto médio
    for code, mid in CS_RANGE_MIDPOINTS_MM.items():
        out[n == code] = mid
    # 000-200 mm reais (teto de plausibilidade)
    valid = n.between(0, TUMOR_SIZE_CEILING_MM)
    out[valid] = n[valid]
    return out


def tumor_size_diffuse_flag(size: pd.Series) -> pd.Series:
    """Indicador de doença difusa/inflamatória (código 998). Fica guardado à parte e não
    entra no modelo. O tamanho em si vira NaN em `decode_tumor_size`.
    """
    n = pd.to_numeric(size, errors="coerce")
    return (n == TUMOR_SIZE_DIFFUSE).rename("tumor_size_diffuse")
