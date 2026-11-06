"""Leitura do CSV bruto do SEER.

Tudo é lido como texto (`dtype=str`, sem o pandas adivinhar o que é nulo), porque é a
limpeza, mais adiante, que decide o que conta como ausente no SEER. Se o pandas
convertesse antes, esconderia justamente os casos que precisam ser examinados.

Além da leitura completa e da amostrada, há uma leitura em blocos: a Etapa 0 conta os
valores das 1,36 milhão de linhas sem carregar o CSV inteiro na memória.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, RAW_CSV_PATH, READ_CSV_KWARGS


def read_header(path: Path = RAW_CSV_PATH) -> list[str]:
    """Lê apenas o cabeçalho (nomes de coluna, na ordem do arquivo)."""
    return list(pd.read_csv(path, nrows=0, **READ_CSV_KWARGS).columns)


def count_data_rows(path: Path = RAW_CSV_PATH) -> int:
    """Conta as linhas de dados (sem o cabeçalho) numa leitura binária rápida, para não
    depender de um N fixo no código caso o export do SEER*Stat mude.
    """
    with open(path, "rb") as f:
        n_lines = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))
    return n_lines - 1  # desconta o header


def iter_raw_chunks(
    chunksize: int = 200_000, path: Path = RAW_CSV_PATH
) -> Iterator[pd.DataFrame]:
    """Itera o CSV em blocos de `chunksize` linhas (todas as colunas como string).

    Usado para contar valores e conferir invariantes sem carregar o arquivo inteiro;
    quem chama soma as contagens de cada bloco.
    """
    yield from pd.read_csv(path, chunksize=chunksize, **READ_CSV_KWARGS)


def load_raw_full(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Carrega o CSV inteiro (cerca de 1,36 milhão de linhas) na memória. Para testes
    rápidos, `load_raw_sample` é melhor.
    """
    return pd.read_csv(path, **READ_CSV_KWARGS)


def load_raw_sample(
    sample_n: int, seed: int = RANDOM_SEED, path: Path = RAW_CSV_PATH
) -> pd.DataFrame:
    """Amostra `sample_n` linhas do CSV sem carregar o arquivo inteiro.

    Usa `skiprows` com índices sorteados: o leitor ainda percorre o arquivo, mas não
    guarda as linhas não sorteadas. Assim dá para testar em segundos, com pouca memória,
    antes de rodar na base completa.
    """
    n_total = count_data_rows(path)
    sample_n = min(sample_n, n_total)
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(n_total, size=sample_n, replace=False).tolist())

    def _skip(row_idx: int) -> bool:
        if row_idx == 0:  # header
            return False
        return (row_idx - 1) not in chosen

    return pd.read_csv(path, skiprows=_skip, **READ_CSV_KWARGS)
