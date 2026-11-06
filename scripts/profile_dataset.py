"""Etapa 0 — conferir se o CSV é a mesma base descrita no trabalho.

Lê o CSV inteiro, confere o cabeçalho (os 22 nomes de coluna) e compara as contagens
(total de linhas, status vital, sexo, eras, invariantes de linfonodos e grau) com os
valores guardados no config.

Critério de aceite: todas as contagens batem. Se não baterem, o CSV mudou; o script
termina com código diferente de 0 e não faz sentido seguir com essa base.

Uso:
    python scripts/profile_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# O console do Windows (cp1252) não mostra alguns acentos e símbolos; isto força UTF-8
# para os prints não quebrarem.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.config import RAW_COLUMNS, RAW_CSV_PATH
from src.checks.invariants import all_passed, check_profile, format_checks
from src.data.loading import read_header
from src.data.profiling import profile_csv


def check_header() -> bool:
    """Confere que o cabeçalho do CSV é exatamente as 22 colunas esperadas, na ordem."""
    measured = read_header()
    expected = RAW_COLUMNS
    if measured == expected:
        print(f"[header] OK — {len(measured)} colunas, nomes e ordem conferem.")
        return True

    print("[header] FALHA — cabeçalho diferente das 22 colunas esperadas:")
    print(f"         esperado: {len(expected)} colunas | medido: {len(measured)} colunas")
    for i, exp in enumerate(expected):
        got = measured[i] if i < len(measured) else "<ausente>"
        if got != exp:
            print(f"         pos {i}: esperado {exp!r} | medido {got!r}")
    for extra in measured[len(expected):]:
        print(f"         coluna extra no CSV: {extra!r}")
    return False


def main() -> int:
    print(f"CSV: {RAW_CSV_PATH}")
    if not RAW_CSV_PATH.exists():
        print(f"[erro] CSV não encontrado em {RAW_CSV_PATH}. Ajuste RAW_CSV_PATH no config.")
        return 2

    header_ok = check_header()

    print("\nPerfilando o CSV inteiro em streaming (pode levar ~1 min)...")
    report = profile_csv()
    checks = check_profile(report)

    print("\n=== Etapa 0 — perfil esperado vs. medido ===")
    print(format_checks(checks))

    profile_ok = all_passed(checks)
    n_fail = sum(1 for c in checks if not c.ok)

    print("\n=== Resultado ===")
    print(f"cabeçalho: {'OK' if header_ok else 'FALHA'}")
    print(f"invariantes: {len(checks) - n_fail}/{len(checks)} OK" + (f" — {n_fail} FALHA(S)" if n_fail else ""))

    if header_ok and profile_ok:
        print("\nCRITÉRIO DE ACEITE DA ETAPA 0: PASSOU [OK]")
        print("O CSV corresponde ao perfil esperado da base. Pode avançar para a Etapa 1.")
        return 0

    print("\nCRITÉRIO DE ACEITE DA ETAPA 0: FALHOU [X]")
    print("O CSV diverge do perfil registrado. Não prossiga com esta base: confira se o")
    print("export do SEER*Stat é o mesmo usado no trabalho.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
