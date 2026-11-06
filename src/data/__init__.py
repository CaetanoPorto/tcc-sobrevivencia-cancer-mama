"""Camada de dados: leitura, reconstrução de era, coorte, limpeza SEER, alvo.

A ordem de execução importa: a era é reconstruída antes de limpar os `Blank(s)`, os
pares de colunas são unificados antes de decodificar os códigos especiais, e nenhum
ajuste (`fit`) acontece antes do split. Quem garante essa ordem é `src/data/build.py`.
"""
