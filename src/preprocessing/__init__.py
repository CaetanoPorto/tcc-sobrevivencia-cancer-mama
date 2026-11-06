"""Pré-processamento: split e transformação (ajustada só no treino).

É aqui que fica a proteção contra vazamento de dados. O split (Etapa 4) divide a coorte
antes de qualquer `fit`, e a transformação (Etapa 5) é ajustada apenas no treino. Uma
versão anterior do código ajustava antes do split e chegava a C-index de 1,0, que é
sinal claro de vazamento.
"""
