"""Modelos de sobrevivência. Todos usam a mesma matriz de variáveis pré-processada e o
mesmo split, para que os C-index possam ser comparados entre si.

Baselines (Etapa 6): Cox, Cox+splines (lifelines), RSF, Gradient Boosting (scikit-survival).
Profundos (Etapa 7): DeepSurv, DeepHit (pycox).
"""
