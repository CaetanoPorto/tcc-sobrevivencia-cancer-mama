# Análise de sobrevivência em câncer de mama: deep learning vs. modelos clássicos (SEER)

**Autor:** Guilherme Porto — Informática Biomédica, FMRP-USP
**Orientadora:** Profa. Gleici

Código do Trabalho de Conclusão de Curso. Ele compara modelos clássicos de
sobrevivência com modelos de deep learning na coorte de câncer de mama do SEER
(2000–2022). O desfecho é o óbito por qualquer causa.

| Grupo | Modelo | Biblioteca |
|---|---|---|
| Clássicos | Cox de riscos proporcionais (ridge) | `lifelines` |
| | Cox com splines nas variáveis contínuas | `lifelines` |
| | Random Survival Forest | `scikit-survival` |
| | Gradient Boosting de sobrevivência | `scikit-survival` |
| Deep learning | DeepSurv | `pycox` |
| | DeepHit | `pycox` |

---

## Por onde começar

**`scripts/run_evaluation.py`** gera os resultados finais do trabalho. Ele treina os seis
modelos no mesmo split e produz a tabela comparativa (C-index de Harrell, Uno e
Antolini, Brier e IBS, com intervalo bootstrap), a calibração por decil, as métricas por
era, a validação temporal e a análise de sensibilidade.

Os outros scripts geram os números citados na Metodologia:

| Script | O que gera | Onde aparece no TCC |
|---|---|---|
| `scripts/profile_dataset.py` | Conferência da base: contagens, eras e invariantes | Descrição da base |
| `scripts/run_cohort.py` | Fluxo de inclusão e exclusão da coorte | Metodologia |
| `scripts/run_baselines.py` | Teste de proporcionalidade de Schoenfeld (e os modelos clássicos isolados) | Metodologia |
| `scripts/run_preprocess.py` | Checagens de vazamento e sondagem de era | Metodologia |
| `scripts/run_deep.py` | DeepSurv e DeepHit com 5 sementes, avaliados na validação | Metodologia (estabilidade entre sementes) |
| `scripts/run_evaluation.py` | Resultados finais | Resultados |

Todos os scripts usam o mesmo código de `src/`, então os passos são sempre os mesmos.

---

## Organização

```
├── scripts/                os scripts que geram os resultados (tabela acima)
├── src/
│   ├── config.py           caminhos, nomes de coluna, regras de codificação e hiperparâmetros
│   ├── data/               leitura, reconstrução da era, coorte, limpeza e desfecho
│   ├── preprocessing/      split e pré-processamento (ajustado só no treino)
│   ├── models/             Cox, Cox+splines, RSF, GBS, DeepSurv, DeepHit
│   ├── train/              LR finder e treino com early stopping (redes)
│   ├── evaluate/           métricas, bootstrap, calibração, era, validação temporal
│   └── checks/             critérios de aceite de cada etapa e checagens de vazamento
├── notebook/
│   └── colab_deep.ipynb    roteiro de execução no Google Colab (GPU)
├── dados/
│   └── breast_cancer.dic   dicionário do export do SEER*Stat (as 22 colunas)
└── requirements.txt        versões das bibliotecas
```

Para ler o código, a ordem mais natural é:

1. `src/config.py`
2. `src/data/build.py`, que chama em sequência `era.py` → `cohort.py` → `target.py` → `cleaning.py`
3. `src/preprocessing/split.py` e `transform.py`
4. `src/models/`
5. `scripts/run_evaluation.py`, que junta tudo

---

## Dados

Os dados **não estão no repositório**, porque o termo de uso do SEER não permite
redistribuí-los. A base é um export do SEER*Stat (17 registros, 2000–2022, 1.365.329
linhas e 22 colunas), salvo como `breast_cancer.csv`. O arquivo `dados/breast_cancer.dic`
descreve as colunas.

Para rodar, indique o caminho do CSV numa variável de ambiente:

```bash
export SEER_CSV_PATH="/caminho/para/breast_cancer.csv"
```

---

## Como rodar

### No Google Colab (GPU), onde os modelos são treinados

Abra `notebook/colab_deep.ipynb` e rode as células em ordem. O notebook clona este
repositório, monta o Google Drive (onde está o CSV) e executa:

```bash
python scripts/profile_dataset.py
python scripts/run_deep.py --full --seeds 5
python scripts/run_evaluation.py --sample-n 100000 --eval-set test --n-boot 200   # comparação principal
python scripts/run_evaluation.py --full --eval-set test --n-boot 200              # sensibilidade, base inteira
```

### No computador (CPU)

```bash
python -m venv .venv
source .venv/bin/activate          # no Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/profile_dataset.py                     # confere a base
python scripts/run_evaluation.py --sample-n 15000     # teste rápido, roda na validação
```

Os scripts rodam por padrão numa amostra, para testar em poucos minutos. A base inteira
só é usada com `--full`.

---

## Principais decisões de método (onde estão no código)

- **Desfecho: óbito por qualquer causa.** O export não traz a causa do óbito.
  (`src/config.py`, `src/data/target.py`)
- **Reconstrução da era de diagnóstico.** O export não tem o ano de diagnóstico, mas as
  colunas que só existem a partir de certo ano deixam `Blank(s)` nos períodos
  anteriores. O padrão desses vazios identifica a era (2000-03, 2004-09, 2010-15,
  2016-17, 2018-22). Isso é feito antes de qualquer limpeza. (`src/data/era.py`)
- **A era não é variável preditiva.** A mortalidade bruta muda muito entre eras só por
  diferença de tempo de acompanhamento. A era entra apenas como estrato do split e como
  critério de coorte. (`src/preprocessing/split.py`, `src/evaluate/temporal.py`)
- **Códigos especiais do SEER.** Números altos são códigos, não quantidades (ex.: 98
  linfonodos positivos = nenhum linfonodo examinado; 999 mm = tamanho desconhecido).
  Eles são decodificados antes de qualquer padronização. (`src/data/sentinels.py`)
- **Proteção contra vazamento de dados.** O split acontece antes de qualquer `fit`, e
  todo o pré-processamento é ajustado só no treino. (`src/preprocessing/`,
  `src/checks/leakage.py`)
- **Teste usado uma única vez.** O desenvolvimento é feito na validação. O teste só é
  aberto na rodada final de `run_evaluation.py --eval-set test`.
- **Amostra fixa de 100 mil para a comparação principal.** O Gradient Boosting de
  sobrevivência não escala para a base inteira; para comparar os seis modelos nos
  mesmos dados, a tabela principal usa uma amostra fixa. Os outros cinco modelos também
  rodam na base inteira, como análise de sensibilidade.
- **Métricas.** C-index de Harrell (para comparar com a literatura), C-index de Uno
  (principal, corrige pela censura), C-index de Antolini (o adequado para o DeepHit),
  Brier e IBS, com intervalo de confiança por bootstrap. (`src/evaluate/`)

---

## Ambiente

Versões em `requirements.txt` (numpy 1.26.4, pandas 2.2.2, scikit-learn 1.5.1,
lifelines 0.29.0, scikit-survival 0.23.1, torch 2.3.1, pycox 0.2.3). O treino dos
modelos profundos é feito no Google Colab com GPU T4. Semente principal: 42.
