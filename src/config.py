"""Configuração central do projeto.

Reúne num só lugar os caminhos, os nomes das 22 colunas do export do SEER, a regra de
reconstrução da era de diagnóstico, as contagens medidas na base e os hiperparâmetros
de todos os modelos. Assim nenhum módulo usa um nome de coluna, semente ou limiar
diferente dos outros.

Os nomes de coluna estão escritos exatamente como no cabeçalho do CSV (com acentos,
parênteses e espaços), porque é assim que o pandas os lê.

O arquivo segue a ordem das etapas do pipeline: dados e coorte primeiro, depois
limpeza, pré-processamento, modelos e avaliação.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------

RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Fixa a semente em todas as fontes de aleatoriedade (random, numpy e torch).
    É chamada no início de cada script, para que cada execução seja reproduzível.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# O CSV do SEER não fica no repositório (os dados não podem ser redistribuídos pelo
# termo de uso do SEER). Para rodar em outra máquina ou no Colab, basta definir a
# variável de ambiente SEER_CSV_PATH com o caminho do CSV; não é preciso editar aqui.
_DEFAULT_CSV = PROJECT_ROOT / "TCC_ML_DL-20260707T020648Z-3-001" / "TCC_ML_DL" / "breast_cancer.csv"
RAW_CSV_PATH = Path(os.environ.get("SEER_CSV_PATH", _DEFAULT_CSV))
RAW_DIC_PATH = PROJECT_ROOT / "breast_cancer.dic"

# Tudo é lido como texto, sem deixar o pandas decidir o que é nulo. No SEER existem
# tipos diferentes de "vazio" (campo que não existia na época, informação desconhecida,
# código numérico especial), e quem decide o que cada um significa é a limpeza.
READ_CSV_KWARGS: dict = dict(dtype=str, keep_default_na=False, na_values=[])

# ---------------------------------------------------------------------------
# Colunas do export — as 22 colunas na ordem em que aparecem no CSV
# ---------------------------------------------------------------------------

# Desfecho (são o alvo, nunca entram como variável explicativa)
COL_SURVIVAL_MONTHS = "Survival months"
COL_VITAL_STATUS = "Vital status recode (study cutoff used)"

# Filtro de coorte
COL_SEX = "Sex"

# Demográficas
COL_AGE = "Age recode with <1 year olds and 90+"
COL_RACE = "Race recode (W, B, AI, API)"

# Grau (duas colunas: uma até 2017, outra a partir de 2018)
COL_GRADE_THRU_2017 = "Grade Recode (thru 2017)"
COL_GRADE_2018 = "Derived Summary Grade 2018 (2018+)"

# Estadiamento e subtipo
COL_STAGE = "Combined Summary Stage with Expanded Regional Codes (2004+)"
COL_BREAST_SUBTYPE = "Breast Subtype (2010+)"

# Receptores hormonais e HER2 (cada um com versão antiga e versão 2018+)
COL_ER_OLD = "ER Status Recode Breast Cancer (1990+)"
COL_ER_NEW = "Estrogen Receptor Summary (2018+)"
COL_PR_OLD = "PR Status Recode Breast Cancer (1990+)"
COL_PR_NEW = "Progesterone Receptor Summary (2018+)"
COL_HER2_OLD = "Derived HER2 Recode (2010+)"
COL_HER2_NEW = "HER2 Overall Summary Recode (2018+)"

# Tamanho do tumor (esquema CS até 2015 e Summary a partir de 2016)
COL_TUMOR_SIZE_CS = "CS tumor size (2004-2015)"
COL_TUMOR_SIZE_SUMMARY = "Tumor Size Summary (2016+)"

# Linfonodos
COL_NODES_POSITIVE = "Regional nodes positive (1988+)"
COL_NODES_EXAMINED = "Regional nodes examined (1988+)"

# Tratamento
COL_SURGERY = "RX Summ--Surg Prim Site (1998+)"
COL_RADIATION = "Radiation recode"
COL_CHEMOTHERAPY = "Chemotherapy recode (yes, no/unk)"

# Ordem exata do cabeçalho do CSV, conferida em scripts/profile_dataset.py (Etapa 0).
RAW_COLUMNS: list[str] = [
    COL_SURVIVAL_MONTHS,
    COL_VITAL_STATUS,
    COL_AGE,
    COL_SEX,
    COL_RACE,
    COL_GRADE_THRU_2017,
    COL_GRADE_2018,
    COL_STAGE,
    COL_BREAST_SUBTYPE,
    COL_ER_OLD,
    COL_PR_OLD,
    COL_HER2_OLD,
    COL_ER_NEW,
    COL_PR_NEW,
    COL_HER2_NEW,
    COL_TUMOR_SIZE_CS,
    COL_TUMOR_SIZE_SUMMARY,
    COL_NODES_POSITIVE,
    COL_NODES_EXAMINED,
    COL_SURGERY,
    COL_RADIATION,
    COL_CHEMOTHERAPY,
]

# ---------------------------------------------------------------------------
# Desfecho e coorte
# ---------------------------------------------------------------------------

# Desfecho = óbito por qualquer causa (sobrevivência global). Dead=1 é o evento e
# Alive=0 é censura no corte de 31/12/2022. Não é óbito por câncer de mama, porque o
# export usado não traz a causa do óbito.
EVENT_MAP: dict[str, int] = {"Dead": 1, "Alive": 0}

# Texto usado em `Survival months` quando o tempo é desconhecido (6.052 casos, todos óbitos).
SURVIVAL_UNKNOWN_TOKEN = "Unknown"

SEX_KEEP_VALUE = "Female"

# ---------------------------------------------------------------------------
# Reconstrução da era de diagnóstico
# ---------------------------------------------------------------------------
# O export não traz `Year of diagnosis`, mas o período fica registrado no padrão de
# `Blank(s)` das variáveis que só passaram a existir em certo ano. A era é um
# confundidor estrutural: é usada para estratificar o split e para restringir a coorte,
# mas nunca como variável preditiva.

BLANK_TOKEN = "Blank(s)"
BREAST_SUBTYPE_NA_TOKEN = "Recode not available"  # subtipo não existe antes de 2010

ERA_2000_2003 = "2000-2003"
ERA_2004_2009 = "2004-2009"
ERA_2010_2015 = "2010-2015"
ERA_2016_2017 = "2016-2017"
ERA_2018_2022 = "2018-2022"

ERA_LABELS: list[str] = [
    ERA_2000_2003,
    ERA_2004_2009,
    ERA_2010_2015,
    ERA_2016_2017,
    ERA_2018_2022,
]

# Acompanhamento máximo possível (em meses) em cada era até o corte de 31/12/2022.
# Serve para validar a reconstrução: o máximo de `Survival months` em cada era
# reconstruída bate exatamente com esses cinco valores.
ERA_MAX_FOLLOWUP: dict[str, int] = {
    ERA_2000_2003: 275,
    ERA_2004_2009: 227,
    ERA_2010_2015: 155,
    ERA_2016_2017: 83,
    ERA_2018_2022: 59,
}

# ---------------------------------------------------------------------------
# Split treino/validação/teste
# ---------------------------------------------------------------------------

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
CENSORING_RATE_TOLERANCE = 0.03  # diferença máx. de taxa de censura entre folds
ERA_BALANCE_TOLERANCE = 0.01  # diferença máx. de proporção de uma era entre folds

# ---------------------------------------------------------------------------
# Limiares de sanidade contra vazamento de dados
# ---------------------------------------------------------------------------

SANITY_CINDEX_MAX = 0.95
SANITY_CINDEX_MIN = 0.55
SANITY_TRAIN_TEST_GAP_MAX = 0.05

# ===========================================================================
# Contagens medidas na base (varredura completa do CSV em 22/07/2026)
# ---------------------------------------------------------------------------
# São contagens exatas, não estimativas. Se o CSV for exportado de novo e estes
# números não baterem, a Etapa 0 falha e o pipeline para, em vez de seguir com uma base
# diferente da descrita no trabalho.
# ===========================================================================

EXPECTED_N_ROWS = 1_365_329

EXPECTED_VITAL_STATUS_COUNTS: dict[str, int] = {"Alive": 920_439, "Dead": 444_890}

EXPECTED_SURVIVAL_UNKNOWN = 6_052  # `Survival months == "Unknown"`, todos `Dead`
EXPECTED_SURVIVAL_ZERO = 16_918  # `Survival months == "0000"` (8.482 Dead / 8.436 Alive)

EXPECTED_SEX_COUNTS: dict[str, int] = {"Female": 1_355_045, "Male": 10_284}

EXPECTED_ERA_COUNTS: dict[str, int] = {
    ERA_2000_2003: 205_049,
    ERA_2004_2009: 322_229,
    ERA_2010_2015: 361_127,
    ERA_2016_2017: 130_721,
    ERA_2018_2022: 346_203,
}

# Invariante dos linfonodos: `nodes_positive == "98"` se e somente se
# `nodes_examined == "00"`, em exatamente 198.635 registros, sem nenhuma exceção.
EXPECTED_NODE_INVARIANT_COUNT = 198_635

# Complementaridade perfeita do grau: as duas colunas nunca estão ambas preenchidas
# nem ambas vazias; a soma dos `Blank(s)` de cada uma é N.
EXPECTED_GRADE_BLANK_COUNTS: dict[str, int] = {
    COL_GRADE_THRU_2017: 346_203,  # Blank(s) exatamente na era 2018–2022
    COL_GRADE_2018: 1_019_126,  # Blank(s) em todas as eras anteriores a 2018
}

# --- Fluxo de coorte (contagens medidas) ---
# Base bruta 1.365.329 − 10.284 (homens) = 1.355.045 (coorte feminina)
#            1.355.045 − 5.988 (tempo desconhecido, feminino) = 1.349.057 (analítica)
# Dos 6.052 tempos desconhecidos no total, 64 são de homens (já saem no filtro de sexo).
EXPECTED_UNKNOWN_TIME_TOTAL = 6_052
EXPECTED_UNKNOWN_TIME_MALE = 64
EXPECTED_UNKNOWN_TIME_FEMALE = 5_988  # removidos da coorte feminina, 100% óbitos
EXPECTED_ANALYTIC_COHORT = 1_349_057

# ===========================================================================
# Etapa 2 — limpeza segundo as regras de codificação do SEER
# ===========================================================================

# --- Pares de colunas antiga/nova (unificadas, com prioridade para a mais nova) ---
# O vocabulário das duas é igualado antes de juntar, senão "Positive" e "ER positive"
# virariam categorias diferentes. Nesse ponto o `Blank(s)` estrutural pode ser tratado
# como ausente, porque a era já foi reconstruída antes.


@dataclass(frozen=True)
class RedundantPair:
    old_col: str
    new_col: str
    output_col: str
    old_value_map: dict[str, str] = field(default_factory=dict)
    new_value_map: dict[str, str] = field(default_factory=dict)


GRADE_VALUE_MAP = {
    "Well differentiated; Grade I": "1",
    "Moderately differentiated; Grade II": "2",
    "Poorly differentiated; Grade III": "3",
    "Undifferentiated; anaplastic; Grade IV": "4",
}

REDUNDANT_PAIRS: list[RedundantPair] = [
    RedundantPair(COL_ER_OLD, COL_ER_NEW, "er_status",
                  new_value_map={"ER positive": "Positive", "ER negative": "Negative"}),
    RedundantPair(COL_PR_OLD, COL_PR_NEW, "pr_status",
                  new_value_map={"PR positive": "Positive", "PR negative": "Negative"}),
    RedundantPair(COL_HER2_OLD, COL_HER2_NEW, "her2_status",
                  new_value_map={"HER2 positive": "Positive", "HER2 negative; equivocal": "Negative"}),
    RedundantPair(COL_GRADE_THRU_2017, COL_GRADE_2018, "tumor_grade",
                  old_value_map=GRADE_VALUE_MAP),
    RedundantPair(COL_TUMOR_SIZE_CS, COL_TUMOR_SIZE_SUMMARY, "tumor_size_mm"),
]

# --- Textos que indicam informação realmente não obtida -> NaN nas categóricas ---
# Não inclui Blank(s) (estrutural, já resolvido ao unir os pares) nem os estados
# clínicos de radiação (None/Unknown, Recommended..., Refused), que viram grupo próprio.
REAL_NULL_TOKENS: set[str] = {
    "Unknown",
    "Recode not available",
    "Borderline/Unknown",
    "Not documented; Cannot be determined; Not assessed or unknown if assessed",
    "Not documented; Indeterminate; Not assessed or unknown if assessed",
    "Test ordered, results not in chart",
}
# Grau 2018+: "9" = desconhecido; as letras são sistemas de graduação alternativos sem
# equivalência direta com Nottingham, então são tratadas como faltante em vez de
# forçar uma conversão.
GRADE_NULL_TOKENS: set[str] = {"9", "A", "B", "C", "D", "H", "L", "M"}

# --- Linfonodos: três variáveis derivadas ---
NODE_COUNT_MAX = 90  # 00-90 são contagens; 95/96/97/98/99 são códigos
NODE_STATUS_NEGATIVE = "negativo"
NODE_STATUS_POSITIVE = "positivo"
NODE_STATUS_NOT_ASSESSED = "nao_avaliado"
NODE_STATUS_UNKNOWN = "desconhecido"
NODE_STATUS_LEVELS = [
    NODE_STATUS_NEGATIVE, NODE_STATUS_POSITIVE, NODE_STATUS_NOT_ASSESSED, NODE_STATUS_UNKNOWN,
]

# --- Tamanho do tumor: recuperar as faixas do esquema CS + teto de plausibilidade ---
TUMOR_SIZE_CEILING_MM = 200  # sem `Primary Site` no export, é um teto razoável para mama
# Ponto médio (mm) das faixas 991-995 do esquema CS (faixas de 1 cm: 991 -> 5 mm,
# 992 -> 15 mm, e assim por diante).
# 996 (5-10 cm) e 997 (>10 cm) são tratadas como desconhecidas (384 registros, 0,03%).
CS_RANGE_MIDPOINTS_MM: dict[int, int] = {991: 5, 992: 15, 993: 25, 994: 35, 995: 45}
TUMOR_SIZE_MICROSCOPIC = 990  # foco microscópico -> NaN (não é uma medida)
TUMOR_SIZE_DIFFUSE = 998  # doença difusa/inflamatória -> NaN (e um indicador à parte)
TUMOR_SIZE_UNKNOWN = 999

# --- Idade -> ponto médio numérico da faixa (idade é praticamente contínua) ---
AGE_MIDPOINTS: dict[str, float] = {
    "00 years": 0,
    "01-04 years": 2.5,
    "05-09 years": 7,
    "10-14 years": 12,
    "15-19 years": 17,
    "20-24 years": 22,
    "25-29 years": 27,
    "30-34 years": 32,
    "35-39 years": 37,
    "40-44 years": 42,
    "45-49 years": 47,
    "50-54 years": 52,
    "55-59 years": 57,
    "60-64 years": 62,
    "65-69 years": 67,
    "70-74 years": 72,
    "75-79 years": 77,
    "80-84 years": 82,
    "85-89 years": 87,
    "90+ years": 92,
}

# --- Cirurgia: 48 códigos -> 4 famílias (pela codificação do SEER: 20-24 e 40-59) ---
SURGERY_NONE = "nenhuma"
SURGERY_CONSERVATIVE = "conservadora"  # códigos 20-24 (mastectomia parcial/lumpectomia)
SURGERY_MASTECTOMY = "mastectomia"  # códigos 40-59 (simples/radical modificada/radical)
SURGERY_OTHER_UNKNOWN = "outra_desconhecida"  # o resto (10-19, 30-39, 60-90, 99)
SURGERY_GROUP_LEVELS = [
    SURGERY_NONE, SURGERY_CONSERVATIVE, SURGERY_MASTECTOMY, SURGERY_OTHER_UNKNOWN,
]

# --- Radiação: 8 categorias -> 4 níveis ---
RADIATION_NONE_UNKNOWN = "nenhuma_desconhecida"
RADIATION_BEAM = "feixe"
RADIATION_IMPLANT = "implante_isotopo"
RADIATION_REFUSED = "recusou"
RADIATION_GROUP_LEVELS = [
    RADIATION_NONE_UNKNOWN, RADIATION_BEAM, RADIATION_IMPLANT, RADIATION_REFUSED,
]
# "Radiation NOS" e "Combination" indicam que houve radiação, mas sem separar a
# modalidade; foram agrupadas em "feixe", que é a modalidade dominante.
# Os textos estão exatamente como aparecem no CSV (inclusive o espaço duplo em
# "Radiation, NOS  method...").
RADIATION_GROUP_MAP: dict[str, str] = {
    "None/Unknown": RADIATION_NONE_UNKNOWN,
    "Recommended, unknown if administered": RADIATION_NONE_UNKNOWN,
    "Beam radiation": RADIATION_BEAM,
    "Radiation, NOS  method or source not specified": RADIATION_BEAM,
    "Combination of beam with implants or isotopes": RADIATION_BEAM,
    "Radioactive implants (includes brachytherapy) (1988+)": RADIATION_IMPLANT,
    "Radioisotopes (1988+)": RADIATION_IMPLANT,
    "Refused (1988+)": RADIATION_REFUSED,
}

# --- Variáveis finais que saem da limpeza (Etapa 2) ---
NUMERIC_FEATURES: list[str] = ["node_count", "nodes_examined_n", "tumor_size_mm", "age_mid"]
CATEGORICAL_FEATURES: list[str] = [
    "node_status", "race", "stage", "tumor_grade",
    "er_status", "pr_status", "her2_status",
    "surgery_group", "radiation_group", "chemotherapy",
]
# Ordinais (a ordem é usada na codificação da Etapa 5).
STAGE_ORDER = ["in_situ", "localized", "regional", "distant"]
GRADE_ORDER = ["1", "2", "3", "4"]

# --- Grupos de pré-processamento da Etapa 5 (ajustado só no treino) ---
# Ordinais: codificação ordinal (preserva a ordem) + imputação pela mediana + escala.
ORDINAL_ENCODE: dict[str, list[str]] = {"stage": STAGE_ORDER, "tumor_grade": GRADE_ORDER}

# Categóricas nominais -> categoria "desconhecido" explícita + one-hot.
# Inclui her2_status. Antes de 2010 o HER2 não era registrado, mas tratá-lo como ER/PR
# preserva o valor preditivo do HER2 na coorte principal (2010-2022), onde o faltante é
# de fato desconhecido, em vez de preencher o HER2 com a moda.
# Na coorte completa, "her2 = desconhecido" acaba indicando diagnóstico antes de 2010;
# isso é tratado restringindo a coorte por era, e não na imputação.
NOMINAL_DESCONHECIDO: list[str] = [
    "node_status", "race", "er_status", "pr_status", "her2_status",
    "surgery_group", "radiation_group", "chemotherapy",
]
DESCONHECIDO_LEVEL = "desconhecido"

# Numéricas e ordinais cujo faltante depende da era (`tumor_size_mm` e `stage` antes de
# 2004) são imputadas pela mediana, sem criar indicador de faltante. Um indicador ali
# funcionaria como um "marcador de era" dentro das variáveis.

# breast_subtype fica guardado, mas não entra no modelo: usamos ER, PR e HER2 separados,
# e o subtipo é uma combinação exata deles (incluir os dois geraria colinearidade).
KEPT_NOT_MODELED: list[str] = ["breast_subtype"]

# ===========================================================================
# Etapa 3 — restrição de era (coorte de análise)
# ===========================================================================
# Como a era é um confundidor estrutural, a análise principal pode ser restrita a um
# período; a exploração e as análises de sensibilidade usam a base completa.
# None = base completa. A restrição a 2010-2022 ainda está em decisão com a orientação,
# por isso o padrão continua sendo a base completa.
ERA_RESTRICTION: list[str] | None = None

# Período candidato para a análise principal: a partir de 2010 o subtipo molecular está
# disponível e o tratamento é mais próximo da prática atual.
# Na base bruta são 838.051 casos em 2010-2022 (361.127 + 130.721 + 346.203). Na coorte
# analítica (feminina, tempo conhecido) são 828.121 (356.865 + 129.245 + 342.011).
ERA_MAIN_COHORT_RECOMMENDED: list[str] = [ERA_2010_2015, ERA_2016_2017, ERA_2018_2022]

# ===========================================================================
# Etapa 6 — modelos clássicos (baselines)
# ===========================================================================


@dataclass
class CoxConfig:
    penalizer: float = 0.1  # penalização ridge (L2), estabiliza o ajuste com muitas dummies


@dataclass
class SplineConfig:
    n_knots: int = 4
    degree: int = 3  # splines cúbicas nas variáveis contínuas, para captar não linearidade


@dataclass
class RSFConfig:
    n_estimators: int = 100
    min_samples_leaf: int = 20  # folhas maiores: treino mais rápido e menos sobreajuste
    max_features: str = "sqrt"


@dataclass
class GBSConfig:
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    subsample: float = 0.8


# C-index de Uno (IPCW): truncado em tau = percentil 95 dos tempos de treino, para a
# ponderação pela censura não ficar instável na cauda (a base tem cerca de 67% de censura).
UNO_TAU_PERCENTILE = 95

# A faixa de sanidade do C-index (SANITY_CINDEX_MIN/MAX e SANITY_TRAIN_TEST_GAP_MAX)
# está definida mais acima.

# Amostra fixa comum para a tabela comparativa final. O tamanho foi escolhido pelo
# limite do Gradient Boosting, que não escala para a base inteira. Todos os modelos são
# comparados nessa mesma amostra, com semente fixa.
COMPARISON_SAMPLE_N = 100_000

# ===========================================================================
# Etapa 7 — modelos profundos (DeepSurv, DeepHit — pycox)
# ===========================================================================


@dataclass
class MLPConfig:
    num_layers: int = 2
    hidden_units: int = 32
    dropout: float = 0.2
    batch_norm: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 100
    patience: int = 10  # early stopping por perda de validação
    learning_rate: float = 1e-3  # valor padrão; quando usado, o LR finder sugere outro


@dataclass
class DeepHitConfig:
    num_time_bins: int = 20  # discretização do tempo por quantis (inclui um intervalo para t=0)
    scheme: str = "quantiles"  # intervalos iguais ficariam vazios na cauda (até 275 meses)
    alpha: float = 0.2  # peso entre a verossimilhança e a perda de ranking do DeepHit
    sigma: float = 0.1  # parâmetro de escala da perda de ranking

# Número mínimo de sementes para cada modelo profundo.
MIN_SEEDS = 3

# ===========================================================================
# Etapa 8 — avaliação final
# ===========================================================================
EVAL_HORIZONS = [12, 36, 60, 120]  # meses = 1, 3, 5, 10 anos (Brier)
CALIBRATION_HORIZON = 60  # curva de calibração por decil em 5 anos
N_BOOTSTRAP = 200  # reamostragens para os intervalos de confiança
