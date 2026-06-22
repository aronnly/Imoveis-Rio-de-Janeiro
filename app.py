from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# -----------------------------------------------------------------------------
# Configuração geral
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Painel ITBI — preço/m² corrigido",
    layout="wide",
    initial_sidebar_state="auto",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.main .block-container {
    padding-top: 1.35rem;
    padding-left: clamp(1.2rem, 3vw, 3.2rem);
    padding-right: clamp(1.2rem, 3vw, 3.2rem);
    max-width: 1520px;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.72rem;
}

[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 12px;
    padding: 4px 4px 0 4px;
    margin: 12px 0 26px 0;
    background: transparent;
    box-shadow: none;
    overflow: hidden;
}

[data-testid="stDataFrame"] {
    margin: 10px 0 24px 0;
}

[data-testid="stDataFrame"] div[role="gridcell"],
[data-testid="stDataFrame"] div[role="columnheader"] {
    font-size: 0.92rem;
}

.block-note {
    margin: 8px 0 16px 0;
    color: #64748B;
    font-size: 0.92rem;
}

/* Remove setas de delta dos KPIs — ficam feias no Streamlit default */
[data-testid="stMetricDelta"] svg { display: none; }

/* Tabs com fonte mais pesada */
button[data-baseweb="tab"] { font-weight: 500 !important; }

/* Padding interno das métricas */
[data-testid="metric-container"] {
    padding: 14px 16px !important;
    background: rgba(15, 23, 42, 0.20);
    border-radius: 8px;
    border: 1px solid rgba(148, 163, 184, 0.18);
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
    gap: 14px;
    margin: 18px 0 26px 0;
}
.kpi-card {
    border: 1px solid rgba(148, 163, 184, 0.18);
    background: rgba(15, 23, 42, 0.16);
    border-radius: 14px;
    padding: 15px 16px 14px 16px;
    min-height: 112px;
    overflow: hidden;
}
.kpi-label {
    color: #64748B;
    font-size: 0.86rem;
    line-height: 1.18rem;
    margin-bottom: 8px;
    white-space: normal;
}
.kpi-value {
    color: #E5E7EB;
    font-size: clamp(1.22rem, 1.55vw, 1.80rem);
    line-height: 1.08;
    font-weight: 650;
    letter-spacing: -0.035em;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}
.kpi-value.long {
    font-size: clamp(1.08rem, 1.24vw, 1.42rem);
    letter-spacing: -0.045em;
}
.kpi-sublabel {
    color: #94A3B8;
    font-size: 0.78rem;
    line-height: 1.1rem;
    margin-top: 8px;
    white-space: normal;
}
.kpi-pill {
    display: inline-block;
    font-size: clamp(1.14rem, 1.34vw, 1.55rem);
    line-height: 1.05;
    font-weight: 650;
    letter-spacing: -0.025em;
    border-radius: 999px;
    padding: 10px 15px;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}
.kpi-negative {
    color: #7F1D1D;
    background: #FEE2E2;
    border: 1px solid #FCA5A5;
}
.kpi-positive {
    color: #14532D;
    background: #DCFCE7;
    border: 1px solid #86EFAC;
}
.kpi-neutral {
    color: #334155;
    background: #F1F5F9;
    border: 1px solid #CBD5E1;
}
@media (max-width: 1500px) {
    .kpi-grid { grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
}
@media (max-width: 760px) {
    .kpi-grid { grid-template-columns: 1fr; }
}

/* Evita a duplicação visual dos rótulos extremos no select_slider de período.
   Mantém apenas os valores selecionados em destaque. */
[data-testid="stSliderTickBar"],
[data-testid="stTickBar"],
.stSlider [class*="TickBar"],
.stSlider [class*="tickBar"] {
    display: none !important;
}

/* -------------------------------------------------------------------------
   Responsividade — aplicada antes do carregamento pesado do painel
   ------------------------------------------------------------------------- */
html { -webkit-text-size-adjust: 100%; }
img, svg, canvas { max-width: 100%; }

[data-testid="stAppViewContainer"] {
    overflow-x: hidden;
}

[data-testid="stHorizontalBlock"] {
    gap: clamp(0.55rem, 1.6vw, 1rem);
}

[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plot-container,
[data-testid="stPlotlyChart"] .svg-container {
    max-width: 100% !important;
}

/* Tabs: no celular viram uma faixa rolável, sem quebrar layout. */
div[data-testid="stTabs"] div[role="tablist"] {
    overflow-x: auto;
    overflow-y: hidden;
    flex-wrap: nowrap;
    scrollbar-width: thin;
    -webkit-overflow-scrolling: touch;
}
div[data-testid="stTabs"] button[role="tab"] {
    white-space: nowrap;
    flex: 0 0 auto;
}

/* Controles e botões devem ocupar largura útil em telas menores. */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"],
[data-testid="stMultiSelect"],
.stDateInput,
.stButton > button,
.stDownloadButton > button {
    max-width: 100%;
}

/* Dataframes e gráficos largos podem rolar horizontalmente sem estourar a tela. */
[data-testid="stDataFrame"],
[data-testid="stTable"],
[data-testid="stPlotlyChart"] {
    max-width: 100%;
}

@media (max-width: 1100px) {
    .main .block-container {
        max-width: 100%;
        padding-left: 1.05rem;
        padding-right: 1.05rem;
    }
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap;
    }
    [data-testid="column"] {
        min-width: min(100%, 320px) !important;
    }
}

@media (max-width: 760px) {
    .main .block-container {
        padding-top: 0.72rem;
        padding-left: 0.72rem;
        padding-right: 0.72rem;
    }

    h1 { font-size: 1.52rem !important; line-height: 1.18 !important; }
    h2 { font-size: 1.30rem !important; line-height: 1.22 !important; }
    h3 { font-size: 1.12rem !important; line-height: 1.25 !important; }
    h4 { font-size: 1.02rem !important; line-height: 1.28 !important; }
    p, li, label, [data-testid="stMarkdownContainer"] {
        font-size: 0.94rem;
        line-height: 1.35;
    }

    [data-testid="stSidebar"] {
        max-width: 92vw !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.52rem;
    }

    /* No celular, colunas viram blocos empilhados. */
    [data-testid="stHorizontalBlock"] {
        display: block !important;
    }
    [data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        margin-bottom: 0.8rem;
    }

    [data-testid="stPlotlyChart"] {
        border-radius: 10px;
        padding: 2px 2px 0 2px;
        margin: 10px 0 18px 0;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }

    /* Mantém gráficos densos legíveis: cabe na tela, mas aceita rolagem horizontal se Plotly precisar. */
    [data-testid="stPlotlyChart"] .js-plotly-plot,
    [data-testid="stPlotlyChart"] .plot-container,
    [data-testid="stPlotlyChart"] .svg-container {
        width: 100% !important;
        max-width: 100% !important;
    }

    div[data-testid="stTabs"] div[role="tablist"] {
        gap: 0.25rem;
        padding-bottom: 0.25rem;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        padding: 0.45rem 0.65rem !important;
        font-size: 0.88rem !important;
    }

    [data-testid="metric-container"] {
        padding: 10px 12px !important;
    }
    .kpi-grid {
        grid-template-columns: 1fr !important;
        gap: 10px;
        margin: 12px 0 18px 0;
    }
    .kpi-card {
        min-height: auto;
        padding: 12px 13px;
    }
    .kpi-value, .kpi-value.long, .kpi-pill {
        font-size: 1.08rem !important;
        white-space: normal;
        word-break: break-word;
    }

    [data-testid="stDataFrame"] {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }
    [data-testid="stDataFrame"] div[role="gridcell"],
    [data-testid="stDataFrame"] div[role="columnheader"] {
        font-size: 0.80rem;
    }

    .stButton > button,
    .stDownloadButton > button {
        width: 100%;
    }
}

@media (max-width: 430px) {
    .main .block-container {
        padding-left: 0.45rem;
        padding-right: 0.45rem;
    }
    [data-testid="stPlotlyChart"] {
        margin-left: -0.15rem;
        margin-right: -0.15rem;
    }
    .kpi-label, .kpi-sublabel {
        font-size: 0.76rem;
    }
}

</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
SHEET_NAME = "Base_Normalizada"

DATA_FILE_STEM = "ITBI_mapa_excel_valores_corrigidos_IPCA_mar_2026"
GEO_FILE_NAME = "geocoded_logradouros.csv"

DATE_COL = "data_ref"
TX_COL = "total_transações"
VALUE_COL = "média_valor_imóvel_corrigido_mar_2026"
AREA_COL = "média_área_construída"
PRICE_COL = "preco_m2_imovel_corrigido_mar_2026"
TX_VALUE_COL = "média_valor_transação_corrigido_mar_2026"
TX_PRICE_COL = "preco_m2_transacao_corrigido_mar_2026"

TEXT_COLS = [
    "bairro",
    "logradouro",
    "uso",
    "principais_tipologias",
    "principal_transação_mercado",
    "obs_qualidade",
    "endereco_mapa",
]

NUMERIC_COLS = [
    TX_COL,
    VALUE_COL,
    AREA_COL,
    PRICE_COL,
    TX_VALUE_COL,
    TX_PRICE_COL,
    "média_percentual_transferido",
    "média_valor_transação",
    "média_valor_imóvel",
    "preco_m2_transacao",
    "preco_m2_imovel",
    "índice_ipca_mês",
    "fator_ipca_para_mar_2026",
]

REQUIRED_COLS = {
    DATE_COL,
    "bairro",
    "logradouro",
    "uso",
    "principais_tipologias",
    TX_COL,
    VALUE_COL,
    AREA_COL,
    PRICE_COL,
}

CESTAS_BAIRROS: dict[str, list[str]] = {
    "Personalizado": [],
    "Zona Sul": [
        "Botafogo",
        "Catete",
        "Copacabana",
        "Flamengo",
        "Glória",
        "Ipanema",
        "Lagoa",
        "Laranjeiras",
        "Leblon",
    ],
    "Grande Tijuca": [
        "Alto da Boa Vista",
        "Andaraí",
        "Grajaú",
        "Maracanã",
        "Praça da Bandeira",
        "Tijuca",
        "Vila Isabel",
    ],
    "Centro expandido": [
        "Centro",
        "Cidade Nova",
        "Estácio",
        "Gamboa",
        "Glória",
        "Lapa",
        "Saúde",
        "Santa Teresa",
        "Santo Cristo",
    ],
    "Barra/Recreio/Jacarepaguá": [
        "Barra da Tijuca",
        "Camorim",
        "Curicica",
        "Freguesia (Jacarepaguá)",
        "Jacarepaguá",
        "Recreio dos Bandeirantes",
        "Vargem Grande",
        "Vargem Pequena",
    ],
    "Zona Norte ferroviária": [
        "Abolição",
        "Cachambi",
        "Del Castilho",
        "Engenho de Dentro",
        "Engenho Novo",
        "Méier",
        "Piedade",
        "Todos os Santos",
    ],
}

AREA_BINS = [0, 30, 55, 75, 100, 150, 250, np.inf]
AREA_LABELS = [
    "até 30 m²",
    "30–55 m²",
    "55–75 m²",
    "75–100 m²",
    "100–150 m²",
    "150–250 m²",
    "acima de 250 m²",
]

ROLLING_WINDOW_OPTIONS = [12, 24, 36, 48, 60]
BENCHMARK_SERIES_NAME = "Média do recorte de comparação"

# Paleta orientada por Storytelling with Data:
# contexto em cinza, um destaque principal e cores fortes só quando codificam uma mensagem.
PAPER    = "#0E1117"
PLOT_BG  = "#111827"
INK      = "#E5E7EB"
MUTED    = "#94A3B8"
GRID     = "#2A3441"
CONTEXT  = "#64748B"
ACCENT   = "#60A5FA"
ACCENT2  = "#2DD4BF"
BENCHMARK = "#F59E0B"
VOLUME   = "#475569"
POSITIVE = "#34D399"
NEGATIVE = "#F87171"
CAUTION  = "#FBBF24"
BORDO_1  = "#334155"
BORDO_2  = "#93C5FD"
BORDO_3  = "#60A5FA"
WARNING  = "#F59E0B"
HOT_COOL_SCALE = [
    [0.00, "#93C5FD"],  # frio / baixo
    [0.25, "#60A5FA"],
    [0.50, "#FDE68A"],
    [0.75, "#F59E0B"],
    [1.00, "#DC2626"],  # quente / alto
]

PLOT_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter, DM Sans, Arial, sans-serif", color=INK, size=13),
        title=dict(font=dict(size=18, color=INK), x=0.0, xanchor="left", y=0.98),
        colorway=[
            ACCENT, ACCENT2, "#A78BFA", "#FBBF24",
            "#67E8F9", "#F9A8D4", "#CBD5E1", "#86EFAC", "#FDBA74", "#818CF8",
        ],
        margin=dict(l=84, r=52, t=92, b=76),
        separators=",.",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12, color=INK),
            itemwidth=34,
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            bordercolor="#334155",
            font=dict(color="#F8FAFC", family="Inter, DM Sans, Arial, sans-serif", size=12),
        ),
    )
)

# -----------------------------------------------------------------------------
# Formatação e caminhos
# -----------------------------------------------------------------------------


def br_money(value: float | int | None, prefix: str = "R$ ", decimals: int = 0) -> str:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return "—"
    return prefix + f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def br_number(value: float | int | None, decimals: int = 0) -> str:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return "—"
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def parse_br_number(value: object, default: float = 0.0) -> float:
    """Converte entrada leiga em PT-BR, como '600.000' ou '600.000,50'."""
    if value is None:
        return float(default)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if np.isfinite(value) else float(default)
    raw = str(value).strip()
    if not raw:
        return float(default)
    cleaned = (
        raw.replace("R$", "")
        .replace(" ", "")
        .replace(".", "")
        .replace(",", ".")
    )
    try:
        parsed = float(cleaned)
        return parsed if np.isfinite(parsed) else float(default)
    except ValueError:
        return float(default)


def br_pct(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return "—"
    return f"{value:.{decimals}%}".replace(".", ",")


def normalize_transfer_ratio(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.where(values > 1.5, values / 100.0, values), index=series.index)


def br_transfer_pct(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return "—"
    ratio = float(value / 100.0) if value > 1.5 else float(value)
    return br_pct(ratio, decimals)


def find_data_file() -> Path | None:
    candidates = [
        BASE_DIR / "data" / f"{DATA_FILE_STEM}.csv",
        BASE_DIR / f"{DATA_FILE_STEM}.csv",
        BASE_DIR / "data" / f"{DATA_FILE_STEM}.csv.gz",
        BASE_DIR / f"{DATA_FILE_STEM}.csv.gz",
        BASE_DIR / "data" / f"{DATA_FILE_STEM}.xlsx",
        BASE_DIR / f"{DATA_FILE_STEM}.xlsx",
        BASE_DIR / "data" / f"{DATA_FILE_STEM}(6).xlsx",
        BASE_DIR / f"{DATA_FILE_STEM}(6).xlsx",
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = sorted(BASE_DIR.rglob(f"{DATA_FILE_STEM}*.csv"))
    if matches:
        return matches[0]
    matches = sorted(BASE_DIR.rglob(f"{DATA_FILE_STEM}*.csv.gz"))
    if matches:
        return matches[0]
    matches = sorted(BASE_DIR.rglob(f"{DATA_FILE_STEM}*.xlsx"))
    return matches[0] if matches else None


def find_geo_file() -> Path | None:
    candidates = [
        BASE_DIR / "data" / GEO_FILE_NAME,
        BASE_DIR / GEO_FILE_NAME,
        BASE_DIR / "data" / "geocoded_logradouros(2).csv",
        BASE_DIR / "geocoded_logradouros(2).csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(BASE_DIR.rglob("geocoded_logradouros*.csv"))
    return matches[0] if matches else None


# -----------------------------------------------------------------------------
# Carga, limpeza e validação
# -----------------------------------------------------------------------------


def assign_area_labels_exclusive(df: pd.DataFrame) -> pd.DataFrame:
    """Recalcula faixa_area com limite superior exclusivo.

    Exemplo: 55 m² entra em "55–75 m²", não em "30–55 m²".
    Isso evita dupla contagem visual nas faixas de transição.
    """
    out = df.copy()
    out["faixa_area"] = pd.cut(
        out[AREA_COL],
        bins=AREA_BINS,
        labels=AREA_LABELS,
        include_lowest=True,
        right=False,
    ).astype("string")
    return out


@st.cache_data(show_spinner="Carregando base corrigida pelo IPCA...")
def load_data(path: str) -> pd.DataFrame:
    data_path = Path(path)
    if data_path.suffix.lower() == ".csv" or data_path.name.lower().endswith(".csv.gz"):
        df = pd.read_csv(data_path, encoding="utf-8-sig", compression="infer")
    else:
        df = pd.read_excel(data_path, sheet_name=SHEET_NAME, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError("Colunas obrigatórias ausentes: " + ", ".join(sorted(missing)))

    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, "bairro", "logradouro"]).copy()

    df["valor_total_corrigido"] = df[VALUE_COL] * df[TX_COL]
    df["area_total"] = df[AREA_COL] * df[TX_COL]
    df["preco_m2_corrigido_linha"] = df["valor_total_corrigido"] / df["area_total"]
    df.loc[~np.isfinite(df["preco_m2_corrigido_linha"]), "preco_m2_corrigido_linha"] = np.nan

    df["periodo_mes"] = df[DATE_COL].dt.to_period("M").dt.to_timestamp()
    df["ano"] = df[DATE_COL].dt.year
    df = assign_area_labels_exclusive(df)

    if "endereco_mapa" not in df.columns:
        df["endereco_mapa"] = pd.NA

    fallback_address = (
        df["logradouro"].astype("string")
        + ", "
        + df["bairro"].astype("string")
        + ", Rio de Janeiro, RJ, Brasil"
    )
    df["endereco_mapa"] = df["endereco_mapa"].fillna(fallback_address)

    return df


@st.cache_data(show_spinner=False)
def load_geo(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()

    geo = pd.read_csv(path, encoding="utf-8-sig")
    geo.columns = [str(c).strip() for c in geo.columns]

    # Defesa contra CSV exportado com a linha inteira envolvida por aspas.
    # Nesse caso o pandas cria a coluna "bairro", mas joga o restante da linha
    # dentro dela, deixando latitude/longitude vazias.
    if {"bairro", "latitude", "longitude"}.issubset(geo.columns):
        malformed = geo["latitude"].isna().mean() > 0.95 and geo["bairro"].astype(str).str.contains(",").mean() > 0.80
        if malformed:
            import csv
            rows: list[list[str]] = []
            with open(path, "r", encoding="utf-8-sig", newline="") as fh:
                for line in fh:
                    raw = line.strip()
                    if not raw:
                        continue
                    if raw.startswith('"') and raw.endswith('"'):
                        raw = raw[1:-1].replace('""', '"')
                    rows.append(next(csv.reader([raw])))
            if rows:
                header, data_rows = rows[0], rows[1:]
                geo = pd.DataFrame(data_rows, columns=header)
                geo.columns = [str(c).strip() for c in geo.columns]

    required = {"bairro", "logradouro", "latitude", "longitude"}
    if not required.issubset(geo.columns):
        return pd.DataFrame()

    for col in ["bairro", "logradouro", "endereco_mapa", "status_geocodificacao"]:
        if col in geo.columns:
            geo[col] = geo[col].astype("string").str.strip()

    geo["latitude"] = pd.to_numeric(geo["latitude"], errors="coerce")
    geo["longitude"] = pd.to_numeric(geo["longitude"], errors="coerce")
    return geo.dropna(subset=["latitude", "longitude"]).drop_duplicates(["bairro", "logradouro"])


# -----------------------------------------------------------------------------
# Estatística descritiva
# -----------------------------------------------------------------------------


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    data = pd.DataFrame({"v": values, "w": weights}).replace([np.inf, -np.inf], np.nan).dropna()
    data = data[data["w"] > 0]
    if data.empty:
        return np.nan
    data = data.sort_values("v")
    cutoff = data["w"].sum() / 2
    return float(data.loc[data["w"].cumsum() >= cutoff, "v"].iloc[0])


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = pd.DataFrame({"v": values, "w": weights}).replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[valid["w"] > 0]
    if valid.empty:
        return np.nan
    return float(np.average(valid["v"], weights=valid["w"]))


def aggregate_metrics(df: pd.DataFrame, group_cols: Sequence[str] | None = None) -> pd.DataFrame:
    group_cols = list(group_cols or [])

    if df.empty:
        cols = group_cols + [
            "valor_total_corrigido",
            "area_total",
            "total_transações",
            "preco_m2_corrigido",
            "mediana_preco_m2",
            "preco_m2_media_simples",
            "linhas",
        ]
        return pd.DataFrame(columns=cols)

    if group_cols:
        grouped = df.groupby(group_cols, dropna=False, observed=True)
        base = grouped.agg(
            valor_total_corrigido=("valor_total_corrigido", "sum"),
            area_total=("area_total", "sum"),
            total_transações=(TX_COL, "sum"),
            preco_m2_media_simples=(PRICE_COL, "mean"),
            linhas=(PRICE_COL, "size"),
        ).reset_index()
        med = grouped[[PRICE_COL, "area_total"]].apply(lambda g: weighted_median(g[PRICE_COL], g["area_total"]))
        med = med.rename("mediana_preco_m2").reset_index()
        out = base.merge(med, on=group_cols, how="left")
    else:
        out = pd.DataFrame(
            [
                {
                    "valor_total_corrigido": df["valor_total_corrigido"].sum(),
                    "area_total": df["area_total"].sum(),
                    "total_transações": df[TX_COL].sum(),
                    "preco_m2_media_simples": df[PRICE_COL].mean(),
                    "mediana_preco_m2": weighted_median(df[PRICE_COL], df["area_total"]),
                    "linhas": len(df),
                }
            ]
        )

    out["preco_m2_corrigido"] = out["valor_total_corrigido"] / out["area_total"]
    out.loc[~np.isfinite(out["preco_m2_corrigido"]), "preco_m2_corrigido"] = np.nan

    ordered = group_cols + [
        "total_transações",
        "preco_m2_corrigido",
        "mediana_preco_m2",
        "preco_m2_media_simples",
        "valor_total_corrigido",
        "area_total",
        "linhas",
    ]
    return out[ordered]


def add_period(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    data = df.copy()
    if freq == "Mensal":
        data["periodo"] = data[DATE_COL].dt.to_period("M").dt.to_timestamp()
    elif freq == "Trimestral":
        data["periodo"] = data[DATE_COL].dt.to_period("Q").dt.to_timestamp()
    else:
        data["periodo"] = data[DATE_COL].dt.to_period("Y").dt.to_timestamp()
    return data


def time_group(df: pd.DataFrame, freq: str, group_cols: Sequence[str] | None = None) -> pd.DataFrame:
    group_cols = list(group_cols or [])
    data = add_period(df, freq)
    return aggregate_metrics(data, ["periodo"] + group_cols).sort_values(["periodo"] + group_cols)


def calc_variation(ts: pd.DataFrame, min_trans: int = 1) -> float:
    valid = ts[(ts["total_transações"] >= min_trans) & ts["preco_m2_corrigido"].notna()].sort_values("periodo")
    if len(valid) < 2:
        return np.nan
    first = valid.iloc[0]["preco_m2_corrigido"]
    last = valid.iloc[-1]["preco_m2_corrigido"]
    if not first or first <= 0:
        return np.nan
    return float(last / first - 1)


def index_series(ts: pd.DataFrame, group_cols: Sequence[str] | None = None, value_col: str = "preco_m2_corrigido") -> pd.DataFrame:
    group_cols = list(group_cols or [])
    if not group_cols:
        out = ts.sort_values("periodo").copy()
        valid = out[value_col].replace([np.inf, -np.inf], np.nan).dropna()
        base = valid.iloc[0] if not valid.empty else np.nan
        out["indice_100"] = out[value_col] / base * 100 if base and base > 0 else np.nan
        return out

    frames = []
    for _, part in ts.groupby(group_cols, dropna=False, observed=True):
        chunk = part.sort_values("periodo").copy()
        valid = chunk[value_col].replace([np.inf, -np.inf], np.nan).dropna()
        base = valid.iloc[0] if not valid.empty else np.nan
        chunk["indice_100"] = chunk[value_col] / base * 100 if base and base > 0 else np.nan
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True) if frames else ts.assign(indice_100=np.nan)




def drawdown_series(indexed: pd.DataFrame, series_col: str | None = None) -> pd.DataFrame:
    if indexed.empty or "periodo" not in indexed.columns or "indice_100" not in indexed.columns:
        return pd.DataFrame()

    base = indexed.copy()
    base["rendimento_pct"] = base["indice_100"] / 100.0 - 1.0
    base["wealth_index"] = 1.0 + base["rendimento_pct"]

    if not series_col:
        base = base.sort_values("periodo").copy()
        base["peak"] = base["wealth_index"].cummax()
        base["drawdown"] = base["wealth_index"] / base["peak"] - 1.0
        return base

    frames = []
    for _, part in base.groupby(series_col, observed=True, dropna=False):
        chunk = part.sort_values("periodo").copy()
        chunk["peak"] = chunk["wealth_index"].cummax()
        chunk["drawdown"] = chunk["wealth_index"] / chunk["peak"] - 1.0
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def rolling_returns_by_series(
    ts: pd.DataFrame,
    window_months: int,
    min_trans: int,
    *,
    group_col: str | None = None,
    series_name: str = BENCHMARK_SERIES_NAME,
    value_col: str = "preco_m2_corrigido",
) -> pd.DataFrame:
    """
    Calcula retorno real por janelas móveis mensais.

    A entrada já deve estar filtrada pelo período escolhido pelo usuário.
    Para janela de 12 meses, por exemplo, cada ponto compara o mês final
    com o mesmo mês 12 meses antes. Meses sem preço válido ou com volume
    inferior ao mínimo são ignorados.
    """
    required_cols = {"periodo", "total_transações", value_col}
    if group_col:
        required_cols.add(group_col)
    if ts.empty or not required_cols.issubset(ts.columns):
        return pd.DataFrame(
            columns=[
                "serie",
                "periodo",
                "inicio_janela",
                "fim_janela",
                "janela_meses",
                "preco_inicio",
                "preco_fim",
                "retorno_janela",
                "ganho_positivo",
                "transacoes_inicio",
                "transacoes_fim",
            ]
        )

    work = ts.copy()
    work["serie"] = work[group_col].astype(str) if group_col else series_name
    work["periodo"] = pd.to_datetime(work["periodo"], errors="coerce")
    work = work.dropna(subset=["periodo", value_col, "total_transações"])
    work = work[(work[value_col] > 0) & (work["total_transações"] >= min_trans)].copy()
    if work.empty:
        return pd.DataFrame(columns=["serie", "periodo", "inicio_janela", "fim_janela", "janela_meses", "preco_inicio", "preco_fim", "retorno_janela", "ganho_positivo", "transacoes_inicio", "transacoes_fim"])

    records: list[dict[str, object]] = []
    for serie, part in work.groupby("serie", dropna=False, observed=True):
        part = part.sort_values("periodo").copy()
        part["periodo_m"] = part["periodo"].dt.to_period("M")
        part = part.drop_duplicates("periodo_m", keep="last").set_index("periodo_m")

        for end_period, end_row in part.iterrows():
            start_period = end_period - window_months
            if start_period not in part.index:
                continue

            start_row = part.loc[start_period]
            start_price = float(start_row[value_col])
            end_price = float(end_row[value_col])
            if not np.isfinite(start_price) or not np.isfinite(end_price) or start_price <= 0:
                continue

            retorno = end_price / start_price - 1
            records.append(
                {
                    "serie": str(serie),
                    "periodo": end_period.to_timestamp(),
                    "inicio_janela": start_period.to_timestamp(),
                    "fim_janela": end_period.to_timestamp(),
                    "janela_meses": int(window_months),
                    "preco_inicio": start_price,
                    "preco_fim": end_price,
                    "retorno_janela": float(retorno),
                    "ganho_positivo": bool(retorno > 0),
                    "transacoes_inicio": float(start_row["total_transações"]),
                    "transacoes_fim": float(end_row["total_transações"]),
                }
            )

    return pd.DataFrame.from_records(records)


def rolling_gain_summary(rolling: pd.DataFrame) -> pd.DataFrame:
    if rolling.empty:
        return pd.DataFrame(
            columns=[
                "serie",
                "janelas_validas",
                "janelas_com_ganho",
                "probabilidade_ganho",
                "retorno_medio",
                "retorno_mediano",
                "melhor_janela",
                "pior_janela",
                "retorno_mais_recente",
                "inicio_mais_recente",
                "fim_mais_recente",
            ]
        )

    base = rolling.copy()
    grouped = base.groupby("serie", observed=True)
    summary = grouped.agg(
        janelas_validas=("retorno_janela", "size"),
        janelas_com_ganho=("ganho_positivo", "sum"),
        probabilidade_ganho=("ganho_positivo", "mean"),
        retorno_medio=("retorno_janela", "mean"),
        retorno_mediano=("retorno_janela", "median"),
        melhor_janela=("retorno_janela", "max"),
        pior_janela=("retorno_janela", "min"),
    ).reset_index()

    recent_idx = base.sort_values("periodo").groupby("serie", observed=True).tail(1).index
    recent = base.loc[recent_idx, ["serie", "retorno_janela", "inicio_janela", "fim_janela"]].rename(
        columns={
            "retorno_janela": "retorno_mais_recente",
            "inicio_janela": "inicio_mais_recente",
            "fim_janela": "fim_mais_recente",
        }
    )

    summary = summary.merge(recent, on="serie", how="left")
    return summary.sort_values(["probabilidade_ganho", "retorno_mediano", "janelas_validas"], ascending=[False, False, False])


def rolling_summary_for_windows(
    ts_bairro: pd.DataFrame,
    ts_benchmark: pd.DataFrame,
    windows: Sequence[int],
    min_trans: int,
) -> pd.DataFrame:
    frames = []
    for window in windows:
        rolling_bairros = rolling_returns_by_series(ts_bairro, window, min_trans, group_col="bairro")
        rolling_bench = rolling_returns_by_series(ts_benchmark, window, min_trans, series_name=BENCHMARK_SERIES_NAME)
        rolling = pd.concat([rolling_bairros, rolling_bench], ignore_index=True)
        if rolling.empty:
            continue
        summary = rolling_gain_summary(rolling)
        summary.insert(0, "janela_meses", window)
        frames.append(summary)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def month_span(start: pd.Timestamp, end: pd.Timestamp) -> int:
    start_p = pd.Timestamp(start).to_period("M")
    end_p = pd.Timestamp(end).to_period("M")
    return int((end_p.year - start_p.year) * 12 + (end_p.month - start_p.month))

def apply_outlier_filter(df: pd.DataFrame, col: str, low_q: float, high_q: float) -> tuple[pd.DataFrame, float, float]:
    valid = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty or low_q <= 0 and high_q >= 1:
        return df, np.nan, np.nan
    lo, hi = valid.quantile([low_q, high_q])
    filtered = df[(df[col].isna()) | ((df[col] >= lo) & (df[col] <= hi))].copy()
    return filtered, float(lo), float(hi)


def rank_by_volume(df: pd.DataFrame, col: str, top_n: int) -> list[str]:
    if df.empty or col not in df.columns:
        return []
    ranking = df.groupby(col, observed=True)[TX_COL].sum().sort_values(ascending=False)
    return ranking.head(top_n).index.astype(str).tolist()


def ordered_series_last_value(df: pd.DataFrame, series_col: str, value_col: str) -> list[str]:
    """Ordena séries pelo valor mais recente, em ordem decrescente.
    Isso melhora a leitura do hover unificado e da legenda."""
    if df.empty or series_col not in df.columns or value_col not in df.columns:
        return []
    ordered = (
        df.sort_values([series_col, "periodo"])
        .groupby(series_col, observed=True)[value_col]
        .last()
        .sort_values(ascending=False)
    )
    return ordered.index.astype(str).tolist()


def ranking_logradouros(df: pd.DataFrame, min_trans: int) -> pd.DataFrame:
    rank = aggregate_metrics(df, ["bairro", "logradouro", "uso"])
    rank = rank[rank["total_transações"] >= min_trans].copy()
    return rank.sort_values("preco_m2_corrigido", ascending=False)


def ranking_bairros(df: pd.DataFrame, min_trans: int) -> pd.DataFrame:
    rank = aggregate_metrics(df, ["bairro"])
    rank = rank[rank["total_transações"] >= min_trans].copy()
    return rank.sort_values("preco_m2_corrigido", ascending=False)




MONTH_LABELS_PT = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


def br_pct_signed(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return "—"
    sign = "+" if value >= 0 else "−"
    return sign + br_pct(abs(value), decimals)


def calc_spread_mediana_media(row: pd.Series) -> float:
    media = row.get("preco_m2_corrigido", np.nan)
    mediana = row.get("mediana_preco_m2", np.nan)
    if pd.isna(media) or pd.isna(mediana) or not np.isfinite(media) or media == 0:
        return np.nan
    return (mediana - media) / media


def calc_recent_trend_6m(filtered: pd.DataFrame) -> float:
    """Compara o preço/m² médio dos 6 últimos meses disponíveis com os 6 meses anteriores."""
    ts = time_group(filtered, "Mensal").sort_values("periodo")
    ts = ts[(ts["total_transações"] > 0) & ts["preco_m2_corrigido"].notna()].copy()
    if len(ts) < 12:
        return np.nan
    last_6 = ts.tail(6)["preco_m2_corrigido"].mean()
    prev_6 = ts.iloc[-12:-6]["preco_m2_corrigido"].mean()
    if pd.isna(last_6) or pd.isna(prev_6) or prev_6 == 0:
        return np.nan
    return (last_6 / prev_6) - 1


def monthly_price_band(raw_df: pd.DataFrame) -> pd.DataFrame:
    """P25/P75 mensal calculado diretamente no campo de preço/m² linha a linha."""
    if raw_df.empty:
        return pd.DataFrame(columns=["periodo", "p25_preco_m2", "p75_preco_m2"])
    data = raw_df[[DATE_COL, PRICE_COL]].dropna().copy()
    if data.empty:
        return pd.DataFrame(columns=["periodo", "p25_preco_m2", "p75_preco_m2"])
    data["periodo"] = data[DATE_COL].dt.to_period("M").dt.to_timestamp()
    band = (
        data.groupby("periodo", observed=True)[PRICE_COL]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "p25_preco_m2", 0.75: "p75_preco_m2"})
        .reset_index()
        .sort_values("periodo")
    )
    return band


def valuation_ranking(ts_bairro: pd.DataFrame, min_trans: int) -> pd.DataFrame:
    """Variação real acumulada por bairro entre o primeiro e o último período com volume mínimo."""
    rows: list[dict[str, object]] = []
    if ts_bairro.empty:
        return pd.DataFrame(
            columns=[
                "bairro",
                "periodo_inicial",
                "periodo_final",
                "preco_m2_inicial",
                "preco_m2_final",
                "variacao_real_acumulada",
                "total_transações",
            ]
        )

    for bairro, part in ts_bairro.groupby("bairro", observed=True):
        valid = (
            part[(part["total_transações"] >= min_trans) & part["preco_m2_corrigido"].notna()]
            .sort_values("periodo")
            .copy()
        )
        if len(valid) < 2:
            continue
        first = valid.iloc[0]
        last = valid.iloc[-1]
        start_price = first["preco_m2_corrigido"]
        end_price = last["preco_m2_corrigido"]
        if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
            continue
        rows.append(
            {
                "bairro": str(bairro),
                "periodo_inicial": first["periodo"],
                "periodo_final": last["periodo"],
                "preco_m2_inicial": start_price,
                "preco_m2_final": end_price,
                "variacao_real_acumulada": (end_price / start_price) - 1,
                "total_transações": valid["total_transações"].sum(),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "bairro",
                "periodo_inicial",
                "periodo_final",
                "preco_m2_inicial",
                "preco_m2_final",
                "variacao_real_acumulada",
                "total_transações",
            ]
        )
    return out.sort_values("variacao_real_acumulada", ascending=True)


def seasonality_table(filtered: pd.DataFrame) -> pd.DataFrame:
    """Média de preço/m² e de transações por mês calendário no período filtrado."""
    if filtered.empty:
        return pd.DataFrame(columns=["mes_num", "mes", "media_preco_m2", "media_transacoes"])

    monthly = time_group(filtered, "Mensal").copy()
    if monthly.empty:
        return pd.DataFrame(columns=["mes_num", "mes", "media_preco_m2", "media_transacoes"])

    monthly["mes_num"] = monthly["periodo"].dt.month
    seasonal = (
        monthly.groupby("mes_num", observed=True)
        .agg(
            media_preco_m2=("preco_m2_corrigido", "mean"),
            media_transacoes=("total_transações", "mean"),
        )
        .reset_index()
    )
    seasonal["mes"] = seasonal["mes_num"].map(lambda m: MONTH_LABELS_PT[int(m) - 1])
    seasonal = seasonal.sort_values("mes_num")
    return seasonal[["mes_num", "mes", "media_preco_m2", "media_transacoes"]]


def ranking_aceleracao_recente(filtered: pd.DataFrame, min_trans: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    monthly = time_group(filtered, "Mensal", ["bairro"]).sort_values(["bairro", "periodo"])
    if monthly.empty:
        return pd.DataFrame(columns=["bairro", "aceleracao_recente", "tendencia_ultimos_6m", "tendencia_6m_anteriores", "total_transações"])

    for bairro, part in monthly.groupby("bairro", observed=True):
        part = part[(part["total_transações"] >= min_trans) & part["preco_m2_corrigido"].notna()].copy()
        if len(part) < 12:
            continue
        last_6 = part.tail(6)["preco_m2_corrigido"].mean()
        prev_6 = part.iloc[-12:-6]["preco_m2_corrigido"].mean()
        if pd.isna(last_6) or pd.isna(prev_6) or prev_6 == 0:
            continue
        tendencia = (last_6 / prev_6) - 1
        if len(part) >= 18:
            base_prev = part.iloc[-18:-12]["preco_m2_corrigido"].mean()
            tendencia_prev = (prev_6 / base_prev) - 1 if pd.notna(base_prev) and base_prev != 0 else np.nan
        else:
            tendencia_prev = np.nan
        aceleracao = tendencia - tendencia_prev if pd.notna(tendencia_prev) else tendencia
        rows.append({"bairro": str(bairro), "aceleracao_recente": aceleracao, "tendencia_ultimos_6m": tendencia, "tendencia_6m_anteriores": tendencia_prev, "total_transações": part["total_transações"].sum()})
    out = pd.DataFrame(rows)
    return out.sort_values("aceleracao_recente", ascending=False) if not out.empty else pd.DataFrame(columns=["bairro", "aceleracao_recente", "tendencia_ultimos_6m", "tendencia_6m_anteriores", "total_transações"])


def transfer_quality_summary(filtered: pd.DataFrame) -> pd.DataFrame:
    if filtered.empty or "média_percentual_transferido" not in filtered.columns:
        return pd.DataFrame(columns=["categoria", "transações", "participação"])
    ratio = normalize_transfer_ratio(filtered["média_percentual_transferido"])
    tx = filtered[TX_COL].fillna(0)
    parcial = float(tx[(ratio < 0.999)].sum())
    integral = float(tx[(ratio.isna()) | (ratio >= 0.999)].sum())
    total = parcial + integral
    out = pd.DataFrame({"categoria": ["Transferência integral", "Transferência parcial"], "transações": [integral, parcial]})
    out["participação"] = out["transações"] / total if total else np.nan
    return out


def make_transfer_quality_chart(summary: pd.DataFrame) -> go.Figure:
    if summary.empty:
        return go.Figure()
    colors = [ACCENT, BORDO_2]
    fig = go.Figure(go.Bar(x=summary["participação"], y=summary["categoria"], orientation="h", marker=dict(color=colors), text=summary["participação"].map(lambda v: br_pct(v, 1)), textposition="auto", cliponaxis=False, customdata=summary["transações"], hovertemplate="%{y}<br>Participação: %{x:.1%}<br>Transações: %{customdata:,.0f}<extra></extra>"))
    fig.update_layout(title=None, height=330, showlegend=False, margin=dict(l=90, r=86, t=42, b=64))
    fig.update_xaxes(title_text="Participação no volume de transações", tickformat=".0%", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title_text="", showgrid=False)
    fig = clean_axes(fig, y_money=False)
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    return fig


def make_area_price_distribution_chart(filtered: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Distribuição de preço/m² por faixa de área.

    A base do painel já vem agregada por logradouro/período; por isso,
    usar média_área_construída como eixo X em scatter cria compressão visual
    e sugere granularidade transacional que não existe. O box plot por
    faixa_area é mais honesto com a estrutura da base.
    """
    if filtered.empty:
        return go.Figure()

    base = assign_area_labels_exclusive(filtered)
    top_bairros = rank_by_volume(base, "bairro", top_n=top_n)
    plot = base[base["bairro"].isin(top_bairros)].copy()
    plot = plot[["bairro", "faixa_area", PRICE_COL, TX_COL, DATE_COL, "logradouro"]].dropna()
    plot = plot[plot["faixa_area"].isin(AREA_LABELS)].copy()
    if plot.empty:
        return go.Figure()

    fig = px.box(
        plot,
        x="faixa_area",
        y=PRICE_COL,
        color="bairro",
        points="suspectedoutliers",
        category_orders={"faixa_area": AREA_LABELS, "bairro": top_bairros},
        custom_data=["bairro", "logradouro", TX_COL, DATE_COL],
        labels={
            "faixa_area": "Faixa de área construída",
            PRICE_COL: "Preço/m² corrigido",
            "bairro": "Bairro",
        },
    )
    fig.update_traces(
        boxmean=True,
        quartilemethod="linear",
        boxpoints="outliers",
        jitter=0,
        pointpos=0,
        marker=dict(size=4, opacity=0.55),
        line=dict(width=1.1),
        hovertemplate=(
            "Faixa: %{x}<br>"
            "Preço/m²: R$ %{y:,.0f}<br>"
            "Bairro: %{customdata[0]}<br>"
            "Logradouro: %{customdata[1]}<br>"
            "Transações do registro: %{customdata[2]:,.0f}<br>"
            "Data: %{customdata[3]|%d-%m-%Y}<extra></extra>"
        ),
    )
    fig.update_layout(
        title=None,
        height=640,
        hovermode="closest",
        boxmode="group",
        margin=dict(l=78, r=220, t=60, b=108),
        legend=dict(
            title="Bairro",
            orientation="v",
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12, color=INK),
        ),
    )
    fig.update_yaxes(title_text="Preço/m² corrigido", tickprefix="R$ ")
    fig.update_xaxes(title_text="Faixa de área construída", categoryorder="array", categoryarray=AREA_LABELS, tickangle=-20)
    return clean_axes(fig)


# Alias mantido para não quebrar chamadas antigas em versões já editadas do app.
def make_area_price_scatter(filtered: pd.DataFrame, top_n: int = 10) -> go.Figure:
    return make_area_price_distribution_chart(filtered, top_n=top_n)


def calcular_percentis_area_bairro(
    df_linha: pd.DataFrame,
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Calcula P25/P75/P90 do preço/m² por bairro × faixa de área.

    Usa o campo PRICE_COL linha a linha. A faixa de área é recalculada com
    limite superior exclusivo para impedir dupla contagem em pontos de corte
    como 55 m² e 75 m².
    """
    group_cols = list(group_cols or ["bairro", "faixa_area"])
    result_cols = group_cols + ["p25_preco_m2", "p75_preco_m2", "p90_preco_m2"]
    if df_linha.empty or PRICE_COL not in df_linha.columns:
        return pd.DataFrame(columns=result_cols)

    work = assign_area_labels_exclusive(df_linha)
    required = [c for c in group_cols if c in work.columns] + [PRICE_COL]
    if len(required) != len(group_cols) + 1:
        return pd.DataFrame(columns=result_cols)

    work = work.dropna(subset=required).copy()
    work = work[work[PRICE_COL] > 0].copy()
    if work.empty:
        return pd.DataFrame(columns=result_cols)

    pct = (
        work.groupby(group_cols, dropna=False, observed=True)[PRICE_COL]
        .quantile([0.25, 0.75, 0.90])
        .unstack()
    )
    pct = pct.rename(columns={0.25: "p25_preco_m2", 0.75: "p75_preco_m2", 0.90: "p90_preco_m2"})
    pct = pct.reset_index()
    return pct[result_cols]


def adicionar_percentis_area_bairro(
    df_agregado: pd.DataFrame,
    df_linha: pd.DataFrame | None = None,
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Garante p25_preco_m2, p75_preco_m2 e p90_preco_m2 no agregado."""
    group_cols = list(group_cols or ["bairro", "faixa_area"])
    needed = ["p25_preco_m2", "p75_preco_m2", "p90_preco_m2"]
    out = df_agregado.copy()
    if all(col in out.columns for col in needed):
        return out
    if df_linha is None or df_linha.empty:
        for col in needed:
            if col not in out.columns:
                out[col] = np.nan
        return out

    pct = calcular_percentis_area_bairro(df_linha, group_cols=group_cols)
    if pct.empty:
        for col in needed:
            if col not in out.columns:
                out[col] = np.nan
        return out

    cols_to_drop = [col for col in needed if col in out.columns]
    if cols_to_drop:
        out = out.drop(columns=cols_to_drop)
    return out.merge(pct, on=group_cols, how="left")


def interpretar_area_bairro(
    df: pd.DataFrame,
    min_trans: int = 10,
) -> dict[str, dict[str, str]]:
    interpretacoes: dict[str, dict[str, str]] = {}
    required = {
        "faixa_area",
        "bairro",
        "mediana_preco_m2",
        "preco_m2_corrigido",
        "total_transações",
        "p25_preco_m2",
        "p75_preco_m2",
        "p90_preco_m2",
    }
    if df.empty or not required.issubset(df.columns):
        return interpretacoes

    work = df.copy()
    work = work[work["total_transações"] >= min_trans].copy()
    work = work.dropna(
        subset=[
            "bairro",
            "faixa_area",
            "mediana_preco_m2",
            "preco_m2_corrigido",
            "p25_preco_m2",
            "p75_preco_m2",
            "p90_preco_m2",
        ]
    )
    if work.empty:
        return interpretacoes

    for _, row in work.iterrows():
        bairro = str(row["bairro"])
        faixa = str(row["faixa_area"])
        mediana = float(row["mediana_preco_m2"])
        media = float(row["preco_m2_corrigido"])
        p25 = float(row["p25_preco_m2"])
        p75 = float(row["p75_preco_m2"])
        p90 = float(row["p90_preco_m2"])
        total_transacoes = float(row["total_transações"])

        if not all(np.isfinite(v) for v in [mediana, media, p25, p75, p90, total_transacoes]) or mediana <= 0:
            continue

        faixa_iqr = p75 - p25
        frase_1 = (
            f"A maioria dos imóveis de {faixa} em {bairro} foi negociada entre "
            f"{br_money(p25)} e {br_money(p75)}/m²."
        )
        if faixa_iqr > 0.5 * mediana:
            frase_1 += " — uma faixa ampla, indicando imóveis bem variados nesse segmento."
        elif faixa_iqr < 0.1 * mediana:
            frase_1 += " — uma faixa estreita, indicando imóveis bastante homogêneos."

        frase_2 = f"O valor mais comum foi {br_money(mediana)}/m²."
        if abs(mediana - media) / mediana > 0.15:
            frase_2 += f" Alguns negócios fora do padrão puxaram a média para {br_money(media)}/m²."

        frases = [frase_1, frase_2]
        if p90 > p75 * 1.4:
            frases.append(f"Alguns imóveis atípicos chegaram a {br_money(p90)}/m² ou mais.")
        frases.append(f"Baseado em {br_number(total_transacoes)} negócios registrados.")

        interpretacoes.setdefault(bairro, {})[faixa] = " ".join(frases)

    return interpretacoes


def render_interpretacao_area(
    interpretacoes: dict[str, dict[str, str]],
    bairros_selecionados: list[str],
    faixas_ordenadas: list[str],
    min_trans: int,
) -> None:
    bairros = [str(b) for b in bairros_selecionados]
    if not bairros:
        bairros = sorted(interpretacoes.keys())

    for bairro in bairros:
        with st.expander(str(bairro), expanded=False):
            textos_bairro = interpretacoes.get(str(bairro), {})
            exibiu = False
            for faixa in faixas_ordenadas:
                texto = textos_bairro.get(str(faixa))
                if texto:
                    texto_md = str(texto).replace("$", r"\$")
                    st.markdown(f"**{faixa}**  \n{texto_md}")
                    exibiu = True
            if not exibiu:
                st.caption("Dados insuficientes para este bairro no recorte atual.")


def make_product_mix_chart(filtered: pd.DataFrame, top_n: int = 10) -> go.Figure:
    if filtered.empty:
        return go.Figure()

    base = assign_area_labels_exclusive(filtered)
    top_bairros = rank_by_volume(base, "bairro", top_n=top_n)
    base = base[base["bairro"].isin(top_bairros)].copy()
    base = base[base["faixa_area"].isin(AREA_LABELS)].copy()

    mix = (
        base.groupby(["bairro", "faixa_area"], observed=True)[TX_COL]
        .sum()
        .reset_index(name="transações")
    )
    if mix.empty:
        return go.Figure()

    totals = mix.groupby("bairro", observed=True)["transações"].sum().rename("total_bairro").reset_index()
    mix = mix.merge(totals, on="bairro", how="left")
    mix["participação"] = mix["transações"] / mix["total_bairro"]
    order_map = {label: idx for idx, label in enumerate(AREA_LABELS)}
    mix["ordem"] = mix["faixa_area"].map(order_map)
    mix = mix.sort_values(["ordem", "bairro"])

    fig = px.bar(
        mix,
        x="bairro",
        y="participação",
        color="faixa_area",
        category_orders={"faixa_area": AREA_LABELS, "bairro": top_bairros},
        labels={"bairro": "Bairro", "participação": "Participação", "faixa_area": "Faixa de área"},
    )
    fig.update_traces(
        customdata=np.stack([mix["transações"]], axis=-1),
        hovertemplate=(
            "Bairro: %{x}<br>"
            "Faixa: %{fullData.name}<br>"
            "Participação: %{y:.1%}<br>"
            "Transações da faixa: %{customdata[0]:,.0f}<extra></extra>"
        ),
    )
    fig.update_layout(
        title=None,
        height=590,
        barmode="stack",
        hovermode="closest",
        margin=dict(l=72, r=42, t=60, b=128),
        legend=dict(
            title="Faixa de área",
            orientation="h",
            x=0,
            y=-0.18,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12, color=INK),
        ),
    )
    fig.update_yaxes(title_text="Participação no bairro", tickformat=".0%", range=[0, 1])
    fig.update_xaxes(title_text="", tickangle=-20)
    return clean_axes(fig, y_money=False)


def make_acceleration_chart(rank: pd.DataFrame, top_n: int = 12) -> go.Figure:
    if rank.empty:
        return go.Figure()
    half = max(1, top_n // 2)
    plot = pd.concat([rank.head(half), rank.tail(half)]).drop_duplicates("bairro")
    plot = plot.sort_values("aceleracao_recente", ascending=True)
    colors = np.where(plot["aceleracao_recente"] >= 0, POSITIVE, NEGATIVE)
    fig = go.Figure(go.Bar(x=plot["aceleracao_recente"], y=plot["bairro"], orientation="h", marker=dict(color=colors), text=plot["aceleracao_recente"].map(lambda v: br_pct_signed(v, 1)), textposition="inside", insidetextanchor="middle", textfont=dict(color="#FFFFFF", size=11), cliponaxis=False, customdata=np.stack([plot["tendencia_ultimos_6m"], plot["tendencia_6m_anteriores"], plot["total_transações"]], axis=-1), hovertemplate="<b>%{y}</b><br>Aceleração recente: %{text}<br>Tendência últimos 6 meses: %{customdata[0]:.1%}<br>Tendência 6 meses anteriores: %{customdata[1]:.1%}<br>Transações: %{customdata[2]:,.0f}<extra></extra>"))
    fig.add_vline(x=0, line_width=1.2, line_dash="solid", line_color="rgba(148,163,184,0.70)")
    fig.update_layout(title=None, height=max(440, 30 * len(plot) + 150), showlegend=False, margin=dict(l=132, r=96, t=48, b=64))
    fig.update_xaxes(title_text="Aceleração recente", tickformat=".0%", showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(title_text="", showgrid=False)
    fig = clean_axes(fig, y_money=False)
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    return fig


def budget_finder(rank_bairros: pd.DataFrame, budget_max: float, target_area: float, min_trans: int) -> pd.DataFrame:
    if rank_bairros.empty or budget_max <= 0 or target_area <= 0:
        return pd.DataFrame(columns=["bairro", "mediana_preco_m2", "preco_m2_corrigido", "custo_estimado", "bairro_alcancavel", "total_transações"])
    base = rank_bairros.copy()
    base = base[base["total_transações"] >= min_trans].copy()
    base["orcamento_maximo"] = budget_max
    base["metragem_desejada"] = target_area
    base["orcamento_por_m2"] = budget_max / target_area
    base["custo_estimado"] = base["mediana_preco_m2"] * target_area
    base["bairro_alcancavel"] = np.where(base["custo_estimado"] <= budget_max, "Sim", "Não")
    return base.sort_values(["bairro_alcancavel", "custo_estimado", "total_transações"], ascending=[False, True, False])


def make_budget_finder_map(budget_df: pd.DataFrame, geo: pd.DataFrame) -> go.Figure:
    if budget_df.empty or geo.empty:
        return go.Figure()
    bairro_centers = geo.groupby("bairro", observed=True)[["latitude", "longitude"]].mean().reset_index()
    plot = budget_df.merge(bairro_centers, on="bairro", how="left").dropna(subset=["latitude", "longitude"])
    if plot.empty:
        return go.Figure()
    plot["alcancavel_flag"] = plot["bairro_alcancavel"].map({"Sim": 1, "Não": 0})
    colors = [[0.0, NEGATIVE], [0.4999, NEGATIVE], [0.5, POSITIVE], [1.0, POSITIVE]]
    fig = go.Figure(go.Scattermapbox(lat=plot["latitude"], lon=plot["longitude"], mode="markers+text", text=plot["bairro"], textposition="top center", marker=dict(size=16, color=plot["alcancavel_flag"], colorscale=colors, cmin=0, cmax=1, showscale=False, opacity=0.86), customdata=np.stack([plot["mediana_preco_m2"], plot["custo_estimado"], plot["total_transações"], plot["bairro_alcancavel"]], axis=-1), hovertemplate="<b>%{text}</b><br>Preço mediano/m²: R$ %{customdata[0]:,.0f}<br>Custo estimado: R$ %{customdata[1]:,.0f}<br>Transações: %{customdata[2]:,.0f}<br>Alcançável: %{customdata[3]}<extra></extra>"))
    fig.update_layout(
        mapbox=dict(style="carto-positron", zoom=9.9, center=dict(lat=float(plot["latitude"].mean()), lon=float(plot["longitude"].mean()))),
        height=520,
        margin=dict(l=8, r=8, t=8, b=8),
        template=PLOT_TEMPLATE,
        paper_bgcolor=PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(color=INK, family="Inter, DM Sans, Arial, sans-serif"),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="#334155", font=dict(color="#F8FAFC", family="Inter", size=12)),
    )
    return fig


def render_orientation_panel() -> None:
    with st.expander("Como ler este painel", expanded=False):
        st.markdown(
            """
**ITBI** é o imposto pago quando há transmissão de propriedade imobiliária. Como ele é registrado a partir de transações efetivas, a base tende a ser mais próxima do mercado realizado do que anúncios de venda.

**Preço corrigido pelo IPCA** significa que os valores antigos foram trazidos para o poder de compra de março de 2026. Exemplo: se um imóvel aparece como R$ 10.000/m² corrigido, ele está expresso em dinheiro de março de 2026, permitindo comparar períodos diferentes sem confundir inflação com valorização real.

**Preço médio ponderado** divide o valor total movimentado pela área total registrada. Ele dá mais peso aos imóveis maiores ou aos conjuntos de transações com maior área. **Mediana** é o ponto central: metade das transações fica abaixo e metade fica acima. Quando média e mediana se afastam, pode haver imóveis muito caros ou muito baratos puxando a média.

Use os filtros laterais para restringir período, bairros, uso, tipologia, tamanho do imóvel e logradouro. Quanto mais específico o filtro, maior o risco de a amostra ficar pequena; por isso, observe também a quantidade de transações.

Os preços são por metro quadrado de **área construída registrada** na base, não necessariamente a área total percebida pelo comprador.
"""
        )

# -----------------------------------------------------------------------------
# Gráficos — desenho deliberadamente limpo
# -----------------------------------------------------------------------------


def clean_axes(fig: go.Figure, *, y_money: bool = True) -> go.Figure:
    """Aplica acabamento visual sem destruir eixos já configurados.

    Versões anteriores aplicavam prefixo "R$" em todos os eixos Y. Em gráficos
    com subplots isso fazia o volume de transações aparecer como dinheiro. Aqui
    o prefixo monetário só é aplicado ao eixo Y que ainda não tem prefixo definido.
    """
    fig.update_layout(
        template=PLOT_TEMPLATE,
        separators=",.",
        paper_bgcolor=PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter, DM Sans, Arial, sans-serif", color=INK, size=13),
        title=None,
        title_font=dict(size=18, color=INK),
        hoverlabel=dict(
            bgcolor="#0F172A",
            bordercolor="#334155",
            font=dict(color="#F8FAFC", family="Inter, DM Sans, Arial, sans-serif", size=12),
        ),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    fig.update_xaxes(
        showline=False,
        ticks="outside",
        tickcolor=GRID,
        showgrid=False,
        zeroline=False,
        automargin=True,
        separatethousands=True,
    )
    fig.update_yaxes(
        showline=False,
        showgrid=True,
        gridcolor=GRID,
        ticks="outside",
        tickcolor=GRID,
        zeroline=False,
        separatethousands=True,
        automargin=True,
    )
    money_terms = ("r$", "preço", "valor", "custo", "m²", "m2")
    for axis_name in fig.layout:
        if str(axis_name).startswith("yaxis"):
            axis = fig.layout[axis_name]
            title_text = str(getattr(getattr(axis, "title", None), "text", "") or "").lower()
            is_money_axis = any(term in title_text for term in money_terms)
            if y_money and is_money_axis and getattr(axis, "tickprefix", None) in (None, ""):
                axis.tickprefix = "R$ "
            axis.separatethousands = True
            axis.automargin = True
    return fig

def make_price_volume_chart(ts: pd.DataFrame, title: str, raw_df: pd.DataFrame | None = None) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.055,
    )

    if raw_df is not None and not raw_df.empty:
        band = monthly_price_band(raw_df)
        if not band.empty:
            fig.add_trace(
                go.Scatter(
                    x=band["periodo"],
                    y=band["p75_preco_m2"],
                    mode="lines",
                    name="P75 preço/m²",
                    line=dict(width=0, color=CONTEXT),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=band["periodo"],
                    y=band["p25_preco_m2"],
                    mode="lines",
                    name="Faixa típica P25–P75",
                    line=dict(width=0, color=CONTEXT),
                    fill="tonexty",
                    fillcolor="rgba(148, 163, 184, 0.18)",
                    customdata=np.array(list(zip(band["p25_preco_m2"], band["p75_preco_m2"])), dtype=object),
                    hovertemplate=(
                        "%{x|%d/%m/%Y}<br>"
                        "Faixa típica P25–P75<br>"
                        "P25: R$ %{customdata[0]:,.0f}/m²<br>"
                        "P75: R$ %{customdata[1]:,.0f}/m²<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=ts["periodo"],
            y=ts["preco_m2_corrigido"],
            mode="lines+markers",
            name="Preço/m² médio ponderado",
            line=dict(width=2.2, color=ACCENT),
            marker=dict(size=5, color=ACCENT),
            hovertemplate="%{x|%d/%m/%Y}<br>Preço/m²: R$ %{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ts["periodo"],
            y=ts["total_transações"],
            name="Transações",
            marker=dict(color=VOLUME, line=dict(width=0)),
            hovertemplate="%{x|%d/%m/%Y}<br>Transações: %{y:,.0f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=None,
        height=590,
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=88, r=48, t=64, b=80),
    )
    fig.update_yaxes(title_text="R$/m²", row=1, col=1, tickprefix="R$ ", gridcolor=GRID)
    fig.update_yaxes(title_text="Transações", row=2, col=1, tickprefix=None, gridcolor=GRID, title_standoff=12)
    fig.update_xaxes(title_text="Período", row=2, col=1, automargin=True)
    return clean_axes(fig)


def make_valuation_ranking_chart(rank: pd.DataFrame, title: str, top_n: int = 25) -> go.Figure:
    if rank.empty:
        return go.Figure()

    plot = rank.sort_values("variacao_real_acumulada", ascending=True).tail(top_n).copy()
    plot["label"] = plot["variacao_real_acumulada"].map(lambda v: br_pct_signed(v, 1))
    colors = np.where(plot["variacao_real_acumulada"] >= 0, POSITIVE, NEGATIVE)

    fig = go.Figure(
        go.Bar(
            x=plot["variacao_real_acumulada"],
            y=plot["bairro"],
            orientation="h",
            marker=dict(color=colors),
            text=plot["label"],
            textposition="auto",
            cliponaxis=False,
            customdata=np.array(
                list(zip(plot["periodo_inicial"], plot["periodo_final"], plot["preco_m2_inicial"], plot["preco_m2_final"], plot["total_transações"])),
                dtype=object,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Variação real: %{text}<br>"
                "Período: %{customdata[0]|%m/%Y} → %{customdata[1]|%m/%Y}<br>"
                "Preço inicial: R$ %{customdata[2]:,.0f}/m²<br>"
                "Preço final: R$ %{customdata[3]:,.0f}/m²<br>"
                "Transações nos períodos válidos: %{customdata[4]:,.0f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_width=1.2, line_dash="solid", line_color="rgba(148,163,184,0.70)")
    fig.update_layout(
        title=None,
        height=max(460, 28 * len(plot) + 160),
        hovermode="y unified",
        showlegend=False,
        margin=dict(l=132, r=96, t=48, b=62),
    )
    fig.update_traces(textposition="inside", textfont=dict(color="#FFFFFF", size=11))
    fig.update_xaxes(title_text="Variação real acumulada", tickformat=".0%", automargin=True, showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(title_text="", automargin=True, showgrid=False)
    fig = clean_axes(fig, y_money=False)
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    return fig


def make_seasonality_chart(seasonal: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.055,
    )

    if seasonal.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=seasonal["mes"],
            y=seasonal["media_preco_m2"],
            mode="lines+markers",
            name="Preço/m² médio",
            line=dict(width=2.2, color=ACCENT),
            marker=dict(size=6, color=ACCENT),
            hovertemplate="%{x}<br>Preço/m² médio: R$ %{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=seasonal["mes"],
            y=seasonal["media_transacoes"],
            mode="lines+markers",
            name="Transações médias",
            line=dict(width=2.0, color=ACCENT2),
            marker=dict(size=6, color=ACCENT2),
            hovertemplate="%{x}<br>Transações médias: %{y:,.0f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=None,
        height=570,
        hovermode="x unified",
        showlegend=True,
        margin=dict(l=78, r=42, t=54, b=72),
    )
    fig.update_xaxes(categoryorder="array", categoryarray=MONTH_LABELS_PT, automargin=True)
    fig.update_yaxes(title_text="R$/m²", row=1, col=1, tickprefix="R$ ", gridcolor=GRID)
    fig.update_yaxes(title_text="Transações", row=2, col=1, tickprefix=None, gridcolor=GRID, title_standoff=12)
    fig.update_xaxes(title_text="Mês do ano", row=2, col=1)
    return clean_axes(fig)


def make_level_comparison_chart(
    ts_bairro: pd.DataFrame,
    ts_bench: pd.DataFrame,
    title: str,
    top_n: int,
    selected_bairros: list[str] | None = None,
) -> go.Figure:
    if ts_bairro.empty:
        return go.Figure()

    selected_bairros = [str(b) for b in (selected_bairros or [])]
    if selected_bairros:
        plot = ts_bairro[ts_bairro["bairro"].isin(selected_bairros)].copy()
    else:
        top_bairros = (
            ts_bairro.groupby("bairro", observed=True)["total_transações"].sum().sort_values(ascending=False).head(top_n).index
        )
        plot = ts_bairro[ts_bairro["bairro"].isin(top_bairros)].copy()

    fig = go.Figure()
    order_bairros = ordered_series_last_value(plot, "bairro", "preco_m2_corrigido")
    for bairro in order_bairros:
        part = plot[plot["bairro"].astype(str) == str(bairro)].sort_values("periodo")
        fig.add_trace(
            go.Scatter(
                x=part["periodo"],
                y=part["preco_m2_corrigido"],
                mode="lines+markers",
                name=str(bairro),
                line=dict(width=2.0),
                marker=dict(size=5),
                hovertemplate=f"{bairro}<br>%{{x|%d/%m/%Y}}<br>R$ %{{y:,.0f}}/m²<extra></extra>",
            )
        )

    if not ts_bench.empty:
        fig.add_trace(
            go.Scatter(
                x=ts_bench["periodo"],
                y=ts_bench["preco_m2_corrigido"],
                mode="lines",
                name="Média do recorte de comparação",
                line=dict(width=3.4, color=BENCHMARK, dash="dash"),
                hovertemplate="Média do recorte de comparação<br>%{x|%d/%m/%Y}<br>R$ %{y:,.0f}/m²<extra></extra>",
            )
        )

    fig.update_layout(
        title=None,
        height=610,
        hovermode="x unified",
        margin=dict(l=84, r=52, t=96, b=146),
        legend=dict(
            title=None,
            orientation="h",
            x=0.0,
            y=-0.18,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12, color=INK),
            traceorder="normal",
        ),
    )
    fig.update_yaxes(title_text="R$/m² corrigido", title_standoff=16)
    fig.update_xaxes(title_text="Período", title_standoff=14)
    return clean_axes(fig)


def make_index_chart(
    ts_bairro: pd.DataFrame,
    ts_bench: pd.DataFrame,
    title: str,
    top_n: int,
    selected_bairros: list[str] | None = None,
) -> go.Figure:
    selected_bairros = [str(b) for b in (selected_bairros or [])]
    if selected_bairros:
        indexed = index_series(ts_bairro[ts_bairro["bairro"].isin(selected_bairros)], ["bairro"])
    else:
        top_bairros = (
            ts_bairro.groupby("bairro", observed=True)["total_transações"].sum().sort_values(ascending=False).head(top_n).index
        )
        indexed = index_series(ts_bairro[ts_bairro["bairro"].isin(top_bairros)], ["bairro"])
    bench_indexed = index_series(ts_bench)

    fig = go.Figure()
    order_bairros = ordered_series_last_value(indexed, "bairro", "indice_100")
    for bairro in order_bairros:
        part = indexed[indexed["bairro"].astype(str) == str(bairro)].sort_values("periodo")
        part = part.copy()
        part["rendimento_pct"] = part["indice_100"] / 100.0 - 1.0
        fig.add_trace(
            go.Scatter(
                x=part["periodo"],
                y=part["rendimento_pct"],
                mode="lines+markers",
                name=str(bairro),
                line=dict(width=2.0),
                marker=dict(size=5),
                hovertemplate=f"{bairro}<br>%{{x|%d/%m/%Y}}<br>Rendimento acumulado: %{{y:.1%}}<extra></extra>",
            )
        )

    if not bench_indexed.empty:
        bench_indexed = bench_indexed.copy()
        bench_indexed["rendimento_pct"] = bench_indexed["indice_100"] / 100.0 - 1.0
        fig.add_trace(
            go.Scatter(
                x=bench_indexed["periodo"],
                y=bench_indexed["rendimento_pct"],
                mode="lines",
                name="Média do recorte de comparação",
                line=dict(width=3.4, color=BENCHMARK, dash="dash"),
                hovertemplate="Média do recorte de comparação<br>%{x|%d/%m/%Y}<br>Rendimento acumulado: %{y:.1%}<extra></extra>",
            )
        )

    fig.add_hline(y=0, line_width=1.8, line_dash="solid", line_color=NEGATIVE)
    fig.update_layout(
        title=None,
        height=610,
        hovermode="x unified",
        margin=dict(l=84, r=52, t=96, b=146),
        legend=dict(
            title=None,
            orientation="h",
            x=0.0,
            y=-0.18,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12, color=INK),
            traceorder="normal",
        ),
    )
    fig.update_yaxes(title_text="Rendimento acumulado", tickformat=".0%", tickprefix=None, title_standoff=16)
    fig.update_xaxes(title_text="Período", title_standoff=14)
    return clean_axes(fig, y_money=False)


def make_drawdown_chart(
    ts_bairro: pd.DataFrame,
    ts_bench: pd.DataFrame,
    title: str,
    *,
    top_n: int,
    selected_bairros: list[str] | None = None,
) -> go.Figure:
    selected_bairros = [str(b) for b in (selected_bairros or [])]
    if selected_bairros:
        source = ts_bairro[ts_bairro["bairro"].isin(selected_bairros)].copy()
    else:
        top_bairros = (
            ts_bairro.groupby("bairro", observed=True)["total_transações"].sum().sort_values(ascending=False).head(top_n).index
        )
        source = ts_bairro[ts_bairro["bairro"].isin(top_bairros)].copy()

    indexed = index_series(source, ["bairro"]) if not source.empty else pd.DataFrame()
    dd_bairros = drawdown_series(indexed, "bairro") if not indexed.empty else pd.DataFrame()
    bench_indexed = index_series(ts_bench)
    dd_bench = drawdown_series(bench_indexed) if not bench_indexed.empty else pd.DataFrame()

    fig = go.Figure()
    order_bairros = ordered_series_last_value(indexed, "bairro", "indice_100") if not indexed.empty else []
    for bairro in order_bairros:
        part = dd_bairros[dd_bairros["bairro"].astype(str) == str(bairro)].sort_values("periodo")
        fig.add_trace(
            go.Scatter(
                x=part["periodo"],
                y=part["drawdown"],
                mode="lines+markers",
                name=str(bairro),
                line=dict(width=2.0),
                marker=dict(size=5),
                hovertemplate=f"{bairro}<br>%{{x|%d/%m/%Y}}<br>Drawdown: %{{y:.1%}}<extra></extra>",
            )
        )

    if not dd_bench.empty:
        fig.add_trace(
            go.Scatter(
                x=dd_bench["periodo"],
                y=dd_bench["drawdown"],
                mode="lines",
                name="Média do recorte de comparação",
                line=dict(width=3.4, color=BENCHMARK, dash="dash"),
                hovertemplate="Média do recorte de comparação<br>%{x|%d/%m/%Y}<br>Drawdown: %{y:.1%}<extra></extra>",
            )
        )

    fig.add_hline(y=0, line_width=1.8, line_dash="solid", line_color=NEGATIVE)
    fig.update_layout(
        title=None,
        height=540,
        hovermode="x unified",
        margin=dict(l=84, r=52, t=48, b=146),
        legend=dict(
            title=None,
            orientation="h",
            x=0.0,
            y=-0.18,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12, color=INK),
            traceorder="normal",
        ),
    )
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", title_standoff=16)
    fig.update_xaxes(title_text="Período", title_standoff=14)
    return clean_axes(fig, y_money=False)


def make_rolling_return_chart(
    rolling: pd.DataFrame,
    summary: pd.DataFrame,
    title: str,
    *,
    top_n: int,
    selected_bairros: list[str] | None = None,
) -> go.Figure:
    if rolling.empty:
        return go.Figure()

    prob_lookup = {
        row["serie"]: row["probabilidade_ganho"]
        for _, row in summary.iterrows()
        if pd.notna(row.get("probabilidade_ganho"))
    }

    selected_bairros = [str(b) for b in (selected_bairros or [])]
    if selected_bairros:
        selected_series = set(selected_bairros + [BENCHMARK_SERIES_NAME])
    else:
        valid_order = (
            summary[summary["serie"] != BENCHMARK_SERIES_NAME]
            .sort_values(["janelas_validas", "probabilidade_ganho"], ascending=[False, False])
            .head(top_n)["serie"]
            .astype(str)
            .tolist()
        )
        selected_series = set(valid_order + [BENCHMARK_SERIES_NAME])
    plot = rolling[rolling["serie"].isin(selected_series)].copy()

    fig = go.Figure()
    non_bench = plot[plot["serie"] != BENCHMARK_SERIES_NAME].copy()
    order_series = ordered_series_last_value(non_bench, "serie", "retorno_janela")
    if BENCHMARK_SERIES_NAME in plot["serie"].astype(str).unique().tolist():
        order_series = order_series + [BENCHMARK_SERIES_NAME]

    for serie in order_series:
        part = plot[plot["serie"].astype(str) == str(serie)].sort_values("periodo")
        prob = prob_lookup.get(str(serie), np.nan)
        prob_label = br_pct(prob, 0) if pd.notna(prob) else "—"
        is_benchmark = str(serie) == BENCHMARK_SERIES_NAME
        fig.add_trace(
            go.Scatter(
                x=part["periodo"],
                y=part["retorno_janela"],
                mode="lines+markers",
                name=str(serie),
                line=(
                    dict(width=3.4, dash="dash", color=BENCHMARK)
                    if is_benchmark
                    else dict(width=2.0, dash="solid")
                ),
                marker=(dict(size=7, color=BENCHMARK) if is_benchmark else dict(size=5)),
                customdata=np.array(
                    list(zip(part["inicio_janela"], part["fim_janela"], part["preco_inicio"], part["preco_fim"])),
                    dtype=object,
                ),
                hovertemplate=(
                    f"{serie}<br>"
                    "Janela: %{customdata[0]|%m/%Y} → %{customdata[1]|%m/%Y}<br>"
                    "Retorno real: %{y:.1%}<br>"
                    "Preço início: R$ %{customdata[2]:,.0f}/m²<br>"
                    "Preço fim: R$ %{customdata[3]:,.0f}/m²<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=0, line_width=1.2, line_dash="dot", line_color=GRID)
    fig.update_layout(
        title=None,
        height=660,
        hovermode="x unified",
        margin=dict(l=84, r=52, t=110, b=160),
        legend=dict(
            title=None,
            orientation="h",
            x=0.0,
            y=-0.18,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12, color=INK),
            traceorder="normal",
        ),
    )
    fig.update_yaxes(title_text="Retorno real da janela", tickformat=".0%", tickprefix=None, title_standoff=16)
    fig.update_xaxes(title_text="Mês final da janela", title_standoff=14)
    return clean_axes(fig, y_money=False)


def make_gain_quadrant_chart(
    rolling_summary: pd.DataFrame,
    window_months: int,
) -> go.Figure:
    required_cols = {
        "serie",
        "probabilidade_ganho",
        "retorno_mediano",
        "janelas_validas",
        "melhor_janela",
        "pior_janela",
    }
    if rolling_summary.empty or not required_cols.issubset(rolling_summary.columns):
        return go.Figure()

    plot = rolling_summary[list(required_cols)].copy()
    plot["serie"] = plot["serie"].astype(str)
    numeric_cols = [
        "probabilidade_ganho",
        "retorno_mediano",
        "janelas_validas",
        "melhor_janela",
        "pior_janela",
    ]
    for col in numeric_cols:
        plot[col] = pd.to_numeric(plot[col], errors="coerce")
    plot = plot.dropna(subset=["probabilidade_ganho", "retorno_mediano", "janelas_validas"])
    if plot.empty:
        return go.Figure()

    def zone_color(prob: float) -> str:
        if prob < 0.50:
            return NEGATIVE
        if prob < 0.65:
            return CAUTION
        if prob < 0.75:
            return ACCENT2
        return ACCENT

    bench_mask = plot["serie"].eq(BENCHMARK_SERIES_NAME)
    bairros = plot[~bench_mask].copy()
    benchmark = plot[bench_mask].copy()

    if not bairros.empty:
        min_windows = bairros["janelas_validas"].min()
        max_windows = bairros["janelas_validas"].max()
        bairros["marker_size"] = 8 + 20 * (bairros["janelas_validas"] - min_windows) / (max_windows - min_windows + 1)
        label_threshold = bairros["janelas_validas"].quantile(0.50)
        bairros["label_text"] = np.where(bairros["janelas_validas"] >= label_threshold, bairros["serie"], "")
        bairros["marker_color"] = bairros["probabilidade_ganho"].map(zone_color)
    else:
        bairros["marker_size"] = pd.Series(dtype=float)
        bairros["label_text"] = pd.Series(dtype=str)
        bairros["marker_color"] = pd.Series(dtype=str)

    fig = go.Figure()

    zone_specs = [
        (0.00, 0.50, "#1F2937", "Predomínio<br>abaixo da inflação", 0.25),
        (0.50, 0.65, "#172033", "Inconclusivo", 0.575),
        (0.65, 0.75, "#122B36", "Consistente", 0.70),
        (0.75, 1.00, "#102A43", "Forte", 0.875),
    ]
    for x0, x1, fillcolor, label, xpos in zone_specs:
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=fillcolor,
            opacity=0.28,
            layer="below",
            line_width=0,
        )
        fig.add_annotation(
            x=xpos,
            y=1.01,
            xref="x",
            yref="paper",
            text=label,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=11, color=MUTED),
            align="center",
        )

    for xref in [0.50, 0.65, 0.75]:
        fig.add_vline(x=xref, line_dash="dot", line_color=GRID, line_width=1)
    fig.add_hline(y=0.0, line_dash="solid", line_color=GRID, line_width=1)
    fig.add_hline(y=0.08, line_dash="dash", line_color=BENCHMARK, line_width=1.8)
    fig.add_annotation(
        x=1.0,
        y=0.08,
        xref="paper",
        yref="y",
        text="Mínimo estimado após<br>custos de transação (~8%)",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=10, color=BENCHMARK),
        align="right",
    )

    if not bairros.empty:
        labeled = bairros[bairros["label_text"].astype(str).ne("")].copy()
        unlabeled = bairros[bairros["label_text"].astype(str).eq("")].copy()

        if not unlabeled.empty:
            fig.add_trace(
                go.Scatter(
                    x=unlabeled["probabilidade_ganho"],
                    y=unlabeled["retorno_mediano"],
                    mode="markers",
                    text=unlabeled["serie"],
                    marker=dict(
                        size=unlabeled["marker_size"],
                        color=unlabeled["marker_color"],
                        opacity=0.88,
                        line=dict(width=1, color="#CBD5E1"),
                    ),
                    customdata=np.stack(
                        [
                            unlabeled["janelas_validas"].round(0).astype(int),
                            unlabeled["melhor_janela"],
                            unlabeled["pior_janela"],
                        ],
                        axis=-1,
                    ),
                    name="Bairros",
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Prob. de ganho: %{x:.1%}<br>"
                        "Retorno mediano: %{y:.1%}<br>"
                        "Janelas válidas: %{customdata[0]}<br>"
                        "Melhor janela: %{customdata[1]:.1%}<br>"
                        "Pior janela: %{customdata[2]:.1%}<extra></extra>"
                    ),
                )
            )

        if not labeled.empty:
            fig.add_trace(
                go.Scatter(
                    x=labeled["probabilidade_ganho"],
                    y=labeled["retorno_mediano"],
                    mode="markers+text",
                    text=labeled["serie"],
                    textposition="top center",
                    textfont=dict(size=10, color=INK),
                    marker=dict(
                        size=labeled["marker_size"],
                        color=labeled["marker_color"],
                        opacity=0.88,
                        line=dict(width=1, color="#CBD5E1"),
                    ),
                    customdata=np.stack(
                        [
                            labeled["janelas_validas"].round(0).astype(int),
                            labeled["melhor_janela"],
                            labeled["pior_janela"],
                        ],
                        axis=-1,
                    ),
                    name="Bairros com rótulo",
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Prob. de ganho: %{x:.1%}<br>"
                        "Retorno mediano: %{y:.1%}<br>"
                        "Janelas válidas: %{customdata[0]}<br>"
                        "Melhor janela: %{customdata[1]:.1%}<br>"
                        "Pior janela: %{customdata[2]:.1%}<extra></extra>"
                    ),
                )
            )

    if not benchmark.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark["probabilidade_ganho"],
                y=benchmark["retorno_mediano"],
                mode="markers+text",
                text=benchmark["serie"],
                textposition="top center",
                textfont=dict(size=10, color=INK),
                marker=dict(
                    size=18,
                    color=BENCHMARK,
                    symbol="diamond",
                    opacity=0.95,
                    line=dict(width=1.2, color="#CBD5E1"),
                ),
                customdata=np.stack(
                    [
                        benchmark["janelas_validas"].round(0).astype(int),
                        benchmark["melhor_janela"],
                        benchmark["pior_janela"],
                    ],
                    axis=-1,
                ),
                name=BENCHMARK_SERIES_NAME,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Prob. de ganho: %{x:.1%}<br>"
                    "Retorno mediano: %{y:.1%}<br>"
                    "Janelas válidas: %{customdata[0]}<br>"
                    "Melhor janela: %{customdata[1]:.1%}<br>"
                    "Pior janela: %{customdata[2]:.1%}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=None,
        height=610,
        hovermode="closest",
        showlegend=False,
        margin=dict(l=78, r=54, t=64, b=82),
    )
    fig.update_xaxes(
        title_text="Proporção de períodos com ganho real",
        tickformat=".0%",
        range=[0, 1],
        dtick=0.1,
    )
    fig.update_yaxes(
        title_text="Retorno mediano real no período",
        tickformat=".0%",
        zeroline=False,
    )
    return clean_axes(fig, y_money=False)


def render_rolling_probability_cards(
    rolling_summary: pd.DataFrame,
    bairros_selecionados: list[str],
) -> None:
    """Mostra P(ganho) dos bairros selecionados como números simples."""
    bairros = [str(b) for b in bairros_selecionados if str(b) != BENCHMARK_SERIES_NAME][:3]
    if not bairros:
        return

    lookup = {}
    if not rolling_summary.empty and {"serie", "probabilidade_ganho"}.issubset(rolling_summary.columns):
        lookup = {
            str(row["serie"]): row["probabilidade_ganho"]
            for _, row in rolling_summary.iterrows()
            if pd.notna(row.get("probabilidade_ganho"))
        }

    cols = st.columns(3)
    for idx, col in enumerate(cols):
        if idx >= len(bairros):
            col.empty()
            continue
        bairro = bairros[idx]
        prob = lookup.get(bairro, np.nan)
        col.metric(bairro, br_pct(prob, 1) if pd.notna(prob) else "—")


def make_small_multiples(ts_bairro: pd.DataFrame, title: str, top_n: int, common_y: bool = True) -> go.Figure:
    top_bairros = (
        ts_bairro.groupby("bairro", observed=True)["total_transações"].sum().sort_values(ascending=False).head(top_n).index
    )
    plot = ts_bairro[ts_bairro["bairro"].isin(top_bairros)].copy()
    if plot.empty:
        return go.Figure()

    fig = px.line(
        plot,
        x="periodo",
        y="preco_m2_corrigido",
        facet_col="bairro",
        facet_col_wrap=3,
        markers=True,
        title=None,
        labels={"periodo": "", "preco_m2_corrigido": "R$/m²"},
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("bairro=", ""), font_size=12, font_color=INK))
    fig.update_traces(line_width=2.0, marker_size=4.5, hovertemplate="%{x|%d/%m/%Y}<br>R$ %{y:,.0f}/m²<extra></extra>")
    if common_y:
        fig.update_yaxes(matches="y")
    else:
        fig.update_yaxes(matches=None)
    fig.update_layout(
        height=max(560, 285 * int(np.ceil(len(top_bairros) / 3))),
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=72, r=42, t=92, b=72),
    )
    fig.update_yaxes(title_standoff=12)
    return clean_axes(fig)



def make_rank_dotplot(rank: pd.DataFrame, title: str, top_n: int = 25, show_bairro: bool = True) -> go.Figure:
    plot = rank.head(top_n).sort_values("preco_m2_corrigido", ascending=True).copy()
    if plot.empty:
        return go.Figure()

    logradouro = plot["logradouro"].astype(str).str.replace(r"^\s*R\$\s*", "", regex=True).str.strip()
    if show_bairro and "bairro" in plot.columns:
        plot["label"] = logradouro + " — " + plot["bairro"].astype(str)
    else:
        plot["label"] = logradouro

    fig = go.Figure(
        go.Scatter(
            x=plot["preco_m2_corrigido"],
            y=plot["label"],
            mode="markers",
            marker=dict(
                size=np.clip(np.sqrt(plot["total_transações"]) * 2.4, 6, 24),
                color=plot["preco_m2_corrigido"],
                colorscale=HOT_COOL_SCALE,
                reversescale=False,
                showscale=True,
                colorbar=dict(
                    title=dict(text="R$/m²", side="top", font=dict(color=INK, size=12)),
                    tickprefix="R$ ",
                    tickformat=",.0f",
                    tickfont=dict(color=INK, size=11),
                    thickness=14,
                    len=0.72,
                    outlinewidth=0,
                    bgcolor="rgba(0,0,0,0)",
                ),
                line=dict(width=0.6, color="#CBD5E1"),
            ),
            customdata=np.stack([plot["total_transações"], plot["mediana_preco_m2"]], axis=-1),
            hovertemplate="%{y}<br>Preço/m²: R$ %{x:,.0f}<br>Mediana: R$ %{customdata[1]:,.0f}<br>Transações: %{customdata[0]:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(title=None, height=max(500, 26 * len(plot) + 150), showlegend=False, margin=dict(l=92, r=92, t=46, b=66))
    fig.update_xaxes(title_text="R$/m² corrigido", tickprefix="R$ ", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title_text="", showgrid=True, gridcolor="rgba(148,163,184,0.18)", tickson="boundaries")
    fig = clean_axes(fig, y_money=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    return fig


def make_heatmap(heat: pd.DataFrame, title: str) -> go.Figure:
    if heat.empty:
        return go.Figure()

    period_col = "periodo_heatmap" if "periodo_heatmap" in heat.columns else "ano"
    x_label = "Período" if period_col == "periodo_heatmap" else "Ano"
    piv = heat.pivot(index="bairro", columns=period_col, values="preco_m2_corrigido")
    fig = px.imshow(
        piv,
        aspect="auto",
        color_continuous_scale=HOT_COOL_SCALE,
        labels=dict(x=x_label, y="Bairro", color="R$/m²"),
        title=None,
    )
    fig.update_layout(
        height=max(460, 28 * len(piv.index) + 170),
        margin=dict(l=96, r=160, t=64, b=72),
        coloraxis_colorbar=dict(
            title="R$/m²",
            tickprefix="R$ ",
            tickformat=",.0f",
            thickness=16,
            len=0.70,
            x=1.04,
            y=0.5,
            xanchor="left",
            outlinewidth=0,
        ),
    )
    fig.update_traces(hovertemplate="Bairro: %{y}<br>Período: %{x}<br>R$ %{z:,.0f}/m²<extra></extra>")
    return clean_axes(fig, y_money=False)


def make_area_chart(
    df: pd.DataFrame,
    min_trans: int,
    *,
    top_n: int = 8,
    zone_df: pd.DataFrame | None = None,
    zone_name: str | None = None,
) -> go.Figure:
    """
    Barras agrupadas por bairro em cada faixa de área. As linhas ficam
    reservadas apenas para médias: média do filtro e, opcionalmente, média
    de uma grande zona escolhida pelo usuário.
    """
    order = {label: i for i, label in enumerate(AREA_LABELS)}
    fig = go.Figure()

    bairro_rank = (
        df.groupby("bairro", observed=True)[TX_COL]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .astype(str)
        .tolist()
        if not df.empty else []
    )

    area_bairros = aggregate_metrics(df[df["bairro"].isin(bairro_rank)], ["bairro", "faixa_area"])
    area_bairros = area_bairros[area_bairros["total_transações"] >= min_trans].copy()
    area_bairros["ordem"] = area_bairros["faixa_area"].map(order)
    area_bairros = area_bairros.sort_values(["bairro", "ordem"])

    for bairro, part in area_bairros.groupby("bairro", observed=True):
        fig.add_trace(
            go.Bar(
                x=part["faixa_area"],
                y=part["preco_m2_corrigido"],
                name=str(bairro),
                opacity=0.62,
                customdata=part["total_transações"],
                hovertemplate=f"{bairro}<br>%{{x}}<br>R$ %{{y:,.0f}}/m²<br>Transações: %{{customdata:,.0f}}<extra></extra>",
            )
        )

    # Média do filtro do usuário: linha de referência principal.
    area_media = aggregate_metrics(df, ["faixa_area"])
    area_media = area_media[area_media["total_transações"] >= min_trans].copy()
    area_media["ordem"] = area_media["faixa_area"].map(order)
    area_media = area_media.sort_values("ordem")
    if not area_media.empty:
        fig.add_trace(
            go.Scatter(
                x=area_media["faixa_area"],
                y=area_media["preco_m2_corrigido"],
                mode="lines+markers",
                name="Média do filtro",
                line=dict(color=BENCHMARK, width=3.6, dash="dash"),
                marker=dict(size=8, color=BENCHMARK),
                customdata=area_media["total_transações"],
                hovertemplate="Média do filtro<br>%{x}<br>R$ %{y:,.0f}/m²<br>Transações: %{customdata:,.0f}<extra></extra>",
            )
        )

    # Média de uma grande zona escolhida pelo usuário: linha comparativa.
    if zone_df is not None and zone_name and not zone_df.empty:
        area_zone = aggregate_metrics(zone_df, ["faixa_area"])
        area_zone = area_zone[area_zone["total_transações"] >= min_trans].copy()
        area_zone["ordem"] = area_zone["faixa_area"].map(order)
        area_zone = area_zone.sort_values("ordem")
        if not area_zone.empty:
            fig.add_trace(
                go.Scatter(
                    x=area_zone["faixa_area"],
                    y=area_zone["preco_m2_corrigido"],
                    mode="lines+markers",
                    name=f"Média — {zone_name}",
                    line=dict(color="#7C3AED", width=3.2, dash="dot"),
                    marker=dict(size=8, color="#7C3AED"),
                    customdata=area_zone["total_transações"],
                    hovertemplate=f"Média — {zone_name}<br>%{{x}}<br>R$ %{{y:,.0f}}/m²<br>Transações: %{{customdata:,.0f}}<extra></extra>",
                )
            )

    fig.update_layout(
        title=None,
        height=660,
        hovermode="x unified",
        barmode="group",
        bargap=0.18,
        bargroupgap=0.08,
        margin=dict(l=84, r=300, t=64, b=104),
        legend=dict(
            title="Bairros e médias",
            orientation="v",
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12, color=INK),
        ),
    )
    fig.update_xaxes(title_text="Faixa de área", categoryorder="array", categoryarray=AREA_LABELS, title_standoff=14)
    fig.update_yaxes(title_text="R$/m² corrigido", title_standoff=16)
    return clean_axes(fig)

def make_map(mapa: pd.DataFrame, color_by: str) -> go.Figure:
    plot = mapa.copy()
    if plot.empty:
        return go.Figure()

    marker_size = 11
    marker_opacity = 0.76

    if color_by == "Volume":
        color_col = "total_transações"
        color_title = "Transações"
        # Mais volume = tom mais escuro/saturado; menos volume = tom mais claro.
        color_scale = [[0.0, "#FDE68A"], [0.50, "#F59E0B"], [1.0, "#7C2D12"]]
        tickprefix = ""
    else:
        color_col = "preco_m2_corrigido"
        color_title = "R$/m²"
        # Mais caro = quente; mais barato = frio.
        color_scale = HOT_COOL_SCALE
        tickprefix = "R$ "

    low = plot[color_col].quantile(0.05)
    high = plot[color_col].quantile(0.95)
    if pd.isna(low) or pd.isna(high) or low == high:
        low = plot[color_col].min()
        high = plot[color_col].max()
    if pd.isna(low) or pd.isna(high) or low == high:
        low, high = 0, 1

    lat_min, lat_max = plot["latitude"].min(), plot["latitude"].max()
    lon_min, lon_max = plot["longitude"].min(), plot["longitude"].max()
    center_lat = float((lat_min + lat_max) / 2)
    center_lon = float((lon_min + lon_max) / 2)
    lat_span = max(float(lat_max - lat_min), 0.08)
    lon_span = max(float(lon_max - lon_min), 0.08)
    zoom = 10.6
    if max(lat_span, lon_span) > 0.45:
        zoom = 9.7
    elif max(lat_span, lon_span) > 0.25:
        zoom = 10.2

    customdata = np.stack([
        plot["bairro"].astype(str),
        plot["preco_m2_corrigido"].astype(float),
        plot["mediana_preco_m2"].astype(float),
        plot["total_transações"].astype(float),
    ], axis=-1)

    fig = go.Figure()
    fig.add_trace(
        go.Scattermapbox(
            lat=plot["latitude"],
            lon=plot["longitude"],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=plot[color_col],
                colorscale=color_scale,
                cmin=low,
                cmax=high,
                opacity=marker_opacity,
                sizemode="diameter",
                colorbar=dict(
                    title=dict(text=color_title, side="top", font=dict(color="#E5E7EB", size=12)),
                    tickprefix=tickprefix,
                    tickformat=",.0f",
                    tickfont=dict(color="#E5E7EB", size=11),
                    thickness=18,
                    len=0.82,
                    x=1.01,
                    y=0.5,
                    xanchor="left",
                    xpad=8,
                    outlinewidth=0,
                    bgcolor=PAPER,
                ),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Bairro: %{customdata[0]}<br>"
                "Preço/m²: R$ %{customdata[1]:,.0f}<br>"
                "Mediana: R$ %{customdata[2]:,.0f}<br>"
                "Transações: %{customdata[3]:,.0f}<extra></extra>"
            ),
            text=plot["logradouro"].astype(str),
        )
    )

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            zoom=zoom,
            center=dict(lat=center_lat, lon=center_lon),
            domain=dict(x=[0.0, 0.90], y=[0.0, 1.0]),
        ),
        height=660,
        margin=dict(l=8, r=128, t=8, b=8),
        template=PLOT_TEMPLATE,
        paper_bgcolor=PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(color=INK, family="Inter, DM Sans, Arial, sans-serif"),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="#334155", font=dict(color="#F8FAFC", family="Inter", size=12)),
        autosize=True,
        uirevision="mapa-itbi",
    )
    return fig

# -----------------------------------------------------------------------------
# Filtros e UI
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Filters:
    start: pd.Timestamp
    end: pd.Timestamp
    bairros: list[str]
    usos: list[str]
    tipologias: list[str]
    logradouros: list[str]
    faixas_area: list[str]
    low_q: float
    high_q: float
    remove_outliers: bool
    exclude_partial_transfers: bool


def apply_filters(
    df: pd.DataFrame,
    filters: Filters,
    *,
    use_bairro: bool = True,
    use_logradouro: bool = True,
    use_outlier: bool = True,
) -> tuple[pd.DataFrame, float, float]:
    out = df.copy()
    out = out[(out[DATE_COL] >= filters.start) & (out[DATE_COL] <= filters.end)]

    if use_bairro and filters.bairros:
        out = out[out["bairro"].isin(filters.bairros)]
    if filters.usos:
        out = out[out["uso"].isin(filters.usos)]
    if filters.tipologias:
        out = out[out["principais_tipologias"].isin(filters.tipologias)]
    if use_logradouro and filters.logradouros:
        out = out[out["logradouro"].isin(filters.logradouros)]
    if filters.faixas_area:
        out = out[out["faixa_area"].isin(filters.faixas_area)]

    out = out[(out["area_total"] > 0) & (out["valor_total_corrigido"] > 0)].copy()

    if filters.exclude_partial_transfers and "média_percentual_transferido" in out.columns:
        ratio = normalize_transfer_ratio(out["média_percentual_transferido"])
        out = out[ratio.isna() | (ratio >= 0.999)].copy()

    lo = hi = np.nan
    if use_outlier and filters.remove_outliers:
        out, lo, hi = apply_outlier_filter(out, PRICE_COL, filters.low_q, filters.high_q)
    return out, lo, hi


def render_metric_note(row: pd.Series) -> None:
    st.caption(
        "Métrica central: soma do valor corrigido ÷ soma da área estimada. "
        "A média simples de preço/m² aparece apenas como diagnóstico porque pode distorcer bairros com volumes diferentes."
    )



def render_kpis(
    filtered: pd.DataFrame,
    ts_region: pd.DataFrame,
    min_trans: int,
    *,
    rj_general_median_m2: float | None = None,
) -> None:
    metrics = aggregate_metrics(filtered)
    row = metrics.iloc[0]
    variation = calc_variation(ts_region, min_trans)
    trend_6m = calc_recent_trend_6m(filtered)
    spread = calc_spread_mediana_media(row)
    imovel_tipico = row["preco_m2_corrigido"] * 65 if pd.notna(row["preco_m2_corrigido"]) else np.nan

    def pill_class(value: float | int | None, *, negative_is_bad: bool = True) -> str:
        if value is None or pd.isna(value) or not np.isfinite(value):
            return "kpi-neutral"
        if value < 0:
            return "kpi-negative" if negative_is_bad else "kpi-positive"
        if value > 0:
            return "kpi-positive" if negative_is_bad else "kpi-negative"
        return "kpi-neutral"

    if pd.isna(variation):
        gain_class = "kpi-neutral"
        gain_value = "—"
    elif variation < 0:
        gain_class = "kpi-negative"
        gain_value = br_pct(variation)
    else:
        gain_class = "kpi-positive"
        gain_value = br_pct(variation)

    if pd.isna(spread):
        spread_value = "—"
        spread_class = "kpi-neutral"
    else:
        spread_value = br_pct_signed(spread)
        spread_class = pill_class(spread)

    if pd.isna(trend_6m):
        trend_value = "—"
        trend_class = "kpi-neutral"
    else:
        trend_value = br_pct_signed(trend_6m)
        trend_class = pill_class(trend_6m)

    rj_sublabel = ""
    if rj_general_median_m2 is not None and pd.notna(rj_general_median_m2) and np.isfinite(rj_general_median_m2):
        rj_sublabel = f"vs {br_money(rj_general_median_m2)}/m² (RJ geral)"

    cards = [
        {
            "label": "Preço/m² (médio pond.)",
            "value": br_money(row["preco_m2_corrigido"]),
            "kind": "value",
            "sublabel": rj_sublabel,
        },
        {"label": "Mediana ponderada", "value": br_money(row["mediana_preco_m2"]), "kind": "value"},
        {"label": "Transações", "value": br_number(row["total_transações"]), "kind": "value"},
        {"label": "Imóvel típico (65 m²)", "value": br_money(imovel_tipico), "kind": "value"},
        {"label": "Ganho real s/ IPCA", "value": gain_value, "kind": gain_class},
        {"label": "Spread mediana/média", "value": spread_value, "kind": spread_class},
        {"label": "Tendência recente (6 meses)", "value": trend_value, "kind": trend_class},
    ]

    html_cards = []
    for card in cards:
        label = card["label"]
        value = card["value"]
        kind = card["kind"]
        sublabel = card.get("sublabel", "")
        sublabel_html = f'<div class="kpi-sublabel">{sublabel}</div>' if sublabel else ""

        if kind == "value":
            value_class = "kpi-value long" if len(str(value)) >= 12 else "kpi-value"
            html_cards.append(
                f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="{value_class}">{value}</div>{sublabel_html}</div>'
            )
        else:
            html_cards.append(
                f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-pill {kind}">{value}</div>{sublabel_html}</div>'
            )

    st.markdown('<div class="kpi-grid">' + ''.join(html_cards) + '</div>', unsafe_allow_html=True)
    render_metric_note(row)


def dataframe_config() -> dict:
    return {
        "preco_m2_corrigido": st.column_config.NumberColumn("Preço/m² ponderado", format="R$ %.0f"),
        "mediana_preco_m2": st.column_config.NumberColumn("Mediana ponderada", format="R$ %.0f"),
        "preco_m2_media_simples": st.column_config.NumberColumn("Média simples", format="R$ %.0f"),
        "valor_total_corrigido": st.column_config.NumberColumn("Valor total corrigido", format="R$ %.0f"),
        "area_total": st.column_config.NumberColumn("Área agregada", format="%.0f m²"),
        "total_transações": st.column_config.NumberColumn("Transações", format="%.0f"),
        "participação_volume": st.column_config.NumberColumn("Part. volume", format="%.1%%"),
    }




def rolling_dataframe_config() -> dict:
    return {
        "janela_meses": st.column_config.NumberColumn("Janela", format="%d meses"),
        "janelas_validas": st.column_config.NumberColumn("Janelas válidas", format="%d"),
        "janelas_com_ganho": st.column_config.NumberColumn("Janelas com ganho", format="%d"),
        "probabilidade_ganho": st.column_config.NumberColumn("Prob. de ganho", format="%.1%%"),
        "retorno_medio": st.column_config.NumberColumn("Retorno médio", format="%.1%%"),
        "retorno_mediano": st.column_config.NumberColumn("Retorno mediano", format="%.1%%"),
        "melhor_janela": st.column_config.NumberColumn("Melhor janela", format="%.1%%"),
        "pior_janela": st.column_config.NumberColumn("Pior janela", format="%.1%%"),
        "retorno_mais_recente": st.column_config.NumberColumn("Retorno mais recente", format="%.1%%"),
        "inicio_mais_recente": st.column_config.DateColumn("Início mais recente", format="MM/YYYY"),
        "fim_mais_recente": st.column_config.DateColumn("Fim mais recente", format="MM/YYYY"),
    }

def render_download_table(df: pd.DataFrame, file_name: str, label: str) -> None:
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label, data=csv, file_name=file_name, mime="text/csv")



# Colunas exibidas em linguagem de público leigo.
# A base e os downloads continuam numéricos; só a exibição no dashboard é formatada.
FRIENDLY_COLUMN_NAMES = {
    "bairro": "Bairro",
    "bairro / referência": "Bairro / referência",
    "logradouro": "Logradouro",
    "uso": "Uso",
    "principais_tipologias": "Tipologia principal",
    "faixa_area": "Faixa de área",
    "data_ref": "Data",
    "total_transações": "Transações",
    "participação_volume": "Participação no volume",
    "preco_m2_corrigido": "Preço médio/m²",
    "mediana_preco_m2": "Preço mediano/m²",
    "preco_m2_media_simples": "Média simples/m²",
    "valor_total_corrigido": "Valor total corrigido",
    "area_total": "Área total",
    "linhas": "Registros",
    "janela_meses": "Janela móvel",
    "janelas_validas": "Janelas válidas",
    "janelas_com_ganho": "Janelas com ganho",
    "probabilidade_ganho": "Probabilidade de ganho",
    "retorno_medio": "Retorno médio",
    "retorno_mediano": "Retorno mediano",
    "melhor_janela": "Melhor janela",
    "pior_janela": "Pior janela",
    "retorno_mais_recente": "Retorno mais recente",
    "inicio_mais_recente": "Início da janela recente",
    "fim_mais_recente": "Fim da janela recente",
    "obs_qualidade": "Observação de qualidade",
    "média_área_construída": "Área média",
    "média_valor_imóvel_corrigido_mar_2026": "Valor médio corrigido",
    "preco_m2_imovel_corrigido_mar_2026": "Preço/m² corrigido",
    "média_valor_transação_corrigido_mar_2026": "Valor de transação corrigido",
    "preco_m2_transacao_corrigido_mar_2026": "Preço/m² transação",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "rank_volume": "Rank volume",
    "variacao_real_acumulada": "Variação real acumulada",
    "spread_mediana_media": "Spread mediana/média",
    "tendencia_recente_6m": "Tendência recente 6m",
    "share_transacoes": "Participação das transações",
    "aceleracao_recente": "Aceleração recente",
    "orcamento_maximo": "Orçamento máximo",
    "metragem_desejada": "Metragem desejada",
    "orcamento_por_m2": "Orçamento por m²",
    "custo_estimado": "Custo estimado",
    "bairro_alcancavel": "Bairro alcançável",
}

MONEY_COLUMNS = {
    "preco_m2_corrigido",
    "mediana_preco_m2",
    "preco_m2_media_simples",
    "valor_total_corrigido",
    "média_valor_imóvel_corrigido_mar_2026",
    "preco_m2_imovel_corrigido_mar_2026",
    "média_valor_transação_corrigido_mar_2026",
    "preco_m2_transacao_corrigido_mar_2026",
    "média_valor_transação",
    "média_valor_imóvel",
    "preco_m2_transacao",
    "preco_m2_imovel",
    "orcamento_maximo",
    "orcamento_por_m2",
    "custo_estimado",
}

PERCENT_COLUMNS = {
    "participação_volume",
    "probabilidade_ganho",
    "retorno_medio",
    "retorno_mediano",
    "melhor_janela",
    "pior_janela",
    "retorno_mais_recente",
    "média_percentual_transferido",
    "variacao_real_acumulada",
    "spread_mediana_media",
    "tendencia_recente_6m",
    "aceleracao_recente",
    "share_transacoes",
}

COUNT_COLUMNS = {"total_transações", "linhas", "janelas_validas", "janelas_com_ganho", "rank_volume"}
AREA_COLUMNS = {"area_total", "média_área_construída"}
DATE_COLUMNS = {DATE_COL, "inicio_mais_recente", "fim_mais_recente", "periodo", "periodo_mes"}


def br_date(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return pd.Timestamp(value).strftime("%d-%m-%Y")


def br_area(value: float | int | None, decimals: int = 0) -> str:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return "—"
    return f"{br_number(value, decimals)} m²"


def friendly_dataframe(df: pd.DataFrame, *, max_text: int = 70) -> pd.DataFrame:
    """Formata tabelas para leitura leiga sem destruir os dados originais."""
    if df.empty:
        return df.rename(columns=FRIENDLY_COLUMN_NAMES)

    out = df.copy()
    for col in out.columns:
        if col in MONEY_COLUMNS:
            out[col] = out[col].map(lambda v: br_money(v, decimals=0))
        elif col == "média_percentual_transferido":
            out[col] = out[col].map(lambda v: br_transfer_pct(v, decimals=1))
        elif col in PERCENT_COLUMNS:
            out[col] = out[col].map(lambda v: br_pct(v, decimals=1))
        elif col in AREA_COLUMNS:
            out[col] = out[col].map(lambda v: br_area(v, decimals=0))
        elif col in COUNT_COLUMNS:
            out[col] = out[col].map(lambda v: br_number(v, decimals=0))
        elif col == "janela_meses":
            out[col] = out[col].map(lambda v: "—" if pd.isna(v) else f"{int(v)} meses")
        elif col in DATE_COLUMNS or pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].map(br_date)
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: br_number(v, decimals=2))
        elif pd.api.types.is_integer_dtype(out[col]):
            out[col] = out[col].map(lambda v: br_number(v, decimals=0))
        elif out[col].dtype == "object" or str(out[col].dtype).startswith("string"):
            out[col] = out[col].astype("string").str.slice(0, max_text)

    return out.rename(columns=FRIENDLY_COLUMN_NAMES)


def friendly_column_config(display_df: pd.DataFrame) -> dict:
    config = {}
    large_cols = {
        "Bairro",
        "Bairro / referência",
        "Logradouro",
        "Tipologia principal",
        "Observação de qualidade",
        "Bairro alcançável",
    }
    medium_cols = {
        "Faixa de área",
        "Janela móvel",
        "Início da janela recente",
        "Fim da janela recente",
        "Preço médio/m²",
        "Preço mediano/m²",
        "Custo estimado",
        "Transações",
    }
    for col in display_df.columns:
        if col in large_cols:
            config[col] = st.column_config.TextColumn(col, width="large")
        elif col in medium_cols:
            config[col] = st.column_config.TextColumn(col, width="medium")
    return config


def show_friendly_dataframe(df: pd.DataFrame, *, height: int | None = None) -> None:
    display_df = friendly_dataframe(df)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=height,
        column_config=friendly_column_config(display_df),
    )





def make_affordability_chart(rank_bairros: pd.DataFrame, areas: list[float] = [45, 65, 90, 150, 250]) -> go.Figure:
    """
    Barras agrupadas: para cada bairro, mostra o custo estimado de um imóvel
    de diferentes metragens com base no preço/m² ponderado.
    Ideal para público leigo entender o patamar de entrada e de imóveis maiores.
    """
    plot = rank_bairros.head(12).copy()
    fig = go.Figure()
    colors = [ACCENT, ACCENT2, "#7C3AED", "#B45309", "#9D174D"]
    for cor, area in zip(colors, areas):
        fig.add_trace(go.Bar(
            name=f"{int(area)} m²",
            x=plot["bairro"],
            y=plot["preco_m2_corrigido"] * area,
            marker_color=cor,
            hovertemplate=f"<b>%{{x}}</b><br>{int(area)} m²: R$ %{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        title=None,
        barmode="group",
        height=560,
        hovermode="x unified",
        legend=dict(title="Metragem do imóvel", orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom"),
        margin=dict(l=84, r=42, t=92, b=124),
    )
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(title_text="Valor estimado (R$)", tickprefix="R$ ")
    return clean_axes(fig)

def date_input_compat(label: str, value):
    try:
        return st.date_input(label, value=value, format="DD-MM-YYYY")
    except TypeError:
        return st.date_input(label, value=value)


def slug_key(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")[:80]


def select_bairros_checkbox_dropdown(
    label: str,
    options: list[str],
    default: list[str],
    key: str,
    max_items: int = 3,
) -> list[str]:
    """Popover/expander com checkboxes e botão Aplicar.

    Os checkboxes ficam dentro de um st.form. Assim, marcar/desmarcar opções
    não recalcula os gráficos imediatamente; a seleção só altera o estado
    aplicado quando o usuário clica em Aplicar.
    """
    options = [str(x) for x in options]
    default = [str(x) for x in default if str(x) in options][:max_items]
    state_key = f"{key}_selected"
    sync_key = f"{key}_sync_signature"
    checkbox_prefix = f"{key}_chk_"

    if state_key not in st.session_state:
        st.session_state[state_key] = default

    applied = [str(x) for x in st.session_state.get(state_key, []) if str(x) in options][:max_items]
    if not applied:
        applied = default
        st.session_state[state_key] = applied

    # Sincroniza os checkboxes com a seleção aplicada antes da criação dos widgets.
    # Isso evita que um rascunho não aplicado continue aparecendo como se estivesse ativo.
    applied_signature = tuple(applied)
    if st.session_state.get(sync_key) != applied_signature:
        for option in options:
            st.session_state[f"{checkbox_prefix}{slug_key(option)}"] = option in applied
        st.session_state[sync_key] = applied_signature

    contexto = st.popover(label) if hasattr(st, "popover") else st.expander(label, expanded=False)
    with contexto:
        st.caption(f"Marque até {max_items} bairros. O gráfico só será atualizado após clicar em Aplicar.")
        with st.form(key=f"{key}_form", clear_on_submit=False):
            staged_flags: dict[str, bool] = {}
            for option in options:
                cb_key = f"{checkbox_prefix}{slug_key(option)}"
                staged_flags[option] = st.checkbox(option, key=cb_key)

            submitted = st.form_submit_button("Aplicar", type="primary")

        if submitted:
            staged = [opt for opt in options if staged_flags.get(opt, False)]
            if len(staged) > max_items:
                st.warning(f"Máximo de {max_items} bairros. Apliquei os {max_items} primeiros marcados.")
                staged = staged[:max_items]
            st.session_state[state_key] = staged
            # força a sincronização visual dos checkboxes no próximo render
            st.session_state[sync_key] = None
            applied = staged

    applied = [str(x) for x in st.session_state.get(state_key, []) if str(x) in options][:max_items]
    st.caption("Bairros aplicados: " + (", ".join(applied) if applied else "nenhum bairro"))
    return applied


def main() -> None:
    data_path = find_data_file()
    if not data_path:
        st.error(
            "Arquivo de dados não encontrado. Coloque o CSV ou Excel em `data/` com nome começando por "
            f"`{DATA_FILE_STEM}`."
        )
        st.stop()

    try:
        df = load_data(str(data_path))
    except Exception as exc:
        st.error(f"Falha ao carregar a base: {exc}")
        st.stop()

    geo_path = find_geo_file()
    geo = load_geo(str(geo_path) if geo_path else None)

    rj_general_median_m2 = weighted_median(df[PRICE_COL], df["area_total"]) if not df.empty else np.nan

    st.markdown("## Mercado imobiliário — Rio de Janeiro")
    st.markdown(
        "<span style='color:#6B6B6B;font-size:0.93rem'>"
        "Preços corrigidos pelo IPCA até março de 2026 · Fonte: ITBI/SMF-Rio"
        "</span>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Filtros")
        periodo_min_ts = df[DATE_COL].min().to_period("M").to_timestamp()
        periodo_max_ts = df[DATE_COL].max().to_period("M").to_timestamp()
        month_options = list(pd.date_range(periodo_min_ts, periodo_max_ts, freq="MS").date)

        st.markdown("**Período**")
        start_date_raw, end_date_raw = st.select_slider(
            "Arraste para selecionar início e fim",
            options=month_options,
            value=(month_options[0], month_options[-1]),
            format_func=lambda d: pd.Timestamp(d).strftime("%d-%m-%Y"),
            help="A base é mensal. Use também os campos abaixo se preferir inserir ou selecionar datas específicas.",
            label_visibility="collapsed",
        )

        c_ini, c_fim = st.columns(2)
        with c_ini:
            start_box = date_input_compat("Data de início", start_date_raw)
        with c_fim:
            end_box = date_input_compat("Data de término", end_date_raw)

        start = pd.Timestamp(start_box).to_period("M").to_timestamp()
        end = pd.Timestamp(end_box).to_period("M").to_timestamp() + pd.offsets.MonthEnd(0)
        if start > end:
            st.warning("A data de início estava maior que a data final. O painel ajustou a ordem automaticamente.")
            start, end = pd.Timestamp(end_box).to_period("M").to_timestamp(), pd.Timestamp(start_box).to_period("M").to_timestamp() + pd.offsets.MonthEnd(0)

        cesta = st.selectbox("Cesta de bairros", list(CESTAS_BAIRROS.keys()), index=0)
        bairros_all = sorted(df["bairro"].dropna().unique().tolist())

        if CESTAS_BAIRROS[cesta]:
            default_bairros = [b for b in CESTAS_BAIRROS[cesta] if b in bairros_all]
        else:
            default_bairros = rank_by_volume(df, "bairro", top_n=6)

        bairros_sel = st.multiselect("Bairros", bairros_all, default=default_bairros)

        uso_all = sorted(df["uso"].dropna().unique().tolist())
        uso_default = [u for u in uso_all if str(u).strip().upper() == "RESIDENCIAL"] or uso_all
        uso_sel = st.multiselect("Uso", uso_all, default=uso_default)

        tip_all = sorted(df["principais_tipologias"].dropna().unique().tolist())
        tip_default = [t for t in tip_all if "APARTAMENTO" in str(t).strip().upper()]
        tip_sel = st.multiselect("Tipologia", tip_all, default=tip_default[:1])

        area_all = [label for label in AREA_LABELS if label in set(df["faixa_area"].dropna())]
        area_sel = st.multiselect("Faixa de área construída", area_all, default=[])

        if bairros_sel:
            log_candidates = sorted(df[df["bairro"].isin(bairros_sel)]["logradouro"].dropna().unique().tolist())
        else:
            log_candidates = sorted(df["logradouro"].dropna().unique().tolist())
        log_sel = st.multiselect("Logradouro", log_candidates, default=[])

        st.divider()
        st.header("Leitura")
        freq = st.radio("Agregação temporal", ["Mensal", "Trimestral", "Anual"], index=1, horizontal=True)
        benchmark_kind = st.radio(
            "Referência de comparação",
            ["Recorte selecionado", "Geral sem filtro de bairro"],
            index=1,
            help="A referência geral mantém período, uso, tipologia e faixa de área, mas remove o filtro de bairro/logradouro.",
        )
        min_trans = st.number_input("Mínimo de transações para rankings/sinais", min_value=1, value=10, step=1)
        top_n = st.slider("Máximo de séries simultâneas", min_value=3, max_value=18, value=9, step=1)

        common_y = st.checkbox("Pequenos múltiplos com mesma escala vertical", value=True)

        st.divider()
        st.header("Tratamento")
        remove_outliers = st.checkbox("Remover preços extremos", value=True)
        outlier_options = {
            "Conservador (remove 1% extremos)": (0.01, 0.99),
            "Moderado (remove 2% extremos)": (0.02, 0.98),
            "Amplo (sem remoção)": (0.00, 1.00),
        }
        if remove_outliers:
            outlier_profile = st.select_slider(
                "Tratamento de preços extremos",
                options=list(outlier_options.keys()),
                value="Conservador (remove 1% extremos)",
                help="Remove os preços/m² muito fora do padrão antes dos cálculos. A opção conservadora costuma preservar mais dados.",
            )
            q_low, q_high = outlier_options[outlier_profile]
        else:
            q_low, q_high = 0.0, 1.0

        exclude_partial_transfers = st.toggle(
            "Excluir transferências parciais (<100%)",
            value=False,
            help="Transferências parciais podem refletir herança, doação ou separação e distorcer o preço/m² médio de mercado.",
        )

    filters = Filters(
        start=start,
        end=end,
        bairros=bairros_sel,
        usos=uso_sel,
        tipologias=tip_sel,
        logradouros=log_sel,
        faixas_area=area_sel,
        low_q=q_low,
        high_q=q_high,
        remove_outliers=remove_outliers,
        exclude_partial_transfers=exclude_partial_transfers,
    )

    filtered, outlier_lo, outlier_hi = apply_filters(df, filters)
    if filtered.empty:
        st.warning("Nenhum registro no recorte selecionado.")
        st.stop()

    df_sem_filtro_bairro, _, _ = apply_filters(df, filters, use_bairro=False, use_logradouro=False)
    if df_sem_filtro_bairro.empty:
        df_sem_filtro_bairro = filtered.copy()

    if benchmark_kind == "Geral sem filtro de bairro":
        benchmark_df = df_sem_filtro_bairro.copy()
    else:
        benchmark_df = filtered.copy()

    if benchmark_df.empty:
        benchmark_df = filtered.copy()

    compare_candidates = rank_by_volume(filtered, "bairro", top_n=max(3, filtered["bairro"].nunique()))
    compare_default = compare_candidates[:3]

    ts_region = time_group(filtered, freq)
    ts_benchmark = time_group(benchmark_df, freq)
    ts_bairro = time_group(filtered, freq, ["bairro"])

    # Janelas móveis são sempre mensais, independentemente da agregação temporal
    # escolhida para os demais gráficos. O período de origem continua sendo o
    # filtro do usuário.
    ts_benchmark_monthly = time_group(benchmark_df, "Mensal")
    ts_bairro_monthly = time_group(filtered, "Mensal", ["bairro"])

    render_orientation_panel()
    render_kpis(filtered, ts_region, min_trans, rj_general_median_m2=rj_general_median_m2)

    tabs = st.tabs(
        [
            "Visão geral",
            "Bairros lado a lado",
            "Vale a pena investir?",
            "Cada bairro no seu ritmo",
            "Mapa",
            "Rua a rua",
            "Por tamanho de imóvel",
            "Metodologia",
            "Qualidade dos dados",
        ]
    )

    with tabs[0]:
        st.subheader("Preço e volume sem eixo duplo")
        st.plotly_chart(
            make_price_volume_chart(ts_region, "Preço/m² corrigido e volume — recorte selecionado", raw_df=filtered),
            use_container_width=True,
            theme=None,
        )
        st.caption(
            "A faixa azul sombreada mostra o intervalo típico do preço/m² em cada mês. "
            "Ela ajuda a separar a tendência média da dispersão normal dos imóveis negociados."
        )

        st.markdown("#### Ranking de valorização real")
        valorizacao = valuation_ranking(ts_bairro, min_trans)
        if valorizacao.empty:
            st.info("Não há bairros com pelo menos dois períodos válidos para calcular valorização acumulada.")
        else:
            st.plotly_chart(
                make_valuation_ranking_chart(
                    valorizacao,
                    "Variação real acumulada por bairro — último período vs primeiro período válido",
                    top_n=min(top_n, 25),
                ),
                use_container_width=True,
                theme=None,
            )
            st.caption(
                "Variação real já desconta inflação (IPCA). Um bairro com +10% aqui cresceu 10% acima da inflação no período."
            )

        st.markdown("---")
        rank_bairros = ranking_bairros(filtered, min_trans)
        st.plotly_chart(make_affordability_chart(rank_bairros), use_container_width=True, theme=None)
        st.caption(
            "Estimativa simples: preço/m² ponderado do bairro multiplicado pela metragem. "
            "Serve para comparar ordem de grandeza; não substitui avaliação por rua, tipologia, andar e estado do imóvel."
        )
        st.caption(
            "Para comparar: o custo médio de aluguel residencial em RJ gira em torno de 0,4–0,6% do valor do imóvel ao mês "
            "(yield bruto estimado de mercado). Este painel não calcula aluguel — use como referência de ordem de grandeza."
        )

        st.markdown("#### Budget finder")
        bf1, bf2 = st.columns(2)
        budget_input = bf1.text_input("Orçamento máximo (R$)", value="600.000")
        budget_max = parse_br_number(budget_input, default=600000.0)
        target_area = bf2.number_input("Metragem desejada (m²)", min_value=20.0, value=65.0, step=5.0, format="%.0f")
        st.caption(f"Orçamento informado: {br_money(budget_max)} · Metragem desejada: {br_area(target_area)}")
        budget_df = budget_finder(rank_bairros, budget_max, target_area, min_trans)
        if budget_df.empty:
            st.info("Não há bairros suficientes para o budget finder no recorte atual.")
        else:
            cols_bf = st.columns([1.15, 0.85])
            with cols_bf[0]:
                show_friendly_dataframe(budget_df[["bairro", "mediana_preco_m2", "preco_m2_corrigido", "custo_estimado", "bairro_alcancavel", "total_transações"]].head(20))
            with cols_bf[1]:
                if not geo.empty:
                    st.plotly_chart(make_budget_finder_map(budget_df[budget_df["bairro_alcancavel"] == "Sim"], geo), use_container_width=True, theme=None)
            st.caption("O budget finder usa o preço mediano por m² do bairro para estimar se a metragem desejada cabe no orçamento informado. É um filtro de triagem, não um substituto para análise por rua ou padrão do imóvel.")

        st.markdown("#### Bairros com maior aceleração recente")
        acc_rank = ranking_aceleracao_recente(filtered, min_trans)
        if acc_rank.empty:
            st.info("Não há bairros com pelo menos 12 meses válidos para calcular aceleração recente.")
        else:
            st.plotly_chart(make_acceleration_chart(acc_rank, top_n=min(12, max(6, top_n))), use_container_width=True, theme=None)

        st.markdown("#### Sazonalidade")
        sazonalidade = seasonality_table(filtered)
        if sazonalidade.empty:
            st.info("Não há dados suficientes para calcular sazonalidade no recorte atual.")
        else:
            st.plotly_chart(
                make_seasonality_chart(sazonalidade, "Sazonalidade por mês do ano"),
                use_container_width=True,
                theme=None,
            )
            st.caption(
                "Mostra se há meses historicamente mais movimentados ou com preços mais altos. Útil para decidir quando comprar ou lançar."
            )

        c1, c2 = st.columns([1.05, 0.95])
        with c1:
            st.markdown("#### Decomposição por bairro")
            decomp = aggregate_metrics(filtered, ["bairro"]).sort_values("total_transações", ascending=False)
            decomp["rank_volume"] = np.arange(1, len(decomp) + 1)
            total_tx = decomp["total_transações"].sum()
            total_tx_sem_bairro = df_sem_filtro_bairro[TX_COL].sum()
            decomp["participação_volume"] = decomp["total_transações"] / total_tx if total_tx else np.nan
            show_friendly_dataframe(
                decomp[
                    [
                        "rank_volume",
                        "bairro",
                        "total_transações",
                        "participação_volume",
                        "preco_m2_corrigido",
                        "mediana_preco_m2",
                        "preco_m2_media_simples",
                        "valor_total_corrigido",
                        "area_total",
                    ]
                ]
            )
            concentração = decomp["total_transações"].sum() / total_tx_sem_bairro if total_tx_sem_bairro else np.nan
            st.caption(
                f"Os {len(decomp)} bairros acima concentram {br_pct(concentração)} das transações do recorte sem filtro de bairro."
            )
            render_download_table(decomp, "decomposicao_bairros.csv", "Baixar decomposição")

        with c2:
            st.markdown("#### Top logradouros por preço/m²")
            rank = ranking_logradouros(filtered, min_trans)
            show_friendly_dataframe(
                rank.head(25)[
                    [
                        "bairro",
                        "logradouro",
                        "uso",
                        "total_transações",
                        "preco_m2_corrigido",
                        "mediana_preco_m2",
                        "preco_m2_media_simples",
                    ]
                ]
            )
            render_download_table(rank, "ranking_logradouros.csv", "Baixar ranking")

    with tabs[1]:
        st.subheader("Comparação com a média do recorte")
        st.caption(
            "A linha âmbar tracejada é a média agregada do recorte de comparação. Use o nível absoluto para comparar patamar de preço e o rendimento acumulado para comparar trajetória percentual."
        )
        bairros_compare_tab = select_bairros_checkbox_dropdown(
            "Selecionar bairros para comparação",
            compare_candidates,
            compare_default,
            key="bairros_compare_checkbox",
            max_items=3,
        )
        if not bairros_compare_tab:
            bairros_compare_tab = compare_default
        st.caption(
            f"Bairros exibidos: {', '.join(bairros_compare_tab) if bairros_compare_tab else 'nenhum bairro disponível no filtro atual'}."
        )
        st.plotly_chart(
            make_level_comparison_chart(
                ts_bairro,
                ts_benchmark,
                "Nível de preço/m² por bairro — linha âmbar = média de comparação",
                top_n,
                selected_bairros=bairros_compare_tab,
            ),
            use_container_width=True,
            theme=None,
        )
        st.plotly_chart(
            make_index_chart(
                ts_bairro,
                ts_benchmark,
                "Rendimento acumulado desde o início do período",
                top_n,
                selected_bairros=bairros_compare_tab,
            ),
            use_container_width=True,
            theme=None,
        )

        st.markdown("#### Drawdown histórico")
        st.plotly_chart(
            make_drawdown_chart(
                ts_bairro_monthly,
                ts_benchmark_monthly,
                "Drawdown histórico",
                top_n=top_n,
                selected_bairros=bairros_compare_tab,
            ),
            use_container_width=True,
            theme=None,
        )
        st.caption("Drawdown mede a queda percentual em relação ao pico anterior do próprio bairro. Quanto mais negativo, maior foi a perda acumulada desde o topo.")

        heat_freq = st.radio("Agrupamento do heatmap", ["Mensal", "Trimestral", "Anual"], horizontal=True)
        heat_base = filtered.copy()
        if heat_freq == "Mensal":
            heat_base["periodo_heatmap"] = heat_base[DATE_COL].dt.to_period("M").astype(str)
        elif heat_freq == "Trimestral":
            heat_base["periodo_heatmap"] = heat_base[DATE_COL].dt.to_period("Q").astype(str).str.replace("Q", " T", regex=False)
        else:
            heat_base["periodo_heatmap"] = heat_base[DATE_COL].dt.year.astype(str)

        heat = aggregate_metrics(heat_base, ["bairro", "periodo_heatmap"])
        heat = heat[heat["total_transações"] >= min_trans]
        if not heat.empty:
            st.plotly_chart(
                make_heatmap(heat, f"Heatmap {heat_freq.lower()} — preço/m² corrigido"),
                use_container_width=True,
                theme=None,
            )

    with tabs[2]:
        st.subheader("Vale a pena investir?")
        c_sel_rolling, c_window_rolling = st.columns([0.68, 0.32])
        with c_sel_rolling:
            bairros_rolling_tab = select_bairros_checkbox_dropdown(
                "Selecionar bairros para comparação",
                compare_candidates,
                compare_default,
                key="bairros_rolling_checkbox",
                max_items=3,
            )
        with c_window_rolling:
            rolling_window = st.selectbox(
                "Janela móvel",
                ROLLING_WINDOW_OPTIONS,
                index=0,
                format_func=lambda m: f"{m} meses",
                key="rolling_window_tab_select",
                help="Escolha a janela diretamente nesta análise.",
            )
        if not bairros_rolling_tab:
            bairros_rolling_tab = compare_default

        if month_span(filters.start, filters.end) < rolling_window:
            st.info(
                f"O período filtrado tem menos de {rolling_window} meses completos. "
                "Amplie o intervalo no filtro lateral ou escolha uma janela menor."
            )
        else:
            rolling_bairros = rolling_returns_by_series(
                ts_bairro_monthly,
                rolling_window,
                min_trans,
                group_col="bairro",
            )
            rolling_bench = rolling_returns_by_series(
                ts_benchmark_monthly,
                rolling_window,
                min_trans,
                series_name=BENCHMARK_SERIES_NAME,
            )
            rolling = pd.concat([rolling_bairros, rolling_bench], ignore_index=True)
            rolling_summary = rolling_gain_summary(rolling)

            if rolling.empty:
                st.warning(
                    "Não há janelas válidas para o filtro atual. Reduza o mínimo de transações, amplie o período ou selecione uma janela menor."
                )
            else:
                st.plotly_chart(
                    make_gain_quadrant_chart(rolling_summary, rolling_window),
                    use_container_width=True,
                    theme=None,
                )
                st.caption(
                    "A linha de 8% aproxima custos de transação usuais de compra e venda, como ITBI, corretagem, escritura e registros. "
                    "Retornos reais abaixo desse patamar podem desaparecer depois desses custos."
                )

                st.markdown("#### Evolução temporal das janelas")
                st.plotly_chart(
                    make_rolling_return_chart(
                        rolling,
                        rolling_summary,
                        f"Retorno real por janela móvel de {rolling_window} meses — bairros e média de comparação",
                        top_n=top_n,
                        selected_bairros=bairros_rolling_tab,
                    ),
                    use_container_width=True,
                    theme=None,
                )

                k1, k2, k3 = st.columns(3)
                bench_row = rolling_summary[rolling_summary["serie"] == BENCHMARK_SERIES_NAME]
                if not bench_row.empty:
                    bench_row = bench_row.iloc[0]
                    k1.metric("Prob. de ganho — média comparação", br_pct(bench_row["probabilidade_ganho"]))
                    k2.metric("Janelas válidas", br_number(bench_row["janelas_validas"]))
                    k3.metric("Retorno mediano", br_pct(bench_row["retorno_mediano"]))
                else:
                    k1.metric("Prob. de ganho — média comparação", "—")
                    k2.metric("Janelas válidas", "—")
                    k3.metric("Retorno mediano", "—")

                render_rolling_probability_cards(rolling_summary, bairros_rolling_tab)

                st.markdown("#### Detalhamento das janelas")
                show_friendly_dataframe(
                    rolling_summary[
                        [
                            "serie",
                            "janelas_validas",
                            "janelas_com_ganho",
                            "probabilidade_ganho",
                            "retorno_medio",
                            "retorno_mediano",
                            "melhor_janela",
                            "pior_janela",
                            "retorno_mais_recente",
                            "inicio_mais_recente",
                            "fim_mais_recente",
                        ]
                    ].rename(columns={"serie": "bairro / referência"})
                )

                st.markdown("#### Resumo das janelas 12, 24, 36, 48 e 60 meses")
                all_windows_summary = rolling_summary_for_windows(
                    ts_bairro_monthly,
                    ts_benchmark_monthly,
                    ROLLING_WINDOW_OPTIONS,
                    min_trans,
                )
                if not all_windows_summary.empty:
                    show_friendly_dataframe(
                        all_windows_summary[
                            [
                                "janela_meses",
                                "serie",
                                "janelas_validas",
                                "probabilidade_ganho",
                                "retorno_mediano",
                                "retorno_mais_recente",
                                "inicio_mais_recente",
                                "fim_mais_recente",
                            ]
                        ].rename(columns={"serie": "bairro / referência"})
                    )
                    render_download_table(all_windows_summary, "janelas_moveis_probabilidade_ganho.csv", "Baixar resumo das janelas")

    with tabs[3]:
        st.subheader("Pequenos múltiplos")
        st.caption(
            "A comparação fica mais legível quando cada bairro ocupa seu próprio painel e usa a mesma escala vertical."
        )
        st.plotly_chart(
            make_small_multiples(ts_bairro, "Preço/m² corrigido por bairro", top_n, common_y),
            use_container_width=True,
            theme=None,
        )

    with tabs[4]:
        st.subheader("Mapa por logradouro")
        if geo.empty:
            st.warning(
                "Arquivo de coordenadas não encontrado. Gere `data/geocoded_logradouros.csv` ou baixe o template abaixo."
            )
            template = filtered[["bairro", "logradouro", "endereco_mapa"]].drop_duplicates().sort_values(
                ["bairro", "logradouro"]
            )
            template["latitude"] = np.nan
            template["longitude"] = np.nan
            template["status_geocodificacao"] = "pendente"
            st.download_button(
                "Baixar template de geocodificação",
                template.to_csv(index=False).encode("utf-8-sig"),
                "template_geocodificacao_logradouros.csv",
                "text/csv",
            )
            st.code("py geocode_logradouros.py --limit 500")
        else:
            mapa = aggregate_metrics(filtered, ["bairro", "logradouro"])
            mapa = mapa.merge(
                geo[["bairro", "logradouro", "latitude", "longitude"]],
                on=["bairro", "logradouro"],
                how="left",
            ).dropna(subset=["latitude", "longitude"])
            mapa = mapa[mapa["total_transações"] >= min_trans].copy()

            # Exclui logradouros de morro, comunidade e topônimos irrelevantes para mercado imobiliário formal
            _EXCLUIR_PADROES = re.compile(
                r"\b(morro|comunidade|alto do|alto da|vila\s+\w+\s*(do|da|de)\s+morro|pico|parque\s+nacional|"
                r"estrada\s+(das\s+)?canoas|estrada\s+das\s+furnas|alto\s+da\s+boa\s+vista|"
                r"floresta\s+da\s+tijuca|parque\s+lage|serr[ao])\b",
                flags=re.IGNORECASE,
            )
            mapa = mapa[~mapa["logradouro"].astype(str).str.contains(_EXCLUIR_PADROES, na=False)].copy()

            if mapa.empty:
                st.info("Não há logradouros geocodificados suficientes para o filtro atual.")
            else:
                color_by = st.radio("Cor do mapa", ["Preço/m²", "Volume"], horizontal=True)
                st.caption(
                    "Cor representa a variável escolhida. Os pontos têm tamanho fixo; volume aparece no hover ou quando você seleciona Volume na cor. "
                    "A escala de cor usa percentis 5–95 para reduzir achatamento por outliers."
                )
                st.plotly_chart(make_map(mapa, color_by), use_container_width=True, theme=None)
                show_friendly_dataframe(
                    mapa.sort_values("preco_m2_corrigido", ascending=False).head(200)
                )

    with tabs[5]:
        st.subheader("Dentro do bairro")
        bairros_para_detalhe = sorted(filtered["bairro"].dropna().unique().tolist())
        bairro_detalhe = st.selectbox("Bairro para detalhar", bairros_para_detalhe)
        df_bairro = filtered[filtered["bairro"] == bairro_detalhe]

        st.markdown(f"#### Logradouros de {bairro_detalhe}")
        rank_log = ranking_logradouros(df_bairro, min_trans)
        st.plotly_chart(
            make_rank_dotplot(rank_log, f"Top logradouros — {bairro_detalhe}", top_n=25, show_bairro=False),
            use_container_width=True,
            theme=None,
        )
        show_friendly_dataframe(
            rank_log.head(60)[
                [
                    "logradouro",
                    "uso",
                    "total_transações",
                    "preco_m2_corrigido",
                    "mediana_preco_m2",
                    "preco_m2_media_simples",
                    "valor_total_corrigido",
                    "area_total",
                ]
            ]
        )



    with tabs[6]:
        st.subheader("Segmentação por metragem")
        st.caption(
            "Esta aba evita comparar apartamentos pequenos e grandes como se fossem o mesmo produto. "
            "Use-a junto com tipologia e uso."
        )
        zonas_para_comparar = [nome for nome in CESTAS_BAIRROS.keys() if nome != "Personalizado"]
        zona_area = st.selectbox("Comparar com média de uma grande zona", ["Nenhuma"] + zonas_para_comparar)
        zone_df = None
        if zona_area != "Nenhuma":
            zone_filters = Filters(
                start=filters.start,
                end=filters.end,
                bairros=CESTAS_BAIRROS[zona_area],
                usos=filters.usos,
                tipologias=filters.tipologias,
                logradouros=[],
                faixas_area=filters.faixas_area,
                low_q=filters.low_q,
                high_q=filters.high_q,
                remove_outliers=filters.remove_outliers,
                exclude_partial_transfers=filters.exclude_partial_transfers,
            )
            zone_df, _, _ = apply_filters(df, zone_filters, use_bairro=True, use_logradouro=False)

        st.plotly_chart(make_area_chart(filtered, min_trans, top_n=top_n, zone_df=zone_df, zone_name=None if zona_area == "Nenhuma" else zona_area), use_container_width=True, theme=None)

        st.markdown("#### Área × preço/m²")
        st.plotly_chart(make_area_price_distribution_chart(filtered, top_n=min(10, max(6, top_n))), use_container_width=True, theme=None)
        st.caption("A base está agregada por logradouro/período, não por transação individual. Por isso, o gráfico usa faixa de área no eixo X e box plot de preço/m² no eixo Y, evitando a falsa impressão de precisão de um scatter com área média quase constante.")

        area_interpretacao_table = aggregate_metrics(assign_area_labels_exclusive(filtered), ["bairro", "faixa_area"])
        area_interpretacao_table = adicionar_percentis_area_bairro(
            area_interpretacao_table,
            filtered,
            group_cols=["bairro", "faixa_area"],
        )
        st.markdown("#### O que esses números dizem")
        interpretacoes = interpretar_area_bairro(area_interpretacao_table, min_trans)
        render_interpretacao_area(
            interpretacoes,
            bairros_sel,
            AREA_LABELS,
            min_trans,
        )

        st.markdown("#### Mix de produto por bairro")
        st.plotly_chart(make_product_mix_chart(filtered, top_n=min(10, max(6, top_n))), use_container_width=True, theme=None)
        st.caption("As barras empilhadas mostram o perfil de oferta de cada bairro por faixa de área: studios, apartamentos médios ou imóveis grandes. As faixas usam limite superior exclusivo: 55 m² entra em 55–75 m², não em 30–55 m².")

        area_table = aggregate_metrics(assign_area_labels_exclusive(filtered), ["faixa_area", "bairro", "uso", "principais_tipologias"])
        area_table = adicionar_percentis_area_bairro(
            area_table,
            filtered,
            group_cols=["bairro", "faixa_area"],
        )
        area_table = area_table[area_table["total_transações"] >= min_trans].sort_values(
            ["faixa_area", "total_transações"], ascending=[True, False]
        )
        show_friendly_dataframe(area_table)
        render_download_table(area_table, "segmentacao_area_tipologia.csv", "Baixar segmentação")

    with tabs[7]:
        st.subheader("Metodologia")
        st.markdown(
            """
##### Variáveis principais
- **valor_total_corrigido**: valor total da transação trazido a preços reais de março/2026 via IPCA.
- **area_construida**: área construída registrada na base.
- **preco_m2_corrigido**: `valor_total_corrigido / area_construida`.
- **mediana_preco_m2**: mediana do preço/m² no recorte; reduz sensibilidade a extremos.
- **preco_m2_corrigido (média ponderada)**: `soma(valor_total_corrigido) / soma(area_construida)` no grupo analisado.
- **total_transações**: soma das ocorrências/registros válidos no agrupamento.
- **valor_total_corrigido** e **area_total**: somatórios usados para compor médias ponderadas.

##### Como os principais cálculos são feitos
- **Valorização real acumulada por bairro**: compara o último período válido com o primeiro período válido do mesmo bairro, já com valores corrigidos pela inflação.
- **Rendimento acumulado (%)**: `preço_m2_corrigido do período / preço_m2_corrigido inicial - 1`.
- **Janelas móveis (12, 24, 36, 48, 60 meses)**: calculam o retorno real para cada janela completa dentro do período filtrado.
- **Probabilidade de ganho real**: proporção das janelas em que o retorno foi maior que zero.
- **Retorno mediano**: mediana dos retornos das janelas; é a medida central da experiência histórica.
- **Heatmap**: mostra o preço/m² real por bairro ao longo do tempo.
- **Budget finder**: estima um custo de compra usando o preço mediano por m² do bairro multiplicado pela metragem desejada.

##### Faixa típica, percentis e boxplots
- A faixa **P25–P75** representa o intervalo interquartil típico do preço/m².
- Nos gráficos **Área × preço/m²**, os whiskers e outliers seguem a lógica clássica do **IQR** do boxplot.
- O painel pode remover outliers extremos por quantis quando o usuário ativa esse filtro lateral.

##### Por que usar 8% como taxa mínima de retorno aceitável?
Os **8%** não vieram de um número arbitrário. Eles foram usados como uma **taxa mínima simplificada de fricção total da operação**, isto é, uma aproximação do custo de comprar e depois vender um imóvel urbano no Rio.

A decomposição usada no painel é esta:
- **3,0% de ITBI** na compra;
- **0,5% de escritura**;
- **0,5% de registro e despesas cartoriais/operacionais**;
- **4,0% de corretagem / custo de intermediação e liquidez na saída**.

Soma da hipótese-base:
- **3,0% + 0,5% + 0,5% + 4,0% = 8,0%**.

Ou seja, a lógica é: para que o investimento realmente gere ganho econômico líquido, o imóvel precisa superar pelo menos essa barreira aproximada de custos de entrada e saída. Por isso:
- retorno real **abaixo de 8%** sugere que o investidor pode ter apenas coberto custos;
- retorno real **acima de 8%** sugere criação líquida de valor, antes de considerar custo de capital e risco.

Essa taxa é uma **hipótese analítica padronizada do painel**, não uma verdade universal. Em operações reais, o total pode ficar abaixo ou acima disso, dependendo de desconto na compra, corretagem efetiva, custo jurídico, forma de escritura e liquidez do ativo.

##### Como ler o gráfico de quadrantes de retorno
O gráfico usado na aba **Vale a pena investir?** cruza duas informações históricas das **janelas móveis** escolhidas (12, 24, 36, 48 ou 60 meses):
- **Eixo X — Probabilidade de ganho real**: proporção das janelas em que o retorno foi **maior que zero**, isto é, em que o bairro ficou **acima da inflação** no período.
- **Eixo Y — Retorno mediano real no período**: mediana dos retornos reais observados nas janelas daquele bairro.
- **Tamanho da bolha**: número de janelas válidas disponíveis para o cálculo.
- **Losango laranja**: média do recorte de comparação.
- **Linha horizontal de 8%**: referência mínima estimada para superar custos de transação.

##### Por que as regiões se chamam “Abaixo da inflação”, “Inconclusivo”, “Consistente” e “Forte”?
Esses nomes derivam da **probabilidade histórica de o bairro entregar ganho real**, isto é, retorno acima da inflação:
- **Abaixo da inflação**: probabilidade **menor que 50%**. Historicamente, o bairro passou **mais vezes abaixo da inflação do que acima**.
- **Inconclusivo**: probabilidade entre **50% e 65%**. Há vantagem sobre o zero real, mas ainda sem evidência suficientemente robusta.
- **Consistente**: probabilidade entre **65% e 75%**. O bairro superou a inflação com frequência alta e relativamente estável.
- **Forte**: probabilidade **acima de 75%**. O histórico mostra predominância clara de janelas com ganho real.

Esses cortes (**50%, 65% e 75%**) são **faixas analíticas do painel**, usadas para transformar a leitura estatística em categorias interpretáveis. Eles não são uma lei econômica; são convenções de leitura para separar:
- ausência de evidência (<50%),
- evidência fraca/intermediária (50–65%),
- evidência consistente (65–75%),
- evidência forte (>75%).

A interpretação correta combina os dois eixos:
- um bairro pode ter **probabilidade alta**, mas **retorno mediano baixo**;
- ou pode ter **retorno mediano alto**, mas com **probabilidade baixa**, o que sugere comportamento mais errático.

Por isso o gráfico não mede apenas “quanto subiu”, mas **com que frequência o ganho real aconteceu** e **quão relevante ele foi em termos de retorno mediano**.
            """
        )

    with tabs[8]:
        st.subheader("Qualidade dos dados")
        st.markdown(
            """
O georreferenciamento deste painel foi feito com **geopy**.

O dataset primário pode ser consultado na página do **Data.Rio**:
[ITBI — base no Data.Rio](https://www.data.rio/datasets/5e4dda4d33f44b1eb9246559b281d1b8_8/about)
            """
        )



if __name__ == "__main__":
    main()
