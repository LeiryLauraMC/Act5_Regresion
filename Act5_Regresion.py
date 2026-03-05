import streamlit as st
import pandas as pd
import numpy as np
import requests
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA · Educación Global",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# PALETA
# ──────────────────────────────────────────────────────────────────────────────
C1 = "#7EB5D6"
C2 = "#E8A090"
C3 = "#A8C8A0"
C4 = "#C4A8C8"
C5 = "#F0C890"
C6 = "#90C8D8"
PALETTE  = [C1, C2, C3, C4, C5, C6]
COL_TIT  = "#2E4A6B"
COL_REF  = "#9E3A3A"
BG_PLOT  = "#FDFAF6"
PAPER_BG = "#F7F3EE"

PLOTLY_BASE = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=BG_PLOT,
    font=dict(family="Georgia, serif", color="#3A3228", size=11),
    title_font=dict(family="Georgia, serif", color=COL_TIT, size=13),
    legend=dict(bgcolor="rgba(253,250,246,0.88)", bordercolor="#D8D0C8",
                borderwidth=1, font=dict(size=10)),
    margin=dict(l=60, r=30, t=60, b=50),
    hoverlabel=dict(bgcolor="#fff", bordercolor="#C8C0B8",
                    font=dict(family="Georgia, serif", size=11)),
)

AXIS_STYLE = dict(gridcolor="#E2DBD4", gridwidth=0.5,
                  linecolor="#C8C0B8", linewidth=0.8,
                  tickfont=dict(size=10), zeroline=False)

def apply_layout(fig, title="", height=430):
    fig.update_layout(**PLOTLY_BASE, title=title, height=height)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# ESTILOS GLOBALES
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #F7F3EE !important;
    font-family: 'Source Sans 3', sans-serif;
}
[data-testid="stHeader"] { background-color: #F7F3EE !important; }

/* ── hero ── */
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(125deg, #1E3350 0%, #2E4A6B 30%, #3D6B55 68%, #7A6848 100%);
    border-radius: 18px;
    padding: 2.6rem 3rem 2.4rem;
    margin-bottom: 1.8rem;
    color: #fff;
    box-shadow: 0 8px 36px rgba(30,51,80,.30);
}
.hero::after {
    content: '';
    position: absolute;
    right: 0; top: 0; bottom: 0;
    width: 46%;
    background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI1MjAiIGhlaWdodD0iMjIwIiB2aWV3Qm94PSIwIDAgNTIwIDIyMCI+CiAgPGNpcmNsZSBjeD0iNDIiIGN5PSIxODUiIHI9IjUuNSIgZmlsbD0iIzdFQjVENiIgb3BhY2l0eT0iMC44MiIvPgogIDxjaXJjbGUgY3g9IjYwIiBjeT0iMTcyIiByPSI0IiBmaWxsPSIjN0VCNUQ2IiBvcGFjaXR5PSIwLjY1Ii8+CiAgPGNpcmNsZSBjeD0iNzUiIGN5PSIxNjMiIHI9IjYiIGZpbGw9IiM5MEM4RDgiIG9wYWNpdHk9IjAuNzUiLz4KICA8Y2lyY2xlIGN4PSI1NSIgY3k9IjE1NSIgcj0iMy41IiBmaWxsPSIjN0VCNUQ2IiBvcGFjaXR5PSIwLjU1Ii8+CiAgPGNpcmNsZSBjeD0iODgiIGN5PSIxNDgiIHI9IjUiIGZpbGw9IiM5MEM4RDgiIG9wYWNpdHk9IjAuNzAiLz4KICA8Y2lyY2xlIGN4PSIxMDAiIGN5PSIxMzgiIHI9IjQuNSIgZmlsbD0iIzdFQjVENiIgb3BhY2l0eT0iMC43OCIvPgogIDxjaXJjbGUgY3g9IjExOCIgY3k9IjEyOCIgcj0iNSIgZmlsbD0iI0E4QzhBMCIgb3BhY2l0eT0iMC43MiIvPgogIDxjaXJjbGUgY3g9IjExMCIgY3k9IjExOCIgcj0iMy41IiBmaWxsPSIjOTBDOEQ4IiBvcGFjaXR5PSIwLjYwIi8+CiAgPGNpcmNsZSBjeD0iMTMyIiBjeT0iMTEwIiByPSI1LjUiIGZpbGw9IiNBOEM4QTAiIG9wYWNpdHk9IjAuNzgiLz4KICA8Y2lyY2xlIGN4PSIxNDgiIGN5PSIxMDAiIHI9IjQiIGZpbGw9IiNDNEE4QzgiIG9wYWNpdHk9IjAuNjUiLz4KICA8Y2lyY2xlIGN4PSIxNjUiIGN5PSI5MCIgcj0iNSIgZmlsbD0iI0E4QzhBMCIgb3BhY2l0eT0iMC43MiIvPgogIDxjaXJjbGUgY3g9IjE1NSIgY3k9IjgwIiByPSIzLjUiIGZpbGw9IiNDNEE4QzgiIG9wYWNpdHk9IjAuNTUiLz4KICA8Y2lyY2xlIGN4PSIxODAiIGN5PSI3MiIgcj0iNS41IiBmaWxsPSIjRjBDODkwIiBvcGFjaXR5PSIwLjc1Ii8+CiAgPGNpcmNsZSBjeD0iMTk1IiBjeT0iNjMiIHI9IjQuNSIgZmlsbD0iI0U4QTA5MCIgb3BhY2l0eT0iMC42OCIvPgogIDxjaXJjbGUgY3g9IjIxMCIgY3k9IjU1IiByPSI1IiBmaWxsPSIjRjBDODkwIiBvcGFjaXR5PSIwLjcyIi8+CiAgPGNpcmNsZSBjeD0iMjI1IiBjeT0iNDciIHI9IjQiIGZpbGw9IiNFOEEwOTAiIG9wYWNpdHk9IjAuNzgiLz4KICA8Y2lyY2xlIGN4PSIyNDAiIGN5PSI0MCIgcj0iNS41IiBmaWxsPSIjRThBMDkwIiBvcGFjaXR5PSIwLjgwIi8+CiAgPGNpcmNsZSBjeD0iMTMwIiBjeT0iMTQ1IiByPSIzLjUiIGZpbGw9IiNDNEE4QzgiIG9wYWNpdHk9IjAuNTAiLz4KICA8Y2lyY2xlIGN4PSIxNzAiIGN5PSIxMzAiIHI9IjQiIGZpbGw9IiNGMEM4OTAiIG9wYWNpdHk9IjAuNTUiLz4KICA8Y2lyY2xlIGN4PSIyMDAiIGN5PSIxMDUiIHI9IjMiIGZpbGw9IiNFOEEwOTAiIG9wYWNpdHk9IjAuNDUiLz4KICA8Y2lyY2xlIGN4PSI4NSIgY3k9IjE3MCIgcj0iMyIgZmlsbD0iIzkwQzhEOCIgb3BhY2l0eT0iMC40NSIvPgogIDxsaW5lIHgxPSIzNSIgeTE9IjE5MiIgeDI9IjI0OCIgeTI9IjM0IiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC41NSkiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWRhc2hhcnJheT0iNiA0IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz4KICA8bGluZSB4MT0iMzAiIHkxPSIzMCIgeDI9IjMwIiB5Mj0iMjAwIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC4zMCkiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz4KICA8bGluZSB4MT0iMzAiIHkxPSIyMDAiIHgyPSIyNjAiIHkyPSIyMDAiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjMwKSIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgogIDxyZWN0IHg9IjI5NSIgeT0iMTQwIiB3aWR0aD0iMjIiIGhlaWdodD0iNTUiIHJ4PSIzIiBmaWxsPSIjN0VCNUQ2IiBvcGFjaXR5PSIwLjU1Ii8+CiAgPHJlY3QgeD0iMzI1IiB5PSIxMTAiIHdpZHRoPSIyMiIgaGVpZ2h0PSI4NSIgcng9IjMiIGZpbGw9IiNBOEM4QTAiIG9wYWNpdHk9IjAuNTUiLz4KICA8cmVjdCB4PSIzNTUiIHk9IjgwIiB3aWR0aD0iMjIiIGhlaWdodD0iMTE1IiByeD0iMyIgZmlsbD0iI0M0QThDOCIgb3BhY2l0eT0iMC41NSIvPgogIDxyZWN0IHg9IjM4NSIgeT0iNTUiIHdpZHRoPSIyMiIgaGVpZ2h0PSIxNDAiIHJ4PSIzIiBmaWxsPSIjRjBDODkwIiBvcGFjaXR5PSIwLjU1Ii8+CiAgPHJlY3QgeD0iNDE1IiB5PSIzNSIgd2lkdGg9IjIyIiBoZWlnaHQ9IjE2MCIgcng9IjMiIGZpbGw9IiNFOEEwOTAiIG9wYWNpdHk9IjAuNjAiLz4KICA8bGluZSB4MT0iMjg4IiB5MT0iMjAwIiB4Mj0iNDQ4IiB5Mj0iMjAwIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC4yOCkiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz4KICA8Y2lyY2xlIGN4PSI0NzgiIGN5PSI2MCIgcj0iMTgiIGZpbGw9IiM3RUI1RDYiIG9wYWNpdHk9IjAuMjIiLz4KICA8Y2lyY2xlIGN4PSI1MDAiIGN5PSI2MCIgcj0iMTgiIGZpbGw9IiNBOEM4QTAiIG9wYWNpdHk9IjAuMjIiLz4KICA8Y2lyY2xlIGN4PSI0NzgiIGN5PSI4MiIgcj0iMTgiIGZpbGw9IiNGMEM4OTAiIG9wYWNpdHk9IjAuMjIiLz4KICA8Y2lyY2xlIGN4PSI1MDAiIGN5PSI4MiIgcj0iMTgiIGZpbGw9IiNFOEEwOTAiIG9wYWNpdHk9IjAuMjIiLz4KICA8Y2lyY2xlIGN4PSI0NzgiIGN5PSIxMDQiIHI9IjE4IiBmaWxsPSIjQzRBOEM4IiBvcGFjaXR5PSIwLjIyIi8+CiAgPGNpcmNsZSBjeD0iNTAwIiBjeT0iMTA0IiByPSIxOCIgZmlsbD0iIzdFQjVENiIgb3BhY2l0eT0iMC4yMiIvPgo8L3N2Zz4=");
    background-repeat: no-repeat;
    background-position: center right;
    background-size: contain;
    opacity: 0.88;
    pointer-events: none;
    border-radius: 0 18px 18px 0;
    -webkit-mask-image: linear-gradient(to right, transparent 0%, rgba(0,0,0,0.4) 18%, rgba(0,0,0,1) 42%);
    mask-image: linear-gradient(to right, transparent 0%, rgba(0,0,0,0.4) 18%, rgba(0,0,0,1) 42%);
}
.hero-content { position: relative; z-index: 1; max-width: 58%; }
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem; font-weight: 700;
    margin: 0 0 .22rem; letter-spacing: .015em; line-height: 1.2;
}
.hero .author {
    font-family: 'Source Sans 3', sans-serif;
    font-size: .85rem; font-weight: 400;
    letter-spacing: .1em; text-transform: uppercase;
    opacity: .68; margin: 0 0 .7rem; display: block;
}
.hero p { font-size: .95rem; opacity: .82; margin: 0; line-height: 1.6; }
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,.13);
    border: 1px solid rgba(255,255,255,.26);
    border-radius: 3px; padding: .15rem .68rem;
    font-size: .67rem; margin-right: .38rem; margin-top: .72rem;
    letter-spacing: .07em; text-transform: uppercase;
}

/* ── tabs ── */
[data-testid="stTabs"] > div:first-child {
    gap: 4px; background: #EDE8E0; border-radius: 10px;
    padding: 5px 7px; margin-bottom: 1.4rem;
}
button[data-baseweb="tab"] {
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: .88rem !important; font-weight: 600 !important;
    letter-spacing: .05em !important; text-transform: uppercase !important;
    color: #6B5E4C !important; background: transparent !important;
    border-radius: 7px !important; padding: .42rem 1.05rem !important;
    border: none !important; transition: all .18s ease !important;
}
button[data-baseweb="tab"]:hover { background: rgba(255,255,255,.5) !important; color: #2E4A6B !important; }
button[aria-selected="true"][data-baseweb="tab"] {
    background: #fff !important; color: #2E4A6B !important;
    box-shadow: 0 1px 6px rgba(46,74,107,.13) !important;
}

/* ── section titles ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem; font-weight: 700; color: #2E4A6B;
    border-left: 4px solid #4E7A5E; padding-left: .75rem; margin-bottom: 1rem;
}
.sub-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.08rem; font-weight: 600; color: #4A3C2C;
    margin-top: 1.4rem; margin-bottom: .5rem;
    padding-bottom: .3rem; border-bottom: 1px solid #E0D8CE;
}

/* ── metric cards ── */
.card-row { display: flex; gap: 14px; flex-wrap: wrap; margin: 1rem 0; }
.metric-card {
    flex: 1; min-width: 140px; background: #fff; border-radius: 12px;
    padding: 1rem 1.2rem; box-shadow: 0 1px 8px rgba(0,0,0,.06);
    border-top: 3px solid #4E7A5E; text-align: center;
}
.metric-card .label { font-size: .70rem; color: #9E8C78; text-transform: uppercase; letter-spacing: .07em; margin-bottom: .3rem; }
.metric-card .value { font-family: 'Playfair Display', serif; font-size: 1.55rem; font-weight: 700; color: #2E4A6B; }
.metric-card .sub { font-size: .74rem; color: #A89880; margin-top: .15rem; }

/* ── info / answer boxes ── */
.info-box {
    background: #fff; border-left: 3px solid #A8997E;
    border-radius: 0 8px 8px 0; padding: .85rem 1.1rem; margin: .7rem 0;
    font-size: .93rem; color: #4A3E33; box-shadow: 0 1px 5px rgba(0,0,0,.04);
}
.info-box strong { color: #2E4A6B; }
.answer-box {
    background: #EDF2F8; border-left: 3px solid #2E4A6B;
    border-radius: 0 8px 8px 0; padding: .85rem 1.1rem; margin: .7rem 0; color: #2C3E57;
}
.answer-box h4 { font-family: 'Playfair Display', serif; margin: 0 0 .32rem; font-size: .97rem; color: #2E4A6B; }

/* ── custom tables ── */
.custom-table {
    width: 100%; border-collapse: collapse;
    font-family: 'Source Sans 3', sans-serif; font-size: .88rem; color: #3A3228;
    margin: .5rem 0 1.2rem; border-radius: 10px; overflow: hidden;
    box-shadow: 0 1px 8px rgba(0,0,0,.08);
}
.custom-table thead tr { background: #2E4A6B; color: #fff; text-transform: uppercase; letter-spacing: .06em; font-size: .73rem; }
.custom-table thead th { padding: .65rem 1rem; text-align: left; font-weight: 600; border: none; }
.custom-table tbody tr:nth-child(even) { background: #F0EBE4; }
.custom-table tbody tr:nth-child(odd)  { background: #FAF7F3; }
.custom-table tbody tr:hover { background: #E4EDF6; transition: background .15s; }
.custom-table tbody td { padding: .55rem 1rem; border-bottom: 1px solid #E8E0D8; vertical-align: middle; }
.custom-table tbody tr:last-child td { border-bottom: none; }
.custom-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
.custom-table td.highlight { color: #2E4A6B; font-weight: 600; }

/* ── filtros sidebar ── */
.filter-label { font-size: .78rem; font-weight: 600; color: #6B5E4C; text-transform: uppercase; letter-spacing: .06em; margin-bottom: .2rem; }

/* ── misc ── */
hr { border: none; border-top: 1px solid #DDD4C8; margin: 1.3rem 0; }
[data-testid="stSpinner"] > div > div { border-top-color: #2E4A6B !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS DE TABLA HTML
# ──────────────────────────────────────────────────────────────────────────────
def render_table(df, num_cols=None, highlight_col=None):
    num_cols = num_cols or []
    html = '<table class="custom-table"><thead><tr>'
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            cls = ""
            if col in num_cols: cls += " num"
            if highlight_col and col == highlight_col: cls += " highlight"
            html += f'<td class="{cls.strip()}">{row[col]}</td>'
        html += "</tr>"
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# DESCARGA Y PROCESAMIENTO
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def descargar_indicador(indicador, nombre, inicio=2010, fin=2023):
    url    = f"https://api.worldbank.org/v2/country/all/indicator/{indicador}"
    params = {"format": "json", "date": f"{inicio}:{fin}", "per_page": 20000}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 1 and data[1]:
            rows = []
            for rec in data[1]:
                if rec["value"] is not None and rec["countryiso3code"]:
                    rows.append({
                        "pais": rec["country"]["value"],
                        "codigo_pais": rec["countryiso3code"],
                        "year": int(rec["date"]),
                        nombre: rec["value"],
                    })
            return pd.DataFrame(rows)
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def cargar_datos():
    def mas_reciente(df, col):
        if df.empty:
            return pd.DataFrame()
        return (df.sort_values("year", ascending=False)
                  .groupby("codigo_pais").first().reset_index()
                  [["pais", "codigo_pais", col]])

    df_g = mas_reciente(descargar_indicador("SE.XPD.TOTL.GD.ZS",       "gasto_educacion_pib"), "gasto_educacion_pib")
    df_e = mas_reciente(descargar_indicador("UIS.EA.MEAN.1T6.AG25T99", "anios_escolaridad"),    "anios_escolaridad")
    if df_e.empty:
        df_e = mas_reciente(descargar_indicador("SE.SCH.LIFE", "anios_escolaridad"), "anios_escolaridad")
    df_p = mas_reciente(descargar_indicador("NY.GDP.PCAP.PP.CD",        "pib_per_capita"),      "pib_per_capita")
    df_a = mas_reciente(descargar_indicador("SE.ADT.LITR.ZS",           "tasa_alfabetizacion"), "tasa_alfabetizacion")

    df = (df_g
          .merge(df_e[["codigo_pais", "anios_escolaridad"]],   on="codigo_pais", how="inner")
          .merge(df_p[["codigo_pais", "pib_per_capita"]],      on="codigo_pais", how="inner")
          .merge(df_a[["codigo_pais", "tasa_alfabetizacion"]], on="codigo_pais", how="inner"))
    df = df.dropna().copy()
    df["log_pib_per_capita"] = np.log(df["pib_per_capita"])
    return df

@st.cache_data(show_spinner=False)
def ajustar_modelos(df_json):
    df    = pd.read_json(df_json)
    vars_ = ["gasto_educacion_pib", "anios_escolaridad", "pib_per_capita", "tasa_alfabetizacion"]
    df    = df.dropna(subset=vars_ + ["log_pib_per_capita"])
    y     = df["anios_escolaridad"].values

    X1    = df[["gasto_educacion_pib"]].values
    X1tr, X1te, y1tr, y1te = train_test_split(X1, y, test_size=0.2, random_state=42)
    m1    = LinearRegression().fit(X1tr, y1tr)
    y1p   = m1.predict(X1te)
    r2_1  = r2_score(y1te, y1p)
    rmse_1 = np.sqrt(mean_squared_error(y1te, y1p))
    mae_1  = mean_absolute_error(y1te, y1p)
    cv1   = cross_val_score(LinearRegression(), X1, y, cv=5, scoring="r2")
    sm1   = sm.OLS(df["anios_escolaridad"], sm.add_constant(df["gasto_educacion_pib"])).fit()

    vars2  = ["gasto_educacion_pib", "log_pib_per_capita", "tasa_alfabetizacion"]
    X2     = df[vars2].values
    X2tr, X2te, y2tr, y2te = train_test_split(X2, y, test_size=0.2, random_state=42)
    m2    = LinearRegression().fit(X2tr, y2tr)
    y2p   = m2.predict(X2te)
    r2_2  = r2_score(y2te, y2p)
    rmse_2 = np.sqrt(mean_squared_error(y2te, y2p))
    mae_2  = mean_absolute_error(y2te, y2p)
    cv2   = cross_val_score(LinearRegression(), X2, y, cv=5, scoring="r2")
    sm2   = sm.OLS(df["anios_escolaridad"], sm.add_constant(df[vars2])).fit()

    X2sc   = StandardScaler().fit_transform(df[vars2])
    m2std  = LinearRegression().fit(X2sc, df["anios_escolaridad"])
    importancia = sorted(
        zip(["Gasto educ. (% PIB)", "log(PIB per cápita)", "Tasa alfabetización"], m2std.coef_),
        key=lambda x: abs(x[1]), reverse=True
    )
    return dict(
        m1=m1, m2=m2, sm1=sm1, sm2=sm2,
        r2_1=r2_1, rmse_1=rmse_1, mae_1=mae_1, cv1=cv1,
        r2_2=r2_2, rmse_2=rmse_2, mae_2=mae_2, cv2=cv2,
        y1te=list(y1te), y1p=list(y1p), y2te=list(y2te), y2p=list(y2p),
        importancia=importancia, vars2=vars2,
    )

# ──────────────────────────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-content">
    <h1>Educación y Escolaridad Global</h1>
    <span class="author">Leiry Laura Mares Cure</span>
    <p>Análisis Exploratorio de Datos &mdash; Regresión Lineal</p>
    <div><span class="badge">Banco Mundial</span><span class="badge">Regresión Lineal</span><span class="badge">2010 – 2023</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CARGA
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner("Descargando datos del Banco Mundial..."):
    df_full = cargar_datos()

if df_full.empty:
    st.error("No se pudieron descargar los datos. Verifica tu conexión a internet.")
    st.stop()

variables   = ["gasto_educacion_pib", "anios_escolaridad", "pib_per_capita", "tasa_alfabetizacion"]
nombres_var = ["Gasto Educ. (% PIB)", "Años Escolaridad", "PIB per cápita (USD)", "Alfabetización (%)"]
VAR_MAP     = dict(zip(nombres_var, variables))
COLOR_MAP   = dict(zip(variables, [C1, C2, C3, C4]))

res = ajustar_modelos(df_full.to_json())

# ──────────────────────────────────────────────────────────────────────────────
# FILTROS GLOBALES (sidebar)
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filtros globales")
    st.markdown("---")

    rango_gasto = st.slider(
        "Gasto en educación (% PIB)",
        float(df_full["gasto_educacion_pib"].min()),
        float(df_full["gasto_educacion_pib"].max()),
        (float(df_full["gasto_educacion_pib"].min()),
         float(df_full["gasto_educacion_pib"].max())),
        step=0.1,
    )
    rango_escolaridad = st.slider(
        "Años de escolaridad",
        float(df_full["anios_escolaridad"].min()),
        float(df_full["anios_escolaridad"].max()),
        (float(df_full["anios_escolaridad"].min()),
         float(df_full["anios_escolaridad"].max())),
        step=0.5,
    )
    rango_pib = st.slider(
        "PIB per cápita (USD)",
        float(df_full["pib_per_capita"].min()),
        float(df_full["pib_per_capita"].max()),
        (float(df_full["pib_per_capita"].min()),
         float(df_full["pib_per_capita"].max())),
        step=500.0,
    )
    rango_alfa = st.slider(
        "Tasa de alfabetización (%)",
        float(df_full["tasa_alfabetizacion"].min()),
        float(df_full["tasa_alfabetizacion"].max()),
        (float(df_full["tasa_alfabetizacion"].min()),
         float(df_full["tasa_alfabetizacion"].max())),
        step=1.0,
    )

    paises_lista = sorted(df_full["pais"].unique())
    paises_sel   = st.multiselect("Filtrar por país (opcional)", paises_lista)

    st.markdown("---")
    st.caption(f"Países disponibles: {len(df_full)}")

# Aplicar filtros
mask = (
    df_full["gasto_educacion_pib"].between(*rango_gasto) &
    df_full["anios_escolaridad"].between(*rango_escolaridad) &
    df_full["pib_per_capita"].between(*rango_pib) &
    df_full["tasa_alfabetizacion"].between(*rango_alfa)
)
if paises_sel:
    mask &= df_full["pais"].isin(paises_sel)

df = df_full[mask].copy()

if df.empty:
    st.warning("No hay países que cumplan los filtros seleccionados. Ajusta los rangos.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Q — Question",
    "U — Understand",
    "E — Explore",
    "S — Study",
    "T — Tell",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB Q
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">Preguntas de Investigación</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    En este EDA (análisis exploratorio de datos) se busca conocer patrones globales entre el gasto en educación, el desarrollo económico, la alfabetización y los años de escolaridad, aportando evidencia empírica para la toma de decisiones en política pública, basados en indicadores educativos del Banco Mundial.
    </div>
    """, unsafe_allow_html=True)

    preguntas = [
        ("P1", "Relación lineal",           "¿Existe una relación lineal significativa entre el gasto público en educación (% del PIB) y los años promedio de escolaridad de un país?"),
        ("P2", "Influencia del PIB",         "¿El PIB per cápita tiene mayor influencia que el gasto en educación sobre los años promedio de escolaridad de la población?"),
        ("P3", "Correlación alfabetización", "¿Cuál es la correlación entre la tasa de alfabetización y los años promedio de escolaridad, y cómo se compara con las demás variables?"),
        ("P4", "Modelo múltiple vs simple",  "¿Un modelo de regresión múltiple predice significativamente mejor los años de escolaridad que uno basado únicamente en el gasto en educación?"),
    ]
    border_colors = [C1, C3, C2, C4]
    cols = st.columns(2)
    for i, (code, title, text) in enumerate(preguntas):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:#fff;border-radius:12px;padding:1.2rem 1.4rem;
                        border-top:4px solid {border_colors[i]};
                        box-shadow:0 1px 8px rgba(0,0,0,.06);margin-bottom:1rem;">
              <span style="font-family:'Playfair Display',serif;font-size:1.55rem;
                           color:{border_colors[i]};font-weight:700;">{code}</span>
              <span style="font-size:.69rem;text-transform:uppercase;letter-spacing:.08em;
                           color:#9E8C78;margin-left:.5rem;">{title}</span>
              <p style="margin:.5rem 0 0;color:#4A3E33;font-size:.93rem;line-height:1.55;">{text}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Variables del Estudio</div>', unsafe_allow_html=True)
    render_table(pd.DataFrame({
        "Variable": ["Gasto en educación (% PIB)", "Años promedio de escolaridad",
                     "PIB per cápita (PPA, USD)", "Tasa de alfabetización"],
        "Rol": ["Independiente", "Dependiente", "Independiente", "Independiente"],
        "Indicador Banco Mundial": ["SE.XPD.TOTL.GD.ZS", "UIS.EA.MEAN.1T6.AG25T99",
                                    "NY.GDP.PCAP.PP.CD", "SE.ADT.LITR.ZS"],
    }), highlight_col="Rol")

# ══════════════════════════════════════════════════════════════════════════════
# TAB U
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">Comprensión de los Datos</div>', unsafe_allow_html=True)

    n_paises = len(df)
    st.markdown(f"""
    <div class="card-row">
      <div class="metric-card"><div class="label">Países seleccionados</div>
        <div class="value">{n_paises}</div><div class="sub">de {len(df_full)} totales</div></div>
      <div class="metric-card"><div class="label">Variables de análisis</div>
        <div class="value">4</div><div class="sub">Educación · Economía</div></div>
      <div class="metric-card"><div class="label">Gasto promedio</div>
        <div class="value">{df['gasto_educacion_pib'].mean():.1f}%</div><div class="sub">del PIB</div></div>
      <div class="metric-card"><div class="label">Escolaridad promedio</div>
        <div class="value">{df['anios_escolaridad'].mean():.1f}</div><div class="sub">años</div></div>
      <div class="metric-card"><div class="label">PIB per cápita promedio</div>
        <div class="value">{df['pib_per_capita'].mean()/1000:.1f}K</div><div class="sub">USD PPA</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Estadísticos Descriptivos</div>', unsafe_allow_html=True)

    stats_rows = ["Media", "Mediana", "Desv. Estándar", "Mínimo",
                  "P25 (Q1)", "P75 (Q3)", "Máximo", "Asimetría", "Curtosis"]
    stats_data = {"Estadístico": stats_rows}
    for var, nombre in zip(variables, nombres_var):
        c = df[var]
        stats_data[nombre] = [
            f"{c.mean():.3f}", f"{c.median():.3f}", f"{c.std():.3f}", f"{c.min():.3f}",
            f"{c.quantile(0.25):.3f}", f"{c.quantile(0.75):.3f}", f"{c.max():.3f}",
            f"{c.skew():.3f}", f"{c.kurtosis():.3f}",
        ]
    render_table(pd.DataFrame(stats_data), num_cols=nombres_var, highlight_col="Estadístico")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Distribuciones Generales</div>', unsafe_allow_html=True)

    # Filtro de variable para histogramas
    var_hist = st.selectbox("Variable a visualizar:", nombres_var, key="hist_var")
    var_hist_key = VAR_MAP[var_hist]

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        nbins = st.slider("Número de bins:", 5, 60, 25, key="nbins")
        show_kde = st.checkbox("Mostrar curva KDE", value=True, key="kde_check")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df[var_hist_key], nbinsx=nbins,
        marker_color=COLOR_MAP[var_hist_key], opacity=0.68,
        marker_line=dict(color="white", width=0.5),
        name="Frecuencia", hovertemplate=f"{var_hist}: %{{x}}<br>Frecuencia: %{{y}}<extra></extra>"
    ))
    if show_kde:
        kde_x = np.linspace(df[var_hist_key].min(), df[var_hist_key].max(), 300)
        kde   = stats.gaussian_kde(df[var_hist_key].dropna())
        scale = len(df[var_hist_key]) * (df[var_hist_key].max() - df[var_hist_key].min()) / nbins
        fig_hist.add_trace(go.Scatter(
            x=kde_x, y=kde(kde_x) * scale,
            mode="lines", line=dict(color=COL_TIT, width=2),
            name="KDE", hovertemplate="KDE: %{y:.2f}<extra></extra>"
        ))
    media   = df[var_hist_key].mean()
    mediana = df[var_hist_key].median()
    fig_hist.add_vline(x=media,   line_dash="dash", line_color=COL_REF,   line_width=1.8,
                       annotation_text=f"Media: {media:.2f}",   annotation_position="top right")
    fig_hist.add_vline(x=mediana, line_dash="solid", line_color=COL_TIT, line_width=1.8,
                       annotation_text=f"Mediana: {mediana:.2f}", annotation_position="top left")
    apply_layout(fig_hist, title=f"Distribución — {var_hist}", height=400)
    fig_hist.update_xaxes(title_text=var_hist)
    fig_hist.update_yaxes(title_text="Frecuencia")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Histogramas de las 4 variables en subplots
    st.markdown('<div class="sub-title">Vista Comparativa de Todas las Variables</div>', unsafe_allow_html=True)
    fig_all = make_subplots(rows=2, cols=2,
                            subplot_titles=nombres_var,
                            vertical_spacing=0.15, horizontal_spacing=0.1)
    for i, (var, nombre, color) in enumerate(zip(variables, nombres_var, [C1, C2, C3, C4])):
        r, c = (i // 2) + 1, (i % 2) + 1
        fig_all.add_trace(
            go.Histogram(x=df[var], nbinsx=25, marker_color=color, opacity=0.65,
                         marker_line=dict(color="white", width=0.4),
                         name=nombre, showlegend=False,
                         hovertemplate=f"{nombre}: %{{x}}<br>Frecuencia: %{{y}}<extra></extra>"),
            row=r, col=c
        )
    fig_all.update_layout(**PLOTLY_BASE, height=520,
                          title="Distribución de las Cuatro Variables")
    fig_all.update_xaxes(**AXIS_STYLE)
    fig_all.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_all, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB E — EXPLORE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">Exploración Individual de Variables</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        var_exp = st.selectbox("Variable:", nombres_var, key="exp_var")
    with col_f2:
        color_by = st.selectbox("Colorear por:", ["Ninguno"] + nombres_var, key="exp_color")
    with col_f3:
        chart_type = st.selectbox("Tipo de gráfico:", ["Histograma", "Boxplot", "Violín", "Q-Q Plot"], key="exp_type")

    var_exp_key   = VAR_MAP[var_exp]
    color_exp_key = VAR_MAP[color_by] if color_by != "Ninguno" else None

    if chart_type == "Histograma":
        fig_exp = px.histogram(
            df, x=var_exp_key, nbins=25,
            color_discrete_sequence=[COLOR_MAP[var_exp_key]],
            hover_data={"pais": True, var_exp_key: True},
            labels={var_exp_key: var_exp, "pais": "País"},
        )
        fig_exp.update_traces(
            marker_line_color="white", marker_line_width=0.5, opacity=0.68,
            hovertemplate="País: %{customdata[0]}<br>" + var_exp + ": %{x:.2f}<extra></extra>"
        )
        sw_p = stats.shapiro(df[var_exp_key].dropna())[1]
        fig_exp.add_vline(x=df[var_exp_key].mean(),   line_dash="dash",  line_color=COL_REF,  line_width=1.8)
        fig_exp.add_vline(x=df[var_exp_key].median(), line_dash="solid", line_color=COL_TIT, line_width=1.8)
        apply_layout(fig_exp, title=f"Histograma — {var_exp}   |   Shapiro-Wilk p = {sw_p:.4f}", height=430)
        fig_exp.update_xaxes(title_text=var_exp)
        fig_exp.update_yaxes(title_text="Frecuencia")

    elif chart_type == "Boxplot":
        if color_exp_key:
            q33 = df[color_exp_key].quantile(0.33)
            q66 = df[color_exp_key].quantile(0.66)
            df["_grupo"] = pd.cut(df[color_exp_key],
                                  bins=[-np.inf, q33, q66, np.inf],
                                  labels=["Bajo", "Medio", "Alto"])
            fig_exp = px.box(
                df, y=var_exp_key, color="_grupo",
                color_discrete_sequence=[C1, C3, C2],
                points="outliers",
                hover_data={"pais": True},
                labels={var_exp_key: var_exp, "_grupo": color_by},
            )
            df.drop(columns=["_grupo"], inplace=True)
        else:
            fig_exp = px.box(
                df, y=var_exp_key, points="outliers",
                color_discrete_sequence=[COLOR_MAP[var_exp_key]],
                hover_data={"pais": True},
                labels={var_exp_key: var_exp},
            )
        apply_layout(fig_exp, title=f"Boxplot — {var_exp}", height=430)
        fig_exp.update_yaxes(title_text=var_exp)

    elif chart_type == "Violín":
        if color_exp_key:
            q33 = df[color_exp_key].quantile(0.33)
            q66 = df[color_exp_key].quantile(0.66)
            df["_grupo"] = pd.cut(df[color_exp_key],
                                  bins=[-np.inf, q33, q66, np.inf],
                                  labels=["Bajo", "Medio", "Alto"])
            fig_exp = px.violin(
                df, y=var_exp_key, color="_grupo",
                color_discrete_sequence=[C1, C3, C2],
                box=True, points="outliers",
                hover_data={"pais": True},
                labels={var_exp_key: var_exp, "_grupo": color_by},
            )
            df.drop(columns=["_grupo"], inplace=True)
        else:
            fig_exp = px.violin(
                df, y=var_exp_key, box=True, points="outliers",
                color_discrete_sequence=[COLOR_MAP[var_exp_key]],
                hover_data={"pais": True},
                labels={var_exp_key: var_exp},
            )
        apply_layout(fig_exp, title=f"Violín — {var_exp}", height=430)
        fig_exp.update_yaxes(title_text=var_exp)

    else:  # Q-Q Plot
        data_sorted  = np.sort(df[var_exp_key].dropna())
        n            = len(data_sorted)
        quantiles_th = stats.norm.ppf(np.linspace(0.5/n, 1 - 0.5/n, n),
                                      loc=data_sorted.mean(), scale=data_sorted.std())
        fig_exp = go.Figure()
        fig_exp.add_trace(go.Scatter(
            x=quantiles_th, y=data_sorted, mode="markers",
            marker=dict(color=COLOR_MAP[var_exp_key], size=6, opacity=0.7,
                        line=dict(color="white", width=0.4)),
            name="Datos", hovertemplate="Teórico: %{x:.2f}<br>Observado: %{y:.2f}<extra></extra>"
        ))
        mn = min(quantiles_th.min(), data_sorted.min())
        mx = max(quantiles_th.max(), data_sorted.max())
        fig_exp.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color=COL_REF, dash="dash", width=1.8),
            name="Referencia normal"
        ))
        apply_layout(fig_exp, title=f"Q-Q Plot — {var_exp}", height=430)
        fig_exp.update_xaxes(title_text="Cuantiles teóricos (normal)")
        fig_exp.update_yaxes(title_text="Cuantiles observados")

    st.plotly_chart(fig_exp, use_container_width=True)

    # Boxplots comparativos
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Boxplots Comparativos — Todas las Variables</div>', unsafe_allow_html=True)

    fig_box = make_subplots(rows=1, cols=4, subplot_titles=nombres_var,
                            horizontal_spacing=0.06)
    for i, (var, nombre, color) in enumerate(zip(variables, nombres_var, [C1, C2, C3, C4])):
        fig_box.add_trace(
            go.Box(y=df[var], name=nombre, marker_color=color, opacity=0.7,
                   boxmean=True, showlegend=False,
                   hovertemplate=nombre + ": %{y:.2f}<extra></extra>"),
            row=1, col=i+1
        )
    fig_box.update_layout(**PLOTLY_BASE, height=430, title="Distribución y Valores Atípicos")
    fig_box.update_xaxes(**AXIS_STYLE)
    fig_box.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_box, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB S — STUDY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">Relaciones, Correlaciones y Modelos</div>', unsafe_allow_html=True)

    # ── Heatmap de correlación ──
    st.markdown('<div class="sub-title">Matriz de Correlación</div>', unsafe_allow_html=True)

    corr_method = st.radio("Método:", ["Pearson", "Spearman"], horizontal=True, key="corr_method")
    corr_mat    = df[variables].corr(method=corr_method.lower()).round(3)

    fig_heat = px.imshow(
        corr_mat,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        labels=dict(color="Correlación"),
        x=nombres_var, y=nombres_var,
    )
    fig_heat.update_traces(textfont_size=11)
    apply_layout(fig_heat, title=f"Correlación de {corr_method}", height=420)
    fig_heat.update_coloraxes(colorbar=dict(thickness=14, len=0.85,
                                            tickfont=dict(size=10)))
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Scatter interactivo ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Diagrama de Dispersión Interactivo</div>', unsafe_allow_html=True)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        eje_x = st.selectbox("Eje X:", nombres_var, index=0, key="sc_x")
    with col_s2:
        eje_y = st.selectbox("Eje Y:", nombres_var, index=1, key="sc_y")
    with col_s3:
        color_sc = st.selectbox("Color:", ["Ninguno"] + nombres_var, key="sc_color")
    with col_s4:
        show_trendline = st.checkbox("Línea de tendencia", value=True, key="sc_trend")

    x_key = VAR_MAP[eje_x]
    y_key = VAR_MAP[eje_y]
    c_key = VAR_MAP[color_sc] if color_sc != "Ninguno" else None

    scatter_df = df.copy()
    x_label = eje_x

    fig_sc = px.scatter(
        scatter_df,
        x=x_key, y=y_key,
        color=c_key,
        hover_name="pais",
        hover_data={x_key: ":.2f", y_key: ":.2f"},
        color_continuous_scale="RdBu_r" if c_key else None,
        color_discrete_sequence=PALETTE,
        trendline="ols" if show_trendline else None,
        trendline_color_override=COL_REF,
        opacity=0.72,
        labels={x_key: x_label, y_key: eje_y, c_key: color_sc if c_key else ""},
    )
    fig_sc.update_traces(
        marker=dict(size=7, line=dict(color="white", width=0.5)),
        selector=dict(mode="markers")
    )
    r_val, p_val = stats.pearsonr(scatter_df[x_key].dropna(), scatter_df[y_key].dropna())
    apply_layout(fig_sc, title=f"{x_label} vs {eje_y}   |   r = {r_val:.3f}   p = {p_val:.4f}", height=460)
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Pairplot interactivo ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Matriz de Dispersión (Pairplot)</div>', unsafe_allow_html=True)

    vars_pair = st.multiselect(
        "Variables a incluir:",
        nombres_var, default=nombres_var, key="pair_vars"
    )
    if len(vars_pair) >= 2:
        keys_pair  = [VAR_MAP[v] for v in vars_pair]
        # Etiquetas cortas para evitar solapamiento en los ejes
        short_labels = {
            "gasto_educacion_pib":  "Gasto Educ.<br>(% PIB)",
            "anios_escolaridad":    "Años<br>Escolaridad",
            "pib_per_capita":       "PIB per<br>cápita (USD)",
            "tasa_alfabetizacion":  "Alfabet.<br>(%)",
        }
        labels_pair = {k: short_labels.get(k, k) for k in keys_pair}

        fig_pair = px.scatter_matrix(
            df, dimensions=keys_pair,
            hover_name="pais",
            color_discrete_sequence=[C1],
            opacity=0.55,
            labels=labels_pair,
        )
        fig_pair.update_traces(
            marker=dict(size=4, line=dict(color="white", width=0.3)),
            diagonal_visible=True,
        )

        n_vars = len(keys_pair)
        altura = max(600, n_vars * 170)

        fig_pair.update_layout(
            **PLOTLY_BASE,
            title="Matriz de Dispersión",
            height=altura,
            margin=dict(l=110, r=40, t=60, b=110),
        )
        fig_pair.update_xaxes(
            **AXIS_STYLE,
            tickangle=30,
            tickfont=dict(size=9),
            title_font=dict(size=10),
            title_standoff=18,
        )
        fig_pair.update_yaxes(
            **AXIS_STYLE,
            tickfont=dict(size=9),
            title_font=dict(size=10),
            title_standoff=18,
        )
        st.plotly_chart(fig_pair, use_container_width=True)
    else:
        st.info("Selecciona al menos 2 variables para ver la matriz de dispersión.")

    # ── Modelos ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Modelo 1 — Regresión Lineal Simple</div>', unsafe_allow_html=True)

    m1   = res["m1"]
    y1te = np.array(res["y1te"])
    y1p  = np.array(res["y1p"])

    col_m1a, col_m1b, col_m1c = st.columns(3)

    with col_m1a:
        xl    = np.linspace(df["gasto_educacion_pib"].min(), df["gasto_educacion_pib"].max(), 200).reshape(-1, 1)
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=df["gasto_educacion_pib"], y=df["anios_escolaridad"],
            mode="markers", name="Países",
            marker=dict(color=C1, size=6, opacity=0.65, line=dict(color="white", width=0.4)),
            text=df["pais"], hovertemplate="País: %{text}<br>Gasto: %{x:.2f}%<br>Escolaridad: %{y:.2f}<extra></extra>"
        ))
        fig_r.add_trace(go.Scatter(
            x=xl.flatten(), y=m1.predict(xl).flatten(),
            mode="lines", name=f"Regresión (R²={res['r2_1']:.3f})",
            line=dict(color=COL_REF, width=2, dash="dash")
        ))
        apply_layout(fig_r, title="Línea de Regresión", height=350)
        fig_r.update_xaxes(title_text="Gasto Educ. (% PIB)")
        fig_r.update_yaxes(title_text="Años de Escolaridad")
        st.plotly_chart(fig_r, use_container_width=True)

    with col_m1b:
        fig_rv = go.Figure()
        fig_rv.add_trace(go.Scatter(
            x=y1te, y=y1p, mode="markers", name="Test",
            marker=dict(color=C2, size=6, opacity=0.65, line=dict(color="white", width=0.4)),
            hovertemplate="Real: %{x:.2f}<br>Predicho: %{y:.2f}<extra></extra>"
        ))
        lim = [min(y1te.min(), y1p.min())-0.5, max(y1te.max(), y1p.max())+0.5]
        fig_rv.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Referencia",
                                    line=dict(color=COL_REF, dash="dash", width=1.6)))
        apply_layout(fig_rv, title="Real vs Predicho", height=350)
        fig_rv.update_xaxes(title_text="Valores Reales")
        fig_rv.update_yaxes(title_text="Valores Predichos")
        st.plotly_chart(fig_rv, use_container_width=True)

    with col_m1c:
        res1 = y1te - y1p
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=y1p, y=res1, mode="markers", name="Residuos",
            marker=dict(color=C3, size=6, opacity=0.65, line=dict(color="white", width=0.4)),
            hovertemplate="Predicho: %{x:.2f}<br>Residuo: %{y:.2f}<extra></extra>"
        ))
        fig_res.add_hline(y=0, line_dash="dash", line_color=COL_REF, line_width=1.6)
        apply_layout(fig_res, title="Análisis de Residuos", height=350)
        fig_res.update_xaxes(title_text="Valores Predichos")
        fig_res.update_yaxes(title_text="Residuos")
        st.plotly_chart(fig_res, use_container_width=True)

    st.markdown('<div class="sub-title">Modelo 2 — Regresión Lineal Múltiple</div>', unsafe_allow_html=True)

    y2te = np.array(res["y2te"])
    y2p  = np.array(res["y2p"])
    res2 = y2te - y2p

    col_m2a, col_m2b, col_m2c = st.columns(3)

    with col_m2a:
        fig_rv2 = go.Figure()
        fig_rv2.add_trace(go.Scatter(
            x=y2te, y=y2p, mode="markers", name="Test",
            marker=dict(color=C2, size=6, opacity=0.65, line=dict(color="white", width=0.4)),
            hovertemplate="Real: %{x:.2f}<br>Predicho: %{y:.2f}<extra></extra>"
        ))
        lim2 = [min(y2te.min(), y2p.min())-0.5, max(y2te.max(), y2p.max())+0.5]
        fig_rv2.add_trace(go.Scatter(x=lim2, y=lim2, mode="lines", name="Referencia",
                                     line=dict(color=COL_REF, dash="dash", width=1.6)))
        apply_layout(fig_rv2, title=f"Real vs Predicho — R² = {res['r2_2']:.3f}", height=350)
        fig_rv2.update_xaxes(title_text="Valores Reales")
        fig_rv2.update_yaxes(title_text="Valores Predichos")
        st.plotly_chart(fig_rv2, use_container_width=True)

    with col_m2b:
        fig_res2 = go.Figure()
        fig_res2.add_trace(go.Scatter(
            x=y2p, y=res2, mode="markers", name="Residuos",
            marker=dict(color=C3, size=6, opacity=0.65, line=dict(color="white", width=0.4)),
            hovertemplate="Predicho: %{x:.2f}<br>Residuo: %{y:.2f}<extra></extra>"
        ))
        fig_res2.add_hline(y=0, line_dash="dash", line_color=COL_REF, line_width=1.6)
        apply_layout(fig_res2, title="Análisis de Residuos", height=350)
        fig_res2.update_xaxes(title_text="Valores Predichos")
        fig_res2.update_yaxes(title_text="Residuos")
        st.plotly_chart(fig_res2, use_container_width=True)

    with col_m2c:
        kde_x   = np.linspace(min(res2) - 0.5, max(res2) + 0.5, 300)
        kde_val = stats.gaussian_kde(res2)(kde_x)
        sw_p    = stats.shapiro(res2)[1]
        fig_rd  = go.Figure()
        fig_rd.add_trace(go.Histogram(
            x=res2, nbinsx=20, histnorm="probability density",
            marker_color=C4, opacity=0.60,
            marker_line=dict(color="white", width=0.4), name="Residuos"
        ))
        fig_rd.add_trace(go.Scatter(
            x=kde_x, y=kde_val, mode="lines",
            line=dict(color=COL_TIT, width=2), name="KDE"
        ))
        fig_rd.add_vline(x=0, line_dash="dash", line_color=COL_REF, line_width=1.6)
        apply_layout(fig_rd, title=f"Distribución de Residuos  |  Shapiro-Wilk p = {sw_p:.4f}", height=350)
        fig_rd.update_xaxes(title_text="Residuos")
        fig_rd.update_yaxes(title_text="Densidad")
        st.plotly_chart(fig_rd, use_container_width=True)

    # ── Comparación de modelos ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comparación de Modelos</div>', unsafe_allow_html=True)

    render_table(pd.DataFrame({
        "Métrica": ["R² (test)", "R² Ajustado", "RMSE (test)", "MAE (test)",
                    "AIC", "BIC", "R² CV 5-fold", "Predictoras"],
        "Modelo 1 (Simple)": [
            f"{res['r2_1']:.4f}", f"{res['sm1'].rsquared_adj:.4f}",
            f"{res['rmse_1']:.4f}", f"{res['mae_1']:.4f}",
            f"{res['sm1'].aic:.2f}", f"{res['sm1'].bic:.2f}",
            f"{res['cv1'].mean():.4f} ± {res['cv1'].std():.4f}", "1"],
        "Modelo 2 (Múltiple)": [
            f"{res['r2_2']:.4f}", f"{res['sm2'].rsquared_adj:.4f}",
            f"{res['rmse_2']:.4f}", f"{res['mae_2']:.4f}",
            f"{res['sm2'].aic:.2f}", f"{res['sm2'].bic:.2f}",
            f"{res['cv2'].mean():.4f} ± {res['cv2'].std():.4f}", "3"],
    }), num_cols=["Modelo 1 (Simple)", "Modelo 2 (Múltiple)"], highlight_col="Métrica")

    col_c1, col_c2, col_c3 = st.columns(3)
    metricas = [
        ([res["r2_1"], res["r2_2"]],             "R²",   "Coef. Determinación (mayor → mejor)"),
        ([res["rmse_1"], res["rmse_2"]],          "RMSE", "Error Cuadrático Medio (menor → mejor)"),
        ([res["cv1"].mean(), res["cv2"].mean()],  "R² CV","Validación Cruzada 5-fold (mayor → mejor)"),
    ]
    for col, (vals, ylabel, titulo) in zip([col_c1, col_c2, col_c3], metricas):
        with col:
            fig_cmp = go.Figure(go.Bar(
                x=["Modelo 1\n(Simple)", "Modelo 2\n(Múltiple)"],
                y=vals, marker_color=[C1, C2], opacity=0.78,
                marker_line=dict(color="white", width=0.5),
                text=[f"{v:.4f}" for v in vals], textposition="outside",
                hovertemplate="%{x}<br>" + ylabel + ": %{y:.4f}<extra></extra>"
            ))
            apply_layout(fig_cmp, title=titulo, height=340)
            fig_cmp.update_yaxes(title_text=ylabel,
                                  range=[0, max(vals) * 1.35])
            st.plotly_chart(fig_cmp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB T — TELL
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">Síntesis, Conclusiones y Respuestas</div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-title">Panel Resumen del Análisis</div>', unsafe_allow_html=True)

    vars_corr  = ["gasto_educacion_pib", "pib_per_capita", "tasa_alfabetizacion", "log_pib_per_capita"]
    names_corr = ["Gasto Educ. (% PIB)", "PIB per cápita", "Tasa Alfab.", "log(PIB per cápita)"]
    correlaciones = [df[v].corr(df["anios_escolaridad"]) for v in vars_corr]

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        fig_corr_bar = go.Figure(go.Bar(
            x=correlaciones, y=names_corr, orientation="h",
            marker_color=[C1 if c > 0 else C2 for c in correlaciones], opacity=0.75,
            marker_line=dict(color="white", width=0.5),
            text=[f"{v:.3f}" for v in correlaciones], textposition="outside",
            hovertemplate="%{y}<br>r = %{x:.3f}<extra></extra>"
        ))
        fig_corr_bar.add_vline(x=0, line_color="#7A6E65", line_width=0.8)
        apply_layout(fig_corr_bar, title="Correlaciones con Años de Escolaridad", height=340)
        fig_corr_bar.update_xaxes(title_text="Correlación de Pearson")
        st.plotly_chart(fig_corr_bar, use_container_width=True)

    with col_t2:
        r2_adj = [res["sm1"].rsquared_adj, res["sm2"].rsquared_adj]
        fig_r2 = go.Figure(go.Bar(
            x=["Modelo 1\n(Simple)", "Modelo 2\n(Múltiple)"],
            y=r2_adj, marker_color=[C1, C2], opacity=0.75,
            marker_line=dict(color="white", width=0.5),
            text=[f"{v:.4f}" for v in r2_adj], textposition="outside",
            hovertemplate="%{x}<br>R² Adj = %{y:.4f}<extra></extra>"
        ))
        apply_layout(fig_r2, title="R² Ajustado — Comparación de Modelos", height=340)
        fig_r2.update_yaxes(title_text="R² Ajustado", range=[0, max(r2_adj)*1.4])
        st.plotly_chart(fig_r2, use_container_width=True)

    col_t3, col_t4 = st.columns(2)

    with col_t3:
        y_all   = res["m2"].predict(df[res["vars2"]].values)
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=df["anios_escolaridad"], y=y_all, mode="markers",
            text=df["pais"],
            marker=dict(color=C2, size=6, opacity=0.58, line=dict(color="white", width=0.4)),
            hovertemplate="País: %{text}<br>Real: %{x:.2f}<br>Predicho: %{y:.2f}<extra></extra>"
        ))
        lim_all = [min(df["anios_escolaridad"].min(), y_all.min())-0.5,
                   max(df["anios_escolaridad"].max(), y_all.max())+0.5]
        fig_pred.add_trace(go.Scatter(x=lim_all, y=lim_all, mode="lines",
                                      line=dict(color=COL_REF, dash="dash", width=1.6),
                                      name="Referencia"))
        apply_layout(fig_pred, title="Predicción del Mejor Modelo (todos los países)", height=340)
        fig_pred.update_xaxes(title_text="Valores Reales")
        fig_pred.update_yaxes(title_text="Predichos (Modelo 2)")
        st.plotly_chart(fig_pred, use_container_width=True)

    with col_t4:
        imp_names  = [x[0] for x in res["importancia"]]
        imp_vals   = [x[1] for x in res["importancia"]]
        fig_imp = go.Figure(go.Bar(
            x=[abs(v) for v in imp_vals], y=imp_names, orientation="h",
            marker_color=[C2 if v < 0 else C1 for v in imp_vals], opacity=0.75,
            marker_line=dict(color="white", width=0.5),
            text=[f"{v:.3f}" for v in imp_vals], textposition="outside",
            hovertemplate="%{y}<br>Coef. estand. = %{text}<extra></extra>"
        ))
        apply_layout(fig_imp, title="Importancia de Variables (Modelo 2)", height=340)
        fig_imp.update_xaxes(title_text="|Coeficiente Estandarizado|")
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Respuestas ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Respuestas a las Preguntas de Investigación</div>', unsafe_allow_html=True)

    mejora_r2   = ((res["r2_2"] - res["r2_1"]) / abs(res["r2_1"])) * 100 if res["r2_1"] != 0 else 0
    mejora_rmse = ((res["rmse_1"] - res["rmse_2"]) / res["rmse_1"]) * 100

    respuestas = [
        ("P1", "Relación lineal gasto — escolaridad",
         f"La relación es débil a moderada. El Modelo 1 alcanza un R² ajustado de "
         f"{res['sm1'].rsquared_adj:.4f}, indicando que el gasto como porcentaje del PIB "
         f"no es suficiente por sí solo para explicar la variabilidad en los años de escolaridad."),
        ("P2", "PIB per cápita vs gasto en educación",
         f"Sí. El PIB per cápita (en escala logarítmica) presenta una correlación "
         f"considerablemente más fuerte con la escolaridad. Los coeficientes "
         f"estandarizados del Modelo 2 confirman que el desarrollo económico es el predictor dominante."),
        ("P3", "Correlación tasa de alfabetización",
         f"La tasa de alfabetización muestra una correlación positiva significativa "
         f"(r = {df['tasa_alfabetizacion'].corr(df['anios_escolaridad']):.3f}) con la escolaridad. "
         f"Sin embargo, la correlación con log(PIB) tiende a ser mayor, "
         f"confirmando el rol preponderante del contexto económico."),
        ("P4", "Modelo múltiple vs modelo simple",
         f"Sí, significativamente. El Modelo 2 supera al Modelo 1 en todas las métricas: "
         f"R² mejora de {res['r2_1']:.4f} a {res['r2_2']:.4f} "
         f"(+{mejora_r2:.1f}%), y el RMSE se reduce un {mejora_rmse:.1f}%."),
    ]
    resp_colors = [C1, C3, C2, C4]
    cols = st.columns(2)
    for i, (code, titulo, texto) in enumerate(respuestas):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:#fff;border-radius:12px;padding:1.1rem 1.35rem;
                        border-left:4px solid {resp_colors[i]};
                        box-shadow:0 1px 8px rgba(0,0,0,.06);margin-bottom:1rem;min-height:118px;">
              <span style="font-family:'Playfair Display',serif;font-size:.70rem;
                           text-transform:uppercase;letter-spacing:.1em;
                           color:{resp_colors[i]};font-weight:700;">{code}</span>
              <div style="font-family:'Playfair Display',serif;font-weight:700;
                          font-size:.96rem;color:#2E4A6B;margin:.28rem 0 .45rem;">{titulo}</div>
              <p style="margin:0;color:#4A3E33;font-size:.89rem;line-height:1.55;">{texto}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Tabla resumen ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Tabla Resumen de Resultados</div>', unsafe_allow_html=True)

    mejor_modelo = "Modelo 2 (Regresión Múltiple)" if res["r2_2"] > res["r2_1"] else "Modelo 1 (Regresión Simple)"
    render_table(pd.DataFrame({
        "Aspecto": [
            "Variable dependiente", "R² Ajustado — Modelo 1", "R² Ajustado — Modelo 2",
            "RMSE — Modelo 1", "RMSE — Modelo 2",
            "Variable más influyente", "Mejora Modelo 2 vs 1", "Mejor modelo",
        ],
        "Resultado": [
            "Años promedio de escolaridad",
            f"{res['sm1'].rsquared_adj:.4f}", f"{res['sm2'].rsquared_adj:.4f}",
            f"{res['rmse_1']:.4f}", f"{res['rmse_2']:.4f}",
            res["importancia"][0][0],
            f"{mejora_r2:+.1f}% en R²", mejor_modelo,
        ],
    }), highlight_col="Aspecto")

    st.markdown("""
    <div style="text-align:center;margin-top:2.2rem;color:#A89880;font-size:.79rem;letter-spacing:.03em;line-height:2;">
      Fuente de datos: <strong>Banco Mundial</strong> — World Bank Open Data (data.worldbank.org)
      &nbsp;&nbsp;·&nbsp;&nbsp;
      Framework de análisis: <strong>QUEST</strong>
      (Question &rarr; Understand &rarr; Explore &rarr; Study &rarr; Tell)
    </div>
    """, unsafe_allow_html=True)
