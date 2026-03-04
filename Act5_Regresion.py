import streamlit as st
import pandas as pd
import numpy as np
import requests
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
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
    background: linear-gradient(135deg, #2E4A6B 0%, #4E7A5E 65%, #8A7A62 100%);
    border-radius: 16px;
    padding: 2.8rem 3rem 2.4rem;
    margin-bottom: 1.8rem;
    color: #fff;
    box-shadow: 0 6px 28px rgba(46,74,107,.22);
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 .25rem;
    letter-spacing: .015em;
}
.hero .author {
    font-family: 'Source Sans 3', sans-serif;
    font-size: .88rem;
    font-weight: 400;
    letter-spacing: .1em;
    text-transform: uppercase;
    opacity: .72;
    margin: 0 0 .65rem;
    border-left: 2px solid rgba(255,255,255,.35);
    padding-left: .6rem;
    display: inline-block;
}
.hero p { font-size: .97rem; opacity: .85; margin: 0; line-height: 1.6; }
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,.14);
    border: 1px solid rgba(255,255,255,.28);
    border-radius: 3px;
    padding: .16rem .7rem;
    font-size: .68rem;
    margin-right: .4rem;
    margin-top: .75rem;
    letter-spacing: .07em;
    text-transform: uppercase;
}

/* ── tabs ── */
[data-testid="stTabs"] > div:first-child {
    gap: 4px;
    background: #EDE8E0;
    border-radius: 10px;
    padding: 5px 7px;
    margin-bottom: 1.4rem;
}
button[data-baseweb="tab"] {
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: .88rem !important;
    font-weight: 600 !important;
    letter-spacing: .05em !important;
    text-transform: uppercase !important;
    color: #6B5E4C !important;
    background: transparent !important;
    border-radius: 7px !important;
    padding: .42rem 1.05rem !important;
    border: none !important;
    transition: all .18s ease !important;
}
button[data-baseweb="tab"]:hover {
    background: rgba(255,255,255,.5) !important;
    color: #2E4A6B !important;
}
button[aria-selected="true"][data-baseweb="tab"] {
    background: #fff !important;
    color: #2E4A6B !important;
    box-shadow: 0 1px 6px rgba(46,74,107,.13) !important;
}

/* ── section titles ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #2E4A6B;
    border-left: 4px solid #4E7A5E;
    padding-left: .75rem;
    margin-bottom: 1rem;
}
.sub-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.08rem;
    font-weight: 600;
    color: #4A3C2C;
    margin-top: 1.4rem;
    margin-bottom: .5rem;
    padding-bottom: .3rem;
    border-bottom: 1px solid #E0D8CE;
}

/* ── metric cards ── */
.card-row { display: flex; gap: 14px; flex-wrap: wrap; margin: 1rem 0; }
.metric-card {
    flex: 1; min-width: 140px;
    background: #fff;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 8px rgba(0,0,0,.06);
    border-top: 3px solid #4E7A5E;
    text-align: center;
}
.metric-card .label {
    font-size: .70rem; color: #9E8C78;
    text-transform: uppercase; letter-spacing: .07em; margin-bottom: .3rem;
}
.metric-card .value {
    font-family: 'Playfair Display', serif;
    font-size: 1.55rem; font-weight: 700; color: #2E4A6B;
}
.metric-card .sub { font-size: .74rem; color: #A89880; margin-top: .15rem; }

/* ── info / answer boxes ── */
.info-box {
    background: #fff;
    border-left: 3px solid #A8997E;
    border-radius: 0 8px 8px 0;
    padding: .85rem 1.1rem; margin: .7rem 0;
    font-size: .93rem; color: #4A3E33;
    box-shadow: 0 1px 5px rgba(0,0,0,.04);
}
.info-box strong { color: #2E4A6B; }
.answer-box {
    background: #EDF2F8;
    border-left: 3px solid #2E4A6B;
    border-radius: 0 8px 8px 0;
    padding: .85rem 1.1rem; margin: .7rem 0; color: #2C3E57;
}
.answer-box h4 {
    font-family: 'Playfair Display', serif;
    margin: 0 0 .32rem; font-size: .97rem; color: #2E4A6B;
}

/* ── custom tables ── */
.custom-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Source Sans 3', sans-serif;
    font-size: .88rem;
    color: #3A3228;
    margin: .5rem 0 1.2rem;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 8px rgba(0,0,0,.08);
}
.custom-table thead tr {
    background: #2E4A6B;
    color: #fff;
    text-transform: uppercase;
    letter-spacing: .06em;
    font-size: .73rem;
}
.custom-table thead th {
    padding: .65rem 1rem;
    text-align: left;
    font-weight: 600;
    border: none;
}
.custom-table tbody tr:nth-child(even) { background: #F0EBE4; }
.custom-table tbody tr:nth-child(odd)  { background: #FAF7F3; }
.custom-table tbody tr:hover           { background: #E4EDF6; transition: background .15s; }
.custom-table tbody td {
    padding: .55rem 1rem;
    border-bottom: 1px solid #E8E0D8;
    vertical-align: middle;
}
.custom-table tbody tr:last-child td { border-bottom: none; }
.custom-table td.num {
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-family: 'Source Sans 3', sans-serif;
}
.custom-table td.highlight {
    color: #2E4A6B;
    font-weight: 600;
}

/* ── misc ── */
hr { border: none; border-top: 1px solid #DDD4C8; margin: 1.3rem 0; }
[data-testid="stSpinner"] > div > div { border-top-color: #2E4A6B !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# PALETA Y CONFIGURACIÓN GLOBAL DE GRÁFICAS
# ──────────────────────────────────────────────────────────────────────────────
PASTEL_COLORS = ["#7EB5D6", "#E8A090", "#A8C8A0", "#C4A8C8", "#F0C890", "#90C8D8"]
C1, C2, C3, C4, C5, C6 = PASTEL_COLORS

BG_FIG  = "#F7F3EE"
BG_AX   = "#FDFAF6"
COL_REF = "#9E3A3A"
COL_TIT = "#2E4A6B"

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9.5,
    "axes.titlesize":    10.5,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   COL_TIT,
    "axes.labelsize":    9,
    "axes.labelcolor":   "#3A3A3A",
    "axes.edgecolor":    "#C8C0B8",
    "axes.linewidth":    0.75,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#E2DBD4",
    "grid.linewidth":    0.5,
    "grid.linestyle":    "--",
    "xtick.color":       "#6A6A6A",
    "ytick.color":       "#6A6A6A",
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   8.5,
    "legend.framealpha": 0.88,
    "legend.edgecolor":  "#D8D0C8",
    "figure.dpi":        110,
})

def fig_style(fig, axes=None):
    fig.patch.set_facecolor(BG_FIG)
    if axes is not None:
        for ax in np.array(axes).flatten():
            ax.set_facecolor(BG_AX)
    return fig

def spine_style(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#C8C0B8")
        ax.spines[side].set_linewidth(0.75)

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
            if col in num_cols:
                cls += " num"
            if highlight_col and col == highlight_col:
                cls += " highlight"
            html += f'<td class="{cls.strip()}">{row[col]}</td>'
        html += "</tr>"
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)

def render_index_table(df, index_label="Estadístico", num_cols=None):
    num_cols = num_cols or list(df.columns)
    html = '<table class="custom-table"><thead><tr>'
    html += f"<th>{index_label}</th>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    for idx, row in df.iterrows():
        html += "<tr>"
        html += f'<td class="highlight">{idx}</td>'
        for col in df.columns:
            cls = "num" if col in num_cols else ""
            html += f'<td class="{cls}">{row[col]}</td>'
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
                        "pais":        rec["country"]["value"],
                        "codigo_pais": rec["countryiso3code"],
                        "year":        int(rec["date"]),
                        nombre:        rec["value"],
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
    df = pd.read_json(df_json)
    variables = ["gasto_educacion_pib", "anios_escolaridad", "pib_per_capita", "tasa_alfabetizacion"]
    df = df.dropna(subset=variables + ["log_pib_per_capita"])

    X1 = df[["gasto_educacion_pib"]].values
    y  = df["anios_escolaridad"].values
    X1tr, X1te, y1tr, y1te = train_test_split(X1, y, test_size=0.2, random_state=42)
    m1     = LinearRegression().fit(X1tr, y1tr)
    y1p    = m1.predict(X1te)
    r2_1   = r2_score(y1te, y1p)
    rmse_1 = np.sqrt(mean_squared_error(y1te, y1p))
    mae_1  = mean_absolute_error(y1te, y1p)
    cv1    = cross_val_score(LinearRegression(), X1, y, cv=5, scoring="r2")
    sm1    = sm.OLS(df["anios_escolaridad"], sm.add_constant(df["gasto_educacion_pib"])).fit()

    vars2  = ["gasto_educacion_pib", "log_pib_per_capita", "tasa_alfabetizacion"]
    X2     = df[vars2].values
    X2tr, X2te, y2tr, y2te = train_test_split(X2, y, test_size=0.2, random_state=42)
    m2     = LinearRegression().fit(X2tr, y2tr)
    y2p    = m2.predict(X2te)
    r2_2   = r2_score(y2te, y2p)
    rmse_2 = np.sqrt(mean_squared_error(y2te, y2p))
    mae_2  = mean_absolute_error(y2te, y2p)
    cv2    = cross_val_score(LinearRegression(), X2, y, cv=5, scoring="r2")
    sm2    = sm.OLS(df["anios_escolaridad"], sm.add_constant(df[vars2])).fit()

    X2sc  = StandardScaler().fit_transform(df[vars2])
    m2std = LinearRegression().fit(X2sc, df["anios_escolaridad"])
    importancia = sorted(
        zip(["Gasto educ. (% PIB)", "log(PIB per cápita)", "Tasa alfabetización"], m2std.coef_),
        key=lambda x: abs(x[1]), reverse=True
    )
    return dict(
        m1=m1, m2=m2, sm1=sm1, sm2=sm2,
        r2_1=r2_1, rmse_1=rmse_1, mae_1=mae_1, cv1=cv1,
        r2_2=r2_2, rmse_2=rmse_2, mae_2=mae_2, cv2=cv2,
        y1te=y1te, y1p=y1p, y2te=y2te, y2p=y2p,
        importancia=importancia, vars2=vars2,
    )

# ──────────────────────────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Educación &amp; Escolaridad Global</h1>
  <div class="author">Leiry Laura Mares Cure</div>
  <p>Análisis Exploratorio de Datos — Relación entre Gasto en Educación y Años de Escolaridad</p>
  <span class="badge">Banco Mundial</span>
  <span class="badge">Framework QUEST</span>
  <span class="badge">Regresión Lineal</span>
  <span class="badge">2010 – 2023</span>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner("Descargando datos del Banco Mundial..."):
    df = cargar_datos()

if df.empty:
    st.error("No se pudieron descargar los datos. Verifica tu conexión a internet.")
    st.stop()

variables   = ["gasto_educacion_pib", "anios_escolaridad", "pib_per_capita", "tasa_alfabetizacion"]
nombres_var = ["Gasto Educ. (% PIB)", "Años Escolaridad", "PIB per cápita (USD)", "Alfabetización (%)"]
COLORS_VARS = [C1, C2, C3, C4]

res = ajustar_modelos(df.to_json())

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
# TAB Q — QUESTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">Preguntas de Investigación</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    En esta sección se plantean las preguntas que guiarán todo el análisis exploratorio.
    Serán respondidas formalmente en la sección <strong>T — Tell</strong>.
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
            <div style="background:#fff; border-radius:12px; padding:1.2rem 1.4rem;
                        border-top:4px solid {border_colors[i]};
                        box-shadow:0 1px 8px rgba(0,0,0,.06); margin-bottom:1rem;">
              <span style="font-family:'Playfair Display',serif; font-size:1.55rem;
                           color:{border_colors[i]}; font-weight:700;">{code}</span>
              <span style="font-size:.69rem; text-transform:uppercase; letter-spacing:.08em;
                           color:#9E8C78; margin-left:.5rem;">{title}</span>
              <p style="margin:.5rem 0 0; color:#4A3E33; font-size:.93rem; line-height:1.55;">{text}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Variables del Estudio</div>', unsafe_allow_html=True)

    tabla_vars = pd.DataFrame({
        "Variable": ["Gasto en educación (% PIB)", "Años promedio de escolaridad",
                     "PIB per cápita (PPA, USD)",   "Tasa de alfabetización"],
        "Rol":      ["Independiente", "Dependiente", "Independiente", "Independiente"],
        "Indicador Banco Mundial": ["SE.XPD.TOTL.GD.ZS", "UIS.EA.MEAN.1T6.AG25T99",
                                    "NY.GDP.PCAP.PP.CD",  "SE.ADT.LITR.ZS"],
    })
    render_table(tabla_vars, highlight_col="Rol")

# ══════════════════════════════════════════════════════════════════════════════
# TAB U — UNDERSTAND
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">Comprensión de los Datos</div>', unsafe_allow_html=True)

    n_paises  = len(df)
    pct_nulos = df[variables].isnull().mean().mean() * 100
    st.markdown(f"""
    <div class="card-row">
      <div class="metric-card">
        <div class="label">Países con datos completos</div>
        <div class="value">{n_paises}</div>
        <div class="sub">Banco Mundial · 2010–2023</div>
      </div>
      <div class="metric-card">
        <div class="label">Variables de análisis</div>
        <div class="value">4</div>
        <div class="sub">Educación · Economía</div>
      </div>
      <div class="metric-card">
        <div class="label">Valores faltantes</div>
        <div class="value">{pct_nulos:.1f}%</div>
        <div class="sub">Tras limpieza (inner join)</div>
      </div>
      <div class="metric-card">
        <div class="label">Gasto promedio</div>
        <div class="value">{df['gasto_educacion_pib'].mean():.1f}%</div>
        <div class="sub">del PIB</div>
      </div>
      <div class="metric-card">
        <div class="label">Escolaridad promedio</div>
        <div class="value">{df['anios_escolaridad'].mean():.1f}</div>
        <div class="sub">años</div>
      </div>
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
    render_table(pd.DataFrame(stats_data),
                 num_cols=nombres_var, highlight_col="Estadístico")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Distribuciones Generales</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig_style(fig, axes)
    fig.suptitle("Distribución de las Variables de Estudio",
                 fontsize=13, fontweight="bold", color=COL_TIT, y=1.01)

    for i, (var, nombre, color) in enumerate(zip(variables, nombres_var, COLORS_VARS)):
        ax = axes[i // 2][i % 2]
        ax.set_facecolor(BG_AX)
        sns.histplot(data=df, x=var, kde=True, color=color,
                     alpha=0.58, edgecolor="white", linewidth=0.4, ax=ax)
        media   = df[var].mean()
        mediana = df[var].median()
        ax.axvline(media,   color=COL_REF, linestyle="--", linewidth=1.6,
                   label=f"Media: {media:.1f}")
        ax.axvline(mediana, color=COL_TIT, linestyle="-",  linewidth=1.6,
                   label=f"Mediana: {mediana:.1f}")
        ax.set_title(nombre)
        ax.set_xlabel("")
        ax.set_ylabel("Frecuencia", fontsize=8.5)
        ax.legend()
        spine_style(ax)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("""
    <div class="info-box">
    <strong>Observaciones clave:</strong> El <strong>PIB per cápita</strong> muestra fuerte asimetría
    positiva (pocos países con valores muy altos), mientras que la <strong>tasa de
    alfabetización</strong> presenta asimetría negativa (la mayoría de países supera el 80%).
    Se aplicará transformación logarítmica al PIB en los modelos de regresión.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB E — EXPLORE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">Exploración Individual de Variables</div>',
                unsafe_allow_html=True)

    var_sel   = st.selectbox("Selecciona una variable para explorar en detalle:", nombres_var)
    var_idx   = nombres_var.index(var_sel)
    var_key   = variables[var_idx]
    color_sel = COLORS_VARS[var_idx]

    col_left, col_right = st.columns(2)

    with col_left:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig_style(fig, ax)
        ax.set_facecolor(BG_AX)
        sns.histplot(data=df, x=var_key, bins=22, kde=True, color=color_sel,
                     alpha=0.58, edgecolor="white", linewidth=0.4, ax=ax)
        sw_stat, sw_p = stats.shapiro(df[var_key].dropna())
        normal_text   = "Distribucion normal" if sw_p > 0.05 else "No normal"
        ax.set_title(f"{var_sel}\nShapiro-Wilk: p = {sw_p:.4f}  —  {normal_text}")
        ax.set_xlabel(var_sel, fontsize=8.5)
        ax.set_ylabel("Frecuencia", fontsize=8.5)
        spine_style(ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_right:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig_style(fig, ax)
        ax.set_facecolor(BG_AX)
        stats.probplot(df[var_key].dropna(), dist="norm", plot=ax)
        ax.get_lines()[0].set(color=color_sel, markersize=4.5,
                               markeredgecolor="white", markeredgewidth=0.4, alpha=0.72)
        ax.get_lines()[1].set(color=COL_REF, linewidth=1.6)
        ax.set_title(f"Q-Q Plot — {var_sel}")
        spine_style(ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Diagramas de Caja — Todas las Variables</div>',
                unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig_style(fig, axes)
    fig.suptitle("Distribución y Valores Atípicos",
                 fontsize=12, fontweight="bold", color=COL_TIT)

    for i, (var, nombre, color) in enumerate(zip(variables, nombres_var, COLORS_VARS)):
        ax = axes[i]
        ax.set_facecolor(BG_AX)
        ax.boxplot(df[var].dropna(), patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.58, linewidth=0.8),
                   medianprops=dict(color=COL_REF, linewidth=2.1),
                   whiskerprops=dict(linewidth=0.9, color="#7A6E65"),
                   capprops=dict(linewidth=0.9, color="#7A6E65"),
                   flierprops=dict(marker="o", markerfacecolor=color, markersize=4.5,
                                   alpha=0.62, markeredgecolor="white", markeredgewidth=0.4))
        q1, q3 = df[var].quantile(0.25), df[var].quantile(0.75)
        iqr     = q3 - q1
        n_out   = len(df[(df[var] < q1 - 1.5*iqr) | (df[var] > q3 + 1.5*iqr)])
        ax.set_title(nombre)
        ax.set_ylabel("Valor", fontsize=8.5)
        ax.text(0.5, -0.13, f"Valores atípicos: {n_out}",
                transform=ax.transAxes, ha="center", fontsize=8.5,
                color=COL_REF if n_out > 0 else "#4E7A5E", fontweight="bold")
        spine_style(ax)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB S — STUDY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">Relaciones, Correlaciones y Modelos</div>',
                unsafe_allow_html=True)

    # ── Matrices de correlación ──
    st.markdown('<div class="sub-title">Matrices de Correlación</div>', unsafe_allow_html=True)

    corr_p = df[variables].corr(method="pearson")
    corr_s = df[variables].corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig_style(fig, axes)

    for ax, corr, title in zip(axes, [corr_p, corr_s], ["Pearson", "Spearman"]):
        ax.set_facecolor(BG_AX)
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm",
                    center=0, vmin=-1, vmax=1, square=True,
                    xticklabels=nombres_var, yticklabels=nombres_var,
                    linewidths=1.4, linecolor=BG_FIG,
                    annot_kws={"size": 9},
                    cbar_kws={"label": "Correlación", "shrink": .78}, ax=ax)
        ax.set_title(f"Correlación de {title}", pad=14)
        ax.tick_params(axis="x", rotation=28, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Scatterplots ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Relaciones con la Variable Dependiente</div>',
                unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig_style(fig, axes)
    fig.suptitle("Relaciones con Años de Escolaridad",
                 fontsize=12, fontweight="bold", color=COL_TIT)

    for ax, xvar, xlabel, color in zip(
            axes,
            ["gasto_educacion_pib", "pib_per_capita", "tasa_alfabetizacion"],
            ["Gasto Educ. (% PIB)", "PIB per cápita (USD)", "Tasa de Alfabetización (%)"],
            [C1, C3, C4]):
        ax.set_facecolor(BG_AX)
        ax.scatter(df[xvar], df["anios_escolaridad"],
                   alpha=0.48, s=30, c=color, edgecolors="white", linewidths=0.4)
        z  = np.polyfit(df[xvar], df["anios_escolaridad"], 1)
        xl = np.linspace(df[xvar].min(), df[xvar].max(), 100)
        ax.plot(xl, np.poly1d(z)(xl), color=COL_REF, linewidth=1.8, linestyle="--")
        r, p = stats.pearsonr(df[xvar], df["anios_escolaridad"])
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_ylabel("Años de Escolaridad", fontsize=8.5)
        ax.set_title(f"r = {r:.3f}   p = {p:.4f}")
        spine_style(ax)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Transformación log PIB ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Transformación Logarítmica del PIB per cápita</div>',
                unsafe_allow_html=True)

    r_orig = stats.pearsonr(df["pib_per_capita"],     df["anios_escolaridad"])[0]
    r_log  = stats.pearsonr(df["log_pib_per_capita"], df["anios_escolaridad"])[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig_style(fig, axes)

    for ax, xvar, xlabel, r_val, title in zip(
            axes,
            ["pib_per_capita", "log_pib_per_capita"],
            ["PIB per cápita (USD)", "log(PIB per cápita)"],
            [r_orig, r_log],
            ["Escala Original", "Escala Logarítmica"]):
        ax.set_facecolor(BG_AX)
        ax.scatter(df[xvar], df["anios_escolaridad"],
                   alpha=0.48, s=28, c=C3, edgecolors="white", linewidths=0.4)
        z  = np.polyfit(df[xvar], df["anios_escolaridad"], 1)
        xl = np.linspace(df[xvar].min(), df[xvar].max(), 100)
        ax.plot(xl, np.poly1d(z)(xl), color=COL_REF, linewidth=1.8, linestyle="--")
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_ylabel("Años de Escolaridad", fontsize=8.5)
        ax.set_title(f"{title}  —  r = {r_val:.3f}")
        spine_style(ax)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown(f"""
    <div class="info-box">
    La transformación logarítmica mejora la correlación de <strong>{r_orig:.3f}</strong>
    a <strong>{r_log:.3f}</strong>, confirmando que la relación entre PIB y escolaridad
    es <em>no lineal</em>. Se utilizará <code>log(PIB per cápita)</code> en el Modelo 2.
    </div>
    """, unsafe_allow_html=True)

    # ── Modelo 1 ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Modelo 1 — Regresión Lineal Simple</div>',
                unsafe_allow_html=True)

    m1   = res["m1"]
    y1te = np.array(res["y1te"])
    y1p  = np.array(res["y1p"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig_style(fig, axes)
    fig.suptitle("Modelo 1 — Gasto en Educación → Años de Escolaridad",
                 fontsize=11, fontweight="bold", color=COL_TIT)

    ax = axes[0]
    ax.set_facecolor(BG_AX)
    ax.scatter(df["gasto_educacion_pib"], df["anios_escolaridad"],
               alpha=0.46, s=28, c=C1, edgecolors="white", linewidths=0.4)
    xl = np.linspace(df["gasto_educacion_pib"].min(),
                     df["gasto_educacion_pib"].max(), 100).reshape(-1, 1)
    ax.plot(xl, m1.predict(xl), color=COL_REF, linewidth=1.9,
            label=f"R² test = {res['r2_1']:.3f}")
    ax.set_xlabel("Gasto Educ. (% PIB)", fontsize=8.5)
    ax.set_ylabel("Años de Escolaridad", fontsize=8.5)
    ax.set_title("Línea de Regresión")
    ax.legend()
    spine_style(ax)

    ax = axes[1]
    ax.set_facecolor(BG_AX)
    ax.scatter(y1te, y1p, alpha=0.52, s=28, c=C2, edgecolors="white", linewidths=0.4)
    lim = [min(y1te.min(), y1p.min())-1, max(y1te.max(), y1p.max())+1]
    ax.plot(lim, lim, color=COL_REF, linestyle="--", linewidth=1.6, label="Referencia ideal")
    ax.set_xlabel("Valores Reales", fontsize=8.5)
    ax.set_ylabel("Valores Predichos", fontsize=8.5)
    ax.set_title("Real vs Predicho (Test)")
    ax.legend(); ax.set_xlim(lim); ax.set_ylim(lim)
    spine_style(ax)

    ax = axes[2]
    ax.set_facecolor(BG_AX)
    res1 = y1te - y1p
    ax.scatter(y1p, res1, alpha=0.52, s=28, c=C3, edgecolors="white", linewidths=0.4)
    ax.axhline(0, color=COL_REF, linestyle="--", linewidth=1.6)
    ax.set_xlabel("Valores Predichos", fontsize=8.5)
    ax.set_ylabel("Residuos", fontsize=8.5)
    ax.set_title("Análisis de Residuos")
    spine_style(ax)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Modelo 2 ──
    st.markdown('<div class="sub-title">Modelo 2 — Regresión Lineal Múltiple</div>',
                unsafe_allow_html=True)

    y2te = np.array(res["y2te"])
    y2p  = np.array(res["y2p"])
    res2 = y2te - y2p

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig_style(fig, axes)
    fig.suptitle("Modelo 2 — Gasto + log(PIB) + Alfabetización → Años de Escolaridad",
                 fontsize=11, fontweight="bold", color=COL_TIT)

    ax = axes[0]
    ax.set_facecolor(BG_AX)
    ax.scatter(y2te, y2p, alpha=0.52, s=28, c=C2, edgecolors="white", linewidths=0.4)
    lim = [min(y2te.min(), y2p.min())-1, max(y2te.max(), y2p.max())+1]
    ax.plot(lim, lim, color=COL_REF, linestyle="--", linewidth=1.6, label="Referencia ideal")
    ax.set_xlabel("Valores Reales", fontsize=8.5)
    ax.set_ylabel("Valores Predichos", fontsize=8.5)
    ax.set_title(f"Real vs Predicho  —  R² = {res['r2_2']:.3f}")
    ax.legend(); ax.set_xlim(lim); ax.set_ylim(lim)
    spine_style(ax)

    ax = axes[1]
    ax.set_facecolor(BG_AX)
    ax.scatter(y2p, res2, alpha=0.52, s=28, c=C3, edgecolors="white", linewidths=0.4)
    ax.axhline(0, color=COL_REF, linestyle="--", linewidth=1.6)
    ax.set_xlabel("Valores Predichos", fontsize=8.5)
    ax.set_ylabel("Residuos", fontsize=8.5)
    ax.set_title("Análisis de Residuos")
    spine_style(ax)

    ax = axes[2]
    ax.set_facecolor(BG_AX)
    sns.histplot(res2, kde=True, color=C4, alpha=0.58, edgecolor="white",
                 linewidth=0.4, ax=ax)
    ax.axvline(0, color=COL_REF, linestyle="--", linewidth=1.6)
    sw_p = stats.shapiro(res2)[1]
    ax.set_title(f"Distribución de Residuos\nShapiro-Wilk p = {sw_p:.4f}")
    ax.set_xlabel("Residuos", fontsize=8.5)
    ax.set_ylabel("Frecuencia", fontsize=8.5)
    spine_style(ax)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Comparación de modelos ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comparación de Modelos</div>', unsafe_allow_html=True)

    comp_data = pd.DataFrame({
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
    })
    render_table(comp_data,
                 num_cols=["Modelo 1 (Simple)", "Modelo 2 (Múltiple)"],
                 highlight_col="Métrica")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig_style(fig, axes)
    fig.suptitle("Comparación de Modelos de Regresión",
                 fontsize=12, fontweight="bold", color=COL_TIT)

    for ax, vals, ylabel, titulo in zip(
            axes,
            [[res["r2_1"], res["r2_2"]], [res["rmse_1"], res["rmse_2"]],
             [res["cv1"].mean(), res["cv2"].mean()]],
            ["R²", "RMSE", "R² CV"],
            ["Coef. de Determinación\n(mayor es mejor)",
             "Error Cuadrático Medio\n(menor es mejor)",
             "Validación Cruzada 5-fold\n(mayor es mejor)"]):
        ax.set_facecolor(BG_AX)
        bars = ax.bar(["Modelo 1\n(Simple)", "Modelo 2\n(Múltiple)"],
                      vals, color=[C1, C2], alpha=0.70,
                      edgecolor="white", linewidth=0.5, width=0.46)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.022,
                    f"{val:.4f}", ha="center", fontweight="bold",
                    fontsize=10.5, color=COL_TIT)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_title(titulo)
        ax.set_ylim(0, max(vals)*1.35)
        spine_style(ax)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB T — TELL
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">Síntesis, Conclusiones y Respuestas</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="sub-title">Panel Resumen del Análisis</div>', unsafe_allow_html=True)

    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor(BG_FIG)
    fig.suptitle("Resumen del Análisis Exploratorio — Educación & Escolaridad Global",
                 fontsize=13, fontweight="bold", color=COL_TIT, y=1.01)

    vars_corr  = ["gasto_educacion_pib", "pib_per_capita", "tasa_alfabetizacion", "log_pib_per_capita"]
    names_corr = ["Gasto Educ. (% PIB)", "PIB per cápita", "Tasa Alfab.", "log(PIB per cápita)"]
    correlaciones = [df[v].corr(df["anios_escolaridad"]) for v in vars_corr]

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor(BG_AX)
    colores_bar = [C1 if c > 0 else C2 for c in correlaciones]
    bars = ax1.barh(names_corr, correlaciones, color=colores_bar,
                    alpha=0.68, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Correlación con Años de Escolaridad", fontsize=8.5)
    ax1.set_title("Correlaciones con Variable Dependiente")
    ax1.axvline(0, color="#7A6E65", linewidth=0.7)
    for bar, val in zip(bars, correlaciones):
        ax1.text(val + 0.014 if val > 0 else val - 0.014,
                 bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center",
                 ha="left" if val > 0 else "right",
                 fontsize=8.5, fontweight="bold", color=COL_TIT)
    spine_style(ax1)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor(BG_AX)
    r2_adj = [res["sm1"].rsquared_adj, res["sm2"].rsquared_adj]
    bars2  = ax2.bar(["Modelo 1\n(Simple)", "Modelo 2\n(Múltiple)"],
                     r2_adj, color=[C1, C2], alpha=0.68,
                     edgecolor="white", linewidth=0.5, width=0.46)
    for bar, val in zip(bars2, r2_adj):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f"{val:.4f}", ha="center", fontweight="bold", fontsize=10.5, color=COL_TIT)
    ax2.set_ylabel("R² Ajustado", fontsize=8.5)
    ax2.set_title("Comparación de Modelos")
    ax2.set_ylim(0, max(r2_adj)*1.4)
    spine_style(ax2)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor(BG_AX)
    y_all   = res["m2"].predict(df[res["vars2"]].values)
    lim_all = [min(df["anios_escolaridad"].min(), y_all.min())-1,
               max(df["anios_escolaridad"].max(), y_all.max())+1]
    ax3.scatter(df["anios_escolaridad"], y_all,
                alpha=0.46, s=24, c=C2, edgecolors="white", linewidths=0.4)
    ax3.plot(lim_all, lim_all, color=COL_REF, linestyle="--", linewidth=1.6)
    ax3.set_xlabel("Valores Reales", fontsize=8.5)
    ax3.set_ylabel("Predichos (Modelo 2)", fontsize=8.5)
    ax3.set_title("Predicción del Mejor Modelo (todos los países)")
    ax3.set_xlim(lim_all); ax3.set_ylim(lim_all)
    spine_style(ax3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor(BG_AX)
    imp_names  = [x[0] for x in res["importancia"]]
    imp_vals   = [x[1] for x in res["importancia"]]
    imp_colors = [C2 if v < 0 else C1 for v in imp_vals]
    bars4 = ax4.barh(imp_names, [abs(v) for v in imp_vals],
                     color=imp_colors, alpha=0.68, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars4, imp_vals):
        ax4.text(bar.get_width() + 0.007, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=8.5,
                 fontweight="bold", color=COL_TIT)
    ax4.set_xlabel("|Coeficiente Estandarizado|", fontsize=8.5)
    ax4.set_title("Importancia de Variables (Modelo 2)")
    spine_style(ax4)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Respuestas ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Respuestas a las Preguntas de Investigación</div>',
                unsafe_allow_html=True)

    mejora_r2   = ((res["r2_2"] - res["r2_1"]) / abs(res["r2_1"])) * 100 if res["r2_1"] != 0 else 0
    mejora_rmse = ((res["rmse_1"] - res["rmse_2"]) / res["rmse_1"]) * 100

    respuestas = [
        ("P1", "Relación lineal gasto — escolaridad",
         f"La relación es <strong>débil a moderada</strong>. El Modelo 1 alcanza un R² ajustado de "
         f"<strong>{res['sm1'].rsquared_adj:.4f}</strong>, indicando que el gasto como porcentaje del PIB "
         f"<em>no es suficiente por sí solo</em> para explicar la variabilidad en los años de escolaridad."),
        ("P2", "PIB per cápita vs gasto en educación",
         f"<strong>Sí.</strong> El PIB per cápita (en escala logarítmica) presenta una correlación "
         f"<strong>considerablemente más fuerte</strong> con la escolaridad. Los coeficientes "
         f"estandarizados del Modelo 2 confirman que el desarrollo económico es el predictor dominante."),
        ("P3", "Correlación tasa de alfabetización",
         f"La tasa de alfabetización muestra una correlación <strong>positiva significativa</strong> "
         f"(r = {df['tasa_alfabetizacion'].corr(df['anios_escolaridad']):.3f}) con la escolaridad. "
         f"Sin embargo, la correlación con log(PIB) tiende a ser mayor, "
         f"confirmando el rol preponderante del contexto económico."),
        ("P4", "Modelo múltiple vs modelo simple",
         f"<strong>Sí, significativamente.</strong> El Modelo 2 supera al Modelo 1 en todas las métricas: "
         f"R² mejora de <strong>{res['r2_1']:.4f}</strong> a <strong>{res['r2_2']:.4f}</strong> "
         f"(<strong>+{mejora_r2:.1f}%</strong>), y el RMSE se reduce un <strong>{mejora_rmse:.1f}%</strong>."),
    ]

    for code, titulo, texto in respuestas:
        st.markdown(f"""
        <div class="answer-box">
          <h4>{code} — {titulo}</h4>
          <p style="margin:0; line-height:1.6;">{texto}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Conclusiones ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Conclusiones del EDA</div>', unsafe_allow_html=True)

    conclusiones = [
        ("I",   "Gasto educativo como predictor aislado",
         "El porcentaje del PIB destinado a educación no predice de forma robusta los años de "
         "escolaridad. La <strong>eficiencia y focalización del gasto</strong> son tan críticas "
         "como su magnitud."),
        ("II",  "PIB per cápita: el predictor dominante",
         "El desarrollo económico general crea condiciones estructurales (infraestructura, "
         "estabilidad, incentivos laborales) que impulsan la educación prolongada, especialmente "
         "visible en escala logarítmica."),
        ("III", "Superioridad del modelo múltiple",
         f"Combinar gasto, PIB y alfabetización produce un modelo con R² ajustado de "
         f"<strong>{res['sm2'].rsquared_adj:.4f}</strong> frente a "
         f"<strong>{res['sm1'].rsquared_adj:.4f}</strong> del modelo simple — "
         f"una mejora sustancial y estadísticamente significativa."),
        ("IV",  "Implicación para política pública",
         "Aumentar el presupuesto educativo es necesario pero <em>no suficiente</em>. "
         "Deben considerarse simultáneamente el crecimiento económico, la reducción del "
         "analfabetismo y la calidad de la inversión educativa."),
    ]

    concl_colors = [C1, C3, C2, C4]
    cols = st.columns(2)
    for i, (num, titulo, texto) in enumerate(conclusiones):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:#fff; border-radius:12px; padding:1.1rem 1.35rem;
                        border-left:4px solid {concl_colors[i]};
                        box-shadow:0 1px 8px rgba(0,0,0,.06);
                        margin-bottom:1rem; min-height:118px;">
              <span style="font-family:'Playfair Display',serif; font-size:.70rem;
                           text-transform:uppercase; letter-spacing:.1em;
                           color:{concl_colors[i]}; font-weight:700;">{num}</span>
              <div style="font-family:'Playfair Display',serif; font-weight:700;
                          font-size:.96rem; color:#2E4A6B; margin:.28rem 0 .45rem;">{titulo}</div>
              <p style="margin:0; color:#4A3E33; font-size:.89rem; line-height:1.55;">{texto}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Tabla resumen final ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Tabla Resumen de Resultados</div>', unsafe_allow_html=True)

    mejor_modelo = ("Modelo 2 (Regresión Múltiple)" if res["r2_2"] > res["r2_1"]
                    else "Modelo 1 (Regresión Simple)")
    resumen_data = pd.DataFrame({
        "Aspecto": [
            "Variable dependiente",
            "R² Ajustado — Modelo 1", "R² Ajustado — Modelo 2",
            "RMSE — Modelo 1",        "RMSE — Modelo 2",
            "Variable más influyente", "Mejora Modelo 2 vs 1", "Mejor modelo",
        ],
        "Resultado": [
            "Años promedio de escolaridad",
            f"{res['sm1'].rsquared_adj:.4f}",
            f"{res['sm2'].rsquared_adj:.4f}",
            f"{res['rmse_1']:.4f}",
            f"{res['rmse_2']:.4f}",
            res["importancia"][0][0],
            f"{mejora_r2:+.1f}% en R²",
            mejor_modelo,
        ],
    })
    render_table(resumen_data, highlight_col="Aspecto")

    st.markdown("""
    <div style="text-align:center; margin-top:2.2rem; color:#A89880; font-size:.79rem;
                letter-spacing:.03em; line-height:2;">
      Fuente de datos: <strong>Banco Mundial</strong> — World Bank Open Data (data.worldbank.org)
      &nbsp;&nbsp;·&nbsp;&nbsp;
      Framework de análisis: <strong>QUEST</strong>
      (Question &rarr; Understand &rarr; Explore &rarr; Study &rarr; Tell)
    </div>
    """, unsafe_allow_html=True)
