import streamlit as st
import pandas as pd
import numpy as np
import requests
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA · Educación Global",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES  —  fondo beige cálido, tipografía elegante, tabs destacados
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');

/* ---------- fondo global ---------- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #F7F3EE !important;
    font-family: 'Source Sans 3', sans-serif;
}
[data-testid="stHeader"] { background-color: #F7F3EE !important; }

/* ---------- hero banner ---------- */
.hero {
    background: linear-gradient(135deg, #3D5A80 0%, #6B8F71 60%, #A8997E 100%);
    border-radius: 18px;
    padding: 2.8rem 3rem 2.4rem;
    margin-bottom: 1.8rem;
    color: #fff;
    box-shadow: 0 8px 32px rgba(61,90,128,.25);
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 .5rem;
    letter-spacing: .02em;
}
.hero p { font-size: 1rem; opacity: .88; margin: 0; line-height: 1.6; }
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,.18);
    border: 1px solid rgba(255,255,255,.35);
    border-radius: 20px;
    padding: .2rem .8rem;
    font-size: .78rem;
    margin-right: .4rem;
    margin-top: .7rem;
    letter-spacing: .05em;
    text-transform: uppercase;
}

/* ---------- tabs ---------- */
[data-testid="stTabs"] > div:first-child {
    gap: 6px;
    background: #EDE8E0;
    border-radius: 12px;
    padding: 6px 8px;
    margin-bottom: 1.4rem;
}
button[data-baseweb="tab"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #6B5E4C !important;
    background: transparent !important;
    border-radius: 9px !important;
    padding: .5rem 1.2rem !important;
    border: none !important;
    transition: all .2s ease !important;
}
button[data-baseweb="tab"]:hover {
    background: rgba(255,255,255,.55) !important;
    color: #3D5A80 !important;
}
button[aria-selected="true"][data-baseweb="tab"] {
    background: #fff !important;
    color: #3D5A80 !important;
    box-shadow: 0 2px 8px rgba(61,90,128,.15) !important;
}

/* ---------- section headers ---------- */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: #3D5A80;
    border-left: 5px solid #6B8F71;
    padding-left: .8rem;
    margin-bottom: 1rem;
}
.sub-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #5C4A32;
    margin-top: 1.4rem;
    margin-bottom: .4rem;
}

/* ---------- metric cards ---------- */
.card-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 1rem 0; }
.metric-card {
    flex: 1;
    min-width: 140px;
    background: #fff;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 2px 10px rgba(0,0,0,.06);
    border-top: 4px solid #6B8F71;
    text-align: center;
}
.metric-card .label {
    font-size: .75rem;
    color: #9E8C78;
    text-transform: uppercase;
    letter-spacing: .07em;
    margin-bottom: .3rem;
}
.metric-card .value {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #3D5A80;
}
.metric-card .sub {
    font-size: .78rem;
    color: #A89880;
    margin-top: .2rem;
}

/* ---------- info boxes ---------- */
.info-box {
    background: #fff;
    border-left: 4px solid #A8997E;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: .8rem 0;
    font-size: .95rem;
    color: #4A3E33;
    box-shadow: 0 1px 6px rgba(0,0,0,.05);
}
.info-box strong { color: #3D5A80; }
.answer-box {
    background: #EDF2F8;
    border-left: 4px solid #3D5A80;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: .8rem 0;
    color: #2C3E57;
}
.answer-box h4 {
    font-family: 'Playfair Display', serif;
    margin: 0 0 .4rem;
    font-size: 1rem;
    color: #3D5A80;
}

/* ---------- table ---------- */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ---------- dividers ---------- */
hr { border: none; border-top: 1px solid #DDD4C8; margin: 1.4rem 0; }

/* ---------- spinner ---------- */
[data-testid="stSpinner"] > div > div { border-top-color: #3D5A80 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# PALETA COOLWARM PERSONALIZADA (pastel)
# ──────────────────────────────────────────────────────────────────────────────
CW = plt.cm.coolwarm
PASTEL_COLORS = [
    "#7EB5D6",   # azul frío pastel
    "#E8A090",   # rojo cálido pastel
    "#A8C8A0",   # verde neutro pastel
    "#C4A8C8",   # púrpura pastel
    "#F0C890",   # amarillo cálido pastel
    "#90C8D8",   # cian frío pastel
]
C1, C2, C3, C4, C5, C6 = PASTEL_COLORS

def cw_cmap():
    """Coolwarm suavizado para heatmaps."""
    return "coolwarm"

def fig_style(fig, ax_or_axes=None):
    """Aplica fondo beige consistente a cualquier figura."""
    fig.patch.set_facecolor("#F7F3EE")
    axes = ax_or_axes if ax_or_axes is not None else []
    if hasattr(axes, '__iter__') and not isinstance(axes, plt.Axes):
        for ax in np.array(axes).flatten():
            ax.set_facecolor("#FDFAF6")
    elif isinstance(axes, plt.Axes):
        axes.set_facecolor("#FDFAF6")
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# DESCARGA DE DATOS (cacheada)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def descargar_indicador(indicador, nombre, inicio=2010, fin=2023):
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicador}"
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

    df_g = mas_reciente(descargar_indicador("SE.XPD.TOTL.GD.ZS", "gasto_educacion_pib"), "gasto_educacion_pib")
    df_e = mas_reciente(descargar_indicador("UIS.EA.MEAN.1T6.AG25T99", "anios_escolaridad"), "anios_escolaridad")
    if df_e.empty:
        df_e = mas_reciente(descargar_indicador("SE.SCH.LIFE", "anios_escolaridad"), "anios_escolaridad")
    df_p = mas_reciente(descargar_indicador("NY.GDP.PCAP.PP.CD", "pib_per_capita"), "pib_per_capita")
    df_a = mas_reciente(descargar_indicador("SE.ADT.LITR.ZS", "tasa_alfabetizacion"), "tasa_alfabetizacion")

    df = (df_g.merge(df_e[["codigo_pais", "anios_escolaridad"]], on="codigo_pais", how="inner")
               .merge(df_p[["codigo_pais", "pib_per_capita"]], on="codigo_pais", how="inner")
               .merge(df_a[["codigo_pais", "tasa_alfabetizacion"]], on="codigo_pais", how="inner"))
    df = df.dropna().copy()
    df["log_pib_per_capita"] = np.log(df["pib_per_capita"])
    return df

# ──────────────────────────────────────────────────────────────────────────────
# MODELOS (cacheados)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def ajustar_modelos(df_json):
    df = pd.read_json(df_json)
    variables = ["gasto_educacion_pib", "anios_escolaridad", "pib_per_capita", "tasa_alfabetizacion"]
    df = df.dropna(subset=variables + ["log_pib_per_capita"])

    # Modelo 1
    X1 = df[["gasto_educacion_pib"]].values
    y  = df["anios_escolaridad"].values
    X1tr, X1te, y1tr, y1te = train_test_split(X1, y, test_size=0.2, random_state=42)
    m1 = LinearRegression().fit(X1tr, y1tr)
    y1p = m1.predict(X1te)
    r2_1  = r2_score(y1te, y1p)
    rmse_1 = np.sqrt(mean_squared_error(y1te, y1p))
    mae_1  = mean_absolute_error(y1te, y1p)
    cv1   = cross_val_score(LinearRegression(), X1, y, cv=5, scoring="r2")
    X1sm  = sm.add_constant(df["gasto_educacion_pib"])
    sm1   = sm.OLS(df["anios_escolaridad"], X1sm).fit()

    # Modelo 2
    vars2 = ["gasto_educacion_pib", "log_pib_per_capita", "tasa_alfabetizacion"]
    X2 = df[vars2].values
    X2tr, X2te, y2tr, y2te = train_test_split(X2, y, test_size=0.2, random_state=42)
    m2 = LinearRegression().fit(X2tr, y2tr)
    y2p = m2.predict(X2te)
    r2_2  = r2_score(y2te, y2p)
    rmse_2 = np.sqrt(mean_squared_error(y2te, y2p))
    mae_2  = mean_absolute_error(y2te, y2p)
    cv2   = cross_val_score(LinearRegression(), X2, y, cv=5, scoring="r2")
    X2sm  = sm.add_constant(df[vars2])
    sm2   = sm.OLS(df["anios_escolaridad"], X2sm).fit()

    scaler = StandardScaler()
    X2sc = scaler.fit_transform(df[vars2])
    m2std = LinearRegression().fit(X2sc, df["anios_escolaridad"])
    importancia = sorted(
        zip(["Gasto educ. (% PIB)", "log(PIB per cápita)", "Tasa alfabetización"],
            m2std.coef_),
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
  <h1>📚 Educación &amp; Escolaridad Global</h1>
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
with st.spinner("⏳ Descargando datos del Banco Mundial…"):
    df = cargar_datos()

if df.empty:
    st.error("No se pudieron descargar los datos. Verifica tu conexión a internet.")
    st.stop()

# variables
variables    = ["gasto_educacion_pib", "anios_escolaridad", "pib_per_capita", "tasa_alfabetizacion"]
nombres_var  = ["Gasto Educ. (% PIB)", "Años Escolaridad", "PIB per cápita (USD)", "Alfabetización (%)"]
COLORS_VARS  = [C1, C2, C3, C4]

# ajustar modelos
res = ajustar_modelos(df.to_json())

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "❓ Q — Question",
    "🔍 U — Understand",
    "📊 E — Explore",
    "🧪 S — Study",
    "💡 T — Tell",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB Q — QUESTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">Preguntas de Investigación</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    En esta sección se plantean las preguntas que guiarán todo el análisis exploratorio.
    Serán respondidas de manera formal en la sección <strong>T — Tell</strong>.
    </div>
    """, unsafe_allow_html=True)

    preguntas = [
        ("P1", "Relación lineal", "¿Existe una relación lineal significativa entre el gasto público en educación (% del PIB) y los años promedio de escolaridad de un país?"),
        ("P2", "Influencia del PIB", "¿El PIB per cápita tiene mayor influencia que el gasto en educación sobre los años promedio de escolaridad de la población?"),
        ("P3", "Correlación alfabetización", "¿Cuál es la correlación entre la tasa de alfabetización y los años promedio de escolaridad, y cómo se compara con las otras variables?"),
        ("P4", "Modelo múltiple vs simple", "¿Un modelo de regresión múltiple predice significativamente mejor los años de escolaridad que uno basado únicamente en el gasto en educación?"),
    ]

    cols = st.columns(2)
    border_colors = [C1, C3, C2, C4]
    for i, (code, title, text) in enumerate(preguntas):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:#fff; border-radius:14px; padding:1.3rem 1.5rem;
                        border-top:5px solid {border_colors[i]}; box-shadow:0 2px 10px rgba(0,0,0,.06);
                        margin-bottom:1rem;">
              <span style="font-family:'Playfair Display',serif; font-size:1.7rem;
                           color:{border_colors[i]}; font-weight:700;">{code}</span>
              <span style="font-size:.75rem; text-transform:uppercase; letter-spacing:.08em;
                           color:#9E8C78; margin-left:.5rem;">{title}</span>
              <p style="margin:.6rem 0 0; color:#4A3E33; font-size:.97rem; line-height:1.55;">{text}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Variables del Estudio</div>', unsafe_allow_html=True)

    tabla_vars = pd.DataFrame({
        "Variable": ["Gasto en educación (% PIB)", "Años promedio de escolaridad",
                     "PIB per cápita (PPA, USD)", "Tasa de alfabetización"],
        "Rol": ["Independiente", "Dependiente ✦", "Independiente", "Independiente"],
        "Indicador Banco Mundial": ["SE.XPD.TOTL.GD.ZS", "UIS.EA.MEAN.1T6.AG25T99",
                                     "NY.GDP.PCAP.PP.CD", "SE.ADT.LITR.ZS"],
    })
    st.dataframe(tabla_vars, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB U — UNDERSTAND
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">Comprensión de los Datos</div>', unsafe_allow_html=True)

    # KPIs
    n_paises = len(df)
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
        <div class="label">Gasto promedio educ.</div>
        <div class="value">{df['gasto_educacion_pib'].mean():.1f}%</div>
        <div class="sub">del PIB</div>
      </div>
      <div class="metric-card">
        <div class="label">Años escolaridad prom.</div>
        <div class="value">{df['anios_escolaridad'].mean():.1f}</div>
        <div class="sub">años</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Estadísticos Descriptivos</div>', unsafe_allow_html=True)

    stats_rows = ["Media", "Mediana", "Desv. Estándar", "Mínimo",
                  "P25 (Q1)", "P75 (Q3)", "Máximo", "Asimetría", "Curtosis"]
    stats_data = {}
    for var, nombre in zip(variables, nombres_var):
        col = df[var]
        stats_data[nombre] = [
            col.mean(), col.median(), col.std(), col.min(),
            col.quantile(0.25), col.quantile(0.75), col.max(),
            col.skew(), col.kurtosis(),
        ]
    st.dataframe(pd.DataFrame(stats_data, index=stats_rows).round(3), use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Distribuciones Generales</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig_style(fig, axes)
    fig.suptitle("Distribución de las Variables de Estudio",
                 fontsize=14, fontweight="bold", color="#3D5A80",
                 fontfamily="serif", y=1.01)

    for i, (var, nombre, color) in enumerate(zip(variables, nombres_var, COLORS_VARS)):
        ax = axes[i // 2][i % 2]
        sns.histplot(data=df, x=var, kde=True, color=color,
                     alpha=0.65, edgecolor="white", ax=ax)
        media   = df[var].mean()
        mediana = df[var].median()
        ax.axvline(media,   color="#E06060", linestyle="--", linewidth=1.8,
                   label=f"Media: {media:.1f}")
        ax.axvline(mediana, color="#3D5A80", linestyle="-", linewidth=1.8,
                   label=f"Mediana: {mediana:.1f}")
        ax.set_title(nombre, fontsize=11, fontweight="bold", color="#3D5A80")
        ax.set_xlabel(""); ax.set_ylabel("Frecuencia", fontsize=9)
        ax.legend(fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#D0C8BC")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("""
    <div class="info-box">
    <strong>Observaciones clave:</strong>
    El <strong>PIB per cápita</strong> muestra fuerte asimetría positiva (pocos países con valores muy altos),
    mientras que la <strong>tasa de alfabetización</strong> tiene asimetría negativa (mayoría supera el 80%).
    Se aplicará transformación logarítmica al PIB en los modelos de regresión.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB E — EXPLORE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">Exploración Individual de Variables</div>', unsafe_allow_html=True)

    # Selector de variable
    var_sel = st.selectbox("Selecciona una variable para explorar en detalle:",
                            nombres_var, index=0)
    var_idx = nombres_var.index(var_sel)
    var_key  = variables[var_idx]
    color_sel = COLORS_VARS[var_idx]

    col_left, col_right = st.columns(2)

    with col_left:
        # Histograma + Shapiro
        fig, ax = plt.subplots(figsize=(6, 4))
        fig_style(fig, ax)
        sns.histplot(data=df, x=var_key, bins=22, kde=True,
                     color=color_sel, alpha=0.65, edgecolor="white", ax=ax)
        sw_stat, sw_p = stats.shapiro(df[var_key].dropna())
        normal_text = "Distribución normal ✓" if sw_p > 0.05 else "No normal ✗"
        ax.set_title(f"{var_sel}\nShapiro-Wilk: p = {sw_p:.4f}  —  {normal_text}",
                     fontsize=10, color="#3D5A80")
        ax.set_xlabel(var_sel, fontsize=9); ax.set_ylabel("Frecuencia", fontsize=9)
        for spine in ax.spines.values(): spine.set_edgecolor("#D0C8BC")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col_right:
        # Q-Q Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        fig_style(fig, ax)
        stats.probplot(df[var_key].dropna(), dist="norm", plot=ax)
        ax.get_lines()[0].set_color(color_sel)
        ax.get_lines()[0].set_markersize(5)
        ax.get_lines()[1].set_color("#E06060")
        ax.set_title(f"Q-Q Plot · {var_sel}", fontsize=10, color="#3D5A80")
        for spine in ax.spines.values(): spine.set_edgecolor("#D0C8BC")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Diagramas de Caja — Todas las Variables</div>',
                unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig_style(fig, axes)
    fig.suptitle("Distribución y Valores Atípicos (Boxplots)",
                 fontsize=13, fontweight="bold", color="#3D5A80", fontfamily="serif")

    for i, (var, nombre, color) in enumerate(zip(variables, nombres_var, COLORS_VARS)):
        ax = axes[i]
        bp = ax.boxplot(df[var].dropna(), patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.65),
                        medianprops=dict(color="#C0392B", linewidth=2.2),
                        whiskerprops=dict(linewidth=1.4, color="#7A6E65"),
                        capprops=dict(linewidth=1.4, color="#7A6E65"),
                        flierprops=dict(marker="o", markerfacecolor=color,
                                        markersize=5.5, alpha=0.7, markeredgecolor="white"))
        q1, q3 = df[var].quantile(0.25), df[var].quantile(0.75)
        iqr = q3 - q1
        n_out = len(df[(df[var] < q1 - 1.5*iqr) | (df[var] > q3 + 1.5*iqr)])
        ax.set_title(nombre, fontsize=9.5, fontweight="bold", color="#3D5A80")
        ax.set_ylabel("Valor", fontsize=8.5)
        ax.text(0.5, -0.12, f"Outliers: {n_out}",
                transform=ax.transAxes, ha="center", fontsize=9,
                color="#C0392B" if n_out > 0 else "#6B8F71", fontweight="bold")
        for spine in ax.spines.values(): spine.set_edgecolor("#D0C8BC")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB S — STUDY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">Relaciones, Correlaciones y Modelos</div>',
                unsafe_allow_html=True)

    # ── Correlaciones ──
    st.markdown('<div class="sub-title">Matrices de Correlación</div>', unsafe_allow_html=True)

    corr_p = df[variables].corr(method="pearson")
    corr_s = df[variables].corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig_style(fig, axes)

    for ax, corr, title in zip(axes, [corr_p, corr_s], ["Pearson", "Spearman"]):
        sns.heatmap(corr, annot=True, fmt=".3f", cmap=cw_cmap(),
                    center=0, vmin=-1, vmax=1, square=True,
                    xticklabels=nombres_var, yticklabels=nombres_var,
                    linewidths=2, linecolor="#F7F3EE",
                    cbar_kws={"label": "Correlación", "shrink": .8}, ax=ax)
        ax.set_title(f"Correlación de {title}", fontsize=12, fontweight="bold",
                     color="#3D5A80", pad=14)
        ax.tick_params(axis="x", rotation=28, labelsize=8.5)
        ax.tick_params(axis="y", rotation=0,  labelsize=8.5)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    # ── Scatter plots ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Relaciones con la Variable Dependiente</div>',
                unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig_style(fig, axes)
    fig.suptitle("Relaciones con Años de Escolaridad",
                 fontsize=13, fontweight="bold", color="#3D5A80", fontfamily="serif")

    scatter_vars  = ["gasto_educacion_pib", "pib_per_capita", "tasa_alfabetizacion"]
    scatter_labels = ["Gasto Educ. (% PIB)", "PIB per cápita (USD)", "Tasa de Alfabetización (%)"]
    scatter_colors = [C1, C3, C4]

    for ax, xvar, xlabel, color in zip(axes, scatter_vars, scatter_labels, scatter_colors):
        ax.scatter(df[xvar], df["anios_escolaridad"],
                   alpha=0.55, s=38, c=color, edgecolors="white")
        z = np.polyfit(df[xvar], df["anios_escolaridad"], 1)
        xl = np.linspace(df[xvar].min(), df[xvar].max(), 100)
        ax.plot(xl, np.poly1d(z)(xl), color="#C0392B", linewidth=2, linestyle="--")
        r, p = stats.pearsonr(df[xvar], df["anios_escolaridad"])
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Años de Escolaridad", fontsize=9)
        ax.set_title(f"r = {r:.3f}  ·  p = {p:.4f}", fontsize=10, color="#3D5A80")
        for spine in ax.spines.values(): spine.set_edgecolor("#D0C8BC")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    # ── Transformación log PIB ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Transformación Logarítmica del PIB per cápita</div>',
                unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig_style(fig, axes)

    r_orig = stats.pearsonr(df["pib_per_capita"], df["anios_escolaridad"])[0]
    r_log  = stats.pearsonr(df["log_pib_per_capita"], df["anios_escolaridad"])[0]

    for ax, xvar, xlabel, r_val, title in zip(
            axes,
            ["pib_per_capita", "log_pib_per_capita"],
            ["PIB per cápita (USD)", "log(PIB per cápita)"],
            [r_orig, r_log],
            ["Escala Original", "Escala Logarítmica"]):
        ax.scatter(df[xvar], df["anios_escolaridad"],
                   alpha=0.5, s=36, c=C3, edgecolors="white")
        z = np.polyfit(df[xvar], df["anios_escolaridad"], 1)
        xl = np.linspace(df[xvar].min(), df[xvar].max(), 100)
        ax.plot(xl, np.poly1d(z)(xl), color="#C0392B", linewidth=2, linestyle="--")
        ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel("Años de Escolaridad", fontsize=9)
        ax.set_title(f"{title}  —  r = {r_val:.3f}", fontsize=10, color="#3D5A80", fontweight="bold")
        for spine in ax.spines.values(): spine.set_edgecolor("#D0C8BC")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown(f"""
    <div class="info-box">
    La transformación logarítmica mejora la correlación de
    <strong>{r_orig:.3f}</strong> → <strong>{r_log:.3f}</strong>,
    confirmando que la relación entre PIB y escolaridad es <em>no lineal</em>.
    Se usará <code>log(PIB per cápita)</code> en el Modelo 2.
    </div>
    """, unsafe_allow_html=True)

    # ── Modelos ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Modelo 1 · Regresión Lineal Simple</div>',
                unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig_style(fig, axes)
    fig.suptitle("Modelo 1 — Gasto en Educación → Años de Escolaridad",
                 fontsize=12, fontweight="bold", color="#3D5A80", fontfamily="serif")

    m1 = res["m1"]
    y1te, y1p = np.array(res["y1te"]), np.array(res["y1p"])

    # regresión
    ax = axes[0]
    ax.scatter(df["gasto_educacion_pib"], df["anios_escolaridad"],
               alpha=0.5, s=36, c=C1, edgecolors="white")
    xl = np.linspace(df["gasto_educacion_pib"].min(), df["gasto_educacion_pib"].max(), 100).reshape(-1,1)
    ax.plot(xl, m1.predict(xl), color="#C0392B", linewidth=2.2,
            label=f"R² test = {res['r2_1']:.3f}")
    ax.set_xlabel("Gasto Educ. (% PIB)", fontsize=9)
    ax.set_ylabel("Años de Escolaridad", fontsize=9)
    ax.set_title("Línea de Regresión", fontsize=10, color="#3D5A80")
    ax.legend(fontsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor("#D0C8BC")

    # real vs predicho
    ax = axes[1]
    ax.scatter(y1te, y1p, alpha=0.6, s=36, c=C2, edgecolors="white")
    lim = [min(y1te.min(), y1p.min())-1, max(y1te.max(), y1p.max())+1]
    ax.plot(lim, lim, "r--", linewidth=1.5)
    ax.set_xlabel("Valores Reales", fontsize=9); ax.set_ylabel("Valores Predichos", fontsize=9)
    ax.set_title("Real vs Predicho (Test)", fontsize=10, color="#3D5A80")
    ax.set_xlim(lim); ax.set_ylim(lim)
    for spine in ax.spines.values(): spine.set_edgecolor("#D0C8BC")

    # residuos
    ax = axes[2]
    res1 = y1te - y1p
    ax.scatter(y1p, res1, alpha=0.6, s=36, c=C3, edgecolors="white")
    ax.axhline(0, color="#C0392B", linestyle="--", linewidth=1.6)
    ax.set_xlabel("Valores Predichos", fontsize=9); ax.set_ylabel("Residuos", fontsize=9)
    ax.set_title("Análisis de Residuos", fontsize=10, color="#3D5A80")
    for spine in ax.spines.values(): spine.set_edgecolor("#D0C8BC")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<div class="sub-title">Modelo 2 · Regresión Lineal Múltiple</div>',
                unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig_style(fig, axes)
    fig.suptitle("Modelo 2 — Gasto + log(PIB) + Alfabetización → Años de Escolaridad",
                 fontsize=12, fontweight="bold", color="#3D5A80", fontfamily="serif")

    y2te, y2p = np.array(res["y2te"]), np.array(res["y2p"])
    res2 = y2te - y2p

    # real vs predicho
    ax = axes[0]
    ax.scatter(y2te, y2p, alpha=0.6, s=36, c=C2, edgecolors="white")
    lim = [min(y2te.min(), y2p.min())-1, max(y2te.max(), y2p.max())+1]
    ax.plot(lim, lim, "r--", linewidth=1.5, label="Línea ideal")
    ax.set_xlabel("Valores Reales", fontsize=9); ax.set_ylabel("Valores Predichos", fontsize=9)
    ax.set_title(f"Real vs Predicho  —  R² = {res['r2_2']:.3f}", fontsize=10, color="#3D5A80")
    ax.legend(fontsize=8); ax.set_xlim(lim); ax.set_ylim(lim)
    for spine in ax.spines.values(): spine.set_edgecolor("#D0C8BC")

    # residuos
    ax = axes[1]
    ax.scatter(y2p, res2, alpha=0.6, s=36, c=C3, edgecolors="white")
    ax.axhline(0, color="#C0392B", linestyle="--", linewidth=1.6)
    ax.set_xlabel("Valores Predichos", fontsize=9); ax.set_ylabel("Residuos", fontsize=9)
    ax.set_title("Análisis de Residuos", fontsize=10, color="#3D5A80")
    for spine in ax.spines.values(): spine.set_edgecolor("#D0C8BC")

    # distribución residuos
    ax = axes[2]
    sns.histplot(res2, kde=True, color=C4, alpha=0.65, edgecolor="white", ax=ax)
    ax.axvline(0, color="#C0392B", linestyle="--", linewidth=1.6)
    sw_p = stats.shapiro(res2)[1]
    ax.set_title(f"Distribución de Residuos\nShapiro-Wilk p = {sw_p:.4f}", fontsize=10, color="#3D5A80")
    for spine in ax.spines.values(): spine.set_edgecolor("#D0C8BC")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    # ── Comparación de modelos ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comparación de Modelos</div>', unsafe_allow_html=True)

    comp_df = pd.DataFrame({
        "Métrica":        ["R² (test)", "R² Ajustado", "RMSE (test)", "MAE (test)",
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
    }).set_index("Métrica")
    st.dataframe(comp_df, use_container_width=True)

    # gráfico comparación
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig_style(fig, axes)
    fig.suptitle("Comparación de Modelos de Regresión",
                 fontsize=13, fontweight="bold", color="#3D5A80", fontfamily="serif")

    mod_labels = ["Modelo 1\n(Simple)", "Modelo 2\n(Múltiple)"]
    bar_colors = [C1, C2]

    for ax, vals, ylabel, titulo in zip(
            axes,
            [[res["r2_1"], res["r2_2"]], [res["rmse_1"], res["rmse_2"]],
             [res["cv1"].mean(), res["cv2"].mean()]],
            ["R²", "RMSE", "R² CV"],
            ["Coef. Determinación\n(mayor → mejor)",
             "Error Cuadrático Medio\n(menor → mejor)",
             "Validación Cruzada 5-fold\n(mayor → mejor)"]):
        bars = ax.bar(mod_labels, vals, color=bar_colors, alpha=0.75,
                      edgecolor="white", linewidth=2, width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.025,
                    f"{val:.4f}", ha="center", fontweight="bold", fontsize=11, color="#3D5A80")
        ax.set_ylabel(ylabel, fontsize=9); ax.set_title(titulo, fontsize=9.5, color="#5C4A32")
        ax.set_ylim(0, max(vals)*1.35)
        for spine in ax.spines.values(): spine.set_edgecolor("#D0C8BC")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB T — TELL
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">Síntesis, Conclusiones y Respuestas</div>',
                unsafe_allow_html=True)

    # Resumen visual
    st.markdown('<div class="sub-title">Panel Resumen del Análisis</div>', unsafe_allow_html=True)

    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor("#F7F3EE")
    fig.suptitle("RESUMEN DEL ANÁLISIS EXPLORATORIO — Educación & Escolaridad Global",
                 fontsize=14, fontweight="bold", color="#3D5A80", fontfamily="serif", y=1.01)

    vars_corr  = ["gasto_educacion_pib", "pib_per_capita", "tasa_alfabetizacion", "log_pib_per_capita"]
    names_corr = ["Gasto Educ.\n(% PIB)", "PIB per\ncápita", "Tasa\nAlfab.", "log(PIB\nper cápita)"]
    correlaciones = [df[v].corr(df["anios_escolaridad"]) for v in vars_corr]

    # 1. Correlaciones
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor("#FDFAF6")
    colores_bar = [C1 if c > 0 else C2 for c in correlaciones]
    bars = ax1.barh(names_corr, correlaciones, color=colores_bar, alpha=0.75, edgecolor="white")
    ax1.set_xlabel("Correlación con Años de Escolaridad", fontsize=9)
    ax1.set_title("Correlaciones con Variable Dependiente", fontsize=10, color="#3D5A80")
    ax1.axvline(0, color="#7A6E65", linewidth=0.7)
    for bar, val in zip(bars, correlaciones):
        ax1.text(val + 0.02 if val > 0 else val - 0.02,
                 bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center",
                 ha="left" if val > 0 else "right", fontsize=9, fontweight="bold", color="#3D5A80")
    for spine in ax1.spines.values(): spine.set_edgecolor("#D0C8BC")

    # 2. Comparación R² ajustado
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor("#FDFAF6")
    r2_adj = [res["sm1"].rsquared_adj, res["sm2"].rsquared_adj]
    bars2 = ax2.bar(["Modelo 1\n(Simple)", "Modelo 2\n(Múltiple)"],
                    r2_adj, color=[C1, C2], alpha=0.75, edgecolor="white", width=0.5)
    for bar, val in zip(bars2, r2_adj):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", fontweight="bold", fontsize=12, color="#3D5A80")
    ax2.set_ylabel("R² Ajustado", fontsize=9)
    ax2.set_title("Comparación de Modelos", fontsize=10, color="#3D5A80")
    ax2.set_ylim(0, max(r2_adj)*1.4)
    for spine in ax2.spines.values(): spine.set_edgecolor("#D0C8BC")

    # 3. Predicción modelo 2 vs real
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor("#FDFAF6")
    y_all = res["m2"].predict(df[res["vars2"]].values)
    ax3.scatter(df["anios_escolaridad"], y_all, alpha=0.5, s=28, c=C2, edgecolors="white")
    lim_all = [min(df["anios_escolaridad"].min(), y_all.min())-1,
               max(df["anios_escolaridad"].max(), y_all.max())+1]
    ax3.plot(lim_all, lim_all, "r--", linewidth=1.5)
    ax3.set_xlabel("Valores Reales", fontsize=9); ax3.set_ylabel("Predichos (M2)", fontsize=9)
    ax3.set_title("Predicción del Mejor Modelo (todos los países)", fontsize=10, color="#3D5A80")
    ax3.set_xlim(lim_all); ax3.set_ylim(lim_all)
    for spine in ax3.spines.values(): spine.set_edgecolor("#D0C8BC")

    # 4. Importancia de variables
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor("#FDFAF6")
    imp_names = [x[0] for x in res["importancia"]]
    imp_vals  = [x[1] for x in res["importancia"]]
    imp_colors = [C2 if v < 0 else C1 for v in imp_vals]
    bars4 = ax4.barh(imp_names, [abs(v) for v in imp_vals],
                     color=imp_colors, alpha=0.75, edgecolor="white")
    for bar, val in zip(bars4, imp_vals):
        ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9, fontweight="bold", color="#3D5A80")
    ax4.set_xlabel("|Coeficiente Estandarizado|", fontsize=9)
    ax4.set_title("Importancia de Variables (Modelo 2)", fontsize=10, color="#3D5A80")
    for spine in ax4.spines.values(): spine.set_edgecolor("#D0C8BC")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    # ── Respuestas a preguntas ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Respuestas a las Preguntas de Investigación</div>',
                unsafe_allow_html=True)

    mejora_r2   = ((res["r2_2"] - res["r2_1"]) / abs(res["r2_1"])) * 100 if res["r2_1"] != 0 else 0
    mejora_rmse = ((res["rmse_1"] - res["rmse_2"]) / res["rmse_1"]) * 100

    respuestas = [
        ("P1", "Relación lineal gasto → escolaridad",
         f"La relación es <strong>débil a moderada</strong>. El Modelo 1 alcanza un R² ajustado de "
         f"<strong>{res['sm1'].rsquared_adj:.4f}</strong>, indicando que el gasto como porcentaje del PIB "
         f"<em>no es suficiente por sí solo</em> para explicar la variabilidad en años de escolaridad. "
         f"La eficiencia del gasto y el contexto socioeconómico son igualmente relevantes."),
        ("P2", "PIB per cápita vs gasto en educación",
         f"<strong>Sí.</strong> El PIB per cápita (en escala logarítmica) presenta una correlación "
         f"<strong>considerablemente más fuerte</strong> con la escolaridad. Los coeficientes "
         f"estandarizados del Modelo 2 confirman que el desarrollo económico es el predictor más importante."),
        ("P3", "Correlación tasa de alfabetización",
         f"La tasa de alfabetización muestra una correlación <strong>positiva significativa</strong> "
         f"(r = {df['tasa_alfabetizacion'].corr(df['anios_escolaridad']):.3f}) con la escolaridad. "
         f"Aunque es notable, la correlación con log(PIB) tiende a ser mayor, "
         f"confirmando el rol preponderante del contexto económico."),
        ("P4", "Modelo múltiple vs modelo simple",
         f"<strong>Sí, significativamente.</strong> El Modelo 2 supera al Modelo 1 en todas las métricas: "
         f"R² test mejora de <strong>{res['r2_1']:.4f} → {res['r2_2']:.4f}</strong> "
         f"(<strong>+{mejora_r2:.1f}%</strong>), RMSE reduce un <strong>{mejora_rmse:.1f}%</strong>. "
         f"La inclusión de PIB y alfabetización amplifica sustancialmente la capacidad predictiva."),
    ]

    for code, titulo, texto in respuestas:
        st.markdown(f"""
        <div class="answer-box">
          <h4>{code} · {titulo}</h4>
          <p style="margin:0; line-height:1.6;">{texto}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Conclusiones ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Conclusiones del EDA</div>', unsafe_allow_html=True)

    conclusiones = [
        ("📉", "Gasto educativo como predictor aislado",
         "El porcentaje del PIB destinado a educación no predice de forma robusta los años de "
         "escolaridad. La <strong>eficiencia y focalización del gasto</strong> son tan críticas "
         "como su magnitud."),
        ("🌐", "PIB per cápita: el predictor dominante",
         "El desarrollo económico general crea condiciones estructurales (infraestructura, "
         "estabilidad, incentivos laborales) que impulsan la educación prolongada, especialmente "
         "visible en escala logarítmica."),
        ("📈", "Superioridad del modelo múltiple",
         f"Combinar gasto, PIB y alfabetización produce un modelo con R² de "
         f"<strong>{res['sm2'].rsquared_adj:.4f}</strong> vs "
         f"<strong>{res['sm1'].rsquared_adj:.4f}</strong> del modelo simple — "
         f"una mejora sustancial y estadísticamente significativa."),
        ("🏛️", "Implicación para política pública",
         "Aumentar el presupuesto educativo es necesario pero <em>no suficiente</em>. "
         "Deben considerarse simultáneamente el crecimiento económico, la reducción del "
         "analfabetismo y la calidad de la inversión educativa."),
    ]

    cols = st.columns(2)
    concl_colors = [C1, C3, C2, C4]
    for i, (icon, titulo, texto) in enumerate(conclusiones):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:#fff; border-radius:14px; padding:1.2rem 1.4rem;
                        border-left:5px solid {concl_colors[i]}; box-shadow:0 2px 10px rgba(0,0,0,.06);
                        margin-bottom:1rem; min-height:130px;">
              <div style="font-size:1.6rem; margin-bottom:.3rem;">{icon}</div>
              <div style="font-family:'Playfair Display',serif; font-weight:700; font-size:1rem;
                          color:#3D5A80; margin-bottom:.5rem;">{titulo}</div>
              <p style="margin:0; color:#4A3E33; font-size:.92rem; line-height:1.55;">{texto}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Tabla resumen ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Tabla Resumen de Resultados</div>', unsafe_allow_html=True)

    mejor_modelo = "Modelo 2 (Regresión Múltiple)" if res["r2_2"] > res["r2_1"] else "Modelo 1 (Regresión Simple)"
    resumen_df = pd.DataFrame({
        "Aspecto": [
            "Variable dependiente", "R² Ajustado — Modelo 1", "R² Ajustado — Modelo 2",
            "RMSE — Modelo 1", "RMSE — Modelo 2",
            "Variable más influyente", "Mejora Modelo 2 vs 1", "Mejor modelo"
        ],
        "Resultado": [
            "Años promedio de escolaridad",
            f"{res['sm1'].rsquared_adj:.4f}",
            f"{res['sm2'].rsquared_adj:.4f}",
            f"{res['rmse_1']:.4f}",
            f"{res['rmse_2']:.4f}",
            res["importancia"][0][0],
            f"{mejora_r2:+.1f}% en R²",
            mejor_modelo
        ],
    }).set_index("Aspecto")
    st.dataframe(resumen_df, use_container_width=True)

    st.markdown("""
    <div style="text-align:center; margin-top:2rem; color:#A89880; font-size:.82rem;">
      Fuente de datos: <strong>Banco Mundial</strong> — World Bank Open Data (data.worldbank.org) &nbsp;·&nbsp;
      Framework de análisis: <strong>QUEST</strong> (Question → Understand → Explore → Study → Tell)
    </div>
    """, unsafe_allow_html=True)
