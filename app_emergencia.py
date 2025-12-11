# ===============================================================
# 🌾 PREDWEEM vK3 — ANN + Clasificador funcional K=3 (DTW K-Medoids)
# Versión completa con riesgo, animación, comparación observada
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle, requests, xml.etree.ElementTree as ET
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------
# CONFIG STREAMLIT
# ---------------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM vK3 – ANN + Clasificador funcional K=3",
    layout="wide"
)

# Ocultar menú/herramientas
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------------
# FUNCIONES SEGURAS
# ---------------------------------------------------------------
def safe(fn, msg):
    try:
        return fn()
    except Exception as e:
        st.error(f"{msg}: {e}")
        return None

# ---------------------------------------------------------------
# API METEOBAHIA (7 días) — OPCIONAL
# ---------------------------------------------------------------
API_URL = "https://meteobahia.com.ar/scripts/forecast/for-ta.xml"

def _to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return None

@st.cache_data(ttl=900)
def fetch_forecast():
    r = requests.get(API_URL, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.content)

    rows = []
    for d in root.findall(".//forecast/tabular/day"):
        fecha  = d.find("fecha").get("value")
        tmax   = d.find("tmax").get("value")
        tmin   = d.find("tmin").get("value")
        prec   = d.find("precip").get("value")
        rows.append({
            "Fecha": pd.to_datetime(fecha),
            "TMAX": _to_float(tmax),
            "TMIN": _to_float(tmin),
            "Prec": _to_float(prec)
        })

    df = pd.DataFrame(rows).sort_values("Fecha").head(7)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    return df


# ===============================================================
# 🔥 ANN — MODELO DE EMERGENCIA
# ===============================================================
class PracticalANNModel:
    """
    ANN oficial PREDWEEM — produce EMERREL cruda diaria
    """
    def __init__(self, IW, bIW, LW, bLW):
        self.IW = IW
        self.bIW = bIW
        self.LW = LW
        self.bLW = bLW
        # Rango de entrenamiento original
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2*(X - self.input_min)/(self.input_max - self.input_min)-1

    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        emer = []
        for x in Xn:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = self.LW @ a1 + self.bLW
            emer.append(np.tanh(z2))
        emer = (np.array(emer) + 1) / 2
        emer_ac = np.cumsum(emer)
        emerrel = np.diff(emer_ac, prepend=0)
        return emerrel, emer_ac


@st.cache_resource
def load_ann():
    IW  = np.load(BASE/"IW.npy")
    bIW = np.load(BASE/"bias_IW.npy")
    LW  = np.load(BASE/"LW.npy")
    bLW = np.load(BASE/"bias_out.npy")
    return PracticalANNModel(IW, bIW, LW, bLW)


modelo_ann = safe(lambda: load_ann(), "❌ No se pudieron cargar los pesos ANN")
if modelo_ann is None:
    st.stop()


# ===============================================================
# POST-PROCESO EMERGENCIA
# ===============================================================
def postprocess_emergence(emerrel_raw, smooth=True, window=3, clip_zero=True):
    emer = np.array(emerrel_raw, dtype=float)

    if clip_zero:
        emer = np.maximum(emer, 0)

    if smooth and window > 1:
        w = np.ones(window) / window
        emer = np.convolve(emer, w, mode="same")

    emerac = np.cumsum(emer)
    return emer, emerac


# ===============================================================
# UI — Cargar datos meteorológicos
# ===============================================================
st.title("🌾 PREDWEEM vK3 – ANN + Clasificador funcional K=3")

with st.sidebar:
    st.header("Ajustes de emergencia")
    use_smoothing = st.checkbox("Suavizar EMERREL", True)
    window_size   = st.slider("Ventana de suavizado", 1, 9, 3)
    clip_zero     = st.checkbox("Recortar negativos a 0", True)

st.subheader("📂 Carga de datos meteorológicos")

op_meteo = st.radio(
    "Fuente de datos:",
    ["Usar meteo_daily.csv interno", "Subir archivo externo (CSV/XLSX)"]
)

df = None

# ---------------------------------------------------------------
# OPCIÓN 1 → meteo_daily.csv interno
# ---------------------------------------------------------------
if op_meteo == "Usar meteo_daily.csv interno":
    file_path = BASE/"meteo_daily.csv"
    if not file_path.exists():
        st.error("❌ No se encontró meteo_daily.csv.")
        st.stop()

    df = pd.read_csv(file_path, parse_dates=["Fecha"])

    if "Julian_days" not in df.columns:
        df["Julian_days"] = df["Fecha"].dt.dayofyear

    df = df.sort_values("Julian_days")


# ---------------------------------------------------------------
# OPCIÓN 2 → archivo externo
# ---------------------------------------------------------------
else:
    up = st.file_uploader("Subir archivo meteorológico", type=["csv", "xlsx", "xls"])
    if up is not None:

        try:
            if up.name.lower().endswith(".csv"):
                df_raw = pd.read_csv(up, dtype=str)
            else:
                df_raw = pd.read_excel(up, dtype=str)
        except Exception as e:
            st.error(f"Error leyendo archivo: {e}")
            st.stop()

        df_raw.columns = [c.strip().lower() for c in df_raw.columns]

        col_map = {}

        for c in df_raw.columns:
            if c in ["jd","dia_juliano","julian","diajuliano"]:
                col_map["jd"] = c
            if c in ["tmin","tempmin","min","t_min"]:
                col_map["tmin"] = c
            if c in ["tmax","tempmax","max","t_max"]:
                col_map["tmax"] = c
            if c in ["prec","lluvia","ppt","rain"]:
                col_map["prec"] = c

        required = {"jd","tmin","tmax","prec"}

        if not required.issubset(col_map.keys()):
            st.error(f"❌ Columnas requeridas: {required}")
            st.stop()

        def to_float(x):
            try:
                return float(str(x).replace(",","."))
            except:
                return np.nan

        year_default = pd.Timestamp.today().year

        df = pd.DataFrame({
            "Julian_days": df_raw[col_map["jd"]].astype(int),
            "TMIN": df_raw[col_map["tmin"]].apply(to_float),
            "TMAX": df_raw[col_map["tmax"]].apply(to_float),
            "Prec": df_raw[col_map["prec"]].apply(to_float)
        })

        df["Fecha"] = pd.to_datetime(df["Julian_days"], format="%j") \
                        .apply(lambda x: x.replace(year=year_default))

        df = df.sort_values("Julian_days")

# ---------------------------------------------------------------
# VALIDACIÓN
# ---------------------------------------------------------------
if df is None:
    st.warning("Subí un archivo o seleccioná una fuente para continuar.")
    st.stop()

st.success("Datos meteorológicos cargados correctamente.")
st.dataframe(df.head(), use_container_width=True)

# ===============================================================
# ANN → EMERREL cruda → post-proceso
# ===============================================================
X = df[["Julian_days","TMAX","TMIN","Prec"]].to_numpy(float)
emerrel_raw, emerac_raw = modelo_ann.predict(X)

emerrel, emerac = postprocess_emergence(
    emerrel_raw,
    smooth=use_smoothing,
    window=window_size,
    clip_zero=clip_zero
)

df["EMERREL"] = emerrel
df["EMERAC"]  = emerac

dias   = df["Julian_days"].to_numpy()
fechas = df["Fecha"].to_numpy()

# ===============================================================
# 🔥 MAPA DE RIESGO — MODERNO E INTERACTIVO
# ===============================================================
st.subheader("🔥 Mapa moderno e interactivo de riesgo de emergencia")

if "Riesgo" not in df.columns:
    max_emerrel = df["EMERREL"].max()
    if max_emerrel > 0:
        df["Riesgo"] = df["EMERREL"] / max_emerrel
    else:
        df["Riesgo"] = 0.0

if "Nivel_riesgo" not in df.columns:

    def clasificar_riesgo(r):
        if r <= 0.10:  return "Nulo"
        if r <= 0.33:  return "Bajo"
        if r <= 0.66:  return "Medio"
        return "Alto"

    df["Nivel_riesgo"] = df["Riesgo"].apply(clasificar_riesgo)

df_risk = df.copy()
df_risk["Fecha_str"] = df_risk["Fecha"].dt.strftime("%d-%b")

if df_risk["Riesgo"].max() > 0:
    idx_max_riesgo = df_risk["Riesgo"].idxmax()
    fecha_max_riesgo = df_risk.loc[idx_max_riesgo, "Fecha"]
    valor_max_riesgo = df_risk.loc[idx_max_riesgo, "Riesgo"]
else:
    fecha_max_riesgo = None
    valor_max_riesgo = None

with st.sidebar:
    st.markdown("### 🎨 Estilo del mapa de riesgo")
    cmap = st.selectbox(
        "Mapa de colores",
        ["viridis","plasma","cividis","turbo","magma","inferno","cool","warm"],
        index=0
    )
    tipo_barra = st.radio(
        "Tipo de visualización",
        ["Rectángulo suave (recomendado)", "Barras tipo timeline"],
        index=0
    )

if tipo_barra == "Rectángulo suave (recomendado)":
    fig = go.Figure(
        data=go.Heatmap(
            z=[df_risk["Riesgo"].values],
            x=df_risk["Fecha"],
            y=["Riesgo"],
            colorscale=cmap,
            zmin=0, zmax=1,
            showscale=True,
            hovertemplate="<b>%{x|%d-%b}</b><br>Riesgo: %{z:.2f}<extra></extra>"
        )
    )
    fig.update_yaxes(showticklabels=False)
else:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_risk["Fecha"],
        y=df_risk["Riesgo"],
        marker=dict(color=df_risk["Riesgo"], colorscale=cmap, cmin=0, cmax=1),
        hovertemplate="<b>%{x|%d-%b}</b><br>Riesgo: %{y:.2f}<extra></extra>"
    ))
    fig.update_yaxes(range=[0,1], title="Riesgo")

if fecha_max_riesgo is not None:
    fig.add_annotation(
        x=fecha_max_riesgo,
        y=1.05 if tipo_barra != "Rectángulo suave (recomendado)" else 0.6,
        text=f"⬆ Máximo riesgo ({valor_max_riesgo:.2f})",
        showarrow=False,
        font=dict(size=12, color="red")
    )

fig.update_layout(
    height=260,
    margin=dict(l=30,r=30,t=40,b=20),
    title="Mapa interactivo de riesgo diario de emergencia (0–1)"
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("📋 Tabla detallada de riesgo"):
    st.dataframe(
        df_risk[["Fecha","EMERREL","Riesgo","Nivel_riesgo"]],
        use_container_width=True
    )

# ===============================================================
# 🎬 ANIMACIÓN TEMPORAL DEL RIESGO
# ===============================================================
st.subheader("🎬 Animación temporal del riesgo día por día")

df_anim = df.copy()
df_anim["Fecha_str"] = df_anim["Fecha"].dt.strftime("%d-%b")

with st.sidebar:
    cmap_anim = st.selectbox(
        "Mapa de colores para animación",
        ["viridis","plasma","cividis","turbo","magma","inferno","icefire","rdbu"],
        key="anim_cmap"
    )

fig_anim = px.scatter(
    df_anim,
    x="Fecha",
    y="Riesgo",
    animation_frame="Fecha_str",
    range_y=[0,1],
    color="Riesgo",
    color_continuous_scale=cmap_anim,
    size=[12]*len(df_anim),
    hover_data={"Fecha_str":True,"Riesgo":":.2f"},
    labels={"Fecha":"Fecha","Riesgo":"Riesgo (0–1)"}
)

fig_anim.add_trace(go.Scatter(
    x=df_anim["Fecha"],
    y=df_anim["Riesgo"],
    mode="lines",
    line=dict(color="gray", width=1.5)
))

fig_anim.update_layout(
    title="Evolución diaria del riesgo de emergencia",
    height=450
)

fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 300

st.plotly_chart(fig_anim, use_container_width=True)

# ===============================================================
# 🔍 GRÁFICOS EMERREL y EMERAC (Fecha real)
# ===============================================================
st.subheader("🔍 EMERGENCIA diaria y acumulada — ANN vs post-proceso")

col_er, col_ac = st.columns(2)

with col_er:
    fig_er, ax_er = plt.subplots(figsize=(5,4))
    ax_er.plot(fechas, emerrel_raw, label="EMERREL cruda (ANN)", color="red", alpha=0.6)
    ax_er.plot(fechas, emerrel,     label="EMERREL procesada",   color="blue", linewidth=2)

    ax_er.set_xlabel("Fecha")
    ax_er.set_ylabel("EMERREL")
    ax_er.set_title("EMERREL — ANN vs procesada")
    ax_er.legend()
    fig_er.autofmt_xdate()
    st.pyplot(fig_er)

with col_ac:
    fig_ac, ax_ac = plt.subplots(figsize=(5,4))

    if emerac_raw[-1] > 0:
        ax_ac.plot(fechas, emerac_raw/emerac_raw[-1], label="EMERAC cruda (norm)", color="orange", alpha=0.6)
    else:
        ax_ac.plot(fechas, emerac_raw,                 label="EMERAC cruda", color="orange", alpha=0.6)

    if emerac[-1] > 0:
        ax_ac.plot(fechas, emerac/emerac[-1], label="EMERAC procesada (norm)", color="green", linewidth=2)
    else:
        ax_ac.plot(fechas, emerac,             label="EMERAC procesada", color="green", linewidth=2)

    ax_ac.set_xlabel("Fecha")
    ax_ac.set_ylabel("EMERAC (0–1)")
    ax_ac.set_title("EMERAC — ANN vs procesada")
    ax_ac.legend()
    fig_ac.autofmt_xdate()
    st.pyplot(fig_ac)

# ===============================================================
# 🔥 CLASIFICADOR FUNCIONAL K=3 (DTW + K-Medoids)
# ===============================================================
st.header("🌾 Clasificación funcional K=3 basada en curvas EMERREL (DTW)")

cluster_path = BASE/"modelo_clusters_k3.pkl"

if not cluster_path.exists():
    st.error("❌ No se encontró modelo_clusters_k3.pkl en el directorio local.")
    st.stop()

with open(cluster_path, "rb") as f:
    cluster_model = pickle.load(f)

names_k3        = cluster_model["names"]
labels_k3       = np.array(cluster_model["labels_k3"])
medoids_k3      = cluster_model["medoids_k3"]
DTW_hist        = np.array(cluster_model["DTW_matrix"])
JD_COMMON       = np.array(cluster_model["JD_common"])
curves_interp   = np.array(cluster_model["curves_interp"])   # matriz (N, T)

# ===============================================================
# FUNCIONES AUXILIARES K=3
# ===============================================================
def dtw_distance(a, b):
    na, nb = len(a), len(b)
    dp = np.full((na+1, nb+1), np.inf)
    dp[0,0] = 0
    for i in range(1, na+1):
        for j in range(1, nb+1):
            cost = abs(a[i-1] - b[j-1])
            dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return dp[na, nb]


def interpolate_curve(jd, y, jd_common):
    return np.interp(jd_common, jd, y)

# Normalizar EMERREL simulada
if emerrel.max() > 0:
    emerrel_norm = emerrel / emerrel.max()
else:
    emerrel_norm = emerrel.copy()

curve_interp_year = interpolate_curve(dias, emerrel_norm, JD_COMMON)

# Medoides por cluster
med0 = curves_interp[medoids_k3[0]]   # Cluster 0
med1 = curves_interp[medoids_k3[1]]   # Cluster 1
med2 = curves_interp[medoids_k3[2]]   # Cluster 2

d0 = dtw_distance(curve_interp_year, med0)
d1 = dtw_distance(curve_interp_year, med1)
d2 = dtw_distance(curve_interp_year, med2)

dist_vector = np.array([d0, d1, d2])
cluster_pred = int(np.argmin(dist_vector))

# Definición ordenada de patrones (1 = temprano compacto)
cluster_names = {
    1: "🌱 Temprano / Compacto",
    0: "🌾 Intermedio / Bimodal",
    2: "🍂 Tardío / Extendido"
}

cluster_colors = {
    1: "green",
    0: "blue",
    2: "orange"
}

cluster_desc = {
    1: (
        "Patrón temprano, compacto y altamente concentrado.\n"
        "- Emergencia dominante en febrero–marzo.\n"
        "- Pico marcado antes de abril.\n"
        "- Requiere manejo anticipado (residuales tempranos, monitoreo intenso)."
    ),
    0: (
        "Patrón mixto/bimodal con dos pulsos.\n"
        "- Pulso temprano moderado + pulso otoñal fuerte.\n"
        "- Exige doble estrategia: residual temprano + postemergente táctico en otoño.\n"
        "- Es uno de los patrones más desafiantes para el manejo."
    ),
    2: (
        "Patrón tardío/extenso con prolongada emergencia otoñal.\n"
        "- Actividad principal abril–junio.\n"
        "- Cola larga que puede extender el riesgo hasta invierno.\n"
        "- Requiere monitoreo extendido y flexibilidad en el control postemergente."
    )
}

st.markdown(f"""
## 🎯 Patrón asignado por análisis funcional K=3:
### <span style='color:{cluster_colors.get(cluster_pred, "black")}; font-size:30px;'>
{cluster_names.get(cluster_pred, f"Cluster {cluster_pred}")}
</span>
""", unsafe_allow_html=True)

st.info(cluster_desc.get(cluster_pred, "Patrón no documentado en la descripción."))

# ===============================================================
# 🔧 Función robusta para convertir fechas
# ===============================================================
def safe_to_date(x):
    if x is None:
        return "No definido"
    try:
        return str(pd.to_datetime(x).date())
    except:
        pass
    try:
        jd = int(x)
        year = pd.Timestamp.today().year
        fecha = pd.to_datetime(f"{jd}", format="%j").replace(year=year)
        return str(fecha.date())
    except:
        pass
    return str(x)

# ===============================================================
# 🔍 Métricas de emergencia y pico
# ===============================================================
peak = emerrel.max() if len(emerrel) > 0 else 0

if len(emerrel) > 0:
    idx_peak = int(np.argmax(emerrel))
    dia_peak = fechas[idx_peak]
else:
    dia_peak = None

fecha_pico_segura = safe_to_date(dia_peak)

if emerrel.sum() > 0:
    frac_temprana = emerrel[dias < 90].sum() / emerrel.sum()
    frac_tardia   = emerrel[dias > 120].sum() / emerrel.sum()
else:
    frac_temprana = 0.0
    frac_tardia   = 0.0

st.subheader("📋 Resumen del diagnóstico funcional")

resumen_diagnostico = {
    "Patrón asignado": cluster_names.get(cluster_pred, f"Cluster {cluster_pred}"),
    "Cluster ID": int(cluster_pred),
    "Pico máximo (EMERREL)": float(peak),
    "Fecha del pico": fecha_pico_segura,
    "Proporción temprana (< JD 90)": round(frac_temprana, 3),
    "Proporción tardía (> JD 120)": round(frac_tardia, 3),
}
st.write(resumen_diagnostico)

# ===============================================================
# 🧠 Interpretación agronómica del patrón K=3
# ===============================================================
st.subheader("🧠 Interpretación agronómica del patrón K=3")

if cluster_pred == 1:
    # Temprano / Compacto
    if frac_temprana > 0.60:
        st.success("🌱 **Año muy temprano**, con >60% de emergencia en la primera ventana crítica.")
    else:
        st.warning("🌱 Año temprano, pero con una distribución algo más extendida de lo normal.")

elif cluster_pred == 2:
    # Tardío / Extendido
    if frac_tardia > 0.40:
        st.error("🍂 **Año altamente tardío**, con fuerte concentración de emergencia hacia invierno.")
    else:
        st.warning("🍂 Año tardío, pero con menor extensión que otros casos históricos.")

elif cluster_pred == 0:
    # Intermedio / Bimodal
    if frac_temprana > 0.40 and frac_tardia > 0.25:
        st.info("🌾 **Año bimodal clásico**, con pulsos temprano y tardío bien marcados.")
    else:
        st.info("🌾 Patrón intermedio, con menor dominancia de uno de los pulsos.")

# ===============================================================
# 📈 Gráficos de patrones vs año
# ===============================================================
st.subheader("📈 Curva del año vs medoide asignado")

fig_cmp, ax_cmp = plt.subplots(figsize=(9,5))

ax_cmp.plot(JD_COMMON, curve_interp_year,
            label="Año evaluado (normalizado)", color="black", linewidth=3)

med_dict = {0: med0, 1: med1, 2: med2}

ax_cmp.plot(JD_COMMON, med_dict[cluster_pred],
            label=f"Medoide del patrón {cluster_pred}",
            color=cluster_colors.get(cluster_pred, "gray"),
            linewidth=3, linestyle="--")

ax_cmp.set_xlabel("Día Juliano (grilla unificada)")
ax_cmp.set_ylabel("EMERREL normalizada")
ax_cmp.legend()
st.pyplot(fig_cmp)

st.subheader("🌈 Los tres patrones funcionales (medoides)")

fig_all, ax_all = plt.subplots(figsize=(9,5))

ax_all.plot(JD_COMMON, med0, label="Patrón 0 — Intermedio/Bimodal", color="blue")
ax_all.plot(JD_COMMON, med1, label="Patrón 1 — Temprano/Compacto",   color="green")
ax_all.plot(JD_COMMON, med2, label="Patrón 2 — Tardío/Extendido",    color="orange")
ax_all.plot(JD_COMMON, curve_interp_year, label="Año evaluado", color="black", linewidth=2)

ax_all.set_xlabel("Día Juliano")
ax_all.set_ylabel("EMERREL normalizada")
ax_all.legend()
st.pyplot(fig_all)

# ===============================================================
# 📏 Distancias DTW
# ===============================================================
st.subheader("📏 Distancias DTW a los 3 patrones")
st.write({
    "Patrón 0 – Intermedio/Bimodal": float(d0),
    "Patrón 1 – Temprano/Compacto": float(d1),
    "Patrón 2 – Tardío/Extendido": float(d2)
})

# ===============================================================
# 🌾 DESCRIPCIÓN AGRONÓMICA DETALLADA DE LOS 3 PATRONES K=3
# ===============================================================

st.subheader("🌱 Descripción agronómica ampliada del patrón asignado")

descripcion_agronomica_detallada = {
    1: """
### 🟢 **Patrón 1 — Temprano / Compacto**
Este patrón representa los años de **mayor riesgo inicial** para la competencia y fallas de control.

#### 🔬 Dinámica de emergencia
- Emergencia **muy concentrada en un corto período** (generalmente 20–35 días).
- Pico marcado **entre fines de febrero y mediados de marzo**.
- Casi nula emergencia posterior a abril.
- Relación fuerte con:
  - precipitaciones de verano,
  - suelos con buena humedad superficial (post-nap),
  - temperaturas estables y templadas en febrero.

#### 🎯 Implicancias para el manejo
- **La ventana crítica ocurre muy temprano**, por lo que:
  - Los **herbicidas residuales pre-siembra o pre-emergentes** deben estar activos desde fines de febrero.
  - Los **tratamientos postemergentes** pierden efectividad si se aplican después del pico.
- Recomendado en cultivos de invierno:
  - Residuales de alta persistencia.
  - Monitoreo inmediato en la primera quincena de marzo.
- Riesgo elevado de:
  - Lotes sucios tempranos.
  - Interferencia inicial con cultivos de siembra otoñal temprana.

#### 📌 Síntesis agronómica
Un patrón que **recompensa el manejo anticipado** y castiga la demora; si se controla temprano, el año puede ser fácil.
""",

    0: """
### 🔵 **Patrón 0 — Intermedio / Bimodal**
Es el patrón **más complejo** desde el manejo debido a su dualidad.

#### 🔬 Dinámica de emergencia
- Dos picos bien reconocibles:
  - **Uno temprano** (marzo).
  - **Uno tardío** (mediados de mayo o incluso junio).
- Entre ambos picos se observa una meseta o período de baja actividad.
- Alta variabilidad interanual dentro del grupo.
- Asociado a:
  - alternancia de ciclos húmedo–seco,
  - temperaturas otoñales erráticas,
  - rearme de humedad superficial tardío.

#### 🎯 Implicancias para el manejo
- Requiere **doble estrategia**:
  1. **Protección residual temprana**, especialmente si hay cultivos de fina.
  2. **Refuerzos post-emergentes** o residuales de segunda ventana hacia mayo–junio.
- El mayor desafío:
  - Percepción engañosa: luego del primer pico parece que el año “termina”, pero llega el **segundo pulso fuerte**.
- Importante:
  - Mantener monitoreo durante todo abril–mayo.
  - Considerar productos con persistencia media-alta.

#### 📌 Síntesis agronómica
Patrón “trampas” para el manejo: **si no se atiende el segundo pulso**, el lote se descontrola. Manejo escalonado obligatorio.
""",

    2: """
### 🟠 **Patrón 2 — Tardío / Extendido**
Años donde la emergencia mayor ocurre **tarde y durante un largo período**.

#### 🔬 Dinámica de emergencia
- Emergencia creciente a partir de abril.
- Pico marcado en **mayo** (incluso junio en algunos años).
- Cola extensa que puede llegar a julio.
- Asociado a:
  - otoños húmedos,
  - años fríos con baja evaporación,
  - suelos que retienen humedad por largos períodos.

#### 🎯 Implicancias para el manejo
- Los residuales aplicados en febrero–marzo **no alcanzan** a cubrir la ventana efectiva.
- Se vuelve imprescindible:
  - Programar **postemergentes estratégicos** en varias rondas.
  - Flexibilidad en fechas de aplicación (no depender de un único tratamiento).
  - Mantener monitoreo continuo durante mayo y junio.
- Impactos económicos:
  - Aumenta el costo del control.
  - Impacta cultivos tardíos (cebada sembrada tarde, verdeos, pasturas).

#### 📌 Síntesis agronómica
Año “largo y agotador”: la emergencia **no es alta en intensidad**, pero sí en **duración**, exigiendo persistencia del manejo.
"""
}

# Mostrar descripción final
st.markdown(descripcion_agronomica_detallada.get(
    cluster_pred,
    "No hay descripción disponible para este patrón."
))


