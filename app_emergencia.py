
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
import plotly.express as px
import plotly.graph_objects as go

st.subheader("🔥 Mapa moderno e interactivo de riesgo de emergencia")

# ---------------------------------------------------------------
# Cálculo del riesgo (0–1 normalizado)
# ---------------------------------------------------------------
if "Riesgo" not in df.columns:
    max_emerrel = df["EMERREL"].max()
    if max_emerrel > 0:
        df["Riesgo"] = df["EMERREL"] / max_emerrel
    else:
        df["Riesgo"] = 0.0

# ---------------------------------------------------------------
# Clasificación del nivel de riesgo
# ---------------------------------------------------------------
if "Nivel_riesgo" not in df.columns:

    def clasificar_riesgo(r):
        if r <= 0.10:  return "Nulo"
        if r <= 0.33:  return "Bajo"
        if r <= 0.66:  return "Medio"
        return "Alto"

    df["Nivel_riesgo"] = df["Riesgo"].apply(clasificar_riesgo)

# ---------------------------------------------------------------
# Preparación del dataframe
# ---------------------------------------------------------------
df_risk = df.copy()
df_risk["Fecha_str"] = df_risk["Fecha"].dt.strftime("%d-%b")

# Día de riesgo máximo
if df_risk["Riesgo"].max() > 0:
    idx_max_riesgo = df_risk["Riesgo"].idxmax()
    fecha_max_riesgo = df_risk.loc[idx_max_riesgo, "Fecha"]
    valor_max_riesgo = df_risk.loc[idx_max_riesgo, "Riesgo"]
else:
    fecha_max_riesgo = None
    valor_max_riesgo = None

# ---------------------------------------------------------------
# Opciones visuales en sidebar
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Construcción del gráfico
# ---------------------------------------------------------------
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

# Anotación del máximo
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

# Línea base
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

# Velocidad animación
fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 300

st.plotly_chart(fig_anim, use_container_width=True)


# ===============================================================
# 🔍 GRÁFICOS EMERREL y EMERAC (Fecha real)
# ===============================================================
st.subheader("🔍 EMERGENCIA diaria y acumulada — ANN vs post-proceso")

col_er, col_ac = st.columns(2)

# -----------------------------
# EMERREL
# -----------------------------
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

# -----------------------------
# EMERAC
# -----------------------------
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

# ---------------------------------------------------------------
# Cargar el archivo modelo_clusters_k3.pkl desde BASE
# ---------------------------------------------------------------
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


# ---------------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------------
def dtw_distance(a, b):
    """
    DTW simple para dos curvas 1D.
    """
    na, nb = len(a), len(b)
    dp = np.full((na+1, nb+1), np.inf)
    dp[0,0] = 0
    for i in range(1, na+1):
        for j in range(1, nb+1):
            cost = abs(a[i-1] - b[j-1])
            dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return dp[na, nb]


def interpolate_curve(jd, y, jd_common):
    """
    Interpola la curva EMERREL del año a la misma grilla usada para clusterizar.
    """
    return np.interp(jd_common, jd, y)


# ---------------------------------------------------------------
# PREPARAR curva del año
# ---------------------------------------------------------------
curve_interp_year = interpolate_curve(dias, emerrel, JD_COMMON)

# Obtener medoides reales
medA = curves_interp[medoids_k3[0]]
medB = curves_interp[medoids_k3[1]]
medC = curves_interp[medoids_k3[2]]

# Distancias DTW
dA = dtw_distance(curve_interp_year, medA)
dB = dtw_distance(curve_interp_year, medB)
dC = dtw_distance(curve_interp_year, medC)

dist_vector = np.array([dA, dB, dC])
cluster_pred = int(np.argmin(dist_vector))


# ---------------------------------------------------------------
# Nombres y colores de clusters
# ---------------------------------------------------------------
cluster_names = {
    2: "🌱 Temprano / Compacto",
    1: "🍂 Tardío / Extendido",
    0: "🌾 Intermedio / Bimodal"
}

cluster_colors = {
    2: "green",
    1: "orange",
    0: "blue"
}

cluster_desc = {
    2: "Curvas muy concentradas, 1–2 pulsos, emergencia temprana (feb–abr).",
    1: "Curvas extendidas con cola larga, riesgo elevado tardío (abr–jul).",
    0: "Curvas mixtas/bimodales: pulso temprano + rebrote otoñal claro."
}

# ---------------------------------------------------------------
# MOSTRAR RESULTADO
# ---------------------------------------------------------------
st.markdown(f"""
## 🎯 Patrón asignado por análisis funcional K=3:
### <span style='color:{cluster_colors[cluster_pred]}; font-size:30px;'>
{cluster_names[cluster_pred]}
</span>
""", unsafe_allow_html=True)

st.info(cluster_desc[cluster_pred])


# ---------------------------------------------------------------
# GRÁFICO — Curva del año vs SU medoide
# ---------------------------------------------------------------
st.subheader("📈 Curva del año vs medoide asignado")

fig_cmp, ax_cmp = plt.subplots(figsize=(9,5))

ax_cmp.plot(JD_COMMON, curve_interp_year,
            label="Año evaluado", color="black", linewidth=3)

medoids_dict = {0: medA, 1: medB, 2: medC}
ax_cmp.plot(JD_COMMON, medoids_dict[cluster_pred],
            label=f"Medoide patrón {cluster_pred}", 
            color=cluster_colors[cluster_pred],
            linewidth=3, linestyle="--")

ax_cmp.set_xlabel("Día Juliano (grilla unificada)")
ax_cmp.set_ylabel("EMERREL normalizada")
ax_cmp.legend()
st.pyplot(fig_cmp)


# ---------------------------------------------------------------
# GRÁFICO — Los 3 patrones juntos
# ---------------------------------------------------------------
st.subheader("🌈 Los tres patrones funcionales (medoides)")

fig_all, ax_all = plt.subplots(figsize=(9,5))

ax_all.plot(JD_COMMON, medA, label="Patrón 0 — Intermedio/Bimodal", color="blue")
ax_all.plot(JD_COMMON, medB, label="Patrón 1 — Tardío/Extendido",   color="orange")
ax_all.plot(JD_COMMON, medC, label="Patrón 2 — Temprano/Compacto",  color="green")
ax_all.plot(JD_COMMON, curve_interp_year, label="Año evaluado", color="black", linewidth=3)

ax_all.set_xlabel("Día Juliano")
ax_all.set_ylabel("EMERREL interpolada")
ax_all.legend()
st.pyplot(fig_all)


# ---------------------------------------------------------------
# Mostrar distancias numéricas
# ---------------------------------------------------------------
st.subheader("📏 Distancias DTW a los 3 patrones")
st.write({
    "Patrón 0 – Intermedio/Bimodal": float(dA),
    "Patrón 1 – Tardío/Extendido": float(dB),
    "Patrón 2 – Temprano/Compacto": float(dC)
})


# ===============================================================
# 🌱 DIAGNÓSTICO AGRONÓMICO SEGÚN EL PATRÓN K=3
# ===============================================================
st.header("🧠 Diagnóstico agronómico integrado (según patrón K=3)")

pat = cluster_pred  # alias

# ---------------------------------------------------------------
# DEFINICIONES AGRONÓMICAS POR CLUSTER
# ---------------------------------------------------------------
diagnostico = {
    2: {
        "titulo": "🌱 PATRÓN TEMPRANO / COMPACTO",
        "color": "green",
        "resumen": """
        Emergencia muy concentrada entre febrero y abril. 
        Uno o dos pulsos dominantes y poca cola tardía.
        Este patrón implica una ventana crítica MUY temprana y necesidad de actuar
        preventivamente (residuales y monitoreo intensivo temprano).
        """,
        "manejo": [
            "Aplicar residuales **antes del 10 de marzo**.",
            "Monitoreo intensivo durante febrero–marzo.",
            "Control postemergente casi siempre anticipado.",
            "Años favorables para lograr supresión temprana y reducir carga anual."
        ]
    },

    1: {
        "titulo": "🍂 PATRÓN TARDÍO / EXTENDIDO",
        "color": "orange",
        "resumen": """
        Curvas extensas o multimodales con cola prolongada hacia otoño-invierno.
        Alta proporción de emergencia después de abril.
        Requiere una estrategia de manejo prolongada y flexible.
        """,
        "manejo": [
            "Monitoreo extendido hasta **junio/julio**.",
            "Refuerzo obligatorio con **postemergente tardío**.",
            "Años de mayor riesgo de escapes.",
            "Evitar quedar con el lote 'descubierto' después de abril."
        ]
    },

    0: {
        "titulo": "🌾 PATRÓN INTERMEDIO / BIMODAL",
        "color": "blue",
        "resumen": """
        Emergencia mixta: un pulso temprano moderado y un segundo pulso otoñal fuerte.
        Indica un año de comportamiento dual: no es totalmente temprano ni totalmente tardío.
        """,
        "manejo": [
            "Usar un **residual temprano**, pero sin confiarse.",
            "Planificar **una segunda intervención** en otoño si el rebrote es alto.",
            "Aumenta la incertidumbre: monitoreo cada 7 días desde febrero a mayo.",
            "Importante ajustar estrategias según lluvias otoñales."
        ]
    }
}

# ---------------------------------------------------------------
# MOSTRAR DIAGNÓSTICO
# ---------------------------------------------------------------
titulo = diagnostico[pat]["titulo"]
color  = diagnostico[pat]["color"]

st.markdown(
    f"## <span style='color:{color}; font-size:28px;'>{titulo}</span>",
    unsafe_allow_html=True
)

st.markdown("### 📌 Resumen agronómico del patrón")
st.write(diagnostico[pat]["resumen"])

st.markdown("### 🔧 Recomendaciones de manejo prioritarias")

for rec in diagnostico[pat]["manejo"]:
    st.markdown(f"- {rec}")

# ---------------------------------------------------------------
# EXTRA: INTENSIDAD DE RIESGO SEGÚN LA CURVA REAL
# ---------------------------------------------------------------
st.subheader("🔍 Evaluación fina de intensidad emergente")

# Cálculo simple de indicadores
peak = emerrel.max()
dia_peak = fechas[np.argmax(emerrel)]

frac_tardia = emerrel[dias > 120].sum() / emerrel.sum() if emerrel.sum() > 0 else 0
frac_temprana = emerrel[dias < 90].sum() / emerrel.sum() if emerrel.sum() > 0 else 0

st.write({
    "Pico máximo (EMERREL)": float(peak),
    "Fecha del pico": str(dia_peak.date()),
    "Proporción temprana (< JD 90)": round(frac_temprana, 3),
    "Proporción tardía (> JD 120)": round(frac_tardia, 3)
})

# Comentarios interpretativos
if pat == 2:
    if frac_temprana > 0.60:
        st.success("Año **muy temprano**, con >60% de emergencia en la primera ventana crítica.")
    else:
        st.warning("Año temprano, pero con distribución un poco más extendida de lo esperado.")

elif pat == 1:
    if frac_tardia > 0.40:
        st.error("Año **altamente tardío**, gran presión de emergencia hacia invierno.")
    else:
        st.warning("Año tardío, pero con menor cola de lo habitual.")

elif pat == 0:
    if frac_temprana > 0.40 and frac_tardia > 0.25:
        st.info("Año **bimodal clásico**, con pulsos temprano y tardío bien marcados.")
    else:
        st.info("Patrón intermedio, aunque con menor fuerza en uno de los pulsos.")

# ---------------------------------------------------------------
# FIN DE LA APLICACIÓN
# ---------------------------------------------------------------
st.markdown("---")
st.markdown("""
### ✔ Aplicación finalizada  
Esta versión de **PREDWEEM vK3** incorpora ANN + análisis funcional completo + DTW K-Medoids  
y reemplaza totalmente al clasificador basado en percentiles.

Para comentarios o mejoras, simplemente indicá lo que necesitás.
""")






