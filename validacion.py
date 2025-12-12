# ===============================================================
# 🌾 PREDWEEM v7.2 — ANN + Clasificación robusta con datos parciales
# - ANN → EMERREL diaria
# - Post-proceso: recorte negativos, suavizado opcional, acumulado
# - Percentiles d25–d95 calculados sobre la curva disponible (truncada)
# - Clasificación Temprano / Extendido + confianza (ALTA / MEDIA / BAJA)
# - Momento crítico en fecha calendario real
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, requests, xml.etree.ElementTree as ET
from pathlib import Path

# ===============================================================
# 🔧 CONFIG STREAMLIT
# ===============================================================
st.set_page_config(
    page_title="PREDWEEM v7.2 – Emergencia + Patrón (datos parciales)",
    layout="wide",
)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ===============================================================
# 🔧 FUNCIONES SEGURAS
# ===============================================================
def safe(fn, msg):
    try:
        return fn()
    except Exception as e:
        st.error(f"{msg}: {e}")
        return None

# ===============================================================
# 🔧 API METEOBAHIA (7 días) — OPCIONAL
# ===============================================================
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
            "Prec": _to_float(prec),
        })

    df = pd.DataFrame(rows).sort_values("Fecha").head(7)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    return df

# ===============================================================
# 🔧 ANN — Modelo de predicción emergencia
# ===============================================================
class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW = IW
        self.bIW = bIW
        self.LW = LW
        self.bLW = bLW
        # rango de entrenamiento original
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2*(X - self.input_min)/(self.input_max - self.input_min)-1

    def predict(self, Xreal):
        """
        Devuelve EMERREL cruda de la ANN y EMERAC cruda (cumsum).
        El post-procesamiento se hace por fuera.
        """
        Xn = self.normalize(Xreal)
        emer = []
        for x in Xn:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = self.LW @ a1 + self.bLW
            emer.append(np.tanh(z2))
        emer = (np.array(emer) + 1) / 2    # 0–1 (diario, crudo)
        emer_ac = np.cumsum(emer)          # acumulada cruda
        emerrel = np.diff(emer_ac, prepend=0)
        return emerrel, emer_ac

@st.cache_resource
def load_ann():
    IW  = np.load(BASE/"IW.npy")
    bIW = np.load(BASE/"bias_IW.npy")
    LW  = np.load(BASE/"LW.npy")
    bLW = np.load(BASE/"bias_out.npy")
    return PracticalANNModel(IW, bIW, LW, bLW)

modelo_ann = safe(lambda: load_ann(), "Error cargando pesos ANN")
if modelo_ann is None:
    st.stop()

# ===============================================================
# 🔧 POST-PROCESO EMERGENCIA (suavizado + recorte, SIN reescalar a 1)
# ===============================================================
def postprocess_emergence(emerrel_raw,
                          smooth=True,
                          window=3,
                          clip_zero=True):
    """
    Toma EMERREL cruda de la ANN y devuelve:
    - emerrel_proc: EMERREL suavizada / recortada
    - emerac_proc : EMERAC acumulada (no forzada a terminar en 1)
    """
    emer = np.array(emerrel_raw, dtype=float)

    # 1) Recortar posibles negativos
    if clip_zero:
        emer = np.maximum(emer, 0.0)

    # 2) Suavizado por media móvil
    if smooth and len(emer) > 1 and window > 1:
        window = int(window)
        window = max(1, min(window, len(emer)))
        if window > 1:
            kernel = np.ones(window, dtype=float) / window
            emer = np.convolve(emer, kernel, mode="same")

    # 3) EMERAC acumulada
    emerac = np.cumsum(emer)

    return emer, emerac

# ===============================================================
# 🔧 CARGAR MODELO DE CLUSTERS
# ===============================================================
def load_cluster_model():
    local_path = BASE/"modelo_cluster_d25_d50_d75_d95.pkl"
    alt_path   = Path("/mnt/data/modelo_cluster_d25_d50_d75_d95.pkl")

    if local_path.exists():
        path = local_path
    elif alt_path.exists():
        path = alt_path
    else:
        raise FileNotFoundError("modelo_cluster_d25_d50_d75_d95.pkl no encontrado")

    with open(path, "rb") as f:
        data = pickle.load(f)

    scaler        = data["scaler"]
    model         = data["model"]
    centroides    = data["centroides"]       # numpy (2,4)
    metricas_hist = data.get("metricas_hist", data.get("metricas", {}))
    labels_hist   = data.get("labels_hist",  data.get("labels", {}))

    return scaler, model, metricas_hist, labels_hist, centroides

cluster_pack = safe(lambda: load_cluster_model(),
    "Error cargando modelo_cluster_d25_d50_d75_d95.pkl")

if cluster_pack is None:
    st.stop()
else:
    scaler_cl, model_cl, metricas_hist, labels_hist, centroides = cluster_pack

# ===============================================================
# 🔧 FUNCIONES D25–D95 (sobre curva truncada)
# ===============================================================
def calc_percentiles_trunc(dias, emerac):
    """
    Calcula d25–d95 tomando como referencia el máximo disponible
    (curva potencialmente truncada).
    """
    if emerac.max() == 0:
        return None
    y = emerac / emerac.max()   # normaliza respecto a lo emergido hasta la fecha
    d25 = np.interp(0.25, y, dias)
    d50 = np.interp(0.50, y, dias)
    d75 = np.interp(0.75, y, dias)
    d95 = np.interp(0.95, y, dias)
    return d25, d50, d75, d95

def curva(vals):
    d25, d50, d75, d95 = vals
    x = np.array([d25, d50, d75, d95])
    y = np.array([0.25, 0.50, 0.75, 0.95])
    dias = np.arange(20, 200)
    curva = np.interp(dias, x, y)
    return dias, curva

# ===============================================================
# 🔧 RADAR MULTISERIES
# ===============================================================
def radar_multiseries(values_dict, labels, title):

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)

    colors = {
        "Año evaluado": "blue",
        "Temprano": "green",
        "Extendido": "orange"
    }

    for name, vals in values_dict.items():
        vals2 = list(vals) + [vals[0]]
        c = colors.get(name, None)
        ax.plot(angles, vals2, lw=2.5, label=name, color=c)
        ax.fill(angles, vals2, alpha=0.15, color=c)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0.1))

    return fig

# ===============================================================
# 🔧 UI PRINCIPAL
# ===============================================================
st.title("🌾 PREDWEEM v7.2 — ANN + Clasificación robusta con datos parciales")

# ---- Controles de post-proceso en el sidebar ----
with st.sidebar:
    st.header("Ajustes de emergencia")
    use_smoothing = st.checkbox("Suavizar EMERREL", value=True)
    window_size   = st.slider("Ventana de suavizado (días)", 1, 9, 3)
    clip_zero     = st.checkbox("Recortar negativos a 0", value=True)

st.subheader("📂 Carga de datos meteorológicos")

op_meteo = st.radio(
    "Fuente de datos meteorológicos:",
    ["Usar meteo_daily.csv interno", "Subir archivo externo (CSV/XLSX)"]
)

df = None

# ===============================================================
# 🚀 OPCIÓN 1 — USAR meteo_daily.csv INTERNO
# ===============================================================
if op_meteo == "Usar meteo_daily.csv interno":

    file_path = BASE / "meteo_daily.csv"
    if not file_path.exists():
        st.error("❌ No se encontró meteo_daily.csv en el directorio de la app.")
        st.stop()

    # Este archivo YA contiene Fecha → lectura directa
    df = pd.read_csv(file_path, parse_dates=["Fecha"])
    
    # Asegurar columna JD
    if "Julian_days" not in df.columns:
        df["Julian_days"] = df["Fecha"].dt.dayofyear
        
    df = df.sort_values("Julian_days")

# ===============================================================
# 🚀 SUBIR ARCHIVO METEOROLÓGICO EXTERNO (formato flexible)
# ===============================================================
else:
    up = st.file_uploader(
        "Subir archivo meteorológico externo",
        type=["csv", "xlsx", "xls"]
    )

    if up is not None:

        # ---- Lectura flexible según formato ----
        try:
            if up.name.lower().endswith(".csv"):
                df_raw = pd.read_csv(up, dtype=str)
            else:
                df_raw = pd.read_excel(up, dtype=str)
        except Exception as e:
            st.error(f"❌ Error leyendo el archivo: {e}")
            st.stop()

        # ---- Normalizar nombres de columnas ----
        df_raw.columns = [c.strip().lower() for c in df_raw.columns]

        # ---- Mapeo flexible (acepta JD o jd o Jd, etc.) ----
        col_map = {}
        for c in df_raw.columns:
            if c in ["jd", "dia_juliano", "julian", "diajuliano"]:
                col_map["jd"] = c
            if c in ["tmin", "tempmin", "min", "t_min"]:
                col_map["tmin"] = c
            if c in ["tmax", "tempmax", "max", "t_max"]:
                col_map["tmax"] = c
            if c in ["prec", "lluvia", "ppt", "rain"]:
                col_map["prec"] = c

        required = {"jd", "tmin", "tmax", "prec"}

        if not required.issubset(set(col_map.keys())):
            st.error(f"❌ El archivo debe contener las columnas: {required}")
            st.stop()

        # ---- Conversión coma→punto ----
        def to_float(x):
            try:
                return float(str(x).replace(",", "."))
            except:
                return np.nan

        # ---- Construcción del DataFrame estandarizado ----
        df = pd.DataFrame({
            "Julian_days": df_raw[col_map["jd"]].astype(int),
            "TMIN": df_raw[col_map["tmin"]].apply(to_float),
            "TMAX": df_raw[col_map["tmax"]].apply(to_float),
            "Prec": df_raw[col_map["prec"]].apply(to_float)
        })

        # ---- Generar Fecha a partir de JD ----
        year_default = pd.Timestamp.today().year
        df["Fecha"] = pd.to_datetime(df["Julian_days"], format="%j") \
                        .apply(lambda x: x.replace(year=year_default))

        df = df.sort_values("Julian_days")

# ===============================================================
# 🚀 VALIDACIÓN FINAL
# ===============================================================
if df is None:
    st.warning("Subí un archivo o seleccioná una fuente para continuar.")
    st.stop()

st.success("✅ Datos meteorológicos cargados correctamente.")
st.dataframe(df.head(), use_container_width=True)

# ===============================================================
# 🔧 ANN → EMERREL cruda + POST-PROCESO
# ===============================================================
X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
emerrel_raw, emerac_raw = modelo_ann.predict(X)

emerrel, emerac = postprocess_emergence(
    emerrel_raw,
    smooth=use_smoothing,
    window=window_size,
    clip_zero=clip_zero,
)

df["EMERREL"] = emerrel
df["EMERAC"]  = emerac

dias   = df["Julian_days"].to_numpy()
fechas = df["Fecha"].to_numpy()


# ===============================================================
# 🔥 MAPA DE RIESGO — VERSIÓN MODERNA E INTERACTIVA (SEGURO)
# ===============================================================
import plotly.express as px
import plotly.graph_objects as go

st.subheader("🔥 Mapa moderno e interactivo de riesgo de emergencia")

# ---------------------------------------------------------------
# 🛡️ Validación: asegurar que EMERREL está disponible
# ---------------------------------------------------------------
if "EMERREL" not in df.columns:
    st.error("No se encontró la columna EMERREL. Asegurate de ejecutar la ANN antes del mapa de riesgo.")
    st.stop()

# ---------------------------------------------------------------
# 🛡️ Crear columna Riesgo si no existe
# ---------------------------------------------------------------
if "Riesgo" not in df.columns:
    max_emerrel = df["EMERREL"].max()
    if max_emerrel > 0:
        df["Riesgo"] = df["EMERREL"] / max_emerrel
    else:
        df["Riesgo"] = 0.0

# ---------------------------------------------------------------
# 🛡️ Crear columna Nivel_riesgo si no existe
# ---------------------------------------------------------------
if "Nivel_riesgo" not in df.columns:
    def clasificar_riesgo(r):
        if r <= 0.10:
            return "Nulo"
        elif r <= 0.33:
            return "Bajo"
        elif r <= 0.66:
            return "Medio"
        else:
            return "Alto"
    df["Nivel_riesgo"] = df["Riesgo"].apply(clasificar_riesgo)

# ---------------------------------------------------------------
# Copia segura para el gráfico
# ---------------------------------------------------------------
df_risk = df.copy()
df_risk["Fecha_str"] = df_risk["Fecha"].dt.strftime("%d-%b")

# Día con riesgo máximo — protegido
if df_risk["Riesgo"].max() > 0:
    idx_max_riesgo = df_risk["Riesgo"].idxmax()
    fecha_max_riesgo = df_risk.loc[idx_max_riesgo, "Fecha"]
    valor_max_riesgo = df_risk.loc[idx_max_riesgo, "Riesgo"]
else:
    fecha_max_riesgo = None
    valor_max_riesgo = None

# ---------------------------------------------------------------
# 🟦 Sidebar visual
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎨 Estilo del mapa de riesgo")
    cmap = st.selectbox(
        "Mapa de colores",
        ["viridis", "plasma", "cividis", "turbo", "magma", "inferno", "cool", "warm"],
        index=0
    )
    tipo_barra = st.radio(
        "Modo de visualización",
        ["Rectángulo suave (recomendado)", "Barras finas tipo timeline"],
        index=0
    )

# ---------------------------------------------------------------
# 🔥 Generación del gráfico
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
            hovertemplate="<b>%{x|%d-%b}</b><br>Riesgo: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(showticklabels=False)

else:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_risk["Fecha"],
            y=df_risk["Riesgo"],
            marker=dict(color=df_risk["Riesgo"], colorscale=cmap, cmin=0, cmax=1),
            hovertemplate="<b>%{x|%d-%b}</b><br>Riesgo: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, 1], title="Riesgo")

# ---------------------------------------------------------------
# ⭐ Anotación segura
# ---------------------------------------------------------------
if fecha_max_riesgo is not None:
    fig.add_annotation(
        x=fecha_max_riesgo,
        y=1.05 if tipo_barra != "Rectángulo suave (recomendado)" else 0.6,
        text=f"⬆ Máximo riesgo ({valor_max_riesgo:.2f})",
        showarrow=False,
        font=dict(size=12, color="red")
    )

fig.update_layout(
    height=250,
    margin=dict(l=30, r=30, t=40, b=20),
    title="Mapa interactivo de riesgo diario de emergencia (0–1)",
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("📋 Tabla detallada de riesgo diario"):
    st.dataframe(
        df_risk[["Fecha", "EMERREL", "Riesgo", "Nivel_riesgo"]],
        use_container_width=True
    )


# ===============================================================
# 🎬 ANIMACIÓN DEL RIESGO DE EMERGENCIA DÍA A DÍA
# ===============================================================
import plotly.express as px
import plotly.graph_objects as go

st.subheader("🎬 Animación temporal del riesgo de emergencia (día por día)")

# ---------------------------------------------------------------
# 🛡 Validación
# ---------------------------------------------------------------
if "Riesgo" not in df.columns:
    st.error("No existe la columna Riesgo. Asegurate de ejecutar el cálculo previo.")
    st.stop()

# Preparación del DataFrame para animación
df_anim = df.copy()
df_anim["Fecha_str"] = df_anim["Fecha"].dt.strftime("%d-%b")

# ---------------------------------------------------------------
# 🎨 Selector de paleta de colores
# ---------------------------------------------------------------
with st.sidebar:
    cmap_anim = st.selectbox(
        "Mapa de colores para la animación",
        ["viridis", "plasma", "cividis", "turbo", "magma", "inferno", "icefire", "rdbu"],
        index=0,
        key="anim_cmap"
    )

# ---------------------------------------------------------------
# 🎬 Gráfico animado
# ---------------------------------------------------------------
fig_anim = px.scatter(
    df_anim,
    x="Fecha",
    y="Riesgo",
    animation_frame="Fecha_str",
    range_y=[0, 1],
    color="Riesgo",
    color_continuous_scale=cmap_anim,
    size=[12]*len(df_anim),   # puntos uniformes
    hover_data={"Fecha_str": True, "Riesgo": ":.2f"},
    labels={"Fecha": "Fecha calendario", "Riesgo": "Riesgo de emergencia (0–1)"}
)

# Línea base de riesgo completo
fig_anim.add_trace(
    go.Scatter(
        x=df_anim["Fecha"],
        y=df_anim["Riesgo"],
        mode="lines",
        line=dict(color="gray", width=1.5),
        name="Riesgo acumulado"
    )
)

# Mejora estética
fig_anim.update_layout(
    title="Evolución diaria del riesgo de emergencia",
    height=450,
    margin=dict(l=20, r=20, t=50, b=20),
)

# ---------------------------------------------------------------
# Controlar velocidad de animación
# ---------------------------------------------------------------
fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 300  # 300 ms entre frames

# Mostrar animación
st.plotly_chart(fig_anim, use_container_width=True)





# ===============================================================
# 🔧 GRÁFICOS MOSTRATIVOS EMERREL / EMERAC — AHORA EN FECHAS REALES
# ===============================================================
st.subheader("🔍 EMERGENCIA diaria y acumulada — Cruda vs Procesada (Fecha real)")

col_er, col_ac = st.columns(2)

# -------------------------------
# 🔵 EMERREL cruda vs procesada
# -------------------------------
with col_er:
    fig_er, ax_er = plt.subplots(figsize=(5,4))
    ax_er.plot(fechas, emerrel_raw, label="EMERREL cruda (ANN)", color="red", alpha=0.6)
    ax_er.plot(fechas, emerrel,     label="EMERREL procesada",   color="blue", linewidth=2)
    
    ax_er.set_xlabel("Fecha calendario real")
    ax_er.set_ylabel("EMERREL (fracción diaria)")
    ax_er.set_title("EMERREL: ANN vs post-proceso (en fechas reales)")
    
    ax_er.legend()
    fig_er.autofmt_xdate()
    st.pyplot(fig_er)

# -------------------------------
# 🟢 EMERAC cruda vs procesada
# -------------------------------
with col_ac:
    fig_ac, ax_ac = plt.subplots(figsize=(5,4))

    # Normalizadas si corresponde
    if emerac_raw[-1] > 0:
        ax_ac.plot(fechas, emerac_raw/emerac_raw[-1],
                   label="EMERAC cruda (normalizada)",
                   color="orange", alpha=0.6)
    else:
        ax_ac.plot(fechas, emerac_raw,
                   label="EMERAC cruda",
                   color="orange", alpha=0.6)

    if emerac[-1] > 0:
        ax_ac.plot(fechas, emerac/emerac[-1],
                   label="EMERAC procesada (normalizada)",
                   color="green", linewidth=2)
    else:
        ax_ac.plot(fechas, emerac,
                   label="EMERAC procesada",
                   color="green", linewidth=2)

    ax_ac.set_xlabel("Fecha calendario real")
    ax_ac.set_ylabel("EMERAC (0–1 relativo al período)")
    ax_ac.set_title("EMERAC: ANN vs post-proceso (en fechas reales)")

    ax_ac.legend()
    fig_ac.autofmt_xdate()
    st.pyplot(fig_ac)

# ===============================================================
# 🔧 COMPARACIÓN CON DATOS OBSERVADOS INDEPENDIENTES (EMERREL)
# ===============================================================
st.subheader("📂 Comparación con emergencia observada independiente (opcional)")

st.markdown("""
Podés subir un archivo con **emergencia relativa diaria observada** para 
compararla con la curva simulada por la ANN.

**Formato esperado (ejemplo 2015.xlsx):**
- Columna 1: `dia juliano` (entero, 1–365)
- Columna 2: `emer` (emergencia relativa diaria, fracción 0–1)
""")

file_obs = st.file_uploader(
    "Subir archivo de emergencia observada (xlsx/xls/csv)",
    type=["xlsx", "xls", "csv"],
    key="file_obs_emergencia"
)

if file_obs is not None:
    # ---------- Lectura robusta ----------
    try:
        if file_obs.name.lower().endswith((".xlsx", ".xls")):
            df_obs = pd.read_excel(file_obs)
        else:
            df_obs = pd.read_csv(file_obs)
    except Exception as e:
        st.error(f"Error leyendo archivo observado: {e}")
        df_obs = None

    if df_obs is not None:
        # Normalizar nombres de columnas
        cols_lower = {c.lower(): c for c in df_obs.columns}

        # Columna JD
        if "dia juliano" in cols_lower:
            col_jd = cols_lower["dia juliano"]
        elif "jd" in cols_lower:
            col_jd = cols_lower["jd"]
        else:
            # Asumir primera columna
            col_jd = df_obs.columns[0]

        # Columna EMERREL diaria
        if "emer" in cols_lower:
            col_emer = cols_lower["emer"]
        elif "emerrel" in cols_lower:
            col_emer = cols_lower["emerrel"]
        else:
            # Asumir segunda columna
            if len(df_obs.columns) > 1:
                col_emer = df_obs.columns[1]
            else:
                st.error("No se pudo identificar la columna de emergencia relativa (emer).")
                col_emer = None

        if col_emer is not None:
            # ---------- Procesamiento observado ----------
            df_obs = df_obs[[col_jd, col_emer]].copy()
            df_obs.columns = ["JD_obs", "EMERREL_obs"]

            # Ordenar por JD por seguridad
            df_obs = df_obs.sort_values("JD_obs")

            # Recortar negativos y calcular EMERAC observada
            df_obs["EMERREL_obs"] = df_obs["EMERREL_obs"].astype(float)
            df_obs["EMERREL_obs"] = df_obs["EMERREL_obs"].clip(lower=0.0)
            df_obs["EMERAC_obs"]  = df_obs["EMERREL_obs"].cumsum()

            max_obs = df_obs["EMERAC_obs"].max()
            if max_obs > 0:
                df_obs["EMERAC_obs_norm"] = df_obs["EMERAC_obs"] / max_obs
            else:
                df_obs["EMERAC_obs_norm"] = 0.0

            # ---------- EMERAC simulada normalizada ----------
            df_sim = df.copy()
            max_sim = df_sim["EMERAC"].max()
            if max_sim > 0:
                df_sim["EMERAC_sim_norm"] = df_sim["EMERAC"] / max_sim
            else:
                df_sim["EMERAC_sim_norm"] = 0.0

            # ---------- Emparejar por día juliano ----------
            merged = pd.merge(
                df_obs,
                df_sim[["Julian_days", "EMERAC_sim_norm"]],
                left_on="JD_obs",
                right_on="Julian_days",
                how="inner"
            )

            if len(merged) < 3:
                st.warning(
                    "Muy pocos puntos en común entre la curva observada y la simulada "
                    "(< 3 días coincidentes). No se calcula RMSE."
                )
            else:
                # ---------- Cálculo de RMSE ----------
                dif = merged["EMERAC_obs_norm"] - merged["EMERAC_sim_norm"]
                rmse = float(np.sqrt(np.mean(dif**2)))

                st.markdown("### 📏 Comparación EMERAC normalizada (observada vs simulada)")

                # ---------- Gráfico comparativo ----------
                fig_cmp, ax_cmp = plt.subplots(figsize=(9, 5))
                ax_cmp.plot(
                    merged["JD_obs"],
                    merged["EMERAC_obs_norm"],
                    label="EMERAC observada (normalizada)",
                    linewidth=2.5
                )
                ax_cmp.plot(
                    merged["JD_obs"],
                    merged["EMERAC_sim_norm"],
                    label="EMERAC simulada (normalizada)",
                    linewidth=2.5,
                    linestyle="--"
                )
                ax_cmp.set_xlabel("Día juliano")
                ax_cmp.set_ylabel("EMERAC normalizada (0–1)")
                ax_cmp.set_title("Curva observada vs simulada (EMERAC normalizada)")
                ax_cmp.legend()
                st.pyplot(fig_cmp)

                st.success(
                    f"**RMSE entre EMERAC observada y simulada (0–1):** {rmse:.3f}"
                )

                # Opcional: mostrar tabla resumida
                with st.expander("Ver datos emparejados (JD, EMERAC obs, EMERAC sim)", expanded=False):
                    st.dataframe(
                        merged[["JD_obs", "EMERAC_obs_norm", "EMERAC_sim_norm"]],
                        use_container_width=True
                    )

# ===============================================================
# 🔧 COBERTURA TEMPORAL Y CALIDAD DE INFORMACIÓN
# ===============================================================
st.subheader("🗓️ Cobertura temporal de los datos")

JD_START = int(dias.min())
JD_END   = int(dias.max())
TEMPORADA_MAX = 241  # 1-ene → 1-sep, aprox. temporada completa
cobertura = (JD_END - JD_START + 1) / TEMPORADA_MAX

st.write({
    "Fecha inicio datos": str(df["Fecha"].iloc[0].date()),
    "Fecha fin datos":    str(df["Fecha"].iloc[-1].date()),
    "JD inicio": JD_START,
    "JD fin":    JD_END,
    "Cobertura relativa de temporada (~1-ene a 1-oct)": f"{cobertura*100:.1f} %",
})








