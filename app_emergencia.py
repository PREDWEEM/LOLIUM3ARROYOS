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
    window_size   = st.slider("Ventana de suavizado (días)", min_value=1, max_value=9, value=3, step=1)
    clip_zero     = st.checkbox("Recortar negativos a 0", value=True)

fuente = st.radio("Fuente de datos:", [
    "Histórico (meteo_daily.csv)",
    "Subir archivo CSV"
])

df = None
if fuente == "Histórico (meteo_daily.csv)":
    if not (BASE/"meteo_daily.csv").exists():
        st.error("No se encontró meteo_daily.csv")
        st.stop()
    df = pd.read_csv(BASE/"meteo_daily.csv", parse_dates=["Fecha"])
else:
    up = st.file_uploader("Subir meteo_history.csv", type=["csv"])
    if up:
        df = pd.read_csv(up, parse_dates=["Fecha"])

if df is None:
    st.stop()

df["Julian_days"] = df["Fecha"].dt.dayofyear
df = df.sort_values("Fecha")

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
# 🔧 GRÁFICOS MOSTRATIVOS EMERREL / EMERAC
# ===============================================================
st.subheader("🔍 EMERGENCIA diaria y acumulada — Cruda vs Procesada")

col_er, col_ac = st.columns(2)

with col_er:
    fig_er, ax_er = plt.subplots(figsize=(5,4))
    ax_er.plot(dias, emerrel_raw, label="EMERREL cruda (ANN)", color="red", alpha=0.6)
    ax_er.plot(dias, emerrel,     label="EMERREL procesada",   color="blue", linewidth=2)
    ax_er.set_xlabel("Día juliano")
    ax_er.set_ylabel("EMERREL (fracción diaria)")
    ax_er.set_title("EMERREL: ANN vs post-proceso")
    ax_er.legend()
    st.pyplot(fig_er)

with col_ac:
    fig_ac, ax_ac = plt.subplots(figsize=(5,4))
    if emerac_raw[-1] > 0:
        ax_ac.plot(dias, emerac_raw/emerac_raw[-1], label="EMERAC cruda (normalizada)", color="orange", alpha=0.6)
    else:
        ax_ac.plot(dias, emerac_raw, label="EMERAC cruda", color="orange", alpha=0.6)
    if emerac[-1] > 0:
        ax_ac.plot(dias, emerac/emerac[-1], label="EMERAC procesada (normalizada)", color="green", linewidth=2)
    else:
        ax_ac.plot(dias, emerac, label="EMERAC procesada", color="green", linewidth=2)
    ax_ac.set_xlabel("Día juliano")
    ax_ac.set_ylabel("EMERAC (0–1 relativo al período)")
    ax_ac.set_title("EMERAC: ANN vs post-proceso")
    ax_ac.legend()
    st.pyplot(fig_ac)

# ===============================================================
# 🔧 COBERTURA TEMPORAL Y CALIDAD DE INFORMACIÓN
# ===============================================================
st.subheader("🗓️ Cobertura temporal de los datos")

JD_START = int(dias.min())
JD_END   = int(dias.max())
TEMPORADA_MAX = 274  # 1-ene → 1-oct, aprox. temporada completa
cobertura = (JD_END - JD_START + 1) / TEMPORADA_MAX

st.write({
    "Fecha inicio datos": str(df["Fecha"].iloc[0].date()),
    "Fecha fin datos":    str(df["Fecha"].iloc[-1].date()),
    "JD inicio": JD_START,
    "JD fin":    JD_END,
    "Cobertura relativa de temporada (~1-ene a 1-oct)": f"{cobertura*100:.1f} %",
})

# ===============================================================
# 🔧 PERCENTILES SOBRE CURVA TRUNCADA
# ===============================================================
res = calc_percentiles_trunc(dias, emerac)

if res is None:
    st.error("No se pudieron calcular percentiles.")
    st.stop()

d25, d50, d75, d95 = res

st.subheader("📌 Percentiles simulados del año (sobre lo emergido hasta la fecha)")
st.write({
    "d25 (del período observado)": round(d25, 1),
    "d50 (del período observado)": round(d50, 1),
    "d75 (del período observado)": round(d75, 1),
    "d95 (del período observado)": round(d95, 1)
})

# ===============================================================
# 🔧 CLASIFICACIÓN + CONFIANZA
# ===============================================================
entrada_sc = scaler_cl.transform([[d25, d50, d75, d95]])
cl = int(model_cl.predict(entrada_sc)[0])

nombres = {1: "🌱 Temprano / Compacto", 0: "🌾 Extendido / Lento"}
colors  = {1: "green", 0: "orange"}

st.markdown(f"""
## 🎯 Patrón del año (basado en datos parciales):
### <span style='color:{colors[cl]}; font-size:30px;'>{nombres[cl]}</span>
""", unsafe_allow_html=True)

# ===============================================================
# 🔧 CURVAS COMPARATIVAS
# ===============================================================
st.subheader("Curva del año vs centroides históricos (forma normalizada)")

dias_x, curva_x    = curva([d25, d50, d75, d95])
dias_ext, curva_ext   = curva(centroides[0])
dias_temp, curva_temp = curva(centroides[1])

fig, ax = plt.subplots(figsize=(9,5))
ax.plot(dias_x,   curva_x,   lw=3, label="Año evaluado (parcial)",   color="blue")
ax.plot(dias_temp, curva_temp, lw=2, label="Centroide Temprano",     color="green")
ax.plot(dias_ext,  curva_ext,  lw=2, label="Centroide Extendido",    color="orange")
ax.set_xlabel("Día juliano (escala normalizada)")
ax.set_ylabel("EMERAC relativa (0–1)")
ax.legend()
st.pyplot(fig)

# ===============================================================
# 🔧 RADAR MULTISERIES
# ===============================================================
st.subheader("Radar comparativo del patrón")

vals_year = [d25, d50, d75, d95]
vals_temp = list(centroides[1])
vals_ext  = list(centroides[0])

fig_rad = radar_multiseries(
    {
        "Año evaluado": vals_year,
        "Temprano": vals_temp,
        "Extendido": vals_ext
    },
    labels=["d25", "d50", "d75", "d95"],
    title="Radar — Año Evaluado (parcial) vs Temprano vs Extendido"
)

st.pyplot(fig_rad)

# ===============================================================
# 🔧 CERTEZA TEMPORAL DEL PATRÓN + MOMENTO CRÍTICO (FECHA REAL)
# ===============================================================
st.subheader("📈 Certeza temporal del patrón (día por día)")

probs_temp = []
probs_ext  = []
dias_eval  = []
fechas_eval = []

for i in range(5, len(df)):

    dias_parc   = dias[:i]
    emerac_parc = emerac[:i]
    fechas_parc = df["Fecha"].iloc[:i]

    res_parc = calc_percentiles_trunc(dias_parc, emerac_parc)
    if res_parc is None:
        continue

    d25_p, d50_p, d75_p, d95_p = res_parc

    entrada_sc_parc = scaler_cl.transform([[d25_p, d50_p, d75_p, d95_p]])

    # Distancias a centroides en espacio escaleado
    d_ext = np.linalg.norm(
        entrada_sc_parc - model_cl.cluster_centers_[0].reshape(1, -1)
    )
    d_temp = np.linalg.norm(
        entrada_sc_parc - model_cl.cluster_centers_[1].reshape(1, -1)
    )

    # Probabilidades ~ inverso de la distancia
    if d_ext == 0 and d_temp == 0:
        prob_temp = 0.5
        prob_ext  = 0.5
    else:
        w_ext  = 1.0 / (d_ext + 1e-9)
        w_temp = 1.0 / (d_temp + 1e-9)
        s = w_ext + w_temp
        prob_temp = w_temp / s
        prob_ext  = w_ext / s

    dias_eval.append(dias_parc[-1])
    fechas_eval.append(fechas_parc.iloc[-1])
    probs_temp.append(prob_temp)
    probs_ext.append(prob_ext)

# ----- Determinar patrón resultante (cl ya calculado arriba) -----
if cl == 1:
    probs_clase   = probs_temp
    nombre_clase  = "Temprano / Compacto"
    color_clase   = "green"
else:
    probs_clase   = probs_ext
    nombre_clase  = "Extendido / Lento"
    color_clase   = "orange"

# ----- Momento crítico y máxima certeza -----
UMBRAL = 0.8  # umbral de decisión

idx_crit = next((i for i, p in enumerate(probs_clase) if p >= UMBRAL), None)
idx_max  = int(np.argmax(probs_clase)) if len(probs_clase) > 0 else None

fecha_crit = None
prob_crit  = None
if idx_crit is not None:
    fecha_crit = fechas_eval[idx_crit]
    prob_crit  = probs_clase[idx_crit]

fecha_max = None
prob_max  = None
if idx_max is not None:
    fecha_max = fechas_eval[idx_max]
    prob_max  = probs_clase[idx_max]

# ----- Gráfico con fechas reales -----
figp, axp = plt.subplots(figsize=(9,5))

axp.plot(fechas_eval, probs_temp, label="Probabilidad Temprano",  color="green",  lw=2.0)
axp.plot(fechas_eval, probs_ext,  label="Probabilidad Extendido", color="orange", lw=2.0)

if fecha_crit is not None:
    axp.axvline(fecha_crit, color=color_clase, linestyle="--", linewidth=2,
                label=f"Momento crítico ({nombre_clase})")

if fecha_max is not None and (fecha_crit is None or fecha_max != fecha_crit):
    axp.axvline(fecha_max, color="blue", linestyle=":", linewidth=2,
                label="Fecha máxima certeza")

axp.set_ylim(0,1)
axp.set_xlabel("Fecha calendario real")
axp.set_ylabel("Probabilidad")
axp.set_title("Evolución de la certeza del patrón")
axp.legend()
figp.autofmt_xdate()
st.pyplot(figp)

# ===============================================================
# 🔧 RESUMEN DE MOMENTO CRÍTICO + CONFIANZA GLOBAL
# ===============================================================
st.markdown("### 🧠 Momento crítico de definición del patrón")

if fecha_crit is not None:
    st.write(
        f"- **Patrón resultante:** {nombre_clase}  \n"
        f"- **Momento crítico (primer día con prob ≥ {UMBRAL:.0%}):** "
        f"**{fecha_crit.strftime('%d-%b')}**  \n"
        f"- **Probabilidad en ese día:** {prob_crit:.2f}  \n"
        f"- **Fecha de máxima certeza:** {fecha_max.strftime('%d-%b')} "
        f"(prob = {prob_max:.2f})"
    )
elif fecha_max is not None:
    st.write(
        f"- **Patrón resultante:** {nombre_clase}  \n"
        f"- No se alcanza el umbral de {UMBRAL:.0%}, "
        f"pero la máxima certeza se logra el "
        f"**{fecha_max.strftime('%d-%b')}** con probabilidad **{prob_max:.2f}**."
    )
else:
    st.info("No se pudo calcular la evolución de probabilidad del patrón.")

# ----- Evaluación de confianza global (ALTA / MEDIA / BAJA) -----
if prob_max is not None:
    # Regla heurística combinando cobertura temporal + probabilidad máxima
    if cobertura >= 0.7 and prob_max >= 0.8:
        nivel_conf = "ALTA"
        color_conf = "green"
    elif cobertura >= 0.4 and prob_max >= 0.65:
        nivel_conf = "MEDIA"
        color_conf = "orange"
    else:
        nivel_conf = "BAJA"
        color_conf = "red"

    st.markdown(
        f"### 🔒 Nivel de confianza de la clasificación: "
        f"<span style='color:{color_conf}; font-size:26px;'>{nivel_conf}</span>",
        unsafe_allow_html=True
    )
    st.write(
        f"- **Cobertura temporal:** {cobertura*100:.1f} % de la temporada estimada  \n"
        f"- **Probabilidad máxima del patrón resultante:** {prob_max:.2f}"
    )
else:
    st.info("No se pudo estimar un nivel de confianza para la clasificación.")


# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Comparación Temprano vs Tardío — PREDWEEM",
    layout="wide"
)

st.title("🌾 Comparación de Patrones — PREDWEEM (EMERAC 0–1 y EMERREL)")

# ============================================================
# CONFIGURACIÓN DE PATRONES Y ARCHIVOS
# ============================================================

tempranos = ["2008","2012","2013","2025"]
tardios   = ["2009","2010","2011","2014","2015","2023","2024"]

files = {yr: f"/mnt/data/{yr}.xlsx" for yr in tempranos + tardios}

# ============================================================
# FUNCIONES
# ============================================================

def jd_to_date(jd, year=2025):
    """Convertir día Juliano → fecha calendario."""
    return datetime(year,1,1) + timedelta(days=int(jd)-1)

def load_data(path):
    """Carga emergencias y devuelve df con EMERAC y EMERREL."""
    df = pd.read_excel(path, header=None)
    df.columns=["jd","emerrel"]
    df=df.sort_values("jd")
    df["emerac"]=df["emerrel"].cumsum()
    total=df["emerac"].iloc[-1]

    df["emerac_rel"] = df["emerac"] / total if total > 0 else 0
    df["date"] = df["jd"].apply(jd_to_date)

    return df

# ============================================================
# GRÁFICO EMERAC RELATIVA (0–1) — FECHA CALENDARIO
# ============================================================

st.subheader("📈 EMERAC Relativa (0–1) — Eje temporal en fecha calendario")

fig1, ax1 = plt.subplots(figsize=(12,6))

for yr in tempranos:
    df = load_data(files[yr])
    ax1.plot(df["date"], df["emerac_rel"], color="tab:blue", label=f"{yr} (Temprano)")

for yr in tardios:
    df = load_data(files[yr])
    ax1.plot(df["date"], df["emerac_rel"], color="tab:red", label=f"{yr} (Tardío)")

ax1.set_ylim(0,1)
ax1.set_xlabel("Fecha calendario (2025 como referencia)")
ax1.set_ylabel("Emergencia acumulada relativa (0–1)")
ax1.set_title("EMERAC Relativa — Tempranos (Azul) vs Tardíos (Rojo)")
ax1.grid(True)
ax1.legend()

st.pyplot(fig1)

# ============================================================
# GRÁFICO EMERREL (Emergencia diaria)
# ============================================================

st.subheader("📉 EMERREL — Emergencia Relativa Diaria")

fig2, ax2 = plt.subplots(figsize=(12,6))

for yr in tempranos:
    df = load_data(files[yr])
    ax2.plot(df["date"], df["emerrel"], color="tab:blue", label=f"{yr} (Temprano)")

for yr in tardios:
    df = load_data(files[yr])
    ax2.plot(df["date"], df["emerrel"], color="tab:red", label=f"{yr} (Tardío)")

ax2.set_xlabel("Fecha calendario (2025 como referencia)")
ax2.set_ylabel("Emergencia relativa diaria")
ax2.set_title("EMERREL — Tempranos vs Tardíos")
ax2.grid(True)
ax2.legend()

st.pyplot(fig2)

# ============================================================
# TABLA COMPARATIVA
# ============================================================

st.subheader("📊 Tabla comparativa de patrones")

data = {
    "Rasgo": [
        "Inicio",
        "Velocidad",
        "Concentración",
        "Fecha 80% EMERAC",
        "Fecha d95 (fin del proceso)",
        "Implicancias de manejo"
    ],
    "Patrón Temprano": [
        "Febrero – inicio de marzo",
        "Muy rápida",
        "1–2 pulsos concentrados",
        "Abril",
        "Fin de abril – inicios de mayo",
        "Residuales + control temprano (antes del 10 de marzo)"
    ],
    "Patrón Tardío / Extendido": [
        "Mitad de marzo – abril",
        "Lenta y escalonada",
        "2–4 pulsos, forma extendida",
        "Mayo – junio",
        "Junio – agosto",
        "Monitoreo prolongado + postemergente tardío"
    ]
}

df_table = pd.DataFrame(data)
st.dataframe(df_table, use_container_width=True)

# DESCARGA DE TABLA
csv = df_table.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Descargar tabla en CSV",
    data=csv,
    file_name="comparacion_patrones.csv",
    mime="text/csv"
)

