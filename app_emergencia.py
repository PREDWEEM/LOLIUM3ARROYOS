# ===============================================================
# 🌾 PREDWEEM v7.1 — ANN + Clasificación Temprano/Extendido
# EMERREL postprocesada → EMERAC suave, monotónica y normalizada
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
    page_title="PREDWEEM v7.1 – Emergencia + Patrón",
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
# 🔧 API METEOBAHIA (7 días)
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
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2*(X - self.input_min)/(self.input_max - self.input_min)-1

    def predict(self, Xreal):
        """
        Devuelve EMERREL crudo de la red (sin post-procesamiento)
        y EMERAC crudo (simple cumsum).
        El post-procesamiento se hace afuera.
        """
        Xn = self.normalize(Xreal)
        emer = []
        for x in Xn:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = self.LW @ a1 + self.bLW
            emer.append(np.tanh(z2))
        emer = (np.array(emer) + 1) / 2           # 0–1 (diario, crudo)
        emer_ac = np.cumsum(emer)                 # acumulada cruda
        emerrel = np.diff(emer_ac, prepend=0)     # debería coincidir con emer
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
# 🔧 POST-PROCESO EMERGENCIA (suavizado + recorte + normalización)
# ===============================================================
def postprocess_emergence(emerrel_raw,
                          smooth=True,
                          window=3,
                          rescale_to_one=True,
                          clip_zero=True):
    """
    Toma EMERREL cruda de la ANN y devuelve:
    - emerrel_proc: EMERREL suavizada y consistente
    - emerac_proc : EMERAC acumulada, monotónica, terminando en 1 (si rescale=True)
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

    # 4) Reescalar para que EMERAC final sea 1
    if rescale_to_one and emerac[-1] > 0:
        emerac = emerac / emerac[-1]
        emer = np.diff(emerac, prepend=0)

    return emer, emerac

# ===============================================================
# 🔧 CARGAR MODELO DE CLUSTERS (SIN PANDAS)
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
    metricas_hist = data["metricas_hist"]    # dict
    labels_hist   = data["labels_hist"]      # dict

    return scaler, model, metricas_hist, labels_hist, centroides

cluster_pack = safe(lambda: load_cluster_model(),
    "Error cargando modelo_cluster_d25_d50_d75_d95.pkl")

if cluster_pack is None:
    st.stop()
else:
    scaler_cl, model_cl, metricas_hist, labels_hist, centroides = cluster_pack

# ===============================================================
# 🔧 FUNCIONES D25–D95
# ===============================================================
def calc_percentiles(dias, emerac):
    if emerac.max() == 0:
        return None
    y = emerac / emerac.max()
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
        ax.plot(angles, vals2, lw=2.5, label=name, color=colors[name])
        ax.fill(angles, vals2, alpha=0.15, color=colors[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0.1))

    return fig

# ===============================================================
# 🔧 UI PRINCIPAL
# ===============================================================
st.title("🌾 PREDWEEM v7.1 — ANN + Clasificación Temprano/Extendido")

# ---- Controles de post-proceso en el sidebar ----
with st.sidebar:
    st.header("Ajustes de emergencia")
    use_smoothing = st.checkbox("Suavizar EMERREL", value=True)
    window_size = st.slider("Ventana de suavizado (días)", min_value=1, max_value=9, value=3, step=1)
    rescale_to_one = st.checkbox("Forzar EMERAC final = 1", value=True)
    clip_zero = st.checkbox("Recortar negativos a 0", value=True)

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
# 🔧 ANN → EMERREL CRUDA + POST-PROCESO → EMERREL / EMERAC
# ===============================================================
X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
emerrel_raw, emerac_raw = modelo_ann.predict(X)

emerrel, emerac = postprocess_emergence(
    emerrel_raw,
    smooth=use_smoothing,
    window=window_size,
    rescale_to_one=rescale_to_one,
    clip_zero=clip_zero,
)

df["EMERREL"] = emerrel
df["EMERAC"] = emerac

# ===============================================================
# 🔧 PERCENTILES
# ===============================================================
dias = df["Julian_days"].to_numpy()
res = calc_percentiles(dias, emerac)

if res is None:
    st.error("No se pudieron calcular percentiles.")
    st.stop()

d25, d50, d75, d95 = res

st.subheader("📌 Percentiles simulados del año (sobre EMERAC post-procesada)")
st.write({
    "d25": round(d25, 1),
    "d50": round(d50, 1),
    "d75": round(d75, 1),
    "d95": round(d95, 1)
})

# ===============================================================
# 🔧 CLASIFICACIÓN
# ===============================================================
entrada_sc = scaler_cl.transform([[d25, d50, d75, d95]])
cl = int(model_cl.predict(entrada_sc)[0])

nombres = {1: "🌱 Temprano / Compacto", 0: "🌾 Extendido / Lento"}
colors  = {1: "green", 0: "orange"}

st.markdown(f"""
## 🎯 Patrón del año:
### <span style='color:{colors[cl]}; font-size:30px;'>{nombres[cl]}</span>
""", unsafe_allow_html=True)

# ===============================================================
# 🔧 CURVAS COMPARATIVAS
# ===============================================================
st.subheader("Curva del año vs centroides históricos")

dias_x, curva_x    = curva([d25, d50, d75, d95])
dias_ext, curva_ext   = curva(centroides[0])
dias_temp, curva_temp = curva(centroides[1])

fig, ax = plt.subplots(figsize=(9,5))
ax.plot(dias_x,   curva_x,   lw=3, label="Año evaluado",       color="blue")
ax.plot(dias_temp, curva_temp, lw=2, label="Centroide Temprano",   color="green")
ax.plot(dias_ext,  curva_ext,  lw=2, label="Centroide Extendido",  color="orange")
ax.set_xlabel("Día juliano")
ax.set_ylabel("EMERAC (0–1)")
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
    title="Radar — Año Evaluado vs Temprano vs Extendido"
)

st.pyplot(fig_rad)

# ===============================================================
# 🔧 GRÁFICO DE CERTEZA TEMPORAL DEL PATRÓN + MOMENTO CRÍTICO
# ===============================================================
st.subheader("📈 Certeza temporal del patrón (día por día)")

probs_temp = []
probs_ext  = []
dias_eval  = []

for i in range(5, len(df)):

    dias_parc   = dias[:i]
    emerac_parc = emerac[:i]

    res_parc = calc_percentiles(dias_parc, emerac_parc)
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

dia_crit = None
prob_crit = None
if idx_crit is not None:
    dia_crit  = dias_eval[idx_crit]
    prob_crit = probs_clase[idx_crit]

dia_max = None
prob_max = None
if idx_max is not None:
    dia_max  = dias_eval[idx_max]
    prob_max = probs_clase[idx_max]

# ----- Gráfico de evolución de probabilidad -----
figp, axp = plt.subplots(figsize=(9,5))
axp.plot(dias_eval, probs_temp, label="Probabilidad Temprano",  color="green",  lw=2.0)
axp.plot(dias_eval, probs_ext,  label="Probabilidad Extendido", color="orange", lw=2.0)

# Líneas verticales para momento crítico y máxima certeza
if dia_crit is not None:
    axp.axvline(dia_crit, color=color_clase, linestyle="--", linewidth=2,
                label=f"Momento crítico ({nombre_clase})")

if dia_max is not None and (dia_crit is None or dia_max != dia_crit):
    axp.axvline(dia_max, color="blue", linestyle=":", linewidth=2,
                label="Día de máxima certeza")

axp.set_ylim(0,1)
axp.set_xlabel("Día juliano")
axp.set_ylabel("Probabilidad")
axp.set_title("Evolución de la certeza del patrón")
axp.legend()
st.pyplot(figp)

# ----- Resumen numérico del momento crítico -----
st.markdown("### 🧠 Momento crítico de definición del patrón")

if dia_crit is not None:
    st.write(
        f"- **Patrón resultante:** {nombre_clase}  \n"
        f"- **Momento crítico (primer día con prob ≥ {UMBRAL:.0%}):** "
        f"día juliano **{int(dia_crit)}**  \n"
        f"- **Probabilidad en ese día:** {prob_crit:.2f}  \n"
        f"- **Día de máxima certeza:** {int(dia_max)} "
        f"(prob = {prob_max:.2f})"
    )
elif dia_max is not None:
    st.write(
        f"- **Patrón resultante:** {nombre_clase}  \n"
        f"- No se alcanza el umbral de {UMBRAL:.0%}, "
        f"pero la máxima certeza se logra en el día juliano "
        f"**{int(dia_max)}** con probabilidad **{prob_max:.2f}**."
    )
else:
    st.info("No se pudo calcular la evolución de probabilidad del patrón.")

# ===============================================================
# 🔧 GRÁFICOS MOSTRATIVOS — EMERREL cruda vs procesada
# ===============================================================
st.subheader("🔍 Comparación EMERREL — Cruda vs Procesada")

fig_er, ax_er = plt.subplots(figsize=(10,4))
ax_er.plot(df["Julian_days"], emerrel_raw, label="EMERREL cruda (ANN)", color="red", alpha=0.6)
ax_er.plot(df["Julian_days"], emerrel,     label="EMERREL procesada", color="blue", linewidth=2)
ax_er.set_xlabel("Día juliano")
ax_er.set_ylabel("EMERREL (fracción diaria)")
ax_er.set_title("EMERREL: salida ANN vs post-proceso")
ax_er.legend()
st.pyplot(fig_er)

# ===============================================================
# 🔧 GRÁFICOS MOSTRATIVOS — EMERAC cruda vs procesada
# ===============================================================
st.subheader("🔍 Comparación EMERAC — Cruda vs Procesada")

fig_ac, ax_ac = plt.subplots(figsize=(10,4))
ax_ac.plot(df["Julian_days"], emerac_raw/emerac_raw[-1], label="EMERAC cruda (normalizada)", color="orange", alpha=0.6)
ax_ac.plot(df["Julian_days"], emerac,                    label="EMERAC procesada", color="green", linewidth=2)
ax_ac.set_xlabel("Día juliano")
ax_ac.set_ylabel("EMERAC (0–1)")
ax_ac.set_title("EMERAC: acumulado ANN vs post-proceso")
ax_ac.legend()
st.pyplot(fig_ac)
