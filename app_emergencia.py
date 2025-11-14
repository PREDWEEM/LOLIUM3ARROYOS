# ===============================================================
# 🌾 PREDWEEM v7 — ANN + Clasificación Temprano/Extendido
# Integración ANN + Clustering d25–d95 (sin Pandas en el modelo)
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
    page_title="PREDWEEM v7 – Emergencia + Patrón",
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
# 🔧 API METEOBAHIA (7 días)  — OPCIONAL
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
        return 2*(X - self.input_min)/(self.input_max - self.input_min) - 1

    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        emer = []
        for x in Xn:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = self.LW @ a1 + self.bLW
            emer.append(np.tanh(z2))
        emer = (np.array(emer) + 1) / 2  # 0–1
        emer_ac = np.cumsum(emer)
        emerrel = np.diff(emer_ac, prepend=0)
        return emerrel, emer_ac

@st.cache_resource
def load_ann():
    IW  = np.load(BASE / "IW.npy")
    bIW = np.load(BASE / "bias_IW.npy")
    LW  = np.load(BASE / "LW.npy")
    bLW = np.load(BASE / "bias_out.npy")
    return PracticalANNModel(IW, bIW, LW, bLW)

modelo_ann = safe(lambda: load_ann(), "Error cargando pesos ANN")
if modelo_ann is None:
    st.stop()

# ===============================================================
# 🔧 CARGAR MODELO DE CLUSTERS (SIN PANDAS EN EL PKL)
# ===============================================================
def load_cluster_model():
    local_path = BASE / "modelo_cluster_d25_d50_d75_d95.pkl"
    alt_path   = Path("/mnt/data/modelo_cluster_d25_d50_d75_d95.pkl")

    if local_path.exists():
        path = local_path
    elif alt_path.exists():
        path = alt_path
    else:
        raise FileNotFoundError("modelo_cluster_d25_d50_d75_d95.pkl no encontrado")

    with open(path, "rb") as f:
        data = pickle.load(f)

    scaler     = data["scaler"]
    model      = data["model"]
    centroides = data["centroides"]  # numpy (2,4)

    # Soportar ambas versiones de clave: 'metricas_hist' vs 'metricas'
    metricas_hist = data.get("metricas_hist", data.get("metricas", {}))
    labels_hist   = data.get("labels_hist", data.get("labels", {}))

    return scaler, model, metricas_hist, labels_hist, centroides

cluster_pack = safe(
    lambda: load_cluster_model(),
    "Error cargando modelo_cluster_d25_d50_d75_d95.pkl"
)

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

    fig = plt.figure(figsize=(6, 6))
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
st.title("🌾 PREDWEEM v7 — ANN + Clasificación Temprano/Extendido")

fuente = st.radio("Fuente de datos:", [
    "Histórico (meteo_daily.csv)",
    "Subir archivo CSV"
])

df = None
if fuente == "Histórico (meteo_daily.csv)":
    if not (BASE / "meteo_daily.csv").exists():
        st.error("No se encontró meteo_daily.csv")
        st.stop()
    df = pd.read_csv(BASE / "meteo_daily.csv", parse_dates=["Fecha"])
else:
    up = st.file_uploader("Subir meteo_history.csv", type=["csv"])
    if up:
        df = pd.read_csv(up, parse_dates=["Fecha"])

if df is None:
    st.stop()

df["Julian_days"] = df["Fecha"].dt.dayofyear
df = df.sort_values("Fecha")

# ===============================================================
# 🔧 ANN → EMERREL + EMERAC
# ===============================================================
X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
emerrel, emerac = modelo_ann.predict(X)

df["EMERREL"] = emerrel
df["EMERAC"] = emerac

# ===============================================================
# 🔧 PERCENTILES DEL AÑO EVALUADO
# ===============================================================
dias = df["Julian_days"].to_numpy()
res = calc_percentiles(dias, emerac)

if res is None:
    st.error("No se pudieron calcular percentiles.")
    st.stop()

d25, d50, d75, d95 = res

st.subheader("📌 Percentiles simulados del año")
st.write({
    "d25": round(d25, 1),
    "d50": round(d50, 1),
    "d75": round(d75, 1),
    "d95": round(d95, 1)
})

# ===============================================================
# 🔧 CLASIFICACIÓN (Temprano vs Extendido)
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
# 🔧 CURVAS COMPARATIVAS (Año vs Centroides)
# ===============================================================
st.subheader("Curva del año vs centroides históricos")

dias_x, curva_x   = curva([d25, d50, d75, d95])
dias_ext, curva_ext   = curva(centroides[0])
dias_temp, curva_temp = curva(centroides[1])

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(dias_x,   curva_x,   lw=3, label="Año evaluado",      color="blue")
ax.plot(dias_temp, curva_temp, lw=2, label="Centroide Temprano",  color="green")
ax.plot(dias_ext,  curva_ext,  lw=2, label="Centroide Extendido", color="orange")
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
# 🔧 GRÁFICO DE CERTEZA TEMPORAL DEL PATRÓN
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

    # Distancias en el espacio escaleado
    centro_ext  = model_cl.cluster_centers_[0].reshape(1, -1)
    centro_temp = model_cl.cluster_centers_[1].reshape(1, -1)

    d_ext  = np.linalg.norm(entrada_sc_parc - centro_ext)
    d_temp = np.linalg.norm(entrada_sc_parc - centro_temp)

    # Probabilidades ~ inverso de la distancia
    if d_ext == 0 and d_temp == 0:
        p_temp = 0.5
        p_ext  = 0.5
    else:
        w_ext  = 1.0 / (d_ext + 1e-9)
        w_temp = 1.0 / (d_temp + 1e-9)
        s = w_ext + w_temp
        p_temp = w_temp / s
        p_ext  = w_ext / s

    dias_eval.append(dias_parc[-1])
    probs_temp.append(p_temp)
    probs_ext.append(p_ext)

figp, axp = plt.subplots(figsize=(9, 5))
axp.plot(dias_eval, probs_temp, label="Probabilidad Temprano",  color="green",  lw=2.5)
axp.plot(dias_eval, probs_ext,  label="Probabilidad Extendido", color="orange", lw=2.5)

axp.set_ylim(0, 1)
axp.set_xlabel("Día juliano")
axp.set_ylabel("Probabilidad")
axp.set_title("Evolución de la certeza del patrón")
axp.legend()
st.pyplot(figp)

# ===============================================================
# 🔧 CLASIFICACIÓN DE AÑOS HISTÓRICOS (USANDO EL MODELO)
# ===============================================================
st.subheader("📚 Clasificación de años históricos (modelo de clusters)")

if isinstance(metricas_hist, dict) and len(metricas_hist) > 0:
    filas = []
    for k, v in metricas_hist.items():
        # k puede ser año (int o str); v = [d25,d50,d75,d95] o similar
        try:
            d25_h, d50_h, d75_h, d95_h = v
        except Exception:
            continue

        entrada_sc_h = scaler_cl.transform([[d25_h, d50_h, d75_h, d95_h]])
        cl_h = int(model_cl.predict(entrada_sc_h)[0])

        centro_ext  = model_cl.cluster_centers_[0].reshape(1, -1)
        centro_temp = model_cl.cluster_centers_[1].reshape(1, -1)

        d_ext_h  = np.linalg.norm(entrada_sc_h - centro_ext)
        d_temp_h = np.linalg.norm(entrada_sc_h - centro_temp)

        # probas históricas similares a las actuales
        w_ext_h  = 1.0 / (d_ext_h + 1e-9)
        w_temp_h = 1.0 / (d_temp_h + 1e-9)
        s_h = w_ext_h + w_temp_h
        p_temp_h = w_temp_h / s_h
        p_ext_h  = w_ext_h / s_h

        filas.append({
            "Año": k,
            "d25": round(d25_h, 1),
            "d50": round(d50_h, 1),
            "d75": round(d75_h, 1),
            "d95": round(d95_h, 1),
            "Cluster": cl_h,
            "Patrón": nombres.get(cl_h, str(cl_h)),
            "Prob_Temprano": round(float(p_temp_h), 3),
            "Prob_Extendido": round(float(p_ext_h), 3),
        })

    if filas:
        df_hist = pd.DataFrame(filas)

        col1, col2 = st.columns([3, 1])

        with col1:
            st.dataframe(
                df_hist.sort_values("Año"),
                use_container_width=True
            )

        with col2:
            resumen = (
                df_hist.groupby("Patrón")
                .size()
                .reset_index(name="N_años")
            )
            st.markdown("**Resumen por patrón**")
            st.dataframe(resumen, use_container_width=True)
    else:
        st.info("No hay métricas históricas válidas en el modelo de clusters.")
else:
    st.info("El archivo de clusters no contiene 'metricas_hist' / 'metricas'.")

# ===============================================================
# FIN DEL SCRIPT
# ===============================================================


