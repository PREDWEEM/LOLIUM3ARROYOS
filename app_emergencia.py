# ===============================================================
# 🌾 PREDWEEM v7 — ANN + Clasificación Temprano/Extendido
# Integración ANN + Modelo de Clusters d25–d95
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle, requests, time, xml.etree.ElementTree as ET
from pathlib import Path

# ======================
# CONFIGURACIÓN STREAMLIT
# ======================
st.set_page_config(
    page_title="PREDWEEM v7 – Emergencia + Patrón",
    layout="wide",
)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ======================
# FUNCIONES SEGURAS
# ======================
def safe(fn, msg):
    try:
        return fn()
    except Exception as e:
        st.error(f"{msg}: {e}")
        return None

# ======================
# API METEOBahia (7 días)
# ======================
API_URL = "https://meteobahia.com.ar/scripts/forecast/for-ta.xml"

def _to_float(x):
    try: return float(str(x).replace(",", "."))
    except: return None

@st.cache_data(ttl=900)
def fetch_forecast():
    r = requests.get(API_URL, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.content)

    days = root.findall(".//forecast/tabular/day")
    rows = []
    for d in days:
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

# ======================
# ANN MODEL DE PREDICCIÓN
# ======================
class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW = IW
        self.bIW = bIW
        self.LW = LW
        self.bLW = bLW
        self.input_min = np.array([1,0,-7,0])
        self.input_max = np.array([300,41,25.5,84])

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
        emer = np.array(emer)
        emer = (emer+1)/2
        emer_ac = np.cumsum(emer)
        emerrel = np.diff(emer_ac, prepend=0)
        return emerrel, emer_ac

@st.cache_resource
def load_ann():
    IW = np.load(BASE/"IW.npy")
    bIW = np.load(BASE/"bias_IW.npy")
    LW = np.load(BASE/"LW.npy")
    bLW = np.load(BASE/"bias_out.npy")
    return PracticalANNModel(IW,bIW,LW,bLW)

modelo_ann = safe(lambda: load_ann(), "Error cargando pesos ANN")
if modelo_ann is None:
    st.stop()

# ======================
# CARGAR MODELO DE CLUSTERS
# ======================
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

    return data["scaler"], data["model"], data["metricas"], data["centroides"]

cluster_pack = safe(
    lambda: load_cluster_model(),
    "Error cargando modelo_cluster_d25_d50_d75_d95.pkl"
)

if cluster_pack is None:
    st.stop()
else:
    scaler_cl, model_cl, metricas_hist, centroides = cluster_pack

# ======================
# FUNCIONES D25–D95
# ======================
def calc_percentiles(dias, emerac):
    if emerac.max() == 0:
        return None
    y = emerac/emerac.max()
    d25 = np.interp(0.25, y, dias)
    d50 = np.interp(0.50, y, dias)
    d75 = np.interp(0.75, y, dias)
    d95 = np.interp(0.95, y, dias)
    return d25,d50,d75,d95

def curva(vals):
    d25,d50,d75,d95 = vals
    x = np.array([d25,d50,d75,d95])
    y = np.array([0.25,0.50,0.75,0.95])
    dias = np.arange(20,200)
    curva = np.interp(dias, x, y)
    return dias, curva

def radar(vals, labels, title, color):
    vals = list(vals); vals += vals[:1]
    ang = np.linspace(0,2*np.pi,len(vals))
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(ang, vals, color=color, lw=3)
    ax.fill(ang, vals, color=color, alpha=0.3)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    return fig

# ======================
# UI PRINCIPAL
# ======================
st.title("🌾 PREDWEEM v7 — ANN + Clasificación Temprano/Extendido")

fuente = st.radio("Fuente de datos", [
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
    up = st.file_uploader("Subir archivo meteo_history.csv", type=["csv"])
    if up:
        df = pd.read_csv(up, parse_dates=["Fecha"])

if df is None:
    st.stop()

df["Julian_days"] = df["Fecha"].dt.dayofyear
df = df.sort_values("Fecha")

# ======================
# ANN → EMERREL + EMERAC
# ======================
X = df[["Julian_days","TMAX","TMIN","Prec"]].to_numpy(float)
emerrel, emerac = modelo_ann.predict(X)

df["EMERREL"] = emerrel
df["EMERAC"] = emerac

# ======================
# CALCULAR PERCENTILES
# ======================
dias = df["Julian_days"].to_numpy()
res = calc_percentiles(dias, emerac)
if res is None:
    st.error("No se pudieron calcular percentiles")
    st.stop()

d25,d50,d75,d95 = res

st.subheader("📌 Percentiles simulados")
st.write({
    "d25": round(d25,1),
    "d50": round(d50,1),
    "d75": round(d75,1),
    "d95": round(d95,1)
})

# ======================
# CLASIFICACIÓN
# ======================
entrada_sc = scaler_cl.transform([[d25,d50,d75,d95]])
cl = int(model_cl.predict(entrada_sc)[0])

nombres = {1:"🌱 Temprano / Compacto", 0:"🌾 Extendido / Lento"}
colors  = {1:"green", 0:"orange"}

st.markdown(f"""
## 🎯 Patrón:
### <span style='color:{colors[cl]}; font-size:30px;'>{nombres[cl]}</span>
""", unsafe_allow_html=True)

# ======================
# CURVAS COMPARATIVAS
# ======================
st.subheader("Curvas comparativas")

dias_x, curva_x = curva([d25,d50,d75,d95])
dias0, curva0 = curva(centroides.loc[0].values)
dias1, curva1 = curva(centroides.loc[1].values)

fig, ax = plt.subplots(figsize=(9,5))
ax.plot(dias_x, curva_x, lw=3, label="Año simulado", color="blue")
ax.plot(dias1, curva1, lw=2, label="Centroide Temprano", color="green")
ax.plot(dias0, curva0, lw=2, label="Centroide Extendido", color="orange")
ax.set_xlabel("Día juliano")
ax.set_ylabel("EMERAC (0–1)")
ax.legend()
st.pyplot(fig)

# ======================
# RADAR
# ======================
st.subheader("Radar del patrón")
fig_rad = radar([d25,d50,d75,d95], ["d25","d50","d75","d95"], "Radar", "blue")
st.pyplot(fig_rad)

# ======================
# FIN
# ======================
