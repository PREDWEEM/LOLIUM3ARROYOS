# app_emergencia_v7.py
# ===============================================================
# 🌾 PREDWEEM v7 — ANN + Clasificación anticipada Temprano/Extendido
# ---------------------------------------------------------------
# - Usa ANN para EMERREL/EMERAC
# - Usa meteo local + API
# - Clasifica patrón histórico por d25, d50, d75, d95 simulados
# - Modelo corregido: modelo_cluster_d25_d50_d75_d95.pkl
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import requests, time, xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pickle

# ===============================================================
# 🔧 CONFIGURACIÓN STREAMLIT
# ===============================================================
st.set_page_config(
    page_title="PREDWEEM v7 – EMERGENCIA + PATRÓN",
    layout="wide"
)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ===============================================================
# 🔧 UTILS
# ===============================================================

def safe(fn, msg):
    try:
        return fn()
    except Exception as e:
        st.error(msg)
        return None

# ===============================================================
# 🔧 MODELO ANN
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
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def tansig(self, x):
        return np.tanh(x)

    def forward(self, x):
        z1 = self.IW.T @ x + self.bIW
        a1 = self.tansig(z1)
        z2 = self.LW @ a1 + self.bLW
        return self.tansig(z2)

    def predict_df(self, X_real):
        Xn = self.normalize(X_real)
        emer = np.array([self.forward(x) for x in Xn])
        emer = (emer + 1) / 2
        emer_ac = np.cumsum(emer)
        emerrel = np.diff(emer_ac, prepend=0)

        return emerrel, emer_ac

# ===============================================================
# 🔧 CARGAR ANN
# ===============================================================

def load_ann():
    IW = np.load(BASE/"IW.npy")
    bIW = np.load(BASE/"bias_IW.npy")
    LW = np.load(BASE/"LW.npy")
    bLW = np.load(BASE/"bias_out.npy")
    return PracticalANNModel(IW, bIW, LW, bLW)

modelo_ann = safe(lambda: load_ann(), "Error cargando ANN")
if modelo_ann is None:
    st.stop()

# ===============================================================
# 🔧 MODELO CLUSTER (NUEVO CORREGIDO)
# ===============================================================

def load_cluster_model():
    with open(BASE/"modelo_cluster_d25_d50_d75_d95.pkl", "rb") as f:
        data = pickle.load(f)
    return data["scaler"], data["model"], data["metricas"], data["centroides"]

scaler_cl, model_cl, metricas_hist, centroides = safe(
    lambda: load_cluster_model(),
    "Error cargando modelo_cluster_d25_d50_d75_d95.pkl"
)

if scaler_cl is None:
    st.stop()

# ===============================================================
# 🔧 FUNCS DE CLASIFICACIÓN
# ===============================================================

def calcular_percentiles(dias, emerac):
    if emerac.max() == 0:
        return None

    y = emerac / emerac.max()
    d25 = np.interp(0.25, y, dias)
    d50 = np.interp(0.50, y, dias)
    d75 = np.interp(0.75, y, dias)
    d95 = np.interp(0.95, y, dias)
    return d25, d50, d75, d95

def curva_centroidal(vals):
    d25, d50, d75, d95 = vals
    x = np.array([d25,d50,d75,d95])
    y = np.array([0.25,0.50,0.75,0.95])
    dias = np.arange(20,200)
    curva = np.interp(dias, x, y)
    return dias, curva

def radar(vals, labels, title, color):
    vals = list(vals)
    vals.append(vals[0])
    ang = np.linspace(0,2*np.pi,len(vals))
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(ang, vals, color=color, lw=3)
    ax.fill(ang, vals, color=color, alpha=0.3)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    return fig

# ===============================================================
# 🔧 UI – FUENTE DE DATOS
# ===============================================================
st.title("🌾 PREDWEEM v7 — EMERGENCIA + CLASIFICACIÓN DEL PATRÓN")

fuente = st.radio(
    "Fuente de datos climáticos",
    ["Histórico local (meteo_daily.csv)", "Subir archivo"]
)

df_meteo = None

if fuente == "Histórico local (meteo_daily.csv)":
    if not (BASE/"meteo_daily.csv").exists():
        st.error("No se encontró meteo_daily.csv")
        st.stop()
    df_meteo = pd.read_csv(BASE/"meteo_daily.csv", parse_dates=["Fecha"])

else:
    up = st.file_uploader("Subí archivo meteo_history.csv", type=["csv"])
    if up is not None:
        df_meteo = pd.read_csv(up, parse_dates=["Fecha"])

if df_meteo is None:
    st.stop()

df_meteo["Julian_days"] = df_meteo["Fecha"].dt.dayofyear
df_meteo = df_meteo.sort_values("Fecha")

# ===============================================================
# 🔧 ANN → EMERREL / EMERAC
# ===============================================================

X = df_meteo[["Julian_days","TMAX","TMIN","Prec"]].to_numpy(float)

emerrel, emerac = modelo_ann.predict_df(X)

df_meteo["EMERREL"] = emerrel
df_meteo["EMERAC"] = emerac

# ===============================================================
# 🔧 CLASIFICACIÓN DEL PATRÓN
# ===============================================================

dias = df_meteo["Julian_days"].to_numpy()
res = calcular_percentiles(dias, emerac)

if res is None:
    st.error("No se pudo calcular percentiles (curva vacía).")
    st.stop()

d25, d50, d75, d95 = res

st.subheader("Percentiles simulados:")
st.write({
    "d25": round(d25,1),
    "d50": round(d50,1),
    "d75": round(d75,1),
    "d95": round(d95,1)
})

# CLUSTER
entrada = np.array([[d25,d50,d75,d95]])
entrada_sc = scaler_cl.transform(entrada)
cl = int(model_cl.predict(entrada_sc)[0])

nombres = {
    1: "🌱 Temprano / Compacto",
    0: "🌾 Extendido / Lento"
}
colors = {1:"green", 0:"orange"}

st.markdown(f"""
## 🎯 **Patrón del año (ANN + clima):**  
### <span style='color:{colors[cl]}; font-size:30px;'>{nombres[cl]}</span>
""", unsafe_allow_html=True)

# ===============================================================
# 🔧 GRAFICAR CURVAS
# ===============================================================

st.subheader("Curva del año vs centroides históricos")

dias_x, curva_x = curva_centroidal([d25,d50,d75,d95])

dias0, curva0 = curva_centroidal(centroides.loc[0].values)
dias1, curva1 = curva_centroidal(centroides.loc[1].values)

fig, ax = plt.subplots(figsize=(9,5))
ax.plot(dias_x, curva_x, label="Año actual (simulado)", lw=3, color="blue")
ax.plot(dias1, curva1, label="Centroide Temprano", color="green")
ax.plot(dias0, curva0, label="Centroide Extendido", color="orange")
ax.set_xlabel("Día juliano")
ax.set_ylabel("Emergencia acumulada (0–1)")
ax.set_title("Comparación del patrón")
ax.legend()
st.pyplot(fig)

# ===============================================================
# 🔧 RADAR
# ===============================================================

st.subheader("Radar del patrón del año")

rad = radar([d25,d50,d75,d95], ["d25","d50","d75","d95"], "Radar del año", "blue")
st.pyplot(rad)

# ===============================================================
# FIN
