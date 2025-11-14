# app_patron_meteo_v7.py
# ===============================================================
# 🌾 PREDWEEM v7 – CLASIFICACIÓN DIRECTA DESDE METEO
# ---------------------------------------------------------------
# Entrada: meteo_history.csv (Fecha, Julian_days, TMAX, TMIN, Prec)
# Salida: Patrón Temprano / Extendido + gráficos
# Usa:
#   - ANN (IW.npy, LW.npy, bias_IW.npy, bias_out.npy)
#   - modelo_cluster_d25_d50_d75_d95.pkl
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# ===============================================================
# 💠 CONFIG STREAMLIT
# ===============================================================
st.set_page_config(
    page_title="PREDWEEM – Patrón Histórico (ANN + d25–d50–d75–d95)",
    layout="wide",
)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ===============================================================
# 💠 MODELO ANN
# ===============================================================
class PracticalANNModel:
    def __init__(self, IW, bias_IW, LW, bias_out):
        self.IW = IW
        self.bias_IW = bias_IW
        self.LW = LW
        self.bias_out = bias_out
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def tansig(self, x): 
        return np.tanh(x)

    def normalize_input(self, X_real):
        return 2 * (X_real - self.input_min) / (self.input_max - self.input_min) - 1

    def desnorm_out(self, y):
        return (y + 1) / 2

    def predict(self, X_real):
        X_norm = self.normalize_input(X_real)
        z1 = (self.IW.T @ X_norm.T).T + self.bias_IW
        a1 = np.tanh(z1)
        z2 = (self.LW @ a1.T).T + self.bias_out
        out_norm = np.tanh(z2)
        emer = self.desnorm_out(out_norm)
        emerac = np.cumsum(emer)
        emerrel = np.diff(emerac, prepend=0)
        return emerrel, emerac

# ===============================================================
# 💠 CARGA DE ARCHIVOS ANN
# ===============================================================
@st.cache_resource
def load_ann():
    IW = np.load(BASE/"IW.npy")
    bias_IW = np.load(BASE/"bias_IW.npy")
    LW = np.load(BASE/"LW.npy")
    bias_out = np.load(BASE/"bias_out.npy")
    return PracticalANNModel(IW, bias_IW, LW, bias_out)

modelo_ann = load_ann()

# ===============================================================
# 💠 CLUSTER d25–d50–d75–d95
# ===============================================================
@st.cache_resource
def load_cluster():
    with open(BASE/"modelo_cluster_d25_d50_d75_d95.pkl", "rb") as f:
        data = pickle.load(f)
    return data["scaler"], data["model"], data["metricas"]

scaler_clust, model_clust, metricas_hist = load_cluster()
centroides = metricas_hist.groupby("cluster")[["d25","d50","d75","d95"]].mean()

# ===============================================================
# 💠 FUNCIONES AUXILIARES
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

def curva_representativa(d25,d50,d75,d95):
    x = np.array([d25,d50,d75,d95])
    y = np.array([0.25,0.50,0.75,0.95])
    dias = np.arange(20,200)
    curva = np.interp(dias, x, y)
    return dias, curva

def radar(vals, labels, title, color):
    vals = list(vals)
    vals += vals[:1]  
    angles = np.linspace(0, 2*np.pi, len(vals), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals, color=color, lw=3)
    ax.fill(angles, vals, color=color, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    return fig

# ===============================================================
# 💠 UI PRINCIPAL
# ===============================================================

st.title("🌾 PREDWEEM – CLASIFICACIÓN DEL PATRÓN HISTÓRICO (solo clima)")

archivo = st.file_uploader(
    "Cargar archivo meteo_history.csv",
    type=["csv"],
    accept_multiple_files=False
)

if archivo is None:
    st.stop()

df = pd.read_csv(archivo, parse_dates=["Fecha"])
df = df.sort_values("Fecha")
df["Julian_days"] = df["Fecha"].dt.dayofyear

st.success("Archivo meteorológico cargado correctamente.")

# ===============================================================
# 💠 ANN → EMERREL / EMERAC
# ===============================================================

X = df[["Julian_days","TMAX","TMIN","Prec"]].to_numpy(float)
emerrel, emerac = modelo_ann.predict(X)

df["EMERREL"] = emerrel
df["EMERAC"] = emerac

# ===============================================================
# 💠 CALCULAR PERCENTILES
# ===============================================================

dias = df["Julian_days"].to_numpy()
res = calcular_percentiles(dias, emerac)

if res is None:
    st.error("No se pudo calcular percentiles. Verificar datos.")
    st.stop()

d25, d50, d75, d95 = res

st.subheader("📌 Percentiles obtenidos:")
st.write({
    "d25": round(d25,2),
    "d50": round(d50,2),
    "d75": round(d75,2),
    "d95": round(d95,2)
})

# ===============================================================
# 💠 CLASIFICAR
# ===============================================================

X_in = np.array([[d25,d50,d75,d95]])
X_sc = scaler_clust.transform(X_in)
cluster = int(model_clust.predict(X_sc)[0])

etiqueta = {1: "🌱 Temprano / Compacto", 0: "🌾 Extendido / Lento"}
color  = {1: "green", 0: "orange"}

st.markdown(f"""
## 🎯 **Patrón determinado:**  
### <span style='color:{color[cluster]}; font-size:30px;'>{etiqueta[cluster]}</span>
""", unsafe_allow_html=True)

# ===============================================================
# 💠 GRAFICAR CURVA DEL AÑO VS CENTROIDES
# ===============================================================

c0 = centroides.loc[0]
c1 = centroides.loc[1]

dias_x, curva_x = curva_representativa(d25,d50,d75,d95)
dias0, curva0   = curva_representativa(c0["d25"],c0["d50"],c0["d75"],c0["d95"])
dias1, curva1   = curva_representativa(c1["d25"],c1["d50"],c1["d75"],c1["d95"])

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(dias_x, curva_x, label="Año analizado", lw=3, color="blue")
ax.plot(dias1, curva1, label="Centroide Temprano", lw=2, color="green")
ax.plot(dias0, curva0, label="Centroide Extendido", lw=2, color="orange")
ax.set_title("Curva proyectada del año vs patrones típicos")
ax.set_xlabel("Día juliano")
ax.set_ylabel("Emergencia acumulada (0–1)")
ax.legend()
st.pyplot(fig)

# ===============================================================
# 💠 RADAR
# ===============================================================

labels = ["d25","d50","d75","d95"]
fig_r = radar([d25,d50,d75,d95], labels, "Radar del año analizado", "blue")
st.pyplot(fig_r)

# ===============================================================
# FIN DEL SCRIPT

