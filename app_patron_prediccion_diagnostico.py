# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificación 2 categorías (CONCENTRADA / EXTENDIDA) basada en AUC (1-ene → 1-may)

import os, cv2, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import streamlit as st

# =============== CONFIGURACIÓN ===============
st.set_page_config(page_title="PREDWEEM — Patrones (AUC 1-ene→1-may)", layout="wide")
st.title("🌾 Clasificación de patrones por área bajo la curva (AUC)")

st.markdown("""
La app analiza las curvas de emergencia entre **1 de enero y 1 de mayo (JD 1–121)**  
y clasifica los patrones en dos categorías:
- **🟢 CONCENTRADA** → emergencia rápida (mayor parte del área antes del JD 100)  
- **🟠 EXTENDIDA** → emergencia prolongada o tardía (área distribuida más allá del JD 100)
""")

# =============== CONTROLES LATERALES ===============
with st.sidebar:
    st.header("Ajustes de extracción")
    left = st.slider("Recorte izquierdo (px)", 0, 250, 60)
    right = st.slider("Recorte derecho (px)", 0, 250, 40)
    top = st.slider("Recorte superior (px)", 0, 250, 40)
    bottom = st.slider("Recorte inferior (px)", 0, 250, 70)
    thr_dark = st.slider("Umbral de oscuridad", 0, 255, 70)
    canny_low = st.slider("Canny low", 0, 200, 30)
    canny_high = st.slider("Canny high", 50, 300, 120)
    win = st.slider("Ventana Savitzky-Golay", 3, 51, 9, 2)
    poly = st.slider("Orden polinomio", 1, 5, 2)
    x_min_fig = st.number_input("JD mínimo del gráfico", value=0)
    x_max_fig = st.number_input("JD máximo del gráfico", value=400)
    normalize_area = st.checkbox("Normalizar área total (AUC=1)", True)
    show_debug = st.checkbox("Mostrar depuración", False)

JD_CUTOFF = 121
EPS = 1e-9

# =============== FUNCIONES AUXILIARES ===============
def read_image(file):
    data = file.read() if hasattr(file, "read") else file
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

def crop_roi(img, left, right, top, bottom):
    h, w = img.shape[:2]
    return img[top:h-bottom, left:w-right].copy()

def extract_curve(img, thr_dark, canny_lo, canny_hi):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray <= thr_dark).astype(np.uint8) * 255
    edges = cv2.Canny(mask, canny_lo, canny_hi)
    h, w = edges.shape
    xs, ys = [], []
    for x in range(w):
        ys_col = np.where(edges[:, x] > 0)[0]
        if ys_col.size > 0:
            ys.append(np.median(ys_col[:3]))
            xs.append(x)
    return np.array(xs), np.array(ys)

def to_series(xs, ys, w, h, x_min, x_max):
    x_jd = x_min + (xs / (w-1)) * (x_max - x_min)
    y_val = (h - ys) / (h-1)
    if y_val.max() > 0: y_val /= y_val.max()
    return x_jd, y_val

def restrict(x, y):
    mask = (x >= 1) & (x <= JD_CUTOFF)
    return x[mask], y[mask]

def regularize(x, y):
    x_new = np.arange(1, JD_CUTOFF+1)
    y_new = np.interp(x_new, x, y)
    return x_new, y_new

def smooth(y, win, poly):
    if len(y) < win: return y
    if win % 2 == 0: win += 1
    return savgol_filter(y, win, poly)

def auc(y): return float(np.trapz(y))

# =============== CLASIFICACIÓN BASADA EN ÁREA ===============
def classify_by_auc(x, y):
    total_auc = auc(y)
    idx_100 = np.where(x <= 100)[0]
    auc_pre100 = auc(y[idx_100]) if idx_100.size else 0
    share_pre100 = auc_pre100 / (total_auc + EPS)
    if share_pre100 >= 0.70 and total_auc <= 0.6:
        pattern, color = "CONCENTRADA", "green"
    else:
        pattern, color = "EXTENDIDA", "orange"
    prob = abs(share_pre100 - 0.70) * 1.5 + 0.5
    return pattern, prob, dict(auc_total=total_auc, share_pre100=share_pre100, color=color)

# =============== PROCESAMIENTO DE IMÁGENES ===============
files = st.file_uploader("Subí las imágenes (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=True)
if not files: st.stop()

rows, series = [], {}
for f in files:
    try:
        img = read_image(f)
        roi = crop_roi(img, left, right, top, bottom)
        xs, ys = extract_curve(roi, thr_dark, canny_low, canny_high)
        if xs.size == 0: raise ValueError("Curva no detectada")
        h, w = roi.shape[:2]
        x, y = to_series(xs, ys, w, h, x_min_fig, x_max_fig)
        x, y = restrict(x, y)
        x, y = regularize(x, y)
        y = smooth(y, win, poly)
        if normalize_area and auc(y) > 0:
            y /= auc(y)
        pat, prob, info = classify_by_auc(x, y)
        year = os.path.splitext(f.name)[0]
        rows.append({
            "año": year,
            "AUC_total": round(info["auc_total"],3),
            "%_área_pre100": round(info["share_pre100"]*100,1),
            "patrón": pat,
            "probabilidad": round(prob,2)
        })
        series[year] = (x, y, info["color"])
    except Exception as e:
        rows.append({"año": f.name, "patrón": f"ERROR: {e}"})

# =============== RESULTADOS ==================
df = pd.DataFrame(rows)
st.subheader("Resultados basados en área bajo la curva (AUC)")
st.dataframe(df, use_container_width=True)

st.download_button("⬇️ Descargar CSV", df.to_csv(index=False).encode("utf-8"),
                   file_name="patrones_auc.csv")

# =============== GRÁFICO COMPARATIVO ==================
st.subheader("Curvas normalizadas (1–121)")
fig, ax = plt.subplots(figsize=(9,4))
for y, (xx,yy,col) in series.items():
    ax.plot(xx, yy, label=y, color=col)
ax.axvline(100, color="gray", ls="--", lw=1)
ax.axvline(121, color="black", ls="--", lw=1)
ax.set_xlim(1, JD_CUTOFF)
ax.set_xlabel("Día juliano (JD)")
ax.set_ylabel("Emergencia relativa")
ax.legend(ncol=6, fontsize=8)
st.pyplot(fig, clear_figure=True)

# =============== CRITERIOS ===============
st.markdown("""
### Criterios de clasificación (basados en el área)
| Categoría | Condición principal | Interpretación agronómica |
|------------|--------------------|---------------------------|
| **🟢 CONCENTRADA** | ≥ 70 % del área total antes del JD 100 y AUC ≤ 0.6 | Emergencia rápida y sincronizada (efectiva con residuales cortos) |
| **🟠 EXTENDIDA** | < 70 % del área antes del JD 100 o AUC > 0.6 | Emergencia prolongada o tardía (requiere control complementario postemergente) |
""")

