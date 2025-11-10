# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificación 2 categorías (CONCENTRADA / EXTENDIDA)
# Usa solo información 1-ene → 1-may (JD 1–121)

import os, math, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import streamlit as st

# ========================= CONFIGURACIÓN =========================
st.set_page_config(page_title="PREDWEEM — Patrones 2 categorías", layout="wide")
st.title("🌾 Clasificación de patrones (CONCENTRADA / EXTENDIDA) — 1 ene → 1 may")

st.markdown("""
Subí las imágenes de las curvas de emergencia.  
La app analiza **solo los días julianos 1–121 (1 de enero → 1 de mayo)**  
y clasifica cada año como **CONCENTRADA** o **EXTENDIDA**.
""")

# -------------------- CONTROLES LATERALES --------------------
with st.sidebar:
    st.header("Extracción y análisis")
    left_margin   = st.slider("Recorte izquierdo (px)", 0, 250, 60)
    right_margin  = st.slider("Recorte derecho (px)",   0, 250, 40)
    top_margin    = st.slider("Recorte superior (px)",  0, 250, 40)
    bottom_margin = st.slider("Recorte inferior (px)",  0, 250, 70)
    thr_dark      = st.slider("Umbral de oscuridad", 0, 255, 70)
    canny_low     = st.slider("Canny low", 0, 200, 30)
    canny_high    = st.slider("Canny high", 50, 300, 120)
    win           = st.slider("Suavizado Savitzky-Golay", 3, 51, 9, step=2)
    poly          = st.slider("Orden polinomio", 1, 5, 2)
    prominence    = st.slider("Prominencia de picos", 0.0, 10.0, 0.8, 0.1)
    distance      = st.slider("Distancia mínima entre picos (JD)", 1, 60, 10)
    height_thr    = st.slider("Altura mínima relativa", 0.0, 1.0, 0.05, 0.01)
    x_min_fig     = st.number_input("JD mínimo del gráfico", value=0)
    x_max_fig     = st.number_input("JD máximo del gráfico", value=400)
    normalize_area= st.checkbox("Normalizar área (AUC=1)", True)
    show_debug    = st.checkbox("Mostrar depuración", False)

JD_CUTOFF = 121
EPS = 1e-9

# -------------------- FUNCIONES AUXILIARES --------------------
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
        y_col = np.where(edges[:, x] > 0)[0]
        if y_col.size > 0:
            ys.append(np.median(y_col[:3]))
            xs.append(x)
    return np.array(xs), np.array(ys)

def to_series(xs, ys, w, h, x_min, x_max):
    x_jd = x_min + (xs / (w-1)) * (x_max - x_min)
    y_val = (h - ys) / (h-1)
    if y_val.max() > 0:
        y_val /= y_val.max()
    return x_jd, y_val

def restrict(x, y):
    mask = (x >= 1) & (x <= JD_CUTOFF)
    return x[mask], y[mask]

def regularize(x, y):
    x_new = np.arange(1, JD_CUTOFF+1)
    y_new = np.interp(x_new, x, y)
    return x_new, y_new

def smooth(y, win, poly):
    if len(y) < win:
        return y
    if win % 2 == 0: win += 1
    return savgol_filter(y, win, poly)

def auc(y):
    return float(np.trapz(y))

# -------------------- CLASIFICACIÓN 2 CATEGORÍAS --------------------
def classify_2cats(x, y, prominence, distance, height_thr):
    peaks, props = find_peaks(y, prominence=prominence, distance=distance, height=height_thr)
    n_p = len(peaks)
    jd_main = x[peaks[np.argmax(props["prominences"])]] if n_p else np.nan
    dur = x[np.where(y > 0.02)[0][-1]] - x[np.where(y > 0.02)[0][0]] if np.any(y > 0.02) else 0
    m100 = np.where(x >= 100)[0]
    share_after_100 = y[m100].sum() / (y.sum() + EPS)
    # REGLAS
    score_conc = (n_p <= 2)*0.4 + (dur <= 40)*0.4 + (share_after_100 < 0.2)*0.2
    score_ext  = (n_p >= 3)*0.3 + (dur > 40)*0.4 + (share_after_100 >= 0.2)*0.3
    pattern = "CONCENTRADA" if score_conc >= score_ext else "EXTENDIDA"
    prob = abs(score_conc - score_ext) + 0.5
    return pattern, prob, dict(n_p=n_p, dur=dur, share=share_after_100)

# -------------------- PROCESAMIENTO --------------------
files = st.file_uploader("Subí las imágenes (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=True)
if not files: st.stop()

rows, series = [], {}
for f in files:
    try:
        img = read_image(f)
        roi = crop_roi(img, left_margin, right_margin, top_margin, bottom_margin)
        xs, ys = extract_curve(roi, thr_dark, canny_low, canny_high)
        if xs.size == 0: raise ValueError("Curva no detectada")
        h, w = roi.shape[:2]
        x, y = to_series(xs, ys, w, h, x_min_fig, x_max_fig)
        x, y = restrict(x, y)
        x, y = regularize(x, y)
        y = smooth(y, win, poly)
        if normalize_area and auc(y)>0: y /= auc(y)
        pat, prob, info = classify_2cats(x, y, prominence, distance, height_thr)
        year = os.path.splitext(f.name)[0]
        rows.append({
            "año": year, "n_picos": info["n_p"],
            "duración_días": round(info["dur"],1),
            "masa_JD>100": round(info["share"],3),
            "patrón": pat, "probabilidad": round(prob,2)
        })
        series[year] = (x,y)
    except Exception as e:
        rows.append({"año": f.name, "patrón": f"ERROR: {e}"})

# -------------------- RESULTADOS --------------------
df = pd.DataFrame(rows)
st.subheader("Resultados (JD 1–121)")
st.dataframe(df, use_container_width=True)

st.download_button("⬇️ Descargar CSV", df.to_csv(index=False).encode("utf-8"),
                   file_name="patrones_concentrada_extendida.csv")

# -------------------- GRÁFICO COMPARATIVO --------------------
st.subheader("Curvas normalizadas (1–121)")
fig, ax = plt.subplots(figsize=(9,4))
for y, (xx,yy) in series.items():
    ax.plot(xx, yy, label=y)
ax.axvline(100, color="gray", ls="--", lw=1)
ax.axvline(121, color="black", ls="--", lw=1)
ax.set_xlim(1, JD_CUTOFF)
ax.set_xlabel("Día juliano (JD)")
ax.set_ylabel("Emergencia relativa")
ax.legend(ncol=6, fontsize=8)
st.pyplot(fig, clear_figure=True)

st.markdown("""
### Criterios de clasificación
| Categoría | Rasgos dominantes (1–121) |
|------------|---------------------------|
| **🟢 CONCENTRADA** | ≤ 2 picos, duración ≤ 40 d, masa JD>100 < 0.20 |
| **🟠 EXTENDIDA**   | ≥ 3 picos o duración > 40 d o masa JD>100 ≥ 0.20 |
""")
