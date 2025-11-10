# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificación CONCENTRADA / EXTENDIDA (≥50 % AUC antes JD 121)
# con preprocesamiento + previsualización de calibración y curva detectada

import os, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import streamlit as st

# =============== CONFIGURACIÓN ===============
st.set_page_config(page_title="PREDWEEM — Clasificación AUC calibrada (v2)", layout="wide")
st.title("🌾 Clasificación de patrones — AUC (≥50 % antes JD121, con calibración visual)")

st.markdown("""
Analiza las curvas de emergencia entre **1 de enero y 1 de mayo (JD 1–121)**  
e incluye una **previsualización calibrada** para verificar la detección del eje X y la curva antes de clasificar:
- 🟢 **CONCENTRADA** → ≥ 50 % del área total antes del JD 121  
- 🟠 **EXTENDIDA** → < 50 % del área total antes del JD 121
""")

# =============== PARÁMETROS ==================
with st.sidebar:
    st.header("⚙️ Parámetros de extracción")
    thr_dark = st.slider("Umbral de oscuridad", 0, 255, 70)
    canny_low = st.slider("Canny low", 0, 200, 30)
    canny_high = st.slider("Canny high", 50, 300, 120)
    win = st.slider("Ventana suavizado", 3, 51, 9, step=2)
    poly = st.slider("Orden polinomio", 1, 5, 2)
    normalize_area = st.checkbox("Normalizar área total (AUC=1)", True)
    show_precal = st.checkbox("Mostrar previsualización de calibración", True)
    show_curve = st.checkbox("Mostrar curva detectada sobre imagen", True)

JD_CUTOFF = 121
EPS = 1e-9

# =============== FUNCIONES ==================
def read_image(file):
    data = file.read() if hasattr(file, "read") else file
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

def preprocess_calibrate_x(img, thr_dark):
    """Detecta el rango visible del eje X automáticamente"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray <= thr_dark).astype(np.uint8) * 255
    cols = np.sum(mask > 0, axis=0)
    x_min = np.argmax(cols > 5)
    x_max = len(cols) - np.argmax(cols[::-1] > 5)
    w = img.shape[1]
    jd_min_real = 0
    jd_max_real = 365
    if x_max - x_min < w * 0.8:
        jd_max_real = (x_max - x_min) / w * 365
    return int(x_min), int(x_max), jd_min_real, jd_max_real

def extract_curve(img, thr_dark, c_lo, c_hi):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray <= thr_dark).astype(np.uint8) * 255
    edges = cv2.Canny(mask, c_lo, c_hi)
    h, w = edges.shape
    xs, ys = [], []
    for x in range(w):
        y_col = np.where(edges[:, x] > 0)[0]
        if y_col.size:
            ys.append(np.median(y_col[:3]))
            xs.append(x)
    return np.array(xs), np.array(ys)

def to_series(xs, ys, h, jd_min, jd_max, x_min_px, x_max_px):
    span_px = max(1, x_max_px - x_min_px)
    x_jd = jd_min + ((xs - x_min_px)/span_px)*(jd_max - jd_min)
    y = (h - ys)/(h-1)
    if y.max() > 0: y /= y.max()
    return x_jd, y

def restrict(x,y):
    mask=(x>=1)&(x<=JD_CUTOFF)
    return x[mask], y[mask]

def regularize(x,y):
    xg=np.arange(1,JD_CUTOFF+1)
    yg=np.interp(xg,x,y)
    return xg, yg

def smooth(y,win,poly):
    if len(y)<win: return y
    if win%2==0: win+=1
    return savgol_filter(y,win,poly)

def auc(y): return float(np.trapz(y))

def classify_auc50(x,y):
    total=auc(y)
    share_before121 = auc(y[x<=121])/(total+EPS)
    if share_before121>=0.50:
        patt, col="CONCENTRADO","green"
    else:
        patt, col="EXTENDIDO","orange"
    prob=round(abs(share_before121-0.50)*1.5+0.5,2)
    return patt, prob, dict(share=share_before121,total=total,col=col)

# =============== PROCESAMIENTO ===============
files = st.file_uploader("Subí las imágenes (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=True)
if not files: st.stop()

rows, series = [], {}

for f in files:
    try:
        img = read_image(f)
        x_min_px, x_max_px, jd_min_real, jd_max_real = preprocess_calibrate_x(img, thr_dark)

        # --- PREVISUALIZACIÓN DE CALIBRACIÓN ---
        if show_precal:
            fig_pre, ax_pre = plt.subplots(figsize=(6, 3))
            ax_pre.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax_pre.axvline(x_min_px, color='red', linestyle='--', label='Inicio eje X')
            ax_pre.axvline(x_max_px, color='red', linestyle='--', label='Fin eje X')
            ax_pre.set_title(f"{f.name} — Rango detectado: {jd_min_real:.0f} → {jd_max_real:.0f} JD")
            ax_pre.legend()
            st.pyplot(fig_pre, clear_figure=True)

        # --- EXTRACCIÓN DE CURVA ---
        xs, ys = extract_curve(img, thr_dark, canny_low, canny_high)
        if xs.size == 0: raise ValueError("Curva no detectada")
        h = img.shape[0]

        # --- CURVA DETECTADA SOBRE IMAGEN ---
        if show_curve:
            fig_curv, ax_curv = plt.subplots(figsize=(6, 3))
            ax_curv.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax_curv.plot(xs, ys, color='yellow', linewidth=1)
            ax_curv.set_title("Curva detectada (línea amarilla)")
            st.pyplot(fig_curv, clear_figure=True)

        # --- CALIBRACIÓN + CLASIFICACIÓN ---
        x, y = to_series(xs, ys, h, jd_min_real, jd_max_real, x_min_px, x_max_px)
        x, y = restrict(x, y)
        x, y = regularize(x, y)
        y = smooth(y, win, poly)
        if normalize_area and auc(y) > 0: y /= auc(y)
        patt, prob, info = classify_auc50(x, y)
        year = os.path.splitext(f.name)[0]

        rows.append({
            "año": year,
            "JD_min_real": jd_min_real,
            "JD_max_real": round(jd_max_real,1),
            "AUC_total": round(info["total"],3),
            "%_área ≤121": round(info["share"]*100,1),
            "patrón": patt,
            "probabilidad": prob
        })
        series[year]=(x,y,info["col"])

    except Exception as e:
        rows.append({"año": f.name, "patrón": f"ERROR: {e}"})

# =============== RESULTADOS ===============
df=pd.DataFrame(rows)
st.subheader("Resultados (calibración automática y visual)")
st.dataframe(df,use_container_width=True)
st.download_button("⬇️ Descargar CSV", df.to_csv(index=False).encode("utf-8"),
                   file_name="patrones_auc50_calibrado_v2.csv")

# =============== GRÁFICO ==================
fig,ax=plt.subplots(figsize=(9,4))
for y,(xx,yy,col) in series.items():
    ax.plot(xx,yy,label=y,color=col)
ax.axvline(121,color="black",ls="--",lw=1)
ax.set_xlabel("Día juliano (JD calibrado)")
ax.set_ylabel("Emergencia relativa")
ax.legend(ncol=6,fontsize=8)
st.pyplot(fig,clear_figure=True)

st.markdown("""
### 🌾 Criterio de clasificación
| Categoría | Condición | Interpretación |
|:-----------|:-----------|:---------------|
| **🟢 CONCENTRADO** | ≥ 50 % del área total antes del JD 121 | Emergencia predominantemente temprana |
| **🟠 EXTENDIDO** | < 50 % del área total antes del JD 121 | Emergencia más tardía o escalonada |

> El eje X se calibró automáticamente a partir del rango visible y se puede verificar visualmente antes del análisis.
""")

