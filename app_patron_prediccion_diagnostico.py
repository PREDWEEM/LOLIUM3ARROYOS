# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificación CONCENTRADA / EXTENDIDA (≥50 % AUC antes JD 121)
# incluye calibración manual del eje X (JD real por gráfico)

import os, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import streamlit as st

# =============== CONFIGURACIÓN ===============
st.set_page_config(page_title="PREDWEEM — Clasificación AUC calibrada", layout="wide")
st.title("🌾 Clasificación de patrones — AUC (≥50 % antes JD 121, calibración del eje X)")

st.markdown("""
Analiza las curvas de emergencia entre **1 de enero y 1 de mayo (JD 1–121)**  
y clasifica según el área acumulada (AUC):
- **🟢 CONCENTRADA** → ≥ 50 % del área total antes del JD 121  
- **🟠 EXTENDIDA** → < 50 % del área total antes del JD 121  
Incluye calibración manual del eje X → relaciona los píxeles con los días julianos reales.
""")

# =============== CONTROLES ===============
with st.sidebar:
    st.header("📐 Calibración del eje X (JD real)")
    st.markdown("Define los valores reales que corresponden al eje horizontal del gráfico.")
    jd_min_real = st.number_input("JD mínimo visible", value=1)
    jd_max_real = st.number_input("JD máximo visible", value=365)

    st.header("🧭 Extracción de curva")
    left = st.slider("Recorte izquierdo (px)", 0, 250, 60)
    right = st.slider("Recorte derecho (px)", 0, 250, 40)
    top = st.slider("Recorte superior (px)", 0, 250, 40)
    bottom = st.slider("Recorte inferior (px)", 0, 250, 70)
    thr_dark = st.slider("Umbral de oscuridad", 0, 255, 70)
    canny_low = st.slider("Canny low", 0, 200, 30)
    canny_high = st.slider("Canny high", 50, 300, 120)
    win = st.slider("Ventana suavizado", 3, 51, 9, step=2)
    poly = st.slider("Orden polinomio", 1, 5, 2)
    normalize_area = st.checkbox("Normalizar área total (AUC = 1)", True)

JD_CUTOFF = 121
EPS = 1e-9

# =============== FUNCIONES ==================
def read_image(file):
    data = file.read() if hasattr(file, "read") else file
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

def crop_roi(img, l, r, t, b):
    h, w = img.shape[:2]
    return img[t:h-b, l:w-r].copy()

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

def to_series(xs, ys, w, h, jd_min, jd_max):
    x_jd = jd_min + (xs/(w-1))*(jd_max - jd_min)
    y = (h - ys)/(h-1)
    if y.max()>0: y/=y.max()
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

# =============== CLASIFICACIÓN 50 % ===============
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
        roi = crop_roi(img,left,right,top,bottom)
        xs,ys = extract_curve(roi,thr_dark,canny_low,canny_high)
        if xs.size==0: raise ValueError("Curva no detectada")
        h,w = roi.shape[:2]
        x,y = to_series(xs,ys,w,h,jd_min_real,jd_max_real)
        x,y = restrict(x,y)
        x,y = regularize(x,y)
        y = smooth(y,win,poly)
        if normalize_area and auc(y)>0: y/=auc(y)
        patt,prob,info = classify_auc50(x,y)
        year = os.path.splitext(f.name)[0]
        rows.append({
            "año":year,
            "AUC_total":round(info["total"],3),
            "%_área ≤ 121":round(info["share"]*100,1),
            "patrón":patt,
            "probabilidad":prob
        })
        series[year]=(x,y,info["col"])
    except Exception as e:
        rows.append({"año":f.name,"patrón":f"ERROR: "+str(e)})

# =============== RESULTADOS ===============
df=pd.DataFrame(rows)
st.subheader("Resultados (criterio ≥ 50 % AUC antes JD 121)")
st.dataframe(df,use_container_width=True)
st.download_button("⬇️ Descargar CSV", df.to_csv(index=False).encode("utf-8"),
                   file_name="patrones_auc50_calibrado.csv")

# =============== GRÁFICO ==================
fig,ax=plt.subplots(figsize=(9,4))
for y,(xx,yy,col) in series.items():
    ax.plot(xx,yy,label=y,color=col)
ax.axvline(121,color="black",ls="--",lw=1)
ax.set_xlim(jd_min_real,jd_max_real)
ax.set_xlabel("Día juliano (JD)")
ax.set_ylabel("Emergencia relativa")
ax.legend(ncol=6,fontsize=8)
st.pyplot(fig,clear_figure=True)

st.markdown(f"""
### 🌾 Criterio de clasificación (calibrado)
| Categoría | Condición | Interpretación |
|:-----------|:-----------|:---------------|
| **🟢 CONCENTRADO** | ≥ 50 % del área total antes del JD 121 | Emergencia predominantemente temprana |
| **🟠 EXTENDIDO** | < 50 % del área total antes del JD 121 | Emergencia más tardía o escalonada |

> Rango de JD usado en esta calibración: **{jd_min_real} → {jd_max_real}**
""")

