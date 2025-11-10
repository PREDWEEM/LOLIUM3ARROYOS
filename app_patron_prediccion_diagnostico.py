# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificación CONCENTRADA / EXTENDIDA (≥50 % AUC antes JD121)
# con calibración MANUAL mediante SLIDER sobre el gráfico

import os, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import streamlit as st

# ===== CONFIGURACIÓN =====
st.set_page_config(page_title="PREDWEEM — Clasificación con calibración visual", layout="wide")
st.title("🌾 Clasificación de patrones — Calibración manual interactiva del eje X (slider sobre gráfico)")

st.markdown("""
Ajustá **visualmente** la escala temporal (JD mínimo / máximo) directamente sobre cada gráfico.  
- Mové los *sliders* para que las líneas rojas coincidan con los límites del eje X real.  
- Observá en tiempo real cómo cambia la calibración.  
- Luego el sistema clasifica según el **AUC (≥50 % antes JD121)**.
""")

JD_CUTOFF = 121
EPS = 1e-9
CALIB_FILE = "calibracion_patrones.csv"

# ===== FUNCIONES =====
def read_image(file):
    data = file.read() if hasattr(file, "read") else file
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

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
    return np.array(xs), np.array(ys), h, w

def to_series(xs, ys, h, jd_min, jd_max, w):
    x_jd = jd_min + (xs / (w - 1)) * (jd_max - jd_min)
    y = (h - ys) / (h - 1)
    if y.max() > 0: y /= y.max()
    return x_jd, y

def restrict(x, y):
    mask = (x >= 1) & (x <= JD_CUTOFF)
    return x[mask], y[mask]

def regularize(x, y):
    xg = np.arange(1, JD_CUTOFF + 1)
    yg = np.interp(xg, x, y)
    return xg, yg

def smooth(y, win, poly):
    if len(y) < win: return y
    if win % 2 == 0: win += 1
    return savgol_filter(y, win, poly)

def auc(y): return float(np.trapz(y))

def classify_auc50(x, y):
    total = auc(y)
    share = auc(y[x <= JD_CUTOFF]) / (total + EPS)
    if share >= 0.50:
        patt, col = "CONCENTRADO", "green"
    else:
        patt, col = "EXTENDIDO", "orange"
    prob = round(abs(share - 0.50) * 1.5 + 0.5, 2)
    return patt, prob, dict(share=share, total=total, col=col)

# ===== PANEL DE AJUSTES =====
with st.sidebar:
    st.header("🧭 Detección y suavizado")
    thr_dark = st.slider("Umbral de oscuridad", 0, 255, 70)
    canny_low = st.slider("Canny low", 0, 200, 30)
    canny_high = st.slider("Canny high", 50, 300, 120)
    win = st.slider("Ventana suavizado", 3, 51, 9, step=2)
    poly = st.slider("Orden polinomio", 1, 5, 2)
    normalize_area = st.checkbox("Normalizar área total (AUC = 1)", True)
    auto_save = st.checkbox("Guardar calibraciones automáticamente", True)

# ===== LEER CALIBRACIONES =====
if os.path.exists(CALIB_FILE):
    df_calib = pd.read_csv(CALIB_FILE)
else:
    df_calib = pd.DataFrame(columns=["imagen", "JD_min", "JD_max"])

# ===== PROCESAMIENTO =====
files = st.file_uploader("📤 Subí las imágenes (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
if not files: st.stop()

rows, series = [], {}

for f in files:
    st.subheader(f"🖼️ {f.name}")
    try:
        img = read_image(f)
        xs, ys, h, w = extract_curve(img, thr_dark, canny_low, canny_high)
        if xs.size == 0:
            st.error("⚠️ No se detectó la curva. Ajustá umbral o filtros Canny.")
            continue

        # Calibración previa si existe
        prev = df_calib[df_calib["imagen"] == f.name]
        if not prev.empty:
            jd_min_def, jd_max_def = int(prev["JD_min"].iloc[0]), int(prev["JD_max"].iloc[0])
        else:
            jd_min_def, jd_max_def = 0, 365

        # === SLIDER DE CALIBRACIÓN ===
        st.markdown("### 🔧 Calibración visual del eje X")
        fig_cal, ax_cal = plt.subplots(figsize=(6, 3))
        ax_cal.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax_cal.plot(xs, ys, color='yellow', lw=1)
        st.pyplot(fig_cal, clear_figure=True)

        jd_min, jd_max = st.slider(f"Escala de días julianos para {f.name}",
                                   min_value=0, max_value=400,
                                   value=(jd_min_def, jd_max_def), step=5)

        # === VISUALIZAR CON LÍNEAS DE CALIBRACIÓN ===
        fig_lines, ax_lines = plt.subplots(figsize=(6, 3))
        ax_lines.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax_lines.axvline((jd_min / 400) * w, color='red', ls='--', label=f"JD {jd_min}")
        ax_lines.axvline((jd_max / 400) * w, color='red', ls='--', label=f"JD {jd_max}")
        ax_lines.plot(xs, ys, color='yellow', lw=1)
        ax_lines.legend(fontsize=7)
        ax_lines.set_title("Curva y límites de calibración (rojo)")
        st.pyplot(fig_lines, clear_figure=True)

        # Guardar calibración
        if auto_save:
            df_calib = df_calib[df_calib["imagen"] != f.name]
            df_calib.loc[len(df_calib)] = [f.name, jd_min, jd_max]
            df_calib.to_csv(CALIB_FILE, index=False)

        # === CLASIFICACIÓN ===
        x, y = to_series(xs, ys, h, jd_min, jd_max, w)
        x, y = restrict(x, y)
        x, y = regularize(x, y)
        y = smooth(y, win, poly)
        if normalize_area and auc(y) > 0: y /= auc(y)

        patt, prob, info = classify_auc50(x, y)
        year = os.path.splitext(f.name)[0]

        rows.append({
            "año": year,
            "JD_min": jd_min,
            "JD_max": jd_max,
            "AUC_total": round(info["total"], 3),
            "%_área ≤121": round(info["share"] * 100, 1),
            "patrón": patt,
            "probabilidad": prob
        })
        series[year] = (x, y, info["col"])

        st.success(f"**{year}** → {patt} ({prob:.2f}) — {info['share']*100:.1f}% del área antes del JD121")

    except Exception as e:
        st.error(f"Error procesando {f.name}: {e}")

# ===== RESULTADOS =====
if rows:
    df = pd.DataFrame(rows)
    st.subheader("📊 Resultados (AUC ≥50 % antes JD121)")
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Descargar CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="patrones_auc50_slider.csv")

    fig, ax = plt.subplots(figsize=(9, 4))
    for y, (xx, yy, col) in series.items():
        ax.plot(xx, yy, label=y, color=col)
    ax.axvline(121, color="black", ls="--", lw=1)
    ax.set_xlabel("Día juliano (JD calibrado)")
    ax.set_ylabel("Emergencia relativa")
    ax.legend(ncol=6, fontsize=8)
    st.pyplot(fig, clear_figure=True)

st.markdown("""
### 🌾 Criterio de clasificación
| Categoría | Condición | Interpretación |
|:-----------|:-----------|:---------------|
| **🟢 CONCENTRADO** | ≥ 50 % del área total antes del JD 121 | Emergencia temprana y sincronizada |
| **🟠 EXTENDIDO** | < 50 % del área total antes del JD 121 | Emergencia prolongada o escalonada |

> Las líneas **rojas** del gráfico marcan los límites de calibración JD.  
> Los valores ajustados se guardan automáticamente en `calibracion_patrones.csv`.
""")
