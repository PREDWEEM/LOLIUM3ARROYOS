# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador robusto (curva negra + seguimiento preciso hasta 1° mayo)
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

# ========= CONFIGURACIÓN STREAMLIT =========
st.set_page_config(page_title="Clasificador PREDWEEM — Preciso (1° mayo)", layout="wide")
st.title("🌾 Clasificador PREDWEEM — Precisión fotométrica (curva negra, corte al 1° mayo)")

st.markdown("""
Analiza gráficos de **emergencia relativa** con eje X en **días julianos (0–300)**.  
Detecta automáticamente la **curva negra** principal, reconstruye su forma exacta y la usa para determinar el **patrón histórico (P1, P1b, P2, P3)**.  
Solo se usa información hasta el **1° de mayo (día juliano 121)** para clasificar.
""")

# ========= SIDEBAR =========
st.sidebar.header("⚙️ Parámetros de análisis")
height_thr = st.sidebar.slider("Umbral mínimo de altura", 0.05, 0.5, 0.22, 0.01)
dist_min = st.sidebar.slider("Distancia mínima entre picos (≈ días julianos)", 10, 80, 35, 5)
window_smooth = st.sidebar.slider("Ventana de suavizado (px)", 5, 51, 15, 2)
poly_order = st.sidebar.slider("Orden del filtro", 1, 3, 2, 1)

uploaded = st.file_uploader("📤 Cargar imagen (.png o .jpg)", type=["png", "jpg"])

# ========= PROCESAMIENTO =========
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(uploaded, caption="📈 Imagen original", use_container_width=True)

    # --- Conversión a gris e inversión ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray_inv = 255 - gray
    gray_norm = cv2.GaussianBlur(gray_inv, (3, 3), 0)
    gray_norm = gray_norm.astype(float) / 255.0

    # --- Seguimiento vertical de línea negra (píxel más oscuro por columna) ---
    curve_y = []
    for i in range(w):
        col = gray_norm[:, i]
        if np.max(col) > 0.25:  # ignora columnas sin trazo
            y_pos = np.argmax(col)
            curve_y.append(h - y_pos)
        else:
            curve_y.append(np.nan)
    curve_y = np.array(curve_y)

    # --- Interpolación de huecos y normalización ---
    curve_y = pd.Series(curve_y).interpolate(limit_direction="both").to_numpy()
    y_norm = (curve_y - np.nanmin(curve_y)) / (np.nanmax(curve_y) - np.nanmin(curve_y) + 1e-6)

    # --- Eje temporal juliano ---
    x_julian = np.linspace(0, 300, len(y_norm))

    # --- Corte al 1° mayo (día juliano 121) ---
    mask_corte = x_julian <= 121
    x_sub, y_sub = x_julian[mask_corte], y_norm[mask_corte]

    # --- Suavizado leve (sin deformar) ---
    y_smooth = savgol_filter(y_sub, window_smooth, poly_order)

    # ========= CLASIFICACIÓN =========
    def clasificar(curva, thr, dist):
        peaks, props = find_peaks(curva, height=thr, distance=dist)
        heights = props.get("peak_heights", [])
        n = len(peaks)

        if n == 0:
            return "-", 0.0, [], [], 0, 0, 0, 0

        mean_sep = np.mean(np.diff(peaks)) if n > 1 else 0
        std_sep = np.std(np.diff(peaks)) if n > 2 else 0
        hmax = float(np.max(heights))
        hmean = float(np.mean(heights))
        ratio_minor = float(np.min(heights) / (hmax + 1e-6)) if len(heights) > 1 else 0.0

        # --- Criterios revisados ---
        if n == 1:
            tipo = "P1"
        elif n == 2:
            if mean_sep < 70 and ratio_minor < 0.35:
                tipo = "P1b"
            elif mean_sep >= 70 and ratio_minor >= 0.35:
                tipo = "P2"
            else:
                tipo = "P1b"
        elif n >= 3:
            tipo = "P3"
        else:
            tipo = "P1b"

        conf = ((hmax - hmean * 0.4) / (hmax + 1e-6)) * np.exp(-0.005 * std_sep)
        prob = float(np.clip(conf, 0.0, 1.0))

        return tipo, prob, peaks, heights, mean_sep, std_sep, hmax, hmean

    tipo, prob, peaks, heights, mean_sep, std_sep, hmax, hmean = clasificar(y_smooth, height_thr, dist_min)
    nivel = "🔵 Alta" if prob > 0.75 else "🟠 Media" if prob > 0.45 else "🔴 Baja"

    # ========= VISUALIZACIÓN =========
    st.subheader("📊 Curva reconstruida hasta 1° mayo (JD ≤ 121)")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_julian, y_norm, color="lightgray", lw=1.2, label="Curva completa (0–300)")
    ax.plot(x_sub, y_smooth, color="royalblue", lw=2, label="Tramo ≤ 1° mayo (analizado)")

    if len(peaks):
        ax.plot(x_sub[peaks], y_smooth[peaks], "ro", label="Picos detectados")
        for i, p in enumerate(peaks):
            ax.text(x_sub[p], min(1.02, y_smooth[p] + 0.03), f"{x_sub[p]:.0f}", fontsize=8, rotation=45, ha="center")

    ax.axvline(121, color="red", linestyle="--", lw=1.2, label="1° mayo (JD 121)")
    ax.set_title(f"Clasificación al 1° mayo: {tipo} ({nivel}, prob={prob:.2f})")
    ax.set_xlabel("Día juliano")
    ax.set_ylabel("Emergencia relativa (normalizada)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # ========= DESCRIPCIÓN =========
    st.subheader("🌾 Descripción agronómica")
    st.markdown(f"""
    **Tipo detectado:** {tipo}  
    **Probabilidad:** {prob:.2f} ({nivel})  
    **Separación media:** {mean_sep:.1f}  
    **Proporción pico menor/mayor:** {heights[-1]/heights[0] if len(heights) > 1 else 0:.2f}  

    **Interpretación:**
    - **P1:** Emergencia única, compacta y temprana.  
    - **P1b:** Pico principal temprano + repunte leve posterior (como 2008).  
    - **P2:** Dos cohortes bien separadas y comparables.  
    - **P3:** Emergencia prolongada y continua.

    🔎 *Solo se utiliza información hasta el día juliano 121 (1° mayo); los picos posteriores no influyen en la clasificación.*
    """)

    # ========= VISTA DE VERIFICACIÓN =========
    st.subheader("🔍 Verificación del seguimiento de línea")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.imshow(gray, cmap="gray")
    ax2.plot(np.arange(len(y_norm)), h - y_norm * h, color="red", lw=1)
    ax2.set_title("Seguimiento del trazo negro (línea reconstruida)")
    st.pyplot(fig2)

else:
    st.info("📂 Cargá una imagen con eje X en días julianos (0–300). El análisis se corta automáticamente al 1° mayo (JD 121).")

