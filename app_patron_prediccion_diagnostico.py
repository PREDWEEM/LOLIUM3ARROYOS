# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador robusto (detección de curva negra, límite al 1° mayo)
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

# ========== CONFIGURACIÓN STREAMLIT ==========
st.set_page_config(page_title="Clasificador PREDWEEM — Robustizado (1° mayo)", layout="wide")
st.title("🌾 Clasificador PREDWEEM — Corte al 1° de mayo (detección robusta de curvas negras)")

st.markdown("""
Analiza **gráficos históricos en escala de grises o curvas ANN** para identificar el patrón de emergencia (P1, P1b, P2, P3).  
Usa sólo la información hasta el **1° de mayo (día juliano 121)** y corrige detecciones falsas mediante morfología y suavizado.
""")

# ========== SIDEBAR ==========
st.sidebar.header("⚙️ Parámetros de detección")

height_thr = st.sidebar.slider("Umbral mínimo de altura", 0.05, 0.5, 0.22, 0.01)
dist_min = st.sidebar.slider("Distancia mínima entre picos (px ≈ días julianos)", 10, 80, 35, 5)
gamma_corr = st.sidebar.slider("Corrección gamma", 0.2, 1.0, 0.4, 0.1)
gain = st.sidebar.slider("Ganancia de contraste", 0.5, 3.0, 1.5, 0.1)

uploaded = st.file_uploader("📤 Cargar imagen del gráfico (.png o .jpg)", type=["png", "jpg"])

# ========== PROCESAMIENTO ==========
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(uploaded, caption="📈 Imagen original", use_container_width=True)

    # Convertir a gris y realzar contraste
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    # Binarización adaptativa inversa
    mask = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 41, 12
    )

    # Morfología para suavizar y cerrar huecos
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    st.image(mask, caption="🧭 Curva procesada (limpia y continua)", use_container_width=True)

    # Promedio vertical y suavizado
    curve = np.mean(mask, axis=0)
    curve_smooth = savgol_filter(curve, 35, 3)
    curve_smooth = (curve_smooth - curve_smooth.min()) / (curve_smooth.max() - curve_smooth.min() + 1e-6)
    curve_smooth = np.clip(curve_smooth ** gamma_corr * gain, 0, 1)

    # Escala temporal en días julianos
    x_julian = np.linspace(0, 300, len(curve_smooth))

    # Corte al 1° de mayo (día juliano 121)
    mask_corte = x_julian <= 121
    x_sub, y_sub = x_julian[mask_corte], curve_smooth[mask_corte]

    # ========== CLASIFICACIÓN ==========
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

        # Lógica revisada (P1b más estricta)
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

    tipo, prob, peaks, heights, mean_sep, std_sep, hmax, hmean = clasificar(y_sub, height_thr, dist_min)
    nivel = "🔵 Alta" if prob > 0.75 else "🟠 Media" if prob > 0.45 else "🔴 Baja"

    # ========== VISUALIZACIÓN ==========
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_julian, curve_smooth, color="lightgray", lw=1.5, label="Curva completa")
    ax.plot(x_sub, y_sub, color="royalblue", lw=2, label="Tramo ≤ 1-mayo")
    if len(peaks):
        ax.plot(x_sub[peaks], y_sub[peaks], "ro", label="Picos detectados")
    ax.axvline(121, color="red", linestyle="--", lw=1.2, label="1° mayo (JD 121)")
    ax.set_title(f"Clasificación al 1° mayo: {tipo} ({nivel}, prob={prob:.2f})")
    ax.set_xlabel("Día juliano")
    ax.set_ylabel("Emergencia relativa (normalizada)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # ========== DESCRIPCIÓN ==========
    st.markdown(f"""
    ### 🌾 Clasificación
    - **Tipo detectado:** {tipo}  
    - **Probabilidad:** {prob:.2f} ({nivel})  
    - **Separación media entre picos:** {mean_sep:.1f}  
    - **Proporción pico menor / mayor:** {ratio_minor:.2f}

    **Interpretación agronómica:**
    - **P1:** Emergencia única, compacta y temprana.  
    - **P1b:** Pico principal temprano + pequeño repunte posterior (como 2008).  
    - **P2:** Dos cohortes bien separadas y de magnitud similar.  
    - **P3:** Emergencia prolongada con múltiples cohortes.

    🔎 *El análisis se limita al día juliano 121 (1° mayo); eventos posteriores no se usan para clasificar.*
    """)

else:
    st.info("Cargá una imagen (.png o .jpg) con el eje X en días julianos (0–300) para analizar hasta el 1° de mayo.")


