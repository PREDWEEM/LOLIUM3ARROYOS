# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador unificado automático (azul, negro o áreas)
import streamlit as st
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from datetime import datetime, timedelta, date
from pathlib import Path
import pandas as pd

# ========= CONFIGURACIÓN =========
st.set_page_config(page_title="Clasificador PREDWEEM — Automático", layout="wide")
st.title("🌾 Clasificador PREDWEEM — Detección automática de curvas")

st.markdown("""
Compatible con gráficos tipo **EMERREL (curva azul ANN)**, **curvas negras históricas (2008–2012)**  
y **áreas multicolor (verde/amarillo/rojo)**.  
Permite ajustar el eje temporal y generar una descripción agronómica completa del patrón detectado.
""")

# ========= SIDEBAR =========
st.sidebar.header("⚙️ Parámetros de análisis")

# --- Tipo de gráfico ---
modo = st.sidebar.radio(
    "🎨 Tipo de gráfico a analizar:",
    ["Detección automática",
     "Curva azul (ANN / PREDWEEM)",
     "Curva en negro (formato histórico)",
     "Áreas de color (verde/amarillo/rojo)"],
    index=0
)

# --- Parámetros de detección ---
st.sidebar.subheader("🎨 Parámetros de color (solo modo azul)")
h_min = st.sidebar.slider("Hue mínimo (H)", 70, 130, 80)
h_max = st.sidebar.slider("Hue máximo (H)", 110, 160, 150)
s_min = st.sidebar.slider("Saturación mínima (S)", 0, 255, 30)
v_min = st.sidebar.slider("Brillo mínimo (V)", 0, 255, 160)

# --- Picos ---
st.sidebar.subheader("📈 Detección de picos")
height_thr = st.sidebar.slider("Umbral mínimo de altura", 0.01, 0.5, 0.18, 0.01)
dist_min = st.sidebar.slider("Distancia mínima entre picos", 5, 80, 20, 5)
gamma_corr = st.sidebar.slider("Realce de contraste (γ)", 0.2, 1.0, 0.4, 0.1)
gain = st.sidebar.slider("Ganancia de contraste", 0.5, 3.0, 1.5, 0.1)

# --- Escala temporal ---
st.sidebar.subheader("📅 Escala temporal")
year_ref = st.sidebar.number_input("Año de referencia", min_value=2000, max_value=2100, value=2025)
fecha_inicio = st.sidebar.date_input("Fecha inicial", date(year_ref, 2, 1))
fecha_fin = st.sidebar.date_input("Fecha final", date(year_ref, 8, 18))

offset_dias = st.sidebar.slider("🧭 Desplazamiento (± días)", -60, 60, 0, 1)
escala_factor = st.sidebar.slider("Escala temporal (%)", 50, 150, 100, 5)

# ========= CARGA =========
uploaded = st.file_uploader("📤 Cargar imagen (.png o .jpg)", type=["png", "jpg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(uploaded, caption="📈 Imagen original", use_container_width=True)

    # ========= DETECCIÓN AUTOMÁTICA =========
    modo_auto = None
    if modo == "Detección automática":
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_mean = np.mean(img_hsv[:, :, 0])
        s_mean = np.mean(img_hsv[:, :, 1])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()

        if 80 < h_mean < 130 and s_mean > 60:
            modo_auto = "Curva azul (ANN / PREDWEEM)"
        elif contrast > 40 and s_mean < 40:
            modo_auto = "Curva en negro (formato histórico)"
        else:
            modo_auto = "Áreas de color (verde/amarillo/rojo)"
        st.sidebar.success(f"Modo detectado automáticamente: **{modo_auto}**")
        modo = modo_auto

    # ========= CONSTRUCCIÓN DE MÁSCARA =========
    if modo.startswith("Curva azul"):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([h_min, s_min, v_min])
        upper_blue = np.array([h_max, 255, 255])
        mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

    elif modo.startswith("Curva en negro"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, mask = cv2.threshold(gray_blur, 80, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (1, 9), 0)
        h, w = mask.shape
        mask = mask[int(h * 0.15):int(h * 0.95), :]

    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, mask = cv2.threshold(gray_blur, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(mask)
        mask = cv2.GaussianBlur(mask, (1, 7), 0)
        h, w = mask.shape
        mask = mask[int(h * 0.25):int(h * 0.9), :]

    st.image(mask, caption="🧭 Curva detectada (máscara base)", use_container_width=True)

    # ========= EXTRACCIÓN Y SUAVIZADO =========
    curve = np.mean(mask, axis=0)
    curve_smooth = cv2.GaussianBlur(curve.reshape(1, -1), (1, 9), 0).flatten()
    curve_smooth = (curve_smooth - curve_smooth.min()) / (curve_smooth.max() - curve_smooth.min() + 1e-6)
    curve_smooth = np.clip(curve_smooth ** gamma_corr * gain, 0, 1)

    total_dias = (fecha_fin - fecha_inicio).days
    dias_reales = int(total_dias * (escala_factor / 100))
    fecha_fin_adj = fecha_inicio + timedelta(days=dias_reales)
    fechas = pd.date_range(start=fecha_inicio, end=fecha_fin_adj, periods=len(curve_smooth))
    fechas = fechas + timedelta(days=offset_dias)

    # ========= CLASIFICACIÓN =========
    def clasificar(curva, thr, dist):
        peaks, props = find_peaks(curva, height=thr, distance=dist)
        heights = props.get("peak_heights", [])
        n = len(peaks)
        mean_sep = np.mean(np.diff(peaks)) if n > 1 else 0
        std_sep = np.std(np.diff(peaks)) if n > 2 else 0
        hmax, hmean = (heights.max() if len(heights) else 0), (np.mean(heights) if len(heights) else 0)

        if n == 1: tipo = "P1"
        elif n == 2 and mean_sep < 50: tipo = "P1b"
        elif n == 2: tipo = "P2"
        else: tipo = "P3"

        conf = ((hmax - hmean * 0.4) / (hmax + 1e-6)) * np.exp(-0.008 * std_sep) if hmax > 0 else 0.0
        prob = float(np.clip(conf, 0.0, 1.0))
        return tipo, prob, peaks, heights

    tipo, prob, peaks, heights = clasificar(curve_smooth, height_thr, dist_min)
    nivel = "🔵 Alta" if prob > 0.75 else "🟠 Media" if prob > 0.45 else "🔴 Baja"

    # ========= VISUALIZACIÓN =========
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fechas, curve_smooth, color="royalblue", lw=2)
    if len(peaks):
        ax.plot(fechas[peaks], curve_smooth[peaks], "ro")
    ax.set_title(f"Patrón detectado: {tipo} ({nivel}, prob={prob:.2f})")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ========= DESCRIPCIÓN =========
    st.markdown(f"""
    ### 🌾 Descripción del patrón
    **Tipo:** {tipo}  
    **Probabilidad:** {prob:.2f} ({nivel})  

    **Interpretación agronómica:**  
    - **P1:** emergencia rápida y concentrada.  
    - **P1b:** pico temprano + pequeño repunte posterior.  
    - **P2:** dos cohortes separadas (bimodal).  
    - **P3:** emergencia prolongada y escalonada.  
    """)

else:
    st.info("Cargá una imagen (.png o .jpg) para iniciar la clasificación.")

