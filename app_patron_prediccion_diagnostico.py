# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador automático con límite temporal (1-may)
import streamlit as st
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from datetime import date
from pathlib import Path
import pandas as pd

# ========= CONFIGURACIÓN =========
st.set_page_config(page_title="Clasificador PREDWEEM — Corte 1° de mayo", layout="wide")
st.title("🌾 Clasificador PREDWEEM — Análisis limitado al 1° de mayo")

st.markdown("""
Clasifica curvas de emergencia (ANN o históricas) **usando únicamente la información disponible hasta el 1° de mayo (día juliano 121)**.  
Compatible con curvas negras históricas (2008–2012) y curvas azules ANN PREDWEEM.
""")

# ========= SIDEBAR =========
st.sidebar.header("⚙️ Parámetros de análisis")

modo = st.sidebar.radio(
    "🎨 Tipo de gráfico a analizar:",
    ["Detección automática", "Curva azul (ANN / PREDWEEM)", "Curva en negro (formato histórico)"],
    index=0
)

# Parámetros
height_thr = st.sidebar.slider("Umbral mínimo de altura", 0.01, 0.5, 0.15, 0.01)
dist_min = st.sidebar.slider("Distancia mínima entre picos", 5, 80, 20, 5)
gamma_corr = st.sidebar.slider("Corrección gamma", 0.2, 1.0, 0.4, 0.1)
gain = st.sidebar.slider("Ganancia", 0.5, 3.0, 1.5, 0.1)

# ========= CARGA =========
uploaded = st.file_uploader("📤 Cargar imagen (.png o .jpg)", type=["png", "jpg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(uploaded, caption="📈 Imagen original", use_container_width=True)

    # ========= DETECCIÓN AUTOMÁTICA =========
    if modo == "Detección automática":
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_mean, s_mean = np.mean(img_hsv[:, :, 0]), np.mean(img_hsv[:, :, 1])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        if 80 < h_mean < 130 and s_mean > 60:
            modo = "Curva azul (ANN / PREDWEEM)"
        elif contrast > 40 and s_mean < 40:
            modo = "Curva en negro (formato histórico)"
        else:
            modo = "Curva en negro (formato histórico)"
        st.sidebar.success(f"Modo detectado: **{modo}**")

    # ========= MÁSCARA SEGÚN MODO =========
    if modo.startswith("Curva azul"):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([80, 30, 160])
        upper_blue = np.array([150, 255, 255])
        mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.GaussianBlur(mask, (3, 7), 0)
        h, w = mask.shape
        mask = mask[int(h * 0.15):int(h * 0.95), :]

    st.image(mask, caption="🧭 Curva detectada (máscara base)", use_container_width=True)

    # ========= EXTRACCIÓN Y NORMALIZACIÓN =========
    curve = np.mean(mask, axis=0)
    curve_smooth = cv2.GaussianBlur(curve.reshape(1, -1), (1, 9), 0).flatten()
    curve_smooth = (curve_smooth - curve_smooth.min()) / (curve_smooth.max() - curve_smooth.min() + 1e-6)
    curve_smooth = np.clip(curve_smooth ** gamma_corr * gain, 0, 1)

    # Simular eje en días julianos (0–300)
    x_julian = np.linspace(0, 300, len(curve_smooth))

    # === Recorte al 1° de mayo (día juliano 121) ===
    mask_corte = x_julian <= 121
    x_sub = x_julian[mask_corte]
    y_sub = curve_smooth[mask_corte]

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

    tipo, prob, peaks, heights = clasificar(y_sub, height_thr, dist_min)
    nivel = "🔵 Alta" if prob > 0.75 else "🟠 Media" if prob > 0.45 else "🔴 Baja"

    # ========= GRÁFICO =========
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_julian, curve_smooth, color="gray", lw=1.5, label="Curva completa")
    ax.plot(x_sub, y_sub, color="royalblue", lw=2, label="Tramo analizado (≤ 1-may)")
    if len(peaks):
        ax.plot(x_sub[peaks], y_sub[peaks], "ro", label="Picos detectados")
    ax.axvline(121, color="red", linestyle="--", lw=1.2, label="1-may (día 121)")
    ax.set_title(f"Clasificación hasta 1° de mayo: {tipo} ({nivel}, prob={prob:.2f})")
    ax.set_xlabel("Día juliano")
    ax.set_ylabel("Emergencia relativa (normalizada)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # ========= DESCRIPCIÓN =========
    st.markdown(f"""
    ### 🌾 Clasificación al 1° de mayo
    **Tipo detectado:** {tipo}  
    **Probabilidad:** {prob:.2f} ({nivel})  

    **Interpretación agronómica:**  
    - **P1:** emergencia rápida y concentrada.  
    - **P1b:** pico temprano + repunte posterior leve.  
    - **P2:** dos cohortes separadas.  
    - **P3:** emergencia prolongada, varias cohortes.

    🔎 *El análisis se limitó al 1° de mayo (día juliano 121); los eventos posteriores no fueron considerados.*
    """)

else:
    st.info("Cargá una imagen (.png o .jpg) con el eje X en días julianos para analizar hasta el 1° de mayo.")
