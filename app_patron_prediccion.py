# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador automático de patrón histórico (imagen tipo gráfico)
import streamlit as st
import cv2, os, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from datetime import datetime
from pathlib import Path

# ======== CONFIGURACIÓN STREAMLIT ========
st.set_page_config(page_title="Clasificador PREDWEEM", layout="wide")
st.title("🌾 Clasificador automático del patrón histórico — Imágenes tipo EMERREL")

st.markdown("""
Este módulo detecta los **picos de emergencia (EMERREL)** a partir de una imagen del gráfico,
usando el **1 de mayo como fecha crítica (JD≈121)** para clasificar entre:
**P1, P1b, P2, P3**, con una estimación de **probabilidad de éxito** y registro automático.
""")

# Carpeta de salida
OUT_DIR = Path("resultados_clasif")
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUT_DIR / "hist_patrones.csv"

uploaded = st.file_uploader("📤 Cargar imagen (.png o .jpg)", type=["png", "jpg"])

# ======== PROCESAMIENTO ========
if uploaded:
    # Leer imagen
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- Máscara azul (curva EMERREL) ---
    lower_blue = np.array([90, 50, 70])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

    # --- Extraer curva promedio (1D) ---
    curve = np.mean(mask, axis=0)
    curve = np.ravel(curve)
    curve = cv2.GaussianBlur(curve.reshape(1, -1), (1, 9), 0).flatten()
    curve = (curve - curve.min()) / (curve.max() - curve.min() + 1e-6)
    curve = curve ** 0.5  # realce de contraste leve

    # --- Detección de picos más sensible ---
    peaks, props = find_peaks(curve, height=0.10, distance=25)
    heights = props.get("peak_heights", [])
    n_picos = len(peaks)
    mean_sep = np.mean(np.diff(peaks)) if n_picos > 1 else 0
    std_sep = np.std(np.diff(peaks)) if n_picos > 2 else 0
    hmax, hmean = heights.max() if len(heights) else 0, np.mean(heights) if len(heights) else 0

    # --- Clasificación heurística ---
    if n_picos == 1:
        tipo, desc = "P1", "Emergencia temprana y compacta"
    elif n_picos == 2 and mean_sep < 50:
        tipo, desc = "P1b", "Temprana con repunte corto"
    elif n_picos == 2:
        tipo, desc = "P2", "Bimodal"
    else:
        tipo, desc = "P3", "Extendida o multimodal"

    # --- Probabilidad ajustada ---
    conf = ((hmax - hmean * 0.5) / (hmax + 0.01)) * np.exp(-0.010 * std_sep)
    prob = round(max(0.0, min(1.0, conf)), 3)

    if prob > 0.75:
        nivel, color_box = "🔵 Alta", "#c8f7c5"  # verde claro
    elif prob > 0.45:
        nivel, color_box = "🟠 Media", "#fff3b0"  # amarillo
    else:
        nivel, color_box = "🔴 Baja", "#ffcccc"  # rosado

    # ======== VISUALIZACIÓN ========
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.image(uploaded, caption="📈 Imagen original analizada", use_container_width=True)
        st.markdown(f"<div style='background-color:{color_box}; padding:10px; border-radius:10px;'>"
                    f"<b>Tipo de patrón:</b> {tipo}<br>"
                    f"<b>Descripción:</b> {desc}<br>"
                    f"<b>Probabilidad:</b> {nivel} ({prob:.2f})<br>"
                    f"<b>N° de picos detectados:</b> {n_picos}</div>", unsafe_allow_html=True)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(curve, color='royalblue', linewidth=2)
        if len(peaks):
            ax.plot(peaks, curve[peaks], "ro")

        # Línea del 1 de mayo (JD ≈ 121)
        jd_mayo = int(len(curve) * 121 / 300)
        ax.axvline(jd_mayo, color='red', linestyle='--', linewidth=1.5, label="1 de mayo (JD≈121)")
        ax.legend(loc='upper right')

        ax.set_title(f"Curva detectada — {tipo}")
        ax.set_xlabel("Eje temporal relativo (0–300)")
        ax.set_ylabel("Intensidad normalizada")
        st.pyplot(fig)

    # ======== REGISTRO ========
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [now, uploaded.name, tipo, prob, nivel, n_picos]
    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Fecha análisis", "Archivo", "Tipo patrón", "Probabilidad", "Nivel", "N° picos"])
        writer.writerow(row)

    st.success(f"📄 Registro guardado en **{CSV_PATH}**")

    # Mostrar historial
    if CSV_PATH.exists():
        df = np.genfromtxt(CSV_PATH, delimiter=",", dtype=str, skip_header=1)
        if len(df) > 0:
            st.subheader("📚 Historial de clasificaciones")
            st.dataframe(df)
