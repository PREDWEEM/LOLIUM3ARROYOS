
# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador de patrón histórico con diagnóstico visual
import streamlit as st
import cv2, os, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from datetime import datetime
from pathlib import Path

# ======== CONFIGURACIÓN STREAMLIT ========
st.set_page_config(page_title="Clasificador PREDWEEM — Diagnóstico", layout="wide")
st.title("🌾 Clasificador de patrón histórico — Modo Diagnóstico")

st.markdown("""
Esta versión muestra todos los pasos intermedios del análisis de imagen:
1. Máscara azul detectada  
2. Curva promedio extraída  
3. Curva suavizada y normalizada  
4. Picos detectados  
5. Clasificación + probabilidad  
---
🧠 Permite depurar umbrales y asegurar que la detección es correcta.
""")

# Carpeta de salida
OUT_DIR = Path("resultados_clasif")
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUT_DIR / "hist_patrones.csv"

uploaded = st.file_uploader("📤 Cargar imagen (.png o .jpg)", type=["png", "jpg"])

if uploaded:
    # --- Leer imagen ---
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    st.image(uploaded, caption="📈 Imagen original analizada", use_container_width=True)

    # --- Detección de color (rango ampliado) ---
    lower_blue = np.array([80, 30, 40])
    upper_blue = np.array([150, 255, 255])
    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

    st.image(mask, caption="🎨 Máscara azul detectada (región de la curva)", use_container_width=True)

    # --- Extracción de curva ---
    curve = np.mean(mask, axis=0)
    curve = np.ravel(curve)
    st.line_chart(curve)
    st.caption("🔍 Curva original extraída (antes de suavizado)")

    # --- Suavizado y normalización ---
    curve_smooth = cv2.GaussianBlur(curve.reshape(1, -1), (1, 9), 0).flatten()
    curve_smooth = (curve_smooth - curve_smooth.min()) / (curve_smooth.max() - curve_smooth.min() + 1e-6)
    curve_smooth = curve_smooth ** 0.4  # realce de contraste
    curve_smooth = np.clip(curve_smooth * 1.5, 0, 1)

    # --- Detección de picos ---
    peaks, props = find_peaks(curve_smooth, height=0.08, distance=20)
    heights = props.get("peak_heights", [])
    n_picos = len(peaks)
    mean_sep = np.mean(np.diff(peaks)) if n_picos > 1 else 0
    std_sep = np.std(np.diff(peaks)) if n_picos > 2 else 0
    hmax, hmean = (heights.max() if len(heights) else 0), (np.mean(heights) if len(heights) else 0)

    # --- Clasificación heurística ---
    if n_picos == 1:
        tipo, desc = "P1", "Emergencia temprana y compacta"
    elif n_picos == 2 and mean_sep < 50:
        tipo, desc = "P1b", "Temprana con repunte corto"
    elif n_picos == 2:
        tipo, desc = "P2", "Bimodal"
    else:
        tipo, desc = "P3", "Extendida o multimodal"

    # --- Probabilidad (recalibrada) ---
    conf = ((hmax - hmean * 0.4) / (hmax + 0.01)) * np.exp(-0.008 * std_sep)
    prob = round(max(0.0, min(1.0, conf)), 3)

    if prob > 0.75:
        nivel, color_box = "🔵 Alta", "#c8f7c5"
    elif prob > 0.45:
        nivel, color_box = "🟠 Media", "#fff3b0"
    else:
        nivel, color_box = "🔴 Baja", "#ffcccc"

    # --- Visualización de picos detectados ---
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(curve_smooth, color='royalblue', linewidth=2, label="Curva suavizada")
    if len(peaks):
        ax.plot(peaks, curve_smooth[peaks], "ro", label="Picos detectados")

    # Línea del 1 de mayo (JD≈121)
    jd_mayo = int(len(curve_smooth) * 121 / 300)
    ax.axvline(jd_mayo, color='red', linestyle='--', linewidth=1.5, label="1 de mayo (JD≈121)")
    ax.axhline(0.08, color='gray', linestyle='--', alpha=0.4, label="Umbral=0.08")
    ax.legend(loc='upper right')
    ax.set_xlabel("Eje temporal relativo (0–300)")
    ax.set_ylabel("Intensidad normalizada")
    ax.set_title(f"Curva detectada — {tipo}")
    st.pyplot(fig)

    # --- Mostrar resultados numéricos ---
    st.markdown(f"<div style='background-color:{color_box}; padding:10px; border-radius:10px;'>"
                f"<b>Tipo de patrón:</b> {tipo}<br>"
                f"<b>Descripción:</b> {desc}<br>"
                f"<b>Probabilidad:</b> {nivel} ({prob:.2f})<br>"
                f"<b>N° picos:</b> {n_picos}<br>"
                f"<b>hmax:</b> {hmax:.2f} | <b>hmean:</b> {hmean:.2f}<br>"
                f"<b>mean_sep:</b> {mean_sep:.1f} | <b>std_sep:</b> {std_sep:.1f}</div>", 
                unsafe_allow_html=True)

    # --- Guardar registro CSV ---
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [now, uploaded.name, tipo, prob, nivel, n_picos]
    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Fecha análisis", "Archivo", "Tipo patrón", "Probabilidad", "Nivel", "N° picos"])
        writer.writerow(row)

    st.success(f"📄 Registro guardado en **{CSV_PATH}**")

    if CSV_PATH.exists():
        df = np.genfromtxt(CSV_PATH, delimiter=",", dtype=str, skip_header=1)
        if len(df) > 0:
            st.subheader("📚 Historial de clasificaciones")
            st.dataframe(df)
