
# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador automático de patrones históricos (con registro CSV)
import streamlit as st
import cv2, os, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from datetime import datetime

st.set_page_config(page_title="Clasificador de patrones PREDWEEM", layout="wide")
st.title("🌾 Clasificación automática del patrón histórico (con probabilidad y registro)")

uploaded = st.file_uploader("📤 Cargar imagen (.png, .jpg)", type=["png", "jpg"])

CSV_PATH = "hist_patrones.csv"

if uploaded:
    # --- Leer imagen ---
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # --- Extraer canal azul ---
    blue = img[:,:,0].astype(float)
    curve = np.mean(blue, axis=0)
    curve = (curve.max() - curve) / (curve.max() - curve.min())
    curve = (curve - curve.min()) / (curve.max() - curve.min())

    # --- Detectar picos ---
    peaks, props = find_peaks(curve, height=0.2, distance=15)
    heights = props.get("peak_heights", [])
    n_picos = len(peaks)
    mean_sep = np.mean(np.diff(peaks)) if n_picos > 1 else 0
    std_sep = np.std(np.diff(peaks)) if n_picos > 2 else 0
    hmax, hmean = heights.max() if len(heights) else 0, np.mean(heights) if len(heights) else 0

    # --- Clasificación ---
    if n_picos == 1:
        tipo, desc = "P1", "Emergencia temprana y compacta"
    elif n_picos == 2 and mean_sep < 30:
        tipo, desc = "P1b", "Temprana con repunte corto"
    elif n_picos == 2:
        tipo, desc = "P2", "Bimodal"
    else:
        tipo, desc = "P3", "Extendida o multimodal"

    # --- Probabilidad ---
    conf = ((hmax - hmean) / (hmax + 0.01)) * np.exp(-0.02 * std_sep)
    prob = round(max(0.0, min(1.0, conf)), 3)
    if prob > 0.75:
        nivel = "🔵 Alta"
    elif prob > 0.45:
        nivel = "🟠 Media"
    else:
        nivel = "🔴 Baja"

    # --- Mostrar resultados ---
    st.image(uploaded, caption="Imagen analizada")
    fig, ax = plt.subplots()
    ax.plot(curve, color='royalblue', linewidth=2)
    if len(peaks):
        ax.plot(peaks, curve[peaks], "ro")
    ax.set_title(f"{tipo} — {desc}")
    ax.set_xlabel("Eje temporal relativo")
    ax.set_ylabel("EMERREL (normalizado)")
    st.pyplot(fig)

    st.success(f"✅ Patrón: **{tipo}** ({desc})")
    st.info(f"🔍 Probabilidad: **{nivel} ({prob:.2f})**  |  Nº picos: {n_picos}")

    # --- Guardar registro CSV ---
    row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), uploaded.name, tipo, prob, nivel, n_picos]
    file_exists = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Fecha análisis", "Archivo", "Tipo patrón", "Probabilidad", "Nivel", "N° picos"])
        writer.writerow(row)

    st.success(f"📄 Registro guardado en **{CSV_PATH}**")
