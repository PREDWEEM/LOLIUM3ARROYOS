# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador automático de patrones históricos (Streamlit)
import streamlit as st
import cv2, os, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from datetime import datetime
from pathlib import Path

# ========== CONFIGURACIÓN ==========
st.set_page_config(
    page_title="Clasificador PREDWEEM",
    layout="wide",
    menu_items={"Get help": None, "Report a bug": None, "About": None}
)

# Carpeta de resultados
OUT_DIR = Path("resultados_clasif")
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUT_DIR / "hist_patrones.csv"

# ========== INTERFAZ ==========
st.title("🌾 Clasificador automático de patrón histórico — PREDWEEM")
st.markdown("""
Este módulo permite **clasificar automáticamente un patrón de emergencia**
a partir de una imagen del gráfico EMERREL.  
La clasificación usa el **1 de mayo** como fecha crítica y asigna un tipo:
**P1, P1b, P2 o P3**, con una probabilidad de éxito.
""")

uploaded = st.file_uploader("📤 Cargar imagen del gráfico (.png o .jpg)", type=["png", "jpg"])

# ========== PROCESAMIENTO ==========
if uploaded:
    # Leer imagen como array OpenCV
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Extraer canal azul (curva principal)
    blue = img[:, :, 0].astype(float)
    curve = np.mean(blue, axis=0)
    curve = (curve.max() - curve) / (curve.max() - curve.min())
    curve = (curve - curve.min()) / (curve.max() - curve.min())

    # Detectar picos
    peaks, props = find_peaks(curve, height=0.2, distance=15)
    heights = props.get("peak_heights", [])
    n_picos = len(peaks)
    mean_sep = np.mean(np.diff(peaks)) if n_picos > 1 else 0
    std_sep = np.std(np.diff(peaks)) if n_picos > 2 else 0
    hmax, hmean = heights.max() if len(heights) else 0, np.mean(heights) if len(heights) else 0

    # Clasificación heurística
    if n_picos == 1:
        tipo, desc = "P1", "Emergencia temprana y compacta"
    elif n_picos == 2 and mean_sep < 30:
        tipo, desc = "P1b", "Temprana con repunte corto"
    elif n_picos == 2:
        tipo, desc = "P2", "Bimodal"
    else:
        tipo, desc = "P3", "Extendida o multimodal"

    # Probabilidad de éxito
    conf = ((hmax - hmean) / (hmax + 0.01)) * np.exp(-0.02 * std_sep)
    prob = round(max(0.0, min(1.0, conf)), 3)
    if prob > 0.75:
        nivel = "🔵 Alta"
    elif prob > 0.45:
        nivel = "🟠 Media"
    else:
        nivel = "🔴 Baja"

    # ========== VISUALIZACIÓN ==========
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.image(uploaded, caption="📈 Imagen original analizada", use_container_width=True)
        st.metric("Tipo de patrón", tipo, desc)
        st.metric("Probabilidad", f"{prob:.2f}", nivel)
        st.write(f"**N° de picos detectados:** {n_picos}")

    with col2:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(curve, color='royalblue', linewidth=2)
        if len(peaks):
            ax.plot(peaks, curve[peaks], "ro")
        ax.set_title(f"Curva normalizada — {tipo}")
        ax.set_xlabel("Eje temporal relativo")
        ax.set_ylabel("EMERREL (0–1)")
        st.pyplot(fig)

    # ========== GUARDAR REGISTRO ==========
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

