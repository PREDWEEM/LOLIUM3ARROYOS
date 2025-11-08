# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador interactivo con ajuste fino del eje X
import streamlit as st
import cv2, os, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from datetime import datetime, timedelta, date
from pathlib import Path
import pandas as pd

# ======== CONFIGURACIÓN STREAMLIT ========
st.set_page_config(page_title="Clasificador PREDWEEM — Ajuste fino eje X", layout="wide")
st.title("🌾 Clasificador de patrón histórico — Ajuste fino del eje temporal")

st.markdown("""
Permite **ajustar el eje X** a la curva detectada:
- 🕹️ Desplazá la curva horizontalmente (offset de días)
- 🔍 Cambiá la escala temporal (compresión o estiramiento)
- 📆 Ajustá manualmente las fechas de inicio y fin

Así podés **alinear los picos detectados con las fechas reales** (por ejemplo, corregir un pico mal posicionado).
""")

# ======== SIDEBAR ========
st.sidebar.header("⚙️ Parámetros de análisis")

# --- Color (HSV) ---
st.sidebar.subheader("🎨 Detección de color azul (curva EMERREL)")
h_min = st.sidebar.slider("Hue mínimo (H)", 70, 130, 80)
h_max = st.sidebar.slider("Hue máximo (H)", 110, 160, 150)
s_min = st.sidebar.slider("Saturación mínima (S)", 0, 255, 30)
v_min = st.sidebar.slider("Brillo mínimo (V)", 0, 255, 160)

# --- Curva y picos ---
st.sidebar.subheader("📈 Detección de picos")
height_thr = st.sidebar.slider("Umbral mínimo de altura", 0.01, 0.5, 0.18, 0.01)
dist_min = st.sidebar.slider("Distancia mínima entre picos", 5, 80, 10, 5)
gamma_corr = st.sidebar.slider("Realce de contraste (γ)", 0.2, 1.0, 0.4, 0.1)
gain = st.sidebar.slider("Ganancia de contraste", 0.5, 3.0, 1.5, 0.1)

# --- Escala temporal ---
st.sidebar.subheader("📅 Escala temporal")
year_ref = st.sidebar.number_input("Año de referencia", min_value=2000, max_value=2100, value=2025)
fecha_inicio = st.sidebar.date_input("Fecha inicial", date(year_ref, 2, 1))
fecha_fin = st.sidebar.date_input("Fecha final", date(year_ref, 8, 18))
fecha_mayo = date(year_ref, 5, 1)

# --- Ajuste fino del eje X ---
st.sidebar.subheader("🧭 Ajuste fino del eje X")
offset_dias = st.sidebar.slider("Desplazamiento temporal (± días)", -60, 60, 38, 1)
escala_factor = st.sidebar.slider("Escala temporal (%)", 50, 150, 100, 5)

# ======== SALIDA ========
OUT_DIR = Path("resultados_clasif")
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUT_DIR / "hist_patrones.csv"

uploaded = st.file_uploader("📤 Cargar imagen (.png o .jpg)", type=["png", "jpg"])

if uploaded:
    # --- Leer imagen ---
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- Detección de color ajustable ---
    lower_blue = np.array([h_min, s_min, v_min])
    upper_blue = np.array([h_max, 255, 255])
    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

    st.image(uploaded, caption="📈 Imagen original analizada", use_container_width=True)
    st.image(mask, caption="🎨 Máscara azul detectada (curva EMERREL)", use_container_width=True)

    # --- Extracción y suavizado de curva ---
    curve = np.mean(mask, axis=0)
    curve = np.ravel(curve)
    curve_smooth = cv2.GaussianBlur(curve.reshape(1, -1), (1, 9), 0).flatten()
    curve_smooth = (curve_smooth - curve_smooth.min()) / (curve_smooth.max() - curve_smooth.min() + 1e-6)
    curve_smooth = curve_smooth ** gamma_corr
    curve_smooth = np.clip(curve_smooth * gain, 0, 1)

    # --- Ajuste de escala y desplazamiento ---
    total_dias = (fecha_fin - fecha_inicio).days
    dias_reales = int(total_dias * (escala_factor / 100))
    fecha_fin_adj = fecha_inicio + timedelta(days=dias_reales)
    fechas = pd.date_range(start=fecha_inicio, end=fecha_fin_adj, periods=len(curve_smooth))
    fechas = fechas + timedelta(days=offset_dias)

    # --- Detección de picos ---
    peaks, props = find_peaks(curve_smooth, height=height_thr, distance=dist_min)
    heights = props.get("peak_heights", [])
    n_picos = len(peaks)
    mean_sep = np.mean(np.diff(peaks)) if n_picos > 1 else 0
    std_sep = np.std(np.diff(peaks)) if n_picos > 2 else 0
    hmax, hmean = (heights.max() if len(heights) else 0), (np.mean(heights) if len(heights) else 0)

    # --- Clasificación del patrón ---
    if n_picos == 1:
        tipo, desc = "P1", "Emergencia temprana y compacta"
    elif n_picos == 2 and mean_sep < 50:
        tipo, desc = "P1b", "Temprana con repunte corto"
    elif n_picos == 2:
        tipo, desc = "P2", "Bimodal"
    else:
        tipo, desc = "P3", "Extendida o multimodal"

    # --- Probabilidad ---
    conf = ((hmax - hmean * 0.4) / (hmax + 0.01)) * np.exp(-0.008 * std_sep)
    prob = round(max(0.0, min(1.0, conf)), 3)
    if prob > 0.75:
        nivel, color_box = "🔵 Alta", "#c8f7c5"
    elif prob > 0.45:
        nivel, color_box = "🟠 Media", "#fff3b0"
    else:
        nivel, color_box = "🔴 Baja", "#ffcccc"

    # --- Visualización del gráfico ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fechas, curve_smooth, color='royalblue', linewidth=2, label="Curva suavizada")
    if len(peaks):
        ax.plot(fechas[peaks], curve_smooth[peaks], "ro", label="Picos detectados")

    # Sombras predictiva / posterior al corte
    ax.axvspan(fechas.min(), fecha_mayo, color='lightblue', alpha=0.15, label="Periodo predictivo (≤1 mayo)")
    ax.axvspan(fecha_mayo, fechas.max(), color='lightcoral', alpha=0.15, label="Posterior al corte (≥1 mayo)")
    ax.axvline(fecha_mayo, color='red', linestyle='--', linewidth=1.5, label="1 de mayo")
    ax.axhline(height_thr, color='gray', linestyle='--', alpha=0.4, label=f"Umbral={height_thr:.2f}")

    ax.legend(loc='upper right')
    ax.set_xlabel(f"Fecha calendario ajustada ({fechas.min().strftime('%d-%b')} → {fechas.max().strftime('%d-%b')})")
    ax.set_ylabel("Intensidad normalizada")
    ax.set_title(f"Curva detectada — {tipo} (Año {year_ref})")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # --- Resultados numéricos ---
    st.markdown(f"<div style='background-color:{color_box}; padding:10px; border-radius:10px;'>"
                f"<b>Tipo de patrón:</b> {tipo}<br>"
                f"<b>Descripción:</b> {desc}<br>"
                f"<b>Probabilidad:</b> {nivel} ({prob:.2f})<br>"
                f"<b>N° picos:</b> {n_picos}<br>"
                f"<b>hmax:</b> {hmax:.2f} | <b>hmean:</b> {hmean:.2f}<br>"
                f"<b>mean_sep:</b> {mean_sep:.1f} | <b>std_sep:</b> {std_sep:.1f}</div>", 
                unsafe_allow_html=True)

    # --- Descripción agronómica ---
    st.subheader("🧩 Descripción del patrón detectado")
    if len(peaks):
        fechas_picos = [fechas[p].date() for p in peaks]
        picos_post_mayo = [f for f in fechas_picos if f > fecha_mayo]
    else:
        picos_post_mayo = []

    if tipo == "P1":
        interpretacion = "Emergencia temprana concentrada (una cohorte dominante antes de mayo)."
    elif tipo == "P1b":
        interpretacion = "Emergencia temprana con pequeño repunte luego del 1° de mayo."
    elif tipo == "P2":
        interpretacion = "Emergencia bimodal: dos pulsos bien definidos antes y después del 1° de mayo."
    else:
        interpretacion = "Emergencia extendida o multimodal: prolongada en el tiempo."

    if picos_post_mayo:
        fechas_str = ", ".join([f.strftime("%d-%b") for f in picos_post_mayo])
        st.info(f"📆 Picos posteriores al 1° de mayo: **{fechas_str}**")
    else:
        st.info("📆 No se detectaron picos posteriores al 1° de mayo.")
    st.write(interpretacion)

    # --- Registro CSV ---
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [now, uploaded.name, tipo, prob, nivel, n_picos]
    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Fecha análisis", "Archivo", "Tipo patrón", "Probabilidad", "Nivel", "N° picos"])
        writer.writerow(row)

    st.success(f"📄 Registro guardado en **{CSV_PATH}**")
