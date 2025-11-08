# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador unificado (curva azul + áreas de color)
import streamlit as st
import cv2, os, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from datetime import datetime, timedelta, date
from pathlib import Path
import pandas as pd

# ========= CONFIGURACIÓN =========
st.set_page_config(page_title="Clasificador PREDWEEM — Unificado", layout="wide")
st.title("🌾 Clasificador PREDWEEM — Detección adaptable con ajuste fino de eje temporal")

st.markdown("""
Compatible con gráficos tipo **EMERREL (curva azul)** y **áreas multicolor (verde/amarillo/rojo)**.  
Permite ajustar el eje X manualmente y generar una descripción agronómica completa del patrón detectado.
""")

# ========= SIDEBAR =========
st.sidebar.header("⚙️ Parámetros de análisis")

# --- Tipo de gráfico ---
modo = st.sidebar.radio(
    "🎨 Tipo de gráfico a analizar:",
    ["Curva azul (ANN / PREDWEEM)", "Áreas de color (verde/amarillo/rojo)"],
    index=0
)

# --- Color HSV (solo curva azul) ---
st.sidebar.subheader("🎨 Detección de color azul (solo modo curva)")
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
fecha_mayo = date(year_ref, 5, 1)

# --- Ajuste manual ---
st.sidebar.subheader("🧭 Ajuste fino del eje X")
offset_dias = st.sidebar.slider("Desplazamiento (± días)", -60, 60, 0, 1)
escala_factor = st.sidebar.slider("Escala temporal (%)", 50, 150, 100, 5)

# --- Autoajuste ---
autoajuste = st.sidebar.button("⚡ Autoajustar a rango típico (1-mar → 20-jul)")

# ========= SALIDA =========
OUT_DIR = Path("resultados_clasif"); OUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUT_DIR / "hist_patrones.csv"

uploaded = st.file_uploader("📤 Cargar imagen (.png o .jpg)", type=["png", "jpg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(uploaded, caption="📈 Imagen original", use_container_width=True)

    # ========= DETECCIÓN DE CURVA SEGÚN MODO =========
    if modo.startswith("Curva azul"):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([h_min, s_min, v_min])
        upper_blue = np.array([h_max, 255, 255])
        mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, mask = cv2.threshold(gray_blur, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(mask)
        mask = cv2.GaussianBlur(mask, (1, 7), 0)
        h, w = mask.shape
        mask = mask[int(h * 0.25):int(h * 0.9), :]

    st.caption(f"Modo de lectura activo: **{modo}**")
    st.image(mask, caption="🧭 Curva procesada / máscara base", use_container_width=True)

    # ========= EXTRACCIÓN Y SUAVIZADO =========
    curve = np.mean(mask, axis=0)
    curve = np.ravel(curve)
    curve_smooth = cv2.GaussianBlur(curve.reshape(1, -1), (1, 9), 0).flatten()
    curve_smooth = (curve_smooth - curve_smooth.min()) / (curve_smooth.max() - curve_smooth.min() + 1e-6)
    curve_smooth = np.clip(curve_smooth ** gamma_corr * gain, 0, 1)

    # ========= AJUSTE TEMPORAL =========
    total_dias = (fecha_fin - fecha_inicio).days
    dias_reales = int(total_dias * (escala_factor / 100))
    fecha_fin_adj = fecha_inicio + timedelta(days=dias_reales)
    fechas = pd.date_range(start=fecha_inicio, end=fecha_fin_adj, periods=len(curve_smooth))
    fechas = fechas + timedelta(days=offset_dias)

    # ========= DETECCIÓN DE PICOS =========
    peaks, props = find_peaks(curve_smooth, height=height_thr, distance=dist_min)
    heights = props.get("peak_heights", [])
    n_picos = len(peaks)
    mean_sep = np.mean(np.diff(peaks)) if n_picos > 1 else 0
    std_sep = np.std(np.diff(peaks)) if n_picos > 2 else 0
    hmax, hmean = (heights.max() if len(heights) else 0), (np.mean(heights) if len(heights) else 0)

    # --- Autoajuste opcional ---
    if autoajuste and len(peaks) > 1:
        f1, f2 = fechas[peaks[0]], fechas[peaks[-1]]
        dur_curva = (f2 - f1).days
        nueva_escala = (142 / max(dur_curva, 1)) * 100
        nuevo_offset = (fecha_inicio + timedelta(days=28) - f1).days
        st.sidebar.success(f"Autoajuste aplicado: Escala {nueva_escala:.0f}%, Offset {nuevo_offset:+d} días")
        escala_factor, offset_dias = nueva_escala, nuevo_offset

          # ========= FUNCIÓN DE CLASIFICACIÓN =========
    from scipy.signal import find_peaks
    import numpy as np
    
    def clasificar(curva, thr, dist):
        """
        Detecta picos y clasifica el patrón según cantidad y separación.
        Devuelve: tipo, probabilidad, peaks, heights, mean_sep, std_sep, hmax, hmean
        """
        peaks, props = find_peaks(curva, height=thr, distance=dist)
        heights = props.get("peak_heights", [])
        n = len(peaks)
        mean_sep = np.mean(np.diff(peaks)) if n > 1 else 0.0
        std_sep  = np.std(np.diff(peaks)) if n > 2 else 0.0
        hmax     = (heights.max() if len(heights) else 0.0)
        hmean    = (np.mean(heights) if len(heights) else 0.0)
    
        if n == 1:
            tipo = "P1"
        elif n == 2 and mean_sep < 50:
            tipo = "P1b"
        elif n == 2:
            tipo = "P2"
        else:
            tipo = "P3"
    
        conf = ((hmax - hmean * 0.4) / (hmax + 1e-6)) * np.exp(-0.008 * std_sep) if hmax > 0 else 0.0
        prob = float(np.clip(conf, 0.0, 1.0))
        return tipo, prob, peaks, heights, mean_sep, std_sep, hmax, hmean
          
         
    # ========= CLASIFICACIÓN GLOBAL (solo 1-feb → 1-may) =========
    fecha_febrero = date(year_ref, 2, 1)
    fecha_mayo = date(year_ref, 5, 1)
    
    # Creamos un DataFrame temporal para facilitar el recorte
    df_curva = pd.DataFrame({"fecha": fechas, "valor": curve_smooth})
    mask_periodo = (df_curva["fecha"].dt.date >= fecha_febrero) & (df_curva["fecha"].dt.date <= fecha_mayo)
    df_periodo = df_curva.loc[mask_periodo].reset_index(drop=True)
    
    if len(df_periodo) > 10:
        # Solo se clasifica usando el tramo 1-feb → 1-may
        tipo, prob, peaks, heights, mean_sep, std_sep, hmax, hmean = clasificar(
            df_periodo["valor"].to_numpy(), height_thr, dist_min
        )
        fechas_sub = df_periodo["fecha"].to_numpy()
        nivel = "🔵 Alta" if prob > 0.75 else "🟠 Media" if prob > 0.45 else "🔴 Baja"
    else:
        tipo, prob, nivel = "-", 0.0, "—"
        peaks, heights, mean_sep, std_sep, hmax, hmean = [], [], 0, 0, 0, 0
        fechas_sub = []
    
    # ========= VISUALIZACIÓN =========
    fig, ax = plt.subplots(figsize=(11, 4))
    
    # Curva completa en gris (referencia)
    ax.plot(fechas, curve_smooth, color='lightgray', linewidth=1.5, label="Curva completa")
    
    # Tramo analizado (1-feb → 1-may) en azul
    ax.plot(df_periodo["fecha"], df_periodo["valor"], color='royalblue', linewidth=2, label="Tramo 1-feb → 1-may")
    
    # Convertimos las fechas de submuestra a datetime para evitar errores de tipo numpy.datetime64
    fechas_sub = pd.to_datetime(fechas_sub).to_pydatetime() if len(fechas_sub) else []
    
    # Marcamos los picos detectados
    if len(peaks):
        ax.plot([fechas_sub[p] for p in peaks],
                [df_periodo["valor"].iloc[p] for p in peaks],
                "ro", label="Picos detectados")
        for p in peaks:
            fecha_pico = pd.to_datetime(fechas_sub[p]).to_pydatetime()
            ax.text(
                fecha_pico,
                min(1.0, df_periodo["valor"].iloc[p] + 0.03),
                fecha_pico.strftime("%d-%b"),
                rotation=45,
                fontsize=8,
                ha='center'
            )
    
    # Líneas de referencia: 1-febrero y 1-mayo
    ax.axvline(fecha_febrero, color='green', linestyle='--', linewidth=1, label="Inicio (1-feb)")
    ax.axvline(fecha_mayo, color='red', linestyle='--', linewidth=1.5, label="Fin (1-may)")
    
    # Línea de umbral de altura
    ax.axhline(height_thr, color='gray', linestyle='--', alpha=0.4)
    
    # Etiquetas y estética
    ax.legend(loc='upper right')
    ax.set_xlabel("Fecha calendario ajustada")
    ax.set_ylabel("Intensidad normalizada")
    ax.set_title(f"Clasificación (solo 1-feb→1-may): {tipo} ({nivel}, {prob:.2f})")
    plt.xticks(rotation=45)
    st.pyplot(fig)

      # ========= DESCRIPCIÓN COMPLETA =========
    st.subheader("🌾 Descripción completa del patrón detectado")

    if len(peaks):
        fechas_picos = [fechas[p].date() for p in peaks]
        primer_pico, ultimo_pico = fechas_picos[0], fechas_picos[-1]
        duracion = (ultimo_pico - primer_pico).days if len(fechas_picos) > 1 else 0
        picos_pre = [f for f in fechas_picos if f <= fecha_mayo]
        picos_post = [f for f in fechas_picos if f > fecha_mayo]
        resumen_tiempo = (
            f"La curva presenta **{n_picos} picos principales** entre "
            f"**{primer_pico.strftime('%d-%b')}** y **{ultimo_pico.strftime('%d-%b')}**, "
            f"con una duración efectiva aproximada de **{duracion} días**."
        )
        if picos_pre and picos_post:
            resumen_tiempo += " Se observan pulsos tanto **antes como después del 1° de mayo**, indicando continuidad de emergencia."
        elif picos_pre:
            resumen_tiempo += " La emergencia se concentró **antes del 1° de mayo**."
        else:
            resumen_tiempo += " La emergencia principal ocurrió **después del 1° de mayo**."
    else:
        resumen_tiempo = "No se detectaron picos definidos en la curva."

    # Caracterización según tipo
    caracteristicas = {
        "P1": "Emergencia rápida y concentrada, asociada a condiciones tempranas favorables.",
        "P1b": "Emergencia principal temprana con leve repunte posterior.",
        "P2": "Dos pulsos de emergencia separados: bimodal, con reactivación otoñal.",
        "P3": "Emergencia prolongada y sostenida en el tiempo, con múltiples cohortes."
    }[tipo]

    nivel_texto = (
        "La **probabilidad de clasificación es alta**, con buena correspondencia histórica."
        if prob >= 0.75 else
        "La **probabilidad es moderada**, posible sesgo temporal; revisar eje X."
        if prob >= 0.45 else
        "La **probabilidad es baja**, posible ruido o desfase en la curva."
    )

    descripcion_final = (
        f"{resumen_tiempo}\n\n"
        f"El patrón identificado es **{tipo}**: {caracteristicas}\n\n"
        f"{nivel_texto}\n\n"
        "📊 **Interpretación agronómica:**\n"
        "Los patrones P1–P1b requieren control temprano, mientras que P2–P3 "
        "demandan manejo residual o prolongado durante el otoño-invierno."
    )

    st.markdown(descripcion_final)

    # ========= GRÁFICO ILUSTRATIVO DE PATRONES HISTÓRICOS =========
    st.subheader("📚 Ejemplos ilustrativos de patrones históricos")
    
    # Curvas sintéticas (0–140 días ~ feb–jun)
    x = np.linspace(0, 140, 400)
    x_date = pd.date_range(start=date(year_ref, 2, 1), periods=len(x))
    
    # Patrones modelo (formas típicas)
    p1  = np.exp(-0.5*((x-25)/12)**2)
    p1b = np.exp(-0.5*((x-25)/12)**2) + 0.4*np.exp(-0.5*((x-75)/15)**2)
    p2  = np.exp(-0.5*((x-25)/12)**2) + np.exp(-0.5*((x-90)/12)**2)
    p3  = 0.8*np.exp(-0.5*((x-25)/20)**2) + 0.7*np.exp(-0.5*((x-70)/25)**2) + 0.6*np.exp(-0.5*((x-115)/25)**2)
    
    # Normalizar y escalar
    def norm(v): return (v - v.min()) / (v.max() - v.min())
    p1, p1b, p2, p3 = map(norm, [p1, p1b, p2, p3])
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(x_date, p1,  color='royalblue',  lw=2.5, label="P1 — Concentrado temprano")
    ax2.plot(x_date, p1b, color='teal',       lw=2.5, label="P1b — Repunte leve posterior")
    ax2.plot(x_date, p2,  color='orange',     lw=2.5, label="P2 — Bimodal otoñal")
    ax2.plot(x_date, p3,  color='crimson',    lw=2.5, label="P3 — Prolongado/múltiple")
    
    ax2.set_title("Ejemplos de patrones históricos de emergencia (curvas sintéticas)")
    ax2.set_xlabel("Fecha calendario (año de referencia)")
    ax2.set_ylabel("Emergencia relativa (normalizada)")
    ax2.legend(loc='upper right', fontsize=9)
    ax2.axvline(pd.to_datetime(fecha_mayo), color='red', linestyle='--', lw=1, label="1-may")
    ax2.grid(alpha=0.25)
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    
    st.markdown("""
    **Interpretación visual:**
    - 🔵 **P1:** emergencia única y concentrada, temprano (marzo–abril).  
    - 🟢 **P1b:** pico temprano y un leve repunte posterior.  
    - 🟠 **P2:** bimodal, con un segundo pulso otoñal más marcado.  
    - 🔴 **P3:** emergencia prolongada, sin pausa clara, hasta el invierno.  
    """)
