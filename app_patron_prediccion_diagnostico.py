# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador robusto (final) con ventana ≤1 mayo o completa
import streamlit as st
import cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from datetime import datetime
from pathlib import Path

# ========== CONFIGURACIÓN STREAMLIT ==========
st.set_page_config(page_title="Clasificador PREDWEEM — Versión final", layout="wide")
st.title("🌾 Clasificador PREDWEEM — Versión final con detección adaptativa")

st.markdown("""
Analiza gráficos de **emergencia relativa (curva negra)** con eje X en **días julianos (0–300)**.  
Detecta el patrón (P1, P1b, P2, P3) según forma y separación temporal.  
Permite elegir analizar hasta **1° mayo (JD 121)** o el **patrón completo (0–300)**.  
Genera automáticamente un **CSV resumen** con los resultados.
""")

# ========= SIDEBAR =========
st.sidebar.header("⚙️ Parámetros de análisis")
height_thr = st.sidebar.slider("Umbral mínimo de altura", 0.05, 0.5, 0.22, 0.01)
dist_min = st.sidebar.slider("Distancia mínima entre picos", 10, 80, 35, 5)
window_smooth = st.sidebar.slider("Ventana de suavizado", 5, 51, 15, 2)
poly_order = st.sidebar.slider("Orden del filtro", 1, 3, 2, 1)

st.sidebar.subheader("🕒 Ventana de análisis")
modo_ventana = st.sidebar.radio(
    "Rango temporal:",
    ["≤ 1-mayo (JD 121)", "Patrón completo (0–300)"],
    index=0
)

# ========= ARCHIVO SUBIDO =========
uploaded = st.file_uploader("📤 Cargar imagen (.png o .jpg)", type=["png", "jpg"])
OUT_DIR = Path("resultados_clasif"); OUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUT_DIR / "resumen_clasificaciones.csv"

# ========= CLASIFICADOR PRINCIPAL =========
def clasificar(curva, x_axis, thr=0.22, min_dist_px=35,
               prom_thr=0.12, merge_window_days=18):
    """
    Clasifica patrón de emergencia con detección global + tardía adaptativa
    """
    # --- Global ---
    peaks_g, props_g = find_peaks(curva, height=thr, distance=min_dist_px, prominence=prom_thr)
    h_g = props_g.get("peak_heights", np.array([]))

    # --- Tardía (100–121) ---
    late = (x_axis >= 100) & (x_axis <= 121)
    peaks_l = np.array([])
    if np.any(late):
        ylate = curva[late]
        if np.nanmax(ylate) > 0:
            thr_l = max(0.10, 0.35 * float(np.nanmax(ylate)))
            prom_l = max(0.05, 0.60 * prom_thr)
            dist_l = max(12, int(min_dist_px * 0.6))
            idx_off = np.where(late)[0][0]
            peaks_l0, _ = find_peaks(ylate, height=thr_l, distance=dist_l, prominence=prom_l)
            peaks_l = peaks_l0 + idx_off

    # --- Unión ---
    peaks = np.unique(np.concatenate([peaks_g, peaks_l]).astype(int))
    if peaks.size == 0:
        return "-", 0.0, np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0

    heights = curva[peaks]

    # --- Fusión cercana ---
    px_per_day = len(curva) / (x_axis[-1] - x_axis[0] + 1e-6)
    merge_px = int(max(3, merge_window_days * px_per_day))
    keep, i = [], 0
    while i < len(peaks):
        j = i + 1; group = [i]
        while j < len(peaks) and (peaks[j] - peaks[j-1]) <= merge_px:
            group.append(j); j += 1
        best = group[np.argmax(heights[group])]; keep.append(best); i = j
    peaks = peaks[keep]; heights = heights[keep]

    # --- Filtro JD < 30 ---
    mask_init = x_axis[peaks] > 30
    peaks = peaks[mask_init]; heights = heights[mask_init]
    n = len(peaks)
    if n == 0:
        return "-", 0.0, np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0

    # --- Métricas ---
    jd = x_axis[peaks]
    sep_days = np.diff(jd) if n > 1 else np.array([0.0])
    mean_sep = float(np.mean(sep_days)) if n > 1 else 0.0
    std_sep  = float(np.std (sep_days)) if n > 2 else 0.0
    hmax  = float(np.max(heights)); hmean = float(np.mean(heights))
    ratio_minor = float(np.min(heights) / (hmax + 1e-6)) if n > 1 else 0.0

    def area_local(idx, win_days=8):
        half = int(max(2, win_days * px_per_day))
        a = max(0, peaks[idx] - half); b = min(len(curva)-1, peaks[idx] + half)
        return float(np.trapz(curva[a:b+1]))
    areas = np.array([area_local(i) for i in range(n)])
    area_ratio_minor = float(np.min(areas) / (np.max(areas) + 1e-6)) if n > 1 else 0.0

    # --- Reglas ---
    if n == 1:
        tipo = "P1"
    elif n == 2:
        if (mean_sep >= 70) and (max(ratio_minor, area_ratio_minor) >= 0.40):
            tipo = "P2"
        else:
            tipo = "P1b"
    else:
        tipo = "P3"

    conf = ((hmax - 0.4 * hmean) / (hmax + 1e-6)) * np.exp(-0.004 * std_sep)
    prob = float(np.clip(conf, 0.0, 1.0))
    return tipo, prob, peaks, heights, mean_sep, std_sep, hmax, hmean


# ========= PROCESAMIENTO DE IMAGEN =========
if uploaded:
    fname = uploaded.name
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(uploaded, caption=f"📈 Imagen original: {fname}", use_container_width=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    crop_left, crop_top, crop_bottom = int(w*0.08), int(h*0.10), int(h*0.05)
    gray = gray[crop_top:h-crop_bottom, crop_left:w]; h, w = gray.shape
    gray_inv = 255 - gray
    gray_norm = cv2.GaussianBlur(gray_inv, (3,3), 0).astype(float)/255.0

    # Seguimiento robusto
    curve_y = []
    for i in range(w):
        col = gray_norm[:, i]
        if np.count_nonzero(col > 0.25) < h * 0.05:
            curve_y.append(np.nan)
            continue
        y_pos = np.argmax(col)
        curve_y.append(h - y_pos)
    curve_y = np.array(curve_y)
    curve_y = pd.Series(curve_y).interpolate(limit_direction="both").to_numpy()
    y_norm = (curve_y - np.nanmin(curve_y)) / (np.nanmax(curve_y) - np.nanmin(curve_y) + 1e-6)

    x_julian = np.linspace(0, 300, len(y_norm))
    corte = 121 if modo_ventana.startswith("≤ 1-mayo") else 300
    mask_corte = x_julian <= corte
    x_sub, y_sub = x_julian[mask_corte], y_norm[mask_corte]

    # Suavizado seguro
    win = max(7, min(window_smooth, len(y_sub) - (1 - len(y_sub) % 2)))
    if win % 2 == 0: win += 1
    y_smooth = savgol_filter(y_sub, win, poly_order)

    tipo, prob, peaks, heights, mean_sep, std_sep, hmax, hmean = clasificar(
        y_smooth, x_sub, thr=height_thr, min_dist_px=dist_min,
        prom_thr=0.12, merge_window_days=18
    )
    nivel = "🔵 Alta" if prob > 0.75 else "🟠 Media" if prob > 0.45 else "🔴 Baja"

    # ========= GRAFICAR =========
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_julian, y_norm, color="lightgray", lw=1.2, label="Curva completa (0–300)")
    ax.plot(x_sub, y_smooth, color="royalblue", lw=2, label="Tramo analizado")
    if len(peaks):
        ax.plot(x_sub[peaks], y_smooth[peaks], "ro", label="Picos detectados")
        for p in peaks:
            ax.text(x_sub[p], min(1.02, y_smooth[p] + 0.03), f"{x_sub[p]:.0f}", fontsize=8, rotation=45, ha="center")
    ax.axvline(121, color="red", linestyle="--", lw=1.2, label="1° mayo (JD 121)")
    ax.set_title(f"Clasificación: {tipo} ({nivel}, prob={prob:.2f})")
    ax.set_xlabel("Día juliano"); ax.set_ylabel("Emergencia relativa (normalizada)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    st.pyplot(fig)

    # ========= EXPORTAR RESULTADO =========
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    row = pd.DataFrame([[fname, tipo, prob, modo_ventana, ts]],
                       columns=["imagen", "tipo", "probabilidad", "ventana", "fecha"])
    if CSV_PATH.exists():
        df_old = pd.read_csv(CSV_PATH)
        df_new = pd.concat([df_old, row], ignore_index=True)
    else:
        df_new = row
    df_new.to_csv(CSV_PATH, index=False)
    st.success(f"✅ Clasificación guardada en {CSV_PATH.name}")

else:
    st.info("📂 Cargá una imagen con eje X en días julianos (0–300).")

