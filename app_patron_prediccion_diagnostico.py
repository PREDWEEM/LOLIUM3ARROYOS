# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador robusto con separación temporal y prominence
import streamlit as st
import cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

# ========= CONFIGURACIÓN STREAMLIT =========
st.set_page_config(page_title="Clasificador PREDWEEM — Preciso (1° mayo o completo)", layout="wide")
st.title("🌾 Clasificador PREDWEEM — Precisión fotométrica (curva negra, eje en días julianos)")

st.markdown("""
Analiza gráficos de **emergencia relativa (curva negra)** con eje X en **días julianos (0–300)**.  
Detecta la forma exacta de la curva y clasifica el patrón (P1, P1b, P2, P3).  
Podés elegir entre analizar solo hasta **1° mayo (JD 121)** o el **patrón completo (0–300)**.
""")

# ========= SIDEBAR =========
st.sidebar.header("⚙️ Parámetros de análisis")
height_thr = st.sidebar.slider("Umbral mínimo de altura", 0.05, 0.5, 0.22, 0.01)
dist_min = st.sidebar.slider("Distancia mínima entre picos", 10, 80, 35, 5)
window_smooth = st.sidebar.slider("Ventana de suavizado (px)", 5, 51, 15, 2)
poly_order = st.sidebar.slider("Orden del filtro", 1, 3, 2, 1)

st.sidebar.subheader("🕒 Ventana de análisis")
modo_ventana = st.sidebar.radio(
    "Rango temporal para clasificar:",
    ["≤ 1-mayo (JD 121)", "Patrón completo (0-300)"],
    index=0
)

uploaded = st.file_uploader("📤 Cargar imagen (.png o .jpg)", type=["png", "jpg"])

# ========= PROCESAMIENTO =========
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(uploaded, caption="📈 Imagen original", use_container_width=True)

    # --- Conversión a gris y recorte de ejes ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    crop_left = int(w * 0.08); crop_top = int(h * 0.10); crop_bottom = int(h * 0.05)
    gray = gray[crop_top:h - crop_bottom, crop_left:w]
    h, w = gray.shape

    # --- Invertimos para seguir el trazo negro ---
    gray_inv = 255 - gray
    gray_norm = cv2.GaussianBlur(gray_inv, (3, 3), 0)
    gray_norm = gray_norm.astype(float) / 255.0

    # --- Seguimiento robusto de la línea ---
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

    # --- Construcción del eje juliano ---
    x_julian = np.linspace(0, 300, len(y_norm))
    corte = 121 if modo_ventana.startswith("≤ 1-mayo") else 300
    mask_corte = x_julian <= corte
    x_sub, y_sub = x_julian[mask_corte], y_norm[mask_corte]
    y_smooth = savgol_filter(y_sub, window_smooth, poly_order)

    # ========= CLASIFICACIÓN =========
    def clasificar(curva, x_axis, thr=0.22, min_dist_px=35,
                   prom_thr=0.12, merge_window_days=18):
        peaks, props = find_peaks(curva, height=thr, distance=min_dist_px, prominence=prom_thr)
        heights = props.get("peak_heights", np.array([]))
        promin  = props.get("prominences",  np.array([]))

        # Fusión de picos muy cercanos
        if len(peaks) >= 2:
            px_per_day = len(curva) / (x_axis[-1] - x_axis[0] + 1e-6)
            merge_px = int(max(3, merge_window_days * px_per_day))
            keep = []; i = 0
            while i < len(peaks):
                j = i + 1; group = [i]
                while j < len(peaks) and (peaks[j] - peaks[j-1]) <= merge_px:
                    group.append(j); j += 1
                best = group[np.argmax(heights[group])]; keep.append(best); i = j
            keep = np.array(sorted(list(set(keep))))
            peaks, heights, promin = peaks[keep], heights[keep], promin[keep]

        # Filtro: eliminar picos falsos < JD 30
        peaks_valid = []
        for p in peaks:
            if x_axis[p] > 30:
                peaks_valid.append(p)
        peaks = np.array(peaks_valid)
        heights = heights[:len(peaks)]

        n = len(peaks)
        if n == 0:
            return "-", 0.0, np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0

        jd = x_axis[peaks]
        sep_days = np.diff(jd) if n > 1 else np.array([0.0])
        mean_sep = float(np.mean(sep_days)) if n > 1 else 0.0
        std_sep  = float(np.std (sep_days)) if n > 2 else 0.0
        hmax  = float(np.max(heights))
        hmean = float(np.mean(heights))
        ratio_minor = float(np.min(heights) / (hmax + 1e-6)) if n > 1 else 0.0

        # Áreas locales para comparar magnitudes
        def area_local(idx, win_days=8):
            pxpd = len(curva) / (x_axis[-1] - x_axis[0] + 1e-6)
            half = int(max(2, win_days * pxpd))
            a = max(0, peaks[idx] - half); b = min(len(curva)-1, peaks[idx] + half)
            return float(np.trapz(curva[a:b+1]))
        areas = np.array([area_local(i) for i in range(n)])
        area_ratio_minor = float(np.min(areas) / (np.max(areas) + 1e-6)) if n > 1 else 0.0

        # Reglas P1 / P1b / P2 / P3
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

    tipo, prob, peaks, heights, mean_sep, std_sep, hmax, hmean = clasificar(
        y_smooth, x_sub, thr=height_thr, min_dist_px=dist_min,
        prom_thr=0.12, merge_window_days=18
    )
    nivel = "🔵 Alta" if prob > 0.75 else "🟠 Media" if prob > 0.45 else "🔴 Baja"

    # ========= VISUALIZACIÓN =========
    st.subheader(f"📊 Curva reconstruida ({modo_ventana})")
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

    # ========= DESCRIPCIÓN =========
    st.subheader("🌾 Descripción agronómica")
    st.markdown(f"""
    **Tipo detectado:** {tipo}  
    **Probabilidad:** {prob:.2f} ({nivel})  
    **Separación media:** {mean_sep:.1f} días  
    **Relación de magnitud (2º/1º pico):** {max(heights[-1]/(heights[0]+1e-6),0):.2f}

    **Interpretación:**
    - **P1:** Emergencia única, compacta y temprana.  
    - **P1b:** Pico principal temprano + repunte leve posterior.  
    - **P2:** Dos cohortes separadas (bimodal) con magnitudes comparables.  
    - **P3:** Emergencia prolongada y continua.  

    🔎 *La clasificación depende de la ventana seleccionada. Si el segundo pico ocurre después de 1° mayo, usar “Patrón completo”.*
    """)

    # ========= Seguimiento de línea =========
    st.subheader("🔍 Seguimiento del trazo detectado")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.imshow(gray, cmap="gray"); ax2.plot(np.arange(len(y_norm)), h - y_norm * h, color="red", lw=1)
    ax2.set_title("Área útil del gráfico (ejes recortados y trazo seguido)")
    st.pyplot(fig2)

else:
    st.info("📂 Cargá una imagen con eje X en días julianos (0–300). Podés elegir analizar ≤ 1 mayo o todo el patrón.")


