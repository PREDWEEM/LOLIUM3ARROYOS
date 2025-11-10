# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador de patrones con información 1-ene → 1-may (JD 1–121)
# - Procesa imágenes tipo EMERREL (línea negra + ejes)
# - Extrae la curva, restringe a JD≤121 y clasifica P1 / P1b / P2 / P3
# - Incluye controles de recorte y umbrales para robustez entre figuras

import io, os, math, base64
from typing import Dict, Tuple, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
import cv2
from scipy.signal import find_peaks, savgol_filter
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------- Config Streamlit -----------------------
st.set_page_config(page_title="PREDWEEM — Patrones (1-ene→1-may)", layout="wide")
st.title("🌾 Clasificación de patrones (usando solo 1 de enero → 1 de mayo)")

st.markdown("""
Subí las imágenes de las curvas históricas. La app extrae la **línea negra**, analiza **solo JD ≤ 121** y clasifica en **P1, P1b, P2 o P3**.
""")

# ----------------------- Parámetros UI --------------------------
with st.sidebar:
    st.header("Ajustes de extracción")
    st.caption("Usá estos controles si la curva no se detecta correctamente.")
    left_margin   = st.slider("Recorte izquierdo (px)",  0, 250, 60, 1)
    right_margin  = st.slider("Recorte derecho (px)",    0, 250, 40, 1)
    top_margin    = st.slider("Recorte superior (px)",   0, 250, 40, 1)
    bottom_margin = st.slider("Recorte inferior (px)",   0, 250, 70, 1)

    thresh_dark   = st.slider("Umbral de oscuridad (0=negro…255=blanco)", 0, 255, 70, 1)
    canny_low     = st.slider("Canny — low", 0, 200, 30, 1)
    canny_high    = st.slider("Canny — high", 50, 300, 120, 1)

    st.header("Suavizado y picos")
    win = st.slider("Ventana Savitzky-Golay (ímpar)", 3, 51, 9, 2)
    poly = st.slider("Orden polinomio", 1, 5, 2, 1)
    prominence = st.slider("Prominencia de picos", 0.0, 10.0, 0.8, 0.1)
    distance   = st.slider("Distancia mínima entre picos (JD)", 1, 60, 10, 1)
    height_thr = st.slider("Altura mínima relativa (0–1)", 0.0, 1.0, 0.05, 0.01)

    st.header("Eje temporal de la figura")
    x_min_fig = st.number_input("JD mínimo mostrado en la figura", value=0, step=10)
    x_max_fig = st.number_input("JD máximo mostrado en la figura", value=400, step=10)

    st.header("Salida")
    normalize_area = st.checkbox("Normalizar área (AUC=1 en ROI)", value=True)
    show_debug = st.checkbox("Ver depuración por imagen", value=False)

# ----------------------- Constantes -----------------------------
JD_CUTOFF = 121  # 1 de mayo
EPS = 1e-9

@dataclass
class Features:
    year: str
    n_peaks: int
    jd_peaks: List[float]
    jd_main: float
    duration: float
    auc_1_121: float
    share_after_100: float
    pattern: str
    prob: float
    note: str

# ----------------------- Utilidades ----------------------------
def read_image(file) -> np.ndarray:
    data = file.read() if hasattr(file, "read") else file
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def crop_roi(img, left, right, top, bottom):
    h, w = img.shape[:2]
    x1 = np.clip(left, 0, w-1)
    x2 = np.clip(w - right, 1, w)
    y1 = np.clip(top, 0, h-1)
    y2 = np.clip(h - bottom, 1, h)
    return img[y1:y2, x1:x2].copy()

def extract_curve_xy(img_rgb: np.ndarray,
                     thr_dark: int,
                     canny_lo: int, canny_hi: int) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve vectores x_px, y_px (en píxeles dentro del ROI) de la curva.
       Estrategia: detectar bordes oscuros (línea negra), luego para cada columna
       tomar el y más alto donde hay borde (la `cresta` de la curva)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # Enmascarar pixeles claros para enfocarnos en trazos oscuros
    dark_mask = (gray <= thr_dark).astype(np.uint8) * 255
    edges = cv2.Canny(dark_mask, canny_lo, canny_hi)

    h, w = edges.shape
    xs, ys = [], []
    for x in range(w):
        ys_col = np.where(edges[:, x] > 0)[0]
        if ys_col.size > 0:
            # Elegir el valor con mayor vecindad (reduce ruido) → mediana de los 3 más altos
            top_candidates = ys_col[:min(3, ys_col.size)]
            ys.append(int(np.median(top_candidates)))
            xs.append(x)
    if len(xs) < 5:
        return np.array([]), np.array([])
    return np.array(xs, dtype=float), np.array(ys, dtype=float)

def pixels_to_series(xs_px, ys_px, w_px, h_px,
                     x_min_fig, x_max_fig) -> Tuple[np.ndarray, np.ndarray]:
    """Mapea píxeles a eje X (JD) y Y (emergencia relativa), invierte Y para que arriba sea mayor.
       Y queda escalado a [0,1] por max en el ROI."""
    # X en JD:
    x_jd = x_min_fig + (xs_px / max(w_px-1, 1)) * (x_max_fig - x_min_fig)
    # Y: invertir eje vertical, normalizar
    y_val = (h_px - 1 - ys_px).astype(float)
    # Normalización por máximo (para heurísticas de picos relativas)
    if y_val.max() > 0:
        y_val = y_val / (y_val.max() + EPS)
    return x_jd, y_val

def restrict_jd(x_jd, y, jd_max=JD_CUTOFF):
    mask = (x_jd >= 1) & (x_jd <= jd_max)
    return x_jd[mask], y[mask]

def regularize_series(x, y, step=1.0):
    """Re-muestrea la serie a pasos de 1 JD con interpolación lineal."""
    if len(x) < 3:
        return np.array([]), np.array([])
    x_uniform = np.arange(max(1, int(np.floor(x.min()))),
                          int(np.ceil(x.max())) + 1, step, dtype=float)
    y_uniform = np.interp(x_uniform, x, y)
    return x_uniform, y_uniform

def smooth(y, win, poly):
    if len(y) < max(win, poly+2):
        return y
    if win % 2 == 0:
        win = win + 1
    return savgol_filter(y, window_length=win, polyorder=poly, mode="interp")

def area_under_curve(y):
    return float(np.trapz(y))

# ----------------------- Clasificación --------------------------
def classify_by_rules(x, y, prominence=0.8, distance=10, height_thr=0.05) -> Tuple[str, float, Dict]:
    """Clasifica P1 / P1b / P2 / P3 usando SOLO x≤121.
       Devuelve (patrón, probabilidad [0-1], extras)"""
    # Detectar picos en la zona
    peaks, props = find_peaks(y, prominence=prominence, distance=distance, height=height_thr)
    jd_peaks = x[peaks].tolist()
    n_peaks = len(jd_peaks)
    jd_main = float(jd_peaks[np.argmax(props["prominences"])]) if n_peaks else float('nan')

    # Duración efectiva (entre primer y último JD con y > umbral bajo)
    thr = max(0.02, height_thr/2)
    idx_pos = np.where(y > thr)[0]
    duration = float(x[idx_pos[-1]] - x[idx_pos[0]]) if idx_pos.size else 0.0

    # Masa después de JD=100 (clave para discriminar P1 vs P2 al 1-may)
    m100 = np.where(x >= 100)[0]
    share_after_100 = float(y[m100].sum() / (y.sum() + EPS)) if m100.size else 0.0

    # AUC 1–121 (si ya está restringido)
    auc = area_under_curve(y)

    # ---------- Reglas heurísticas (solo info temprana) ----------
    # P1: 1 pico dominante en 60–110, duración corta, baja masa >100
    score_p1 = 0.0
    if n_peaks == 1:
        score_p1 += 0.5
    if 60 <= (jd_main if not math.isnan(jd_main) else 0) <= 110:
        score_p1 += 0.2
    if duration <= 40:
        score_p1 += 0.2
    if share_after_100 <= 0.15:
        score_p1 += 0.1

    # P1b: 2–3 picos tempranos compactos y baja masa >100
    score_p1b = 0.0
    if n_peaks in (2, 3):
        score_p1b += 0.4
    if duration <= 60:
        score_p1b += 0.2
    if share_after_100 <= 0.25:
        score_p1b += 0.2
    # si todos los picos < 115 JD
    if n_peaks >= 2 and max(jd_peaks) < 115:
        score_p1b += 0.2

    # P2: presencia de 2 picos con cola hacia fin de abril o masa >100 elevada
    score_p2 = 0.0
    if n_peaks >= 2:
        score_p2 += 0.3
    if share_after_100 >= 0.20:
        score_p2 += 0.4
    if 100 <= (jd_main if not math.isnan(jd_main) else 0) <= 121 and duration >= 40:
        score_p2 += 0.3

    # P3: muy disperso (duración larga) o ≥3 picos y masa >100
    score_p3 = 0.0
    if n_peaks >= 3:
        score_p3 += 0.4
    if duration >= 80:
        score_p3 += 0.3
    if share_after_100 >= 0.30:
        score_p3 += 0.3

    scores = {"P1": score_p1, "P1b": score_p1b, "P2": score_p2, "P3": score_p3}
    pattern = max(scores, key=scores.get)
    # Probabilidad/certeza: softmax simple de los scores
    vals = np.array(list(scores.values()), dtype=float)
    if vals.max() <= 0:
        prob = 0.25
    else:
        e = np.exp(vals - vals.max())
        prob = float(e[list(scores.keys()).index(pattern)] / (e.sum() + EPS))

    note = (
        f"n_picos={n_peaks}, jd_main={jd_main:.1f} "
        f"duración={duration:.1f}d, masa>100={share_after_100:.2f}, AUC_1_121={auc:.3f}"
    )
    extras = dict(
        n_peaks=n_peaks, jd_peaks=jd_peaks, jd_main=jd_main, duration=duration,
        share_after_100=share_after_100, auc_1_121=auc, scores=scores
    )
    return pattern, prob, extras

def process_image(file, params) -> Tuple[Features, pd.DataFrame, Dict]:
    """Devuelve Features, serie (df) y artefactos para debug."""
    img = read_image(file)
    roi = crop_roi(img, params["left"], params["right"], params["top"], params["bottom"])
    xs_px, ys_px = extract_curve_xy(roi, params["thr"], params["canny_lo"], params["canny_hi"])
    if xs_px.size == 0:
        raise RuntimeError("No se pudo detectar la curva. Ajustá recortes/umbrales en la barra lateral.")

    h, w = roi.shape[:2]
    x_jd, y_rel = pixels_to_series(xs_px, ys_px, w, h, params["x_min"], params["x_max"])
    x_jd, y_rel = restrict_jd(x_jd, y_rel, JD_CUTOFF)
    x_uni, y_uni = regularize_series(x_jd, y_rel, step=1.0)
    if x_uni.size == 0:
        raise RuntimeError("No quedaron datos dentro del rango JD 1–121.")

    y_s = smooth(y_uni, params["win"], params["poly"])
    if params["normalize"]:
        auc = area_under_curve(y_s)
        if auc > 0:
            y_s = y_s / auc

    # Clasificar
    patt, prob, extra = classify_by_rules(
        x_uni, y_s, prominence=params["prom"], distance=params["dist"], height_thr=params["hthr"]
    )

    # Nombre/año heurístico
    name = getattr(file, "name", "imagen")
    year = os.path.splitext(os.path.basename(name))[0]

    feat = Features(
        year=year,
        n_peaks=extra["n_peaks"],
        jd_peaks=extra["jd_peaks"],
        jd_main=extra["jd_main"],
        duration=extra["duration"],
        auc_1_121=extra["auc_1_121"],
        share_after_100=extra["share_after_100"],
        pattern=patt,
        prob=prob,
        note=f"{extra['scores']} | {patt} ({prob:.2f})"
    )

    df = pd.DataFrame({"JD": x_uni, "EMERREL": y_s})
    debug = dict(roi=roi, xs_px=xs_px, ys_px=ys_px, x=x_uni, y=y_s, peaks=extra["jd_peaks"])
    return feat, df, debug

# ----------------------- UI: carga y procesamiento --------------
files = st.file_uploader(
    "Arrastrá o seleccioná una o más imágenes (PNG/JPG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if not files:
    st.info("Subí imágenes para comenzar. (Sugerencia: las figuras 2008–2016, 2023 que mostraste).")
    st.stop()

params = dict(
    left=left_margin, right=right_margin, top=top_margin, bottom=bottom_margin,
    thr=thresh_dark, canny_lo=canny_low, canny_hi=canny_high,
    x_min=x_min_fig, x_max=x_max_fig, win=win, poly=poly,
    prom=prominence, dist=distance, hthr=height_thr, normalize=normalize_area
)

rows = []
series_per_file: Dict[str, pd.DataFrame] = {}
debug_plots = {}

for f in files:
    try:
        feat, df, dbg = process_image(f, params)
        rows.append({
            "archivo": getattr(f, "name", "imagen"),
            "año": feat.year,
            "n_picos": feat.n_peaks,
            "JD_picos": ";".join([f"{p:.0f}" for p in feat.jd_peaks]),
            "JD_pico_max": f"{feat.jd_main:.0f}" if not math.isnan(feat.jd_main) else "",
            "duración_días": round(feat.duration, 1),
            "AUC_1_121": round(feat.auc_1_121, 3),
            "masa_JD>100": round(feat.share_after_100, 3),
            "patrón": feat.pattern,
            "probabilidad": round(feat.prob, 2),
            "nota": feat.note
        })
        series_per_file[feat.year] = df
        debug_plots[feat.year] = dbg
    except Exception as e:
        rows.append({
            "archivo": getattr(f, "name", "imagen"),
            "año": os.path.splitext(getattr(f, "name", "imagen"))[0],
            "n_picos": "", "JD_picos": "", "JD_pico_max": "", "duración_días": "",
            "AUC_1_121": "", "masa_JD>100": "", "patrón": "ERROR", "probabilidad": "",
            "nota": f"ERROR: {type(e).__name__}: {e}"
        })

# ----------------------- Tabla de resultados --------------------
res = pd.DataFrame(rows)
st.subheader("Resultados (solo JD 1–121)")
st.dataframe(res, use_container_width=True)

# Descarga CSV
csv = res.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar resultados (CSV)", csv, file_name="patrones_enero_mayo.csv", mime="text/csv")

# ----------------------- Gráfico comparativo --------------------
st.subheader("Comparación de series (normalizadas en 1–121)")
fig, ax = plt.subplots(figsize=(9, 4))
for year, df in series_per_file.items():
    ax.plot(df["JD"], df["EMERREL"], label=year)
ax.axvline(100, color="gray", linestyle="--", linewidth=1)
ax.axvline(121, color="black", linestyle="--", linewidth=1)
ax.set_xlim(1, JD_CUTOFF)
ax.set_xlabel("Día juliano (JD)")
ax.set_ylabel("Emergencia relativa (normalizada)")
ax.legend(ncol=6, fontsize=8)
st.pyplot(fig, clear_figure=True)

# ----------------------- Depuración por imagen ------------------
if show_debug:
    st.subheader("Depuración por imagen (ROI, picos, etc.)")
    for year, dbg in debug_plots.items():
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"**{year} — ROI**")
            st.image(cv2.cvtColor(dbg["roi"], cv2.COLOR_BGR2RGB), clamp=True)
        with col2:
            st.markdown(f"**{year} — Serie 1–121 con picos**")
            x, y = dbg["x"], dbg["y"]
            peaks = dbg["peaks"]
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.plot(x, y, lw=1.5)
            for p in peaks:
                ax2.axvline(p, color="red", linestyle="--", alpha=0.7)
            ax2.set_xlim(1, JD_CUTOFF)
            ax2.set_xlabel("JD")
            ax2.set_ylabel("Emergencia rel.")
            st.pyplot(fig2, clear_figure=True)

# ----------------------- Leyenda de patrones --------------------
st.markdown("""
### Criterios (reglas heurísticas con info hasta 1-may)
- **P1**: 1 pico dominante (JD 60–110), **duración < 40 d**, masa JD>100 **≤ 0.15**.  
- **P1b**: 2–3 picos **tempranos** y compactos, **duración ≤ 60 d**, masa JD>100 **≤ 0.25**.  
- **P2**: ≥2 picos y **cola hacia fin de abril**, masa JD>100 **≥ 0.20** o duración **≥ 40 d**.  
- **P3**: patrón **muy extendido** (duración ≥ 80 d) o **≥3 picos** con masa JD>100 **≥ 0.30**.  

> La **probabilidad** reportada es una softmax de los puntajes internos; sirve como **certeza relativa** de la clasificación con la información disponible hasta el 1-de-mayo.
""")
