# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Calibración por CLICS sobre el gráfico (Plotly)
# Clasificación: CONCENTRADO (≥50% AUC antes JD121) vs EXTENDIDO
# Guarda calibración por imagen (CSV)

import os, cv2, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# =================== CONFIG ===================
st.set_page_config(page_title="PREDWEEM — Calibración por clics", layout="wide")
st.title("🌾 Clasificación por AUC con **calibración por clics sobre el gráfico**")

st.markdown("""
1) Hacé **dos clics** en el gráfico: el primero marca el **mínimo** y el segundo el **máximo** del eje X (días julianos).  
2) La app guarda la calibración por imagen y clasifica según **AUC ≥ 50%** antes de **JD 121**.  
3) Podés limpiar los clics y volver a marcar.  
""")

CALIB_FILE = "calibracion_patrones_clicks.csv"   # persistencia por imagen
JD_CUTOFF = 121
YEAR_JD_MAX = 365
EPS = 1e-9

# =================== UTILS ===================
def read_image(file):
    data = file.read() if hasattr(file, "read") else file
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

def extract_curve(img_bgr, thr_dark, c_lo, c_hi):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray <= thr_dark).astype(np.uint8) * 255
    edges = cv2.Canny(mask, c_lo, c_hi)
    h, w = edges.shape
    xs, ys = [], []
    for x in range(w):
        y_col = np.where(edges[:, x] > 0)[0]
        if y_col.size:
            ys.append(int(np.median(y_col[:3])))
            xs.append(x)
    return np.array(xs), np.array(ys), h, w

def to_series_from_clicks(xs_px, ys_px, h, w, x_min_click_px, x_max_click_px):
    """
    Mapear píxeles -> JD usando los dos clics como extremos visibles:
    - Asumimos eje anual 0..365
    - jd(x) = (x / (w-1)) * 365
    - El usuario define el sub-rango visible marcando x_min_click_px y x_max_click_px
    """
    jd_all = (xs_px / max(1, (w - 1))) * YEAR_JD_MAX   # 0..365
    # Escalar Y (invertido y normalizado en ROI)
    y = (h - 1 - ys_px).astype(float)
    if y.max() > 0:
        y /= y.max()
    # Filtrar al rango visible marcado por clics
    jd_min = (x_min_click_px / max(1, (w - 1))) * YEAR_JD_MAX
    jd_max = (x_max_click_px / max(1, (w - 1))) * YEAR_JD_MAX
    mask = (jd_all >= jd_min) & (jd_all <= jd_max)
    return jd_all[mask], y[mask], float(jd_min), float(jd_max)

def regularize_1_121(x_jd, y):
    if x_jd.size < 3:
        return np.array([]), np.array([])
    xg = np.arange(1, JD_CUTOFF + 1, 1.0)
    yg = np.interp(xg, x_jd, y, left=0.0, right=0.0)
    return xg, yg

def smooth(y, win, poly):
    if len(y) < win: 
        return y
    if win % 2 == 0: 
        win += 1
    return savgol_filter(y, win, poly, mode="interp")

def auc(y):
    return float(np.trapz(y))

def classify_auc50(x, y):
    total = auc(y)
    share = auc(y[x <= JD_CUTOFF]) / (total + EPS)
    patt = "CONCENTRADO" if share >= 0.50 else "EXTENDIDO"
    col = "green" if patt == "CONCENTRADO" else "orange"
    prob = round(abs(share - 0.50) * 1.5 + 0.5, 2)
    return patt, prob, dict(share=share, total=total, col=col)

def load_calib():
    if os.path.exists(CALIB_FILE):
        try:
            df = pd.read_csv(CALIB_FILE)
            return df
        except Exception:
            return pd.DataFrame(columns=["imagen", "x_min_px", "x_max_px"])
    return pd.DataFrame(columns=["imagen", "x_min_px", "x_max_px"])

def save_calib(df):
    df.to_csv(CALIB_FILE, index=False)

# =================== SIDEBAR ===================
with st.sidebar:
    st.header("Parámetros de extracción y suavizado")
    thr_dark   = st.slider("Umbral de oscuridad", 0, 255, 70)
    canny_low  = st.slider("Canny low", 0, 200, 30)
    canny_high = st.slider("Canny high", 50, 300, 120)
    win        = st.slider("Ventana Savitzky-Golay", 3, 51, 9, step=2)
    poly       = st.slider("Orden polinomio", 1, 5, 2)
    normalize_area = st.checkbox("Normalizar AUC en 1 (1–121)", True)
    auto_save  = st.checkbox("Guardar calibración automáticamente", True)

# =================== FILES ===================
files = st.file_uploader("📤 Subí imágenes (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
if not files:
    st.stop()

df_calib = load_calib()
rows, series = [], {}

# =================== LOOP POR IMAGEN ===================
for f in files:
    st.markdown("---")
    st.subheader(f"🖼️ {f.name}")

    try:
        img_bgr = read_image(f)
        xs_px, ys_px, h, w = extract_curve(img_bgr, thr_dark, canny_low, canny_high)
        if xs_px.size == 0:
            st.error("⚠️ No se detectó la curva. Ajustá umbral o Canny.")
            continue

        # Figura interactiva con Plotly (imagen + curva amarilla)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        fig = px.imshow(img_rgb, binary_string=True)
        fig.update_xaxes(showgrid=False, showticklabels=False, range=[0, w])
        fig.update_yaxes(showgrid=False, showticklabels=False, range=[h, 0])
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=320, title="Hacé 2 clics: JD mínimo y JD máximo")

        # overlay de la curva (en coordenadas de píxel)
        fig.add_trace(go.Scatter(
            x=xs_px, y=ys_px,
            mode="lines", line=dict(width=2, color="yellow"),
            name="Curva detectada"
        ))

        # Cargar calibración previa (si existe)
        prev = df_calib[df_calib["imagen"] == f.name]
        preset_lines = []
        if not prev.empty:
            x_min_prev = float(prev["x_min_px"].iloc[0])
            x_max_prev = float(prev["x_max_px"].iloc[0])
            preset_lines = [x_min_prev, x_max_prev]
            fig.add_vline(x=x_min_prev, line=dict(color="red", dash="dash"), annotation_text=f"prev min px={int(x_min_prev)}")
            fig.add_vline(x=x_max_prev, line=dict(color="red", dash="dash"), annotation_text=f"prev max px={int(x_max_prev)}")

        # Capturar clics (lista de puntos)
        st.caption("🔎 Indicá con **dos clics** los extremos del eje X real (min → max).")
        clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=330, override_width=None)
        # clicks es lista de dicts [{"x": float_px, "y": float_px, ...}, ...]

        # Controles para manejar la selección
        cols = st.columns([1,1,1])
        with cols[0]:
            reset = st.button(f"🧹 Limpiar clics ({f.name})")
        with cols[1]:
            use_prev = st.button(f"↩️ Usar calibración previa ({f.name})") if not prev.empty else st.empty()
        with cols[2]:
            save_btn = st.button(f"💾 Guardar calibración ({f.name})")

        # Armar x_min_px / x_max_px desde clics o preset
        x_min_px, x_max_px = None, None
        if reset:
            st.session_state[f"clicks_{f.name}"] = []

        # Mantener clicks en session_state
        key = f"clicks_{f.name}"
        if key not in st.session_state:
            st.session_state[key] = []

        # Agregar clics nuevos (máx 2)
        if clicks:
            for c in clicks:
                if "x" in c:
                    st.session_state[key].append(float(c["x"]))
            st.session_state[key] = st.session_state[key][:2]

        # Usar calibración previa
        if use_prev and not prev.empty:
            st.session_state[key] = preset_lines[:2]

        # Mostrar estado de selección
        sel = st.session_state[key]
        if len(sel) == 2:
            x_min_px, x_max_px = sorted(sel)
            st.success(f"Selección: x_min_px={int(x_min_px)} | x_max_px={int(x_max_px)}")
        elif len(sel) == 1:
            st.info(f"Primer clic en x={int(sel[0])}. Falta el segundo clic.")
        else:
            st.warning("Esperando dos clics… (o usá la calibración previa si existe).")

        # Guardar calibración si corresponde
        if (save_btn or auto_save) and x_min_px is not None and x_max_px is not None:
            df_calib = df_calib[df_calib["imagen"] != f.name]
            df_calib.loc[len(df_calib)] = [f.name, x_min_px, x_max_px]
            save_calib(df_calib)
            if save_btn:
                st.success("✅ Calibración guardada.")

        # Si no hay dos clics aún, no seguimos
        if x_min_px is None or x_max_px is None:
            continue

        # ====== CONVERSIÓN A JD, RESTRICCIÓN 1–121, SUAVIZADO ======
        x_jd, y_raw, jd_min, jd_max = to_series_from_clicks(xs_px, ys_px, h, w, x_min_px, x_max_px)
        # recorte 1..121 y remuestreo entero
        mask_roi = (x_jd >= 1) & (x_jd <= JD_CUTOFF)
        x_roi, y_roi = x_jd[mask_roi], y_raw[mask_roi]
        x_uni, y_uni = regularize_1_121(x_roi, y_roi)
        if x_uni.size == 0:
            st.error("No quedaron datos en 1–121 con la calibración seleccionada.")
            continue
        y_s = smooth(y_uni, win, poly)
        if normalize_area and auc(y_s) > 0:
            y_s = y_s / auc(y_s)

        # ====== CLASIFICACIÓN ======
        patt, prob, info = classify_auc50(x_uni, y_s)
        year = os.path.splitext(f.name)[0]
        rows.append({
            "año": year,
            "JD_min_sel": round(jd_min, 1),
            "JD_max_sel": round(jd_max, 1),
            "AUC_total": round(info["total"], 3),
            "%_área ≤121": round(info["share"] * 100, 1),
            "patrón": patt,
            "probabilidad": prob
        })
        series[year] = (x_uni, y_s, info["col"])

        st.success(f"**{year}** → {patt} ({prob:.2f}) — {info['share']*100:.1f}% del área antes de JD 121")

    except Exception as e:
        st.error(f"Error procesando {f.name}: {e}")

# =================== TABLA + GRÁFICO ===================
if rows:
    df_out = pd.DataFrame(rows)
    st.subheader("📊 Resultados (AUC ≥50% antes JD121)")
    st.dataframe(df_out, use_container_width=True)
    st.download_button("⬇️ Descargar CSV", df_out.to_csv(index=False).encode("utf-8"),
                       file_name="patrones_auc50_clicks.csv")

    # gráfico comparativo 1–121
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    for y, (xx, yy, col) in series.items():
        ax2.plot(xx, yy, label=y, color=col)
    ax2.axvline(121, color="black", ls="--", lw=1)
    ax2.set_xlim(1, JD_CUTOFF)
    ax2.set_xlabel("Día juliano (JD)")
    ax2.set_ylabel("Emergencia relativa (1–121)")
    ax2.legend(ncol=6, fontsize=8)
    st.pyplot(fig2, clear_figure=True)

st.markdown("""
### Criterio de clasificación
- **🟢 CONCENTRADO:** AUC(≤ JD121) ≥ 50%  
- **🟠 EXTENDIDO:** AUC(≤ JD121) < 50%  

🖱️ *Usá dos clics sobre el gráfico para fijar el rango visible de X (min→max).  
La calibración se guarda en* `calibracion_patrones_clicks.csv`.
""")
