# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Calibración visual con clics + JD reales
# Clasificación CONCENTRADO / EXTENDIDO (AUC ≥ 50 % antes JD121)
# Zoom + desplazamiento + líneas guía

import streamlit as st

# ====== CHEQUEO DE DEPENDENCIAS ======
try:
    from streamlit_plotly_events import plotly_events
except ImportError:
    st.warning("""
    ⚠️ Falta la librería **streamlit-plotly-events**.
    Instalala ejecutando:

    ```bash
    pip install streamlit-plotly-events
    ```
    Si usás **Streamlit Cloud**, agregala en tu archivo `requirements.txt`.
    """)
    st.stop()

import os, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import savgol_filter

# ====== CONFIGURACIÓN ======
st.set_page_config(page_title="PREDWEEM — Calibración JD real", layout="wide")
st.title("🌾 Clasificación de patrones — Calibración visual con JD reales")

st.markdown("""
🧭 **Modo de uso:**
1. Hacé **2 clics** sobre el gráfico (inicio y fin del eje X visible).  
2. Luego ingresá los **valores reales de día juliano (JD)** correspondientes.  
3. Verás **líneas rojas** en el gráfico indicando tus puntos seleccionados.  
4. La app guardará la calibración en `calibracion_patrones_clicks.csv`.  
5. Clasifica automáticamente según **AUC ≥ 50 % antes JD 121**.
""")

CALIB_FILE = "calibracion_patrones_clicks.csv"
JD_CUTOFF = 121
EPS = 1e-9

# ====== FUNCIONES ======
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

def map_to_jd(xs_px, ys_px, h, x_min_px, x_max_px, jd_min, jd_max):
    """Mapea píxeles → JD según puntos y valores reales."""
    y = (h - 1 - ys_px).astype(float)
    if y.max() > 0: y /= y.max()
    jd = jd_min + (xs_px - x_min_px) * (jd_max - jd_min) / max(1, (x_max_px - x_min_px))
    mask = (jd >= jd_min) & (jd <= jd_max)
    return jd[mask], y[mask]

def regularize(x, y):
    if x.size < 3: return np.array([]), np.array([])
    xg = np.arange(1, JD_CUTOFF + 1, 1.0)
    yg = np.interp(xg, x, y, left=0.0, right=0.0)
    return xg, yg

def smooth(y, win, poly):
    if len(y) < win: return y
    if win % 2 == 0: win += 1
    return savgol_filter(y, win, poly, mode="interp")

def auc(y): return float(np.trapz(y))

def classify_auc50(x, y):
    total = auc(y)
    share = auc(y[x <= JD_CUTOFF]) / (total + EPS)
    patt = "CONCENTRADO" if share >= 0.50 else "EXTENDIDO"
    col = "green" if patt == "CONCENTRADO" else "orange"
    prob = round(abs(share - 0.50) * 1.5 + 0.5, 2)
    return patt, prob, dict(share=share, total=total, col=col)

def load_calib():
    if os.path.exists(CALIB_FILE):
        try: return pd.read_csv(CALIB_FILE)
        except: return pd.DataFrame(columns=["imagen", "x_min_px", "x_max_px", "JD_min", "JD_max"])
    return pd.DataFrame(columns=["imagen", "x_min_px", "x_max_px", "JD_min", "JD_max"])

def save_calib(df): df.to_csv(CALIB_FILE, index=False)

# ====== SIDEBAR ======
with st.sidebar:
    st.header("🎛️ Parámetros")
    thr_dark = st.slider("Umbral de oscuridad", 0, 255, 70)
    canny_low = st.slider("Canny low", 0, 200, 30)
    canny_high = st.slider("Canny high", 50, 300, 120)
    win = st.slider("Ventana Savitzky-Golay", 3, 51, 9, step=2)
    poly = st.slider("Orden polinomio", 1, 5, 2)
    normalize_area = st.checkbox("Normalizar área total (AUC=1)", True)
    auto_save = st.checkbox("Guardar calibración automáticamente", True)

# ====== ARCHIVOS ======
files = st.file_uploader("📤 Subí imágenes (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
if not files: st.stop()
df_calib = load_calib()
rows, series = [], {}

# ====== LOOP ======
for f in files:
    st.markdown("---")
    st.subheader(f"🖼️ {f.name}")

    img_bgr = read_image(f)
    xs, ys, h, w = extract_curve(img_bgr, thr_dark, canny_low, canny_high)
    if xs.size == 0:
        st.error("⚠️ No se detectó la curva. Ajustá los filtros.")
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ====== GRÁFICO CON ZOOM ======
    fig = px.imshow(img_rgb)
    fig.update_xaxes(showticklabels=False, range=[0, w])
    fig.update_yaxes(showticklabels=False, range=[h, 0])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                             line=dict(color="yellow", width=2), name="Curva detectada"))
    fig.update_layout(
        title=dict(
            text="🖱️ Hacé 2 clics (inicio y fin eje X visible). Podés hacer zoom antes.",
            x=0.02, xanchor="left"),
        height=750,
        width=None,
        margin=dict(l=0, r=0, t=50, b=0),
        dragmode="zoom",
        hovermode=False,
        xaxis=dict(fixedrange=False),
        yaxis=dict(fixedrange=False)
    )

    clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=760)

    key = f"clicks_{f.name}"
    if key not in st.session_state: st.session_state[key] = []
    if clicks:
        for c in clicks:
            if "x" in c:
                st.session_state[key].append(float(c["x"]))
        st.session_state[key] = st.session_state[key][:2]

    sel = st.session_state[key]
    if len(sel) == 2:
        x_min_px, x_max_px = sorted(sel)
        st.success(f"📍 Píxeles seleccionados: {int(x_min_px)} → {int(x_max_px)}")
    else:
        st.info("👉 Hacé 2 clics sobre el gráfico (mín y máx del eje X).")
        continue

    # ====== VALORES JD REALES ======
    cols = st.columns(2)
    with cols[0]:
        jd_min = st.number_input(f"Valor JD mínimo real ({f.name})", min_value=0.0, max_value=400.0, value=0.0, step=1.0)
    with cols[1]:
        jd_max = st.number_input(f"Valor JD máximo real ({f.name})", min_value=jd_min+1, max_value=400.0, value=365.0, step=1.0)

    # Mostrar líneas guía en la imagen original
    fig_lines = px.imshow(img_rgb)
    fig_lines.update_xaxes(showticklabels=False, range=[0, w])
    fig_lines.update_yaxes(showticklabels=False, range=[h, 0])
    fig_lines.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="yellow", width=2)))
    fig_lines.add_vline(x=x_min_px, line=dict(color="red", dash="dash"), annotation_text=f"JD {jd_min}")
    fig_lines.add_vline(x=x_max_px, line=dict(color="red", dash="dash"), annotation_text=f"JD {jd_max}")
    fig_lines.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_lines, use_container_width=True)

    if auto_save:
        df_calib = df_calib[df_calib["imagen"] != f.name]
        df_calib.loc[len(df_calib)] = [f.name, x_min_px, x_max_px, jd_min, jd_max]
        save_calib(df_calib)

    # ====== CLASIFICACIÓN ======
    x_jd, y_raw = map_to_jd(xs, ys, h, x_min_px, x_max_px, jd_min, jd_max)
    xg, yg = regularize(x_jd, y_raw)
    yg = smooth(yg, win, poly)
    if normalize_area and auc(yg) > 0: yg /= auc(yg)
    patt, prob, info = classify_auc50(xg, yg)
    year = os.path.splitext(f.name)[0]
    rows.append({
        "año": year, "JD_min": round(jd_min, 1), "JD_max": round(jd_max, 1),
        "%_área ≤121": round(info["share"] * 100, 1),
        "patrón": patt, "probabilidad": prob
    })
    series[year] = (xg, yg, info["col"])
    st.success(f"**{year}** → {patt} ({prob:.2f}) — {info['share']*100:.1f}% del área antes de JD121")

# ====== RESULTADOS ======
if rows:
    df = pd.DataFrame(rows)
    st.subheader("📊 Resultados (AUC ≥ 50 % antes JD121)")
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Descargar CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="patrones_auc50_clicks_valores_reales.csv")

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    for y, (xx, yy, col) in series.items():
        ax2.plot(xx, yy, label=y, color=col)
    ax2.axvline(121, color="black", ls="--", lw=1)
    ax2.set_xlabel("Día juliano (JD calibrado)")
    ax2.set_ylabel("Emergencia relativa")
    ax2.legend(ncol=6, fontsize=8)
    st.pyplot(fig2, clear_figure=True)
