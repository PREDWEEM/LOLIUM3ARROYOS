# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Calibración por CLICS sobre el gráfico (Plotly)
# Instalación automática de dependencias faltantes
# Clasificación CONCENTRADO/EXTENDIDO según AUC ≥50 % antes JD121

import os, sys, subprocess

# ====== 🔧 Instalación automática de dependencias ======
required = ["streamlit-plotly-events", "opencv-python", "plotly", "scipy", "pandas", "matplotlib", "numpy"]
for pkg in required:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Instalando {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# ====== Librerías principales ======
import cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# ====== CONFIGURACIÓN ======
st.set_page_config(page_title="PREDWEEM — Calibración por clics", layout="wide")
st.title("🌾 Clasificación por AUC con calibración por clics sobre el gráfico")

st.markdown("""
1️⃣ Hacé **dos clics** sobre el gráfico: el primero marca el **mínimo (JD_min)** y el segundo el **máximo (JD_max)** del eje X real.  
2️⃣ La app guarda la calibración por imagen y clasifica el patrón según **AUC ≥ 50 % antes de JD 121**.  
3️⃣ Podés limpiar o reutilizar calibraciones previas guardadas automáticamente.
""")

CALIB_FILE = "calibracion_patrones_clicks.csv"
JD_CUTOFF = 121
YEAR_JD_MAX = 365
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

def to_series(xs_px, ys_px, h, w, x_min_px, x_max_px):
    jd_all = (xs_px / max(1, (w - 1))) * YEAR_JD_MAX
    y = (h - 1 - ys_px).astype(float)
    if y.max() > 0: y /= y.max()
    jd_min = (x_min_px / max(1, (w - 1))) * YEAR_JD_MAX
    jd_max = (x_max_px / max(1, (w - 1))) * YEAR_JD_MAX
    mask = (jd_all >= jd_min) & (jd_all <= jd_max)
    return jd_all[mask], y[mask], float(jd_min), float(jd_max)

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
        try:
            return pd.read_csv(CALIB_FILE)
        except:
            return pd.DataFrame(columns=["imagen", "x_min_px", "x_max_px"])
    return pd.DataFrame(columns=["imagen", "x_min_px", "x_max_px"])

def save_calib(df): df.to_csv(CALIB_FILE, index=False)

# ====== SIDEBAR ======
with st.sidebar:
    st.header("🧭 Detección y suavizado")
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

# ====== LOOP PRINCIPAL ======
for f in files:
    st.markdown("---")
    st.subheader(f"🖼️ {f.name}")

    try:
        img_bgr = read_image(f)
        xs, ys, h, w = extract_curve(img_bgr, thr_dark, canny_low, canny_high)
        if xs.size == 0:
            st.error("⚠️ No se detectó la curva. Ajustá umbral o filtros Canny.")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        fig = px.imshow(img_rgb)
        fig.update_xaxes(showticklabels=False, range=[0, w])
        fig.update_yaxes(showticklabels=False, range=[h, 0])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="yellow", width=2), name="Curva detectada"))
        fig.update_layout(height=320, title="Hacé 2 clics: JD mínimo y JD máximo")

        clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=330)

        key = f"clicks_{f.name}"
        if key not in st.session_state: st.session_state[key] = []

        # guardar clics
        if clicks:
            for c in clicks:
                if "x" in c:
                    st.session_state[key].append(float(c["x"]))
            st.session_state[key] = st.session_state[key][:2]

        # mostrar resultado
        sel = st.session_state[key]
        if len(sel) == 2:
            x_min_px, x_max_px = sorted(sel)
            st.success(f"Selección: x_min_px={int(x_min_px)}, x_max_px={int(x_max_px)}")
        else:
            st.info("Hacé 2 clics sobre el gráfico (min y max).")
            continue

        # guardar calibración
        if auto_save:
            df_calib = df_calib[df_calib["imagen"] != f.name]
            df_calib.loc[len(df_calib)] = [f.name, x_min_px, x_max_px]
            save_calib(df_calib)

        # ====== PROCESO Y CLASIFICACIÓN ======
        x_jd, y_raw, jd_min, jd_max = to_series(xs, ys, h, w, x_min_px, x_max_px)
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

    except Exception as e:
        st.error(f"Error procesando {f.name}: {e}")

# ====== RESULTADOS ======
if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Descargar CSV", df.to_csv(index=False).encode("utf-8"), file_name="patrones_auc50_clicks.csv")

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    for y, (xx, yy, col) in series.items():
        ax2.plot(xx, yy, label=y, color=col)
    ax2.axvline(121, color="black", ls="--", lw=1)
    ax2.set_xlabel("Día juliano (JD)")
    ax2.set_ylabel("Emergencia relativa")
    ax2.legend(ncol=6, fontsize=8)
    st.pyplot(fig2, clear_figure=True)

