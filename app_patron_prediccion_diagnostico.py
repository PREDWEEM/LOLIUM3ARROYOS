# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador de patrones históricos (modelo base 2008–2012)
import streamlit as st
import cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt, pickle
from pathlib import Path
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ========= CONFIGURACIÓN =========
st.set_page_config(page_title="PREDWEEM — Clasificador de patrones históricos", layout="wide")
st.title("🌾 Clasificador de patrones históricos — Entrenamiento y predicción")

st.markdown("""
Este modelo se entrena con curvas históricas (2008–2012 + `newplot(7)` dentro de **P3**)  
y permite clasificar nuevas imágenes de emergencia relativa según el **patrón histórico** detectado:
- 🟦 **P1:** Emergencia rápida y compacta.  
- 🟩 **P1b:** Emergencia temprana con pequeño repunte posterior.  
- 🟧 **P2:** Emergencia bimodal (dos pulsos bien separados).  
- 🟥 **P3:** Emergencia extendida o prolongada.
""")

# ========= FUNCIONES AUXILIARES =========
def extraer_curva(path_img):
    """Extrae la curva negra principal de una figura de emergencia (eje X: días julianos)."""
    img = cv2.imread(str(path_img))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray = gray[int(h*0.1):int(h*0.95), int(w*0.08):]  # recorte margen
    inv = 255 - gray
    inv = cv2.GaussianBlur(inv, (3,3), 0)
    y_curve = []
    for i in range(inv.shape[1]):
        col = inv[:, i]
        if np.count_nonzero(col > 30) < inv.shape[0]*0.05:
            y_curve.append(np.nan)
            continue
        y_pos = np.argmax(col)
        y_curve.append(inv.shape[0] - y_pos)
    y_curve = pd.Series(y_curve).interpolate(limit_direction="both").to_numpy()
    y_norm = (y_curve - np.min(y_curve)) / (np.max(y_curve) - np.min(y_curve) + 1e-6)
    x_jd = np.linspace(0, 300, len(y_norm))
    return x_jd, y_norm

def extraer_features(x, y):
    """Extrae descriptores numéricos de la curva."""
    y_smooth = pd.Series(y).rolling(window=7, min_periods=1, center=True).mean()
    peaks, props = find_peaks(y_smooth, height=0.1, distance=8)
    h = props.get("peak_heights", [])
    n_peaks = len(peaks)
    sep = np.diff(x[peaks]) if n_peaks > 1 else [0]
    dur = x[np.nanargmax(y)] - x[np.nanargmin(y)]
    features = {
        "n_peaks": n_peaks,
        "mean_sep": np.mean(sep) if n_peaks>1 else 0,
        "std_sep": np.std(sep) if n_peaks>1 else 0,
        "max_h": np.max(h) if n_peaks>0 else 0,
        "mean_h": np.mean(h) if n_peaks>0 else 0,
        "first_peak": x[peaks[0]] if n_peaks>0 else 0,
        "last_peak": x[peaks[-1]] if n_peaks>0 else 0,
        "span": dur,
    }
    return np.array(list(features.values())), features.keys()

# ========= DATOS HISTÓRICOS =========
DATASET = {
    "2008.png": "P1b",
    "2009.png": "P2",
    "2010.png": "P3",
    "2011.png": "P1b",
    "2012.png": "P1",
    "newplot (7).png": "P3"
}

modelo_path = Path("modelo_patrones.pkl")

# ========= ENTRENAMIENTO =========
st.sidebar.header("⚙️ Entrenamiento del modelo")
if st.sidebar.button("🏋️ Entrenar modelo con históricos"):
    X, y = [], []
    for fname, label in DATASET.items():
        p = Path(fname)
        if p.exists():
            x, yv = extraer_curva(p)
            feats, _ = extraer_features(x, yv)
            X.append(feats)
            y.append(label)
    X, y = np.array(X), np.array(y)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42))
    ])
    pipe.fit(X, y)
    pickle.dump(pipe, open(modelo_path, "wb"))
    st.success(f"✅ Modelo entrenado y guardado en {modelo_path}")
    df_train = pd.DataFrame({"imagen": list(DATASET.keys()), "patrón": list(DATASET.values())})
    st.dataframe(df_train)

# ========= CLASIFICACIÓN =========
st.header("📈 Clasificación de una nueva curva")

uploaded = st.file_uploader("Cargar imagen de curva (.png o .jpg)", type=["png","jpg"])
if uploaded and modelo_path.exists():
    tmp = Path("temp_upload.png")
    tmp.write_bytes(uploaded.read())

    model = pickle.load(open(modelo_path, "rb"))
    x, y = extraer_curva(tmp)
    feats, f_names = extraer_features(x, y)

    pred = model.predict([feats])[0]
    prob = np.max(model.predict_proba([feats]))

    st.success(f"📊 Patrón detectado: **{pred}** — Probabilidad: {prob:.2f}")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, y, color="black", lw=1.5)
    ax.set_title(f"Clasificación: {pred} (prob={prob:.2f})")
    ax.set_xlabel("Día Juliano"); ax.set_ylabel("Emergencia relativa (0–1)")
    st.pyplot(fig)

    feat_table = pd.DataFrame([feats], columns=f_names).T
    st.markdown("### 🔍 Características extraídas")
    st.dataframe(feat_table.style.format("{:.2f}"))

elif uploaded:
    st.error("⚠️ Primero entrená el modelo antes de clasificar.")

# ========= LIMPIEZA OPCIONAL =========
if st.sidebar.button("🧹 Borrar modelo entrenado"):
    if modelo_path.exists():
        modelo_path.unlink()
        st.warning("Modelo eliminado. Podés volver a entrenarlo.")

