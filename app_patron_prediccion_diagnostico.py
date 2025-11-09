# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador (JD 1–121) con asignación manual de patrones
import streamlit as st
import cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt, pickle
from pathlib import Path
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="PREDWEEM — Entrenamiento manual + diagnóstico", layout="wide")
st.title("🌾 Clasificador PREDWEEM — Asignación manual + Diagnóstico de estabilidad (JD 1–121)")

st.markdown("""
Podés cargar varias imágenes históricas y **asignar manualmente el patrón** (P1, P1b, P2, P3)  
antes de entrenar el modelo.  
Solo se usa el tramo **JD 1–121 (1 enero → 1 mayo)** para el análisis.
""")

# ========= FUNCIONES =========
def extraer_curva(path_img):
    img = cv2.imread(str(path_img))
    if img is None:
        raise ValueError("No se pudo leer la imagen.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray = gray[int(h*0.1):int(h*0.95), int(w*0.08):]
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
    y_smooth = pd.Series(y).rolling(window=7, min_periods=1, center=True).mean()
    peaks, props = find_peaks(y_smooth, height=0.1, distance=8)
    h = props.get("peak_heights", [])
    n_peaks = len(peaks)
    sep = np.diff(x[peaks]) if n_peaks > 1 else [0]
    dur = (x[-1] - x[0]) if len(x) > 1 else 0
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

modelo_path = Path("modelo_patrones_jd121.pkl")

# ========= CARGA DE IMÁGENES =========
st.header("📥 Asignación manual de patrones para entrenamiento")
uploaded_train = st.file_uploader("Cargar imágenes históricas (.png o .jpg)", type=["png","jpg"], accept_multiple_files=True)

train_data = []
if uploaded_train:
    st.info("Revisá cada imagen y seleccioná el patrón correspondiente:")
    cols = st.columns(2)
    for i, file in enumerate(uploaded_train):
        with cols[i % 2]:
            tmp = Path(f"temp_{file.name}")
            tmp.write_bytes(file.read())
            st.image(str(tmp), caption=file.name, use_container_width=True)
            label = st.selectbox(f"🧭 Patrón para {file.name}",
                                 ["P1", "P1b", "P2", "P3"],
                                 key=f"label_{file.name}")
            train_data.append((tmp, label))

# ========= ENTRENAMIENTO =========
if st.button("🏋️ Entrenar modelo con asignación manual (JD 1–121)"):
    if not train_data:
        st.error("⚠️ No se cargaron imágenes o etiquetas.")
    else:
        X, y = [], []
        for path_img, label in train_data:
            try:
                x, yv = extraer_curva(path_img)
                mask = (x >= 1) & (x <= 121)
                x, yv = x[mask], yv[mask]
                feats, _ = extraer_features(x, yv)
                X.append(feats); y.append(label)
            except Exception as e:
                st.warning(f"Error procesando {path_img.name}: {e}")
        if len(X) < 2:
            st.error("❌ No hay suficientes curvas válidas.")
        else:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(n_estimators=300, random_state=42))
            ])
            model.fit(np.array(X), np.array(y))
            pickle.dump(model, open(modelo_path, "wb"))
            st.success(f"✅ Modelo entrenado con {len(X)} imágenes etiquetadas manualmente.")
            
            df_train = pd.DataFrame({"Imagen": [p.name for p,_ in train_data],
                                     "Patrón asignado": y})
            st.dataframe(df_train)

            with open(modelo_path, "rb") as f:
                st.download_button("💾 Descargar modelo entrenado (.pkl)",
                                   f, file_name="modelo_patrones_jd121.pkl",
                                   mime="application/octet-stream")

# ========= CLASIFICACIÓN =========
st.header("📈 Clasificación y diagnóstico de estabilidad")
uploaded_pred = st.file_uploader("Cargar imagen para clasificación (.png o .jpg)", type=["png","jpg"])

if uploaded_pred and modelo_path.exists():
    tmp = Path("temp_pred.png"); tmp.write_bytes(uploaded_pred.read())
    model = pickle.load(open(modelo_path, "rb"))

    x, yv = extraer_curva(tmp)
    mask_121 = (x >= 1) & (x <= 121)
    x, yv = x[mask_121], yv[mask_121]
    feats, f_names = extraer_features(x, yv)
    pred = model.predict([feats])[0]
    prob = np.max(model.predict_proba([feats]))
    st.success(f"📊 Patrón detectado (JD 1–121): **{pred}** — Probabilidad: {prob:.2f}")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, yv, color="black", lw=1.5)
    ax.axvline(121, color="red", linestyle="--", lw=1)
    ax.text(123, 0.9, "1 mayo", color="red", fontsize=9, va="center")
    ax.set_xlabel("Día Juliano"); ax.set_ylabel("Emergencia relativa")
    ax.set_title(f"Clasificación principal: {pred} (prob={prob:.2f})")
    st.pyplot(fig)

    # -------- Diagnóstico --------
    st.subheader("🧭 Diagnóstico de estabilidad (JD 60–121)")
    cortes = [60, 90, 105, 121]
    resultados = []
    for c in cortes:
        mask = (x >= 1) & (x <= c)
        feats_c, _ = extraer_features(x[mask], yv[mask])
        pred_c = model.predict([feats_c])[0]
        prob_c = np.max(model.predict_proba([feats_c]))
        resultados.append((c, pred_c, prob_c))
    df_diag = pd.DataFrame(resultados, columns=["JD_final","Patrón","Probabilidad"])
    st.dataframe(df_diag.style.format({"Probabilidad":"{:.2f}"}))

    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.plot(df_diag["JD_final"], df_diag["Probabilidad"], "o-", color="royalblue", lw=2)
    for i,row in df_diag.iterrows():
        ax2.text(row["JD_final"], row["Probabilidad"]+0.02, row["Patrón"], ha="center", fontsize=9)
    ax2.axhline(0.75, color="green", linestyle="--", lw=1, alpha=0.6, label="Alta certeza (≥0.75)")
    ax2.axhline(0.45, color="orange", linestyle="--", lw=1, alpha=0.5, label="Media (≥0.45)")
    ax2.set_xlabel("JD límite usado para clasificación")
    ax2.set_ylabel("Probabilidad")
    ax2.set_title("Evolución temporal de la clasificación (1 ene → 1 may)")
    ax2.legend(); ax2.grid(alpha=0.3)
    st.pyplot(fig2)

# ========= LIMPIAR =========
if st.sidebar.button("🧹 Borrar modelo entrenado"):
    if modelo_path.exists():
        modelo_path.unlink()
        st.sidebar.warning("🗑️ Modelo eliminado.")
