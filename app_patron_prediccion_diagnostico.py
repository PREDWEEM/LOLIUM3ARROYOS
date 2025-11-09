# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador (JD 1–121) con diagnóstico temporal + guardado/carga de modelo
import streamlit as st
import cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt, pickle
from pathlib import Path
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ========= CONFIGURACIÓN =========
st.set_page_config(page_title="PREDWEEM — Clasificador con guardado/carga de modelo", layout="wide")
st.title("🌾 Clasificador PREDWEEM — Diagnóstico y gestión de modelo (JD 1–121)")

st.markdown("""
El modelo analiza **solo los días julianos 1–121 (1 enero → 1 mayo)**  
y permite **entrenar, guardar, cargar y diagnosticar estabilidad temporal**  
de la clasificación de patrones de emergencia.
""")

# ========= FUNCIONES BASE =========
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

# ========= SIDEBAR: OPCIONES DE MODELO =========
st.sidebar.header("⚙️ Gestión de modelo")

modo_modelo = st.sidebar.radio("Seleccioná una opción:",
                               ["Entrenar nuevo modelo", "Cargar modelo guardado (.pkl)"],
                               index=0)

modelo = None

# --- Opción 1: Entrenamiento ---
if modo_modelo == "Entrenar nuevo modelo":
    uploaded_train = st.sidebar.file_uploader("Cargar imágenes para entrenamiento", type=["png","jpg"], accept_multiple_files=True)
    if st.sidebar.button("🏋️ Entrenar modelo (JD 1–121)"):
        if not uploaded_train:
            st.error("No se cargaron imágenes.")
        else:
            X, y, etiquetas = [], [], {}
            for f in uploaded_train:
                label = st.sidebar.selectbox(f"Etiqueta para {f.name}", ["P1","P1b","P2","P3"], key=f.name)
                tmp = Path(f"tmp_{f.name}")
                tmp.write_bytes(f.read())
                x, yv = extraer_curva(tmp)
                mask = (x >= 1) & (x <= 121)
                x, yv = x[mask], yv[mask]
                feats, _ = extraer_features(x, yv)
                X.append(feats); y.append(label)
                etiquetas[f.name] = label

            X, y = np.array(X), np.array(y)
            modelo = Pipeline([
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(n_estimators=300, random_state=42))
            ])
            modelo.fit(X, y)
            pickle.dump(modelo, open(modelo_path, "wb"))

            st.success(f"✅ Modelo entrenado con {len(X)} curvas.")
            df_train = pd.DataFrame(list(etiquetas.items()), columns=["Imagen", "Patrón"])
            st.dataframe(df_train)

            # 💾 Descargar modelo
            with open(modelo_path, "rb") as f:
                st.download_button("💾 Descargar modelo entrenado (.pkl)",
                                   f, file_name="modelo_patrones_jd121.pkl",
                                   mime="application/octet-stream")

# --- Opción 2: Cargar modelo guardado ---
elif modo_modelo == "Cargar modelo guardado (.pkl)":
    uploaded_model = st.sidebar.file_uploader("Cargar archivo .pkl", type=["pkl"])
    if uploaded_model:
        modelo_path = Path("modelo_cargado.pkl")
        modelo_path.write_bytes(uploaded_model.read())
        modelo = pickle.load(open(modelo_path, "rb"))
        st.sidebar.success("✅ Modelo cargado correctamente.")

# ========= CLASIFICACIÓN + DIAGNÓSTICO =========
st.header("📈 Clasificación y diagnóstico (solo JD 1–121)")
uploaded_pred = st.file_uploader("Cargar imagen para clasificación (.png o .jpg)", type=["png","jpg"])

if uploaded_pred and modelo_path.exists():
    tmp = Path("temp_pred.png"); tmp.write_bytes(uploaded_pred.read())
    modelo = pickle.load(open(modelo_path, "rb"))

    x, yv = extraer_curva(tmp)
    mask_121 = (x >= 1) & (x <= 121)
    x, yv = x[mask_121], yv[mask_121]

    feats, f_names = extraer_features(x, yv)
    pred = modelo.predict([feats])[0]
    prob = np.max(modelo.predict_proba([feats]))

    st.success(f"📊 Patrón detectado (JD 1–121): **{pred}** — Probabilidad: {prob:.2f}")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, yv, color="black", lw=1.5)
    ax.axvline(121, color="red", linestyle="--", lw=1)
    ax.text(123, 0.9, "1 mayo", color="red", fontsize=9, va="center")
    ax.set_xlabel("Día Juliano"); ax.set_ylabel("Emergencia relativa")
    ax.set_title(f"Clasificación principal: {pred} (prob={prob:.2f})")
    st.pyplot(fig)

    # -------- Diagnóstico temporal --------
    st.subheader("🧭 Diagnóstico de estabilidad de clasificación")
    cortes = [60, 90, 105, 121]
    resultados = []
    for c in cortes:
        mask = (x >= 1) & (x <= c)
        feats_c, _ = extraer_features(x[mask], yv[mask])
        pred_c = modelo.predict([feats_c])[0]
        prob_c = np.max(modelo.predict_proba([feats_c]))
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
    ax2.set_title("Evolución temporal de certeza de clasificación")
    ax2.legend(); ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    # 📅 Identificar punto de estabilización
    estables = df_diag[df_diag["Probabilidad"] >= 0.75]
    if not estables.empty:
        jd_estable = int(estables.iloc[0]["JD_final"])
        st.success(f"✅ El patrón se estabiliza alrededor del **JD {jd_estable}**, "
                   f"equivalente al {jd_estable}° día del año (~{jd_estable//30+1:02d}-mes).")
    else:
        st.warning("⚠️ La clasificación no alcanza alta certeza (≥0.75) antes de JD 121.")

# ========= LIMPIAR MODELO =========
if st.sidebar.button("🧹 Borrar modelo"):
    if modelo_path.exists():
        modelo_path.unlink()
        st.sidebar.warning("🗑️ Modelo eliminado. Podés entrenar o cargar otro.")
