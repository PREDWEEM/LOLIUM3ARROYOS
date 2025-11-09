# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador de patrones históricos (recorte 1–121 JD)
import streamlit as st
import cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt, pickle
from pathlib import Path
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="PREDWEEM — Clasificador (JD 1–121)", layout="wide")
st.title("🌾 Clasificador PREDWEEM — Basado en días julianos 1–121 (1 enero → 1 mayo)")

st.markdown("""
Solo se utiliza la información **comprendida entre JD 1 y JD 121**  
para extraer las características y clasificar el patrón de emergencia.  
Ideal para predicciones **anticipadas antes del 1° de mayo**.
""")

# ========= FUNCIONES =========
def extraer_curva(path_img):
    """Extrae la curva negra principal de una figura de emergencia (eje X: días julianos)."""
    img = cv2.imread(str(path_img))
    if img is None:
        raise ValueError("No se pudo leer la imagen.")
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

    # 🟢 Recortar a ventana 1–121
    mask = (x_jd >= 1) & (x_jd <= 121)
    return x_jd[mask], y_norm[mask]

def extraer_features(x, y):
    """Extrae descriptores numéricos de la curva (solo JD 1–121)."""
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
st.header("📥 Cargar imágenes para entrenamiento")

uploaded_files = st.file_uploader(
    "Seleccioná una o más imágenes (.png o .jpg)",
    type=["png", "jpg"],
    accept_multiple_files=True
)

train_data = []
if uploaded_files:
    st.info("Seleccioná el tipo de patrón para cada imagen cargada (solo se usará JD 1–121).")
    for file in uploaded_files:
        label = st.selectbox(
            f"🖼️ {file.name} — seleccionar patrón:",
            ["P1", "P1b", "P2", "P3"],
            key=file.name
        )
        temp_path = Path(f"temp_{file.name}")
        temp_path.write_bytes(file.read())
        train_data.append((temp_path, label))
        st.image(str(temp_path), caption=f"{file.name} — {label}", use_container_width=True)

# ========= ENTRENAMIENTO =========
if st.button("🏋️ Entrenar modelo con imágenes cargadas (solo JD 1–121)"):
    if not train_data:
        st.error("⚠️ No se cargaron imágenes para entrenamiento.")
    else:
        X, y = [], []
        for path_img, label in train_data:
            try:
                x, yv = extraer_curva(path_img)
                feats, _ = extraer_features(x, yv)
                if np.any(np.isnan(feats)) or np.all(feats == 0):
                    st.warning(f"⚠️ {path_img.name} tiene datos no válidos. Omitido.")
                    continue
                X.append(feats)
                y.append(label)
            except Exception as e:
                st.warning(f"Error al procesar {path_img.name}: {e}")

        if len(X) < 2:
            st.error("❌ No hay suficientes curvas válidas para entrenar el modelo.")
        else:
            X, y = np.array(X), np.array(y)
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(n_estimators=300, random_state=42))
            ])
            pipe.fit(X, y)
            pickle.dump(pipe, open(modelo_path, "wb"))
            st.success(f"✅ Modelo entrenado correctamente (ventana JD 1–121).")
            df_train = pd.DataFrame({"imagen": [p.name for p,_ in train_data], "patrón": y})
            st.dataframe(df_train)

# ========= CLASIFICACIÓN =========
st.header("📈 Clasificar nueva curva (usando solo JD 1–121)")
uploaded_pred = st.file_uploader("Cargar imagen para clasificación (.png o .jpg)", type=["png","jpg"])

if uploaded_pred and modelo_path.exists():
    tmp = Path("temp_upload_pred.png")
    tmp.write_bytes(uploaded_pred.read())
    model = pickle.load(open(modelo_path, "rb"))

    try:
        x, yv = extraer_curva(tmp)
        feats, f_names = extraer_features(x, yv)
        pred = model.predict([feats])[0]
        prob = np.max(model.predict_proba([feats]))
        st.success(f"📊 Patrón detectado: **{pred}** — Probabilidad: {prob:.2f}")

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, yv, color="black", lw=1.5)
        ax.set_title(f"Clasificación (JD 1–121): {pred} (prob={prob:.2f})")
        ax.set_xlabel("Día Juliano"); ax.set_ylabel("Emergencia relativa (0–1)")
        ax.axvline(121, color="red", linestyle="--", lw=1)
        ax.text(123, 0.9, "1 mayo", color="red", fontsize=9, va="center")
        st.pyplot(fig)

        feat_table = pd.DataFrame([feats], columns=f_names).T
        st.markdown("### 🔍 Características extraídas (solo JD 1–121)")
        st.dataframe(feat_table.style.format("{:.2f}"))

    except Exception as e:
        st.error(f"❌ Error al analizar la imagen: {e}")

elif uploaded_pred:
    st.error("⚠️ Primero entrená el modelo con al menos dos imágenes.")

# ========= LIMPIAR MODELO =========
if st.sidebar.button("🧹 Borrar modelo entrenado"):
    if modelo_path.exists():
        modelo_path.unlink()
        st.warning("🗑️ Modelo eliminado. Podés volver a entrenarlo.")

