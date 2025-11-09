# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador (JD 1–121) con asignación manual y guardado del modelo
import streamlit as st
import cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt, pickle
from pathlib import Path
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="PREDWEEM — Entrenamiento manual + guardado modelo", layout="wide")
st.title("🌾 Clasificador PREDWEEM — Entrenamiento manual y guardado del modelo (JD 1–121)")

st.markdown("""
Podés **asignar manualmente el patrón** (P1, P1b, P2, P3) a cada imagen antes de entrenar.  
El modelo se entrena solo con los días julianos **1–121 (1 enero → 1 mayo)**  
y luego podrás **descargar el modelo entrenado (.pkl)** para reutilizarlo o guardarlo.
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

# ========= CARGA Y ASIGNACIÓN MANUAL =========
st.header("📥 Cargar imágenes y asignar manualmente el patrón")
uploaded_train = st.file_uploader("Seleccioná las imágenes históricas (.png o .jpg)", type=["png","jpg"], accept_multiple_files=True)

train_data = []
if uploaded_train:
    st.info("Seleccioná manualmente el patrón correspondiente a cada imagen:")
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

# ========= ENTRENAMIENTO Y GUARDADO =========
if st.button("🏋️ Entrenar y guardar modelo (JD 1–121)"):
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

            # 💾 OPCIÓN DE DESCARGA Y GUARDADO LOCAL
            with open(modelo_path, "rb") as f:
                st.download_button(
                    label="💾 Descargar modelo entrenado (.pkl)",
                    data=f,
                    file_name="modelo_patrones_jd121.pkl",
                    mime="application/octet-stream"
                )

            st.info("""
            El modelo se guardó localmente en el servidor y también podés descargarlo 
            con el botón anterior para usarlo más adelante en otra sesión.
            """)

# ========= OPCIÓN DE CARGA DE MODELO GUARDADO =========
st.header("📂 Cargar modelo previamente guardado")
uploaded_model = st.file_uploader("Cargar archivo de modelo (.pkl)", type=["pkl"])
if uploaded_model:
    modelo_path = Path("modelo_cargado.pkl")
    modelo_path.write_bytes(uploaded_model.read())
    st.success("✅ Modelo cargado correctamente. Ya podés clasificar nuevas imágenes.")

# ========= CLASIFICACIÓN =========
st.header("📈 Clasificar imagen con el modelo activo")
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
