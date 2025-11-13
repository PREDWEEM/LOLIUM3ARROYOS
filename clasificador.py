# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM–METEO v5.5
# Clasificación del patrón histórico desde meteorología
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump, load

# ===============================================================
# STREAMLIT UI
# ===============================================================

st.set_page_config(page_title="PREDWEEM–METEO v5.5", layout="wide")
st.title("🌾 PREDWEEM–METEO v5.5 — Clasificación de Patrón Histórico Desde Meteorología")


# ===============================================================
# AUXILIARES
# ===============================================================

def safe_mean(arr):
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if len(arr) > 0 else 0.0

def safe_idx(arr, idx):
    if len(arr) == 0:
        return 0.0
    idx = min(idx, len(arr)-1)
    return float(arr[idx])

def max_run(vec):
    vec = np.array(vec).astype(int)
    m = c = 0
    for v in vec:
        c = c + 1 if v else 0
        m = max(m, c)
    return int(m)


# ===============================================================
# CARGA DE METEOROLOGÍA — FORMATO REAL DEL USUARIO
# ===============================================================

def cargar_meteo_xlsx(file):
    book = pd.read_excel(file, sheet_name=None)
    out = {}
    problemas = []

    for name, df in book.items():

        if not isinstance(df, pd.DataFrame):
            continue

        # detectar año desde el nombre de la hoja
        try:
            year = int(re.findall(r"\d{4}", str(name))[0])
        except:
            continue

        # Normalizar nombres de columnas
        df.columns = [c.strip().lower() for c in df.columns]

        # Requeridos EXACTOS según formato enviado
        required = ["jd", "tmin", "tmax", "prec"]
        if not all(col in df.columns for col in required):
            problemas.append(name)
            continue

        # Convertir JD
        df["jd"] = pd.to_numeric(df["jd"], errors="coerce")
        df = df.dropna(subset=["jd"])
        df = df[(df["jd"] >= 1) & (df["jd"] <= 365)]

        if df.empty:
            problemas.append(name)
            continue

        # COMA DECIMAL → PUNTO
        for c in ["tmin", "tmax", "prec"]:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )

        df = df.dropna(subset=["tmin", "tmax", "prec"])
        df = df.sort_values("jd").reset_index(drop=True)

        out[year] = df

    if problemas:
        st.warning(f"⚠ Pestañas ignoradas por no tener columnas JD/Tmin/Tmax/prec: {problemas}")

    return out


# ===============================================================
# FEATURES METEOROLÓGICAS
# ===============================================================

def features_meteo(df):
    tmin = df["tmin"].values
    tmax = df["tmax"].values
    tmed = (tmin + tmax) / 2
    prec = df["prec"].values

    tmed = np.nan_to_num(tmed)
    prec = np.nan_to_num(prec)

    gdd5 = np.cumsum(np.maximum(tmed - 5, 0))
    gdd3 = np.cumsum(np.maximum(tmed - 3, 0))

    return {
        "gdd5_120": safe_idx(gdd5, 119),
        "gdd3_120": safe_idx(gdd3, 119),
        "pp_120": float(np.sum(prec[:120])),
        "tmed_14may": safe_mean(tmed[136:150]),
        "tmed_28may": safe_mean(tmed[122:150]),
        "FM_pp": float(np.sum(prec[31:151])),
        "ev10_FM": int(np.sum(prec[31:151] >= 10)),
        "ev20_FM": int(np.sum(prec[31:151] >= 20)),
        "dryrun_FM": max_run(prec[31:151] < 1),
        "wetrun_FM": max_run(prec[31:151] >= 5)
    }


# ===============================================================
# DTW + K-MEDOIDS
# ===============================================================

def dtw(a, b):
    n, m = len(a), len(b)
    D = np.full((n+1,m+1), np.inf)
    D[0,0] = 0
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost = (a[i-1]-b[j-1])**2
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(np.sqrt(D[n,m]))


def k_medoids(curves, K=3, seed=42):
    rng = np.random.default_rng(seed)
    N = len(curves)
    K = min(K, N)
    med_idx = list(rng.choice(N, size=K, replace=False))

    # Matriz DTW
    D = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            d = dtw(curves[i], curves[j])
            D[i,j] = D[j,i] = d

    for _ in range(50):
        assign = np.argmin(D[:, med_idx], axis=1)
        new_meds = []
        for k in range(K):
            members = np.where(assign == k)[0]
            if len(members) == 0:
                new_meds.append(med_idx[k])
                continue
            sub = D[np.ix_(members, members)]
            sums = sub.sum(axis=1)
            best = members[np.argmin(sums)]
            new_meds.append(best)
        if new_meds == med_idx:
            break
        med_idx = new_meds

    clusters = {k: [] for k in range(K)}
    assign = np.argmin(D[:, med_idx], axis=1)
    for i in range(N):
        clusters[int(assign[i])].append(i)

    return med_idx, clusters, D


# ===============================================================
# CARGA ROBUSTA DE CURVAS
# ===============================================================

def cargar_curvas(files):
    curvas = []
    años = []

    for f in files:
        try:
            df = pd.read_excel(f, header=None)
        except:
            st.error(f"Error leyendo {f.name}")
            continue

        if df.shape[1] < 2:
            st.error(f"{f.name} no tiene dos columnas.")
            continue

        dias_raw = df.iloc[:, 0].values
        vals_raw = df.iloc[:, 1].values
        diario = np.zeros(365)

        for d, v in zip(dias_raw, vals_raw):
            try:
                d_num = float(str(d).replace(",", "."))
                d_int = int(d_num)
                if 1 <= d_int <= 365:
                    diario[d_int - 1] = float(str(v).replace(",", "."))
            except:
                continue

        acum = np.cumsum(diario)
        maxv = acum.max()
        if maxv > 0:
            acum = acum / maxv

        curva = np.maximum.accumulate(acum)
        curvas.append(curva)

        y4 = re.findall(r"(\d{4})", f.name)
        años.append(int(y4[0]) if y4 else f.name)

    return curvas, años


# ===============================================================
# ENTRENAMIENTO
# ===============================================================

def entrenar_modelo(curvas, meteo_dict, años, K=4):
    medoids, clusters, D = k_medoids(curvas, K)

    X_meteo = []
    y_labels = []

    for i, año in enumerate(años):
        if año not in meteo_dict:
            continue

        feats = features_meteo(meteo_dict[año])
        X_meteo.append(list(feats.values()))

        dist_to_meds = [dtw(curvas[i], curvas[m]) for m in medoids]
        y_labels.append(int(np.argmin(dist_to_meds)))

    X_meteo = np.nan_to_num(np.array(X_meteo, float))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_meteo)

    clf = GradientBoostingClassifier()
    clf.fit(Xs, y_labels)

    return clf, scaler, list(feats.keys()), medoids, clusters


# ===============================================================
# STREAMLIT — ENTRENAMIENTO
# ===============================================================

st.header("🧪 Entrenamiento del Modelo")

meteo_file = st.file_uploader("📘 Cargar meteorología (pestañas)", type=["xlsx"])
curva_files = st.file_uploader("📈 Cargar curvas históricas", type=["xlsx"], accept_multiple_files=True)

K = st.slider("Número de patrones (K)", 2, 10, 4)

if st.button("🚀 ENTRENAR"):
    if not meteo_file or not curva_files:
        st.error("Faltan archivos.")
        st.stop()

    meteo_dict = cargar_meteo_xlsx(meteo_file)
    curvas, años = cargar_curvas(curva_files)

    clf, scaler, cols, medoids, clusters = entrenar_modelo(curvas, meteo_dict, años, K)

    st.success("Modelo entrenado correctamente.")

    bundle = {
        "clf": clf,
        "scaler": scaler,
        "cols": cols,
        "medoids": medoids,
        "curvas": curvas,
        "años": años
    }
    dump(bundle, "modelo_predweem_meteo.joblib")

    st.download_button(
        "💾 Descargar modelo",
        data=open("modelo_predweem_meteo.joblib", "rb").read(),
        file_name="modelo_predweem_meteo.joblib"
    )


# ===============================================================
# STREAMLIT — PREDICCIÓN
# ===============================================================

st.header("🔮 Predicción del Patrón Histórico")

modelo_file = st.file_uploader("📦 Cargar modelo entrenado", type=["joblib"], key="modfile")
meteo_nueva = st.file_uploader("🌦️ Cargar meteorología nueva", type=["xlsx"], key="metnew")

if st.button("🔍 PREDECIR"):
    if not (modelo_file and meteo_nueva):
        st.error("Cargar modelo y meteorología nueva.")
        st.stop()

    bundle = load(modelo_file)
    clf = bundle["clf"]
    scaler = bundle["scaler"]
    cols = bundle["cols"]
    medoids_idx = bundle["medoids"]
    curvas_hist = bundle["curvas"]

    metneo = cargar_meteo_xlsx(meteo_nueva)

    if not metneo:
        st.error("❌ No se detectaron pestañas válidas en la meteorología nueva.")
        st.stop()

    año_new = sorted(metneo.keys())[0]
    df_new = metneo[año_new]

    feats = features_meteo(df_new)
    X = np.array([[feats[c] for c in cols]])
    Xs = scaler.transform(X)

    proba = clf.predict_proba(Xs)[0]
    pred = int(np.argmax(proba))

    st.success(f"Patrón más probable: **C{pred}** con probabilidad {proba[pred]:.2f}")

    dias = np.arange(1, 366)
    curva_proto = curvas_hist[medoids_idx[pred]]

    dfp = pd.DataFrame({"Día": dias, "Valor": curva_proto})

    chart = alt.Chart(dfp).mark_line(color="blue").encode(
        x="Día",
        y=alt.Y("Valor", scale=alt.Scale(domain=[0,1]))
    ).properties(title=f"Patrón C{pred} (medoid)")

    st.altair_chart(chart, use_container_width=True)
