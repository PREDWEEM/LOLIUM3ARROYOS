
# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM–METEO v1.0 — Streamlit App
# ===============================================================
# Predice el patrón histórico más probable SOLO desde meteorología.
# Usa:
#   ✔ curvas históricas anuales (XLSX)
#   ✔ meteo por pestañas (XLSX)
#   ✔ extracción de features de curvas y meteo
#   ✔ clustering DTW (k-medoids)
#   ✔ GradientBoostingClassifier
#   ✔ Predicción del patrón más probable
#   ✔ Probabilidades completas
#   ✔ Exportación del modelo
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import io
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# ===============================================================
# 0) CONFIG STREAMLIT
# ===============================================================

st.set_page_config(
    page_title="🌾 PREDWEEM–METEO v1.0",
    layout="wide"
)

st.title("🌾 PREDWEEM–METEO v1.0 — Predicción del patrón histórico desde meteorología")


# ===============================================================
# 1) FUNCIONES GENERALES
# ===============================================================

def cargar_curva_acumulada(file):
    df = pd.read_excel(file, header=None)
    dias = df.iloc[:,0].values
    vals = df.iloc[:,1].values

    diario = np.zeros(365)
    for d,v in zip(dias, vals):
        try:
            dd = int(d)
            if 1 <= dd <= 365:
                diario[dd-1] = float(v)
        except:
            continue

    acum = np.cumsum(diario)
    if acum[-1] == 0:
        return np.zeros(365)
    return acum / acum[-1]


def cargar_meteo_xlsx(file):
    book = pd.read_excel(file, sheet_name=None)
    out = {}
    for name, df in book.items():
        try:
            year = int(re.findall(r"\d{4}", str(name))[0])
        except:
            continue

        df = df.rename(columns={
            "TMIN": "tmin", "TMAX": "tmax", "Prec": "prec",
            "Julian_days": "jd", "Día juliano": "jd"
        })

        for c in ["jd","tmin","tmax","prec"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["jd"])
        df = df.sort_values("jd")
        df = df[(df["jd"] >= 1) & (df["jd"] <= 274)]
        df = df.reset_index(drop=True)

        if all(c in df.columns for c in ["tmin","tmax","prec"]):
            out[year] = df

    return out


def max_run(mask):
    c = m = 0
    for v in mask:
        c = c + 1 if v else 0
        m = max(m, c)
    return m


def features_meteo(df):

    tmin = df["tmin"].values
    tmax = df["tmax"].values
    tmed = (tmin + tmax)/2
    prec = df["prec"].values

    gdd5 = np.cumsum(np.maximum(tmed - 5, 0))
    gdd3 = np.cumsum(np.maximum(tmed - 3, 0))

    return {
        "gdd5_120": gdd5[min(119,len(gdd5)-1)],
        "gdd3_120": gdd3[min(119,len(gdd3)-1)],
        "pp_120": np.sum(prec[:120]),
        "tmed_14may": np.mean(tmed[136:150]) if len(tmed)>150 else np.nan,
        "tmed_28may": np.mean(tmed[122:150]) if len(tmed)>150 else np.nan,
        "ev10_FM": np.sum(prec[31:151] >= 10),
        "ev20_FM": np.sum(prec[31:151] >= 20),
        "FM_pp": np.sum(prec[31:151]),
        "dryrun_FM": max_run(prec[31:151] < 1),
        "wetrun_FM": max_run(prec[31:151] >= 5)
    }


def features_curva(curva):
    idx_inicio = int(np.argmax(curva > 0)) + 1
    frac_120 = curva[119] if len(curva)>=120 else curva[-1]
    tramo = curva[29:121]
    tasa = np.nanmean(np.diff(tramo))
    return {
        "inicio": idx_inicio,
        "frac_120": frac_120,
        "tasa_30_120": tasa
    }




# ===============================================================
# 2) DTW + K-MEDOIDS
# ===============================================================

def dtw(a,b):
    n,m = len(a), len(b)
    D = np.full((n+1,m+1), np.inf)
    D[0,0] = 0
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost = (a[i-1]-b[j-1])**2
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return np.sqrt(D[n,m])


def k_medoids(curvas, K=3):
    N = len(curvas)
    rng = np.random.default_rng(0)
    medoids = rng.choice(N, K, replace=False)

    for _ in range(40):
        clusters = {k:[] for k in range(K)}
        for i in range(N):
            d = [dtw(curvas[i], curvas[m]) for m in medoids]
            k = np.argmin(d)
            clusters[k].append(i)

        new_medoids=[]
        for k in range(K):
            if not clusters[k]:
                new_medoids.append(medoids[k])
                continue

            sub = clusters[k]
            subD = np.zeros((len(sub), len(sub)))
            for i,p in enumerate(sub):
                for j,q in enumerate(sub):
                    subD[i,j] = dtw(curvas[p], curvas[q])
            new_medoids.append(sub[np.argmin(subD.sum(axis=1))])

        new_medoids = np.array(new_medoids)
        if np.all(new_medoids == medoids):
            break
        medoids = new_medoids

    return medoids, clusters


# ===============================================================
# 3) ENTRENAMIENTO COMPLETO
# ===============================================================

def entrenar_modelo(curvas, meteo_dict, años, K=3):

    Fcurva = pd.DataFrame([features_curva(c) for c in curvas])
    Fmeteo = pd.DataFrame([features_meteo(meteo_dict[y]) for y in años])

    medoids, clusters = k_medoids(curvas, K=K)

    y_labels = np.zeros(len(curvas), dtype=int)
    for k,lista in clusters.items():
        for i in lista:
            y_labels[i] = k

    X = pd.concat([Fmeteo, Fcurva], axis=1)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    clf = GradientBoostingClassifier().fit(Xs, y_labels)

    return clf, scaler, X.columns, medoids, clusters


# ===============================================================
# 4) PREDICCIÓN NUEVO AÑO
# ===============================================================

def predecir_patron(df_meteo, clf, scaler, cols):
    f = features_meteo(df_meteo)
    X = pd.DataFrame([f])[cols]
    Xs = scaler.transform(X)
    proba = clf.predict_proba(Xs)[0]
    k = int(np.argmax(proba))
    return k, proba


# ===============================================================
# 5) INTERFAZ STREAMLIT
# ===============================================================

st.header("📘 Entrenamiento del modelo")
meteo_file = st.file_uploader("Cargar meteorología multianual (pestañas por año)", type=["xlsx"])
curvas_files = st.file_uploader("Cargar curvas históricas (XLSX por año)", type=["xlsx"], accept_multiple_files=True)

K = st.slider("Número de patrones (clusters K)", 2, 6, 3)

if st.button("🚀 Entrenar modelo"):
    if not (meteo_file and curvas_files):
        st.error("Faltan archivos")
        st.stop()

    meteo_dict = cargar_meteo_xlsx(meteo_file)

    años_curvas = []
    curvas = []
    for f in curvas_files:
        y = int(re.findall(r"\d{4}", f.name)[0])
        curva = cargar_curva_acumulada(f)
        if y in meteo_dict:
            años_curvas.append(y)
            curvas.append(curva)

    clf, scaler, cols, medoids, clusters = entrenar_modelo(curvas, meteo_dict, años_curvas, K=K)

    st.success("Modelo entrenado correctamente")

    buf = io.BytesIO()
    dump({
        "clf": clf,
        "scaler": scaler,
        "cols": cols,
        "medoids": medoids,
        "clusters": clusters,
        "años": años_curvas
    }, buf)

    st.download_button(
        "💾 Descargar modelo entrenado (.joblib)",
        data=buf.getvalue(),
        file_name="predweem_meteo_v1.joblib"
    )


# ===============================================================
# 6) PREDICCIÓN NUEVO AÑO
# ===============================================================

st.header("🔮 Predicción del patrón histórico desde meteorología nueva")

modelo_file = st.file_uploader("Cargar modelo entrenado (.joblib)", type=["joblib"], key="model_upl")
meteo_nueva_file = st.file_uploader("Cargar meteorología NUEVA (XLSX)", type=["xlsx"], key="meteo_new")

if st.button("🔍 Predecir patrón"):
    if not (modelo_file and meteo_nueva_file):
        st.error("Faltan archivos")
        st.stop()

    M = load(modelo_file)
    clf = M["clf"]
    scaler = M["scaler"]
    cols = M["cols"]

    dfm = list(pd.read_excel(meteo_nueva_file, sheet_name=None).values())[0]
    dfm = dfm.rename(columns={
        "TMIN": "tmin", "TMAX": "tmax", "Prec": "prec",
        "Julian_days": "jd", "Día juliano": "jd"
    })
    dfm = dfm.dropna(subset=["jd"]).sort_values("jd")

    k, proba = predecir_patron(dfm, clf, scaler, cols)

    st.subheader(f"🎯 Patrón más probable: **C{k}**")
    st.write("Probabilidades:")
    for i,p in enumerate(proba):
        st.write(f"- C{i}: {p:.3f}")






