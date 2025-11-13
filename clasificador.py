# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM–METEO v5.3 — Clasificación del patrón histórico
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump, load

st.set_page_config(page_title="PREDWEEM–METEO v5.3", layout="wide")
st.title("🌾 PREDWEEM–METEO v5.3 — Clasificación del Patrón Desde Meteorología")


# ===============================================================
# UTILIDADES SEGURO-SIN-NAN
# ===============================================================

def safe_mean(arr):
    arr = np.array(arr, float)
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
# CARGA METEO ROBUSTA
# ===============================================================

def cargar_meteo_xlsx(file):
    book = pd.read_excel(file, sheet_name=None)
    out = {}
    problemas = []

    JD_VARIANTS = ["jd","Julian_days","Julian_day","dia","day","julian","JD","Día","dia juliano"]

    for name, df in book.items():
        try:
            year = int(re.findall(r"\d{4}", str(name))[0])
        except:
            continue

        if not isinstance(df, pd.DataFrame):
            continue

        df.columns = [str(c).strip() for c in df.columns]

        # JD
        jd_col = None
        for cand in JD_VARIANTS:
            for col in df.columns:
                if col.lower() == cand.lower():
                    jd_col = col
                    break
            if jd_col:
                break

        # Si no hay JD → intentar fecha
        if jd_col is None:
            fecha_cols = [c for c in df.columns if "fec" in c.lower()]
            if fecha_cols:
                try:
                    fch = pd.to_datetime(df[fecha_cols[0]], errors="coerce", dayfirst=True)
                    df["jd"] = fch.dt.dayofyear
                    jd_col = "jd"
                except:
                    problemas.append(name)
                    continue
            else:
                problemas.append(name)
                continue

        df["jd"] = pd.to_numeric(df[jd_col], errors="coerce")
        df = df.dropna(subset=["jd"])
        df = df[(df["jd"] >= 1) & (df["jd"] <= 274)]
        if df.empty:
            problemas.append(name)
            continue

        # Detectar variables
        def buscar(cands):
            for cand in cands:
                for col in df.columns:
                    if col.lower() == cand.lower():
                        return col
            return None

        col_tmin = buscar(["tmin","temperatura minima","min"])
        col_tmax = buscar(["tmax","temperatura maxima","max"])
        col_prec = buscar(["prec","pp","lluvia","rain"])

        if not (col_tmin and col_tmax and col_prec):
            problemas.append(name)
            continue

        df = df.rename(columns={
            col_tmin: "tmin",
            col_tmax: "tmax",
            col_prec: "prec"
        })

        for c in ["tmin","tmax","prec"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["tmin","tmax","prec"])
        if df.empty:
            problemas.append(name)
            continue

        out[year] = df.sort_values("jd").reset_index(drop=True)

    return out


# ===============================================================
# FEATURES METEO
# ===============================================================

def features_meteo(df):
    tmin = df["tmin"].values
    tmax = df["tmax"].values
    prec = df["prec"].values
    tmin = np.nan_to_num(tmin)
    tmax = np.nan_to_num(tmax)
    prec = np.nan_to_num(prec)

    tmed = (tmin + tmax) / 2
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

def k_medoids(curves, K=4):
    rng = np.random.default_rng(42)
    N = len(curves)
    med = rng.choice(N, size=min(K,N), replace=False).tolist()

    # matriz distancia
    D = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            d = dtw(curves[i], curves[j])
            D[i,j] = D[j,i] = d

    # refinamiento
    for _ in range(40):
        assign = np.argmin(D[:, med], axis=1)
        new = []
        for k in range(len(med)):
            members = np.where(assign==k)[0]
            if len(members)==0:
                new.append(med[k])
                continue
            sub = D[np.ix_(members, members)]
            best = members[np.argmin(sub.sum(axis=1))]
            new.append(best)
        if new == med:
            break
        med = new

    clusters = {k: [] for k in range(len(med))}
    assign = np.argmin(D[:, med], axis=1)
    for i in range(N):
        clusters[int(assign[i])].append(i)

    return med, clusters, D


# ===============================================================
# CARGA CURVAS — FIX DEFINITIVO
# ===============================================================

def cargar_curvas(files):
    curvas = []
    años = []

    for f in files:
        try:
            df = pd.read_excel(f, header=None)
        except:
            st.error(f"❌ Error leyendo {f.name}")
            continue

        if df.shape[1] < 2:
            st.error(f"❌ {f.name} no tiene 2 columnas.")
            continue

        dias = df.iloc[:,0].values
        vals = df.iloc[:,1].values

        diario = np.zeros(365, float)

        for d, v in zip(dias, vals):
            try:
                dnum = float(d)
                dint = int(dnum)
                if 1 <= dint <= 365:
                    diario[dint-1] = float(v)
            except:
                continue

        acum = np.cumsum(diario)
        maxv = acum.max() if acum.max() > 0 else 1
        curva = acum / maxv
        curva = np.maximum.accumulate(curva)

        curvas.append(curva)

        y4 = re.findall(r"(\d{4})", f.name)
        años.append(int(y4[0]) if y4 else f.name)

    return curvas, años


# ===============================================================
# ENTRENAMIENTO
# ===============================================================

def entrenar_modelo(curvas, meteo_dict, años, K):

    medoids, clusters, D = k_medoids(curvas, K)

    X = []
    y = []

    for i, año in enumerate(años):
        if año not in meteo_dict:
            continue
        feats = features_meteo(meteo_dict[año])
        X.append(list(feats.values()))
        dist = [dtw(curvas[i], curvas[m]) for m in medoids]
        y.append(int(np.argmin(dist)))

    X = np.array(X, float)
    X = np.nan_to_num(X)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = GradientBoostingClassifier()
    clf.fit(Xs, y)

    return clf, scaler, list(feats.keys()), medoids


# ===============================================================
# STREAMLIT UI — ENTRENAR
# ===============================================================

st.header("🧪 Entrenamiento")

meteo_file = st.file_uploader("📘 Meteorología multianual (XLSX con pestañas)", type=["xlsx"])
curva_files = st.file_uploader("📈 Curvas históricas", type=["xlsx"], accept_multiple_files=True)
K = st.slider("Número de patrones (K)", 2, 10, 4)

if st.button("🚀 ENTRENAR"):
    if not (meteo_file and curva_files):
        st.error("❌ Faltan archivos")
        st.stop()

    meteo_dict = cargar_meteo_xlsx(meteo_file)
    curvas, años = cargar_curvas(curva_files)

    clf, scaler, cols, medoids = entrenar_modelo(curvas, meteo_dict, años, K)

    bundle = {
        "clf": clf,
        "scaler": scaler,
        "cols": cols,
        "curvas": curvas,
        "medoids": medoids
    }

    dump(bundle, "modelo_predweem_meteo.joblib")

    st.success("Modelo entrenado con éxito ✔")
    st.download_button(
        "💾 Descargar modelo",
        data=open("modelo_predweem_meteo.joblib","rb").read(),
        file_name="modelo_predweem_meteo.joblib"
    )


# ===============================================================
# STREAMLIT UI — PREDICCIÓN
# ===============================================================

st.header("🔮 Predicción del Patrón")

modelo_file = st.file_uploader("📦 Modelo entrenado", type=["joblib"], key="mf")
meteo_new = st.file_uploader("🌦️ Meteorología nueva", type=["xlsx"], key="mn")

if st.button("🔍 PREDECIR"):
    if not (modelo_file and meteo_new):
        st.error("❌ Cargar modelo y meteorología")
        st.stop()

    bundle = load(modelo_file)
    clf = bundle["clf"]
    scaler = bundle["scaler"]
    cols = bundle["cols"]
    curvas_hist = bundle["curvas"]
    medoids = bundle["medoids"]

    metneo = cargar_meteo_xlsx(meteo_new)
    año_new = list(metneo.keys())[0]

    feats = features_meteo(metneo[año_new])

    X = np.array([[feats[c] for c in cols]], float)
    Xs = scaler.transform(X)

    proba = clf.predict_proba(Xs)[0]
    pred = int(np.argmax(proba))

    st.success(f"Patrón más probable: **C{pred}** con prob = {proba[pred]:.2f}")

    curva_proto = curvas_hist[medoids[pred]]
    dias = np.arange(1,366)

    dfp = pd.DataFrame({"Día": dias, "Valor": curva_proto})

    chart = alt.Chart(dfp).mark_line(color="blue").encode(
        x="Día:Q",
        y=alt.Y("Valor:Q", scale=alt.Scale(domain=[0,1]))
    ).properties(title=f"Patrón C{pred} (Medoid)")

    st.altair_chart(chart, use_container_width=True)


