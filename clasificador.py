# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM–METEO v1.1 — Streamlit App (con FIX robusto)
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

st.set_page_config(page_title="🌾 PREDWEEM–METEO v1.1", layout="wide")
st.title("🌾 PREDWEEM–METEO v1.1 — Predicción del patrón histórico desde meteorología (robusto)")


# ===============================================================
# 1) FUNCIONES GENERALES
# ===============================================================

def cargar_curva_acumulada(file):
    """Lee XLSX (día, valor) y normaliza a emergencia acumulada 0–1."""
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


# ===============================================================
# 2) FIX: CARGA ROBUSTA DE METEOROLOGÍA
# ===============================================================

def cargar_meteo_xlsx(file):
    book = pd.read_excel(file, sheet_name=None)
    out = {}
    problemas = []

    JD_VARIANTS = [
        "jd", "JD", "Julian_days", "Julian_Day",
        "día juliano", "dia juliano", "Día juliano",
        "DiaJuliano", "Day", "dia", "Dia"
    ]

    for name, df in book.items():

        if not isinstance(df, pd.DataFrame):
            continue

        # Detectar año desde el nombre de la pestaña
        try:
            year = int(re.findall(r"\d{4}", str(name))[0])
        except:
            continue

        # Normalizar columnas
        df.columns = [str(c).strip() for c in df.columns]

        # === DETECTAR COLUMNA JD ===
        jd_col = None
        for cand in JD_VARIANTS:
            for col in df.columns:
                if col.lower() == cand.lower():
                    jd_col = col
                    break
            if jd_col:
                break

        # Si no existe JD → intentar detectar desde columna fecha
        if jd_col is None:
            date_cols = [c for c in df.columns if "fec" in c.lower() or "date" in c.lower()]
            if date_cols:
                try:
                    fecha = pd.to_datetime(df[date_cols[0]], errors="coerce", dayfirst=True)
                    df["jd"] = fecha.dt.dayofyear
                    jd_col = "jd"
                except:
                    problemas.append(name)
                    continue
            else:
                problemas.append(name)
                continue

        # Convertir JD
        df["jd"] = pd.to_numeric(df[jd_col], errors="coerce")
        df = df.dropna(subset=["jd"])
        df = df[(df["jd"] >= 1) & (df["jd"] <= 274)]
        if df.empty:
            problemas.append(name)
            continue

        # === DETECTAR TMIN, TMAX, PREC con múltiples variantes ===
        def find_col(df, names):
            for n in names:
                for c in df.columns:
                    if c.lower() == n.lower():
                        return c
            return None

        col_tmin = find_col(df, ["tmin","TMIN","min","mínima"])
        col_tmax = find_col(df, ["tmax","TMAX","max","máxima"])
        col_prec = find_col(df, ["prec","pp","rain","lluvia"])

        if not (col_tmin and col_tmax and col_prec):
            problemas.append(name)
            continue

        # Normalizar nombres
        df = df.rename(columns={
            col_tmin: "tmin",
            col_tmax: "tmax",
            col_prec: "prec"
        })

        # A numérico
        for c in ["tmin","tmax","prec"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["tmin","tmax","prec"])
        if df.empty:
            problemas.append(name)
            continue

        df = df.sort_values("jd").reset_index(drop=True)

        out[year] = df

    # Reportar pestañas descartadas
    if problemas:
        st.warning(f"Pestañas ignoradas (sin datos válidos): {problemas}")

    return out


# ===============================================================
# 3) FEATURES
# ===============================================================

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
# 4) DTW + K-MEDOIDS
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
            dm = [dtw(curvas[i], curvas[m]) for m in medoids]
            k = np.argmin(dm)
            clusters[k].append(i)

        new = []
        for k,lista in clusters.items():
            if not lista:
                new.append(medoids[k])
                continue
            subD = np.zeros((len(lista),len(lista)))
            for i,p in enumerate(lista):
                for j,q in enumerate(lista):
                    subD[i,j] = dtw(curvas[p], curvas[q])
            new.append(lista[np.argmin(subD.sum(axis=1))])

        if np.all(new == medoids):
            break
        medoids = new

    return medoids, clusters


# ===============================================================
# 5) ENTRENAMIENTO
# ===============================================================

def entrenar_modelo(curvas, meteo_dict, años, K=3):

    Fcurva = pd.DataFrame([features_curva(c) for c in curvas])
    Fmeteo = pd.DataFrame([features_meteo(meteo_dict[y]) for y in años])

    medoids, clusters = k_medoids(curvas, K)

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
# 6) PREDICCIÓN
# ===============================================================

def predecir_patron(df_meteo, clf, scaler, cols):
    f = features_meteo(df_meteo)
    X = pd.DataFrame([f])[cols]
    Xs = scaler.transform(X)
    proba = clf.predict_proba(Xs)[0]
    k = int(np.argmax(proba))
    return k, proba


# ===============================================================
# 7) INTERFAZ STREAMLIT
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

    años = []
    curvas = []

    for f in curvas_files:
        y = int(re.findall(r"\d{4}", f.name)[0])
        if y not in meteo_dict:
            st.warning(f"⚠ Año {y}: no hay meteorología asociada → ignorado")
            continue
        curva = cargar_curva_acumulada(f)
        años.append(y)
        curvas.append(curva)

    clf, scaler, cols, medoids, clusters = entrenar_modelo(curvas, meteo_dict, años, K)

    st.success("Modelo entrenado correctamente ✔")

    # Exportar modelo
    buf = io.BytesIO()
    dump({"clf": clf, "scaler": scaler, "cols": cols, "medoids": medoids, "clusters": clusters, "años": años}, buf)

    st.download_button(
        "💾 Descargar modelo entrenado (.joblib)",
        data=buf.getvalue(),
        file_name="predweem_meteo_v1_1.joblib"
    )


# ===============================================================
# 8) PREDICCIÓN NUEVA
# ===============================================================

st.header("🔮 Predicción del patrón desde meteorología nueva")

modelo_file = st.file_uploader("Cargar modelo entrenado (.joblib)", type=["joblib"], key="mod")
meteo_nueva_file = st.file_uploader("Cargar meteorología NUEVA (XLSX)", type=["xlsx"], key="newmet")

if st.button("🔍 Predecir patrón"):
    if not (modelo_file and meteo_nueva_file):
        st.error("Faltan archivos")
        st.stop()

    M = load(modelo_file)
    clf = M["clf"]
    scaler = M["scaler"]
    cols = M["cols"]

    # Cargar SOLO la primera pestaña de la meteo nueva
    dfm = list(pd.read_excel(meteo_nueva_file, sheet_name=None).values())[0]

    # NORMALIZACIÓN por si faltan columnas
    dfm.columns = [str(c).strip() for c in dfm.columns]
    dfm = dfm.rename(columns={
        "TMIN":"tmin","TMAX":"tmax","Prec":"prec",
        "Julian_days":"jd","Día juliano":"jd"
    })
    dfm["jd"] = pd.to_numeric(dfm["jd"], errors="coerce")
    dfm = dfm.dropna(subset=["jd"])
    dfm = dfm.sort_values("jd").reset_index(drop=True)

    # Predecir
    k, proba = predecir_patron(dfm, clf, scaler, cols)

    st.subheader(f"🎯 Patrón más probable: **C{k}**")
    st.write("Probabilidades estimadas:")
    for i,p in enumerate(proba):
        st.write(f"- C{i}: {p:.3f}")



