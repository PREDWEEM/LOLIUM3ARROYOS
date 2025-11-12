# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Mixture-of-Prototypes (DTW + Monotone)
# Versión extendida con comparación de escenarios
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import re, io, joblib
from io import BytesIO
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="PREDWEEM — Mixture-of-Prototypes (DTW + Comparación)", layout="wide")
st.title("🌾 PREDWEEM — Mixture-of-Prototypes (DTW + Monotone + Comparación de Escenarios)")

JD_MAX = 274
XRANGE = (1, JD_MAX)

# ===============================================================
# UTILIDADES
# ===============================================================
def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    ren = {
        "temperatura minima":"tmin","t_min":"tmin","t min":"tmin","mínima":"tmin","min":"tmin",
        "temperatura maxima":"tmax","t_max":"tmax","t max":"tmax","máxima":"tmax","max":"tmax",
        "precipitacion":"prec","precip":"prec","pp":"prec","lluvia":"prec","rain":"prec",
        "dia juliano":"jd","día juliano":"jd","julian_days":"jd","dia":"jd","día":"jd",
        "fecha":"fecha","date":"fecha"
    }
    for k,v in ren.items():
        if k in df.columns:
            df = df.rename(columns={k:v})
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
    for c in ["tmin","tmax","prec","jd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def ensure_jd_1_to_274(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "jd" not in df.columns:
        if "fecha" in df.columns and df["fecha"].notna().any():
            y0 = int(df["fecha"].dt.year.mode().iloc[0])
            df = df[(df["fecha"] >= f"{y0}-01-01") & (df["fecha"] <= f"{y0}-10-01")].copy().sort_values("fecha")
            df["jd"] = df["fecha"].dt.dayofyear - pd.Timestamp(f"{y0}-01-01").dayofyear + 1
        else:
            df["jd"] = np.arange(1, len(df) + 1)
    df = (df.set_index("jd")
            .reindex(range(1, JD_MAX+1))
            .interpolate()
            .ffill().bfill()
            .reset_index())
    return df

def curva_desde_xlsx_anual(file) -> np.ndarray:
    df = pd.read_excel(file, header=None)
    if df.shape[1] < 2:
        df = pd.read_excel(file)
    col0 = pd.to_numeric(df.iloc[:,0], errors="coerce")
    col1 = pd.to_numeric(df.iloc[:,1], errors="coerce")

    if col0.isna().mean() > 0.5:
        fch = pd.to_datetime(df.iloc[:,0], errors="coerce", dayfirst=True)
        jd  = fch.dt.dayofyear
        val = pd.to_numeric(df.iloc[:,1], errors="coerce").fillna(0.0)
    else:
        jd  = col0.astype("Int64")
        val = col1.fillna(0.0)

    jd_clean = jd.dropna().astype(int).sort_values().unique()
    paso = int(np.median(np.diff(jd_clean))) if len(jd_clean)>1 else 7

    daily = np.zeros(365, dtype=float)
    if paso == 1:
        for d,v in zip(jd,val):
            if pd.notna(d) and 1 <= int(d) <= 365:
                daily[int(d)-1] += float(v)
    else:
        for d,v in zip(jd,val):
            if pd.notna(d) and 1 <= int(d) <= 365:
                daily[int(d)-1] += float(v)
        daily = np.convolve(daily, np.ones(7)/7, mode="same")

    acum = np.cumsum(daily)
    if np.nanmax(acum) == 0:
        return np.zeros(JD_MAX, dtype=float)
    curva = (acum / np.nanmax(acum))[:JD_MAX]
    return np.maximum.accumulate(curva)

def emerg_rel_7d_from_acum(y_acum: np.ndarray) -> np.ndarray:
    inc = np.diff(np.insert(y_acum, 0, 0.0))
    return np.convolve(inc, np.ones(7)/7, mode="same")

# ===============================================================
# BLOQUE ROBUSTO DE FEATURES METEOROLÓGICAS
# ===============================================================
def build_features_meteo(dfm: pd.DataFrame):
    dfm = standardize_cols(dfm)
    dfm = ensure_jd_1_to_274(dfm)
    tmin = dfm["tmin"].to_numpy(float)
    tmax = dfm["tmax"].to_numpy(float)
    tmed = (tmin + tmax) / 2.0
    prec = dfm["prec"].to_numpy(float)
    jd = dfm["jd"].to_numpy(int)

    mask_FM = (jd >= 32) & (jd <= 151)  # Feb–May aprox
    gdd5 = np.cumsum(np.maximum(tmed - 5, 0))
    gdd3 = np.cumsum(np.maximum(tmed - 3, 0))

    if not np.any(mask_FM):
        mask_FM = np.ones_like(jd, dtype=bool)

    safe = lambda arr, func=np.ptp: float(func(arr[mask_FM])) if np.any(~np.isnan(arr[mask_FM])) else 0.0
    f = {}
    f["gdd5_FM"] = safe(gdd5)
    f["gdd3_FM"] = safe(gdd3)

    pf = prec[mask_FM]
    if pf.size == 0 or np.all(np.isnan(pf)):
        pf = np.zeros(1)
    f["pp_FM"] = np.nansum(pf)
    f["ev10_FM"] = int(np.nansum(pf >= 10))
    f["ev20_FM"] = int(np.nansum(pf >= 20))

    dry = np.nan_to_num(pf < 1, nan=0).astype(int)
    wet = np.nan_to_num(pf >= 5, nan=0).astype(int)

    def longest_run(x):
        c = m = 0
        for v in x:
            c = c + 1 if v == 1 else 0
            m = max(m, c)
        return m
    f["dry_run_FM"] = longest_run(dry)
    f["wet_run_FM"] = longest_run(wet)

    def ma(x, w):
        k = np.ones(w) / w
        return np.convolve(x, k, "same")

    idx_may = np.clip(151, 0, len(tmed) - 1)
    f["tmed14_May"] = float(ma(tmed, 14)[idx_may])
    f["tmed28_May"] = float(ma(tmed, 28)[idx_may])

    idx_120 = min(119, len(tmed) - 1)
    f["gdd5_120"] = float(gdd5[idx_120])
    f["pp_120"] = float(np.nansum(prec[: idx_120 + 1]))

    return dfm, f

# ===============================================================
# DTW + K-MEDOIDS
# ===============================================================
def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    D = np.full((n+1, m+1), np.inf, dtype=float)
    D[0,0] = 0.0
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            cost = (ai - b[j-1])**2
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(np.sqrt(D[n,m]))

def k_medoids_dtw(curves: list, K: int, max_iter: int = 50, seed: int = 42):
    rng = np.random.default_rng(seed)
    N = len(curves)
    if K > N: K = N
    idx = rng.choice(N, size=K, replace=False)
    medoid_idx = list(idx)
    D = np.zeros((N,N), float)
    for i in range(N):
        for j in range(i+1, N):
            d = dtw_distance(curves[i], curves[j])
            D[i,j] = D[j,i] = d
    for _ in range(max_iter):
        assign = np.argmin(D[:, medoid_idx], axis=1)
        new_medoids = []
        for k in range(K):
            members = np.where(assign==k)[0]
            if len(members)==0:
                new_medoids.append(medoid_idx[k]); continue
            subD = D[np.ix_(members, members)]
            sums = subD.sum(axis=1)
            chosen = members[np.argmin(sums)]
            new_medoids.append(chosen)
        if new_medoids == medoid_idx: break
        medoid_idx = new_medoids
    clusters = {k: [] for k in range(K)}
    assign = np.argmin(D[:, medoid_idx], axis=1)
    for i in range(N): clusters[int(assign[i])].append(i)
    return medoid_idx, clusters, D

# ===============================================================
# STREAMLIT TABS
# ===============================================================
tabs = st.tabs(["🧪 Entrenar prototipos", "🔮 Predecir nueva curva", "📊 Evaluar"])

# ===============================================================
# TAB 2 — PREDICCIÓN
# ===============================================================
with tabs[1]:
    st.subheader("🔮 Predicción y comparación con escenarios históricos")
    modelo_file = st.file_uploader("📦 Modelo (predweem_mixture_dtw.joblib)", type=["joblib"])
    meteo_file  = st.file_uploader("📘 Meteorología nueva (XLSX)", type=["xlsx","xls"])
    btn_pred = st.button("🚀 Predecir")

    if btn_pred:
        if not (modelo_file and meteo_file):
            st.error("Cargá el modelo y la meteo.")
            st.stop()

        bundle = joblib.load(modelo_file)
        xsc = bundle["xsc"]; feat_names = bundle["feat_names"]; clf = bundle["clf"]
        protos = bundle["protos"]; regs_shift = bundle["regs_shift"]; regs_scale = bundle["regs_scale"]
        K = protos.shape[0]

        dfm = pd.read_excel(meteo_file)
        dfm, f = build_features_meteo(dfm)
        X = np.array([[f[k] for k in sorted(feat_names)]], float)
        Xs = xsc.transform(X)

        proba = clf.predict_proba(Xs)[0]
        k_hat = int(np.argmax(proba))
        shift = float(regs_shift[k_hat].predict(Xs)[0]) if k_hat in regs_shift else 0.0
        scale = float(regs_scale[k_hat].predict(Xs)[0]) if k_hat in regs_scale else 1.0
        scale = float(np.clip(scale, 0.9, 1.1))

        t = np.arange(1, JD_MAX+1, dtype=float)
        def warp_curve(proto, sh, sc):
            tp = (t - sh)/max(sc, 1e-6)
            tp = np.clip(tp, 1, JD_MAX)
            yv = np.interp(tp, np.arange(1, JD_MAX+1, dtype=float), proto)
            return np.maximum.accumulate(np.clip(yv, 0, 1))

        mix = np.zeros(JD_MAX, float)
        for k in range(K):
            yk = warp_curve(protos[k], shift if k==k_hat else 0.0, scale if k==k_hat else 1.0)
            mix += float(proba[k]) * yk
        mix = np.maximum.accumulate(np.clip(mix, 0, 1))

        dias = np.arange(1, JD_MAX+1)
        chart = alt.Chart(pd.DataFrame({"Día":dias,"Pred":mix})).mark_line(color="orange", strokeWidth=3).encode(
            x="Día", y=alt.Y("Pred", title="Emergencia acumulada (0–1)")
        ).properties(height=400, title=f"Curva predicha (C{k_hat}, conf {proba.max():.2f})")
        st.altair_chart(chart, use_container_width=True)

        # ===========================================================
        # 🔁 Comparar con todos los escenarios posibles
        # ===========================================================
        comparaciones = []
        for k in range(K):
            proto = protos[k]
            rmse = float(np.sqrt(np.mean((mix - proto)**2)))
            mae = float(np.mean(np.abs(mix - proto)))
            comparaciones.append((k, rmse, mae, float(proba[k])))
        df_cmp = pd.DataFrame(comparaciones, columns=["Cluster","RMSE_vs_pred","MAE_vs_pred","Probabilidad"])
        df_cmp["Similitud (%)"] = 100 * (1 - df_cmp["RMSE_vs_pred"]/df_cmp["RMSE_vs_pred"].max())
        st.markdown("### 🔁 Comparación de la curva predicha vs escenarios históricos")
        st.dataframe(df_cmp.sort_values("RMSE_vs_pred"), use_container_width=True)

        # Mostrar todas las curvas
        dfp = []
        for k in range(K):
            dfp.append(pd.DataFrame({"Día":dias,"Valor":protos[k],"Serie":f"Escenario {k}"}))
        dfp.append(pd.DataFrame({"Día":dias,"Valor":mix,"Serie":"Predicción"}))
        dfp = pd.concat(dfp)
        chart_cmp = alt.Chart(dfp).mark_line().encode(
            x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE))),
            y=alt.Y("Valor:Q", title="Emergencia acumulada (0–1)", scale=alt.Scale(domain=[0,1])),
            color="Serie:N"
        ).properties(height=420, title="Predicción vs escenarios históricos")
        st.altair_chart(chart_cmp, use_container_width=True)








