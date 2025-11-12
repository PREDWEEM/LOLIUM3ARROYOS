
# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Mixture-of-Prototypes (DTW + Monotone + Comparación)
# ===============================================================
# - Entrenamiento (k-medoids DTW + GradientBoosting)
# - Predicción desde meteorología nueva
# - Comparación (RMSE/MAE) vs TODOS los escenarios (prototipos)
# - Ponderación opcional Feb–May en la similitud
# - Evaluación por año (RMSE, MAE) + gráfico
# - Monotonía garantizada (acumulado)
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import re, io, joblib
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="PREDWEEM — Mixture-of-Prototypes", layout="wide")
st.title("🌾 PREDWEEM — Mixture-of-Prototypes (DTW + Monotone + Comparación)")

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
        if k in df.columns: df = df.rename(columns={k:v})
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
    for c in ["tmin","tmax","prec","jd"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
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
    df = (df.set_index("jd").reindex(range(1, JD_MAX+1)).interpolate().ffill().bfill().reset_index())
    return df

def curva_desde_xlsx_anual(file) -> np.ndarray:
    df = pd.read_excel(file, header=None)
    if df.shape[1] < 2: df = pd.read_excel(file)
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
    for d,v in zip(jd,val):
        if pd.notna(d) and 1 <= int(d) <= 365:
            daily[int(d)-1] += float(v)
    if paso > 1:  # suaviza series semanales
        daily = np.convolve(daily, np.ones(7)/7, mode="same")

    acum = np.cumsum(daily)
    if np.nanmax(acum) == 0:
        return np.zeros(JD_MAX)
    curva = (acum / np.nanmax(acum))[:JD_MAX]
    return np.maximum.accumulate(curva)

def emerg_rel_7d_from_acum(y_acum: np.ndarray) -> np.ndarray:
    inc = np.diff(np.insert(y_acum, 0, 0.0))
    return np.convolve(inc, np.ones(7)/7, mode="same")

# ===============================================================
# FEATURES METEOROLÓGICAS (ROBUSTO)
# ===============================================================
def build_features_meteo(dfm: pd.DataFrame):
    dfm = standardize_cols(dfm)
    dfm = ensure_jd_1_to_274(dfm)
    tmin = dfm["tmin"].to_numpy(float)
    tmax = dfm["tmax"].to_numpy(float)
    tmed = (tmin + tmax) / 2.0
    prec = dfm["prec"].to_numpy(float)
    jd = dfm["jd"].to_numpy(int)

    mask_FM = (jd >= 32) & (jd <= 151)  # Feb–May
    gdd5 = np.cumsum(np.maximum(tmed - 5, 0))
    gdd3 = np.cumsum(np.maximum(tmed - 3, 0))

    def longest_run(x):
        c = m = 0
        for v in x:
            c = c + 1 if v == 1 else 0; m = max(m, c)
        return m

    def ma_series(x, w):
        return pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()

    # Safe stats (si FM vacío, usan todo el año)
    if not np.any(mask_FM): mask_FM = np.ones_like(jd, dtype=bool)
    pf = prec[mask_FM]; pf = pf if (pf.size>0 and not np.all(np.isnan(pf))) else np.zeros(1)

    f = {}
    f["gdd5_FM"]  = float(np.ptp(gdd5[mask_FM])) if np.any(~np.isnan(gdd5[mask_FM])) else 0.0
    f["gdd3_FM"]  = float(np.ptp(gdd3[mask_FM])) if np.any(~np.isnan(gdd3[mask_FM])) else 0.0
    f["pp_FM"]    = float(np.nansum(pf))
    f["ev10_FM"]  = int(np.nansum(pf >= 10))
    f["ev20_FM"]  = int(np.nansum(pf >= 20))
    dry           = np.nan_to_num(pf < 1, nan=0).astype(int)
    wet           = np.nan_to_num(pf >= 5, nan=0).astype(int)
    f["dry_run_FM"] = longest_run(dry)
    f["wet_run_FM"] = longest_run(wet)

    tmed14 = ma_series(tmed, 14)
    tmed28 = ma_series(tmed, 28)
    idx_may = np.clip(151, 0, len(tmed) - 1)
    f["tmed14_May"] = float(tmed14[idx_may])
    f["tmed28_May"] = float(tmed28[idx_may])

    idx_120 = min(119, len(tmed) - 1)
    f["gdd5_120"] = float(gdd5[idx_120])
    f["pp_120"]   = float(np.nansum(prec[: idx_120 + 1]))
    return dfm, f

# ===============================================================
# DTW + K-MEDOIDS
# ===============================================================
def dtw_distance(a, b):
    n, m = len(a), len(b)
    D = np.full((n+1, m+1), np.inf, dtype=float)
    D[0,0] = 0.0
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            cost = (ai - b[j-1])**2
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(np.sqrt(D[n,m]))

def k_medoids_dtw(curves, K, max_iter=50, seed=42):
    rng = np.random.default_rng(seed)
    N = len(curves)
    if K > N: K = N
    medoid_idx = list(rng.choice(N, size=K, replace=False))

    # Distancias DTW
    D = np.zeros((N, N), float)
    for i in range(N):
        for j in range(i+1, N):
            d = dtw_distance(curves[i], curves[j])
            D[i,j] = D[j,i] = d

    # Iterar
    for _ in range(max_iter):
        assign = np.argmin(D[:, medoid_idx], axis=1)
        new_medoids = []
        for k in range(K):
            members = np.where(assign == k)[0]
            if len(members) == 0:
                new_medoids.append(medoid_idx[k]); continue
            subD = D[np.ix_(members, members)]
            sums = subD.sum(axis=1)
            chosen = members[np.argmin(sums)]
            new_medoids.append(chosen)
        if new_medoids == medoid_idx: break
        medoid_idx = new_medoids

    return medoid_idx, D

# ===============================================================
# STREAMLIT — TABS
# ===============================================================
tabs = st.tabs(["🧪 Entrenar modelo", "🔮 Predecir & comparar", "📊 Evaluar"])

# ----------------------- TAB 1: ENTRENAMIENTO ------------------
with tabs[0]:
    st.subheader("🧪 Entrenamiento del modelo (k-medoids DTW + GB)")
    meteo_book = st.file_uploader("📘 Meteorología multianual (una hoja por año)", type=["xlsx","xls"])
    curvas_files = st.file_uploader("📈 Curvas históricas (XLSX por año)", type=["xlsx","xls"], accept_multiple_files=True)
    K = st.slider("Número de patrones (K)", 2, 6, 4, 1)
    seed = st.number_input("Semilla aleatoria", 0, 99999, 42)
    btn_train = st.button("🚀 Entrenar modelo")

    if btn_train:
        if not (meteo_book and curvas_files):
            st.error("Faltan archivos."); st.stop()

        # Meteorología por año
        sheets = pd.read_excel(meteo_book, sheet_name=None)
        meteo_dict = {}
        for name, df in sheets.items():
            df = standardize_cols(df); df = ensure_jd_1_to_274(df)
            try: year = int(re.findall(r"\d{4}", str(name))[0])
            except: continue
            if all(c in df.columns for c in ["tmin","tmax","prec"]):
                meteo_dict[year] = df[["jd","tmin","tmax","prec"]].copy()

        # Curvas por año
        curves_by_year = {}
        for f in curvas_files:
            y4 = re.findall(r"(\d{4})", f.name)
            if not y4: continue
            year = int(y4[0])
            curves_by_year[year] = curva_desde_xlsx_anual(f)

        common = sorted(set(meteo_dict) & set(curves_by_year))
        if len(common) < 3:
            st.error("Muy pocos años comunes (≥3 recomendado)."); st.stop()

        curves = [curves_by_year[y] for y in common]
        medoid_idx, D = k_medoids_dtw(curves, K=K, seed=seed)
        protos = [curves[i] for i in medoid_idx]

        # Features + etiquetas de cluster (asignación por distancia)
        feat_names, feats, labels = None, [], []
        assign = np.argmin(D[:, medoid_idx], axis=1)  # 0..K-1
        for i, y in enumerate(common):
            _, f = build_features_meteo(meteo_dict[y])
            if feat_names is None: feat_names = sorted(f.keys())
            feats.append([f[k] for k in feat_names])
            labels.append(int(assign[i]))
        X = np.array(feats, float); y_lab = np.array(labels, int)

        # Clasificador patrón
        xsc = StandardScaler().fit(X); Xs = xsc.transform(X)
        clf = GradientBoostingClassifier(random_state=seed).fit(Xs, y_lab)

        # Warps por cluster
        regs_shift, regs_scale = {}, {}
        t = np.arange(1, JD_MAX+1, dtype=float)
        def warp(proto, sh, sc):
            tp = (t - sh)/max(sc, 1e-6); tp = np.clip(tp, 1, JD_MAX)
            yv = np.interp(tp, np.arange(1, JD_MAX+1, dtype=float), proto)
            return np.maximum.accumulate(np.clip(yv, 0, 1))

        for k in range(K):
            idxs = np.where(y_lab == k)[0]
            shifts, scales, Xk = [], [], []
            for i in idxs:
                curv = curves[i]
                best = (0, 1, 1e9)
                for sh in range(-20, 21, 5):
                    for sc in [0.9, 0.95, 1.0, 1.05, 1.1]:
                        cand = warp(protos[k], sh, sc)
                        rmse = float(np.sqrt(np.mean((cand - curv)**2)))
                        if rmse < best[2]: best = (float(sh), float(sc), rmse)
                shifts.append(best[0]); scales.append(best[1]); Xk.append(Xs[i])
            if Xk:
                Xk = np.vstack(Xk)
                regs_shift[k] = GradientBoostingRegressor(random_state=seed).fit(Xk, np.array(shifts))
                regs_scale[k] = GradientBoostingRegressor(random_state=seed).fit(Xk, np.array(scales))

        # Bundle + descarga
        bundle = {"xsc": xsc, "feat_names": feat_names, "clf": clf,
                  "protos": np.vstack(protos),
                  "regs_shift": regs_shift, "regs_scale": regs_scale}
        st.session_state["bundle"] = bundle

        buf = io.BytesIO(); joblib.dump(bundle, buf)
        st.download_button("💾 Descargar modelo (.joblib)", data=buf.getvalue(),
                           file_name=f"predweem_mixture_dtw_K{K}.joblib",
                           mime="application/octet-stream")

        # Plot prototipos
        dias = np.arange(1, JD_MAX+1)
        dfp = pd.concat([pd.DataFrame({"Día":dias,"Valor":protos[i],"Serie":f"Proto {i}"}) for i in range(K)])
        chart = alt.Chart(dfp).mark_line().encode(
            x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE))),
            y=alt.Y("Valor:Q", title="Emergencia acumulada (0–1)", scale=alt.Scale(domain=[0,1])),
            color="Serie:N"
        ).properties(height=420, title="Prototipos (medoids DTW)")
        st.altair_chart(chart, use_container_width=True)

# ----------------------- TAB 2: PREDICCIÓN ---------------------
with tabs[1]:
    st.subheader("🔮 Predicción y comparación con escenarios")
    modelo_file = st.file_uploader("📦 Modelo (.joblib)", type=["joblib"])
    meteo_file  = st.file_uploader("📘 Meteorología nueva (XLSX)", type=["xlsx","xls"])
    peso_feb_may = st.slider("Peso Feb–May en similitud", 1.0, 4.0, 2.0, 0.5,
                             help=">1 aumenta la importancia de la ventana crítica (JD 32–151) en la comparación.")
    btn_pred = st.button("🚀 Predecir")

    if btn_pred:
        if not (modelo_file and meteo_file):
            st.error("Faltan archivos."); st.stop()
        bundle = joblib.load(modelo_file)
        xsc = bundle["xsc"]; feat_names = bundle["feat_names"]; clf = bundle["clf"]
        protos = bundle["protos"]; regs_shift = bundle["regs_shift"]; regs_scale = bundle["regs_scale"]
        K = protos.shape[0]

        dfm = pd.read_excel(meteo_file)
        dfm, f = build_features_meteo(dfm)
        X = np.array([[f[k] for k in sorted(feat_names)]], float)
        Xs = xsc.transform(X)

        proba = clf.predict_proba(Xs)[0]; k_hat = int(np.argmax(proba))
        shift = float(regs_shift.get(k_hat).predict(Xs)[0]) if k_hat in regs_shift else 0.0
        scale = float(regs_scale.get(k_hat).predict(Xs)[0]) if k_hat in regs_scale else 1.0
        scale = float(np.clip(scale, 0.9, 1.1))

        t = np.arange(1, JD_MAX+1, dtype=float)
        def warp(proto, sh, sc):
            tp = (t - sh)/max(sc, 1e-6); tp = np.clip(tp, 1, JD_MAX)
            yv = np.interp(tp, np.arange(1, JD_MAX+1, dtype=float), proto)
            return np.maximum.accumulate(np.clip(yv, 0, 1))

        mix = np.zeros(JD_MAX, float)
        for k in range(K):
            yk = warp(protos[k], shift if k==k_hat else 0.0, scale if k==k_hat else 1.0)
            mix += float(proba[k]) * yk
        mix = np.maximum.accumulate(np.clip(mix, 0, 1))

        dias = np.arange(1, JD_MAX+1)
        df_pred = pd.DataFrame({"Día":dias, "Emergencia_predicha": mix,
                                "Emergencia_relativa_7d": emerg_rel_7d_from_acum(mix)})
        st.download_button("⬇️ Descarga curva predicha (CSV)",
                           df_pred.to_csv(index=False).encode("utf-8"),
                           file_name="curva_predicha.csv", mime="text/csv")

        st.altair_chart(
            alt.Chart(df_pred).mark_line(color="#e67300", strokeWidth=2.5).encode(
                x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE))),
                y=alt.Y("Emergencia_predicha:Q", title="Emergencia acumulada (0–1)", scale=alt.Scale(domain=[0,1]))
            ).properties(height=420, title=f"Predicción (C{k_hat} • conf {proba.max():.2f} • shift {shift:+.1f} • scale {scale:.3f})"),
            use_container_width=True
        )

        # ---- Comparación con TODOS los escenarios (ponderación Feb–May) ----
        mask_FM = (dias >= 32) & (dias <= 151)
        w = np.ones_like(dias, dtype=float)
        w[mask_FM] = peso_feb_may

        comps = []
        for k in range(K):
            err = (mix - protos[k])**2
            rmse_w = float(np.sqrt(np.sum(err * w) / np.sum(w)))
            mae_w  = float(np.sum(np.abs(mix - protos[k]) * w) / np.sum(w))
            comps.append((k, rmse_w, mae_w, float(proba[k])))

        df_cmp = pd.DataFrame(comps, columns=["Cluster","RMSE_pond","MAE_pond","Probabilidad"])
        df_cmp["Similitud_%"] = 100 * (1 - df_cmp["RMSE_pond"] / df_cmp["RMSE_pond"].max().clip(min=1e-9))
        st.markdown("### 🔁 Comparación (ponderada Feb–May)")
        st.dataframe(df_cmp.sort_values("RMSE_pond"), use_container_width=True)

        st.download_button("⬇️ Descarga comparación (CSV)",
                           df_cmp.to_csv(index=False).encode("utf-8"),
                           file_name="comparacion_escenarios.csv", mime="text/csv")

        dfp = pd.concat(
            [pd.DataFrame({"Día":dias,"Valor":protos[i],"Serie":f"Escenario {i}"}) for i in range(K)]
            + [pd.DataFrame({"Día":dias,"Valor":mix,"Serie":"Predicción"})]
        )
        chart_cmp = alt.Chart(dfp).mark_line().encode(
            x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE))),
            y=alt.Y("Valor:Q", title="Emergencia acumulada (0–1)", scale=alt.Scale(domain=[0,1])),
            color="Serie:N"
        ).properties(height=440, title="Predicción vs escenarios históricos")
        st.altair_chart(chart_cmp, use_container_width=True)

# ----------------------- TAB 3: EVALUACIÓN ---------------------
with tabs[2]:
    st.subheader("📊 Evaluación histórica (holdout por año)")
    curvas_eval = st.file_uploader("📈 Curvas históricas (XLSX por año)", type=["xlsx","xls"], accept_multiple_files=True)
    meteo_book_eval = st.file_uploader("📘 Meteorología multianual (XLSX)", type=["xlsx","xls"])
    modelo_eval = st.file_uploader("📦 Modelo (.joblib)", type=["joblib"])
    btn_eval = st.button("🔎 Evaluar")

    if btn_eval:
        if not (curvas_eval and meteo_book_eval and modelo_eval):
            st.error("Faltan archivos."); st.stop()

        bundle = joblib.load(modelo_eval)
        xsc = bundle["xsc"]; feat_names = bundle["feat_names"]; clf = bundle["clf"]
        protos = bundle["protos"]; regs_shift = bundle["regs_shift"]; regs_scale = bundle["regs_scale"]
        K = protos.shape[0]

        sheets = pd.read_excel(meteo_book_eval, sheet_name=None)
        meteo_dict = {}
        for name, df in sheets.items():
            df = standardize_cols(df); df = ensure_jd_1_to_274(df)
            try: year = int(re.findall(r"\d{4}", str(name))[0])
            except: continue
            if all(c in df.columns for c in ["tmin","tmax","prec"]):
                meteo_dict[year] = df[["jd","tmin","tmax","prec"]].copy()

        curves_eval = {}
        for f in curvas_eval:
            y4 = re.findall(r"(\d{4})", f.name)
            if not y4: continue
            year = int(y4[0])
            curves_eval[year] = curva_desde_xlsx_anual(f)

        common = sorted(set(meteo_dict) & set(curves_eval))
        if not common:
            st.error("No hay años en común."); st.stop()

        dias = np.arange(1, JD_MAX+1)
        rows=[]
        t = np.arange(1, JD_MAX+1, dtype=float)
        def warp(proto, sh, sc):
            tp = (t - sh)/max(sc, 1e-6); tp = np.clip(tp, 1, JD_MAX)
            yv = np.interp(tp, np.arange(1, JD_MAX+1, dtype=float), proto)
            return np.maximum.accumulate(np.clip(yv, 0, 1))

        for y in common:
            dfm, f = build_features_meteo(meteo_dict[y])
            X = np.array([[f[k] for k in sorted(feat_names)]], float)
            Xs = xsc.transform(X)
            proba = clf.predict_proba(Xs)[0]; k_hat = int(np.argmax(proba))
            shift = float(regs_shift.get(k_hat).predict(Xs)[0]) if k_hat in regs_shift else 0.0
            scale = float(regs_scale.get(k_hat).predict(Xs)[0]) if k_hat in regs_scale else 1.0
            scale = float(np.clip(scale, 0.9, 1.1))

            mix = np.zeros(JD_MAX, float)
            for k in range(K):
                yk = warp(protos[k], shift if k==k_hat else 0.0, scale if k==k_hat else 1.0)
                mix += float(proba[k]) * yk
            mix = np.maximum.accumulate(np.clip(mix, 0, 1))

            y_true = curves_eval[y]
            rmse = float(np.sqrt(np.mean((y_true - mix)**2)))
            mae  = float(np.mean(np.abs(y_true - mix)))
            rows.append((int(y), rmse, mae, k_hat, float(proba.max()), shift, scale))

        dfm_eval = pd.DataFrame(rows, columns=["Año","RMSE","MAE","Cluster","Conf","Shift_d","Scale"]).sort_values("Año")
        st.dataframe(dfm_eval, use_container_width=True)

        yopt = st.selectbox("Ver año:", options=[int(y) for y in common])
        # recomputo para graficar
        dfm, f = build_features_meteo(meteo_dict[yopt])
        X = np.array([[f[k] for k in sorted(feat_names)]], float); Xs = xsc.transform(X)
        proba = clf.predict_proba(Xs)[0]; k_hat = int(np.argmax(proba))
        shift = float(regs_shift.get(k_hat).predict(Xs)[0]) if k_hat in regs_shift else 0.0
        scale = float(regs_scale.get(k_hat).predict(Xs)[0]) if k_hat in regs_scale else 1.0
        scale = float(np.clip(scale, 0.9, 1.1))

        def warp_g(proto): return warp(proto, shift, scale)
        mix = np.zeros(JD_MAX, float)
        for k in range(K):
            yk = warp(protos[k]) if k==k_hat else protos[k]
            mix += float(proba[k]) * yk
        mix = np.maximum.accumulate(np.clip(mix, 0, 1))

        df_plot = pd.DataFrame({"Día":dias,"Emergencia real":curves_eval[yopt],"Emergencia predicha":mix}).melt("Día","Serie","Valor")
        chart = alt.Chart(df_plot).mark_line().encode(
            x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE))),
            y=alt.Y("Valor:Q", title="Emergencia acumulada (0–1)", scale=alt.Scale(domain=[0,1])),
            color="Serie:N"
        ).properties(height=420, title=f"Detalle {yopt} (C{k_hat} • conf {proba.max():.2f} • shift {shift:+.1f} • scale {scale:.3f})")
        st.altair_chart(chart, use_container_width=True)

