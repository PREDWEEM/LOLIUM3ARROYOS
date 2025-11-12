# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Mixture-of-Prototypes (DTW + forma monótona)
# - Aprende K prototipos (medoids) con k-medoids bajo DTW (sin libs extra)
# - Clasifica patrón desde meteo (GradientBoostingClassifier)
# - Genera curva como mezcla convexa de prototipos + pequeño warp
# - Garantiza monotonía acumulando incrementos >= 0
# - Trabaja en JD 1..274 (1-ene → 1-oct)
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

st.set_page_config(page_title="PREDWEEM — Mixture-of-Prototypes (DTW)", layout="wide")
st.title("🌾 PREDWEEM — Mixture-of-Prototypes (DTW + monotone) • JD 1..274")

JD_MAX = 274
XRANGE = (1, JD_MAX)

# ---------------------------------------------------------------
# Utilidades genéricas
# ---------------------------------------------------------------
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
    # Lee [día/fecha, valor] (diario o semanal) → curva acumulada 0..1 (JD 1..274)
    df = pd.read_excel(file, header=None)
    if df.shape[1] < 2:
        df = pd.read_excel(file)
    col0 = pd.to_numeric(df.iloc[:,0], errors="coerce")
    col1 = pd.to_numeric(df.iloc[:,1], errors="coerce")
    if col0.isna().mean() > 0.5:
        # col0 parece fecha
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
    # Asegurar estricta monotonía (acumulado no decreciente)
    return np.maximum.accumulate(curva)

def emerg_rel_7d_from_acum(y_acum: np.ndarray) -> np.ndarray:
    inc = np.diff(np.insert(y_acum, 0, 0.0))
    return np.convolve(inc, np.ones(7)/7, mode="same")

# ---------------------------------------------------------------
# DTW y k-medoids (sin dependencias externas)
# ---------------------------------------------------------------
def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Simple DTW L2 (O(n^2)) — suficiente para 274 puntos y decenas de años
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
    idx = rng.choice(N, size=K, replace=False)
    medoid_idx = list(idx)
    # Pre-compute distance matrix (N x N) — N suele ser chico (años)
    D = np.zeros((N,N), float)
    for i in range(N):
        for j in range(i+1, N):
            d = dtw_distance(curves[i], curves[j])
            D[i,j] = D[j,i] = d

    for _ in range(max_iter):
        # Asignar a medoid más cercano
        assign = np.argmin(D[:, medoid_idx], axis=1)
        # Recalcular medoids por cluster
        new_medoids = []
        for k in range(K):
            members = np.where(assign==k)[0]
            if len(members)==0:
                new_medoids.append(medoid_idx[k])
                continue
            # medoid = argmin suma de distancias dentro del cluster
            subD = D[np.ix_(members, members)]
            sums = subD.sum(axis=1)
            chosen = members[np.argmin(sums)]
            new_medoids.append(chosen)
        if new_medoids == medoid_idx:
            break
        medoid_idx = new_medoids
    # Construir salida
    clusters = {k: [] for k in range(K)}
    assign = np.argmin(D[:, medoid_idx], axis=1)
    for i in range(N): clusters[int(assign[i])].append(i)
    return medoid_idx, clusters, D

# ---------------------------------------------------------------
# Features meteorológicas → patrón
# ---------------------------------------------------------------
def build_features_meteo(dfm: pd.DataFrame):
    dfm = standardize_cols(dfm)
    dfm = ensure_jd_1_to_274(dfm)
    tmin = dfm["tmin"].to_numpy(float)
    tmax = dfm["tmax"].to_numpy(float)
    tmed = (tmin + tmax)/2.0
    prec = dfm["prec"].to_numpy(float)
    jd = dfm["jd"].to_numpy(int)

    mask_FM = (jd>=32) & (jd<=151)  # Feb–May aprox
    gdd5 = np.cumsum(np.maximum(tmed-5,0))
    gdd3 = np.cumsum(np.maximum(tmed-3,0))
    f = {}
    f["gdd5_FM"] = gdd5[mask_FM].ptp()
    f["gdd3_FM"] = gdd3[mask_FM].ptp()
    pf = prec[mask_FM]
    f["pp_FM"] = pf.sum()
    f["ev10_FM"] = int((pf>=10).sum())
    f["ev20_FM"] = int((pf>=20).sum())
    dry = (pf<1).astype(int); wet=(pf>=5).astype(int)
    # rachas
    def longest_run(x):
        c=m=0
        for v in x: c = c+1 if v==1 else 0; m=max(m,c)
        return m
    f["dry_run_FM"] = longest_run(dry)
    f["wet_run_FM"] = longest_run(wet)
    # ventanas móviles
    def ma(x,w): k=np.ones(w)/w; return np.convolve(x,k,"same")
    f["tmed14_May"] = ma(tmed,14)[151]
    f["tmed28_May"] = ma(tmed,28)[151]
    f["gdd5_120"] = gdd5[119]
    f["pp_120"]   = prec[:120].sum()
    return dfm, f

# ---------------------------------------------------------------
# App — Tabs
# ---------------------------------------------------------------
tabs = st.tabs(["🧪 Entrenar prototipos + clasificador", "🔮 Predecir nueva curva", "📊 Evaluar"])

with tabs[0]:
    st.subheader("🧪 Entrenamiento (k-medoids DTW + mezcla de prototipos)")
    st.markdown("Subí **meteorología multianual** y **curvas históricas** (XLSX por año).")
    meteo_book = st.file_uploader("📘 Meteorología multianual (una hoja por año)", type=["xlsx","xls"])
    curvas_files = st.file_uploader("📈 Curvas históricas (XLSX por año, acumulada o semanal)", type=["xlsx","xls"], accept_multiple_files=True)

    K = st.slider("Número de prototipos/patrones (K)", 2, 6, 4, 1)
    seed = st.number_input("Semilla", 0, 99999, 42)
    btn_train = st.button("🚀 Entrenar")

    if btn_train:
        if not (meteo_book and curvas_files):
            st.error("Cargá ambos conjuntos: meteo y curvas.")
            st.stop()

        # --- leer meteo por año ---
        sheets = pd.read_excel(meteo_book, sheet_name=None)
        meteo_dict = {}
        for name, df in sheets.items():
            df = standardize_cols(df)
            df = ensure_jd_1_to_274(df)
            try:
                year = int(re.findall(r"\d{4}", str(name))[0])
            except:
                if "fecha" in df.columns and df["fecha"].notna().any():
                    year = int(df["fecha"].dt.year.mode().iloc[0])
                else:
                    year = None
            if year and all(c in df.columns for c in ["tmin","tmax","prec"]):
                meteo_dict[year] = df[["jd","tmin","tmax","prec"]].copy()

        if not meteo_dict:
            st.error("⛔ No se detectó meteorología válida por año.")
            st.stop()

        # --- leer curvas por año ---
        years_list, curves_list = [], []
        for f in curvas_files:
            y4 = re.findall(r"(\d{4})", f.name)
            year = int(y4[0]) if y4 else None
            if year is None: continue
            curva = np.maximum.accumulate(curva_desde_xlsx_anual(f))
            if curva.max()>0:
                curves_list.append(curva[:JD_MAX])
                years_list.append(year)

        if not years_list:
            st.error("⛔ No se detectaron curvas válidas.")
            st.stop()

        # intersección meteo–curvas
        common = [y for y in years_list if y in meteo_dict]
        if len(common) < 3:
            st.error("⛔ Muy pocos años en común. Se recomienda ≥5.")
            st.stop()

        # ordenar por año
        common = sorted(common)
        curves = [curves_list[years_list.index(y)] for y in common]

        # --- k-medoids bajo DTW ---
        st.info("🧮 Calculando k-medoids (DTW)...")
        medoid_idx, clusters, D = k_medoids_dtw(curves, K=K, max_iter=50, seed=seed)
        protos = [curves[i] for i in medoid_idx]

        # --- features y etiqueta de cluster por año ---
        feat_rows, y_labels = [], []
        feat_names = None
        for y in common:
            dfm, f = build_features_meteo(meteo_dict[y])
            if feat_names is None: feat_names = sorted(f.keys())
            feat_rows.append([f[k] for k in feat_names])
        # cluster label por año = cluster al que fue asignada su curva
        # reconstruimos asignación desde D y medoid_idx
        assign = np.argmin(D[:, np.array(medoid_idx)], axis=1)  # índice cluster 0..K-1
        y_labels = assign.astype(int)

        X = np.array(feat_rows, float)
        y = y_labels
        xsc = StandardScaler().fit(X)
        Xs = xsc.transform(X)

        # --- clasificador de patrón (cluster) ---
        clf = GradientBoostingClassifier(random_state=seed)
        clf.fit(Xs, y)

        # --- warp leve (shift + scale en tiempo) por cluster ---
        # Para ajustar timings sin salir de la forma del prototipo.
        # Entrenamos dos regresores por cluster: shift (δt) y scale (α) en [0.9..1.1]
        regs_shift = {}
        regs_scale = {}
        t = np.arange(1, JD_MAX+1, dtype=float)

        def warp_curve(proto, shift, scale):
            # interpola la curva del prototipo en tiempo transformado t' = (t - shift)/scale
            tp = (t - shift)/max(scale, 1e-6)
            tp = np.clip(tp, 1, JD_MAX)
            y = np.interp(tp, np.arange(1, JD_MAX+1, dtype=float), proto)
            # garantizar monotonía (acumulado de incrementos >=0)
            return np.maximum.accumulate(np.clip(y, 0, 1))

        # construir target de (shift,scale) aproximando cada curva al medoid de su cluster
        for k in range(K):
            idx = np.where(y==k)[0]
            if len(idx)==0:
                regs_shift[k] = GradientBoostingRegressor(random_state=seed)
                regs_scale[k] = GradientBoostingRegressor(random_state=seed)
                continue
            proto = protos[k]
            shifts, scales, Xk = [], [], []
            for ii in idx:
                curv = curves[ii]
                # buscar shift/scale grosero que minimice RMSE vs proto (rejilla pequeña)
                best = (0.0, 1.0, 1e9)
                for sh in range(-20, 21, 5):       # ±20 días
                    for sc in [0.9, 0.95, 1.0, 1.05, 1.1]:
                        cand = warp_curve(proto, sh, sc)
                        rmse = float(np.sqrt(np.mean((cand - curv)**2)))
                        if rmse < best[2]:
                            best = (float(sh), float(sc), rmse)
                shifts.append(best[0]); scales.append(best[1]); Xk.append(Xs[ii])
            Xk = np.vstack(Xk)
            regs_shift[k] = GradientBoostingRegressor(random_state=seed).fit(Xk, np.array(shifts))
            regs_scale[k] = GradientBoostingRegressor(random_state=seed).fit(Xk, np.array(scales))

        # --- guardar bundle ---
        bundle = {
            "xsc": xsc,
            "feat_names": feat_names,
            "clf": clf,
            "protos": np.vstack(protos),  # K x 274
            "regs_shift": regs_shift,
            "regs_scale": regs_scale
        }
        st.success(f"✅ Entrenamiento OK. K={K} prototipos.")
        st.session_state["mix_bundle"] = bundle

        # descarga
        buf = io.BytesIO()
        joblib.dump(bundle, buf)
        st.download_button(
            "💾 Descargar modelo (joblib)",
            data=buf.getvalue(),
            file_name=f"predweem_mixture_dtw_K{K}.joblib",
            mime="application/octet-stream"
        )

        # vista rápida prototipos
        dias = np.arange(1, JD_MAX+1)
        dfp = []
        for k,proto in enumerate(protos):
            dfp.append(pd.DataFrame({"Día": dias, "Valor": proto, "Serie": f"Proto {k}"}))
        dfp = pd.concat(dfp)
        chart = alt.Chart(dfp).mark_line().encode(
            x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE))),
            y=alt.Y("Valor:Q", title="Emergencia acumulada (0–1)", scale=alt.Scale(domain=[0,1])),
            color="Serie:N"
        ).properties(height=420, title="Prototipos (medoids DTW)")
        st.altair_chart(chart, use_container_width=True)

with tabs[1]:
    st.subheader("🔮 Predicción a partir de meteorología nueva")
    modelo_file = st.file_uploader("📦 Modelo (predweem_mixture_dtw_*.joblib)", type=["joblib"])
    meteo_file  = st.file_uploader("📘 Meteorología nueva (XLSX)", type=["xlsx","xls"])
    btn_pred = st.button("🚀 Predecir")

    if btn_pred:
        if not (modelo_file and meteo_file):
            st.error("Cargá el modelo y la meteo.")
            st.stop()
        bundle = joblib.load(modelo_file)
        xsc = bundle["xsc"]; feat_names = bundle["feat_names"]; clf = bundle["clf"]
        protos = bundle["protos"]
        regs_shift = bundle["regs_shift"]; regs_scale = bundle["regs_scale"]
        K = protos.shape[0]

        # features desde meteo
        dfm = pd.read_excel(meteo_file)
        dfm, f = build_features_meteo(dfm)
        X = np.array([[f[k] for k in feat_names]], float)
        Xs = xsc.transform(X)

        # patrón (cluster) + probabilidades
        proba = clf.predict_proba(Xs)[0]  # shape (K,)
        k_hat = int(np.argmax(proba))

        # warp predicho por cluster
        shift = float(regs_shift[k_hat].predict(Xs)[0]) if k_hat in regs_shift else 0.0
        scale = float(regs_scale[k_hat].predict(Xs)[0]) if k_hat in regs_scale else 1.0
        scale = float(np.clip(scale, 0.9, 1.1))

        # construir mezcla convexa de prototipos (suaviza borde de decisión)
        mix = np.zeros(JD_MAX, float)
        t = np.arange(1, JD_MAX+1, dtype=float)

        def warp_curve(proto, sh, sc):
            tp = (t - sh)/max(sc, 1e-6)
            tp = np.clip(tp, 1, JD_MAX)
            y = np.interp(tp, np.arange(1, JD_MAX+1, dtype=float), proto)
            return np.maximum.accumulate(np.clip(y, 0, 1))

        for k in range(K):
            yk = warp_curve(protos[k], shift if k==k_hat else 0.0, scale if k==k_hat else 1.0)
            mix += float(proba[k]) * yk
        mix = np.maximum.accumulate(np.clip(mix, 0, 1))

        # mostrar
        dias = np.arange(1, JD_MAX+1)
        df_pred = pd.DataFrame({"Día": dias, "Emergencia predicha": mix})
        df_proba = pd.DataFrame({"Cluster": [f"C{k}" for k in range(K)], "Probabilidad": proba}).sort_values("Probabilidad", ascending=False)

        chart = alt.Chart(df_pred).mark_line(color="#e67300", strokeWidth=2.5).encode(
            x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE)), title="Día juliano (1–274)"),
            y=alt.Y("Emergencia predicha:Q", scale=alt.Scale(domain=[0,1]), title="Emergencia acumulada (0–1)")
        ).properties(height=420, title=f"Predicción (cluster C{k_hat} · conf {proba.max():.2f} · shift {shift:+.1f} d · scale {scale:.3f})")
        st.altair_chart(chart, use_container_width=True)

        st.dataframe(df_proba, use_container_width=True)
        rel7 = emerg_rel_7d_from_acum(mix)
        out = pd.DataFrame({"Día": dias, "Emergencia_predicha": mix, "Emergencia_relativa_7d": rel7})
        st.download_button("⬇️ Descargar curva (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="curva_predicha_mixture_dtw.csv", mime="text/csv")

with tabs[2]:
    st.subheader("📊 Evaluación histórica (holdout por año vs prototipo)")
    st.markdown("Subí nuevamente las curvas históricas y el modelo para evaluar RMSE/MAE por año.")
    curvas_eval = st.file_uploader("📈 Curvas históricas (XLSX por año)", type=["xlsx","xls"], accept_multiple_files=True, key="eval_cur")
    meteo_book_eval = st.file_uploader("📘 Meteorología multianual (XLSX)", type=["xlsx","xls"], key="eval_met")
    modelo_eval = st.file_uploader("📦 Modelo (joblib)", type=["joblib"], key="eval_model")
    btn_eval = st.button("🔎 Evaluar")

    if btn_eval:
        if not (curvas_eval and meteo_book_eval and modelo_eval):
            st.error("Faltan archivos.")
            st.stop()
        bundle = joblib.load(modelo_eval)
        xsc = bundle["xsc"]; feat_names = bundle["feat_names"]; clf = bundle["clf"]
        protos = bundle["protos"]; regs_shift = bundle["regs_shift"]; regs_scale = bundle["regs_scale"]
        K = protos.shape[0]

        sheets = pd.read_excel(meteo_book_eval, sheet_name=None)
        meteo_dict = {}
        for name, df in sheets.items():
            df = standardize_cols(df); df = ensure_jd_1_to_274(df)
            try:
                year = int(re.findall(r"\d{4}", str(name))[0])
            except:
                year = None
            if year and all(c in df.columns for c in ["tmin","tmax","prec"]):
                meteo_dict[year] = df[["jd","tmin","tmax","prec"]].copy()

        curves_eval = {}
        for f in curvas_eval:
            y4 = re.findall(r"(\d{4})", f.name)
            year = int(y4[0]) if y4 else None
            if year is None: continue
            curva = np.maximum.accumulate(curva_desde_xlsx_anual(f))
            curves_eval[year] = curva

        common = sorted(set(meteo_dict.keys()) & set(curves_eval.keys()))
        if not common:
            st.error("No hay años en común.")
            st.stop()

        dias = np.arange(1, JD_MAX+1)
        rows=[]
        for y in common:
            dfm,_f = build_features_meteo(meteo_dict[y])
            X = np.array([[_f[k] for k in feat_names]], float)
            Xs = xsc.transform(X)
            proba = clf.predict_proba(Xs)[0]
            k_hat = int(np.argmax(proba))
            shift = float(regs_shift[k_hat].predict(Xs)[0]) if k_hat in regs_shift else 0.0
            scale = float(regs_scale[k_hat].predict(Xs)[0]) if k_hat in regs_scale else 1.0
            scale = float(np.clip(scale, 0.9, 1.1))

            # mezcla con warp en cluster dominante
            t = np.arange(1, JD_MAX+1, dtype=float)
            def warp_curve(proto, sh, sc):
                tp = (t - sh)/max(sc,1e-6); tp = np.clip(tp,1,JD_MAX)
                y = np.interp(tp, np.arange(1,JD_MAX+1,dtype=float), proto)
                return np.maximum.accumulate(np.clip(y,0,1))
            mix = np.zeros(JD_MAX, float)
            for k in range(K):
                yk = warp_curve(protos[k], shift if k==k_hat else 0.0, scale if k==k_hat else 1.0)
                mix += float(proba[k]) * yk
            mix = np.maximum.accumulate(np.clip(mix,0,1))

            y_true = curves_eval[y]
            rmse = float(np.sqrt(np.mean((y_true - mix)**2)))
            mae  = float(np.mean(np.abs(y_true - mix)))
            rows.append((int(y), rmse, mae, k_hat, float(proba.max()), shift, scale))
        dfm = pd.DataFrame(rows, columns=["Año","RMSE","MAE","Cluster","Conf","Shift_d","Scale"])
        st.dataframe(dfm.sort_values("Año"), use_container_width=True)

        # detalle gráfico
        yopt = st.selectbox("Ver año:", options=[int(y) for y in common])
        # recomputar para graficar
        dfm_, f_ = build_features_meteo(meteo_dict[yopt])
        X = np.array([[f_[k] for k in feat_names]], float); Xs = xsc.transform(X)
        proba = clf.predict_proba(Xs)[0]; k_hat = int(np.argmax(proba))
        shift = float(regs_shift[k_hat].predict(Xs)[0]) if k_hat in regs_shift else 0.0
        scale = float(regs_scale[k_hat].predict(Xs)[0]) if k_hat in regs_scale else 1.0
        scale = float(np.clip(scale, 0.9, 1.1))
        t = np.arange(1, JD_MAX+1, dtype=float)
        def warp(proto, sh, sc):
            tp = (t - sh)/max(sc,1e-6); tp=np.clip(tp,1,JD_MAX)
            y = np.interp(tp, np.arange(1,JD_MAX+1,dtype=float), proto)
            return np.maximum.accumulate(np.clip(y,0,1))
        mix = np.zeros(JD_MAX, float)
        for k in range(K):
            yk = warp(protos[k], shift if k==k_hat else 0.0, scale if k==k_hat else 1.0)
            mix += float(proba[k]) * yk
        mix = np.maximum.accumulate(np.clip(mix,0,1))

        df_plot = pd.DataFrame({
            "Día": dias,
            "Emergencia real": curves_eval[yopt],
            "Emergencia predicha": mix
        }).melt("Día", var_name="Serie", value_name="Valor")
        chart = alt.Chart(df_plot).mark_line().encode(
            x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE))),
            y=alt.Y("Valor:Q", title="Emergencia acumulada (0–1)", scale=alt.Scale(domain=[0,1])),
            color="Serie:N"
        ).properties(height=420, title=f"Detalle {yopt} (C{k_hat} • conf {proba.max():.2f} • shift {shift:+.1f} • scale {scale:.3f})")
        st.altair_chart(chart, use_container_width=True)

