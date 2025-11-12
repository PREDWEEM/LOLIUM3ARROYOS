# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Mixture-of-Prototypes (DTW + Monotone + Comparación)
# v3 — Muestra el AÑO del patrón más similar (origen del medoid)
# ===============================================================
# - Entrenamiento: k-medoids + DTW + GradientBoosting
# - Predicción: mezcla convexa con warp temporal (shift/scale)
# - Comparación (RMSE/MAE) ponderada Feb–May
# - Resaltado del patrón más similar + texto interpretativo con AÑO
# - Descargas: modelo, curva predicha, tabla de comparaciones
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
XRANGE  = (1, JD_MAX)

# ===============================================================
# Auxiliares
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
    if df.shape[1] < 2:
        df = pd.read_excel(file)
    col0 = pd.to_numeric(df.iloc[:,0], errors="coerce")
    col1 = pd.to_numeric(df.iloc[:,1], errors="coerce")

    # Si la primera col no es día juliano, asumimos índice simple
    if col0.isna().mean() > 0.5:
        jd  = pd.Series(np.arange(1, len(df)+1))
        val = col1.fillna(0.0)
    else:
        jd  = col0.fillna(method="ffill")
        val = col1.fillna(0.0)

    jd_clean = jd.dropna().astype(int).sort_values().unique()
    paso = int(np.median(np.diff(jd_clean))) if len(jd_clean)>1 else 7

    daily = np.zeros(365, dtype=float)
    for d,v in zip(jd, val):
        if pd.notna(d) and 1 <= int(d) <= 365:
            daily[int(d)-1] += float(v)

    # Si paso es semanal u otra frecuencia, suavizamos a ~diario
    if paso > 1:
        daily = np.convolve(daily, np.ones(7)/7, mode="same")

    acum = np.cumsum(daily)
    if np.nanmax(acum) == 0:
        return np.zeros(JD_MAX)
    curva = (acum / np.nanmax(acum))[:JD_MAX]
    return np.maximum.accumulate(np.clip(curva, 0, 1))

def emerg_rel_7d_from_acum(y):
    inc = np.diff(np.insert(y, 0, 0.0))
    return np.convolve(inc, np.ones(7)/7, mode="same")

def build_features_meteo(dfm):
    dfm = standardize_cols(dfm)
    dfm = ensure_jd_1_to_274(dfm)
    tmin, tmax, prec = dfm["tmin"], dfm["tmax"], dfm["prec"]
    tmed = (tmin + tmax) / 2
    jd   = dfm["jd"].to_numpy(int)

    mask_FM = (jd >= 32) & (jd <= 151)   # Feb–May
    gdd5 = np.cumsum(np.maximum(tmed - 5, 0))

    # métricas base (robustas)
    f = {}
    f["pp_FM"]      = float(np.nansum(prec[mask_FM])) if np.any(mask_FM) else float(np.nansum(prec))
    f["gdd5_FM"]    = float(np.ptp(gdd5[mask_FM]))    if np.any(mask_FM) else float(np.ptp(gdd5))
    f["tmed14_May"] = float(pd.Series(tmed).rolling(14, min_periods=1).mean().iloc[min(150, len(tmed)-1)])
    f["tmed28_May"] = float(pd.Series(tmed).rolling(28, min_periods=1).mean().iloc[min(150, len(tmed)-1)])
    f["pp_120"]     = float(np.nansum(prec[:min(120,len(prec))]))
    return dfm, f

# --------------------------- DTW & K-medoids --------------------
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

    # Optimización simple
    for _ in range(max_iter):
        assign = np.argmin(D[:, medoid_idx], axis=1)
        new_meds = []
        for k in range(K):
            members = np.where(assign == k)[0]
            if len(members) == 0:
                new_meds.append(medoid_idx[k]); continue
            subD = D[np.ix_(members, members)]
            sums = subD.sum(axis=1)
            new_meds.append(members[np.argmin(sums)])
        if new_meds == medoid_idx: break
        medoid_idx = new_meds

    return medoid_idx, D

# ===============================================================
# UI
# ===============================================================
tabs = st.tabs(["🧪 Entrenar", "🔮 Predecir & Comparar"])

# --------------------------- ENTRENAMIENTO ----------------------
with tabs[0]:
    st.header("🧪 Entrenamiento del modelo")
    meteo_book  = st.file_uploader("📘 Meteorología multianual (una hoja por año)", type=["xlsx","xls"])
    curvas_files= st.file_uploader("📈 Curvas históricas (XLSX por año)", type=["xlsx","xls"], accept_multiple_files=True)
    K           = st.slider("Número de patrones (K)", 2, 12, 4)
    seed        = st.number_input("Semilla", 0, 99999, 42)
    btn_train   = st.button("🚀 Entrenar modelo")

    if btn_train:
        if not (meteo_book and curvas_files):
            st.error("Faltan archivos."); st.stop()

        # Meteorología por año
        sheets = pd.read_excel(meteo_book, sheet_name=None)
        meteo_dict, years_meteo = {}, []
        for name, df in sheets.items():
            df = standardize_cols(df); df = ensure_jd_1_to_274(df)
            y4 = re.findall(r"(\d{4})", str(name))
            if not y4: continue
            year = int(y4[0])
            if all(c in df.columns for c in ["tmin","tmax","prec"]):
                meteo_dict[year] = df[["jd","tmin","tmax","prec"]].copy()
                years_meteo.append(year)

        # Curvas por año
        curves_by_year = {}
        for f in curvas_files:
            y4 = re.findall(r"(\d{4})", f.name)
            if not y4: continue
            year = int(y4[0])
            curves_by_year[year] = curva_desde_xlsx_anual(f)

        # Intersección
        common = sorted(set(meteo_dict) & set(curves_by_year))
        if len(common) < 3:
            st.error("Se requieren ≥3 años en común para entrenar."); st.stop()

        curves = [curves_by_year[y] for y in common]
        medoid_idx, D = k_medoids_dtw(curves, K=K, seed=seed)
        protos = [curves[i] for i in medoid_idx]
        proto_years = [common[i] for i in medoid_idx]   # 🔴 AÑOS de los prototipos

        # Features + asignaciones
        feat_names, feats, labels = None, [], []
        assign = np.argmin(D[:, medoid_idx], axis=1)
        for i, y in enumerate(common):
            _, f = build_features_meteo(meteo_dict[y])
            if feat_names is None: feat_names = sorted(f.keys())
            feats.append([f[k] for k in feat_names])
            labels.append(int(assign[i]))
        X = np.array(feats, float); y_lab = np.array(labels, int)

        # Clasificador de patrón
        xsc = StandardScaler().fit(X); Xs = xsc.transform(X)
        clf = GradientBoostingClassifier(random_state=seed).fit(Xs, y_lab)

        # Warps por cluster (shift/scale)
        regs_shift, regs_scale = {}, {}
        t = np.arange(1, JD_MAX+1, dtype=float)
        def warp(proto, sh, sc):
            tp = (t - sh)/max(sc, 1e-6); tp = np.clip(tp, 1, JD_MAX)
            yv = np.interp(tp, np.arange(1, JD_MAX+1, dtype=float), proto)
            return np.maximum.accumulate(np.clip(yv, 0, 1))

        for k in range(K):
            idxs = np.where(y_lab == k)[0]; shifts, scales, Xk = [], [], []
            for i in idxs:
                curv = curves[i]
                best = (0, 1, 1e9)
                for sh in range(-20, 21, 5):
                    for sc in [0.9, 0.95, 1.00, 1.05, 1.10]:
                        cand = warp(protos[k], sh, sc)
                        rmse = float(np.sqrt(np.mean((cand - curv)**2)))
                        if rmse < best[2]: best = (float(sh), float(sc), rmse)
                shifts.append(best[0]); scales.append(best[1]); Xk.append(Xs[i])
            if Xk:
                Xk = np.vstack(Xk)
                regs_shift[k] = GradientBoostingRegressor(random_state=seed).fit(Xk, np.array(shifts))
                regs_scale[k] = GradientBoostingRegressor(random_state=seed).fit(Xk, np.array(scales))

        # Bundle + descarga
        bundle = {
            "xsc": xsc,
            "feat_names": feat_names,
            "clf": clf,
            "protos": np.vstack(protos),
            "proto_years": np.array(proto_years, int),   # 🔴 guardamos AÑOS
            "regs_shift": regs_shift,
            "regs_scale": regs_scale
        }
        st.session_state["bundle"] = bundle

        buf = io.BytesIO(); joblib.dump(bundle, buf)
        st.download_button("💾 Descargar modelo (.joblib)", data=buf.getvalue(),
                           file_name=f"predweem_mixture_dtw_K{K}.joblib",
                           mime="application/octet-stream")

        # Mapa de prototipos con año
        dias = np.arange(1, JD_MAX+1)
        dfp = pd.concat([
            pd.DataFrame({"Día":dias,"Valor":protos[i],
                          "Serie":f"Escenario {i} ({proto_years[i]})"})
            for i in range(K)
        ])
        chart = alt.Chart(dfp).mark_line().encode(
            x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE))),
            y=alt.Y("Valor:Q", title="Emergencia acumulada (0–1)", scale=alt.Scale(domain=[0,1])),
            color="Serie:N"
        ).properties(height=420, title="Prototipos (medoids DTW) con año de origen")
        st.altair_chart(chart, use_container_width=True)

# ------------------------ PREDICCIÓN ----------------------------
with tabs[1]:
    st.header("🔮 Predicción y comparación")
    modelo_file = st.file_uploader("📦 Modelo (.joblib)", type=["joblib"])
    meteo_file  = st.file_uploader("📘 Meteorología nueva (XLSX)", type=["xlsx","xls"])
    peso_fm     = st.slider("Peso Feb–May en similitud", 1.0, 4.0, 2.0, 0.5,
                            help=">1 aumenta la importancia de JD 32–151.")
    btn_pred    = st.button("🚀 Predecir")

    if btn_pred:
        if not (modelo_file and meteo_file):
            st.error("Faltan archivos."); st.stop()

        bundle = joblib.load(modelo_file)
        xsc         = bundle["xsc"]
        feat_names  = bundle["feat_names"]
        clf         = bundle["clf"]
        protos      = bundle["protos"]
        proto_years = bundle.get("proto_years", np.arange(protos.shape[0]))  # fallback
        regs_shift  = bundle["regs_shift"]
        regs_scale  = bundle["regs_scale"]
        K           = protos.shape[0]

        dfm = pd.read_excel(meteo_file)
        dfm, f = build_features_meteo(dfm)
        X = np.array([[f[k] for k in sorted(feat_names)]], float)
        Xs = xsc.transform(X)

        proba  = clf.predict_proba(Xs)[0]
        k_hat  = int(np.argmax(proba))
        shift  = float(regs_shift.get(k_hat).predict(Xs)[0]) if k_hat in regs_shift else 0.0
        scale  = float(regs_scale.get(k_hat).predict(Xs)[0]) if k_hat in regs_scale else 1.0
        scale  = float(np.clip(scale, 0.9, 1.1))

        t = np.arange(1, JD_MAX+1, dtype=float)
        def warp(proto, sh, sc):
            tp = (t - sh)/max(sc, 1e-6); tp = np.clip(tp, 1, JD_MAX)
            yv = np.interp(tp, np.arange(1, JD_MAX+1, dtype=float), proto)
            return np.maximum.accumulate(np.clip(yv, 0, 1))

        # mezcla convexa con warp solo en cluster ganador
        mix = np.zeros(JD_MAX, float)
        for k in range(K):
            yk = warp(protos[k], shift if k==k_hat else 0.0, scale if k==k_hat else 1.0)
            mix += float(proba[k]) * yk
        mix = np.maximum.accumulate(np.clip(mix, 0, 1))

        dias = np.arange(1, JD_MAX+1)
        st.altair_chart(
            alt.Chart(pd.DataFrame({"Día":dias,"Pred":mix}))
            .mark_line(color="#E67300", strokeWidth=2.5)
            .encode(x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE))),
                    y=alt.Y("Pred:Q", title="Emergencia acumulada (0–1)", scale=alt.Scale(domain=[0,1])) )
            .properties(height=420,
                        title=f"Predicción (C{k_hat} • conf {proba.max():.2f} • shift {shift:+.1f} • scale {scale:.3f})"),
            use_container_width=True
        )

        # ---- Comparación ponderada Feb–May ----
        mask = (dias >= 32) & (dias <= 151)
        w = np.ones_like(dias, dtype=float); w[mask] = peso_fm

        comps = []
        for k in range(K):
            err = (mix - protos[k])**2
            rmse_w = float(np.sqrt(np.sum(err*w) / np.sum(w)))
            mae_w  = float(np.sum(np.abs(mix - protos[k]) * w) / np.sum(w))
            comps.append((k, int(proto_years[k]), rmse_w, mae_w, float(proba[k])))

        df_cmp = pd.DataFrame(comps, columns=["Cluster","Año_proto","RMSE_pond","MAE_pond","Probabilidad"])
        df_cmp["Similitud_%"] = 100 * (1 - df_cmp["RMSE_pond"] / df_cmp["RMSE_pond"].max().clip(min=1e-9))

        # ganador y texto interpretativo con AÑO
        best_row = df_cmp.loc[df_cmp["RMSE_pond"].idxmin()]
        best_idx = int(best_row["Cluster"])
        best_year= int(best_row["Año_proto"])
        best_sim = float(best_row["Similitud_%"])
        best_prob= float(best_row["Probabilidad"])

        st.success(f"🏆 Patrón más similar: **Escenario {best_idx} (año {best_year})** — "
                   f"Similitud: **{best_sim:.1f}%**, "
                   f"Probabilidad (clasificador): **{best_prob:.2f}**")

        # Tabla + descarga
        st.markdown("### 🔁 Comparación (ponderada Feb–May)")
        st.dataframe(df_cmp.sort_values("RMSE_pond"), use_container_width=True)
        st.download_button("⬇️ Descargar comparación (CSV)",
                           df_cmp.to_csv(index=False).encode("utf-8"),
                           file_name="comparacion_escenarios.csv", mime="text/csv")

        # Gráfico: resaltar patrón más similar + año en etiqueta
        dfp = []
        for k in range(K):
            tipo = "Más similar" if k == best_idx else "Otros"
            dfp.append(pd.DataFrame({
                "Día": dias,
                "Valor": protos[k],
                "Serie": f"Escenario {k} ({int(proto_years[k])})",
                "Tipo": tipo
            }))
        dfp.append(pd.DataFrame({"Día": dias, "Valor": mix, "Serie": "Predicción", "Tipo": "Predicción"}))
        dfp = pd.concat(dfp)

        highlight = alt.condition(
            alt.datum.Tipo == "Más similar", alt.value("#0072B2"),
            alt.condition(alt.datum.Tipo == "Predicción", alt.value("#E67300"), alt.value("#CCCCCC"))
        )
        width_cond = alt.condition(
            alt.datum.Tipo == "Más similar", alt.value(3),
            alt.condition(alt.datum.Tipo == "Predicción", alt.value(3), alt.value(1))
        )

        chart_cmp = alt.Chart(dfp).mark_line().encode(
            x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE))),
            y=alt.Y("Valor:Q", title="Emergencia acumulada (0–1)", scale=alt.Scale(domain=[0,1])),
            color=highlight,
            strokeWidth=width_cond,
            tooltip=["Serie","Tipo","Valor"]
        ).properties(
            height=440,
            title=f"Predicción vs Escenarios — resaltado Escenario {best_idx} ({best_year})"
        )
        st.altair_chart(chart_cmp, use_container_width=True)

        # Descarga curva predicha
        df_pred = pd.DataFrame({
            "Día": dias,
            "Emergencia_predicha": mix,
            "Emergencia_relativa_7d": emerg_rel_7d_from_acum(mix)
        })
        st.download_button("⬇️ Descargar curva predicha (CSV)",
                           df_pred.to_csv(index=False).encode("utf-8"),
                           file_name="curva_predicha.csv", mime="text/csv")

