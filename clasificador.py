# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Mixture-of-Prototypes (DTW + Monotone + Comparación)
# v4.1 — Corrección curva_desde_xlsx_anual + Compatibilidad Altair 5.x
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
# 🧩 FUNCIONES AUXILIARES
# ===============================================================
def standardize_cols(df):
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


def ensure_jd_1_to_274(df):
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


def curva_desde_xlsx_anual(file):
    """Lee una curva XLSX (día, valor) y devuelve acumulada 0–1 (JD 1–274)."""
    df = pd.read_excel(file, header=None)
    if df.shape[1] < 2:
        df = pd.read_excel(file)

    col0 = pd.to_numeric(df.iloc[:,0], errors="coerce")
    col1 = pd.to_numeric(df.iloc[:,1], errors="coerce")

    # ✅ Corrección segura del TypeError
    jd = col0.copy()
    if jd.isna().any():
        jd = jd.where(~jd.isna(), np.arange(1, len(df)+1))

    val = col1.fillna(0.0)
    jd_clean = jd.dropna().unique()
    paso = int(np.median(np.diff(np.sort(jd_clean)))) if len(jd_clean) > 1 else 7

    daily = np.zeros(365)
    for d,v in zip(jd,val):
        if pd.notna(d) and 1 <= int(d) <= 365:
            daily[int(d)-1] += float(v)

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
    jd = dfm["jd"].to_numpy(int)
    mask_FM = (jd >= 32) & (jd <= 151)
    gdd5 = np.cumsum(np.maximum(tmed - 5, 0))
    f = {}
    f["pp_FM"] = np.nansum(prec[mask_FM]) if np.any(mask_FM) else np.nansum(prec)
    f["gdd5_FM"] = np.ptp(gdd5[mask_FM]) if np.any(mask_FM) else np.ptp(gdd5)
    f["tmed14_May"] = float(pd.Series(tmed).rolling(14,min_periods=1).mean().iloc[min(150,len(tmed)-1)])
    f["tmed28_May"] = float(pd.Series(tmed).rolling(28,min_periods=1).mean().iloc[min(150,len(tmed)-1)])
    f["pp_120"] = np.nansum(prec[:min(120,len(prec))])
    return dfm, f


def dtw_distance(a,b):
    n,m=len(a),len(b)
    D=np.full((n+1,m+1),np.inf); D[0,0]=0
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost=(a[i-1]-b[j-1])**2
            D[i,j]=cost+min(D[i-1,j],D[i,j-1],D[i-1,j-1])
    return np.sqrt(D[n,m])


def k_medoids_dtw(curves,K,max_iter=50,seed=42):
    rng=np.random.default_rng(seed)
    N=len(curves)
    if K>N: K=N
    medoids=list(rng.choice(N,size=K,replace=False))
    D=np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            d=dtw_distance(curves[i],curves[j])
            D[i,j]=D[j,i]=d
    for _ in range(max_iter):
        assign=np.argmin(D[:,medoids],axis=1)
        new=[]
        for k in range(K):
            idx=np.where(assign==k)[0]
            if len(idx)==0: new.append(medoids[k]); continue
            subD=D[np.ix_(idx,idx)]
            new.append(idx[np.argmin(subD.sum(1))])
        if new==medoids: break
        medoids=new
    return medoids,D

# ===============================================================
# 🧭 INTERFAZ STREAMLIT
# ===============================================================
tabs = st.tabs(["🧪 Entrenar modelo", "🔮 Predecir & Comparar"])

# --------------------------- ENTRENAMIENTO ----------------------
with tabs[0]:
    st.header("🧪 Entrenamiento del modelo")
    meteo = st.file_uploader("📘 Meteorología multianual (una hoja por año)", type=["xlsx"])
    curvas = st.file_uploader("📈 Curvas históricas (XLSX por año)", type=["xlsx"], accept_multiple_files=True)
    K = st.slider("Número de patrones", 2, 6, 4)
    seed = st.number_input("Semilla aleatoria", 0, 99999, 42)

    if st.button("🚀 Entrenar modelo"):
        sheets = pd.read_excel(meteo, sheet_name=None)
        meteo_dict = {int(re.findall(r"\d{4}", n)[0]): ensure_jd_1_to_274(d)
                      for n, d in sheets.items() if re.findall(r"\d{4}", n)}
        curves_dict = {int(re.findall(r"\d{4}", f.name)[0]): curva_desde_xlsx_anual(f)
                       for f in curvas}
        common = sorted(set(meteo_dict) & set(curves_dict))
        if len(common) < 3:
            st.error("⚠️ Se necesitan al menos 3 años coincidentes."); st.stop()

        curves = [curves_dict[y] for y in common]
        medoids, D = k_medoids_dtw(curves, K, seed=seed)
        protos = [curves[i] for i in medoids]
        proto_years = [common[i] for i in medoids]

        feats, labels, feat_names = [], [], None
        assign = np.argmin(D[:, medoids], axis=1)
        for i, y in enumerate(common):
            _, f = build_features_meteo(meteo_dict[y])
            if feat_names is None: feat_names = sorted(f)
            feats.append([f[k] for k in feat_names])
            labels.append(assign[i])
        X = np.array(feats)
        xsc = StandardScaler().fit(X)
        Xs = xsc.transform(X)
        clf = GradientBoostingClassifier(random_state=seed).fit(Xs, labels)

        regs_shift, regs_scale = {}, {}
        t = np.arange(1, JD_MAX+1)
        def warp(proto, sh, sc):
            tp = np.clip((t - sh)/max(sc, 1e-6), 1, JD_MAX)
            yv = np.interp(tp, np.arange(1, JD_MAX+1), proto)
            return np.maximum.accumulate(np.clip(yv, 0, 1))
        for k in range(K):
            idx = np.where(assign==k)[0]; Xk=[]; shs=[]; scs=[]
            for i in idx:
                best=(0,1,1e9)
                for sh in range(-20,21,5):
                    for sc in [0.9,0.95,1,1.05,1.1]:
                        rmse=np.sqrt(np.mean((warp(protos[k],sh,sc)-curves[i])**2))
                        if rmse<best[2]: best=(sh,sc,rmse)
                Xk.append(Xs[i]); shs.append(best[0]); scs.append(best[1])
            if Xk:
                Xk=np.vstack(Xk)
                regs_shift[k]=GradientBoostingRegressor().fit(Xk,shs)
                regs_scale[k]=GradientBoostingRegressor().fit(Xk,scs)

        bundle = {"xsc": xsc, "feat_names": feat_names, "clf": clf,
                  "protos": np.vstack(protos), "proto_years": np.array(proto_years),
                  "regs_shift": regs_shift, "regs_scale": regs_scale}
        buf = io.BytesIO(); joblib.dump(bundle, buf)
        st.download_button("💾 Descargar modelo", data=buf.getvalue(), file_name="predweem_model_v4_1.joblib")

        dias = np.arange(1, JD_MAX+1)
        dfp = pd.concat([
            pd.DataFrame({"Día": dias, "Valor": protos[i], "Serie": f"Escenario {i} ({proto_years[i]})"})
            for i in range(K)
        ])
        st.altair_chart(alt.Chart(dfp).mark_line().encode(x="Día", y="Valor", color="Serie"), use_container_width=True)


# --------------------------- PREDICCIÓN -------------------------
with tabs[1]:
    st.header("🔮 Predicción y comparación")
    modelo_file = st.file_uploader("📦 Modelo entrenado (.joblib)", type=["joblib"])
    meteo_file = st.file_uploader("📘 Meteorología nueva (XLSX)", type=["xlsx"])
    peso_fm = st.slider("Peso Feb–May", 1.0, 4.0, 2.0, 0.5)

    if st.button("🚀 Predecir"):
        bundle = joblib.load(modelo_file)
        xsc, feat_names, clf = bundle["xsc"], bundle["feat_names"], bundle["clf"]
        protos, proto_years = bundle["protos"], bundle["proto_years"]
        regs_shift, regs_scale = bundle["regs_shift"], bundle["regs_scale"]
        K = protos.shape[0]

        df, f = build_features_meteo(pd.read_excel(meteo_file))
        X = np.array([[f[k] for k in sorted(feat_names)]], float)
        Xs = xsc.transform(X)
        proba = clf.predict_proba(Xs)[0]; k_hat = int(np.argmax(proba))
        shift = float(regs_shift[k_hat].predict(Xs)[0]) if k_hat in regs_shift else 0
        scale = float(regs_scale[k_hat].predict(Xs)[0]) if k_hat in regs_scale else 1
        scale = np.clip(scale, 0.9, 1.1)

        t = np.arange(1, JD_MAX+1)
        def warp(proto, sh, sc):
            tp = np.clip((t - sh)/max(sc, 1e-6), 1, JD_MAX)
            yv = np.interp(tp, np.arange(1, JD_MAX+1), proto)
            return np.maximum.accumulate(np.clip(yv, 0, 1))
        mix = np.zeros(JD_MAX)
        for k in range(K):
            mix += proba[k] * warp(protos[k], shift if k==k_hat else 0, scale if k==k_hat else 1)

        dias = np.arange(1, JD_MAX+1)
        mask = (dias >= 32) & (dias <= 151)
        w = np.ones_like(dias); w[mask] = peso_fm
        comps = []
        for k in range(K):
            rmse = np.sqrt(np.sum((mix-protos[k])**2 * w) / np.sum(w))
            comps.append((k, int(proto_years[k]), rmse, float(proba[k])))
        dfc = pd.DataFrame(comps, columns=["Cluster", "Año_proto", "RMSE", "Probabilidad"])
        dfc["Similitud_%"] = 100*(1 - dfc["RMSE"]/dfc["RMSE"].max())
        best = int(dfc.loc[dfc["RMSE"].idxmin(), "Cluster"])
        best_year = int(dfc.loc[dfc["RMSE"].idxmin(), "Año_proto"])

        st.success(f"🏆 Patrón más similar: Escenario {best} (año {best_year}) "
                   f"— Similitud {dfc.loc[dfc['RMSE'].idxmin(),'Similitud_%']:.1f}%")

        # gráfico seguro
        dfp = []
        for k in range(K):
            tipo = "Más similar" if k==best else "Otros"
            dfp.append(pd.DataFrame({"Día": dias, "Valor": protos[k],
                                     "Serie": f"Escenario {k} ({int(proto_years[k])})", "Tipo": tipo}))
        dfp.append(pd.DataFrame({"Día": dias, "Valor": mix, "Serie": "Predicción", "Tipo": "Predicción"}))
        dfp = pd.concat(dfp)

        color_scale = alt.Color("Tipo:N",
                                scale=alt.Scale(domain=["Predicción","Más similar","Otros"],
                                                range=["#E67300","#0072B2","#CCCCCC"]),
                                legend=alt.Legend(title="Tipo de curva"))
        stroke_scale = alt.Size("Tipo:N",
                                scale=alt.Scale(domain=["Predicción","Más similar","Otros"],
                                                range=[3,3,1]),
                                legend=None)
        chart_cmp = alt.Chart(dfp).mark_line().encode(
            x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE))),
            y=alt.Y("Valor:Q", title="Emergencia acumulada (0–1)", scale=alt.Scale(domain=[0,1])),
            color=color_scale, size=stroke_scale, tooltip=["Serie","Tipo","Valor"]
        ).properties(height=440,
                     title=f"Predicción vs Escenarios — resaltado Escenario {best} ({best_year})")
        st.altair_chart(chart_cmp, use_container_width=True)
        st.dataframe(dfc.sort_values("RMSE"), use_container_width=True)
