# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Clasificador de patrones de emergencia (v4.3)
# ===============================================================
# - Entrena patrones prototipo (K-medoids sobre curvas históricas)
# - Construye regresores para ajuste temporal (shift, scale)
# - Clasifica nuevas curvas meteorológicas y predice la emergencia
# - Compara la predicción con todos los patrones (RMSE + DTW)
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import re, io, joblib
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# ======================
# ⚙️ CONFIGURACIÓN
# ======================
st.set_page_config(page_title="PREDWEEM — Clasificador de Patrones v4.3", layout="wide")
st.title("🌾 PREDWEEM — Clasificador y Comparador de Patrones (v4.3)")

JD_MAX = 274
XRANGE = (1, JD_MAX)

# ===============================================================
# 🧩 FUNCIONES AUXILIARES
# ===============================================================
def standardize_cols(df):
    """Normaliza nombres y detecta columnas meteorológicas automáticamente."""
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

    # Detección automática
    if not any(c.startswith("tmin") for c in df.columns):
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].dropna().between(-10,20).mean()>0.5:
                df = df.rename(columns={c:"tmin"}); break
    if not any(c.startswith("tmax") for c in df.columns):
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].dropna().between(10,45).mean()>0.5:
                df = df.rename(columns={c:"tmax"}); break
    if not any(c.startswith("prec") for c in df.columns):
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].dropna().between(0,200).mean()>0.5:
                df = df.rename(columns={c:"prec"}); break

    for c in ["tmin","tmax","prec","jd"]:
        if c in df.columns and isinstance(df[c], pd.Series):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "fecha" in df.columns and isinstance(df["fecha"], pd.Series):
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)

    return df


def ensure_jd_1_to_274(df):
    """Asegura existencia y validez de columna jd (1D, 1–274)."""
    df = df.copy()
    df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)

    if "jd" not in df.columns:
        if "fecha" in df.columns and df["fecha"].notna().any():
            y0 = int(df["fecha"].dt.year.mode().iloc[0])
            df = df[(df["fecha"] >= f"{y0}-01-01") & (df["fecha"] <= f"{y0}-10-01")].copy().sort_values("fecha")
            df["jd"] = df["fecha"].dt.dayofyear - pd.Timestamp(f"{y0}-01-01").dayofyear + 1
        else:
            df["jd"] = np.arange(1, len(df) + 1)

    if isinstance(df["jd"], pd.DataFrame):
        df["jd"] = df["jd"].iloc[:, 0]
    df["jd"] = pd.to_numeric(df["jd"], errors="coerce").astype("Int64")

    jd_range = np.arange(1, JD_MAX + 1)
    df = (df.set_index("jd")
          .reindex(jd_range)
          .interpolate()
          .ffill()
          .bfill()
          .reset_index())
    return df


def curva_desde_xlsx_anual(file):
    """Convierte un XLSX anual a curva acumulada 0–1."""
    df = pd.read_excel(file, header=None)
    if df.shape[1] < 2:
        df = pd.read_excel(file)
    col0 = pd.to_numeric(df.iloc[:,0], errors="coerce")
    col1 = pd.to_numeric(df.iloc[:,1], errors="coerce")

    jd = col0.copy()
    if jd.isna().any():
        jd = jd.where(~jd.isna(), np.arange(1, len(df)+1))
    val = col1.fillna(0.0)

    daily = np.zeros(365)
    for d,v in zip(jd,val):
        if pd.notna(d) and 1 <= int(d) <= 365:
            daily[int(d)-1] += float(v)
    acum = np.cumsum(daily)
    if np.nanmax(acum)==0:
        return np.zeros(JD_MAX)
    curva = (acum / np.nanmax(acum))[:JD_MAX]
    return np.maximum.accumulate(np.clip(curva,0,1))


def build_features_meteo(dfm):
    """Extrae rasgos meteorológicos resumidos."""
    dfm = standardize_cols(dfm)
    dfm = ensure_jd_1_to_274(dfm)
    tmin, tmax, prec = dfm["tmin"], dfm["tmax"], dfm["prec"]
    tmed = (tmin + tmax) / 2
    jd = dfm["jd"].to_numpy(int)
    mask_FM = (jd>=32)&(jd<=151)
    gdd5 = np.cumsum(np.maximum(tmed-5,0))
    f = {
        "pp_FM": np.nansum(prec[mask_FM]) if np.any(mask_FM) else np.nansum(prec),
        "gdd5_FM": np.ptp(gdd5[mask_FM]) if np.any(mask_FM) else np.ptp(gdd5),
        "tmed14_May": float(pd.Series(tmed).rolling(14,min_periods=1).mean().iloc[min(150,len(tmed)-1)]),
        "tmed28_May": float(pd.Series(tmed).rolling(28,min_periods=1).mean().iloc[min(150,len(tmed)-1)]),
        "pp_120": np.nansum(prec[:min(120,len(prec))])
    }
    return dfm, f


def dtw_distance(a,b):
    n,m=len(a),len(b)
    D=np.full((n+1,m+1),np.inf); D[0,0]=0
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost=(a[i-1]-b[j-1])**2
            D[i,j]=cost+min(D[i-1,j],D[i,j-1],D[i-1,j-1])
    return np.sqrt(D[n,m])


def warp(proto,sh,sc):
    t=np.arange(1,JD_MAX+1)
    tp=np.clip((t-sh)/max(sc,1e-6),1,JD_MAX)
    yv=np.interp(tp,np.arange(1,JD_MAX+1),proto)
    return np.maximum.accumulate(np.clip(yv,0,1))

# ===============================================================
# 🧭 INTERFAZ STREAMLIT
# ===============================================================
tabs = st.tabs(["🧪 Entrenar modelo","🔮 Predecir y comparar"])

# ----------------------- ENTRENAMIENTO --------------------------
with tabs[0]:
    st.header("🧪 Entrenamiento")
    meteo = st.file_uploader("📘 Archivo meteorológico (una hoja por año)", type=["xlsx"])
    curvas = st.file_uploader("📈 Curvas históricas (una por año)", type=["xlsx"], accept_multiple_files=True)
    K = st.slider("Número de patrones",2,6,4)
    seed = st.number_input("Semilla aleatoria",0,99999,42)

    if st.button("🚀 Entrenar"):
        sheets = pd.read_excel(meteo,sheet_name=None)
        meteo_dict={int(re.findall(r"\d{4}",n)[0]):ensure_jd_1_to_274(d)
                    for n,d in sheets.items() if re.findall(r"\d{4}",n)}
        curves_dict={int(re.findall(r"\d{4}",f.name)[0]):curva_desde_xlsx_anual(f)
                     for f in curvas}
        common=sorted(set(meteo_dict)&set(curves_dict))
        if len(common)<3: st.error("⚠️ Se necesitan al menos 3 años coincidentes."); st.stop()

        curves=[curves_dict[y] for y in common]
        # matriz de distancias
        D=np.zeros((len(curves),len(curves)))
        for i in range(len(curves)):
            for j in range(i+1,len(curves)):
                D[i,j]=D[j,i]=dtw_distance(curves[i],curves[j])

        # medoids
        medoids=list(np.argsort(D.sum(1))[:K])
        protos=[curves[i] for i in medoids]
        proto_years=[common[i] for i in medoids]

        feats,labels,feat_names=[],[],None
        assign=np.argmin(D[:,medoids],axis=1)
        for i,y in enumerate(common):
            _,f=build_features_meteo(meteo_dict[y])
            if feat_names is None: feat_names=sorted(f)
            feats.append([f[k] for k in feat_names])
            labels.append(assign[i])
        X=np.array(feats)
        xsc=StandardScaler().fit(X)
        Xs=xsc.transform(X)
        clf=GradientBoostingClassifier(random_state=seed).fit(Xs,labels)

        regs_shift,regs_scale={},{}
        for k in range(K):
            idx=np.where(assign==k)[0]; Xk=[];shs=[];scs=[]
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

        bundle={"xsc":xsc,"feat_names":feat_names,"clf":clf,
                "protos":np.vstack(protos),"proto_years":np.array(proto_years),
                "regs_shift":regs_shift,"regs_scale":regs_scale}
        buf=io.BytesIO(); joblib.dump(bundle,buf)
        st.download_button("💾 Descargar modelo entrenado",data=buf.getvalue(),
                           file_name="predweem_v43.joblib",mime="application/octet-stream")
        st.success(f"✅ Modelo entrenado con {K} patrones.")


# ----------------------- PREDICCIÓN -----------------------------
with tabs[1]:
    st.header("🔮 Predicción y comparación")
    modelo_file=st.file_uploader("📦 Modelo entrenado (.joblib)",type=["joblib"])
    meteo_file=st.file_uploader("📘 Meteorología nueva (xlsx)",type=["xlsx"])
    peso_fm=st.slider("Peso Feb–May",1.0,4.0,2.0,0.5)

    if st.button("🚀 Predecir"):
        bundle=joblib.load(modelo_file)
        xsc,feat_names,clf=bundle["xsc"],bundle["feat_names"],bundle["clf"]
        protos,proto_years=bundle["protos"],bundle["proto_years"]
        regs_shift,regs_scale=bundle["regs_shift"],bundle["regs_scale"]
        K=protos.shape[0]

        df,f=build_features_meteo(pd.read_excel(meteo_file))
        X=np.array([[f[k] for k in sorted(feat_names)]],float)
        Xs=xsc.transform(X)
        proba=clf.predict_proba(Xs)[0]
        k_hat=int(np.argmax(proba))
        shift=float(regs_shift[k_hat].predict(Xs)[0]) if k_hat in regs_shift else 0
        scale=float(regs_scale[k_hat].predict(Xs)[0]) if k_hat in regs_scale else 1
        scale=np.clip(scale,0.9,1.1)

        mix=np.zeros(JD_MAX)
        for k in range(K):
            mix+=proba[k]*warp(protos[k],shift if k==k_hat else 0,scale if k==k_hat else 1)

        dias=np.arange(1,JD_MAX+1)
        mask=(dias>=32)&(dias<=151); w=np.ones_like(dias); w[mask]=peso_fm

        comps=[]
        for k in range(K):
            rmse=np.sqrt(np.sum((mix-protos[k])**2*w)/np.sum(w))
            dtw_sim=dtw_distance(mix,protos[k])
            comps.append((k,int(proto_years[k]),rmse,dtw_sim,float(proba[k])))
        dfc=pd.DataFrame(comps,columns=["Cluster","Año_proto","RMSE","DTW","Probabilidad"])
        dfc["Similitud_%"]=100*(1-dfc["DTW"]/dfc["DTW"].max())
        best=int(dfc.loc[dfc["RMSE"].idxmin(),"Cluster"])
        best_year=int(dfc.loc[dfc["RMSE"].idxmin(),"Año_proto"])

        st.sidebar.subheader("🧭 Selección manual de patrón")
        manual_sel=st.sidebar.selectbox(
            "Elegir patrón a destacar",
            ["Automático (más similar)"]+[f"Escenario {i} ({int(proto_years[i])})" for i in range(K)]
        )
        if manual_sel.startswith("Escenario"):
            best=int(re.findall(r"\d+",manual_sel)[0]); best_year=int(proto_years[best])

        st.success(f"🏆 Patrón destacado: Escenario {best} ({best_year})")

        dfp=[]
        for k in range(K):
            tipo="Destacado" if k==best else "Otros"
            dfp.append(pd.DataFrame({"Día":dias,"Valor":protos[k],
                                     "Serie":f"Escenario {k} ({int(proto_years[k])})","Tipo":tipo}))
        dfp.append(pd.DataFrame({"Día":dias,"Valor":mix,"Serie":"Predicción","Tipo":"Predicción"}))
        dfp=pd.concat(dfp)

        color_scale=alt.Color("Tipo:N",
                              scale=alt.Scale(domain=["Predicción","Destacado","Otros"],
                                              range=["#E67300","#0072B2","#CCCCCC"]))
        stroke_scale=alt.Size("Tipo:N",
                              scale=alt.Scale(domain=["Predicción","Destacado","Otros"],
                                              range=[3,3,1]),legend=None)

        chart=alt.Chart(dfp).mark_line().encode(
            x=alt.X("Día:Q",scale=alt.Scale(domain=list(XRANGE))),
            y=alt.Y("Valor:Q",title="Emergencia acumulada (0–1)",scale=alt.Scale(domain=[0,1])),
            color=color_scale,size=stroke_scale,tooltip=["Serie","Tipo","Valor"]
        ).properties(height=440,
                     title=f"Predicción vs Escenarios — resaltado Escenario {best} ({best_year})")
        st.altair_chart(chart,use_container_width=True)
        st.dataframe(dfc.sort_values("RMSE"),use_container_width=True)

