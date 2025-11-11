# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Predicción de curva de emergencia acumulada (1-ene → 1-may) desde meteorología diaria
# Actualización: lectura automática de curvas históricas desde GitHub RAW
# ---------------------------------------------------------------
# - Lee archivo meteorológico (1 hoja por año)
# - Descarga curvas de emergencia acumulada (1 archivo XLSX por año) desde GitHub
# - Empareja por año y entrena un modelo MLPRegressor multisalida
# - Permite predecir curva para un nuevo año solo con meteorología
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import io, re, joblib, requests, math
from io import BytesIO
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============== CONFIGURACIÓN STREAMLIT ==============
st.set_page_config(page_title="PREDWEEM · Curva de emergencia desde meteo", layout="wide")
st.title("🌾 PREDWEEM — Predicción de la curva de emergencia acumulada (1-ene → 1-may)")

# ============== FUNCIONES AUXILIARES ==============
COLMAP_METEO = {
    "fecha": ["fecha", "date"],
    "jd": ["dia juliano", "julian_days", "jd"],
    "tmin": ["temperatura minima", "tmin", "t_min"],
    "tmax": ["temperatura maxima", "tmax", "t_max"],
    "prec": ["precipitacion", "pp", "rain", "prec"]
}

def standardize_cols(df):
    df.columns = [c.lower().strip() for c in df.columns]
    ren = {}
    for k, aliases in COLMAP_METEO.items():
        for a in aliases:
            if a in df.columns:
                ren[a] = k
                break
    df = df.rename(columns=ren)
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
    for c in ["tmin","tmax","prec","jd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def slice_jan_to_may(df):
    if "fecha" in df.columns:
        y = int(df["fecha"].dt.year.mode().iloc[0])
        m = (df["fecha"] >= f"{y}-01-01") & (df["fecha"] <= f"{y}-05-01")
        df = df.loc[m].copy().sort_values("fecha")
        if "jd" not in df.columns:
            df["jd"] = np.arange(1, len(df)+1)
    else:
        df = df[df["jd"].between(1,121)].copy().sort_values("jd")
    return df

def infer_year(name, df):
    if "fecha" in df.columns:
        try:
            return int(df["fecha"].dt.year.mode().iloc[0])
        except:
            pass
    nums = re.findall(r"\d{4}", str(name))
    return int(nums[0]) if nums else None

def load_meteo_sheets(uploaded_xlsx):
    sheets = pd.read_excel(uploaded_xlsx, sheet_name=None)
    out = {}
    for name, df in sheets.items():
        df = standardize_cols(df)
        df = slice_jan_to_may(df)
        y = infer_year(name, df)
        if y:
            df = df.set_index("jd").reindex(range(1,122)).interpolate().fillna(0).reset_index()
            out[y] = df[["jd","tmin","tmax","prec"]]
    return out

def load_curve_from_github(years, base_url):
    curvas_dict = {}
    for y in years:
        url = f"{base_url}/{y}.xlsx"
        try:
            r = requests.get(url)
            if r.status_code != 200:
                st.warning(f"No se encontró {y}.xlsx (HTTP {r.status_code})")
                continue
            df = pd.read_excel(BytesIO(r.content), header=None)
            dias = pd.to_numeric(df.iloc[:,0], errors="coerce").to_numpy()
            vals = pd.to_numeric(df.iloc[:,1], errors="coerce").to_numpy()
            daily = np.zeros(365)
            for d,v in zip(dias, vals):
                if not np.isnan(d) and 1<=int(d)<=365 and not np.isnan(v):
                    daily[int(d)-1] = v
            acum = np.cumsum(daily)
            curva = acum / (acum[-1] if acum[-1]!=0 else 1)
            curvas_dict[y] = curva[:121]
        except Exception as e:
            st.error(f"Error leyendo {y}: {e}")
    return curvas_dict

def build_xy(meteo_dict, curvas_dict):
    common = sorted(set(meteo_dict.keys()) & set(curvas_dict.keys()))
    X, Y, years = [], [], []
    for y in common:
        dfm = meteo_dict[y]
        x = np.concatenate([dfm["tmin"], dfm["tmax"], dfm["prec"]])
        X.append(x)
        Y.append(curvas_dict[y])
        years.append(y)
    return np.array(X), np.array(Y), np.array(years)

def rmse(a,b): return np.sqrt(mean_squared_error(a,b))

# ============== CARGA DE DATOS ==============
st.sidebar.header("1️⃣ Cargar meteorología (una hoja por año)")
meteo_file = st.sidebar.file_uploader("📂 Archivo Excel meteorológico", type=["xlsx","xls"])

st.sidebar.header("2️⃣ Curvas históricas desde GitHub")
base_url = st.sidebar.text_input(
    "📦 URL base RAW del repositorio",
    value="https://raw.githubusercontent.com/PREDWEEM/LOLium3arroyos/main"
)
btn_download = st.sidebar.button("📥 Descargar curvas desde GitHub")

seed = st.sidebar.number_input("Seed", 0, 99999, 42)
neurons = st.sidebar.slider("Neuronas por capa", 16, 256, 64, 16)
max_iter = st.sidebar.slider("Iteraciones", 200, 3000, 800, 100)
btn_fit = st.sidebar.button("🚀 Entrenar modelo")

# ============== PROCESAMIENTO ==============
meteo_dict, curvas_dict = {}, {}
if meteo_file:
    meteo_dict = load_meteo_sheets(meteo_file)
    st.success(f"Meteorología cargada ({len(meteo_dict)} años).")

if btn_download and meteo_dict:
    curvas_dict = load_curve_from_github(meteo_dict.keys(), base_url)
    st.success(f"Descargadas {len(curvas_dict)} curvas desde GitHub.")

if meteo_dict and curvas_dict:
    X, Y, years = build_xy(meteo_dict, curvas_dict)
    st.write(f"📈 Dataset combinado: {len(years)} años comunes")
    st.write(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # ========== Entrenamiento Leave-One-Year-Out ==========
    if btn_fit:
        kf = KFold(n_splits=len(years))
        metrics = []
        preds = []
        xsc = StandardScaler()
        ysc = StandardScaler()
        for train, test in kf.split(X):
            Xtr, Xte = X[train], X[test]
            Ytr, Yte = Y[train], Y[test]
            Xtr_s = xsc.fit_transform(Xtr)
            Xte_s = xsc.transform(Xte)
            Ytr_s = ysc.fit_transform(Ytr)
            mlp = MLPRegressor(hidden_layer_sizes=(neurons,), max_iter=max_iter, random_state=seed)
            mlp.fit(Xtr_s, Ytr_s)
            Yhat_s = mlp.predict(Xte_s)
            Yhat = ysc.inverse_transform(Yhat_s)
            m_rmse = rmse(Yte[0], Yhat[0])
            m_mae = mean_absolute_error(Yte[0], Yhat[0])
            metrics.append((years[test][0], m_rmse, m_mae))
            preds.append((years[test][0], Yte[0], Yhat[0]))
        dfm = pd.DataFrame(metrics, columns=["Año","RMSE","MAE"]).sort_values("Año")
        st.dataframe(dfm)

        st.subheader("Comparación curva real vs predicha")
        year_sel = st.selectbox("Seleccionar año", dfm["Año"])
        for y, yt, yp in preds:
            if y == year_sel:
                dias = np.arange(1,122)
                plot_df = pd.DataFrame({
                    "Día": np.concatenate([dias,dias]),
                    "Valor": np.concatenate([yt,yp]),
                    "Serie": ["Real"]*len(dias)+["Predicha"]*len(dias)
                })
                chart = alt.Chart(plot_df).mark_line().encode(
                    x="Día:Q",
                    y=alt.Y("Valor:Q", scale=alt.Scale(domain=[0,1])),
                    color="Serie:N"
                )
                st.altair_chart(chart, use_container_width=True)
                break

        # Entrenamiento final con todos los años
        xsc.fit(X); ysc.fit(Y)
        mlp_final = MLPRegressor(hidden_layer_sizes=(neurons,), max_iter=max_iter, random_state=seed)
        mlp_final.fit(xsc.transform(X), ysc.transform(Y))
        bundle = {"xsc":xsc, "ysc":ysc, "mlp":mlp_final}
        buf = io.BytesIO(); joblib.dump(bundle, buf)
        st.download_button("💾 Descargar modelo entrenado (.joblib)", buf.getvalue(),
                           file_name="modelo_curva_emergencia.joblib")

# ============== PREDICCIÓN NUEVO AÑO ==============
st.markdown("---")
st.header("🔮 Predicción para un nuevo año (solo meteorología)")

meteo_pred = st.file_uploader("Subí archivo meteorológico del nuevo año", type=["xlsx","xls"], key="new")
modelo_up = st.file_uploader("Cargá modelo entrenado (.joblib)", type=["joblib"])

if st.button("Predecir curva"):
    if not meteo_pred or not modelo_up:
        st.error("Faltan archivos.")
    else:
        try:
            bundle = joblib.load(modelo_up)
            xsc, ysc, mlp = bundle["xsc"], bundle["ysc"], bundle["mlp"]
            df = pd.read_excel(meteo_pred)
            df = standardize_cols(df)
            df = slice_jan_to_may(df)
            df = df.set_index("jd").reindex(range(1,122)).interpolate().fillna(0).reset_index()
            xnew = np.concatenate([df["tmin"], df["tmax"], df["prec"]]).reshape(1,-1)
            yhat_s = mlp.predict(xsc.transform(xnew))
            yhat = ysc.inverse_transform(yhat_s)[0]
            yhat = np.clip(np.maximum.accumulate(yhat),0,1)
            dfp = pd.DataFrame({"Día":np.arange(1,122),"Emergencia predicha":yhat})
            chart = alt.Chart(dfp).mark_line(color="orange").encode(
                x="Día:Q", y=alt.Y("Emergencia predicha:Q", scale=alt.Scale(domain=[0,1]))
            )
            st.altair_chart(chart, use_container_width=True)
            st.download_button("⬇️ Descargar curva (CSV)",
                               dfp.to_csv(index=False).encode("utf-8"),
                               file_name="curva_predicha.csv")
        except Exception as e:
            st.error(f"Error en predicción: {e}")

