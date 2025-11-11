# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Generador + Entrenador + Predictor
# Curvas de emergencia acumulada desde GitHub + modelo MLP de predicción
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests, re, io, joblib
from io import BytesIO
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===============================================================
# 🔧 CONFIGURACIÓN
# ===============================================================
st.set_page_config(page_title="PREDWEEM — Predicción de Curvas de Emergencia", layout="wide")
st.title("🌾 PREDWEEM — Generador y predicción de curvas de emergencia acumulada (1-ene → 1-may)")

# ===============================================================
# 🔹 PESTAÑAS PRINCIPALES
# ===============================================================
tabs = st.tabs([
    "📈 Generar curvas desde GitHub",
    "🤖 Entrenar modelo predictivo",
    "🔮 Predecir nuevo año"
])

# ===============================================================
# 📈 TAB 1 — GENERADOR DE CURVAS AUTOMÁTICAS
# ===============================================================
with tabs[0]:
    st.subheader("📦 Generar curvas automáticamente desde GitHub")

    base_url = st.text_input(
        "URL base RAW del repositorio",
        value="https://raw.githubusercontent.com/PREDWEEM/LOLium3arroyos/main"
    )
    btn_gen = st.button("🚀 Generar curvas")

    def listar_archivos_github(base_url):
        return [f"{base_url}/{y}.xlsx" for y in range(2008, 2031)]

    def descargar_y_procesar(url):
        try:
            r = requests.get(url)
            if r.status_code != 200:
                return None, None
            df = pd.read_excel(BytesIO(r.content), header=None)
            dias = pd.to_numeric(df.iloc[:,0], errors="coerce").to_numpy()
            vals = pd.to_numeric(df.iloc[:,1], errors="coerce").to_numpy()
            daily = np.zeros(365)
            for d,v in zip(dias, vals):
                if not np.isnan(d) and 1 <= int(d) <= 365 and not np.isnan(v):
                    daily[int(d)-1] = v
            acum = np.cumsum(daily)
            if acum[-1] == 0: return None, None
            curva = acum / acum[-1]
            curva = curva[:121]
            anio = int(re.findall(r"(\d{4})", url)[0])
            return anio, curva
        except Exception:
            return None, None

    if btn_gen:
        st.info("Descargando curvas desde GitHub...")
        urls = listar_archivos_github(base_url)
        curvas = {}
        for url in urls:
            anio, curva = descargar_y_procesar(url)
            if anio and curva is not None:
                curvas[anio] = curva

        if not curvas:
            st.error("No se pudieron generar curvas. Revisá la URL o los archivos.")
            st.stop()

        st.success(f"✅ Se generaron {len(curvas)} curvas.")
        st.session_state["curvas_github"] = curvas

        # === GRAFICAR ===
        dias = np.arange(1,122)
        data = []
        for y, curva in curvas.items():
            for d, v in zip(dias, curva):
                data.append({"Día": d, "Año": y, "Emergencia acumulada": v})
        df = pd.DataFrame(data)
        curva_media = df.groupby("Día")["Emergencia acumulada"].mean().reset_index()
        curva_media["Año"] = "Promedio"
        df_total = pd.concat([df, curva_media])

        chart = alt.Chart(df_total).mark_line().encode(
            x=alt.X("Día:Q", title="Día juliano (1–121)"),
            y=alt.Y("Emergencia acumulada:Q", title="Emergencia acumulada normalizada (0–1)"),
            color="Año:N",
            size=alt.condition(alt.datum.Año == "Promedio", alt.value(3), alt.value(1))
        ).properties(height=450)

        st.altair_chart(chart, use_container_width=True)

        df_wide = df.pivot(index="Día", columns="Año", values="Emergencia acumulada").fillna(0)
        csv = df_wide.to_csv().encode("utf-8")
        st.download_button("⬇️ Descargar curvas (CSV)", csv, "curvas_emergencia_github.csv", mime="text/csv")

# ===============================================================
# 🤖 TAB 2 — ENTRENAMIENTO DEL MODELO
# ===============================================================
with tabs[1]:
    st.subheader("🤖 Entrenar modelo predictivo a partir de meteorología")

    # === CARGA METEOROLOGÍA ===
    meteo_file = st.file_uploader("📂 Cargar archivo meteorológico (una hoja por año)", type=["xlsx","xls"])
    seed = st.number_input("Seed aleatoria", 0, 99999, 42)
    neurons = st.slider("Neuronas por capa", 16, 256, 64, 16)
    max_iter = st.slider("Iteraciones", 200, 3000, 800, 100)
    btn_fit = st.button("🚀 Entrenar modelo")

    # === FUNCIONES ===
    def standardize_cols(df):
        df.columns = [c.lower().strip() for c in df.columns]
        ren = {"temperatura minima":"tmin","tmin":"tmin",
               "temperatura maxima":"tmax","tmax":"tmax",
               "precipitacion":"prec","pp":"prec"}
        for k,v in ren.items():
            if k in df.columns: df = df.rename(columns={k:v})
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
        for c in ["tmin","tmax","prec"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def slice_jan_to_may(df):
        if "fecha" in df.columns:
            y = int(df["fecha"].dt.year.mode().iloc[0])
            m = (df["fecha"] >= f"{y}-01-01") & (df["fecha"] <= f"{y}-05-01")
            df = df.loc[m].copy().sort_values("fecha")
            df["jd"] = np.arange(1, len(df)+1)
        return df

    def load_meteo_sheets(uploaded_xlsx):
        sheets = pd.read_excel(uploaded_xlsx, sheet_name=None)
        out = {}
        for name, df in sheets.items():
            df = standardize_cols(df)
            df = slice_jan_to_may(df)
            try: year = int(re.findall(r"\d{4}", name)[0])
            except: year = int(df["fecha"].dt.year.mode().iloc[0])
            df = df.set_index("jd").reindex(range(1,122)).interpolate().fillna(0).reset_index()
            out[year] = df[["jd","tmin","tmax","prec"]]
        return out

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

    # === PROCESAMIENTO ===
    meteo_dict = {}
    curvas_dict = st.session_state.get("curvas_github", {})

    if meteo_file:
        meteo_dict = load_meteo_sheets(meteo_file)
        st.success(f"✅ Meteorología cargada ({len(meteo_dict)} años).")

    # === ENTRENAMIENTO ===
    if btn_fit and meteo_dict and curvas_dict:
        X, Y, years = build_xy(meteo_dict, curvas_dict)
        kf = KFold(n_splits=len(years))
        metrics, preds = [], []
        xsc, ysc = StandardScaler(), StandardScaler()
        for train, test in kf.split(X):
            Xtr, Xte = X[train], X[test]
            Ytr, Yte = Y[train], Y[test]
            Xtr_s, Xte_s = xsc.fit_transform(Xtr), xsc.transform(Xte)
            Ytr_s = ysc.fit_transform(Ytr)
            mlp = MLPRegressor(hidden_layer_sizes=(neurons,), max_iter=max_iter, random_state=seed)
            mlp.fit(Xtr_s, Ytr_s)
            Yhat = ysc.inverse_transform(mlp.predict(Xte_s))
            metrics.append((years[test][0], rmse(Yte[0], Yhat[0]), mean_absolute_error(Yte[0], Yhat[0])))
            preds.append((years[test][0], Yte[0], Yhat[0]))

        dfm = pd.DataFrame(metrics, columns=["Año","RMSE","MAE"]).sort_values("Año")
        st.dataframe(dfm, use_container_width=True)

        st.session_state["bundle"] = {"xsc": xsc, "ysc": ysc, "mlp": mlp}
        st.success("✅ Modelo entrenado y guardado en sesión.")

        buf = io.BytesIO()
        joblib.dump(st.session_state["bundle"], buf)
        st.download_button(
            "⬇️ Descargar modelo entrenado (.joblib)",
            data=buf.getvalue(),
            file_name="modelo_curva_emergencia.joblib",
            mime="application/octet-stream"
        )

# ===============================================================
# 🔮 TAB 3 — PREDICCIÓN NUEVO AÑO
# ===============================================================
with tabs[2]:
    st.subheader("🔮 Predicción de nueva curva a partir de meteorología")

    meteo_pred = st.file_uploader("📂 Meteorología nueva (xlsx)", type=["xlsx","xls"], key="pred")
    modelo_up = st.file_uploader("📦 Modelo entrenado (.joblib)", type=["joblib"])

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
                yhat = ysc.inverse_transform(mlp.predict(xsc.transform(xnew)))[0]
                yhat = np.clip(np.maximum.accumulate(yhat),0,1)
                dfp = pd.DataFrame({"Día":np.arange(1,122),"Emergencia predicha":yhat})
                chart = alt.Chart(dfp).mark_line(color="orange").encode(
                    x="Día:Q", y=alt.Y("Emergencia predicha:Q", scale=alt.Scale(domain=[0,1]))
                )
                st.altair_chart(chart, use_container_width=True)
                st.download_button(
                    "⬇️ Descargar curva predicha (CSV)",
                    dfp.to_csv(index=False).encode("utf-8"),
                    file_name="curva_predicha.csv"
                )
            except Exception as e:
                st.error(f"Error en la predicción: {e}")
