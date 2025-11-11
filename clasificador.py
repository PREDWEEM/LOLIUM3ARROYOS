# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Predicción de Curvas de Emergencia Acumulada (hasta JD 300)
# ===============================================================
# Usa datos meteorológicos diarios (Tmin, Tmax, Prec) desde 1 enero a JD 300
# para predecir la curva de emergencia acumulada completa (0–1).
# Incluye:
#   - Generación automática de curvas históricas desde GitHub
#   - Entrenamiento MLP multisalida (300 días)
#   - Descarga del modelo entrenado (.joblib)
#   - Predicción extendida y comparación con histórico
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
# ⚙️ CONFIGURACIÓN GENERAL
# ===============================================================
st.set_page_config(page_title="PREDWEEM — Curvas hasta JD300", layout="wide")
st.title("🌾 PREDWEEM — Predicción de curvas de emergencia acumulada (1-ene → JD 300)")

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
            curva = curva[:300]     # 🔸 hasta día 300
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

        st.success(f"✅ Se generaron {len(curvas)} curvas (0–300 días).")
        st.session_state["curvas_github"] = curvas

        dias = np.arange(1,301)
        data = []
        for y, curva in curvas.items():
            for d, v in zip(dias, curva):
                data.append({"Día": d, "Año": y, "Emergencia acumulada": v})
        df = pd.DataFrame(data)
        curva_media = df.groupby("Día")["Emergencia acumulada"].mean().reset_index()
        curva_media["Año"] = "Promedio"
        df_total = pd.concat([df, curva_media])

        chart = alt.Chart(df_total).mark_line().encode(
            x=alt.X("Día:Q", title="Día juliano (1–300)"),
            y=alt.Y("Emergencia acumulada:Q", title="Emergencia acumulada (0–1)"),
            color="Año:N",
            size=alt.condition(alt.datum.Año == "Promedio", alt.value(3), alt.value(1))
        ).properties(height=450)
        st.altair_chart(chart, use_container_width=True)

        df_wide = df.pivot(index="Día", columns="Año", values="Emergencia acumulada").fillna(0)
        st.download_button(
            "⬇️ Descargar curvas (CSV)",
            df_wide.to_csv().encode("utf-8"),
            "curvas_emergencia_github_300.csv",
            mime="text/csv"
        )

# ===============================================================
# 🤖 TAB 2 — ENTRENAMIENTO DEL MODELO
# ===============================================================
with tabs[1]:
    st.subheader("🤖 Entrenar modelo predictivo (enero → octubre)")

    meteo_file = st.file_uploader("📂 Cargar archivo meteorológico (una hoja por año)", type=["xlsx","xls"])
    seed = st.number_input("Seed aleatoria", 0, 99999, 42)
    neurons = st.slider("Neuronas por capa", 32, 256, 128, 16)
    max_iter = st.slider("Iteraciones", 300, 5000, 1500, 100)
    btn_fit = st.button("🚀 Entrenar modelo")

    def standardize_cols(df):
        df.columns = [c.lower().strip() for c in df.columns]
        ren = {"temperatura minima":"tmin","tmin":"tmin",
               "temperatura maxima":"tmax","tmax":"tmax",
               "precipitacion":"prec","pp":"prec","rain":"prec"}
        for k,v in ren.items():
            if k in df.columns: df = df.rename(columns={k:v})
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
        for c in ["tmin","tmax","prec"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def slice_jan_to_oct(df):
        if "fecha" in df.columns:
            y = int(df["fecha"].dt.year.mode().iloc[0])
            m = (df["fecha"] >= f"{y}-01-01") & (df["fecha"] <= f"{y}-10-27")
            df = df.loc[m].copy().sort_values("fecha")
            df["jd"] = np.arange(1, len(df)+1)
        return df

    def load_meteo_sheets(uploaded_xlsx):
        sheets = pd.read_excel(uploaded_xlsx, sheet_name=None)
        out = {}
        for name, df in sheets.items():
            df = standardize_cols(df)
            df = slice_jan_to_oct(df)
            try:
                year = int(re.findall(r"\d{4}", name)[0])
            except:
                year = int(df["fecha"].dt.year.mode().iloc[0]) if "fecha" in df.columns else None
            if "jd" not in df.columns:
                if "fecha" in df.columns:
                    df["jd"] = df["fecha"].dt.dayofyear - df["fecha"].dt.dayofyear.iloc[0] + 1
                else:
                    df["jd"] = np.arange(1, len(df) + 1)
            df = df.set_index("jd").reindex(range(1,301)).interpolate().fillna(0).reset_index()
            if all(c in df.columns for c in ["tmin","tmax","prec"]):
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

    meteo_dict = {}
    curvas_dict = st.session_state.get("curvas_github", {})

    if meteo_file:
        meteo_dict = load_meteo_sheets(meteo_file)
        st.success(f"✅ Meteorología cargada ({len(meteo_dict)} años).")

    if btn_fit and meteo_dict and curvas_dict:
        X, Y, years = build_xy(meteo_dict, curvas_dict)
        for i in range(Y.shape[0]):
            Y[i] = Y[i] / (Y[i][-1] if Y[i][-1]!=0 else 1)
        kf = KFold(n_splits=len(years))
        metrics = []
        xsc, ysc = StandardScaler(), StandardScaler()
        for train, test in kf.split(X):
            Xtr, Xte = X[train], X[test]
            Ytr, Yte = Y[train], Y[test]
            Xtr_s, Xte_s = xsc.fit_transform(Xtr), xsc.transform(Xte)
            Ytr_s = ysc.fit_transform(Ytr)
            mlp = MLPRegressor(hidden_layer_sizes=(neurons,), max_iter=max_iter, random_state=seed)
            mlp.fit(Xtr_s, Ytr_s)
            Yhat = ysc.inverse_transform(mlp.predict(Xte_s))
            metrics.append((years[test][0], np.sqrt(mean_squared_error(Yte[0], Yhat[0])), mean_absolute_error(Yte[0], Yhat[0])))
        dfm = pd.DataFrame(metrics, columns=["Año","RMSE","MAE"]).sort_values("Año")
        st.dataframe(dfm, use_container_width=True)
        st.session_state["bundle"] = {"xsc": xsc, "ysc": ysc, "mlp": mlp}
        st.success("✅ Modelo entrenado y guardado en sesión.")
        buf = io.BytesIO()
        joblib.dump(st.session_state["bundle"], buf)
        st.download_button(
            "⬇️ Descargar modelo entrenado (.joblib)",
            data=buf.getvalue(),
            file_name="modelo_curva_emergencia_300.joblib",
            mime="application/octet-stream"
        )

# ===============================================================
# 🔮 TAB 3 — PREDICCIÓN NUEVO AÑO (con comparación histórica + eje secundario)
# ===============================================================
with tabs[2]:
    st.subheader("🔮 Predicción extendida (hasta JD 300)")

    curvas_hist = st.session_state.get("curvas_github", {})
    show_hist_ref = st.checkbox("Mostrar banda histórica (min–max) y promedio", value=True)

    meteo_pred = st.file_uploader("📂 Meteorología nueva (xlsx)", type=["xlsx","xls"], key="pred")
    modelo_up  = st.file_uploader("📦 Modelo entrenado (.joblib)", type=["joblib"])

    if st.button("Predecir curva"):
        if not meteo_pred or not modelo_up:
            st.error("Faltan archivos.")
        else:
            try:
                # --- Cargar modelo entrenado ---
                bundle = joblib.load(modelo_up)
                xsc, ysc, mlp = bundle["xsc"], bundle["ysc"], bundle["mlp"]

                # --- Procesar meteorología ---
                df = pd.read_excel(meteo_pred)
                df = standardize_cols(df)
                df = slice_jan_to_oct(df)
                if "jd" not in df.columns:
                    if "fecha" in df.columns:
                        df["jd"] = df["fecha"].dt.dayofyear - df["fecha"].dt.dayofyear.iloc[0] + 1
                    else:
                        df["jd"] = np.arange(1, len(df)+1)
                df = df.set_index("jd").reindex(range(1,301)).interpolate().fillna(0).reset_index()

                # --- Predicción de curva acumulada ---
                xnew = np.concatenate([df["tmin"], df["tmax"], df["prec"]]).reshape(1,-1)
                yhat = ysc.inverse_transform(mlp.predict(xsc.transform(xnew)))[0]
                yhat = np.maximum.accumulate(yhat)
                yhat = yhat / (yhat[-1] if yhat[-1] != 0 else 1.0)
                yhat = np.clip(yhat, 0, 1)
                dias = np.arange(1,301)
                df_pred = pd.DataFrame({"Día": dias, "Emergencia predicha": yhat})

                # --- Emergencia relativa semanal (media móvil 7 días) ---
                rel = np.convolve(np.diff(np.insert(yhat,0,0)), np.ones(7)/7, mode="same")
                df_rel = pd.DataFrame({"Día": dias, "Emergencia relativa semanal": rel})

                # --- Capa base: acumulada + histórico ---
                layers = []
                if show_hist_ref and isinstance(curvas_hist, dict) and len(curvas_hist) > 0:
                    H = np.vstack([v[:300] for _, v in sorted(curvas_hist.items()) if len(v) >= 300])
                    y_min, y_max = np.nanmin(H, axis=0), np.nanmax(H, axis=0)
                    y_mean = np.nanmean(H, axis=0)
                    df_band = pd.DataFrame({"Día": dias, "Min": y_min, "Max": y_max})
                    df_mean = pd.DataFrame({"Día": dias, "Promedio histórico": y_mean})

                    area_hist = alt.Chart(df_band).mark_area(opacity=0.15).encode(
                        x=alt.X("Día:Q", title="Día juliano (1–300)"),
                        y=alt.Y("Min:Q", title="Emergencia acumulada (0–1)"),
                        y2="Max:Q"
                    )
                    line_mean = alt.Chart(df_mean).mark_line(color="black", strokeWidth=2).encode(
                        x="Día:Q", y="Promedio histórico:Q"
                    )
                    layers += [area_hist, line_mean]

                # --- Línea de emergencia acumulada predicha ---
                line_pred = alt.Chart(df_pred).mark_line(color="orange", strokeWidth=2.5).encode(
                    x=alt.X("Día:Q", title="Día juliano (1–300)"),
                    y=alt.Y("Emergencia predicha:Q", title="Emergencia acumulada (0–1)",
                            scale=alt.Scale(domain=[0,1]))
                )

                # --- Emergencia relativa semanal (eje secundario) ---
                area_rel = alt.Chart(df_rel).mark_area(color="steelblue", opacity=0.25).encode(
                    x="Día:Q",
                    y=alt.Y("Emergencia relativa semanal:Q",
                            axis=alt.Axis(title="Emergencia relativa semanal",
                                          titleColor="steelblue"))
                )
                line_rel = alt.Chart(df_rel).mark_line(color="steelblue", strokeDash=[5,3]).encode(
                    x="Día:Q",
                    y="Emergencia relativa semanal:Q"
                )

                layers += [line_pred, area_rel, line_rel]

                # --- Combinación con ejes independientes ---
                chart = alt.layer(*layers).resolve_scale(
                    y='independent'
                ).properties(
                    height=480,
                    title="Curva predicha vs. histórico + Emergencia relativa semanal"
                )

                st.altair_chart(chart, use_container_width=True)

                # --- Descarga CSV comparativo ---
                out = pd.DataFrame({"Día": dias,
                                    "Emergencia_predicha": yhat,
                                    "Emergencia_relativa": rel})
                st.download_button(
                    "⬇️ Descargar curva y emergencia relativa (CSV)",
                    out.to_csv(index=False).encode("utf-8"),
                    file_name="curva_predicha_relativa_300.csv",
                    mime="text/csv"
                )

                # --- Métricas comparativas ---
                if show_hist_ref and len(curvas_hist) > 0:
                    rmse = np.sqrt(mean_squared_error(y_mean, yhat))
                    mae  = mean_absolute_error(y_mean, yhat)
                    st.caption(f"📏 Comparación con promedio histórico — RMSE: **{rmse:.3f}**, MAE: **{mae:.3f}**")

            except Exception as e:
                st.error(f"Error en la predicción: {e}")

