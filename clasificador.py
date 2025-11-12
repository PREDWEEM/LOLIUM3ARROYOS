# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Curvas de Emergencia (hasta 1 de octubre · JD 274)
# ===============================================================
# - Genera curvas históricas desde GitHub RAW
#   · Detecta frecuencia: diaria → semanal (auto)
#   · Normaliza y recorta a JD 274 (1/oct)
# - Entrena MLP multisalida (Tmin, Tmax, Prec → curva 0..1)
# - Predice curva nueva y:
#   · Muestra la banda histórica min–max y promedio
#   · Calcula emergencia relativa semanal
#   · Identifica el patrón más plausible (correlación / RMSE)
#   · Grafica la curva predicha junto al patrón más cercano
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

# =========================
# ⚙️ CONFIG GENERAL
# =========================
st.set_page_config(page_title="PREDWEEM — Curvas hasta 1/oct (JD 274)", layout="wide")
st.title("🌾 PREDWEEM — Generador, Entrenador y Predictor (1-ene → 1-oct, JD 274)")

JD_MAX = 274
XRANGE = (1, JD_MAX)

# =========================
# 🔧 UTILIDADES
# =========================
def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).lower().strip() for c in df.columns]
    ren = {
        "temperatura minima": "tmin", "t_min": "tmin", "t min": "tmin",
        "tminima": "tmin", "min": "tmin", "mínima": "tmin", "tmin": "tmin",
        "temperatura maxima": "tmax", "t_max": "tmax", "t max": "tmax",
        "tmaxima": "tmax", "max": "tmax", "máxima": "tmax", "tmax": "tmax",
        "precipitacion": "prec", "precip": "prec", "pp": "prec",
        "rain": "prec", "lluvia": "prec", "prec": "prec",
        "dia juliano": "jd", "día juliano": "jd", "julian_days": "jd",
        "dia": "jd", "día": "jd", "fecha": "fecha"
    }
    for k, v in ren.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
    for c in ["tmin", "tmax", "prec", "jd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def slice_jan_to_oct1(df: pd.DataFrame) -> pd.DataFrame:
    if "fecha" in df.columns and df["fecha"].notna().any():
        y_mode = df["fecha"].dt.year.mode()
        if len(y_mode) > 0 and not np.isnan(y_mode.iloc[0]):
            y = int(y_mode.iloc[0])
            m = (df["fecha"] >= f"{y}-01-01") & (df["fecha"] <= f"{y}-10-01")
            df = df.loc[m].copy().sort_values("fecha")
            if "jd" not in df.columns:
                df["jd"] = np.arange(1, len(df) + 1)
    return df

def build_xy(meteo_dict: dict, curvas_dict: dict):
    common = sorted(set(meteo_dict.keys()) & set(curvas_dict.keys()))
    X, Y, years = [], [], []
    for y in common:
        dfm = meteo_dict[y]
        x = np.concatenate([dfm["tmin"], dfm["tmax"], dfm["prec"]])
        X.append(x)
        Y.append(curvas_dict[y])
        years.append(y)
    return np.array(X), np.array(Y), np.array(years)

def emerg_rel_semanal_desde_acum(y_acum: np.ndarray) -> np.ndarray:
    inc_diario = np.diff(np.insert(y_acum, 0, 0.0))
    rel = np.convolve(inc_diario, np.ones(7) / 7, mode="same")
    return rel

# =========================
# 🧭 PESTAÑAS
# =========================
tabs = st.tabs(["📈 Generar curvas desde GitHub", "🤖 Entrenar modelo", "🔮 Predecir nuevo año"])

# ===============================================================
# 📈 TAB 1
# ===============================================================
with tabs[0]:
    st.subheader("📦 Generar curvas automáticamente desde GitHub")
    base_url = st.text_input("URL base RAW del repositorio",
                             value="https://raw.githubusercontent.com/PREDWEEM/LOLium3arroyos/main")
    btn_gen = st.button("🚀 Generar curvas")

    def listar_archivos_github(base_url: str):
        return [f"{base_url}/{y}.xlsx" for y in range(2008, 2031)]

    def descargar_y_procesar(url: str):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                return None, None
            df = pd.read_excel(BytesIO(r.content), header=None)
            dias = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().astype(int).to_numpy()
            vals = pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0).to_numpy()
            if len(dias) == 0 or len(vals) == 0:
                return None, None
            paso = int(np.median(np.diff(np.unique(np.sort(dias))))) if len(dias) > 1 else 7
            if paso == 1:
                semanas_idx = np.arange(0, len(vals), 7)
                vals = np.array([vals[i:i + 7].mean() for i in semanas_idx])
                dias = np.arange(1, len(vals) * 7 + 1, 7)
            daily = np.zeros(365, dtype=float)
            for d, v in zip(dias, vals):
                if 1 <= int(d) <= 365:
                    daily[int(d) - 1] = float(v)
            acum = np.cumsum(daily)
            if acum[-1] == 0:
                return None, None
            curva = acum / acum[-1]
            curva = curva[:JD_MAX]
            anio = int(re.findall(r"(\d{4})", url)[0])
            return anio, curva
        except Exception:
            return None, None

    if btn_gen:
        st.info("Descargando curvas desde GitHub...")
        curvas = {}
        for url in listar_archivos_github(base_url):
            anio, curva = descargar_y_procesar(url)
            if anio and curva is not None:
                curvas[anio] = curva
        if not curvas:
            st.error("No se pudieron generar curvas.")
        else:
            st.session_state["curvas_github"] = curvas
            st.success(f"✅ {len(curvas)} curvas generadas (JD 1–{JD_MAX})")

# ===============================================================
# 🤖 TAB 2
# ===============================================================
with tabs[1]:
    st.subheader("🤖 Entrenar modelo (1-ene → 1-oct, JD 274)")
    meteo_file = st.file_uploader("📂 Archivo meteorológico (una hoja por año)", type=["xlsx", "xls"])
    seed = st.number_input("Seed aleatoria", 0, 99999, 42)
    neurons = st.slider("Neuronas", 32, 256, 128, 16)
    max_iter = st.slider("Iteraciones", 300, 5000, 1500, 100)
    btn_fit = st.button("🚀 Entrenar modelo")
    curvas_dict = st.session_state.get("curvas_github", {})

    if meteo_file:
        sheets = pd.read_excel(meteo_file, sheet_name=None)
        meteo_dict = {}
        for name, dfm in sheets.items():
            dfm = standardize_cols(dfm)
            dfm = slice_jan_to_oct1(dfm)
            try:
                year = int(re.findall(r"\d{4}", name)[0])
            except:
                year = None
            if year and all(c in dfm.columns for c in ["tmin", "tmax", "prec"]):
                dfm = dfm.set_index("jd").reindex(range(1, JD_MAX + 1)).interpolate().fillna(0).reset_index()
                meteo_dict[year] = dfm[["jd", "tmin", "tmax", "prec"]]
        st.session_state["meteo_dict"] = meteo_dict
        st.success(f"✅ {len(meteo_dict)} años cargados.")

    if btn_fit and "meteo_dict" in st.session_state and curvas_dict:
        X, Y, years = build_xy(st.session_state["meteo_dict"], curvas_dict)
        for i in range(Y.shape[0]):
            Y[i] = Y[i] / (Y[i][-1] if Y[i][-1] != 0 else 1)
        kf = KFold(n_splits=len(years))
        metrics = []
        xsc, ysc = StandardScaler(), StandardScaler()
        for tr, te in kf.split(X):
            mlp = MLPRegressor(hidden_layer_sizes=(neurons,), max_iter=max_iter, random_state=seed)
            mlp.fit(xsc.fit_transform(X[tr]), ysc.fit_transform(Y[tr]))
            Yhat = ysc.inverse_transform(mlp.predict(xsc.transform(X[te])))
            rmse = float(np.sqrt(mean_squared_error(Y[te][0], Yhat[0])))
            mae = float(mean_absolute_error(Y[te][0], Yhat[0]))
            metrics.append((int(years[te][0]), rmse, mae))
        st.dataframe(pd.DataFrame(metrics, columns=["Año", "RMSE", "MAE"]).sort_values("Año"))
        mlp.fit(xsc.fit_transform(X), ysc.fit_transform(Y))
        st.session_state["bundle"] = {"xsc": xsc, "ysc": ysc, "mlp": mlp}
        buf = io.BytesIO()
        joblib.dump(st.session_state["bundle"], buf)
        st.download_button("⬇️ Descargar modelo (.joblib)", buf.getvalue(),
                           file_name=f"modelo_curva_emergencia_{JD_MAX}.joblib",
                           mime="application/octet-stream")
        st.success("✅ Modelo entrenado y guardado en sesión.")

# ===============================================================
# 🔮 TAB 3 — PREDICCIÓN Y CLASIFICACIÓN DE PATRÓN
# ===============================================================
with tabs[2]:
    st.subheader("🔮 Predicción y detección del patrón más plausible")
    curvas_hist = st.session_state.get("curvas_github", {})
    meteo_pred = st.file_uploader("📂 Meteorología nueva (xlsx)", type=["xlsx", "xls"], key="pred")
    modelo_up = st.file_uploader("📦 Modelo entrenado (.joblib)", type=["joblib"])
    show_hist_ref = st.checkbox("Mostrar banda histórica", value=True)

    if st.button("Predecir curva"):
        if not meteo_pred or not modelo_up:
            st.error("Faltan archivos.")
        else:
            try:
                bundle = joblib.load(modelo_up)
                xsc, ysc, mlp = bundle["xsc"], bundle["ysc"], bundle["mlp"]
                df = pd.read_excel(meteo_pred)
                df = standardize_cols(df)
                df = slice_jan_to_oct1(df)
                df = df.set_index("jd").reindex(range(1, JD_MAX + 1)).interpolate().fillna(0).reset_index()
                xnew = np.concatenate([df["tmin"], df["tmax"], df["prec"]]).reshape(1, -1)
                yhat = ysc.inverse_transform(mlp.predict(xsc.transform(xnew)))[0]
                yhat = np.maximum.accumulate(yhat)
                yhat = yhat / yhat[-1] if yhat[-1] != 0 else yhat
                dias = np.arange(1, JD_MAX + 1)

                # --- Clasificación de patrón ---
                if curvas_hist:
                    H = np.vstack([v[:JD_MAX] for v in curvas_hist.values()])
                    anios = np.array(list(curvas_hist.keys()))
                    corrs = [np.corrcoef(yhat, h)[0, 1] for h in H]
                    rmses = [np.sqrt(np.mean((yhat - h)**2)) for h in H]
                    best_idx = int(np.argmax(corrs))
                    mejor_anio = anios[best_idx]
                    rmax, rmse_best = corrs[best_idx], rmses[best_idx]

                    patrones = {2008: "P1", 2009: "P1b", 2010: "P2", 2011: "P3",
                                2012: "P1", 2013: "P2", 2014: "P3", 2015: "P1b",
                                2023: "P2", 2024: "P3", 2025: "P1"}

                    st.markdown("### 🧩 Clasificación del patrón más plausible")
                    st.write(f"**Año más similar:** {mejor_anio} (r = {rmax:.3f}, RMSE = {rmse_best:.3f})")
                    if mejor_anio in patrones:
                        st.success(f"🌾 Patrón más plausible: **{patrones[mejor_anio]}** ({mejor_anio})")

                    # --- Gráfico comparativo ---
                    df_comp = pd.DataFrame({
                        "Día": dias,
                        "Predicha": yhat,
                        "Histórica más similar": curvas_hist[mejor_anio][:JD_MAX]
                    }).melt("Día", var_name="Serie", value_name="Emergencia")

                    chart = alt.Chart(df_comp).mark_line().encode(
                        x=alt.X("Día:Q", title="Día juliano (1–274)"),
                        y=alt.Y("Emergencia:Q", title="Emergencia acumulada (0–1)", scale=alt.Scale(domain=[0, 1])),
                        color="Serie:N"
                    ).properties(height=460, title=f"Curva predicha vs patrón {patrones.get(mejor_anio, mejor_anio)}")
                    st.altair_chart(chart, use_container_width=True)

            except Exception as e:
                st.error(f"Error en la predicción: {e}")

