# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Curvas de Emergencia (hasta 1 de octubre · JD 274)
# ===============================================================
# - Genera curvas históricas desde GitHub RAW
#   · Detecta frecuencia: diaria → semanal (auto) si paso=1 día
#   · Normaliza y recorta a JD 274 (1/oct)
# - Entrena MLP multisalida (Tmin, Tmax, Prec → curva 0..1 de 274 días)
# - Predice curva nueva y muestra:
#   · Banda histórica min–max y promedio
#   · Eje Y secundario con emergencia relativa semanal (media móvil 7d)
#   · Patrón más plausible (según correlación / RMSE)
# - Descargas: curvas CSV, modelo .joblib, comparación CSV
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

JD_MAX = 274  # 1 de octubre
XRANGE = (1, JD_MAX)

# =========================
# 🔧 UTILIDADES GLOBALES
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
# 📈 TAB 1 — GENERADOR DE CURVAS AUTOMÁTICAS (GitHub RAW)
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
        urls = listar_archivos_github(base_url)
        curvas = {}
        for url in urls:
            anio, curva = descargar_y_procesar(url)
            if anio and curva is not None:
                curvas[anio] = curva
        if not curvas:
            st.error("No se pudieron generar curvas.")
        else:
            st.success(f"✅ Se generaron {len(curvas)} curvas (JD 1–{JD_MAX}).")
            st.session_state["curvas_github"] = curvas

# ===============================================================
# 🤖 TAB 2 — ENTRENAMIENTO DEL MODELO
# ===============================================================
with tabs[1]:
    st.subheader("🤖 Entrenar modelo (1-ene → 1-oct, JD 274)")
    meteo_file = st.file_uploader("📂 Cargar archivo meteorológico (una hoja por año)", type=["xlsx", "xls"])
    seed = st.number_input("Seed aleatoria", 0, 99999, 42)
    neurons = st.slider("Neuronas por capa", 32, 256, 128, 16)
    max_iter = st.slider("Iteraciones", 300, 5000, 1500, 100)
    btn_fit = st.button("🚀 Entrenar modelo")

    curvas_dict = st.session_state.get("curvas_github", {})
    meteo_dict = {}

    if meteo_file:
        sheets = pd.read_excel(meteo_file, sheet_name=None)
        out = {}
        for name, dfm in sheets.items():
            dfm = standardize_cols(dfm)
            dfm = slice_jan_to_oct1(dfm)
            try:
                year = int(re.findall(r"\d{4}", name)[0])
            except:
                year = int(dfm["fecha"].dt.year.mode().iloc[0]) if "fecha" in dfm.columns and dfm["fecha"].notna().any() else None
            if "jd" not in dfm.columns:
                dfm["jd"] = np.arange(1, len(dfm) + 1)
            dfm = dfm.set_index("jd").reindex(range(1, JD_MAX + 1)).interpolate().fillna(0).reset_index()
            if all(c in dfm.columns for c in ["tmin", "tmax", "prec"]):
                out[year] = dfm[["jd", "tmin", "tmax", "prec"]]
        meteo_dict = {k: v for k, v in out.items() if k is not None}
        st.success(f"✅ Meteorología cargada ({len(meteo_dict)} años).")

    if btn_fit and meteo_dict and curvas_dict:
        X, Y, years = build_xy(meteo_dict, curvas_dict)
        for i in range(Y.shape[0]):
            Y[i] = Y[i] / (Y[i][-1] if Y[i][-1] != 0 else 1)
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
            rmse = float(np.sqrt(mean_squared_error(Yte[0], Yhat[0])))
            mae = float(mean_absolute_error(Yte[0], Yhat[0]))
            metrics.append((int(years[test][0]), rmse, mae))
        dfm = pd.DataFrame(metrics, columns=["Año", "RMSE", "MAE"]).sort_values("Año")
        st.dataframe(dfm, use_container_width=True)
        xsc.fit(X); ysc.fit(Y)
        mlp_final = MLPRegressor(hidden_layer_sizes=(neurons,), max_iter=max_iter, random_state=seed)
        mlp_final.fit(xsc.transform(X), ysc.transform(Y))
        st.session_state["bundle"] = {"xsc": xsc, "ysc": ysc, "mlp": mlp_final}
        st.success("✅ Modelo entrenado y guardado en sesión.")
        buf = io.BytesIO()
        joblib.dump(st.session_state["bundle"], buf)
        st.download_button("⬇️ Descargar modelo entrenado (.joblib)",
                           data=buf.getvalue(),
                           file_name=f"modelo_curva_emergencia_{JD_MAX}.joblib",
                           mime="application/octet-stream")

# ===============================================================
# 🔮 TAB 3 — PREDICCIÓN NUEVO AÑO + CLASIFICACIÓN DE PATRÓN
# ===============================================================
with tabs[2]:
    st.subheader("🔮 Predicción (1-ene → 1-oct, JD 274) + Clasificación del patrón más plausible")
    curvas_hist = st.session_state.get("curvas_github", {})
    show_hist_ref = st.checkbox("Mostrar banda histórica", value=True)
    meteo_pred = st.file_uploader("📂 Meteorología nueva (xlsx)", type=["xlsx", "xls"], key="pred")
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
                df = slice_jan_to_oct1(df)
                df = df.set_index("jd").reindex(range(1, JD_MAX + 1)).interpolate().fillna(0).reset_index()
                xnew = np.concatenate([df["tmin"], df["tmax"], df["prec"]]).reshape(1, -1)
                yhat = ysc.inverse_transform(mlp.predict(xsc.transform(xnew)))[0]
                yhat = np.maximum.accumulate(yhat)
                yhat = yhat / (yhat[-1] if yhat[-1] != 0 else 1.0)
                yhat = np.clip(yhat, 0, 1)
                dias = np.arange(1, JD_MAX + 1)

                # Emergencia relativa semanal
                rel = emerg_rel_semanal_desde_acum(yhat)
                df_rel = pd.DataFrame({"Día": dias, "Emergencia relativa semanal": rel})

                # Mostrar patrón más plausible
                if curvas_hist:
                    H = np.vstack([v[:JD_MAX] for v in curvas_hist.values()])
                    anios = np.array(list(curvas_hist.keys()))
                    corrs = [np.corrcoef(yhat, h)[0, 1] for h in H]
                    rmses = [np.sqrt(np.mean((yhat - h) ** 2)) for h in H]
                    idx_best_corr = int(np.argmax(corrs))
                    idx_best_rmse = int(np.argmin(rmses))
                    mejor_corr = anios[idx_best_corr]
                    mejor_rmse = anios[idx_best_rmse]
                    st.markdown("### 🧩 Clasificación del patrón más plausible")
                    st.markdown(f"""
                    - **Mayor correlación:** {mejor_corr} (r = {corrs[idx_best_corr]:.3f})  
                    - **Menor RMSE:** {mejor_rmse} (RMSE = {rmses[idx_best_rmse]:.3f})
                    """)
                    patrones = {2008: "P1", 2009: "P1b", 2010: "P2", 2011: "P3",
                                2012: "P1", 2013: "P2", 2014: "P3", 2015: "P1b",
                                2023: "P2", 2024: "P3", 2025: "P1"}
                    if mejor_corr in patrones:
                        st.success(f"🌾 **Patrón más plausible:** {patrones[mejor_corr]} ({mejor_corr})")
                    else:
                        st.info(f"🌾 Patrón más plausible: año {mejor_corr} (sin etiqueta)")

            except Exception as e:
                st.error(f"Error en la predicción: {e}")
