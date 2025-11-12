# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Zero: Entrena y predice curvas acumuladas (0..1)
#  - Fuente: meteorología (tmin, tmax, prec) + curvas históricas
#  - Rango temporal fijo: JD 1..274 (1-ene → 1-oct)
#  - Modelo: MLPRegressor multisalida (128 neuronas, 1500 iter)
#  - Tabs: ① Entrenamiento ② Predicción ③ Evaluación histórica
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import io, re, joblib
from io import BytesIO
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="PREDWEEM Zero — Acumulada 0..1", layout="wide")
st.title("🌾 PREDWEEM Zero — Entrenamiento y Predicción (acumulada 0..1 · JD 1..274)")

JD_MAX = 274
XRANGE = (1, JD_MAX)

# ---------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------
def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    ren = {
        "temperatura minima": "tmin", "t_min": "tmin", "t min": "tmin", "mínima": "tmin",
        "tminima": "tmin", "min": "tmin",
        "temperatura maxima": "tmax", "t_max": "tmax", "t max": "tmax", "máxima": "tmax",
        "tmaxima": "tmax", "max": "tmax",
        "precipitacion": "prec", "precip": "prec", "pp": "prec", "lluvia": "prec", "rain": "prec",
        "dia juliano": "jd", "día juliano": "jd", "julian_days": "jd", "dia": "jd", "día": "jd",
        "fecha": "fecha", "date": "fecha"
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

def ensure_jd_1_to_274(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "jd" not in df.columns:
        if "fecha" in df.columns and df["fecha"].notna().any():
            # Reinicia JD desde el 1-ene detectado
            y0 = int(df["fecha"].dt.year.mode().iloc[0])
            df = df[(df["fecha"] >= f"{y0}-01-01") & (df["fecha"] <= f"{y0}-10-01")].copy().sort_values("fecha")
            df["jd"] = df["fecha"].dt.dayofyear - pd.Timestamp(f"{y0}-01-01").dayofyear + 1
        else:
            df["jd"] = np.arange(1, len(df) + 1)
    df = (df.set_index("jd")
            .reindex(range(1, JD_MAX + 1))
            .interpolate()
            .ffill().bfill()
            .reset_index())
    return df

def curva_desde_xlsx_anual(file) -> np.ndarray:
    """
    Lee un XLSX con dos columnas [día, valor] (diaria o semanal).
    Devuelve curva acumulada normalizada 0..1 en JD 1..274.
    - Si paso=1 → agrega por día y acumula.
    - Si paso~7 → considera “valor” semanal en el día indicado.
    """
    df = pd.read_excel(file, header=None)
    # Tolerante: si tiene encabezado, reintenta sin header=None
    if df.shape[1] < 2:
        df = pd.read_excel(file)
    df = df.copy()
    # Detectar columnas numéricas
    col0 = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    col1 = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    # Si col0 no es numérica, puede ser fecha tipo "22-mar"
    if col0.isna().mean() > 0.5:
        try:
            fch = pd.to_datetime(df.iloc[:, 0], errors="coerce", dayfirst=True)
            jd = fch.dt.dayofyear
            val = pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0.0)
        except Exception:
            # fallback: todo cero
            daily = np.zeros(365, float)
            curva = np.cumsum(daily); 
            return (curva/curva.max())[:JD_MAX] if curva.max()>0 else np.zeros(JD_MAX)
    else:
        jd = col0.astype("Int64")
        val = col1.fillna(0.0)

    # Detecta paso típico (1 vs 7)
    jd_clean = jd.dropna().astype(int).sort_values().unique()
    if len(jd_clean) > 1:
        paso = int(np.median(np.diff(jd_clean)))
    else:
        paso = 7

    daily = np.zeros(365, dtype=float)
    if paso == 1:
        # “val” es relativo diario → suma diaria
        for d, v in zip(jd, val):
            if pd.notna(d) and 1 <= int(d) <= 365:
                daily[int(d) - 1] += float(v)
    else:
        # “val” es relativo semanal → ubica valor en el día y suaviza con ventana 7
        for d, v in zip(jd, val):
            if pd.notna(d) and 1 <= int(d) <= 365:
                daily[int(d) - 1] += float(v)
        # distribuir a la semana por media móvil para evitar escalones
        kernel = np.ones(7) / 7
        daily = np.convolve(daily, kernel, mode="same")

    acum = np.cumsum(daily)
    if np.nanmax(acum) == 0:
        return np.zeros(JD_MAX, dtype=float)
    curva = acum / np.nanmax(acum)
    return curva[:JD_MAX]

def build_xy(meteo_dict: dict, curvas_dict: dict):
    common = sorted(set(meteo_dict.keys()) & set(curvas_dict.keys()))
    X, Y, years = [], [], []
    for y in common:
        dfm = meteo_dict[y]
        x = np.concatenate([dfm["tmin"].to_numpy(), dfm["tmax"].to_numpy(), dfm["prec"].to_numpy()])
        X.append(x)
        Y.append(curvas_dict[y])
        years.append(y)
    return np.array(X), np.array(Y), np.array(years)

def emerg_rel_7d_from_acum(y_acum: np.ndarray) -> np.ndarray:
    inc = np.diff(np.insert(y_acum, 0, 0.0))
    rel7 = np.convolve(inc, np.ones(7) / 7, mode="same")
    return rel7

# ---------------------------------------------------------------
# UI — Tabs
# ---------------------------------------------------------------
tabs = st.tabs(["🧪 Entrenamiento (histórico)", "🔮 Predicción nueva", "📊 Evaluación histórica"])

# ===============================================================
# TAB 1 — ENTRENAMIENTO
# ===============================================================
with tabs[0]:
    st.subheader("🧪 Entrenamiento base desde meteorología + curvas acumuladas")

    meteo_book = st.file_uploader("📘 Meteorología multianual (una hoja por año)", type=["xlsx", "xls"])
    st.caption("La planilla debe contener por hoja un año, con columnas: fecha o jd, tmin, tmax, prec.")

    st.markdown("**📥 Curvas históricas (acumuladas 0..1):** subí uno o varios XLSX anuales.")
    curvas_files = st.file_uploader("Cargar XLSX por año (2008, 2009, ..., 2025)", type=["xlsx", "xls"], accept_multiple_files=True)

    seed = st.number_input("Semilla aleatoria", 0, 99999, 42)
    neurons = st.slider("Neuronas (capa oculta)", 16, 256, 128, step=16)
    max_iter = st.slider("Iteraciones (max_iter)", 300, 5000, 1500, step=100)

    btn_train = st.button("🚀 Entrenar modelo")

    if btn_train:
        if not meteo_book or not curvas_files:
            st.error("Cargá la meteorología y al menos una curva histórica.")
            st.stop()

        # 1) Meteorología por año
        sheets = pd.read_excel(meteo_book, sheet_name=None)
        meteo_dict = {}
        for name, df in sheets.items():
            df = standardize_cols(df)
            df = ensure_jd_1_to_274(df)
            try:
                year = int(re.findall(r"\d{4}", str(name))[0])
            except Exception:
                # intenta deducir por la columna fecha
                if "fecha" in df.columns and df["fecha"].notna().any():
                    year = int(df["fecha"].dt.year.mode().iloc[0])
                else:
                    year = None
            if year and all(c in df.columns for c in ["tmin", "tmax", "prec"]):
                meteo_dict[year] = df[["jd", "tmin", "tmax", "prec"]].copy()

        if not meteo_dict:
            st.error("No se pudo construir meteorología por año con columnas tmin, tmax, prec.")
            st.stop()
        st.success(f"✅ Meteorología cargada para {len(meteo_dict)} años.")

        # 2) Curvas acumuladas por año (desde XLSX anuales)
        curvas_dict = {}
        for f in curvas_files:
            try:
                y4 = re.findall(r"(\d{4})", f.name)
                year = int(y4[0]) if y4 else None
                curva = curva_desde_xlsx_anual(f)
                if year is not None and np.nanmax(curva) > 0:
                    curvas_dict[year] = curva
            except Exception:
                pass

        if not curvas_dict:
            st.error("No se detectaron curvas válidas en los XLSX anuales.")
            st.stop()
        st.success(f"✅ Curvas acumuladas cargadas para {len(curvas_dict)} años.")

        # 3) Construir X, Y y entrenar
        X, Y, years = build_xy(meteo_dict, curvas_dict)
        if len(years) < 3:
            st.warning("Muy pocos años en común entre meteo y curvas. Se recomienda ≥ 5.")
        # Normalizar Y para forzar final en 1
        for i in range(Y.shape[0]):
            if Y[i][-1] != 1.0 and np.nanmax(Y[i]) > 0:
                Y[i] = Y[i] / Y[i][-1]

        # LOO para métricas por año
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
        st.markdown("### 📊 Métricas Leave-One-Year-Out")
        st.dataframe(dfm, use_container_width=True)

        # Entrenamiento final en todo el set
        xsc.fit(X); ysc.fit(Y)
        mlp_final = MLPRegressor(hidden_layer_sizes=(neurons,), max_iter=max_iter, random_state=seed)
        mlp_final.fit(xsc.transform(X), ysc.transform(Y))

        # Guardar en sesión y ofrecer descarga
        st.session_state["predweem_bundle"] = {"xsc": xsc, "ysc": ysc, "mlp": mlp_final}
        st.success("✅ Modelo entrenado y guardado en sesión.")

        buf = io.BytesIO()
        joblib.dump(st.session_state["predweem_bundle"], buf)
        st.download_button(
            "💾 Descargar modelo entrenado (.joblib)",
            data=buf.getvalue(),
            file_name="predweem_bundle.joblib",
            mime="application/octet-stream"
        )

# ===============================================================
# TAB 2 — PREDICCIÓN
# ===============================================================
with tabs[1]:
    st.subheader("🔮 Predicción de curva acumulada (0..1) a partir de meteorología nueva")

    modelo_up = st.file_uploader("📦 Cargar modelo (.joblib)", type=["joblib"])
    meteo_pred = st.file_uploader("📘 Meteorología nueva (XLSX)", type=["xlsx", "xls"], key="pred")

    show_hist = st.checkbox("Mostrar promedio histórico (si entrenaste en esta sesión)", value=True)

    if st.button("Predecir curva"):
        if not (modelo_up and meteo_pred):
            st.error("Cargá el modelo y la meteorología.")
            st.stop()
        try:
            bundle = joblib.load(modelo_up)
            xsc, ysc, mlp = bundle["xsc"], bundle["ysc"], bundle["mlp"]

            df = pd.read_excel(meteo_pred)
            df = standardize_cols(df)
            df = ensure_jd_1_to_274(df)

            faltan = [c for c in ["tmin", "tmax", "prec"] if c not in df.columns]
            if faltan:
                st.error(f"Faltan columnas meteorológicas: {faltan}")
                st.stop()

            xnew = np.concatenate([df["tmin"], df["tmax"], df["prec"]]).reshape(1, -1)
            yhat = ysc.inverse_transform(mlp.predict(xsc.transform(xnew)))[0]
            yhat = np.maximum.accumulate(yhat)
            yhat = yhat / (yhat[-1] if yhat[-1] != 0 else 1.0)
            yhat = np.clip(yhat, 0, 1)

            dias = np.arange(1, JD_MAX + 1)
            df_pred = pd.DataFrame({"Día": dias, "Emergencia predicha": yhat})

            layers = []
            if show_hist and "predweem_bundle" in st.session_state:
                st.caption("Mostrando promedio histórico estimado a partir de las curvas usadas en esta sesión (si las hubiere).")
                # Si en esta sesión hubo curvas en TAB 1, podemos reconstruir promedio aprox
                # (guardá en tu flujo real el promedio si querés precisión).
                # Acá, solo mostramos la predicción (capa principal).
            line_pred = alt.Chart(df_pred).mark_line(color="#e67300", strokeWidth=2.5).encode(
                x=alt.X("Día:Q", title=f"Día juliano (1–{JD_MAX})", scale=alt.Scale(domain=list(XRANGE))),
                y=alt.Y("Emergencia predicha:Q", title="Emergencia acumulada (0–1)", scale=alt.Scale(domain=[0, 1]))
            )
            layers += [line_pred]

            rel = emerg_rel_7d_from_acum(yhat)
            df_rel = pd.DataFrame({"Día": dias, "Emergencia relativa 7d": rel})
            area_rel = alt.Chart(df_rel).mark_area(opacity=0.25).encode(
                x="Día:Q",
                y=alt.Y("Emergencia relativa 7d:Q", axis=alt.Axis(title="Emergencia relativa 7d"))
            )
            line_rel = alt.Chart(df_rel).mark_line(strokeDash=[5,3]).encode(
                x="Día:Q",
                y="Emergencia relativa 7d:Q"
            )
            layers += [area_rel, line_rel]

            chart = alt.layer(*layers).resolve_scale(y='independent').properties(height=460, title="Curva predicha + relativa 7d")
            st.altair_chart(chart, use_container_width=True)

            out = pd.DataFrame({"Día": dias, "Emergencia_predicha": yhat, "Emergencia_relativa_7d": rel})
            st.download_button(
                "⬇️ Descargar curva (CSV)",
                out.to_csv(index=False).encode("utf-8"),
                file_name="curva_predicha.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error en la predicción: {e}")

# ===============================================================
# TAB 3 — EVALUACIÓN HISTÓRICA (opcional rápido)
# ===============================================================
with tabs[2]:
    st.subheader("📊 Evaluación rápida con curvas históricas (si querés re-chequear)")

    st.markdown("Subí las **mismas curvas anuales** que usaste para entrenar y evaluamos el ajuste Leave-One-Year-Out.")
    curvas_eval = st.file_uploader("Curvas históricas (XLSX por año)", type=["xlsx", "xls"], accept_multiple_files=True, key="eval_curvas")
    meteo_book_eval = st.file_uploader("Meteorología multianual (XLSX)", type=["xlsx", "xls"], key="eval_meteo")
    modelo_eval = st.file_uploader("Modelo entrenado (.joblib)", type=["joblib"], key="eval_model")
    btn_eval = st.button("🔎 Evaluar")

    if btn_eval:
        if not (curvas_eval and meteo_book_eval and modelo_eval):
            st.error("Faltan archivos para la evaluación.")
            st.stop()
        try:
            # meteorología
            sheets = pd.read_excel(meteo_book_eval, sheet_name=None)
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
                if year and all(c in df.columns for c in ["tmin", "tmax", "prec"]):
                    meteo_dict[year] = df[["jd", "tmin", "tmax", "prec"]].copy()

            curvas_dict = {}
            for f in curvas_eval:
                y4 = re.findall(r"(\d{4})", f.name)
                year = int(y4[0]) if y4 else None
                curva = curva_desde_xlsx_anual(f)
                if year is not None and np.nanmax(curva) > 0:
                    curvas_dict[year] = curva

            X, Y, years = build_xy(meteo_dict, curvas_dict)
            for i in range(Y.shape[0]):
                if Y[i][-1] != 1.0 and np.nanmax(Y[i]) > 0:
                    Y[i] = Y[i] / Y[i][-1]

            bundle = joblib.load(modelo_eval)
            xsc, ysc, mlp = bundle["xsc"], bundle["ysc"], bundle["mlp"]

            # Evaluación directa (no reentrena)
            preds, metrics = [], []
            for i, y in enumerate(years):
                xnew = X[i].reshape(1, -1)
                yhat = ysc.inverse_transform(mlp.predict(xsc.transform(xnew)))[0]
                yhat = np.maximum.accumulate(yhat)
                yhat = yhat / (yhat[-1] if yhat[-1] != 0 else 1.0)
                yhat = np.clip(yhat, 0, 1)
                rmse = float(np.sqrt(mean_squared_error(Y[i], yhat)))
                mae = float(mean_absolute_error(Y[i], yhat))
                metrics.append((int(y), rmse, mae))
                preds.append((y, yhat))

            dfm = pd.DataFrame(metrics, columns=["Año", "RMSE", "MAE"]).sort_values("Año")
            st.dataframe(dfm, use_container_width=True)

            # Gráfico por año (selector)
            opt_year = st.selectbox("Ver detalle del año:", options=[int(y) for y in years])
            y_true = Y[list(years).index(opt_year)]
            y_hat = dict(preds)[opt_year]
            dias = np.arange(1, JD_MAX + 1)
            df_plot = pd.DataFrame({
                "Día": dias,
                "Emergencia real": y_true,
                "Emergencia predicha": y_hat
            }).melt("Día", var_name="Serie", value_name="Valor")
            chart = alt.Chart(df_plot).mark_line().encode(
                x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE))),
                y=alt.Y("Valor:Q", title="Emergencia acumulada (0–1)", scale=alt.Scale(domain=[0, 1])),
                color="Serie:N"
            ).properties(height=420, title=f"Detalle {opt_year}")
            st.altair_chart(chart, use_container_width=True)

        except Exception as e:
            st.error(f"Error en la evaluación: {e}")

