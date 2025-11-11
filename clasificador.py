# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Predicción de curva de emergencia acumulada (1-ene → 1-may) desde meteorología diaria
# - Meteo: Excel con hojas por año (fecha / jd / tmin / tmax / prec)
# - Curvas históricas: archivos por año (día juliano, emergencia diaria); se acumula y normaliza 0..1
# - Modelo: MLPRegressor multi-salida -> predice vector de 121 días (0..1)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import io, re, joblib, math
from pathlib import Path
from datetime import datetime

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="PREDWEEM · Predicción de curva de emergencia", layout="wide")
st.title("🌾 Predicción de la curva de emergencia acumulada (1-ene → 1-may) desde meteorología diaria")

# =============== Utilidades =================

COLMAP_METEO = {
    "fecha": ["fecha", "date"],
    "jd": ["dia juliano", "julian_days", "julian", "jd", "Julian_days"],
    "tmin": ["temperatura minima", "tmin", "t_min", "t. min", "tmin (°c)"],
    "tmax": ["temperatura maxima", "tmax", "t_max", "t. max", "tmax (°c)"],
    "prec": ["precipitacion", "pp", "rain", "prec", "prec (mm)"]
}

def standardize_cols(df: pd.DataFrame, colmap=COLMAP_METEO) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    found = {}
    for key, aliases in colmap.items():
        match = None
        for a in aliases:
            a_ = a.lower().strip()
            if a_ in cols:
                match = cols[a_]
                break
        # si no matchea exacto, intentar por contiene
        if match is None:
            for c in df.columns:
                if key in c.lower():
                    match = c
                    break
        if match is not None:
            found[key] = match
    dfx = df.rename(columns={v: k for k, v in found.items()}).copy()
    if "fecha" in dfx.columns:
        dfx["fecha"] = pd.to_datetime(dfx["fecha"], errors="coerce", dayfirst=True)
    for c in ["jd", "tmin", "tmax", "prec"]:
        if c in dfx.columns:
            dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
    return dfx

def slice_jan_to_may1(df: pd.DataFrame) -> pd.DataFrame:
    if "fecha" in df.columns and df["fecha"].notna().any():
        y = int(df["fecha"].dt.year.mode().iloc[0])
        start = pd.Timestamp(y, 1, 1)
        end = pd.Timestamp(y, 5, 1)
        m = (df["fecha"] >= start) & (df["fecha"] <= end)
        out = df.loc[m].copy().sort_values("fecha")
        if "jd" not in out.columns:
            out["jd"] = np.arange(1, len(out)+1)  # fallback
        return out
    # fallback por JD
    if "jd" in df.columns:
        out = df[(df["jd"] >= 1) & (df["jd"] <= 121)].copy().sort_values("jd")
        return out
    return df.copy()

def infer_year_from_sheet(name: str, df: pd.DataFrame) -> int | None:
    # 1) intentar con fecha
    if "fecha" in df.columns and df["fecha"].notna().any():
        try:
            return int(df["fecha"].dt.year.mode().iloc[0])
        except:
            pass
    # 2) nombre de hoja
    nums = re.findall(r"\d{4}", str(name))
    if nums:
        return int(nums[0])
    # 3) por jd ~ 1..365 no alcanza, devolvemos None
    return None

def load_meteo_sheets(uploaded_xlsx) -> dict:
    """Devuelve {anio: DataFrame con columnas estandar tmin,tmax,prec,jd (fecha opcional)} recortado a 1-ene..1-may"""
    sheets = pd.read_excel(uploaded_xlsx, sheet_name=None)
    out = {}
    for sheet_name, df in sheets.items():
        df = standardize_cols(df)
        df = slice_jan_to_may1(df)
        y = infer_year_from_sheet(sheet_name, df)
        if y is None:
            # intentar por frecuencia de jd
            y = int(st.session_state.get("tmp_year_counter", 2000))
            st.session_state["tmp_year_counter"] = y + 1
        # asegurar columnas
        for c in ["tmin", "tmax", "prec"]:
            if c not in df.columns:
                df[c] = np.nan
        # reindex por jd de 1..121 (rellenar faltantes con ffill/0 según corresponda)
        if "jd" in df.columns:
            df = df.sort_values("jd")
            jd_full = pd.DataFrame({"jd": np.arange(1, 122)})
            df = jd_full.merge(df, on="jd", how="left")
            # completar meteorología: ffill para T y 0 para prec
            for c in ["tmin", "tmax"]:
                df[c] = df[c].interpolate(limit_direction="both")
            df["prec"] = df["prec"].fillna(0.0)
        out[y] = df[["jd", "tmin", "tmax", "prec"]].copy()
    return out

def load_curve_file(file) -> tuple[int, np.ndarray] | None:
    """Lee un xlsx con dos columnas (dia, valor_diario) y devuelve (anio, curva_acumulada_norm[1..121])"""
    try:
        df = pd.read_excel(file, header=None)
    except Exception as e:
        st.error(f"Error leyendo {file.name}: {e}")
        return None
    if df.shape[1] < 2:
        st.error(f"{file.name}: se esperan 2 columnas (día juliano, emergencia diaria).")
        return None
    dias = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    vals = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    # vector diario 1..365
    daily = np.zeros(365, dtype=float)
    for d, v in zip(dias, vals):
        if not np.isnan(d) and 1 <= int(d) <= 365 and not np.isnan(v):
            daily[int(d)-1] = float(v)
    acum = np.cumsum(daily)
    final = acum[-1]
    curva = (acum / final) if final > 0 else acum
    # recorte 1..121
    curva_121 = curva[:121]
    # inferir año por nombre de archivo
    nums = re.findall(r"\d{4}", file.name)
    if nums:
        anio = int(nums[0])
    else:
        anio = None
    return (anio, curva_121)

def build_xy(meteo_dict: dict, curvas_dict: dict):
    """Empareja por año -> X: (121x3) aplanado; Y: (121)"""
    common_years = sorted(list(set(meteo_dict.keys()) & set(curvas_dict.keys())))
    rows_x, rows_y, years = [], [], []
    for y in common_years:
        dfm = meteo_dict[y]
        # features diarias concatenadas: [tmin_1..121, tmax_1..121, prec_1..121]
        x = np.concatenate([dfm["tmin"].to_numpy()[:121],
                            dfm["tmax"].to_numpy()[:121],
                            dfm["prec"].to_numpy()[:121]], axis=0)
        # target: curva acumulada normalizada 1..121
        yvec = curvas_dict[y][:121]
        rows_x.append(x)
        rows_y.append(yvec)
        years.append(y)
    return np.array(rows_x, dtype=float), np.array(rows_y, dtype=float), np.array(years, dtype=int)

def rmse(a, b): return math.sqrt(mean_squared_error(a, b))
def mase(a, b): # scaled by naive seasonal lag-1 difference over train target
    # simple MASE-like: divide MAE por MAE de desplazamiento 1
    denom = mean_absolute_error(a[1:], a[:-1]) + 1e-9
    return mean_absolute_error(a, b)/denom

def curve_metrics(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

# =============== Sidebar: Carga ===============
st.sidebar.header("1) Cargar meteorología (Excel con hojas por año)")
meteo_file = st.sidebar.file_uploader("Subí el Excel de meteorología", type=["xlsx","xls"])

st.sidebar.header("2) Cargar curvas históricas (uno por año)")
curve_files = st.sidebar.file_uploader(
    "Subí varios XLSX (uno por año). 2 columnas: día juliano, emergencia diaria",
    type=["xlsx","xls"],
    accept_multiple_files=True
)

seed = st.sidebar.number_input("Seed", 0, 99999, 42)
hidden_size = st.sidebar.slider("Neuronas por capa", 16, 256, 64, 16)
layers = st.sidebar.selectbox("Arquitectura MLP", ["(64,)", "(128,)", "(64,32)", "(128,64)"], index=2)
lr = st.sidebar.selectbox("Learning rate", ["constant","adaptive","invscaling"], index=1)
max_iter = st.sidebar.slider("Max iter", 200, 3000, 800, 100)
btn_fit = st.sidebar.button("Entrenar (Leave-One-Year-Out)")
btn_save = st.sidebar.button("Guardar modelo (.joblib)")

# =============== Construcción dataset ===============
meteo_dict, curvas_dict = {}, {}
if meteo_file:
    try:
        meteo_dict = load_meteo_sheets(meteo_file)
        st.success(f"Meteorología: {len(meteo_dict)} años detectados.")
    except Exception as e:
        st.error(f"Error procesando meteorología: {e}")

if curve_files:
    for f in curve_files:
        res = load_curve_file(f)
        if res is None: 
            continue
        anio, curva = res
        if anio is None:
            st.warning(f"No se pudo inferir año desde el nombre: {f.name} (se omite).")
            continue
        curvas_dict[anio] = curva
    st.success(f"Curvas: {len(curvas_dict)} años válidos.")

if meteo_dict and curvas_dict:
    X, Y, years = build_xy(meteo_dict, curvas_dict)
    st.subheader("Años en común")
    st.write(list(years))

    st.subheader("Dimensiones del dataset")
    st.write(f"X: {X.shape}  (n_años x 363 features = 121*3)")
    st.write(f"Y: {Y.shape}  (n_años x 121)")

# =============== Modelo y Validación ===============
@st.cache_data(show_spinner=False)
def loocv_train(X, Y, years, hidden_layer_sizes, seed=42, lr="adaptive", max_iter=800):
    # Pipeline: escala X e Y por separado (Y entre 0–1 ya normalizada, igual escalamos suavemente)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    # KFold leave-one-year-out
    kf = KFold(n_splits=len(years), shuffle=False)
    preds = []
    metrics = []
    for train_idx, test_idx in kf.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        Ytr, Yte = Y[train_idx], Y[test_idx]
        # Escalado
        Xtr_s = x_scaler.fit_transform(Xtr)
        Xte_s = x_scaler.transform(Xte)
        Ytr_s = y_scaler.fit_transform(Ytr)
        # MLP multi-output (Y shape: n_años x 121)
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate=lr,
            max_iter=max_iter,
            early_stopping=True,
            random_state=seed
        )
        mlp.fit(Xtr_s, Ytr_s)
        Yhat_s = mlp.predict(Xte_s)
        # desescalar
        Yhat = y_scaler.inverse_transform(Yhat_s)
        preds.append((test_idx[0], Yhat[0], Yte[0]))
        m = curve_metrics(Yte[0], Yhat[0])
        metrics.append((years[test_idx][0], m["RMSE"], m["MAE"]))
    # Entrena final sobre todo el dataset
    Xs = x_scaler.fit_transform(X)
    Ys = y_scaler.fit_transform(Y)
    final_mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        learning_rate=lr,
        max_iter=max_iter,
        early_stopping=True,
        random_state=seed
    )
    final_mlp.fit(Xs, Ys)
    bundle = {"x_scaler": x_scaler, "y_scaler": y_scaler, "mlp": final_mlp}
    return preds, metrics, bundle

if btn_fit and (meteo_dict and curvas_dict):
    hls = eval(layers)  # e.g. "(64,32)" -> tuple
    with st.spinner("Entrenando con validación Leave-One-Year-Out..."):
        preds, metrics, bundle = loocv_train(X, Y, years, hls, seed=seed, lr=lr, max_iter=max_iter)
    st.success("Entrenamiento terminado.")

    st.subheader("Métricas por año (LOO)")
    dfm = pd.DataFrame(metrics, columns=["año","RMSE","MAE"]).sort_values("año")
    st.dataframe(dfm, use_container_width=True)

    # Gráfico comparativo por un año (selección)
    st.subheader("Comparación curva real vs predicha (LOO)")
    sel_year = st.selectbox("Año a visualizar", list(dfm["año"]))
    # buscar
    y_true = y_pred = None
    for idx, yhat, yte in preds:
        if years[idx] == sel_year:
            y_true = yte; y_pred = yhat; break

    if y_true is not None:
        dias = np.arange(1, 122)
        plot_df = pd.DataFrame({
            "Día": np.concatenate([dias, dias]),
            "Valor": np.concatenate([y_true, y_pred]),
            "Serie": ["Real"]*len(dias) + ["Predicha"]*len(dias)
        })
        # banda histórica (opcional) a partir de Y
        y_min = np.nanmin(Y, axis=0)
        y_max = np.nanmax(Y, axis=0)
        band_df = pd.DataFrame({"Día": dias, "Min": y_min, "Max": y_max})

        area = alt.Chart(band_df).mark_area(opacity=0.15).encode(
            x="Día:Q", y="Min:Q", y2="Max:Q"
        )
        lines = alt.Chart(plot_df).mark_line().encode(
            x=alt.X("Día:Q", title="Día juliano (1..121)"),
            y=alt.Y("Valor:Q", title="Emergencia acumulada (0–1)", scale=alt.Scale(domain=[0,1])),
            color=alt.Color("Serie:N", scale=alt.Scale(range=["black","orange"]))
        )
        st.altair_chart(area + lines, use_container_width=True)

    # Guardar en sesión para exportar
    st.session_state["bundle"] = bundle

elif btn_fit:
    st.error("Cargá meteorología y curvas históricas primero.")

# =============== Guardar modelo ===============
if btn_save:
    bundle = st.session_state.get("bundle", None)
    if bundle is None and meteo_dict and curvas_dict:
        st.warning("No hay modelo en memoria. Entrená primero.")
    elif bundle is None:
        st.warning("Todavía no se entrenó el modelo.")
    else:
        buf = io.BytesIO()
        joblib.dump(bundle, buf)
        st.download_button("⬇️ Descargar modelo (.joblib)", data=buf.getvalue(),
                           file_name="modelo_curva_emergencia.joblib",
                           mime="application/octet-stream")

st.markdown("---")
st.header("🔮 Predicción para un año nuevo (solo meteorología)")

new_meteo = st.file_uploader("Subí Excel de meteorología con una hoja (o elegí la hoja) del año a predecir", type=["xlsx","xls"], key="newmet")

sheet_name = st.text_input("Nombre de hoja (opcional). Si queda vacío se toma la primera.", value="")
model_up = st.file_uploader("Cargá un modelo entrenado (.joblib)", type=["joblib"])

if st.button("Predecir curva"):
    if not new_meteo or not model_up:
        st.error("Cargá la meteorología y el modelo.")
    else:
        try:
            bundle = joblib.load(model_up)
            x_scaler = bundle["x_scaler"]; y_scaler = bundle["y_scaler"]; mlp = bundle["mlp"]
        except Exception as e:
            st.error(f"No se pudo cargar el modelo: {e}")
            st.stop()
        try:
            # leer hoja específica o primera
            if sheet_name.strip():
                sheets = pd.read_excel(new_meteo, sheet_name=None)
                if sheet_name not in sheets:
                    st.error(f"No existe hoja '{sheet_name}' en el archivo.")
                    st.stop()
                df = sheets[sheet_name]
            else:
                df = pd.read_excel(new_meteo)

            df = standardize_cols(df)
            df = slice_jan_to_may1(df)

            # reindex 1..121
            if "jd" in df.columns:
                jd_full = pd.DataFrame({"jd": np.arange(1,122)})
                df = jd_full.merge(df, on="jd", how="left")
            for c in ["tmin","tmax"]:
                df[c] = df[c].interpolate(limit_direction="both")
            df["prec"] = df["prec"].fillna(0.0)

            xnew = np.concatenate([df["tmin"].to_numpy()[:121],
                                   df["tmax"].to_numpy()[:121],
                                   df["prec"].to_numpy()[:121]], axis=0).reshape(1, -1)
            xnew_s = x_scaler.transform(xnew)
            yhat_s = mlp.predict(xnew_s)
            yhat = y_scaler.inverse_transform(yhat_s)[0]
            # recortar y saturar 0..1 y monótona no decreciente
            yhat = np.clip(yhat, 0, 1)
            yhat = np.maximum.accumulate(yhat)

            dias = np.arange(1, 122)
            df_pred = pd.DataFrame({"Día": dias, "Emergencia predicha": yhat})

            st.subheader("Curva de emergencia acumulada predicha")
            line = alt.Chart(df_pred).mark_line(color="orange").encode(
                x=alt.X("Día:Q", title="Día juliano (1..121)"),
                y=alt.Y("Emergencia predicha:Q", title="Emergencia acumulada (0–1)", scale=alt.Scale(domain=[0,1]))
            )
            st.altair_chart(line, use_container_width=True)

            csv = df_pred.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar curva predicha (CSV)", data=csv, file_name="curva_emergencia_predicha.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error en la predicción: {e}")

