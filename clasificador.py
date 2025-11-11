# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador de patrón futuro de emergencia a partir de meteorología (1-ene → 1-may)
import streamlit as st
import pandas as pd
import numpy as np
import io, re, joblib
from datetime import datetime
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="PREDWEEM · Patrón por meteorología (ene→may)", layout="wide")
st.title("🌾 Clasificador de patrón de emergencia a partir de meteorología (1-ene → 1-may)")

# ---------- Utilidades ----------
COLMAP = {
    "fecha": ["fecha", "date", "Fecha"],
    "jd": ["dia juliano", "julian_days", "julian", "jd", "Julian_days"],
    "tmin": ["temperatura minima", "tmin", "t_min"],
    "tmax": ["temperatura maxima", "tmax", "t_max"],
    "prec": ["precipitacion", "pp", "rain", "prec"]
}

def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    out = {}
    for key, aliases in COLMAP.items():
        found = None
        for al in aliases:
            al_ = al.lower().strip()
            if al_ in cols:
                found = cols[al_]
                break
        if found is None:
            # si falta alguna, intentar por regex suave
            for c in df.columns:
                if key in c.lower():
                    found = c
                    break
        if found is None:
            # crear columna vacía si no está (mejor avisar)
            out[key] = None
        else:
            out[key] = found
    # Renombrar a nombres estándar
    keep = {v: k for k, v in out.items() if v is not None}
    dfx = df.rename(columns=keep).copy()
    # tipos
    if "fecha" in dfx.columns:
        dfx["fecha"] = pd.to_datetime(dfx["fecha"], errors="coerce", dayfirst=True)
    if "jd" in dfx.columns:
        dfx["jd"] = pd.to_numeric(dfx["jd"], errors="coerce")
    for c in ["tmin","tmax","prec"]:
        if c in dfx.columns:
            dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
    return dfx

def slice_jan_to_may1(df: pd.DataFrame) -> pd.DataFrame:
    if "fecha" in df.columns and df["fecha"].notna().any():
        y = int(df["fecha"].dt.year.mode().iloc[0])
        start = pd.Timestamp(year=y, month=1, day=1)
        end = pd.Timestamp(year=y, month=5, day=1)
        m = (df["fecha"]>=start) & (df["fecha"]<=end)
        return df.loc[m].sort_values("fecha")
    else:
        # fallback por JD
        m = (df["jd"]>=1) & (df["jd"]<=121)
        return df.loc[m].sort_values("jd")

def gdd(series_tmin, series_tmax, base=5.0):
    tmean = (series_tmin + series_tmax)/2.0
    return np.clip(tmean - base, 0, None).sum()

def rain_events(prec, thr=1.0):
    # nro de días con precip > thr
    return np.sum(np.nan_to_num(prec, nan=0.0) > thr)

def longest_dry_spell(prec, thr=1.0):
    # racha seca más larga (días consecutivos con precip ≤ thr)
    p = np.nan_to_num(prec, nan=0.0)
    longest = curr = 0
    for v in p:
        if v <= thr:
            curr += 1
            longest = max(longest, curr)
        else:
            curr = 0
    return longest

def summarize_year(df: pd.DataFrame) -> dict:
    # Estadísticos 1-ene → 1-may
    tmin = df["tmin"].to_numpy()
    tmax = df["tmax"].to_numpy()
    prec = df["prec"].to_numpy()

    feats = {}
    # Temperaturas
    feats["tmin_mean"] = np.nanmean(tmin); feats["tmax_mean"] = np.nanmean(tmax)
    feats["tmin_p10"] = np.nanpercentile(tmin,10); feats["tmin_p90"]=np.nanpercentile(tmin,90)
    feats["tmax_p10"] = np.nanpercentile(tmax,10); feats["tmax_p90"]=np.nanpercentile(tmax,90)
    feats["gdd_b5"]  = gdd(df["tmin"], df["tmax"], base=5.0)
    feats["gdd_b10"] = gdd(df["tmin"], df["tmax"], base=10.0)

    # Precipitaciones
    feats["pp_sum"] = np.nansum(prec)
    feats["pp_mean"] = np.nanmean(prec)
    feats["pp_events_gt1"] = rain_events(prec,1.0)
    feats["dryspell_max"] = longest_dry_spell(prec,1.0)
    # Intensidad: p95 y suma de días >10 mm
    feats["pp_p95"] = np.nanpercentile(prec,95)
    feats["pp_days_gt10"] = int(np.sum(np.nan_to_num(prec,0)>10.0))

    # Interacciones simples
    feats["gdd_b5_per_event"] = feats["gdd_b5"] / (feats["pp_events_gt1"]+1e-6)
    feats["pp_sum_per_dry"] = feats["pp_sum"] / (feats["dryspell_max"]+1e-6)
    return feats

def year_from_df(df):
    if "fecha" in df.columns and df["fecha"].notna().any():
        return int(df["fecha"].dt.year.mode().iloc[0])
    # fallback si no hay fecha
    return int(st.session_state.get("tmp_year_counter", 2000))

# ---------- Sidebar: carga de datos ----------
st.sidebar.header("1) Cargar meteorología por año (1-ene → 1-may)")
files = st.sidebar.file_uploader(
    "Subí 1+ archivos (CSV/XLSX). Un archivo por año.",
    type=["csv","xlsx","xls"],
    accept_multiple_files=True
)

st.sidebar.header("2) Etiquetas de patrón (opcional)")
labels_file = st.sidebar.file_uploader(
    "CSV de etiquetas con columnas: anio, patron",
    type=["csv"]
)

st.sidebar.header("3) Configuración")
kfold = st.sidebar.slider("K-Fold estratificado", 3, 10, 5)
seed = st.sidebar.number_input("Seed", 0, 99999, 42)
btn_train = st.sidebar.button("Entrenar modelo")
btn_export = st.sidebar.button("Exportar modelo entrenado (.joblib)")

# ---------- Armar dataset ----------
rows = []
raw_years = {}
if files:
    for up in files:
        try:
            if up.name.lower().endswith(".csv"):
                df = pd.read_csv(up)
            else:
                df = pd.read_excel(up)
        except Exception as e:
            st.error(f"Error leyendo {up.name}: {e}")
            continue
        df = standardize_cols(df)
        df = slice_jan_to_may1(df)
        yr = year_from_df(df)
        raw_years[yr] = df.copy()

        feats = summarize_year(df)
        feats["anio"] = yr
        rows.append(feats)

X = pd.DataFrame(rows).sort_values("anio") if rows else pd.DataFrame()

# ---------- Etiquetas ----------
y = None
if not X.empty:
    st.subheader("Resumen de features por año (1-ene → 1-may)")
    st.dataframe(X.set_index("anio"))

    if labels_file is not None:
        lab = pd.read_csv(labels_file)
        lab.columns = [c.strip().lower() for c in lab.columns]
        if not {"anio","patron"}.issubset(lab.columns):
            st.error("El CSV de etiquetas debe tener columnas: anio, patron")
        else:
            X = X.merge(lab[["anio","patron"]], on="anio", how="left")
            y = X["patron"]

    # Asignación manual si faltan etiquetas
    if "patron" not in X.columns or X["patron"].isna().any():
        st.subheader("Asignación manual de patrón por año")
        patrones = ["P1","P1b","P2","P3"]
        pats = {}
        for yr in X["anio"].tolist():
            default = None
            if "patron" in X.columns:
                val = X.loc[X["anio"]==yr, "patron"].values[0]
                default = val if pd.notna(val) else None
            pats[yr] = st.selectbox(f"Año {yr}", opciones := patrones, index=(opciones.index(default) if default in patrones else 0), key=f"pat_{yr}")
        X["patron"] = X["anio"].map(pats)
        y = X["patron"]

# ---------- Entrenamiento ----------
@st.cache_data(show_spinner=False)
def train_model(Xdf: pd.DataFrame, target: pd.Series, seed=42, k=5):
    feats = [c for c in Xdf.columns if c not in ("anio","patron")]
    Xmat = Xdf[feats].to_numpy(dtype=float)
    yvec = target.astype(str).to_numpy()

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(random_state=seed))
    ])

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(pipe, Xmat, yvec, cv=cv, method="predict")
    y_proba = cross_val_predict(pipe, Xmat, yvec, cv=cv, method="predict_proba")

    # Fit final sobre todo el dataset
    pipe.fit(Xmat, yvec)
    classes = list(pipe.named_steps["clf"].classes_)
    return pipe, feats, classes, y_pred, y_proba

model_bundle = None

if btn_train and (X.empty or y is None):
    st.error("Faltan datos o etiquetas. Cargá al menos 1 archivo por año y definí el patrón por año.")
elif btn_train:
    with st.spinner("Entrenando..."):
        pipe, feats, classes, y_pred, y_proba = train_model(X, y, seed=seed, k=kfold)
        model_bundle = {"model": pipe, "features": feats, "classes": classes}
    st.success("Modelo entrenado.")
    st.subheader("Métricas (CV estratificado)")
    st.text(classification_report(X["patron"], y_pred, digits=3))
    st.write("Matriz de confusión:")
    st.dataframe(pd.DataFrame(confusion_matrix(X["patron"], y_pred), index=classes, columns=classes))

    # Predicciones por año (promedios de CV no triviales; mostramos del fit final)
    Xmat_all = X[feats].to_numpy(float)
    proba_all = pipe.predict_proba(Xmat_all)
    df_pred = pd.DataFrame(proba_all, columns=[f"proba_{c}" for c in classes])
    out = pd.concat([X[["anio","patron"]].reset_index(drop=True), df_pred], axis=1)
    st.subheader("Predicciones por año (probas del modelo final)")
    st.dataframe(out)

    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar predicciones (CSV)", data=csv, file_name="predicciones_patron.csv", mime="text/csv")

# ---------- Exportar modelo ----------
if btn_export:
    if model_bundle is None:
        st.warning("Primero entrená el modelo.")
    else:
        buf = io.BytesIO()
        joblib.dump(model_bundle, buf)
        st.download_button("⬇️ Descargar modelo (.joblib)", data=buf.getvalue(), file_name="modelo_patron_meteo.joblib", mime="application/octet-stream")

st.markdown("---")
st.subheader("Predicción rápida con el modelo (inferencia)")
up_pred = st.file_uploader("Subí un archivo de meteorología (1-ene → 1-may) para predecir el patrón", type=["csv","xlsx","xls"], key="pred_file")
col1, col2 = st.columns(2)
with col1:
    modelo_up = st.file_uploader("O cargá un modelo .joblib previamente entrenado", type=["joblib"])
with col2:
    base_gdd = st.number_input("Base GDD para cálculo (solo por registro)", value=5.0, step=0.5)

if st.button("Predecir patrón"):
    if modelo_up is None:
        st.error("Cargá un modelo .joblib")
    elif up_pred is None:
        st.error("Cargá el archivo de meteorología a predecir")
    else:
        try:
            bundle = joblib.load(modelo_up)
            model = bundle["model"]; feats = bundle["features"]; classes = bundle["classes"]
        except Exception as e:
            st.error(f"No se pudo cargar el modelo: {e}")
            st.stop()
        try:
            if up_pred.name.lower().endswith(".csv"):
                dfp = pd.read_csv(up_pred)
            else:
                dfp = pd.read_excel(up_pred)
            dfp = standardize_cols(dfp)
            dfp = slice_jan_to_may1(dfp)
            # recalcular features con la misma receta
            feats_row = summarize_year(dfp)
            Xnew = pd.DataFrame([feats_row])[feats].to_numpy(float)
            proba = model.predict_proba(Xnew)[0]
            idx = int(np.argmax(proba))
            st.success(f"Patrón estimado: **{classes[idx]}** (confianza {proba[idx]*100:.1f}%)")
            st.write(pd.DataFrame([dict(zip([f"proba_{c}" for c in classes], proba))]))
        except Exception as e:
            st.error(f"Error en la predicción: {e}")

