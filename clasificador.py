# -*- coding: utf-8 -*-
# 🌾 PREDWEEM — Clasificador de patrón futuro (LOLium) a partir de meteorología (1-ene → 1-may, hojas por año)
import streamlit as st
import pandas as pd
import numpy as np
import io, re, joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="PREDWEEM · Clasificador LOLIUM", layout="wide")
st.title("🌾 Clasificador de patrón futuro de emergencia — LOLIUM (1-ene → 1-may)")

# ==============================
# 🔹 Funciones auxiliares
# ==============================
def standardize_cols(df):
    mapping = {
        "fecha": ["fecha", "date"],
        "jd": ["dia juliano", "julian_days", "jd"],
        "tmin": ["temperatura minima", "tmin", "t_min"],
        "tmax": ["temperatura maxima", "tmax", "t_max"],
        "prec": ["precipitacion", "pp", "rain", "prec"]
    }
    df.columns = [c.strip().lower() for c in df.columns]
    renames = {}
    for key, aliases in mapping.items():
        for al in aliases:
            if al in df.columns:
                renames[al] = key
                break
    df = df.rename(columns=renames)
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
    return df

def slice_jan_to_may(df):
    if "fecha" in df.columns:
        y = int(df["fecha"].dt.year.mode().iloc[0])
        start, end = pd.Timestamp(y,1,1), pd.Timestamp(y,5,1)
        return df[(df["fecha"]>=start)&(df["fecha"]<=end)]
    elif "jd" in df.columns:
        return df[(df["jd"]>=1)&(df["jd"]<=121)]
    return df

def gdd(tmin, tmax, base):
    tmean = (tmin + tmax) / 2.0
    return np.clip(tmean - base, 0, None).sum()

def rain_events(prec, thr=1.0):
    return np.sum(np.nan_to_num(prec,0) > thr)

def longest_dry_spell(prec, thr=1.0):
    p = np.nan_to_num(prec,0)
    longest = curr = 0
    for v in p:
        if v <= thr:
            curr += 1
            longest = max(longest, curr)
        else:
            curr = 0
    return longest

def summarize_year(df):
    tmin, tmax, prec = df["tmin"], df["tmax"], df["prec"]
    f = {}
    f["tmin_mean"] = np.nanmean(tmin)
    f["tmax_mean"] = np.nanmean(tmax)
    f["gdd_b5"]  = gdd(tmin,tmax,5)
    f["gdd_b10"] = gdd(tmin,tmax,10)
    f["pp_sum"]  = np.nansum(prec)
    f["pp_events"] = rain_events(prec,1)
    f["dryspell_max"] = longest_dry_spell(prec,1)
    f["pp_p95"]  = np.nanpercentile(prec,95)
    f["pp_days_gt10"] = int(np.sum(np.nan_to_num(prec,0)>10))
    f["pp_mean"] = np.nanmean(prec)
    f["pp_sum_per_event"] = f["pp_sum"]/(f["pp_events"]+1e-6)
    return f

# ==============================
# 🔹 Cargar archivo de meteorología
# ==============================
uploaded = st.file_uploader("📂 Subí el archivo meteorológico con todas las hojas (una por año)", type=["xlsx","xls"])

if uploaded:
    try:
        sheets = pd.read_excel(uploaded, sheet_name=None)
    except Exception as e:
        st.error(f"Error leyendo archivo: {e}")
        st.stop()

    rows = []
    st.success(f"Archivo cargado con {len(sheets)} hojas (años detectados).")
    for nombre, df in sheets.items():
        df = standardize_cols(df)
        df = slice_jan_to_may(df)
        try:
            if "fecha" in df.columns:
                y = int(df["fecha"].dt.year.mode().iloc[0])
            else:
                y = int(nombre)
        except:
            y = int(re.sub(r"[^0-9]", "", nombre)) if re.search(r"[0-9]+", nombre) else None
        if y is None:
            continue
        f = summarize_year(df)
        f["anio"] = y
        rows.append(f)

    X = pd.DataFrame(rows).sort_values("anio")
    st.subheader("📊 Variables meteorológicas resumen (1-ene → 1-may)")
    st.dataframe(X.set_index("anio"))

    # ===========================
    # 🔹 Cargar o asignar patrones
    # ===========================
    st.markdown("---")
    st.subheader("🎯 Asignación de patrones históricos (P1, P1b, P2, P3)")
    patrones = ["P1","P1b","P2","P3"]
    pat_dict = {}
    for y in X["anio"]:
        pat_dict[y] = st.selectbox(f"Año {y}", patrones, key=f"pat_{y}")
    X["patron"] = X["anio"].map(pat_dict)

    # ===========================
    # 🔹 Entrenamiento
    # ===========================
    if st.button("🚀 Entrenar modelo"):
        feats = [c for c in X.columns if c not in ("anio","patron")]
        Xmat = X[feats].to_numpy(float)
        y = X["patron"].astype(str)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(random_state=42))
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(pipe, Xmat, y, cv=cv)
        pipe.fit(Xmat, y)
        st.success("✅ Modelo entrenado con validación cruzada (5-fold)")
        st.text(classification_report(y, y_pred))
        st.write("Matriz de confusión:")
        st.dataframe(pd.DataFrame(confusion_matrix(y, y_pred), index=np.unique(y), columns=np.unique(y)))

        # Guardar modelo
        model_bundle = {"model": pipe, "features": feats, "classes": list(pipe.named_steps["clf"].classes_)}
        buf = io.BytesIO()
        joblib.dump(model_bundle, buf)
        st.download_button("⬇️ Descargar modelo entrenado (.joblib)", data=buf.getvalue(),
                           file_name="modelo_patron_lolium.joblib", mime="application/octet-stream")

    # ===========================
    # 🔹 Predicción nueva
    # ===========================
    st.markdown("---")
    st.subheader("🔮 Predicción de patrón para un nuevo año")
    new_file = st.file_uploader("Subí un archivo de meteorología (1-ene → 1-may)", type=["xlsx","xls","csv"], key="pred_new")
    model_up = st.file_uploader("Cargá el modelo entrenado (.joblib)", type=["joblib"])

    if st.button("🔎 Predecir patrón"):
        if not new_file or not model_up:
            st.error("Cargá ambos archivos.")
        else:
            try:
                bundle = joblib.load(model_up)
                model, feats, classes = bundle["model"], bundle["features"], bundle["classes"]
                if new_file.name.endswith(".csv"):
                    dfp = pd.read_csv(new_file)
                else:
                    dfp = pd.read_excel(new_file)
                dfp = standardize_cols(dfp)
                dfp = slice_jan_to_may(dfp)
                feats_row = summarize_year(dfp)
                Xnew = pd.DataFrame([feats_row])[feats].to_numpy(float)
                proba = model.predict_proba(Xnew)[0]
                pred = classes[np.argmax(proba)]
                st.success(f"🌾 Patrón estimado: **{pred}** (confianza {proba.max()*100:.1f}%)")
                st.dataframe(pd.DataFrame([dict(zip(classes, proba))]))
            except Exception as e:
                st.error(f"Error en predicción: {e}")

