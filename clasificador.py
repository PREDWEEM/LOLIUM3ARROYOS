# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Predicción de patrón a partir de datos meteorológicos
# ===============================================================
# Usa bundle P2F (clasificador + generador de curvas) para predecir
# el patrón histórico más plausible según datos meteorológicos nuevos.
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib, io, matplotlib.pyplot as plt

JD_MAX = 274

# ===============================================================
# 🔧 FUNCIONES BASE
# ===============================================================
def standardize_cols(df):
    df.columns = [str(c).lower().strip() for c in df.columns]
    ren = {
        "dia juliano": "jd", "julian_days": "jd", "día": "jd", "dia": "jd",
        "t min": "tmin", "t_min": "tmin", "temperatura mínima": "tmin",
        "t max": "tmax", "t_max": "tmax", "temperatura máxima": "tmax",
        "precipitacion": "prec", "pp": "prec", "lluvia": "prec"
    }
    for k, v in ren.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    return df

def richards(t, K, r, t0, v):
    return K / (1 + v * np.exp(-r * (t - t0))) ** (1 / v)

def two_stage_richards(t, p):
    a1, r1, t01, v1, a2, r2, t02, v2 = p
    y1 = a1 * richards(t, 1, r1, t01, v1)
    y2 = a2 * richards(t, 1, r2, t02, v2)
    y = np.maximum.accumulate(y1 + y2)
    if y.max() > 0:
        y /= y.max()
    return np.clip(y, 0, 1)

def longest_run(x):
    c = m = 0
    for v in x:
        c = c + 1 if v == 1 else 0
        m = max(m, c)
    return m

def build_features(dfm):
    dfm = standardize_cols(dfm)
    if "jd" not in dfm.columns:
        dfm["jd"] = np.arange(1, len(dfm) + 1)
    dfm = (
        dfm.set_index("jd")
        .reindex(range(1, JD_MAX + 1))
        .interpolate()
        .ffill()
        .bfill()
        .reset_index()
    )

    tmin = dfm["tmin"].to_numpy(float)
    tmax = dfm["tmax"].to_numpy(float)
    tmed = (tmin + tmax) / 2.0
    prec = dfm["prec"].to_numpy(float)
    jd = dfm["jd"].to_numpy(int)

    mask = (jd >= 32) & (jd <= 151)
    gdd5 = np.cumsum(np.maximum(tmed - 5, 0))
    gdd3 = np.cumsum(np.maximum(tmed - 3, 0))

    feat = {}
    feat["gdd5_FM"] = gdd5[mask].ptp()
    feat["gdd3_FM"] = gdd3[mask].ptp()
    pf = prec[mask]
    feat["pp_FM"] = pf.sum()
    feat["ev10_FM"] = int((pf >= 10).sum())
    feat["ev20_FM"] = int((pf >= 20).sum())
    dry = (pf < 1).astype(int)
    wet = (pf >= 5).astype(int)
    feat["dry_run_FM"] = longest_run(dry)
    feat["wet_run_FM"] = longest_run(wet)

    def ma(x, w):
        k = np.ones(w) / w
        return np.convolve(x, k, "same")

    feat["tmed14_May"] = ma(tmed, 14)[151]
    feat["tmed28_May"] = ma(tmed, 28)[151]
    feat["gdd5_120"] = gdd5[119]
    feat["pp_120"] = prec[:120].sum()
    return dfm, feat

# ===============================================================
# 🌾 STREAMLIT UI
# ===============================================================
st.set_page_config(page_title="PREDWEEM — Patrón meteorológico", layout="wide")
st.title("🌾 PREDWEEM — Predicción de patrón a partir de meteorología nueva")

st.sidebar.header("📂 Entradas requeridas")
bundle_file = st.sidebar.file_uploader("Modelo P2F (.joblib)", type=["joblib"])
meteo_option = st.sidebar.radio(
    "Fuente de meteorología:",
    ["📁 Archivo local (XLSX)", "🌐 GitHub RAW"],
)

if meteo_option == "📁 Archivo local (XLSX)":
    meteo_file = st.sidebar.file_uploader("Cargar datos meteorológicos", type=["xlsx", "xls"])
else:
    meteo_url = st.sidebar.text_input(
        "URL RAW del archivo en GitHub",
        value="https://raw.githubusercontent.com/PREDWEEM/LOLium3arroyos/main/datos%20LOLIUM%20METEORO.xlsx",
    )

btn_run = st.sidebar.button("🚀 Predecir patrón")

# ===============================================================
# 🔮 PREDICCIÓN
# ===============================================================
if btn_run:
    if not bundle_file:
        st.error("❌ Falta el modelo P2F (.joblib)")
        st.stop()

    try:
        bundle = joblib.load(bundle_file)
        xsc = bundle["xsc"]
        feat_names = bundle["feat_names"]
        clf = bundle["clf"]
        regs = bundle["regs"]
        protos = bundle["protos"]

        # --- Leer meteo ---
        if meteo_option == "📁 Archivo local (XLSX)":
            dfm = pd.read_excel(meteo_file)
        else:
            dfm = pd.read_excel(meteo_url)

        dfm, feat = build_features(dfm)
        X = np.array([[feat[k] for k in feat_names]], float)
        Xs = xsc.transform(X)

        # --- Clasificación del patrón ---
        proba = clf.predict_proba(Xs)[0]
        clases = clf.classes_
        pat = clases[np.argmax(proba)]

        # --- Predicción de parámetros ---
        regs_pat = regs[pat]
        p = np.array([r.predict(Xs)[0] for r in regs_pat])
        t = np.arange(1, JD_MAX + 1, dtype=float)
        y = two_stage_richards(t, p)
        y = 0.8 * y + 0.2 * protos[pat][:JD_MAX]

        # --- Visualización ---
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t, y, color="darkorange", lw=2.5, label=f"Curva predicha ({pat})")
        ax.plot(t, protos[pat][:JD_MAX], color="steelblue", lw=2, ls="--", alpha=0.6, label=f"Prototipo {pat}")
        ax.set_xlim(1, JD_MAX)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Día juliano (1–274)")
        ax.set_ylabel("Emergencia acumulada (0–1)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        st.pyplot(fig)

        # --- Resultados ---
        st.success(f"✅ Patrón predicho: **{pat}** (confianza {proba.max():.2f})")
        dfp = pd.DataFrame({"Patrón": clases, "Probabilidad": proba}).sort_values("Probabilidad", ascending=False)
        st.dataframe(dfp, use_container_width=True)

        # --- Descarga ---
        out = pd.DataFrame({"Día": t, "Emergencia_predicha": y})
        st.download_button(
            "⬇️ Descargar curva predicha (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name=f"curva_predicha_{pat}.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")

