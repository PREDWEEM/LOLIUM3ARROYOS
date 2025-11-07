# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Predicción de patrón histórico (diagnóstico al 1 de mayo)
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="🌾 PREDWEEM — Patrón histórico", layout="wide")

st.title("🌾 PREDWEEM — Diagnóstico del patrón histórico de emergencia")
st.markdown("""
Esta herramienta analiza los resultados del modelo **PREDWEEM** (archivo CSV con columnas `Fecha`, `Julian_days`, `Nivel de EMERREL`, `EMEAC (%)`)
y predice con alta precisión el **patrón histórico de emergencia** (P1, P1b, P2 o P3),
utilizando la información disponible hasta el **1 de mayo (JD≈121)**.
""")

# ===============================================================
# 📂 CARGA DE ARCHIVO
# ===============================================================
uploaded = st.file_uploader("📤 Cargá el archivo CSV de resultados de emergencia PREDWEEM", type=["csv"])
if uploaded is None:
    st.info("Esperando archivo... Subí tu CSV con los resultados del modelo.")
    st.stop()

# ===============================================================
# 🧾 LECTURA Y PROCESAMIENTO
# ===============================================================
df = pd.read_csv(uploaded)
df.columns = [c.strip() for c in df.columns]
df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
col_ac = [c for c in df.columns if "EMEAC" in c.upper()][0]
df["Emer_AC"] = pd.to_numeric(df[col_ac], errors="coerce")
if df["Emer_AC"].max() > 1.01:
    df["Emer_AC"] /= 100.0
df["Emer_Rel"] = df["Emer_AC"].diff().fillna(0).clip(lower=0)
df["JD"] = df["Fecha"].dt.dayofyear

# ===============================================================
# 📆 CORTE AL 1 DE MAYO
# ===============================================================
fecha_corte = datetime(df["Fecha"].dt.year.iloc[0], 5, 1)
jd_corte = 121
df_corte = df[df["JD"] <= jd_corte]

st.subheader(f"📅 Análisis hasta el 1 de mayo (JD {jd_corte})")
st.write(f"Período analizado: {df_corte['Fecha'].min().date()} → {df_corte['Fecha'].max().date()}")
st.write(f"Días considerados: {len(df_corte)}")

# ===============================================================
# 🧠 CLASIFICADOR DE PATRONES
# ===============================================================
def normalize(v): s = v.sum(); return v / s if s > 0 else v

def features(df):
    jd, rel = df["JD"].values, df["Emer_Rel"].values
    thr = 0.3 * (rel.max() if len(rel) else 0)
    peaks = np.where((rel[1:-1] > rel[:-2]) & (rel[1:-1] > rel[2:]) & (rel[1:-1] >= thr))[0] + 1
    n = len(peaks)
    jd50 = np.interp(0.5, df["Emer_AC"], jd) if df["Emer_AC"].max() > 0.5 else np.nan
    late = normalize(rel)[jd > 160].sum()
    return dict(n_peaks=n, jd50=float(jd50), late_share=float(late))

def classify(f):
    n, jd50, late = f["n_peaks"], f["jd50"], f["late_share"]
    if n >= 3 or late > 0.20: return "P3"
    if n == 2 and jd50 < 120: return "P2"
    if n >= 2 or late > 0.05: return "P1b"
    return "P1"

feat = features(df_corte)
patron_pred = classify(feat)

st.success(f"**Patrón histórico estimado al 1 de mayo:** {patron_pred}")
st.json(feat)

# ===============================================================
# 📈 DEFINICIÓN DE PATRONES HISTÓRICOS CANÓNICOS
# ===============================================================
JD = np.arange(1, 301)

def gaussian(x, mu, sigma, amp): return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def normalize_shape(v): return v / v.sum()

# Curvas canónicas
shape_P1  = normalize_shape(gaussian(JD, 70, 10, 1.0))
shape_P1b = normalize_shape(gaussian(JD, 85, 12, 0.85) + gaussian(JD, 200, 25, 0.25))
shape_P2  = normalize_shape(gaussian(JD, 60, 6, 1.0) + gaussian(JD, 140, 12, 0.3))
shape_P3  = normalize_shape(gaussian(JD, 100, 15, 0.35) + gaussian(JD, 160, 20, 0.30)
                            + gaussian(JD, 220, 22, 0.25) + gaussian(JD, 270, 10, 0.20))

patterns = {"P1": shape_P1, "P1b": shape_P1b, "P2": shape_P2, "P3": shape_P3}
colors   = {"P1": "tab:blue", "P1b": "tab:orange", "P2": "tab:green", "P3": "tab:red"}

# ===============================================================
# 📊 GRAFICO COMPARATIVO
# ===============================================================
fig, ax = plt.subplots(figsize=(10,5))

# Emergencia real
ax.plot(df["JD"], df["Emer_AC"], color="black", lw=2.5, label="Emergencia acumulada (real)")
ax.bar(df["JD"], df["Emer_Rel"], color="gray", alpha=0.3, label="Emergencia diaria")

# Línea de diagnóstico (1 mayo)
ax.axvline(jd_corte, color="orange", ls="--", lw=2, label="Diagnóstico (1 mayo)")

# Patrones históricos
for k, v in patterns.items():
    ax.plot(JD, np.cumsum(v), color=colors[k], lw=1.5, alpha=0.5,
            label=f"Patrón {k}")

ax.set_xlim(0, min(300, df["JD"].max() + 10))
ax.set_ylim(0, 1.05)
ax.set_xlabel("Día juliano (JD)")
ax.set_ylabel("Emergencia acumulada")
ax.set_title(f"Patrón estimado: {patron_pred}")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, ls="--", alpha=0.4)

st.pyplot(fig)

# ===============================================================
# 📤 EXPORTACIÓN
# ===============================================================
resumen = pd.DataFrame([{
    "Fecha_corte": fecha_corte.date(),
    "JD_corte": jd_corte,
    "Patron_predicho": patron_pred,
    **feat
}])

csv_out = resumen.to_csv(index=False).encode("utf-8")
st.download_button("💾 Descargar resumen de diagnóstico", data=csv_out,
                   file_name="prediccion_patron_1mayo.csv", mime="text/csv")

st.caption("Versión PREDWEEM v2.0 — Diagnóstico de patrones fenológicos con comparación histórica.")

