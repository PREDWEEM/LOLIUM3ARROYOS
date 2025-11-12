# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Clasificador & Fine-Tuning 2021 (Streamlit)
# ===============================================================
# - Carga archivos meteorológicos y de emergencia (subida directa)
# - Normaliza encabezados (fecha, jd, Tmin, Tmax, Prec, etc.)
# - Genera 'jd' automáticamente si falta
# - Muestra curva real vs predicha original y ajustada
# - Calcula métricas y permite descargar el modelo calibrado
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io, joblib
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# ⚙️ CONFIGURACIÓN
# ===============================
st.set_page_config(page_title="🌾 PREDWEEM — Clasificador 2021", layout="wide")
st.title("🌾 PREDWEEM — Clasificador y Ajuste fino (2021)")

JD_MAX = 274

# ===============================
# 🧹 NORMALIZADOR DE COLUMNAS
# ===============================
def standardize_cols(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    ren = {
        "t min": "tmin", "t_min": "tmin", "temperatura mínima": "tmin", "tminima": "tmin",
        "t max": "tmax", "t_max": "tmax", "temperatura máxima": "tmax", "tmaxima": "tmax",
        "precip": "prec", "lluvia": "prec", "pp": "prec", "precipitacion": "prec",
        "dia juliano": "jd", "día juliano": "jd", "dia": "jd", "día": "jd",
        "julian_days": "jd", "n_dia": "jd", "diajul": "jd", "diajuliano": "jd",
        "fecha": "fecha", "date": "fecha"
    }
    for k, v in ren.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    return df

# ===============================
# 📂 CARGA DE ARCHIVOS
# ===============================
st.sidebar.header("📂 Archivos de entrada")

meteo_file = st.sidebar.file_uploader("Archivo meteorológico (2021)", type=["xlsx", "xls"])
emer_file  = st.sidebar.file_uploader("Archivo de emergencia (2021)", type=["xlsx", "xls"])
modelo_file = st.sidebar.file_uploader("Modelo original (.joblib)", type=["joblib"])

btn_run = st.sidebar.button("🚀 Ejecutar ajuste fino")

# ===============================
# 🧭 PROCESAMIENTO
# ===============================
if btn_run:
    if not all([meteo_file, emer_file, modelo_file]):
        st.error("⚠️ Cargá los tres archivos: meteorología, emergencia y modelo.")
        st.stop()

    try:
        # --- Lectura
        meteo = pd.read_excel(meteo_file)
        emer_week = pd.read_excel(emer_file)
        meteo = standardize_cols(meteo)
        emer_week = standardize_cols(emer_week)

        # --- JD automático
        if "jd" not in meteo.columns:
            if "fecha" in meteo.columns:
                meteo["fecha"] = pd.to_datetime(meteo["fecha"], errors="coerce")
                meteo = meteo.dropna(subset=["fecha"])
                meteo["jd"] = meteo["fecha"].dt.dayofyear
                st.info("⚙️ Columna 'jd' generada desde 'fecha'.")
            else:
                meteo["jd"] = np.arange(1, len(meteo) + 1)
                st.info("⚙️ Columna 'jd' creada secuencialmente (1..n).")

        meteo["jd"] = pd.to_numeric(meteo["jd"], errors="coerce")
        meteo = meteo.dropna(subset=["jd"]).astype({"jd": int})
        meteo = (meteo.set_index("jd")
                      .reindex(range(1, JD_MAX + 1))
                      .ffill().bfill().reset_index())

        # --- Emergencia
        emer_week["fecha"] = pd.to_datetime(emer_week["fecha"], errors="coerce")
        emer_week = emer_week.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)
        emer_week["jd"] = emer_week["fecha"].dt.dayofyear
        num_cols = emer_week.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            st.error("❌ No se encontró columna numérica de emergencia relativa.")
            st.stop()
        emer_week["emer_rel"] = emer_week[num_cols[0]]
        emer_week["emer_acum"] = emer_week["emer_rel"].cumsum()
        emer_week["emer_acum"] /= emer_week["emer_acum"].max()
        jd_daily = np.arange(1, JD_MAX + 1)
        emer_obs_daily = np.interp(jd_daily, emer_week["jd"], emer_week["emer_acum"])

        # --- Cargar modelo
        bundle = joblib.load(modelo_file)
        xsc, ysc, mlp = bundle["xsc"], bundle["ysc"], bundle["mlp"]
        lr_orig = getattr(mlp, "learning_rate_init", 1e-3)

        # --- Predicción original
        xnew = np.concatenate([meteo["tmin"], meteo["tmax"], meteo["prec"]]).reshape(1, -1)
        y_pred_before = ysc.inverse_transform(mlp.predict(xsc.transform(xnew)))[0]
        y_pred_before = np.maximum.accumulate(y_pred_before)
        y_pred_before /= y_pred_before[-1] if y_pred_before[-1] != 0 else 1
        y_pred_before = np.clip(y_pred_before, 0, 1)

        rmse_before = float(np.sqrt(mean_squared_error(emer_obs_daily, y_pred_before)))
        corr_before = float(np.corrcoef(emer_obs_daily, y_pred_before)[0, 1])
        r2_before = float(r2_score(emer_obs_daily, y_pred_before))

        # --- Fine-tuning puntual
        mlp.warm_start = True
        old_max_iter = mlp.max_iter
        old_lr = mlp.learning_rate_init
        mlp.max_iter = 200
        mlp.learning_rate_init = lr_orig

        Y_target = emer_obs_daily.reshape(1, -1)
        Y_target_s = ysc.transform(Y_target)
        mlp.fit(xsc.transform(xnew), Y_target_s)

        mlp.max_iter = old_max_iter
        mlp.learning_rate_init = old_lr

        # --- Predicción ajustada
        y_pred_after = ysc.inverse_transform(mlp.predict(xsc.transform(xnew)))[0]
        y_pred_after = np.maximum.accumulate(y_pred_after)
        y_pred_after /= y_pred_after[-1] if y_pred_after[-1] != 0 else 1
        y_pred_after = np.clip(y_pred_after, 0, 1)

        rmse_after = float(np.sqrt(mean_squared_error(emer_obs_daily, y_pred_after)))
        corr_after = float(np.corrcoef(emer_obs_daily, y_pred_after)[0, 1])
        r2_after = float(r2_score(emer_obs_daily, y_pred_after))

        # --- Mostrar métricas
        st.success(f"✅ Fine-tuning completado con éxito.")
        st.markdown(f"""
        **Antes:** r={corr_before:.3f}, RMSE={rmse_before:.3f}, R²={r2_before:.3f}  
        **Después:** r={corr_after:.3f}, RMSE={rmse_after:.3f}, R²={r2_after:.3f}
        """)

        # --- Gráfico comparativo
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(jd_daily, emer_obs_daily, "-", color="tab:orange", lw=2.0, label="Real 2021 (acumulada)")
        ax.scatter(emer_week["jd"], emer_week["emer_acum"], color="tab:orange", s=28, alpha=0.8, label="Puntos semanales")
        ax.plot(jd_daily, y_pred_before, "-", color="tab:blue", lw=2.0, label=f"Predicción original (r={corr_before:.3f})")
        ax.plot(jd_daily, y_pred_after, "-", color="tab:green", lw=2.4, label=f"Predicción ajustada (r={corr_after:.3f})")
        ax.set_xlim(1, JD_MAX)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Día Juliano (1–274)")
        ax.set_ylabel("Emergencia acumulada (0–1)")
        ax.set_title("🌾 PREDWEEM — Ajuste fino 2021")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        st.pyplot(fig)

        # --- Descargar modelo ajustado
        buf = io.BytesIO()
        joblib.dump({"xsc": xsc, "ysc": ysc, "mlp": mlp}, buf)
        st.download_button(
            "💾 Descargar modelo ajustado (.joblib)",
            data=buf.getvalue(),
            file_name="modelo_curva_emergencia_274_finetuned2021_2025-11-12.joblib",
            mime="application/octet-stream"
        )

    except Exception as e:
        st.error(f"❌ Error durante el procesamiento: {e}")
