# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Clasificador y Ajuste Fino 2021 (robusto)
# ===============================================================
# - Busca automáticamente los archivos meteorológicos y de emergencia
# - Carga y normaliza los datos (corrige mayúsculas, acentos, nombres)
# - Ajusta el modelo .joblib original con los datos 2021 (fine-tuning puntual)
# - Muestra curva real vs predicha original vs ajustada (con fechas y puntos semanales)
# - Guarda el nuevo modelo calibrado como modelo_curva_emergencia_274_finetuned2021_2025-11-12.joblib
# ===============================================================

import os, re, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# ⚙️ CONFIGURACIÓN GENERAL
# ===============================
JD_MAX = 274
BASE_DIR = Path(__file__).parent
MODEL_IN  = BASE_DIR / "modelo_curva_emergencia_274.joblib"
MODEL_OUT = BASE_DIR / "modelo_curva_emergencia_274_finetuned2021_2025-11-12.joblib"

# ===============================
# 🔍 FUNCIÓN ROBUSTA DE BÚSQUEDA
# ===============================
def find_excel_file(patterns, search_dir="."):
    """
    Busca archivos Excel que contengan alguno de los patrones.
    Ej: find_excel_file(["meteo", "2021"], BASE_DIR)
    """
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.lower().endswith((".xlsx", ".xls")):
                for pat in patterns:
                    if re.search(pat.lower(), f.lower()):
                        return Path(root) / f
    return None

# ===============================
# 📦 DETECCIÓN AUTOMÁTICA DE ARCHIVOS
# ===============================
meteo_path = find_excel_file(["meteo", "2021"], BASE_DIR)
emer_path  = find_excel_file(["emer", "emergencia", "2021"], BASE_DIR)

if not meteo_path:
    raise FileNotFoundError("❌ No se encontró el archivo meteorológico (debe contener 'meteo' o '2021').")
if not emer_path:
    raise FileNotFoundError("❌ No se encontró el archivo de emergencia (debe contener 'emer' o 'emergencia' o '2021').")

print(f"✅ Archivo meteorológico encontrado: {meteo_path.name}")
print(f"✅ Archivo de emergencia encontrado: {emer_path.name}")

# ===============================
# 🧹 NORMALIZACIÓN DE COLUMNAS
# ===============================
def standardize_cols(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    ren = {
        "t min": "tmin", "t_min": "tmin", "temperatura mínima": "tmin", "tminima": "tmin",
        "t max": "tmax", "t_max": "tmax", "temperatura máxima": "tmax", "tmaxima": "tmax",
        "precip": "prec", "lluvia": "prec", "pp": "prec", "precipitacion": "prec",
        "dia juliano": "jd", "día juliano": "jd", "dia": "jd", "día": "jd", "julian_days": "jd",
        "fecha": "fecha"
    }
    for k, v in ren.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    return df

# ===============================
# 📂 CARGA DE DATOS
# ===============================
meteo = pd.read_excel(meteo_path)
emer_week = pd.read_excel(emer_path)

meteo = standardize_cols(meteo)
emer_week = standardize_cols(emer_week)

# ===============================
# 📊 LIMPIEZA Y AJUSTE
# ===============================
# Meteorología
if "jd" not in meteo.columns:
    raise ValueError("El archivo meteorológico debe tener columna de día juliano (jd).")
meteo["jd"] = pd.to_numeric(meteo["jd"], errors="coerce")
meteo = meteo.dropna(subset=["jd"])
meteo["jd"] = meteo["jd"].astype(int)
meteo = (meteo.set_index("jd")
              .reindex(range(1, JD_MAX + 1))
              .ffill().bfill().reset_index())

# Emergencia semanal
if "fecha" not in emer_week.columns:
    emer_week.insert(0, "fecha", np.nan)
emer_week["fecha"] = pd.to_datetime(emer_week["fecha"], errors="coerce")
emer_week = emer_week.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)
emer_week["jd"] = emer_week["fecha"].dt.dayofyear
emer_week.columns = ["fecha", "emer_rel", "jd"] if len(emer_week.columns) == 3 else emer_week.columns

emer_week["emer_acum"] = emer_week["emer_rel"].cumsum()
emer_week["emer_acum"] /= emer_week["emer_acum"].max()

jd_daily = np.arange(1, JD_MAX + 1)
emer_obs_daily = np.interp(jd_daily, emer_week["jd"], emer_week["emer_acum"])

# ===============================
# 🔮 CARGA Y AJUSTE DEL MODELO
# ===============================
print("📦 Cargando modelo original...")
bundle = joblib.load(MODEL_IN)
xsc, ysc, mlp = bundle["xsc"], bundle["ysc"], bundle["mlp"]
lr_orig = getattr(mlp, "learning_rate_init", 1e-3)

# Predicción antes
xnew = np.concatenate([meteo["tmin"], meteo["tmax"], meteo["prec"]]).reshape(1, -1)
y_pred_before = ysc.inverse_transform(mlp.predict(xsc.transform(xnew)))[0]
y_pred_before = np.maximum.accumulate(y_pred_before)
y_pred_before /= y_pred_before[-1] if y_pred_before[-1] != 0 else 1
y_pred_before = np.clip(y_pred_before, 0, 1)

# Métricas antes
rmse_before = float(np.sqrt(mean_squared_error(emer_obs_daily, y_pred_before)))
corr_before = float(np.corrcoef(emer_obs_daily, y_pred_before)[0, 1])
r2_before = float(r2_score(emer_obs_daily, y_pred_before))
print(f"📊 Antes: r={corr_before:.3f} | RMSE={rmse_before:.3f} | R²={r2_before:.3f}")

# Fine-tuning
print("🔧 Ejecutando fine-tuning puntual con 2021...")
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

# Predicción después
y_pred_after = ysc.inverse_transform(mlp.predict(xsc.transform(xnew)))[0]
y_pred_after = np.maximum.accumulate(y_pred_after)
y_pred_after /= y_pred_after[-1] if y_pred_after[-1] != 0 else 1
y_pred_after = np.clip(y_pred_after, 0, 1)

rmse_after = float(np.sqrt(mean_squared_error(emer_obs_daily, y_pred_after)))
corr_after = float(np.corrcoef(emer_obs_daily, y_pred_after)[0, 1])
r2_after = float(r2_score(emer_obs_daily, y_pred_after))
print(f"📈 Después: r={corr_after:.3f} | RMSE={rmse_after:.3f} | R²={r2_after:.3f}")

# ===============================
# 💾 GUARDAR MODELO AJUSTADO
# ===============================
joblib.dump({"xsc": xsc, "ysc": ysc, "mlp": mlp}, MODEL_OUT)
print(f"💾 Modelo ajustado guardado como: {MODEL_OUT.name}")

# ===============================
# 📉 GRAFICAR RESULTADOS
# ===============================
ref_year = 2021
month_starts = pd.date_range(f"{ref_year}-02-01", f"{ref_year}-10-01", freq="MS")
month_jd = (month_starts - pd.Timestamp(f"{ref_year}-01-01")).days + 1
try:
    month_lbl = [d.strftime("%-d-%b") for d in month_starts]
except:
    month_lbl = [d.strftime("%#d-%b") for d in month_starts]

plt.figure(figsize=(11,6))
plt.plot(jd_daily, emer_obs_daily, "-", color="tab:orange", lw=2.0, label="Real 2021 (acumulada)")
plt.scatter(emer_week["jd"], emer_week["emer_acum"], color="tab:orange", s=28, alpha=0.8, label="Puntos semanales")
plt.plot(jd_daily, y_pred_before, "-", color="tab:blue", lw=2.0, label=f"Predicción original (r={corr_before:.3f}, RMSE={rmse_before:.3f})")
plt.plot(jd_daily, y_pred_after, "-", color="tab:green", lw=2.4, label=f"Predicción ajustada (r={corr_after:.3f}, RMSE={rmse_after:.3f})")
plt.xlim(1, JD_MAX)
plt.ylim(0, 1.02)
plt.xlabel("Día Juliano (1–274)")
plt.ylabel("Emergencia acumulada (0–1)")
plt.title("🌾 Ajuste fino 2021 — Curva real vs predicha original y ajustada")
ax = plt.gca()
ax.set_xticks(month_jd)
ax.set_xticklabels(month_lbl, rotation=30)
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
