# -*- coding: utf-8 -*-
# Fine-tuning puntual con datos 2021 para PREDWEEM (JD 1..274)
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

JD_MAX = 274
MODEL_IN  = "modelo_curva_emergencia_274.joblib"
MODEL_OUT = "modelo_curva_emergencia_274_finetuned2021_2025-11-12.joblib"
METEO_2021_PATH = "2021.xlsx"
EMER_2021_PATH  = "2021 emer.xlsx"

# ---------- 1) Cargar datos ----------
meteo = pd.read_excel(METEO_2021_PATH)
meteo.columns = [str(c).lower().strip() for c in meteo.columns]
if "dia juliano" in meteo.columns:
    meteo = meteo.rename(columns={"dia juliano": "jd"})
meteo["jd"] = pd.to_numeric(meteo["jd"], errors="coerce")
meteo = meteo.dropna(subset=["jd"])
meteo["jd"] = meteo["jd"].astype(int)
meteo = (meteo.set_index("jd")
              .reindex(range(1, JD_MAX+1))
              .ffill().bfill().reset_index())

emer_week = pd.read_excel(EMER_2021_PATH)
emer_week.columns = ["fecha", "emer_rel"]
emer_week["fecha"] = pd.to_datetime(emer_week["fecha"], errors="coerce")
emer_week = emer_week.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)
emer_week["jd"] = emer_week["fecha"].dt.dayofyear

emer_week["emer_acum"] = emer_week["emer_rel"].cumsum()
if emer_week["emer_acum"].max() == 0:
    raise ValueError("La serie semanal es toda cero; no se puede normalizar.")
emer_week["emer_acum"] = emer_week["emer_acum"] / emer_week["emer_acum"].max()

jd_daily = np.arange(1, JD_MAX+1)
emer_obs_daily = np.interp(jd_daily, emer_week["jd"], emer_week["emer_acum"])

# ---------- 2) Cargar modelo original ----------
bundle = joblib.load(MODEL_IN)
xsc, ysc, mlp = bundle["xsc"], bundle["ysc"], bundle["mlp"]
lr_orig = getattr(mlp, "learning_rate_init", 1e-3)

# ---------- 3) Predicción ANTES ----------
xnew = np.concatenate([meteo["tmin"], meteo["tmax"], meteo["prec"]]).reshape(1, -1)
y_pred_before = ysc.inverse_transform(mlp.predict(xsc.transform(xnew)))[0]
y_pred_before = np.maximum.accumulate(y_pred_before)
if y_pred_before[-1] != 0: y_pred_before = y_pred_before / y_pred_before[-1]
y_pred_before = np.clip(y_pred_before, 0, 1)

rmse_before = float(np.sqrt(mean_squared_error(emer_obs_daily, y_pred_before)))
corr_before = float(np.corrcoef(emer_obs_daily, y_pred_before)[0, 1])
r2_before   = float(r2_score(emer_obs_daily, y_pred_before))

print("[ANTES]  r = %.3f | RMSE = %.3f | R2 = %.3f" % (corr_before, rmse_before, r2_before))

# ---------- 4) Fine-tuning puntual ----------
mlp.warm_start = True
old_max_iter = mlp.max_iter
old_lr = mlp.learning_rate_init
mlp.max_iter = 200
mlp.learning_rate_init = lr_orig  # mantener LR original

Y_target = emer_obs_daily.reshape(1, -1)
Y_target_s = ysc.transform(Y_target)

mlp.fit(xsc.transform(xnew), Y_target_s)

# Restaurar hiperparámetros que cambiamos temporalmente
mlp.max_iter = old_max_iter
mlp.learning_rate_init = old_lr

# ---------- 5) Predicción DESPUÉS ----------
y_pred_after = ysc.inverse_transform(mlp.predict(xsc.transform(xnew)))[0]
y_pred_after = np.maximum.accumulate(y_pred_after)
if y_pred_after[-1] != 0: y_pred_after = y_pred_after / y_pred_after[-1]
y_pred_after = np.clip(y_pred_after, 0, 1)

rmse_after = float(np.sqrt(mean_squared_error(emer_obs_daily, y_pred_after)))
corr_after = float(np.corrcoef(emer_obs_daily, y_pred_after)[0, 1])
r2_after   = float(r2_score(emer_obs_daily, y_pred_after))

print("[DESPUÉS] r = %.3f | RMSE = %.3f | R2 = %.3f" % (corr_after, rmse_after, r2_after))

# ---------- 6) Guardar modelo ajustado ----------
joblib.dump({"xsc": xsc, "ysc": ysc, "mlp": mlp}, MODEL_OUT)
print("✅ Modelo guardado en:", MODEL_OUT)

# ---------- 7) Gráfico comparativo ----------
ref_year = 2021
month_starts = pd.date_range(f"{ref_year}-02-01", f"{ref_year}-10-01", freq="MS")  # Feb→Oct
month_jd = (month_starts - pd.Timestamp(f"{ref_year}-01-01")).days + 1
# En Windows usa %#d; en Linux/Mac usa %-d — probamos ambos:
fmt = "%-d-%b"
try:
    month_lbl = [d.strftime(fmt) for d in month_starts]
except:
    month_lbl = [d.strftime("%#d-%b") for d in month_starts]

plt.figure(figsize=(11,6))
plt.plot(jd_daily, emer_obs_daily, "-", color="tab:orange", lw=2.0, label="Real 2021 (acumulada)")
plt.scatter(emer_week["jd"], emer_week["emer_acum"], color="tab:orange", s=28, alpha=0.8, label="Puntos semanales reales")
plt.plot(jd_daily, y_pred_before, "-", color="tab:blue",  lw=2.0, label=f"Predicción original (r={corr_before:.3f}, RMSE={rmse_before:.3f})")
plt.plot(jd_daily, y_pred_after,  "-", color="tab:green", lw=2.4, label=f"Predicción ajustada (r={corr_after:.3f}, RMSE={rmse_after:.3f})")
plt.xlim(1, JD_MAX); plt.ylim(0, 1.02)
plt.xlabel("Día Juliano (1–274)"); plt.ylabel("Emergencia acumulada (0–1)")
plt.title("Ajuste fino con datos 2021 — Predicción original vs ajustada vs real")
ax = plt.gca()
ax.set_xticks(month_jd); ax.set_xticklabels(month_lbl, rotation=30)
plt.grid(True, alpha=0.3); plt.legend(loc="lower right"); plt.tight_layout()
plt.show()

