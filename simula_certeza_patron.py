# simula_certeza_patron.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# UTILIDADES DE CLASIFICACIÓN (idénticas a pattern_update.py)
# =====================================================
def to_julian(d):
    d = pd.to_datetime(d)
    return d.dt.dayofyear

def rel_from_ac(df):
    ac = df["Emer_AC"].clip(0, 1).values
    rel = np.diff(np.r_[0, ac])
    rel[rel < 0] = 0
    df["Emer_Rel"] = rel
    return df

def normalize_daily(rel):
    s = rel.sum()
    return rel / s if s > 0 else rel

def pattern_features(df):
    jd = to_julian(df["Fecha"])
    rel = df["Emer_Rel"].values
    thr = 0.30 * (rel.max() if len(rel) else 0)
    peaks = np.where((rel[1:-1] > rel[:-2]) & (rel[1:-1] > rel[2:]) & (rel[1:-1] >= thr))[0] + 1
    n_peaks = len(peaks)
    jd50 = np.interp(0.5, df["Emer_AC"], jd) if df["Emer_AC"].max() > 0.5 else np.nan
    late_share = normalize_daily(rel)[jd > 160].sum()
    return dict(n_peaks=n_peaks, jd50=float(jd50), late_share=float(late_share))

def classify_pattern(feat):
    n, jd50, late = feat["n_peaks"], feat["jd50"], feat["late_share"]
    if n >= 3 or late > 0.20: return "P3"
    if n == 2 and (jd50 < 120): return "P2"
    if n >= 2 or late > 0.05:  return "P1b"
    return "P1"

# =====================================================
# 1. LEER ARCHIVO HISTÓRICO COMPLETO (AÑO ACTUAL FINALIZADO)
# =====================================================
df = pd.read_csv("/mount/src/lolium3arroyos/data/historico_pronostico_resultados_rango.csv")
df.columns = [c.strip() for c in df.columns]
df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
col_ac = [c for c in df.columns if "EMEAC" in c.upper()][0]
df["Emer_AC"] = pd.to_numeric(df[col_ac], errors="coerce")
if df["Emer_AC"].max() > 1.01:
    df["Emer_AC"] /= 100.0
df = rel_from_ac(df)

# Clasificación real (curva completa)
feat_final = pattern_features(df)
patron_real = classify_pattern(feat_final)
print(f"✅ Patrón real del año: {patron_real}")

# =====================================================
# 2. SIMULAR CORTES TEMPORALES (cada 10 días)
# =====================================================
jd_max = int(df["Fecha"].dt.dayofyear.max())
intervalos = np.arange(40, jd_max, 10)
resultados = []

for jd_corte in intervalos:
    df_corte = df[df["Fecha"].dt.dayofyear <= jd_corte].copy()
    if len(df_corte) < 10: continue
    feat = pattern_features(df_corte)
    label = classify_pattern(feat)
    prob_aprox = (1 - abs(feat.get("late_share",0) - 0.1)) * 0.9  # proxy simple
    certeza = 1.0 if label == patron_real else 0.0
    resultados.append({
        "JD_corte": jd_corte,
        "Fecha_corte": df_corte["Fecha"].iloc[-1],
        "Patron_pred": label,
        "Certeza_pred": certeza,
        "Prob_aprox": prob_aprox
    })

df_res = pd.DataFrame(resultados)

# =====================================================
# 3. GRAFICAR CURVA DE CERTEZA TEMPORAL
# =====================================================
plt.figure(figsize=(10,5))
plt.plot(df_res["Fecha_corte"], df_res["Prob_aprox"], "o-", label="Probabilidad aproximada de acierto")
plt.scatter(df_res["Fecha_corte"], df_res["Certeza_pred"], c="red", label="Acierto real (1=correcto)")
plt.title(f"Certeza temporal de predicción del patrón histórico — Patrón real: {patron_real}")
plt.ylabel("Probabilidad / Acierto")
plt.xlabel("Fecha de corte")
plt.grid(True, ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("certeza_temporal_patron.png", dpi=300)
print("✅ Análisis completado — gráfico guardado como 'certeza_temporal_patron.png'")

