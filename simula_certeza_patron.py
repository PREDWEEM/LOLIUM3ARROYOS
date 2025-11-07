# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Simulador de certeza temporal del patrón histórico
# ===============================================================
# Calcula la evolución temporal de la certeza de clasificación
# del patrón (P1, P1b, P2, P3) según la fecha de corte.
# Incluye detección automática de la fecha óptima y etiquetas visuales.
# ---------------------------------------------------------------

import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===============================================================
# 🔍 BLOQUE ROBUSTO DE LECTURA
# ===============================================================
def find_csv_candidates(patterns, roots):
    cands = []
    for root in roots:
        for pat in patterns:
            cands += glob.glob(str(Path(root) / pat), recursive=True)
    uniq = []
    seen = set()
    for p in cands:
        q = os.path.abspath(p)
        if q not in seen:
            uniq.append(q)
            seen.add(q)
    return uniq

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOTS = [
    HERE, HERE / "data",
    HERE.parent, HERE.parent / "data",
    Path("/mount/src/lolium3arroyos/data"),
    Path("/mnt/data"),
]
PATTERNS = [
    "*historico*pronostico*resultados*rango*.csv",
    "*Histórico*Pronóstico*resultados*rango*.csv",
    "*rango*.csv",
]

CANDS = find_csv_candidates(PATTERNS, ROOTS)
if not CANDS:
    print("No se encontró el archivo CSV. Carpetas exploradas:")
    for r in ROOTS:
        try:
            print(f"📁 {r} → {[p.name for p in Path(r).glob('*')]}")
        except: pass
    raise FileNotFoundError("❌ No se encontró el archivo de resultados históricos.")

CSV_PATH = CANDS[0]
print(f"✅ Usando archivo: {CSV_PATH}")

# ===============================================================
# 📈 LECTURA Y PREPARACIÓN DE DATOS
# ===============================================================
df = pd.read_csv(CSV_PATH, encoding="utf-8")
df.columns = [c.strip() for c in df.columns]
df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

col_ac = [c for c in df.columns if "EMEAC" in c.upper()][0]
df["Emer_AC"] = pd.to_numeric(df[col_ac], errors="coerce")
if df["Emer_AC"].max() > 1.01:
    df["Emer_AC"] /= 100.0

ac = df["Emer_AC"].clip(0,1).values
rel = np.diff(np.r_[0, ac])
rel[rel < 0] = 0
df["Emer_Rel"] = rel
df["JD"] = df["Fecha"].dt.dayofyear

# ===============================================================
# 🧠 CLASIFICADOR DE PATRONES
# ===============================================================
def normalize(v): s = v.sum(); return v / s if s > 0 else v
def features(df):
    jd, rel = df["JD"].values, df["Emer_Rel"].values
    thr = 0.30 * (rel.max() if len(rel) else 0)
    peaks = np.where((rel[1:-1]>rel[:-2])&(rel[1:-1]>rel[2:])&(rel[1:-1]>=thr))[0]+1
    n = len(peaks)
    jd50 = np.interp(0.5, df["Emer_AC"], jd) if df["Emer_AC"].max()>0.5 else np.nan
    late = normalize(rel)[jd>160].sum()
    return dict(n_peaks=n, jd50=float(jd50), late_share=float(late))

def classify(f):
    n, jd50, late = f["n_peaks"], f["jd50"], f["late_share"]
    if n>=3 or late>0.20: return "P3"
    if n==2 and jd50<120: return "P2"
    if n>=2 or late>0.05: return "P1b"
    return "P1"

# ===============================================================
# ⚙️ SIMULACIÓN TEMPORAL
# ===============================================================
feat_full = features(df)
pat_real = classify(feat_full)
print(f"🌾 Patrón real del año: {pat_real}")

jd_max = int(df["JD"].max())
intervalos = np.arange(40, jd_max, 10)
resultados = []

for jd_corte in intervalos:
    sub = df[df["JD"] <= jd_corte]
    if len(sub)<10: continue
    feat = features(sub)
    label = classify(feat)
    certeza = 1 if label==pat_real else 0
    prob_aprox = min(1.0, df.loc[df["JD"]<=jd_corte,"Emer_AC"].max()*1.3)
    resultados.append({
        "JD_corte": jd_corte,
        "Fecha_corte": sub["Fecha"].iloc[-1],
        "Patron_pred": label,
        "Certeza_pred": certeza,
        "Prob_aprox": prob_aprox
    })

df_res = pd.DataFrame(resultados)

# ===============================================================
# 🧭 FECHA ÓPTIMA
# ===============================================================
fecha_opt = None
if not df_res.empty:
    df_res["Estable"] = df_res["Patron_pred"].eq(df_res["Patron_pred"].shift())
    opt = df_res[(df_res["Prob_aprox"]>=0.8) & (df_res["Estable"])]
    if not opt.empty:
        fecha_opt = opt.iloc[0]["Fecha_corte"]
        print(f"📅 Fecha óptima detectada: {fecha_opt.date()} (prob ≥ 0.8)")

# ===============================================================
# 📊 GRÁFICO
# ===============================================================
plt.figure(figsize=(11,5))
plt.plot(df_res["Fecha_corte"], df_res["Prob_aprox"], "o-", color="tab:blue", label="Probabilidad de acierto")
plt.scatter(df_res["Fecha_corte"], df_res["Certeza_pred"], c="red", label="Acierto real (1=correcto)")

# Etiquetas de patrón sobre los puntos
for i, row in df_res.iterrows():
    plt.text(row["Fecha_corte"], row["Prob_aprox"]+0.03, row["Patron_pred"],
             ha='center', va='bottom', fontsize=8, color='darkslategray', rotation=45)

if fecha_opt is not None:
    plt.axvline(fecha_opt, color="green", linestyle="--", lw=2,
                label=f"Fecha óptima ≈ {fecha_opt.date()}")

plt.title(f"Certeza temporal del patrón — Patrón real: {pat_real}")
plt.ylabel("Probabilidad / Acierto")
plt.xlabel("Fecha de corte")
plt.grid(True, ls="--", alpha=0.5)
plt.ylim(0, 1.1)
plt.legend()
plt.tight_layout()

out_png = "certeza_temporal_patron.png"
plt.savefig(out_png, dpi=300)

# ===============================================================
# 💾 EXPORTACIÓN ROBUSTA (Excel si posible, CSV si no)
# ===============================================================
try:
    import openpyxl
    out_xlsx = "certeza_temporal_patron.xlsx"
    df_res.to_excel(out_xlsx, index=False)
    print(f"✅ Exportado correctamente a Excel: {out_xlsx}")
except ImportError:
    out_csv = "certeza_temporal_patron.csv"
    df_res.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"⚠️ 'openpyxl' no está instalado — exportado como CSV: {out_csv}")

print(f"✅ Análisis completado\n📈 {out_png}")
