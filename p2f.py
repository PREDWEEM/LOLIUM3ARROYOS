# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Entrenamiento híbrido P2F (LOLium)
# ===============================================================
# Genera p2f_bundle.joblib combinando:
#  - curvas_emergencia_github_274.csv
#  - datos LOLIUM METEORO.xlsx (una hoja por año)
#  - patrones_por_anio.csv
# ===============================================================

import numpy as np, pandas as pd, joblib, re
from pathlib import Path
from scipy.optimize import curve_fit
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

JD_MAX = 274

# === FUNCIONES DE CURVA ===
def richards(t, K, r, t0, v):
    return K / (1 + v * np.exp(-r * (t - t0))) ** (1 / v)

def two_stage_richards(t, p):
    a1, r1, t01, v1, a2, r2, t02, v2 = p
    y1 = a1 * richards(t, 1, r1, t01, v1)
    y2 = a2 * richards(t, 1, r2, t02, v2)
    y = np.clip(np.maximum.accumulate(y1 + y2), 0, 1)
    if y.max() > 0:
        y /= y.max()
    return y

def fit_params(curva):
    t = np.arange(1, JD_MAX + 1)
    y = np.clip(np.maximum.accumulate(curva), 0, 1)
    y /= y[-1] if y[-1] != 0 else 1
    p0 = [0.7, 0.08, 120, 1.0, 0.6, 0.06, 165, 1.0]
    bounds = ([0, 0, 60, 0.2, 0, 0, 120, 0.2], [1.5, 1, 200, 5.0, 1.5, 1, 260, 5.0])
    try:
        popt, _ = curve_fit(lambda tt, *pp: two_stage_richards(tt, pp),
                            t, y, p0=p0, bounds=bounds, maxfev=20000)
        return popt
    except Exception:
        return np.array(p0, float)

# === FUNCIONES DE FEATURES METEOROLÓGICAS ===
def standardize_cols(df):
    df.columns = [str(c).lower().strip() for c in df.columns]
    ren = {
        "t min": "tmin", "t_max": "tmax", "t max": "tmax",
        "precipitacion": "prec", "pp": "prec", "lluvia": "prec",
        "dia juliano": "jd", "día": "jd", "dia": "jd", "julian_days": "jd"
    }
    for k, v in ren.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    return df

def build_features(df):
    tmin = df["tmin"].to_numpy(float)
    tmax = df["tmax"].to_numpy(float)
    tmed = (tmin + tmax) / 2.0
    prec = df["prec"].to_numpy(float)
    jd = df["jd"].to_numpy(int)

    mask = (jd >= 32) & (jd <= 151)
    gdd5 = np.cumsum(np.maximum(tmed - 5, 0))
    gdd3 = np.cumsum(np.maximum(tmed - 3, 0))

    f = {}
    f["gdd5_FM"] = gdd5[mask].ptp()
    f["gdd3_FM"] = gdd3[mask].ptp()
    pf = prec[mask]
    f["pp_FM"] = pf.sum()
    f["ev10_FM"] = int((pf >= 10).sum())
    f["ev20_FM"] = int((pf >= 20).sum())

    dry = (pf < 1).astype(int)
    wet = (pf >= 5).astype(int)
    def longest_run(x):
        c = m = 0
        for v in x: c = c + 1 if v == 1 else 0; m = max(m, c)
        return m
    f["dry_run_FM"] = longest_run(dry)
    f["wet_run_FM"] = longest_run(wet)

    def ma(x, w): return np.convolve(x, np.ones(w)/w, "same")
    f["tmed14_May"] = ma(tmed, 14)[151]
    f["tmed28_May"] = ma(tmed, 28)[151]
    f["gdd5_120"] = gdd5[119]
    f["pp_120"] = prec[:120].sum()
    return f

# === CARGA DE ARCHIVOS ===
def load_meteo_dict(path_xlsx):
    sheets = pd.read_excel(path_xlsx, sheet_name=None)
    out = {}
    for name, df in sheets.items():
        df = standardize_cols(df)
        if "jd" not in df.columns:
            df["jd"] = np.arange(1, len(df) + 1)
        df = (df.set_index("jd").reindex(range(1, JD_MAX + 1)).interpolate().ffill().bfill().reset_index())
        try:
            year = int(re.findall(r"\d{4}", str(name))[0])
        except:
            year = None
        if year:
            out[year] = df[["jd", "tmin", "tmax", "prec"]]
    return out

def load_curvas_dict(csv_path):
    df = pd.read_csv(csv_path)
    jdcol = [c for c in df.columns if c.lower() in ["día", "dia", "day", "jd"]][0]
    ycols = [c for c in df.columns if c != jdcol]
    curvas = {}
    for c in ycols:
        y = np.array(df[c].values[:JD_MAX], float)
        if np.nanmax(y) > 0:
            y = np.maximum.accumulate(y)
            y /= y[-1]
            curvas[int(c)] = y
    return curvas

# === ENTRENAMIENTO ===
def main():
    curvas_csv = "curvas_emergencia_github_274.csv"
    meteo_xlsx = "datos LOLIUM METEORO.xlsx"
    patrones_csv = "patrones_por_anio.csv"

    curvas = load_curvas_dict(curvas_csv)
    meteo = load_meteo_dict(meteo_xlsx)
    patrones = pd.read_csv(patrones_csv)

    años = sorted(set(curvas) & set(meteo) & set(patrones["anio"]))
    X, Ypat, Params = [], [], []
    for y in años:
        feats = build_features(meteo[y])
        X.append([feats[k] for k in sorted(feats.keys())])
        curva = curvas[y]
        Params.append(fit_params(curva))
        patron = patrones.loc[patrones["anio"] == y, "patron"].iloc[0]
        Ypat.append(patron)

    feat_names = sorted(build_features(next(iter(meteo.values()))).keys())
    X = np.array(X, float)
    Y = np.array(Ypat)
    P = np.array(Params, float)

    xsc = StandardScaler().fit(X)
    Xs = xsc.transform(X)

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xs, Y)

    regs = {}
    for pat in np.unique(Y):
        idx = np.where(Y == pat)[0]
        regs[pat] = []
        for j in range(P.shape[1]):
            r = GradientBoostingRegressor(random_state=42)
            r.fit(Xs[idx], P[idx, j])
            regs[pat].append(r)

    protos = {}
    for pat in np.unique(Y):
        curvs = np.vstack([curvas[y] for i, y in enumerate(años) if Y[i] == pat])
        protos[pat] = np.median(curvs, axis=0)

    bundle = {"xsc": xsc, "feat_names": feat_names, "clf": clf, "regs": regs, "protos": protos}
    joblib.dump(bundle, "p2f_bundle.joblib")
    print("✅ Modelo P2F guardado: p2f_bundle.joblib")

if __name__ == "__main__":
    main()

