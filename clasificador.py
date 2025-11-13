
import pandas as pd
import numpy as np
import re
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import euclidean
from joblib import dump, load


def cargar_curva_acumulada(path):
    df = pd.read_excel(path, header=None)
    dias = df.iloc[:,0].values
    vals = df.iloc[:,1].values

    diario = np.zeros(365)
    for d,v in zip(dias, vals):
        if 1 <= int(d) <= 365:
            diario[int(d)-1] = v

    acum = np.cumsum(diario)
    if acum[-1] == 0:
        return np.zeros(365)

    return acum / acum[-1]

def cargar_meteo_xlsx(path):
    book = pd.read_excel(path, sheet_name=None)
    out = {}
    for name, df in book.items():
        if not isinstance(df, pd.DataFrame): 
            continue
        try:
            year = int(re.findall(r"\d{4}", str(name))[0])
        except:
            continue

        df = df.rename(columns={
            "TMIN":"tmin", "TMAX":"tmax", "Prec":"prec",
            "Julian_days":"jd", "Día juliano":"jd"
        })

        df["jd"] = pd.to_numeric(df["jd"], errors="coerce")
        df["tmin"] = pd.to_numeric(df["tmin"], errors="coerce")
        df["tmax"] = pd.to_numeric(df["tmax"], errors="coerce")
        df["prec"] = pd.to_numeric(df["prec"], errors="coerce")

        df = df.sort_values("jd").reset_index(drop=True)
        df = df[(df["jd"] >= 1) & (df["jd"] <= 274)]

        out[year] = df
    return out

def features_meteo(df):
    tmin = df["tmin"].values
    tmax = df["tmax"].values
    tmed = (tmin + tmax)/2
    prec = df["prec"].values
    jd = df["jd"].values
    
    gdd5 = np.cumsum(np.maximum(tmed - 5, 0))
    gdd3 = np.cumsum(np.maximum(tmed - 3, 0))

    return {
        "gdd5_120": gdd5[min(119,len(gdd5)-1)],
        "gdd3_120": gdd3[min(119,len(gdd3)-1)],
        "pp_120": np.sum(prec[:120]),
        "tmed_14may": np.mean(tmed[136:150]) if len(tmed)>150 else np.nan,
        "tmed_28may": np.mean(tmed[122:150]) if len(tmed)>150 else np.nan,
        "ev10_FM": np.sum(prec[31:151] >= 10),
        "ev20_FM": np.sum(prec[31:151] >= 20),
        "FM_pp": np.sum(prec[31:151]),
        "dryrun_FM": max_run(prec[31:151] < 1),
        "wetrun_FM": max_run(prec[31:151] >= 5)
    }

def max_run(mask):
    c = m = 0
    for v in mask:
        c = c+1 if v else 0
        m = max(m,c)
    return m

def features_curva(curva):

    idx_inicio = np.argmax(curva > 0)
    frac_1_120 = curva[119] if len(curva)>=120 else curva[-1]

    tramo = curva[29:121]  # JD 30–120
    tasas = np.diff(tramo)
    tasa_30_120 = np.nanmean(tasas)

    return {
        "inicio": idx_inicio + 1,
        "frac_120": frac_1_120,
        "tasa_30_120": tasa_30_120,
        "dia_10": int(np.argmax(curva > 0.10)),
        "dia_25": int(np.argmax(curva > 0.25)),
        "dia_50": int(np.argmax(curva > 0.50)),
        "dia_75": int(np.argmax(curva > 0.75)),
    }

def dtw(a,b):
    n,m = len(a), len(b)
    D = np.full((n+1,m+1), np.inf)
    D[0,0]=0
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost = (a[i-1]-b[j-1])**2
            D[i,j] = cost + min(D[i-1,j],D[i,j-1],D[i-1,j-1])
    return np.sqrt(D[n,m])

def k_medoids(curvas, K=3):
    N = len(curvas)
    rng = np.random.default_rng(0)
    medoids = rng.choice(N,K,replace=False)

    for _ in range(40):
        clusters = {k:[] for k in range(K)}
        for i in range(N):
            d = [dtw(curvas[i], curvas[m]) for m in medoids]
            k = np.argmin(d)
            clusters[k].append(i)

        new_medoids=[]
        for k in range(K):
            if not clusters[k]:
                new_medoids.append(medoids[k])
                continue
            subD = np.zeros((len(clusters[k]),len(clusters[k])))
            for i,p in enumerate(clusters[k]):
                for j,q in enumerate(clusters[k]):
                    subD[i,j]=dtw(curvas[p],curvas[q])
            new_medoids.append(clusters[k][np.argmin(subD.sum(axis=1))])

        if np.all(new_medoids == medoids):
            break
        medoids = new_medoids

    return medoids, clusters


def entrenar_modelo(curvas, meteo_dict, años):
    # Extraer features de curvas
    Fcurva=[]
    for curva in curvas:
        Fcurva.append(features_curva(curva))
    Fcurva = pd.DataFrame(Fcurva)

    # Extraer features meteo
    Fmet=[]
    for y in años:
        Fmet.append(features_meteo(meteo_dict[y]))
    Fmet = pd.DataFrame(Fmet)

    # Generar patrones con DTW
    medoids, clusters = k_medoids(curvas, K=3)

    etiquetas=[]
    for k,lista in clusters.items():
        for i in lista:
            etiquetas.append((i,k))
    etiquetas = sorted(etiquetas)     # (índice curva, cluster)
    y_labels = np.array([k for _,k in etiquetas])

    # Construir X
    X = pd.concat([Fmet.reset_index(drop=True),
                   Fcurva.reset_index(drop=True)], axis=1)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    clf = GradientBoostingClassifier().fit(Xs, y_labels)

    return clf, scaler, medoids, clusters, X.columns

def predecir_patron(df_meteo, clf, scaler, cols):
    f = features_meteo(df_meteo)
    X = pd.DataFrame([f])[cols]
    Xs = scaler.transform(X)
    proba = clf.predict_proba(Xs)[0]
    k = np.argmax(proba)
    return k, proba










