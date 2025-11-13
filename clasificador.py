# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM v5.2 — Mixture-of-Prototypes
# ===============================================================
# Mejoras principales vs v5.1:
# - DTW ponderado: mayor peso a JD 30–121 (ventana crítica de patrón)
# - Banda Sakoe–Chiba (±band) para evitar alineamientos imposibles
# - Embedding de forma de la curva:
#     * inicio_emergencia (primer día con E>0)
#     * frac_1_120 (fracción acumulada al JD 120)
#     * tasa_prom_30_120 (tasa media diaria en 30–120)
#     * max_inc_30_120 (máx incremento diario en 30–120)
#     * dia_max_inc_30_120 (JD de ese máximo)
#     * skew_inc_30_120 (asimetría incrementos 30–120)
#     * kurt_inc_30_120 (curtosis incrementos 30–120)
# - Clasificador combinado METEO + EMBEDDINGS DE CURVA
# - Detección de outliers (LocalOutlierFactor) antes del k-medoids
# - Opción de cargar manualmente la fecha de inicio de emergencia
#   (dato medido a campo) en la predicción
# - Curvas siempre monotónicas 0–1 en JD 1..274
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import re, io, joblib

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

from scipy.stats import skew, kurtosis

# ---------------------------------------------------------------
# CONFIGURACIÓN GENERAL STREAMLIT
# ---------------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM v5.2 — Mixture-of-Prototypes (DTW ponderado)",
    layout="wide"
)
st.title("🌾 PREDWEEM v5.2 — DTW ponderado + Embeddings de curva")

JD_MAX = 274          # Trabajamos hasta el 1 de octubre
XRANGE = (1, JD_MAX)  # Rango del eje X en gráficos

# ===============================================================
# UTILIDADES GENERALES
# ===============================================================
def _make_unique(names):
    """
    Hace únicos los nombres de columna sin usar APIs internas de pandas.
    Si hay columnas repetidas, les agrega .1, .2, etc.
    """
    seen, out = {}, []
    for n in names:
        if n not in seen:
            seen[n] = 0
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}.{seen[n]}")
    return out

def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas y trata de mapearlos a:
    - tmin, tmax, prec, jd, fecha
    """
    df = df.copy()
    df.columns = _make_unique([str(c).lower().strip() for c in df.columns])
    ren = {
        "temperatura minima":"tmin","t_min":"tmin","t min":"tmin","mínima":"tmin","min":"tmin",
        "temperatura maxima":"tmax","t_max":"tmax","t max":"tmax","máxima":"tmax","max":"tmax",
        "precipitacion":"prec","precip":"prec","pp":"prec","lluvia":"prec","rain":"prec",
        "dia juliano":"jd","día juliano":"jd","julian_days":"jd","dia":"jd","día":"jd",
        "fecha":"fecha","date":"fecha"
    }
    for k,v in ren.items():
        if k in df.columns:
            df = df.rename(columns={k:v})
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
    for c in ["tmin","tmax","prec","jd"]:
        if c in df.columns and isinstance(df[c], pd.Series):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def ensure_jd_1_to_274(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que la tabla tenga una columna 'jd' (día juliano) y la
    reindexa al rango 1..274 con interpolación y relleno.
    """
    df = df.copy()
    df.columns = _make_unique(df.columns)
    if "jd" not in df.columns:
        # Si hay fecha, derivar jd; si no, asumir secuencia 1..n
        if "fecha" in df.columns and df["fecha"].notna().any():
            y0 = int(df["fecha"].dt.year.mode().iloc[0])
            df = df[(df["fecha"] >= f"{y0}-01-01") & (df["fecha"] <= f"{y0}-10-01")].copy().sort_values("fecha")
            df["jd"] = df["fecha"].dt.dayofyear - pd.Timestamp(f"{y0}-01-01").dayofyear + 1
        else:
            df["jd"] = np.arange(1, len(df) + 1, dtype=int)
    if isinstance(df["jd"], pd.DataFrame):
        df["jd"] = df["jd"].iloc[:,0]
    df["jd"] = pd.to_numeric(df["jd"], errors="coerce").astype("Int64")

    jd_range = np.arange(1, JD_MAX+1)
    df = (df.set_index("jd")
            .reindex(jd_range)
            .interpolate()
            .ffill().bfill()
            .reset_index())
    return df

# ===============================================================
# CURVAS DE EMERGENCIA (HISTÓRICA O REAL) DESDE XLSX
# ===============================================================
def curva_desde_xlsx_anual(file) -> np.ndarray:
    """
    Lee XLSX con dos columnas [día/fecha, valor] (diaria o semanal) y
    devuelve curva acumulada 0..1 (JD 1..274). Si la serie es semanal,
    suaviza con ventana de 7 días.
    """
    df = pd.read_excel(file, header=None)
    if df.shape[1] < 2:
        df = pd.read_excel(file)

    col0 = pd.to_numeric(df.iloc[:,0], errors="coerce")
    col1 = pd.to_numeric(df.iloc[:,1], errors="coerce").fillna(0.0)

    if col0.isna().mean() > 0.5:
        # Primera columna es fecha
        fch = pd.to_datetime(df.iloc[:,0], errors="coerce", dayfirst=True)
        jd  = fch.dt.dayofyear
        val = col1
    else:
        jd  = col0.astype("Int64")
        val = col1

    jd_clean = jd.dropna().astype(int).sort_values().unique()
    paso = int(np.median(np.diff(jd_clean))) if len(jd_clean)>1 else 7

    # Arreglo diario año completo (365)
    daily = np.zeros(365, dtype=float)
    for d,v in zip(jd,val):
        if pd.notna(d) and 1 <= int(d) <= 365:
            daily[int(d)-1] += float(v)
    # Si es semanal u otra frecuencia >1, suavizar
    if paso > 1:
        daily = np.convolve(daily, np.ones(7)/7, mode="same")

    # Acumulada y normalización a 0..1
    acum = np.cumsum(daily)
    if np.nanmax(acum) == 0:
        return np.zeros(JD_MAX, dtype=float)
    curva = (acum / np.nanmax(acum))[:JD_MAX]
    # Asegurar monotonía no decreciente
    return np.maximum.accumulate(np.clip(curva,0,1))

def emerg_rel_7d_from_acum(y_acum: np.ndarray) -> np.ndarray:
    """
    Emergencia relativa semanal: deriva la acumulada diaria y aplica
    promedio móvil de 7 días.
    """
    inc = np.diff(np.insert(y_acum, 0, 0.0))
    return np.convolve(inc, np.ones(7)/7, mode="same")

def frac_curva_1_120(curva: np.ndarray) -> float:
    """
    Fracción de emergencia acumulada al día juliano 120.
    Dado que la curva está normalizada 0–1, es simplemente E(120).
    (Se usa como factor de clasificación y diagnóstico).
    """
    if len(curva) == 0:
        return 0.0
    idx_120 = min(119, len(curva)-1)  # JD120 → índice 119
    return float(curva[idx_120])

def detectar_inicio_emergencia(curva: np.ndarray) -> int:
    """
    Detecta el JD de inicio de emergencia.
    Definición:
      ➜ primer JD donde la emergencia acumulada supera 0.

    Si la curva nunca supera 0 ⇒ devuelve 999 (desconocido).
    """
    idx = np.where(curva > 0)[0]
    if len(idx) == 0:
        return 999
    return int(idx[0] + 1)  # índice → día juliano

def analizar_incrementos_30_120(curva: np.ndarray):
    """
    Analiza el comportamiento de la curva entre JD 30 y JD 120:
      - tasa_promedio_30_120: incremento medio diario
      - max_incremento_30_120: mayor ∆E en un día
      - dia_max_incremento_30_120: JD donde ocurre ese máximo ∆E

    Se usa como factor de clasificación y también para diagnóstico.
    """
    i1, i2 = 29, 119  # JD30=idx29, JD120=idx119
    segmento = curva[i1:i2+1]

    if len(segmento) < 2:
        return {
            "tasa_promedio_30_120": 0.0,
            "max_incremento_30_120": 0.0,
            "dia_max_incremento_30_120": 30
        }

    # Incrementos diarios en ese segmento
    inc = np.diff(segmento)

    tasa_promedio = (segmento[-1] - segmento[0]) / (i2 - i1)
    max_inc = float(np.max(inc))
    idx_max_inc = int(np.argmax(inc))
    dia_max_inc = 30 + idx_max_inc   # convertir a día juliano

    return {
        "tasa_promedio_30_120": float(tasa_promedio),
        "max_incremento_30_120": max_inc,
        "dia_max_incremento_30_120": dia_max_inc
    }

def embedding_curva_forma(curva: np.ndarray) -> dict:
    """
    Genera un embedding de forma para la curva de emergencia:
    - inicio_emergencia
    - frac_1_120
    - tasa_prom_30_120
    - max_inc_30_120
    - dia_max_inc_30_120
    - skew_inc_30_120
    - kurt_inc_30_120

    Este embedding se usa para:
    - mejorar la agrupación (2008, 2013, etc.)
    - alimentar el clasificador junto con las features meteo.
    """
    inicio = detectar_inicio_emergencia(curva)
    frac120 = frac_curva_1_120(curva)
    anal = analizar_incrementos_30_120(curva)

    # Incrementos en 30–120 para skewness / kurtosis
    i1, i2 = 29, 119
    segmento = curva[i1:i2+1]
    if len(segmento) < 3:
        inc = np.zeros(3)
    else:
        inc = np.diff(segmento)
    # Evitar NaNs
    if np.allclose(inc, 0):
        skew_inc = 0.0
        kurt_inc = 0.0
    else:
        skew_inc = float(skew(inc, bias=False, nan_policy="omit"))
        kurt_inc = float(kurtosis(inc, fisher=True, bias=False, nan_policy="omit"))

    emb = {
        "inicio_emergencia": float(inicio),
        "frac_1_120": float(frac120),
        "tasa_prom_30_120": float(anal["tasa_promedio_30_120"]),
        "max_inc_30_120": float(anal["max_incremento_30_120"]),
        "dia_max_inc_30_120": float(anal["dia_max_incremento_30_120"]),
        "skew_inc_30_120": skew_inc,
        "kurt_inc_30_120": kurt_inc
    }
    return emb

# ===============================================================
# DTW PONDERADO + BANDA SAKOE–CHIBA (v5.2)
# ===============================================================
def dtw_distance_weighted(a: np.ndarray,
                          b: np.ndarray,
                          band: int = 10,
                          w_focus: float = 3.0) -> float:
    """
    Distancia DTW ponderada entre dos curvas de emergencia acumulada,
    usando únicamente el segmento JD 30–121 (inclusive) y una banda
    de Sakoe–Chiba ±band.

    - Tramo 30–121 tiene peso w_focus (por defecto 3.0)
    - Resto del tramo considerado tiene peso 1.0
    (En esta implementación trabajamos SOLO con 30–121, así que el
    peso es w_focus constante; la estructura permite extenderlo).

    Parámetros
    ----------
    a, b : np.ndarray
        Curvas acumuladas (longitud ≥ 121).
    band : int
        Semiancho de la banda Sakoe–Chiba (en días).
    w_focus : float
        Peso aplicado al tramo 30–121 (ventana crítica).
    """
    # Recortar a ventana 30–121 (índices 29..120)
    a_seg = a[29:121]
    b_seg = b[29:121]

    n, m = len(a_seg), len(b_seg)
    # Pesos: aquí constante = w_focus (se podría hacer variable por día)
    w = w_focus

    # Matriz de costes acumulados
    D = np.full((n+1, m+1), np.inf, dtype=float)
    D[0, 0] = 0.0

    for i in range(1, n+1):
        # Banda Sakoe–Chiba alrededor de la diagonal
        j_min = max(1, i - band)
        j_max = min(m, i + band)
        ai = a_seg[i-1]
        for j in range(j_min, j_max+1):
            diff = ai - b_seg[j-1]
            cost = w * (diff * diff)
            D[i, j] = cost + min(D[i-1, j],    # inserción
                                 D[i, j-1],    # eliminación
                                 D[i-1, j-1])  # match

    dist = float(np.sqrt(D[n, m]))
    return dist

# ===============================================================
# FEATURES METEOROLÓGICAS + EMBEDDINGS + OUTLIERS + K-MEDOIDS
# ===============================================================

# ------------------------------#
# 1) FEATURES METEOROLÓGICAS
# ------------------------------#
FEATURE_ORDER = [
    "gdd5_FM","gdd3_FM","pp_FM","ev10_FM","ev20_FM",
    "dry_run_FM","wet_run_FM","tmed14_May","tmed28_May","gdd5_120","pp_120"
]

def _longest_run(binary_vec: np.ndarray) -> int:
    """
    Longitud máxima de racha consecutiva de 1s (por ejemplo,
    días secos o húmedos consecutivos).
    """
    m = c = 0
    for v in binary_vec:
        c = c + 1 if v == 1 else 0
        m = max(m, c)
    return int(m)

def build_features_meteo(dfm: pd.DataFrame):
    """
    A partir de la serie meteo diaria (tmin, tmax, prec, jd) en 1..274,
    calcula un conjunto de features agroclimáticos agregados:
    - GDD base 5 y 3 en Feb–May
    - Precipitaciones totales, eventos ≥10 mm, ≥20 mm
    - Racha seca y húmeda más larga
    - Tmed 14 y 28 días centrada en mayo
    - GDD5 acumulado al JD 120
    - Precipitación acumulada al JD 120
    """
    dfm = standardize_cols(dfm)
    dfm = ensure_jd_1_to_274(dfm)
    tmin = dfm["tmin"].astype(float).to_numpy()
    tmax = dfm["tmax"].astype(float).to_numpy()
    tmed = (tmin + tmax) / 2.0
    prec = dfm["prec"].astype(float).to_numpy()
    jd   = dfm["jd"].astype(int).to_numpy()

    # Ventana Feb–May (aprox JD 32–151)
    mask_FM = (jd >= 32) & (jd <= 151)
    gdd5 = np.cumsum(np.maximum(tmed - 5, 0))
    gdd3 = np.cumsum(np.maximum(tmed - 3, 0))

    if not np.any(mask_FM):
        # Si algo raro, usar todo el rango
        mask_FM = np.ones_like(jd, dtype=bool)

    pf = prec[mask_FM]
    if pf.size == 0 or np.all(np.isnan(pf)):
        pf = np.zeros(1)

    f = {}
    f["gdd5_FM"]   = float(np.ptp(gdd5[mask_FM])) if np.any(~np.isnan(gdd5[mask_FM])) else 0.0
    f["gdd3_FM"]   = float(np.ptp(gdd3[mask_FM])) if np.any(~np.isnan(gdd3[mask_FM])) else 0.0
    f["pp_FM"]     = float(np.nansum(pf))
    f["ev10_FM"]   = int(np.nansum(pf >= 10))
    f["ev20_FM"]   = int(np.nansum(pf >= 20))
    dry            = np.nan_to_num(pf < 1, nan=0).astype(int)
    wet            = np.nan_to_num(pf >= 5, nan=0).astype(int)
    f["dry_run_FM"]= _longest_run(dry)
    f["wet_run_FM"]= _longest_run(wet)

    # Tmed suavizada alrededor de mayo (14 y 28 días)
    def ma(x, w):
        k = np.ones(w) / w
        return np.convolve(x, k, "same")
    idx_may = min(150, len(tmed)-1)
    f["tmed14_May"] = float(ma(tmed, 14)[idx_may])
    f["tmed28_May"] = float(ma(tmed, 28)[idx_may])

    # Estado a JD 120
    idx_120 = min(119, len(tmed) - 1)
    f["gdd5_120"] = float(gdd5[idx_120])
    f["pp_120"]   = float(np.nansum(prec[: idx_120 + 1]))

    # Orden consistente
    f = {k: f[k] for k in FEATURE_ORDER}
    return dfm, f

# ------------------------------#
# 2) EMBEDDINGS DE CURVA
# ------------------------------#
EMBED_FEAT_NAMES = [
    "inicio_emergencia",
    "frac_1_120",
    "tasa_prom_30_120",
    "max_inc_30_120",
    "dia_max_inc_30_120",
    "skew_inc_30_120",
    "kurt_inc_30_120"
]

def build_embedding_matrix(curves: list, years: list):
    """
    Construye una matriz de embeddings de forma (uno por curva)
    y devuelve:
      - emb_matrix: np.ndarray [N x len(EMBED_FEAT_NAMES)]
      - emb_list: lista de dicts emb por año (para debug/diagnóstico)
    """
    embs = []
    emb_dicts = []
    for curva, y in zip(curves, years):
        emb = embedding_curva_forma(curva)
        emb_dicts.append({"year": int(y), **emb})
        row = [emb[k] for k in EMBED_FEAT_NAMES]
        embs.append(row)
    emb_matrix = np.array(embs, float)
    return emb_matrix, emb_dicts

def detectar_outliers_embeddings(emb_matrix: np.ndarray,
                                 n_neighbors: int = None,
                                 contamination: float = 0.25):
    """
    Detecta outliers en el espacio de embeddings usando
    LocalOutlierFactor.

    Devuelve:
      - mask_inliers: boolean array (True = inlier)
      - lof_scores: array de scores (negativos, más negativo = más outlier)
    """
    n_samples = emb_matrix.shape[0]
    if n_neighbors is None:
        n_neighbors = max(5, min(20, int(0.4 * n_samples)))
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        metric="euclidean"
    )
    y_pred = lof.fit_predict(emb_matrix)  # -1 outlier, 1 inlier
    mask_inliers = (y_pred == 1)
    scores = lof.negative_outlier_factor_
    return mask_inliers, scores

# ------------------------------#
# 3) K-MEDOIDS CON DTW PONDERADO
# ------------------------------#
def k_medoids_dtw_weighted(curves: list,
                           K: int,
                           max_iter: int = 50,
                           seed: int = 42,
                           band: int = 10,
                           w_focus: float = 3.0):
    """
    Agrupa curvas en K clusters usando k-medoids con distancia DTW
    ponderada (dtw_distance_weighted) sobre JD 30–121.

    Parámetros
    ----------
    curves : list of np.ndarray
        Curvas de emergencia acumulada (0..1, longitud ≥ 121).
    K : int
        Número de prototipos.
    band : int
        Banda Sakoe–Chiba (±días).
    w_focus : float
        Peso aplicado al tramo crítico (aquí 30–121).

    Devuelve
    --------
    medoid_idx : list[int]
        Índices de las curvas elegidas como prototipos.
    clusters : dict[int, list[int]]
        Asignación de cada curva a su cluster (por índice).
    D : np.ndarray
        Matriz de distancias DTW ponderadas (N x N).
    """
    rng = np.random.default_rng(seed)
    N = len(curves)
    if N == 0:
        raise ValueError("No hay curvas para agrupar.")
    if K > N:
        K = N

    idx = rng.choice(N, size=K, replace=False)
    medoid_idx = list(idx)

    # Matriz de distancias (simétrica)
    D = np.zeros((N, N), float)
    for i in range(N):
        for j in range(i+1, N):
            d = dtw_distance_weighted(curves[i], curves[j],
                                      band=band, w_focus=w_focus)
            D[i, j] = D[j, i] = d

    # Iterar hasta convergencia de medoids
    for _ in range(max_iter):
        assign = np.argmin(D[:, medoid_idx], axis=1)
        new_medoids = []
        for k in range(K):
            members = np.where(assign == k)[0]
            if len(members) == 0:
                # Si queda vacío, mantener el medoid actual
                new_medoids.append(medoid_idx[k])
                continue
            subD = D[np.ix_(members, members)]
            sums = subD.sum(axis=1)
            chosen = members[np.argmin(sums)]
            new_medoids.append(chosen)
        if new_medoids == medoid_idx:
            break
        medoid_idx = new_medoids

    clusters = {k: [] for k in range(K)}
    assign = np.argmin(D[:, medoid_idx], axis=1)
    for i in range(N):
        clusters[int(assign[i])].append(i)
    return medoid_idx, clusters, D

# ------------------------------#
# 4) WARP + MEZCLA CONVEXA
# ------------------------------#
def warp_curve(proto: np.ndarray, shift: float, scale: float) -> np.ndarray:
    """
    Aplica un warp simple a la curva prototipo:
    - shift: desplazamiento horizontal en días
    - scale: escalado del eje temporal (compresión/estiramiento)
    """
    t = np.arange(1, JD_MAX+1, dtype=float)
    tp = (t - shift) / max(scale, 1e-6)
    tp = np.clip(tp, 1, JD_MAX)
    yv = np.interp(tp, np.arange(1, JD_MAX+1, dtype=float), proto)
    return np.maximum.accumulate(np.clip(yv, 0, 1))

def mezcla_convexa(protos: np.ndarray,
                   proba: np.ndarray,
                   k_hat: int,
                   shift: float,
                   scale: float) -> np.ndarray:
    """
    Construye la curva predicha como mezcla convexa de todos los prototipos,
    aplicando el warp (shift/scale) sólo al patrón más probable.
    """
    K = protos.shape[0]
    mix = np.zeros(JD_MAX, float)
    for k in range(K):
        yk = warp_curve(
            protos[k],
            shift if k == k_hat else 0.0,
            scale if k == k_hat else 1.0
        )
        mix += float(proba[k]) * yk
    return np.maximum.accumulate(np.clip(mix, 0, 1))

# ------------------------------#
# 5) LISTAS DE FEATURES GLOBALES
# ------------------------------#
# Entrada a regresores que predicen EMBEDDINGS desde meteo
REG_INPUT_FEAT_NAMES = FEATURE_ORDER[:]  # sólo features meteo

# Entrada al clasificador final: METEO + EMBEDDINGS
CLF_FEAT_NAMES = FEATURE_ORDER[:] + EMBED_FEAT_NAMES[:]

# ===============================================================
# APP — TABS (v5.2)
# ===============================================================
tabs = st.tabs([
    "🧪 Entrenar prototipos + clasificador",
    "🔮 Identificar patrones y predecir",
    "📊 Comparar curva real vs predicha"
])

# ---------------------------------------------------------------
# TAB 1 — ENTRENAMIENTO COMPLETO (v5.2)
# ---------------------------------------------------------------
with tabs[0]:
    st.subheader("🧪 Entrenamiento (DTW ponderado + Embeddings + Outliers)")
    st.markdown("""
    **Flujo v5.2**:
    1. Se leen curvas históricas (emergencia acumulada por año).
    2. Se calculan *embeddings de forma* de cada curva:
       - inicio de emergencia (E>0 desde JD 1),
       - fracción 1–120,
       - tasas y picos de incremento entre JD 30–120,
       - skewness y kurtosis de los incrementos.
    3. Se detectan **outliers** en el espacio de embeddings (LocalOutlierFactor).
    4. Se aplica **k-medoids** con **DTW ponderado (30–121)** sobre curvas inliers.
    5. Se calculan:
       - Prototipos (medoids),
       - Años asignados a cada patrón.
    6. Se entrenan:
       - Regresores METEO→EMBEDDINGS,
       - Clasificador final METEO+EMB,
       - Regresores de warp (shift/scale) por patrón.
    """)

    meteo_book = st.file_uploader("📘 Meteorología multianual (una hoja por año)", type=["xlsx","xls"])
    curvas_files = st.file_uploader(
        "📈 Curvas históricas (XLSX por año, acumulada o semanal)",
        type=["xlsx","xls"],
        accept_multiple_files=True
    )

    col_par1, col_par2 = st.columns(2)
    with col_par1:
        K = st.slider("Número de prototipos/patrones (K)", 2, 10, 6, 1)
        seed = st.number_input("Semilla aleatoria", 0, 99999, 42)
    with col_par2:
        band = st.slider("Banda Sakoe–Chiba (± días)", 3, 20, 10, 1)
        w_focus = st.slider("Peso DTW tramo 30–121", 1.0, 5.0, 3.0, 0.5)

    btn_train = st.button("🚀 Entrenar modelo v5.2")

    if btn_train:
        if not (meteo_book and curvas_files):
            st.error("Cargá **meteorología** y **curvas históricas**."); st.stop()

        # 1) Leer METEOROLOGÍA por año
        sheets = pd.read_excel(meteo_book, sheet_name=None)
        meteo_dict = {}
        for name, df in sheets.items():
            df = standardize_cols(df)
            df = ensure_jd_1_to_274(df)
            try:
                year = int(re.findall(r"\d{4}", str(name))[0])
            except:
                year = None
            if year and all(c in df.columns for c in ["tmin", "tmax", "prec"]):
                meteo_dict[year] = df[["jd", "tmin", "tmax", "prec"]].copy()

        if not meteo_dict:
            st.error("⛔ No se detectó meteorología válida por año."); st.stop()
        st.success(f"✅ Meteorología válida: {len(meteo_dict)} años")

        # 2) Leer CURVAS históricas por año
        years_list, curves_list = [], []
        for f in curvas_files:
            y4 = re.findall(r"(\d{4})", f.name)
            year = int(y4[0]) if y4 else None
            if year is None:
                continue
            curva = np.maximum.accumulate(curva_desde_xlsx_anual(f))
            if curva.max() > 0:
                curves_list.append(curva[:JD_MAX])
                years_list.append(year)

        if not years_list:
            st.error("⛔ No se detectaron curvas válidas."); st.stop()

        # 3) Intersección meteo–curvas
        common_years = sorted([y for y in years_list if y in meteo_dict])
        if len(common_years) < 4:
            st.error("⛔ Muy pocos años en común (se recomienda ≥ 5)."); st.stop()

        curves = [curves_list[years_list.index(y)] for y in common_years]

        st.info(f"📆 Años comunes entre meteo y curvas: {common_years}")

        # 4) Embeddings de forma de curva + detección de outliers
        st.markdown("### 1️⃣ Embeddings de forma + Outliers")

        emb_matrix, emb_dicts = build_embedding_matrix(curves, common_years)
        mask_inliers, lof_scores = detectar_outliers_embeddings(
            emb_matrix,
            n_neighbors=None,
            contamination=0.25  # ~25% pueden ser atípicos
        )

        years_inliers = [y for y, m in zip(common_years, mask_inliers) if m]
        years_outliers = [y for y, m in zip(common_years, mask_inliers) if not m]

        st.write(f"✅ Años usados como **inliers** (para prototipos): {years_inliers}")
        if years_outliers:
            st.warning(f"⚠ Años detectados como **outliers** (no se usan para prototipos): {years_outliers}")

        df_emb = pd.DataFrame(emb_dicts)
        df_emb["lof_score"] = lof_scores
        st.markdown("**Embeddings de forma por año (para diagnóstico):**")
        st.dataframe(df_emb.round(4), use_container_width=True)

        if len(years_inliers) < 3:
            st.error("⛔ Demasiados outliers: quedan <3 años inliers para entrenar."); st.stop()

        curves_inliers = [curves[common_years.index(y)] for y in years_inliers]
        emb_inliers = emb_matrix[[common_years.index(y) for y in years_inliers], :]

        # 5) K-medoids con DTW ponderado (solo inliers)
        st.markdown("### 2️⃣ k-medoids con DTW ponderado (tramo 30–121)")
        K_eff = min(K, len(curves_inliers))
        medoid_idx, clusters, D_in = k_medoids_dtw_weighted(
            curves_inliers,
            K=K_eff,
            max_iter=50,
            seed=seed,
            band=band,
            w_focus=w_focus
        )
        protos = [curves_inliers[i] for i in medoid_idx]

        assign_inliers = np.zeros(len(curves_inliers), dtype=int)
        for k_cl, members in clusters.items():
            for idx_i in members:
                assign_inliers[idx_i] = int(k_cl)

        year_to_cluster = {y: int(assign_inliers[i]) for i, y in enumerate(years_inliers)}

        cluster_years = {k: [] for k in range(K_eff)}
        for y in years_inliers:
            k_cl = year_to_cluster[y]
            cluster_years[k_cl].append(int(y))

        st.success(f"✅ k-medoids OK. K efectivo = {K_eff} prototipos.")

        # 6) Features METEO + Embeddings para inliers
        st.markdown("### 3️⃣ Construcción de features METEO + EMBEDDINGS (inliers)")

        feat_rows_meteo = []
        for y in years_inliers:
            dfy, f_m = build_features_meteo(meteo_dict[y])
            feat_rows_meteo.append([f_m[k] for k in FEATURE_ORDER])

        X_meteo_in = np.array(feat_rows_meteo, float)         # [N_in x len(FEATURE_ORDER)]
        X_emb_in = emb_inliers                                # [N_in x len(EMBED_FEAT_NAMES)]
        y_lbl = np.array([year_to_cluster[y] for y in years_inliers], dtype=int)

        xsc_meteo = StandardScaler().fit(X_meteo_in)
        X_meteo_in_s = xsc_meteo.transform(X_meteo_in)

        X_clf_in = np.hstack([X_meteo_in, X_emb_in])
        xsc_clf = StandardScaler().fit(X_clf_in)
        X_clf_in_s = xsc_clf.transform(X_clf_in)

        # 7) Regresores METEO → EMBEDDINGS
        st.markdown("### 4️⃣ Regresores METEO → EMBEDDINGS de forma")

        embed_regressors = {}
        for j, emb_name in enumerate(EMBED_FEAT_NAMES):
            y_emb = X_emb_in[:, j]
            reg = GradientBoostingRegressor(random_state=seed)
            reg.fit(X_meteo_in_s, y_emb)
            embed_regressors[emb_name] = reg

        st.success("✅ Regresores de embeddings entrenados.")

        # 8) Clasificador final METEO+EMB
        st.markdown("### 5️⃣ Clasificador final (METEO + EMBEDDINGS)")
        clf = GradientBoostingClassifier(random_state=seed)
        clf.fit(X_clf_in_s, y_lbl)
        st.success("✅ Clasificador entrenado.")

        # 9) Warps shift/scale por patrón
        st.markdown("### 6️⃣ Ajuste de warp (shift/scale) por patrón")

        regs_shift, regs_scale = {}, {}
        K_final = K_eff

        for k_cl in range(K_final):
            idx_k = np.where(y_lbl == k_cl)[0]
            if len(idx_k) == 0:
                continue
            proto = protos[k_cl]
            shifts, scales, Xk = [], [], []

            for ii in idx_k:
                curv = curves_inliers[ii]
                best = (0.0, 1.0, 1e9)
                for sh in range(-20, 21, 5):       # ±20 días
                    for sc in [0.9, 0.95, 1.0, 1.05, 1.1]:
                        cand = warp_curve(proto, sh, sc)
                        rmse = float(np.sqrt(np.mean((cand - curv)**2)))
                        if rmse < best[2]:
                            best = (float(sh), float(sc), rmse)
                shifts.append(best[0])
                scales.append(best[1])
                Xk.append(X_clf_in_s[ii])

            Xk = np.vstack(Xk)
            reg_sh = GradientBoostingRegressor(random_state=seed)
            reg_sc = GradientBoostingRegressor(random_state=seed)
            reg_sh.fit(Xk, np.array(shifts))
            reg_sc.fit(Xk, np.array(scales))

            regs_shift[k_cl] = reg_sh
            regs_scale[k_cl] = reg_sc

        st.success("✅ Warps (shift/scale) ajustados por patrón.")

        # 10) Guardar bundle completo v5.2
        st.markdown("### 7️⃣ Bundle final v5.2")

        bundle = {
            "version": "5.2",
            "JD_MAX": JD_MAX,
            "bandsakoe": band,
            "w_focus": w_focus,
            # Escaladores
            "xsc_meteo": xsc_meteo,
            "xsc_clf": xsc_clf,
            # Listas de features
            "feat_names_meteo": FEATURE_ORDER[:],
            "feat_names_embed": EMBED_FEAT_NAMES[:],
            "clf_feat_names": CLF_FEAT_NAMES[:],
            # Modelos
            "clf": clf,
            "protos": np.vstack(protos),        # K_final x JD_MAX
            "regs_shift": regs_shift,
            "regs_scale": regs_scale,
            "embed_regressors": embed_regressors,
            # Info de clusters
            "cluster_years": cluster_years,
            "years_inliers": years_inliers,
            "years_outliers": years_outliers,
            # Embeddings para diagnóstico
            "embeddings_inliers": emb_inliers,
        }

        st.success(f"✅ Entrenamiento v5.2 completo. K_final = {K_final} patrones.")

        buf = io.BytesIO()
        joblib.dump(bundle, buf)
        st.download_button(
            "💾 Descargar modelo v5.2 (joblib)",
            data=buf.getvalue(),
            file_name=f"predweem_v52_mixture_dtw_K{K_final}.joblib",
            mime="application/octet-stream"
        )

        # 11) Vista rápida de prototipos
        dias = np.arange(1, JD_MAX+1)
        dfp = []
        for k_cl, proto in enumerate(protos):
            yrs_txt = ", ".join(map(str, cluster_years.get(k_cl, []))) if cluster_years.get(k_cl) else "—"
            dfp.append(pd.DataFrame({
                "Día": dias,
                "Valor": proto,
                "Serie": f"Proto {k_cl} · años: {yrs_txt}"
            }))
        dfp = pd.concat(dfp)
        chart = alt.Chart(dfp).mark_line().encode(
            x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE))),
            y=alt.Y("Valor:Q", title="Emergencia acumulada (0–1)",
                    scale=alt.Scale(domain=[0, 1])),
            color="Serie:N"
        ).properties(
            height=420,
            title="Prototipos (medoids DTW ponderado 30–121)"
        )
        st.altair_chart(chart, use_container_width=True)

# ---------------------------------------------------------------
# TAB 2 — PREDICCIÓN (v5.2 con embeddings + DTW ponderado)
# ---------------------------------------------------------------
with tabs[1]:

    st.subheader("🔮 Predicción de patrón y curva con meteorología nueva (v5.2)")
    st.markdown("""
    En esta versión:
    - Se toman **features meteorológicas**
    - Se predicen **embeddings de forma** (inicio, fracción 1–120, tasas, skewness…)
    - Se combina en un vector METEO+EMB
    - Se predice el **patrón más probable**
    - Se ajusta **shift** y **scale**
    - Se genera la **curva predicha** completa (1–274)
    - Se grafica con **emergencia relativa semanal** (eje secundario)

    Además, podés **forzar manualmente la fecha de inicio de emergencia**
    (dato de campo) para ayudar a la clasificación de patrones.
    """)

    modelo_file = st.file_uploader("📦 Modelo v5.2 (joblib)", type=["joblib"], key="pred_model")
    meteo_file  = st.file_uploader("📘 Meteorología nueva (XLSX)", type=["xlsx","xls"], key="pred_meteo")

    st.markdown("### ➤ Cargar manualmente la fecha de inicio de emergencia (opcional)")
    inicio_manual = st.number_input(
        "Fecha de inicio de emergencia (JD), si se conoce a campo:",
        min_value=1,
        max_value=300,
        value=1,
        help="Si no se conoce, dejar 999 (predicción sólo con meteorología)."
    )

    btn_pred = st.button("🚀 Ejecutar predicción v5.2")

    if btn_pred:
        if not (modelo_file and meteo_file):
            st.error("Cargá el modelo y la meteorología."); st.stop()

        bundle = joblib.load(modelo_file)

        JD_MAX = bundle["JD_MAX"]
        xsc_meteo = bundle["xsc_meteo"]
        xsc_clf   = bundle["xsc_clf"]

        feat_names_meteo = bundle["feat_names_meteo"]
        feat_names_embed = bundle["feat_names_embed"]
        clf_feat_names   = bundle["clf_feat_names"]

        clf           = bundle["clf"]
        protos        = bundle["protos"]
        regs_shift    = bundle["regs_shift"]
        regs_scale    = bundle["regs_scale"]
        embed_regs    = bundle["embed_regressors"]
        cluster_years = bundle.get("cluster_years", {})

        K = protos.shape[0]

        # Meteorología nueva
        dfm = pd.read_excel(meteo_file)
        dfm, f_new = build_features_meteo(dfm)

        X_m = np.array([[f_new[k] for k in feat_names_meteo]], float)
        X_m_s = xsc_meteo.transform(X_m)

        # Predicción de EMBEDDINGS desde METEO
        emb_pred = {}
        for emb in feat_names_embed:
            emb_pred[emb] = float(embed_regs[emb].predict(X_m_s)[0])

        # Si el usuario cargó inicio de emergencia, lo imponemos
        if inicio_manual != 1:
            emb_pred["inicio_emergencia"] = int(inicio_manual)

        # Vector METEO + EMBEDDINGS
        X_clf = np.array([[f_new[k] for k in feat_names_meteo] +
                          [emb_pred[k] for k in feat_names_embed]], float)
        X_clf_s = xsc_clf.transform(X_clf)

        # Clasificación de patrón
        proba = clf.predict_proba(X_clf_s)[0]
        top_idx = np.argsort(proba)[::-1]
        k_hat = int(top_idx[0])
        conf  = float(proba[k_hat])
        yrs_k = cluster_years.get(k_hat, [])

        # Warp shift/scale
        if k_hat in regs_shift:
            shift = float(regs_shift[k_hat].predict(X_clf_s)[0])
        else:
            shift = 0.0

        if k_hat in regs_scale:
            scale = float(regs_scale[k_hat].predict(X_clf_s)[0])
        else:
            scale = 1.0

        scale = float(np.clip(scale, 0.9, 1.1))

        # Curva predicha
        mix = mezcla_convexa(
            protos,
            proba,
            k_hat,
            shift=shift,
            scale=scale
        )
        proto_hat = protos[k_hat]
        rel7 = emerg_rel_7d_from_acum(mix)

        # Gráfico
        dias = np.arange(1, JD_MAX+1)
        df_plot = pd.DataFrame({
            "Día": dias,
            "Predicción": mix,
            "Patrón más probable": proto_hat,
            "Emergencia_relativa_7d": rel7
        })

        base = alt.Chart(df_plot).encode(
            x=alt.X("Día:Q", scale=alt.Scale(domain=[1, JD_MAX]))
        )

        curvas = base.transform_fold(
            ["Predicción", "Patrón más probable"],
            as_=["Serie", "Valor"]
        ).mark_line(strokeWidth=2).encode(
            y=alt.Y("Valor:Q",
                    title="Emergencia acumulada (0–1)",
                    scale=alt.Scale(domain=[0, 1])),
            color="Serie:N",
            tooltip=["Serie:N", alt.Tooltip("Valor:Q", format=".3f"), "Día:Q"]
        )

        max_rel = float(np.nanmax(rel7)) if np.isfinite(np.nanmax(rel7)) else 1.0
        rel_area = base.mark_area(opacity=0.35).encode(
            y=alt.Y("Emergencia_relativa_7d:Q",
                    axis=alt.Axis(title="Emergencia relativa 7d"),
                    scale=alt.Scale(domain=[0, max_rel * 1.1]))
        )

        chart = alt.layer(curvas, rel_area).resolve_scale(y="independent").properties(
            height=420,
            title=f"Predicción v5.2 — Patrón C{k_hat} (conf {conf:.2f}, shift {shift:+1.1f}, scale {scale:.3f})"
        )

        st.altair_chart(chart, use_container_width=True)

        # Tabla de probabilidades por patrón
        rows = []
        for k_i in range(K):
            yrs_txt = ", ".join(map(str, cluster_years.get(k_i, []))) if cluster_years.get(k_i) else "—"
            rows.append((f"C{k_i}", float(proba[k_i]), yrs_txt))
        df_proba = pd.DataFrame(rows, columns=["Patrón", "Probabilidad", "Años"])
        df_proba = df_proba.sort_values("Probabilidad", ascending=False).reset_index(drop=True)

        st.markdown("### 🔢 Probabilidades por patrón")
        st.dataframe(df_proba.style.format({"Probabilidad": "{:.3f}"}), use_container_width=True)

        # Descarga de curva predicha
        out = pd.DataFrame({
            "Día": dias,
            "Emergencia_predicha": mix,
            "Patrón_mas_probable": proto_hat,
            "Emergencia_relativa_7d": rel7
        })
        st.download_button(
            "⬇️ Descargar curva predicha (CSV)",
            out.to_csv(index=False).encode("utf-8"),
            file_name="curva_predicha_v52.csv",
            mime="text/csv"
        )

# ---------------------------------------------------------------
# TAB 3 — Comparación curva REAL vs PREDICHA (v5.2)
# ---------------------------------------------------------------
with tabs[2]:

    st.subheader("📊 Comparación curva real vs curva predicha (RMSE, MAE) — PREDWEEM v5.2")

    st.markdown("""
    Este módulo permite:
    - Cargar una **curva real** (XLSX con JD y valores)
    - Cargar la **meteorología** correspondiente
    - Cargar el **modelo v5.2**
    - Generar la **curva predicha completa**
    - Compararla con la curva real
    - Calcular **RMSE** y **MAE**
    """)

    modelo_eval = st.file_uploader("📦 Modelo v5.2 (joblib)", type=["joblib"], key="eval_model_v52")
    meteo_eval  = st.file_uploader("📘 Meteorología del año a evaluar (XLSX)", type=["xlsx","xls"], key="eval_meteo_v52")
    curva_real  = st.file_uploader("📈 Curva real (XLSX)", type=["xlsx","xls"], key="eval_curva_v52")

    btn_eval = st.button("🔎 Ejecutar comparación")

    if btn_eval:
        if not (modelo_eval and meteo_eval and curva_real):
            st.error("Faltan uno o más archivos."); st.stop()

        bundle = joblib.load(modelo_eval)

        JD_MAX = bundle["JD_MAX"]
        xsc_meteo = bundle["xsc_meteo"]
        xsc_clf   = bundle["xsc_clf"]

        feat_names_meteo = bundle["feat_names_meteo"]
        feat_names_embed = bundle["feat_names_embed"]
        clf_feat_names   = bundle["clf_feat_names"]

        clf           = bundle["clf"]
        protos        = bundle["protos"]
        regs_shift    = bundle["regs_shift"]
        regs_scale    = bundle["regs_scale"]
        embed_regs    = bundle["embed_regressors"]
        cluster_years = bundle.get("cluster_years", {})

        K = protos.shape[0]

        # Meteorología
        dfm_eval, f_eval = build_features_meteo(pd.read_excel(meteo_eval))
        X_m_e = np.array([[f_eval[k] for k in feat_names_meteo]], float)
        X_m_e_s = xsc_meteo.transform(X_m_e)

        # Embeddings predichos desde meteo
        emb_pred_eval = {emb: float(embed_regs[emb].predict(X_m_e_s)[0])
                         for emb in feat_names_embed}

        X_clf_e = np.array([[f_eval[k] for k in feat_names_meteo] +
                            [emb_pred_eval[k] for k in feat_names_embed]], float)
        X_clf_e_s = xsc_clf.transform(X_clf_e)

        # Clasificación de patrón
        proba_e = clf.predict_proba(X_clf_e_s)[0]
        k_hat_e = int(np.argmax(proba_e))
        conf_e  = float(proba_e[k_hat_e])

        # Warp shift/scale
        shift_e = float(regs_shift[k_hat_e].predict(X_clf_e_s)[0]) if k_hat_e in regs_shift else 0.0
        scale_e = float(regs_scale[k_hat_e].predict(X_clf_e_s)[0]) if k_hat_e in regs_scale else 1.0
        scale_e = float(np.clip(scale_e, 0.9, 1.1))

        # Curva predicha
        mix_e = mezcla_convexa(
            protos,
            proba_e,
            k_hat_e,
            shift=shift_e,
            scale=scale_e
        )

        # Curva REAL
        curva_r = curva_desde_xlsx_anual(curva_real)[:JD_MAX]
        curva_r = np.maximum.accumulate(np.clip(curva_r, 0, 1))

        # Métricas
        rmse = float(np.sqrt(np.mean((curva_r - mix_e)**2)))
        mae  = float(np.mean(np.abs(curva_r - mix_e)))

        st.success(f"RMSE = {rmse:.4f}     |     MAE = {mae:.4f}")

        # Gráfico real vs predicha
        dias = np.arange(1, JD_MAX+1)
        df_cmp = pd.DataFrame({
            "Día": dias,
            "Real": curva_r,
            "Predicha": mix_e
        }).melt("Día", var_name="Serie", value_name="Valor")

        chart_cmp = (
            alt.Chart(df_cmp)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("Día:Q", scale=alt.Scale(domain=[1, JD_MAX])),
                y=alt.Y("Valor:Q", title="Emergencia acumulada (0–1)",
                        scale=alt.Scale(domain=[0, 1])),
                color="Serie:N"
            )
            .properties(
                height=420,
                title=f"Comparación Real vs Predicha (Patrón C{k_hat_e}, conf {conf_e:.2f}, shift {shift_e:+1.1f}, scale {scale_e:.3f})"
            )
        )

        st.altair_chart(chart_cmp, use_container_width=True)

        # Descarga
        out_cmp = pd.DataFrame({
            "Día": dias,
            "Real": curva_r,
            "Predicha": mix_e
        })
        st.download_button(
            "⬇️ Descargar comparación (CSV)",
            out_cmp.to_csv(index=False).encode("utf-8"),
            file_name="comparacion_real_vs_predicha_v52.csv",
            mime="text/csv"
        )





