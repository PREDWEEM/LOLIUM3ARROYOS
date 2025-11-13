# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM v5.3 — Mixture-of-Prototypes (DTW ponderado + ILN)
# ===============================================================
# Mejoras v5.3:
# - Clasificación basada en la forma de la curva entre JD 30–121
# - Inclusión de:
#     * inicio_emergencia (primer día con E>0)
#     * frac_1_120 (fracción acumulada a JD 120)
#     * tasa_prom_30_120 (tasa media 30–120)
#     * max_inc_30_120 (máximo incremento diario 30–120)
#     * dia_max_inc_30_120 (JD de ese máximo)
#     * skew_inc_30_120 (asimetría de incrementos 30–120)
#     * kurt_inc_30_120 (curtosis de incrementos 30–120)
# - NUEVO: ILN (Increment Local Normalized) → vector de 12 rasgos
#   que captura la distribución temporal del incremento 30–121,
#   clave para matchear casos como 2008 ~ 2013.
# - DTW ponderado (mayor peso tramo JD 30–121) + banda Sakoe–Chiba
# - Clasificador METEO + EMBEDDINGS
# - Comparación curva real vs predicha (RMSE/MAE)
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
    page_title="PREDWEEM v5.3 — Mixture-of-Prototypes (DTW ponderado + ILN)",
    layout="wide"
)
st.title("🌾 PREDWEEM v5.3 — DTW ponderado + Embeddings de curva (ILN)")

JD_MAX = 274
XRANGE = (1, JD_MAX)

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
    Asegura que la tabla tenga 'jd' (día juliano) y reindexa al rango 1..274
    con interpolación y relleno.
    """
    df = df.copy()
    df.columns = _make_unique(df.columns)
    if "jd" not in df.columns:
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
# CURVAS DE EMERGENCIA DESDE XLSX
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

    daily = np.zeros(365, dtype=float)
    for d,v in zip(jd,val):
        if pd.notna(d) and 1 <= int(d) <= 365:
            daily[int(d)-1] += float(v)
    if paso > 1:
        daily = np.convolve(daily, np.ones(7)/7, mode="same")

    acum = np.cumsum(daily)
    if np.nanmax(acum) == 0:
        return np.zeros(JD_MAX, dtype=float)
    curva = (acum / np.nanmax(acum))[:JD_MAX]
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
    Como la curva está normalizada 0–1, es simplemente E(120).
    """
    if len(curva) == 0:
        return 0.0
    idx_120 = min(119, len(curva)-1)
    return float(curva[idx_120])

def detectar_inicio_emergencia(curva: np.ndarray) -> int:
    """
    Detecta el JD de inicio de emergencia:
      ➜ primer JD donde la emergencia acumulada supera 0.
    Si la curva nunca supera 0 ⇒ devuelve 999 (desconocido).
    """
    idx = np.where(curva > 0)[0]
    if len(idx) == 0:
        return 999
    return int(idx[0] + 1)  # índice → día juliano

def analizar_incrementos_30_120(curva: np.ndarray):
    """
    Analiza el tramo JD 30–120:
      - tasa_promedio_30_120: incremento medio diario
      - max_incremento_30_120: mayor ∆E en un día
      - dia_max_incremento_30_120: JD donde ocurre ese máximo ∆E
    """
    i1, i2 = 29, 119  # JD30=idx29, JD120=idx119
    segmento = curva[i1:i2+1]

    if len(segmento) < 2:
        return {
            "tasa_promedio_30_120": 0.0,
            "max_incremento_30_120": 0.0,
            "dia_max_incremento_30_120": 30
        }

    inc = np.diff(segmento)
    tasa_promedio = (segmento[-1] - segmento[0]) / (i2 - i1)
    max_inc = float(np.max(inc))
    idx_max_inc = int(np.argmax(inc))
    dia_max_inc = 30 + idx_max_inc

    return {
        "tasa_promedio_30_120": float(tasa_promedio),
        "max_incremento_30_120": max_inc,
        "dia_max_incremento_30_120": dia_max_inc
    }

def embedding_curva_forma(curva: np.ndarray) -> dict:
    """
    Embedding de forma de la curva:
      - inicio_emergencia
      - frac_1_120
      - tasa_prom_30_120
      - max_inc_30_120
      - dia_max_inc_30_120
      - skew_inc_30_120
      - kurt_inc_30_120
    """
    inicio = detectar_inicio_emergencia(curva)
    frac120 = frac_curva_1_120(curva)
    anal = analizar_incrementos_30_120(curva)

    i1, i2 = 29, 119
    segmento = curva[i1:i2+1]
    if len(segmento) < 3:
        inc = np.zeros(3)
    else:
        inc = np.diff(segmento)

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

def embedding_ILN(curva: np.ndarray, n_bins: int = 12) -> np.ndarray:
    """
    ILN (Increment Local Normalized):
    Huella temporal de la distribución relativa de los incrementos
    de emergencia en el tramo JD 30–121.
    - Calcula incrementos diarios.
    - Recorta 30–121.
    - Normaliza para que sumen 1.
    - Agrupa en n_bins y suma la "energía" en cada bin.

    Devuelve un vector de longitud n_bins.
    """
    inc = np.diff(np.insert(curva, 0, 0.0))
    inc_win = inc[29:121]  # JD 30–121

    total = inc_win.sum() + 1e-9
    inc_norm = inc_win / total

    bins = np.array_split(inc_norm, n_bins)
    iln = np.array([b.sum() for b in bins], dtype=float)
    return iln

# ===============================================================
# DTW PONDERADO + BANDA SAKOE–CHIBA
# ===============================================================
def dtw_distance_weighted(a: np.ndarray,
                          b: np.ndarray,
                          band: int = 10,
                          w_focus: float = 3.0) -> float:
    """
    Distancia DTW ponderada entre dos curvas de emergencia acumulada,
    usando únicamente el segmento JD 30–121 y una banda de Sakoe–Chiba
    ±band en torno a la diagonal.

    - Tramo 30–121 tiene peso w_focus.
    """
    a_seg = a[29:121]
    b_seg = b[29:121]

    n, m = len(a_seg), len(b_seg)
    w = w_focus

    D = np.full((n+1, m+1), np.inf, dtype=float)
    D[0, 0] = 0.0

    for i in range(1, n+1):
        j_min = max(1, i - band)
        j_max = min(m, i + band)
        ai = a_seg[i-1]
        for j in range(j_min, j_max+1):
            diff = ai - b_seg[j-1]
            cost = w * (diff * diff)
            D[i, j] = cost + min(
                D[i-1, j],   # inserción
                D[i, j-1],   # eliminación
                D[i-1, j-1]  # match
            )

    return float(np.sqrt(D[n, m]))

# ===============================================================
# FEATURES METEOROLÓGICAS + EMBEDDINGS + OUTLIERS + K-MEDOIDS
# ===============================================================
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
    calcula un conjunto de features agroclimáticos agregados.
    """
    dfm = standardize_cols(dfm)
    dfm = ensure_jd_1_to_274(dfm)
    tmin = dfm["tmin"].astype(float).to_numpy()
    tmax = dfm["tmax"].astype(float).to_numpy()
    tmed = (tmin + tmax) / 2.0
    prec = dfm["prec"].astype(float).to_numpy()
    jd   = dfm["jd"].astype(int).to_numpy()

    mask_FM = (jd >= 32) & (jd <= 151)
    gdd5 = np.cumsum(np.maximum(tmed - 5, 0))
    gdd3 = np.cumsum(np.maximum(tmed - 3, 0))

    if not np.any(mask_FM):
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

    def ma(x, w):
        k = np.ones(w) / w
        return np.convolve(x, k, "same")
    idx_may = min(150, len(tmed)-1)
    f["tmed14_May"] = float(ma(tmed, 14)[idx_may])
    f["tmed28_May"] = float(ma(tmed, 28)[idx_may])

    idx_120 = min(119, len(tmed) - 1)
    f["gdd5_120"] = float(gdd5[idx_120])
    f["pp_120"]   = float(np.nansum(prec[: idx_120 + 1]))

    f = {k: f[k] for k in FEATURE_ORDER}
    return dfm, f

# Lista base de embeddings (sin ILN todavía)
EMBED_FEAT_NAMES = [
    "inicio_emergencia",
    "frac_1_120",
    "tasa_prom_30_120",
    "max_inc_30_120",
    "dia_max_inc_30_120",
    "skew_inc_30_120",
    "kurt_inc_30_120"
]

# Agregar ILN (12 bins) a la lista de embeddings
EMBED_ILN_NAMES = [f"iln_{i}" for i in range(12)]
EMBED_FEAT_NAMES = EMBED_FEAT_NAMES + EMBED_ILN_NAMES

def build_embedding_matrix(curves: list, years: list):
    """
    Construye matriz de embeddings de forma para cada curva y año:
    - Rasgos morfológicos escalares
    - ILN (vector 12 bins)
    """
    embs = []
    emb_dicts = []
    for curva, y in zip(curves, years):
        base_emb = embedding_curva_forma(curva)
        iln_vec = embedding_ILN(curva, n_bins=12)

        # combinar en dict
        full_emb = base_emb.copy()
        for i, v in enumerate(iln_vec):
            full_emb[f"iln_{i}"] = float(v)

        row = [full_emb[k] for k in EMBED_FEAT_NAMES]
        embs.append(row)
        emb_dicts.append({"year": int(y), **full_emb})

    emb_matrix = np.array(embs, float)
    return emb_matrix, emb_dicts

def detectar_outliers_embeddings(emb_matrix: np.ndarray,
                                 n_neighbors: int = None,
                                 contamination: float = 0.25):
    """
    Detecta outliers en el espacio de embeddings usando LocalOutlierFactor.
    Devuelve:
      - mask_inliers: boolean array (True = inlier)
      - scores: negative_outlier_factor_ (más negativo = más outlier)
    """
    n_samples = emb_matrix.shape[0]
    if n_neighbors is None:
        n_neighbors = max(5, min(20, int(0.4 * n_samples)))
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        metric="euclidean"
    )
    y_pred = lof.fit_predict(emb_matrix)
    mask_inliers = (y_pred == 1)
    scores = lof.negative_outlier_factor_
    return mask_inliers, scores

def k_medoids_dtw_weighted(curves: list,
                           K: int,
                           max_iter: int = 50,
                           seed: int = 42,
                           band: int = 10,
                           w_focus: float = 3.0):
    """
    k-medoids con DTW ponderado (30–121) sobre las curvas inliers.
    Devuelve:
      - medoid_idx
      - clusters
      - D (matriz de distancias)
    """
    rng = np.random.default_rng(seed)
    N = len(curves)
    if N == 0:
        raise ValueError("No hay curvas para agrupar.")
    if K > N:
        K = N

    idx = rng.choice(N, size=K, replace=False)
    medoid_idx = list(idx)

    D = np.zeros((N, N), float)
    for i in range(N):
        for j in range(i+1, N):
            d = dtw_distance_weighted(curves[i], curves[j],
                                      band=band, w_focus=w_focus)
            D[i, j] = D[j, i] = d

    for _ in range(max_iter):
        assign = np.argmin(D[:, medoid_idx], axis=1)
        new_medoids = []
        for k in range(K):
            members = np.where(assign == k)[0]
            if len(members) == 0:
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

def warp_curve(proto: np.ndarray, shift: float, scale: float) -> np.ndarray:
    """
    Warp simple de la curva prototipo:
      - shift: desplazamiento horizontal
      - scale: escala del eje temporal
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
    Curva predicha = mezcla convexa de prototipos,
    aplicando warp sólo al patrón más probable.
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

# Listas de features para los modelos
REG_INPUT_FEAT_NAMES = FEATURE_ORDER[:]           # METEO → EMBEDDINGS
CLF_FEAT_NAMES       = FEATURE_ORDER[:] + EMBED_FEAT_NAMES[:]

# ===============================================================
# DEFINICIÓN DE TABS (el contenido va en los bloques siguientes)
# ===============================================================
tabs = st.tabs([
    "🧪 Entrenar prototipos + clasificador",
    "🔮 Identificar patrones y predecir",
    "📊 Comparar curva real vs predicha"
])


# ===============================================================
# TAB 1 — ENTRENAMIENTO
# ===============================================================
with tabs[0]:
    st.subheader("🧪 Entrenamiento (k-medoids DTW ponderado + embeddings de curva)")
    st.markdown("""
    Subí:
    - 📘 **Meteorología multianual** (una hoja por año: tmin, tmax, prec, fecha/jd)
    - 📈 **Curvas históricas de emergencia** (un archivo XLSX por año)

    El modelo:
    1. Construye **embeddings de forma** (incluye ILN 30–121).
    2. Detecta **outliers morfológicos** y permite excluirlos.
    3. Aplica **k-medoids con DTW ponderado** sobre las curvas inliers.
    4. Entrena:
        - Reglas **METEO → EMBEDDINGS** (regresión)
        - Clasificador **METEO+EMB → patrón k-medoids**
        - Reglas **METEO → warp (shift/scale)** del patrón más probable.
    """)

    meteo_book = st.file_uploader(
        "📘 Meteorología multianual (una hoja por año)",
        type=["xlsx", "xls"],
        key="train_meteo"
    )
    curvas_files = st.file_uploader(
        "📈 Curvas históricas (XLSX por año, acumulada o semanal)",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        key="train_curves"
    )

    col_k, col_band, col_w, col_seed = st.columns([1, 1, 1, 1])
    with col_k:
        K = st.slider("Número de prototipos/patrones (K)", 2, 10, 5, 1)
    with col_band:
        band = st.slider("Banda Sakoe–Chiba (± días)", 5, 30, 10, 1)
    with col_w:
        w_focus = st.slider("Peso tramo JD 30–121 en DTW", 1.0, 6.0, 3.0, 0.5)
    with col_seed:
        seed = st.number_input("Semilla aleatoria", 0, 99999, 42)

    btn_train = st.button("🚀 Entrenar modelo PREDWEEM v5.3")

    if btn_train:
        if not (meteo_book and curvas_files):
            st.error("⚠️ Cargá ambos conjuntos: meteorología multianual y curvas históricas.")
            st.stop()

        # -----------------------------------------------------------
        # 1) Leer meteorología por año
        # -----------------------------------------------------------
        sheets = pd.read_excel(meteo_book, sheet_name=None)
        meteo_dict = {}
        for name, df in sheets.items():
            df = standardize_cols(df)
            df = ensure_jd_1_to_274(df)
            # Intentar deducir año a partir del nombre de la hoja
            year = None
            try:
                year = int(re.findall(r"\d{4}", str(name))[0])
            except Exception:
                if "fecha" in df.columns and df["fecha"].notna().any():
                    year = int(df["fecha"].dt.year.mode().iloc[0])
            if year and all(c in df.columns for c in ["tmin", "tmax", "prec"]):
                meteo_dict[year] = df[["jd", "tmin", "tmax", "prec"]].copy()

        if not meteo_dict:
            st.error("⛔ No se detectó meteorología válida por año.")
            st.stop()

        st.success(f"✅ Meteorología válida detectada para {len(meteo_dict)} años.")

        # -----------------------------------------------------------
        # 2) Leer curvas de emergencia por año
        # -----------------------------------------------------------
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
            st.error("⛔ No se detectaron curvas válidas (revisar nombres con año en el archivo).")
            st.stop()

        # -----------------------------------------------------------
        # 3) Intersección meteo–curvas
        # -----------------------------------------------------------
        common_years = sorted([y for y in years_list if y in meteo_dict])
        if len(common_years) < 4:
            st.error("⛔ Muy pocos años en común entre meteo y curvas (se recomienda ≥ 5).")
            st.stop()

        curves = [curves_list[years_list.index(y)] for y in common_years]
        st.info(f"📅 Años utilizados en el entrenamiento: {common_years}")

        # -----------------------------------------------------------
        # 4) Embeddings de forma + ILN para cada curva
        # -----------------------------------------------------------
        emb_matrix, emb_dicts = build_embedding_matrix(curves, common_years)
        df_emb = pd.DataFrame(emb_dicts)
        st.markdown("### 🔬 Embeddings de forma por año (incluye ILN 30–121)")
        st.dataframe(df_emb.set_index("year"), use_container_width=True)

        # -----------------------------------------------------------
        # 5) Detección de outliers en espacio de embeddings
        # -----------------------------------------------------------
        mask_inliers, scores = detectar_outliers_embeddings(
            emb_matrix,
            contamination=0.25
        )
        df_out = pd.DataFrame({
            "Año": common_years,
            "is_inlier": mask_inliers,
            "score_LOF": scores
        }).sort_values("Año")

        st.markdown("### 🧹 Detección de outliers morfológicos (LOF sobre embeddings)")
        st.dataframe(df_out, use_container_width=True)

        use_outlier_filter = st.checkbox(
            "Excluir outliers morfológicos (recomendado)",
            value=True
        )

        if use_outlier_filter:
            idx_use = np.where(mask_inliers)[0]
        else:
            idx_use = np.arange(len(common_years))

        if len(idx_use) < 3:
            st.error("⛔ Tras filtrar outliers, quedan muy pocos años para entrenar (min=3).")
            st.stop()

        years_inliers = [common_years[i] for i in idx_use]
        curves_in = [curves[i] for i in idx_use]
        emb_in = emb_matrix[idx_use, :]

        st.success(f"✅ Años usados para clustering (inliers): {years_inliers}")

        # -----------------------------------------------------------
        # 6) k-medoids con DTW ponderado 30–121
        # -----------------------------------------------------------
        st.markdown("### 🧮 Clustering k-medoids con DTW ponderado (segmento 30–121)")
        medoid_idx, clusters, D = k_medoids_dtw_weighted(
            curves_in,
            K=K,
            max_iter=50,
            seed=seed,
            band=band,
            w_focus=w_focus
        )
        protos = [curves_in[i] for i in medoid_idx]

        # asignación por año inlier
        assign_labels = np.argmin(D[:, np.array(medoid_idx)], axis=1)
        cluster_years = {k: [] for k in range(K)}
        for i, y in enumerate(years_inliers):
            cluster_years[int(assign_labels[i])].append(int(y))

        # -----------------------------------------------------------
        # 7) Features METEO por año
        # -----------------------------------------------------------
        meteo_feats_by_year = {}
        X_meteo_rows = []
        for y in years_inliers:
            dfm_y, f_y = build_features_meteo(meteo_dict[y])
            meteo_feats_by_year[y] = f_y
            X_meteo_rows.append([f_y[k] for k in FEATURE_ORDER])

        X_meteo = np.array(X_meteo_rows, float)  # shape: (n_inliers, len(FEATURE_ORDER))

        # -----------------------------------------------------------
        # 8) Reglas METEO → EMBEDDINGS (regresión)
        # -----------------------------------------------------------
        st.markdown("### 🔗 Aprendizaje METEO → Embeddings de forma")

        # escalador para regresión (input sólo meteoclimático)
        xsc_reg = StandardScaler().fit(X_meteo)
        X_meteo_reg = xsc_reg.transform(X_meteo)

        reg_embed = {}
        for j, feat_name in enumerate(EMBED_FEAT_NAMES):
            y_target = emb_in[:, j]
            reg = GradientBoostingRegressor(random_state=seed)
            reg.fit(X_meteo_reg, y_target)
            reg_embed[feat_name] = reg

        st.success(f"✅ Entrenados {len(EMBED_FEAT_NAMES)} regresores METEO → EMBEDDINGS.")

        # -----------------------------------------------------------
        # 9) Clasificador METEO + EMBEDDINGS → patrón
        # -----------------------------------------------------------
        st.markdown("### 🧬 Clasificador METEO+Embeddings → Patrón k-medoids")

        # Construimos matriz de entrada para el clasificador:
        # concatenamos features meteo + embeddings verdaderos
        Z_clf = np.hstack([X_meteo, emb_in])  # (n, len(FEATURE_ORDER)+len(EMBED_FEAT_NAMES))
        y_clf = assign_labels.astype(int)

        xsc_clf = StandardScaler().fit(Z_clf)
        Z_clf_scaled = xsc_clf.transform(Z_clf)

        clf = GradientBoostingClassifier(random_state=seed)
        clf.fit(Z_clf_scaled, y_clf)

        st.success("✅ Clasificador de patrón entrenado.")

        # -----------------------------------------------------------
        # 10) Reglas METEO → warp (shift/scale) del prototipo
        # -----------------------------------------------------------
        st.markdown("### 🕓 Ajuste temporal (warp) de prototipos por patrón")

        regs_shift = {}
        regs_scale = {}

        # usaremos las features de regresión (X_meteo_reg) para ajustar shift/scale
        for k in range(K):
            idx_k = np.where(assign_labels == k)[0]
            if len(idx_k) == 0:
                # cluster vacío, se crea dummy
                regs_shift[k] = GradientBoostingRegressor(random_state=seed)
                regs_scale[k] = GradientBoostingRegressor(random_state=seed)
                continue

            proto_k = protos[k]
            shifts_k = []
            scales_k = []
            Xk_reg = []

            for idx_i in idx_k:
                curva_i = curves_in[idx_i]
                # búsqueda gruesa de shift/scale que minimice RMSE frente al prototipo
                best = (0.0, 1.0, 1e9)
                for sh in range(-20, 21, 5):  # ±20 días en pasos de 5
                    for sc in [0.9, 0.95, 1.0, 1.05, 1.1]:
                        cand = warp_curve(proto_k, sh, sc)
                        rmse = float(np.sqrt(np.mean((cand - curva_i) ** 2)))
                        if rmse < best[2]:
                            best = (float(sh), float(sc), rmse)
                shifts_k.append(best[0])
                scales_k.append(best[1])
                Xk_reg.append(X_meteo_reg[idx_i])  # mismas features meteoclimáticas

            Xk_reg = np.vstack(Xk_reg)

            reg_sh = GradientBoostingRegressor(random_state=seed)
            reg_sc = GradientBoostingRegressor(random_state=seed)
            reg_sh.fit(Xk_reg, np.array(shifts_k))
            reg_sc.fit(Xk_reg, np.array(scales_k))

            regs_shift[k] = reg_sh
            regs_scale[k] = reg_sc

        st.success("✅ Reglas de warp (shift/scale) por patrón entrenadas.")

        # -----------------------------------------------------------
        # 11) Guardar bundle del modelo
        # -----------------------------------------------------------
        assign_by_year = {int(y): int(lbl) for y, lbl in zip(years_inliers, assign_labels)}

        bundle = {
            # escaladores
            "xsc_reg": xsc_reg,
            "xsc_clf": xsc_clf,

            # nombres de features
            "meteo_feat_names": FEATURE_ORDER[:],
            "embed_feat_names": EMBED_FEAT_NAMES[:],

            # modelos regresores y clasificador
            "reg_embed": reg_embed,        # dict feat_name -> GBR
            "clf": clf,                    # METEO+EMB → patrón

            # prototipos y info de clustering
            "protos": np.vstack(protos),   # K x 274
            "cluster_years": cluster_years,
            "assign_by_year": assign_by_year,
            "years_inliers": years_inliers,

            # reglas de warp
            "regs_shift": regs_shift,
            "regs_scale": regs_scale,

            # parámetros DTW
            "dtw_band": band,
            "dtw_w_focus": w_focus
        }

        st.session_state["predweem_v53_bundle"] = bundle

        buf = io.BytesIO()
        joblib.dump(bundle, buf)
        st.download_button(
            "💾 Descargar modelo entrenado (PREDWEEM_v5_3.joblib)",
            data=buf.getvalue(),
            file_name="PREDWEEM_v5_3.joblib",
            mime="application/octet-stream"
        )

        # -----------------------------------------------------------
        # 12) Visualización de prototipos vs años asignados
        # -----------------------------------------------------------
        dias = np.arange(1, JD_MAX + 1)
        dfp = []
        for k, proto in enumerate(protos):
            years_txt = ", ".join(map(str, cluster_years.get(k, []))) if cluster_years.get(k) else "—"
            serie_name = f"C{k} · años: {years_txt}"
            dfp.append(pd.DataFrame({
                "Día": dias,
                "Valor": proto,
                "Serie": serie_name
            }))
        dfp = pd.concat(dfp, ignore_index=True)

        chart = alt.Chart(dfp).mark_line().encode(
            x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE))),
            y=alt.Y("Valor:Q", title="Emergencia acumulada (0–1)",
                    scale=alt.Scale(domain=[0, 1])),
            color="Serie:N"
        ).properties(
            height=420,
            title="Prototipos (medoids DTW ponderado 30–121) y años asociados"
        )
        st.altair_chart(chart, use_container_width=True)

# ===============================================================
# TAB 2 — PREDICCIÓN (FIX COMPLETO ALTair + Sanitización)
# ===============================================================
with tabs[1]:
    st.subheader("🔮 Identificar patrón y predecir curva de emergencia")

    modelo_file = st.file_uploader(
        "📦 Modelo PREDWEEM v5.3 (.joblib)",
        type=["joblib"],
        key="pred_model_v53"
    )

    meteo_file = st.file_uploader(
        "📘 Meteorología nueva (XLSX)",
        type=["xlsx","xls"],
        key="pred_meteo_v53"
    )

    inicio_manual = st.number_input(
        "📍 Fecha de inicio de emergencia (JD) medida a campo (opcional):",
        min_value=1,
        max_value=300,
        value=300,  # valor sentinela
        help="Dejar en 300 si no se conoce la fecha real."
    )
    usa_inicio_manual = (inicio_manual < 300)

    btn_pred = st.button("🚀 Ejecutar predicción", key="btn_pred_v53")

    if btn_pred:

        # -----------------------------------------------------------
        # 0) Validación inicial
        # -----------------------------------------------------------
        if not (modelo_file and meteo_file):
            st.error("Cargá el modelo y la meteorología.")
            st.stop()

        # -----------------------------------------------------------
        # 1) Cargar modelo entrenado v5.3
        # -----------------------------------------------------------
        bundle = joblib.load(modelo_file)
        xsc_reg = bundle["xsc_reg"]
        xsc_clf = bundle["xsc_clf"]
        reg_embed = bundle["reg_embed"]
        clf = bundle["clf"]

        protos = bundle["protos"]               # K x 274
        regs_shift = bundle["regs_shift"]
        regs_scale = bundle["regs_scale"]

        meteo_feat_names = bundle["meteo_feat_names"]
        embed_feat_names = bundle["embed_feat_names"]

        # Asegurar prototipos (sanitización)
        protos = np.array([np.nan_to_num(p[:JD_MAX], nan=0.0) for p in protos])

        K = protos.shape[0]
        dias = np.arange(1, JD_MAX + 1)

        # -----------------------------------------------------------
        # 2) Leer meteorología nueva
        # -----------------------------------------------------------
        dfm = pd.read_excel(meteo_file)
        dfm, f_new = build_features_meteo(dfm)

        X_new = np.array([[f_new[k] for k in meteo_feat_names]], float)
        X_new_reg = xsc_reg.transform(X_new)

        # -----------------------------------------------------------
        # 3) Predecir EMBEDDINGS (METEO → EMBEDDING CURVA)
        # -----------------------------------------------------------
        emb_pred = {}
        for feat in embed_feat_names:
            emb_pred[feat] = float(reg_embed[feat].predict(X_new_reg)[0])

        # Si hay inicio manual → reemplazar
        if usa_inicio_manual and "inicio_emergencia" in emb_pred:
            emb_pred["inicio_emergencia"] = float(inicio_manual)

        # vector METEO+EMBEDDINGS
        Z_new = np.array([
            list(f_new.values()) +
            [emb_pred[k] for k in embed_feat_names]
        ], float)
        Z_new_scaled = xsc_clf.transform(Z_new)

        # -----------------------------------------------------------
        # 4) Clasificar patrón
        # -----------------------------------------------------------
        proba = clf.predict_proba(Z_new_scaled)[0]
        k_hat = int(np.argmax(proba))
        conf = float(proba[k_hat])

        st.markdown(f"### 🎯 Patrón predicho: **C{k_hat}** · Prob = {conf:.3f}")

        # -----------------------------------------------------------
        # 5) Predecir shift/scale
        # -----------------------------------------------------------
        shift = float(regs_shift[k_hat].predict(X_new_reg)[0])
        scale = float(regs_scale[k_hat].predict(X_new_reg)[0])
        scale = float(np.clip(scale, 0.9, 1.1))

        st.markdown(f"""
        **Warp temporal predicho:**  
        - shift = {shift:+.1f} días  
        - scale = {scale:.3f}
        """)

        # -----------------------------------------------------------
        # 6) Construir curva predicha (mezcla convexa)
        # -----------------------------------------------------------
        mix = mezcla_convexa(protos, proba, k_hat, shift, scale)

        # sanitizar mix
        mix = np.nan_to_num(mix[:JD_MAX], nan=0.0)
        mix = np.maximum.accumulate(np.clip(mix, 0, 1))

        # emergencia relativa 7d
        rel7 = emerg_rel_7d_from_acum(mix)
        rel7 = np.nan_to_num(rel7[:JD_MAX], nan=0.0)

        # prototipo sanitizado
        proto_hat = np.nan_to_num(protos[k_hat][:JD_MAX], nan=0.0)

            # ================= GRÁFICO SEGURO ALTair ======================
        # Sanitizar y forzar misma longitud
        n = min(len(dias), len(mix), len(proto_hat), len(rel7))
        day = np.arange(1, n + 1)
        
        pred = np.nan_to_num(mix[:n], nan=0.0)
        proto = np.nan_to_num(proto_hat[:n], nan=0.0)
        rel7_safe = np.nan_to_num(rel7[:n], nan=0.0)
        
        df_plot = pd.DataFrame({
            "day": day,
            "pred": pred,
            "proto": proto,
            "rel7": rel7_safe
        })
        
        # Long-form para las líneas (pred vs proto)
        df_lines = df_plot.melt(
            id_vars="day",
            value_vars=["pred", "proto"],
            var_name="series",
            value_name="value"
        )
        
        st.markdown("### 📈 Curva predicha vs patrón más probable")
        
        base = alt.Chart(df_plot).properties(height=420)
        
        # Líneas de emergencia acumulada
        chart_lines = alt.Chart(df_lines).mark_line(strokeWidth=2).encode(
            x=alt.X("day:Q", scale=alt.Scale(domain=[1, n]), title="Día juliano"),
            y=alt.Y("value:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    title="Emergencia acumulada (0–1)"),
            color=alt.Color(
                "series:N",
                title="Serie",
                scale=alt.Scale(domain=["pred", "proto"],
                                range=["#1f77b4", "#ff7f0e"]),  # colores estándar
                legend=alt.Legend(labelExpr="datum.label == 'pred' ? 'Predicción' : 'Patrón'")
            ),
            tooltip=[
                alt.Tooltip("series:N", title="Serie"),
                alt.Tooltip("value:Q", title="Emergencia", format=".3f"),
                alt.Tooltip("day:Q", title="Día juliano")
            ]
        )
        
        # Área de emergencia relativa (eje secundario)
        max_rel = float(np.nanmax(rel7_safe))
        if not np.isfinite(max_rel) or max_rel <= 0:
            max_rel = 1.0
        
        chart_rel = alt.Chart(df_plot).mark_area(opacity=0.3).encode(
            x=alt.X("day:Q"),
            y=alt.Y("rel7:Q",
                    axis=alt.Axis(title="Emergencia relativa semanal"),
                    scale=alt.Scale(domain=[0, max_rel * 1.1]))
        )
        
        chart = alt.layer(chart_lines, chart_rel).resolve_scale(
            y="independent"
        ).properties(
            title=f"Predicción final (C{k_hat} · prob {conf:.2f})"
        )
        
        st.altair_chart(chart, use_container_width=True)


        # -----------------------------------------------------------
        # 9) Tabla de probabilidades
        # -----------------------------------------------------------
        df_proba = pd.DataFrame({
            "Cluster": [f"C{i}" for i in range(K)],
            "Probabilidad": proba
        }).sort_values("Probabilidad", ascending=False)

        st.markdown("### 🔢 Probabilidades de cada patrón")
        st.dataframe(df_proba.style.format({"Probabilidad": "{:.3f}"}))

        # -----------------------------------------------------------
        # 10) Descargar CSV
        # -----------------------------------------------------------
        st.session_state["mix_last_pred"] = mix  # importante para TAB 3

        df_out = pd.DataFrame({
            "Día": dias,
            "Emergencia_predicha": mix,
            "Patrón_C_hat": proto_hat,
            "Relativa_7d": rel7
        })

        st.download_button(
            "⬇️ Descargar curva predicha (CSV)",
            df_out.to_csv(index=False).encode("utf-8"),
            file_name="predweem_v53_prediccion_fix.csv",
            mime="text/csv"
        )








# ===============================================================
# TAB 3 — COMPARAR CURVA REAL vs CURVA PREDICHA
# ===============================================================
with tabs[2]:
    st.subheader("📊 Comparación curva REAL vs curva PREDICHA (RMSE/MAE)")

    st.markdown("""
    ### ¿Cómo funciona este módulo?
    1. Primero ejecutá la predicción en el TAB 2.  
    2. Luego subí la curva **real** (XLSX) del mismo año.  
    3. El sistema:
       - Lee la curva real (diaria o semanal)  
       - La transforma en curva acumulada 0–1  
       - Calcula **RMSE** y **MAE** contra la curva predicha  
       - Grafica ambas curvas superpuestas  
    """)

    curva_real_file = st.file_uploader(
        "📘 Cargar curva REAL (XLSX)",
        type=["xlsx", "xls"],
        key="real_curve"
    )

    btn_comp = st.button("🔍 Comparar curvas")

    if btn_comp:

        # -------------------------------------------
        # 1) Verificar que la predicción ya esté hecha
        # -------------------------------------------
        if "mix_last_pred" not in st.session_state:
            st.error("⚠ Antes debés ejecutar una predicción en el TAB 2.")
            st.stop()

        y_pred = st.session_state["mix_last_pred"]
        dias = np.arange(1, JD_MAX + 1)

        if curva_real_file is None:
            st.error("Cargá una curva REAL para comparar.")
            st.stop()

        # -------------------------------------------
        # 2) Leer curva REAL
        # -------------------------------------------
        y_real = curva_desde_xlsx_anual(curva_real_file)
        y_real = y_real[:JD_MAX]
        y_real = np.maximum.accumulate(np.clip(y_real, 0, 1))

        # -------------------------------------------
        # 3) Calcular errores
        # -------------------------------------------
        rmse = float(np.sqrt(np.mean((y_real - y_pred)**2)))
        mae = float(np.mean(np.abs(y_real - y_pred)))

        st.markdown(f"""
        ## 🧮 Métricas de comparación  
        **RMSE:** {rmse:.4f}  
        **MAE:** {mae:.4f}  
        """)

        # -------------------------------------------
        # 4) Graficar REAL vs PREDICHA
        # -------------------------------------------
        df_cmp = pd.DataFrame({
            "Día": dias,
            "Real": y_real,
            "Predicha": y_pred
        }).melt("Día", var_name="Serie", value_name="Valor")

        chart_cmp = alt.Chart(df_cmp).mark_line(strokeWidth=2).encode(
            x=alt.X("Día:Q", scale=alt.Scale(domain=[1, JD_MAX])),
            y=alt.Y("Valor:Q", scale=alt.Scale(domain=[0, 1]),
                    title="Emergencia acumulada (0–1)"),
            color="Serie:N",
            tooltip=["Serie", alt.Tooltip("Valor:Q", format=".3f"), "Día:Q"]
        ).properties(
            height=420,
            title=f"Curva REAL vs PREDICHA  •  RMSE={rmse:.4f} • MAE={mae:.4f}"
        )

        st.altair_chart(chart_cmp, use_container_width=True)

        # -------------------------------------------
        # 5) Exportar resultados
        # -------------------------------------------
        df_out = pd.DataFrame({
            "Día": dias,
            "Real": y_real,
            "Predicha": y_pred
        })

        st.download_button(
            "⬇️ Descargar comparación (CSV)",
            df_out.to_csv(index=False).encode("utf-8"),
            file_name="comparacion_real_vs_predicha.csv",
            mime="text/csv"
        )

# ===============================================================
# 🔍 MÓDULO DE VALIDACIÓN LEAVE-ONE-OUT (LOOCV)
# ===============================================================

def predweem_train_single(meteo_dict, curves_dict, K=5, seed=42):
    """
    ENTRENAMIENTO COMPLETO PREDWEEM (sin similitud climática)
    usado dentro del LOOCV.
    """
    years = sorted(curves_dict.keys())
    curves = [curves_dict[y] for y in years]

    # Clustering k-medoids DTW
    medoid_idx, clusters, D = k_medoids_dtw(curves, K=K, seed=seed)
    protos = [curves[i] for i in medoid_idx]

    # Features
    feat_rows = []
    for y in years:
        _, f = build_features_meteo(meteo_dict[y])
        feat_rows.append([f[k] for k in FEATURE_ORDER])

    X = np.array(feat_rows, float)
    y_lbl = np.argmin(D[:, np.array(medoid_idx)], axis=1)  # cluster real
    xsc = StandardScaler().fit(X)
    Xs = xsc.transform(X)

    # Clasificador
    clf = GradientBoostingClassifier(random_state=seed)
    clf.fit(Xs, y_lbl)

    # Warping (shift/scale)
    regs_shift, regs_scale = {}, {}
    for k in range(K):
        idx = np.where(y_lbl == k)[0]
        if len(idx) == 0:
            continue
        proto = protos[k]
        shifts, scales, Xk_rows = [], [], []
        for ii in idx:
            curv = curves[ii]
            best = (0.0, 1.0, 1e9)
            for sh in range(-20, 21, 5):
                for sc in [0.9, 0.95, 1.0, 1.05, 1.1]:
                    cand = warp_curve(proto, sh, sc)
                    rmse = float(np.sqrt(np.mean((cand - curv)**2)))
                    if rmse < best[2]:
                        best = (sh, sc, rmse)
            shifts.append(best[0])
            scales.append(best[1])
            Xk_rows.append(Xs[ii])

        Xk = np.vstack(Xk_rows)
        regs_shift[k] = GradientBoostingRegressor().fit(Xk, np.array(shifts))
        regs_scale[k] = GradientBoostingRegressor().fit(Xk, np.array(scales))

    return {
        "xsc": xsc,
        "clf": clf,
        "protos": np.vstack(protos),
        "regs_shift": regs_shift,
        "regs_scale": regs_scale,
        "assign": y_lbl,
        "years": years
    }

# ---------------------------------------------------------------
# LOOCV
# ---------------------------------------------------------------

def predweem_loocv(meteo_dict, curves_dict, K=5):
    rows = []
    all_years = sorted(curves_dict.keys())

    for y_test in all_years:
        # Conjuntos train y test
        train_curves = {y: curves_dict[y] for y in all_years if y != y_test}
        train_meteo  = {y: meteo_dict[y]  for y in all_years if y != y_test}

        # Entrenar con TRAIN
        model = predweem_train_single(train_meteo, train_curves, K=K)

        # Extraer partes del modelo
        xsc = model["xsc"]
        clf = model["clf"]
        protos = model["protos"]
        regs_shift = model["regs_shift"]
        regs_scale = model["regs_scale"]

        # Procesar año de test
        dfm, f_new = build_features_meteo(meteo_dict[y_test])
        X  = np.array([[f_new[k] for k in FEATURE_ORDER]], float)
        Xs = xsc.transform(X)

        # Clasificación
        proba = clf.predict_proba(Xs)[0]
        k_hat = int(np.argmax(proba))
        conf  = proba[k_hat]

        # Warp predicho
        shift = regs_shift[k_hat].predict(Xs)[0] if k_hat in regs_shift else 0.0
        scale = regs_scale[k_hat].predict(Xs)[0] if k_hat in regs_scale else 1.0
        scale = float(np.clip(scale, 0.9, 1.1))

        # Curva predicha
        mix = mezcla_convexa(protos, proba, k_hat, shift, scale)

        # Curva real
        y_real = curves_dict[y_test][:len(mix)]

        # Métricas
        rmse = float(np.sqrt(np.mean((y_real - mix)**2)))
        mae  = float(np.mean(np.abs(y_real - mix)))

        rows.append((
            y_test,         # Año de validación
            k_hat,          # Patrón predicho
            conf,           # Probabilidad
            rmse,
            mae,
            float(shift),
            float(scale)
        ))

    df = pd.DataFrame(rows, columns=[
        "Año_validado",
        "Patrón_predicho",
        "Confianza",
        "RMSE",
        "MAE",
        "Shift",
        "Scale"
    ])

    return df

# ===============================================================
# 🔍 GRÁFICOS PARA VALIDACIÓN LOOCV
# ===============================================================

import matplotlib.pyplot as plt
import seaborn as sns

def plot_loocv_results(df_loocv, curves_dict, meteo_dict, K):
    st.subheader("📊 Resultados LOOCV")

    st.dataframe(df_loocv.style.format({
        "RMSE": "{:.4f}",
        "MAE": "{:.4f}",
        "Confianza": "{:.3f}",
        "Shift": "{:+.2f}",
        "Scale": "{:.3f}"
    }), use_container_width=True)

    # ===========================================================
    # 📉 Distribución de errores
    # ===========================================================
    st.markdown("### 📉 Distribución de errores RMSE y MAE")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(df_loocv["RMSE"], bins=10, kde=True, ax=ax[0], color="royalblue")
    ax[0].set_title("Distribución RMSE")
    ax[0].set_xlabel("RMSE")

    sns.histplot(df_loocv["MAE"], bins=10, kde=True, ax=ax[1], color="darkorange")
    ax[1].set_title("Distribución MAE")
    ax[1].set_xlabel("MAE")

    st.pyplot(fig)

    # ===========================================================
    # 🎯 RMSE vs Confianza
    # ===========================================================
    st.markdown("### 🎯 Relación RMSE vs Confianza de la predicción")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=df_loocv,
        x="Confianza",
        y="RMSE",
        size="MAE",
        hue="Patrón_predicho",
        palette="tab10",
        sizes=(50, 300),
    )
    plt.title("RMSE vs Confianza")
    plt.xlabel("Confianza del patrón predicho")
    plt.ylabel("RMSE")
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)

    # ===========================================================
    # 🟥 Clasificación correcta vs incorrecta
    # ===========================================================
    if "Patrón_real" in df_loocv.columns:
        df_loocv["Correcto"] = df_loocv["Patrón_real"] == df_loocv["Patrón_predicho"]

    else:
        # si no tenemos patrón real, estimar usando clustering completo
        df_loocv["Correcto"] = df_loocv["RMSE"] < df_loocv["RMSE"].median()

    st.markdown("### 🟥 Tasa de acierto del modelo")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(
        data=df_loocv,
        x="Correcto",
        palette=["tomato", "seagreen"]
    )
    plt.title("Clasificación correcta vs incorrecta")
    plt.xlabel("¿Correctamente clasificado?")
    plt.ylabel("Cantidad de años")
    st.pyplot(fig)

    # ===========================================================
    # 🔥 Heatmap RMSE por año (similitud real vs predicho)
    # ===========================================================
    st.markdown("### 🔥 Matriz de error (RMSE por año)")

    df_r = df_loocv.pivot_table(
        index="Año_validado",
        values="RMSE"
    )

    fig, ax = plt.subplots(figsize=(5, len(df_r) * 0.4 + 2))
    sns.heatmap(df_r, cmap="viridis", annot=True, fmt=".3f", cbar=True)
    plt.title("RMSE por año validado")
    st.pyplot(fig)

    # ===========================================================
    # 📈 Selección interactiva de un año para ver curva real vs predicha
    # ===========================================================
    st.markdown("### 📈 Curva real vs predicha (seleccionar año)")

    year_sel = st.selectbox("Seleccionar año:", df_loocv["Año_validado"].tolist())

    # Buscar fila seleccionada
    row = df_loocv[df_loocv["Año_validado"] == year_sel].iloc[0]

    # Generar nuevamente la predicción para graficar
    dfm, f_new = build_features_meteo(meteo_dict[year_sel])
    X  = np.array([[f_new[k] for k in FEATURE_ORDER]], float)

    # recalcular predicción con el modelo entrenado sin ese año
    train_curves = {y: curves_dict[y] for y in curves_dict if y != year_sel}
    train_meteo  = {y: meteo_dict[y] for y in meteo_dict  if y != year_sel}

    model = predweem_train_single(train_meteo, train_curves, K=K)

    xsc = model["xsc"]
    clf = model["clf"]
    protos = model["protos"]
    regs_shift = model["regs_shift"]
    regs_scale = model["regs_scale"]

    Xs = xsc.transform(X)
    proba = clf.predict_proba(Xs)[0]
    k_hat = int(np.argmax(proba))

    shift = regs_shift[k_hat].predict(Xs)[0] if k_hat in regs_shift else 0.0
    scale = regs_scale[k_hat].predict(Xs)[0]  if k_hat in regs_scale else 1.0
    scale = float(np.clip(scale, 0.9, 1.1))

    mix = mezcla_convexa(protos, proba, k_hat, shift, scale)
    y_real = curves_dict[year_sel]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_real, label=f"Real {year_sel}", color="black", linewidth=2)
    ax.plot(mix, label=f"Predicción C{k_hat}", color="dodgerblue", linestyle="--", linewidth=2)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Curva real vs predicha – Año {year_sel}")
    ax.legend()
    st.pyplot(fig)



















