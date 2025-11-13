# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM v5.1 — Mixture-of-Prototypes (DTW + Monotone)
# ===============================================================
# - K prototipos (k-medoids con DTW, sin libs extra)
# - Clasificador meteo (+ inicio_emergencia) → patrón (GradientBoostingClassifier)
# - Curva predicha = mezcla convexa de prototipos + warp (shift/scale)
# - Monotonía garantizada (acumulado de incrementos ≥ 0)
# - Identifica años por patrón (cluster_years)
# - Clasificación de patrones basada SOLO en la curva entre JD 30–121 (DTW)
# - Variable fenológica adicional: día de inicio de la emergencia (inicio_emergencia)
# - Módulo para comparar curva real vs predicha (RMSE/MAE)
# - Incluye fracción de emergencia acumulada entre JD 1–120 (salida diagnóstica)
# - Rango JD 1..274 (1-ene → 1-oct)
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import re, io, joblib
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------
# CONFIGURACIÓN GENERAL STREAMLIT
# ---------------------------------------------------------------
st.set_page_config(page_title="PREDWEEM v5.1 — Mixture-of-Prototypes (DTW)", layout="wide")
st.title("🌾 PREDWEEM v5.1 — Mixture-of-Prototypes (DTW + Monotone)")

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

def frac_curva_1_120(y_acum: np.ndarray) -> float:
    """
    Fracción de emergencia acumulada al día juliano 120.
    Dado que la curva está normalizada 0–1, es simplemente E(120).
    (Se usa como salida diagnóstica).
    """
    if len(y_acum) == 0:
        return 0.0
    idx_120 = min(119, len(y_acum)-1)  # JD120 → índice 119
    return float(y_acum[idx_120])

def detectar_inicio_emergencia(curva: np.ndarray) -> int:
    """
    Detecta el día juliano de inicio de la emergencia.

    Definición (según tu criterio):
      ➜ primer día (JD) donde la emergencia acumulada es > 0,
         contando desde el día juliano 1.

    Si nunca supera 0, devuelve 999 (indicador de 'desconocido').
    """
    idx = np.where(curva > 0)[0]
    if len(idx) == 0:
        return 999
    return int(idx[0] + 1)  # índice 0-based → JD (1-based)

# ===============================================================
# FEATURES METEOROLÓGICAS (robusto)
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

# ===============================================================
# DTW + K-MEDOIDS (SIN DEPENDENCIAS EXTERNAS)
# ===> Importante: usa sólo el tramo JD 30–121 para comparar curvas
# ===============================================================
def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distancia DTW entre dos curvas de emergencia acumulada,
    usando únicamente el segmento JD 30–121 (inclusive).
    Esto asegura que la clasificación de patrones se base sólo
    en la parte temprana de la curva.
    """
    # Recortar a ventana 30–121 (índices 29..120)
    a_seg = a[29:121]
    b_seg = b[29:121]

    n, m = len(a_seg), len(b_seg)
    D = np.full((n+1, m+1), np.inf, dtype=float)
    D[0,0] = 0.0
    for i in range(1, n+1):
        ai = a_seg[i-1]
        for j in range(1, m+1):
            cost = (ai - b_seg[j-1])**2
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(np.sqrt(D[n,m]))

def k_medoids_dtw(curves: list, K: int, max_iter: int = 50, seed: int = 42):
    """
    Agrupa curvas en K clusters usando k-medoids con distancia DTW
    basada sólo en JD 30–121. Devuelve:
    - índices de los medoids (prototipos),
    - asignación de miembros a clusters,
    - matriz de distancias DTW.
    """
    rng = np.random.default_rng(seed)
    N = len(curves)
    if K > N:
        K = N
    idx = rng.choice(N, size=K, replace=False)
    medoid_idx = list(idx)

    # Matriz de distancias (simétrica)
    D = np.zeros((N,N), float)
    for i in range(N):
        for j in range(i+1, N):
            d = dtw_distance(curves[i], curves[j])
            D[i,j] = D[j,i] = d

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

# ===============================================================
# BUNDLE HELPERS — warp + mezcla convexa
# ===============================================================
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

def mezcla_convexa(protos: np.ndarray, proba: np.ndarray, k_hat: int, shift: float, scale: float) -> np.ndarray:
    """
    Construye la curva predicha como mezcla convexa de todos los prototipos,
    aplicando el warp (shift/scale) sólo al patrón más probable.
    """
    K = protos.shape[0]
    mix = np.zeros(JD_MAX, float)
    for k in range(K):
        yk = warp_curve(protos[k], shift if k==k_hat else 0.0,
                        scale if k==k_hat else 1.0)
        mix += float(proba[k]) * yk
    return np.maximum.accumulate(np.clip(mix, 0, 1))

# ===============================================================
# APP — TABS
# ===============================================================
tab1, tab2, tab3 = st.tabs([
    "🧪 Entrenar prototipos + clasificador",
    "🔮 Identificar patrones y predecir",
    "📈 Comparar Real vs Predicción"
])

# ---------------------------------------------------------------
# TAB 1 — ENTRENAMIENTO
# ---------------------------------------------------------------
with tab1:
    st.subheader("🧪 Entrenamiento (k-medoids DTW + mezcla de prototipos)")
    st.markdown("""
    Subí:
    - **Meteorología multianual** (una hoja por año)
    - **Curvas históricas de emergencia** (1 archivo XLSX por año)

    El modelo:
    1. Aprende K prototipos de curva (k-medoids con DTW entre JD 30–121)
    2. Asigna cada año a un patrón (cluster)
    3. Construye un clasificador meteo + inicio_emergencia → patrón
    4. Ajusta warps (shift/scale) por cluster
    """)

    meteo_book = st.file_uploader("📘 Meteorología multianual (una hoja por año)", type=["xlsx","xls"])
    curvas_files = st.file_uploader("📈 Curvas históricas (XLSX por año, acumulada o semanal)",
                                    type=["xlsx","xls"], accept_multiple_files=True)

    K = st.slider("Número de prototipos/patrones (K)", 2, 10, 10, 1)
    seed = st.number_input("Semilla", 0, 99999, 42)
    btn_train = st.button("🚀 Entrenar")

    if btn_train:
        if not (meteo_book and curvas_files):
            st.error("Cargá ambos conjuntos: meteo y curvas.")
            st.stop()

        # 1) Leer meteo por año
        sheets = pd.read_excel(meteo_book, sheet_name=None)
        meteo_dict = {}
        for name, df in sheets.items():
            df = standardize_cols(df)
            df = ensure_jd_1_to_274(df)
            try:
                year = int(re.findall(r"\d{4}", str(name))[0])
            except:
                year = None
            if year and all(c in df.columns for c in ["tmin","tmax","prec"]):
                meteo_dict[year] = df[["jd","tmin","tmax","prec"]].copy()

        if not meteo_dict:
            st.error("⛔ No se detectó meteorología válida por año.")
            st.stop()
        st.success(f"✅ Meteorología válida: {len(meteo_dict)} años")

        # 2) Leer curvas por año
        years_list, curves_list = [], []
        curves_dict = {}   # para acceso por año
        for f in curvas_files:
            y4 = re.findall(r"(\d{4})", f.name)
            year = int(y4[0]) if y4 else None
            if year is None:
                continue
            curva = np.maximum.accumulate(curva_desde_xlsx_anual(f))
            if curva.max() > 0:
                curva = curva[:JD_MAX]
                curves_list.append(curva)
                years_list.append(year)
                curves_dict[year] = curva
        if not years_list:
            st.error("⛔ No se detectaron curvas válidas.")
            st.stop()

        # 3) Intersección meteo–curvas
        common_years = sorted([y for y in years_list if y in meteo_dict])
        if len(common_years) < 3:
            st.error("⛔ Muy pocos años en común (se recomienda ≥ 5).")
            st.stop()
        curves = [curves_dict[y] for y in common_years]

        # 4) Detectar inicio de emergencia por año (desde curva real)
        inicio_year = {}
        for y in common_years:
            curva = curves_dict[y]
            inicio_year[y] = detectar_inicio_emergencia(curva)
        st.write("📍 Día de inicio de emergencia por año:", inicio_year)

        # 5) k-medoids (DTW sobre JD 30–121)
        st.info("🧮 Calculando k-medoids (DTW, JD 30–121)...")
        medoid_idx, clusters, D = k_medoids_dtw(curves, K=K, max_iter=50, seed=seed)
        protos = [curves[i] for i in medoid_idx]

        # 6) Features desde meteo + inicio_emergencia + etiqueta de cluster por año
        feat_rows = []
        FEAT_EXT = FEATURE_ORDER + ["inicio_emergencia"]  # orden de features del clasificador
        for y in common_years:
            _, f_meteo = build_features_meteo(meteo_dict[y])
            # unir meteo + inicio_emergencia
            f_all = f_meteo.copy()
            f_all["inicio_emergencia"] = inicio_year[y]
            feat_rows.append([f_all[k] for k in FEAT_EXT])

        assign = np.argmin(D[:, np.array(medoid_idx)], axis=1)  # índice cluster 0..K-1

        # Años por cluster para interpretación
        cluster_years = {k: [] for k in range(K)}
        for i, y in enumerate(common_years):
            cluster_years[int(assign[i])].append(int(y))

        X = np.array(feat_rows, float)
        y_lbl = assign.astype(int)
        xsc = StandardScaler().fit(X)
        Xs  = xsc.transform(X)

        # 7) Clasificador de patrón (meteo + inicio_emergencia → cluster)
        clf = GradientBoostingClassifier(random_state=seed)
        clf.fit(Xs, y_lbl)
        feat_names_ext = FEAT_EXT[:]   # guardar orden de columnas del clasificador

        # 8) Warps (shift/scale) por cluster
        regs_shift, regs_scale = {}, {}
        for k in range(K):
            idx = np.where(y_lbl == k)[0]
            if len(idx) == 0:
                continue
            proto = protos[k]
            shifts, scales, Xk = [], [], []
            for ii in idx:
                curv = curves[ii]
                best = (0.0, 1.0, 1e9)
                # Búsqueda gruesa de shift y scale
                for sh in range(-20, 21, 5):       # ±20 días
                    for sc in [0.9, 0.95, 1.0, 1.05, 1.1]:
                        cand = warp_curve(proto, sh, sc)
                        rmse = float(np.sqrt(np.mean((cand - curv)**2)))
                        if rmse < best[2]:
                            best = (float(sh), float(sc), rmse)
                shifts.append(best[0])
                scales.append(best[1])
                Xk.append(Xs[ii])
            Xk = np.vstack(Xk)
            regs_shift[k] = GradientBoostingRegressor(random_state=seed).fit(Xk, np.array(shifts))
            regs_scale[k] = GradientBoostingRegressor(random_state=seed).fit(Xk, np.array(scales))

        # 9) Guardar bundle
        bundle = {
            "xsc": xsc,
            "feat_names": feat_names_ext,  # incluye inicio_emergencia
            "clf": clf,
            "protos": np.vstack(protos),   # K x 274
            "regs_shift": regs_shift,
            "regs_scale": regs_scale,
            "cluster_years": cluster_years
        }
        st.success(f"✅ Entrenamiento OK. K={K} prototipos.")
        st.session_state["mix_bundle"] = bundle

        buf = io.BytesIO()
        joblib.dump(bundle, buf)
        st.download_button(
            "💾 Descargar modelo (joblib)",
            data=buf.getvalue(),
            file_name=f"predweem_v51_mixture_dtw_K{K}.joblib",
            mime="application/octet-stream"
        )

        # 10) Vista rápida de prototipos
        dias = np.arange(1, JD_MAX+1)
        dfp = []
        for k, proto in enumerate(protos):
            years_txt = ", ".join(map(str, cluster_years.get(k, []))) if cluster_years.get(k) else "—"
            dfp.append(pd.DataFrame({
                "Día": dias,
                "Valor": proto,
                "Serie": f"Proto {k} · años: {years_txt}"
            }))
        dfp = pd.concat(dfp)
        chart = alt.Chart(dfp).mark_line().encode(
            x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE))),
            y=alt.Y("Valor:Q", title="Emergencia acumulada (0–1)", scale=alt.Scale(domain=[0,1])),
            color="Serie:N"
        ).properties(
            height=420,
            title="Prototipos (medoids DTW, clasificación basada en JD 30–121)"
        )
        st.altair_chart(chart, use_container_width=True)

# ---------------------------------------------------------------
# TAB 2 — PREDICCIÓN
# ---------------------------------------------------------------
with tab2:
    st.subheader("🔮 Identificación de patrones y predicción a partir de meteorología nueva")

    st.markdown("""
    Cargá:
    - Un **modelo entrenado** (.joblib)  
    - La **meteorología diaria** del año que querés analizar  

    Opcionalmente podés ingresar el **día de inicio de emergencia medido a campo**.
    Si lo dejás en 0, el modelo usará 999 como valor 'desconocido'.
    """)

    modelo_file = st.file_uploader("📦 Modelo (predweem_v51_mixture_dtw_*.joblib)", type=["joblib"])
    meteo_file  = st.file_uploader("📘 Meteorología nueva (XLSX)", type=["xlsx","xls"])
    inicio_manual = st.number_input(
        "📍 Día de inicio de emergencia medido a campo (0 = desconocido)",
        min_value=0, max_value=JD_MAX, value=0, step=1
    )

    btn_pred = st.button("🚀 Analizar y predecir")

    if btn_pred:
        if not (modelo_file and meteo_file):
            st.error("Cargá el modelo y la meteo.")
            st.stop()

        # --- Cargar modelo ---
        bundle = joblib.load(modelo_file)
        xsc = bundle["xsc"]
        feat_names = bundle["feat_names"]   # incluye inicio_emergencia
        clf = bundle["clf"]
        protos = bundle["protos"]
        regs_shift = bundle["regs_shift"]
        regs_scale = bundle["regs_scale"]
        cluster_years = bundle.get("cluster_years", {})
        K = protos.shape[0]

        # --- Features desde meteo nueva ---
        dfm = pd.read_excel(meteo_file)
        dfm, f_new = build_features_meteo(dfm)

        # Construimos diccionario de features extendido (meteo + inicio_emergencia)
        features_all = f_new.copy()
        if inicio_manual > 0:
            features_all["inicio_emergencia"] = float(inicio_manual)
        else:
            features_all["inicio_emergencia"] = 999.0  # valor 'desconocido' / neutro

        # Vector X respetando el orden feat_names
        xrow = [features_all[k] for k in feat_names]
        X = np.array([xrow], float)
        Xs = xsc.transform(X)

        # --- Probabilidades de cada patrón ---
        proba  = clf.predict_proba(Xs)[0]  # shape (K,)
        top_idx = np.argsort(proba)[::-1]
        k_hat = int(top_idx[0])

        # --- Warp predicho para el patrón más probable ---
        if k_hat in regs_shift:
            shift = float(regs_shift[k_hat].predict(Xs)[0])
        else:
            shift = 0.0
        if k_hat in regs_scale:
            scale = float(regs_scale[k_hat].predict(Xs)[0])
        else:
            scale = 1.0
        scale = float(np.clip(scale, 0.9, 1.1))

        # --- Curva predicha (mezcla convexa) y patrón más probable ---
        mix = mezcla_convexa(protos, proba, k_hat, shift, scale)
        proto_hat = protos[k_hat]

        # --- Emergencia relativa semanal (sobre la predicción) ---
        rel7 = emerg_rel_7d_from_acum(mix)

        # --- Fracción de la curva entre JD 1–120 (predicha) ---
        frac120_pred = frac_curva_1_120(mix)

        st.markdown(f"**Fracción acumulada predicha al JD 120:** `{frac120_pred:.3f}`")

        # --- Gráfico: Predicción + Patrón más probable + Relativa 7d ---
        dias = np.arange(1, JD_MAX + 1)
        df_plot = pd.DataFrame({
            "Día": dias,
            "Predicción": mix,
            "Patrón más probable": proto_hat,
            "Emergencia_relativa_7d": rel7
        })

        base = alt.Chart(df_plot).encode(
            x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE)))
        )

        curva_lineas = base.transform_fold(
            ["Predicción", "Patrón más probable"], as_=["Serie", "Valor"]
        ).mark_line(strokeWidth=2).encode(
            y=alt.Y("Valor:Q", title="Emergencia acumulada (0–1)",
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Serie:N", scale=alt.Scale(scheme="tableau10")),
            tooltip=["Serie:N", alt.Tooltip("Valor:Q", format=".3f"), "Día:Q"]
        )

        max_rel = float(np.nanmax(rel7)) if np.isfinite(np.nanmax(rel7)) else 1.0
        barra_rel = base.mark_area(opacity=0.35).encode(
            y=alt.Y("Emergencia_relativa_7d:Q",
                    axis=alt.Axis(title="Emergencia relativa semanal", titleColor="#666"),
                    scale=alt.Scale(domain=[0, max_rel * 1.1]))
        )

        chart = alt.layer(curva_lineas, barra_rel).resolve_scale(y='independent').properties(
            height=420,
            title=(
                f"Predicción (C{k_hat} • conf {proba[k_hat]:.2f} • "
                f"shift {shift:+.1f}d • scale {scale:.3f} • inicio_emergencia={features_all['inicio_emergencia']:.0f})"
            )
        )
        st.altair_chart(chart, use_container_width=True)

        # --- Tabla de probabilidades por patrón (años del cluster) ---
        rows = []
        for k in range(K):
            years_txt = ", ".join(map(str, cluster_years.get(k, []))) if cluster_years.get(k) else "—"
            rows.append((f"C{k}", float(proba[k]), years_txt))
        df_proba = pd.DataFrame(rows, columns=["Cluster","Probabilidad","Años (cluster)"]) \
                    .sort_values("Probabilidad", ascending=False).reset_index(drop=True)
        st.markdown("### 🔢 Probabilidades por patrón")
        st.dataframe(df_proba.style.format({"Probabilidad": "{:.3f}"}), use_container_width=True)

        # --- Descarga predicción (incluye patrón más probable, relativa 7d y fracción 1–120) ---
        out = pd.DataFrame({
            "Día": dias,
            "Emergencia_predicha": mix,
            "Patrón_mas_probable": proto_hat,
            "Emergencia_relativa_7d": rel7
        })
        out["Frac_1_120_pred"] = frac120_pred  # mismo valor en todas las filas, para referencia
        out["inicio_emergencia_usado"] = features_all["inicio_emergencia"]

        st.download_button(
            "⬇️ Descargar curvas (CSV)",
            out.to_csv(index=False).encode("utf-8"),
            file_name="curva_predicha_vs_patron.csv",
            mime="text/csv"
        )

# ---------------------------------------------------------------
# TAB 3 — COMPARAR CURVA REAL VS PREDICHA (RMSE/MAE)
# ---------------------------------------------------------------
with tab3:
    st.subheader("📈 Comparar curva real vs curva predicha (RMSE/MAE)")

    st.markdown("""
    Cargá:
    - Un **modelo entrenado** (.joblib)  
    - La **meteorología del año** que querés evaluar  
    - La **curva real de emergencia** de ese mismo año (XLSX, diaria o semanal)

    El sistema:
    1. Calcula **inicio_emergencia real** a partir de la curva
    2. Usa meteo + inicio_emergencia real → patrón
    3. Construye la curva predicha
    4. Calcula **RMSE/MAE**
    5. Compara fracción al JD 120 entre real y predicha
    """)

    modelo_cmp = st.file_uploader("📦 Modelo", type=["joblib"], key="cmp_model")
    meteo_cmp  = st.file_uploader("📘 Meteorología del año", type=["xlsx","xls"], key="cmp_meteo")
    curva_real_file = st.file_uploader("📈 Curva real (XLSX)", type=["xlsx","xls"], key="cmp_curva")

    btn_cmp = st.button("🚀 Comparar")

    if btn_cmp:
        if not (modelo_cmp and meteo_cmp and curva_real_file):
            st.error("Falta cargar modelo, meteorología o curva real.")
            st.stop()

        # --- Cargar modelo ---
        bundle = joblib.load(modelo_cmp)
        xsc = bundle["xsc"]
        feat_names = bundle["feat_names"]   # incluye inicio_emergencia
        clf = bundle["clf"]
        protos = bundle["protos"]
        regs_shift = bundle["regs_shift"]
        regs_scale = bundle["regs_scale"]
        cluster_years = bundle.get("cluster_years", {})
        K = protos.shape[0]

        # --- Cargar curva real ---
        curva_real = np.maximum.accumulate(curva_desde_xlsx_anual(curva_real_file))[:JD_MAX]
        rel7_real = emerg_rel_7d_from_acum(curva_real)
        frac120_real = frac_curva_1_120(curva_real)
        inicio_real = detectar_inicio_emergencia(curva_real)

        # --- Cargar y procesar meteo ---
        dfm = pd.read_excel(meteo_cmp)
        dfm, f_new = build_features_meteo(dfm)

        # Construimos features (meteo + inicio_emergencia real)
        features_all = f_new.copy()
        features_all["inicio_emergencia"] = float(inicio_real)

        xrow = [features_all[k] for k in feat_names]
        X = np.array([xrow], float)
        Xs = xsc.transform(X)

        # --- Clasificación ---
        proba = clf.predict_proba(Xs)[0]
        k_hat = int(np.argmax(proba))

        # --- Warps ---
        if k_hat in regs_shift:
            shift = float(regs_shift[k_hat].predict(Xs)[0])
        else:
            shift = 0.0
        if k_hat in regs_scale:
            scale = float(regs_scale[k_hat].predict(Xs)[0])
        else:
            scale = 1.0
        scale = float(np.clip(scale, 0.9, 1.1))

        # --- Curva predicha ---
        curva_pred = mezcla_convexa(protos, proba, k_hat, shift, scale)
        rel7_pred = emerg_rel_7d_from_acum(curva_pred)
        frac120_pred = frac_curva_1_120(curva_pred)

        # --- RMSE & MAE ---
        rmse = float(np.sqrt(np.mean((curva_real - curva_pred)**2)))
        mae  = float(np.mean(np.abs(curva_real - curva_pred)))

        st.success(f"✅ RMSE = {rmse:.4f} — MAE = {mae:.4f}")
        st.markdown(
            f"- **Fracción real al JD 120:** `{frac120_real:.3f}`\n\n"
            f"- **Fracción predicha al JD 120:** `{frac120_pred:.3f}`\n\n"
            f"- **inicio_emergencia real usado:** JD `{inicio_real}`"
        )

        # --- Gráfico comparativo ---
        dias = np.arange(1, JD_MAX+1)
        df_cmp = pd.DataFrame({
            "Día": dias,
            "Real": curva_real,
            "Predicción": curva_pred,
            "Relativa real 7d": rel7_real,
            "Relativa pred 7d": rel7_pred
        })

        base = alt.Chart(df_cmp).encode(
            x=alt.X("Día:Q", scale=alt.Scale(domain=[1, JD_MAX]))
        )

        lineas = base.transform_fold(
            ["Real", "Predicción"], as_=["Serie", "Valor"]
        ).mark_line(strokeWidth=2).encode(
            y=alt.Y("Valor:Q", title="Emergencia acumulada (0–1)",
                    scale=alt.Scale(domain=[0,1])),
            color="Serie:N"
        )

        max_rel = max(float(rel7_real.max()), float(rel7_pred.max()))
        areas = base.transform_fold(
            ["Relativa real 7d", "Relativa pred 7d"],
            as_=["Serie", "Valor"]
        ).mark_area(opacity=0.35).encode(
            y=alt.Y("Valor:Q",
                    axis=alt.Axis(title="Emergencia relativa semanal"),
                    scale=alt.Scale(domain=[0, max_rel*1.1])),
            color="Serie:N"
        )

        chart = alt.layer(lineas, areas).resolve_scale(y='independent').properties(
            height=420,
            title=(
                f"Comparación Real vs Predicción (C{k_hat} • conf {proba[k_hat]:.2f} • "
                f"shift {shift:+.1f}d • scale {scale:.3f})"
            )
        )
        st.altair_chart(chart, use_container_width=True)

        # --- Exportar ---
        out = df_cmp.copy()
        out["Error_abs"] = np.abs(curva_real - curva_pred)
        out["Frac_1_120_real"] = frac120_real
        out["Frac_1_120_pred"] = frac120_pred
        out["inicio_emergencia_real"] = inicio_real
        out["RMSE_global"] = rmse
        out["MAE_global"] = mae

        st.download_button(
            "⬇️ Descargar comparación (CSV)",
            out.to_csv(index=False).encode("utf-8"),
            file_name="comparacion_real_vs_pred.csv",
            mime="text/csv"
        )






