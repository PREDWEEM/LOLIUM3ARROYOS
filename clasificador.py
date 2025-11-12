# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Curvas de Emergencia (hasta 1 de octubre · JD 274)
# ===============================================================
# - Genera curvas históricas desde GitHub RAW
#   · Detecta frecuencia: diaria → semanal (auto) si paso=1 día
#   · Normaliza y recorta a JD 274 (1/oct)
# - Entrena MLP multisalida (Tmin, Tmax, Prec → curva 0..1 de 274 días)
# - Predice curva nueva y muestra:
#   · Banda histórica min–max y promedio
#   · Eje Y secundario con emergencia relativa semanal (media móvil 7d)
# - Genera patrones sintéticos (nuevos escenarios meteorológicos no vistos)
# - Descargas: curvas CSV, modelo .joblib, comparación CSV, patrones sintéticos CSV
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests, re, io, joblib
from io import BytesIO
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# ⚙️ CONFIG GENERAL
# =========================
st.set_page_config(page_title="PREDWEEM — Curvas hasta 1/oct (JD 274)", layout="wide")
st.title("🌾 PREDWEEM — Generador, Entrenador y Predictor (1-ene → 1-oct, JD 274)")

JD_MAX = 274  # 1 de octubre (no bisiesto)
XRANGE = (1, JD_MAX)

# =========================
# 🔧 UTILIDADES GLOBALES (robustas)
# =========================
def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres y tipos para meteorología (robusta a headers numéricos/NaN)."""
    # Headers como strings "limpios"
    cols = []
    for c in df.columns:
        try:
            c = str(c).strip().lower()
        except Exception:
            c = f"col_{len(cols)}"
        cols.append(c)
    df.columns = cols

    ren = {
        # Tmin
        "temperatura minima": "tmin", "t_min": "tmin", "t min": "tmin",
        "mínima": "tmin", "min": "tmin", "tminima": "tmin", "tmin": "tmin",
        # Tmax
        "temperatura maxima": "tmax", "t_max": "tmax", "t max": "tmax",
        "máxima": "tmax", "max": "tmax", "tmaxima": "tmax", "tmax": "tmax",
        # Prec
        "precipitacion": "prec", "precip": "prec", "pp": "prec",
        "rain": "prec", "lluvia": "prec", "prec": "prec",
        # JD / fecha
        "dia juliano": "jd", "día juliano": "jd", "julian_days": "jd",
        "julian day": "jd", "dia": "jd", "día": "jd",
        "fecha": "fecha"
    }
    for k, v in ren.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)

    for c in ["tmin", "tmax", "prec", "jd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def slice_jan_to_oct1(df: pd.DataFrame) -> pd.DataFrame:
    """Recorta 1-ene → 1-oct (inclusive) y garantiza jd incremental si falta."""
    if "fecha" in df.columns and df["fecha"].notna().any():
        y_mode = df["fecha"].dt.year.mode()
        if len(y_mode) > 0 and not pd.isna(y_mode.iloc[0]):
            y = int(y_mode.iloc[0])
            m = (df["fecha"] >= f"{y}-01-01") & (df["fecha"] <= f"{y}-10-01")
            df = df.loc[m].copy().sort_values("fecha")
    # Construir JD si no hay
    if "jd" not in df.columns or df["jd"].isna().all():
        if "fecha" in df.columns and df["fecha"].notna().any():
            start = df["fecha"].dropna().iloc[0]
            df["jd"] = (df["fecha"] - start).dt.days + 1
        else:
            df["jd"] = np.arange(1, len(df) + 1)

    # Reindexar a 1..JD_MAX con interpolación
    df = (df.set_index("jd")
            .sort_index()
            .reindex(range(1, JD_MAX + 1)))
    for c in ["tmin", "tmax", "prec"]:
        if c in df.columns:
            df[c] = df[c].interpolate().fillna(0.0).astype("float64")
    df = df.reset_index().rename(columns={"index": "jd"})
    return df


def to_weekly_by_jd(jd: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convierte una serie diaria a semanal por agrupación de semanas naturales desde JD=1.
    Semana = floor((jd-1)/7). Devuelve jd_sem (1,8,15,...) y valores promedio por semana.
    """
    s = pd.DataFrame({"jd": jd, "v": values}).dropna()
    s["week_id"] = ((s["jd"].astype(int) - 1) // 7).astype(int)
    g = s.groupby("week_id", as_index=False)["v"].mean()
    jd_sem = 1 + g["week_id"].to_numpy() * 7
    vals_sem = g["v"].to_numpy()
    return jd_sem, vals_sem


def build_xy(meteo_dict: dict, curvas_dict: dict):
    """Construye matrices X (meteo) y Y (curva) alineadas por año."""
    common = sorted(set(meteo_dict.keys()) & set(curvas_dict.keys()))
    X, Y, years = [], [], []
    for y in common:
        dfm = meteo_dict[y]
        x = np.concatenate([dfm["tmin"], dfm["tmax"], dfm["prec"]])
        X.append(x)
        Y.append(curvas_dict[y])
        years.append(y)
    return np.array(X, dtype="float64"), np.array(Y, dtype="float64"), np.array(years)


def emerg_rel_semanal_desde_acum(y_acum: np.ndarray) -> np.ndarray:
    """Emergencia relativa semanal (media móvil 7d) desde la acumulada."""
    inc_diario = np.diff(np.insert(y_acum, 0, 0.0))
    rel = np.convolve(inc_diario, np.ones(7) / 7, mode="same")
    return rel

# =========================
# 🧭 PESTAÑAS
# =========================
tabs = st.tabs([
    "📈 Generar curvas desde GitHub",
    "🤖 Entrenar modelo",
    "🔮 Predecir nuevo año",
    "🧬 Generar nuevos patrones"
])

# ===============================================================
# 📈 TAB 1 — GENERADOR DE CURVAS AUTOMÁTICAS (GitHub RAW)
# ===============================================================
with tabs[0]:
    st.subheader("📦 Generar curvas automáticamente desde GitHub")
    st.markdown("Convierte **automáticamente** series diarias → **semanales** si el paso es de 1 día.")

    base_url = st.text_input(
        "URL base RAW del repositorio",
        value="https://raw.githubusercontent.com/PREDWEEM/LOLium3arroyos/main"
    )
    btn_gen = st.button("🚀 Generar curvas")

    def listar_archivos_github(base_url: str):
        """Candidatos 2008..2030 (GitHub RAW no lista índice)."""
        return [f"{base_url}/{y}.xlsx" for y in range(2008, 2031)]

    @st.cache_data(show_spinner=False)
    def descargar_y_procesar(url: str):
        """Lee [día, valor] desde Excel, detecta frecuencia por JD y normaliza a curva 0..1 (JD 1..274)."""
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                return None, None
            df = pd.read_excel(BytesIO(r.content), header=None)

            # Expectativa mínima: 2 columnas (día, valor)
            col0 = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            col1 = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            dias = col0.dropna().astype(int).to_numpy()
            vals = col1.fillna(0).to_numpy()
            if len(dias) == 0 or len(vals) == 0:
                return None, None

            # Paso típico por diferencias únicas
            paso = int(np.median(np.diff(np.unique(np.sort(dias))))) if len(dias) > 1 else 7

            if paso == 1:
                jd_sem, vals_sem = to_weekly_by_jd(dias, vals)
                dias, vals = jd_sem, vals_sem

            # Construir serie diaria completa y acumular
            daily = np.zeros(365, dtype="float64")
            for d, v in zip(dias, vals):
                if 1 <= int(d) <= 365:
                    daily[int(d) - 1] = float(v)

            acum = np.cumsum(daily)
            if acum[-1] == 0:
                return None, None

            curva = (acum / acum[-1])[:JD_MAX]
            anio_m = re.findall(r"(\d{4})", url)
            if not anio_m:
                return None, None
            anio = int(anio_m[0])
            return anio, curva
        except Exception:
            return None, None

    if btn_gen:
        st.info("Descargando curvas desde GitHub...")
        urls = listar_archivos_github(base_url)
        curvas = {}
        for url in urls:
            anio, curva = descargar_y_procesar(url)
            if anio and curva is not None:
                curvas[anio] = curva

        if not curvas:
            st.error("No se pudieron generar curvas. Revisá la URL o los archivos en el repo.")
        else:
            st.success(f"✅ Se generaron {len(curvas)} curvas (JD 1–{JD_MAX}).")
            st.session_state["curvas_github"] = curvas

            dias = np.arange(1, JD_MAX + 1)
            data = []
            for y, c in sorted(curvas.items()):
                for d, v in zip(dias, c):
                    data.append({"Día": d, "Año": y, "Emergencia acumulada": v})
            df = pd.DataFrame(data)
            curva_media = df.groupby("Día")["Emergencia acumulada"].mean().reset_index()
            curva_media["Año"] = "Promedio"
            df_total = pd.concat([df, curva_media], ignore_index=True)

            chart = alt.Chart(df_total).mark_line().encode(
                x=alt.X("Día:Q", title=f"Día juliano (1–{JD_MAX})", scale=alt.Scale(domain=list(XRANGE))),
                y=alt.Y("Emergencia acumulada:Q", title="Emergencia acumulada (0–1)"),
                color="Año:N",
                size=alt.condition(alt.datum.Año == "Promedio", alt.value(3), alt.value(1))
            ).properties(height=440)
            st.altair_chart(chart, use_container_width=True)

            df_wide = df.pivot(index="Día", columns="Año", values="Emergencia acumulada").sort_index(axis=1).fillna(0)
            st.download_button(
                "⬇️ Descargar curvas históricas (CSV)",
                df_wide.to_csv().encode("utf-8"),
                f"curvas_emergencia_github_{JD_MAX}.csv",
                mime="text/csv"
            )

# ===============================================================
# 🤖 TAB 2 — ENTRENAMIENTO DEL MODELO
# ===============================================================
with tabs[1]:
    st.subheader("🤖 Entrenar modelo (1-ene → 1-oct, JD 274)")

    meteo_file = st.file_uploader("📂 Cargar archivo meteorológico (una hoja por año)", type=["xlsx", "xls"])
    seed = st.number_input("Seed aleatoria", 0, 99999, 42)
    neurons = st.slider("Neuronas por capa", 32, 256, 128, 16)
    max_iter = st.slider("Iteraciones", 300, 5000, 1500, 100)
    btn_fit = st.button("🚀 Entrenar modelo")

    curvas_dict = st.session_state.get("curvas_github", {})
    meteo_dict = {}

    # --- construir meteo_dict de forma robusta ---
    if meteo_file:
        sheets = pd.read_excel(meteo_file, sheet_name=None)
        tmp = {}
        for name, dfm in sheets.items():
            dfm = standardize_cols(dfm)
            dfm = slice_jan_to_oct1(dfm)
            # año por hoja o contenido
            year = None
            m = re.findall(r"\d{4}", str(name))
            if m:
                year = int(m[0])
            elif "fecha" in dfm.columns and dfm["fecha"].notna().any():
                year = int(dfm["fecha"].dt.year.mode().iloc[0])
            # asegurar columnas y rango
            has_cols = all(c in dfm.columns for c in ["tmin", "tmax", "prec", "jd"])
            if year and has_cols:
                use = (dfm[["jd", "tmin", "tmax", "prec"]]
                       .set_index("jd")
                       .reindex(range(1, JD_MAX + 1))
                       .interpolate()
                       .fillna(0.0)
                       .reset_index())
                for c in ["tmin", "tmax", "prec"]:
                    use[c] = use[c].astype("float64")
                tmp[year] = use
        meteo_dict = tmp
        st.success(f"✅ Meteorología cargada ({len(meteo_dict)} años).")

    # --- Entrenamiento LOYO ---
    if btn_fit and meteo_dict and curvas_dict:
        # Filtrar a años comunes con curvas
        anos_comunes = sorted(set(meteo_dict.keys()) & set(curvas_dict.keys()))
        if not anos_comunes:
            st.error("No hay intersección entre años de meteorología y curvas históricas.")
            st.stop()

        # X, Y como float64
        X, Y, years = build_xy({k: meteo_dict[k] for k in anos_comunes},
                               {k: curvas_dict[k] for k in anos_comunes})
        # Garantizar curva termina en 1 y es monótona
        for i in range(Y.shape[0]):
            if Y[i, -1] != 0:
                Y[i] = Y[i] / Y[i, -1]
            Y[i] = np.clip(np.maximum.accumulate(Y[i]), 0, 1)

        kf = KFold(n_splits=len(years))
        metrics = []
        xsc, ysc = StandardScaler(), StandardScaler()
        for train, test in kf.split(X):
            Xtr, Xte = X[train], X[test]
            Ytr, Yte = Y[train], Y[test]
            Xtr_s, Xte_s = xsc.fit_transform(Xtr), xsc.transform(Xte)
            Ytr_s = ysc.fit_transform(Ytr)

            mlp = MLPRegressor(hidden_layer_sizes=(neurons,), max_iter=max_iter, random_state=seed)
            mlp.fit(Xtr_s, Ytr_s)
            Yhat = ysc.inverse_transform(mlp.predict(Xte_s))[0]
            Yhat = np.clip(np.maximum.accumulate(Yhat), 0, 1)

            rmse = float(np.sqrt(mean_squared_error(Yte[0], Yhat)))
            mae = float(mean_absolute_error(Yte[0], Yhat))
            metrics.append((int(years[test][0]), rmse, mae))

        dfm = pd.DataFrame(metrics, columns=["Año", "RMSE", "MAE"]).sort_values("Año")
        st.dataframe(dfm, use_container_width=True)

        # Fit final
        xsc.fit(X); ysc.fit(Y)
        mlp_final = MLPRegressor(hidden_layer_sizes=(neurons,), max_iter=max_iter, random_state=seed)
        mlp_final.fit(xsc.transform(X), ysc.transform(Y))
        st.session_state["bundle"] = {"xsc": xsc, "ysc": ysc, "mlp": mlp_final}
        st.success("✅ Modelo entrenado y guardado en sesión.")

        buf = io.BytesIO()
        joblib.dump(st.session_state["bundle"], buf)
        st.download_button("⬇️ Descargar modelo entrenado (.joblib)",
                           data=buf.getvalue(),
                           file_name=f"modelo_curva_emergencia_{JD_MAX}.joblib",
                           mime="application/octet-stream")

# ===============================================================
# 🔮 TAB 3 — PREDICCIÓN NUEVO AÑO (con histórico + eje secundario)
# ===============================================================
with tabs[2]:
    st.subheader("🔮 Predicción (1-ene → 1-oct, JD 274) con histórico + eje secundario")

    curvas_hist = st.session_state.get("curvas_github", {})
    show_hist_ref = st.checkbox("Mostrar banda histórica (min–max) y promedio", value=True)

    meteo_pred = st.file_uploader("📂 Meteorología nueva (xlsx)", type=["xlsx", "xls"], key="pred")
    modelo_up  = st.file_uploader("📦 Modelo entrenado (.joblib)", type=["joblib"])

    if st.button("Predecir curva"):
        if not meteo_pred or not modelo_up:
            st.error("Faltan archivos.")
        else:
            try:
                bundle = joblib.load(modelo_up)
                xsc, ysc, mlp = bundle["xsc"], bundle["ysc"], bundle["mlp"]

                df = pd.read_excel(meteo_pred)
                df = standardize_cols(df)
                df = slice_jan_to_oct1(df)

                if "jd" not in df.columns:
                    if "fecha" in df.columns and df["fecha"].notna().any():
                        df["jd"] = df["fecha"].dt.dayofyear - df["fecha"].dt.dayofyear.iloc[0] + 1
                    else:
                        df["jd"] = np.arange(1, len(df) + 1)

                df = df.set_index("jd").reindex(range(1, JD_MAX + 1)).interpolate().fillna(0).reset_index()

                xnew = np.concatenate([df["tmin"], df["tmax"], df["prec"]]).reshape(1, -1)
                yhat = ysc.inverse_transform(mlp.predict(xsc.transform(xnew)))[0]
                yhat = np.maximum.accumulate(yhat)
                yhat = yhat / (yhat[-1] if yhat[-1] != 0 else 1.0)
                yhat = np.clip(yhat, 0, 1)

                dias = np.arange(1, JD_MAX + 1)
                df_pred = pd.DataFrame({"Día": dias, "Emergencia predicha": yhat})

                # Emergencia relativa semanal (eje secundario)
                rel = emerg_rel_semanal_desde_acum(yhat)
                df_rel = pd.DataFrame({"Día": dias, "Emergencia relativa semanal": rel})

                # Capas históricas
                layers = []
                if show_hist_ref and isinstance(curvas_hist, dict) and len(curvas_hist) > 0:
                    H = np.vstack([v[:JD_MAX] for _, v in sorted(curvas_hist.items()) if len(v) >= JD_MAX])
                    y_min, y_max = np.nanmin(H, axis=0), np.nanmax(H, axis=0)
                    y_mean = np.nanmean(H, axis=0)
                    df_band = pd.DataFrame({"Día": dias, "Min": y_min, "Max": y_max})
                    df_mean = pd.DataFrame({"Día": dias, "Promedio histórico": y_mean})

                    area_hist = alt.Chart(df_band).mark_area(opacity=0.15).encode(
                        x=alt.X("Día:Q", title=f"Día juliano (1–{JD_MAX})", scale=alt.Scale(domain=list(XRANGE))),
                        y=alt.Y("Min:Q", title="Emergencia acumulada (0–1)"),
                        y2="Max:Q"
                    )
                    line_mean = alt.Chart(df_mean).mark_line(color="black", strokeWidth=2).encode(
                        x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE))),
                        y="Promedio histórico:Q"
                    )
                    layers += [area_hist, line_mean]

                # Acumulada predicha (eje Y izquierdo)
                line_pred = alt.Chart(df_pred).mark_line(color="orange", strokeWidth=2.5).encode(
                    x=alt.X("Día:Q", title=f"Día juliano (1–{JD_MAX})", scale=alt.Scale(domain=list(XRANGE))),
                    y=alt.Y("Emergencia predicha:Q", title="Emergencia acumulada (0–1)",
                            scale=alt.Scale(domain=[0, 1]))
                )

                # Relativa semanal (eje Y derecho)
                area_rel = alt.Chart(df_rel).mark_area(color="steelblue", opacity=0.25).encode(
                    x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE))),
                    y=alt.Y("Emergencia relativa semanal:Q",
                            axis=alt.Axis(title="Emergencia relativa semanal", titleColor="steelblue"))
                )
                line_rel = alt.Chart(df_rel).mark_line(color="steelblue", strokeDash=[5, 3]).encode(
                    x=alt.X("Día:Q", scale=alt.Scale(domain=list(XRANGE))),
                    y="Emergencia relativa semanal:Q"
                )

                layers += [line_pred, area_rel, line_rel]

                chart = alt.layer(*layers).resolve_scale(
                    y='independent'
                ).properties(
                    height=480,
                    title="Curva predicha vs. histórico + Emergencia relativa semanal"
                )
                st.altair_chart(chart, use_container_width=True)

                # Descargas (predicción + relativa)
                out = pd.DataFrame({"Día": dias,
                                    "Emergencia_predicha": yhat,
                                    "Emergencia_relativa_semanal": rel})
                st.download_button(
                    "⬇️ Descargar curva y emergencia relativa (CSV)",
                    out.to_csv(index=False).encode("utf-8"),
                    file_name=f"curva_predicha_relativa_{JD_MAX}.csv",
                    mime="text/csv"
                )

                if show_hist_ref and isinstance(curvas_hist, dict) and len(curvas_hist) > 0:
                    df_comp = pd.DataFrame({
                        "Día": dias,
                        "Predicha": yhat,
                        "Hist_Min": y_min,
                        "Hist_Promedio": y_mean,
                        "Hist_Max": y_max
                    })
                    st.download_button(
                        "⬇️ Descargar comparación Predicha vs Histórica (CSV)",
                        df_comp.to_csv(index=False).encode("utf-8"),
                        file_name=f"comparacion_predicha_vs_historico_{JD_MAX}.csv",
                        mime="text/csv"
                    )
                    rmse = float(np.sqrt(mean_squared_error(y_mean, yhat)))
                    mae  = float(mean_absolute_error(y_mean, yhat))
                    st.caption(f"📏 Contra promedio histórico — RMSE: **{rmse:.3f}**, MAE: **{mae:.3f}**")

            except Exception as e:
                st.error(f"Error en la predicción: {e}")

# ===============================================================
# 🧬 TAB 4 — GENERACIÓN DE NUEVOS PATRONES (meteorología sintética)
# ===============================================================
with tabs[3]:
    st.subheader("🧬 Generar nuevos patrones (meteorología sintética → curva predicha)")
    st.markdown("""
    A partir del **modelo entrenado**, podés **simular condiciones meteorológicas no observadas**
    combinando promedios diarios históricos con perturbaciones aleatorias (ruido gaussiano).
    """)

    bundle = st.session_state.get("bundle", None)
    if not bundle:
        st.warning("⚠️ Primero entrená un modelo en la pestaña **🤖 Entrenar modelo**.")
    else:
        xsc, ysc, mlp = bundle["xsc"], bundle["ysc"], bundle["mlp"]

        meteo_hist_for_stats = st.file_uploader(
            "📂 Cargar meteo histórica para estimar medias/desvíos (mismo formato del entrenamiento)",
            type=["xlsx", "xls"],
            key="meteo_hist_gen"
        )

        if meteo_hist_for_stats is not None:
            sheets = pd.read_excel(meteo_hist_for_stats, sheet_name=None)
            frames = []
            for df in sheets.values():
                df = standardize_cols(df)
                df = slice_jan_to_oct1(df)
                if all(c in df.columns for c in ["tmin", "tmax", "prec"]):
                    frames.append(df[["tmin", "tmax", "prec"]].iloc[:JD_MAX])

            if not frames:
                st.error("No se encontraron columnas tmin/tmax/prec válidas en las hojas.")
            else:
                # Construir matrices (n_años, JD_MAX) por variable
                arr_tmin = np.vstack([f["tmin"].to_numpy(dtype="float64") for f in frames])
                arr_tmax = np.vstack([f["tmax"].to_numpy(dtype="float64") for f in frames])
                arr_prec = np.vstack([f["prec"].to_numpy(dtype="float64") for f in frames])

                mu_tmin, sd_tmin = arr_tmin.mean(axis=0), arr_tmin.std(axis=0)
                mu_tmax, sd_tmax = arr_tmax.mean(axis=0), arr_tmax.std(axis=0)
                mu_prec, sd_prec = arr_prec.mean(axis=0), arr_prec.std(axis=0)

                st.success(f"✅ Estadísticos diarios calculados a partir de {len(frames)} años.")

                col1, col2, col3 = st.columns(3)
                with col1:
                    n_scen = st.slider("Cantidad de escenarios a generar", 1, 100, 10, 1)
                with col2:
                    noise_scale = st.slider("Intensidad del ruido (σ multiplicador)", 0.1, 3.0, 1.0, 0.1)
                with col3:
                    clip_prec_0 = st.checkbox("Forzar precipitación ≥ 0", value=True)

                st.caption("Tip: aumentá el ruido para explorar escenarios más extremos; reducilo para escenarios conservadores.")
                if st.button("🚀 Generar patrones sintéticos"):
                    dias = np.arange(1, JD_MAX + 1)
                    curves = []
                    for i in range(n_scen):
                        # Generar secuencias sintéticas día a día
                        tmin_syn = mu_tmin + np.random.randn(JD_MAX) * sd_tmin * noise_scale
                        tmax_syn = mu_tmax + np.random.randn(JD_MAX) * sd_tmax * noise_scale
                        prec_syn = mu_prec + np.random.randn(JD_MAX) * sd_prec * noise_scale
                        if clip_prec_0:
                            prec_syn = np.clip(prec_syn, 0, None)

                        xnew = np.concatenate([tmin_syn, tmax_syn, prec_syn]).reshape(1, -1)

                        # Predicción con el modelo
                        yhat = ysc.inverse_transform(mlp.predict(xsc.transform(xnew)))[0]
                        yhat = np.maximum.accumulate(yhat)
                        yhat = yhat / (yhat[-1] if yhat[-1] != 0 else 1)
                        yhat = np.clip(yhat, 0, 1)
                        curves.append(yhat)

                    # Graficar todos los patrones generados
                    df_plot = pd.DataFrame(curves).T
                    df_plot["Día"] = dias
                    df_plot = df_plot.melt(id_vars="Día", var_name="Escenario", value_name="Emergencia acumulada")
                    chart = alt.Chart(df_plot).mark_line().encode(
                        x=alt.X("Día:Q", title=f"Día juliano (1–{JD_MAX})", scale=alt.Scale(domain=list(XRANGE))),
                        y=alt.Y("Emergencia acumulada:Q", title="Emergencia (0–1)", scale=alt.Scale(domain=[0, 1])),
                        color="Escenario:N"
                    ).properties(height=440, title="Patrones de emergencia generados sintéticamente")
                    st.altair_chart(chart, use_container_width=True)

                    # Exportación
                    csv = df_plot.pivot(index="Día", columns="Escenario", values="Emergencia acumulada").to_csv().encode("utf-8")
                    st.download_button("⬇️ Descargar patrones generados (CSV)", csv, "patrones_sinteticos.csv", mime="text/csv")
