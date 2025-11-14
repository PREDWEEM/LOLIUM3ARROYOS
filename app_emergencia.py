
# app_emergencia_v7.py
# ===============================================================
# 🌾 PREDWEEM v7 — ANN + Clasificación anticipada (solo clima)
# - Predicción de EMERREL/EMERAC con ANN
# - Histórico local + API MeteoBahía (7 días)
# - Clasificación Temprano / Extendido usando d25, d50, d90
#   simulados a partir de la ANN (clasificación anticipada)
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import requests, time, xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pickle
from typing import Callable, Any

# ========= LOCKDOWN STREAMLIT (sin menú, sin toolbar, sin badges) =========
st.set_page_config(
    page_title="PREDWEEM v7 - EMERGENCIA + CLASIFICACIÓN ANTICIPADA",
    layout="wide",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": None
    }
)
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header [data-testid="stToolbar"] {visibility: hidden;}
    .viewerBadge_container__1QSob {visibility: hidden;}
    .st-emotion-cache-9aoz2h {visibility: hidden;}
    .stAppDeployButton {display: none;}
    </style>
    """,
    unsafe_allow_html=True
)
# ========= FIN LOCKDOWN =========

BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ========= Utilidades de error seguro =========
def safe_run(fn: Callable[[], Any], user_msg: str):
    try:
        return fn()
    except Exception:
        st.error(user_msg)
        return None

# =================== Modelo ANN ===================
class PracticalANNModel:
    def __init__(self, IW, bias_IW, LW, bias_out, low=0.02, medium=0.079):
        self.IW = IW
        self.bias_IW = bias_IW
        self.LW = LW
        self.bias_out = bias_out
        # Orden esperado: [Julian_days, TMAX, TMIN, Prec]
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])
        self.low_thr = low
        self.med_thr = medium

    def tansig(self, x):
        return np.tanh(x)

    def normalize_input(self, X_real):
        return 2 * (X_real - self.input_min) / (self.input_max - self.input_min) - 1

    def desnormalizar_salida(self, y_norm, ymin=-1, ymax=1):
        return (y_norm - ymin) / (ymax - ymin)

    def _predict_single(self, x_norm):
        z1 = self.IW.T @ x_norm + self.bias_IW
        a1 = self.tansig(z1)
        z2 = self.LW @ a1 + self.bias_out
        return self.tansig(z2)

    def _clasificar(self, valor):
        if valor < self.low_thr:
            return "Bajo"
        elif valor <= self.med_thr:
            return "Medio"
        else:
            return "Alto"

    def predict_df(self, df_meteo: pd.DataFrame) -> pd.DataFrame:
        """Devuelve DataFrame con EMERREL(0-1), riesgo, EMERAC, etc."""
        X_real = df_meteo[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(dtype=float)
        X_norm = self.normalize_input(X_real)
        emerrel_pred = np.array([self._predict_single(x) for x in X_norm])
        emerrel_desnorm = self.desnormalizar_salida(emerrel_pred)
        emer_cumsum = np.cumsum(emerrel_desnorm)
        valor_max_emeac = 8.05  # calibra total EMERAC
        emer_ac = emer_cumsum / valor_max_emeac
        emerrel_diff = np.diff(emer_ac, prepend=0)
        riesgo = np.array([self._clasificar(v) for v in emerrel_diff])
        out = pd.DataFrame({
            "EMERREL(0-1)": emerrel_diff,
            "Nivel_Emergencia_relativa": riesgo,
        })
        out["EMERAC(0-1)"] = emer_ac
        return out

# =================== Pronóstico API ===================
API_URL = "https://meteobahia.com.ar/scripts/forecast/for-ta.xml"
API_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://meteobahia.com.ar/",
    "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
}
FORECAST_DAYS_LIMIT = 7  # usar solo primeros 7 días de pronóstico

def _to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return None

@st.cache_data(ttl=15*60, show_spinner=False)
def fetch_forecast(url: str = API_URL, retries: int = 3, backoff: int = 2) -> pd.DataFrame:
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=API_HEADERS, timeout=30)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            days = root.findall(".//forecast/tabular/day")
            rows = []
            for d in days:
                fecha  = d.find("./fecha")
                tmax   = d.find("./tmax")
                tmin   = d.find("./tmin")
                precip = d.find("./precip")
                fval = fecha.get("value") if fecha is not None else None
                if not fval:
                    continue
                rows.append({
                    "Fecha": pd.to_datetime(fval).normalize(),
                    "TMAX": _to_float(tmax.get("value")) if tmax is not None else None,
                    "TMIN": _to_float(tmin.get("value")) if tmin is not None else None,
                    "Prec": _to_float(precip.get("value")) if precip is not None else 0.0,
                })
            if not rows:
                raise RuntimeError("XML sin días válidos.")
            df = pd.DataFrame(rows).sort_values("Fecha").reset_index(drop=True)
            df = df.head(FORECAST_DAYS_LIMIT).copy()
            df["Julian_days"] = df["Fecha"].dt.dayofyear
            return df[["Fecha", "Julian_days", "TMAX", "TMIN", "Prec"]]
        except Exception as e:
            last_err = e
            time.sleep(backoff * (i + 1))
    raise RuntimeError("No se pudo obtener el pronóstico desde la API.")

# =================== Utilidades ===================
def validar_columnas(df: pd.DataFrame) -> tuple[bool, str]:
    req = {"Fecha", "Julian_days", "TMAX", "TMIN", "Prec"}
    faltan = req - set(df.columns)
    if faltan:
        return False, f"Faltan columnas: {', '.join(sorted(faltan))}"
    return True, ""

def obtener_colores(niveles: pd.Series):
    m = niveles.map({"Bajo": "green", "Medio": "orange", "Alto": "red"})
    return m.fillna("gray")

def detectar_fuera_rango(X_real: np.ndarray, input_min: np.ndarray, input_max: np.ndarray) -> bool:
    out = (X_real < input_min) | (X_real > input_max)
    return bool(np.any(out))

@st.cache_data(show_spinner=False)
def load_weights(base_dir: Path):
    IW = np.load(base_dir / "IW.npy")
    bias_IW = np.load(base_dir / "bias_IW.npy")
    LW = np.load(base_dir / "LW.npy")
    bias_out = np.load(base_dir / "bias_out.npy")
    return IW, bias_IW, LW, bias_out

# ============= CLASIFICADOR TEMPRANO / EXTENDIDO (d25, d50, d90) =============
CLUSTER_MODEL_FILE = "modelo_cluster_d25_d50_d90.pkl"

@st.cache_resource(show_spinner=False)
def load_cluster_model(base_dir: Path):
    path = base_dir / CLUSTER_MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(str(path))
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["scaler"], data["model"]

def calcular_d25_d50_d90(dias: np.ndarray, emerac: np.ndarray):
    """Calcula d25, d50, d90 a partir de EMERAC (0-1) simulada."""
    if len(dias) < 3:
        return None, None, None
    y = emerac.copy()
    # asegurar que EMERAC esté entre 0 y 1
    if y.max() <= 0:
        return None, None, None
    y = y / y.max()
    try:
        d25 = np.interp(0.25, y, dias)
        d50 = np.interp(0.50, y, dias)
        d90 = np.interp(0.90, y, dias)
    except Exception:
        return None, None, None
    return float(d25), float(d50), float(d90)

def generar_curva_centroides(d25, d50, d90):
    x = np.array([d25, d50, d90])
    y = np.array([0.25, 0.50, 0.90])
    dias = np.arange(20, 200)
    curva = np.interp(dias, x, y)
    return dias, curva

# =================== UI ===================
st.title("🌾 PREDWEEM v7 — EMERGENCIA + CLASIFICACIÓN ANTICIPADA")

st.sidebar.header("Fuente de datos")
fuente = st.sidebar.radio(
    "Elegí la fuente",
    ["Histórico local + Pronóstico (API)", "Subir histórico + usar Pronóstico (API)"]
)

st.sidebar.header("Configuración EMEAC")
umbral_usuario = st.sidebar.number_input(
    "Umbral de EMEAC para 100%", min_value=1.2, max_value=3.0,
    value=2.70, step=0.01, format="%.2f"
)

st.sidebar.header("Clasificación anticipada")
dia_corte = st.sidebar.slider(
    "Día juliano de corte para clasificación anticipada",
    min_value=40, max_value=200, value=121, step=1,
    help="Ej: 121 ≈ 1 de mayo"
)

st.sidebar.header("Validaciones")
mostrar_fuera_rango = st.sidebar.checkbox(
    "Avisar datos fuera de rango de entrenamiento ANN", value=False
)

if st.sidebar.button("Forzar recarga de datos"):
    st.cache_data.clear()
    st.cache_resource.clear()

# Pesos modelo ANN
def _cargar_pesos():
    return load_weights(BASE_DIR)

pesos = safe_run(
    _cargar_pesos,
    "Error al cargar archivos del modelo ANN. Verifique que IW.npy, bias_IW.npy, LW.npy y bias_out.npy estén junto al script."
)
if pesos is None:
    st.stop()

IW, bias_IW, LW, bias_out = pesos
modelo = PracticalANNModel(IW, bias_IW, LW, bias_out)

# Modelo de cluster Temprano/Extendido
cluster_model_data = safe_run(
    lambda: load_cluster_model(BASE_DIR),
    f"No se pudo cargar el modelo de patrón {CLUSTER_MODEL_FILE}. Colocá el archivo junto al script si querés clasificar Temprano/Extendido."
)
if cluster_model_data is not None:
    scaler_clust, model_clust = cluster_model_data
else:
    scaler_clust = model_clust = None

# =================== Cargar histórico ===================
df_hist = None
hist_path_default = BASE_DIR / "meteo_daily.csv"

if fuente == "Histórico local + Pronóstico (API)":
    def _leer_hist_local():
        return pd.read_csv(hist_path_default, parse_dates=["Fecha"])
    if hist_path_default.exists():
        df_hist = safe_run(_leer_hist_local, "No se pudo leer el histórico local.")
    else:
        st.warning("No se encontró el histórico local (meteo_daily.csv). Podés subir un CSV en la otra opción.")
else:
    up = st.file_uploader(
        "Subí el histórico (.csv) con columnas: Fecha, Julian_days, TMAX, TMIN, Prec",
        type=["csv"]
    )
    if up is not None:
        def _leer_hist_upload():
            return pd.read_csv(up, parse_dates=["Fecha"])
        df_hist = safe_run(_leer_hist_upload, "No se pudo leer el CSV subido.")

# Validar/limpiar histórico
if df_hist is not None:
    ok, msg = validar_columnas(df_hist)
    if not ok:
        st.error(f"Histórico inválido: {msg}")
        df_hist = None
    else:
        cols_num = ["Julian_days", "TMAX", "TMIN", "Prec"]
        df_hist[cols_num] = df_hist[cols_num].apply(pd.to_numeric, errors="coerce")
        df_hist = df_hist.dropna(subset=cols_num).copy()
        df_hist["Fecha"] = pd.to_datetime(df_hist["Fecha"]).dt.normalize()
        df_hist["Julian_days"] = df_hist["Fecha"].dt.dayofyear
        df_hist = df_hist.sort_values("Fecha").reset_index(drop=True)

# =================== Pronóstico (API) ===================
df_fcst = safe_run(fetch_forecast, "Fallo al obtener el pronóstico desde la API.")

# =================== Combinar ===================
dfs = []
if df_hist is not None and df_fcst is not None:
    today = pd.Timestamp.today().normalize()
    df_hist_past = df_hist[df_hist["Fecha"] < today].copy()
    df_fcst_today_fwd = df_fcst[df_fcst["Fecha"] >= today].copy()
    df_all = pd.concat([df_hist_past, df_fcst_today_fwd], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["Fecha"], keep="last").sort_values("Fecha").reset_index(drop=True)
    df_all["Julian_days"] = df_all["Fecha"].dt.dayofyear
    dfs.append(("Histórico+Pronóstico", df_all))
elif df_fcst is not None and df_hist is None:
    st.info("Usando solo Pronóstico (API) porque no hay histórico válido disponible.")
    dfs.append(("Solo_Pronóstico", df_fcst))
elif df_hist is not None and df_fcst is None:
    st.info("Usando solo Histórico porque falló el pronóstico de la API.")
    dfs.append(("Solo_Histórico", df_hist))
else:
    st.stop()

# =================== Bloque centroides históricos fijos ===================
metricas_hist = pd.DataFrame([
    {"d_25":61,"d_50":66,"d_90":156,"cluster":1,"archivo":"2009"},
    {"d_25":77,"d_50":92,"d_90":115,"cluster":1,"archivo":"2014"},
    {"d_25":79,"d_50":83,"d_90":132,"cluster":1,"archivo":"2011"},
    {"d_25":113,"d_50":119,"d_90":151,"cluster":1,"archivo":"2015"},
    {"d_25":63,"d_50":67,"d_90":107,"cluster":0,"archivo":"2013"},
    {"d_25":75,"d_50":91,"d_90":132,"cluster":1,"archivo":"2024"},
    {"d_25":67,"d_50":85,"d_90":131,"cluster":1,"archivo":"2023"},
    {"d_25":60,"d_50":65,"d_90":135,"cluster":1,"archivo":"2008"},
    {"d_25":37,"d_50":48,"d_90":86,"cluster":0,"archivo":"2012"}
])
centroides = metricas_hist.groupby("cluster")[["d_25","d_50","d_90"]].mean()

# =================== Procesamiento y gráficos ===================
def procesar_escenario(nombre, df_all):
    st.markdown(f"## Escenario: **{nombre}**")

    df_all = df_all.sort_values("Fecha").reset_index(drop=True)
    X_real = df_all[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(dtype=float)

    if mostrar_fuera_rango and detectar_fuera_rango(X_real, modelo.input_min, modelo.input_max):
        st.info("⚠️ Hay valores fuera del rango de entrenamiento de la ANN.")

    # ANN → EMERREL / EMERAC
    pred_ann = modelo.predict_df(df_all)
    pred = df_all.copy()
    pred["EMERREL(0-1)"] = pred_ann["EMERREL(0-1)"]
    pred["Nivel_Emergencia_relativa"] = pred_ann["Nivel_Emergencia_relativa"]
    pred["EMERAC(0-1)"] = pred_ann["EMERAC(0-1)"]
    pred["Fecha"] = pd.to_datetime(pred["Fecha"])

    # EMEAC (escala 0–100) con distintos umbrales
    pred["EMEAC_min"] = pred["EMERAC(0-1)"] / 1.2 * 100
    pred["EMEAC_max"] = pred["EMERAC(0-1)"] / 3.0 * 100
    pred["EMEAC_ajust"] = pred["EMERAC(0-1)"] / umbral_usuario * 100

    years = pred["Fecha"].dt.year.unique()
    yr = int(years[0]) if len(years) == 1 else int(
        st.sidebar.selectbox(
            f"Año a mostrar ({nombre})",
            sorted(years),
            key=f"year_select_{nombre}"
        )
    )

    # Filtrar año
    pred_year = pred[pred["Fecha"].dt.year == yr].copy()
    if pred_year.empty:
        st.warning(f"No hay datos para el año {yr} en {nombre}.")
        return

    # ---------- CLASIFICACIÓN ANTICIPADA ----------
    st.subheader("Clasificación anticipada del patrón (ANN + clima)")

    # recorte hasta día de corte
    pred_corte = pred_year[pred_year["Julian_days"] <= dia_corte].copy()
    if len(pred_corte) < 5 or cluster_model_data is None:
        st.info("No hay suficientes datos hasta el día de corte, o el modelo de cluster no está disponible.")
    else:
        dias_corte = pred_corte["Julian_days"].to_numpy()
        emerac_corte = pred_corte["EMERAC(0-1)"].to_numpy()
        d25, d50, d90 = calcular_d25_d50_d90(dias_corte, emerac_corte)

        if d25 is None:
            st.info("No se pudieron calcular d25, d50, d90 en el rango elegido.")
        else:
            X_in = np.array([[d25, d50, d90]])
            X_sc = scaler_clust.transform(X_in)
            cl = int(model_clust.predict(X_sc)[0])

            nombres_cl = {0: "🌱 Temprano / Compacto", 1: "🌾 Extendido / Prolongado"}
            color_cl = {0: "green", 1: "orange"}
            patron = nombres_cl[cl]

            st.markdown(
                f"### Año {yr} — Día de corte: {dia_corte}  "
                f"<br>Patrón proyectado: <span style='color:{color_cl[cl]}; font-size:26px;'>{patron}</span>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"- d25 ≈ **{d25:.1f}**  "
                f"- d50 ≈ **{d50:.1f}**  "
                f"- d90 ≈ **{d90:.1f}**"
            )

            # Gráfico EMERAC simulada (hasta corte) + marcas
            fig_ac, ax_ac = plt.subplots(figsize=(8,4))
            y_norm = emerac_corte.copy()
            y_norm = y_norm / y_norm.max() if y_norm.max() > 0 else y_norm
            ax_ac.plot(dias_corte, y_norm, label="EMERAC simulada (0-1)", linewidth=2)
            ax_ac.axvline(d25, color="gray", linestyle="--", label="d25")
            ax_ac.axvline(d50, color="black", linestyle="--", label="d50")
            ax_ac.axvline(d90, color="red", linestyle="--", label="d90")
            ax_ac.set_xlabel("Día juliano")
            ax_ac.set_ylabel("EMERAC normalizada (0–1)")
            ax_ac.set_title("Emergencia acumulada simulada (ANN) hasta día de corte")
            ax_ac.legend()
            st.pyplot(fig_ac)

            # Curvas representativas de centroides vs año proyectado
            d25_0, d50_0, d90_0 = centroides.loc[0]["d_25"], centroides.loc[0]["d_50"], centroides.loc[0]["d_90"]
            d25_1, d50_1, d90_1 = centroides.loc[1]["d_25"], centroides.loc[1]["d_50"], centroides.loc[1]["d_90"]

            dias_x, curva_x = generar_curva_centroides(d25, d50, d90)
            dias0, curva0 = generar_curva_centroides(d25_0, d50_0, d90_0)
            dias1, curva1 = generar_curva_centroides(d25_1, d50_1, d90_1)

            fig_pat, ax_pat = plt.subplots(figsize=(8,4))
            ax_pat.plot(dias_x, curva_x, label="Proyección año actual", linewidth=3, color="blue")
            ax_pat.plot(dias0, curva0, label="Centroide Temprano", linewidth=2, color="green")
            ax_pat.plot(dias1, curva1, label="Centroide Extendido", linewidth=2, color="orange")
            ax_pat.set_xlabel("Día juliano")
            ax_pat.set_ylabel("Emergencia acumulada (0–1)")
            ax_pat.set_title("Comparación con patrones históricos (curvas típicas)")
            ax_pat.legend()
            st.pyplot(fig_pat)

    # ---------- EMERGENCIA RELATIVA Y ACUMULADA (1/feb → 1/sep) ----------
    st.subheader("EMERGENCIA RELATIVA Y ACUMULADA (1/feb → 1/sep)")

    fecha_inicio_rango = pd.Timestamp(year=yr, month=2, day=1)
    fecha_fin_rango    = pd.Timestamp(year=yr, month=9, day=1)
    mask = (pred_year["Fecha"] >= fecha_inicio_rango) & (pred_year["Fecha"] <= fecha_fin_rango)
    pred_vis = pred_year.loc[mask].copy()
    if pred_vis.empty:
        st.warning(f"No hay datos entre {fecha_inicio_rango.date()} y {fecha_fin_rango.date()} para {nombre}.")
        return

    # EMERREL para rango
    pred_vis["EMERREL_MA5_rango"] = pred_vis["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()
    colores_vis = obtener_colores(pred_vis["Nivel_Emergencia_relativa"])

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("#### EMERREL diaria (0–1)")
        fig_er = go.Figure()
        fig_er.add_bar(
            x=pred_vis["Fecha"], y=pred_vis["EMERREL(0-1)"],
            marker=dict(color=colores_vis.tolist()),
            hovertemplate=("Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.3f}<br>Nivel: %{customdata}"),
            customdata=pred_vis["Nivel_Emergencia_relativa"],
            name="EMERREL (0-1)",
        )
        fig_er.add_trace(go.Scatter(
            x=pred_vis["Fecha"], y=pred_vis["EMERREL_MA5_rango"],
            mode="lines", name="Media móvil 5 días",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>MA5: %{y:.3f}<extra></extra>"
        ))
        low_thr = float(modelo.low_thr); med_thr = float(modelo.med_thr)
        fig_er.add_trace(go.Scatter(
            x=[fecha_inicio_rango, fecha_fin_rango], y=[low_thr, low_thr],
            mode="lines", line=dict(color="green", dash="dot"),
            name=f"Bajo (≤ {low_thr:.3f})", hoverinfo="skip"))
        fig_er.add_trace(go.Scatter(
            x=[fecha_inicio_rango, fecha_fin_rango], y=[med_thr, med_thr],
            mode="lines", line=dict(color="orange", dash="dot"),
            name=f"Medio (≤ {med_thr:.3f})", hoverinfo="skip"))
        fig_er.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="red", dash="dot"),
            name=f"Alto (> {med_thr:.3f})",
            hoverinfo="skip", showlegend=True))
        fig_er.update_layout(
            xaxis_title="Fecha", yaxis_title="EMERREL (0-1)",
            hovermode="x unified", legend_title="Referencias", height=450
        )
        fig_er.update_xaxes(range=[fecha_inicio_rango, fecha_fin_rango], dtick="M1", tickformat="%b")
        fig_er.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_er, use_container_width=True, theme="streamlit")

    with col_g2:
        st.markdown("#### EMERGENCIA acumulada (EMEAC %)")
        fig_ac2 = go.Figure()
        fig_ac2.add_trace(go.Scatter(
            x=pred_vis["Fecha"], y=pred_vis["EMEAC_max"],
            mode="lines", line=dict(width=0), name="Máximo (escenario)",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"))
        fig_ac2.add_trace(go.Scatter(
            x=pred_vis["Fecha"], y=pred_vis["EMEAC_min"],
            mode="lines", line=dict(width=0), fill="tonexty", name="Mínimo (escenario)",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"))
        fig_ac2.add_trace(go.Scatter(
            x=pred_vis["Fecha"], y=pred_vis["EMEAC_ajust"],
            mode="lines", name="Umbral ajustable",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Ajustable: %{y:.1f}%<extra></extra>",
            line=dict(width=2.5)))
        for nivel in [25, 50, 75, 90]:
            fig_ac2.add_hline(y=nivel, line_dash="dash", opacity=0.6, annotation_text=f"{nivel}%")
        fig_ac2.update_layout(
            xaxis_title="Fecha", yaxis_title="EMEAC (%)",
            yaxis=dict(range=[0, 100]),
            hovermode="x unified", legend_title="Referencias", height=450
        )
        fig_ac2.update_xaxes(range=[fecha_inicio_rango, fecha_fin_rango], dtick="M1", tickformat="%b")
        st.plotly_chart(fig_ac2, use_container_width=True, theme="streamlit")

    # Tabla resumen del rango
    st.subheader(f"Resultados (1/feb → 1/sep) - {nombre}, año {yr}")
    tabla = pred_vis[["Fecha", "Julian_days", "Nivel_Emergencia_relativa", "EMEAC_ajust"]].rename(
        columns={
            "Nivel_Emergencia_relativa": "Nivel de EMERREL",
            "EMEAC_ajust": "EMEAC (%) ajustable"
        }
    )
    st.dataframe(tabla, use_container_width=True)
    csv = tabla.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"Descargar resultados (1/feb–1/sep) - {nombre} {yr}",
        csv,
        f"{nombre}_resultados_{yr}.csv",
        "text/csv"
    )

# Ejecutar para cada escenario combinado
for nombre, df in dfs:
    procesar_escenario(nombre, df)
