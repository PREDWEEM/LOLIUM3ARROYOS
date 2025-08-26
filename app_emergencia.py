# app_emergencia.py (actualizado para histórico local + pronóstico API, con LOCKDOWN + 7 días de API)
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import requests, time, xml.etree.ElementTree as ET

# ========= LOCKDOWN STREAMLIT (sin menú, sin toolbar, sin badges) =========
st.set_page_config(
    page_title="PREDICCIÓN EMERGENCIA AGRÍCOLA - LOLIUM sp.",
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
    /* Menú hamburguesa */
    #MainMenu {visibility: hidden;}
    /* Footer */
    footer {visibility: hidden;}
    /* Toolbar superior (puede mostrar "View source" o "Manage app") */
    header [data-testid="stToolbar"] {visibility: hidden;}
    /* Badges / botones de despliegue */
    .viewerBadge_container__1QSob {visibility: hidden;}
    .st-emotion-cache-9aoz2h {visibility: hidden;}
    .stAppDeployButton {display: none;}
    </style>
    """,
    unsafe_allow_html=True
)
# ========= FIN LOCKDOWN =========

# ========= Utilidades de error seguro (mensajes genéricos en UI) =========
from typing import Callable, Any
def safe_run(fn: Callable[[], Any], user_msg: str):
    try:
        return fn()
    except Exception:
        st.error(user_msg)
        return None
# ========= FIN utilidades =========

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

    def tansig(self, x): return np.tanh(x)

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
        if valor < self.low_thr: return "Bajo"
        elif valor <= self.med_thr: return "Medio"
        else: return "Alto"

    def predict(self, X_real):
        X_norm = self.normalize_input(X_real)
        emerrel_pred = np.array([self._predict_single(x) for x in X_norm])
        emerrel_desnorm = self.desnormalizar_salida(emerrel_pred)
        emerrel_cumsum = np.cumsum(emerrel_desnorm)
        valor_max_emeac = 8.05
        emer_ac = emerrel_cumsum / valor_max_emeac
        emerrel_diff = np.diff(emer_ac, prepend=0)
        riesgo = np.array([self._clasificar(v) for v in emerrel_diff])
        return pd.DataFrame({
            "EMERREL(0-1)": emerrel_diff,
            "Nivel_Emergencia_relativa": riesgo
        })

# =================== Pronóstico API ===================
API_URL = "https://meteobahia.com.ar/scripts/forecast/for-ta.xml"
API_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://meteobahia.com.ar/",
    "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
}
FORECAST_DAYS_LIMIT = 7  # fijo en código: usar solo los primeros 7 días de pronóstico

def _to_float(x):
    try: return float(str(x).replace(",", "."))
    except: return None

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
                if not fval: continue
                rows.append({
                    "Fecha": pd.to_datetime(fval).normalize(),
                    "TMAX": _to_float(tmax.get("value")) if tmax is not None else None,
                    "TMIN": _to_float(tmin.get("value")) if tmin is not None else None,
                    "Prec": _to_float(precip.get("value")) if precip is not None else 0.0,
                })
            if not rows: raise RuntimeError("XML sin días válidos.")
            df = pd.DataFrame(rows).sort_values("Fecha").reset_index(drop=True)
            # === limitar a los primeros 7 días de pronóstico ===
            df = df.head(FORECAST_DAYS_LIMIT).copy()
            df["Julian_days"] = df["Fecha"].dt.dayofyear
            return df[["Fecha", "Julian_days", "TMAX", "TMIN", "Prec"]]
        except Exception as e:
            last_err = e
            time.sleep(backoff*(i+1))
    raise RuntimeError("No se pudo obtener el pronóstico desde la API.")

# =================== Utilidades ===================
def validar_columnas(df: pd.DataFrame) -> tuple[bool, str]:
    req = {"Fecha", "Julian_days", "TMAX", "TMIN", "Prec"}
    faltan = req - set(df.columns)
    if faltan: return False, f"Faltan columnas: {', '.join(sorted(faltan))}"
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

# =================== UI ===================
st.title("PREDICCIÓN EMERGENCIA AGRÍCOLA - LOLIUM sp. TRES ARROYOS")

st.sidebar.header("Fuente de datos")
fuente = st.sidebar.radio(
    "Elegí la fuente",
    ["Histórico local + Pronóstico (API)", "Subir histórico + usar Pronóstico (API)"]
)

st.sidebar.header("Configuración")
umbral_usuario = st.sidebar.number_input(
    "Umbral de EMEAC para 100%", min_value=1.2, max_value=3.0, value=2.70, step=0.01, format="%.2f"
)

st.sidebar.header("Validaciones")
mostrar_fuera_rango = st.sidebar.checkbox("Avisar datos fuera de rango de entrenamiento", value=False)

if st.sidebar.button("Forzar recarga de datos"):
    st.cache_data.clear()

# Pesos modelo (mensaje genérico si falta algo)
def _cargar_pesos():
    base = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    return load_weights(base)
pesos = safe_run(
    _cargar_pesos,
    "Error al cargar archivos del modelo. Verifique que IW.npy, bias_IW.npy, LW.npy y bias_out.npy estén junto al script."
)
if pesos is None:
    st.stop()
IW, bias_IW, LW, bias_out = pesos
modelo = PracticalANNModel(IW, bias_IW, LW, bias_out)

# =================== Cargar histórico ===================
df_hist = None
hist_path_default = Path("meteo_daily.csv")

if fuente == "Histórico local + Pronóstico (API)":
    def _leer_hist_local():
        return pd.read_csv(hist_path_default, parse_dates=["Fecha"])
    if hist_path_default.exists():
        df_hist = safe_run(_leer_hist_local, "No se pudo leer el histórico local.")
    else:
        st.warning("No se encontró el histórico local. Subí un CSV en la opción de carga para combinar con la API.")
elif fuente == "Subir histórico + usar Pronóstico (API)":
    up = st.file_uploader("Subí el histórico (.csv) con columnas: Fecha, Julian_days, TMAX, TMIN, Prec", type=["csv"])
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

# =================== Procesamiento y gráficos ===================
def plot_and_table(nombre, df):
    df = df.sort_values("Fecha").reset_index(drop=True)
    X_real = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(dtype=float)
    fechas = pd.to_datetime(df["Fecha"])

    if mostrar_fuera_rango and detectar_fuera_rango(X_real, modelo.input_min, modelo.input_max):
        st.info("⚠️ Hay valores fuera del rango de entrenamiento de la red.")

    pred = modelo.predict(X_real)
    pred["Fecha"] = fechas
    pred["Julian_days"] = df["Julian_days"]
    pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()
    pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()

    pred["EMEAC (0-1) - mínimo"] = pred["EMERREL acumulado"] / 1.2
    pred["EMEAC (0-1) - máximo"] = pred["EMERREL acumulado"] / 3.0
    pred["EMEAC (0-1) - ajustable"] = pred["EMERREL acumulado"] / umbral_usuario
    pred["EMEAC (%) - mínimo"] = pred["EMEAC (0-1) - mínimo"] * 100
    pred["EMEAC (%) - máximo"] = pred["EMEAC (0-1) - máximo"] * 100
    pred["EMEAC (%) - ajustable"] = pred["EMEAC (0-1) - ajustable"] * 100

    years = pred["Fecha"].dt.year.unique()
    yr = int(years[0]) if len(years) == 1 else int(st.sidebar.selectbox("Año a mostrar (reinicio 1/feb → 1/sep)", sorted(years), key=f"year_select_{nombre}"))

    fecha_inicio_rango = pd.Timestamp(year=yr, month=2, day=1)
    fecha_fin_rango    = pd.Timestamp(year=yr, month=9, day=1)
    mask = (pred["Fecha"] >= fecha_inicio_rango) & (pred["Fecha"] <= fecha_fin_rango)
    pred_vis = pred.loc[mask].copy()
    if pred_vis.empty:
        st.warning(f"No hay datos entre {fecha_inicio_rango.date()} y {fecha_fin_rango.date()} para {nombre}.")
        return

    pred_vis["EMERREL acumulado (reiniciado)"] = pred_vis["EMERREL(0-1)"].cumsum()
    pred_vis["EMEAC (0-1) - mínimo (rango)"]    = pred_vis["EMERREL acumulado (reiniciado)"] / 1.2
    pred_vis["EMEAC (0-1) - máximo (rango)"]    = pred_vis["EMERREL acumulado (reiniciado)"] / 3.0
    pred_vis["EMEAC (0-1) - ajustable (rango)"] = pred_vis["EMERREL acumulado (reiniciado)"] / umbral_usuario
    pred_vis["EMEAC (%) - mínimo (rango)"]      = pred_vis["EMEAC (0-1) - mínimo (rango)"] * 100
    pred_vis["EMEAC (%) - máximo (rango)"]      = pred_vis["EMEAC (0-1) - máximo (rango)"] * 100
    pred_vis["EMEAC (%) - ajustable (rango)"]   = pred_vis["EMEAC (0-1) - ajustable (rango)"] * 100

    pred_vis["EMERREL_MA5_rango"] = pred_vis["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()
    colores_vis = obtener_colores(pred_vis["Nivel_Emergencia_relativa"])

    st.subheader("EMERGENCIA RELATIVA DIARIA - TRES ARROYOS")
    fig_er = go.Figure()
    fig_er.add_bar(
        x=pred_vis["Fecha"], y=pred_vis["EMERREL(0-1)"],
        marker=dict(color=colores_vis.tolist()),
        hovertemplate=("Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.3f}<br>Nivel: %{customdata}"),
        customdata=pred_vis["Nivel_Emergencia_relativa"], name="EMERREL (0-1)",
    )
    fig_er.add_trace(go.Scatter(
        x=pred_vis["Fecha"], y=pred_vis["EMERREL_MA5_rango"],
        mode="lines", name="Media móvil 5 días (rango)",
        hovertemplate="Fecha: %{x|%d-%b-%Y}<br>MA5: %{y:.3f}<extra></extra>"
    ))
    fig_er.add_trace(go.Scatter(
        x=pred_vis["Fecha"], y=pred_vis["EMERREL_MA5_rango"],
        mode="lines", line=dict(width=0), fill="tozeroy",
        fillcolor="rgba(135, 206, 250, 0.3)", name="Área MA5",
        hoverinfo="skip", showlegend=False
    ))
    low_thr = float(modelo.low_thr); med_thr = float(modelo.med_thr)
    fig_er.add_trace(go.Scatter(x=[fecha_inicio_rango, fecha_fin_rango], y=[low_thr, low_thr],
        mode="lines", line=dict(color="green", dash="dot"),
        name=f"Bajo (≤ {low_thr:.3f})", hoverinfo="skip"))
    fig_er.add_trace(go.Scatter(x=[fecha_inicio_rango, fecha_fin_rango], y=[med_thr, med_thr],
        mode="lines", line=dict(color="orange", dash="dot"),
        name=f"Medio (≤ {med_thr:.3f})", hoverinfo="skip"))
    fig_er.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
        line=dict(color="red", dash="dot"), name=f"Alto (> {med_thr:.3f})",
        hoverinfo="skip", showlegend=True))
    fig_er.update_layout(xaxis_title="Fecha", yaxis_title="EMERREL (0-1)",
                         hovermode="x unified", legend_title="Referencias", height=650)
    fig_er.update_xaxes(range=[fecha_inicio_rango, fecha_fin_rango], dtick="M1", tickformat="%b")
    fig_er.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig_er, use_container_width=True, theme="streamlit")

    st.subheader("EMERGENCIA ACUMULADA DIARIA - TRES ARROYOS")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - máximo (rango)"],
                             mode="lines", line=dict(width=0), name="Máximo (reiniciado)",
                             hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - mínimo (rango)"],
                             mode="lines", line=dict(width=0), fill="tonexty", name="Mínimo (reiniciado)",
                             hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - ajustable (rango)"],
                             mode="lines", name="Umbral ajustable (reiniciado)",
                             hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Ajustable: %{y:.1f}%<extra></extra>",
                             line=dict(width=2.5)))
    fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - mínimo (rango)"],
                             mode="lines", name="Umbral mínimo (reiniciado)",
                             line=dict(dash="dash", width=1.5),
                             hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - máximo (rango)"],
                             mode="lines", name="Umbral máximo (reiniciado)",
                             line=dict(dash="dash", width=1.5),
                             hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"))
    for nivel in [25, 50, 75, 90]:
        fig.add_hline(y=nivel, line_dash="dash", opacity=0.6, annotation_text=f"{nivel}%")
    fig.update_layout(xaxis_title="Fecha", yaxis_title="EMEAC (%)", yaxis=dict(range=[0, 100]),
                      hovermode="x unified", legend_title="Referencias", height=600)
    fig.update_xaxes(range=[fecha_inicio_rango, fecha_fin_rango], dtick="M1", tickformat="%b")
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    st.subheader(f"Resultados (1/feb → 1/sep) - {nombre}")
    col_emeac = "EMEAC (%) - ajustable (rango)" if "EMEAC (%) - ajustable (rango)" in pred_vis.columns else "EMEAC (%) - ajustable"
    tabla = pred_vis[["Fecha", "Julian_days", "Nivel_Emergencia_relativa", col_emeac]].rename(
        columns={"Nivel_Emergencia_relativa": "Nivel de EMERREL", col_emeac: "EMEAC (%)"}
    )
    st.dataframe(tabla, use_container_width=True)
    csv = tabla.to_csv(index=False).encode("utf-8")
    st.download_button(f"Descargar resultados (rango) - {nombre}", csv, f"{nombre}_resultados_rango.csv", "text/csv")

for nombre, df in dfs:
    plot_and_table(nombre, df)
