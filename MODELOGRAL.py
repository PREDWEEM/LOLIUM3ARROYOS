# -*- coding: utf-8 -*-
# 📈 Emergencia acumulada histórica — análisis comparativo por día juliano
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
from pathlib import Path
from io import StringIO

# ============ CONFIG ============ #
st.set_page_config(page_title="Emergencia Acumulada Histórica", layout="wide")
st.title("Análisis histórico de emergencia acumulada")

st.markdown(
    """
- Carga automática de archivos anuales (Excel) en el directorio actual.  
- Exclusiones: **2010** y **2015**.  
- La curva de cada año se **normaliza** a 1 (100%) en el día 365.  
- Elegí el **día juliano** y un **umbral** (por defecto 50%) para ver estadísticas.  
- Se muestra el **promedio**, bandas **P10–P90** e **IQR (P25–P75)**.
"""
)

# ============ UTILIDADES ============ #
def _parse_year_from_name(name: str) -> str:
    m = re.match(r"^(\d+)", name)
    return m.group(1) if m else name

def _is_excluded(yr: str) -> bool:
    # excluir 2010 y 2015
    try:
        y = int(yr)
        return y in (2010, 2015)
    except:
        return False

@st.cache_data
def cargar_curvas_desde_excels(directorio: str = "."):
    """
    Busca *.xlsx en 'directorio', arma curvas normalizadas (365 días),
    y devuelve (dict_año->np.array(365,), listado_años_ordenado).
    """
    p = Path(directorio)
    archivos = sorted([f for f in p.glob("*.xlsx")])

    curvas = {}
    for f in archivos:
        anno = _parse_year_from_name(f.stem)
        if _is_excluded(anno):
            continue
        try:
            # Asumimos dos columnas: día juliano y valor diario
            df = pd.read_excel(f, header=None)
        except Exception as e:
            st.warning(f"⚠️ No se pudo leer {f.name}: {e}")
            continue

        if df.empty or df.shape[1] < 2:
            st.warning(f"⚠️ Formato inesperado en {f.name}; se requieren 2 columnas (día, valor).")
            continue

        # Coerción segura
        dias = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
        vals = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
        mask = ~np.isnan(dias) & ~np.isnan(vals)
        dias, vals = dias[mask], vals[mask]

        # Vector diario (365)
        v = np.zeros(365, dtype=float)
        for d, val in zip(dias, vals):
            try:
                idx = int(d) - 1
                if 0 <= idx < 365:
                    v[idx] += float(val)  # si hay duplicados en un día, se suma
            except:
                pass

        curva_acum = np.cumsum(v)
        fin = curva_acum[-1]
        if fin > 0:
            curva_norm = curva_acum / fin
        else:
            # si no hubo datos > 0, dejamos todo en 0
            curva_norm = curva_acum

        curvas[anno] = curva_norm

    # ordenar por año numérico si es posible
    def _key(a):
        try:
            return int(a)
        except:
            return a

    anios = sorted(curvas.keys(), key=_key)
    return curvas, anios

# ============ LECTURA ============ #
curvas_dict, anios_disponibles = cargar_curvas_desde_excels(".")
if not anios_disponibles:
    st.error("No se encontraron archivos .xlsx válidos en el directorio actual (excluyendo 2010 y 2015).")
    st.stop()

# ============ CONTROLES ============ #
colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    anios_sel = st.multiselect(
        "Años a incluir",
        options=anios_disponibles,
        default=anios_disponibles,
        key="ms_anios",
        help="Podés filtrar años para el análisis y las curvas."
    )
with colB:
    dia_sel = st.slider("Día juliano", min_value=1, max_value=365, value=180, key="k_dia")
with colC:
    umbral = st.slider("Umbral (fracción)", min_value=0.05, max_value=0.95, value=0.50, step=0.05, key="k_umbral")

if not anios_sel:
    st.warning("Seleccioná al menos un año.")
    st.stop()

# ============ ARMADO MATRIZ (n_años x 365) ============ #
mat = np.vstack([curvas_dict[a] for a in anios_sel])  # shape: (N, 365)
N = mat.shape[0]

# Estadísticas por día seleccionado
idx = dia_sel - 1
valores_dia = mat[:, idx]
media = float(np.mean(valores_dia))
desv  = float(np.std(valores_dia))
mediana = float(np.median(valores_dia))
p10 = float(np.percentile(valores_dia, 10))
p25 = float(np.percentile(valores_dia, 25))
p75 = float(np.percentile(valores_dia, 75))
p90 = float(np.percentile(valores_dia, 90))
prob_sup = float(np.mean(valores_dia > umbral))

st.markdown(
    f"""
**Resultados para el día {dia_sel}:**  
- Promedio: **{media*100:.1f}%** (± {desv*100:.1f}%)  
- Mediana: **{mediana*100:.1f}%**  
- Percentiles: **P10={p10*100:.1f}%**, **P25={p25*100:.1f}%**, **P75={p75*100:.1f}%**, **P90={p90*100:.1f}%**  
- Probabilidad de superar **{umbral*100:.0f}%**: **{prob_sup*100:.1f}%**  
"""
)

# ============ DATAFRAME para Altair ============ #
dias = np.arange(1, 366)
df_lineas = []
for a in anios_sel:
    curva = curvas_dict[a]
    df_lineas.append(
        pd.DataFrame({"Día": dias, "Fracción": curva, "Serie": a})
    )
df_lineas = pd.concat(df_lineas, ignore_index=True)

# Curva promedio y bandas
curva_mean = mat.mean(axis=0)
curva_p10  = np.percentile(mat, 10, axis=0)
curva_p25  = np.percentile(mat, 25, axis=0)
curva_p75  = np.percentile(mat, 75, axis=0)
curva_p90  = np.percentile(mat, 90, axis=0)

df_stats = pd.DataFrame({
    "Día": dias,
    "mean": curva_mean,
    "p10":  curva_p10,
    "p25":  curva_p25,
    "p75":  curva_p75,
    "p90":  curva_p90
})

# ============ GRÁFICO PRINCIPAL ============ #
base = alt.Chart().encode(
    x=alt.X("Día:Q", title="Día del año")
)

band_p10_90 = base.mark_area(opacity=0.15).encode(
    y=alt.Y("p10:Q", title="Fracción acumulada"),
    y2="p90:Q",
).transform_lookup(
    lookup="Día",
    from_=alt.LookupData(df_stats, "Día", ["p10", "p90"])
)

band_iqr = base.mark_area(opacity=0.25).encode(
    y="p25:Q",
    y2="p75:Q",
).transform_lookup(
    lookup="Día",
    from_=alt.LookupData(df_stats, "Día", ["p25", "p75"])
)

line_mean = base.mark_line(size=3).encode(
    y="mean:Q",
    color=alt.value("#000")  # negro para destacar el promedio
).transform_lookup(
    lookup="Día",
    from_=alt.LookupData(df_stats, "Día", ["mean"])
)

line_years = alt.Chart(df_lineas).mark_line(opacity=0.6).encode(
    x="Día:Q",
    y=alt.Y("Fracción:Q", title="Fracción acumulada (0–1)", scale=alt.Scale(domain=[0, 1])),
    color=alt.Color("Serie:N", title="Año")
)

rule_day = alt.Chart(pd.DataFrame({"Día": [dia_sel]})).mark_rule(color="red", strokeDash=[4,4]).encode(x="Día:Q")

chart = (band_p10_90 + band_iqr + line_years + line_mean + rule_day).properties(height=420)
st.altair_chart(chart, use_container_width=True)

# ============ HISTOGRAMA (distribución del día seleccionado) ============ #
st.subheader("Distribución en el día seleccionado")
df_hist = pd.DataFrame({"Fracción": valores_dia, "Año": anios_sel})
hist = alt.Chart(df_hist).mark_bar().encode(
    x=alt.X("Fracción:Q", bin=alt.Bin(maxbins=20), title="Fracción acumulada en el día seleccionado"),
    y=alt.Y("count():Q", title="Frecuencia"),
    tooltip=[alt.Tooltip("count():Q", title="Frecuencia")]
).properties(height=220)
rule_thr = alt.Chart(pd.DataFrame({"Fracción": [umbral]})).mark_rule(color="red", strokeDash=[4,4]).encode(x="Fracción:Q")
st.altair_chart(hist + rule_thr, use_container_width=True)

# ============ TABLA Y DESCARGA ============ #
st.subheader("Valores por año en el día seleccionado")
df_resumen = pd.DataFrame({
    "Año": anios_sel,
    "Fracción_día": valores_dia,
    "Porcentaje_día": valores_dia * 100
}).sort_values("Año", key=lambda s: s.map(lambda x: int(x) if str(x).isdigit() else x))

st.dataframe(
    df_resumen.style.format({"Fracción_día": "{:.3f}", "Porcentaje_día": "{:.1f}"}),
    use_container_width=True
)

# CSV de descarga (resumen del día)
csv_buf = StringIO()
df_resumen.to_csv(csv_buf, index=False)
st.download_button(
    "⬇️ Descargar resumen (CSV)",
    data=csv_buf.getvalue(),
    file_name=f"resumen_dia_{dia_sel}.csv",
    mime="text/csv",
    key="dl_resumen"
)

# CSV con todas las curvas seleccionadas (365 filas por año)
st.subheader("Descarga de todas las curvas seleccionadas (normalizadas)")
df_todas = []
for a in anios_sel:
    df_todas.append(pd.DataFrame({"Año": a, "Día": dias, "Fracción": curvas_dict[a]}))
df_todas = pd.concat(df_todas, ignore_index=True)

csv_buf2 = StringIO()
df_todas.to_csv(csv_buf2, index=False)
st.download_button(
    "⬇️ Descargar curvas (CSV)",
    data=csv_buf2.getvalue(),
    file_name="curvas_normalizadas.csv",
    mime="text/csv",
    key="dl_curvas"
)

st.caption("Bandas sombreadas: IQR (P25–P75, más oscuro) y P10–P90 (más claro). La línea negra es el promedio.")

