
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re

# =============================================
# 🌾 Análisis histórico de emergencia acumulada
# =============================================

st.set_page_config(page_title="Emergencia Acumulada Histórica", layout="centered")
st.title("Análisis histórico de emergencia acumulada y emergencia relativa semanal")

# === FUNCIÓN DE CARGA DE DATOS ===
@st.cache_data
def cargar_datos_normalizados():
    archivos = ["2008.xlsx", "2009.xlsx", "2011.xlsx", "2012.xlsx",
                "2013.xlsx", "2014.xlsx", "2023.xlsx", "2024.xlsx", "2025.xlsx", "2015.xlsx"]
    curvas, etiquetas = [], []
    for archivo in archivos:
        try:
            datos = pd.read_excel(archivo, header=None)
        except Exception as e:
            st.error(f"Error al leer {archivo}: {e}")
            continue
        if datos.empty:
            continue
        dias = datos.iloc[:, 0].values
        valores = datos.iloc[:, 1].values
        valores_diarios = np.zeros(365)
        for d, val in zip(dias, valores):
            dia_idx = int(d) - 1
            if 0 <= dia_idx < 365:
                valores_diarios[dia_idx] = val
        curva_acumulada = np.cumsum(valores_diarios)
        valor_final = curva_acumulada[-1]
        curva_norm = curva_acumulada / valor_final if valor_final != 0 else curva_acumulada
        anno = re.match(r"^(\d+)", archivo)
        etiqueta_anno = anno.group(1) if anno else archivo
        curvas.append(curva_norm)
        etiquetas.append(etiqueta_anno)
    curvas = np.array(curvas)
    return curvas, etiquetas

# === CARGA DE DATOS ===
curvas_historicas, etiquetas_annos = cargar_datos_normalizados()
if curvas_historicas.size == 0:
    st.error("No se encontraron datos históricos para procesar.")
    st.stop()

# === SLIDER DE DÍA JULIANO ===
dia_seleccionado = st.slider(
    "Seleccione el día juliano",
    min_value=1, max_value=365, value=180, key="dia_slider"
)

# === ESTADÍSTICAS ===
idx = dia_seleccionado - 1
valores_dia = curvas_historicas[:, idx]
media = valores_dia.mean()
desviacion = valores_dia.std()
prob_supera_50 = (valores_dia > 0.5).mean()

st.markdown(f"**Resultados para el día {dia_seleccionado}:**")
st.write(f"- Emergencia acumulada promedio: **{media*100:.1f}%** (± {desviacion*100:.1f}%).")
st.write(f"- Probabilidad de superar 50% del total anual: **{prob_supera_50*100:.1f}%**.")

# === PREPARAR DATOS ===
dias = np.arange(1, 366)
data_graf = []
for curva, anno in zip(curvas_historicas, etiquetas_annos):
    for d, valor in zip(dias, curva):
        data_graf.append({"Día": d, "Año": anno, "Fracción acumulada": valor})

curva_promedio = curvas_historicas.mean(axis=0)
for d, valor in zip(dias, curva_promedio):
    data_graf.append({"Día": d, "Año": "Promedio", "Fracción acumulada": valor})

df_graf = pd.DataFrame(data_graf)

# === CÁLCULO DE EMERGENCIA RELATIVA SEMANAL ===
emergencia_diaria = np.diff(curva_promedio, prepend=0)
emergencia_relativa = np.convolve(emergencia_diaria, np.ones(7)/7, mode="same")

df_relativa = pd.DataFrame({
    "Día": dias,
    "Emergencia relativa semanal": emergencia_relativa
})

# === GRÁFICO DE EMERGENCIA ACUMULADA (EJE IZQ) ===
base_acum = alt.Chart(df_graf)

curvas = base_acum.transform_filter(
    alt.datum["Año"] != "Promedio"
).mark_line(opacity=0.5).encode(
    x=alt.X("Día:Q", title="Día del año"),
    y=alt.Y("Fracción acumulada:Q",
            title="Fracción acumulada (0–1)",
            axis=alt.Axis(titleColor="steelblue")),
    color=alt.Color("Año:N", title="Año")
)

promedio = base_acum.transform_filter(
    alt.datum["Año"] == "Promedio"
).mark_line(
    color="black", strokeWidth=3
).encode(
    x="Día:Q",
    y="Fracción acumulada:Q"
)

linea_vertical = alt.Chart(pd.DataFrame({"Día": [dia_seleccionado]})).mark_rule(
    color="red", strokeDash=[4, 4]
).encode(x="Día:Q")

# === GRÁFICO DE EMERGENCIA RELATIVA (EJE DER) ===
base_rel = alt.Chart(df_relativa)

area_relativa = base_rel.mark_area(
    color="orange", opacity=0.3
).encode(
    x="Día:Q",
    y=alt.Y("Emergencia relativa semanal:Q",
            axis=alt.Axis(title="Emergencia relativa semanal (0–1)",
                         titleColor="orange"))
)

linea_relativa = base_rel.mark_line(
    color="orange", strokeDash=[6, 3], strokeWidth=2
).encode(
    x="Día:Q",
    y="Emergencia relativa semanal:Q"
)

# === COMBINAR COMO DOS GRÁFICOS INDEPENDIENTES CON EJE SECUNDARIO ===
# Altair no renderiza dos ejes Y en la misma capa, así que usamos un overlay con 'layer' y configure_view(clip=False)
grafico_acum = (curvas + promedio + linea_vertical).properties(height=420)
grafico_rel = (area_relativa + linea_relativa).properties(height=420)

# Overlay con ejes independientes
grafico_final = alt.layer(
    grafico_acum,
    grafico_rel
).resolve_scale(
    y='independent'
).configure_view(clip=False).properties(
    title="Curvas de emergencia acumulada (años históricos) y emergencia relativa semanal (promedio histórico)"
)

# === MOSTRAR ===
st.altair_chart(grafico_final, use_container_width=True)

# === LEYENDA ===
st.caption("""
🟢 **Curvas históricas:** cada año individual.  
⚫ **Curva negra gruesa:** promedio histórico acumulado.  
🟧 **Área naranja:** emergencia relativa semanal (promedio, suavizada 7 días).  
🟧 **Línea discontinua:** tendencia de emergencia relativa semanal.  
🔴 **Línea roja punteada:** día juliano seleccionado.
""")
