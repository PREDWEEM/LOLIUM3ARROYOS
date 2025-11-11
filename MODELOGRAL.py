import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re

# Configuración inicial
st.set_page_config(page_title="Emergencia Acumulada Histórica", layout="centered")
st.title("Análisis histórico de emergencia acumulada")

# === Carga y preparación de datos ===
@st.cache_data
def cargar_datos_normalizados():
    archivos = ["2008.xlsx", "2009+.xlsx", "2011.xlsx", "2012.xlsx",
                "2013.xlsx", "2014.xlsx", "2023.xlsx", "2024.xlsx", "2025.xlsx"]
    curvas = []
    etiquetas = []
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

# === Cargar datos ===
curvas_historicas, etiquetas_annos = cargar_datos_normalizados()

if curvas_historicas.size == 0:
    st.error("No se encontraron datos históricos para procesar.")
    st.stop()

# === Selección del día ===
dia_seleccionado = st.slider(
    "Seleccione el día juliano",
    min_value=1, max_value=365, value=180, key="dia_slider"
)

idx = dia_seleccionado - 1
valores_dia = curvas_historicas[:, idx]
media = valores_dia.mean()
desviacion = valores_dia.std()
prob_supera_50 = (valores_dia > 0.5).mean()

media_pct = media * 100
desviacion_pct = desviacion * 100
prob_pct = prob_supera_50 * 100

st.markdown(f"**Resultados para el día {dia_seleccionado}:**")
st.write(f"- Emergencia acumulada promedio: **{media_pct:.1f}%** del total anual (± {desviacion_pct:.1f}%).")
st.write(f"- Probabilidad de superar 50% del total anual para este día: **{prob_pct:.1f}%**.")

# === Datos para gráfico ===
dias = np.arange(1, 366)
data_graf = []
for curva, anno in zip(curvas_historicas, etiquetas_annos):
    for d, valor in zip(dias, curva):
        data_graf.append({"Día": d, "Año": anno, "Fracción": valor})

curva_promedio = curvas_historicas.mean(axis=0)
for d, valor in zip(dias, curva_promedio):
    data_graf.append({"Día": d, "Año": "Promedio", "Fracción": valor})

df_graf = pd.DataFrame(data_graf)

# === Calcular curva de emergencia relativa semanal ===
# Diferencia semanal (promedio 7 días)
emergencia_diaria = np.diff(curva_promedio, prepend=0)
emergencia_semanal = np.convolve(emergencia_diaria, np.ones(7)/7, mode="same")

df_relativa = pd.DataFrame({
    "Día": dias,
    "Emergencia semanal": emergencia_semanal
})

# === Gráfico de curvas ===
lineas = alt.Chart(df_graf).mark_line().encode(
    x=alt.X("Día:Q", title="Día del año"),
    y=alt.Y("Fracción:Q", title="Fracción acumulada del año", scale=alt.Scale(domain=[0, 1])),
    color=alt.Color("Año:N", title="Año"),
    size=alt.condition(alt.datum.Año == "Promedio", alt.value(3), alt.value(1))
)

linea_vertical = alt.Chart(pd.DataFrame({"Día": [dia_seleccionado]})).mark_rule(
    color="red", strokeDash=[4, 4]
).encode(x="Día:Q")

# Nueva curva: Emergencia relativa semanal (naranja discontinua)
linea_relativa = alt.Chart(df_relativa).mark_line(
    color="orange", strokeDash=[6, 3]
).encode(
    x="Día:Q",
    y=alt.Y("Emergencia semanal:Q", title="Emergencia relativa semanal", axis=alt.Axis(titleColor="orange")),
).interactive()

# Combinar capas y establecer doble eje Y
grafico = alt.layer(lineas, linea_vertical, linea_relativa).resolve_scale(
    y="independent"
)

st.altair_chart(grafico, use_container_width=True)

st.caption("La línea naranja discontinua muestra la emergencia relativa semanal derivada de la curva promedio acumulada.")

