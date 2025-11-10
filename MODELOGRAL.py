import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re

# Configuración inicial de la página (opcional)
st.set_page_config(page_title="Emergencia Acumulada Histórica", layout="centered")

# Título de la aplicación
st.title("Análisis histórico de emergencia acumulada")

# Cargar y preparar datos históricos, excluyendo 2010 y 2015
@st.cache_data  # Cacheamos la carga para no releer archivos en cada interacción
def cargar_datos_normalizados():
    archivos = ["2008.xlsx", "2009+.xlsx", "2011.xlsx", "2012.xlsx", 
                "2013.xlsx", "2014.xlsx", "2023.xlsx", "2024.xlsx", "2025.xlsx"]
    curvas = []    # Lista para almacenar las curvas acumuladas normalizadas de cada año
    etiquetas = [] # Lista para almacenar el nombre/etiqueta de cada curva (año)
    for archivo in archivos:
        # Leer el archivo Excel (sin encabezado, asumiendo dos columnas: día y valor diario)
        try:
            datos = pd.read_excel(archivo, header=None)
        except Exception as e:
            st.error(f"Error al leer {archivo}: {e}")
            continue
        if datos.empty:
            continue
        dias = datos.iloc[:, 0].values    # Columna de días (julianos)
        valores = datos.iloc[:, 1].values # Columna de valores diarios de emergencia
        # Crear vector de valores diarios inicializado en 0 para 365 días
        valores_diarios = np.zeros(365)
        # Rellenar el vector con los valores diarios disponibles
        for d, val in zip(dias, valores):
            dia_idx = int(d) - 1
            if 0 <= dia_idx < 365:
                valores_diarios[dia_idx] = val
        # Asegurar extensión hasta día 365: si faltan días, los valores_diarios ya son 0 (sin incremento adicional)
        # Calcular la curva acumulada sumando los valores diarios día a día
        curva_acumulada = np.cumsum(valores_diarios)
        # Extender la curva con el último valor conocido para días faltantes (ya implícito al no haber incrementos después del último día registrado)
        # Normalizar la curva para que el valor del día 365 sea 1 (100%)
        valor_final = curva_acumulada[-1]  # valor acumulado al día 365 (después de extender con ceros)
        if valor_final == 0:
            curva_norm = curva_acumulada  # evitar división por cero en caso extremo de año sin datos
        else:
            curva_norm = curva_acumulada / valor_final
        # Obtener etiqueta del año a partir del nombre de archivo (ej. "2009+.xlsx" -> "2009")
        anno = re.match(r'^(\d+)', archivo)
        etiqueta_anno = anno.group(1) if anno else archivo
        curvas.append(curva_norm)
        etiquetas.append(etiqueta_anno)
    # Convertir a arreglo numpy para facilitar cálculos estadísticos
    curvas = np.array(curvas)
    return curvas, etiquetas

# Cargar datos normalizados (curvas históricas)
curvas_historicas, etiquetas_annos = cargar_datos_normalizados()

# Verificar que se cargaron datos
if curvas_historicas.size == 0:
    st.error("No se encontraron datos históricos para procesar.")
    st.stop()

# Slider para selección de día juliano
dia_seleccionado = st.slider(
    "Seleccione el día juliano", 
    min_value=1, max_value=365, value=180, 
    key="dia_slider"
)

# Calcular estadísticas para el día seleccionado
# Índice del array (día juliano 1 corresponde al índice 0)
idx = dia_seleccionado - 1
valores_dia = curvas_historicas[:, idx]          # valores de todas las curvas en ese día
media = valores_dia.mean()
desviacion = valores_dia.std()
prob_supera_50 = (valores_dia > 0.5).mean()      # fracción de años con valor > 0.5 (50%)

# Mostrar resultados numéricos con formato adecuado (en porcentaje)
media_pct = media * 100
desviacion_pct = desviacion * 100
prob_pct = prob_supera_50 * 100

st.markdown(f"**Resultados para el día {dia_seleccionado}:**")
st.write(f"- Emergencia acumulada promedio: **{media_pct:.1f}%** del total anual (± {desviacion_pct:.1f}%).")
st.write(f"- Probabilidad de superar 50% del total anual para este día: **{prob_pct:.1f}%**.")

# Preparar datos para la gráfica Altair
# Construimos un DataFrame con columnas: Day, Year, Value (fracción acumulada)
dias = np.arange(1, 366)
data_graf = []
for curva, anno in zip(curvas_historicas, etiquetas_annos):
    for d, valor in zip(dias, curva):
        data_graf.append({"Día": d, "Año": anno, "Fracción": valor})
# Agregar la curva promedio al DataFrame
curva_promedio = curvas_historicas.mean(axis=0)
for d, valor in zip(dias, curva_promedio):
    data_graf.append({"Día": d, "Año": "Promedio", "Fracción": valor})

df_graf = pd.DataFrame(data_graf)

# Crear gráfico de líneas con Altair
lineas = alt.Chart(df_graf).mark_line().encode(
    x=alt.X("Día:Q", title="Día del año"),
    y=alt.Y("Fracción:Q", title="Fracción acumulada del año", scale=alt.Scale(domain=[0, 1])),
    color=alt.Color("Año:N", title="Año"),
    # Hacer la línea de promedio más gruesa usando una condición
    size=alt.condition(alt.datum.Año == "Promedio", alt.value(3), alt.value(1))
)

# Crear línea vertical en el día seleccionado usando mark_rule (línea roja punteada)
linea_vertical = alt.Chart(pd.DataFrame({"Día": [dia_seleccionado]})).mark_rule(color="red", strokeDash=[4, 4]).encode(
    x="Día:Q"
)

# Combinar capas de líneas históricas y promedio con la línea vertical
grafico = lineas + linea_vertical

# Mostrar gráfico en la aplicación
st.altair_chart(grafico, use_container_width=True)



