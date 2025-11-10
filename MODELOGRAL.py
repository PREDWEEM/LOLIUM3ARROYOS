import pandas as pd

# Lista de años a cargar (excluyendo 2010 y 2015)
anios = [2008, 2009, 2011, 2012, 2013, 2014, 2023, 2024, 2025]
datos_por_anio = {}

for anio in anios:
    nombre_archivo = f"{anio}.xlsx"
    # Cargar el Excel en un DataFrame, sin encabezado
    df = pd.read_excel(nombre_archivo, header=None, names=["Dia", "Valor"])
    datos_por_anio[anio] = df

for anio, df in datos_por_anio.items():
    # Suponiendo que la columna "Valor" es el incremento diario de emergencia
    df["Acumulado"] = df["Valor"].cumsum()
    # Normalizar para que el último día tenga valor 1.0 (100%)
    total_anual = df["Acumulado"].iloc[-1]
    if total_anual != 0:
        df["Acumulado"] = df["Acumulado"] / total_anual
# Construir un DataFrame combinado con índice = día, columnas = años
# Primero, crear base de días 1 a 365
dias = range(1, 366)
data_comb = pd.DataFrame(index=dias)

for anio, df in datos_por_anio.items():
    # Asegurar que el índice del DataFrame sea el día para fácil combinación
    df = df.set_index("Dia")
    # Si faltan días (por ejemplo 301-365), extender con último valor
    if df.index.max() < 365:
        ultimo_val = df["Acumulado"].iloc[-1]
        # Reindexar llenando los días faltantes con el último valor
        df = df.reindex(dias, fill_value=ultimo_val)
    else:
        df = df.reindex(dias, method='ffill')  # forward-fill por seguridad
    # Añadir la columna de este año al DataFrame combinado
    data_comb[anio] = df["Acumulado"]
    
# Calcular estadísticos por fila (cada fila es un día)
media_por_dia = data_comb.mean(axis=1)        # media de cada fila (día):contentReference[oaicite:2]{index=2}
std_por_dia   = data_comb.std(axis=1)         # desviación estándar de cada fila

# Ejemplo de slider en Streamlit (etiqueta en español)
dia = st.slider("Seleccione un día juliano", 1, 365, 1)
st.write("Día seleccionado:", dia)

# Obtener estadísticos para el día seleccionado
valor_medio = media_por_dia[dia]
incertidumbre = std_por_dia[dia]

# Mostrar en la interfaz (formato en porcentaje, por ejemplo)
st.metric(label="Emergencia acumulada promedio", value=f"{valor_medio:.2%}")
st.write(f"Rango de incertidumbre (±1σ): {valor_medio - incertidumbre:.2%} – {valor_medio + incertidumbre:.2%}")


import altair as alt

# Preparar datos en formato largo para Altair
df_plot = data_comb.reset_index().melt(id_vars="index", var_name="Año", value_name="Acumulado")
df_plot = df_plot.rename(columns={"index": "Dia"})
# Añadir la serie de promedio como si fuera un "año" más, por conveniencia
df_media = pd.DataFrame({
    "Dia": dias,
    "Año": ["Promedio"] * len(dias),
    "Acumulado": media_por_dia.values
})
df_plot = pd.concat([df_plot, df_media])


# Gráfica de líneas para cada año histórico (excepto promedio)
lineas_historicas = alt.Chart(df_plot[df_plot["Año"] != "Promedio"]).mark_line(
    color="gray",
    opacity=0.5
).encode(
    x=alt.X('Dia:Q', title='Día juliano'),
    y=alt.Y('Acumulado:Q', title='Emergencia acumulada (fracción)'),
    detail='Año:N'  # asegura que se trace una línea por cada año
)

# Línea de promedio
linea_prom = alt.Chart(df_plot[df_plot["Año"] == "Promedio"]).mark_line(
    color='red',
    strokeWidth=3  # más gruesa
).encode(
    x='Dia:Q',
    y='Acumulado:Q',
    tooltip=['Dia', alt.Tooltip('Acumulado:Q', format='.2%')]  # opcional: tooltip
)

# Línea vertical indicadora del día seleccionado
linea_vertical = alt.Chart(pd.DataFrame({'x': [dia]})).mark_rule(
    color='blue',
    strokeDash=[5,5]  # línea punteada por ejemplo
).encode(
    x='x:Q'
)

# Combinar en un solo gráfico
grafico = (lineas_historicas + linea_prom + linea_vertical).properties(width=700, height=400)
st.altair_chart(grafico, use_container_width=True)

umbral = 0.5  # 50%
# Extraer valores acumulados de todos los años en el día seleccionado
valores_en_dia = data_comb.loc[dia, :]  # serie con valores de cada año en ese día
# Calcular proporción de años por encima y por debajo del umbral
por_encima = (valores_en_dia >= umbral).mean()  # esto devuelve k/N porque True=1, False=0
por_debajo = (valores_en_dia < umbral).mean()





