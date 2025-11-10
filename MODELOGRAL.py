import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# Título de la aplicación
st.title("Visualizador de Emergencia Acumulada Histórica")
st.write("Seleccione un día del año para ver el promedio histórico de la emergencia acumulada, su incertidumbre y la probabilidad de haber superado el 50% de la emergencia anual para esa fecha. También se muestra la comparación de las curvas históricas normalizadas.")

# --- Carga y preparación de datos históricos ---

# Años disponibles (según archivos proporcionados), excluyendo 2010 y 2015
years = list(range(2008, 2016)) + [2023, 2024, 2025]  # 2008-2015, luego 2023-2025
exclude_years = [2010, 2015]
years = [y for y in years if y not in exclude_years]

data_by_year = {}  # Diccionario para almacenar la curva acumulada normalizada por año

for year in years:
    # Nombre de archivo (caso especial: 2009 está nombrado como "2009+.xlsx")
    file_name = "2009+.xlsx" if year == 2009 else f"{year}.xlsx"
    try:
        df = pd.read_excel(file_name, header=None, names=["day", "value"])
    except Exception as e:
        # Si el archivo no existe o hay error, continuar con el siguiente año
        print(f"Error al leer datos del año {year}: {e}")
        continue
    
    # Asegurar que los días estén en orden ascendente
    df = df.sort_values("day").reset_index(drop=True)
    
    # Rellenar días faltantes dentro del rango disponible con 0 (si hay brechas internas)
    # Esto cubre el caso de que la serie no tenga algunos días (se asume valor 0 esos días).
    all_days = pd.DataFrame({"day": range(1, int(df["day"].max()) + 1)})
    df = pd.merge(all_days, df, on="day", how="left")
    df["value"] = df["value"].fillna(0.0)
    
    # Extender hasta el día 365 si el último día presente es menor a 365
    last_day = int(df["day"].max())
    if last_day < 365:
        # Obtener el valor acumulado final registrado para mantenerlo constante después
        # (Notar: si el último valor es 0, la serie se quedará en 0 en adelante)
        last_known_value = df.loc[df["day"] == last_day, "value"].iloc[0]
        # Crear dataframe de los días faltantes con value = 0 (sin nuevos eventos después del último día registrado)
        extra_days = pd.DataFrame({
            "day": range(last_day + 1, 366),
            "value": 0.0
        })
        df = pd.concat([df, extra_days], ignore_index=True)
    
    # Calcular la curva de emergencia acumulada (suma acumulada día a día)
    df["cumulative"] = df["value"].cumsum()
    
    # Normalizar la curva acumulada para que el último valor del año sea 1.0 (100%)
    total = df["cumulative"].iloc[-1]
    if total == 0:
        # Si el total anual es 0 (sin emergencias en todo el año), se omite este año de los cálculos
        # (No es posible normalizar una serie cuyo total es cero).
        continue
    df["normalized"] = df["cumulative"] / total
    
    # Guardar solo las columnas necesarias (día y valor normalizado)
    data_by_year[year] = df[["day", "normalized"]]

# Verificar que tengamos datos históricos cargados correctamente
if not data_by_year:
    st.error("No se encontraron datos históricos de emergencias para procesar.")
    st.stop()

# Combinar los datos de todos los años en una sola estructura para cálculos estadísticos
# Creamos un DataFrame con índice de día (1 a 365) y columnas por año, con valores normalizados.
all_days_index = pd.DataFrame({"day": range(1, 366)})
combined = all_days_index.copy()
for year, df in data_by_year.items():
    # Hacemos merge para alinear por día, en caso algún año no tenga todos los días (ya rellenamos hasta 365).
    combined = pd.merge(combined, df, on="day", how="left", suffixes=("", f"_{year}"))
    # La columna recién fusionada se llamará 'normalized' o 'normalized_{year}' según colisiones
    # Para simplificar, la renombramos al año correspondiente.
    col_name = "normalized"
    if col_name in combined.columns:
        # primera combinación, renombrar a año
        combined.rename(columns={col_name: str(year)}, inplace=True)
    else:
        # en combinaciones subsiguientes, la columna vendrá nombrada como normalized_year
        combined.rename(columns={f"normalized_{year}": str(year)}, inplace=True)

# Ahora 'combined' tiene columna "day" y columnas para cada año con valores normalizados.
combined = combined.sort_values("day").reset_index(drop=True).fillna(0.0)  # rellenar NaN (si algún año faltaba algún día, debería ser 0 tras normalización)

combined.set_index("day", inplace=True)  # usar día como índice para facilitar cálculos por día

# Cálculo de estadísticos históricos por día (basado en columnas de años)
mean_by_day = combined.mean(axis=1) * 100  # promedio (%)
std_by_day = combined.std(axis=1) * 100    # desviación estándar (%)
# Probabilidad de superar 50% (0.5 normalizado): porcentaje de años con valor >= 50%
prob_over_half = (combined >= 0.5).mean(axis=1) * 100  # .mean(axis=1) da la fracción de años True

# --- Interfaz de selección de día y visualización de resultados ---

# Slider para seleccionar el día juliano (1 a 365)
selected_day = st.slider("Selecciona el día juliano", min_value=1, max_value=365, value=180)

# Extraer los valores correspondientes al día seleccionado
day_index = selected_day  # (nuestro índice del DataFrame es el número de día)
avg_value = mean_by_day.loc[day_index]
std_value = std_by_day.loc[day_index]
prob_value = prob_over_half.loc[day_index]

# Mostrar las métricas calculadas en tres columnas para alinearlas en una fila
col1, col2, col3 = st.columns(3)
col1.metric("Promedio acumulado", f"{avg_value:.1f}%", help="Valor promedio histórico acumulado hasta el día seleccionado")
col2.metric("Desviación estándar", f"±{std_value:.1f}%", help="Desviación estándar histórica hasta ese día")
col3.metric("Probabilidad > 50%", f"{prob_value:.0f}%", help="Frecuencia histórica de años que superaron 50% del total anual a este día")

# --- Gráfica comparativa de curvas históricas y promedio ---

# Preparar datos en formato largo para Altair (día, año, valor_normalizado)
plot_data = []
for year, df in data_by_year.items():
    plot_df = df.copy()
    plot_df["year"] = str(year)  # convertir año a string para usar como categoría
    plot_data.append(plot_df[["day", "year", "normalized"]])
plot_data = pd.concat(plot_data, ignore_index=True)

# Datos de la curva promedio (día, valor promedio) - usamos la serie mean_by_day calculada
avg_curve = pd.DataFrame({
    "day": mean_by_day.index.values,
    "normalized": mean_by_day.values / 100.0  # dividir por 100 para llevar de % a fracción 0-1
})
# Añadir una etiqueta para la serie promedio (podríamos usar 'year': 'Promedio' para incluir en mismo data, 
# pero la graficamos por separado para estilo diferenciado)

# Crear gráfica Altair
base_lines = alt.Chart(plot_data).mark_line(color='gray', opacity=0.5).encode(
    x=alt.X('day:Q', title='Día del año'),
    y=alt.Y('normalized:Q', title='Porcentaje de la emergencia anual', axis=alt.Axis(format='%')),
    detail='year:N',  # trazar una línea por cada año (agrupadas por categoría 'year')
    order='day:Q'     # asegurar que los puntos de cada línea se conecten en orden de día ascendente
)

# Curva promedio destacada (capa separada, color distinto y trazo más grueso)
avg_line = alt.Chart(avg_curve).mark_line(color='red', size=3).encode(
    x='day:Q',
    y='normalized:Q'
)

# Línea vertical indicando el día seleccionado
vertical_rule = alt.Chart(pd.DataFrame({'day': [selected_day]})).mark_rule(color='red', strokeDash=[4,2]).encode(
    x='day:Q'
)
# (Al usar mark_rule solo con coordenada X, Altair dibuja una línea vertical que abarca toda la altura del gráfico):contentReference[oaicite:6]{index=6}

# Combinar las capas en una sola visualización
chart = alt.layer(base_lines, avg_line, vertical_rule).properties(width='container', height=400)

st.altair_chart(chart, use_container_width=True)



