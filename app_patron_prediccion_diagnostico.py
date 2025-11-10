import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.title("Análisis de Patrones de Emergencia Relativa por Día Juliano")

# 1. Cargar múltiples archivos Excel (.xlsx)
archivos_subidos = st.file_uploader(
    "Cargar archivos Excel con series de emergencia relativa:", 
    type=["xlsx"], accept_multiple_files=True
)
if not archivos_subidos:  # Si no se subió nada, mostrar aviso
    st.info("🔺 Por favor, cargue uno o más archivos Excel para comenzar el análisis.")
    st.stop()

# Listas para acumular datos combinados y resultados resumen
datos_combinados = []
resultados = []

# 2. Procesar cada archivo subido
for archivo in archivos_subidos:
    # Leer datos de Día y Emergencia
    try:
        df = pd.read_excel(archivo, header=None, names=["Dia", "Emergencia"])
    except Exception as e:
        st.error(f"❌ No se pudo leer el archivo {archivo.name}: {e}")
        continue  # Saltar a siguiente archivo si hay error

    # Obtener nombre identificador (por ejemplo, año desde el nombre de archivo)
    nombre = archivo.name
    if nombre.lower().endswith(".xlsx"):
        nombre = nombre[:-5]  # quitar ".xlsx"
    serie_id = nombre  # podría ser año

    # 3. Calcular AUC total y parcial hasta día 121
    auc_total = np.trapz(df["Emergencia"], df["Dia"])
    auc_121   = np.trapz(df[df["Dia"] <= 121]["Emergencia"], df[df["Dia"] <= 121]["Dia"])
    proporcion = auc_121 / auc_total if auc_total != 0 else 0.0

    # 4. Clasificar según proporción (50% umbral)
    if proporcion >= 0.5:
        clasif = "CONCENTRADO"
    else:
        clasif = "EXTENDIDO"

    # Guardar resultados en lista
    resultados.append({
        "Serie (Archivo)": serie_id,
        "AUC_total": auc_total,
        "AUC_<=121": auc_121,
        "Proporción_<=121": proporcion,
        "Clasificación": clasif
    })

    # Preparar datos para gráfico, añadiendo columnas de identificador y clasificación
    df["Serie"] = serie_id
    df["Clasificacion"] = clasif
    datos_combinados.append(df)

# Si no se obtuvo ningún resultado (por ejemplo, todos los archivos fallaron al leer)
if not resultados:
    st.error("No se obtuvieron datos de los archivos proporcionados.")
    st.stop()

# 5. Concatenar datos de todas las series para graficar
datos_comb_df = pd.concat(datos_combinados, ignore_index=True)

# 6. Crear gráfico de líneas múltiples con Altair
chart = alt.Chart(datos_comb_df).mark_line().encode(
    x=alt.X('Dia:Q', title='Día Juliano'),
    y=alt.Y('Emergencia:Q', title='Emergencia relativa'),
    color=alt.Color('Serie:N', title='Serie'),
    strokeDash=alt.StrokeDash('Clasificacion:N', title='Patrón')
).properties(width=700, height=400)
st.altair_chart(chart, use_container_width=True)

# 7. Mostrar tabla de resultados
st.write("## Resultados por serie")
resultados_df = pd.DataFrame(resultados)
# Redondear proporción a 2 decimales para legibilidad
resultados_df["Proporción_<=121"] = resultados_df["Proporción_<=121"].round(2)
st.table(resultados_df)

# 8. Botón de descarga de CSV
csv_data = resultados_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Descargar resumen CSV",
    data=csv_data,
    file_name="resumen_emergencia.csv",
    mime="text/csv"
)
