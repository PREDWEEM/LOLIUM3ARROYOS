import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.title("Análisis de series de emergencia relativa")

st.markdown("""
Suba uno o varios archivos Excel, cada uno con una serie de emergencia relativa (día juliano en la primera columna y valor en la segunda). 
La aplicación detectará automáticamente el tipo de serie (diaria relativa vs. semanal/acumulada), 
generará la curva acumulada normalizada, calculará métricas (AUC total, AUC hasta día 121, proporción al día 121) 
y clasificará cada serie como **CONCENTRADO** o **DISPERSO** según el umbral seleccionado.
""")

# Paso 1: Carga de archivos Excel (múltiples)
archivos = st.file_uploader("Cargar archivos Excel", type=["xlsx"], accept_multiple_files=True)
umbral = st.slider("Umbral para clasificación (CONCENTRADO vs DISPERSO)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if archivos:
    resultados = []  # lista para guardar métricas de cada archivo
    curvas_data = []  # para datos de curvas para graficar (formato largo)
    
    for archivo in archivos:
        nombre = archivo.name
        # Leer el Excel. Asumimos dos columnas: dia juliano y valor
        try:
            df = pd.read_excel(archivo, header=None)
        except Exception as e:
            st.error(f"❌ No se pudo leer {nombre}: {e}")
            continue
        
        # Asegurar que tenemos dos columnas
        if df.shape[1] < 2:
            st.warning(f"El archivo {nombre} no tiene dos columnas, se ignora.")
            continue
        
        # Renombrar columnas para claridad
        df = df.iloc[:, :2]  # tomar solo las dos primeras columnas si hubiera extras
        df.columns = ["dia", "valor"]
        
        # Paso 2: Determinar tipo de serie
        suma_valores = df["valor"].sum(skipna=True)
        es_diaria = False
        # criterio: nombre del archivo contiene '2023', '2024' o '2025' OR suma ~ 1
        if any(str(year) in nombre for year in [2023, 2024, 2025]):
            es_diaria = True
        if suma_valores >= 0.95 and suma_valores <= 1.05:
            # La suma está aproximadamente en 1 (±5%)
            es_diaria = True
        
        if es_diaria:
            # Serie diaria relativa: aplicar cumsum para obtener acumulada
            df_sorted = df.sort_values("dia")
            df_sorted["acumulado"] = df_sorted["valor"].cumsum()
        else:
            # No es diaria relativa (serie semanal o valores absolutos)
            # Verificar si la serie ya es acumulada (monótona no decreciente)
            df_sorted = df.sort_values("dia")
            valores = df_sorted["valor"].values
            # Chequear monotonicidad (permitiendo tolerancia pequeña para flotantes)
            diffs = np.diff(valores)
            if np.all(diffs >= -1e-9):  # si todas las diferencias son >= 0 (monótono creciente)
                # Ya es acumulada
                df_sorted["acumulado"] = df_sorted["valor"]
            else:
                # Es semanal/diaria absoluta: aplicar cumsum para obtener acumulado
                df_sorted["acumulado"] = df_sorted["valor"].cumsum()
        
        # Paso 3: Interpolar a diario (si la secuencia de días tiene huecos)
        # Asegurar que el día 1 está presente; si falta, agregar día 1 con acumulado 0
        if df_sorted["dia"].iloc[0] > 1:
            df_sorted = pd.concat([
                pd.DataFrame({"dia": [1], "acumulado": [0.0]}),
                df_sorted
            ], ignore_index=True)
        # Crear índice completo de días hasta el último día o 365 (lo que sea mayor)
        ultimo_dia = int(df_sorted["dia"].max())
        # Consideramos hasta día 365 por seguridad (año completo)
        if ultimo_dia < 365:
            ultimo_dia = 365
        # Reindexar la serie acumulada con todos los días hasta ultimo_dia
        df_indexed = df_sorted.set_index("dia")["acumulado"]
        df_indexed = df_indexed.reindex(range(1, ultimo_dia+1))
        # Interpolar valores faltantes linealmente entre datos existentes
        df_interpolated = df_indexed.interpolate(method='linear')
        # Rellenar cualquier valor posterior al último dato conocido con el último valor (ffill)
        df_interpolated = df_interpolated.ffill()
        # Rellenar valores antes del primer dato conocido (si aplica) con 0 (bfill)
        df_interpolated = df_interpolated.bfill().fillna(0)
        
        # Paso 4: Normalizar al rango [0,1]
        max_val = df_interpolated.iloc[-1]
        if max_val == 0:
            # Si la serie está toda en cero (caso extremo sin eventos), saltamos
            norm_series = df_interpolated
        else:
            norm_series = df_interpolated / max_val
        
        # Asegurarse que la serie llega hasta día 365 (inclusive)
        if norm_series.index.max() < 365:
            # Extender hasta 365 con el último valor (que sería 1 si max_val>0)
            last_val = norm_series.iloc[-1]
            norm_series = norm_series.reindex(range(1, 366), fill_value=last_val)
        
        # Paso 5: Calcular AUC total, AUC hasta día 121, proporción día 121
        y_values = norm_series.values  # valores de la curva normalizada
        x_days = norm_series.index.values
        # Area total bajo la curva usando método trapezoidal
        auc_total = float(np.trapz(y_values, x_days))
        # Si el día 121 excede los datos, asegurarse de no pasarse
        max_day = norm_series.index.max()
        day121 = 121 if 121 <= max_day else max_day
        auc_121 = float(np.trapz(norm_series.loc[1:day121].values, norm_series.loc[1:day121].index.values))
        # Valor acumulado al día 121
        # Si la serie no llega al 121 (muy improbable), tomar último
        if day121 in norm_series.index:
            prop_121 = float(norm_series.loc[day121])
        else:
            prop_121 = float(norm_series.iloc[-1])
        # Clasificación según umbral
        clasificacion = "CONCENTRADO" if prop_121 >= umbral else "DISPERSO"
        
        # Guardar resultados para la tabla
        resultados.append({
            "Archivo": nombre,
            "AUC_total": auc_total,
            "AUC_dia121": auc_121,
            "Prop_121": prop_121,
            "Clasificación": clasificacion
        })
        
        # Preparar datos para graficar (acumulada normalizada)
        # Usamos nombre del archivo sin extensión como etiqueta de serie
        serie_label = nombre.rsplit('.', 1)[0]
        df_plot = pd.DataFrame({
            "dia": norm_series.index.values,
            "acumulado_norm": norm_series.values,
            "serie": serie_label
        })
        curvas_data.append(df_plot)
    
    if resultados:
        # Combinar datos de todas las curvas para graficar
        curvas_df = pd.concat(curvas_data, ignore_index=True)
        # Paso 6: Gráfico interactivo de curvas acumuladas
        chart = alt.Chart(curvas_df).mark_line().encode(
            x=alt.X("dia:Q", title="Día del año"),
            y=alt.Y("acumulado_norm:Q", title="Fracción acumulada"),
            color=alt.Color("serie:N", title="Serie (Archivo/Año)"),
            tooltip=["serie:N", "dia:Q", alt.Tooltip("acumulado_norm:Q", format=".2f")]
        ).properties(title="Curvas acumuladas normalizadas por año", width=700, height=400)
        st.altair_chart(chart, use_container_width=True)
        
        # Paso 7: Mostrar tabla de resultados
        resultados_df = pd.DataFrame(resultados)
        # Formato: limitar a 3 decimales las columnas numéricas para visualización
        resultados_df_display = resultados_df.copy()
        resultados_df_display["AUC_total"] = resultados_df_display["AUC_total"].map(lambda x: f"{x:.2f}")
        resultados_df_display["AUC_dia121"] = resultados_df_display["AUC_dia121"].map(lambda x: f"{x:.2f}")
        resultados_df_display["Prop_121"] = resultados_df_display["Prop_121"].map(lambda x: f"{x:.3f}")
        st.subheader("Resultados por archivo")
        st.dataframe(resultados_df_display, use_container_width=True)
        
        # Paso 8: Botón de descarga de resultados en CSV
        csv_data = resultados_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar tabla CSV", data=csv_data, file_name="resultados_emergencia.csv", mime="text/csv")
    else:
        st.warning("No se generaron resultados. Verifique que los archivos tengan datos válidos.")
else:
    st.info("⬆ Por favor, cargue uno o varios archivos Excel para comenzar el análisis.")
