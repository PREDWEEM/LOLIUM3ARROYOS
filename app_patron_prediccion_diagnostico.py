import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import streamlit as st

# Título de la aplicación
st.title("Análisis de Patrones de Emergencia Relativa")

# Instrucciones al usuario
st.markdown(
    "Seleccione uno o varios archivos Excel con datos de emergencia relativa. "
    "El script unificará los formatos (diario/semanal), calculará el área bajo la curva (AUC) y clasificará el patrón de emergencia."
)

# Carga de archivos Excel
archivos = st.file_uploader("Suba los archivos Excel de datos", type=["xlsx"], accept_multiple_files=True)

# Lista para recopilar resultados
resultados = []

if archivos:
    for archivo in archivos:
        # Leer datos del Excel (asumiendo dos columnas: Día y Valor de emergencia relativa)
        try:
            df = pd.read_excel(archivo, header=None)
        except Exception as e:
            st.error(f"Error al leer {archivo.name}: {e}")
            continue

        # Asignar nombres de columnas para claridad
        df.columns = ["Dia", "Valor"]
        # Si la primera fila parece ser cabecera textual, intentar recargar sin header
        if isinstance(df.loc[0, "Dia"], str) or isinstance(df.loc[0, "Valor"], str):
            df = pd.read_excel(archivo, header=0)
            df.columns = ["Dia", "Valor"]

        # Extraer arrays de días y valores
        dias = df["Dia"].to_numpy()
        valores = df["Valor"].to_numpy()

        # Asegurarse de que el día 1 esté incluido para iniciar en 0 acumulado
        if dias[0] != 1:
            # Insertar día 1 con valor 0 al inicio si falta
            dias = np.insert(dias, 0, 1)
            valores = np.insert(valores, 0, 0.0)

        # Identificar formato de los datos (diario vs semanal/porcentual)
        formato_diario = valores.sum() < 2  # True si la sumatoria es menor a 2 (fracción diaria)
        # Convertir a fracción si son porcentajes (valores mayores a 1 sugieren porcentajes)
        if not formato_diario and np.nanmax(valores) > 1:
            valores = valores / 100.0  # convertir porcentajes a fracción (0-1)

        # Obtener serie acumulada diaria
        if formato_diario:
            # Datos diarios de fracción: sumamos acumulativamente
            acumulado = np.nancumsum(valores)
        else:
            # Datos semanales/porcentuales:
            # Verificar si la serie es acumulada (monótona) o incremental (no monótona)
            if np.all(np.diff(valores[~np.isnan(valores)]) >= 0):
                # Monótono no decreciente: asumimos que ya es acumulado relativo
                acumulado_puntos = valores.copy()
            else:
                # No monótono: asumimos que son incrementos relativos -> acumulamos
                acumulado_puntos = np.nancumsum(valores)
            # Interpolación a días si los puntos no son diarios
            intervalo_dias = np.diff(dias)
            if np.nanmin(intervalo_dias) > 1:
                # Crear función de interpolación lineal sobre los puntos acumulados
                f_interp = interp1d(dias, acumulado_puntos, kind="linear", fill_value="extrapolate")
                # Generar rango diario desde el primer hasta el último día registrado
                dias_interp = np.arange(int(dias.min()), int(dias.max()) + 1)
                acumulado = f_interp(dias_interp)
                dias = dias_interp  # actualizar días a diarios
            else:
                # Si ya hay datos diarios (posiblemente ya interpolados previamente)
                acumulado = acumulado_puntos

        # Normalizar la curva acumulada a escala 0-1 dividiendo por el total anual (máximo acumulado)
        max_acumulado = np.nanmax(acumulado)
        if max_acumulado == 0 or np.isnan(max_acumulado):
            # Evitar división por cero; si no hay datos, saltar serie
            st.warning(f"La serie {archivo.name} no contiene datos válidos de emergencia.")
            continue
        acumulado_normalizado = acumulado / max_acumulado

        # Asegurarse de que el array de días y acumulado_normalizado tengan el mismo tamaño (por seguridad)
        if len(dias) != len(acumulado_normalizado):
            dias = np.linspace(dias.min(), dias.max(), num=len(acumulado_normalizado))
            # (En la mayoría de casos no será necesario este ajuste si la interpolación se realizó correctamente)

        # Calcular AUC total usando regla del trapecio
        auc_total = np.trapz(acumulado_normalizado, dias)
        # Calcular AUC hasta día 121 (o hasta el último día si el rango acaba antes)
        dia_corte = 121
        if dia_corte <= dias.min():
            auc_hasta_121 = 0.0
            prop_acum_121 = 0.0
        elif dia_corte >= dias.max():
            # Si el corte supera el último día de datos, integrar hasta el final
            auc_hasta_121 = auc_total
            prop_acum_121 = 1.0  # ya está completo para esa fecha (normalizado)
        else:
            # Encontrar índice correspondiente (o interpolar) para el día 121
            if dia_corte in dias:
                idx121 = np.where(dias == dia_corte)[0][0]
                # Integración hasta día 121 inclusive
                auc_hasta_121 = np.trapz(acumulado_normalizado[:idx121+1], dias[:idx121+1])
                prop_acum_121 = acumulado_normalizado[idx121]
            else:
                # Interpolar valor en día 121 si no está exactamente en los datos
                valor_121 = float(np.interp(dia_corte, dias, acumulado_normalizado))
                # Integrar área hasta 121: integrar hasta el día anterior y añadir el trapecio parcial
                idx_below = np.searchsorted(dias, dia_corte) - 1
                auc_hasta_prev = np.trapz(acumulado_normalizado[:idx_below+1], dias[:idx_below+1])
                # Área del trapecio desde el último día conocido hasta el día 121
                x0, y0 = dias[idx_below], acumulado_normalizado[idx_below]
                x1, y1 = dia_corte, valor_121
                auc_hasta_121 = auc_hasta_prev + (y0 + y1) / 2 * (x1 - x0)
                prop_acum_121 = valor_121

        # Clasificación del patrón según proporción acumulada al día 121
        if prop_acum_121 >= 0.5:
            clasificacion = "CONCENTRADO"
        else:
            clasificacion = "DISPERSO"

        # Almacenar resultados de esta serie (usando el nombre del archivo sin extensión como identificador)
        nombre_serie = archivo.name.replace(".xlsx", "")
        resultados.append({
            "Serie": nombre_serie,
            "AUC_total": round(auc_total, 4),
            "AUC_dia121": round(auc_hasta_121, 4),
            "Proporción_121": round(prop_acum_121, 4),
            "Patrón": clasificacion
        })

    # Mostrar resultados en tabla
    resultados_df = pd.DataFrame(resultados)
    st.subheader("Resultados por Serie")
    st.dataframe(resultados_df)

    # Botón de descarga de resultados en CSV
    csv_data = resultados_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar resultados en CSV",
        data=csv_data,
        file_name="resultados_emergencia.csv",
        mime="text/csv"
    )

