import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.title("Curvas de Emergencia Anuales")

# 1. Cargador de archivos Excel múltiple
uploaded_files = st.file_uploader(
    "Sube archivos Excel (.xlsx)", type="xlsx",
    accept_multiple_files=True, key="file_uploader"
)

if uploaded_files:
    curvas_acumuladas = {}  # dict año -> array acumulada
    curvas_no_acumuladas = {}  # dict año -> array no acumulada
    
    for file in uploaded_files:
        # Obtener año del nombre de archivo (asumimos que contiene 'YYYY')
        name = file.name
        year_match = pd.to_datetime(name, errors='ignore', format='%Y')
        if year_match == 'NaT':
            # Extraer dígitos de año
            import re
            m = re.search(r"(\d{4})", name)
            year = m.group(1) if m else name
        else:
            year = str(year_match.year)
        
        # Leer datos sin encabezado, columnas: [día, valor]
        df = pd.read_excel(file, header=None)
        df = df.dropna(how='all')  # eliminar filas vacías si las hubiera
        df.columns = ['dia', 'valor']
        # Aseguramos tipo numérico
        df['dia'] = df['dia'].astype(int)
        df['valor'] = df['valor'].astype(float)
        
        # Ordenar por día (por si acaso)
        df = df.sort_values('dia')
        
        # Valores y días originales
        dias = df['dia'].values
        vals = df['valor'].values
        
        # Extender la serie hasta el día 365 manteniendo último valor
        last_val = vals[-1] if len(vals) > 0 else 0.0
        if dias[-1] < 365:
            # construir un array de 1 a 365
            days_full = np.arange(1, 366)
            vals_full = np.zeros(365)
            # Rellenar los valores conocidos
            for d, v in zip(dias, vals):
                if 1 <= d <= 365:
                    vals_full[d-1] = v
            # A partir del último día, mantener valor constante
            vals_full[dias[-1]:] = last_val
        else:
            # Si hay datos hasta el 365 (o más), simplemente tomar del 1 al 365
            days_full = np.arange(1, 366)
            vals_full = np.zeros(365)
            for d, v in zip(dias, vals):
                if 1 <= d <= 365:
                    vals_full[d-1] = v
            # Si hubiera valores más allá de 365, se ignoran
        
        # Curva no acumulada (ya está en vals_full)
        curva_no_acum = vals_full
        
        # Curva acumulada: cumsum y luego normalizar para que último valor sea 1.0
        curva_acum = np.cumsum(curva_no_acum)
        if curva_acum[-1] != 0:
            curva_acum = curva_acum / curva_acum[-1]
        else:
            # Evitar división por cero
            curva_acum = curva_acum
        
        # Guardar curvas por año
        curvas_no_acumuladas[year] = curva_no_acum
        curvas_acumuladas[year] = curva_acum
    
    # Convertir diccionarios a arrays para cálculos globales
    años = list(curvas_acumuladas.keys())
    data_acum = np.vstack([curvas_acumuladas[a] for a in años])
    
    # 4. Slider para día juliano (única vez)
    dia = st.slider("Selecciona el día juliano:", 1, 365, value=1, key="dia_slider")
    
    # 5. Estadísticas para el día seleccionado
    # Índice en Python: día juliano 1 -> índice 0
    idx = dia - 1
    valores_dia = data_acum[:, idx]  # valores acumulados en ese día, por año
    media = np.mean(valores_dia)
    desviacion = np.std(valores_dia)
    # Probabilidad de superar 50% (>= 0.5)
    prob_supera = np.mean(valores_dia >= 0.5)
    
    st.write(f"Media de la curva **acumulada** en el día {dia}: {media:.3f}")
    st.write(f"Desviación estándar: {desviacion:.3f}")
    st.write(f"Probabilidad de haber superado el 50% del total acumulado en este día: {prob_supera:.2%}")
    
    # 6. Gráfico Altair con todas las curvas
    # Preparar DataFrames largos para Altair
    # Curvas acumuladas (formato largo)
    df_acum = pd.DataFrame({
        'día': np.tile(np.arange(1, 366), len(años)),
        'Valor': np.concatenate([curvas_acumuladas[a] for a in años]),
        'Año': np.repeat(años, 365)
    })
    # Curvas no acumuladas (formato largo)
    df_no_acum = pd.DataFrame({
        'día': np.tile(np.arange(1, 366), len(años)),
        'Valor': np.concatenate([curvas_no_acumuladas[a] for a in años]),
        'Año': np.repeat(años, 365)
    })
    # Curva promedio acumulada
    media_acum = np.mean(data_acum, axis=0)
    df_media = pd.DataFrame({
        'día': np.arange(1, 366),
        'Valor': media_acum
    })
    # Línea vertical para el día seleccionado
    df_vline = pd.DataFrame({'día': [dia]})
    
    # Base para curvas acumuladas: líneas por año
    lineas_acum = alt.Chart(df_acum).mark_line(opacity=0.7).encode(
        x=alt.X('día:Q', title='Día juliano'),
        y=alt.Y('Valor:Q', title='Curva Emergencia'),
        color=alt.Color('Año:N', legend=None)
    )
    # Curvas no acumuladas: podemos usar líneas más finas o semi-transparentes
    lineas_no_acum = alt.Chart(df_no_acum).mark_line(strokeDash=[4,2], opacity=0.5).encode(
        x='día:Q', y='Valor:Q', color=alt.Color('Año:N', legend=None)
    )
    # Curva promedio acumulada: destacada (p.ej. negro punteado)
    linea_media = alt.Chart(df_media).mark_line(color='black', strokeDash=[5,3], size=2).encode(
        x='día:Q', y='Valor:Q'
    )
    # Línea vertical en el día seleccionado (regla roja)
    vline = alt.Chart(df_vline).mark_rule(color='red').encode(x='día:Q')
    
    # Superponer todas las capas
    chart = alt.layer(
        lineas_acum,
        lineas_no_acum,
        linea_media,
        vline
    ).properties(
        width=700, height=400,
        title=f"Curvas acumuladas y no acumuladas por año (Día {dia} seleccionado)"
    )
    
    st.altair_chart(chart, use_container_width=True)

