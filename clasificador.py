# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 ANÁLISIS HISTÓRICO DE EMERGENCIA ACUMULADA
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
import os

st.set_page_config(page_title="Emergencia Acumulada Histórica", layout="centered")
st.title("🌾 Análisis histórico de emergencia acumulada y emergencia relativa semanal")

# ===============================================================
# FUNCIÓN ROBUSTA PARA CARGAR CURVAS ANUALES
# ===============================================================
@st.cache_data
def cargar_curvas_historicas():
    archivos = [
        "2008.xlsx","2009.xlsx","2010.xlsx","2011.xlsx","2012.xlsx",
        "2013.xlsx","2014.xlsx","2015.xlsx","2023.xlsx","2024.xlsx","2025.xlsx"
    ]

    curvas = []
    etiquetas = []

    for archivo in archivos:
        if not os.path.exists(archivo):
            continue

        try:
            df = pd.read_excel(archivo, header=None)
        except Exception as e:
            st.warning(f"No se pudo leer {archivo}: {e}")
            continue

        if df.shape[1] < 2:
            st.warning(f"{archivo} no tiene 2 columnas.")
            continue

        dias_raw = df.iloc[:, 0].values
        vals_raw = df.iloc[:, 1].values

        # Vector diario vacío
        diario = np.zeros(365, dtype=float)

        for d, val in zip(dias_raw, vals_raw):
            try:
                d_int = int(d)
                if 1 <= d_int <= 365:
                    diario[d_int - 1] = float(val)
            except:
                continue

        # Acumulada
        acum = np.cumsum(diario)
        maxv = acum[-1] if acum[-1] > 0 else 1
        curva_norm = acum / maxv

        # Etiqueta del año
        m = re.match(r"(\d+)", archivo)
        anno = m.group(1) if m else archivo

        curvas.append(curva_norm)
        etiquetas.append(anno)

    return np.array(curvas), etiquetas


# ===============================================================
# CARGA DATOS
# ===============================================================
curvas_historicas, etiquetas_annos = cargar_curvas_historicas()

if len(curvas_historicas) == 0:
    st.error("No se encontraron curvas válidas.")
    st.stop()

dias = np.arange(1, 366)

# ===============================================================
# SLIDER DEL DÍA JULIANO
# ===============================================================
dia_sel = st.slider("Seleccione el día juliano", 1, 365, 180)
idx = dia_sel - 1

valores_dia = curvas_historicas[:, idx]
media = valores_dia.mean()
std = valores_dia.std()
prob50 = (valores_dia > 0.5).mean()

st.markdown(f"""
### 📊 Estadísticas para el día **{dia_sel}**
- Emergencia acumulada promedio: **{media*100:.1f}% ± {std*100:.1f}%**
- Probabilidad de superar 50%: **{prob50*100:.1f}%**
""")


# ===============================================================
# PREPARAR DATOS PARA ALTAR
# ===============================================================
data = []

# Curvas individuales
for curva, anio in zip(curvas_historicas, etiquetas_annos):
    for d, v in zip(dias, curva):
        data.append({"Día": d, "Fracción": v, "Año": anio})

# Curva promedio
curva_prom = curvas_historicas.mean(axis=0)
for d, v in zip(dias, curva_prom):
    data.append({"Día": d, "Fracción": v, "Año": "Promedio"})

df = pd.DataFrame(data)

# ===============================================================
# EMERGENCIA RELATIVA SEMANAL (PROMEDIO)
# ===============================================================
emerg_diaria = np.diff(curva_prom, prepend=0)
rel_7d = np.convolve(emerg_diaria, np.ones(7)/7, mode="same")

df_rel = pd.DataFrame({"Día": dias, "Emergencia_rel_7d": rel_7d})

# ===============================================================
# GRÁFICOS ALTAR
# ===============================================================

# Curvas años individuales
g_ind = alt.Chart(df[df["Año"] != "Promedio"]).mark_line(opacity=0.4).encode(
    x="Día:Q",
    y=alt.Y("Fracción:Q", title="Fracción acumulada (0–1)", axis=alt.Axis(titleColor="steelblue")),
    color="Año:N"
)

# Promedio histórico
g_prom = alt.Chart(df[df["Año"] == "Promedio"]).mark_line(
    color="black", strokeWidth=3
).encode(
    x="Día:Q",
    y="Fracción:Q"
)

# Línea vertical (día seleccionado)
g_line = alt.Chart(pd.DataFrame({"Día": [dia_sel]})).mark_rule(
    color="red", strokeDash=[4,4]
).encode(x="Día:Q")

# Emergencia relativa 7 días
g_rel_area = alt.Chart(df_rel).mark_area(
    color="orange", opacity=0.3
).encode(
    x="Día:Q",
    y=alt.Y("Emergencia_rel_7d:Q",
            axis=alt.Axis(title="Emergencia relativa semanal", titleColor="orange"))
)

g_rel_line = alt.Chart(df_rel).mark_line(
    color="orange", strokeDash=[6,3], strokeWidth=2
).encode(
    x="Día:Q",
    y="Emergencia_rel_7d:Q"
)

# Combinación con ejes independientes
grafico_final = alt.layer(
    g_ind, g_prom, g_line, g_rel_area, g_rel_line
).resolve_scale(y="independent").properties(
    height=420,
    title="Curvas históricas de emergencia acumulada + emergencia relativa semanal"
)

# Mostrar
st.altair_chart(grafico_final, use_container_width=True)

# Leyenda
st.caption("""
🟢 **Curvas históricas:** cada año individual  
⚫ **Promedio histórico:** línea negra gruesa  
🟧 **Emergencia relativa semanal:** área y línea naranja  
🔴 **Línea roja:** día juliano seleccionado  
""")

