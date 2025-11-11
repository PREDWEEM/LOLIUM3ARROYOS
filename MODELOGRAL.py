import streamlit as st
import pandas as pd
import altair as alt

st.title("Visualización histórica de emergencia agrícola")

# **Carga de datos**: Especificar los años a incluir y sus archivos
anos = [2008, 2009, 2011, 2012, 2013, 2014, 2023, 2024, 2025]
# Diccionarios para almacenar curvas acumuladas y no acumuladas
curvas_acumuladas = {}
curvas_no_acumuladas = {}

for ano in anos:
    # Leer el archivo Excel correspondiente al año (sin encabezado explícito)
    df = pd.read_excel(f"{ano}.xlsx", header=None, names=["dia", "valor"])
    # Asegurar que los días sean 1-365 (por si faltan días, completaremos luego)
    df = df.astype({"dia": int, "valor": float})
    # Si el último día presente es < 365, extender el DataFrame con valores constantes hasta 365
    if df["dia"].iloc[-1] < 365:
        ultimo_dia = int(df["dia"].iloc[-1])
        ultimo_valor = float(df["valor"].iloc[-1])
        # Crear filas desde ultimo_dia+1 hasta 365 con el ultimo_valor
        dias_faltantes = pd.DataFrame({
            "dia": list(range(ultimo_dia + 1, 366)),
            "valor": [ultimo_valor] * (365 - ultimo_dia)
        })
        df = pd.concat([df, dias_faltantes], ignore_index=True)
    # Ahora df tiene días 1..365 (si originalmente terminaba antes, completado con último valor)

    # Decidir si la serie es diaria (fracciones) o semanal (valores absolutos)
    total_anual = df["valor"].sum()
    es_diario = (abs(total_anual - 1.0) < 0.2) or (ano in [2023, 2024, 2025])
    # (Usamos una tolerancia ~0.2 por si sumas ~0.94 en datos incompletos, etc.)

    if es_diario:
        # Datos diarios (fracción del total por día): aplicar cumsum directamente para curva acumulada
        curva_acum = df["valor"].cumsum()
    else:
        # Datos semanales/absolutos: verificar si ya es acumulada o no
        serie = df["valor"]
        if not serie.is_monotonic_increasing:
            # No es monótona creciente -> son incrementos semanales -> acumulamos
            curva_acum = serie.cumsum()
        else:
            # Ya es creciente -> asumimos que ya es acumulada
            curva_acum = serie.copy()
    # Normalizar la curva acumulada al total anual (último valor)
    total_final = curva_acum.iloc[-1]
    if total_final == 0:
        # Evitar dividir por cero si hubiera año sin datos (no debería ocurrir aquí)
        curva_acum_norm = curva_acum
    else:
        curva_acum_norm = curva_acum / float(total_final)
    # Guardar curva acumulada normalizada (como lista para Altair)
    curvas_acumuladas[ano] = curva_acum_norm.values

    # Calcular curva no acumulada relativa: diferencias diarias de la acumulada normalizada
    valores_relativos = [curva_acum_norm.iloc[0]]
    # Diferenica sucesiva: valor[i] = curva_acum[i] - curva_acum[i-1]
    valores_relativos += list(curva_acum_norm.iloc[1:].values - curva_acum_norm.iloc[:-1].values)
    curvas_no_acumuladas[ano] = valores_relativos

# Convertir datos a formato largo para Altair
data_list = []
for ano in anos:
    # Curva acumulada
    for dia in range(1, 366):
        data_list.append({
            "Año": str(ano),
            "Día": dia,
            "Valor": curvas_acumuladas[ano][dia-1],
            "Tipo": "Acumulada"
        })
    # Curva no acumulada (semanal/diaria)
    for dia in range(1, 366):
        data_list.append({
            "Año": str(ano),
            "Día": dia,
            "Valor": curvas_no_acumuladas[ano][dia-1],
            "Tipo": "Semanal"
        })

df_long = pd.DataFrame(data_list)

# Crear slider para seleccionar día juliano
dia_sel = st.slider("Seleccionar día juliano", min_value=1, max_value=365, value=180)

# **Cálculo de estadísticas para el día seleccionado**:
# Filtrar valores acumulados de todos los años en el día seleccionado
vals_dia = []
for ano in anos:
    # tomar valor acumulado normalizado de ese año al día seleccionado
    val = curvas_acumuladas[ano][dia_sel-1]
    vals_dia.append(val)
promedio = pd.np.mean(vals_dia)  # (pd.np is a shorthand to NumPy)
desvest = pd.np.std(vals_dia, ddof=0)  # desviación estándar poblacional
# Probabilidad de superar 50% (0.5) a ese día
count_superan = sum(1 for v in vals_dia if v >= 0.5)
probabilidad = count_superan / len(vals_dia) if len(vals_dia) > 0 else 0.0

# Mostrar estadísticas en la app
st.write(f"**Emergencia acumulada media (fracción del total anual) al día {dia_sel}:** {promedio:.3f}")
st.write(f"**Desviación estándar:** {desvest:.3f}")
st.write(f"**Probabilidad de haber superado 50% del total anual para este día:** {(probabilidad*100):.1f}%")

# **Construcción del gráfico Altair**
# Gráfico base con todas las curvas (acumuladas y semanales)
chart = alt.Chart(df_long).mark_line().encode(
    x=alt.X("Día:Q", title="Día del año"),
    y=alt.Y("Valor:Q", title="Fracción del total anual"),
    color=alt.Color("Año:N", title="Año"),
    strokeDash=alt.StrokeDash("Tipo:N", title="Tipo de curva")
)

# Destacar la curva promedio acumulada (cálculo previo por día)
# Construir DataFrame para la línea promedio acumulada
prom_cum = []
curva_prom = [0] * 365
# calcular promedio acumulado por día (suma de curvas acumuladas de todos los años / n años)
for dia in range(1, 366):
    valores_dia = [curvas_acumuladas[ano][dia-1] for ano in anos]
    curva_prom[dia-1] = sum(valores_dia) / len(valores_dia)
    prom_cum.append({"Día": dia, "Valor": curva_prom[dia-1]})
df_prom = pd.DataFrame(prom_cum)
linea_promedio = alt.Chart(df_prom).mark_line(color="black", size=3).encode(
    x="Día:Q",
    y="Valor:Q"
)

# Línea vertical en el día seleccionado
linea_vertical = alt.Chart(pd.DataFrame({"Día": [dia_sel]})).mark_rule(
    color="firebrick", strokeWidth=2, strokeDash=[4,2]
).encode(x="Día:Q")

# Combinar las capas: curvas de años + promedio + regla vertical
combined_chart = chart + linea_promedio + linea_vertical
# Ajustar dimensiones y leyendas
combined_chart = combined_chart.properties(width=700, height=400)
combined_chart = combined_chart.configure_legend(
    titleFontSize=11,
    labelFontSize=10
)

st.altair_chart(combined_chart, use_container_width=True)

