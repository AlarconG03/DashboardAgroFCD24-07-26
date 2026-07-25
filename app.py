import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq
from data_processor import procesar_datos
import os

# Configuración inicial
st.set_page_config(page_title="TechLogistics S.A. | DSS", layout="wide")
st.title("📦 TechLogistics S.A. - Sistema de Soporte a la Decisión")

# Carga y caché de datos
@st.cache_data
def load_data():
    return procesar_datos(
        'data/inventario_central_v2.csv',
        'data/transacciones_logistica_v2.csv',
        'data/feedback_clientes_v2.csv'
    )

df, outliers, h_antes, h_despues = load_data()

# --- BARRA LATERAL ---
st.sidebar.header("Filtros Estratégicos")
df_filtrado = df.copy()

fecha_min = df_filtrado['Fecha_Venta'].min().date() if not df_filtrado['Fecha_Venta'].isnull().all() else pd.to_datetime('2025-01-01').date()
fecha_max = df_filtrado['Fecha_Venta'].max().date() if not df_filtrado['Fecha_Venta'].isnull().all() else pd.to_datetime('2026-12-31').date()

fechas = st.sidebar.date_input("Rango de Fechas", [fecha_min, fecha_max])
if len(fechas) == 2:
    df_filtrado = df_filtrado[(df_filtrado['Fecha_Venta'].dt.date >= fechas[0]) & (df_filtrado['Fecha_Venta'].dt.date <= fechas[1])]

categorias = df_filtrado['Categoria'].dropna().unique().tolist()
cat_sel = st.sidebar.multiselect("Categoría", categorias, default=categorias)
if cat_sel:
    df_filtrado = df_filtrado[df_filtrado['Categoria'].isin(cat_sel)]

bodegas = df_filtrado['Bodega_Origen'].dropna().unique().tolist()
bod_sel = st.sidebar.multiselect("Bodega Origen", bodegas, default=bodegas)
if bod_sel:
    df_filtrado = df_filtrado[df_filtrado['Bodega_Origen'].isin(bod_sel)]

if st.sidebar.button("🔄 Refrescar Análisis"):
    st.cache_data.clear()
    st.rerun()

# --- PESTAÑAS (VISUALIZACIÓN PROGRESIVA) ---
tab1, tab2, tab3, tab4 = st.tabs(["🛡️ Auditoría de Datos", "📊 Operaciones y Rentabilidad", "👥 Experiencia del Cliente", "🧠 Insights de IA"])

with tab1:
    st.header("Auditoría de Calidad y Transparencia")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Health Score (Antes)")
        st.json(h_antes)
    with col2:
        st.subheader("Health Score (Después)")
        st.json(h_despues)
    
    st.subheader("Análisis de Ventas Fantasma (SKUs sin catálogo)")
    fantasmas = df[df['Venta_Fantasma'] == True]
    st.warning(f"Se detectaron {len(fantasmas)} transacciones huérfanas.")
    st.dataframe(fantasmas[['Transaccion_ID', 'SKU_ID', 'Cantidad_Vendida', 'Ingreso_Total']].head())
    
    with st.expander("Ver registros excluidos por costos anómalos (Outliers)"):
        st.dataframe(outliers)

with tab2:
    st.header("Rendimiento Operativo")
    
    # Q1: Fuga de Capital y Rentabilidad
    st.subheader("1. SKUs con Margen de Utilidad Negativo")
    df_margen_neg = df_filtrado[df_filtrado['Margen_Utilidad'] < 0]
    if not df_margen_neg.empty:
        fig_q1 = px.bar(
            df_margen_neg.groupby('SKU_ID')['Margen_Utilidad'].sum().reset_index().sort_values('Margen_Utilidad').head(10),
            x='SKU_ID', y='Margen_Utilidad', title="Top 10 SKUs con Pérdidas", color='Margen_Utilidad'
        )
        st.plotly_chart(fig_q1, use_container_width=True)
    else:
        st.success("No hay SKUs con margen negativo en los filtros actuales.")

    # Q3: Análisis de la Venta Invisible
    st.subheader("3. Impacto Financiero de Ventas Invisibles")
    impacto_fantasma = fantasmas['Ingreso_Total'].sum()
    porcentaje_riesgo = (impacto_fantasma / df['Ingreso_Total'].sum()) * 100 if df['Ingreso_Total'].sum() > 0 else 0
    col1, col2 = st.columns(2)
    col1.metric("USD en Riesgo (Ventas sin SKU oficial)", f"${impacto_fantasma:,.2f}")
    col2.metric("% del Ingreso Total Comprometido", f"{porcentaje_riesgo:.2f}%")

    # Q5: Storytelling de Riesgo Operativo
    st.subheader("5. Riesgo Operativo: Auditoría vs Soporte")
    df_q5 = df_filtrado.groupby('Bodega_Origen').agg({'Dias_Desde_Revision': 'mean', 'Tiene_Ticket': 'mean'}).reset_index()
    fig_q5 = px.scatter(
        df_q5, x='Dias_Desde_Revision', y='Tiene_Ticket', size='Dias_Desde_Revision', color='Bodega_Origen',
        title="Días sin revisión de inventario vs Tasa de Tickets de Soporte",
        labels={'Tiene_Ticket': 'Ratio de Tickets', 'Dias_Desde_Revision': 'Días desde Última Revisión'}
    )
    st.plotly_chart(fig_q5, use_container_width=True)

with tab3:
    st.header("Diagnóstico de Experiencia del Cliente")
    
    # Q2: Crisis Logística y Cuellos de Botella
    st.subheader("2. Correlación: Tiempo de Entrega vs NPS por Ciudad")
    df_q2 = df_filtrado.groupby(['Ciudad_Destino', 'Bodega_Origen']).agg({'Tiempo_Entrega':'mean', 'Satisfaccion_NPS':'mean'}).reset_index()
    fig_q2 = px.scatter(
        df_q2, x='Tiempo_Entrega', y='Satisfaccion_NPS', color='Ciudad_Destino', hover_data=['Bodega_Origen'],
        title="Impacto del Tiempo de Entrega en el NPS"
    )
    st.plotly_chart(fig_q2, use_container_width=True)

    # Q4: Diagnóstico de Fidelidad
    st.subheader("4. Paradoja de Fidelidad: Stock vs Sentimiento Negativo")
    df_q4 = df_filtrado.groupby('Categoria').agg({'Stock_Actual': 'mean', 'Satisfaccion_NPS': 'mean'}).reset_index()
    fig_q4 = px.scatter(
        df_q4, x='Stock_Actual', y='Satisfaccion_NPS', color='Categoria', size='Stock_Actual',
        title="Disponibilidad (Stock) vs NPS por Categoría"
    )
    st.plotly_chart(fig_q4, use_container_width=True)

with tab4:
    st.header("Insights Estratégicos con IA (Llama-3)")
    st.markdown("Generación de recomendaciones en tiempo real basadas en los datos filtrados.")
    
    if st.button("Generar Diagnóstico con Groq"):
        try:
            # Resumen estadístico ligero para el prompt
            resumen = f"""
            Ventas totales filtradas: {len(df_filtrado)}. 
            Margen Promedio: {df_filtrado['Margen_Utilidad'].mean():.2f}. 
            NPS Promedio: {df_filtrado['Satisfaccion_NPS'].mean():.2f}.
            Tiempo de entrega promedio: {df_filtrado['Tiempo_Entrega'].mean():.2f} días.
            """
            
            cliente = Groq(api_key=st.secrets["GROQ_API_KEY"])
            respuesta = cliente.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Eres un Consultor Senior de Datos. Responde estrictamente en 3 párrafos con recomendaciones estratégicas de negocio basadas en el siguiente resumen de datos, sin mencionar código."},
                    {"role": "user", "content": resumen}
                ],
                model="llama3-8b-8192",
            )
            st.info(respuesta.choices[0].message.content)
        except Exception as e:
            st.error(f"Error al conectar con Groq: Verifica que la API Key esté configurada correctamente. Detalle: {e}")
