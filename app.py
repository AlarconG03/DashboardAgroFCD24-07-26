"""
app.py
------
TechLogistics S.A. - Dashboard de Consultoría Senior (Challenge 02)
Punto de entrada de la aplicación Streamlit. Orquesta los módulos:
  data_loader -> cleaning -> integration -> features -> visuals -> ai_insights

Ejecutar con: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from modules import data_loader, cleaning, integration, features, visuals, ai_insights

st.set_page_config(
    page_title="TechLogistics S.A. | DSS Consultoría",
    page_icon="📦",
    layout="wide",
)

# ------------------------------------------------------------------
# Sidebar: carga de archivos + filtros + acción de refresco
# ------------------------------------------------------------------
st.sidebar.title("📦 TechLogistics S.A.")
st.sidebar.caption("Sistema de Soporte a la Decisión — Consultoría Senior")

st.sidebar.subheader("1. Carga de datos")
inv_file = st.sidebar.file_uploader("inventario_central_v2.csv", type="csv")
trans_file = st.sidebar.file_uploader("transacciones_logistica_v2.csv", type="csv")
fb_file = st.sidebar.file_uploader("feedback_clientes_v2.csv", type="csv")

st.sidebar.subheader("2. Parámetros de negocio")
sla_dias = st.sidebar.number_input(
    "SLA de entrega prometido (días)", min_value=1, max_value=30, value=5,
    help="No existe un campo explícito de 'días prometidos' en el diccionario de datos; "
         "este es un supuesto de negocio configurable. Ver README.",
)

refresh = st.sidebar.button("🔄 Refrescar Análisis", type="primary", use_container_width=True)

if not (inv_file and trans_file and fb_file):
    st.title("📦 DSS TechLogistics S.A.")
    st.info(
        "Sube los tres archivos en la barra lateral para comenzar: "
        "**inventario_central_v2.csv**, **transacciones_logistica_v2.csv** y "
        "**feedback_clientes_v2.csv**."
    )
    st.stop()

# ------------------------------------------------------------------
# Pipeline: cargar -> limpiar -> integrar -> features
# Se recalcula solo si se pulsa Refrescar o si es la primera carga.
# ------------------------------------------------------------------
if refresh or "master_df" not in st.session_state:
    with st.spinner("Ejecutando pipeline de auditoría, limpieza e integración..."):
        raw, schema_warnings = data_loader.load_all_datasets(inv_file, trans_file, fb_file)

        inv_clean, inv_log, inv_health = cleaning.clean_inventory(raw["inventario"])
        trans_clean, trans_log, trans_health = cleaning.clean_transactions(raw["transacciones"])
        fb_clean, fb_log, fb_health = cleaning.clean_feedback(raw["feedback"])

        master, merge_log = integration.merge_master(inv_clean, trans_clean, fb_clean)
        phantom_summary = integration.phantom_sales_summary(master)
        master = features.build_features(master, sla_dias=sla_dias)

        st.session_state["master_df"] = master
        st.session_state["raw"] = raw
        st.session_state["schema_warnings"] = schema_warnings
        st.session_state["logs"] = {
            "inventario": inv_log, "transacciones": trans_log,
            "feedback": fb_log, "integracion (merge)": merge_log,
        }
        st.session_state["health"] = {"inventario": inv_health, "transacciones": trans_health, "feedback": fb_health}
        st.session_state["phantom_summary"] = phantom_summary

master = st.session_state["master_df"]

# ------------------------------------------------------------------
# Sidebar: filtros dinámicos sobre datos ya integrados
# ------------------------------------------------------------------
st.sidebar.subheader("3. Filtros de análisis")

if "Fecha_Venta" in master.columns and master["Fecha_Venta"].notna().any():
    min_d, max_d = master["Fecha_Venta"].min(), master["Fecha_Venta"].max()
    date_range = st.sidebar.date_input(
        "Rango de fecha de venta", value=(min_d.date(), max_d.date()),
        min_value=min_d.date(), max_value=max_d.date(),
    )
else:
    date_range = None

categorias = sorted(master["Categoria"].dropna().unique()) if "Categoria" in master.columns else []
sel_categorias = st.sidebar.multiselect("Categoría", categorias, default=categorias)

bodegas = sorted(master["Bodega_Origen"].dropna().unique()) if "Bodega_Origen" in master.columns else []
sel_bodegas = st.sidebar.multiselect("Bodega de origen", bodegas, default=bodegas)

canales = sorted(master["Canal_Venta"].dropna().unique()) if "Canal_Venta" in master.columns else []
sel_canales = st.sidebar.multiselect("Canal de venta", canales, default=canales) if canales else []

incluir_fantasma = st.sidebar.checkbox("Incluir Ventas Fantasma en KPIs", value=True)

# Aplicar filtros
df = master.copy()
if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    df = df[(df["Fecha_Venta"] >= start) & (df["Fecha_Venta"] <= end)]
if sel_categorias:
    df = df[df["Categoria"].isin(sel_categorias)]
if sel_bodegas:
    df = df[df["Bodega_Origen"].isin(sel_bodegas)]
if sel_canales:
    df = df[df["Canal_Venta"].isin(sel_canales)]
if not incluir_fantasma:
    df = df[~df["Venta_Fantasma"]]

filtros_aplicados = {
    "rango_fechas": [str(d) for d in date_range] if date_range else None,
    "categorias": sel_categorias,
    "bodegas": sel_bodegas,
    "canales": sel_canales,
    "incluye_venta_fantasma": incluir_fantasma,
}

st.title("📦 TechLogistics S.A. — DSS de Consultoría Senior")
st.caption(f"Última actualización del análisis: {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
           f"{len(df):,} registros tras filtros aplicados")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Ingreso filtrado (USD)", f"${df['Precio_Venta_Final'].sum():,.0f}" if "Precio_Venta_Final" in df.columns else "—")
kpi2.metric("Margen total (USD)", f"${df['Margen_Absoluto_USD'].sum():,.0f}" if "Margen_Absoluto_USD" in df.columns else "—")
kpi3.metric("% Venta Fantasma", f"{df['Venta_Fantasma'].mean()*100:.1f}%" if "Venta_Fantasma" in df.columns else "—")
kpi4.metric("NPS promedio", f"{df['Satisfaccion_NPS'].mean():.1f}" if "Satisfaccion_NPS" in df.columns else "—")

tab_auditoria, tab_operaciones, tab_cliente, tab_ia = st.tabs(
    ["🔍 Auditoría", "🚚 Operaciones", "👥 Cliente", "🤖 Insights de IA"]
)

# ------------------------------------------------------------------
# TAB: Auditoría (Módulo de Transparencia — Antes vs Después)
# ------------------------------------------------------------------
with tab_auditoria:
    st.header("Módulo de Transparencia: Antes vs Después")

    for warn_key, missing_cols in st.session_state["schema_warnings"].items():
        if missing_cols:
            st.warning(f"**{warn_key}**: faltan columnas esperadas del diccionario de datos: {missing_cols}")

    cols = st.columns(3)
    for col, (name, hs) in zip(cols, st.session_state["health"].items()):
        with col:
            st.plotly_chart(visuals.health_gauge(hs["health_score_pct"], f"Health Score — {name}"),
                             use_container_width=True)
            st.metric("Registros antes → después", f"{hs['registros_antes']:,} → {hs['registros_despues']:,}")
            st.metric("Duplicados eliminados", hs["duplicados_eliminados"])
            if hs["outliers_por_columna"]:
                for c, n in hs["outliers_por_columna"].items():
                    st.metric(f"Outliers IQR en {c}", n)

    st.subheader("Bitácora de limpieza aplicada")
    for name, log in st.session_state["logs"].items():
        with st.expander(f"Log — {name}"):
            for line in log:
                st.write(f"• {line}")

    st.subheader("Nulidad por columna (antes vs después)")
    dataset_pick = st.selectbox("Dataset", list(st.session_state["health"].keys()))
    hs_pick = st.session_state["health"][dataset_pick]
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(visuals.null_pct_bar(hs_pick["nulidad_antes_pct"], "Antes"), use_container_width=True)
    with c2:
        st.plotly_chart(visuals.null_pct_bar(hs_pick["nulidad_despues_pct"], "Después"), use_container_width=True)

    st.subheader("Registros excluidos / inconsistencias marcadas")
    flag_suffixes = ("_outlier", "_invalida", "_negativo", "_futura")
    flag_cols = [c for c in df.columns if c.endswith(flag_suffixes) or c in ("Bodega_No_Oficial",)]
    if flag_cols:
        flag = st.selectbox("Ver registros marcados por", flag_cols)
        if st.checkbox("Mostrar solo registros marcados", value=True):
            st.dataframe(df[df[flag] == True], use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
    else:
        st.info("No hay columnas de inconsistencias disponibles en la vista filtrada actual.")

    st.download_button(
        "⬇️ Descargar reporte de limpieza (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="reporte_datos_limpios_techlogistics.csv",
        mime="text/csv",
    )

# ------------------------------------------------------------------
# TAB: Operaciones (P1 margen, P2 logística, P3 venta fantasma)
# ------------------------------------------------------------------
with tab_operaciones:
    st.header("P1 · Fuga de Capital y Rentabilidad")
    fig_rank, fig_box = visuals.margin_leak_charts(df)
    st.plotly_chart(fig_rank, use_container_width=True)
    if fig_box:
        st.plotly_chart(fig_box, use_container_width=True)
    st.caption("¿Pérdida aceptable por volumen o falla crítica de precios? Compare la magnitud del "
               "margen negativo acumulado por SKU contra su volumen de venta antes de decidir.")

    st.divider()
    st.header("P2 · Crisis Logística y Cuellos de Botella")
    fig_corr, fig_scatter = visuals.logistics_bottleneck_charts(df)
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)
    if fig_scatter:
        st.plotly_chart(fig_scatter, use_container_width=True)
    if not fig_corr and not fig_scatter:
        st.info("Se requieren las columnas Tiempo_Entrega, Satisfaccion_NPS y Ciudad_Destino/Bodega_Origen.")

    st.divider()
    st.header("P3 · Análisis de la Venta Invisible (Venta Fantasma)")
    ps = st.session_state["phantom_summary"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Ingreso total (USD)", f"${ps['ingreso_total']:,.0f}")
    m2.metric("Ingreso en riesgo (USD)", f"${ps['ingreso_venta_fantasma']:,.0f}")
    m3.metric("% Ingreso en riesgo", f"{ps['pct_ingreso_en_riesgo']}%")
    donut, fig_city = visuals.phantom_sales_charts(df, ps)
    c1, c2 = st.columns(2)
    c1.plotly_chart(donut, use_container_width=True)
    if fig_city:
        c2.plotly_chart(fig_city, use_container_width=True)

# ------------------------------------------------------------------
# TAB: Cliente (P4 paradoja fidelidad, P5 riesgo operativo)
# ------------------------------------------------------------------
with tab_cliente:
    st.header("P4 · Diagnóstico de Fidelidad")
    fig_paradox = visuals.loyalty_paradox_chart(df)
    if fig_paradox:
        st.plotly_chart(fig_paradox, use_container_width=True)
        st.caption("Categorías en el cuadrante superior-izquierdo (bajo stock, buen rating) no son un "
                   "problema; el cuadrante inferior-derecho (alto stock, bajo rating) sí exige revisión "
                   "de calidad de producto vs. sobrecosto.")
    else:
        st.info("Se requieren las columnas Categoria, Stock_Actual y Rating_Producto.")

    st.divider()
    st.header("P5 · Storytelling de Riesgo Operativo")
    fig_risk = visuals.operational_risk_chart(df)
    if fig_risk:
        st.plotly_chart(fig_risk, use_container_width=True)
        st.caption("Bodegas arriba-a-la-derecha llevan más tiempo sin auditoría física de stock Y "
                   "generan más tickets de soporte: son las que están 'operando a ciegas'.")
    else:
        st.info("Se requieren las columnas Ultima_Revision, Ticket_Soporte y Bodega_Origen.")

# ------------------------------------------------------------------
# TAB: Insights de IA (Groq / Llama-3)
# ------------------------------------------------------------------
with tab_ia:
    st.header("🤖 Recomendación estratégica generada por IA (Llama-3 vía Groq)")
    st.caption("El modelo analiza EXCLUSIVAMENTE el resumen estadístico de los datos ya filtrados "
               "en el sidebar — no tiene acceso al dataset completo.")

    summary = ai_insights.build_filtered_summary(df, filtros_aplicados)
    with st.expander("Ver resumen estadístico enviado al modelo"):
        st.json(summary)

    if st.button("✨ Generar análisis con IA", type="primary"):
        with st.spinner("Consultando a Llama-3 en Groq..."):
            text, error = ai_insights.generate_recommendation(summary)
        if error:
            st.error(error)
        else:
            st.success("Análisis generado con base en los filtros actuales:")
            st.write(text)
