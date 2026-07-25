"""
visuals.py
----------
Una función por cada uno de los 5 Interrogantes Estratégicos Obligatorios
del Challenge, más utilidades para el módulo de Auditoría (Health Score,
Antes vs Después). Todas las funciones retornan figuras de Plotly, listas
para st.plotly_chart().

Elección de gráfico por pregunta (justificación breve, ver README):
  P1 Fuga de capital      -> Bar horizontal (ranking) + box plot por categoría
  P2 Cuellos de botella   -> Heatmap de correlación + scatter ciudad/bodega
  P3 Venta invisible      -> Donut (composición del ingreso) + bar por ciudad
  P4 Paradoja de fidelidad-> Scatter de cuadrantes (stock vs sentimiento)
  P5 Riesgo operativo     -> Scatter con línea de tendencia (antigüedad vs soporte)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------------------
# Auditoría / Transparencia
# ---------------------------------------------------------------------

def null_pct_bar(nulidad_series: pd.Series, title: str):
    fig = px.bar(
        nulidad_series.sort_values(ascending=False),
        labels={"index": "Columna", "value": "% Nulidad"},
        title=title,
    )
    fig.update_layout(showlegend=False, yaxis_title="% Nulidad", xaxis_title="")
    return fig


def health_gauge(pct: float, title: str = "Health Score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        title={"text": title},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#2E7D32" if pct >= 90 else "#F9A825" if pct >= 75 else "#C62828"}},
    ))
    return fig


# ---------------------------------------------------------------------
# P1: Fuga de Capital y Rentabilidad
# ---------------------------------------------------------------------

def margin_leak_charts(df: pd.DataFrame, top_n: int = 15):
    negative = df[df["Margen_Absoluto_USD"] < 0].copy()

    by_sku = (
        negative.groupby("SKU_ID")["Margen_Absoluto_USD"]
        .sum().sort_values().head(top_n).reset_index()
    )
    fig_rank = px.bar(
        by_sku, x="Margen_Absoluto_USD", y="SKU_ID", orientation="h",
        title=f"Top {top_n} SKUs con mayor pérdida acumulada (margen negativo)",
        labels={"Margen_Absoluto_USD": "Margen acumulado (USD)", "SKU_ID": "SKU"},
        color="Margen_Absoluto_USD", color_continuous_scale="Reds_r",
    )

    box_dims = [c for c in ["Categoria", "Canal"] if c in df.columns]
    fig_box = None
    if box_dims:
        dim = "Canal" if "Canal" in box_dims else "Categoria"
        fig_box = px.box(
            df, x=dim, y="Margen_Pct", points=False,
            title=f"Distribución de margen (%) por {dim}",
            labels={"Margen_Pct": "Margen (%)"},
        )
        fig_box.add_hline(y=0, line_dash="dash", line_color="red")

    return fig_rank, fig_box


# ---------------------------------------------------------------------
# P2: Crisis Logística y Cuellos de Botella
# ---------------------------------------------------------------------

def logistics_bottleneck_charts(df: pd.DataFrame):
    dims = [c for c in ["Ciudad_Destino", "Bodega_Origen"] if c in df.columns]
    fig_corr = None
    if dims and "Tiempo_Entrega" in df.columns and "Satisfaccion_NPS" in df.columns:
        rows = []
        for dim in dims:
            for grp, sub in df.dropna(subset=["Tiempo_Entrega", "Satisfaccion_NPS"]).groupby(dim):
                if len(sub) >= 5:
                    corr = sub["Tiempo_Entrega"].corr(sub["Satisfaccion_NPS"])
                    rows.append({"Dimension": dim, "Grupo": grp, "Correlacion": corr, "n": len(sub)})
        corr_df = pd.DataFrame(rows)
        if not corr_df.empty:
            fig_corr = px.bar(
                corr_df.sort_values("Correlacion"),
                x="Correlacion", y="Grupo", color="Dimension", orientation="h",
                title="Correlación Tiempo de Entrega vs NPS (más negativo = peor)",
                labels={"Correlacion": "Correlación (Pearson)"},
            )
            fig_corr.add_vline(x=0, line_dash="dash")

    fig_scatter = None
    if "Ciudad_Destino" in df.columns and "Tiempo_Entrega" in df.columns and "Satisfaccion_NPS" in df.columns:
        fig_scatter = px.scatter(
            df.dropna(subset=["Tiempo_Entrega", "Satisfaccion_NPS"]),
            x="Tiempo_Entrega", y="Satisfaccion_NPS", color="Ciudad_Destino",
            trendline="ols" if df.shape[0] < 5000 else None,
            title="Tiempo de Entrega vs Satisfacción NPS por Ciudad",
            opacity=0.6,
        )
    return fig_corr, fig_scatter


# ---------------------------------------------------------------------
# P3: Análisis de la Venta Invisible (Venta Fantasma)
# ---------------------------------------------------------------------

def phantom_sales_charts(df: pd.DataFrame, summary: dict):
    donut = px.pie(
        names=["Ingreso catalogado", "Ingreso Venta Fantasma (SKU sin inventario)"],
        values=[summary["ingreso_total"] - summary["ingreso_venta_fantasma"], summary["ingreso_venta_fantasma"]],
        hole=0.55, title="Composición del ingreso total: catalogado vs Venta Fantasma",
        color_discrete_sequence=["#2E7D32", "#C62828"],
    )

    fig_city = None
    if "Ciudad_Destino" in df.columns:
        by_city = (
            df[df["Venta_Fantasma"]].groupby("Ciudad_Destino")["Precio_Venta_Final"]
            .sum().sort_values(ascending=False).reset_index()
        )
        fig_city = px.bar(
            by_city, x="Ciudad_Destino", y="Precio_Venta_Final",
            title="Ingreso en riesgo (Venta Fantasma) por ciudad",
            labels={"Precio_Venta_Final": "Ingreso en riesgo (USD)"},
        )
    return donut, fig_city


# ---------------------------------------------------------------------
# P4: Diagnóstico de Fidelidad (paradoja stock alto / sentimiento negativo)
# ---------------------------------------------------------------------

def loyalty_paradox_chart(df: pd.DataFrame):
    if not {"Categoria", "Stock_Actual", "Rating_Producto"}.issubset(df.columns):
        return None
    agg = df.groupby("Categoria").agg(
        Stock_Prom=("Stock_Actual", "mean"),
        Rating_Prom=("Rating_Producto", "mean"),
        Tickets_Soporte=("Ticket_Soporte", "mean") if "Ticket_Soporte" in df.columns else ("Rating_Producto", "size"),
        n=("Rating_Producto", "size"),
    ).reset_index()

    fig = px.scatter(
        agg, x="Stock_Prom", y="Rating_Prom", size="n", color="Tickets_Soporte",
        text="Categoria", color_continuous_scale="RdYlGn_r",
        title="Paradoja de Fidelidad: Stock disponible vs Satisfacción del producto",
        labels={"Stock_Prom": "Stock promedio", "Rating_Prom": "Rating promedio del producto",
                "Tickets_Soporte": "% Tickets soporte"},
    )
    fig.update_traces(textposition="top center")
    fig.add_hline(y=agg["Rating_Prom"].mean(), line_dash="dash", opacity=0.4)
    fig.add_vline(x=agg["Stock_Prom"].mean(), line_dash="dash", opacity=0.4)
    return fig


# ---------------------------------------------------------------------
# P5: Storytelling de Riesgo Operativo (antigüedad de revisión vs soporte)
# ---------------------------------------------------------------------

def operational_risk_chart(df: pd.DataFrame):
    needed = {"Dias_Desde_Revision", "Ticket_Soporte", "Bodega_Origen"}
    if not needed.issubset(df.columns):
        return None
    agg = df.groupby("Bodega_Origen").agg(
        Antiguedad_Prom=("Dias_Desde_Revision", "mean"),
        Tasa_Soporte=("Ticket_Soporte", "mean"),
        n=("Ticket_Soporte", "size"),
    ).reset_index()
    agg["Tasa_Soporte"] = agg["Tasa_Soporte"] * 100

    fig = px.scatter(
        agg, x="Antiguedad_Prom", y="Tasa_Soporte", size="n", text="Bodega_Origen",
        trendline="ols" if len(agg) >= 3 else None,
        title="Bodegas operando 'a ciegas': antigüedad de revisión vs tasa de tickets de soporte",
        labels={"Antiguedad_Prom": "Antigüedad promedio última revisión (días)",
                "Tasa_Soporte": "Tasa de tickets de soporte (%)"},
    )
    fig.update_traces(textposition="top center")
    return fig
