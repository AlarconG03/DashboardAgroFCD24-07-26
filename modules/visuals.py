"""
visuals.py
----------
Una función por cada uno de los 5 Interrogantes Estratégicos Obligatorios
del Challenge, más utilidades para el módulo de Auditoría (Health Score,
Antes vs Después). Todas las funciones retornan figuras de Plotly, listas
para st.plotly_chart().

Elección de gráfico por pregunta (justificación breve, ver README):
  P1 Fuga de capital      -> Bar horizontal (ranking) + box plot por Canal de Venta
  P2 Cuellos de botella   -> Heatmap/bar de correlación + scatter ciudad/bodega
  P3 Venta invisible      -> Donut (composición del ingreso) + bar por ciudad
  P4 Paradoja de fidelidad-> Scatter de cuadrantes (stock vs sentimiento)
  P5 Riesgo operativo     -> Scatter con línea de tendencia (antigüedad vs soporte)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------
# Identidad visual unificada del DSS (paleta TechLogistics)
# ---------------------------------------------------------------------
PALETTE = ["#0B5FFF", "#00B8A9", "#F9A825", "#C62828", "#6A4C93", "#2E7D32"]
NEGATIVE = "#C62828"
POSITIVE = "#2E7D32"
NEUTRAL = "#546E7A"

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALETTE

BASE_LAYOUT = dict(
    font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#1f2937"),
    title=dict(font=dict(size=17, color="#111827")),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(t=70, l=10, r=10, b=10),
    hoverlabel=dict(bgcolor="white", font_size=12),
)


def _style(fig):
    fig.update_layout(**BASE_LAYOUT)
    return fig


# ---------------------------------------------------------------------
# Auditoría / Transparencia
# ---------------------------------------------------------------------

def null_pct_bar(nulidad_series: pd.Series, title: str):
    data = nulidad_series.sort_values(ascending=False)
    fig = px.bar(
        data, orientation="h",
        labels={"index": "", "value": "% Nulidad"},
        title=title, color=data.values, color_continuous_scale="Reds",
        text=[f"{v:.1f}%" for v in data.values],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, coloraxis_showscale=False, yaxis_title="", xaxis_title="% Nulidad")
    return _style(fig)


def health_gauge(pct: float, title: str = "Health Score"):
    color = POSITIVE if pct >= 90 else "#F9A825" if pct >= 75 else NEGATIVE
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 34}},
        title={"text": title, "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.75},
            "bgcolor": "#F3F4F6",
            "steps": [
                {"range": [0, 75], "color": "#FDECEC"},
                {"range": [75, 90], "color": "#FFF6E0"},
                {"range": [90, 100], "color": "#E9F7EF"},
            ],
        },
    ))
    fig.update_layout(margin=dict(t=50, b=10, l=20, r=20), height=220)
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
        labels={"Margen_Absoluto_USD": "Margen acumulado (USD)", "SKU_ID": ""},
        color="Margen_Absoluto_USD", color_continuous_scale=["#7f0000", "#ffb3b3"],
        text=by_sku["Margen_Absoluto_USD"].map(lambda v: f"${v:,.0f}"),
    )
    fig_rank.update_traces(textposition="outside")
    fig_rank.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
    fig_rank = _style(fig_rank)

    dim = "Canal_Venta" if "Canal_Venta" in df.columns else ("Categoria" if "Categoria" in df.columns else None)
    fig_box = None
    if dim:
        fig_box = px.box(
            df.dropna(subset=[dim, "Margen_Pct"]), x=dim, y="Margen_Pct", points=False,
            title=f"Distribución de margen (%) por {'Canal de Venta' if dim == 'Canal_Venta' else dim}",
            labels={"Margen_Pct": "Margen (%)", dim: ""},
            color=dim,
        )
        fig_box.add_hline(y=0, line_dash="dash", line_color=NEGATIVE,
                           annotation_text="Punto de equilibrio", annotation_position="top left")
        fig_box.update_layout(showlegend=False)
        fig_box = _style(fig_box)

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
            corr_df = corr_df.sort_values("Correlacion")
            fig_corr = px.bar(
                corr_df, x="Correlacion", y="Grupo", color="Dimension", orientation="h",
                title="Correlación Tiempo de Entrega vs NPS (más negativo = peor experiencia)",
                labels={"Correlacion": "Correlación (Pearson)", "Grupo": ""},
                hover_data={"n": True},
            )
            fig_corr.add_vline(x=0, line_dash="dash", line_color=NEUTRAL)
            fig_corr = _style(fig_corr)

    fig_scatter = None
    if "Ciudad_Destino" in df.columns and "Tiempo_Entrega" in df.columns and "Satisfaccion_NPS" in df.columns:
        sample = df.dropna(subset=["Tiempo_Entrega", "Satisfaccion_NPS"])
        fig_scatter = px.scatter(
            sample, x="Tiempo_Entrega", y="Satisfaccion_NPS", color="Ciudad_Destino",
            trendline="ols" if len(sample) < 5000 else None,
            title="Tiempo de Entrega vs Satisfacción NPS por Ciudad",
            opacity=0.55,
            labels={"Tiempo_Entrega": "Tiempo de entrega (días)", "Satisfaccion_NPS": "NPS"},
        )
        fig_scatter = _style(fig_scatter)
    return fig_corr, fig_scatter


# ---------------------------------------------------------------------
# P3: Análisis de la Venta Invisible (Venta Fantasma)
# ---------------------------------------------------------------------

def phantom_sales_charts(df: pd.DataFrame, summary: dict):
    donut = px.pie(
        names=["Ingreso catalogado", "Ingreso Venta Fantasma\n(SKU sin inventario)"],
        values=[summary["ingreso_total"] - summary["ingreso_venta_fantasma"], summary["ingreso_venta_fantasma"]],
        hole=0.6, title="Composición del ingreso total: catalogado vs Venta Fantasma",
        color_discrete_sequence=[POSITIVE, NEGATIVE],
    )
    donut.update_traces(textinfo="percent+label", pull=[0, 0.06])
    donut = _style(donut)

    fig_city = None
    if "Ciudad_Destino" in df.columns:
        by_city = (
            df[df["Venta_Fantasma"]].groupby("Ciudad_Destino")["Precio_Venta_Final"]
            .sum().sort_values(ascending=False).reset_index()
        )
        fig_city = px.bar(
            by_city, x="Ciudad_Destino", y="Precio_Venta_Final",
            title="Ingreso en riesgo (Venta Fantasma) por ciudad",
            labels={"Precio_Venta_Final": "Ingreso en riesgo (USD)", "Ciudad_Destino": ""},
            color="Precio_Venta_Final", color_continuous_scale=["#ffcdd2", "#7f0000"],
            text=by_city["Precio_Venta_Final"].map(lambda v: f"${v:,.0f}"),
        )
        fig_city.update_traces(textposition="outside")
        fig_city.update_layout(coloraxis_showscale=False)
        fig_city = _style(fig_city)
    return donut, fig_city


# ---------------------------------------------------------------------
# P4: Diagnóstico de Fidelidad (paradoja stock alto / sentimiento negativo)
# ---------------------------------------------------------------------

def loyalty_paradox_chart(df: pd.DataFrame):
    if not {"Categoria", "Stock_Actual", "Rating_Producto"}.issubset(df.columns):
        return None
    agg_kwargs = dict(
        Stock_Prom=("Stock_Actual", "mean"),
        Rating_Prom=("Rating_Producto", "mean"),
        n=("Rating_Producto", "size"),
    )
    if "Ticket_Soporte" in df.columns:
        agg_kwargs["Tickets_Soporte_Pct"] = ("Ticket_Soporte", "mean")
    agg = df.groupby("Categoria").agg(**agg_kwargs).reset_index()
    if "Tickets_Soporte_Pct" in agg.columns:
        agg["Tickets_Soporte_Pct"] = (agg["Tickets_Soporte_Pct"] * 100).round(1)

    fig = px.scatter(
        agg, x="Stock_Prom", y="Rating_Prom", size="n",
        color="Tickets_Soporte_Pct" if "Tickets_Soporte_Pct" in agg.columns else None,
        text="Categoria", color_continuous_scale="RdYlGn_r",
        title="Paradoja de Fidelidad: Stock disponible vs Satisfacción del producto",
        labels={"Stock_Prom": "Stock promedio", "Rating_Prom": "Rating promedio del producto",
                "Tickets_Soporte_Pct": "% Tickets soporte"},
        size_max=45,
    )
    fig.update_traces(textposition="top center")
    fig.add_hline(y=agg["Rating_Prom"].mean(), line_dash="dash", opacity=0.4,
                   annotation_text="Rating promedio general")
    fig.add_vline(x=agg["Stock_Prom"].mean(), line_dash="dash", opacity=0.4,
                   annotation_text="Stock promedio general")
    return _style(fig)


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
    agg["Tasa_Soporte"] = (agg["Tasa_Soporte"] * 100).round(1)

    fig = px.scatter(
        agg, x="Antiguedad_Prom", y="Tasa_Soporte", size="n", text="Bodega_Origen",
        trendline="ols" if len(agg) >= 3 else None, color="Bodega_Origen",
        title="Bodegas operando 'a ciegas': antigüedad de revisión vs tasa de tickets de soporte",
        labels={"Antiguedad_Prom": "Antigüedad promedio última revisión (días)",
                "Tasa_Soporte": "Tasa de tickets de soporte (%)"},
        size_max=40,
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(showlegend=False)
    return _style(fig)
