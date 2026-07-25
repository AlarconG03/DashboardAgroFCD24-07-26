"""
integration.py
---------------
Fase 2 del pipeline: integración (Left Join estratégico) para construir
la "Sola Fuente de Verdad", incluyendo el diagnóstico de la Venta Fantasma
(SKUs vendidos que no existen en el inventario oficial).
"""

import pandas as pd


def _aggregate_feedback_per_transaction(fb_df: pd.DataFrame):
    """
    Integridad de Identidad (Merging): algunos Transaccion_ID tienen MÁS DE UN
    registro de feedback (p. ej. un cliente comenta y luego abre un ticket
    aparte). Hacer un Left Join directo sobre esa llave no única duplicaría
    las filas de venta y, con ellas, el ingreso -rompiendo la trazabilidad
    exigida por la Guía de Validación-. Por eso se agrega feedback a UNA fila
    por Transaccion_ID antes de integrarlo: promedio para ratings/NPS, 'OR'
    lógico para Ticket_Soporte (si al menos un registro abrió ticket, cuenta
    como ticket abierto), y el primer valor no nulo para texto libre.
    """
    if fb_df is None or "Transaccion_ID" not in fb_df.columns:
        return fb_df, 0

    n_dup_transacciones = int(fb_df["Transaccion_ID"].duplicated().sum())
    if n_dup_transacciones == 0:
        return fb_df, 0

    agg_map = {}
    for col in fb_df.columns:
        if col == "Transaccion_ID":
            continue
        if pd.api.types.is_bool_dtype(fb_df[col]):
            agg_map[col] = "max"
        elif pd.api.types.is_numeric_dtype(fb_df[col]):
            agg_map[col] = "mean"
        else:
            agg_map[col] = "first"

    fb_agg = fb_df.groupby("Transaccion_ID", as_index=False).agg(agg_map)
    return fb_agg, n_dup_transacciones


def merge_master(inv_df: pd.DataFrame, trans_df: pd.DataFrame, fb_df: pd.DataFrame):
    """
    Left Join estratégico:
      transacciones (base) <- inventario (por SKU_ID) <- feedback (por Transaccion_ID, agregado)

    Las ventas cuyo SKU_ID no existe en inventario NO se descartan: se etiquetan
    como 'Venta Fantasma' para que el negocio decida (producto no catalogado
    vs. error de digitación vs. fraude), tal como exige el Criterio de Aceptación
    de Integridad de Identidad de la Guía de Validación.

    Retorna (master, merge_log) donde merge_log es una lista de strings para
    el módulo de Transparencia.
    """
    merge_log = []
    rows_before = len(trans_df)

    master = trans_df.merge(
        inv_df, on="SKU_ID", how="left", suffixes=("", "_inv"), indicator=True
    )
    master["Venta_Fantasma"] = master["_merge"] == "left_only"
    master = master.drop(columns=["_merge"])
    merge_log.append(
        f"Left Join transacciones<-inventario por SKU_ID: {int(master['Venta_Fantasma'].sum())} "
        f"ventas etiquetadas como Venta Fantasma (SKU sin catálogo oficial)"
    )

    # Las ventas fantasma no tienen Categoria/Bodega_Origen (no hay match en
    # inventario). En vez de dejarlas como NaN -y que desaparezcan de los
    # groupby- se etiquetan explícitamente para que sigan siendo visibles
    # en los análisis por categoría/bodega.
    if "Categoria" in master.columns:
        master["Categoria"] = master["Categoria"].fillna("Sin Categoría (Venta Fantasma)")
    if "Bodega_Origen" in master.columns:
        master["Bodega_Origen"] = master["Bodega_Origen"].fillna("Sin Bodega (Venta Fantasma)")

    fb_agg, n_dup = _aggregate_feedback_per_transaction(fb_df)
    if n_dup:
        merge_log.append(
            f"Feedback: {n_dup} Transaccion_ID tenían más de un registro de feedback; "
            f"se agregaron a 1 fila por transacción ANTES del join para no duplicar ingresos"
        )
    if fb_agg is not None and "Transaccion_ID" in fb_agg.columns:
        master = master.merge(fb_agg, on="Transaccion_ID", how="left", suffixes=("", "_fb"))

    rows_after = len(master)
    merge_log.append(
        f"Trazabilidad: {rows_before:,} transacciones antes del join -> {rows_after:,} filas después "
        f"({'OK, sin fan-out' if rows_after == rows_before else 'ALERTA: el conteo cambió, revisar llaves de join'})"
    )

    return master, merge_log


def phantom_sales_summary(master: pd.DataFrame, revenue_col: str = "Precio_Venta_Final"):
    """
    Cuantifica el impacto financiero de las Ventas Fantasma:
    monto en riesgo y % del ingreso total.
    """
    total_revenue = master[revenue_col].sum()
    phantom_revenue = master.loc[master["Venta_Fantasma"], revenue_col].sum()
    pct = round(100 * phantom_revenue / total_revenue, 2) if total_revenue else 0.0

    return {
        "ingreso_total": total_revenue,
        "ingreso_venta_fantasma": phantom_revenue,
        "pct_ingreso_en_riesgo": pct,
        "num_transacciones_fantasma": int(master["Venta_Fantasma"].sum()),
        "num_transacciones_total": len(master),
    }
