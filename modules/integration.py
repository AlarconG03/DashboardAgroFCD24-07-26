"""
integration.py
---------------
Fase 2 del pipeline: integración (Left Join estratégico) para construir
la "Sola Fuente de Verdad", incluyendo el diagnóstico de la Venta Fantasma
(SKUs vendidos que no existen en el inventario oficial).
"""

import pandas as pd


def merge_master(inv_df: pd.DataFrame, trans_df: pd.DataFrame, fb_df: pd.DataFrame):
    """
    Left Join estratégico:
      transacciones (base) <- inventario (por SKU_ID) <- feedback (por Transaccion_ID)

    Las ventas cuyo SKU_ID no existe en inventario NO se descartan: se etiquetan
    como 'Venta Fantasma' para que el negocio decida (producto no catalogado
    vs. error de digitación vs. fraude), tal como exige el Criterio de Aceptación
    de Integridad de Identidad de la Guía de Validación.
    """
    master = trans_df.merge(
        inv_df, on="SKU_ID", how="left", suffixes=("", "_inv"), indicator=True
    )
    master["Venta_Fantasma"] = master["_merge"] == "left_only"
    master = master.drop(columns=["_merge"])

    if fb_df is not None and "Transaccion_ID" in fb_df.columns:
        master = master.merge(fb_df, on="Transaccion_ID", how="left", suffixes=("", "_fb"))

    return master


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
