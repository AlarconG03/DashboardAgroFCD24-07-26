"""
features.py
------------
Variables derivadas (Fase 2, Feature Engineering) requeridas por el Challenge:
  1. Margen de Utilidad
  2. Brecha de Entrega vs Prometido
  3. Ratio de Soporte por Categoría

Nota de supuesto de negocio: el diccionario de datos no incluye un campo de
"días prometidos de entrega" al cliente. Se asume una SLA configurable desde
el sidebar (por defecto 5 días) como benchmark de servicio, documentado aquí
y en el README. Ajuste `sla_dias` si TechLogistics define otro valor oficial.
"""

import pandas as pd


def add_margin(df: pd.DataFrame) -> pd.DataFrame:
    """Margen de Utilidad = (Precio_Venta_Final - Costo_Unitario_USD - Costo_Envio) / Precio_Venta_Final."""
    df = df.copy()
    ingreso = df.get("Precio_Venta_Final", pd.Series(dtype=float))
    costo_unitario = df.get("Costo_Unitario_USD", 0)
    costo_envio = df.get("Costo_Envio", 0)

    df["Costo_Total_Transaccion"] = costo_unitario.fillna(0) + costo_envio.fillna(0)
    df["Margen_Absoluto_USD"] = ingreso - df["Costo_Total_Transaccion"]
    df["Margen_Pct"] = (df["Margen_Absoluto_USD"] / ingreso.replace(0, pd.NA)) * 100
    return df


def add_delivery_gap(df: pd.DataFrame, sla_dias: int = 5) -> pd.DataFrame:
    """Brecha_Entrega_Dias = Tiempo_Entrega real - SLA prometido (supuesto de negocio)."""
    df = df.copy()
    if "Tiempo_Entrega" in df.columns:
        df["Brecha_Entrega_Dias"] = df["Tiempo_Entrega"] - sla_dias
        df["Entrega_Fuera_SLA"] = df["Brecha_Entrega_Dias"] > 0
    return df


def add_support_ratio_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio_Soporte_Categoria: % de transacciones con Ticket_Soporte=True dentro
    de cada Categoria. Se agrega como columna unida al dataframe maestro.
    """
    df = df.copy()
    if "Categoria" in df.columns and "Ticket_Soporte" in df.columns:
        ratio = (
            df.groupby("Categoria")["Ticket_Soporte"]
            .mean()
            .mul(100)
            .rename("Ratio_Soporte_Categoria_Pct")
        )
        df = df.merge(ratio, on="Categoria", how="left")
    return df


def build_features(master: pd.DataFrame, sla_dias: int = 5) -> pd.DataFrame:
    """Aplica las tres variables derivadas sobre el dataframe maestro integrado."""
    df = add_margin(master)
    df = add_delivery_gap(df, sla_dias=sla_dias)
    df = add_support_ratio_by_category(df)
    return df
