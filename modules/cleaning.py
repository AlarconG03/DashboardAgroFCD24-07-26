"""
cleaning.py
-----------
Implementa los casos de prueba de la Guía de Validación (Fase 1 y Fase 2):
- Normalización de ciudades (MED, med, Medellín -> Medellín)
- Detección de outliers por IQR (costos, tiempos de entrega)
- Validación de fechas futuras
- Cálculo de Health Score (antes vs después)
- Deduplicación y edades imposibles en feedback

Cada función retorna también un "log" de lo que hizo, para alimentar el
módulo de Transparencia (Antes vs Después) del dashboard.
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------
# Normalización de ciudades
# ---------------------------------------------------------------------

# Diccionario de mapeo base. Se complementa con una normalización por
# regex (mayúsculas, sin tildes, sin espacios extra) para capturar variantes
# no listadas explícitamente.
CITY_MAP = {
    "MED": "Medellín", "MEDE": "Medellín", "MEDELLIN": "Medellín", "MEDELLÍN": "Medellín",
    "BOG": "Bogotá", "BOGOTA": "Bogotá", "BOGOTÁ": "Bogotá", "BTA": "Bogotá",
    "CALI": "Cali", "CLO": "Cali",
    "BAQ": "Barranquilla", "BARRANQUILLA": "Barranquilla",
    "CTG": "Cartagena", "CARTAGENA": "Cartagena",
    "BUC": "Bucaramanga", "BUCARAMANGA": "Bucaramanga",
    "PEREIRA": "Pereira", "PEI": "Pereira",
    "MANIZALES": "Manizales", "MZL": "Manizales",
}


def _strip_accents(text: str) -> str:
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def normalize_city_column(series: pd.Series):
    """
    Normaliza nombres de ciudad a una única forma canónica por región.
    Retorna (serie_normalizada, mapeo_aplicado_dict).
    """
    applied_map = {}

    def normalize_one(val):
        if pd.isna(val):
            return val
        raw = str(val).strip()
        key = _strip_accents(raw).upper()
        key = re.sub(r"[^A-Z]", "", key)
        canon = CITY_MAP.get(key, raw.title())
        if raw != canon:
            applied_map[raw] = canon
        return canon

    normalized = series.apply(normalize_one)
    return normalized, applied_map


# ---------------------------------------------------------------------
# Outliers por rango intercuartílico (IQR)
# ---------------------------------------------------------------------

def iqr_bounds(series: pd.Series, k: float = 1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper


def flag_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5):
    """
    Agrega una columna booleana '<column>_outlier' marcando outliers por IQR.
    No elimina filas: el consultor decide qué hacer (excluir de KPIs, no del dataset).
    """
    df = df.copy()
    lower, upper = iqr_bounds(df[column].dropna(), k=k)
    df[f"{column}_outlier"] = ~df[column].between(lower, upper)
    return df, lower, upper


# ---------------------------------------------------------------------
# Validación temporal (fechas futuras)
# ---------------------------------------------------------------------

def flag_future_dates(df: pd.DataFrame, date_col: str, reference_date=None):
    """
    Convierte date_col a datetime y marca como inválidas las fechas
    posteriores a la fecha de referencia (por defecto: hoy).
    """
    df = df.copy()
    reference_date = reference_date or pd.Timestamp(datetime.now().date())
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)
    df[f"{date_col}_futura"] = df[date_col] > reference_date
    return df


# ---------------------------------------------------------------------
# Health Score
# ---------------------------------------------------------------------

def health_score(df_before: pd.DataFrame, df_after: pd.DataFrame, outlier_cols=None):
    """
    Calcula métricas de salud de datos: % nulidad por columna (antes/después),
    duplicados eliminados y magnitud de outliers detectados.
    """
    outlier_cols = outlier_cols or []

    null_before = (df_before.isna().mean() * 100).round(2)
    null_after = (df_after.isna().mean() * 100).round(2)

    duplicates_removed = int(df_before.duplicated().sum())

    outlier_summary = {}
    for col in outlier_cols:
        flag_col = f"{col}_outlier"
        if flag_col in df_after.columns:
            outlier_summary[col] = int(df_after[flag_col].sum())

    total_before, total_after = len(df_before), len(df_after)
    pct_health = round(100 * (1 - (df_after.isna().sum().sum() / max(df_after.size, 1))), 2)

    return {
        "registros_antes": total_before,
        "registros_despues": total_after,
        "nulidad_antes_pct": null_before,
        "nulidad_despues_pct": null_after,
        "duplicados_eliminados": duplicates_removed,
        "outliers_por_columna": outlier_summary,
        "health_score_pct": pct_health,
    }


# ---------------------------------------------------------------------
# Pipelines de limpieza por dataset
# ---------------------------------------------------------------------

def clean_inventory(df: pd.DataFrame):
    """
    Limpieza de inventario_central:
    - Costo_Unitario_USD: outliers por IQR (marcados, no eliminados de raíz)
    - Stock_Actual negativo: se marca como inconsistencia (alerta operativa)
    - Ultima_Revision a datetime
    """
    log = []
    df_before = df.copy()
    df = df.copy()

    if "Costo_Unitario_USD" in df.columns:
        df["Costo_Unitario_USD"] = pd.to_numeric(df["Costo_Unitario_USD"], errors="coerce")
        df, low, up = flag_outliers_iqr(df, "Costo_Unitario_USD")
        log.append(f"Costo_Unitario_USD: outliers fuera de [{low:,.2f}, {up:,.2f}] (IQR)")

    if "Stock_Actual" in df.columns:
        df["Stock_Actual"] = pd.to_numeric(df["Stock_Actual"], errors="coerce")
        df["Stock_Actual_negativo"] = df["Stock_Actual"] < 0
        log.append(f"Stock_Actual negativo detectado en {int(df['Stock_Actual_negativo'].sum())} filas")

    if "Ultima_Revision" in df.columns:
        df["Ultima_Revision"] = pd.to_datetime(df["Ultima_Revision"], errors="coerce")
        df["Dias_Desde_Revision"] = (pd.Timestamp(datetime.now().date()) - df["Ultima_Revision"]).dt.days

    if "Lead_Time_Dias" in df.columns:
        df["Lead_Time_Dias"] = pd.to_numeric(df["Lead_Time_Dias"], errors="coerce")

    duplicated_before = df.duplicated(subset=["SKU_ID"]).sum() if "SKU_ID" in df.columns else 0
    if "SKU_ID" in df.columns:
        df = df.drop_duplicates(subset=["SKU_ID"], keep="first")
    log.append(f"Duplicados de SKU_ID eliminados: {int(duplicated_before)}")

    hs = health_score(df_before, df, outlier_cols=["Costo_Unitario_USD"])
    return df, log, hs


def clean_transactions(df: pd.DataFrame, reference_date=None):
    """
    Limpieza de transacciones_logistica:
    - Fecha_Venta: parseo + marca de fechas futuras
    - Tiempo_Entrega: outliers por IQR (hasta 999 días)
    - Ciudad_Destino: normalización
    - Cantidad_Vendida / Precio_Venta_Final / Costo_Envio: a numérico
    """
    log = []
    df_before = df.copy()
    df = df.copy()

    for col in ["Cantidad_Vendida", "Precio_Venta_Final", "Costo_Envio", "Tiempo_Entrega"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Fecha_Venta" in df.columns:
        df = flag_future_dates(df, "Fecha_Venta", reference_date=reference_date)
        n_future = int(df["Fecha_Venta_futura"].sum())
        log.append(f"Transacciones con fecha futura excluidas del análisis temporal: {n_future}")

    if "Tiempo_Entrega" in df.columns:
        df, low, up = flag_outliers_iqr(df, "Tiempo_Entrega")
        log.append(f"Tiempo_Entrega: outliers fuera de [{low:,.1f}, {up:,.1f}] días (IQR)")

    if "Ciudad_Destino" in df.columns:
        df["Ciudad_Destino"], applied_map = normalize_city_column(df["Ciudad_Destino"])
        log.append(f"Ciudades normalizadas: {len(applied_map)} variantes unificadas")

    dup_before = df.duplicated(subset=["Transaccion_ID"]).sum() if "Transaccion_ID" in df.columns else 0
    if "Transaccion_ID" in df.columns:
        df = df.drop_duplicates(subset=["Transaccion_ID"], keep="first")
    log.append(f"Duplicados de Transaccion_ID eliminados: {int(dup_before)}")

    hs = health_score(df_before, df, outlier_cols=["Tiempo_Entrega"])
    return df, log, hs


def clean_feedback(df: pd.DataFrame):
    """
    Limpieza de feedback_clientes:
    - Deduplicación de registros intencionalmente duplicados
    - Edades imposibles (si existe columna Edad): se marcan, no se imputan a ciegas
    - Ticket_Soporte -> booleano estandarizado
    - Satisfaccion_NPS -> numérico
    """
    log = []
    df_before = df.copy()
    df = df.copy()

    dup_before = df.duplicated().sum()
    df = df.drop_duplicates(keep="first")
    log.append(f"Registros duplicados eliminados: {int(dup_before)}")

    age_col = next((c for c in df.columns if c.lower() in ("edad", "age")), None)
    if age_col:
        df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
        df[f"{age_col}_invalida"] = ~df[age_col].between(0, 100)
        log.append(f"Edades imposibles marcadas (fuera de 0-100): {int(df[f'{age_col}_invalida'].sum())}")

    if "Satisfaccion_NPS" in df.columns:
        df["Satisfaccion_NPS"] = pd.to_numeric(df["Satisfaccion_NPS"], errors="coerce")

    if "Ticket_Soporte" in df.columns:
        df["Ticket_Soporte"] = (
            df["Ticket_Soporte"].astype(str).str.strip().str.lower()
            .map({"si": True, "sí": True, "yes": True, "1": True, "true": True})
            .fillna(False)
        )

    for col in ["Rating_Producto", "Rating_Logistica"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    hs = health_score(df_before, df)
    return df, log, hs
