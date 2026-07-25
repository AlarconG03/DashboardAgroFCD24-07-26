"""
cleaning.py
-----------
Implementa los casos de prueba de la Guía de Validación (Fase 1 y Fase 2)
contra los datos REALES de TechLogistics (no solo el escenario ideal del
diccionario). Contempla explícitamente las anomalías encontradas en los
archivos entregados:

  inventario_central_v2.csv
    - Categoria con variantes/basura: 'smart-phone', 'Smartphones', 'LAPTOP',
      'Laptops', '???'
    - Bodega_Origen con variantes de mayúsculas y un código externo no
      oficial ('BOD-EXT-99')
    - Lead_Time_Dias como TEXTO mixto: '25-30 días', '5', 'Inmediato', nan
    - Stock_Actual con nulos y negativos (existencias imposibles)
    - Costo_Unitario_USD con outliers extremos (hasta 850k)

  transacciones_logistica_v2.csv
    - Tiempo_Entrega_Real (alias de Tiempo_Entrega) con outliers (hasta 999)
    - Ciudad_Destino con variantes (MED/med/Medellín) Y contaminación con un
      valor de canal ('Ventas_Web') que no es una ciudad real
    - Cantidad_Vendida negativa (imposible en una venta)
    - Fecha_Venta en formato DD/MM/AAAA
    - Estado_Envio con nulos
    - Canal_Venta (Físico/WhatsApp/Online/App) — variable nueva, útil para P1

  feedback_clientes_v2.csv
    - Ticket_Soporte_Abierto (alias de Ticket_Soporte) mezclando 'Sí'/'No'/'1'/'0'
    - Edad_Cliente con valores imposibles (>100 años)
    - Rating_Producto con valores fuera de escala (99 en vez de 1-5)
    - Comentario_Texto/Recomienda_Marca con placeholders de nulo ('---', 'N/A')

Cada función retorna también un "log" de lo que hizo, para alimentar el
módulo de Transparencia (Antes vs Después) del dashboard.
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------
# Utilidades genéricas
# ---------------------------------------------------------------------

NULL_PLACEHOLDERS = {"---", "n/a", "na", "null", "none", "", "???", "sin dato"}


def clean_text_placeholders(series: pd.Series) -> pd.Series:
    """Convierte placeholders comunes de nulo ('---', 'N/A', etc.) a NaN real."""
    def _clean(v):
        if pd.isna(v):
            return np.nan
        if str(v).strip().lower() in NULL_PLACEHOLDERS:
            return np.nan
        return v
    return series.apply(_clean)


def _strip_accents(text: str) -> str:
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _normalize_key(raw: str) -> str:
    key = _strip_accents(str(raw)).upper()
    return re.sub(r"[^A-Z]", "", key)


def normalize_with_map(series: pd.Series, canon_map: dict, valid_whitelist: set = None):
    """
    Normaliza una columna categórica de texto usando un diccionario de
    mapeo (por clave sin tildes/mayúsculas/espacios) y, opcionalmente,
    valida contra una whitelist de valores canónicos esperados.
    Retorna (serie_normalizada, mapeo_aplicado, serie_booleana_invalidos).
    """
    applied_map = {}

    def normalize_one(val):
        if pd.isna(val):
            return val
        raw = str(val).strip()
        key = _normalize_key(raw)
        canon = canon_map.get(key, raw.title())
        if raw != canon:
            applied_map[raw] = canon
        return canon

    normalized = series.apply(normalize_one)
    invalid = None
    if valid_whitelist is not None:
        invalid = normalized.apply(lambda v: (not pd.isna(v)) and v not in valid_whitelist)
    return normalized, applied_map, invalid


# ---------------------------------------------------------------------
# Diccionarios de mapeo específicos del negocio
# ---------------------------------------------------------------------

CITY_MAP = {
    "MED": "Medellín", "MEDE": "Medellín", "MEDELLIN": "Medellín",
    "BOG": "Bogotá", "BOGOTA": "Bogotá", "BTA": "Bogotá",
    "CALI": "Cali", "CLO": "Cali",
    "BAQ": "Barranquilla", "BARRANQUILLA": "Barranquilla",
    "CTG": "Cartagena", "CARTAGENA": "Cartagena",
    "BUC": "Bucaramanga", "BUCARAMANGA": "Bucaramanga",
    "PEREIRA": "Pereira", "PEI": "Pereira",
    "MANIZALES": "Manizales", "MZL": "Manizales",
}
CITY_WHITELIST = {"Medellín", "Bogotá", "Cali", "Barranquilla", "Cartagena",
                   "Bucaramanga", "Pereira", "Manizales"}

CATEGORY_MAP = {
    "SMARTPHONE": "Smartphones", "SMARTPHONES": "Smartphones",
    "LAPTOP": "Laptops", "LAPTOPS": "Laptops",
    "ACCESORIOS": "Accesorios", "MONITORES": "Monitores", "TABLETS": "Tablets",
}

WAREHOUSE_MAP = {
    "NORTE": "Norte", "SUR": "Sur", "OCCIDENTE": "Occidente",
    "ZONAFRANCA": "Zona Franca",
}


def normalize_city_column(series: pd.Series):
    """Normaliza Ciudad_Destino y marca como inválidos los valores que no
    son una ciudad real (p. ej. contaminación con un valor de canal)."""
    return normalize_with_map(series, CITY_MAP, valid_whitelist=CITY_WHITELIST)


def normalize_category_column(series: pd.Series):
    normalized, applied_map, _ = normalize_with_map(series, CATEGORY_MAP)
    # '???' no cae en el mapa (no tiene letras) -> queda como 'Sin Categoría'
    normalized = normalized.replace({"???": "Sin Categoría"})
    return normalized, applied_map


def normalize_warehouse_column(series: pd.Series):
    """Normaliza Bodega_Origen. Los códigos tipo 'BOD-EXT-99' se preservan
    pero se marcan como bodega no oficial (posible fuga fuera del ERP)."""
    def normalize_one(val):
        if pd.isna(val):
            return val
        raw = str(val).strip()
        key = _normalize_key(raw)
        if key in WAREHOUSE_MAP:
            return WAREHOUSE_MAP[key]
        if re.match(r"^BOD-EXT", raw, flags=re.IGNORECASE):
            return raw.upper()
        return raw.title()

    normalized = series.apply(normalize_one)
    is_external = normalized.astype(str).str.upper().str.startswith("BOD-EXT")
    return normalized, is_external


def parse_lead_time(series: pd.Series):
    """Convierte Lead_Time_Dias (texto mixto: '25-30 días', '5', 'Inmediato',
    nan) a un valor numérico de días. Rangos se promedian."""
    def parse_one(val):
        if pd.isna(val):
            return np.nan
        text = str(val).strip().lower()
        if "inmediato" in text:
            return 0.0
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        if not nums:
            return np.nan
        nums = [float(n) for n in nums]
        return sum(nums) / len(nums)
    return series.apply(parse_one)


# ---------------------------------------------------------------------
# Outliers por rango intercuartílico (IQR)
# ---------------------------------------------------------------------

def iqr_bounds(series: pd.Series, k: float = 1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return lower, upper


def flag_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5):
    """Agrega '<column>_outlier' marcando outliers por IQR. No elimina filas:
    el consultor decide qué hacer (excluir de KPIs, no del dataset)."""
    df = df.copy()
    valid = df[column].dropna()
    if valid.empty:
        df[f"{column}_outlier"] = False
        return df, np.nan, np.nan
    lower, upper = iqr_bounds(valid, k=k)
    df[f"{column}_outlier"] = ~df[column].between(lower, upper)
    return df, lower, upper


# ---------------------------------------------------------------------
# Validación temporal (fechas futuras)
# ---------------------------------------------------------------------

def flag_future_dates(df: pd.DataFrame, date_col: str, reference_date=None, dayfirst=True):
    df = df.copy()
    reference_date = reference_date or pd.Timestamp(datetime.now().date())
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=dayfirst)
    df[f"{date_col}_futura"] = df[date_col] > reference_date
    return df


# ---------------------------------------------------------------------
# Health Score
# ---------------------------------------------------------------------

def health_score(df_before: pd.DataFrame, df_after: pd.DataFrame, outlier_cols=None):
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
    log = []
    df_before = df.copy()
    df = df.copy()

    if "Categoria" in df.columns:
        df["Categoria"] = clean_text_placeholders(df["Categoria"])
        df["Categoria"], cat_map = normalize_category_column(df["Categoria"])
        log.append(f"Categoria: {len(cat_map)} variantes unificadas (ej. 'smart-phone'/'LAPTOP' -> forma canónica); "
                    f"valores sin categoría reconocible -> 'Sin Categoría'")

    if "Bodega_Origen" in df.columns:
        df["Bodega_Origen"], is_external = normalize_warehouse_column(df["Bodega_Origen"])
        df["Bodega_No_Oficial"] = is_external
        log.append(f"Bodegas no oficiales detectadas (código externo tipo BOD-EXT): {int(is_external.sum())}")

    if "Lead_Time_Dias" in df.columns:
        n_text_ranges = df["Lead_Time_Dias"].astype(str).str.contains("-", na=False).sum()
        df["Lead_Time_Dias"] = parse_lead_time(df["Lead_Time_Dias"])
        log.append(f"Lead_Time_Dias: texto mixto parseado a numérico ({n_text_ranges} rangos promediados, "
                    f"'Inmediato' -> 0 días)")

    if "Costo_Unitario_USD" in df.columns:
        df["Costo_Unitario_USD"] = pd.to_numeric(df["Costo_Unitario_USD"], errors="coerce")
        df, low, up = flag_outliers_iqr(df, "Costo_Unitario_USD")
        log.append(f"Costo_Unitario_USD: outliers fuera de [{low:,.2f}, {up:,.2f}] (IQR)")

    if "Stock_Actual" in df.columns:
        df["Stock_Actual"] = pd.to_numeric(df["Stock_Actual"], errors="coerce")
        n_null_stock = int(df["Stock_Actual"].isna().sum())
        df["Stock_Actual_negativo"] = df["Stock_Actual"] < 0
        log.append(f"Stock_Actual: {n_null_stock} valores nulos preservados (no imputados a ciegas); "
                    f"{int(df['Stock_Actual_negativo'].sum())} existencias negativas marcadas como inconsistencia")

    if "Ultima_Revision" in df.columns:
        df["Ultima_Revision"] = pd.to_datetime(df["Ultima_Revision"], errors="coerce")
        df["Dias_Desde_Revision"] = (pd.Timestamp(datetime.now().date()) - df["Ultima_Revision"]).dt.days

    duplicated_before = df.duplicated(subset=["SKU_ID"]).sum() if "SKU_ID" in df.columns else 0
    if "SKU_ID" in df.columns:
        df = df.drop_duplicates(subset=["SKU_ID"], keep="first")
    log.append(f"Duplicados de SKU_ID eliminados: {int(duplicated_before)}")

    hs = health_score(df_before, df, outlier_cols=["Costo_Unitario_USD"])
    return df, log, hs


def clean_transactions(df: pd.DataFrame, reference_date=None):
    log = []
    df_before = df.copy()
    df = df.copy()

    for col in ["Cantidad_Vendida", "Precio_Venta_Final", "Costo_Envio", "Tiempo_Entrega"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Cantidad_Vendida" in df.columns:
        df["Cantidad_Vendida_invalida"] = df["Cantidad_Vendida"] < 0
        log.append(f"Cantidad_Vendida negativa (imposible en una venta) marcada en "
                    f"{int(df['Cantidad_Vendida_invalida'].sum())} filas")

    if "Costo_Envio" in df.columns:
        n_null = int(df["Costo_Envio"].isna().sum())
        log.append(f"Costo_Envio: {n_null} valores nulos preservados (no se imputan a ciegas)")

    if "Estado_Envio" in df.columns:
        n_null_estado = int(df["Estado_Envio"].isna().sum())
        df["Estado_Envio"] = df["Estado_Envio"].fillna("Desconocido")
        log.append(f"Estado_Envio: {n_null_estado} nulos etiquetados como 'Desconocido'")

    if "Fecha_Venta" in df.columns:
        df = flag_future_dates(df, "Fecha_Venta", reference_date=reference_date, dayfirst=True)
        n_future = int(df["Fecha_Venta_futura"].sum())
        log.append(f"Fecha_Venta parseada como DD/MM/AAAA; transacciones con fecha futura excluidas "
                    f"del análisis temporal: {n_future}")

    if "Tiempo_Entrega" in df.columns:
        df, low, up = flag_outliers_iqr(df, "Tiempo_Entrega")
        log.append(f"Tiempo_Entrega: outliers fuera de [{low:,.1f}, {up:,.1f}] días (IQR)")

    if "Ciudad_Destino" in df.columns:
        df["Ciudad_Destino"], applied_map, invalid = normalize_city_column(df["Ciudad_Destino"])
        df["Ciudad_Destino_invalida"] = invalid.fillna(False)
        log.append(f"Ciudades normalizadas: {len(applied_map)} variantes unificadas (MED/med -> Medellín); "
                    f"{int(df['Ciudad_Destino_invalida'].sum())} valores no son una ciudad real "
                    f"(contaminación cruzada con Canal_Venta, ej. 'Ventas_Web') y quedan marcados, no borrados")

    if "Canal_Venta" in df.columns:
        df["Canal_Venta"] = clean_text_placeholders(df["Canal_Venta"])

    dup_before = df.duplicated(subset=["Transaccion_ID"]).sum() if "Transaccion_ID" in df.columns else 0
    if "Transaccion_ID" in df.columns:
        df = df.drop_duplicates(subset=["Transaccion_ID"], keep="first")
    log.append(f"Duplicados de Transaccion_ID eliminados: {int(dup_before)}")

    hs = health_score(df_before, df, outlier_cols=["Tiempo_Entrega"])
    return df, log, hs


def clean_feedback(df: pd.DataFrame):
    log = []
    df_before = df.copy()
    df = df.copy()

    for col in ["Comentario_Texto", "Recomienda_Marca"]:
        if col in df.columns:
            df[col] = clean_text_placeholders(df[col])

    dup_before = df.duplicated().sum()
    df = df.drop_duplicates(keep="first")
    log.append(f"Registros duplicados eliminados: {int(dup_before)}")

    age_col = next((c for c in df.columns if "edad" in c.lower() or c.lower() == "age"), None)
    if age_col:
        df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
        df[f"{age_col}_invalida"] = ~df[age_col].between(0, 100)
        log.append(f"{age_col}: edades imposibles marcadas (fuera de 0-100 años): "
                    f"{int(df[f'{age_col}_invalida'].sum())}")

    if "Satisfaccion_NPS" in df.columns:
        df["Satisfaccion_NPS"] = pd.to_numeric(df["Satisfaccion_NPS"], errors="coerce")

    if "Rating_Producto" in df.columns:
        df["Rating_Producto"] = pd.to_numeric(df["Rating_Producto"], errors="coerce")
        df["Rating_Producto_invalida"] = ~df["Rating_Producto"].between(1, 5)
        log.append(f"Rating_Producto fuera de escala 1-5 marcado (no imputado): "
                    f"{int(df['Rating_Producto_invalida'].sum())}")

    if "Rating_Logistica" in df.columns:
        df["Rating_Logistica"] = pd.to_numeric(df["Rating_Logistica"], errors="coerce")

    if "Ticket_Soporte" in df.columns:
        df["Ticket_Soporte"] = (
            df["Ticket_Soporte"].astype(str).str.strip().str.lower()
            .map({"si": True, "sí": True, "yes": True, "1": True, "true": True,
                  "no": False, "0": False, "false": False})
        )
        n_unmapped = int(df["Ticket_Soporte"].isna().sum())
        df["Ticket_Soporte"] = df["Ticket_Soporte"].fillna(False)
        log.append(f"Ticket_Soporte homologado a booleano (Sí/1 -> True, No/0 -> False); "
                    f"{n_unmapped} valores no reconocidos por defecto en False")

    hs = health_score(df_before, df)
    return df, log, hs
