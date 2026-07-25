"""
data_loader.py
--------------
Carga y validación estructural de los 3 datasets de TechLogistics S.A.
No limpia datos: solo los lee y verifica que el esquema mínimo esperado
(según el Diccionario de Datos) esté presente. La limpieza vive en cleaning.py.
"""

import pandas as pd
import streamlit as st

# Esquema mínimo esperado por dataset, según el Diccionario de Datos del Challenge 02.
SCHEMAS = {
    "inventario": [
        "SKU_ID", "Categoria", "Stock_Actual", "Costo_Unitario_USD",
        "Punto_Reorden", "Lead_Time_Dias", "Bodega_Origen", "Ultima_Revision",
    ],
    "transacciones": [
        "Transaccion_ID", "SKU_ID", "Fecha_Venta", "Cantidad_Vendida",
        "Precio_Venta_Final", "Costo_Envio", "Tiempo_Entrega",
        "Estado_Envio", "Ciudad_Destino",
    ],
    "feedback": [
        "Feedback_ID", "Transaccion_ID", "Rating_Producto",
        "Rating_Logistica", "Satisfaccion_NPS", "Ticket_Soporte",
    ],
}


@st.cache_data(show_spinner=False)
def read_csv(file) -> pd.DataFrame:
    """Lee un CSV subido por el usuario probando separadores/encodings comunes."""
    for sep in [",", ";", "\t"]:
        try:
            file.seek(0)
            df = pd.read_csv(file, sep=sep, encoding="utf-8")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    file.seek(0)
    return pd.read_csv(file, sep=None, engine="python")


def validate_schema(df: pd.DataFrame, dataset_key: str) -> list:
    """Devuelve la lista de columnas esperadas que faltan en el dataframe."""
    expected = SCHEMAS.get(dataset_key, [])
    missing = [c for c in expected if c not in df.columns]
    return missing


def load_all_datasets(inv_file, trans_file, fb_file):
    """
    Carga los tres archivos subidos por el usuario en la barra lateral.
    Retorna un diccionario {nombre: dataframe} y un diccionario de advertencias
    de esquema para mostrar en el módulo de transparencia.
    """
    datasets = {}
    warnings = {}

    if inv_file is not None:
        df = read_csv(inv_file)
        datasets["inventario"] = df
        warnings["inventario"] = validate_schema(df, "inventario")

    if trans_file is not None:
        df = read_csv(trans_file)
        datasets["transacciones"] = df
        warnings["transacciones"] = validate_schema(df, "transacciones")

    if fb_file is not None:
        df = read_csv(fb_file)
        datasets["feedback"] = df
        warnings["feedback"] = validate_schema(df, "feedback")

    return datasets, warnings
