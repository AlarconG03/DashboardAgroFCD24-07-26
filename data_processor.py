import pandas as pd
import numpy as np
from datetime import datetime

def calcular_health_score(df):
    total_filas = len(df)
    if total_filas == 0: return 0
    filas_completas = df.dropna().shape[0]
    return round((filas_completas / total_filas) * 100, 2)

def procesar_datos(f_inv, f_trans, f_feed):
    # 1. Carga de datos (aceptando directamente los archivos de Streamlit)
    inv = pd.read_csv(f_inv)
    trans = pd.read_csv(f_trans)
    feed = pd.read_csv(f_feed)

    health_antes = {
        "Inventario": calcular_health_score(inv),
        "Transacciones": calcular_health_score(trans),
        "Feedback": calcular_health_score(feed)
    }

    # 2. Limpieza de Inventario
    inv = inv.drop_duplicates()
    inv['Stock_Actual'] = np.where(inv['Stock_Actual'] < 0, 0, inv['Stock_Actual'])
    inv['Ultima_Revision'] = pd.to_datetime(inv['Ultima_Revision'], errors='coerce')
    
    # Tratamiento de Outliers de Costo con IQR
    Q1 = inv['Costo_Unitario_USD'].quantile(0.25)
    Q3 = inv['Costo_Unitario_USD'].quantile(0.75)
    IQR = Q3 - Q1
    limite_sup = Q3 + 1.5 * IQR
    inv['Es_Outlier_Costo'] = np.where(inv['Costo_Unitario_USD'] > limite_sup, True, False)

    # 3. Limpieza de Transacciones
    trans = trans.drop_duplicates()
    trans['Fecha_Venta'] = pd.to_datetime(trans['Fecha_Venta'], errors='coerce')
    # Filtro de fechas futuras
    trans = trans[trans['Fecha_Venta'] <= pd.to_datetime(datetime.now())]
    
    # Normalización de variables categóricas (Ciudades)
    diccionario_ciudades = {
        r'(?i)^med.*': 'Medellín',
        r'(?i)^bog.*': 'Bogotá',
        r'(?i)^cal.*': 'Cali',
        r'(?i)^bar.*': 'Barranquilla'
    }
    for patron, reemplazo in diccionario_ciudades.items():
        trans['Ciudad_Destino'] = trans['Ciudad_Destino'].astype(str).str.replace(patron, reemplazo, regex=True)

    # 4. Limpieza de Feedback
    feed = feed.drop_duplicates()
    feed['Satisfaccion_NPS'] = pd.to_numeric(feed['Satisfaccion_NPS'], errors='coerce')
    
    # 5. Integración (Left Join Estratégico)
    df = trans.merge(inv, on='SKU_ID', how='left', indicator='_merge_inv')
    df['Venta_Fantasma'] = df['_merge_inv'] == 'left_only'
    
    df = df.merge(feed, on='Transaccion_ID', how='left')

    # 6. Feature Engineering (Variables Derivadas)
    df['Ingreso_Total'] = df['Precio_Venta_Final'] * df['Cantidad_Vendida']
    df['Costo_Total'] = df['Costo_Unitario_USD'] * df['Cantidad_Vendida']
    df['Margen_Utilidad'] = df['Ingreso_Total'] - df['Costo_Total'] - df['Costo_Envio']
    
    # CORRECCIÓN DE NOMBRES DE VARIABLES
    df['Brecha_Entrega'] = df['Tiempo_Entrega_Real'] - df['Lead_Time_Dias']
    
    # Limpieza robusta del ticket (Maneja variaciones como 'Sí', 'Si', '1', '0', 'No')
    df['Tiene_Ticket'] = df['Ticket_Soporte_Abierto'].astype(str).str.lower().isin(['si', 'sí', '1', 'true']).astype(int)
    
    df['Dias_Desde_Revision'] = (pd.to_datetime(datetime.now()) - df['Ultima_Revision']).dt.days

    health_despues = {
        "Global_Integrado": calcular_health_score(df)
    }

    return df, inv[inv['Es_Outlier_Costo'] == True], health_antes, health_despues
