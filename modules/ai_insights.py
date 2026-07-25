"""
ai_insights.py
---------------
Integración con Groq (modelo Llama-3) para generar recomendaciones
estratégicas en tiempo real, basadas EXCLUSIVAMENTE en el resumen
estadístico de los datos ya filtrados por el usuario en el sidebar
(cumple el Criterio de Aceptación de la Guía de Validación: la IA no
debe alucinar sobre datos que el usuario no seleccionó).

La API Key se lee desde st.secrets, nunca hardcodeada en el código.
"""

import json
import requests
import streamlit as st

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Modelo Llama-3 servido por Groq. Ajustar aquí si Groq deprecida el modelo;
# ver https://console.groq.com/docs/models para el nombre vigente.
GROQ_MODEL = "llama-3.3-70b-versatile"


def get_api_key():
    """Obtiene la API key de Groq desde st.secrets (nunca del código fuente)."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


def build_filtered_summary(df, filtros: dict) -> dict:
    """
    Construye un resumen estadístico compacto SOLO del subconjunto de datos
    ya filtrado por el usuario (fechas, categoría, bodega, ciudad, etc.).
    Este resumen -y nada más- es lo que se envía al modelo.
    """
    summary = {"filtros_aplicados": filtros, "n_registros": len(df)}

    if "Margen_Absoluto_USD" in df.columns:
        summary["margen_total_usd"] = round(float(df["Margen_Absoluto_USD"].sum()), 2)
        summary["pct_transacciones_margen_negativo"] = round(
            float((df["Margen_Absoluto_USD"] < 0).mean() * 100), 2
        )
    if "Venta_Fantasma" in df.columns:
        summary["pct_venta_fantasma"] = round(float(df["Venta_Fantasma"].mean() * 100), 2)
    if "Tiempo_Entrega" in df.columns:
        summary["tiempo_entrega_promedio_dias"] = round(float(df["Tiempo_Entrega"].mean()), 2)
    if "Satisfaccion_NPS" in df.columns:
        summary["nps_promedio"] = round(float(df["Satisfaccion_NPS"].mean()), 2)
    if "Ticket_Soporte" in df.columns:
        summary["tasa_tickets_soporte_pct"] = round(float(df["Ticket_Soporte"].mean() * 100), 2)
    if "Categoria" in df.columns:
        summary["categorias_presentes"] = df["Categoria"].value_counts().head(5).to_dict()
    if "Ciudad_Destino" in df.columns:
        summary["ciudades_presentes"] = df["Ciudad_Destino"].value_counts().head(5).to_dict()
    if "Canal_Venta" in df.columns:
        summary["canales_presentes"] = df["Canal_Venta"].value_counts().to_dict()

    return summary


def generate_recommendation(summary: dict, api_key: str = None, timeout: int = 30):
    """
    Llama a la API de Groq (chat completions, modelo Llama-3) y solicita
    tres párrafos de recomendación estratégica basados solo en `summary`.
    Retorna (texto, error). Si hay error, texto es None.
    """
    api_key = api_key or get_api_key()
    if not api_key:
        return None, (
            "No se encontró GROQ_API_KEY en st.secrets. Configúrala en "
            ".streamlit/secrets.toml (local) o en Settings > Secrets (Streamlit Cloud)."
        )

    system_prompt = (
        "Eres un consultor senior de datos para TechLogistics S.A. Recibirás un resumen "
        "estadístico JSON de los datos YA FILTRADOS por el usuario en un dashboard. "
        "Responde en español, en exactamente 3 párrafos: (1) diagnóstico basado solo en las "
        "cifras dadas, (2) riesgo de negocio principal, (3) una recomendación táctica priorizada. "
        "No inventes cifras que no estén en el resumen. No uses markdown, solo texto plano."
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(summary, ensure_ascii=False, default=str)},
        ],
        "temperature": 0.4,
        "max_tokens": 700,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return text, None
    except requests.exceptions.HTTPError as e:
        return None, f"Error HTTP de Groq ({resp.status_code}): {resp.text[:300]}"
    except Exception as e:
        return None, f"Error al llamar a Groq: {e}"
