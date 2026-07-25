# 📦 DSS TechLogistics S.A. — Dashboard de Consultoría Senior

**Challenge 02 · Fundamentos en Ciencia de Datos · Universidad EAFIT (2026-1)**
Docente: Jorge Iván Padilla-Buriticá

## 1. Descripción del problema

TechLogistics S.A.S. (ficticio) presenta erosión de margen y caída de lealtad de
clientes porque sus tres sistemas —ERP de Inventario, Logística y Feedback— no
"hablan el mismo idioma": SKUs vendidos sin existir en el catálogo oficial
("Venta Fantasma"), costos unitarios atípicos, existencias negativas, fechas de
venta futuras, ciudades escritas de forma inconsistente y tiempos de entrega de
hasta 999 días.

Este repositorio implementa un **Sistema de Soporte a la Decisión (DSS)** en
Streamlit que:

1. Audita y limpia los tres datasets de forma **transparente y trazable**
   (Antes vs Después, Health Score, outliers marcados no borrados a ciegas).
2. Integra los datos en una **Sola Fuente de Verdad** vía Left Join estratégico,
   diagnosticando explícitamente el fenómeno de la Venta Fantasma.
3. Responde con evidencia visual los **5 interrogantes de alta gerencia** del
   Challenge (margen negativo, cuellos de botella logísticos, ingreso en
   riesgo, paradoja de fidelidad, riesgo operativo por bodega).
4. Genera **recomendaciones estratégicas con IA** (Llama-3 vía Groq) basadas
   únicamente en los datos que el usuario haya filtrado en ese momento.

## 2. Arquitectura del código

```
techlogistics_dashboard/
├── app.py                     # Orquestador Streamlit (UI, sidebar, tabs)
├── modules/
│   ├── data_loader.py         # Carga de CSVs + validación de esquema
│   ├── cleaning.py            # IQR, fechas futuras, normalización de ciudad, Health Score
│   ├── integration.py         # Left Join + diagnóstico de Venta Fantasma
│   ├── features.py            # Margen, Brecha de Entrega, Ratio de Soporte
│   ├── visuals.py             # Un gráfico por cada pregunta estratégica
│   └── ai_insights.py         # Cliente de la API de Groq (Llama-3)
├── .streamlit/
│   └── secrets.toml.example   # Plantilla de configuración de la API Key
├── requirements.txt
└── README.md
```

El código está modularizado siguiendo PEP8: la lógica de limpieza, integración,
feature engineering, visualización e IA vive en archivos separados de la capa
de UI (`app.py`), tal como exige el checklist de entrega del Challenge.

## 3. Datasets esperados (no incluidos en el repo)

Súbelos directamente en el sidebar de la app. Deben respetar el esquema del
Diccionario de Datos del curso:

| Archivo | Registros aprox. | Llave |
|---|---|---|
| `inventario_central_v2.csv` | 2,500 | `SKU_ID` |
| `transacciones_logistica_v2.csv` | 10,000 | `Transaccion_ID`, `SKU_ID` |
| `feedback_clientes_v2.csv` | 4,500 | `Feedback_ID`, `Transaccion_ID` |

## 4. Guía de instalación

### 4.1 Requisitos previos
- Python 3.10+
- Una API Key de [Groq](https://console.groq.com) (gratuita)

### 4.2 Instalación local

```bash
git clone https://github.com/<tu-usuario>/<tu-repo>.git
cd <tu-repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4.3 Configurar la API Key de Groq (nunca en el código)

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edita .streamlit/secrets.toml y pega tu GROQ_API_KEY
```

`.streamlit/secrets.toml` está en `.gitignore`: nunca se sube al repositorio.

### 4.4 Ejecutar

```bash
streamlit run app.py
```

Abre `http://localhost:8501`, sube los tres CSV en el sidebar y pulsa
**🔄 Refrescar Análisis**.

## 5. Despliegue en Streamlit Community Cloud

1. Sube este repositorio a GitHub (sin `secrets.toml`, sin los CSV).
2. En [share.streamlit.io](https://share.streamlit.io), conecta el repo y
   selecciona `app.py` como entry point.
3. En **Settings → Secrets**, pega:
   ```toml
   GROQ_API_KEY = "tu_api_key_real"
   ```
4. Despliega. Los datasets se cargan por el usuario en cada sesión (no se
   almacenan en el repo por privacidad y tamaño).

🔗 **App en la nube:** `https://<reemplazar-con-tu-url>.streamlit.app`

## 6. Supuestos de negocio documentados

- **SLA de entrega prometido**: el diccionario de datos no define un campo de
  "días prometidos". Se usa un supuesto configurable en el sidebar (default:
  5 días) para calcular la `Brecha_Entrega_Dias`. Ajusta este valor si
  TechLogistics define un SLA oficial distinto.
- **Venta Fantasma**: las ventas cuyo `SKU_ID` no existe en inventario **no se
  eliminan**; se etiquetan (`Venta_Fantasma = True`) para que el negocio
  decida entre producto no catalogado, error de digitación o fraude.
- **Outliers (IQR)**: se marcan con una columna `<columna>_outlier`, nunca se
  borran de forma automática. El usuario puede excluirlos de los KPIs desde
  la pestaña de Auditoría sin perder trazabilidad.

## 7. Elección de visualizaciones (justificación)

| Pregunta | Gráfico | Por qué |
|---|---|---|
| P1 Fuga de capital | Barra horizontal (ranking) + Box plot por categoría | Ranking para priorizar SKUs; box plot para ver dispersión de margen y detectar sesgo sistemático vs. casos aislados |
| P2 Cuellos de botella | Barras de correlación + Scatter con color por ciudad | La correlación resume "qué tan fuerte" es el problema por zona; el scatter muestra la relación cruda para validar que no sea espuria |
| P3 Venta invisible | Donut (composición) + Barra por ciudad | El donut comunica de inmediato el % de ingreso en riesgo a la junta directiva; la barra localiza dónde actuar primero |
| P4 Paradoja de fidelidad | Scatter de cuadrantes (stock vs. rating, tamaño = volumen) | Los cuadrantes separan visualmente "todo bien" de "alto stock + bajo rating", el foco de la pregunta |
| P5 Riesgo operativo | Scatter con línea de tendencia por bodega | Permite ver si la antigüedad de revisión predice tickets de soporte, bodega por bodega |

## 8. Checklist de cumplimiento (Guía de Validación)

- [x] Left Join estratégico con clasificación de nulos y Venta Fantasma
- [x] Normalización de ciudades vía diccionario de mapeo + regex
- [x] Filtro IQR para costos y tiempos de entrega, con opción "Ver registros excluidos"
- [x] Validación de fechas futuras contra `datetime.now()`
- [x] Sidebar con fecha, categoría, bodega y botón "Refrescar Análisis"
- [x] Pestaña de Transparencia (Antes vs Después, % salud de datos)
- [x] `st.tabs`: Auditoría, Operaciones, Cliente, Insights de IA
- [x] Botón de IA (Groq/Llama-3) basado solo en filtros aplicados
- [x] API Key de Groq vía `st.secrets`, nunca en el código
- [x] Código modularizado (`modules/`)

## 9. Documento de Hallazgos

El informe de consultoría dirigido a la junta directiva (narrativa de negocio,
capturas del dashboard y plan de acción priorizado) se encuentra en el
repositorio como `informe_hallazgos_TechLogistics.docx` — ver plantilla
adjunta con instrucciones de dónde insertar cada captura de pantalla.

## 10. Contacto académico

Profesor: Jorge Iván Padilla-Buriticá | EAFIT | Fundamentos en Ciencia de Datos
| jipadillab@eafit.edu.co
