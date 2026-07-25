# TechLogistics S.A. - Dashboard de Soporte a la Decisión (DSS)

**Autor:** Samuel Alarcón  
**Rol:** Consultor de Datos Senior  
**Curso:** Fundamentos en Ciencia de Datos

## 📌 Descripción del Problema
TechLogistics S.A. enfrenta una severa pérdida de visibilidad operativa que está erosionando su margen de beneficios y lealtad del cliente. La raíz del problema reside en la desconexión de tres sistemas clave: Inventarios, Logística y Feedback. Este proyecto implementa un Sistema de Soporte a la Decisión (DSS) auditable, corrigiendo fallas de catálogo (Ventas Fantasmas), filtrando anomalías de costos y unificando las bases de datos para entregar diagnósticos accionables y soportados por modelos fundacionales de inteligencia artificial.

## 🛠️ Guía de Instalación y Ejecución

1. Clonar el repositorio.
2. Crear un entorno virtual e instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Configurar la clave de API para el modelo de IA:
* Crear una carpeta .streamlit en la raíz.
* Crear el archivo secrets.toml con el contenido: GROQ_API_KEY = "tu_llave_aqui"
4. Ejecutar el dashboard:
   ```Bash
   streamlit run app.py
   ```

## 🌐 Enlace a la Aplicación en la Nube
