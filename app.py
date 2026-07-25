"""
Dashboard: Texto -> Tabla -> EDA
Pega un párrafo de texto con cifras (ventas, producción, población, precios,
etc.), un modelo de IA (Llama 3.3 70B vía Groq) extrae los datos y los
estructura en una tabla, y luego el dashboard hace un EDA cuantitativo,
cualitativo y gráfico sobre esos datos extraídos.
"""

import json
import re

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from groq import Groq

MODEL_ID = "llama-3.3-70b-versatile"

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Texto a Tabla + EDA con IA",
    page_icon="📝",
    layout="wide",
)
sns.set_style("whitegrid")

SYSTEM_PROMPT_EXTRACCION = (
    "Eres un sistema experto en extracción de datos estructurados a partir "
    "de texto libre en español (o cualquier idioma). El usuario te dará un "
    "párrafo que contiene cifras: cantidades, montos, porcentajes, fechas, "
    "nombres de entidades (países, empresas, cultivos, personas, productos, "
    "etc.), categorías, etc.\n\n"
    "Tu tarea es identificar cada entidad o registro mencionado junto con "
    "sus atributos numéricos y categóricos, y devolverlos como una tabla "
    "estructurada.\n\n"
    "Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional, "
    "sin explicaciones y sin backticks, con este formato exacto:\n"
    '{"columnas": ["col1", "col2", ...], "registros": '
    '[{"col1": valor, "col2": valor, ...}, ...]}\n\n'
    "Reglas estrictas:\n"
    "1. Usa nombres de columna cortos, descriptivos, en snake_case y sin "
    "tildes (ej. 'produccion_toneladas', 'crecimiento_pct', 'pais').\n"
    "2. Si un valor es numérico (cifras, porcentajes, montos, años), "
    "extrae SOLO el número, sin símbolos de moneda, sin '%', sin separador "
    "de miles, usando punto como separador decimal. Indica la unidad o el "
    "hecho de que es un porcentaje en el propio nombre de la columna.\n"
    "3. Todas las filas deben tener las mismas columnas; usa null si un "
    "dato no está disponible para un registro.\n"
    "4. Si el texto describe una sola entidad con varias cifras (no una "
    "lista comparativa), igual estructura la información como una tabla "
    "de una sola fila con una columna por cada cifra mencionada.\n"
    "5. No inventes datos, entidades ni cifras que no estén explícita o "
    "razonablemente implícitas en el texto."
)

SYSTEM_PROMPT_ANALISTA = (
    "Eres un analista de datos. Debes interpretar la tabla de datos que se "
    "te entrega en el bloque 'CONTEXTO DE DATOS' (extraída automáticamente "
    "de un texto) y responder las preguntas del usuario o generar "
    "conclusiones ejecutivas.\n\n"
    "Reglas: básate únicamente en las cifras del contexto, no inventes "
    "datos; si algo no se puede responder con la información disponible, "
    "dilo honestamente. Responde en español, de forma clara y concisa, "
    "usando listas o negritas cuando ayuden."
)

TEXTO_EJEMPLO = (
    "En 2023, la producción de café en Colombia alcanzó 11.5 millones de "
    "sacos, con un crecimiento del 8.2% frente al año anterior. Brasil, "
    "el mayor productor mundial, produjo 58 millones de sacos, un aumento "
    "del 3.1%. Vietnam produjo 27 millones de sacos, cayendo un 4.5% por "
    "sequías. Etiopía registró 7.5 millones de sacos con un crecimiento "
    "del 2.0%, mientras que Honduras alcanzó 5.9 millones de sacos, "
    "creciendo 1.8%. El precio internacional promedio del café arábica "
    "cerró en 1.85 dólares por libra."
)

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def extraer_json(texto):
    """Extrae y parsea un bloque JSON de la respuesta del modelo, por si
    viene envuelto en texto adicional o backticks."""
    texto = texto.strip()
    texto = re.sub(r"^```(json)?|```$", "", texto, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", texto, flags=re.DOTALL)
    if match:
        texto = match.group(0)
    return json.loads(texto)


def extraer_tabla_con_ia(client, texto_usuario, temperature=0.2):
    respuesta = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_EXTRACCION},
            {"role": "user", "content": texto_usuario},
        ],
        temperature=temperature,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    contenido = respuesta.choices[0].message.content
    datos = extraer_json(contenido)
    registros = datos.get("registros", [])
    columnas = datos.get("columnas")
    df = pd.DataFrame(registros)
    if columnas:
        columnas_presentes = [c for c in columnas if c in df.columns]
        otras = [c for c in df.columns if c not in columnas_presentes]
        df = df[columnas_presentes + otras]
    return df


def detectar_tipos(df):
    """Determina qué columnas son numéricas (aunque hayan llegado como
    texto/objeto) intentando convertirlas."""
    num_cols, cat_cols = [], []
    for c in df.columns:
        convertido = pd.to_numeric(df[c], errors="coerce")
        tasa_valida = convertido.notna().mean() if len(df) else 0
        if tasa_valida >= 0.7:
            df[c] = convertido
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return df, num_cols, cat_cols


def construir_contexto_generico(df, num_cols, cat_cols, max_categorias=12):
    lineas = []
    lineas.append(f"Tabla extraída del texto: {df.shape[0]} filas, {df.shape[1]} columnas.")
    lineas.append(f"Columnas: {', '.join(df.columns)}.")
    lineas.append("\nDatos completos:")
    lineas.append(df.to_string(index=False))

    if num_cols:
        lineas.append("\nEstadística descriptiva de variables numéricas:")
        desc = df[num_cols].describe().T[["mean", "std", "min", "max"]].round(2)
        for idx, row in desc.iterrows():
            lineas.append(f"- {idx}: promedio={row['mean']}, desv_std={row['std']}, "
                           f"min={row['min']}, max={row['max']}")

    if cat_cols:
        lineas.append("\nVariables categóricas:")
        for c in cat_cols:
            top_vals = df[c].value_counts().head(max_categorias)
            resumen_vals = ", ".join(f"{k} ({v})" for k, v in top_vals.items())
            lineas.append(f"- {c}: {df[c].nunique()} valores únicos. Top: {resumen_vals}")

    return "\n".join(lineas)


def llamar_groq_chat(client, mensajes_historial, contexto_datos, temperature=0.4,
                      max_tokens=900, stream=True):
    system_completo = SYSTEM_PROMPT_ANALISTA + "\n\nCONTEXTO DE DATOS:\n" + contexto_datos
    payload = [{"role": "system", "content": system_completo}] + mensajes_historial
    return client.chat.completions.create(
        model=MODEL_ID,
        messages=payload,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("📝 Panel de control")
groq_api_key = st.sidebar.text_input(
    "Groq API Key", type="password", placeholder="gsk_...",
    help="Se usa únicamente para llamar a la API oficial de Groq desde esta sesión.",
)
st.sidebar.markdown("---")
temperature_extraccion = st.sidebar.slider(
    "Precisión de extracción (menor = más literal)", 0.0, 1.0, 0.2, 0.1
)
st.sidebar.caption(
    "Recomendado dejar bajo (0.0–0.3) para que la IA no invente cifras "
    "al extraer la tabla."
)
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Reiniciar todo"):
    for k in ["df_extraido", "chat_messages"]:
        st.session_state.pop(k, None)
    st.rerun()

st.title("📝 Texto a Tabla + EDA con IA")
st.caption(
    "Pega un párrafo con cifras, la IA (Llama 3.3 70B vía Groq) lo convierte "
    "en una tabla estructurada, y el dashboard hace el EDA cuantitativo, "
    "cualitativo y gráfico automáticamente."
)

if not groq_api_key:
    st.info("👈 Ingresa tu **Groq API Key** en la barra lateral para comenzar.")
    st.stop()

try:
    client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"❌ No se pudo inicializar el cliente de Groq: {e}")
    st.stop()

# =========================================================
# PASO 1: TEXTO -> TABLA
# =========================================================
st.markdown("### 1️⃣ Pega tu párrafo de texto")

col_ej, _ = st.columns([1, 3])
with col_ej:
    if st.button("📋 Usar texto de ejemplo"):
        st.session_state["texto_input"] = TEXTO_EJEMPLO

texto_usuario = st.text_area(
    "Texto con cifras (ventas, producción, población, precios, etc.)",
    value=st.session_state.get("texto_input", ""),
    height=160,
    placeholder="Ej: En 2023, la producción de café en Colombia alcanzó 11.5 "
                "millones de sacos, con un crecimiento del 8.2%...",
    key="texto_input",
)

extraer_btn = st.button("🔎 Extraer tabla con IA", type="primary")

if extraer_btn:
    if not texto_usuario.strip():
        st.warning("⚠️ Pega un texto antes de extraer la tabla.")
    else:
        with st.spinner("Extrayendo datos estructurados con IA..."):
            try:
                df_extraido = extraer_tabla_con_ia(
                    client, texto_usuario, temperature=temperature_extraccion
                )
                if df_extraido.empty:
                    st.error("❌ La IA no logró extraer registros de este texto. "
                              "Intenta con un texto que contenga cifras más explícitas.")
                else:
                    st.session_state["df_extraido"] = df_extraido
                    st.session_state.pop("chat_messages", None)
                    st.success(f"✅ Se extrajeron {df_extraido.shape[0]} registros "
                               f"y {df_extraido.shape[1]} columnas.")
            except Exception as e:
                st.error(f"❌ Error al extraer o interpretar la respuesta de la IA: {e}")

if "df_extraido" not in st.session_state:
    st.info("👆 Pega un texto y presiona **Extraer tabla con IA** para comenzar el EDA.")
    st.stop()

st.markdown("### 2️⃣ Tabla extraída (puedes editarla antes del EDA)")
df_editado = st.data_editor(
    st.session_state["df_extraido"],
    num_rows="dynamic",
    use_container_width=True,
    key="editor_tabla",
)

csv_bytes = df_editado.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Descargar tabla como CSV", data=csv_bytes,
    file_name="tabla_extraida.csv", mime="text/csv",
)

df = df_editado.copy()
df, num_cols, cat_cols = detectar_tipos(df)

st.markdown("---")

# =========================================================
# TABS DE EDA
# =========================================================
tab_cuant, tab_cual, tab_graf, tab_ia = st.tabs(
    ["🔢 EDA Cuantitativo", "🔤 EDA Cualitativo", "📊 EDA Gráfico", "🤖 Interpretación con IA"]
)

# ---------------------------------------------------------
# TAB: EDA CUANTITATIVO
# ---------------------------------------------------------
with tab_cuant:
    st.subheader("Análisis cuantitativo (variables numéricas)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", df.shape[0])
    c2.metric("Columnas numéricas", len(num_cols))
    c3.metric("Columnas categóricas", len(cat_cols))

    if num_cols:
        st.dataframe(
            df[num_cols].describe().T.style.format("{:.2f}"),
            use_container_width=True, key="df_describe_cuant",
        )
        st.markdown("**Rangos y extremos:**")
        for c in num_cols:
            serie = df[c].dropna()
            if len(serie) == 0:
                continue
            idx_max = serie.idxmax()
            idx_min = serie.idxmin()
            etiqueta = cat_cols[0] if cat_cols else None
            texto_max = f"{df.loc[idx_max, etiqueta]}: {serie.loc[idx_max]:.2f}" if etiqueta else f"{serie.loc[idx_max]:.2f}"
            texto_min = f"{df.loc[idx_min, etiqueta]}: {serie.loc[idx_min]:.2f}" if etiqueta else f"{serie.loc[idx_min]:.2f}"
            st.markdown(f"- **{c}** → máximo: {texto_max} · mínimo: {texto_min} · "
                        f"promedio: {serie.mean():.2f}")
    else:
        st.warning("No se detectaron columnas numéricas en la tabla extraída.")

# ---------------------------------------------------------
# TAB: EDA CUALITATIVO
# ---------------------------------------------------------
with tab_cual:
    st.subheader("Análisis cualitativo (variables categóricas)")
    if cat_cols:
        resumen_cat = pd.DataFrame({
            "Columna": cat_cols,
            "Valores únicos": [df[c].nunique() for c in cat_cols],
            "Valor más frecuente": [
                df[c].mode().iloc[0] if not df[c].mode().empty else "-" for c in cat_cols
            ],
            "% Nulos": [round(df[c].isna().mean() * 100, 1) for c in cat_cols],
        })
        st.dataframe(resumen_cat, use_container_width=True, key="df_resumen_cat")

        var_cat_sel = st.selectbox(
            "Ver detalle de una variable categórica", cat_cols, key="select_cat_detalle"
        )
        st.dataframe(
            df[var_cat_sel].value_counts().rename("Frecuencia").to_frame(),
            use_container_width=True, key="df_valuecounts",
        )
    else:
        st.warning("No se detectaron columnas categóricas en la tabla extraída.")

    st.markdown("**Valores faltantes por columna:**")
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False)
    if len(nulos) > 0:
        st.dataframe(nulos.rename("Nulos").to_frame(), use_container_width=True, key="df_nulos")
    else:
        st.success("✅ No hay valores faltantes en la tabla.")

# ---------------------------------------------------------
# TAB: EDA GRÁFICO
# ---------------------------------------------------------
with tab_graf:
    st.subheader("Análisis gráfico")

    col_etiqueta = cat_cols[0] if cat_cols else None

    if num_cols:
        st.markdown("**Ranking por variable numérica**")
        metrica_sel = st.selectbox("Selecciona una variable numérica", num_cols, key="select_metrica_rank")
        if col_etiqueta:
            df_sorted = df[[col_etiqueta, metrica_sel]].dropna().sort_values(metrica_sel, ascending=False)
            fig_rank = px.bar(
                df_sorted, x=col_etiqueta, y=metrica_sel, color=metrica_sel,
                color_continuous_scale="Teal",
                title=f"{metrica_sel} por {col_etiqueta}",
            )
            fig_rank.update_xaxes(tickangle=30)
            st.plotly_chart(fig_rank, use_container_width=True, key="chart_rank")
        else:
            fig_hist = px.histogram(df, x=metrica_sel, nbins=20, marginal="box",
                                     title=f"Distribución de {metrica_sel}")
            st.plotly_chart(fig_hist, use_container_width=True, key="chart_hist_solo")

        colb1, colb2 = st.columns(2)
        with colb1:
            fig_box = px.box(df, y=metrica_sel, points="all", title=f"Boxplot — {metrica_sel}")
            st.plotly_chart(fig_box, use_container_width=True, key="chart_box_metrica")
        with colb2:
            fig, ax = plt.subplots(figsize=(6, 4.2))
            sns.violinplot(y=df[metrica_sel].dropna(), color="#2E8B57", ax=ax)
            ax.set_title(f"Violin plot — {metrica_sel}")
            st.pyplot(fig, clear_figure=True)

        if len(num_cols) >= 2:
            st.markdown("**Matriz de correlación**")
            corr = df[num_cols].corr(numeric_only=True)
            fig_corr = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdYlGn", aspect="auto",
                title="Correlación entre variables numéricas",
            )
            st.plotly_chart(fig_corr, use_container_width=True, key="chart_corr")

            st.markdown("**Relación entre dos variables numéricas**")
            colx, coly = st.columns(2)
            var_x = colx.selectbox("Eje X", num_cols, index=0, key="select_x")
            var_y = coly.selectbox("Eje Y", num_cols, index=min(1, len(num_cols) - 1), key="select_y")
            fig_scatter = px.scatter(
                df, x=var_x, y=var_y, color=col_etiqueta if col_etiqueta else None,
                size=metrica_sel, hover_name=col_etiqueta if col_etiqueta else None,
                title=f"{var_y} vs {var_x}",
            )
            st.plotly_chart(fig_scatter, use_container_width=True, key="chart_scatter")
    else:
        st.warning("No hay columnas numéricas para graficar.")

    if cat_cols:
        st.markdown("**Frecuencia de una variable categórica**")
        var_cat_graf = st.selectbox(
            "Selecciona una variable categórica", cat_cols, key="select_cat_graf"
        )
        conteo = df[var_cat_graf].value_counts().reset_index()
        conteo.columns = [var_cat_graf, "conteo"]
        fig_cat = px.bar(
            conteo, x=var_cat_graf, y="conteo", color="conteo",
            color_continuous_scale="Purples", title=f"Frecuencia de {var_cat_graf}",
        )
        st.plotly_chart(fig_cat, use_container_width=True, key="chart_cat_freq")

# ---------------------------------------------------------
# TAB: INTERPRETACIÓN CON IA
# ---------------------------------------------------------
with tab_ia:
    st.subheader("🤖 Interpreta los datos extraídos conversando con la IA")
    st.caption(
        "El modelo recibe la tabla completa y sus estadísticas descriptivas como "
        "contexto real, y responde basándose en esas cifras."
    )

    contexto_datos = construir_contexto_generico(df, num_cols, cat_cols)
    with st.expander("🔎 Ver el contexto de datos que recibe la IA"):
        st.code(contexto_datos, language="text")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        generar_resumen = st.button(
            "✨ Generar resumen ejecutivo automático",
            use_container_width=True, type="primary",
        )
    with col_btn2:
        temperature_ia = st.slider("Temperatura", 0.0, 1.0, 0.4, 0.1, key="temp_ia")

    if generar_resumen:
        prompt_inicial = (
            "Genera un resumen ejecutivo de los datos extraídos: describe los "
            "hallazgos cuantitativos y cualitativos más relevantes, destaca "
            "valores máximos/mínimos y cualquier patrón interesante, "
            "basándote estrictamente en el CONTEXTO DE DATOS."
        )
        st.session_state.chat_messages.append({"role": "user", "content": prompt_inicial})

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pregunta_usuario = st.chat_input("Pregúntale a la IA sobre esta tabla...")
    if pregunta_usuario:
        st.session_state.chat_messages.append({"role": "user", "content": pregunta_usuario})
        with st.chat_message("user"):
            st.markdown(pregunta_usuario)

    if (st.session_state.chat_messages
            and st.session_state.chat_messages[-1]["role"] == "user"):
        with st.chat_message("assistant"):
            placeholder = st.empty()
            texto_acumulado = ""
            try:
                stream = llamar_groq_chat(
                    client, st.session_state.chat_messages, contexto_datos,
                    temperature=temperature_ia, max_tokens=900, stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    texto_acumulado += delta
                    placeholder.markdown(texto_acumulado + "▌")
                placeholder.markdown(texto_acumulado)
            except Exception as e:
                texto_acumulado = f"❌ Error al llamar a la API de Groq: {e}"
                placeholder.markdown(texto_acumulado)
        st.session_state.chat_messages.append({"role": "assistant", "content": texto_acumulado})

st.sidebar.markdown("---")
st.sidebar.caption("Texto a Tabla + EDA con IA · Groq + Llama 3.3 70B")
