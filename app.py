"""
Dashboard EDA - Impacto del Riego Tecnificado en la Producción Agrícola (Colombia)
Las interpretaciones y conclusiones se generan conversando con un modelo de IA
(Llama 3.3 70B vía Groq) que responde basándose en las cifras reales calculadas
sobre el dataset cargado.

Pregunta de negocio: ¿El sistema de riego tecnificado impacta realmente
la producción por hectárea?
"""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
from groq import Groq

MODEL_ID = "llama-3.3-70b-versatile"

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="EDA Agro Colombia + IA | Riego Tecnificado",
    page_icon="🌾",
    layout="wide",
)
sns.set_style("whitegrid")

SYSTEM_PROMPT_ANALISTA = (
    "Eres un analista de datos agrícolas experto en agricultura colombiana "
    "(cultivos, sistemas de riego y suelos). Tu trabajo es interpretar los "
    "datos que se te entregan en el bloque 'CONTEXTO DE DATOS' y responder "
    "las preguntas del usuario o generar conclusiones ejecutivas.\n\n"
    "Reglas estrictas:\n"
    "1. Basa TODAS tus respuestas únicamente en las cifras del CONTEXTO DE "
    "DATOS proporcionado. Nunca inventes números, cultivos, regiones ni "
    "estadísticas que no estén ahí.\n"
    "2. Si el usuario pregunta algo que no se puede responder con el "
    "contexto disponible, dilo honestamente y sugiere qué información "
    "adicional se necesitaría.\n"
    "3. Cuando interpretes diferencias entre grupos, menciona si son "
    "estadísticamente significativas (valor p) y el tamaño del efecto "
    "cuando estén disponibles, explicando qué significan en términos "
    "sencillos para alguien de negocio.\n"
    "4. Sé claro y accionable: cuando sea razonable, cierra con una "
    "recomendación práctica para el negocio agrícola.\n"
    "5. Responde siempre en español, con tono profesional y cercano. Usa "
    "listas o negritas cuando ayuden a la claridad, pero sé conciso."
)

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def detectar_columna(columnas, palabras_clave):
    for col in columnas:
        for palabra in palabras_clave:
            if palabra.lower() in col.lower():
                return col
    return None


@st.cache_data(show_spinner="Cargando datos...")
def cargar_datos(archivo):
    nombre = archivo.name.lower()
    if nombre.endswith(".csv"):
        return pd.read_csv(archivo, sep=None, engine="python")
    elif nombre.endswith((".xlsx", ".xls")):
        return pd.read_excel(archivo)
    elif nombre.endswith(".json"):
        return pd.read_json(archivo)
    elif nombre.endswith(".parquet"):
        return pd.read_parquet(archivo)
    elif nombre.endswith(".tsv"):
        return pd.read_csv(archivo, sep="\t")
    else:
        raise ValueError("Formato de archivo no soportado.")


@st.cache_data(show_spinner=False)
def generar_datos_demo(n=600, seed=42):
    """Dataset sintético con estructura de agro_colombia.csv, con un
    efecto real de riego tecnificado sobre la producción, para poder
    probar el dashboard sin archivo propio."""
    rng = np.random.default_rng(seed)
    cultivos = ["Café", "Aguacate Hass", "Caña de Azúcar", "Maíz", "Plátano",
                "Papa", "Arroz", "Cacao"]
    regiones = ["Antioquia", "Eje Cafetero", "Valle del Cauca", "Tolima",
                "Cundinamarca", "Huila", "Meta"]
    suelos = ["Franco", "Franco-Arcilloso", "Arcilloso", "Arenoso", "Volcánico"]

    riego = rng.choice(["Tecnificado", "Sin Riego"], size=n, p=[0.42, 0.58])
    cultivo = rng.choice(cultivos, size=n)
    region = rng.choice(regiones, size=n)
    suelo = rng.choice(suelos, size=n)
    hectareas = np.clip(np.round(rng.gamma(3.5, 3.2, size=n), 1), 1, 80)

    base_rend = {
        "Café": 1.4, "Aguacate Hass": 9.0, "Caña de Azúcar": 90.0, "Maíz": 4.5,
        "Plátano": 14.0, "Papa": 18.0, "Arroz": 6.0, "Cacao": 0.8,
    }
    rend_base = np.array([base_rend[c] for c in cultivo])
    efecto_riego = np.where(riego == "Tecnificado", rng.normal(1.35, 0.12, n),
                             rng.normal(1.0, 0.15, n))
    ruido_suelo = rng.normal(1.0, 0.08, n)

    produccion_ha = np.clip(rend_base * efecto_riego * ruido_suelo, 0.05, None)
    produccion_total = produccion_ha * hectareas

    return pd.DataFrame({
        "Cultivo": cultivo,
        "Region": region,
        "Tipo_Suelo": suelo,
        "Sistema_Riego_Tecnificado": riego,
        "Hectareas": hectareas,
        "Produccion_Anual_Ton": np.round(produccion_total, 2),
        "Costo_Insumos_Millon": np.round(hectareas * rng.normal(2.1, 0.4, n), 2),
    })


def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    if dof <= 0:
        return np.nan
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    if pooled_std == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled_std


def interpretar_d(d):
    if pd.isna(d):
        return "no calculable"
    ad = abs(d)
    if ad < 0.2:
        return "insignificante"
    elif ad < 0.5:
        return "pequeño"
    elif ad < 0.8:
        return "moderado"
    else:
        return "grande"


def construir_contexto_datos(df, col_riego, col_metric, metric_label,
                              col_cultivo, col_suelo, col_region, col_area,
                              max_categorias=12):
    """Construye un resumen textual y compacto de los datos y estadísticas
    calculadas, para inyectarlo como contexto real al modelo de IA."""
    lineas = []
    lineas.append(f"Dataset: {df.shape[0]} registros, {df.shape[1]} columnas.")
    lineas.append(f"Variable de sistema de riego: '{col_riego}'.")
    lineas.append(f"Métrica de producción analizada: '{metric_label}' (columna: '{col_metric}').")

    resumen = df.groupby(col_riego)[col_metric].agg(["mean", "median", "std", "count"])
    lineas.append("\nComparativa por sistema de riego (promedio | mediana | desv. estándar | n):")
    for idx, row in resumen.iterrows():
        lineas.append(f"- {idx}: {row['mean']:.2f} | {row['median']:.2f} | "
                       f"{row['std']:.2f} | n={int(row['count'])}")

    grupos = df[col_riego].dropna().unique()
    if len(grupos) == 2:
        g1 = df.loc[df[col_riego] == grupos[0], col_metric].dropna()
        g2 = df.loc[df[col_riego] == grupos[1], col_metric].dropna()
        if len(g1) > 1 and len(g2) > 1:
            t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
            u_stat, p_val_u = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            d = cohens_d(g1, g2)
            diff_pct = ((g1.mean() - g2.mean()) / g2.mean() * 100) if g2.mean() != 0 else np.nan
            lineas.append(f"\nPrueba t de Welch entre '{grupos[0]}' y '{grupos[1]}': "
                           f"t={t_stat:.2f}, valor p={p_val:.4f}.")
            lineas.append(f"Prueba U de Mann-Whitney: valor p={p_val_u:.4f}.")
            lineas.append(f"Tamaño de efecto (d de Cohen): {d:.2f} ({interpretar_d(d)}).")
            lineas.append(f"Diferencia relativa de '{grupos[0]}' respecto a '{grupos[1]}': "
                           f"{diff_pct:.1f}%.")
            lineas.append(f"Significancia estadística (p<0.05): "
                           f"{'SÍ, es significativa' if p_val < 0.05 else 'NO es significativa'}.")
    elif len(grupos) > 2:
        muestras = [df.loc[df[col_riego] == g, col_metric].dropna() for g in grupos]
        f_stat, p_val = stats.f_oneway(*muestras)
        lineas.append(f"\nANOVA entre {len(grupos)} categorías de riego: "
                       f"F={f_stat:.2f}, valor p={p_val:.4f} "
                       f"({'significativo' if p_val < 0.05 else 'no significativo'}).")

    def resumen_categoria(nombre_col, etiqueta):
        agg = (df.groupby([nombre_col, col_riego])[col_metric]
               .mean().round(2).reset_index())
        top = agg.sort_values(col_metric, ascending=False).head(max_categorias)
        lineas.append(f"\nPromedio de {metric_label} por {etiqueta} y sistema de riego "
                       f"(top {len(top)}):")
        for _, row in top.iterrows():
            lineas.append(f"- {row[nombre_col]} / {row[col_riego]}: {row[col_metric]:.2f}")

    if col_cultivo and col_cultivo != "(ninguna)":
        resumen_categoria(col_cultivo, "cultivo")
    if col_suelo and col_suelo != "(ninguna)":
        resumen_categoria(col_suelo, "tipo de suelo")
    if col_region and col_region != "(ninguna)":
        resumen_categoria(col_region, "región")

    if col_area and col_area != "(ninguna)":
        corr = df[[col_area, col_metric]].corr().iloc[0, 1]
        lineas.append(f"\nCorrelación entre área sembrada (hectáreas) y "
                       f"{metric_label}: {corr:.2f}.")

    return "\n".join(lineas)


def llamar_groq(client, mensajes_historial, contexto_datos, temperature=0.4, max_tokens=900, stream=True):
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
# SIDEBAR: API KEY, CARGA DE DATOS Y MAPEO DE COLUMNAS
# =========================================================
st.sidebar.title("🌾 Panel de control")

groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    placeholder="gsk_...",
    help="Se usa únicamente para llamar a la API oficial de Groq desde esta sesión.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Datos")
archivo = st.sidebar.file_uploader(
    "Carga agro_colombia.csv (u otro formato)",
    type=["csv", "xlsx", "xls", "json", "parquet", "tsv"],
)
usar_demo = st.sidebar.checkbox("Usar dataset de demostración", value=(archivo is None))

st.title("🌾 EDA Agro Colombia — Interpretación con IA (Llama 3.3 70B)")
st.markdown(
    "#### Pregunta de negocio: *¿El sistema de riego tecnificado impacta realmente "
    "la producción por hectárea?*"
)
st.caption(
    "Las conclusiones de este dashboard no están escritas a mano: se generan "
    "conversando con un modelo de IA (vía Groq) que interpreta las cifras "
    "reales calculadas sobre tus datos."
)

if archivo is not None:
    try:
        df_raw = cargar_datos(archivo)
        origen = f"archivo cargado: {archivo.name}"
    except Exception as e:
        st.error(f"❌ Error al cargar el archivo: {e}")
        st.stop()
elif usar_demo:
    df_raw = generar_datos_demo()
    origen = "dataset de demostración (sintético, estructura agro_colombia.csv)"
else:
    st.info("👈 Sube **agro_colombia.csv** (u otro formato) o activa el dataset de "
            "demostración en la barra lateral para comenzar.")
    st.stop()

df = df_raw.copy()
df.columns = [c.strip() for c in df.columns]
st.sidebar.success(f"✅ Datos: {df.shape[0]} filas × {df.shape[1]} columnas")

cols = df.columns.tolist()
col_riego_auto = detectar_columna(cols, ["riego", "irrigation", "tecnificado"])
col_prod_auto = detectar_columna(cols, ["produccion", "produc", "yield"])
col_area_auto = detectar_columna(cols, ["hectare", "hectarea", "area", "superficie"])
col_cultivo_auto = detectar_columna(cols, ["cultivo", "crop"])
col_suelo_auto = detectar_columna(cols, ["suelo", "soil"])
col_region_auto = detectar_columna(cols, ["region", "departamento", "zona", "municipio"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Mapeo de columnas")
col_riego = st.sidebar.selectbox(
    "Columna: Sistema de Riego", cols,
    index=cols.index(col_riego_auto) if col_riego_auto in cols else 0,
    key="map_riego",
)
col_prod = st.sidebar.selectbox(
    "Columna: Producción (Ton)", cols,
    index=cols.index(col_prod_auto) if col_prod_auto in cols else 0,
    key="map_prod",
)
col_area = st.sidebar.selectbox(
    "Columna: Área (Hectáreas)", ["(ninguna)"] + cols,
    index=(cols.index(col_area_auto) + 1) if col_area_auto in cols else 0,
    key="map_area",
)
col_cultivo = st.sidebar.selectbox(
    "Columna: Cultivo", ["(ninguna)"] + cols,
    index=(cols.index(col_cultivo_auto) + 1) if col_cultivo_auto in cols else 0,
    key="map_cultivo",
)
col_suelo = st.sidebar.selectbox(
    "Columna: Tipo de Suelo", ["(ninguna)"] + cols,
    index=(cols.index(col_suelo_auto) + 1) if col_suelo_auto in cols else 0,
    key="map_suelo",
)
col_region = st.sidebar.selectbox(
    "Columna: Región / Departamento", ["(ninguna)"] + cols,
    index=(cols.index(col_region_auto) + 1) if col_region_auto in cols else 0,
    key="map_region",
)

if col_area != "(ninguna)":
    df[col_area] = pd.to_numeric(df[col_area], errors="coerce")
    df[col_prod] = pd.to_numeric(df[col_prod], errors="coerce")
    df["Produccion_por_Hectarea"] = df[col_prod] / df[col_area].replace(0, np.nan)
    col_metric = "Produccion_por_Hectarea"
    metric_label = "Producción por Hectárea (Ton/Ha)"
else:
    df[col_prod] = pd.to_numeric(df[col_prod], errors="coerce")
    col_metric = col_prod
    metric_label = "Producción Anual (Ton)"

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Reiniciar chat con la IA"):
    st.session_state.pop("chat_messages", None)
    st.rerun()

# =========================================================
# TABS PRINCIPALES
# =========================================================
tab_datos, tab_eda, tab_pregunta, tab_apoyo, tab_ia = st.tabs(
    ["📂 Datos", "🔍 EDA General", "🎯 Pregunta de Negocio",
     "📊 Gráficas de Apoyo", "🤖 Interpretación con IA"]
)

# ---------------------------------------------------------
# TAB 1: DATOS
# ---------------------------------------------------------
with tab_datos:
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head(20), use_container_width=True, key="df_preview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", df.shape[0])
    c2.metric("Columnas", df.shape[1])
    c3.metric("Duplicados", int(df.duplicated().sum()))

    st.subheader("Tipos de datos")
    st.dataframe(
        pd.DataFrame({"Columna": df.dtypes.index, "Tipo": df.dtypes.astype(str).values}),
        use_container_width=True, key="df_dtypes",
    )

    st.subheader("Valores nulos por columna")
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False)
    if len(nulos) > 0:
        fig_nulos = px.bar(
            x=nulos.values, y=nulos.index, orientation="h",
            labels={"x": "Cantidad de nulos", "y": "Columna"},
            title="Valores faltantes por columna", color=nulos.values,
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_nulos, use_container_width=True, key="chart_nulos")
    else:
        st.success("✅ No se encontraron valores nulos en el dataset.")

# ---------------------------------------------------------
# TAB 2: EDA GENERAL
# ---------------------------------------------------------
with tab_eda:
    st.subheader("Estadística descriptiva (variables numéricas)")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.dataframe(df[num_cols].describe().T, use_container_width=True, key="df_describe")

    st.subheader("Distribución de una variable")
    var_sel = st.selectbox(
        "Selecciona una variable numérica", num_cols,
        index=num_cols.index(col_metric) if col_metric in num_cols else 0,
        key="select_var_num",
    )
    colh1, colh2 = st.columns(2)
    with colh1:
        fig_hist = px.histogram(df, x=var_sel, nbins=30, marginal="box",
                                 title=f"Distribución de {var_sel}")
        st.plotly_chart(fig_hist, use_container_width=True, key="chart_hist")
    with colh2:
        fig_box_all = px.box(df, y=var_sel, title=f"Boxplot general de {var_sel}", points="outliers")
        st.plotly_chart(fig_box_all, use_container_width=True, key="chart_box_all")

    st.subheader("Matriz de correlación")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        fig_corr = px.imshow(
            corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
            title="Correlación entre variables numéricas",
        )
        st.plotly_chart(fig_corr, use_container_width=True, key="chart_corr")
    else:
        st.info("Se necesitan al menos 2 variables numéricas para calcular correlaciones.")

    st.subheader("Variables categóricas")
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    if cat_cols:
        var_cat = st.selectbox("Selecciona una variable categórica", cat_cols, key="select_var_cat")
        conteo = df[var_cat].value_counts().reset_index()
        conteo.columns = [var_cat, "conteo"]
        fig_cat = px.bar(conteo, x=var_cat, y="conteo", title=f"Frecuencia de {var_cat}",
                          color="conteo", color_continuous_scale="Greens")
        st.plotly_chart(fig_cat, use_container_width=True, key="chart_cat")
    else:
        st.info("No se detectaron variables categóricas.")

# ---------------------------------------------------------
# TAB 3: PREGUNTA DE NEGOCIO
# ---------------------------------------------------------
with tab_pregunta:
    st.subheader("¿El sistema de riego tecnificado impacta la producción por hectárea?")

    st.markdown("**Comparativa rápida (promedios por grupo):**")
    resumen = df.groupby(col_riego)[col_metric].mean().rename("Promedio").to_frame()
    resumen["Mediana"] = df.groupby(col_riego)[col_metric].median()
    resumen["Desv. Estándar"] = df.groupby(col_riego)[col_metric].std()
    resumen["N"] = df.groupby(col_riego)[col_metric].count()
    st.table(resumen.style.format({"Promedio": "{:.2f}", "Mediana": "{:.2f}", "Desv. Estándar": "{:.2f}"}))

    st.markdown("### 📦 Visualización clave: Boxplot de Producción — Con Riego vs. Sin Riego")
    fig_box_key = px.box(
        df, x=col_riego, y=col_metric, color=col_riego, points="all",
        title=f"{metric_label} según Sistema de Riego Tecnificado",
        labels={col_riego: "Sistema de Riego", col_metric: metric_label},
    )
    fig_box_key.update_layout(showlegend=False)
    st.plotly_chart(fig_box_key, use_container_width=True, key="chart_box_key")

    with st.expander("Ver versión estática (seaborn)"):
        fig_sns, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x=col_riego, y=col_metric, hue=col_riego, ax=ax,
                    palette="Set2", legend=False)
        ax.set_title(f"{metric_label} según Sistema de Riego")
        st.pyplot(fig_sns, clear_figure=True)

    st.markdown("### 🧪 Prueba estadística")
    grupos = df[col_riego].dropna().unique()
    if len(grupos) == 2:
        g1_name, g2_name = grupos[0], grupos[1]
        g1 = df.loc[df[col_riego] == g1_name, col_metric].dropna()
        g2 = df.loc[df[col_riego] == g2_name, col_metric].dropna()
        t_stat, p_val_t = stats.ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
        d = cohens_d(g1, g2)
        c1, c2, c3 = st.columns(3)
        c1.metric("p-value (t-test)", f"{p_val_t:.4f}")
        c2.metric("d de Cohen", f"{d:.2f}" if not pd.isna(d) else "N/A")
        c3.metric("Significativo (p<0.05)", "Sí" if p_val_t < 0.05 else "No")
    else:
        muestras = [df.loc[df[col_riego] == g, col_metric].dropna() for g in grupos]
        f_stat, p_val = stats.f_oneway(*muestras)
        c1, c2 = st.columns(2)
        c1.metric("Estadístico F (ANOVA)", f"{f_stat:.2f}")
        c2.metric("p-value (ANOVA)", f"{p_val:.4f}")

    st.info(
        "💡 Ve a la pestaña **🤖 Interpretación con IA** para que el modelo Llama 3.3 70B "
        "te explique, en lenguaje de negocio, qué significan estas cifras y responda "
        "tus preguntas de seguimiento."
    )

# ---------------------------------------------------------
# TAB 4: GRÁFICAS DE APOYO
# ---------------------------------------------------------
with tab_apoyo:
    st.subheader("Gráficas complementarias para reforzar el storytelling")

    st.markdown("**Distribución detallada (Violin Plot)**")
    fig_violin = px.violin(
        df, x=col_riego, y=col_metric, color=col_riego, box=True, points="outliers",
        title=f"Distribución de {metric_label} por Sistema de Riego",
    )
    fig_violin.update_layout(showlegend=False)
    st.plotly_chart(fig_violin, use_container_width=True, key="chart_violin")

    if col_area != "(ninguna)":
        st.markdown("**Relación Área vs. Producción, coloreada por Riego**")
        fig_scatter = px.scatter(
            df, x=col_area, y=col_prod, color=col_riego, trendline="ols",
            title="Área Sembrada vs. Producción Anual",
            labels={col_area: "Área (Hectáreas)", col_prod: "Producción (Ton)"},
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key="chart_scatter")

    if col_cultivo != "(ninguna)":
        st.markdown("**Producción media por Cultivo y Sistema de Riego**")
        agg = df.groupby([col_cultivo, col_riego])[col_metric].mean().reset_index()
        fig_bar_cultivo = px.bar(
            agg, x=col_cultivo, y=col_metric, color=col_riego, barmode="group",
            title=f"{metric_label} promedio por Cultivo y Riego",
        )
        st.plotly_chart(fig_bar_cultivo, use_container_width=True, key="chart_bar_cultivo")

    if col_suelo != "(ninguna)":
        st.markdown("**Producción por Tipo de Suelo, según Riego**")
        fig_suelo = px.box(
            df, x=col_suelo, y=col_metric, color=col_riego,
            title=f"{metric_label} por Tipo de Suelo y Sistema de Riego",
        )
        st.plotly_chart(fig_suelo, use_container_width=True, key="chart_suelo")

    if col_region != "(ninguna)":
        st.markdown("**Producción media por Región**")
        agg_reg = df.groupby([col_region, col_riego])[col_metric].mean().reset_index()
        fig_region = px.bar(
            agg_reg, x=col_region, y=col_metric, color=col_riego, barmode="group",
            title=f"{metric_label} promedio por Región y Riego",
        )
        fig_region.update_xaxes(tickangle=45)
        st.plotly_chart(fig_region, use_container_width=True, key="chart_region")

    if col_area == "(ninguna)" and col_cultivo == "(ninguna)" and col_suelo == "(ninguna)" and col_region == "(ninguna)":
        st.info("Selecciona columnas adicionales (área, cultivo, suelo, región) en la "
                 "barra lateral para desbloquear más gráficas.")

# ---------------------------------------------------------
# TAB 5: INTERPRETACIÓN CON IA (CHAT)
# ---------------------------------------------------------
with tab_ia:
    st.subheader("🤖 Conversa con la IA sobre tus resultados")
    st.caption(
        "El modelo recibe, en cada mensaje, un resumen real de las estadísticas "
        "calculadas sobre tu dataset (promedios, prueba t, tamaño de efecto, "
        "desgloses por cultivo/suelo/región) y responde basándose en esas cifras."
    )

    if not groq_api_key:
        st.info("👈 Ingresa tu **Groq API Key** en la barra lateral para activar el "
                "chat de interpretación con IA.")
    else:
        try:
            client = Groq(api_key=groq_api_key)
        except Exception as e:
            st.error(f"❌ No se pudo inicializar el cliente de Groq: {e}")
            st.stop()

        contexto_datos = construir_contexto_datos(
            df, col_riego, col_metric, metric_label,
            col_cultivo, col_suelo, col_region, col_area,
        )

        with st.expander("🔎 Ver el contexto de datos que recibe la IA"):
            st.code(contexto_datos, language="text")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            generar_resumen = st.button(
                "✨ Generar interpretación ejecutiva automática",
                use_container_width=True, type="primary",
            )
        with col_btn2:
            temperature_ia = st.slider("Temperatura", 0.0, 1.0, 0.4, 0.1, key="temp_ia")

        if generar_resumen:
            prompt_inicial = (
                "Genera un resumen ejecutivo de conclusiones sobre si el sistema de "
                "riego tecnificado impacta la producción por hectárea, basado "
                "estrictamente en el CONTEXTO DE DATOS. Incluye: 1) el hallazgo "
                "principal, 2) si es estadísticamente significativo, 3) matices por "
                "cultivo/suelo/región si el contexto lo permite, y 4) una "
                "recomendación de negocio."
            )
            st.session_state.chat_messages.append({"role": "user", "content": prompt_inicial})

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        pregunta_usuario = st.chat_input(
            "Pregúntale a la IA sobre los resultados (ej. '¿Y en el cultivo de café?')"
        )
        if pregunta_usuario:
            st.session_state.chat_messages.append({"role": "user", "content": pregunta_usuario})
            with st.chat_message("user"):
                st.markdown(pregunta_usuario)

        # Si el último mensaje es del usuario y aún no se respondió, generar respuesta
        if (st.session_state.chat_messages
                and st.session_state.chat_messages[-1]["role"] == "user"):
            with st.chat_message("assistant"):
                placeholder = st.empty()
                texto_acumulado = ""
                try:
                    stream = llamar_groq(
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
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": texto_acumulado}
            )

        if not st.session_state.chat_messages:
            st.markdown("##### 💡 Ideas de preguntas:")
            ejemplos = [
                "¿El riego tecnificado realmente aumenta la producción por hectárea?",
                "¿En qué cultivo se nota más el efecto del riego?",
                "¿La diferencia observada es significativa o podría ser azar?",
                "¿Qué recomendarías a un inversionista agrícola con estos datos?",
            ]
            ecols = st.columns(2)
            for i, ej in enumerate(ejemplos):
                if ecols[i % 2].button(ej, use_container_width=True, key=f"ejemplo_{i}"):
                    st.session_state.chat_messages.append({"role": "user", "content": ej})
                    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard EDA + IA · Agro Colombia · Groq + Llama 3.3 70B")
