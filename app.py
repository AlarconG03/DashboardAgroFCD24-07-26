"""
Dashboard EDA - Impacto del Riego Tecnificado en la Producción Agrícola (Colombia)
Pregunta de negocio: ¿El sistema de riego tecnificado impacta realmente la producción por hectárea?
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="EDA Agro Colombia | Riego Tecnificado",
    page_icon="🌾",
    layout="wide",
)

sns.set_style("whitegrid")

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def detectar_columna(columnas, palabras_clave):
    """Busca la primera columna cuyo nombre contenga alguna palabra clave."""
    for col in columnas:
        for palabra in palabras_clave:
            if palabra.lower() in col.lower():
                return col
    return None


@st.cache_data(show_spinner="Cargando datos...")
def cargar_datos(archivo):
    nombre = archivo.name.lower()
    if nombre.endswith(".csv"):
        return pd.read_csv(archivo)
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


def cohens_d(x, y):
    """Tamaño del efecto (d de Cohen) entre dos muestras independientes."""
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


# =========================================================
# SIDEBAR: CARGA DE DATOS
# =========================================================
st.sidebar.title("🌾 Panel de control")
st.sidebar.markdown("Sube el dataset de cultivos, riego y suelos de Colombia.")

archivo = st.sidebar.file_uploader(
    "Carga tu archivo (agro_colombia.csv u otro formato)",
    type=["csv", "xlsx", "xls", "json", "parquet", "tsv"],
)

st.title("🌾 EDA: Impacto del Riego Tecnificado en la Producción Agrícola")
st.markdown(
    "#### Pregunta de negocio: *¿El sistema de riego tecnificado impacta realmente "
    "la producción por hectárea?*"
)

if archivo is None:
    st.info("👈 Sube el archivo **agro_colombia.csv** (u otro formato compatible) desde la barra lateral para comenzar.")
    st.stop()

try:
    df_raw = cargar_datos(archivo)
except Exception as e:
    st.error(f"❌ Error al cargar el archivo: {e}")
    st.stop()

df = df_raw.copy()
st.sidebar.success(f"✅ Datos cargados: {df.shape[0]} filas × {df.shape[1]} columnas")

# =========================================================
# SIDEBAR: MAPEO DE COLUMNAS
# =========================================================
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
    "Columna: Sistema de Riego",
    cols,
    index=cols.index(col_riego_auto) if col_riego_auto in cols else 0,
)
col_prod = st.sidebar.selectbox(
    "Columna: Producción (Ton)",
    cols,
    index=cols.index(col_prod_auto) if col_prod_auto in cols else 0,
)
col_area = st.sidebar.selectbox(
    "Columna: Área (Hectáreas)",
    ["(ninguna)"] + cols,
    index=(cols.index(col_area_auto) + 1) if col_area_auto in cols else 0,
)
col_cultivo = st.sidebar.selectbox(
    "Columna: Cultivo",
    ["(ninguna)"] + cols,
    index=(cols.index(col_cultivo_auto) + 1) if col_cultivo_auto in cols else 0,
)
col_suelo = st.sidebar.selectbox(
    "Columna: Tipo de Suelo",
    ["(ninguna)"] + cols,
    index=(cols.index(col_suelo_auto) + 1) if col_suelo_auto in cols else 0,
)
col_region = st.sidebar.selectbox(
    "Columna: Región / Departamento",
    ["(ninguna)"] + cols,
    index=(cols.index(col_region_auto) + 1) if col_region_auto in cols else 0,
)

# Producción por hectárea (si hay columna de área disponible)
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

# =========================================================
# TABS PRINCIPALES
# =========================================================
tab_datos, tab_eda, tab_pregunta, tab_apoyo, tab_reporte = st.tabs(
    ["📂 Datos", "🔍 EDA General", "🎯 Pregunta de Negocio", "📊 Gráficas de Apoyo", "📝 Reporte y Conclusiones"]
)

# ---------------------------------------------------------
# TAB 1: DATOS
# ---------------------------------------------------------
with tab_datos:
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", df.shape[0])
    c2.metric("Columnas", df.shape[1])
    c3.metric("Duplicados", int(df.duplicated().sum()))

    st.subheader("Tipos de datos")
    st.dataframe(
        pd.DataFrame({"Columna": df.dtypes.index, "Tipo": df.dtypes.astype(str).values}),
        use_container_width=True,
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
        st.plotly_chart(fig_nulos, use_container_width=True)
    else:
        st.success("✅ No se encontraron valores nulos en el dataset.")

# ---------------------------------------------------------
# TAB 2: EDA GENERAL
# ---------------------------------------------------------
with tab_eda:
    st.subheader("Estadística descriptiva (variables numéricas)")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.dataframe(df[num_cols].describe().T, use_container_width=True)

    st.subheader("Distribución de una variable")
    var_sel = st.selectbox("Selecciona una variable numérica", num_cols, index=num_cols.index(col_metric) if col_metric in num_cols else 0)
    colh1, colh2 = st.columns(2)
    with colh1:
        fig_hist = px.histogram(df, x=var_sel, nbins=30, marginal="box", title=f"Distribución de {var_sel}")
        st.plotly_chart(fig_hist, use_container_width=True)
    with colh2:
        fig_box_all = px.box(df, y=var_sel, title=f"Boxplot general de {var_sel}", points="outliers")
        st.plotly_chart(fig_box_all, use_container_width=True)

    st.subheader("Matriz de correlación")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        fig_corr = px.imshow(
            corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
            title="Correlación entre variables numéricas",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Se necesitan al menos 2 variables numéricas para calcular correlaciones.")

    st.subheader("Variables categóricas")
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    if cat_cols:
        var_cat = st.selectbox("Selecciona una variable categórica", cat_cols)
        conteo = df[var_cat].value_counts().reset_index()
        conteo.columns = [var_cat, "conteo"]
        fig_cat = px.bar(conteo, x=var_cat, y="conteo", title=f"Frecuencia de {var_cat}", color="conteo",
                          color_continuous_scale="Greens")
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("No se detectaron variables categóricas.")

# ---------------------------------------------------------
# TAB 3: PREGUNTA DE NEGOCIO
# ---------------------------------------------------------
with tab_pregunta:
    st.subheader("¿El sistema de riego tecnificado impacta la producción por hectárea?")

    # --- Comparativa rápida en tabla (según snippet solicitado) ---
    st.markdown("**Comparativa rápida (promedios por grupo):**")
    resumen = df.groupby(col_riego)[col_metric].mean().rename("Promedio").to_frame()
    resumen["Mediana"] = df.groupby(col_riego)[col_metric].median()
    resumen["Desv. Estándar"] = df.groupby(col_riego)[col_metric].std()
    resumen["N"] = df.groupby(col_riego)[col_metric].count()
    st.table(resumen.style.format({"Promedio": "{:.2f}", "Mediana": "{:.2f}", "Desv. Estándar": "{:.2f}"}))

    # --- Visualización clave: Boxplot ---
    st.markdown("### 📦 Visualización clave: Boxplot de Producción — Con Riego vs. Sin Riego")
    fig_box_key = px.box(
        df, x=col_riego, y=col_metric, color=col_riego, points="all",
        title=f"{metric_label} según Sistema de Riego Tecnificado",
        labels={col_riego: "Sistema de Riego", col_metric: metric_label},
    )
    fig_box_key.update_layout(showlegend=False)
    st.plotly_chart(fig_box_key, use_container_width=True)

    # Versión seaborn/matplotlib de respaldo
    with st.expander("Ver versión estática (seaborn)"):
        fig_sns, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x=col_riego, y=col_metric, ax=ax, palette="Set2")
        ax.set_title(f"{metric_label} según Sistema de Riego")
        st.pyplot(fig_sns)

    # --- Prueba estadística ---
    st.markdown("### 🧪 Prueba estadística")
    grupos = df[col_riego].dropna().unique()

    if len(grupos) == 2:
        g1_name, g2_name = grupos[0], grupos[1]
        g1 = df.loc[df[col_riego] == g1_name, col_metric].dropna()
        g2 = df.loc[df[col_riego] == g2_name, col_metric].dropna()

        t_stat, p_val_t = stats.ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
        u_stat, p_val_u = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        d = cohens_d(g1, g2)

        c1, c2, c3 = st.columns(3)
        c1.metric("p-value (t-test)", f"{p_val_t:.4f}")
        c2.metric("p-value (Mann-Whitney)", f"{p_val_u:.4f}")
        c3.metric("d de Cohen", f"{d:.2f}" if not pd.isna(d) else "N/A")

        diferencia_pct = ((g1.mean() - g2.mean()) / g2.mean()) * 100 if g2.mean() != 0 else np.nan

        significativo = p_val_t < 0.05
        st.session_state["_significativo"] = significativo
        st.session_state["_grupo_mayor"] = g1_name if g1.mean() > g2.mean() else g2_name
        st.session_state["_diferencia_pct"] = diferencia_pct
        st.session_state["_efecto"] = interpretar_d(d)
        st.session_state["_p_val"] = p_val_t

        if significativo:
            st.success(
                f"✅ La diferencia es **estadísticamente significativa** (p = {p_val_t:.4f} < 0.05). "
                f"El grupo **'{st.session_state['_grupo_mayor']}'** presenta un promedio "
                f"{'mayor' if diferencia_pct is not None else ''} en {metric_label.lower()}, "
                f"con un tamaño de efecto **{st.session_state['_efecto']}**."
            )
        else:
            st.warning(
                f"⚠️ La diferencia **no es estadísticamente significativa** (p = {p_val_t:.4f} ≥ 0.05). "
                "No hay evidencia suficiente para afirmar que el riego tecnificado, por sí solo, "
                "explique la diferencia observada en la muestra."
            )
    else:
        st.info(
            f"La columna de riego tiene {len(grupos)} categorías distintas "
            f"({', '.join(map(str, grupos))}). Se muestra el ANOVA correspondiente."
        )
        muestras = [df.loc[df[col_riego] == g, col_metric].dropna() for g in grupos]
        f_stat, p_val = stats.f_oneway(*muestras)
        c1, c2 = st.columns(2)
        c1.metric("Estadístico F", f"{f_stat:.2f}")
        c2.metric("p-value (ANOVA)", f"{p_val:.4f}")
        st.session_state["_significativo"] = p_val < 0.05
        st.session_state["_p_val"] = p_val

# ---------------------------------------------------------
# TAB 4: GRÁFICAS DE APOYO
# ---------------------------------------------------------
with tab_apoyo:
    st.subheader("Gráficas complementarias para reforzar el storytelling")

    # Violin plot
    st.markdown("**Distribución detallada (Violin Plot)**")
    fig_violin = px.violin(
        df, x=col_riego, y=col_metric, color=col_riego, box=True, points="outliers",
        title=f"Distribución de {metric_label} por Sistema de Riego",
    )
    fig_violin.update_layout(showlegend=False)
    st.plotly_chart(fig_violin, use_container_width=True)

    # Scatter Área vs Producción coloreado por riego
    if col_area != "(ninguna)":
        st.markdown("**Relación Área vs. Producción, coloreada por Riego**")
        fig_scatter = px.scatter(
            df, x=col_area, y=col_prod, color=col_riego, trendline="ols",
            title="Área Sembrada vs. Producción Anual",
            labels={col_area: "Área (Hectáreas)", col_prod: "Producción (Ton)"},
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Producción media por Cultivo y Riego
    if col_cultivo != "(ninguna)":
        st.markdown("**Producción media por Cultivo y Sistema de Riego**")
        agg = df.groupby([col_cultivo, col_riego])[col_metric].mean().reset_index()
        fig_bar_cultivo = px.bar(
            agg, x=col_cultivo, y=col_metric, color=col_riego, barmode="group",
            title=f"{metric_label} promedio por Cultivo y Riego",
        )
        st.plotly_chart(fig_bar_cultivo, use_container_width=True)

    # Boxplot por tipo de suelo
    if col_suelo != "(ninguna)":
        st.markdown("**Producción por Tipo de Suelo, según Riego**")
        fig_suelo = px.box(
            df, x=col_suelo, y=col_metric, color=col_riego,
            title=f"{metric_label} por Tipo de Suelo y Sistema de Riego",
        )
        st.plotly_chart(fig_suelo, use_container_width=True)

    # Producción media por región
    if col_region != "(ninguna)":
        st.markdown("**Producción media por Región**")
        agg_reg = df.groupby([col_region, col_riego])[col_metric].mean().reset_index()
        fig_region = px.bar(
            agg_reg, x=col_region, y=col_metric, color=col_riego, barmode="group",
            title=f"{metric_label} promedio por Región y Riego",
        )
        fig_region.update_xaxes(tickangle=45)
        st.plotly_chart(fig_region, use_container_width=True)

    if col_area == "(ninguna)" and col_cultivo == "(ninguna)" and col_suelo == "(ninguna)" and col_region == "(ninguna)":
        st.info("Selecciona columnas adicionales (área, cultivo, suelo, región) en la barra lateral para desbloquear más gráficas.")

# ---------------------------------------------------------
# TAB 5: REPORTE Y CONCLUSIONES
# ---------------------------------------------------------
with tab_reporte:
    st.subheader("📝 Reporte ejecutivo")

    st.markdown(f"**Pregunta de negocio:** ¿El sistema de riego tecnificado impacta realmente la {metric_label.lower()}?")

    st.markdown("**Resumen de hallazgos:**")

    resumen_bullets = []
    resumen_bullets.append(
        f"- El dataset analizado contiene **{df.shape[0]} registros** y **{df.shape[1]} variables**."
    )
    resumen_bullets.append(
        f"- La variable objetivo analizada fue **{metric_label}**, comparada según **{col_riego}**."
    )

    if "_p_val" in st.session_state:
        p_val = st.session_state["_p_val"]
        sig = st.session_state.get("_significativo", False)
        resumen_bullets.append(f"- El valor p obtenido en la prueba estadística fue **{p_val:.4f}**.")
        if sig:
            grupo_mayor = st.session_state.get("_grupo_mayor", "N/A")
            diff = st.session_state.get("_diferencia_pct", np.nan)
            efecto = st.session_state.get("_efecto", "N/A")
            resumen_bullets.append(
                f"- Existe una diferencia **estadísticamente significativa** entre los grupos "
                f"(tamaño de efecto: **{efecto}**)."
            )
            if grupo_mayor != "N/A" and not pd.isna(diff):
                resumen_bullets.append(
                    f"- El grupo **'{grupo_mayor}'** muestra el mayor promedio de {metric_label.lower()}, "
                    f"con una diferencia aproximada de **{abs(diff):.1f}%** frente al otro grupo."
                )
            resumen_bullets.append(
                "- **Conclusión:** la evidencia estadística respalda que el sistema de riego "
                "tecnificado sí está asociado a un cambio real en la producción por hectárea. "
                "Se recomienda profundizar el análisis controlando por variables como tipo de suelo, "
                "cultivo y región para descartar factores de confusión."
            )
        else:
            resumen_bullets.append(
                "- **Conclusión:** con los datos disponibles, no se encontró una diferencia "
                "estadísticamente significativa entre fincas con y sin riego tecnificado. "
                "Esto sugiere que, en esta muestra, el riego por sí solo no explica de forma "
                "concluyente las diferencias en producción; podrían existir otros factores "
                "(suelo, cultivo, clima, manejo agronómico) con mayor influencia."
            )
    else:
        resumen_bullets.append("- Aún no se ha calculado la prueba estadística. Revisa la pestaña 'Pregunta de Negocio'.")

    st.markdown("\n".join(resumen_bullets))

    st.markdown("---")
    st.markdown("**Tabla resumen final:**")
    resumen_final = df.groupby(col_riego)[col_metric].agg(["mean", "median", "std", "count"])
    resumen_final.columns = ["Promedio", "Mediana", "Desv. Estándar", "N"]
    st.table(resumen_final.style.format({"Promedio": "{:.2f}", "Mediana": "{:.2f}", "Desv. Estándar": "{:.2f}"}))

    st.markdown("**Boxplot de referencia (resumen visual del hallazgo principal):**")
    st.plotly_chart(fig_box_key, use_container_width=True)

    st.download_button(
        label="📥 Descargar reporte en texto",
        data="\n".join(resumen_bullets).replace("**", ""),
        file_name="reporte_riego_tecnificado.txt",
        mime="text/plain",
    )

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard EDA · Agro Colombia · Riego Tecnificado")
