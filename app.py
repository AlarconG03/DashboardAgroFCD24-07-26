"""
Bot Conversacional de Cultura General e Historia Mundial
Motor: Groq API — Modelo Llama 3.3 70B (llama-3.3-70b-versatile)
"""

import json
import re

import streamlit as st
from groq import Groq

MODEL_ID = "llama-3.3-70b-versatile"

SYSTEM_PROMPT_CHAT = (
    "Eres un asistente experto en cultura general e historia mundial. "
    "Respondes con precisión, das contexto histórico relevante (fechas, "
    "personajes, causas y consecuencias) y, cuando es útil, agregas datos "
    "curiosos. Si la pregunta del usuario no tiene que ver con cultura "
    "general o historia, respóndela de todas formas de forma breve y "
    "amable, pero intenta reconducir la conversación hacia temas de "
    "historia o cultura general con una pregunta de seguimiento. "
    "Responde siempre en español, de forma clara y estructurada "
    "(usa listas o negritas cuando ayude a la claridad)."
)

SYSTEM_PROMPT_TRIVIA = (
    "Eres un generador de preguntas de trivia de cultura general e "
    "historia mundial (geografía, arte, ciencia, historia antigua, "
    "moderna y contemporánea, mitología, política, etc.). "
    "Debes responder ÚNICAMENTE con un objeto JSON válido, sin texto "
    "adicional, sin backticks ni explicaciones, con este formato exacto:\n"
    '{"pregunta": "...", "opciones": {"A": "...", "B": "...", "C": "...", '
    '"D": "..."}, "respuesta_correcta": "A", "explicacion": "..."}\n'
    "La 'explicacion' debe dar contexto histórico o cultural breve "
    "(máx. 3 líneas) de por qué esa es la respuesta correcta. "
    "Varía el tema y la dificultad en cada pregunta. No repitas preguntas "
    "ya usadas en la conversación."
)

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bot de Cultura General e Historia Mundial",
    page_icon="🏛️",
    layout="centered",
)

st.title("🏛️ Bot de Cultura General e Historia Mundial")
st.caption(f"Potenciado por Groq · Modelo `{MODEL_ID}`")

# ---------------------------------------------------------------------------
# SIDEBAR: API KEY Y CONFIGURACIÓN
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Tu API Key de Groq. No se almacena ni se envía a ningún lado "
             "más que a la API oficial de Groq.",
    )

    st.markdown("---")

    modo = st.radio(
        "Modo de uso",
        ["💬 Chat libre", "🎯 Modo Trivia"],
        help="Chat libre: conversa y pregunta lo que quieras. "
             "Modo Trivia: el bot te hace preguntas de opción múltiple.",
    )

    st.markdown("---")
    st.subheader("Parámetros del modelo")
    temperature = st.slider("Temperatura (creatividad)", 0.0, 1.5, 0.6, 0.1)
    max_tokens = st.slider("Máximo de tokens por respuesta", 128, 2048, 700, 64)

    st.markdown("---")
    if st.button("🗑️ Reiniciar conversación / puntaje"):
        for key in ["messages", "trivia_score", "trivia_total",
                    "pregunta_actual", "respondida"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.markdown("---")
    st.caption(
        "¿No tienes API Key? Consíguela gratis en "
        "[console.groq.com](https://console.groq.com/keys)."
    )

# ---------------------------------------------------------------------------
# ESTADO DE SESIÓN
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "trivia_score" not in st.session_state:
    st.session_state.trivia_score = 0
if "trivia_total" not in st.session_state:
    st.session_state.trivia_total = 0
if "pregunta_actual" not in st.session_state:
    st.session_state.pregunta_actual = None
if "respondida" not in st.session_state:
    st.session_state.respondida = True  # True = lista para generar nueva

# ---------------------------------------------------------------------------
# VALIDACIÓN DE API KEY
# ---------------------------------------------------------------------------
if not api_key:
    st.info(
        "👈 Ingresa tu **Groq API Key** en la barra lateral para comenzar. "
        "Puedes obtenerla gratis en console.groq.com/keys."
    )
    st.stop()

try:
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"❌ No se pudo inicializar el cliente de Groq: {e}")
    st.stop()


def llamar_groq(mensajes, system_prompt, temp=0.6, tokens=700, forzar_json=False):
    """Llama a la API de Groq y devuelve el texto de respuesta."""
    payload = [{"role": "system", "content": system_prompt}] + mensajes
    kwargs = dict(
        model=MODEL_ID,
        messages=payload,
        temperature=temp,
        max_tokens=tokens,
    )
    if forzar_json:
        kwargs["response_format"] = {"type": "json_object"}
    respuesta = client.chat.completions.create(**kwargs)
    return respuesta.choices[0].message.content


def extraer_json(texto):
    """Extrae un bloque JSON de la respuesta del modelo, por si viene
    envuelto en texto o backticks."""
    texto = texto.strip()
    texto = re.sub(r"^```(json)?|```$", "", texto, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", texto, flags=re.DOTALL)
    if match:
        texto = match.group(0)
    return json.loads(texto)


# ---------------------------------------------------------------------------
# MODO 1: CHAT LIBRE
# ---------------------------------------------------------------------------
if modo == "💬 Chat libre":
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pregunta_usuario = st.chat_input(
        "Pregúntame sobre historia mundial o cultura general..."
    )

    if pregunta_usuario:
        st.session_state.messages.append({"role": "user", "content": pregunta_usuario})
        with st.chat_message("user"):
            st.markdown(pregunta_usuario)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            texto_acumulado = ""
            try:
                stream = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
                    + st.session_state.messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    texto_acumulado += delta
                    placeholder.markdown(texto_acumulado + "▌")
                placeholder.markdown(texto_acumulado)
            except Exception as e:
                texto_acumulado = f"❌ Error al llamar a la API de Groq: {e}"
                placeholder.markdown(texto_acumulado)

        st.session_state.messages.append(
            {"role": "assistant", "content": texto_acumulado}
        )

    if not st.session_state.messages:
        st.markdown("##### 💡 Ideas para empezar:")
        ejemplos = [
            "¿Cuáles fueron las causas de la Primera Guerra Mundial?",
            "Cuéntame sobre la caída del Imperio Romano",
            "¿Quién fue Simón Bolívar y por qué es importante?",
            "Explícame la Guerra Fría en pocas palabras",
        ]
        cols = st.columns(2)
        for i, ej in enumerate(ejemplos):
            if cols[i % 2].button(ej, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": ej})
                st.rerun()

# ---------------------------------------------------------------------------
# MODO 2: TRIVIA
# ---------------------------------------------------------------------------
else:
    c1, c2 = st.columns(2)
    c1.metric("✅ Aciertos", st.session_state.trivia_score)
    c2.metric("📊 Preguntas respondidas", st.session_state.trivia_total)

    st.markdown("---")

    # Generar nueva pregunta si corresponde
    if st.session_state.respondida:
        if st.button("🎲 Generar nueva pregunta", type="primary", use_container_width=True):
            with st.spinner("Generando pregunta..."):
                try:
                    historial_previas = [
                        m["content"] for m in st.session_state.messages
                        if m["role"] == "assistant"
                    ][-5:]
                    contexto = (
                        [{"role": "assistant", "content": p} for p in historial_previas]
                        + [{"role": "user", "content": "Genera una nueva pregunta de trivia."}]
                    )
                    raw = llamar_groq(
                        contexto, SYSTEM_PROMPT_TRIVIA,
                        temp=max(temperature, 0.7), tokens=max_tokens,
                        forzar_json=True,
                    )
                    pregunta_json = extraer_json(raw)
                    st.session_state.pregunta_actual = pregunta_json
                    st.session_state.messages.append(
                        {"role": "assistant", "content": raw}
                    )
                    st.session_state.respondida = False
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error generando la pregunta: {e}")

    # Mostrar pregunta activa
    if st.session_state.pregunta_actual and not st.session_state.respondida:
        p = st.session_state.pregunta_actual
        st.markdown(f"### ❓ {p['pregunta']}")

        opcion_elegida = st.radio(
            "Elige tu respuesta:",
            options=list(p["opciones"].keys()),
            format_func=lambda k: f"{k}) {p['opciones'][k]}",
            index=None,
            key=f"radio_{st.session_state.trivia_total}",
        )

        if st.button("✅ Confirmar respuesta", disabled=(opcion_elegida is None)):
            correcta = p["respuesta_correcta"]
            st.session_state.trivia_total += 1
            if opcion_elegida == correcta:
                st.session_state.trivia_score += 1
                st.success(f"¡Correcto! 🎉 La respuesta era **{correcta}) "
                           f"{p['opciones'][correcta]}**")
            else:
                st.error(
                    f"Incorrecto ❌. Tu respuesta: **{opcion_elegida}**. "
                    f"La correcta era **{correcta}) {p['opciones'][correcta]}**"
                )
            st.info(f"📖 **Contexto:** {p['explicacion']}")
            st.session_state.respondida = True

    elif st.session_state.respondida and st.session_state.trivia_total == 0:
        st.markdown(
            "Presiona **Generar nueva pregunta** para comenzar el reto de "
            "cultura general e historia mundial 🌍"
        )

st.sidebar.markdown("---")
st.sidebar.caption("Bot de Cultura General e Historia Mundial · Groq + Llama 3.3 70B")
