import os
import json
import uuid
import pandas as pd
import streamlit as st
from datetime import datetime
from openai import OpenAI
import requests  # para conectar con n8n

# ------------------------------
# Configuración inicial
# ------------------------------
MODEL = "gpt-4o-mini"  # económico y rápido
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta la variable de entorno OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_FILE = os.path.abspath(os.path.join(DATA_PATH, "pisos.csv"))
N8N_WEBHOOK = os.getenv("N8N_WEBHOOK_URL")  # URL del webhook de n8n

REQUIRED_SLOTS = [
    "precio", "barrio_ciudad", "m2", "habitaciones", "banos", "disponibilidad"
]
OPTIONAL_SLOTS = [
    "planta", "ascensor", "amueblado", "mascotas", "estado"
]
ALL_SLOTS = REQUIRED_SLOTS + OPTIONAL_SLOTS


# ------------------------------
# Utilidades de datos
# ------------------------------
def ensure_csv_schema():
    os.makedirs(DATA_PATH, exist_ok=True)
    if not os.path.exists(CSV_FILE):
        cols = [
            "id_piso", "descripcion_original", "descripcion_ia",
            "precio", "barrio_ciudad", "m2", "habitaciones", "banos",
            "planta", "ascensor", "amueblado", "mascotas", "disponibilidad",
            # Campos reservados para M2/M3
            "distancia_metro_m", "score_conectividad",
            "score_visual_global", "fotos_faltantes_sugeridas",
            "created_at"
        ]
        pd.DataFrame(columns=cols).to_csv(CSV_FILE, index=False)


def save_listing(record):
    df = pd.read_csv(CSV_FILE)
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    # Enviar el registro al webhook n8n si está configurado
    if N8N_WEBHOOK:
        try:
            requests.post(N8N_WEBHOOK, json=record, timeout=5)
        except Exception as e:
            print("No se pudo enviar a n8n:", e)


# ------------------------------
# LLM: extracción de campos
# ------------------------------
def extract_slots(description: str) -> dict:
    """Usa el modelo GPT para extraer datos estructurados."""
    system_prompt = (
        "Eres un asistente que extrae campos estructurados de descripciones de pisos en alquiler en España. "
        "Devuelve SOLO un JSON válido con las claves: "
        + ", ".join(ALL_SLOTS)
        + ". Usa null cuando falte el dato. "
        "Normaliza así: precio en euros (int), m2 (int), habitaciones/banos (int), planta (int o null), "
        "ascensor/amueblado/mascotas (true/false/null), disponibilidad en formato ISO YYYY-MM-DD o null, "
        "estado en {'reformado','a reformar','bueno'} o null. "
        "Para barrio_ciudad devuelve 'Barrio, Ciudad' si es posible."
    )
    user_prompt = (
        "Texto del propietario: " + description + "\n\nDevuelve SOLO el JSON, sin texto adicional."
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        start, end = content.find("{"), content.rfind("}") + 1
        data = json.loads(content[start:end])
    for k in ALL_SLOTS:
        data.setdefault(k, None)
    return data


# ------------------------------
# Validaciones y helpers
# ------------------------------
def validate_slots(slots: dict) -> list:
    problems = []
    def to_int(v):
        try:
            return int(v)
        except:
            return None
    m2 = to_int(slots.get("m2"))
    hab = to_int(slots.get("habitaciones"))
    ban = to_int(slots.get("banos"))
    precio = to_int(slots.get("precio"))

    if m2 and m2 < 25:
        problems.append("⚠️ m2 parece demasiado bajo (<25)")
    if hab and m2 and hab > m2 // 8:
        problems.append("⚠️ Demasiadas habitaciones para los m2")
    if ban and ban > 5:
        problems.append("⚠️ Número de baños inusual (>5)")
    if precio is not None and precio <= 0:
        problems.append("⚠️ Precio debe ser mayor que 0")
    return problems


def missing_required(slots: dict) -> list:
    return [k for k in REQUIRED_SLOTS if not slots.get(k)]


def make_questions(slots: dict) -> list:
    q = []
    if not slots.get("precio"):
        q.append("¿Cuál es el precio mensual en euros?")
    if not slots.get("barrio_ciudad"):
        q.append("¿En qué barrio y ciudad está el piso? (Ej.: 'Sant Gervasi, Barcelona')")
    if not slots.get("m2"):
        q.append("¿Cuántos metros cuadrados tiene?")
    if not slots.get("habitaciones"):
        q.append("¿Cuántas habitaciones tiene?")
    if not slots.get("banos"):
        q.append("¿Cuántos baños tiene?")
    if not slots.get("disponibilidad"):
        q.append("¿Desde qué fecha está disponible? (YYYY-MM-DD)")
    return q


def make_summary(slots: dict) -> str:
    asc = slots.get("ascensor")
    asc_txt = "con ascensor" if asc else "sin ascensor"
    amu = slots.get("amueblado")
    amu_txt = "amueblado" if amu else "sin amueblar"
    mas = slots.get("mascotas")
    mas_txt = "se aceptan mascotas" if mas else "no mascotas"
    return (
        f"🏠 Piso en {slots.get('barrio_ciudad') or 'ubicación n/d'} | "
        f"{slots.get('habitaciones') or 'n/d'} hab, {slots.get('m2') or 'n/d'} m², "
        f"{slots.get('banos') or 'n/d'} baños, {slots.get('planta') or 'n/d'}ª, {asc_txt}.\n"
        f"💶 {slots.get('precio') or 'n/d'} €/mes | 📅 Disponible {slots.get('disponibilidad') or 'n/d'} | "
        f"{amu_txt}, {mas_txt}."
    )


# ------------------------------
# Interfaz Streamlit
# ------------------------------
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "slots" not in st.session_state:
        st.session_state.slots = {k: None for k in ALL_SLOTS}
    if "descripcion_original" not in st.session_state:
        st.session_state.descripcion_original = ""


def app():
    st.set_page_config(page_title="RentMatch AI — Alta del piso", page_icon="🏠", layout="wide")
    st.title("M1 — Alta conversacional del piso")
    st.caption("Describe tu piso con tus palabras. Te preguntaré solo lo imprescindible.")
    init_state()
    ensure_csv_schema()

    col1, col2 = st.columns([3, 2])
    with col1:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        prompt = st.chat_input("Escribe aquí la descripción del piso o responde a las preguntas…")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            if not st.session_state.descripcion_original:
                st.session_state.descripcion_original = prompt
                with st.spinner("Analizando descripción…"):
                    extracted = extract_slots(prompt)
                for k in ALL_SLOTS:
                    if extracted.get(k) is not None:
                        st.session_state.slots[k] = extracted[k]
                questions = make_questions(st.session_state.slots)
                if questions:
                    bot = "Gracias. Me faltan algunos datos:\n- " + "\n- ".join(questions)
                else:
                    bot = "Perfecto, ya tengo lo necesario. Revisa la ficha a la derecha y pulsa Guardar."
                st.session_state.messages.append({"role": "assistant", "content": bot})
            else:
                missing = missing_required(st.session_state.slots)
                if missing:
                    field = missing[0]
                    st.session_state.slots[field] = prompt
                    missing = missing_required(st.session_state.slots)
                    if missing:
                        q = make_questions(st.session_state.slots)[0]
                        st.session_state.messages.append({"role": "assistant", "content": q})
                    else:
                        st.session_state.messages.append(
                            {"role": "assistant", "content": "¡Listo! Revisa la ficha y pulsa Guardar."}
                        )
                else:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "He anotado tu comentario. Puedes ajustar en la ficha de la derecha."}
                    )

    with col2:
        st.subheader("Ficha generada")
        with st.expander("Campos (editar si hace falta)", expanded=True):
            for k in ALL_SLOTS:
                v = st.session_state.slots.get(k)
                st.session_state.slots[k] = st.text_input(k, value="" if v is None else str(v))

        st.write("---")
        st.subheader("Resumen para el anuncio")
        st.write(make_summary(st.session_state.slots))
        problems = validate_slots(st.session_state.slots)
        if problems:
            st.warning("; ".join(problems))

        if st.button("Guardar piso en base de datos", type="primary"):
            missing = missing_required(st.session_state.slots)
            if missing:
                st.error("Faltan campos obligatorios: " + ", ".join(missing))
            else:
                rec = {
                    "id_piso": str(uuid.uuid4()),
                    "descripcion_original": st.session_state.descripcion_original,
                    "descripcion_ia": make_summary(st.session_state.slots),
                    **{k: st.session_state.slots.get(k) for k in ALL_SLOTS},
                    "distancia_metro_m": None,
                    "score_conectividad": None,
                    "score_visual_global": None,
                    "fotos_faltantes_sugeridas": None,
                    "created_at": datetime.utcnow().isoformat()
                }
                save_listing(rec)
                st.success("✅ Piso guardado y enviado a n8n (si está configurado).")

        st.write("---")
        st.subheader("Hooks M2 / M3 (simulados)")
        if st.button("Enriquecer entorno (M2 simulado)"):
            df = pd.read_csv(CSV_FILE)
            df.iloc[-1, df.columns.get_loc("distancia_metro_m")] = 350
            df.iloc[-1, df.columns.get_loc("score_conectividad")] = 0.78
            df.to_csv(CSV_FILE, index=False)
            st.info("Se han rellenado datos de entorno (simulados).")

        if st.button("Analizar fotos (M3 simulado)"):
            df = pd.read_csv(CSV_FILE)
            df.iloc[-1, df.columns.get_loc("score_visual_global")] = 0.72
            df.iloc[-1, df.columns.get_loc("fotos_faltantes_sugeridas")] = "fachada, salón, dormitorio principal"
            df.to_csv(CSV_FILE, index=False)
            st.info("Se han rellenado los campos visuales (simulados).")


if __name__ == "__main__":
    app()
