import os
import json
import uuid
import pandas as pd
import streamlit as st
from datetime import datetime
from openai import OpenAI
import requests  # para conectar con n8n
import re

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

SPANISH_MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12
}

def parse_number(text: str) -> int | None:
    """Extrae el primer número razonable del texto (soporta 1.200, 1 200, 1200, 1.200,50 -> 1200)."""
    if not text:
        return None
    m = re.search(r"(\d{1,3}(?:[.\s]\d{3})+|\d+)(?:[.,]\d+)?", text)
    if not m:
        return None
    num = m.group(0)
    # Quitar separadores de miles . o espacios
    num = re.sub(r"[.\s]", "", num)
    # Si hay coma como decimal, ignórala para estos campos (precio, m2)
    num = num.split(",")[0]
    try:
        return int(float(num))
    except:
        return None

def parse_bool(text: str) -> bool | None:
    """Convierte respuestas tipo sí/no en True/False. Busca palabras típicas."""
    if not text:
        return None
    t = text.strip().lower()
    yes = {"si", "sí", "yes", "true", "con", "tiene", "hay", "permitido", "permiten"}
    no = {"no", "false", "sin", "no hay", "no permitido", "no permiten"}
    # señales fuertes
    if any(w in t for w in ["no ", " sin", "no.", "no,", "no\t"]) and not any(w in t for w in ["sí", "si"]):
        return False
    if any(w in t.split() for w in yes):
        return True
    if any(w in t.split() for w in no):
        return False
    return None

def parse_date_es(text: str) -> str | None:
    """
    Intenta devolver YYYY-MM-DD a partir de formatos:
    - YYYY-MM-DD
    - DD/MM/YYYY o DD-MM-YYYY
    - '15 de diciembre de 2025'
    - 'inmediata' -> hoy
    """
    if not text:
        return None
    t = text.strip().lower()

    # inmediata
    if "inmediata" in t or "ya" in t or "hoy" in t:
        return datetime.today().date().isoformat()

    # ISO directo
    m = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$", t)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # DD/MM/YYYY o DD-MM-YYYY
    m = re.match(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})\s*$", t)
    if m:
        dd, mm, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(yyyy, mm, dd).date().isoformat()
        except:
            return None

    # '15 de diciembre de 2025'
    m = re.match(r"^\s*(\d{1,2})\s+de\s+([a-záéíóú]+)\s+de\s+(\d{4})\s*$", t)
    if m:
        dd = int(m.group(1))
        mes = m.group(2).replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u")
        yyyy = int(m.group(3))
        mm = SPANISH_MONTHS.get(mes, 0)
        if mm:
            try:
                return datetime(yyyy, mm, dd).date().isoformat()
            except:
                return None

    return None

def normalize_field(field: str, text: str, *, smart_fallback: bool = bool(int(os.getenv("SMART_FALLBACK", "0")))) -> object:
    """
    Devuelve el valor normalizado para un campo dado.
    No llama a GPT salvo que smart_fallback esté activo y la normalización local falle.
    """
    if text is None:
        return None
    raw = text.strip()

    if field == "precio":
        v = parse_number(raw)
        if v is not None:
            return v
    elif field == "m2":
        v = parse_number(raw)
        if v is not None:
            return v
    elif field in ("habitaciones", "banos", "planta"):
        v = parse_number(raw)
        # Tratamientos semánticos sencillos para planta
        if v is None and field == "planta":
            low = raw.lower()
            if "bajo" in low:
                return 0
            if "principal" in low:
                return 1
        if v is not None:
            return v
    elif field in ("ascensor", "amueblado", "mascotas"):
        v = parse_bool(raw)
        if v is not None:
            return v
    elif field == "disponibilidad":
        v = parse_date_es(raw)
        if v is not None:
            return v
    elif field == "barrio_ciudad":
        # Limpieza básica; más normalización vendrá en M2 si queréis
        return raw

    # Si no pudimos normalizar y tenemos fallback “barato” activado:
    if smart_fallback:
        try:
            prompt = f"Extrae solo el valor atómico para el campo '{field}' a partir del texto entre <t>:</t>. " \
                     f"Responde sin explicación. Si es precio/m2/enteros: un número. Si es booleano: true/false. " \
                     f"Si es fecha: YYYY-MM-DD. Texto: <t>{raw}</t>"
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            ans = resp.choices[0].message.content.strip()
            # Post-procesado mínimo
            if field in ("precio", "m2", "habitaciones", "banos", "planta"):
                return parse_number(ans)
            if field in ("ascensor", "amueblado", "mascotas"):
                return True if "true" in ans.lower() or "sí" in ans.lower() or "si" in ans.lower() else False if "false" in ans.lower() or "no" in ans.lower() else None
            if field == "disponibilidad":
                return parse_date_es(ans) or ans
            return ans
        except Exception:
            pass

    # Último recurso: devolver el texto crudo (como antes)
    return raw

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
    st.markdown(APP_CSS, unsafe_allow_html=True)

    # Header
    colh1, colh2 = st.columns([1, 4])
    with colh1:
        st.write("### 🏠")
    with colh2:
        st.write("# RentMatch AI — M1")
        st.caption("Alta conversacional del piso · Demo Cloud")

    # Sidebar con estado y acciones rápidas
    with st.sidebar:
        st.write("### Progreso")
        done = sum(1 for k in REQUIRED_SLOTS if st.session_state.get("slots", {}).get(k))
        st.progress(done/len(REQUIRED_SLOTS))
        st.write(f"Completados: **{done}/{len(REQUIRED_SLOTS)}**")

        st.write("### Pasos")
        st.markdown(
            "- 1) Describe el piso\n"
            "- 2) Responde a lo que falte\n"
            "- 3) Revisa y **Guarda**"
        )
        st.write("### Enlaces")
        st.markdown(
            "- Supabase (si conectado)\n"
            "- n8n Webhook (si configurado)",
        )
        st.divider()
        st.write("### Info")
        st.markdown("<span class='rm-muted'>Los datos se normalizan localmente y se envían a n8n al guardar.</span>", unsafe_allow_html=True)

    # Estado inicial
    init_state()
    ensure_csv_schema()

    # Tabs principales
    tab_chat, tab_ficha, tab_datos = st.tabs(["💬 Conversación", "🧾 Ficha", "📊 Datos & Hooks"])

    # ===== TAB: Conversación =====
    with tab_chat:
        st.markdown("<div class='rm-card'>", unsafe_allow_html=True)
        st.subheader("Describe tu piso")
        st.caption("Escribe libremente. Solo te preguntaremos lo imprescindible.")

        # Histórico de chat
        for msg in st.session_state.messages:
            role = "🤖" if msg["role"] == "assistant" else "👤"
            st.markdown(f"**{role}** {msg['content']}")

        prompt = st.chat_input("Escribe la descripción o responde a las preguntas…")
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
                    chips = " ".join([f"<span class='rm-chip'>{q}</span>" for q in questions])
                    bot = "Gracias. Me faltan algunos datos:<br/>" + chips
                else:
                    bot = "Perfecto, ya tengo lo necesario. Ve a **Ficha** para revisar y guardar."
                st.session_state.messages.append({"role": "assistant", "content": bot})
            else:
                missing = missing_required(st.session_state.slots)
                if missing:
                    field = missing[0]
                    value = normalize_field(field, prompt)
                    st.session_state.slots[field] = value
                    missing = missing_required(st.session_state.slots)
                    if missing:
                        q = make_questions(st.session_state.slots)[0]
                        st.session_state.messages.append({"role": "assistant", "content": q})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "¡Listo! Revisa la **Ficha** y pulsa **Guardar**."})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "He anotado tu comentario. Ajusta en **Ficha** si lo necesitas."})

        st.markdown("</div>", unsafe_allow_html=True)

    # ===== TAB: Ficha =====
    with tab_ficha:
        st.markdown("<div class='rm-card'>", unsafe_allow_html=True)
        st.subheader("Ficha del anuncio")

        # Campos en dos columnas
        c1, c2 = st.columns(2)
        with c1:
            for k in ["precio", "barrio_ciudad", "m2", "habitaciones", "banos", "disponibilidad"]:
                v = st.session_state.slots.get(k)
                st.session_state.slots[k] = st.text_input(k, value="" if v is None else str(v))
        with c2:
            for k in ["planta", "ascensor", "amueblado", "mascotas", "estado"]:
                v = st.session_state.slots.get(k)
                st.session_state.slots[k] = st.text_input(k, value="" if v is None else str(v))

        st.write("---")
        st.write("**Resumen:**")
        st.info(make_summary(st.session_state.slots))

        problems = validate_slots(st.session_state.slots)
        if problems:
            st.warning(" ; ".join(problems))

        cta_col1, cta_col2 = st.columns([1, 2])
        with cta_col1:
            save_click = st.button("💾 Guardar piso", type="primary")
        with cta_col2:
            st.caption("Se almacenará en CSV (efímero) y se enviará a n8n / Supabase si están configurados.")

        if save_click:
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
                st.success("✅ Guardado correcto. Enviado a n8n/Supabase si procede.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ===== TAB: Datos & Hooks =====
    with tab_datos:
        st.markdown("<div class='rm-card'>", unsafe_allow_html=True)
        st.subheader("Datos recientes y acciones")
        # Mostrar últimas filas del CSV (si existe)
        try:
            df = pd.read_csv(CSV_FILE)
            st.dataframe(df.tail(5), use_container_width=True)
        except Exception:
            st.caption("No hay datos aún.")

        st.write("---")
        st.write("**Hooks de demostración**")
        hc1, hc2 = st.columns(2)
        with hc1:
            if st.button("🛰️ Enriquecer entorno (M2 simulado)"):
                try:
                    df = pd.read_csv(CSV_FILE)
                    df.iloc[-1, df.columns.get_loc("distancia_metro_m")] = 350
                    df.iloc[-1, df.columns.get_loc("score_conectividad")] = 0.78
                    df.to_csv(CSV_FILE, index=False)
                    st.toast("Datos de entorno simulados añadidos.")
                except Exception:
                    st.error("No hay registros para actualizar.")
        with hc2:
            if st.button("🖼️ Analizar fotos (M3 simulado)"):
                try:
                    df = pd.read_csv(CSV_FILE)
                    df.iloc[-1, df.columns.get_loc("score_visual_global")] = 0.72
                    df.iloc[-1, df.columns.get_loc("fotos_faltantes_sugeridas")] = "fachada, salón, dormitorio principal"
                    df.to_csv(CSV_FILE, index=False)
                    st.toast("Campos de análisis visual simulados añadidos.")
                except Exception:
                    st.error("No hay registros para actualizar.")

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    app()
