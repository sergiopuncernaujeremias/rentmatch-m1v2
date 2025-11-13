import os
import json
import uuid
import pandas as pd
import streamlit as st
from datetime import datetime
from openai import OpenAI
import requests
import re

# ------------------------------
# Configuración inicial
# ------------------------------
MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta la variable de entorno OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# CSV solo para debug local, datos reales van a Supabase vía n8n
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_FILE = os.path.abspath(os.path.join(DATA_PATH, "pisos_debug.csv"))
N8N_WEBHOOK = os.getenv("N8N_WEBHOOK_URL")

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
    """Crea CSV de debug si no existe."""
    os.makedirs(DATA_PATH, exist_ok=True)
    if not os.path.exists(CSV_FILE):
        cols = [
            "id_piso", "descripcion_original", "descripcion_ia",
            "precio", "barrio_ciudad", "m2", "habitaciones", "banos",
            "planta", "ascensor", "amueblado", "mascotas", "disponibilidad",
            "distancia_metro_m", "score_conectividad",
            "score_visual_global", "fotos_faltantes_sugeridas",
            "created_at", "webhook_status"
        ]
        pd.DataFrame(columns=cols).to_csv(CSV_FILE, index=False)


def save_listing(record):
    """
    Guarda el registro en Supabase vía n8n webhook.
    CSV solo para debug local.
    Retorna (success: bool, message: str)
    """
    webhook_status = "no_webhook"
    error_msg = None
    
    if N8N_WEBHOOK:
        try:
            response = requests.post(N8N_WEBHOOK, json=record, timeout=10)
            response.raise_for_status()
            webhook_status = "success"
        except requests.exceptions.Timeout:
            webhook_status = "timeout"
            error_msg = "El webhook tardó demasiado en responder (>10s)"
        except requests.exceptions.RequestException as e:
            webhook_status = "error"
            error_msg = f"Error al enviar a n8n: {str(e)}"
    
    # Guardar en CSV local (debug)
    record["webhook_status"] = webhook_status
    try:
        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
    except Exception as e:
        print(f"Warning: No se pudo guardar en CSV debug: {e}")
    
    if webhook_status == "success":
        return True, "✅ Piso guardado correctamente en Supabase"
    elif webhook_status == "no_webhook":
        return False, "⚠️ No hay webhook configurado. Configura N8N_WEBHOOK_URL"
    else:
        return False, f"❌ Error al guardar: {error_msg}"


# ------------------------------
# LLM: extracción de campos (con caché)
# ------------------------------
def extract_slots(description: str) -> dict:
    """Usa GPT para extraer datos estructurados. Se cachea para evitar llamadas repetidas."""
    cache_key = f"extracted_{hash(description)}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
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
    user_prompt = "Texto del propietario: " + description + "\n\nDevuelve SOLO el JSON, sin texto adicional."

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
    
    st.session_state[cache_key] = data
    return data


# ------------------------------
# Validaciones mejoradas
# ------------------------------
def validate_slots(slots: dict) -> tuple:
    """Valida los slots y retorna (is_valid, errors)."""
    errors = []
    warnings = []
    
    def to_int(v):
        try:
            return int(v)
        except:
            return None
    
    m2 = to_int(slots.get("m2"))
    hab = to_int(slots.get("habitaciones"))
    ban = to_int(slots.get("banos"))
    precio = to_int(slots.get("precio"))
    
    # ERRORES CRÍTICOS (bloquean guardado)
    if precio is not None and precio <= 0:
        errors.append("❌ El precio debe ser mayor que 0")
    if m2 is not None and m2 <= 0:
        errors.append("❌ Los m² deben ser mayor que 0")
    if hab is not None and hab <= 0:
        errors.append("❌ Las habitaciones deben ser mayor que 0")
    if ban is not None and ban <= 0:
        errors.append("❌ Los baños deben ser mayor que 0")
    
    # WARNINGS (permiten guardado pero alertan)
    if m2 and m2 < 25:
        warnings.append("⚠️ m² parece bajo (<25). ¿Es correcto?")
    if hab and m2 and hab > m2 // 8:
        warnings.append("⚠️ Muchas habitaciones para los m². Verifica.")
    if ban and ban > 5:
        warnings.append("⚠️ Número de baños inusual (>5). Verifica.")
    
    is_valid = len(errors) == 0
    return is_valid, errors + warnings


def missing_required(slots: dict) -> list:
    """Retorna lista de campos requeridos que faltan."""
    return [k for k in REQUIRED_SLOTS if not slots.get(k)]


# ------------------------------
# Helpers de UI
# ------------------------------
def make_questions(slots: dict) -> list:
    """Genera preguntas para campos faltantes."""
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
    """Genera resumen legible del piso."""
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
# Normalización de campos
# ------------------------------
SPANISH_MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12
}

def parse_number(text: str):
    """Extrae el primer número del texto."""
    if not text:
        return None
    m = re.search(r"(\d{1,3}(?:[.\s]\d{3})+|\d+)(?:[.,]\d+)?", text)
    if not m:
        return None
    num = m.group(0)
    num = re.sub(r"[.\s]", "", num)
    num = num.split(",")[0]
    try:
        return int(float(num))
    except:
        return None

def parse_bool(text: str):
    """Convierte respuestas tipo sí/no en True/False."""
    if not text:
        return None
    t = text.strip().lower()
    yes = {"si", "sí", "yes", "true", "con", "tiene", "hay", "permitido", "permiten"}
    no = {"no", "false", "sin", "no hay", "no permitido", "no permiten"}
    
    if any(w in t for w in ["no ", " sin", "no.", "no,", "no\t"]) and not any(w in t for w in ["sí", "si"]):
        return False
    if any(w in t.split() for w in yes):
        return True
    if any(w in t.split() for w in no):
        return False
    return None

def parse_date_es(text: str):
    """Convierte fechas en español a YYYY-MM-DD."""
    if not text:
        return None
    t = text.strip().lower()

    if "inmediata" in t or "ya" in t or "hoy" in t:
        return datetime.today().date().isoformat()

    m = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$", t)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    m = re.match(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})\s*$", t)
    if m:
        dd, mm, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(yyyy, mm, dd).date().isoformat()
        except:
            return None

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

def normalize_field(field: str, text: str):
    """Normaliza un campo según su tipo."""
    if text is None:
        return None
    raw = text.strip()

    if field == "precio":
        return parse_number(raw)
    elif field == "m2":
        return parse_number(raw)
    elif field in ("habitaciones", "banos", "planta"):
        v = parse_number(raw)
        if v is None and field == "planta":
            low = raw.lower()
            if "bajo" in low:
                return 0
            if "principal" in low:
                return 1
        return v
    elif field in ("ascensor", "amueblado", "mascotas"):
        return parse_bool(raw)
    elif field == "disponibilidad":
        return parse_date_es(raw)
    elif field == "barrio_ciudad":
        return raw
    
    return raw


# ------------------------------
# Interfaz Streamlit
# ------------------------------
def init_state():
    """Inicializa el estado de la sesión."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "slots" not in st.session_state:
        st.session_state.slots = {k: None for k in ALL_SLOTS}
    if "descripcion_original" not in st.session_state:
        st.session_state.descripcion_original = ""
    if "extraction_done" not in st.session_state:
        st.session_state.extraction_done = False

APP_CSS = """
<style>
.rm-card {
  border: 1px solid #e6e6e6; border-radius: 12px; padding: 20px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06); background: #fff;
  height: 100%;
}
.rm-chat-container {
  height: 500px;
  overflow-y: auto;
  padding: 16px;
  background: #fafafa;
  border-radius: 8px;
  margin-bottom: 16px;
}
.rm-message {
  margin-bottom: 12px;
  padding: 10px 14px;
  border-radius: 8px;
  max-width: 85%;
}
.rm-message-user {
  background: #e3f2fd;
  margin-left: auto;
  text-align: right;
}
.rm-message-assistant {
  background: #fff;
  border: 1px solid #e0e0e0;
}
.rm-field-label {
  font-size: 13px;
  font-weight: 600;
  color: #444;
  margin-bottom: 4px;
}
.rm-required {
  color: #d32f2f;
  margin-left: 2px;
}
.stTextInput input {
  background: #fafafa !important;
}
</style>
"""

def app():
    """Aplicación principal."""
    st.set_page_config(
        page_title="RentMatch AI — Alta del piso", 
        page_icon="🏠", 
        layout="wide"
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)

    # Header
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.write("# 🏠")
    with col_title:
        st.write("# RentMatch AI — M1")
        st.caption("Alta conversacional del piso · Vista unificada")

    st.divider()

    # Estado inicial
    init_state()
    ensure_csv_schema()

    # Layout principal: 2 columnas
    col_chat, col_ficha = st.columns([1, 1], gap="large")

    # ========== COLUMNA IZQUIERDA: CHAT ==========
    with col_chat:
        st.markdown("<div class='rm-card'>", unsafe_allow_html=True)
        st.subheader("💬 Conversación")
        
        # Contenedor de mensajes con scroll
        chat_container = st.container()
        with chat_container:
            st.markdown("<div class='rm-chat-container'>", unsafe_allow_html=True)
            
            if len(st.session_state.messages) == 0:
                st.markdown(
                    "<div class='rm-message rm-message-assistant'>"
                    "👋 ¡Hola! Describe tu piso en pocas palabras y yo me encargo del resto."
                    "</div>",
                    unsafe_allow_html=True
                )
            
            for msg in st.session_state.messages:
                if msg["role"] == "assistant":
                    st.markdown(
                        f"<div class='rm-message rm-message-assistant'>🤖 {msg['content']}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='rm-message rm-message-user'>👤 {msg['content']}</div>",
                        unsafe_allow_html=True
                    )
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Input de chat
        prompt = st.chat_input("Escribe aquí tu mensaje...")
        
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Primera descripción: extraer con GPT
            if not st.session_state.extraction_done:
                st.session_state.descripcion_original = prompt
                with st.spinner("🔍 Analizando..."):
                    extracted = extract_slots(prompt)
                
                for k in ALL_SLOTS:
                    if extracted.get(k) is not None:
                        st.session_state.slots[k] = extracted[k]
                
                st.session_state.extraction_done = True
                
                questions = make_questions(st.session_state.slots)
                if questions:
                    bot = f"Perfecto, he capturado varios datos. Me falta: {questions[0]}"
                else:
                    bot = "¡Genial! Ya tengo todo lo necesario. Revisa la ficha a la derecha y guarda cuando quieras."
                st.session_state.messages.append({"role": "assistant", "content": bot})
            
            # Respuestas subsiguientes
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
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": "✅ ¡Perfecto! Ya tenemos todo. Puedes guardar el piso."
                        })
                else:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "Anotado. Si necesitas cambiar algo, edítalo en la ficha."
                    })
            
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

    # ========== COLUMNA DERECHA: FICHA ==========
    with col_ficha:
        st.markdown("<div class='rm-card'>", unsafe_allow_html=True)
        st.subheader("📋 Ficha del Piso")
        
        # Progreso
        done = sum(1 for k in REQUIRED_SLOTS if st.session_state.slots.get(k))
        st.progress(done / len(REQUIRED_SLOTS))
        st.caption(f"Campos completados: {done}/{len(REQUIRED_SLOTS)}")
        
        st.write("---")
        
        # Campos obligatorios
        st.markdown("#### Datos obligatorios")
        
        for k in REQUIRED_SLOTS:
            label = k.replace("_", " ").title()
            st.markdown(
                f"<div class='rm-field-label'>{label} <span class='rm-required'>*</span></div>",
                unsafe_allow_html=True
            )
            v = st.session_state.slots.get(k)
            new_val = st.text_input(
                k, 
                value="" if v is None else str(v), 
                key=f"input_{k}",
                label_visibility="collapsed"
            )
            if new_val:
                st.session_state.slots[k] = normalize_field(k, new_val)
        
        st.write("---")
        
        # Campos opcionales
        st.markdown("#### Datos opcionales")
        
        for k in OPTIONAL_SLOTS:
            label = k.replace("_", " ").title()
            st.markdown(f"<div class='rm-field-label'>{label}</div>", unsafe_allow_html=True)
            v = st.session_state.slots.get(k)
            new_val = st.text_input(
                k, 
                value="" if v is None else str(v), 
                key=f"input_{k}",
                label_visibility="collapsed"
            )
            if new_val:
                st.session_state.slots[k] = normalize_field(k, new_val)
        
        st.write("---")
        
        # Resumen
        st.markdown("#### Resumen")
        st.info(make_summary(st.session_state.slots))
        
        # Validaciones
        is_valid, messages = validate_slots(st.session_state.slots)
        
        if messages:
            for msg in messages:
                if msg.startswith("❌"):
                    st.error(msg)
                else:
                    st.warning(msg)
        
        # Botón guardar
        missing = missing_required(st.session_state.slots)
        can_save = is_valid and len(missing) == 0
        
        if st.button("💾 Guardar Piso", type="primary", disabled=not can_save, use_container_width=True):
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
            
            with st.spinner("📤 Guardando en Supabase..."):
                success, message = save_listing(rec)
            
            if success:
                st.success(message)
                st.balloons()
                
                if st.button("➕ Crear otro piso", use_container_width=True):
                    for key in ["messages", "slots", "descripcion_original", "extraction_done"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            else:
                st.error(message)
        
        if not can_save:
            if missing:
                st.caption(f"⚠️ Faltan campos obligatorios: {', '.join(missing)}")
            elif not is_valid:
                st.caption("⚠️ Corrige los errores antes de guardar")
        
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    app()
    
