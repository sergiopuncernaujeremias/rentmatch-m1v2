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

# Webhooks n8n
N8N_WEBHOOK = os.getenv("N8N_WEBHOOK_URL")  # alta del piso (ya existente)
N8N_WEBHOOK_FOTOS = os.getenv("N8N_WEBHOOK_FOTOS")  # flujo de fotos + scoring
N8N_WEBHOOK_ENRIQUECIMIENTO = os.getenv("N8N_WEBHOOK_ENRIQUECIMIENTO")  # flujo Google Maps / servicios

# Ahora ascensor, amueblado y mascotas son obligatorios
REQUIRED_SLOTS = [
    "precio", "barrio_ciudad", "m2", "habitaciones", "banos",
    "disponibilidad", "ascensor", "amueblado", "mascotas"
]
OPTIONAL_SLOTS = [
    "planta", "estado"
]
ALL_SLOTS = REQUIRED_SLOTS + OPTIONAL_SLOTS

# Campos nuevos de preferencias del arrendador sobre el inquilino
PREF_FIELDS = [
    "max_ocupantes",
    "admite_mascotas_inquilino",
    "edad_minima",
    "tipos_inquilino_preferidos",
    "ingreso_minimo",
    "duracion_minima",
    "duracion_maxima",
    "contrato_estable",
    "autonomo_aceptado",
    "freelance_aceptado",
    "fumadores_permitidos",
    "perfil_tranquilo",
    "normas_especiales",
    "prioridades_seleccion",
    "observaciones_perfil_inquilino",
]


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
            # Preferencias arrendador
            *PREF_FIELDS,
            "created_at",
        ]
        pd.DataFrame(columns=cols).to_csv(CSV_FILE, index=False)
    else:
        # Si ya existe, asegurar que todas las columnas nuevas están presentes
        df = pd.read_csv(CSV_FILE)
        cols_existentes = set(df.columns)
        cols_nuevas = [
            "id_piso", "descripcion_original", "descripcion_ia",
            "precio", "barrio_ciudad", "m2", "habitaciones", "banos",
            "planta", "ascensor", "amueblado", "mascotas", "disponibilidad",
            "distancia_metro_m", "score_conectividad",
            "score_visual_global", "fotos_faltantes_sugeridas",
            *PREF_FIELDS,
            "created_at",
        ]
        for c in cols_nuevas:
            if c not in cols_existentes:
                df[c] = None
        df.to_csv(CSV_FILE, index=False)


def save_listing(record):
    # Guarda en CSV local
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


def is_missing_value(field: str, value):
    """
    Considera faltante:
    - None
    - cadena vacía
    Pero NO considera faltante False o 0 (por ejemplo, 'no ascensor' es válido).
    """
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def missing_required(slots: dict) -> list:
    return [k for k in REQUIRED_SLOTS if is_missing_value(k, slots.get(k))]


# Mapeo campo -> pregunta en lenguaje natural
FIELD_QUESTIONS = {
    "precio": "¿Cuál es el precio mensual en euros?",
    "barrio_ciudad": "¿En qué barrio y ciudad está el piso? (Ej.: 'Sant Gervasi, Barcelona')",
    "m2": "¿Cuántos metros cuadrados tiene?",
    "habitaciones": "¿Cuántas habitaciones tiene?",
    "banos": "¿Cuántos baños tiene?",
    "disponibilidad": "¿Desde qué fecha está disponible? (YYYY-MM-DD)",
    "ascensor": "¿Tiene ascensor el edificio? (sí/no)",
    "amueblado": "¿Está amueblado el piso? (sí/no)",
    "mascotas": "¿Se aceptan mascotas? (sí/no)",
}


def question_for_field(field: str) -> str:
    """Devuelve la pregunta adecuada para un campo concreto."""
    return FIELD_QUESTIONS.get(field, f"¿Puedes facilitar el dato para el campo '{field}'?")


def make_questions(slots: dict) -> list[str]:
    """Devuelve la lista de preguntas pendientes según los campos requeridos que falten."""
    qs = []
    for f in REQUIRED_SLOTS:
        if is_missing_value(f, slots.get(f)):
            qs.append(question_for_field(f))
    return qs


def make_summary(slots: dict) -> str:
    asc = slots.get("ascensor")
    asc_txt = "con ascensor" if asc else "sin ascensor"
    amu = slots.get("amueblado")
    amu_txt = "amueblado" if amu else "sin amueblar"
    mas = slots.get("mascotas")
    mas_txt = "se aceptan mascotas" if mas else "no se aceptan mascotas"
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
        mes = m.group(2).replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
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
            prompt = (
                f"Extrae solo el valor atómico para el campo '{field}' a partir del texto entre <t>:</t>. "
                f"Responde sin explicación. Si es precio/m2/enteros: un número. Si es booleano: true/false. "
                f"Si es fecha: YYYY-MM-DD. Texto: <t>{raw}</t>"
            )
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
# Estado e interfaz
# ------------------------------
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "slots" not in st.session_state:
        st.session_state.slots = {k: None for k in ALL_SLOTS}
    if "descripcion_original" not in st.session_state:
        st.session_state.descripcion_original = ""
    if "last_saved_property_id" not in st.session_state:
        st.session_state.last_saved_property_id = None
    if "last_saved_record" not in st.session_state:
        st.session_state.last_saved_record = None
    if "direccion_completa" not in st.session_state:
        st.session_state.direccion_completa = ""
    # campo que estamos preguntando ahora
    if "current_question_field" not in st.session_state:
        st.session_state.current_question_field = None


APP_CSS = """
<style>
/* Ajustar padding superior de la página para reducir espacio en blanco */
.main .block-container {
  padding-top: 0.5rem;
  padding-bottom: 2rem;
}

/* Contenedor tipo “card” */
.rm-card {
  border: 1px solid #e6e6e6; border-radius: 16px; padding: 16px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.04); background: #fff;
}
.rm-chip {
  display:inline-block; padding:4px 10px; border-radius:999px;
  border:1px solid #e6e6e6; margin-right:6px; font-size:12px;
  background:#fafafa;
}
.rm-muted { color:#666; font-size:12px; }

/* Cabecera tipo hero con imagen de fondo */
.rm-hero {
  background-image: url('https://images.unsplash.com/photo-1523217582562-09d0def993a6?auto=format&fit=crop&w=1600&q=80');
  background-size: cover;
  background-position: center;
  border-radius: 20px;
  margin-bottom: 1.5rem;
  min-height: 150px;
  position: relative;
}
.rm-hero-overlay {
  background: rgba(0,0,0,0.35);
  border-radius: 20px;
  padding: 18px 28px;
  display: flex;
  align-items: center;
  gap: 16px;
  height: 100%;
}
.rm-hero-icon {
  font-size: 40px;
}
.rm-hero-text h1 {
  margin: 0;
  color: #ffffff;
  font-size: 32px;
}
.rm-hero-text p {
  margin: 4px 0 0 0;
  color: #f3f3f3;
  font-size: 14px;
}
</style>
"""


def app():
    st.set_page_config(page_title="RentMatch AI — Alta del piso", page_icon="🏠", layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)

    # Estado inicial + CSV
    init_state()
    ensure_csv_schema()

    # Cabecera con imagen de fondo
    st.markdown(
        """
        <div class="rm-hero">
          <div class="rm-hero-overlay">
            <div class="rm-hero-icon">🏠</div>
            <div class="rm-hero-text">
              <h1>RentMatch AI — M1</h1>
              <p>Alta conversacional del piso · Demo Cloud</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar con estado y acciones rápidas
    with st.sidebar:
        st.write("### Progreso")
        done = sum(1 for k in REQUIRED_SLOTS if not is_missing_value(k, st.session_state.slots.get(k)))
        st.progress(done / len(REQUIRED_SLOTS))
        st.write(f"Completados: **{done}/{len(REQUIRED_SLOTS)}**")

        st.write("### Pasos")
        st.markdown(
            "- 1) Describe y completa los datos del piso\n"
            "- 2) Define tus preferencias sobre inquilinos\n"
            "- 3) Revisa y **Guarda**"
        )
        st.write("### Enlaces")
        st.markdown(
            "- Supabase (si conectado)\n"
            "- n8n Webhook (si configurado)",
        )
        st.divider()
        st.write("### Info")
        st.markdown(
            "<span class='rm-muted'>Los datos se normalizan localmente y se envían a n8n al guardar.</span>",
            unsafe_allow_html=True
        )

    # Pestañas: Paso 1 (piso), Paso 2 (preferencias) y datos
    tab_piso, tab_prefs, tab_datos = st.tabs(
        ["🏠 Paso 1 · Piso", "👥 Paso 2 · Inquilino", "📊 Datos & Hooks"]
    )

    # ===== TAB PASO 1: Conversación (izq) + Ficha (dcha) =====
    with tab_piso:
        col_chat, col_ficha = st.columns([2, 1])

        # -------- Conversación --------
        with col_chat:
            st.markdown("<div class='rm-card'>", unsafe_allow_html=True)
            st.subheader("Describe tu piso")
            st.caption("Escribe libremente. Solo te preguntaremos lo imprescindible.")

            # Histórico de chat
            for msg in st.session_state.messages:
                role_icon = "🤖" if msg["role"] == "assistant" else "👤"
                st.markdown(f"**{role_icon}** {msg['content']}", unsafe_allow_html=True)

            # Primer paso: descripción inicial en un área grande
            if not st.session_state.descripcion_original:
                desc_text = st.text_area(
                    "Escribe aquí la descripción inicial del piso",
                    key="descripcion_inicial_input",
                    height=220,
                    placeholder="Ej.: Piso luminoso de 80m2 en el Eixample, 3 habitaciones..."
                )
                if st.button("Enviar descripción inicial"):
                    if desc_text and desc_text.strip():
                        # Guardar mensaje de usuario
                        st.session_state.messages.append(
                            {"role": "user", "content": desc_text}
                        )
                        st.session_state.descripcion_original = desc_text

                        # Extraer slots con LLM
                        with st.spinner("Analizando descripción…"):
                            extracted = extract_slots(desc_text)
                        for k in ALL_SLOTS:
                            if extracted.get(k) is not None:
                                st.session_state.slots[k] = extracted[k]

                        # Preguntar SOLO el primer campo que falte
                        missing = missing_required(st.session_state.slots)
                        if missing:
                            first_field = missing[0]
                            st.session_state.current_question_field = first_field
                            first_q = question_for_field(first_field)
                            bot = (
                                "Gracias, he leído la descripción. "
                                "Para completar la ficha te iré haciendo algunas preguntas, una a una.\n\n"
                                f"{first_q}"
                            )
                        else:
                            st.session_state.current_question_field = None
                            bot = (
                                "Perfecto, ya tengo lo necesario. "
                                "Revisa la ficha a la derecha, luego pasa al Paso 2 (Inquilino) y guarda el piso."
                            )

                        st.session_state.messages.append(
                            {"role": "assistant", "content": bot}
                        )
            else:
                # Mensajes posteriores: ir rellenando campos que falten, pregunta a pregunta
                prompt = st.chat_input("Responde a las preguntas o añade comentarios…")
                if prompt:
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    current_field = st.session_state.current_question_field

                    if current_field:
                        value = normalize_field(current_field, prompt)
                        st.session_state.slots[current_field] = value

                        # Recalcular campos que faltan
                        missing = missing_required(st.session_state.slots)

                        if current_field in missing:
                            # No se ha podido rellenar bien -> repetir misma pregunta
                            q = question_for_field(current_field)
                            st.session_state.messages.append({"role": "assistant", "content": q})
                        else:
                            # Avanzar al siguiente campo que falte
                            if missing:
                                next_field = missing[0]
                                st.session_state.current_question_field = next_field
                                q = question_for_field(next_field)
                                st.session_state.messages.append({"role": "assistant", "content": q})
                            else:
                                # Ya no faltan campos obligatorios del piso
                                st.session_state.current_question_field = None
                                bot = (
                                    "¡Listo! Ya tengo todos los datos básicos del piso.<br><br>"
                                    "<span style='color:#1f6feb; font-weight:bold;'>1) Revisa la ficha del anuncio a la derecha</span><br>"
                                    "<span style='color:#1f6feb; font-weight:bold;'>2) Ve a la pestaña «👥 Paso 2 · Inquilino»</span><br>"
                                    "<span style='color:#1f6feb; font-weight:bold;'>3) Completa tus preferencias y pulsa «Guardar piso»</span>"
                                )
                                st.session_state.messages.append({"role": "assistant", "content": bot})
                    else:
                        # No hay pregunta activa: tratamos el mensaje como comentario
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": "He anotado tu comentario. Ajusta en la ficha si lo necesitas.",
                            }
                        )

            st.markdown("</div>", unsafe_allow_html=True)

        # -------- Ficha del piso --------
        with col_ficha:
            st.markdown("<div class='rm-card'>", unsafe_allow_html=True)
            st.subheader("Ficha del anuncio")

            # Campos en dos columnas
            c1, c2 = st.columns(2)
            with c1:
                for k in ["precio", "barrio_ciudad", "m2", "habitaciones", "banos", "disponibilidad"]:
                    v = st.session_state.slots.get(k)
                    st.session_state.slots[k] = st.text_input(
                        k, value="" if v is None else str(v)
                    )
            with c2:
                for k in ["planta", "ascensor", "amueblado", "mascotas", "estado"]:
                    v = st.session_state.slots.get(k)
                    st.session_state.slots[k] = st.text_input(
                        k, value="" if v is None else str(v)
                    )

            # Dirección exacta para enriquecimiento (opcional)
            st.session_state.direccion_completa = st.text_input(
                "Dirección completa (para enriquecimiento M2, opcional)",
                value=st.session_state.direccion_completa
            )

            st.write("---")
            st.write("**Resumen generado por la IA:**")
            st.info(make_summary(st.session_state.slots))

            problems = validate_slots(st.session_state.slots)
            if problems:
                st.warning(" ; ".join(problems))

            st.caption(
                "Cuando tengas esta ficha lista, pasa al **Paso 2 · Inquilino** para definir preferencias y guardar el piso."
            )

            st.markdown("</div>", unsafe_allow_html=True)

    # ===== TAB PASO 2: Preferencias sobre el inquilino + Guardar =====
    with tab_prefs:
        st.markdown("<div class='rm-card'>", unsafe_allow_html=True)
        st.subheader("Preferencias sobre el inquilino")

        st.caption(
            "Paso 2 de 2 · Estas preferencias se usarán después para priorizar candidatos cuando haya varias solicitudes."
        )

        # Pequeño resumen del piso arriba, para contexto
        st.write("**Resumen del piso:**")
        st.info(make_summary(st.session_state.slots))

        st.write("---")

        # Widgets de preferencias
        max_ocupantes = st.number_input(
            "Número máximo de ocupantes",
            min_value=1,
            max_value=20,
            step=1,
            key="max_ocupantes"
        )

        admite_mascotas_inquilino_str = st.selectbox(
            "¿Aceptarías inquilinos con mascotas?",
            ["No", "Sí"],
            key="admite_mascotas_inquilino_str"
        )

        edad_minima = st.number_input(
            "Edad mínima de los inquilinos (opcional)",
            min_value=0,
            max_value=120,
            step=1,
            key="edad_minima"
        )

        tipos_inquilino_preferidos_list = st.multiselect(
            "Tipo de inquilino preferido",
            ["Individuo", "Pareja", "Familia", "Estudiantes", "Compartir piso"],
            key="tipos_inquilino_preferidos_list"
        )

        ingreso_minimo = st.number_input(
            "Ingreso mínimo mensual total requerido (opcional, €)",
            min_value=0,
            step=100,
            key="ingreso_minimo"
        )

        duracion_minima = st.number_input(
            "Duración mínima deseada del alquiler (meses)",
            min_value=1,
            max_value=120,
            step=1,
            key="duracion_minima"
        )

        duracion_maxima = st.number_input(
            "Duración máxima aceptada (opcional, meses)",
            min_value=0,
            max_value=240,
            step=1,
            key="duracion_maxima"
        )

        contrato_estable_str = st.selectbox(
            "¿Quieres contrato laboral estable (indefinido/fijo) como preferencia?",
            ["Sí", "No"],
            key="contrato_estable_str"
        )

        autonomo_aceptado_str = st.selectbox(
            "¿Aceptas autónomos?",
            ["Sin preferencia", "Sí", "No"],
            key="autonomo_aceptado_str"
        )

        freelance_aceptado_str = st.selectbox(
            "¿Aceptas freelance / ingresos variables?",
            ["Sin preferencia", "Sí", "No"],
            key="freelance_aceptado_str"
        )

        fumadores_permitidos_str = st.selectbox(
            "¿Permites fumadores dentro de la vivienda?",
            ["Sí", "No"],
            key="fumadores_permitidos_str"
        )

        perfil_tranquilo_str = st.selectbox(
            "¿Prefieres explícitamente un perfil tranquilo?",
            ["Sin preferencia", "Sí", "No"],
            key="perfil_tranquilo_str"
        )

        normas_especiales = st.text_area(
            "Normas especiales de la vivienda / comunidad (opcional)",
            key="normas_especiales"
        )

        prioridades_seleccion = st.selectbox(
            "¿Qué pesa más para ti al elegir un inquilino?",
            [
                "Solvencia",
                "Estabilidad laboral",
                "Duración del contrato",
                "Ausencia de mascotas",
                "Perfil familiar",
                "Rapidez de disponibilidad",
            ],
            key="prioridades_seleccion"
        )

        observaciones_perfil_inquilino = st.text_area(
            "Observaciones sobre el tipo de inquilino deseado (opcional)",
            key="observaciones_perfil_inquilino"
        )

        st.write("---")
        cta_col1, cta_col2 = st.columns([1, 2])
        with cta_col1:
            save_click = st.button("💾 Guardar piso", type="primary")
        with cta_col2:
            st.caption(
                "Se almacenará en CSV (efímero) y se enviará a n8n / Supabase si están configurados."
            )

        if save_click:
            # Validación de campos obligatorios de preferencias
            pref_errors = []
            if not tipos_inquilino_preferidos_list:
                pref_errors.append("Debes seleccionar al menos un tipo de inquilino preferido.")
            if max_ocupantes < 1:
                pref_errors.append("Indica el número máximo de ocupantes.")
            if duracion_minima < 1:
                pref_errors.append("Indica la duración mínima deseada del alquiler (meses).")

            if pref_errors:
                st.error("Faltan datos en preferencias del inquilino: " + " | ".join(pref_errors))
            else:
                missing = missing_required(st.session_state.slots)
                if missing:
                    st.error("Faltan campos obligatorios del piso: " + ", ".join(missing))
                else:
                    # Generar ID del piso
                    id_piso = str(uuid.uuid4())

                    # Campos base del piso (desde slots)
                    precio = st.session_state.slots.get("precio")
                    barrio_ciudad = st.session_state.slots.get("barrio_ciudad")
                    m2 = st.session_state.slots.get("m2")
                    habitaciones = st.session_state.slots.get("habitaciones")
                    banos = st.session_state.slots.get("banos")
                    planta = st.session_state.slots.get("planta")
                    ascensor = st.session_state.slots.get("ascensor")
                    amueblado = st.session_state.slots.get("amueblado")
                    mascotas = st.session_state.slots.get("mascotas")
                    disponibilidad = st.session_state.slots.get("disponibilidad")
                    estado = st.session_state.slots.get("estado")

                    # Campo mascotas obligatorio en BD: si falta, guardarlo como cadena vacía
                    if mascotas is None:
                        mascotas = ""

                    # Construir registro COMPLETO (piso + preferencias)
                    rec = {
                        "id_piso": id_piso,
                        "descripcion_original": st.session_state.descripcion_original,
                        "descripcion_ia": make_summary(st.session_state.slots),

                        # Datos del piso
                        "precio": precio,
                        "barrio_ciudad": barrio_ciudad,
                        "m2": m2,
                        "habitaciones": habitaciones,
                        "banos": banos,
                        "planta": planta,
                        "ascensor": ascensor,
                        "amueblado": amueblado,
                        "mascotas": mascotas,
                        "disponibilidad": disponibilidad,
                        "estado": estado,

                        # Campos M2/M3
                        "distancia_metro_m": None,
                        "score_conectividad": None,
                        "score_visual_global": None,
                        "fotos_faltantes_sugeridas": None,

                        # Preferencias del arrendador
                        "max_ocupantes": int(max_ocupantes),
                        "admite_mascotas_inquilino": True if admite_mascotas_inquilino_str == "Sí" else False,
                        "edad_minima": int(edad_minima) if edad_minima > 0 else None,
                        "tipos_inquilino_preferidos": ",".join(tipos_inquilino_preferidos_list) if tipos_inquilino_preferidos_list else None,
                        "ingreso_minimo": float(ingreso_minimo) if ingreso_minimo > 0 else None,
                        "duracion_minima": int(duracion_minima),
                        "duracion_maxima": int(duracion_maxima) if duracion_maxima > 0 else None,
                        "contrato_estable": True if contrato_estable_str == "Sí" else False,
                        "autonomo_aceptado": None if autonomo_aceptado_str == "Sin preferencia" else (autonomo_aceptado_str == "Sí"),
                        "freelance_aceptado": None if freelance_aceptado_str == "Sin preferencia" else (freelance_aceptado_str == "Sí"),
                        "fumadores_permitidos": True if fumadores_permitidos_str == "Sí" else False,
                        "perfil_tranquilo": None if perfil_tranquilo_str == "Sin preferencia" else (perfil_tranquilo_str == "Sí"),
                        "normas_especiales": normas_especiales or None,
                        "prioridades_seleccion": prioridades_seleccion,
                        "observaciones_perfil_inquilino": observaciones_perfil_inquilino or None,

                        "created_at": datetime.utcnow().isoformat(),
                    }

                    # DEBUG: ver en pantalla qué se está enviando
                    st.write("Payload enviado a n8n / Supabase:")
                    st.json(rec)

                    save_listing(rec)

                    # Guardar en estado para botones posteriores
                    st.session_state.last_saved_property_id = id_piso
                    st.session_state.last_saved_record = rec

                    st.success("✅ Guardado correcto. Enviado a n8n/Supabase si procede.")

        st.write("---")
        st.subheader("Acciones sobre este piso")

        id_piso_guardado = st.session_state.last_saved_property_id
        rec_guardado = st.session_state.last_saved_record

        if not id_piso_guardado or not rec_guardado:
            st.info("Guarda primero el piso para habilitar las acciones adicionales.")
        else:
            # Botón 1: subir fotos y scoring (M3 real vía n8n)
            if st.button("📸 Subir fotos y puntuar calidad"):
                if not N8N_WEBHOOK_FOTOS:
                    st.error("No está configurado N8N_WEBHOOK_FOTOS en el entorno.")
                else:
                    try:
                        payload_fotos = {
                            "id_piso": id_piso_guardado,
                            "barrio_ciudad": rec_guardado.get("barrio_ciudad"),
                            "precio": rec_guardado.get("precio"),
                            "m2": rec_guardado.get("m2"),
                        }
                        resp_fotos = requests.post(N8N_WEBHOOK_FOTOS, json=payload_fotos, timeout=10)
                        resp_fotos.raise_for_status()
                        st.success("He lanzado el flujo de subida de fotos y scoring en n8n.")
                    except Exception as e:
                        st.error(f"Error lanzando el flujo de fotos: {e}")

            # Botón 2: enriquecimiento por Google Maps / servicios (M2 real vía n8n)
            if st.button("📍 Enriquecer datos de la zona (Google Maps)"):
                direccion = st.session_state.direccion_completa or rec_guardado.get("barrio_ciudad")
                if not direccion:
                    st.warning("Necesito una dirección o al menos barrio/ciudad para enriquecer la zona.")
                elif not N8N_WEBHOOK_ENRIQUECIMIENTO:
                    st.error("No está configurado N8N_WEBHOOK_ENRIQUECIMIENTO en el entorno.")
                else:
                    try:
                        payload_enriq = {
                            "id_piso": id_piso_guardado,
                            "direccion": direccion,
                            "barrio_ciudad": rec_guardado.get("barrio_ciudad"),
                            "precio": rec_guardado.get("precio"),
                            "m2": rec_guardado.get("m2"),
                        }
                        resp_enriq = requests.post(
                            N8N_WEBHOOK_ENRIQUECIMIENTO,
                            json=payload_enriq,
                            timeout=10
                        )
                        resp_enriq.raise_for_status()
                        st.success("He lanzado el flujo de enriquecimiento de datos de entorno en n8n.")
                    except Exception as e:
                        st.error(f"Error lanzando el flujo de enriquecimiento: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ===== TAB: Datos & Hooks (simulados) =====
    with tab_datos:
        st.markdown("<div class='rm-card'>", unsafe_allow_html=True)
        st.subheader("Datos recientes y acciones")

        # Mostrar últimas filas del CSV (si existe)
        try:
            df = pd.read_csv(CSV_FILE)
            st.dataframe(df.tail(5), width="stretch")
        except Exception:
            st.caption("No hay datos aún.")

        st.write("---")
        st.write("**Hooks de demostración (simulados en local)**")
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
