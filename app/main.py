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


def make_questions(slots: dict) -> list:
    q = []
    if is_missing_value("precio", slots.get("precio")):
        q.append("¿Cuál es el precio mensual en euros?")
    if is_missing_value("barrio_ciudad", slots.get("barrio_ciudad")):
        q.append("¿En qué barrio y ciudad está el piso? (Ej.: 'Sant Gervasi, Barcelona')")
    if is_missing_value("m2", slots.get("m2")):
        q.append("¿Cuántos metros cuadrados tiene?")
    if is_missing_value("habitaciones", slots.get("habitaciones")):
        q.append("¿Cuántas habitaciones tiene?")
    if is_missing_value("banos", slots.get("banos")):
        q.append("¿Cuántos baños tiene?")
    if is_missing_value("disponibilidad", slots.get("disponibilidad")):
        q.append("¿Desde qué fecha está disponible? (YYYY-MM-DD)")
    if is_missing_value("ascensor", slots.get("ascensor")):
        q.append("¿Tiene ascensor el edificio? (sí/no)")
    if is_missing_value("amueblado", slots.get("amueblado")):
        q.append("¿Está amueblado el piso? (sí/no)")
    if is_missing_value("mascotas", slots.get("mascotas")):
        q.append("¿Se aceptan mascotas? (sí/no)")
    return q


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
  margin: 4px 0
