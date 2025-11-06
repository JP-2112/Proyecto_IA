import os
import io
import time
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# ====== Load environment variables ======
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# Lazy imports for heavy libs to speed up cold start
@st.cache_resource
def get_easyocr_reader():
    import easyocr
    # Spanish + English by default; adjust as needed
    return easyocr.Reader(['es', 'en'], gpu=False)

def do_ocr(image: Image.Image):
    reader = get_easyocr_reader()
    # easyocr expects ndarray (RGB)
    arr = np.array(image.convert("RGB"))
    results = reader.readtext(arr, detail=0)  # list of strings
    return "\n".join(results).strip()

def groq_chat_completion(prompt, model_name, temperature, max_tokens):
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that follows the user's instructions precisely."},
                {"role": "user", "content": prompt},
            ],
            model=model_name,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error calling GROQ API: {e}"

# ===== Hugging Face helpers (robustos por tarea) =====
def hf_post(task: str, inputs: str, model_id: str, extra: dict | None = None):
    """Wrapper estable usando client.post para cualquier task soportada."""
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=HUGGINGFACE_API_KEY)
        payload = {"inputs": inputs}
        if extra:
            payload.update(extra)
        out = client.post(json=payload, task=task, model=model_id)
        # Normalizar salida (puede venir como str, dict o lista)
        if isinstance(out, str):
            return out
        if isinstance(out, list) and out and isinstance(out[0], dict):
            # típicamente [{'summary_text': '...'}] o [{'translation_text': '...'}]
            return out[0].get("summary_text") or out[0].get("translation_text") or str(out[0])
        if isinstance(out, dict):
            return out.get("generated_text") or out.get("summary_text") or out.get("translation_text") or str(out)
        return str(out)
    except Exception as e:
        return f"⚠️ Error calling Hugging Face Inference API: {e}"

def hf_summarization(text: str, model_id="facebook/bart-large-cnn"):
    # Evito kwargs incompatibles entre versiones (max_length/max_new_tokens).
    return hf_post(task="summarization", inputs=text, model_id=model_id)

def hf_translate_es_en(text: str, model_id="Helsinki-NLP/opus-mt-es-en"):
    # Traducción dedicada (mucho más estable que pedirle a un LLM generativo).
    return hf_post(task="translation", inputs=text, model_id=model_id)

def hf_text_generation(prompt: str, model_id="Qwen/Qwen2.5-0.5B-Instruct", temperature=0.3, max_tokens=256):
    # Intento con text-generation. Si el modelo no soporta ese task, hago fallback a un T5 (text2text).
    out = hf_post(
        task="text-generation",
        inputs=prompt,
        model_id=model_id,
        extra={"parameters": {"max_new_tokens": int(max_tokens), "temperature": float(temperature)}},
    )
    # Si el backend se queja de "conversational" u otro task, pruebo text2text con FLAN-T5.
    if "not supported for task" in str(out).lower():
        out = hf_post(
            task="text2text-generation",
            inputs=prompt,
            model_id="google/flan-t5-base",
            extra={"parameters": {"max_new_tokens": int(max_tokens)}},
        )
    return out

# ====== Streamlit UI ======
st.set_page_config(page_title="Taller IA: OCR + LLM", page_icon="🧠", layout="wide")
st.title("🧠 Taller IA: OCR + LLM (Streamlit)")
st.caption("App demo: OCR de imágenes + análisis con LLMs (GROQ / Hugging Face)")

with st.sidebar:
    st.header("⚙️ Configuración")
    provider = st.radio("Proveedor NLP", ["GROQ", "Hugging Face"], index=0)
    temperature = st.slider("Temperature (creatividad)", 0.0, 1.5, 0.3, 0.1)
    max_tokens = st.slider("max_tokens (longitud de respuesta)", 32, 2048, 512, 32)

    if provider == "GROQ":
        groq_model = st.selectbox(
            "Modelo GROQ",
            ["llama-3.1-8b-instant", "llama3-8b-8192", "mixtral-8x7b-32768"],
            index=0
        )
    else:
        hf_model = st.selectbox(
            "Modelo Hugging Face (task depende abajo)",
            [
                "facebook/bart-large-cnn",        # summarization
                "Qwen/Qwen2.5-0.5B-Instruct",     # text-generation
                "Helsinki-NLP/opus-mt-es-en",     # translation
            ],
            index=0
        )

# Persist state for extracted text
if "extracted_text" not in st.session_state:
    st.session_state["extracted_text"] = ""

st.subheader("📷 Módulo 1: Lector de Imágenes (OCR)")

uploaded = st.file_uploader("Sube una imagen (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns([1,1])
with col1:
    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Imagen cargada", use_column_width=True)
        if st.button("🔎 Extraer texto (OCR)"):
            with st.spinner("Ejecutando OCR..."):
                st.session_state["extracted_text"] = do_ocr(image)
            st.success("¡Texto extraído! Revisa el panel derecho.")
with col2:
    st.text_area("📝 Texto extraído (editable)", key="extracted_text", height=300)

st.divider()
st.subheader("🧩 Módulo 2: Cerebro Lingüístico (LLM)")

task = st.selectbox(
    "Selecciona la tarea sobre el texto",
    ["Resumir en 3 puntos clave", "Identificar entidades principales", "Traducir al inglés", "Explicación general"]
)

def build_prompt(task_name: str, text: str):
    if task_name == "Resumir en 3 puntos clave":
        return f"Resume el siguiente texto en exactamente 3 viñetas claras y concisas:\n\nTexto:\n{text}"
    elif task_name == "Identificar entidades principales":
        return (
            "Extrae ENTIDADES (Personas, Organizaciones, Lugares, Fechas) y muéstralas en lista con categoría y valor.\n"
            f"Texto:\n{text}"
        )
    elif task_name == "Traducir al inglés":
        return f"Translate the following Spanish text into clear, natural English:\n\n{text}"
    else:
        return f"Deliver a clear, structured explanation of the following text focusing on the key ideas:\n\n{text}"

# Auto-selección de modelo HF válido por tarea
def pick_hf_model(task_name: str, chosen: str) -> str:
    if task_name == "Resumir en 3 puntos clave":
        return "facebook/bart-large-cnn"
    if task_name == "Traducir al inglés":
        return "Helsinki-NLP/opus-mt-es-en"
    # Entidades / Explicación -> generativo
    # Si el usuario dejó BART (incompatible), forzamos Qwen
    if "bart" in (chosen or "").lower():
        return "Qwen/Qwen2.5-0.5B-Instruct"
    return chosen or "Qwen/Qwen2.5-0.5B-Instruct"

analyze = st.button("🤖 Analizar Texto")

if analyze:
    if not st.session_state["extracted_text"]:
        st.warning("Primero extrae o pega texto en el recuadro de la derecha.")
    else:
        text = st.session_state["extracted_text"]
        with st.spinner(f"Analizando con {provider}..."):
            if provider == "GROQ":
                if not GROQ_API_KEY:
                    st.error("Falta GROQ_API_KEY en tu .env")
                else:
                    prompt = build_prompt(task, text)
                    out = groq_chat_completion(prompt, model_name=groq_model, temperature=temperature, max_tokens=max_tokens)
                    st.markdown("**Salida (GROQ):**")
                    st.markdown(out)
            else:
                if not HUGGINGFACE_API_KEY:
                    st.error("Falta HUGGINGFACE_API_KEY en tu .env")
                else:
                    hf_used = pick_hf_model(task, hf_model)

                    if task == "Resumir en 3 puntos clave":
                        out = hf_summarization(text, model_id=hf_used)
                    elif task == "Traducir al inglés":
                        out = hf_translate_es_en(text, model_id=hf_used)
                    else:
                        prompt = build_prompt(task, text)
                        out = hf_text_generation(
                            prompt,
                            model_id=hf_used,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )

                    st.markdown("**Salida (Hugging Face):**")
                    st.markdown(out)

st.divider()
st.subheader("🧪 Módulo 3: Flexibilidad y Experimentación")
st.markdown(
    "- Cambia **Proveedor NLP** en la barra lateral (GROQ vs Hugging Face)\n"
    "- Ajusta **temperature** y **max_tokens** y observa cómo cambia la salida\n"
    "- Prueba con imágenes de distinta calidad para ver el impacto del OCR"
)

st.info("Consejo: si el modelo OCR tarda en cargar, es normal la primera vez. Gracias al caché, las siguientes ejecuciones serán más rápidas.")
