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

# ====== OCR (EasyOCR cacheado) ======
@st.cache_resource
def get_easyocr_reader():
    import easyocr
    # Español + Inglés por defecto
    return easyocr.Reader(['es', 'en'], gpu=False)

def do_ocr(image: Image.Image):
    reader = get_easyocr_reader()
    arr = np.array(image.convert("RGB"))  # easyocr espera ndarray RGB
    results = reader.readtext(arr, detail=0)  # lista de strings
    return "\n".join(results).strip()

# ====== GROQ helper ======
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

# ====== Hugging Face helpers (compatibles con versiones viejas del hub) ======
def hf_summarization(text: str, model_id="facebook/bart-large-cnn"):
    """
    Resumen robusto usando InferenceClient:
    - Usa client.summarization si existe.
    - Si no existe (versiones viejas), hace fallback a generación con prompt de resumen.
    """
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=HUGGINGFACE_API_KEY)
        if hasattr(client, "summarization"):
            out = client.summarization(text=text, model=model_id)
            if isinstance(out, dict) and "summary_text" in out:
                return out["summary_text"]
            if isinstance(out, list) and out and isinstance(out[0], dict):
                return out[0].get("summary_text", str(out[0]))
            return str(out)
        else:
            # Fallback: resumen vía generación
            return hf_text_generation(
                f"Resume el siguiente texto en exactamente 3 viñetas claras y concisas:\n\n{text}",
                model_id="Qwen/Qwen2.5-0.5B-Instruct",
                temperature=0.2,
                max_tokens=256
            )
    except Exception as e:
        return f"⚠️ Error calling Hugging Face Inference API (summarization): {e}"

def hf_translate_es_en(text: str, model_id="Helsinki-NLP/opus-mt-es-en"):
    """
    Traducción estable:
    - Usa client.translation si está disponible.
    - Si no, hace fallback a un modelo generativo con prompt de traducción.
    """
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=HUGGINGFACE_API_KEY)
        if hasattr(client, "translation"):
            out = client.translation(text=text, model=model_id)
            if isinstance(out, list) and out and isinstance(out[0], dict):
                return out[0].get("translation_text", str(out[0]))
            if isinstance(out, dict) and "translation_text" in out:
                return out["translation_text"]
            return str(out)
        else:
            # Fallback: traducción con LLM generativo
            prompt = f"Translate the following Spanish text into clear, natural English:\n\n{text}"
            return hf_text_generation(
                prompt,
                model_id="Qwen/Qwen2.5-0.5B-Instruct",
                temperature=0.2,
                max_tokens=256
            )
    except Exception as e:
        return f"⚠️ Error calling Hugging Face Inference API (translation): {e}"

def hf_text_generation(prompt: str, model_id="Qwen/Qwen2.5-0.5B-Instruct",
                       temperature=0.3, max_tokens=256):
    """
    Generación de texto:
    - Usa client.text_generation si existe.
    - Si el modelo/cliente no soporta ese método, intenta con FLAN-T5 vía el mismo método
      (versiones viejas lo exponen como text_generation también).
    """
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=model_id, token=HUGGINGFACE_API_KEY)
        if hasattr(client, "text_generation"):
            return client.text_generation(
                prompt,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                do_sample=True,
                stream=False,
            )
        else:
            # Fallback a FLAN-T5 para text2text
            if "flan-t5" not in (model_id or "").lower():
                client = InferenceClient(model="google/flan-t5-base", token=HUGGINGFACE_API_KEY)
            return client.text_generation(
                prompt,
                max_new_tokens=int(max_tokens),
                do_sample=False,
                stream=False,
            )
    except Exception as e:
        return f"⚠️ Error calling Hugging Face Inference API (text-generation): {e}"

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
        # Amplié la lista para cubrir cada tarea con un modelo compatible
        hf_model = st.selectbox(
            "Modelo Hugging Face (task depende abajo)",
            [
                "facebook/bart-large-cnn",        # summarization
                "Qwen/Qwen2.5-0.5B-Instruct",     # text-generation (instruct)
                "Helsinki-NLP/opus-mt-es-en",     # translation (es->en)
                "mistralai/Mistral-7B-Instruct-v0.3",  # visible, pero evitamos si no soporta text_generation
            ],
            index=0
        )

# Persistencia del texto OCR
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

# Prompts base por tarea
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

# Elegir modelo HF compatible según la tarea (con fallback seguro)
def pick_hf_model(task_name: str, chosen: str) -> str:
    if task_name == "Resumir en 3 puntos clave":
        return "facebook/bart-large-cnn"
    if task_name == "Traducir al inglés":
        return "Helsinki-NLP/opus-mt-es-en"
    # Entidades / Explicación -> generativo. Evitar bart/mistral si dan incompatibilidad.
    bad = (chosen or "").lower()
    if "bart" in bad or "mistral" in bad:
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
                    out = groq_chat_completion(
                        prompt,
                        model_name=groq_model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
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
