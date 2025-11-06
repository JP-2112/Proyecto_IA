import os
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
    # Español + Inglés
    return easyocr.Reader(['es', 'en'], gpu=False)

def do_ocr(image: Image.Image):
    reader = get_easyocr_reader()
    import numpy as np
    arr = np.array(image.convert("RGB"))
    results = reader.readtext(arr, detail=0)
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

# ====== Hugging Face helpers con FALLBACK LOCAL (transformers) ======
def _local_summarize(text: str):
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        out = summarizer(text, max_length=130, min_length=30, do_sample=False)
        return out[0]["summary_text"]
    except Exception as e:
        return f"⚠️ Fallback summarization error: {e}"

def _local_translate_es_en(text: str):
    try:
        from transformers import pipeline
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
        out = translator(text, max_length=300)
        return out[0]["translation_text"]
    except Exception as e:
        return f"⚠️ Fallback translation error: {e}"

def _local_generate(prompt: str, max_tokens=256):
    try:
        from transformers import pipeline
        gen = pipeline("text2text-generation", model="google/flan-t5-small")
        out = gen(prompt, max_length=min(512, int(max_tokens) + 64))
        return out[0]["generated_text"]
    except Exception as e:
        return f"⚠️ Fallback generation error: {e}"

def hf_summarization(text: str, model_id="facebook/bart-large-cnn"):
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
        # SDK antiguo → ir a local
        return _local_summarize(text)
    except Exception as e:
        # 403/“no attribute”/“not supported” → local
        if any(x in str(e) for x in ["403", "Forbidden", "no attribute", "not supported"]):
            return _local_summarize(text)
        return f"⚠️ Error calling Hugging Face Inference API (summarization): {e}"

def hf_translate_es_en(text: str, model_id="Helsinki-NLP/opus-mt-es-en"):
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
        return _local_translate_es_en(text)
    except Exception as e:
        if any(x in str(e) for x in ["403", "Forbidden", "no attribute", "not supported"]):
            return _local_translate_es_en(text)
        return f"⚠️ Error calling Hugging Face Inference API (translation): {e}"

def hf_text_generation(prompt: str, model_id="Qwen/Qwen2.5-0.5B-Instruct",
                       temperature=0.3, max_tokens=256):
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
        # SDK antiguo → local
        return _local_generate(prompt, max_tokens=max_tokens)
    except Exception as e:
        if any(x in str(e) for x in ["not supported for task", "403", "Forbidden", "no attribute"]):
            return _local_generate(prompt, max_tokens=max_tokens)
        return f"⚠️ Error calling Hugging Face Inference API (text-generation): {e}"

# ====== Streamlit UI ======
st.set_page_config(page_title="Taller IA: OCR + LLM", page_icon="🧠", layout="wide")
st.title("🧠 Taller IA: OCR + LLM (Streamlit)")
st.caption("App demo: OCR de imágenes + análisis con LLMs (GROQ / Hugging Face con fallback local)")

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
                "mistralai/Mistral-7B-Instruct-v0.3",  # visible (evitamos si da 'conversational')
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

def pick_hf_model(task_name: str, chosen: str) -> str:
    if task_name == "Resumir en 3 puntos clave":
        return "facebook/bart-large-cnn"
    if task_name == "Traducir al inglés":
        return "Helsinki-NLP/opus-mt-es-en"
    # Entidades / Explicación -> generativo (evitar bart/mistral si chocan)
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
                    st.error("Falta GROQ_API_KEY en tu .env/Secrets")
                else:
                    prompt = build_prompt(task, text)
                    out = groq_chat_completion(prompt, model_name=groq_model, temperature=temperature, max_tokens=max_tokens)
                    st.markdown("**Salida (GROQ):**")
                    st.markdown(out)
            else:
                if not HUGGINGFACE_API_KEY:
                    st.warning("Falta HUGGINGFACE_API_KEY en tus Secrets/.env. Usaremos el fallback local si es posible.")
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
