# Taller Final — App OCR + LLM (Streamlit)

Aplicación web **end-to-end** que:
1) Sube una imagen,
2) Extrae texto con **EasyOCR**,
3) Analiza el texto con un **LLM** vía **GROQ** o **Hugging Face**.

## 🚀 Requisitos
- Python 3.10+
- Claves de API:
  - [GROQ](https://groq.com/) → `GROQ_API_KEY`
  - [Hugging Face](https://huggingface.co/) → `HUGGINGFACE_API_KEY`

## 📦 Instalación (una sola vez)
```bash
# 1) Crear entorno (opcional pero recomendado)
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

# 2) Instalar dependencias
pip install -r requirements.txt
```

> **Nota:** La primera ejecución de EasyOCR descarga modelos (~80–100MB). Paciencia 🙂

## 🔐 Claves de API
Copia el archivo `.env.example` a `.env` y pega tus claves:
```env
GROQ_API_KEY="..."
HUGGINGFACE_API_KEY="..."
```

## ▶️ Ejecutar
```bash
streamlit run app.py
```
Abre el enlace local que imprime Streamlit.

## 🧩 Qué incluye (mapeo con el enunciado)
- **Módulo 1 (OCR):** `easyocr.Reader` cacheado con `@st.cache_resource`, upload y previsualización de imágenes, y `st.text_area` para el texto extraído.
- **Módulo 2 (GROQ):** selector de modelo, selector de tarea, y llamada a `client.chat.completions.create(...)`.
- **Módulo 3 (Flexibilidad):** control de `temperature` y `max_tokens`, `radio` para proveedor (GROQ / HF) y ejecución con `huggingface_hub.InferenceClient`.

## 🧪 Pruebas
- Sube una foto nítida de un párrafo (papel/impreso o captura de pantalla).
- Cambia **temperature** para ver creatividad vs. precisión.
- Compara tiempos entre **GROQ** y **Hugging Face**.

## 🛠️ Solución de problemas
- **EasyOCR tarda mucho / falla:** verifica que `opencv-python-headless` no esté en conflicto; en ciertos entornos agregarlo ayuda.
- **Error GROQ:** confirma `GROQ_API_KEY` en `.env` y que el modelo seleccionado esté disponible.
- **Error Hugging Face:** confirma `HUGGINGFACE_API_KEY` y prueba un modelo diferente.
- **Texto se borra al interactuar:** se usa `st.session_state["extracted_text"]` para persistirlo; si limpias caché de Streamlit, se reinicia.

---

Hecho con ❤️ para el Taller Final de IA.
