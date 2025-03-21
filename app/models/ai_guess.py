import os
import io
import re
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

def configure_api_key():
    """Configura la API key desde la variable de entorno."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("La variable de entorno GOOGLE_API_KEY no está definida.")
    genai.configure(api_key=api_key)

def preprocess_image_from_bytes(image_bytes, size=(128, 128), threshold=128):
    """
    Preprocesa la imagen a partir de bytes:
    - Convierte a escala de grises.
    - Redimensiona.
    - Binariza (blanco y negro) según el umbral.
    Retorna un objeto PIL.Image.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize(size)
        image = image.point(lambda p: 255 if p > threshold else 0)
        image = image.convert("1")
        return image
    except Exception as e:
        print(f"Error al preprocesar la imagen: {e}")
        return None

def generate_description_from_pil(image, guessed_words=[], model_name="gemini-1.5-flash"):
    """
    Genera una descripción de la imagen utilizando el modelo Gemini 1.5 Flash.
    Si se proporcionan palabras adivinadas, se añaden al prompt para evitar su repetición.
    El prompt solicita que la respuesta final sea una sola palabra (en minúsculas) sin ningún otro texto,
    formateada dentro de las etiquetas <respuesta> y </respuesta>.
    """
    configure_api_key()
    model = genai.GenerativeModel(model_name)
    prompt_list = []
    if guessed_words:
        filter_text = "No uses las siguientes palabras ya que las haz puesto anterior mente y no son la palabra esperada: " + ", ".join(guessed_words) + ". "
        prompt_list.append(filter_text)
        print(filter_text)
    prompt_list.extend([
        "No repitas las palabras: " + ", ".join(guessed_words) + ".",
        "No uses mayúsculas, signos de puntuación ni caracteres especiales exeptuando la ñ .",
        "Eres un modelo de inteligencia artificial especializado en jugar Pictionary. Se te mostrará una imagen y tu tarea es adivinar lo que representa.",
        "Analiza la imagen y piensa que podria ser de las palabras que tienes a continuacion, tu respuesta debe de ser una sola palabra, sin explicacion ni nada mas solo una palabra",
        "Asegúrate de que el formato de la respuesta final sea exacto, No incluyas ningún otro texto para la respuesta final fuera de la palabra que es la respuesta.",
        "Tu respuesta debe consistir en una sola palabra de la lista anterior, escrita en minúsculas y sin caracteres especiales y sin espacios, solo la palaba bien escrita porfavor con tildes etc, escribe como estan enumeradas anteriormente.",
        "[IMAGEN]",
        image
        ],)
    try:
        response = model.generate_content(
            prompt_list,
            stream=True
        )
        response.resolve()
        full_text = ""
        for chunk in response:
            full_text += chunk.text
        final_answer = full_text
        return full_text
    except Exception as e:
        return f"Error al generar la descripción: {e}"




