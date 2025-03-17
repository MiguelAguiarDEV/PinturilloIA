import os
import json
import base64
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Importamos las funciones de nuestro módulo de IA
from models.ai_guess import preprocess_image_from_bytes, generate_description_from_pil

app = FastAPI()

# Montamos los archivos estáticos en "/static"
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Definimos una ruta para servir el index.html en la raíz
@app.get("/")
async def get_index():
    return FileResponse("frontend/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        try:
            message = json.loads(data)
            if message.get("type") == "draw":
                # Procesa las coordenadas del trazo (opcional)
                x = message.get("x")
                y = message.get("y")
                print("Coordenadas recibidas:", x, y)
                await websocket.send_text("Coordenadas recibidas.")
            elif message.get("type") == "canvasImage":
                # Procesa la imagen del canvas
                image_data = message.get("data")
                header, encoded = image_data.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                
                # Preprocesa la imagen para convertirla en blanco y negro simplificado
                preprocessed_image = preprocess_image_from_bytes(image_bytes)
                if preprocessed_image is None:
                    await websocket.send_text("Error al preprocesar la imagen.")
                    continue
                
                # Genera la descripción (o adivinanza) usando la API de Google Generative AI
                guess = generate_description_from_pil(preprocessed_image)
                await websocket.send_text(f"IA dice: {guess}")
            else:
                await websocket.send_text("Tipo de mensaje desconocido.")
        except json.JSONDecodeError:
            await websocket.send_text("Error al decodificar el mensaje.")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
