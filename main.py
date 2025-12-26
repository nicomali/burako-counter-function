import base64
import json
import os
import time
import logging
from collections import defaultdict
from json import JSONDecodeError
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

# -------------------------------------------------
# Config
# -------------------------------------------------
MAX_IMAGE_BYTES = 2 * 1024 * 1024  # 2 MB
RESIZE_WIDTH = 1280               # ancho máximo para resize
RESIZE_HEIGHT = 960               # alto máximo para resize
RATE_LIMIT = 5                    # requests
RATE_WINDOW = 60                  # seconds
REQUIRED_CLIENT_HEADER = "burako-pwa"
LLM_TIMEOUT = 30                  # seconds
MODEL = "gpt-4o-mini"

# Load environment variables
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

# -------------------------------------------------
# App & Client
# -------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

client = OpenAI(api_key=api_key)

# CORS configuration - allow multiple origins
CORS(app, resources={r"/*": {
    "origins": [
        "*"
    ],
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "X-App-Client", "X-Forwarded-For"]
}})

# -------------------------------------------------
# In-memory rate limit (best effort)
# -------------------------------------------------
requests_log = defaultdict(list)

def is_rate_limited(ip: str) -> bool:
    now = time.time()
    requests_log[ip] = [t for t in requests_log[ip] if now - t < RATE_WINDOW]

    if len(requests_log[ip]) >= RATE_LIMIT:
        return True

    requests_log[ip].append(now)
    return False

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def resize_image(image_bytes: bytes) -> bytes:
    """Redimensiona la imagen manteniendo relación de aspecto"""
    try:
        start_time = time.time()
        original_size = len(image_bytes)
        
        # Abrir y detectar formato
        image = Image.open(BytesIO(image_bytes))
        image_format = image.format
        original_width, original_height = image.size
        
        logging.info(f"[IMG] Original: size={original_size}B, format={image_format}, dimensions={original_width}x{original_height}, mode={image.mode}")
        
        # Redimensionar
        image.thumbnail((RESIZE_WIDTH, RESIZE_HEIGHT), Image.Resampling.LANCZOS)
        resized_width, resized_height = image.size
        
        # Guardar como JPEG
        output = BytesIO()
        image.save(output, format="JPEG", quality=85)
        output.seek(0)
        resized_bytes = output.getvalue()
        resized_size = len(resized_bytes)
        
        elapsed = time.time() - start_time
        reduction = ((original_size - resized_size) / original_size) * 100
        
        logging.info(f"[IMG] Resized: size={resized_size}B, dimensions={resized_width}x{resized_height}, reduction={reduction:.1f}%, time={elapsed:.2f}s")
        
        return resized_bytes
    except Exception as e:
        logging.error(f"[IMG] Error al redimensionar: {type(e).__name__}: {e}")
        return image_bytes


BASE_PROMPT = """
Sos un sistema de visión por computadora especializado en juegos de mesa.

Analizá la imagen proporcionada.
Identificá TODAS las fichas de Burako claramente visibles.

Para cada ficha, extraé:
- number: número entero del 1 al 13 (null si es joker)
- color: uno de ["rojo","azul","verde","negro"] (null si no se distingue)
- joker: true o false

Reglas OBLIGATORIAS:
- Si una ficha es joker, number debe ser null y joker=true
- Si no estás seguro del número o color, usá null
- NO inventes fichas
- NO calcules puntos
- NO describas la imagen
- NO agregues texto fuera del JSON

Formato EXACTO:
{
  "tiles": [
    { "number": 7, "color": "rojo", "joker": false }
  ]
}
"""

RETRY_PROMPT = BASE_PROMPT + """
IMPORTANTE:
La respuesta anterior fue inválida.
Respondé SOLO con el JSON válido, sin texto adicional.
"""

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def call_llm(image_b64: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        timeout=LLM_TIMEOUT,
        messages=[
            {"role": "system", "content": "Detector de fichas de Burako"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content


def parse_and_validate(raw: str):
    data = json.loads(raw)

    if "tiles" not in data or not isinstance(data["tiles"], list):
        raise ValueError("Formato inválido")

    normalized = []
    for t in data["tiles"]:
        normalized.append({
            "number": t.get("number"),
            "color": t.get("color").lower() if isinstance(t.get("color"), str) else None,
            "joker": bool(t.get("joker"))
        })

    return normalized

# -------------------------------------------------
# Endpoint
# -------------------------------------------------
@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    try:
        # ---- Only process POST, skip OPTIONS (CORS preflight) ----
        if request.method == "OPTIONS":
            return "", 200
        
        # ---- Anti-bot header ----
        if request.headers.get("X-App-Client") != REQUIRED_CLIENT_HEADER:
            return jsonify({
                "tiles": [],
                "error": "unauthorized_client"
            }), 200

        # ---- IP & Rate limit ----
        ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        if is_rate_limited(ip):
            return jsonify({
                "tiles": [],
                "error": "rate_limited"
            }), 200

        # ---- File ----
        if "image" not in request.files:
            return jsonify({
                "tiles": [],
                "error": "missing_image"
            }), 200

        file = request.files["image"]

        # ---- Size limit ----
        file.seek(0, 2)
        size = file.tell()
        file.seek(0)
        
        logging.info(f"[REQUEST] filename={file.filename}, original_size={size}B ({size/1024/1024:.2f}MB), content_type={file.content_type}")

        if size > MAX_IMAGE_BYTES:
            logging.warning(f"[REQUEST] Image rejected: size {size}B exceeds limit {MAX_IMAGE_BYTES}B")
            return jsonify({
                "tiles": [],
                "error": "image_too_large"
            }), 200

        image_bytes = file.read()
        
        # ---- Resize image ----
        logging.info("[RESIZE] Starting image resize...")
        image_bytes = resize_image(image_bytes)
        
        # Log base64 size
        image_b64 = base64.b64encode(image_bytes).decode()
        b64_size = len(image_b64)
        logging.info(f"[BASE64] size={b64_size}B ({b64_size/1024:.2f}KB)")

        # ---- LLM call (1st attempt) ----
        logging.info(f"[LLM] Calling LLM (1st attempt) with {len(image_b64)} chars base64")
        llm_start = time.time()
        raw = call_llm(image_b64, BASE_PROMPT)
        llm_elapsed = time.time() - llm_start
        logging.info(f"[LLM] Response received in {llm_elapsed:.2f}s, length={len(raw)}")

        try:
            tiles = parse_and_validate(raw)
            logging.info(f"[PARSE] Successfully parsed {len(tiles)} tiles")
        except (JSONDecodeError, ValueError) as e:
            logging.warning(f"[PARSE] Primer intento inválido: {type(e).__name__}, reintentando...")

            # ---- Retry ----
            logging.info("[LLM] Calling LLM (2nd attempt with retry prompt)")
            llm_start = time.time()
            raw_retry = call_llm(image_b64, RETRY_PROMPT)
            llm_elapsed = time.time() - llm_start
            logging.info(f"[LLM] Retry response received in {llm_elapsed:.2f}s, length={len(raw_retry)}")
            tiles = parse_and_validate(raw_retry)
            logging.info(f"[PARSE] Retry successful, parsed {len(tiles)} tiles")

        logging.info({
            "ip": ip,
            "image_size": size,
            "tiles_detected": len(tiles)
        })

        return jsonify({ "tiles": tiles }), 200

    except Exception as e:
        import traceback
        logging.error(f"[ERROR] Critical error: {type(e).__name__}: {e}")
        logging.error(f"[TRACEBACK] {traceback.format_exc()}")

        return jsonify({
            "tiles": [],
            "error": "vision_failed"
        }), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))