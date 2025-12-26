import base64
import json
import os
import time
import logging
from collections import defaultdict
from json import JSONDecodeError
from functools import wraps

import functions_framework
from flask import Request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------------------------------
# Config
# -------------------------------------------------
MAX_IMAGE_BYTES = 2 * 1024 * 1024  # 2 MB
RATE_LIMIT = 5                    # requests
RATE_WINDOW = 60                  # seconds
REQUIRED_CLIENT_HEADER = "burako-pwa"
LLM_TIMEOUT = 10                  # seconds
MODEL = "gpt-4o-mini"

# Load environment variables
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

# -------------------------------------------------
# Client
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)

client = OpenAI(api_key=api_key)

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
# Prompts
# -------------------------------------------------
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
def add_cors_headers(func):
    """Decorator to add CORS headers to response"""
    @wraps(func)
    def wrapper(request):
        # Handle preflight requests
        if request.method == 'OPTIONS':
            headers = {
                'Access-Control-Allow-Origin': 'https://nicomali.github.io/burako-counter-pwa/',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, X-App-Client, X-Forwarded-For',
                'Access-Control-Max-Age': '3600'
            }
            return '', 204, headers
        
        response = func(request)
        
        # Add CORS headers to response
        if isinstance(response, tuple):
            body, status_code = response[0], response[1]
            headers = response[2] if len(response) > 2 else {}
        else:
            body = response
            status_code = 200
            headers = {}
        
        headers['Access-Control-Allow-Origin'] = 'https://nicomali.github.io/burako-counter-pwa/'
        headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        headers['Access-Control-Allow-Headers'] = 'Content-Type, X-App-Client, X-Forwarded-For'
        
        return body, status_code, headers
    
    return wrapper

@functions_framework.http
@add_cors_headers
def analyze(request: Request):
    try:
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

        if size > MAX_IMAGE_BYTES:
            return jsonify({
                "tiles": [],
                "error": "image_too_large"
            }), 200

        image_bytes = file.read()
        image_b64 = base64.b64encode(image_bytes).decode()

        # ---- LLM call (1st attempt) ----
        raw = call_llm(image_b64, BASE_PROMPT)

        try:
            tiles = parse_and_validate(raw)
        except (JSONDecodeError, ValueError):
            logging.warning("Primer intento inválido, reintentando")

            # ---- Retry ----
            raw_retry = call_llm(image_b64, RETRY_PROMPT)
            tiles = parse_and_validate(raw_retry)

        logging.info({
            "ip": ip,
            "image_size": size,
            "tiles_detected": len(tiles)
        })

        return jsonify({ "tiles": tiles }), 200

    except Exception as e:
        logging.error(f"Error crítico: {e}")

        return jsonify({
            "tiles": [],
            "error": "vision_failed"
        }), 200