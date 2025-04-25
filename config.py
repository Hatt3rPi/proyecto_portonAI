#!/usr/bin/env python

"""
Configuraciones globales para el sistema PortonAI
"""

import os
import logging
from collections import deque
import sys
import json
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Paths base
BASE = os.path.abspath(os.path.dirname(__file__))
MON_PATH = os.path.join(BASE, "scripts", "monitoreo_patentes")
sys.path.insert(0, MON_PATH)

# ---------------------------------------------------------------------
# CONFIGURACIÓN GENERAL: Modo DEBUG
# ---------------------------------------------------------------------
DEBUG_MODE = True  # Cambia a True para activar el modo visual de depuración
ONLINE_MODE = True  # O False si deseas usar un archivo de video

# ---------------------------------------------------------------------
# CONFIGURACIÓN DE LOGGING
# ---------------------------------------------------------------------
if DEBUG_MODE:
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
else:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("yolov8").setLevel(logging.ERROR)

# ---------------------------------------------------------------------
# CONSTANTES Y PARÁMETROS
# ---------------------------------------------------------------------
FPS_DEQUE_MAXLEN = 10

MODELO_OBJETOS = os.path.join(BASE, "modelos", "yolov8n.pt")
MODELO_PATENTE = os.path.join(BASE, "modelos", "modelo_PATENTE.engine")
MODELO_OCR = os.path.join(BASE, "modelos", "modelo_OCR.engine")

# Cargar valores sensibles desde variables de entorno
try:
    TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
except KeyError:
    logging.error("Variable de entorno TELEGRAM_TOKEN no encontrada. Debe establecer esta variable.")
    sys.exit(1)

# Convertir CHAT_ID desde string a set de integers
try:
    chat_id_str = os.environ["TELEGRAM_CHAT_ID"]
    chat_id_str = chat_id_str.replace("{", "").replace("}", "")
    TELEGRAM_CHAT_ID = set(int(id_.strip()) for id_ in chat_id_str.split(",") if id_.strip())
    if not TELEGRAM_CHAT_ID:
        raise ValueError("El conjunto de TELEGRAM_CHAT_ID está vacío")
except Exception as e:
    logging.error(f"Error al procesar TELEGRAM_CHAT_ID: {e}. Debe configurar correctamente esta variable de entorno.")
    sys.exit(1)

CONFIANZA_PATENTE = 60  # porcentaje
DISPLAY_DURATION = 2
SNAPSHOT_LATENCY_MS = 500
SENT_PLATES = {}
UMBRAL_SNAPSHOT_AREA = 1200

# Tiempo mínimo (en segundos) para considerar una nueva detección del mismo vehículo
VEHICLE_MEMORY_TIME = 15
# Umbral de IOU para considerar que es el mismo vehículo
SAME_VEHICLE_IOU_THRESHOLD = 0.3

CALIBRATION_FILE = os.path.join(BASE, "calibration_params.json")

# Valores por defecto basados en la estructura real de calibration_params.json
#   (véase archivo calibration_params.json) :contentReference[oaicite:0]{index=0}
DEFAULT_CALIBRATION_PARAMS = {
    "camera_matrix": [
        [1000.0, 0.0, 320.0],
        [0.0, 1000.0, 240.0],
        [0.0, 0.0, 1.0]
    ],
    "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
    "region_of_interest": {
        "x": 100,
        "y": 150,
        "width": 440,
        "height": 330
    },
    "detection_threshold": 0.65,
    "tracking_parameters": {
        "max_disappeared": 30,
        "max_distance": 50
    },
    "calibration_date": "1970-01-01T00:00:00"
}

# Definición de umbrales para cambiar de modo (para evitar fluctuaciones)
NIGHT_THRESHOLD = 50  # Por debajo de este valor se activa el modo noche
DAY_THRESHOLD = 70    # Por encima de este valor se activa el modo día

# URL para snapshot HD (imagen de mayor resolución)
SNAPSHOT_URL = "http://192.168.0.124/cgi-bin/snapshot.cgi?1"
URL_HD = "rtsp://admin:emilia09@192.168.0.124:554/cam/realmonitor?channel=1&subtype=1&tcp"

# ---------------------------------------------------------------------
# CONSTANTES PARA OCR
# ---------------------------------------------------------------------
WIDTH_OCR = 640
HEIGHT_OCR = 320
AREA_OCR = WIDTH_OCR * HEIGHT_OCR

# ------------------- CONSTANTE PARA CONTROL DE VELOCIDAD -------------------
FAST_AREA_RATE_THRESHOLD = 0.05  # px²/ms; umbral para clasificar "rápido" vs "lento"

# ---------------------------------------------------------------------
# CONFIGURACIÓN DE BACKEND
# ---------------------------------------------------------------------
BACKEND_URL = "https://porton-ia-back-production.up.railway.app/accesLog/registro"
try:
    backend_headers_json = os.environ["BACKEND_HEADERS"]
    BACKEND_HEADERS = json.loads(backend_headers_json)
    if "Content-Type" not in BACKEND_HEADERS or "Authorization" not in BACKEND_HEADERS:
        raise ValueError("BACKEND_HEADERS debe contener 'Content-Type' y 'Authorization'")
except Exception as e:
    logging.error(f"Error al cargar BACKEND_HEADERS: {e}. Debe configurar correctamente esta variable de entorno.")
    sys.exit(1)
