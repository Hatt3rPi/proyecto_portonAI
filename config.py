## archivo: config.py
#!/usr/bin/env python

"""
Configuraciones globales para el sistema PortonAI
"""

import os
import sys
import json
import logging
from collections import deque
from dotenv import load_dotenv
from typing import Optional

# Cargar variables de entorno desde .env
load_dotenv()

# Paths base
BASE = os.path.abspath(os.path.dirname(__file__))

# ---------------------------------------------------------------------
# CONFIGURACIÓN GENERAL: Modo DEBUG y logging
# ---------------------------------------------------------------------
DEBUG_MODE = True  # Cambia a False para producción
if DEBUG_MODE:
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
else:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("yolov8").setLevel(logging.ERROR)

# ---------------------------------------------------------------------
# Modo online / offline
# ---------------------------------------------------------------------
# True = leer la cámara en vivo (RTSP); False = usar un vídeo de prueba
ONLINE_MODE = True

# ---------------------------------------------------------------------
# CONSTANTES Y PARÁMETROS BÁSICOS
# ---------------------------------------------------------------------
FPS_DEQUE_MAXLEN = 20
MODELO_OBJETOS = os.path.join(BASE, "modelos", "yolov8n.pt")
MODELO_PATENTE = os.path.join(BASE, "modelos", "modelo_PATENTE.engine")
MODELO_OCR = os.path.join(BASE, "modelos", "modelo_OCR.engine")

# Variables sensibles via env
try:
    TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
except KeyError:
    logging.error("Variable TELEGRAM_TOKEN no encontrada.")
    sys.exit(1)

try:
    chat_id_str = os.environ["TELEGRAM_CHAT_ID"]
    chat_id_str = chat_id_str.replace("{", "").replace("}", "")
    TELEGRAM_CHAT_ID = set(int(i.strip()) for i in chat_id_str.split(",") if i.strip())
    if not TELEGRAM_CHAT_ID:
        raise ValueError
except Exception:
    logging.error("Error al procesar TELEGRAM_CHAT_ID.")
    sys.exit(1)

CONFIANZA_PATENTE = 70       # porcentaje mínimo para detección
DISPLAY_DURATION = 2         # segundos para mantener en pantalla
SNAPSHOT_LATENCY_MS = 500
SENT_PLATES = {}
UMBRAL_SNAPSHOT_AREA = 1200  # px² mínimo para snapshot HD

VEHICLE_MEMORY_TIME = 15     # seg para considerar vehículo “nuevo”
SAME_VEHICLE_IOU_THRESHOLD = 0.3

CALIBRATION_FILE = os.path.join(BASE, "calibration_params.json")

# ---------------------------------------------------------------------
# ESQUEMA REAL de calibration_params.json y valores por defecto
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# PARÁMETROS DE CONSENSO de OCR
# ---------------------------------------------------------------------
from typing import Optional

# Longitud mínima para incluir una lectura en el consenso
CONSENSUS_MIN_LENGTH: int = 5
# Estrategia para determinar la longitud esperada de la matrícula:
#   'mode'  → el valor más frecuente entre lecturas
#   'median'→ la mediana de las longitudes
#   'fixed' → usar CONSENSUS_FIXED_LENGTH
CONSENSUS_EXPECTED_LENGTH_METHOD: str = "mode"
# Si se usa estrategia 'fixed', aquí se define:
CONSENSUS_FIXED_LENGTH: Optional[int] = None

# ---------------------------------------------------------------------
# UMBRALES DÍA/NOCHE
# ---------------------------------------------------------------------
NIGHT_THRESHOLD = 50   # brillo
DAY_THRESHOLD = 70     # brillo

# ---------------------------------------------------------------------
# URLs y endpoints
# ---------------------------------------------------------------------
SNAPSHOT_URL = "http://192.168.0.124/cgi-bin/snapshot.cgi?1"
URL_HD = "rtsp://admin:emilia09@192.168.0.124:554/cam/realmonitor?channel=1&subtype=0&tcp"

BACKEND_URL = "https://porton-ia-back-production.up.railway.app/accesLog/registro"
try:
    headers_json = os.environ["BACKEND_HEADERS"]
    BACKEND_HEADERS = json.loads(headers_json)
    if "Content-Type" not in BACKEND_HEADERS or "Authorization" not in BACKEND_HEADERS:
        raise ValueError
except Exception:
    logging.error("Error al cargar BACKEND_HEADERS.")
    sys.exit(1)

# ---------------------------------------------------------------------
# CONSTANTES OCR (ROI y velocidad)
# ---------------------------------------------------------------------
WIDTH_OCR = 640
HEIGHT_OCR = 320
AREA_OCR = WIDTH_OCR * HEIGHT_OCR
FAST_AREA_RATE_THRESHOLD = 0.05  # px²/ms

# Configuración de la zona de OCR Stream basado en zona de detección (columna imaginaria)
OCR_STREAM_ZONE = {
    "start_x_pct": 0.07,   # Desde el 7% del ancho
    "end_x_pct": 0.35,     # Hasta el 35% del ancho
    "start_y_pct": 0.10,   # Desde el 10% del alto
    "end_y_pct": 0.85      # Hasta el 85% del alto (o zona de exclusión)
}

# ---------------------------------------------------------------------
# ACTIVACIÓN DE MODALIDADES OCR
# ---------------------------------------------------------------------
OCR_STREAM_ACTIVATED = True    # Activar/desactivar OCR en tiempo real
OCR_SNAPSHOT_ACTIVATED = False  # Activar/desactivar OCR mediante snapshot HD
OCR_OPENAI_ACTIVATED = False   # Activar/desactivar uso de OpenAI para OCR

# ---------------------------------------------------------------------
# CONFIGURACIÓN DE ANÁLISIS QA AVANZADO
# ---------------------------------------------------------------------
QA_ANALISIS_AVANZADO = True  # Activar/desactivar análisis avanzado en QA_mode
QA_ANGULOS_VARIACION = ((-5, 10), 0.5)  # (rango_ángulos, paso) en grados
QA_ESCALAS_VARIACION = (50, 150, 5)  # (min, max, paso) en porcentaje
