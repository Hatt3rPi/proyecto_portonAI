#!/usr/bin/env python
"""
Configuraciones globales para el sistema PortonAI
"""

import os
import logging
from collections import deque
import sys

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
TELEGRAM_TOKEN = "7880232433:AAG_YDSENmmjWVasD4UaE4g1ClkeLt2RZ_o"
TELEGRAM_CHAT_ID = {1567062024}
CONFIANZA_PATENTE = 60   # porcentaje
DISPLAY_DURATION = 2
SNAPSHOT_LATENCY_MS = 500
SENT_PLATES = {}
UMBRAL_SNAPSHOT_AREA = 1200

# Tiempo mínimo (en segundos) para considerar una nueva detección del mismo vehículo
VEHICLE_MEMORY_TIME = 15
# Umbral de IOU para considerar que es el mismo vehículo
SAME_VEHICLE_IOU_THRESHOLD = 0.3

CALIBRATION_FILE = os.path.join(BASE, "calibration_params.json")
DEFAULT_CALIBRATION_PARAMS = {
    "gamma": 1.0,
    "clahe_enabled": False,
    "clahe_clip_limit": 0.0,
    "clahe_tile_grid_size": [8, 8],
    "bilateral_d": 0,
    "bilateral_sigmaColor": 0,
    "bilateral_sigmaSpace": 0
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
FAST_AREA_RATE_THRESHOLD = 0.05   # px²/ms; umbral para clasificar "rápido" vs "lento"

# ---------------------------------------------------------------------
# CONFIGURACIÓN DE BACKEND
# ---------------------------------------------------------------------
BACKEND_URL = "https://porton-ia-back-production.up.railway.app/accesLog/registro"
BACKEND_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
