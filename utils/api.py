"""
Funciones para comunicación con APIs externas, backend y Telegram
"""

import requests
import json
import base64
import cv2
import logging
import time
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, BACKEND_URL, BACKEND_HEADERS, SENT_PLATES

def encode_image(image):
    """
    Codifica una imagen en base64 para envío por API.
    
    Args:
        image: Imagen a codificar (numpy.ndarray)
        
    Returns:
        str: Imagen codificada en base64
    """
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode("utf-8")

def send_backend(plate_text, plate_img):
    """
    Envía datos de detección al backend.
    
    Args:
        plate_text: Texto de la patente
        plate_img: Imagen de la patente
    
    Returns:
        bool: True si se envió correctamente, False en caso contrario
    """
    payload = {
        "id_condominio": "67d8cb2bbccb17ad0c51a547",
        "id_dispositivo": "67d8cb2bbccb17ad0c51a555",
        "patente_leida": plate_text,
        "imagen_patente": encode_image(plate_img),
        "metodo": "IA",
        "resultado": "N/A",
        "tipo_evento": "avistamiento"
    }
    
    try:
        response = requests.post(BACKEND_URL, json=payload, headers=BACKEND_HEADERS, timeout=5)
        response.raise_for_status()
        logging.info(f"Datos enviados correctamente al backend: {plate_text}")
        return True
    except requests.exceptions.RequestException as e:
        logging.warning(f"⚠️ Error al enviar al backend: {e}")
        return False

def send_plate_async(plate_img, frame_img, plate_text, confidence, coords):
    """
    Envía datos de detección de patente a Telegram de forma asíncrona.
    
    Args:
        plate_img: Imagen de la patente
        frame_img: Imagen del frame completo
        plate_text: Texto de la patente
        confidence: Texto con información de confianza
        coords: Coordenadas de la patente
    """
    # Evitar envío de patentes repetidas
    current_time = time.time()
    twenty_minutes = 20 * 60
    
    # Limpiar patentes antiguas
    for key in list(SENT_PLATES.keys()):
        if current_time - SENT_PLATES[key] > twenty_minutes:
            del SENT_PLATES[key]
            
    if plate_text in SENT_PLATES:
        return
        
    # Medir tiempo de procesamiento
    start_proc = time.time()
    processing_time = time.time() - start_proc
    capture_time = time.strftime("%d/%m %H:%M:%S", time.localtime())
    
    # Crear mensaje
    message = f"Patente: {plate_text}\nResultados:\n{confidence}\nFecha captura: {capture_time}\nTiempo de procesamiento: {processing_time:.4f} s"
    
    # Preparar solicitud a Telegram
    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMediaGroup"
    _, plate_buffer = cv2.imencode(".jpg", plate_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    _, frame_buffer = cv2.imencode(".jpg", frame_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    media = [
        {"type": "photo", "media": "attach://plate_image", "caption": message},
        {"type": "photo", "media": "attach://frame_image"}
    ]
    files = {
        "plate_image": ("plate.jpg", plate_buffer.tobytes(), "image/jpeg"),
        "frame_image": ("frame.jpg", frame_buffer.tobytes(), "image/jpeg")
    }
    
    # Enviar a cada chat ID configurado
    chat_ids = list(TELEGRAM_CHAT_ID) if isinstance(TELEGRAM_CHAT_ID, set) else [TELEGRAM_CHAT_ID]
    
    success = False
    for chat_id in chat_ids:
        payload = {"chat_id": chat_id, "media": json.dumps(media)}
        try:
            response = requests.post(telegram_url, data=payload, files=files, timeout=5)
            response.raise_for_status()
            success = True
            logging.info(f"Patente {plate_text} enviada a Telegram (chat_id: {chat_id})")
        except Exception as e:
            logging.warning(f"Error al enviar a Telegram (chat_id {chat_id}): {e}")
    
    # Registrar patente como enviada si al menos un envío fue exitoso
    if success:
        SENT_PLATES[plate_text] = current_time
        return True
    return False
