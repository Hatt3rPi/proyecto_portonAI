"""
Funciones de procesamiento de imágenes para PortonAI
"""

import cv2
import numpy as np
import json
import logging
import os
import math
from datetime import datetime
from PIL import Image
import sys
from config import DEFAULT_CALIBRATION_PARAMS, CALIBRATION_FILE

def resize_image(image, width=None, height=None):
    """Redimensiona una imagen manteniendo su relación de aspecto"""
    dim = None
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def resize_for_inference(frame, max_dim=640):
    """
    Redimensiona una imagen manteniendo su relación de aspecto para inferencia.
    
    Args:
        frame: Imagen a redimensionar
        max_dim: Dimensión máxima (ancho o alto) para la imagen de salida
        
    Returns:
        tuple: (imagen redimensionada, factor_escala_x, factor_escala_y)
    """
    h, w = frame.shape[:2]
    if w >= h:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
    else:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
    resized = cv2.resize(frame, (new_w, new_h))
    scale_x = w / new_w
    scale_y = h / new_h
    return resized, scale_x, scale_y

def preprocess_for_plate_detection(image):
    """Preprocesa una imagen para detección de patentes"""
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro gaussiano para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Normalizar contraste
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized

def load_calibration_params():
    """
    Carga los parámetros de calibración desde un archivo JSON.
    
    Returns:
        dict: Parámetros de calibración
    """
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, "r") as f:
            params = json.load(f)
            logging.info("Parámetros de calibración cargados: %s", params)
            return params
    else:
        logging.info("No se encontró archivo de calibración. Se usan valores por defecto.")
        return DEFAULT_CALIBRATION_PARAMS.copy()

def save_calibration_params(params):
    """
    Guarda los parámetros de calibración en un archivo JSON.
    
    Args:
        params: Diccionario con parámetros a guardar
    """
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(params, f, indent=4)
    logging.info("Parámetros de calibración guardados: %s", params)

def process_image(image):
    """
    Procesa una imagen para visualización, añadiendo timestamp.
    
    Args:
        image: Imagen a procesar
        
    Returns:
        numpy.ndarray: Imagen procesada con timestamp
    """
    height, width = image.shape[:2]
    if width > 1280 or height > 720:
        image = cv2.resize(image, (1280, 720))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(image, timestamp, (20, image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def preprocess_frame(image, calib_params):
    """
    Aplica preprocesamiento a un frame según los parámetros de calibración.
    
    Args:
        image: Imagen a procesar
        calib_params: Diccionario con parámetros de calibración
        
    Returns:
        numpy.ndarray: Imagen procesada
    """
    # En esta implementación simplificada, solo devolvemos una copia
    # En una implementación completa, se aplicarían los ajustes de gamma, CLAHE, etc.
    return image.copy()

def correct_plate_orientation(plate_img):
    """
    Corrige la orientación de una imagen de patente.
    
    Args:
        plate_img: Imagen de la patente
        
    Returns:
        numpy.ndarray: Imagen con orientación corregida
    """
    try:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return plate_img
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        angle = rect[-1]
        if angle < -45:
            angle += 90
        (h, w) = plate_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(plate_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception as e:
        logging.warning(f"Error corrigiendo la orientación de la patente: {e}")
        return plate_img

def is_frame_valid(frame):
    """
    Verifica si un frame es válido para procesamiento.
    
    Args:
        frame: Frame a verificar
        
    Returns:
        bool: True si el frame es válido, False en caso contrario
    """
    return frame is not None and frame.size > 0

def calculate_roi_for_coverage(imagen_hd, center_coords, plate_area, coverage_fraction, target_size=(640, 320)):
    WIDTH_OCR, HEIGHT_OCR = target_size
    AREA_OCR = WIDTH_OCR * HEIGHT_OCR
    
    if plate_area <= 0:
        return None
    s = math.sqrt((coverage_fraction * AREA_OCR) / float(plate_area))
    region_width = WIDTH_OCR / s
    region_height = HEIGHT_OCR / s
    center_x, center_y = center_coords
    x1_roi = int(center_x - region_width / 2)
    y1_roi = int(center_y - region_height / 2)
    x2_roi = int(x1_roi + region_width)
    y2_roi = int(y1_roi + region_height)
    h_img, w_img = imagen_hd.shape[:2]
    
    # --- APLICAR MARGEN DE SEGURIDAD ---
    MARGIN = 20
    if x1_roi < MARGIN or y1_roi < MARGIN or x2_roi > (w_img - MARGIN) or y2_roi > (h_img - MARGIN):
        return None
    
    roi = imagen_hd[y1_roi:y2_roi, x1_roi:x2_roi]
    try:
        roi_resized = cv2.resize(roi, target_size)
    except Exception as e:
        logging.warning(f"Error redimensionando ROI: {e}")
        return None
    return roi_resized
