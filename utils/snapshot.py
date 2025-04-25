"""
Gestión de snapshots para el sistema de detección de patentes
"""

import threading
import logging
import time
import requests
import numpy as np
import cv2
import io
from PIL import Image
import concurrent.futures
from requests.auth import HTTPDigestAuth
from config import SNAPSHOT_URL
import os
import datetime

def fetch_hd_snapshot(url=SNAPSHOT_URL, retries=3, wait_between=1):
    """
    Intenta obtener el snapshot HD con reintentos.
    Si no se logra obtener tras varios intentos, se lanza excepción.
    
    Args:
        url: URL para obtener el snapshot
        retries: Número de intentos antes de fallar
        wait_between: Tiempo en segundos entre reintentos
        
    Returns:
        numpy.ndarray: Imagen HD obtenida
        
    Raises:
        Exception: Si no se pudo obtener snapshot tras los reintentos
    """
    for intento in range(retries):
        try:
            response = requests.get(url, auth=HTTPDigestAuth('admin', 'emilia09'), stream=True, timeout=5)
            response.raise_for_status()
            image_bytes = response.content
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img = pil_img.convert("RGB")
            hd_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            if hd_img is not None and hd_img.size > 0:
                logging.debug(f"fetch_hd_snapshot: Snapshot obtenido en el intento {intento+1}")
                return hd_img
        except Exception as e:
            logging.warning(f"fetch_hd_snapshot: Intento {intento+1} - Error obteniendo snapshot HD: {e}")
            time.sleep(wait_between)
    raise Exception("No se pudo obtener un snapshot HD tras múltiples intentos")

class SnapshotManager:
    """
    Clase para gestionar snapshots asíncronos de la cámara.
    Implementa un sistema de caché y promesas para optimizar las solicitudes.
    """
    def __init__(self):
        """Inicializa el gestor de snapshots"""
        self.running = True
        self.latest_snapshot = None
        self.lock = threading.Lock()
        self.update_event = threading.Event()
        self.request_id = None  # ID de la solicitud actual
        self.future = None      # Future que se resolverá cuando esté listo el snapshot
        self.snapshot_in_progress = False  # Evita múltiples solicitudes simultáneas
        self.thread = threading.Thread(target=self.update_snapshot_loop, daemon=True)
        self.thread.start()
    
    def update_snapshot_loop(self):
        """
        Bucle principal que actualiza los snapshots cuando se solicitan.
        Este método se ejecuta en un thread separado.
        """
        while self.running:
            self.update_event.wait()  # Espera hasta que se solicite actualización
            if not self.running:
                break
            self.update_event.clear()
            with self.lock:
                self.snapshot_in_progress = True
            try:
                # Intenta obtener el snapshot con reintentos
                new_snapshot = fetch_hd_snapshot()
                with self.lock:
                    self.latest_snapshot = new_snapshot
                logging.debug(f"SnapshotManager - id: {self.request_id}, snapshot recibido")
                if self.future is not None:
                    self.future.set_result((self.request_id, new_snapshot))
                    self.future = None
                self.request_id = None
            except Exception as e:
                logging.warning(f"SnapshotManager - id: {self.request_id} error obteniendo el snapshot. Error: {e}")
                if self.future is not None:
                    self.future.set_exception(e)
                    self.future = None
            finally:
                with self.lock:
                    self.snapshot_in_progress = False

    def request_update_future(self, snapshot_id):
        """
        Solicita una actualización del snapshot. Si ya hay uno en progreso, se ignora la nueva solicitud.
        
        Args:
            snapshot_id: Identificador único para esta solicitud
            
        Returns:
            concurrent.futures.Future: Promesa que se resuelve cuando el snapshot está listo
        """
        with self.lock:
            if self.snapshot_in_progress:
                logging.debug("SnapshotManager: Ya hay un snapshot en proceso; se omite nueva solicitud.")
                # Se retorna la misma Future para no duplicar la solicitud.
                return self.future if self.future is not None else concurrent.futures.Future()
            self.request_id = snapshot_id
            self.future = concurrent.futures.Future()
            logging.debug(f"SnapshotManager - id: {self.request_id}, solicitando snapshot (promesa)")
            self.update_event.set()
            return self.future
    
    def get_latest_snapshot(self):
        """
        Obtiene el snapshot más reciente.
        
        Returns:
            numpy.ndarray: El snapshot más reciente o None si no hay ninguno
        """
        with self.lock:
            return self.latest_snapshot.copy() if self.latest_snapshot is not None else None
    
    def stop(self):
        """
        Detiene el thread de actualización de snapshots.
        Debe llamarse al finalizar la aplicación.
        """
        self.running = False
        self.update_event.set()  # Desbloquea el thread de actualización
        self.thread.join(timeout=2)  # Espera a que termine el thread

"""
Funciones para capturar y guardar imágenes de patentes detectadas
"""

def save_plate_snapshot(image, plate_text, confidence, base_path="snapshots"):
    """Guarda una captura de la patente detectada"""
    # Crear directorio si no existe
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # Generar nombre de archivo con timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_path}/plate_{plate_text}_{confidence:.2f}_{timestamp}.jpg"
    
    # Guardar imagen
    cv2.imwrite(filename, image)
    
    return filename

def crop_plate_region(image, bbox, padding=10):
    """Recorta la región de la patente de la imagen original"""
    x, y, w, h = bbox
    
    # Aplicar padding y asegurar que esté dentro de los límites de la imagen
    height, width = image.shape[:2]
    
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)
    
    # Recortar imagen
    cropped = image[y1:y2, x1:x2]
    
    return cropped

def delete_snapshot_file(filepath):
    """
    Elimina un archivo de snapshot si existe
    
    Args:
        filepath: Ruta del archivo a eliminar
        
    Returns:
        bool: True si se eliminó correctamente o no existía, False si hubo error
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logging.debug(f"Snapshot eliminado: {filepath}")
        return True
    except Exception as e:
        logging.error(f"Error al eliminar snapshot {filepath}: {e}")
        return False
