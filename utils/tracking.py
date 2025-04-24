"""
Funciones y clases para el tracking de vehículos y objetos
"""

import cv2
import logging
import time
import numpy as np
from collections import deque
from config import SAME_VEHICLE_IOU_THRESHOLD, VEHICLE_MEMORY_TIME

def create_tracker():
    """
    Crea un tracker CSRT si está disponible. Usa KCF como fallback.
    
    Returns:
        tracker: Un objeto tracker de OpenCV
    """
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        return cv2.legacy.TrackerCSRT_create()
    elif hasattr(cv2, 'TrackerKCF_create'):
        logging.warning("No se encontró CSRT, se usará TrackerKCF como alternativa.")
        return cv2.TrackerKCF_create()
    elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
        logging.warning("No se encontró CSRT, se usará TrackerKCF (legacy) como alternativa.")
        return cv2.legacy.TrackerKCF_create()
    else:
        raise AttributeError("No se encontró ningún tracker disponible en cv2.")

def compute_iou(boxA, boxB):
    """
    Calcula la intersección sobre unión (IoU) entre dos bounding boxes.
    
    Args:
        boxA: Primera caja en formato (x1, y1, x2, y2)
        boxB: Segunda caja en formato (x1, y1, x2, y2)
        
    Returns:
        float: Valor IoU entre 0.0 y 1.0
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Calcular área de intersección
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    # Calcular áreas de cada bounding box
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Calcular IoU
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom != 0 else 0

def find_vehicle_type_for_plate(plate_box, vehicle_boxes, iou_thresh=0.2):
    """
    Encuentra el tipo de vehículo al que pertenece una placa.
    
    Args:
        plate_box: Caja de la placa en formato (x1, y1, x2, y2)
        vehicle_boxes: Lista de diccionarios con claves 'class' y 'bbox'
        iou_thresh: Umbral IoU para considerar coincidencia
        
    Returns:
        str: Tipo de vehículo o None si no se encuentra coincidencia
    """
    for v in vehicle_boxes:
        iou = compute_iou(plate_box, v["bbox"])
        if iou > iou_thresh:
            return v["class"]
    return None

class TrackerManager:
    """
    Clase para gestionar múltiples trackers de vehículos.
    """
    def __init__(self):
        """Inicializa un gestor de trackers"""
        self.trackers = {}  # ID -> {tracker, box, last_seen, vehicle_type}
        self.next_id = 0
    
    def add_tracker(self, frame, bbox, vehicle_type=None):
        """
        Añade un nuevo tracker.
        
        Args:
            frame: Frame actual
            bbox: Bounding box en formato (x, y, w, h)
            vehicle_type: Tipo de vehículo (opcional)
            
        Returns:
            int: ID del tracker creado
        """
        tracker = create_tracker()
        success = tracker.init(frame, bbox)
        
        if not success:
            logging.warning("No se pudo inicializar el tracker")
            return None
            
        tracker_id = self.next_id
        self.next_id += 1
        
        self.trackers[tracker_id] = {
            'tracker': tracker,
            'bbox': bbox,
            'last_seen': time.time(),
            'vehicle_type': vehicle_type
        }
        
        return tracker_id
    
    def update_all(self, frame):
        """
        Actualiza todos los trackers con el frame actual.
        
        Args:
            frame: Frame actual
            
        Returns:
            dict: Mapa de IDs a bounding boxes actualizados
        """
        results = {}
        trackers_to_remove = []
        
        for tracker_id, tracker_data in self.trackers.items():
            # Actualizar tracker
            success, bbox = tracker_data['tracker'].update(frame)
            
            if success:
                # Actualizar datos
                tracker_data['bbox'] = bbox
                tracker_data['last_seen'] = time.time()
                results[tracker_id] = bbox
            else:
                logging.debug(f"Tracker {tracker_id} perdió el seguimiento")
            
            # Eliminar trackers viejos
            if time.time() - tracker_data['last_seen'] > VEHICLE_MEMORY_TIME:
                trackers_to_remove.append(tracker_id)
        
        # Eliminar trackers caducados
        for tracker_id in trackers_to_remove:
            del self.trackers[tracker_id]
            
        return results
    
    def get_tracker_data(self, tracker_id):
        """
        Obtiene datos de un tracker específico.
        
        Args:
            tracker_id: ID del tracker
            
        Returns:
            dict: Datos del tracker o None si no existe
        """
        return self.trackers.get(tracker_id)
    
    def remove_tracker(self, tracker_id):
        """
        Elimina un tracker específico.
        
        Args:
            tracker_id: ID del tracker a eliminar
        """
        if tracker_id in self.trackers:
            del self.trackers[tracker_id]

def update_plate_area_history(plate_entry, current_area):
    """
    Actualiza o crea el historial de áreas (timestamp en ms, área en px²)
    
    Args:
        plate_entry: Diccionario de entrada de placa
        current_area: Área actual de la placa en píxeles cuadrados
    """
    if "area_history" not in plate_entry:
        plate_entry["area_history"] = deque(maxlen=5)
    current_time_ms = int(time.time() * 1000)
    plate_entry["area_history"].append((current_time_ms, current_area))

def compute_smoothed_rate_from_history(area_history):
    """
    Calcula la pendiente (px²/ms) usando regresión lineal sobre el historial
    
    Args:
        area_history: Lista de tuplas (timestamp_ms, area_px2)
        
    Returns:
        float: Pendiente de crecimiento de área o None si no hay suficientes datos
    """
    if len(area_history) < 2:
        return None
    times = np.array([entry[0] for entry in area_history])
    areas = np.array([entry[1] for entry in area_history])
    slope, intercept = np.polyfit(times, areas, 1)
    return slope

def predict_time_to_threshold(current_area, area_history, threshold=1200, snapshot_latency_ms=300):
    """
    Estima el timestamp (en ms) en que el área llegará a threshold, considerando la latencia
    
    Args:
        current_area: Área actual en píxeles cuadrados
        area_history: Lista de tuplas (timestamp_ms, area_px2)
        threshold: Umbral de área a predecir
        snapshot_latency_ms: Latencia del sistema de snapshot
        
    Returns:
        int: Timestamp estimado en ms o None si no es posible estimar
    """
    rate = compute_smoothed_rate_from_history(area_history)
    if rate is None or rate <= 0:
        return None
    time_needed_ms = (threshold - current_area) / rate
    estimated_arrival_ms = int(time.time() * 1000) + time_needed_ms
    return estimated_arrival_ms
