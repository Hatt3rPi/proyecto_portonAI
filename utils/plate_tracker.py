## archivo: plate_tracker.py
"""
Módulo para el seguimiento de placas vehiculares usando un enfoque híbrido
que combina IoU para asociación entre frames y trackers de OpenCV para
mantener la identidad de las placas entre frames.
"""

import time
import uuid
import logging
import cv2
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from .ocr import OCRProcessor
import math
from .image_enhancement import ImageEnhancer  # Importamos el nuevo módulo


# ------------------------------------------------------------------
# 📌  Constantes de deduplicación y cool‑down
# ------------------------------------------------------------------
IOU_MERGE: float = 0.45           # IoU mínimo para fusionar detecciones
MAX_CENTROID_DIST: int = 80       # Distancia máxima entre centroides (px)
SEND_COOLDOWN_SEC: int = 15       # Segundos para evitar re‑envío de la misma placa
# ------------------------------------------------------------------


@dataclass
class PlateTrackInstance:
    """Clase que representa una instancia de placa siendo rastreada."""
    tracker_id: str
    bbox: Tuple[int, int, int, int]
    tracker: Optional[object] = None
    ocr_status: str = 'pending'
    ocr_text: str = ''
    ocr_stream: dict = field(default_factory=dict)
    ocr_conf: float = 0.0
    frame_count: int = 0
    missed_count: int = 0
    matched_detections: int = 0
    last_detection: Optional[Tuple[int, int, int, int]] = None
    detected_at: Optional[float] = None
    metadata: dict = field(default_factory=dict)

def calculate_iou(box1, box2):
    """Calcula la intersección sobre unión (IoU) entre dos cajas."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Verificar que las cajas son válidas
    if x2_1 <= x1_1 or y2_1 <= y1_1 or x2_2 <= x1_2 or y2_2 <= y1_2:
        return 0.0
        
    # Calcular coordenadas de la intersección
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Si no hay intersección
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
        
    # Calcular áreas
    area_i = (x2_i - x1_i) * (y2_i - y1_i)
    area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Retornar IoU
    return area_i / (area_1 + area_2 - area_i)

# Función para determinar si un bbox está en el área de exclusión (footer)
def is_in_exclusion_zone(bbox, frame_shape, exclude_ratio=0.85):
    """
    Determina si un bbox está en la zona de exclusión (footer).
    
    Args:
        bbox: Tupla (x1, y1, x2, y2) con coordenadas del bbox
        frame_shape: Tupla (height, width) del frame
        exclude_ratio: Ratio de altura a partir del cual se considera zona de footer
        
    Returns:
        Boolean: True si está en la zona de exclusión, False en caso contrario
    """
    x1, y1, x2, y2 = bbox
    height, width = frame_shape[:2]
    
    # Considera la zona de exclusión como la parte inferior del frame
    footer_start_y = int(height * exclude_ratio)
    
    # Si el centro del bbox está en la zona de footer
    center_y = (y1 + y2) / 2
    
    # Una detección está en la zona de exclusión si su centro está por debajo del umbral
    return center_y > footer_start_y

# ================== START TRACKING UTILS ==================
# Nuevos parámetros de tracking
MAX_MISSED = 10           # frames permitidos sin detección antes de eliminar
MAX_AGE = 200             # frames máximos de vida de un tracker
OVERLAP_THRESHOLD = 0.3   # IoU umbral para considerar solapamiento

def filter_overlapping_by_size(instances, overlap_thresh=OVERLAP_THRESHOLD):
    """
    Dado un listado de PlateTrackInstance, mantiene solo la instancia
    con mayor área en cada grupo que se solape por encima de overlap_thresh.
    """
    # Orden descendente por área (x2-x1)*(y2-y1)
    sorted_insts = sorted(
        instances,
        key=lambda inst: (inst.bbox[2]-inst.bbox[0])*(inst.bbox[3]-inst.bbox[1]),
        reverse=True
    )
    filtered = []
    for inst in sorted_insts:
        if not any(
            calculate_iou(inst.bbox, kept.bbox) > overlap_thresh
            for kept in filtered
        ):
            filtered.append(inst)
    return filtered
# =================== END TRACKING UTILS ===================
def _bbox_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

class PlateInstance:
    """
    Representa una instancia de placa vehicular detectada y seguida.
    Mantiene el estado de detección, tracking y OCR.
    """
    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float):
        self.id = str(uuid.uuid4())[:8]  # ID único para esta instancia
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence  # Confianza de la detección
        self.detected_at = time.time()  # Timestamp de detección inicial
        self.last_seen = self.detected_at  # Última vez que se vio
        self.tracker = None  # Objeto tracker de OpenCV (será inicializado después)
        self.ocr_text = ""  # Texto OCR
        self.ocr_status = 'pending'  # Estado: pending, processing, completed, failed
        self.ocr_conf = 0.0  # Confianza del OCR
        self.ocr_stream = None  # Resultado OCR stream (dict completo)
        self.matched_detections = 1  # Contador de detecciones emparejadas
        self.ocr_stream_results = []  # Lista de resultados OCR stream para consenso
        self.has_been_in_stream_zone = False  # Indica si alguna vez estuvo en zona de stream


class PlateTrackerManager:
    """
    Gestor de tracking para placas vehiculares.
    Implementa un sistema híbrido que combina detección YOLO con trackers OpenCV.
    """
    
    def __init__(self, 
                 model_ocr=None, 
                 names=None,
                 iou_thresh=0.3,
                 max_missed=5,
                 detect_every=3,
                 min_detection_confidence=0.7,
                 exclude_footer=False,
                 footer_ratio=0.85):
        """
        Inicializa el gestor de tracking híbrido.
        
        Args:
            model_ocr: Modelo OCR (YOLO) para reconocimiento de caracteres
            names: Lista de nombres de clases del modelo OCR
            iou_thresh: Umbral IOU para asociar detecciones con trackers
            max_missed: Máximo número de frames para mantener un tracker sin detecciones
            detect_every: Frecuencia de detección (cada N frames)
            min_detection_confidence: Confianza mínima para considerar una detección válida
            exclude_footer: Si es True, ignora detecciones en la parte inferior del frame
            footer_ratio: Porcentaje vertical donde comienza la zona de exclusión (0.85 = 85%)
        """
        self.active_trackers = {}  # Cambiamos a active_plates para consistencia con el método update
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.detect_every = detect_every
        self.frame_count = 0
        self.min_detection_confidence = min_detection_confidence
        self.exclude_footer = exclude_footer
        self.footer_ratio = footer_ratio
        
        # Inicializar image enhancer con los parámetros correctos
        try:
            from utils.image_enhancement import ImageEnhancer
            self.image_enhancer = ImageEnhancer()  # Sin parámetros adicionales
        except ImportError:
            logging.warning("No se pudo importar ImageEnhancer, funcionando sin mejoramiento de imagen")
            self.image_enhancer = None
            
        # Modelo OCR (opcional, usado para mejorar tracking)
        self.model_ocr = model_ocr
        self.names = names

    def update(self, frame, detections):
        """
        Actualiza los trackers con nuevas detecciones.
        
        Args:
            frame: Frame actual para actualizar los trackers
            detections: Lista de detecciones [(x1,y1,x2,y2), ...]
            
        Returns:
            Diccionario de instancias de PlateTrackInstance activas
        """
        # Incrementar contador de frames
        self.frame_count += 1
        
        # Filtrar detecciones en zona de exclusión si está activado
        if self.exclude_footer:
            h, w = frame.shape[:2]
            footer_y = int(h * self.footer_ratio)
            filtered_dets = []
            for det in detections:
                x1, y1, x2, y2 = det
                # Si la parte inferior del bbox está por debajo de footer_y, excluir
                if y2 <= footer_y:
                    filtered_dets.append(det)
            detections = filtered_dets
        
        # Actualizar trackers existentes con el nuevo frame
        for track_id, instance in list(self.active_trackers.items()):
            # Actualizar tracker con el nuevo frame
            if instance.tracker is not None:
                success, new_bbox = instance.tracker.update(frame)
                if success:
                    # Convertir nuevo bbox (x,y,w,h) a (x1,y1,x2,y2)
                    x, y, w, h = [int(v) for v in new_bbox]
                    instance.bbox = (x, y, x+w, y+h)
                    instance.missed = 0
                else:
                    instance.missed += 1
            else:
                # Sin tracker, solo incrementar contador de pérdida
                instance.missed += 1
                
            # Eliminar tracker si se ha perdido demasiadas veces
            if instance.missed > self.max_missed:
                del self.active_trackers[track_id]
                continue
                
            # Intentar crear un nuevo tracker si no tiene
            if instance.tracker is None and instance.missed < self.max_missed // 2:
                x1, y1, x2, y2 = instance.bbox
                tracker = cv2.legacy.TrackerKCF_create()
                success = tracker.init(frame, (x1, y1, x2-x1, y2-y1))
                if success:
                    instance.tracker = tracker
                        
        # Si es momento de detectar (según detect_every), asociar nuevas detecciones
        if self.frame_count % self.detect_every == 0 and detections:
            self._associate_detections(frame, detections)
                
        # Retornar diccionario de instancias activas
        return self.active_trackers  # Devolvemos active_trackers en lugar de active_instances

    @staticmethod
    def _create_tracker() -> Any:
        """
        Crea un nuevo objeto tracker de OpenCV basado en la disponibilidad.
        Intentará usar los trackers en orden de preferencia.
        """
        # Intentar crear tracker en orden de preferencia
        if hasattr(cv2, 'TrackerCSRT_create'):
            return cv2.TrackerCSRT_create()
        elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            return cv2.legacy.TrackerCSRT_create()
        elif hasattr(cv2, 'TrackerKCF_create'):
            return cv2.TrackerKCF_create()
        elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
            return cv2.legacy.TrackerKCF_create()
        else:
            logging.error("No se encontraron trackers compatibles en OpenCV")
            return None
    
    @staticmethod
    def _iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        """
        Calcula la intersección sobre unión (IoU) entre dos cajas.
        
        Args:
            boxA, boxB: Cajas en formato (x1, y1, x2, y2)
        
        Returns:
            IoU como valor entre 0 y 1.
        """
        # Determinar coordenadas de la intersección
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        # Calcular área de intersección
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        
        # Calcular áreas de ambas cajas
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        # Calcular IoU
        iou = inter_area / float(boxA_area + boxB_area - inter_area)
        
        return iou
    
    def _is_in_footer(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> bool:
        """
        Determina si una detección está en la zona de exclusión inferior (footer).
        
        Args:
            bbox: Caja de la detección (x1, y1, x2, y2)
            frame_shape: Dimensiones del frame (height, width)
        
        Returns:
            True si la detección está mayormente en el footer, False en caso contrario.
        """
        if not self.exclude_footer:
            return False
            
        # Extraer dimensiones
        x1, y1, x2, y2 = bbox
        height, _ = frame_shape[:2]
        
        # Calcular línea de footer
        footer_y = int(height * self.footer_ratio)
        
        # Si el centro de la caja está por debajo de la línea de footer
        box_center_y = (y1 + y2) / 2
        return box_center_y > footer_y
    
    def _associate_detections(self, frame, detections):
        """
        Asocia nuevas detecciones a los trackers existentes o crea nuevos trackers si es necesario.
        
        Args:
            frame: Frame actual con las detecciones
            detections: Lista de detecciones [(x1,y1,x2,y2), ...]
        """
        # Aquí va la lógica para asociar detecciones a trackers existentes
        pass  # Reemplazar con la implementación real
