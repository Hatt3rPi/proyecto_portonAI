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
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from .ocr import OCRProcessor

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

class PlateTrackerManager:
    """
    Administrador de trackers para placas vehiculares usando un enfoque híbrido.
    """
    
    def __init__(self, model_ocr=None, ocr_names=None, iou_thresh=0.5,
                 max_missed=7, detect_every=5, min_detection_confidence=0.7,
                 exclude_footer=True, footer_ratio=0.85,
                 dedup_iou_thresh=0.9):
        """
        Inicializa el administrador de trackers.
        
        Args:
            model_ocr: Modelo OCR para reconocer texto (opcional)
            ocr_names: Nombres/clases para OCR (opcional)
            iou_thresh: Umbral de IoU para asociación
            max_missed: Máximo de frames perdidos antes de eliminar un tracker
            detect_every: Cada cuántos frames realizar detección completa
            min_detection_confidence: Confianza mínima para considerar una detección
            exclude_footer: Si se debe excluir la zona del footer (CUSTOMWARE)
            footer_ratio: Proporción de la altura del frame donde comienza el footer
        """
        self.active_instances = {}
        self.iou_thresh = iou_thresh
        self.dedup_iou_thresh = dedup_iou_thresh
        self.max_missed = max_missed
        self.detect_every = detect_every
        self.frame_count = 0
        self.model_ocr = model_ocr
        self.ocr_names = ocr_names
        self.min_detection_confidence = min_detection_confidence
        self.exclude_footer = exclude_footer
        self.footer_ratio = footer_ratio
        self.known_invalid_words = ['CUSTOMWARE', 'CUSTOM', 'WARE', 'CUSTOMW']
        
        # Verificar disponibilidad de trackers
        self.tracker_type = None
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            self.tracker_type = 'CSRT_LEGACY'
        elif hasattr(cv2, 'TrackerCSRT_create'):
            self.tracker_type = 'CSRT'
        elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
            self.tracker_type = 'KCF_LEGACY'
        elif hasattr(cv2, 'TrackerKCF_create'):
            self.tracker_type = 'KCF'
        
        if self.tracker_type:
            logging.info(f"Usando tracker: {self.tracker_type}")
        else:
            logging.warning("No se encontraron trackers compatibles, usando solo IoU")
    
    def _create_tracker(self):
        """Crea un tracker según la disponibilidad."""
        if self.tracker_type == 'CSRT_LEGACY':
            return cv2.legacy.TrackerCSRT_create()
        elif self.tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        elif self.tracker_type == 'KCF_LEGACY':
            return cv2.legacy.TrackerKCF_create()
        elif self.tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        return None
    
    def _is_valid_detection(self, detection, frame_shape):
        """
        Verifica si una detección es válida según diferentes criterios.
        
        Args:
            detection: Tupla (x1, y1, x2, y2) con coordenadas de la detección
            frame_shape: Forma del frame (height, width)
            
        Returns:
            Boolean: True si la detección es válida, False en caso contrario
        """
        x1, y1, x2, y2 = detection
        
        # 1. Verificar coordenadas válidas
        if x2 <= x1 or y2 <= y1:
            return False
            
        # 2. Verificar tamaño mínimo (evitar detecciones muy pequeñas)
        width, height = x2 - x1, y2 - y1
        min_size = 20
        if width < min_size or height < min_size:
            return False
            
        # 3. Verificar proporción de aspecto típica de placas
        # Las placas chilenas suelen tener una proporción aprox. de 2:1 (ancho:alto)
        aspect_ratio = width / height if height > 0 else 0
        if not (1.5 < aspect_ratio < 5.0):  # Margen amplio para contemplar ángulos
            return False
            
        # 4. Verificar si está en zona de exclusión (footer)
        if self.exclude_footer and is_in_exclusion_zone(detection, frame_shape, self.footer_ratio):
            return False
            
        return True
    
    def update(self, frame, detections):
        """
        Actualiza el tracking basado en el frame actual y las nuevas detecciones.
        Implementa un proceso híbrido de tracking + detección para mantener identidad.
        
        Args:
            frame: Frame actual
            detections: Lista de cajas detectadas (x1, y1, x2, y2)
            
        Returns:
            Diccionario con instancias activas
        """
        self.frame_count += 1
        frame_shape = frame.shape
        
        
        # 1) Filtrar detecciones inválidas (footer, tamaño, aspecto…)
        valid_detections = []
        for det in detections:
            if self._is_valid_detection(det, frame_shape):
                valid_detections.append(det)
        # Reemplazamos detections por las válidas
        detections = valid_detections

        # 2) Dedupe temprana: eliminar detecciones con IoU > dedup_iou_thresh
        unique_dets = []
        for det in detections:
            # si alguna en unique_dets está solapada en exceso, la descartamos
            if not any(calculate_iou(det, ud) > getattr(self, 'dedup_iou_thresh', 0.9)
                       for ud in unique_dets):
                unique_dets.append(det)
        detections = unique_dets

        
        # 3. Actualizar trackers existentes
        for track_id, instance in list(self.active_instances.items()):
            if instance.tracker:
                success, box = instance.tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in box]
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    
                    # Validar coordenadas del tracker y que no esté en la zona de exclusión
                    if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1 and not (
                        self.exclude_footer and is_in_exclusion_zone((x1, y1, x2, y2), frame_shape, self.footer_ratio)
                    ):
                        instance.bbox = (x1, y1, x2, y2)
                        instance.frame_count += 1
                    else:
                        success = False
                
                if not success:
                    instance.missed_count += 1
        
        # 4. Asociación IoU entre detecciones y trackers existentes
        matched_tracks = set()
        matched_detections = set()
        matches = []  # Lista de (track_id, detection_idx, iou)
        
        # 4.1 Calcular todas las coincidencias posibles por IoU
        for track_id, instance in self.active_instances.items():
            for i, detection in enumerate(detections):
                if i in matched_detections:
                    continue
                    
                iou = calculate_iou(instance.bbox, detection)
                if iou >= self.iou_thresh:
                    matches.append((track_id, i, iou))
        
        # 4.2 Ordenar coincidencias por IoU descendente
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # 4.3 Asignar coincidencias de mayor a menor IoU
        for track_id, detection_idx, iou in matches:
            if track_id not in matched_tracks and detection_idx not in matched_detections:
                matched_tracks.add(track_id)
                matched_detections.add(detection_idx)
                
                # Actualizar instancia con la nueva detección
                instance = self.active_instances[track_id]
                instance.bbox = detections[detection_idx]
                instance.missed_count = 0
                instance.matched_detections += 1
                instance.last_detection = detections[detection_idx]
                
                # Inicializar o actualizar el tracker si es necesario
                if self.tracker_type and (not instance.tracker or instance.missed_count > 0):
                    x1, y1, x2, y2 = instance.bbox
                    instance.tracker = self._create_tracker()
                    if instance.tracker:
                        try:
                            instance.tracker.init(frame, (x1, y1, x2-x1, y2-y1))
                        except Exception as e:
                            logging.warning(f"Error al inicializar tracker: {e}")
                            instance.tracker = None
        
        # 5. Crear nuevas instancias para detecciones no asociadas
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                track_id = str(uuid.uuid4())
                x1, y1, x2, y2 = detection
                
                # Validar que la detección es válida
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                new_instance = PlateTrackInstance(
                    tracker_id=track_id,
                    bbox=detection,
                    ocr_status='pending',
                    last_detection=detection,
                    detected_at=time.time()  
                )
                
                # Inicializar tracker
                if self.tracker_type:
                    new_instance.tracker = self._create_tracker()
                    if new_instance.tracker:
                        try:
                            new_instance.tracker.init(frame, (x1, y1, x2-x1, y2-y1))
                        except Exception as e:
                            logging.warning(f"Error al inicializar nuevo tracker: {e}")
                            new_instance.tracker = None
                
                self.active_instances[track_id] = new_instance
        
        # 6. Eliminar trackers con demasiados frames perdidos
        for track_id in list(self.active_instances.keys()):
            if self.active_instances[track_id].missed_count > self.max_missed:
                if self.active_instances[track_id].ocr_status == 'completed':
                    logging.debug(f"Eliminando tracker {track_id} con OCR completado: {self.active_instances[track_id].ocr_text}")
                del self.active_instances[track_id]
        
        # 7. Deduplicar placas si ya tenemos OCR completo y coinciden en texto
        self._deduplicate_by_ocr_text()
        
        # 8. Filtrar instancias con texto OCR inválido (como "CUSTOMWARE")
        self._filter_invalid_ocr_text()
        
        return self.active_instances
    
    def _deduplicate_by_ocr_text(self):
        """Deduplica instancias basándose en el texto OCR si está disponible."""
        # Agrupar por texto OCR
        ocr_groups = {}
        for track_id, instance in self.active_instances.items():
            if instance.ocr_status == 'completed' and instance.ocr_text:
                ocr_text = instance.ocr_text.strip()
                if ocr_text not in ocr_groups:
                    ocr_groups[ocr_text] = []
                ocr_groups[ocr_text].append(track_id)
        
        # Para cada grupo, mantener solo la instancia con más detecciones o mayor confianza
        for ocr_text, track_ids in ocr_groups.items():
            if len(track_ids) > 1:
                # Ordenar por número de detecciones y luego por confianza
                sorted_ids = sorted(track_ids, 
                                   key=lambda tid: (
                                       self.active_instances[tid].matched_detections,
                                       self.active_instances[tid].ocr_conf
                                   ), reverse=True)
                
                # Mantener el mejor, eliminar los demás
                best_id = sorted_ids[0]
                for track_id in sorted_ids[1:]:
                    logging.debug(f"Deduplicando: eliminando {track_id}, manteniendo {best_id} para placa '{ocr_text}'")
                    del self.active_instances[track_id]
    
    def _filter_invalid_ocr_text(self):
        """Elimina instancias con texto OCR que coincida con palabras conocidas inválidas."""
        for track_id in list(self.active_instances.keys()):
            instance = self.active_instances[track_id]
            if instance.ocr_status == 'completed' and instance.ocr_text:
                ocr_text = instance.ocr_text.upper()
                # Verificar si el texto OCR contiene alguna palabra inválida conocida
                if any(invalid_word in ocr_text for invalid_word in self.known_invalid_words):
                    logging.debug(f"Eliminando tracking con texto OCR inválido: '{ocr_text}'")
                    del self.active_instances[track_id]
                    continue
                
                # Verificar si el texto OCR tiene patrones típicos de placas
                # En Chile las placas tienen formatos como "LLLL99" o "LL9999"
                if not (len(ocr_text) >= 5 and any(c.isdigit() for c in ocr_text) and any(c.isalpha() for c in ocr_text)):
                    logging.debug(f"Eliminando tracking con formato no válido de placa: '{ocr_text}'")
                    del self.active_instances[track_id]
