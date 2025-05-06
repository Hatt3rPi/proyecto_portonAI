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
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from .ocr import OCRProcessor
import math          # ← AGREGA



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

class PlateTrackerManager:
    """
    Administrador de trackers para placas vehiculares usando un enfoque híbrido.
    """
    
    def __init__(self, model_ocr=None, ocr_names=None, iou_thresh=0.5,
                 max_missed=7, detect_every=5, min_detection_confidence=0.7,
                 exclude_footer=True, footer_ratio=0.85,
                 dedup_iou_thresh=IOU_MERGE):
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
        self.experimental_mode = False  # Habilitar experimentación con deskewing
        self.deskew_success_rate = 0
        self.deskew_attempt_count = 0
        self.last_send_times = {}
        
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
    
    # ------------------------------------------------------------------
# ▸  DUPLICADOS y COOL-DOWN
# ------------------------------------------------------------------
    def _is_duplicate_detection(self, detection):
        """
        True si la detección se solapa (IoU ≥ IOU_MERGE) o el
        centro está a < MAX_CENTROID_DIST px de un tracker activo.
        """
        for inst in self.active_instances.values():
            if calculate_iou(detection, inst.bbox) >= IOU_MERGE:
                return True
            cx_d, cy_d = _bbox_centroid(detection)
            cx_i, cy_i = _bbox_centroid(inst.bbox)
            if math.hypot(cx_d - cx_i, cy_d - cy_i) < MAX_CENTROID_DIST:
                return True
        return False


    def _on_valid_ocr(self, instance):
        """
        Registra la hora del último envío y devuelve
        False si aún está dentro del cool-down.
        """
        plate = instance.ocr_text.strip().upper()
        if not plate:
            return False
        now = time.time()
        last = self.last_send_times.get(plate, 0)
        if now - last < SEND_COOLDOWN_SEC:
            logging.debug(f"[COOLDOWN] {plate} en cool-down…")
            return False
        self.last_send_times[plate] = now
        return True


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
    
    def _extract_and_deskew_plate(self, frame, bbox):
        """
        Extrae el ROI de la placa y realiza una corrección de perspectiva si es posible.
        
        Args:
            frame: Frame completo
            bbox: Tupla (x1, y1, x2, y2) con las coordenadas del ROI
            
        Returns:
            Tuple con (ROI_original, ROI_deskewed, success)
            donde success indica si la corrección tuvo éxito
        """
        x1, y1, x2, y2 = bbox
        
        # Añadir padding de 20px a cada lado, respetando los límites de la imagen
        padding = 20
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # 1. Extraer ROI original
        ROI_original = frame[y1:y2, x1:x2].copy()
        h_orig, w_orig = ROI_original.shape[:2]
        
        # Si el ROI es demasiado pequeño, no intentar deskew
        if h_orig < 20 or w_orig < 20:
            return ROI_original, ROI_original, False
            
        try:
            # 2. Detectar bordes y contornos
            gray = cv2.cvtColor(ROI_original, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Dilatar bordes para conexión de líneas
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return ROI_original, ROI_original, False
                
            # Encontrar el contorno más grande
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Si el contorno es muy pequeño, ignorar
            if cv2.contourArea(largest_contour) < (h_orig * w_orig * 0.2):
                return ROI_original, ROI_original, False
            
            # 3. Aproximar contorno a un polígono
            epsilon = 0.04 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Buscar un contorno con aproximadamente 4 puntos (rectangular)
            if len(approx) < 4 or len(approx) > 6:
                return ROI_original, ROI_original, False
                
            # Si son más de 4 puntos, quedarse con los 4 más importantes
            if len(approx) > 4:
                # Ordenar vértices por área de contribución (distancia a los bordes)
                def vertex_importance(point):
                    px, py = point[0]
                    # Calcular qué tan cerca está de cada borde
                    dist_to_edges = min(px, py, w_orig - px, h_orig - py)
                    return dist_to_edges
                
                # Ordenar por importancia (más alejados de los bordes son más importantes)
                sorted_vertices = sorted(approx, key=vertex_importance, reverse=True)
                approx = sorted_vertices[:4]
            
            # Asegurarse de que los puntos estén en orden horario
            approx = np.array(sorted(approx, key=lambda p: (p[0][0], p[0][1])))
            
            # Puntos ordenados: superior-izquierda, superior-derecha, inferior-derecha, inferior-izquierda
            pts_src = np.float32([p[0] for p in approx])
            
            # 4. Definir puntos de destino (rectángulo con proporción 2:1)
            # Calcular el ancho y alto destino manteniendo la proporción 2:1 de las placas
            w_dst = 200
            h_dst = 100
            pts_dst = np.float32([[0, 0], [w_dst, 0], [w_dst, h_dst], [0, h_dst]])
            
            # Calcular la matriz de transformación de perspectiva
            H = cv2.getPerspectiveTransform(pts_src, pts_dst)
            
            # Aplicar la transformación
            ROI_deskewed = cv2.warpPerspective(ROI_original, H, (w_dst, h_dst))
            
            return ROI_original, ROI_deskewed, True
            
        except Exception as e:
            logging.warning(f"Error en deskewing de placa: {e}")
            return ROI_original, ROI_original, False

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
        for i, det in enumerate(detections):
            if self._is_valid_detection(det, frame_shape):
                valid_detections.append(det)
            else:
                logging.debug(f"[TRACKER] Detección {i} descartada: {det}")
        
        logging.debug(f"[TRACKER] Frame {self.frame_count}: {len(valid_detections)}/{len(detections)} detecciones válidas")
        
        # Reemplazamos detections por las válidas
        detections = valid_detections

        # 2) Dedupe temprana: eliminar detecciones con IoU > dedup_iou_thresh
        unique_dets = []
        for det in detections:
            # si alguna en unique_dets está solapada en exceso, la descartamos
            if not any(calculate_iou(det, ud) > self.dedup_iou_thresh
                        for ud in unique_dets):
                unique_dets.append(det)
        detections = unique_dets

        # 3. Actualizar trackers existentes
        active_before = len(self.active_instances)
        for track_id, instance in list(self.active_instances.items()):
            old_bbox = instance.bbox
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
                        
                        # Loguear cambio de posición significativo
                        if old_bbox != instance.bbox:
                            dx = x1 - old_bbox[0]
                            dy = y1 - old_bbox[1]
                            #logging.debug(f"[TRACKER] {track_id} movimiento: {old_bbox} -> {instance.bbox} (dx={dx}, dy={dy})")
                    else:
                        success = False
                        #logging.debug(f"[TRACKER] {track_id} tracker fuera de límites o en exclusión: {(x1, y1, x2, y2)}")
                
                if not success:
                    instance.missed_count += 1
                    #logging.debug(f"[TRACKER] {track_id} missed={instance.missed_count}/{self.max_missed}")

        # 3.5 Filtrar solapamientos: mantener solo ROI de mayor tamaño
        filtered = filter_overlapping_by_size(list(self.active_instances.values()))
        self.active_instances = {inst.tracker_id: inst for inst in filtered}

        # 3.6 Eliminar trackers por edad
        for t_id in list(self.active_instances.keys()):
            if self.active_instances[t_id].frame_count > MAX_AGE:
                del self.active_instances[t_id]

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
            if i in matched_detections:
                continue

            # ❌ Omitir si ya existe tracker muy cercano / solapado
            if self._is_duplicate_detection(detection):
                logging.debug(f"[DEDUP] Detección duplicada omitida: {detection}")
                continue

            track_id = str(uuid.uuid4())                  # ←  AHORA sí creamos el ID
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
        
        # EXPERIMENTAL: Procesamiento con deskew para mejorar OCR
        if self.experimental_mode:
            for track_id, inst in list(self.active_instances.items()):
                # Solo procesar instancias pendientes que no hayan sido procesadas antes
                if inst.ocr_status != 'pending' or 'deskew_processed' in inst.metadata:
                    continue
                
                # Validar que el bbox tenga coordenadas válidas
                x1, y1, x2, y2 = inst.bbox
                if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                    continue
                
                # 1. Extraer ROI original y aplicar deskewing
                ROI_original, ROI_deskewed, deskew_success = self._extract_and_deskew_plate(frame, inst.bbox)
                
                self.deskew_attempt_count += 1
                if deskew_success:
                    self.deskew_success_rate += 1
                    
                # Calcular tasa de éxito cada 50 intentos
                if self.deskew_attempt_count % 50 == 0:
                    success_pct = (self.deskew_success_rate / self.deskew_attempt_count) * 100
                    logging.info(f"[EXPERIMENTAL] Tasa de éxito deskew: {success_pct:.1f}% ({self.deskew_success_rate}/{self.deskew_attempt_count})")
                
                # 2. Procesar OCR en ambas versiones
                try:
                    # Marcar como procesado incluso si falla, para no reintentarlo
                    inst.metadata['deskew_processed'] = True
                    
                    if hasattr(self, 'model_ocr') and self.model_ocr is not None and hasattr(self, 'ocr_names') and self.ocr_names is not None:
                        ocr_processor = OCRProcessor(self.model_ocr, self.ocr_names)
                        
                        # 2.1 OCR en ROI original
                        multiscale_orig = ocr_processor.process_multiscale(ROI_original)
                        best_orig = None
                        texto_orig = ""
                        conf_orig = 0.0
                        
                        if multiscale_orig:
                            from .ocr import apply_consensus_voting, is_valid_plate
                            best_orig = apply_consensus_voting(multiscale_orig, min_length=5)
                            if best_orig:
                                texto_orig = best_orig.get("ocr_text", "").strip()
                                conf_orig = best_orig.get("confidence", 0.0)
                                
                        # 2.2 OCR en ROI deskewed (solo si deskew fue exitoso)
                        texto_deskew = ""
                        conf_deskew = 0.0
                        best_deskew = None
                        
                        if deskew_success:
                            multiscale_deskew = ocr_processor.process_multiscale(ROI_deskewed)
                            if multiscale_deskew:
                                best_deskew = apply_consensus_voting(multiscale_deskew, min_length=5)
                                if best_deskew:
                                    texto_deskew = best_deskew.get("ocr_text", "").strip()
                                    conf_deskew = best_deskew.get("confidence", 0.0)
                        
                        # 3. Comparar resultados y escoger el mejor
                        if deskew_success and conf_deskew > conf_orig and is_valid_plate(texto_deskew):
                            mejor_texto = texto_deskew
                            mejor_conf = conf_deskew
                            metodo_elegido = 'deskew'
                        elif is_valid_plate(texto_orig):
                            mejor_texto = texto_orig
                            mejor_conf = conf_orig
                            metodo_elegido = 'orig'
                        else:
                            # No se encontró texto válido
                            continue
                        
                        # 4. Marcar como completado con metadatos
                        inst.ocr_text = mejor_texto
                        inst.ocr_conf = mejor_conf
                        inst.ocr_status = 'completed_experimental'
                        inst.metadata.update({
                            'orig': {'texto': texto_orig, 'conf': conf_orig},
                            'deskew': {'texto': texto_deskew, 'conf': conf_deskew, 'success': deskew_success},
                            'chosen': metodo_elegido,
                            'experiment': 'deskew_vs_multiscale'
                        })
                        
                        # Guardar snapshots para análisis si hay texto reconocido
                        if mejor_texto:
                            try:
                                import os
                                import time as time_module  # Import time with an alias to avoid conflict
                                debug_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "debug")
                                timestamp_ms = int(time_module.time() * 1000)  # Use the imported module
                                
                                # Guardar original
                                orig_path = os.path.join(debug_dir, f"exp_{timestamp_ms}_{track_id}_orig_{texto_orig}.jpg")
                                cv2.imwrite(orig_path, ROI_original)
                                
                                # Guardar deskewed si tuvo éxito
                                if deskew_success:
                                    deskew_path = os.path.join(debug_dir, f"exp_{timestamp_ms}_{track_id}_deskew_{texto_deskew}.jpg")
                                    cv2.imwrite(deskew_path, ROI_deskewed)
                                
                                # Log para QA
                                logging.info(f"[EXPERIMENTAL] Placa {track_id}: orig='{texto_orig}' ({conf_orig:.1f}), deskew='{texto_deskew}' ({conf_deskew:.1f}), elegido='{mejor_texto}' ({metodo_elegido})")
                                
                            except Exception as e:
                                logging.warning(f"Error guardando imágenes experimentales: {e}")
                                
                except Exception as e:
                    logging.warning(f"Error en procesamiento experimental de OCR: {e}")
        
        # 6. Eliminar trackers con demasiados frames perdidos
        for track_id in list(self.active_instances.keys()):
            if self.active_instances[track_id].missed_count > self.max_missed:
                # Antes de eliminar, guardar estadísticas si fue un experimento
                inst = self.active_instances[track_id]
                if inst.ocr_status == 'completed_experimental':
                    chosen = inst.metadata.get('chosen', '')
                    logging.info(f"[EXPERIMENTAL-FINAL] Track {track_id}: método elegido={chosen}, texto={inst.ocr_text}")
                    
                if self.active_instances[track_id].ocr_status == 'completed':
                    logging.debug(f"Eliminando tracker {track_id} con OCR completado: {self.active_instances[track_id].ocr_text}")
                del self.active_instances[track_id]
        
        # 7. Deduplicar placas si ya tenemos OCR completo y coinciden en texto
        self._deduplicate_by_ocr_text()
        
        # 8. Filtrar instancias con texto OCR inválido (como "CUSTOMWARE")
        self._filter_invalid_ocr_text()
        for inst in self.active_instances.values():
            if inst.ocr_status == 'completed' and not inst.metadata.get('cooldown_checked'):
                inst.metadata['cooldown_allowed'] = self._on_valid_ocr(inst)
                inst.metadata['cooldown_checked'] = True

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
                logging.debug(f"[DEDUP] Encontrados {len(track_ids)} trackers con texto '{ocr_text}': {track_ids}")
                
                # Ordenar por número de detecciones y luego por confianza
                sorted_ids = sorted(track_ids, 
                                   key=lambda tid: (
                                       self.active_instances[tid].matched_detections,
                                       self.active_instances[tid].ocr_conf
                                   ), reverse=True)
                
                # Mantener el mejor, eliminar los demás
                best_id = sorted_ids[0]
                for track_id in sorted_ids[1:]:
                    logging.debug(f"[DEDUP] Eliminando {track_id}, manteniendo {best_id} para placa '{ocr_text}'")
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
