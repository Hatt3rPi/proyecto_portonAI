import time
import uuid
import logging
import cv2
from collections import deque
from .ocr import OCRProcessor

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

def create_tracker():
    """
    Crea un tracker de OpenCV. Intenta CSRT, KCF, MOSSE, MIL y MEDIANFLOW.
    Si no se encuentra ninguno, devuelve un tracker dummy que no trackea.
    """
    tracker_constructors = []
    # CSRT
    if hasattr(cv2, 'TrackerCSRT_create'):
        tracker_constructors.append(('CSRT', cv2.TrackerCSRT_create))
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        tracker_constructors.append(('CSRT_legacy', cv2.legacy.TrackerCSRT_create))
    # KCF
    if hasattr(cv2, 'TrackerKCF_create'):
        tracker_constructors.append(('KCF', cv2.TrackerKCF_create))
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
        tracker_constructors.append(('KCF_legacy', cv2.legacy.TrackerKCF_create))
    # MOSSE
    if hasattr(cv2, 'TrackerMOSSE_create'):
        tracker_constructors.append(('MOSSE', cv2.TrackerMOSSE_create))
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMOSSE_create'):
        tracker_constructors.append(('MOSSE_legacy', cv2.legacy.TrackerMOSSE_create))
    # MIL
    if hasattr(cv2, 'TrackerMIL_create'):
        tracker_constructors.append(('MIL', cv2.TrackerMIL_create))
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMIL_create'):
        tracker_constructors.append(('MIL_legacy', cv2.legacy.TrackerMIL_create))
    # MEDIANFLOW
    if hasattr(cv2, 'TrackerMedianFlow_create'):
        tracker_constructors.append(('MEDIANFLOW', cv2.TrackerMedianFlow_create))
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMedianFlow_create'):
        tracker_constructors.append(('MEDIANFLOW_legacy', cv2.legacy.TrackerMedianFlow_create))

    for name, constructor in tracker_constructors:
        try:
            tracker = constructor()
            logging.info(f"Usando tracker {name}")
            return tracker
        except Exception:
            continue

    # Fallback dummy
    logging.warning("No se encontró ningún tracker de OpenCV, usando tracker dummy (sin seguimiento)")
    class DummyTracker:
        def init(self, frame, bbox):
            pass
        def update(self, frame):
            # siempre falla, lo que provocará re-detección
            return False, None
    return DummyTracker()


class PlateInstance:
    def __init__(self, initial_bbox, frame, model_ocr, ocr_names):
        self.id           = str(uuid.uuid4())
        self.bbox         = tuple(map(int, initial_bbox))  # (x,y,w,h)
        self.tracker      = create_tracker()
        self.tracker.init(frame, self.bbox)
        self.last_seen    = time.time()
        self.missed       = 0
        # buffer de cajas para histórico opcional
        self.history      = deque(maxlen=5)
        # estado de OCR
        self.ocr_status   = 'pending'    # pending / completed / failed
        self.ocr_text     = ''
        # instancia de OCR personalizada
        self.ocr_stream = {}
        self.ocr_processor = OCRProcessor(model_ocr, ocr_names)

    def update_tracker(self, frame):
        ok, new_bbox = self.tracker.update(frame)
        if ok and new_bbox is not None:
            self.bbox      = tuple(map(int, new_bbox))
            self.last_seen = time.time()
            self.missed    = 0
            self.history.append(self.bbox)
        else:
            self.missed   += 1
        return ok

    def run_ocr(self, hd_frame):
        if self.ocr_status != 'pending':
            return
        x, y, w, h = self.bbox
        crop = hd_frame[y:y+h, x:x+w]
        try:
            res = self.ocr_processor.process(crop)
            text = res.get("ocr_text", "").strip()
            # sólo marcar completed si tenemos al menos 5 caracteres
            if len(text) >= 5:
                self.ocr_text   = text
                self.ocr_status = 'completed'
            else:
                # lectura demasiado corta: ignorar y seguir en pending
                return
        except Exception:
            self.ocr_status = 'failed'

class PlateTrackerManager:
    def __init__(self, model_ocr, ocr_names,
                 iou_thresh=0.3, max_missed=5, detect_every=5):
        self.instances    = {}   # id -> PlateInstance
        self.iou_thresh   = iou_thresh
        self.max_missed   = max_missed
        self.detect_every = detect_every
        self.frame_count  = 0
        self.model_ocr    = model_ocr
        self.ocr_names    = ocr_names

    def update(self, hd_frame, detections):
        """
        Actualiza cada instancia de placa:
         - Actualiza trackeurs
         - Re-detección cada N frames y asociación por IoU
         - Limpieza de instancias excesivamente perdidas
        """
        self.frame_count += 1

        # 1) Actualizar trackers existentes
        for pid, inst in list(self.instances.items()):
            inst.update_tracker(hd_frame)
            # Eliminar si se perdió demasiado
            if inst.missed > self.max_missed:
                del self.instances[pid]

        # 2) Asociar detecciones nuevas cada detect_every frames
        if self.frame_count % self.detect_every == 0:
            for det in detections:
                best_id, best_iou = None, 0.0
                for pid, inst in self.instances.items():
                    iou = compute_iou(inst.bbox, det)
                    if iou > best_iou:
                        best_iou, best_id = iou, pid
                if best_iou >= self.iou_thresh:
                    # Re-inicializar tracker con nueva bbox
                    inst = self.instances[best_id]
                    inst.tracker.init(hd_frame, det)
                    inst.bbox   = det
                    inst.missed = 0
                else:
                    # Nueva placa detectada
                    new_inst = PlateInstance(det, hd_frame, self.model_ocr, self.ocr_names)
                    self.instances[new_inst.id] = new_inst

        return self.instances
