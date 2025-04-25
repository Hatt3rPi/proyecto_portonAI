import time, uuid
import cv2
from collections import deque
from .ocr import OCRProcessor
from .common import compute_iou  # tu función IoU

def create_tracker():
    # (copia tu implementación de create_tracker aquí)
    ...

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
        # instancia de tu OCR
        self.ocr_processor = OCRProcessor(model_ocr, ocr_names)

    def update_tracker(self, frame):
        ok, new_bbox = self.tracker.update(frame)
        if ok:
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
            res = self.ocr_processor.process(crop)   # asume un método .process()
            text = res.get("ocr_text", "").strip()
            if text:
                self.ocr_text   = text
                self.ocr_status = 'completed'
            else:
                self.ocr_status = 'failed'
        except Exception:
            self.ocr_status = 'failed'

class PlateTrackerManager:
    def __init__(self, model_ocr, ocr_names,
                 iou_thresh=0.3, max_missed=5, detect_every=5):
        self.instances   = {}   # id -> PlateInstance
        self.iou_thresh  = iou_thresh
        self.max_missed  = max_missed
        self.detect_every= detect_every
        self.frame_count = 0
        self.model_ocr   = model_ocr
        self.ocr_names   = ocr_names

    def update(self, hd_frame, detections):
        """
        - hd_frame: frame original en alta resolución
        - detections: lista de bboxes [(x,y,w,h),...] detectadas por YOLO
        Devuelve el dict de instancias activas.
        """
        self.frame_count += 1

        # 1) Primero, actualizar todos los trackers
        for pid, inst in list(self.instances.items()):
            inst.update_tracker(hd_frame)
            # si se perdió demasiado, eliminar
            if inst.missed > self.max_missed:
                del self.instances[pid]

        # 2) Asociar detecciones nuevas _sólo_ cada detect_every frames
        if self.frame_count % self.detect_every == 0:
            for det in detections:
                best_id, best_iou = None, 0
                for pid, inst in self.instances.items():
                    iou = compute_iou(inst.bbox, det)
                    if iou > best_iou:
                        best_iou, best_id = iou, pid
                if best_iou >= self.iou_thresh:
                    # re–inicializar tracker con la caja nueva
                    self.instances[best_id].tracker.init(hd_frame, det)
                    self.instances[best_id].bbox = det
                    self.instances[best_id].missed = 0
                else:
                    # nueva placa
                    inst = PlateInstance(det, hd_frame, self.model_ocr, self.ocr_names)
                    self.instances[inst.id] = inst

        return self.instances
