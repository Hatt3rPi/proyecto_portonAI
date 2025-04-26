#!/usr/bin/env python
"""
Sistema PortonAI - Detección y reconocimiento de patentes vehiculares

Este es el punto de entrada principal del sistema PortonAI que
implementa detección de placas y reconocimiento OCR con tracking híbrido
y procesamiento asíncrono de snapshots/OCR para mantener el bucle de frames
lo más ligero posible.
"""

import os
import sys
import time
import logging
import cv2
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# Importaciones de configuración y modelos
from config import (
    DEBUG_MODE, ONLINE_MODE, URL_HD, CONFIANZA_PATENTE,
    FPS_DEQUE_MAXLEN, UMBRAL_SNAPSHOT_AREA,
    NIGHT_THRESHOLD, DAY_THRESHOLD
)
from models import ModelManager

# Utilidades de sistema y preprocesamiento
from utils.suppression import open_stream_with_suppressed_stderr, suppress_c_stderr
from utils.image_processing import (
    resize_for_inference, preprocess_frame,
    load_calibration_params, is_frame_valid,
    calculate_roi_for_coverage
)
from utils.snapshot import SnapshotManager
from utils.ocr import (
    OCRProcessor, apply_consensus_voting,
    consensus_by_positions, final_consensus,
    is_valid_plate
)
from utils.api import send_backend, send_plate_async

# --- Nuevo: tracking híbrido centrado en placas ---
from utils.plate_tracker import PlateTrackerManager

# ---------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------
if DEBUG_MODE:
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s [%(levelname)s] %(message)s")
else:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(message)s")

# Añadir FileHandler para guardar todos los logs en portonai.log
log_file = os.path.join(os.path.dirname(__file__), "portonai.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.getLogger().addHandler(file_handler)

logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("yolov8").setLevel(logging.ERROR)

# Executor global para snapshots y OCR en background
executor = ThreadPoolExecutor(max_workers=4)


def main():
    """
    Función principal del sistema PortonAI.
    Implementa el bucle principal de procesamiento de video, detección,
    tracking y visualización, mientras delega snapshots/OCR a hilos.
    """
    # -------------------
    # 1. INICIALIZACIÓN
    # -------------------
    # 1.1 Carga de modelos
    manager = ModelManager()
    model_plate = manager.get_plate_model()
    model_ocr = manager.get_ocr_model()
    ocr_processor = OCRProcessor(model_ocr, manager.get_ocr_names())

    # 1.2a Prueba de disponibilidad de trackers OpenCV
    logging.info("Disponibilidad de trackers OpenCV:")
    checks = [
        ("CSRT", hasattr(cv2, 'TrackerCSRT_create')),
        ("CSRT (legacy)", hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create')),
        ("KCF", hasattr(cv2, 'TrackerKCF_create')),
        ("KCF (legacy)", hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'))
    ]
    for name, available in checks:
        symbol = "✔" if available else "✖"
        logging.info(f"  {symbol} {name}")

    # 1.2b Tracking híbrido y snapshot
    plate_manager = PlateTrackerManager(
        model_ocr, manager.get_ocr_names(),
        iou_thresh=0.3, max_missed=5, detect_every=5
    )
    snapshot_manager = SnapshotManager()

    # 1.3 Apertura de stream
    source = URL_HD if ONLINE_MODE else "proyecto_portonAI/debug/test_noche.mp4"
    stream_LD = open_stream_with_suppressed_stderr(source)
    if not stream_LD.isOpened():
        logging.error("No se pudo abrir el stream, reintentando en 2s...")
        time.sleep(2)
        stream_LD = open_stream_with_suppressed_stderr(source)
        if not stream_LD.isOpened():
            logging.error("Error fatal: No se pudo abrir el stream tras reintentar")
            sys.exit(1)
    logging.info("Stream abierto exitosamente")

    # 1.4 Lectura primer frame
    with suppress_c_stderr():
        ret, frame_ld = stream_LD.read()
    if not ret or not is_frame_valid(frame_ld):
        logging.error("No se pudo leer el primer frame del stream")
        sys.exit(1)

    stream_h, stream_w = frame_ld.shape[:2]
    is_offline = not ONLINE_MODE
    if is_offline:
        initial_snapshot = frame_ld.copy()
        snap_h, snap_w = initial_snapshot.shape[:2]
        logging.info(f"[OFFLINE] Dimensiones snapshot (HD): {snap_w}x{snap_h}")
    else:
        # 1.5 Snapshot inicial
        snapshot_id = "initial"
        future = snapshot_manager.request_update_future(snapshot_id)
        try:
            recv_id, initial_snapshot = future.result(timeout=5)
            if initial_snapshot is None:
                raise Exception("Snapshot inicial no disponible")
            if recv_id != snapshot_id:
                logging.warning("Snapshot inicial desincronizado")
        except Exception as e:
            logging.error(f"Error obteniendo snapshot inicial: {e}")
            sys.exit(1)
        snap_h, snap_w = initial_snapshot.shape[:2]
        logging.info(f"Dimensiones del snapshot (HD): {snap_w}x{snap_h}")

    # 1.6 Preparación de UI y auxiliares
    if DEBUG_MODE:
        cv2.namedWindow("PortonAI - Tracking Placas", cv2.WINDOW_NORMAL)
    fps_deque = deque(maxlen=FPS_DEQUE_MAXLEN)
    prev_time = time.time()
    calib_params = load_calibration_params()

    # Pending jobs de snapshot/OCR en background
    pending_jobs = set()
    invalid_frame_count = 0  # contador de frames inválidos consecutivos

    # Ruta de debug para snapshots
    debug_dir = "/home/customware/PortonAI/proyecto_portonAI/debug"
    os.makedirs(debug_dir, exist_ok=True)

    # Función para snapshot y OCR avanzado en background
    def schedule_snapshot_and_ocr(plate_id, inst):
        try:
            # 1) Obtener snapshot HD
            if is_offline:
                hd_snap = frame_ld.copy()
            else:
                _, hd_snap = snapshot_manager.request_update_future(plate_id).result(timeout=5)

            # 2) Refinar detección en HD
            hd_resized, sx, sy = resize_for_inference(hd_snap, max_dim=640)
            refined = model_plate.predict(hd_resized, device='cuda:0', verbose=False)
            box = next((b for b in refined[0].boxes
                        if float(b.conf[0]) * 100 >= CONFIANZA_PATENTE), None)
            if not box:
                return
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            ox1, oy1 = int(x1 * sx), int(y1 * sy)
            ox2, oy2 = int(x2 * sx), int(y2 * sy)
            roi = hd_snap[oy1:oy2, ox1:ox2]

            # 3) Intentar OCR multiescala en el ROI
            multiscale = ocr_processor.process_multiscale(roi)
            best = apply_consensus_voting(multiscale, min_length=5)
            if best is not None:
                candidate_text = best.get("ocr_text", "").strip()
                candidate_conf = best.get("average_conf", 0.0)
            else:
                candidate_text = ""
                candidate_conf = 0.0

            # 4) Si multiescala no arroja valor válido, fallback a OpenAI
            if not (candidate_text and is_valid_plate(candidate_text)):
                result = ocr_processor.process_plate_image(roi, use_openai=True)
                text = result.get("ocr_text", "").strip()
                conf = result.get("confidence", 0.0)
            else:
                text = candidate_text
                conf = candidate_conf

            # 5) Validar y actualizar instancia sólo si es válido
            if is_valid_plate(text):
                inst.ocr_text = text
                inst.ocr_status = 'completed'
                inst.ocr_conf = conf
                logging.info(f"Placa detectada y almacenada: {text}")
                # 6) Guardar snapshot en disco para debug
                timestamp_ms = int(time.time() * 1000)
                filename = f"snapshot_{plate_id}_{timestamp_ms}.jpg"
                cv2.imwrite(os.path.join(debug_dir, filename), roi)
                # 7) Envío de resultados (una sola vez por placa)
                executor.submit(send_plate_async, roi, hd_snap, text, "", inst.bbox)
                send_backend(text, roi)

        except Exception as e:
            logging.warning(f"Error snapshot async placa {plate_id}: {e}")
        finally:
            pending_jobs.discard(plate_id)

    # -------------------
    # 2. BUCLE PRINCIPAL
    # -------------------
    while stream_LD.isOpened():
        try:
            # 2.1 Captura de frame
            with suppress_c_stderr():
                ret, frame_ld = stream_LD.read()
            if not ret or not is_frame_valid(frame_ld):
                invalid_frame_count += 1
                logging.warning("Frame inválido detectado, reintentando...")
                if invalid_frame_count >= 5:
                    logging.info("Reconectando stream tras 5 frames inválidos...")
                    stream_LD.release()
                    time.sleep(1)
                    stream_LD = open_stream_with_suppressed_stderr(source)
                    invalid_frame_count = 0
                time.sleep(0.5)
                continue
            invalid_frame_count = 0

            # 2.2 Preprocesamiento para detección
            inference_frame, scale_x, scale_y = resize_for_inference(frame_ld, max_dim=640)
            inference_frame = preprocess_frame(inference_frame, calib_params)

            # 2.3 (Opcional) Modo día/noche
            avg_brightness = np.mean(frame_ld)

            # 2.4 Detección de placas en frame reducido
            all_detections = []
            results = model_plate.predict(inference_frame, device='cuda:0', verbose=False)
            for box in results[0].boxes:
                conf = float(box.conf[0]) * 100
                if conf < CONFIANZA_PATENTE:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                xh = int(x1 * scale_x)
                yh = int(y1 * scale_y)
                x2h = int(x2 * scale_x)
                y2h = int(y2 * scale_y)
                if DEBUG_MODE:
                    cv2.rectangle(frame_ld, (xh, yh), (x2h, y2h), (0,255,0), 2)
                all_detections.append((xh, yh, x2h, y2h))

            # 2.5 Actualizar tracking híbrido
            active_plates = plate_manager.update(frame_ld, all_detections)

            # 2.6 DIBUJAR TODAS LAS CAJAS Y ETIQUETAS
            vis = frame_ld.copy()
            for pid, inst in active_plates.items():
                x1, y1, x2, y2 = inst.bbox
                if inst.ocr_status == 'completed':
                    color = (0,255,0)
                    label = inst.ocr_text
                else:
                    color = (0,0,255)
                    label = pid[:4]
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Mostrar ventana y FPS si estamos en DEBUG
            if DEBUG_MODE:
                now = time.time()
                fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
                prev_time = now
                fps_deque.append(fps)
                fps_avg = sum(fps_deque) / len(fps_deque)
                cv2.putText(vis, f"FPS: {fps:.1f} Avg: {fps_avg:.1f}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
                cv2.imshow("PortonAI - Tracking Placas", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            # 2.7 OCR EN STREAMING SÓLO PARA PENDIENTES
            for pid, inst in active_plates.items():
                if inst.ocr_status != 'pending':
                    continue
                x1, y1, x2, y2 = inst.bbox
                w, h = x2 - x1, y2 - y1
                if w * h < UMBRAL_SNAPSHOT_AREA:
                    continue
                h_ld, w_ld = frame_ld.shape[:2]
                x1 = max(0, min(x1, w_ld-1))
                y1 = max(0, min(y1, h_ld-1))
                x2 = max(0, min(x2, w_ld))
                y2 = max(0, min(y2, h_ld))
                if x2 <= x1 or y2 <= y1:
                    logging.warning(f"ROI inválido o fuera de rango para placa {pid}: {(x1,y1,x2,y2)}")
                    continue   # o return en snapshot
                crop = frame_ld[y1:y2, x1:x2]
                try:
                    multiscale = ocr_processor.process_multiscale(crop)
                    best = apply_consensus_voting(multiscale, min_length=5)
                    if best is None and multiscale:
                        best = max(multiscale, key=lambda r: r["confidence"])
                    text = (best or {}).get("ocr_text","").strip()
                    if text and is_valid_plate(text):
                        inst.ocr_stream = best
                        inst.ocr_text   = text
                        inst.ocr_status = 'completed'
                        logging.info(f"OCR stream válido placa {pid}: '{text}'")
                    else:
                        logging.debug(f"OCR stream inválido placa {pid}: '{text}'")
                except Exception as e:
                    logging.warning(f"OCR stream error placa {pid}: {e}")

            # 2.8 PROGRAMAR SNAPSHOT+OCR ASÍNCRONO SÓLO PARA PENDIENTES
            for pid, inst in active_plates.items():
                if inst.ocr_status != 'pending' or pid in pending_jobs:
                    continue
                x1, y1, x2, y2 = inst.bbox
                w, h = x2 - x1, y2 - y1
                if w * h >= UMBRAL_SNAPSHOT_AREA:
                    pending_jobs.add(pid)
                    executor.submit(schedule_snapshot_and_ocr, pid, inst)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.warning(f"Error en el ciclo principal: {e}")
            time.sleep(0.5)
            continue

    # 3. Limpieza final
    stream_LD.release()
    if DEBUG_MODE:
        cv2.destroyAllWindows()
    snapshot_manager.stop()
    executor.shutdown()


if __name__ == '__main__':
    # Redirigir stderr para evitar logs de FFmpeg
    FNULL = open(os.devnull, 'w')
    while True:
        try:
            main()
            break
        except Exception as e:
            logging.warning(f"Error crítico, reiniciando el sistema: {e}")
            time.sleep(5)
            with suppress_c_stderr():
                os.execl(sys.executable, sys.executable, *sys.argv)
