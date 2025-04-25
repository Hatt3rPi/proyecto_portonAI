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
    source = URL_HD if ONLINE_MODE else "scripts/monitoreo_patentes/video.mp4"
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
    logging.info(f"Dimensiones del stream (LD): {stream_w}x{stream_h}")

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

    # Función para snapshot y OCR avanzado en background
    def schedule_snapshot_and_ocr(plate_id, inst):
        try:
            # 1) Obtener snapshot HD
            _, hd_snap = snapshot_manager.request_update_future(plate_id).result(timeout=5)

            # 2) Refinar detección en HD
            hd_resized, sx, sy = resize_for_inference(hd_snap, max_dim=640)
            refined = model_plate.predict(hd_resized, device='cuda:0', verbose=False)
            # Tomar la primera caja válida
            box = next((b for b in refined[0].boxes
                        if float(b.conf[0]) * 100 >= CONFIANZA_PATENTE), None)
            if not box:
                return
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            ox1, oy1 = int(x1 * sx), int(y1 * sy)
            ox2, oy2 = int(x2 * sx), int(y2 * sy)
            roi = hd_snap[oy1:oy2, ox1:ox2]

            # 3) OCR con OpenAI
            result = ocr_processor.process_plate_image(roi, use_openai=True)
            text = result.get("ocr_text", "").strip()
            conf = result.get("confidence", 0.0)

            # 4) Validar y actualizar instancia
            if is_valid_plate(text):
                inst.ocr_text = text
                inst.ocr_status = 'completed'
                inst.ocr_conf = conf
                # 5) Envío de resultados
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
                logging.warning("Frame inválido detectado, reintentando...")
                time.sleep(0.5)
                continue

            # 2.2 Preprocesamiento para detección
            inference_frame, scale_x, scale_y = resize_for_inference(frame_ld, max_dim=640)
            inference_frame = preprocess_frame(inference_frame, calib_params)

            # 2.3 (Opcional) Modo día/noche
            avg_brightness = np.mean(frame_ld)
            # (puedes conservar tu lógica de cambio de modo aquí)

            # 2.4 Detección de placas en frame reducido
            all_detections = []
            ocr_candidates = []
            results = model_plate.predict(inference_frame, device='cuda:0', verbose=False)
            for box in results[0].boxes:
                conf = float(box.conf[0]) * 100
                if conf < CONFIANZA_PATENTE:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # escalar a HD
                xh = int(x1 * scale_x)
                yh = int(y1 * scale_y)
                w = int((x2 - x1) * scale_x)
                h = int((y2 - y1) * scale_y)

                # debug: dibujar todas las detecciones
                if DEBUG_MODE:
                    cv2.rectangle(frame_ld, (xh, yh), (xh + w, yh + h), (0,255,0), 2)

                # Tracking para todas las cajas
                all_detections.append((xh, yh, w, h))

                # Solo candidatas para OCR
                if w * h >= 1000:
                    ocr_candidates.append((xh, yh, w, h))

            # 2.5 Actualizar tracking híbrido
            active_plates = plate_manager.update(frame_ld, all_detections)

            # 2.6 OCR en streaming (ligero)
            for pid, inst in active_plates.items():
                if inst.ocr_status == 'pending':
                    x, y, w, h = inst.bbox
                    if w * h >= UMBRAL_SNAPSHOT_AREA:
                        crop = frame_ld[y:y+h, x:x+w]
                        try:
                            multiscale = ocr_processor.process_multiscale(crop)
                            best = apply_consensus_voting(multiscale, min_length=5)
                            if best is None and multiscale:
                                best = max(multiscale, key=lambda r: r["confidence"])
                            text = (best or {}).get("ocr_text","").strip()
                            if text and is_valid_plate(text):
                                inst.ocr_stream = best
                                inst.ocr_text = text
                                inst.ocr_status = 'completed'
                                logging.info(f"OCR stream válido placa {pid}: {text}")
                            else:
                                logging.debug(f"OCR stream inválido placa {pid}: {text}")
                            logging.debug(f"OCR stream placa {pid}: {inst.ocr_stream.get('ocr_text','')}")
                        except Exception as e:
                            inst.ocr_status = 'failed'
                            logging.warning(f"OCR stream error placa {pid}: {e}")

            # 2.7 Programar snapshot+OCR asíncrono para pendientes
            for pid, inst in active_plates.items():
                x, y, w, h = inst.bbox
                if (inst.ocr_status == 'pending'
                        and w * h >= UMBRAL_SNAPSHOT_AREA
                        and pid not in pending_jobs):
                    pending_jobs.add(pid)
                    executor.submit(schedule_snapshot_and_ocr, pid, inst)

            # 2.8 Visualización y FPS
            vis = frame_ld.copy()
            for pid, inst in active_plates.items():
                x, y, w, h = inst.bbox
                color = (0,255,0) if inst.ocr_status == 'completed' else (0,0,255)
                display = inst.ocr_text if inst.ocr_status == 'completed' else pid[:4]
                cv2.rectangle(vis, (x,y), (x+w,y+h), color, 2)
                cv2.putText(vis, display, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Mostrar FPS
            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now
            fps_deque.append(fps)
            fps_avg = sum(fps_deque) / len(fps_deque)

            if DEBUG_MODE:
                cv2.putText(vis, f"FPS: {fps:.1f} Avg: {fps_avg:.1f}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.imshow("PortonAI - Tracking Placas", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

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
