#!/usr/bin/env python
"""
Sistema PortonAI - Detección y reconocimiento de patentes vehiculares

Este es el punto de entrada principal del sistema PortonAI que
implementa detección de placas y reconocimiento OCR con tracking híbrido.
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
from utils.ocr import OCRProcessor, apply_consensus_voting, consensus_by_positions, final_consensus
from utils.api import send_backend, send_plate_async

# --- Nuevo: tracking híbrido centrado en placas ---
from utils.plate_tracker import PlateTrackerManager

# ---------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------
if DEBUG_MODE:
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
else:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("yolov8").setLevel(logging.ERROR)


def main():
    """
    Función principal del sistema PortonAI.
    Implementa el bucle principal de procesamiento de video y detección.
    """
    # -------------------
    # 1. INICIALIZACIÓN
    # -------------------
    # 1.1 Carga de modelos
    manager = ModelManager()
    model_plate   = manager.get_plate_model()
    model_ocr     = manager.get_ocr_model()
    ocr_processor = OCRProcessor(model_ocr, manager.get_ocr_names())

    # 1.2 Tracking híbrido y snapshot
    plate_manager    = PlateTrackerManager(
        model_ocr, manager.get_ocr_names(),
        iou_thresh=0.3, max_missed=5, detect_every=5
    )
    snapshot_manager = SnapshotManager()

    # 1.3 Apertura de stream
    source = URL_HD if ONLINE_MODE else "scripts/monitoreo_patentes/video.mp4"
    cap_hd = open_stream_with_suppressed_stderr(source)
    if not cap_hd.isOpened():
        logging.error("No se pudo abrir el stream, reintentando en 2s...")
        time.sleep(2)
        cap_hd = open_stream_with_suppressed_stderr(source)
        if not cap_hd.isOpened():
            logging.error("Error fatal: No se pudo abrir el stream tras reintentar")
            sys.exit(1)
    logging.info("Stream abierto exitosamente")

    # 1.4 Lectura primer frame
    with suppress_c_stderr():
        ret, frame_hd = cap_hd.read()
    if not ret or not is_frame_valid(frame_hd):
        logging.error("No se pudo leer el primer frame del stream")
        sys.exit(1)
    stream_h, stream_w = frame_hd.shape[:2]
    logging.info(f"Dimensiones del stream (LD): {stream_w}x{stream_h}")

    # 1.5 Snapshot inicial (mismo que antes)
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
    executor   = ThreadPoolExecutor(max_workers=2)
    fps_deque  = deque(maxlen=FPS_DEQUE_MAXLEN)
    prev_time  = time.time()
    calib_params = load_calibration_params()

    # -------------------
    # 2. BUCLE PRINCIPAL
    # -------------------
    while cap_hd.isOpened():
        try:
            # 2.1 Captura de frame
            with suppress_c_stderr():
                ret, frame_hd = cap_hd.read()
            if not ret or not is_frame_valid(frame_hd):
                logging.warning("Frame inválido detectado, reintentando...")
                time.sleep(0.5)
                continue

            # 2.2 Preprocesamiento para detección
            inference_frame, scale_x, scale_y = resize_for_inference(frame_hd, max_dim=640)
            inference_frame = preprocess_frame(inference_frame, calib_params)

            # 2.3 (Opcional) Modo día/noche—puedes conservar o eliminar según prefieras
            avg_brightness = np.mean(frame_hd)
            # if avg_brightness < NIGHT_THRESHOLD: ...

            # 2.4 Detección de placas en frame reducido
            detections = []
            results = model_plate.predict(inference_frame, device='cuda:0', verbose=False)
            for box in results[0].boxes:
                conf = float(box.conf[0]) * 100
                if conf < CONFIANZA_PATENTE:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # escalar a HD
                xh = int(x1 * scale_x)
                yh = int(y1 * scale_y)
                w  = int((x2 - x1) * scale_x)
                h  = int((y2 - y1) * scale_y)
                detections.append((xh, yh, w, h))
                if DEBUG_MODE:
                    cv2.rectangle(frame_hd, (xh, yh), (xh+w, yh+h), (0,255,0), 2)

            # 2.5 Actualizar tracking híbrido
            active_plates = plate_manager.update(frame_hd, detections)

            # 2.6 Ejecutar OCR de stream y encolar snapshot
            plates_for_ocr = []
            for pid, inst in active_plates.items():
                # OCR en streaming, si aún pendiente
                if inst.ocr_status == 'pending' and inst.bbox[2]*inst.bbox[3] >= UMBRAL_SNAPSHOT_AREA:
                    # procesa multiescala directo sobre el recorte
                    x, y, w, h = inst.bbox
                    crop = frame_hd[y:y+h, x:x+w]
                    try:
                        multiscale = ocr_processor.process_multiscale(crop)
                        best = apply_consensus_voting(multiscale, min_length=5)
                        if best is None and multiscale:
                            best = max(multiscale, key=lambda r: r["global_conf"])
                        inst.ocr_stream = best or {"ocr_text":"", "average_conf":0.0}
                        inst.ocr_status = 'completed' if inst.ocr_stream.get("ocr_text","") else 'failed'
                        logging.debug(f"OCR stream placa {pid}: {inst.ocr_stream['ocr_text']}")
                    except Exception as e:
                        inst.ocr_status = 'failed'
                        logging.warning(f"OCR stream error placa {pid}: {e}")

                # Si aún pendiente tras stream, encolar para snapshot
                if inst.ocr_status == 'pending':
                    plates_for_ocr.append(pid)

            # 2.7 Procesar snapshot HD + OCR avanzado
            for pid in plates_for_ocr:
                inst = active_plates[pid]
                try:
                    # 2.7.1 Obtener snapshot HD
                    future = snapshot_manager.request_update_future(pid)
                    _, hd_snap = future.result(timeout=5)
                    if hd_snap is None:
                        raise Exception("Snapshot no disponible")

                    # 2.7.2 Refinar detección de placa en snapshot
                    hd_resized, snap_sx, snap_sy = resize_for_inference(hd_snap, max_dim=640)
                    refined = model_plate.predict(hd_resized, device='cuda:0', verbose=False)
                    refined_box = None
                    for box in refined[0].boxes:
                        if float(box.conf[0])*100 >= CONFIANZA_PATENTE:
                            rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])
                            refined_box = (rx1, ry1, rx2, ry2)
                            break
                    if refined_box is None:
                        raise Exception("No se detectó patente en snapshot HD")

                    # 2.7.3 Extraer ROI HD
                    ox1 = int(refined_box[0]*snap_sx)
                    oy1 = int(refined_box[1]*snap_sy)
                    ox2 = int(refined_box[2]*snap_sx)
                    oy2 = int(refined_box[3]*snap_sy)
                    roi_hd = hd_snap[oy1:oy2, ox1:ox2]

                    # 2.7.4 OCR multiescala sobre snapshot
                    multi = []
                    for pct in range(5, 101, 5):
                        roi = calculate_roi_for_coverage(
                            hd_snap,
                            ((ox1+ox2)/2, (oy1+oy2)/2),
                            (ox2-ox1)*(oy2-oy1),
                            pct/100
                        )
                        if roi is None:
                            continue
                        out = model_ocr.predict(roi, device='cuda:0', verbose=False)
                        res = ocr_processor.process_ocr_result_detailed(out)
                        res["coverage"] = pct
                        multi.append(res)

                    # 2.7.5 Incluir resultado stream si existe
                    stream_res = inst.ocr_stream or {}
                    if stream_res.get("ocr_text",""):
                        multi.append({
                            "ocr_text": stream_res["ocr_text"],
                            "global_conf": stream_res.get("average_conf",0),
                            "predictions": [],
                            "coverage": 100
                        })

                    # 2.7.6 Consensos y resultado final
                    c1 = apply_consensus_voting(multi, min_length=5) or {"ocr_text":"","average_conf":0}
                    c2 = consensus_by_positions(multi, expected_length=len(c1["ocr_text"]) or 6) or {"ocr_text":"","average_conf":0}
                    openai_cons = {"ocr_text":"", "average_conf":0}
                    final = final_consensus([stream_res, c1, c2, openai_cons])

                    # 2.7.7 Actualizar estado y enviar
                    inst.ocr_text     = final["ocr_text"]
                    inst.ocr_status   = 'completed' if final["ocr_text"] else 'failed'
                    inst.ocr_conf     = final["average_conf"]

                    if inst.ocr_status == 'completed':
                        x, y, w, h = inst.bbox
                        crop = frame_hd[y:y+h, x:x+w]
                        executor.submit(send_plate_async, crop, frame_hd, final["ocr_text"], "", inst.bbox)
                        send_backend(final["ocr_text"], crop)
                        logging.info(f"Patente detectada {pid}: {final['ocr_text']}")

                except Exception as e:
                    logging.warning(f"Error OCR snapshot placa {pid}: {e}")

            # 2.8 Visualización y FPS
            vis = frame_hd.copy()
            for pid, inst in active_plates.items():
                x, y, w, h = inst.bbox
                color = (0,255,0) if inst.ocr_status=='completed' else (0,0,255)
                cv2.rectangle(vis, (x,y), (x+w,y+h), color, 2)
                cv2.putText(vis, pid[:4], (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            now = time.time()
            fps = 1.0/(now-prev_time) if now>prev_time else 0.0
            prev_time = now
            fps_deque.append(fps)
            fps_avg = sum(fps_deque)/len(fps_deque)

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
    cap_hd.release()
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
