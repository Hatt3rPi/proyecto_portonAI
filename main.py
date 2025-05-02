#!/usr/bin/env python
"""
Sistema PortonAI - Detección y reconocimiento de patentes vehiculares

Este es el punto de entrada principal del sistema PortonAI que
implementa detección de placas y reconocimiento OCR con tracking híbrido
y procesamiento asíncrono de snapshots/OCR para mantener el bucle de frames
lo más ligero posible.
"""
import argparse
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
    NIGHT_THRESHOLD, DAY_THRESHOLD, OCR_STREAM_ZONE,
    OCR_STREAM_ACTIVATED, OCR_SNAPSHOT_ACTIVATED, OCR_OPENAI_ACTIVATED
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

def is_in_ocr_stream_zone(bbox, frame_shape, zone_def):
    """
    Verifica si un bbox está completamente dentro de la zona de OCR stream.

    Args:
        bbox: (x1, y1, x2, y2) de la detección
        frame_shape: (height, width) del frame
        zone_def: Diccionario con porcentajes de la zona (start_x_pct, end_x_pct, start_y_pct, end_y_pct)

    Returns:
        True si el bbox está completamente dentro de la zona, False en otro caso.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]

    start_x = zone_def["start_x_pct"] * w
    end_x   = zone_def["end_x_pct"] * w
    start_y = zone_def["start_y_pct"] * h
    end_y   = zone_def["end_y_pct"] * h

    return (x1 >= start_x and x2 <= end_x and
            y1 >= start_y and y2 <= end_y)

# Función para snapshot y OCR avanzado en background
def schedule_snapshot_and_ocr(plate_id, inst):
    try:
        # Validar si OCR snapshot está activado
        if not OCR_SNAPSHOT_ACTIVATED:
            logging.debug(f"OCR Snapshot desactivado, omitiendo procesamiento para {plate_id}")
            return

        # Verificar si está en zona OCR 
        inside_zone = is_in_ocr_stream_zone(inst.bbox, frame_ld.shape, OCR_STREAM_ZONE)
        logging.debug(f"[SNAPSHOT-OCR] Instancia {plate_id} bbox={inst.bbox} inside_zone={inside_zone}")
        
        if not inside_zone:
            logging.debug(f"Placa {plate_id} fuera de zona OCR, cancelando procesamiento")
            return

        # 1) Obtener snapshot HD
        if is_offline:
            hd_snap = frame_ld.copy()
        else:
            _, hd_snap = snapshot_manager.request_update_future(plate_id).result(timeout=5)

        # 2) Refinar detección en HD
        hd_resized, sx, sy = resize_for_inference(hd_snap, max_dim=640)
        refined = model_plate.predict(hd_resized, device='cuda:0', verbose=False)
        
        # Usar la caja con mejor confianza que supere el umbral
        boxes = [b for b in refined[0].boxes if float(b.conf[0]) * 100 >= CONFIANZA_PATENTE]
        if not boxes:
            logging.debug(f"No se encontraron placas válidas en el snapshot de {plate_id}")
            return
            
        # Ordenar por confianza descendente
        boxes.sort(key=lambda b: float(b.conf[0]), reverse=True)
        box = boxes[0]  # Tomar la más confiable
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        ox1, oy1 = int(x1 * sx), int(y1 * sy)
        ox2, oy2 = int(x2 * sx), int(y2 * sy)
        
        # Verificar coordenadas
        h_snap, w_snap = hd_snap.shape[:2]
        ox1 = max(0, min(ox1, w_snap-1))
        oy1 = max(0, min(oy1, h_snap-1))
        ox2 = max(0, min(ox2, w_snap))
        oy2 = max(0, min(oy2, h_snap))
        
        if ox2 <= ox1 or oy2 <= oy1:
            logging.warning(f"ROI inválido en snapshot para placa {plate_id}: {(ox1,oy1,ox2,oy2)}")
            return
            
        roi = hd_snap[oy1:oy2, ox1:ox2]
        if roi.size == 0:
            logging.warning(f"ROI vacío para placa {plate_id}")
            return

        # 3) Intentar OCR multiescala en el ROI
        multiscale = ocr_processor.process_multiscale(roi)
        best = apply_consensus_voting(multiscale, min_length=5)
        if best is not None:
            candidate_text = best.get("ocr_text", "").strip()
            candidate_conf = best.get("average_conf", 0.0)
        else:
            candidate_text = ""
            candidate_conf = 0.0

        # 4) Fallback a OpenAI si hace falta y está habilitado
        if OCR_OPENAI_ACTIVATED and not (candidate_text and is_valid_plate(candidate_text)):
            result = ocr_processor.process_plate_image(roi, use_openai=True)
            text = result.get("ocr_text", "").strip()
            conf = result.get("confidence", 0.0)
        else:
            text = candidate_text
            conf = candidate_conf

        # 5) Validar y almacenar -> debe aplicarse para los 3 OCR una vez que tengan un texto válido
        if is_valid_plate(text):
            inst.ocr_text = text
            inst.ocr_status = 'completed'
            inst.ocr_conf = conf
            if inst.detected_at is not None:
                full_time = time.time() - inst.detected_at
                full_time_ms = int(full_time * 1000)
            else:
                full_time_ms = -1  # Por si algo falla, que sepas que es inválido

            logging.info(f"Placa detectada y almacenada: {text}")
            
            # Datos del ROI
            roi_area = (ox2 - ox1) * (oy2 - oy1)
            x_position = ox1

            # Mensaje extendido con tiempo de procesamiento
            print(f"[PLACA] {text} | Área: {roi_area} px² | X inicial: {x_position} | Tiempo detección a OCR: {full_time_ms} ms | Res: {hd_snap.shape[1]}x{hd_snap.shape[0]}")

            # 6) Guardar snapshot en disco para debug
            timestamp_ms = int(time.time() * 1000)
            filename=f"fullframe_{timestamp_ms}_{plate_id}_{text}_ROI.jpg"
            cv2.imwrite(os.path.join(debug_dir, filename), roi)
            # Guardar frame completo en debug
            full_frame_filename = f"fullframe_{timestamp_ms}_{plate_id}_{text}.jpg"
            cv2.imwrite(os.path.join(debug_dir, full_frame_filename), hd_snap)

            # 7) Envío de resultados
            executor.submit(send_plate_async, roi, hd_snap, text, "", inst.bbox)
            send_backend(text, roi)

    except Exception as e:
        logging.warning(f"Error snapshot async placa {plate_id}: {e}")
    finally:
        pending_jobs.discard(plate_id)

def main(video_path=None):
    """
    Función principal del sistema PortonAI.
    Implementa el bucle principal de procesamiento de video, detección,
    tracking y visualización, mientras delega snapshots/OCR a hilos.
    Si se pasa `video_path`, se utilizará ese archivo en lugar de RTSP.
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
        iou_thresh=0.7,  # Aumentado de 0.3 a 0.5 para ser más estricto en la asociación
        max_missed=9,    # Aumentado de 5 a 7 para persistir mejor los trackers existentes
        detect_every=5,
        min_detection_confidence=CONFIANZA_PATENTE * 0.01,  # Convertir porcentaje a decimal
        exclude_footer=True,  # Activar exclusión de footer
        footer_ratio=0.85     # 85% de la altura para comenzar la zona de exclusión
    )
    snapshot_manager = SnapshotManager()

    # 1.3 Apertura de stream o archivo de video
    if video_path:
        source = video_path
        is_offline = True
        logging.info(f"[VIDEO] Modo archivo: {video_path}")
    else:
        source = URL_HD if ONLINE_MODE else os.path.join("proyecto_portonAI", "debug", "test_noche.mp4")
        is_offline = not ONLINE_MODE

    # Abrir la fuente de vídeo (RTSP o archivo local)
    stream_LD = open_stream_with_suppressed_stderr(source)
    if not stream_LD.isOpened():
        logging.error(f"No se pudo abrir la fuente: {source} — reintentando en 2s...")
        time.sleep(2)
        stream_LD = open_stream_with_suppressed_stderr(source)
        if not stream_LD.isOpened():
            logging.error("Error fatal: No se pudo abrir la fuente tras reintentar")
            sys.exit(1)
    logging.info("Fuente de vídeo abierta exitosamente")

    # 1.4 Lectura primer frame
    with suppress_c_stderr():
        ret, frame_ld = stream_LD.read()
    if not ret or not is_frame_valid(frame_ld):
        logging.error("No se pudo leer el primer frame de la fuente")
        sys.exit(1)

    stream_h, stream_w = frame_ld.shape[:2]
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
        logging.info(f"Dimensiones del stream (HD): {stream_w}x{stream_h}")
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
    debug_dir = os.path.join(os.path.dirname(__file__), "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # -------------------
    # 2. BUCLE PRINCIPAL
    # -------------------
    while stream_LD.isOpened():
        try:
            # 2.1 Captura de frame
            with suppress_c_stderr():
                ret, frame_ld = stream_LD.read()
                if not ret:
                    if video_path:
                        # Estamos en modo archivo y el video terminó
                        logging.info("Video finalizado correctamente, cerrando proceso...")
                        break
                    else:
                        # Estamos en RTSP, entonces reintentamos
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
            if not ret or not is_frame_valid(frame_ld):
                invalid_frame_count += 1
                logging.warning("Frame inválido detectado, reintentando...")
                if invalid_frame_count >= 5:
                    logging.info("Reconectando fuente tras 5 frames inválidos...")
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
            
            # Filtrar y ordenar por confianza
            filtered_boxes = []
            for box in results[0].boxes:
                conf = float(box.conf[0]) * 100
                if conf >= CONFIANZA_PATENTE:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    xh = int(x1 * scale_x)
                    yh = int(y1 * scale_y)
                    x2h = int(x2 * scale_x)
                    y2h = int(y2 * scale_y)
                    filtered_boxes.append((xh, yh, x2h, y2h, conf))
            filtered_boxes.sort(key=lambda x: x[4], reverse=True)
            all_detections = [(x[0], x[1], x[2], x[3]) for x in filtered_boxes]

            # Visualización de detecciones brutas
            if DEBUG_MODE and all_detections:
                for i, (xh, yh, x2h, y2h) in enumerate(all_detections):
                    color = (0, 0, 255) if i < 5 else (0, 255, 0)
                    cv2.rectangle(frame_ld, (xh, yh), (x2h, y2h), color, 2)
                    cv2.putText(frame_ld, f"{i+1}", (xh, yh-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 2.5 Actualizar tracking híbrido
            active_plates = plate_manager.update(frame_ld, all_detections)
            # 2.6 Dibujar cajas, etiquetas y zona de exclusión
            vis = frame_ld.copy()
            if DEBUG_MODE:
                h, w = frame_ld.shape[:2]
                footer_y = int(h * plate_manager.footer_ratio)
                cv2.line(vis, (0, footer_y), (w, footer_y), (0, 0, 255), 1)
                cv2.putText(vis, "ZONA DE EXCLUSION", (10, footer_y+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                start_x = int(OCR_STREAM_ZONE["start_x_pct"] * w)
                end_x   = int(OCR_STREAM_ZONE["end_x_pct"] * w)
                start_y = int(OCR_STREAM_ZONE["start_y_pct"] * h)
                end_y   = int(OCR_STREAM_ZONE["end_y_pct"] * h)

                # Línea izquierda
                start_x = int(OCR_STREAM_ZONE["start_x_pct"] * w)
                end_x   = int(OCR_STREAM_ZONE["end_x_pct"] * w)
                start_y = int(OCR_STREAM_ZONE["start_y_pct"] * h)
                end_y   = int(OCR_STREAM_ZONE["end_y_pct"] * h)

                # Línea izquierda
                cv2.line(vis, (start_x, start_y), (start_x, end_y), (255, 255, 0), 2)
                # Línea derecha
                cv2.line(vis, (end_x, start_y), (end_x, end_y), (255, 255, 0), 2)
                # Línea superior
                cv2.line(vis, (start_x, start_y), (end_x, start_y), (255, 255, 0), 2)
                # Línea inferior
                cv2.line(vis, (start_x, end_y), (end_x, end_y), (255, 255, 0), 2)

                # Texto indicador
                cv2.putText(vis, "Zona OCR Stream", (start_x + 5, start_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            for pid, inst in active_plates.items():
                x1, y1, x2, y2 = inst.bbox
                if x2 <= x1 or y2 <= y1:
                    logging.debug(f"[2.6] Ignorando caja inválida para placa {pid}: {(x1,y1,x2,y2)}")
                    continue
                    
                if inst.ocr_status == 'completed':
                    color = (0,255,0)
                    label = inst.ocr_text
                else:
                    if is_in_ocr_stream_zone(inst.bbox, frame_ld.shape, OCR_STREAM_ZONE):
                        color = (255, 255, 0)
                    else:
                        color = (0,0,255)
                    label = pid[:4]  # <<< Esto agrega un label de backup (ej: primeros 4 caracteres del ID)

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


            # Mostrar FPS y ventana en modo DEBUG
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

            # 2.7 OCR en streaming para instancias pendientes
            for pid, inst in active_plates.items():
                # Omitir procesamiento si OCR stream está desactivado
                if not OCR_STREAM_ACTIVATED:
                    continue
                    
                if inst.ocr_status != 'pending':
                    logging.debug(f"[OCR-STREAM] Instancia {pid} omitida, estado={inst.ocr_status}")
                    continue
                
                # Validar que bbox esté completamente dentro de la zona de OCR Stream
                inside_zone = is_in_ocr_stream_zone(inst.bbox, frame_ld.shape, OCR_STREAM_ZONE)
                logging.debug(f"[OCR-STREAM] Instancia {pid} bbox={inst.bbox} inside_zone={inside_zone}")
                
                if not inside_zone:
                    continue

                x1, y1, x2, y2 = inst.bbox
                h_ld, w_ld = frame_ld.shape[:2]
                x1 = max(0, min(x1, w_ld-1))
                y1 = max(0, min(y1, h_ld-1))
                x2 = max(0, min(x2, w_ld))
                y2 = max(0, min(y2, h_ld))
                if x2 <= x1 or y2 <= y1:
                    logging.warning(f"[2.7] ROI inválido para OCR en placa {pid}: {(x1,y1,x2,y2)}")
                    continue
                crop = frame_ld[y1:y2, x1:x2]
                try:
                    logging.debug(f"[OCR-STREAM] Iniciando OCR multiescala para {pid}, ROI={crop.shape}")
                    multiscale = ocr_processor.process_multiscale(crop)
                    best = apply_consensus_voting(multiscale, min_length=5)
                    if best is None and multiscale:
                        best = max(multiscale, key=lambda r: r["confidence"])
                    text = (best or {}).get("ocr_text","").strip()
                    
                    if text:
                        logging.debug(f"[OCR-STREAM] Instancia {pid} resultado OCR: '{text}' válido={is_valid_plate(text)}")
                    else:
                        logging.debug(f"[OCR-STREAM] Instancia {pid} sin texto OCR detectado")
                    
                    if text and is_valid_plate(text):
                        inst.ocr_stream = best
                        inst.ocr_text   = text
                        inst.ocr_status = 'completed'
                        inst.ocr_conf   = best.get("confidence", 0.0)  # Aseguramos guardar la confianza
                        logging.info(f"[2.7] OCR stream válido placa {pid}: '{text}'")
                        
                        # 5) Validar y almacenar para OCR stream (como en snapshot)
                        # - Calcular tiempo de detección a OCR
                        if inst.detected_at is not None:
                            full_time = time.time() - inst.detected_at
                            full_time_ms = int(full_time * 1000)
                        else:
                            full_time_ms = -1
                            
                        # - Datos del ROI
                        roi_area = (x2 - x1) * (y2 - y1)
                        x_position = x1
                        
                        # - Mensaje extendido con tiempo de procesamiento
                        print(f"[PLACA] {text} | Área: {roi_area} px² | X inicial: {x_position} | Tiempo detección a OCR: {full_time_ms} ms | Res: {frame_ld.shape[1]}x{frame_ld.shape[0]}")
                        
                        # 6) Guardar snapshot en disco para debug
                        timestamp_ms = int(time.time() * 1000)
                        # - Guardar ROI
                        filename = f"stream_{timestamp_ms}_{pid}_{text}_ROI.jpg"
                        cv2.imwrite(os.path.join(debug_dir, filename), crop)
                        # - Guardar frame completo
                        full_frame_filename = f"stream_{timestamp_ms}_{pid}_{text}.jpg"
                        cv2.imwrite(os.path.join(debug_dir, full_frame_filename), frame_ld.copy())
                        
                        # 7) Envío de resultados al backend
                        if is_offline:
                            hd_snap = frame_ld.copy()
                            executor.submit(send_plate_async, crop, hd_snap, text, "", inst.bbox)
                            send_backend(text, crop)
                        else:
                            # En modo online intentamos obtener un snapshot HD
                            try:
                                _, hd_snap = snapshot_manager.request_update_future(pid).result(timeout=3)
                                if hd_snap is not None:
                                    executor.submit(send_plate_async, crop, hd_snap, text, "", inst.bbox)
                                    send_backend(text, crop)
                                else:
                                    # Fallback a frame LD
                                    executor.submit(send_plate_async, crop, frame_ld.copy(), text, "", inst.bbox)
                                    send_backend(text, crop)
                            except Exception as e:
                                logging.warning(f"Error obteniendo snapshot para envío: {e}, usando frame LD")
                                # Fallback a frame LD
                                executor.submit(send_plate_async, crop, frame_ld.copy(), text, "", inst.bbox)
                                send_backend(text, crop)
                except Exception as e:
                    logging.warning(f"[2.7] OCR stream error placa {pid}: {e}")

            # 2.8 Programar snapshot+OCR asíncrono para pendientes
            if OCR_SNAPSHOT_ACTIVATED:  # Solo procesar si está habilitado
                for pid, inst in active_plates.items():
                    if inst.ocr_status != 'pending' or pid in pending_jobs:
                        if inst.ocr_status != 'pending':
                            logging.debug(f"[SNAPSHOT] Instancia {pid} omitida, estado={inst.ocr_status}")
                        elif pid in pending_jobs:
                            logging.debug(f"[SNAPSHOT] Instancia {pid} omitida, ya en cola de procesamiento")
                        continue

                    x1, y1, x2, y2 = inst.bbox
                    if x2 <= x1 or y2 <= y1:
                        continue
                    w, h = x2 - x1, y2 - y1

                    # **Verificar**: descartar fuera de zona OCR
                    inside_zone = is_in_ocr_stream_zone((x1, y1, x2, y2), frame_ld.shape, OCR_STREAM_ZONE)
                    logging.debug(f"[SNAPSHOT] Instancia {pid} bbox={(x1,y1,x2,y2)} área={w*h} inside_zone={inside_zone}")
                    
                    if not inside_zone:
                        logging.debug(f"[SNAPSHOT] Omitido snapshot fuera de zona OCR para {pid}: {inst.bbox}")
                        continue

                    if w * h >= UMBRAL_SNAPSHOT_AREA:
                        logging.debug(f"[2.8] Snapshot async solicitado para {pid} con bbox={inst.bbox}")
                        pending_jobs.add(pid)
                        executor.submit(schedule_snapshot_and_ocr, pid, inst)
                    else:
                        logging.debug(f"[SNAPSHOT] Instancia {pid} omitida, área {w*h} < umbral {UMBRAL_SNAPSHOT_AREA}")


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

    # --- Parseo de argumentos ---
    parser = argparse.ArgumentParser(description="PortonAI – detección de patentes")
    parser.add_argument(
        "--video_path",
        type=str,
        help="Ruta a un archivo de video (.dav u otro) para procesar. Si no se indica, usa RTSP."
    )
    args = parser.parse_args()

    # Loop principal con reinicio automático en caso de fallo crítico
    while True:
        try:
            main(video_path=args.video_path)
            break
        except Exception as e:
            logging.warning(f"Error crítico, reiniciando el sistema: {e}")
            time.sleep(5)
            with suppress_c_stderr():
                os.execl(sys.executable, sys.executable, *sys.argv)
