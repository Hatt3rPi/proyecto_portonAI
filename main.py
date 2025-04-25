#!/usr/bin/env python
"""
Sistema PortonAI - Detección y reconocimiento de patentes vehiculares

Este es el punto de entrada principal del sistema PortonAI que
implementa detección de vehículos, patentes y reconocimiento OCR.
"""

import cv2
import numpy as np
import time
import logging
import sys
import os
import uuid
from collections import deque
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Importaciones de módulos propios
from config import (
    DEBUG_MODE, ONLINE_MODE, URL_HD, CONFIANZA_PATENTE,
    FPS_DEQUE_MAXLEN, DISPLAY_DURATION, UMBRAL_SNAPSHOT_AREA, 
    FAST_AREA_RATE_THRESHOLD, VEHICLE_MEMORY_TIME, NIGHT_THRESHOLD, DAY_THRESHOLD
)
from models import ModelManager
from utils.suppression import open_stream_with_suppressed_stderr, suppress_c_stderr
from utils.image_processing import (
    resize_for_inference, preprocess_frame, correct_plate_orientation,
    process_image, load_calibration_params, save_calibration_params,
    is_frame_valid, calculate_roi_for_coverage
)
from utils.tracking import (
    compute_iou, find_vehicle_type_for_plate, 
    update_plate_area_history, compute_smoothed_rate_from_history,
    predict_time_to_threshold
)
from utils.snapshot import SnapshotManager, fetch_hd_snapshot
from utils.ocr import OCRProcessor
from utils.api import send_backend, send_plate_async
from utils.ocr import apply_consensus_voting, consensus_by_positions, final_consensus


def main():
    """
    Función principal del sistema PortonAI.
    Implementa el bucle principal de procesamiento de video y detección.
    """
    # 1 INICIALIZACIÓN
    # 1.1 CARGA DE MODELOS Y CONFIGURACIÓN
    model_manager = ModelManager()
    modelo_OBJETOSy = model_manager.get_object_model()
    modelo_PATENTESy = model_manager.get_plate_model()
    modelo_OCRy = model_manager.get_ocr_model()
    ocr_processor = OCRProcessor()
    
    snapshot_manager = SnapshotManager()
    logging.debug("Intentando abrir stream desde: %s", URL_HD)
    URL_SOURCE = URL_HD if ONLINE_MODE else "scripts/monitoreo_patentes/video.mp4"

    # 1.2 APERTURA DE STREAM Y SNAPSHOT INICIAL
    processed_vehicles = {}
    cap_hd = open_stream_with_suppressed_stderr(URL_SOURCE)
    if not cap_hd.isOpened():
        logging.error("No se pudo abrir el stream, reintentando en 2s...")
        time.sleep(2)
        cap_hd = open_stream_with_suppressed_stderr(URL_SOURCE)
        if not cap_hd.isOpened():
            logging.error("Error fatal: No se pudo abrir el stream tras reintentar")
            sys.exit(1)
    logging.info("Stream abierto exitosamente")

    # 1.2.1 LECTURA PRIMER FRAME
    with suppress_c_stderr():
        ret_hd, frame_hd = cap_hd.read()

    if not ret_hd or not is_frame_valid(frame_hd):
        logging.error("No se pudo leer el primer frame del stream")
        sys.exit(1)
    stream_h, stream_w = frame_hd.shape[:2]
    logging.info(f"Dimensiones del stream (LD): {stream_w}x{stream_h}")
    
    # Obtener snapshot inicial
    snapshot_id = "initial"
    future = snapshot_manager.request_update_future(snapshot_id)
    try:
        recv_id, initial_hd_snapshot = future.result(timeout=5)
        if initial_hd_snapshot is None:
            raise Exception("Snapshot inicial no disponible")
        if recv_id != snapshot_id:
            logging.warning("Snapshot inicial desincronizado")
    except Exception as e:
        logging.error(f"Error obteniendo snapshot inicial: {e}")
        sys.exit(1)

    snap_h, snap_w = initial_hd_snapshot.shape[:2]
    logging.info(f"Dimensiones del snapshot (HD): {snap_w}x{snap_h}")

    inference_frame, scale_x, scale_y = resize_for_inference(frame_hd, max_dim=640)

    # 1.3 PREPARACIÓN DE LA UI
    if DEBUG_MODE:
        cv2.namedWindow("Stream HD con Detección de Patentes", cv2.WINDOW_NORMAL)

    # 1.4 INICIALIZACIÓN DE VARIABLES AUXILIARES
    executor = ThreadPoolExecutor(max_workers=1)
    calib_params = load_calibration_params()
    fps_deque = deque(maxlen=FPS_DEQUE_MAXLEN)
    prev_time = time.time()

    best_plates = {}
    current_mode = "day"

    # 2 BUCLE PRINCIPAL
    while cap_hd.isOpened():
        try:
            # 2.1 CAPTURA Y PREPROCESAMIENTO
            with suppress_c_stderr():
                ret_hd, frame_hd = cap_hd.read()
            if not ret_hd or not is_frame_valid(frame_hd):
                logging.warning("Frame inválido detectado, reintentando...")
                time.sleep(0.5)
                continue

            inference_frame, scale_x, scale_y = resize_for_inference(frame_hd, max_dim=640)
            inference_frame = preprocess_frame(inference_frame, calib_params)

            # 2.2 CHEQUEO MODO DÍA/NOCHE
            avg_brightness = np.mean(frame_hd)
            if current_mode == "day" and avg_brightness < NIGHT_THRESHOLD:
                current_mode = "night"
                logging.info("Cambio a modo NOCHE")
            elif current_mode == "night" and avg_brightness > DAY_THRESHOLD:
                current_mode = "day"
                logging.info("Cambio a modo DÍA")

            # 2.3 DETECCIÓN DE VEHÍCULOS
            if current_mode == "day":
                vehiculo_results = modelo_OBJETOSy.predict(inference_frame, device='cuda:0', verbose=False)
                vehicle_boxes = []
                for box in vehiculo_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    class_name = modelo_OBJETOSy.names[class_id]
                    if class_name in ["car", "motorcycle", "bus", "truck"]:
                        x1_hd = int(x1 * scale_x)
                        y1_hd = int(y1 * scale_y)
                        x2_hd = int(x2 * scale_x)
                        y2_hd = int(y2 * scale_y)
                        if DEBUG_MODE:
                            cv2.rectangle(frame_hd, (x1_hd, y1_hd), (x2_hd, y2_hd), (255, 0, 0), 2)
                            cv2.putText(frame_hd, class_name, (x1_hd, y1_hd - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        vehicle_boxes.append({
                            "class": class_name,
                            "bbox": (x1_hd, y1_hd, x2_hd, y2_hd)
                        })
            else:
                vehicle_boxes = []

            # 2.4 LIMPIEZA DE VEHÍCULOS PROCESADOS
            curr_vehicle_bboxes = [data["bbox"] for data in vehicle_boxes]
            new_processed = {}
            for key, data in processed_vehicles.items():
                for vbox in curr_vehicle_bboxes:
                    if compute_iou(data["bbox"], vbox) > 0.3:
                        new_processed[key] = data
                        break
            processed_vehicles = new_processed

            # 2.5 DETECCIÓN DE PATENTES EN FRAME LD
            results = modelo_PATENTESy.predict(inference_frame, device='cuda:0', verbose=False)
            display_frame = frame_hd.copy()
            current_plates = set()
            plates_for_ocr = []
            
            for box in results[0].boxes:
                if box.conf * 100 < CONFIANZA_PATENTE:
                    continue

                BORDER_MARGIN = 20
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x1 < BORDER_MARGIN or y1 < BORDER_MARGIN or 
                    x2 > (inference_frame.shape[1] - BORDER_MARGIN) or y2 > (inference_frame.shape[0] - BORDER_MARGIN)):
                    continue

                inference_area = (x2 - x1) * (y2 - y1)
                current_bbox = (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))

                # Verificar si este vehículo ya fue procesado exitosamente
                vehicle_already_processed = False
                for hash_id, data in processed_vehicles.items():
                    old_bbox = data["bbox"]
                    iou = compute_iou(current_bbox, old_bbox)
                    if iou > 0.3:
                        vehicle_already_processed = True
                        if DEBUG_MODE:
                            cv2.rectangle(display_frame, 
                                         (current_bbox[0], current_bbox[1]), 
                                         (current_bbox[2], current_bbox[3]),
                                         (0, 255, 255), 2)
                            cv2.putText(display_frame, "PROCESSED", 
                                       (current_bbox[0], current_bbox[1] - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        break

                if vehicle_already_processed:
                    logging.debug(f"Vehículo ya procesado, se omite: área={inference_area}px²")
                    continue

                IOU_THRESHOLD = 0.4
                matched_plate_id = None
                for pid, data in best_plates.items():
                    if compute_iou(current_bbox, data["coords"]) > IOU_THRESHOLD:
                        matched_plate_id = pid
                        break
                if matched_plate_id is None:
                    matched_plate_id = str(uuid.uuid4())
                plate_id = matched_plate_id
                current_plates.add(plate_id)

                # 2.5.1 PROCESAMIENTO DE CADA PLACA
                try:
                    corrected_plate = correct_plate_orientation(frame_hd[current_bbox[1]:current_bbox[3], current_bbox[0]:current_bbox[2]])
                except Exception as e:
                    logging.warning(f"Error al corregir orientación: {e}")
                    corrected_plate = frame_hd[current_bbox[1]:current_bbox[3], current_bbox[0]:current_bbox[2]]

                if plate_id in best_plates:
                    best_plates[plate_id]["coords"] = current_bbox
                    best_plates[plate_id]["last_seen"] = time.time()
                    best_plates[plate_id]["tipo_vehiculo"] = find_vehicle_type_for_plate(current_bbox, vehicle_boxes)
                    best_plates[plate_id]["inference_area"] = inference_area
                    update_plate_area_history(best_plates[plate_id], inference_area)
                    if not best_plates[plate_id].get("locked", False):
                        best_plates[plate_id]["frame_count"] += 1
                        best_plates[plate_id]["image"] = frame_hd[current_bbox[1]:current_bbox[3], current_bbox[0]:current_bbox[2]]
                        best_plates[plate_id]["corrected_plate"] = corrected_plate
                else:
                    best_plates[plate_id] = {
                        "id": plate_id,
                        "text": "",
                        "confidence": 0,
                        "image": frame_hd[current_bbox[1]:current_bbox[3], current_bbox[0]:current_bbox[2]],
                        "corrected_plate": corrected_plate,
                        "coords": current_bbox,
                        "inference_area": inference_area,
                        "chars": [],
                        "last_seen": time.time(),
                        "model_conf": float(box.conf),
                        "locked": False,
                        "frame_count": 1,
                        "tipo_vehiculo": find_vehicle_type_for_plate(current_bbox, vehicle_boxes)
                    }
                    best_plates[plate_id]["area_history"] = deque(maxlen=5)
                    best_plates[plate_id]["area_history"].append((int(time.time()*1000), inference_area))

                # Cálculo de tiempo estimado para que el área alcance un umbral
                predicted_arrival = None
                if best_plates[plate_id]["inference_area"] < UMBRAL_SNAPSHOT_AREA:
                    predicted_arrival = predict_time_to_threshold(
                        best_plates[plate_id]["inference_area"],
                        best_plates[plate_id].get("area_history")
                    )
                if predicted_arrival is not None:
                    target_request_time = predicted_arrival - 500
                    current_time_ms = int(time.time() * 1000)
                    wait_time_ms = target_request_time - current_time_ms
                    if wait_time_ms <= 0:
                        logging.debug(f"Área de patente:{best_plates[plate_id]['inference_area']} px²; Snapshot solicitado de forma anticipada")
                        snapshot_manager.request_update_future(plate_id)
                    if best_plates[plate_id]['inference_area'] >= 1000:
                        logging.debug(f"Área de patente:{best_plates[plate_id]['inference_area']} px²; se estima llegada en {wait_time_ms}ms.")
                else:
                    if best_plates[plate_id]['inference_area'] >= 1000:
                        logging.debug(f"Área de patente:{best_plates[plate_id]['inference_area']} px²; se omite procesamiento OCR.")

                # 2.5.2 OCR STREAM MULTIESCALA
                if best_plates[plate_id]["inference_area"] >= UMBRAL_SNAPSHOT_AREA and not best_plates[plate_id].get("ocr_stream_processed", False):
                    ocr_stream_results_multiscale = ocr_processor.process_multiscale(corrected_plate)
                    stream_consensus = apply_consensus_voting(ocr_stream_results_multiscale, min_length=5)
                    if stream_consensus is None and ocr_stream_results_multiscale:
                        stream_consensus = max(ocr_stream_results_multiscale, key=lambda r: r["global_conf"])
                    elif stream_consensus is None:
                        stream_consensus = {"ocr_text": "", "average_conf": 0.0, "count": 0}
                    logging.debug(f"OCR stream multiescala: consenso obtenido: {stream_consensus['ocr_text']}")
                    best_plates[plate_id]["ocr_stream"] = stream_consensus
                    best_plates[plate_id]["ocr_stream_processed"] = True

                if (not best_plates[plate_id].get("locked", False)):
                    if best_plates[plate_id].get("inference_area", 0) >= UMBRAL_SNAPSHOT_AREA:
                        best_plates[plate_id]['frame_buffer'] = frame_hd.copy()
                        plates_for_ocr.append({
                            "plate_id": plate_id,
                            "coords": current_bbox,
                            "model_conf": best_plates[plate_id]["model_conf"]
                        })

                if DEBUG_MODE:
                    cv2.rectangle(display_frame, (current_bbox[0], current_bbox[1]), (current_bbox[2], current_bbox[3]), 
                                  (0, 255, 0) if inference_area >= UMBRAL_SNAPSHOT_AREA else (0, 255, 255), 2)
                    confidence_text = f"{float(box.conf)*100:.1f}%"
                    cv2.putText(display_frame, confidence_text, (current_bbox[0], current_bbox[1] - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            # 2.6 SNAPSHOT HD Y OCR AVANZADO
            if plates_for_ocr:
                try:
                    current_snapshot_id = plates_for_ocr[0]["plate_id"]
                    rate = compute_smoothed_rate_from_history(best_plates[current_snapshot_id].get("area_history", []))
                    fast_vehicle = (rate is not None and rate > FAST_AREA_RATE_THRESHOLD)

                    # Proceso basado en velocidad del vehículo
                    if fast_vehicle:
                        hd_snapshot = best_plates[current_snapshot_id].get('frame_buffer')
                        if hd_snapshot is None:
                            # Usar OpenAI para análisis de la placa
                            plate_img = best_plates[current_snapshot_id]['corrected_plate']
                            result = ocr_processor.process_plate_image(plate_img, use_openai=True)
                            best_plates[current_snapshot_id]['openai_result'] = result["ocr_text"] or ""
                            
                            if result["ocr_text"]:
                                final_result = {"ocr_text": result["ocr_text"], "average_conf": 100.0}
                            else:
                                final_result = {"ocr_text": "", "average_conf": 0.0}
                                raise Exception("OCR no pudo extraer matrícula")
                    else:
                        # Usar snapshot HD para análisis más detallado
                        snapshot_future = snapshot_manager.request_update_future(current_snapshot_id)
                        _, hd_snapshot = snapshot_future.result(timeout=5)
                        if hd_snapshot is None:
                            raise Exception("Snapshot no disponible")
                            
                    # Guardar snapshot para debug (si es necesario)
                    debug_dir = "debug"
                    if not os.path.exists(debug_dir):
                        os.makedirs(debug_dir)
                    tstamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    snapshot_filename = os.path.join(debug_dir, f"snapshot_{tstamp}.jpg")
                    cv2.imwrite(snapshot_filename, hd_snapshot)
                    logging.debug(f"Snapshot {snapshot_filename} - Paso 1: almacenada")

                    # Análisis del snapshot HD
                    hd_snapshot_resized, snap_scale_x, snap_scale_y = resize_for_inference(hd_snapshot, max_dim=640)
                    logging.debug(f"Snapshot {snapshot_filename} - Paso 2: reconvertida")

                    refined_results = modelo_PATENTESy.predict(hd_snapshot_resized, device='cuda:0', verbose=False)
                    logging.debug(f"Snapshot {snapshot_filename} - Paso 3: enviada a modelo_PATENTES")

                    refined_box = None
                    for box in refined_results[0].boxes:
                        if box.conf * 100 >= CONFIANZA_PATENTE:
                            rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])
                            refined_box = (rx1, ry1, rx2, ry2)
                            break
                    if refined_box is None:
                        logging.debug(f"Snapshot {snapshot_filename} sin detección de patentes")
                        raise Exception("No se detectó patente en snapshot HD")

                    orig_x1 = int(refined_box[0] * snap_scale_x)
                    orig_y1 = int(refined_box[1] * snap_scale_y)
                    orig_x2 = int(refined_box[2] * snap_scale_x)
                    orig_y2 = int(refined_box[3] * snap_scale_y)
                    roi_hd_snapshot_refined = hd_snapshot[orig_y1:orig_y2, orig_x1:orig_x2]

                    plate_area = (orig_x2 - orig_x1) * (orig_y2 - orig_y1)
                    center_x = (orig_x1 + orig_x2) / 2.0
                    center_y = (orig_y1 + orig_y2) / 2.0

                    # OCR multiescala en diferentes regiones de cobertura
                    multi_scale_results = []
                    for porcentaje in range(5, 101, 5):
                        coverage_fraction = porcentaje / 100.0
                        roi_hd = calculate_roi_for_coverage(hd_snapshot, (center_x, center_y), plate_area, coverage_fraction)
                        if roi_hd is None:
                            continue
                        ocr_out = modelo_OCRy.predict(roi_hd, device='cuda:0', verbose=False)
                        result = ocr_processor.process_ocr_result_detailed(ocr_out)
                        result["coverage"] = porcentaje
                        result["roi_image"] = roi_hd
                        multi_scale_results.append(result)

                    # Añadir el resultado de OCR del stream si existe
                    for candidate in plates_for_ocr:
                        pid = candidate["plate_id"]
                        stream_res = best_plates[pid].get("ocr_stream", {})
                        if stream_res.get("ocr_text", "").strip():
                            multi_scale_results.append({
                                "ocr_text":    stream_res["ocr_text"],
                                "global_conf": stream_res.get("average_conf", 0.0),
                                "predictions": [],
                                "coverage":    100
                            })

                    # Aplicar diferentes algoritmos de consenso
                    stream_text = best_plates[plate_id].get('ocr_stream', {}).get('ocr_text', '')
                    consensus1 = apply_consensus_voting(multi_scale_results, min_length=5)
                    if consensus1 is None:
                        consensus1 = {"ocr_text": "", "average_conf": 0.0, "count": 0}
                    expected_length = len(consensus1["ocr_text"]) if consensus1["ocr_text"] != "" else 6
                    consensus2 = consensus_by_positions(multi_scale_results, expected_length=expected_length)
                    if consensus2 is None:
                        consensus2 = {"ocr_text": "", "average_conf": 0.0}

                    # Registro de logs
                    stream_text_display = "S/I" if len(stream_text) <= 5 else stream_text
                    log_text = (
                        f" - Consenso Stream: {stream_text_display}  "
                        f"\n - Consenso Snapshot A: {consensus1['ocr_text']}"
                        f"\n - Consenso Snapshot B: {consensus2['ocr_text']}")

                    # Combinación final de todos los consensos
                    consensos = [
                        best_plates[plate_id].get("ocr_stream", {"ocr_text": "", "average_conf": 0.0}),
                        consensus1,
                        consensus2
                    ]
                    openai_text = best_plates[plate_id].get("openai_result", "")
                    openai_consenso = {
                        "ocr_text": openai_text,
                        "average_conf": 100.0 if openai_text else 0.0
                    }
                    consensos.append(openai_consenso)
                    final_result = final_consensus(consensos)
                    logging.debug(f"Resultado final OCR para la patente {plate_id}: {final_result['ocr_text']} con confianza {final_result['average_conf']:.1f}%")

                    # Actualizar información de la patente
                    best_plates[plate_id].update({
                        "text": final_result["ocr_text"],
                        "confidence": final_result["average_conf"],
                        "last_seen": time.time(),
                        "locked": True
                    })

                    # Enviar resultados si se encontró texto
                    if final_result["ocr_text"] != "":
                        stream_text = best_plates[plate_id].get("ocr_stream", {}).get("ocr_text", "")
                        c1 = consensus1.get("ocr_text", "")
                        c2 = consensus2.get("ocr_text", "")
                        openai_display = openai_text if openai_text else "S/I"
                        log_text = (
                            f"Stream: {stream_text}\n"
                            f"Snapshot A: {c1}\n"
                            f"Snapshot B: {c2}\n"
                            f"OpenAI: {openai_display}"
                        )
                        executor.submit(
                            send_plate_async,
                            roi_hd_snapshot_refined,
                            display_frame,
                            final_result["ocr_text"],
                            log_text,
                            best_plates[plate_id]["coords"]
                        )
                        send_backend(final_result["ocr_text"], roi_hd_snapshot_refined)
                        logging.debug(f"==== Patente detectada con éxito: {final_result['ocr_text']}, se liberan recursos ====")

                        # Agregar a vehículos procesados
                        vehicle_bbox = best_plates[plate_id]["coords"]
                        bbox_hash = f"vehicle_{hash(str(vehicle_bbox))}"
                        processed_vehicles[bbox_hash] = {
                            "timestamp": time.time(),
                            "bbox": vehicle_bbox,
                            "plate": final_result["ocr_text"]
                        }

                        # Limpiar y continuar
                        best_plates.clear()
                        time.sleep(0.5)
                        continue
                        
                except Exception as e:
                    logging.warning(f"Error en el procesamiento OCR vía snapshot: {e}")

            # Eliminar placas que ya no se ven
            current_time = time.time()
            best_plates = { pid: data for pid, data in best_plates.items() if (current_time - data["last_seen"]) <= DISPLAY_DURATION }

            # 2.7 CÁLCULO DE FPS Y VISUALIZACIÓN
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            fps_deque.append(fps)
            fps_avg = sum(fps_deque) / len(fps_deque)

            if DEBUG_MODE:
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame, f"FPS Prom: {fps_avg:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Stream HD con Detección de Patentes", display_frame)
                
                # Manejo de teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("g"):
                    calib_params["gamma"] += 0.1
                    logging.info("Gamma incrementado: %.2f", calib_params["gamma"])
                elif key == ord("h"):
                    calib_params["gamma"] = max(0.1, calib_params["gamma"] - 0.1)
                    logging.info("Gamma decrementado: %.2f", calib_params["gamma"])
                elif key == ord("c"):
                    calib_params["clahe_enabled"] = not calib_params.get("clahe_enabled", True)
                    logging.info("CLAHE activado: %s", calib_params["clahe_enabled"])
                elif key == ord("s"):
                    save_calibration_params(calib_params)
                elif key == ord("r"):
                    logging.info("Reconexión forzada del stream…")
                    cap_hd.release()
                    time.sleep(1)
                    cap_hd = open_stream_with_suppressed_stderr(URL_SOURCE)

        except Exception as e:
            logging.warning(f"Error en el ciclo principal: {e}")
            time.sleep(1)
            continue

    # 3 LIMPIEZA FINAL
    cap_hd.release()
    if DEBUG_MODE:
        cv2.destroyAllWindows()
    snapshot_manager.stop()
    executor.shutdown()

if __name__ == '__main__':
    # Redirigir stderr para evitar mensajes de error de FFmpeg
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
