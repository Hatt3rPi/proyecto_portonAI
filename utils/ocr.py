## archivo: ocr.py
"""
Módulo de procesamiento OCR y consenso para PortonAI
Incluye la clase OCRProcessor y funciones auxiliares para validar placas.
"""

from typing import List, Dict, Any, Optional
import logging
import re
import numpy as np
import cv2

from config import (
    CONSENSUS_MIN_LENGTH,
    CONSENSUS_EXPECTED_LENGTH_METHOD,
    CONSENSUS_FIXED_LENGTH,
    OCR_OPENAI_ACTIVATED,
    QA_ANALISIS_AVANZADO,  # Añadimos la importación de esta variable
    ROI_ANGULO_ROTACION,
    ROI_ESCALA_FACTOR,
    ROI_APLICAR_CORRECCION
)


def is_valid_plate(plate: str) -> bool:
    """
    Valida si el texto dado corresponde a una placa patente válida.
    Lógicas incluidas para:
      - Vehículos estándar (6 chars, bloques AA/BB/99)
      - Vehículos policiales (Z/M/RP/A + 4 dígitos)
      - Bomberos (CB + letra + 3 dígitos)
      - Fuerzas Armadas (EJTO + 4 dígitos)
      - Provisoria (PR + 3 dígitos)
      - Motocicletas (5 chars, XXX?9)
    """
    p = plate.upper().strip()
    # Vehículo estándar: 6 caracteres, bloques 2L-2L|2D-2D-LD
    if re.fullmatch(r"[A-Z]{2}(?:[A-Z]{2}|\d{2})\d{2}", p):
        return True
    # Vehículo policial: Z/M/A + 4 digits o RP +4 digits
    if re.fullmatch(r"(?:Z|M|A)\d{4}|RP\d{4}", p):
        return True
    # Bomberos: CB + letra municipio + 3 dígitos
    if re.fullmatch(r"CB[A-Z]\d{3}", p):
        return True
    # Fuerzas Armadas: EJTO + 4 dígitos
    if re.fullmatch(r"EJTO\d{4}", p):
        return True
    # Provisoria: PR + 3 dígitos
    if re.fullmatch(r"PR\d{3}", p):
        return True
    # Motocicletas: 5 chars, primero 3 letras, cualquier caracter alfanumérico, último dígito
    #if re.fullmatch(r"[A-Z]{3}[A-Z0-9]\d", p):
    #    return True
    return False


def process_ocr_result_detailed(
    ocr_result: Any,
    model_ocr_names: List[str]
) -> Dict[str, Any]:
    """
    Convierte resultados crudos de YOLO (OCR) en un dict estandarizado.
    Args:
        ocr_result: Resultado de llama a modelo_OCR.predict(...)
        model_ocr_names: Lista de nombres de clases (caracteres)

    Returns:
        Dict con:
          - 'ocr_text' (str)
          - 'confidence' (float)
          - 'predictions' (List[Dict])
    """
    if not ocr_result or len(ocr_result) == 0 or not hasattr(ocr_result[0], "boxes"):
        return {"ocr_text": "", "confidence": 0.0, "predictions": []}
    try:
        batch = ocr_result[0]
        preds = []
        for box in batch.boxes:
            try:
                coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], "cpu") else box.xyxy[0]
                x1, y1, x2, y2 = map(int, coords)
                x_center = (x1 + x2) / 2.0
                raw_conf = box.conf[0] if hasattr(box.conf, "__iter__") else box.conf
                conf = float(raw_conf) * 100
                cls_idx = int(box.cls[0]) if hasattr(box, "cls") else 0
                char = model_ocr_names[cls_idx] if cls_idx < len(model_ocr_names) else ""
                preds.append({"x": x_center, "confidence": conf, "char": char})
            except Exception as e:
                logging.warning(f"OCR prediction malformed, se omite: {e}")
        preds.sort(key=lambda p: p["x"])
        confidences = [p["confidence"] for p in preds]
        geom_mean = float(np.prod(confidences) ** (1.0 / len(confidences))) if confidences else 0.0
        text = "".join(p["char"] for p in preds)
        return {"ocr_text": text, "confidence": geom_mean, "predictions": preds}
    except Exception as e:
        logging.error(f"Error en process_ocr_result_detailed: {e}")
        return {"ocr_text": "", "confidence": 0.0, "predictions": []}


def apply_consensus_voting(
    ocr_results: List[Dict[str, Any]],
    min_length: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Vota entre múltiples lecturas OCR para elegir la más frecuente y confiable.
    Filtra también por validez de placa.
    Args:
        ocr_results: Lista de dicts con keys 'ocr_text' y 'confidence'
        min_length: Longitud mínima de texto para considerar; si None, usa CONSENSUS_MIN_LENGTH

    Returns:
        Dict con:
          - 'ocr_text' (str)
          - 'confidence' (float)
          - 'count' (int)
        O None si no hay lecturas válidas.
    """
    length_threshold = min_length or CONSENSUS_MIN_LENGTH
    # Solo textos con longitud mínima y válidos según is_valid_plate
    valid = [r for r in ocr_results
             if len(r.get("ocr_text","")) >= length_threshold
             and is_valid_plate(r.get("ocr_text",""))]
    if not valid:
        if not ocr_results:
            return None
        # Escoge el de mayor confianza, pero validar primero
        candidates = [r for r in ocr_results if is_valid_plate(r.get("ocr_text",""))]
        if not candidates:
            return None
        best = max(candidates, key=lambda r: r.get("confidence", 0.0))
        return {"ocr_text": best.get("ocr_text",""), "confidence": best.get("confidence",0.0), "count": 1}
    tally = {}
    for r in valid:
        txt = r["ocr_text"].upper().strip()
        info = tally.setdefault(txt, {"count":0, "total_conf":0.0})
        info["count"] += 1
        info["total_conf"] += r.get("confidence",0.0)
    best_text, best_info = "", {"count":-1, "avg_conf":-1.0}
    for txt, info in tally.items():
        avg_conf = info["total_conf"] / info["count"]
        if info["count"] > best_info["count"] or (info["count"] == best_info["count"] and avg_conf > best_info["avg_conf"]):
            best_text, best_info = txt, {"count": info["count"], "avg_conf": avg_conf}
    return {"ocr_text": best_text, "confidence": best_info["avg_conf"], "count": best_info["count"]}


def consensus_by_positions(
    ocr_results: List[Dict[str, Any]],
    expected_length: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Construye la matrícula carácter a carácter mediante votación por posición.

    Args:
        ocr_results: Lista de dicts con 'predictions' (x, confidence, char)
        expected_length: Longitud esperada; si None, calculada según config

    Returns:
        Dict con 'ocr_text' y 'confidence', o None si falla.
    """
    filtered = [r for r in ocr_results if r.get("ocr_text","" )]
    if not filtered:
        return None
    if expected_length is None:
        lengths = [len(r["ocr_text"]) for r in filtered]
        method = CONSENSUS_EXPECTED_LENGTH_METHOD.lower()
        try:
            if method == "mode":
                from statistics import mode
                expected_length = mode(lengths)
            elif method == "median":
                expected_length = int(np.median(lengths))
            elif method == "fixed" and CONSENSUS_FIXED_LENGTH:
                expected_length = CONSENSUS_FIXED_LENGTH
            else:
                expected_length = int(np.median(lengths))
        except Exception:
            expected_length = int(np.median(lengths))
    letter_stats = [{} for _ in range(expected_length)]
    for r in filtered:
        for i, pred in enumerate(r.get("predictions", [])):
            if i >= expected_length:
                break
            char = pred.get("char","" ).upper().strip()
            conf = pred.get("confidence",0.0)
            if not char:
                continue
            stats = letter_stats[i]
            entry = stats.setdefault(char, {"count":0, "total_conf":0.0})
            entry["count"]     += 1
            entry["total_conf"] += conf
    result_chars = []
    confidences = []
    for stats in letter_stats:
        if not stats:
            result_chars.append("?")
            confidences.append(0.0)
        else:
            best_char, best_avg = "?", -1.0
            for char, info in stats.items():
                avg_conf = info["total_conf"]/info["count"]
                if avg_conf > best_avg:
                    best_char, best_avg = char, avg_conf
            result_chars.append(best_char)
            confidences.append(best_avg)
    text = "".join(result_chars)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    return {"ocr_text": text, "confidence": avg_conf}


def final_consensus(
    consensus_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Combina múltiples consensos para definir la matrícula final.

    Args:
        consensus_list: Liste de dicts con 'ocr_text' y 'confidence'

    Returns:
        Dict con 'ocr_text' y 'confidence'.
    """
    if not consensus_list:
        return {"ocr_text": "", "confidence": 0.0}
    freq = {}
    for c in consensus_list:
        txt = c.get("ocr_text","" )
        freq[txt] = freq.get(txt,0) + 1
    max_f = max(freq.values(), default=0)
    cands = [txt for txt,f in freq.items() if f==max_f]
    filtered = [c for c in consensus_list if c.get("ocr_text","" ) in cands]
    best = max(filtered, key=lambda c: c.get("confidence",0.0))
    return {"ocr_text": best.get("ocr_text",""), "confidence": best.get("confidence",0.0)}


class OCRProcessor:
    """
    Clase que centraliza flujos de OCR y consenso:
      - process_multiscale: OCR en múltiples escalas desde frame stream
      - process_plate_image: OCR de snapshot o OpenAI
    """
    def __init__(self, model_ocr, model_ocr_names):
        self.model_ocr = model_ocr
        self.names     = model_ocr_names
        # Para guardar los callbacks de progreso
        self.progress_callback = None

    def set_progress_callback(self, callback):
        """Establece un callback para reportar progreso"""
        self.progress_callback = callback

    def process_multiscale(self, plate_img: Any, fixed_angle=None, fixed_scale=None, test_mode=False, 
                          angle_range=None, scale_range=None, angle_step=None, scale_step=None) -> List[Dict[str, Any]]:
        """
        Ejecuta OCR en escalas del 50% al 100% por stream.
        Descarta lecturas no válidas antes de agregarlas.
        Args:
            plate_img: ROI de la placa
            fixed_angle: Ángulo fijo para rotar (None = usar ROI_ANGULO_ROTACION)
            fixed_scale: Factor de escala fijo (None = usar ROI_ESCALA_FACTOR)
            test_mode: Si es True, realiza pruebas en múltiples ángulos/escalas en vez de usar valores fijos
            angle_range: Tupla (min, max) de ángulos a probar en modo test
            scale_range: Tupla (min, max) de factores de escala a probar en modo test
            angle_step: Paso entre ángulos en modo test
            scale_step: Paso entre escalas en modo test
        Returns:
            Lista de dicts procesados por process_ocr_result_detailed
        """
        # Si llega un ROI vacío, no intentamos ningún resize
        if plate_img is None or plate_img.size == 0:
            return []
            
        results = []
        
        # Si estamos en modo prueba, realizamos un barrido de parámetros
        if test_mode and angle_range and scale_range:
            angulos = np.arange(angle_range[0], angle_range[1] + angle_step, angle_step)
            escalas = np.arange(scale_range[0], scale_range[1] + scale_step, scale_step)
            
            for angulo in angulos:
                for escala in escalas:
                    try:
                        # 1. Aplicar rotación
                        if abs(angulo) > 0.1:  # Si el ángulo no es aproximadamente 0
                            h, w = plate_img.shape[:2]
                            center = (w // 2, h // 2)
                            M = cv2.getRotationMatrix2D(center, angulo, 1.0)
                            roi_rotado = cv2.warpAffine(plate_img, M, (w, h), 
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=(0, 0, 0))
                        else:
                            roi_rotado = plate_img.copy()
                        
                        # 2. Aplicar escala
                        if abs(escala/100.0 - 1.0) > 0.01:  # Si el factor no es aproximadamente 1
                            factor = escala/100.0
                            roi_escalado = cv2.resize(roi_rotado, None, fx=factor, fy=factor)
                        else:
                            roi_escalado = roi_rotado
                        
                        # OCR procesamiento
                        ocr_out = self.model_ocr.predict(roi_escalado, device='cuda:0', verbose=False)
                        proc = process_ocr_result_detailed(ocr_out, self.names)
                        
                        text = proc.get('ocr_text', '').strip()
                        valid = is_valid_plate(text)
                        
                        if text and valid:
                            proc['angle'] = angulo
                            proc['scale'] = escala
                            results.append(proc)
                    
                    except Exception:
                        # Suprimir errores completamente
                        pass
            
            # Reportar progreso si hay un callback configurado
            if self.progress_callback:
                self.progress_callback(len(angulos) * len(escalas), len(angulos) * len(escalas))
                
            return results
            
        # Aplicar la corrección de ángulo y escala óptima definida en config.py o parámetros fijos
        if ROI_APLICAR_CORRECCION or fixed_angle is not None or fixed_scale is not None:
            try:
                # Determinar ángulo a usar
                angle_to_use = fixed_angle if fixed_angle is not None else ROI_ANGULO_ROTACION
                
                # 1. Aplicar rotación (si es necesario)
                if abs(angle_to_use) > 0.1:  # Si el ángulo no es aproximadamente 0
                    h, w = plate_img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle_to_use, 1.0)
                    plate_img = cv2.warpAffine(plate_img, M, (w, h), 
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=(0, 0, 0))
                
                # Determinar escala a usar
                scale_to_use = fixed_scale if fixed_scale is not None else ROI_ESCALA_FACTOR
                
                # 2. Aplicar escala óptima
                if abs(scale_to_use - 1.0) > 0.01:  # Si el factor no es aproximadamente 1
                    plate_img = cv2.resize(plate_img, None, fx=scale_to_use, fy=scale_to_use)
                
                # 3. Procesar la imagen corregida
                ocr_out = self.model_ocr.predict(plate_img, device='cuda:0', verbose=False)
                proc = process_ocr_result_detailed(ocr_out, self.names)
                
                text = proc.get('ocr_text', '').strip()
                valid = is_valid_plate(text)
                
                if text and valid:
                    proc['coverage'] = 100  # Marcamos como procesada con parámetros óptimos
                    proc['optimized'] = True
                    proc['angle'] = angle_to_use
                    proc['scale'] = scale_to_use * 100 if scale_to_use < 10 else scale_to_use  # Normalizar escala a porcentaje
                    results.append(proc)
                    return results  # Si tenemos un resultado válido con los parámetros óptimos, lo retornamos directamente
            
            except Exception as e:
                logging.warning(f"Error al aplicar corrección óptima: {e}")
                # Continuamos con el enfoque multi-escala tradicional
        
        # Enfoque multi-escala tradicional (como fallback o si no se aplica corrección)
        total_scales = 11  # Del 50% al 100% en pasos de 5% (11 escalas)
        
        for scale in range(50, 105, 5):
            try:
                fx = fy = scale / 100.0
                roi = cv2.resize(plate_img, None, fx=fx, fy=fy)
                
                # OCR procesamiento
                ocr_out = self.model_ocr.predict(roi, device='cuda:0', verbose=False)
                proc = process_ocr_result_detailed(ocr_out, self.names)
                text = proc.get('ocr_text', '').strip()
                
                # Filtrar resultados inválidos
                valid = is_valid_plate(text)
                if not text or not valid:
                    continue
                
                proc['coverage'] = scale
                proc['optimized'] = False
                proc['angle'] = 0  # Sin rotación en el método tradicional
                proc['scale'] = scale
                results.append(proc)
                
            except Exception:
                # Suprimir errores completamente
                pass
            
            # Reportar progreso si hay un callback configurado
            if self.progress_callback:
                self.progress_callback(scale - 50 + 1, 11)  # +1 porque contamos desde 1
        
        return results

    def process_plate_image(
        self,
        plate_img: Any,
        use_openai: bool = False
    ) -> Dict[str, Any]:
        """
        Ejecuta OCR sobre snapshot HD o usa OpenAI.
        Args:
            plate_img: Imagen de la placa (snapshot o frame)
            use_openai: Si True, usar OpenAI en lugar de YOLO
        Returns:
            Dict resultante con 'ocr_text' y 'confidence'
        """
        if use_openai:
            from scripts.monitoreo_patentes.openai_plate_reader import read_plate_openai
            text = read_plate_openai(plate_img, None) or ""
            if len(text) >= CONSENSUS_MIN_LENGTH and is_valid_plate(text):
                return {"ocr_text": text, "confidence": 100.0}
            logging.debug(f"OCR OpenAI descarta '{text}' inválido")
            return {"ocr_text": "", "confidence": 0.0}
        ocr_out = self.model_ocr.predict(plate_img, device='cuda:0', verbose=False)
        res = process_ocr_result_detailed(ocr_out, self.names)
        text = res.get('ocr_text', '').strip()
        confidence = res.get('confidence', 0.0)
        if len(text) >= CONSENSUS_MIN_LENGTH and is_valid_plate(text):
            return {"ocr_text": text, "confidence": confidence}
        logging.debug(f"OCR tradicional descarta '{text}' inválido")

        # Si está habilitado OpenAI y la confianza es baja, intentar con OpenAI
        if OCR_OPENAI_ACTIVATED and use_openai and confidence < 0.8:
            try:
                from scripts.monitoreo_patentes.openai_plate_reader import read_plate_openai
                text = read_plate_openai(plate_img, None) or ""
                if len(text) >= CONSENSUS_MIN_LENGTH and is_valid_plate(text):
                    return {"ocr_text": text, "confidence": 100.0}
                logging.debug(f"OCR OpenAI descarta '{text}' inválido")
            except Exception as e:
                logging.warning(f"Error procesando con OpenAI: {e}")
        
        return {
            "ocr_text": text,
            "confidence": confidence,
            "raw_output": res
        }
