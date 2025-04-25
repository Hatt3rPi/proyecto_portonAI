"""
Módulo de procesamiento OCR y consenso para PortonAI
Incluye la clase OCRProcessor para gestionar todos los flujos de OCR.
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np
import cv2

from config import (
    CONSENSUS_MIN_LENGTH,
    CONSENSUS_EXPECTED_LENGTH_METHOD,
    CONSENSUS_FIXED_LENGTH
)


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
          - 'ocr_text' (str): Texto concatenado de caracteres detectados
          - 'confidence' (float): Confianza media geométrica [0–100]
          - 'predictions' (List[Dict]): Cada predicción con keys 'x', 'confidence', 'char'
    """
    if not ocr_result or len(ocr_result) == 0 or not hasattr(ocr_result[0], "boxes"):
        return {"ocr_text": "", "confidence": 0.0, "predictions": []}

    try:
        batch = ocr_result[0]
        preds: List[Dict[str, Any]] = []
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
                continue

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
    valid = [r for r in ocr_results if len(r.get("ocr_text","").strip()) >= length_threshold]
    if not valid:
        if not ocr_results:
            return None
        best = max(ocr_results, key=lambda r: r.get("confidence", 0.0))
        return {"ocr_text": best.get("ocr_text",""), "confidence": best.get("confidence",0.0), "count": 1}

    tally: Dict[str, Dict[str, float]] = {}
    for r in valid:
        txt = r["ocr_text"].upper().strip()
        info = tally.setdefault(txt, {"count":0, "total_conf":0.0})
        info["count"]   += 1
        info["total_conf"] += r.get("confidence",0.0)

    best_text, best_info = "", {"count":-1, "avg_conf":-1.0}
    for txt, info in tally.items():
        avg_conf = info["total_conf"] / info["count"]
        if (info["count"] > best_info["count"]) or (info["count"] == best_info["count"] and avg_conf > best_info["avg_conf"]):
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
    filtered = [r for r in ocr_results if r.get("ocr_text","" ).strip()]
    if not filtered:
        return None

    if expected_length is None:
        lengths = [len(r["ocr_text"]) for r in filtered]
        method = CONSENSUS_EXPECTED_LENGTH_METHOD.lower()
        try:
            if method == "mode":
                from statistics import mode, StatisticsError
                expected_length = mode(lengths)
            elif method == "median":
                expected_length = int(np.median(lengths))
            elif method == "fixed" and CONSENSUS_FIXED_LENGTH:
                expected_length = CONSENSUS_FIXED_LENGTH
            else:
                expected_length = int(np.median(lengths))
        except Exception:
            expected_length = int(np.median(lengths))

    letter_stats: List[Dict[str, Dict[str, float]]] = [{} for _ in range(expected_length)]
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
            entry["count"]       += 1
            entry["total_conf"]   += conf

    result_chars: List[str] = []
    confidences:  List[float] = []
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

    freq: Dict[str,int] = {}
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

    def process_multiscale(self, plate_img: Any) -> List[Dict[str, Any]]:
        """
        Ejecuta OCR en escalas del 50% al 100% por stream.
        Args:
            plate_img: ROI de la placa
        Returns:
            Lista de dicts procesados por process_ocr_result_detailed
        """
        results = []
        for scale in range(50, 105, 5):
            try:
                fx = fy = scale / 100.0
                roi = cv2.resize(plate_img, None, fx=fx, fy=fy)
                ocr_out = self.model_ocr.predict(roi, device='cuda:0', verbose=False)
                proc = process_ocr_result_detailed(ocr_out, self.names)
                proc['coverage'] = scale
                results.append(proc)
            except Exception as e:
                logging.warning(f"Error OCR multiescala escala {scale}%: {e}")
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
            if len(text) >= CONSENSUS_MIN_LENGTH:
                return {"ocr_text": text, "confidence": 100.0}
            else:
                logging.debug(f"OCR OpenAI lectura demasiado corta ({len(text)}< {CONSENSUS_MIN_LENGTH}), descartada")
                return {"ocr_text": "", "confidence": 0.0}

        # OCR tradicional con YOLO en escala 100%
        ocr_out = self.model_ocr.predict(plate_img, device='cuda:0', verbose=False)
        res = process_ocr_result_detailed(ocr_out, self.names)
        text = res.get('ocr_text', '').strip()
        if len(text) >= CONSENSUS_MIN_LENGTH:
            return {"ocr_text": text, "confidence": res.get('confidence', 0.0)}
        else:
            logging.debug(f"OCR lectura demasiado corta ({len(text)}< {CONSENSUS_MIN_LENGTH}), descartada")
            return {"ocr_text": "", "confidence": 0.0}
