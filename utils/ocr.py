"""
Procesamiento OCR para reconocimiento de texto en patentes
"""

import numpy as np
import logging
import cv2
import sys
import os
from typing import List, Dict, Any, Optional, Tuple
import concurrent.futures
from models import ModelManager

# Ajuste para importar correctamente el módulo openai_plate_reader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts/monitoreo_patentes')))
from openai_plate_reader import read_plate_openai, process_plate_image as openai_process_plate_image

class OCRProcessor:
    """
    Clase para manejar todos los procesos relacionados con OCR
    """
    def __init__(self):
        """Inicializa el procesador OCR con los modelos necesarios"""
        logging.debug("Inicializando OCRProcessor")
        self.model_manager = ModelManager()
        self.ocr_model = self.model_manager.get_ocr_model()
        self.ocr_names = self.model_manager.get_ocr_names()
    
    def process_ocr_result_detailed(self, ocr_result):
        """
        Procesa un resultado OCR de YOLO en formato detallado
        
        Args:
            ocr_result: Resultado de predicción del modelo OCR
            
        Returns:
            dict: Diccionario con texto OCR y detalles de confianza
        """
        if not ocr_result or len(ocr_result) == 0:
            return {"ocr_text": "", "global_conf": 0.0, "predictions": []}
        
        ocr_result = ocr_result[0]
        predictions = []
        
        for box in ocr_result.boxes:
            if hasattr(box.xyxy[0], "cpu"):
                coords_box = box.xyxy[0].cpu().numpy()
            else:
                coords_box = box.xyxy[0]
                
            x1_box, y1_box, x2_box, y2_box = map(int, coords_box)
            x_center = (x1_box + x2_box) / 2.0
            conf = float(box.conf[0]) if hasattr(box.conf, "__iter__") else float(box.conf)
            cls_index = int(box.cls[0]) if hasattr(box, "cls") else 0
            
            char_pred = self.ocr_names[cls_index] if cls_index < len(self.ocr_names) else ""
            predictions.append({"x": x_center, "confidence": conf, "char": char_pred})
            
        # Ordenar predicciones por posición x
        predictions.sort(key=lambda p: p["x"])
        
        # Calcular confianza global
        confidences = [p["confidence"] for p in predictions]
        if confidences:
            product = 1.0
            for c in confidences:
                product *= c
            global_conf = product ** (1 / len(confidences))
        else:
            global_conf = 0.0
            
        # Obtener texto final
        ocr_text = "".join(p["char"] for p in predictions)
        
        return {"ocr_text": ocr_text, "global_conf": global_conf * 100, "predictions": predictions}
    
    def process_plate_image(self, plate_img, use_openai=False, api_key=None):
        """
        Procesa una imagen de patente, usando OpenAI si se especifica
        
        Args:
            plate_img: Imagen de la patente
            use_openai: Si es True, usa OpenAI para reconocer la patente
            api_key: API Key de OpenAI (opcional)
        
        Returns:
            Diccionario con el resultado del OCR
        """
        if use_openai:
            # Integrar con la función de openai_plate_reader.py
            return openai_process_plate_image(plate_img, use_openai=True)
        else:
            # Usar OCR local
            result = self.process_multiscale(plate_img)
            consensus = self.apply_consensus_voting(result)
            if consensus:
                return consensus
            elif result:
                return max(result, key=lambda r: r["global_conf"])
            else:
                return {"ocr_text": "", "average_conf": 0.0}
    
    def process_multiscale(self, plate_img):
        """
        Procesa una imagen de patente con múltiples escalas
        
        Args:
            plate_img: Imagen de la patente
            
        Returns:
            list: Resultados del OCR en diferentes escalas
        """
        ocr_stream_results_multiscale = []
        for scale in range(50, 105, 5):
            scale_factor = scale / 100.0
            try:
                scaled_roi = cv2.resize(plate_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
                ocr_result = self.ocr_model.predict(scaled_roi, device='cuda:0', verbose=False)
                processed_result = self.process_ocr_result_detailed(ocr_result)
                processed_result["coverage"] = scale
                ocr_stream_results_multiscale.append(processed_result)
            except Exception as e:
                logging.warning(f"Error en OCR stream multiescala con escala {scale}%: {e}")
                
        return ocr_stream_results_multiscale
    
    @staticmethod
    def apply_consensus_voting(ocr_results, min_length=5):
        """
        Aplica un algoritmo de consenso por votación a los resultados OCR
        
        Args:
            ocr_results: Lista de resultados OCR
            min_length: Longitud mínima del texto para considerar en el consenso
            
        Returns:
            dict: Resultado del consenso
        """
        if not ocr_results:
            return None
            
        consensus_dict = {}
        for res in ocr_results:
            text = res["ocr_text"].upper().strip()
            if len(text) < min_length:
                continue
            if text not in consensus_dict:
                consensus_dict[text] = {"count": 0, "total_conf": 0.0}
            consensus_dict[text]["count"] += 1
            consensus_dict[text]["total_conf"] += res["global_conf"]
            
        if not consensus_dict:
            if not ocr_results:
                return None
            best_res = max(ocr_results, key=lambda r: r["global_conf"])
            return {"ocr_text": best_res["ocr_text"].upper().strip(),
                    "average_conf": best_res["global_conf"],
                    "count": 1}
                    
        best_text = None
        best_count = -1
        best_avg_conf = -1.0
        
        for text, info in consensus_dict.items():
            count = info["count"]
            avg_conf = info["total_conf"] / count
            if count > best_count or (count == best_count and avg_conf > best_avg_conf):
                best_text = text
                best_count = count
                best_avg_conf = avg_conf
                
        return {"ocr_text": best_text, "average_conf": best_avg_conf, "count": best_count}
