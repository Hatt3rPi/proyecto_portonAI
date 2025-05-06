## archivo: ocr.py
"""
Módulo de procesamiento OCR y consenso para PortonAI
Incluye la clase OCRProcessor y funciones auxiliares para validar placas.
"""

from typing import List, Dict, Any, Optional, Callable
import logging
import re
import numpy as np
import cv2
import os
import time

from config import (
    CONSENSUS_MIN_LENGTH,
    CONSENSUS_EXPECTED_LENGTH_METHOD,
    CONSENSUS_FIXED_LENGTH,
    OCR_OPENAI_ACTIVATED,
    QA_ANALISIS_AVANZADO,
    ROI_ANGULO_ROTACION,
    ROI_ESCALA_FACTOR,
    ROI_APLICAR_CORRECCION,
    USE_SUPER_RESOLUTION,
    OCR_MIN_CONFIDENCE,
    OCR_MULTISCALE_APPLY_ROTATION
)

# --- Importación condicional para mejorador de imágenes ---
try:
    from utils.image_enhancement import enhance_plate_image
    IMAGE_ENHANCEMENT_AVAILABLE = True
    logging.info("Módulo de mejoramiento de imagen disponible")
except ImportError:
    IMAGE_ENHANCEMENT_AVAILABLE = False
    logging.warning("Módulo de mejoramiento de imagen no disponible")


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
    return False


class OCRProcessor:
    """
    Procesador OCR para reconocimiento de placas vehiculares.
    Soporta múltiples estrategias, incluyendo multiescala, rotación
    y corrección de perspectiva.
    """
    
    def __init__(self, 
                model=None, 
                names=None, 
                apply_rotation=OCR_MULTISCALE_APPLY_ROTATION):
        self.model = model  # Modelo YOLO para OCR
        self.names = names  # Clases del modelo OCR
        self.apply_rotation = apply_rotation
        self.progress_callback = None  # Callback para reportar progreso en QA
        
        # Valores por defecto para multiescala
        self.scales = [80, 90, 100, 110, 120]
        if self.apply_rotation:
            self.rotation_angles = [-5, -2.5, 0, 2.5, 5]
        else:
            self.rotation_angles = [0]
            
    def set_progress_callback(self, callback_fn: Callable):
        """
        Establece una función de callback para reportar progreso.
        Utilizado principalmente en modo QA.
        
        Args:
            callback_fn: Función que recibe (iteración_actual, total_iteraciones)
        """
        self.progress_callback = callback_fn

    def extract_plate_text(self, plate_region: np.ndarray) -> Dict:
        """
        Extrae texto de una imagen de placa vehicular usando el modelo OCR.
        
        Args:
            plate_region: Imagen de la placa recortada
            
        Returns:
            Dict con resultados del OCR
        """
        try:
            # Si el modelo no está cargado, devolver resultado vacío
            if self.model is None:
                return {"ocr_text": "", "confidence": 0.0}
                
            # Inferencia con el modelo OCR
            results = self.model.predict(plate_region, verbose=False)
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                return {"ocr_text": "", "confidence": 0.0}
                
            # Ordenar detecciones por coordenada X
            boxes = results[0].boxes
            box_coords = boxes.xyxy.cpu().numpy()
            
            if len(box_coords) == 0:
                return {"ocr_text": "", "confidence": 0.0}
                
            # Ordenar por posición X para leer de izquierda a derecha
            sorted_indices = np.argsort(box_coords[:, 0])
            
            # Extraer caracteres y confianzas ordenados
            text = ""
            total_conf = 0.0
            
            for idx in sorted_indices:
                cls_id = int(boxes.cls[idx].item())
                conf = float(boxes.conf[idx].item())
                
                if cls_id < len(self.names):
                    char = self.names[cls_id]
                    text += char
                    total_conf += conf
            
            avg_conf = total_conf / len(sorted_indices) if sorted_indices.size > 0 else 0.0
            
            # Empaquetar resultados
            return {
                "ocr_text": text,
                "confidence": avg_conf * 100,  # Convertir a porcentaje
                "raw_results": results
            }
            
        except Exception as e:
            logging.error(f"Error en OCR: {e}")
            return {"ocr_text": "", "confidence": 0.0, "error": str(e)}
            
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rota una imagen por el ángulo especificado.
        
        Args:
            image: Imagen a rotar
            angle: Ángulo en grados
            
        Returns:
            Imagen rotada
        """
        if angle == 0:
            return image.copy()
            
        # Obtener dimensiones
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Matriz de rotación
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Aplicar rotación
        rotated = cv2.warpAffine(
            image, rot_matrix, (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        return rotated
        
    def scale_image(self, image: np.ndarray, scale_percent: float) -> np.ndarray:
        """
        Escala una imagen por el porcentaje especificado.
        
        Args:
            image: Imagen a escalar
            scale_percent: Porcentaje de escala (100 = sin cambios)
            
        Returns:
            Imagen escalada
        """
        if scale_percent == 100:
            return image.copy()
            
        # Calcular nuevas dimensiones
        scale_factor = scale_percent / 100.0
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        
        # Redimensionar imagen
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LINEAR)
        
        return resized
        
    def is_valid_plate(self, text: str) -> bool:
        """
        Verifica si el texto detectado corresponde a una placa válida.
        
        Args:
            text: Texto detectado
            
        Returns:
            True si la placa parece válida, False en caso contrario
        """
        if not text or len(text) < 5:
            return False
            
        # Patrón básico para placas chilenas: LLLLNN (4 letras, 2 números)
        # o patrones LLNNNN (2 letras, 4 números), y otros más nuevos
        
        # Primero eliminar cualquier espacio en blanco
        text = text.strip()
        
        # Patrones comunes de placas chilenas
        patterns = [
            r'^[BCDFGHJKLMNPQRSTVWXYZ]{2}[0-9]{4}$',  # Formato AA1234
            r'^[BCDFGHJKLMNPQRSTVWXYZ]{4}[0-9]{2}$',  # Formato AAAA12
            r'^[BCDFGHJKLMNPQRSTVWXYZ]{3}[0-9]{3}$',  # Formato AAA123
            r'^[BCDFGHJKLMNPQRSTVWXYZ]{3}[0-9]{2}[BCDFGHJKLMNPQRSTVWXYZ]{1}$',  # AAANNA
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
                
        return False

    def process_multiscale(self, 
                          roi: np.ndarray, 
                          scales: List[int] = None,
                          rotation_angles: List[float] = None,
                          track_id: str = None,
                          image_enhancer = None) -> List[Dict]:
        """
        Procesa un ROI de placa utilizando múltiples escalas y ángulos de rotación.
        
        Args:
            roi: Imagen de la placa (región de interés)
            scales: Lista de escalas a probar (en porcentaje, 100 = tamaño original)
            rotation_angles: Lista de ángulos a probar (en grados)
            track_id: ID del tracker para usar el mejorador de imágenes
            image_enhancer: Mejorador de imágenes para aplicar superresolución
            
        Returns:
            Lista de resultados OCR ordenados por confianza
        """
        if roi is None or roi.size == 0:
            return []
            
        # Usar valores por defecto si no se especifican
        if scales is None:
            scales = self.scales
            
        if rotation_angles is None:
            rotation_angles = self.rotation_angles
            
        results = []
        total_iterations = len(scales) * len(rotation_angles)
        current_iteration = 0
        
        # Inicializar contador de iteraciones para callback de progreso
        for scale in scales:
            for angle in rotation_angles:
                # Aplicar transformaciones
                try:
                    # Escalar imagen
                    scaled = self.scale_image(roi, scale)
                    
                    # Rotar imagen
                    if angle != 0:
                        transformed = self.rotate_image(scaled, angle)
                    else:
                        transformed = scaled
                        
                    # Ejecutar OCR
                    ocr_result = self.extract_plate_text(transformed)
                    
                    # Añadir metadatos sobre la transformación
                    ocr_result["scale"] = scale
                    ocr_result["angle"] = angle
                    ocr_result["valid"] = self.is_valid_plate(ocr_result.get("ocr_text", ""))
                    
                    # Añadir a resultados
                    results.append(ocr_result)
                    
                    # Logging detallado
                    text = ocr_result.get("ocr_text", "")
                    conf = ocr_result.get("confidence", 0.0)
                    valid = ocr_result.get("valid", False)
                    logging.debug(f"[OCR-MULTIESCALA] Escala {scale}%: '{text}' válido={valid} conf={conf:.2f}")
                    
                except Exception as e:
                    logging.error(f"Error en proceso multiescala: {e}")
                
                # Actualizar progreso si hay callback
                current_iteration += 1
                if self.progress_callback:
                    self.progress_callback(current_iteration, total_iterations)
        
        logging.debug(f"[OCR-MULTIESCALA] Terminado procesamiento con {len(results)} combinaciones")
        
        # Si no se encontraron resultados válidos y tenemos un enhancer disponible, intentar SR
        if (not any(r.get("valid", False) for r in results) and 
            track_id is not None and image_enhancer is not None):
            try:
                logging.info(f"[OCR-FALLBACK] Intentando método de superresolución para track_id={track_id}")
                enhanced_roi, method = image_enhancer.get_enhanced_roi(track_id)
                if enhanced_roi is not None:
                    # Procesar el ROI mejorado de la misma manera
                    sr_results = self.process_multiscale(enhanced_roi, scales, rotation_angles)
                    # Añadir información sobre el método de mejora
                    for r in sr_results:
                        r["enhancement_method"] = method
                    
                    # Añadir resultados de superresolución a los originales
                    results.extend(sr_results)
                    
                    # Guardar una copia para depuración
                    debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug")
                    if not os.path.exists(debug_dir):
                        os.makedirs(debug_dir)
                    timestamp = int(time.time() * 1000)
                    filename = f"{timestamp}_{track_id}_enhanced_{method}.jpg"
                    cv2.imwrite(os.path.join(debug_dir, filename), enhanced_roi)
                    
                    logging.info(f"[OCR-FALLBACK] {len(sr_results)} resultados adicionales con método {method}")
            except Exception as e:
                logging.error(f"[OCR-FALLBACK] Error en proceso de superresolución: {e}")
        
        # Nueva funcionalidad: si no se obtuvo ningún resultado válido con multiescala,
        # intentar mejorar la imagen con RealESRGAN y volver a procesar
        if not results and IMAGE_ENHANCEMENT_AVAILABLE:
            try:
                logging.info("[OCR] Intentando mejoramiento de imagen con RealESRGAN...")
                # Mejorar imagen usando RealESRGAN
                enhanced_img, success = enhance_plate_image(roi)
                
                if success:
                    logging.info("[OCR] Imagen mejorada exitosamente, procesando con multiescala...")
                    
                    # Intentar OCR en la imagen mejorada con multiescala estándar
                    for scale in scales:
                        for angle in rotation_angles:
                            try:
                                # Escalar imagen
                                scaled = self.scale_image(enhanced_img, scale)
                                
                                # Rotar imagen
                                if angle != 0:
                                    transformed = self.rotate_image(scaled, angle)
                                else:
                                    transformed = scaled
                                    
                                # Ejecutar OCR
                                ocr_result = self.extract_plate_text(transformed)
                                
                                # Añadir metadatos sobre la transformación
                                ocr_result["scale"] = scale
                                ocr_result["angle"] = angle
                                ocr_result["valid"] = self.is_valid_plate(ocr_result.get("ocr_text", ""))
                                ocr_result["enhanced"] = True  # Indicador de que se usó mejoramiento
                                
                                # Añadir a resultados
                                results.append(ocr_result)
                                
                                # Logging detallado
                                text = ocr_result.get("ocr_text", "")
                                conf = ocr_result.get("confidence", 0.0)
                                valid = ocr_result.get("valid", False)
                                logging.debug(f"[OCR-MEJORADO] Escala {scale}%: '{text}' válido={valid} conf={conf:.2f}")
                                
                            except Exception as e:
                                logging.error(f"Error en proceso multiescala mejorado: {e}")
            except Exception as e:
                logging.error(f"[OCR] Error al intentar mejoramiento de imagen: {e}")
        
        # Ordenar resultados por confianza
        sorted_results = sorted(results, key=lambda x: x.get("confidence", 0.0), reverse=True)
        
        return sorted_results

    def process_plate_image(self, 
                           plate_image: np.ndarray,
                           use_multiscale: bool = True,
                           use_openai: bool = False,
                           track_id: str = None,
                           image_enhancer = None) -> Dict:
        """
        Procesa una imagen de placa vehicular con múltiples estrategias.
        
        Args:
            plate_image: Imagen de la placa recortada
            use_multiscale: Si es True, utiliza procesamiento multiescala
            use_openai: Si es True, utiliza OpenAI para reconocimiento
            track_id: ID del tracker para usar el mejorador de imágenes
            image_enhancer: Mejorador de imágenes para aplicar superresolución
            
        Returns:
            Dict con resultados del OCR
        """
        if plate_image is None or plate_image.size == 0:
            return {"ocr_text": "", "confidence": 0.0}
            
        # 1. Método multiescala (es el más completo)
        if use_multiscale:
            results = self.process_multiscale(plate_image, track_id=track_id, image_enhancer=image_enhancer)
            
            if results:
                # Filtrar por resultados válidos
                valid_results = [r for r in results if r.get("valid", False)]
                
                if valid_results:
                    # Ordenar por confianza
                    valid_results.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
                    best = valid_results[0]
                else:
                    # Si no hay resultados válidos, tomar el de mayor confianza
                    results.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
                    best = results[0]
                    
                return best
        
        # 2. Método simple (como fallback)
        result = self.extract_plate_text(plate_image)
        
        # 3. Si tenemos superresolución disponible y aún no ha sido utilizado (sin resultados)
        if (track_id and image_enhancer and 
            (not result or result.get("ocr_text", "") == "" or 
             result.get("confidence", 0) < OCR_MIN_CONFIDENCE)):
            try:
                enhanced_roi, method = image_enhancer.get_enhanced_roi(track_id)
                if enhanced_roi is not None:
                    enhanced_result = self.extract_plate_text(enhanced_roi)
                    enhanced_result["enhancement_method"] = method
                    # Si el resultado mejorado es mejor, usarlo
                    if enhanced_result.get("confidence", 0) > result.get("confidence", 0):
                        result = enhanced_result
            except Exception as e:
                logging.error(f"[OCR] Error en proceso de superresolución: {e}")
        
        # 4. Si está habilitado y configurado, usar OpenAI
        if use_openai and (not result or result.get("confidence", 0) < OCR_MIN_CONFIDENCE):
            try:
                from utils.openai_ocr import recognize_plate_text
                openai_result = recognize_plate_text(plate_image)
                if openai_result:
                    result = {**result, **openai_result, "ocr_method": "openai"}
            except Exception as e:
                logging.error(f"[OCR] Error en OpenAI OCR: {e}")
        
        return result

def apply_consensus_voting(results: List[Dict], min_length: int = CONSENSUS_MIN_LENGTH) -> Optional[Dict]:
    """
    Aplica un algoritmo de consenso para determinar el texto más probable
    entre múltiples resultados OCR de diferentes escalas/ángulos.
    
    Args:
        results: Lista de resultados OCR (Dict) de diferentes escalas
        min_length: Longitud mínima para incluir una lectura en el consenso
        
    Returns:
        El resultado con el mejor consenso, o None si no hay consenso
    """
    if not results:
        return None
        
    # Filtrar resultados vacíos o muy cortos
    valid_results = []
    for r in results:
        text = r.get("ocr_text", "").strip()
        if len(text) >= min_length:
            valid_results.append(r)
            
    if not valid_results:
        return None
        
    # Si solo hay un resultado válido, devolverlo directamente
    if len(valid_results) == 1:
        return valid_results[0]
        
    # Contar frecuencias de cada texto
    text_counts = {}
    for r in valid_results:
        text = r.get("ocr_text", "").strip()
        if text not in text_counts:
            text_counts[text] = []
        text_counts[text].append(r)
        
    # Encontrar el texto más frecuente
    max_count = 0
    consensus_text = None
    consensus_results = []
    
    for text, entries in text_counts.items():
        if len(entries) > max_count:
            max_count = len(entries)
            consensus_text = text
            consensus_results = entries
            
    # Si no hay consenso claro (igual número de ocurrencias), usar el de mayor confianza
    if max_count <= 1:
        return max(valid_results, key=lambda r: r.get("confidence", 0.0))
        
    # Entre los resultados con el mismo texto, elegir el de mayor confianza
    best_result = max(consensus_results, key=lambda r: r.get("confidence", 0.0))
    
    # Calcular confianza promedio para este consenso
    avg_conf = sum(r.get("confidence", 0.0) for r in consensus_results) / len(consensus_results)
    
    # Añadir información de consenso al resultado
    result = dict(best_result)
    result["consensus_count"] = max_count
    result["total_results"] = len(valid_results)
    result["average_conf"] = avg_conf
    
    return result

def consensus_by_positions(results: List[Dict]) -> Optional[str]:
    """
    Aplica un algoritmo de consenso por posiciones para determinar
    el texto más probable caracter por caracter.
    
    Args:
        results: Lista de resultados OCR
        
    Returns:
        Texto consensuado por posiciones, o None si no hay consenso
    """
    if not results:
        return None
        
    # Determinar la longitud esperada de la placa
    if CONSENSUS_EXPECTED_LENGTH_METHOD == 'fixed' and CONSENSUS_FIXED_LENGTH is not None:
        expected_length = CONSENSUS_FIXED_LENGTH
    else:
        # Extraer longitudes de cada resultado
        lengths = [len(r.get("ocr_text", "").strip()) for r in results]
        if not lengths:
            return None
            
        if CONSENSUS_EXPECTED_LENGTH_METHOD == 'mode':
            # Calcular la moda (valor más frecuente)
            counts = {}
            for l in lengths:
                counts[l] = counts.get(l, 0) + 1
            expected_length = max(counts.items(), key=lambda x: x[1])[0]
        else:  # 'median'
            # Calcular la mediana
            sorted_lengths = sorted(lengths)
            mid = len(sorted_lengths) // 2
            if len(sorted_lengths) % 2 == 0:
                expected_length = int((sorted_lengths[mid-1] + sorted_lengths[mid]) / 2)
            else:
                expected_length = sorted_lengths[mid]
    
    # Matriz para conteo de caracteres en cada posición
    char_counts = [{} for _ in range(expected_length)]
    
    # Contar ocurrencias de cada carácter en cada posición
    for r in results:
        text = r.get("ocr_text", "").strip()
        conf = r.get("confidence", 0.0)
        
        for pos, char in enumerate(text):
            if pos >= expected_length:
                break
                
            if char not in char_counts[pos]:
                char_counts[pos][char] = {"count": 0, "conf_sum": 0.0}
                
            char_counts[pos][char]["count"] += 1
            char_counts[pos][char]["conf_sum"] += conf
    
    # Determinar el carácter más probable para cada posición
    consensus = []
    for pos_counts in char_counts:
        if not pos_counts:
            consensus.append("?")  # Sin datos para esta posición
            continue
            
        # Encontrar el carácter con mayor conteo
        max_count = 0
        best_chars = []
        
        for char, data in pos_counts.items():
            if data["count"] > max_count:
                max_count = data["count"]
                best_chars = [char]
            elif data["count"] == max_count:
                best_chars.append(char)
                
        if len(best_chars) == 1:
            consensus.append(best_chars[0])
        else:
            # Desempate por confianza promedio
            best_char = None
            best_avg_conf = 0.0
            
            for char in best_chars:
                data = pos_counts[char]
                avg_conf = data["conf_sum"] / data["count"]
                
                if avg_conf > best_avg_conf:
                    best_avg_conf = avg_conf
                    best_char = char
                    
            consensus.append(best_char if best_char else "?")
    
    return "".join(consensus)

def final_consensus(results: List[Dict]) -> Optional[Dict]:
    """
    Combina los métodos de consenso para obtener el mejor resultado.
    
    Args:
        results: Lista de resultados OCR
        
    Returns:
        El mejor resultado consensuado, o None si no hay consenso
    """
    if not results:
        return None
        
    # Primero intentar consenso por votación
    voting_result = apply_consensus_voting(results)
    
    # Si el consenso por votación es fuerte (al menos 3 coincidencias), usarlo
    if voting_result and voting_result.get("consensus_count", 0) >= 3:
        return voting_result
        
    # Intentar consenso por posiciones
    pos_result = consensus_by_positions(results)
    
    if pos_result and is_valid_plate(pos_result):
        # Crear un resultado nuevo con el texto por posiciones
        best_conf = max(results, key=lambda r: r.get("confidence", 0.0))
        result = dict(best_conf)
        result["ocr_text"] = pos_result
        result["consensus_method"] = "by_positions"
        return result
        
    # Si el consenso por posiciones no es válido o no hay,
    # volver al consenso por votación si existe
    if voting_result:
        return voting_result
        
    # En último caso, devolver el resultado con mayor confianza
    if results:
        return max(results, key=lambda r: r.get("confidence", 0.0))
        
    return None

def process_ocr_result_detailed(results: List[Dict]) -> Dict:
    """
    Procesa los resultados OCR detalladamente, aplicando consenso
    y proveyendo información adicional sobre el proceso.
    
    Args:
        results: Lista de resultados OCR
        
    Returns:
        Dict con información detallada del procesamiento
    """
    # Aplicar consenso
    consensus = final_consensus(results)
    
    # Preparar resultado detallado
    detailed = {
        "timestamp": time.time(),
        "raw_results": results,
        "consensus": consensus,
        "valid_count": sum(1 for r in results if is_valid_plate(r.get("ocr_text", ""))),
        "total_count": len(results)
    }
    
    # Añadir texto y confianza del consenso si existe
    if consensus:
        detailed["text"] = consensus.get("ocr_text", "")
        detailed["confidence"] = consensus.get("confidence", 0.0)
        detailed["is_valid"] = is_valid_plate(detailed["text"])
    else:
        detailed["text"] = ""
        detailed["confidence"] = 0.0
        detailed["is_valid"] = False
        
    return detailed
