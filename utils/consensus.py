"""
Algoritmos de consenso para OCR de patentes vehiculares
"""

import numpy as np
import logging

class PlateConsensus:
    """Clase para generar consenso entre múltiples lecturas de patentes"""
    
    def __init__(self, max_history=10, threshold=0.7):
        self.history = collections.deque(maxlen=max_history)
        self.threshold = threshold
    
    def add_reading(self, plate_text, confidence):
        """Añade una nueva lectura a la historia"""
        self.history.append((plate_text, confidence))
    
    def get_consensus(self):
        """Obtiene la patente consensuada de las últimas lecturas"""
        if not self.history:
            return None, 0.0
        
        # Si solo hay una lectura, devuélvela
        if len(self.history) == 1:
            return self.history[0]
        
        # Contar ocurrencias ponderadas por confianza
        plate_scores = {}
        for plate, conf in self.history:
            plate_scores[plate] = plate_scores.get(plate, 0) + conf
        
        # Encontrar la patente con mayor puntuación
        best_plate = max(plate_scores.items(), key=lambda x: x[1])
        
        # Calcular confianza promedio para esta patente
        confidence = best_plate[1] / sum(1 for p, _ in self.history if p == best_plate[0])
        
        return best_plate[0], confidence

def apply_consensus_voting(ocr_results, min_length=5):
    """
    Aplica votación por consenso entre múltiples resultados OCR.
    
    Args:
        ocr_results: Lista de resultados OCR (dicts con ocr_text y global_conf)
        min_length: Longitud mínima de texto para considerar en el consenso
        
    Returns:
        dict: Resultado de consenso
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
        # Tomar el mejor resultado individual si no hay consenso
        best_res = max(ocr_results, key=lambda r: r["global_conf"])
        return {
            "ocr_text": best_res["ocr_text"].upper().strip(),
            "average_conf": best_res["global_conf"],
            "count": 1
        }
    
    # Encontrar mejor consenso por conteo y luego por confianza
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
    
    return {
        "ocr_text": best_text, 
        "average_conf": best_avg_conf, 
        "count": best_count
    }

def consensus_by_positions(ocr_results, expected_length=None):
    """
    Aplica consenso por posiciones, considerando cada posición de carácter individualmente.
    
    Args:
        ocr_results: Lista de resultados OCR (dicts con ocr_text y predictions)
        expected_length: Longitud esperada del texto (si es None, se calcula por mediana)
        
    Returns:
        dict: Resultado de consenso
    """
    valid_results = [res for res in ocr_results if res["ocr_text"].strip() != ""]
    
    if not valid_results:
        return None
    
    # Determinar longitud esperada
    lengths = [len(res["ocr_text"].strip()) for res in valid_results]
    if expected_length is None:
        expected_length = int(np.median(lengths))
    
    # Filtrar resultados con longitud correcta
    filtered = [res for res in valid_results if len(res["ocr_text"].strip()) == expected_length]
    
    if not filtered:
        return None
    
    # Inicializar estadísticas por posición
    letter_stats = [{} for _ in range(expected_length)]
    
    # Acumular estadísticas por cada resultado
    for res in filtered:
        for i, pred in enumerate(res["predictions"]):
            if i >= expected_length:
                break
                
            char = pred["char"].upper().strip()
            conf = pred["confidence"] * 100
            
            if char == "":
                continue
                
            if char not in letter_stats[i]:
                letter_stats[i][char] = {"count": 0, "total_conf": 0.0}
                
            letter_stats[i][char]["count"] += 1
            letter_stats[i][char]["total_conf"] += conf
    
    # Construir texto final
    consensus_text = ""
    conf_list = []
    
    for stats in letter_stats:
        if not stats:
            consensus_text += "?"
            conf_list.append(0.0)
        else:
            best_char = None
            best_count = -1
            best_avg = -1.0
            
            for char, info in stats.items():
                count = info["count"]
                avg_conf = info["total_conf"] / count
                
                if count > best_count or (count == best_count and avg_conf > best_avg):
                    best_char = char
                    best_count = count
                    best_avg = avg_conf
                    
            consensus_text += best_char if best_char is not None else "?"
            conf_list.append(best_avg)
    
    average_conf = np.mean(conf_list) if conf_list else 0.0
    return {"ocr_text": consensus_text, "average_conf": average_conf}

def final_consensus(consensus_list):
    """
    Combina múltiples consensos y obtiene resultado final.
    
    Args:
        consensus_list: Lista de diccionarios con resultados de consenso
        
    Returns:
        dict: Consenso final
    """
    if not consensus_list:
        return {"ocr_text": "", "average_conf": 0.0, "count": 0}

    # 1) Cuántas veces aparece cada texto
    freq = {}
    for c in consensus_list:
        txt = c.get("ocr_text", "")
        freq[txt] = freq.get(txt, 0) + 1

    # 2) Máxima frecuencia
    max_freq = max(freq.values())

    # 3) Textos con esa frecuencia
    candidates = [txt for txt, f in freq.items() if f == max_freq]

    # 4) Filtrar consensos que estén entre los candidatos
    tied_consensus = [c for c in consensus_list if c.get("ocr_text") in candidates]

    # 5) Elegir el de mayor average_conf
    best = max(tied_consensus, key=lambda c: c.get("average_conf", 0.0))

    return best
