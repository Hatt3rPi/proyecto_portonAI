## archivo: analisis_avanzado.py

"""
Módulo de análisis avanzado para QA_mode que genera mapas de calor
de rendimiento OCR variando ángulos y escalas.
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Configurar backend no interactivo
import matplotlib.pyplot as plt
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
import math
import re
from difflib import SequenceMatcher

def similar(a: str, b: str) -> float:
    """
    Calcula el porcentaje de similitud entre dos cadenas
    Args:
        a, b: Strings a comparar
    Returns:
        Porcentaje de similitud (0-100)
    """
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    
    # Usar SequenceMatcher para strings cortos
    similarity = SequenceMatcher(None, a, b).ratio() * 100
    return similarity


def corregir_rotacion(img: np.ndarray, angulo: float) -> np.ndarray:
    """
    Rota una imagen por el ángulo especificado
    Args:
        img: Imagen a rotar
        angulo: Ángulo en grados
    Returns:
        Imagen rotada
    """
    if angulo == 0:
        return img.copy()
        
    alto, ancho = img.shape[:2]
    centro = (ancho // 2, alto // 2)
    
    # Matriz de rotación
    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    
    # Aplicar rotación
    img_rotada = cv2.warpAffine(img, M, (ancho, alto), 
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))
    return img_rotada


def corregir_distorsion(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Intenta corregir la distorsión de perspectiva de una placa
    Args:
        img: Imagen de la placa
    Returns:
        Tuple con (imagen corregida, éxito)
    """
    try:
        # Convertir a escala de grises
        if len(img.shape) > 2:
            gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gris = img.copy()
            
        # Aplicar threshold adaptativo
        thresh = cv2.adaptiveThreshold(
            gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2)
            
        # Encontrar contornos
        contornos, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        if not contornos:
            return img.copy(), False
            
        # Encontrar el contorno más grande
        c = max(contornos, key=cv2.contourArea)
        
        # Aproximar polígono
        epsilon = 0.02 * cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, epsilon, True)
        
        # Si no se aproxima a un rectángulo
        if len(aprox) < 4:
            return img.copy(), False
            
        # Ordenar puntos del contorno
        puntos = aprox.reshape(len(aprox), 2)
        
        # Encontrar los 4 puntos extremos
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Superior izquierdo: menor suma
        s = puntos.sum(axis=1)
        rect[0] = puntos[np.argmin(s)]
        
        # Inferior derecho: mayor suma
        rect[2] = puntos[np.argmax(s)]
        
        # Superior derecho: menor resta
        diff = np.diff(puntos, axis=1)
        rect[1] = puntos[np.argmin(diff)]
        
        # Inferior izquierdo: mayor resta
        rect[3] = puntos[np.argmax(diff)]
        
        # Dimensiones de la imagen destino
        ancho, alto = 200, 50  # Típica relación para placas chilenas
        
        # Puntos destino
        dst = np.array([
            [0, 0],          # Superior izquierdo
            [ancho-1, 0],    # Superior derecho
            [ancho-1, alto-1], # Inferior derecho
            [0, alto-1]      # Inferior izquierdo
        ], dtype=np.float32)
        
        # Matriz de transformación
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Aplicar transformación
        corregida = cv2.warpPerspective(img, M, (ancho, alto))
        
        return corregida, True
        
    except Exception as e:
        logging.warning(f"Error en corrección de distorsión: {e}")
        return img.copy(), False


def generar_mapa_calor(
    video_path: str,
    placa_esperada: str,
    ocr_processor,
    rango_angulos: Tuple[float, float],
    paso_angulo: float,
    min_escala: int,
    max_escala: int,
    paso_escala: int,
    output_dir: str,
    video_name: str
) -> Dict[str, Any]:
    """
    Genera mapas de calor variando ángulo y escala
    Args:
        video_path: Ruta al video
        placa_esperada: Texto de la placa esperada
        ocr_processor: Procesador OCR
        rango_angulos, paso_angulo: Rango de ángulos y paso
        min_escala, max_escala, paso_escala: Rango de escalas y paso
        output_dir: Directorio de salida para guardar resultados
        video_name: Nombre del video para los archivos de salida
    Returns:
        Dict con matrices de datos para los mapas de calor
    """
    # Capturar ROI original del video
    cap = cv2.VideoCapture(video_path)
    print(f"Analizando video: {os.path.basename(video_path)}")
    print(f"Placa esperada: {placa_esperada}")
    
    # Leer frame central del video (asumiendo que ahí está la placa)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logging.error(f"No se pudo leer el frame del video {video_path}")
        return {}
    
    # Detectar placa en el frame
    # Aquí simplificamos para usar una región fija - en la práctica usaríamos el detector de placas
    h, w = frame.shape[:2]
    roi_x, roi_y = w // 4, h // 3
    roi_w, roi_h = w // 2, h // 3
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    # Crear directorios para resultados
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    video_id = os.path.splitext(video_name)[0]
    
    # Guardar ROI original para referencia
    cv2.imwrite(os.path.join(output_dir, f"{video_id}_roi_original.jpg"), roi)
    
    # Configurar rango de ángulos y escalas
    angulos = np.arange(rango_angulos[0], rango_angulos[1] + paso_angulo, paso_angulo)
    escalas = np.arange(min_escala, max_escala + paso_escala, paso_escala)
    
    # Matrices para almacenar resultados
    n_angulos = len(angulos)
    n_escalas = len(escalas)
    
    # Resultados para ROI con rotación
    similitud_rotacion = np.zeros((n_angulos, n_escalas))
    textos_rotacion = np.empty((n_angulos, n_escalas), dtype=object)
    conf_rotacion = np.zeros((n_angulos, n_escalas))
    
    # Resultados para ROI con rotación + corrección de perspectiva
    similitud_perspectiva = np.zeros((n_angulos, n_escalas))
    textos_perspectiva = np.empty((n_angulos, n_escalas), dtype=object)
    conf_perspectiva = np.zeros((n_angulos, n_escalas))
    
    print(f"Analizando {n_angulos} ángulos × {n_escalas} escalas = {n_angulos * n_escalas} combinaciones")
    total_iter = n_angulos * n_escalas
    iter_actual = 0
    
    # Máximas similitudes encontradas
    max_sim_rot = 0
    max_params_rot = (0, 100)
    max_text_rot = ""
    
    max_sim_persp = 0
    max_params_persp = (0, 100)
    max_text_persp = ""
    
    # Iterar sobre todas las combinaciones de ángulos y escalas
    for i, angulo in enumerate(angulos):
        for j, escala in enumerate(escalas):
            iter_actual += 1
            print(f"Progreso: {iter_actual}/{total_iter} ({iter_actual/total_iter*100:.1f}%)", 
                  end='\r', flush=True)
            
            # 1. Rotación del ROI
            roi_rotado = corregir_rotacion(roi, angulo)
            
            # 2. Escalado
            factor = escala / 100.0
            roi_escalado = cv2.resize(roi_rotado, None, fx=factor, fy=factor)
            
            # 3. OCR en ROI con rotación
            resultados_rot = ocr_processor.process_multiscale(roi_escalado)
            if resultados_rot:
                mejor = max(resultados_rot, key=lambda x: x.get('confidence', 0))
                texto_rot = mejor.get('ocr_text', '')
                conf_rot = mejor.get('confidence', 0)
                # Calcular similitud con placa esperada
                sim_rot = similar(texto_rot, placa_esperada)
                
                similitud_rotacion[i, j] = sim_rot
                textos_rotacion[i, j] = texto_rot
                conf_rotacion[i, j] = conf_rot
                
                # Actualizar máximo si es mejor
                if sim_rot > max_sim_rot:
                    max_sim_rot = sim_rot
                    max_params_rot = (angulo, escala)
                    max_text_rot = texto_rot
            
            # 4. Corrección de perspectiva
            roi_perspectiva, exito = corregir_distorsion(roi_escalado)
            
            # 5. OCR en ROI con rotación + perspectiva
            if exito:
                resultados_persp = ocr_processor.process_multiscale(roi_perspectiva)
                if resultados_persp:
                    mejor = max(resultados_persp, key=lambda x: x.get('confidence', 0))
                    texto_persp = mejor.get('ocr_text', '')
                    conf_persp = mejor.get('confidence', 0)
                    # Calcular similitud con placa esperada
                    sim_persp = similar(texto_persp, placa_esperada)
                    
                    similitud_perspectiva[i, j] = sim_persp
                    textos_perspectiva[i, j] = texto_persp
                    conf_perspectiva[i, j] = conf_persp
                    
                    # Actualizar máximo si es mejor
                    if sim_persp > max_sim_persp:
                        max_sim_persp = sim_persp
                        max_params_persp = (angulo, escala)
                        max_text_persp = texto_persp
    
    print("\n")  # Salto de línea después de la barra de progreso
    
    # Mostrar mejores resultados
    print(f"Mejor resultado con rotación: {max_text_rot} (similitud: {max_sim_rot:.1f}%)")
    print(f"  Parámetros: ángulo={max_params_rot[0]}°, escala={max_params_rot[1]}%")
    
    print(f"Mejor resultado con perspectiva: {max_text_persp} (similitud: {max_sim_persp:.1f}%)")
    print(f"  Parámetros: ángulo={max_params_persp[0]}°, escala={max_params_persp[1]}%")
    
    # Generar y guardar mapas de calor
    plt.figure(figsize=(12, 10))
    
    # Mapa de calor para rotación
    plt.subplot(2, 1, 1)
    plt.title(f"Mapa de calor - Rotación (max: {max_sim_rot:.1f}%)")
    plt.imshow(similitud_rotacion, cmap='hot', interpolation='nearest', 
               origin='lower', aspect='auto')
    plt.colorbar(label="Similitud (%)")
    plt.xlabel("Escala (%)")
    plt.ylabel("Ángulo (°)")
    plt.xticks(np.arange(n_escalas)[::max(1, n_escalas//10)], 
               escalas[::max(1, n_escalas//10)])
    plt.yticks(np.arange(n_angulos)[::max(1, n_angulos//10)], 
               angulos[::max(1, n_angulos//10)])
    
    # Marcar el mejor punto
    best_i_rot = np.where(angulos == max_params_rot[0])[0][0]
    best_j_rot = np.where(escalas == max_params_rot[1])[0][0]
    plt.plot(best_j_rot, best_i_rot, 'wo', markersize=10)
    
    # Mapa de calor para perspectiva
    plt.subplot(2, 1, 2)
    plt.title(f"Mapa de calor - Perspectiva (max: {max_sim_persp:.1f}%)")
    plt.imshow(similitud_perspectiva, cmap='hot', interpolation='nearest', 
               origin='lower', aspect='auto')
    plt.colorbar(label="Similitud (%)")
    plt.xlabel("Escala (%)")
    plt.ylabel("Ángulo (°)")
    plt.xticks(np.arange(n_escalas)[::max(1, n_escalas//10)], 
               escalas[::max(1, n_escalas//10)])
    plt.yticks(np.arange(n_angulos)[::max(1, n_angulos//10)], 
               angulos[::max(1, n_angulos//10)])
    
    # Marcar el mejor punto
    best_i_persp = np.where(angulos == max_params_persp[0])[0][0]
    best_j_persp = np.where(escalas == max_params_persp[1])[0][0]
    plt.plot(best_j_persp, best_i_persp, 'wo', markersize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{video_id}_mapa_calor.png"), dpi=300)
    plt.close()
    
    # Guardar datos para análisis agregado
    resultado = {
        "similitud_rotacion": similitud_rotacion,
        "textos_rotacion": textos_rotacion,
        "conf_rotacion": conf_rotacion,
        "similitud_perspectiva": similitud_perspectiva,
        "textos_perspectiva": textos_perspectiva,
        "conf_perspectiva": conf_perspectiva,
        "angulos": angulos,
        "escalas": escalas,
        "max_params_rot": max_params_rot,
        "max_sim_rot": max_sim_rot,
        "max_text_rot": max_text_rot,
        "max_params_persp": max_params_persp,
        "max_sim_persp": max_sim_persp,
        "max_text_persp": max_text_persp
    }
    
    return resultado


def generar_mapa_calor_agregado(resultados: List[Dict], output_dir: str):
    """
    Genera un mapa de calor agregado con el promedio de todos los videos
    Args:
        resultados: Lista de diccionarios con resultados de cada video
        output_dir: Directorio de salida para guardar resultados
    """
    if not resultados:
        print("No hay resultados para generar mapa de calor agregado")
        return
    
    # Obtener dimensiones de las matrices
    angulos = resultados[0]["angulos"]
    escalas = resultados[0]["escalas"]
    n_angulos = len(angulos)
    n_escalas = len(escalas)
    
    # Inicializar matrices promedio
    sim_rot_avg = np.zeros((n_angulos, n_escalas))
    sim_persp_avg = np.zeros((n_angulos, n_escalas))
    
    # Sumar todas las matrices
    for res in resultados:
        sim_rot_avg += res["similitud_rotacion"]
        sim_persp_avg += res["similitud_perspectiva"]
    
    # Calcular promedio
    n_videos = len(resultados)
    sim_rot_avg /= n_videos
    sim_persp_avg /= n_videos
    
    # Encontrar mejores parámetros
    max_idx_rot = np.unravel_index(np.argmax(sim_rot_avg), sim_rot_avg.shape)
    max_sim_rot = sim_rot_avg[max_idx_rot]
    max_ang_rot = angulos[max_idx_rot[0]]
    max_esc_rot = escalas[max_idx_rot[1]]
    
    max_idx_persp = np.unravel_index(np.argmax(sim_persp_avg), sim_persp_avg.shape)
    max_sim_persp = sim_persp_avg[max_idx_persp]
    max_ang_persp = angulos[max_idx_persp[0]]
    max_esc_persp = escalas[max_idx_persp[1]]
    
    # Generar mapas de calor agregados
    plt.figure(figsize=(12, 10))
    
    # Mapa agregado para rotación
    plt.subplot(2, 1, 1)
    plt.title(f"Mapa de calor agregado - Rotación (max: {max_sim_rot:.1f}%)")
    plt.imshow(sim_rot_avg, cmap='hot', interpolation='nearest', 
               origin='lower', aspect='auto')
    plt.colorbar(label="Similitud promedio (%)")
    plt.xlabel("Escala (%)")
    plt.ylabel("Ángulo (°)")
    plt.xticks(np.arange(n_escalas)[::max(1, n_escalas//10)], 
               escalas[::max(1, n_escalas//10)])
    plt.yticks(np.arange(n_angulos)[::max(1, n_angulos//10)], 
               angulos[::max(1, n_angulos//10)])
    
    # Marcar el mejor punto
    plt.plot(max_idx_rot[1], max_idx_rot[0], 'wo', markersize=10)
    
    # Mapa agregado para perspectiva
    plt.subplot(2, 1, 2)
    plt.title(f"Mapa de calor agregado - Perspectiva (max: {max_sim_persp:.1f}%)")
    plt.imshow(sim_persp_avg, cmap='hot', interpolation='nearest', 
               origin='lower', aspect='auto')
    plt.colorbar(label="Similitud promedio (%)")
    plt.xlabel("Escala (%)")
    plt.ylabel("Ángulo (°)")
    plt.xticks(np.arange(n_escalas)[::max(1, n_escalas//10)], 
               escalas[::max(1, n_escalas//10)])
    plt.yticks(np.arange(n_angulos)[::max(1, n_angulos//10)], 
               angulos[::max(1, n_angulos//10)])
    
    # Marcar el mejor punto
    plt.plot(max_idx_persp[1], max_idx_persp[0], 'wo', markersize=10)
    
    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(os.path.join(output_dir, f"agregado_{timestamp}_mapa_calor.png"), dpi=300)
    plt.close()
    
    # Mostrar resultados óptimos
    print("\n=== RESULTADOS ÓPTIMOS AGREGADOS ===")
    print(f"Mejor configuración con rotación:")
    print(f"  - Ángulo: {max_ang_rot:.2f}°")
    print(f"  - Escala: {max_esc_rot:.2f}%")
    print(f"  - Similitud promedio: {max_sim_rot:.1f}%")
    print()
    print(f"Mejor configuración con perspectiva:")
    print(f"  - Ángulo: {max_ang_persp:.2f}°")
    print(f"  - Escala: {max_esc_persp:.2f}%")
    print(f"  - Similitud promedio: {max_sim_persp:.1f}%")
    
    # Generar archivo de resumen
    with open(os.path.join(output_dir, f"parametros_optimos_{timestamp}.txt"), "w") as f:
        f.write("=== PARÁMETROS ÓPTIMOS PARA OCR ===\n\n")
        f.write(f"Análisis basado en {n_videos} videos\n\n")
        f.write("Configuración con rotación simple:\n")
        f.write(f"  - Ángulo: {max_ang_rot:.2f}°\n")
        f.write(f"  - Escala: {max_esc_rot:.2f}%\n")
        f.write(f"  - Similitud promedio: {max_sim_rot:.1f}%\n\n")
        f.write("Configuración con corrección de perspectiva:\n")
        f.write(f"  - Ángulo: {max_ang_persp:.2f}°\n")
        f.write(f"  - Escala: {max_esc_persp:.2f}%\n")
        f.write(f"  - Similitud promedio: {max_sim_persp:.1f}%\n\n")
        
        # Añadir recomendación
        mejor_metodo = "perspectiva" if max_sim_persp > max_sim_rot else "rotación simple"
        f.write(f"Recomendación: Usar corrección con {mejor_metodo}.\n")
        
        if mejor_metodo == "perspectiva":
            f.write(f"Parámetros recomendados: Ángulo={max_ang_persp:.2f}°, Escala={max_esc_persp:.2f}%\n")
        else:
            f.write(f"Parámetros recomendados: Ángulo={max_ang_rot:.2f}°, Escala={max_esc_rot:.2f}%\n")
