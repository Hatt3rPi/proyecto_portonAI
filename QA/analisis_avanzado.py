## archivo: analisis_avanzado.py

"""
Módulo de análisis avanzado para QA_mode que genera mapas de calor
de rendimiento OCR variando ángulos y escalas.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Configurar backend no interactivo
import matplotlib.pyplot as plt
import logging
import time
import subprocess
from typing import Dict, List, Tuple, Optional, Any
import math
import re
from difflib import SequenceMatcher

# Añadir directorio principal al path para importar módulos del sistema principal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar utilidades necesarias del sistema principal
from utils.image_processing import resize_for_inference, preprocess_frame, load_calibration_params
from utils.suppression import suppress_c_stderr
from config import CONFIANZA_PATENTE, DISPLAY_DURATION, OCR_STREAM_ZONE

def similar(a: str, b: str) -> float:
    """
    Calcula el porcentaje de similitud entre dos cadenas
    Args:
        a, b: Strings a comparar
    Returns:
        Porcentaje de similitud (0-100)
    """
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    
    # Comparación carácter a carácter para placas
    if len(a) == len(b):
        matches = sum(1 for c1, c2 in zip(a, b) if c1 == c2)
        return (matches / len(a)) * 100
    
    # Si longitudes son diferentes, usar SequenceMatcher
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


def ejecutar_main_para_extraer_roi(video_path: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Ejecuta el proceso main.py para analizar el video y obtener el frame y ROI utilizados
    para la detección exitosa de la placa.
    
    Args:
        video_path: Ruta al video a analizar
        
    Returns:
        Tupla con (frame, roi, bbox) donde:
        - frame: El frame completo donde se detectó la placa
        - roi: La región de interés (ROI) donde está la placa
        - bbox: Coordenadas [x1, y1, x2, y2] del bbox en el frame
    """
    # Crear directorio temporal para la salida
    debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug")
    temp_dir = os.path.join(debug_dir, f"qa_temp_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Preparar entorno para capturar los archivos generados
    env = os.environ.copy()
    env["QA_EXTRACT_MODE"] = "1"  # Variable para indicar modo de extracción
    env["QA_OUTPUT_DIR"] = temp_dir
    
    print(f"Ejecutando main.py para extraer ROI de {os.path.basename(video_path)}...")
    
    # Ejecutar el proceso main.py con los argumentos adecuados
    try:
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py"), 
               "--video_path", video_path, "--qa_extract"]
        
        process = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            env=env,
            timeout=60  # 60 segundos máximo para procesar
        )
        
        # Verificar la salida para encontrar los archivos generados
        if process.returncode != 0:
            logging.error(f"Error ejecutando main.py: {process.stderr}")
            raise Exception("Error en la ejecución de main.py")
        
        # Buscar archivos de salida (frame completo y ROI)
        frame_files = [f for f in os.listdir(temp_dir) if f.startswith("frame_") and f.endswith(".jpg")]
        roi_files = [f for f in os.listdir(temp_dir) if f.startswith("roi_") and f.endswith(".jpg")]
        bbox_file = os.path.join(temp_dir, "bbox.txt")
        
        if not frame_files or not roi_files or not os.path.exists(bbox_file):
            raise Exception("No se generaron los archivos esperados")
        
        # Leer los archivos
        frame = cv2.imread(os.path.join(temp_dir, frame_files[0]))
        roi = cv2.imread(os.path.join(temp_dir, roi_files[0]))
        
        with open(bbox_file, 'r') as f:
            bbox = [int(x) for x in f.read().strip().split(',')]
            
        # Limpiar archivos temporales pero mantener directorio para depuración
        for file in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass
        
        print(f"ROI extraído correctamente del video {os.path.basename(video_path)}")
        return frame, roi, bbox
        
    except Exception as e:
        logging.error(f"Error extrayendo ROI: {e}")
        
        # Fallback: extraer un frame central manualmente
        print(f"Utilizando método alternativo para extraer frames del video {os.path.basename(video_path)}...")
        
        # Abrir el video y extraer un frame central
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Intentar extraer frames en diferentes posiciones
        frames = []
        positions = [total_frames // 2, total_frames // 3, 2 * total_frames // 3]
        
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise Exception("No se pudo extraer ningún frame del video")
        
        # Usar el primer frame exitoso
        frame = frames[0]
        
        # Detectar la placa en el frame
        from models import ModelManager
        manager = ModelManager()
        model_plate = manager.get_plate_model()
        
        # Preprocesar frame para detección
        inference_frame, scale_x, scale_y = resize_for_inference(frame, max_dim=640)
        calib_params = load_calibration_params()
        inference_frame = preprocess_frame(inference_frame, calib_params)
        
        # Detectar placas
        results = model_plate.predict(inference_frame, device='cuda:0', verbose=False)
        
        # Encontrar la detección con mejor confianza
        best_box = None
        best_conf = 0
        
        for box in results[0].boxes:
            conf = float(box.conf[0]) * 100
            if conf >= CONFIANZA_PATENTE and conf > best_conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                x1h, y1h = int(x1 * scale_x), int(y1 * scale_y)
                x2h, y2h = int(x2 * scale_x), int(y2 * scale_y)
                best_box = [x1h, y1h, x2h, y2h]
                best_conf = conf
        
        if not best_box:
            # Último recurso: usar una región central como estimación
            h, w = frame.shape[:2]
            x1, y1 = w // 4, h // 3
            x2, y2 = 3 * w // 4, 2 * h // 3
            best_box = [x1, y1, x2, y2]
            
        # Extraer ROI
        x1, y1, x2, y2 = best_box
        roi = frame[y1:y2, x1:x2].copy()
        
        print(f"ROI extraído con método alternativo para {os.path.basename(video_path)}")
        return frame, roi, best_box


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
    print(f"Analizando video: {os.path.basename(video_path)}")
    print(f"Placa esperada: {placa_esperada}")
    
    # Extraer frame y ROI utilizando el mismo proceso del main.py
    try:
        frame, roi, bbox = ejecutar_main_para_extraer_roi(video_path)
        
        # Crear directorio específico para este video
        video_id = os.path.splitext(os.path.basename(video_name))[0]
        video_dir = os.path.join(output_dir, video_id)
        os.makedirs(video_dir, exist_ok=True)
        
        # Guardar frame completo con bbox y ROI original
        frame_with_rect = frame.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame_with_rect, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_with_rect, "ROI Original", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Guardar imágenes para referencia
        cv2.imwrite(os.path.join(video_dir, f"{video_id}_frame.jpg"), frame)
        cv2.imwrite(os.path.join(video_dir, f"{video_id}_frame_with_bbox.jpg"), frame_with_rect)
        cv2.imwrite(os.path.join(video_dir, f"{video_id}_roi_original.jpg"), roi)
        
        print(f"Frame y ROI extraídos y guardados en {video_dir}")
        print(f"Dimensiones del ROI original: {roi.shape[1]}x{roi.shape[0]}")
        
    except Exception as e:
        logging.error(f"Error al extraer frame/ROI: {e}")
        print("Fallback a método de extracción simple...")
        
        # Capturar ROI original del video como fallback
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"No se pudo abrir el video {video_path}")
            return {}
            
        # Leer frame central del video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logging.error(f"No se pudo leer el frame del video {video_path}")
            return {}
        
        # Crear directorios para resultados
        video_id = os.path.splitext(os.path.basename(video_name))[0]
        video_dir = os.path.join(output_dir, video_id)
        os.makedirs(video_dir, exist_ok=True)
        
        # Detección simple para obtener ROI
        h, w = frame.shape[:2]
        # Usar región central como ROI
        roi_x, roi_y = w // 4, h // 3
        roi_w, roi_h = w // 2, h // 3
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # Guardar imágenes para referencia
        cv2.imwrite(os.path.join(video_dir, f"{video_id}_frame.jpg"), frame)
        frame_with_rect = frame.copy()
        cv2.rectangle(frame_with_rect, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 2)
        cv2.putText(frame_with_rect, "ROI Fallback", (roi_x, roi_y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(video_dir, f"{video_id}_frame_with_bbox_fallback.jpg"), frame_with_rect)
        cv2.imwrite(os.path.join(video_dir, f"{video_id}_roi_original.jpg"), roi)
    
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
    
    # Crear directorio para muestras
    samples_dir = os.path.join(video_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
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
            
            # Guardar algunas muestras para depuración
            if i % 5 == 0 and j % 5 == 0:
                sample_path = os.path.join(samples_dir, f"rot_{angulo}_{escala}.jpg")
                cv2.imwrite(sample_path, roi_escalado)
            
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
                
                # Guardar muestras con alta similitud
                if sim_rot > 70 and (i % 3 == 0 and j % 3 == 0):
                    sample_path = os.path.join(samples_dir, f"rot_{angulo}_{escala}_sim{sim_rot:.0f}_{texto_rot}.jpg")
                    cv2.imwrite(sample_path, roi_escalado)
                
                # Actualizar máximo si es mejor
                if sim_rot > max_sim_rot:
                    max_sim_rot = sim_rot
                    max_params_rot = (angulo, escala)
                    max_text_rot = texto_rot
            
            # 4. Corrección de perspectiva
            roi_perspectiva, exito = corregir_distorsion(roi_escalado)
            
            # Guardar algunas muestras para depuración
            if exito and i % 5 == 0 and j % 5 == 0:
                sample_path = os.path.join(samples_dir, f"persp_{angulo}_{escala}.jpg")
                cv2.imwrite(sample_path, roi_perspectiva)
            
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
                    
                    # Guardar muestras con alta similitud
                    if sim_persp > 70 and (i % 3 == 0 and j % 3 == 0):
                        sample_path = os.path.join(samples_dir, f"persp_{angulo}_{escala}_sim{sim_persp:.0f}_{texto_persp}.jpg")
                        cv2.imwrite(sample_path, roi_perspectiva)
                    
                    # Actualizar máximo si es mejor
                    if sim_persp > max_sim_persp:
                        max_sim_persp = sim_persp
                        max_params_persp = (angulo, escala)
                        max_text_persp = texto_persp
    
    print("\n")  # Salto de línea después de la barra de progreso
    
    # Guardar los mejores resultados
    if max_sim_rot > 0:
        ang_best, esc_best = max_params_rot
        roi_rotado = corregir_rotacion(roi, ang_best)
        factor = esc_best / 100.0
        roi_best = cv2.resize(roi_rotado, None, fx=factor, fy=factor)
        best_path = os.path.join(video_dir, f"{video_id}_best_rot_sim{max_sim_rot:.0f}.jpg")
        cv2.imwrite(best_path, roi_best)
        
    if max_sim_persp > 0:
        ang_best, esc_best = max_params_persp
        roi_rotado = corregir_rotacion(roi, ang_best)
        factor = esc_best / 100.0
        roi_esc = cv2.resize(roi_rotado, None, fx=factor, fy=factor)
        roi_best, _ = corregir_distorsion(roi_esc)
        best_path = os.path.join(video_dir, f"{video_id}_best_persp_sim{max_sim_persp:.0f}.jpg")
        cv2.imwrite(best_path, roi_best)
    
    # Mostrar mejores resultados
    print(f"Mejor resultado con rotación: {max_text_rot} (similitud: {max_sim_rot:.1f}%)")
    print(f"  Parámetros: ángulo={max_params_rot[0]}°, escala={max_params_rot[1]}%")
    
    print(f"Mejor resultado con perspectiva: {max_text_persp} (similitud: {max_sim_persp:.1f}%)")
    print(f"  Parámetros: ángulo={max_params_persp[0]}°, escala={max_params_persp[1]}%")
    
    # Generar y guardar mapas de calor
    plt.figure(figsize=(12, 10))
    
    # Configurar rangos de visualización fijos para la escala de colores
    vmin = 0   # 0% similitud
    vmax = 100 # 100% similitud
    
    # Mapa de calor para rotación
    plt.subplot(2, 1, 1)
    plt.title(f"Mapa de calor - Rotación (max: {max_sim_rot:.1f}%)")
    im = plt.imshow(similitud_rotacion, cmap='hot', interpolation='nearest', 
                    origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Similitud (%)")
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
    im = plt.imshow(similitud_perspectiva, cmap='hot', interpolation='nearest', 
                    origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Similitud (%)")
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
    plt.savefig(os.path.join(video_dir, f"{video_id}_mapa_calor.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"{video_id}_mapa_calor.png"), dpi=300)  # Copia en directorio principal
    plt.close()
    
    # Guardar archivo resumen para este video
    with open(os.path.join(video_dir, f"{video_id}_resumen.txt"), "w") as f:
        f.write(f"RESUMEN DE ANÁLISIS PARA: {video_name}\n\n")
        f.write(f"Placa esperada: {placa_esperada}\n\n")
        f.write("=== RESULTADOS CON ROTACIÓN ===\n")
        f.write(f"Mejor texto detectado: {max_text_rot}\n")
        f.write(f"Similitud: {max_sim_rot:.1f}%\n")
        f.write(f"Ángulo óptimo: {max_params_rot[0]}°\n")
        f.write(f"Escala óptima: {max_params_rot[1]}%\n\n")
        f.write("=== RESULTADOS CON PERSPECTIVA ===\n")
        f.write(f"Mejor texto detectado: {max_text_persp}\n")
        f.write(f"Similitud: {max_sim_persp:.1f}%\n")
        f.write(f"Ángulo óptimo: {max_params_persp[0]}°\n")
        f.write(f"Escala óptima: {max_params_persp[1]}%\n")
    
    # Guardar datos para análisis agregado
    resultado = {
        "video": video_name,
        "placa_esperada": placa_esperada,
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
    
    # Configurar rangos de visualización fijos para la escala de colores
    vmin = 0   # 0% similitud
    vmax = 100 # 100% similitud
    
    # Mapa agregado para rotación
    plt.subplot(2, 1, 1)
    plt.title(f"Mapa de calor agregado - Rotación (max: {max_sim_rot:.1f}%)")
    im = plt.imshow(sim_rot_avg, cmap='hot', interpolation='nearest', 
                    origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Similitud promedio (%)")
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
    im = plt.imshow(sim_persp_avg, cmap='hot', interpolation='nearest', 
                    origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Similitud promedio (%)")
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
        
        # Tabla de resultados por video
        f.write("\n\n=== RESULTADOS INDIVIDUALES POR VIDEO ===\n\n")
        f.write(f"{'Video':30} {'Placa':10} {'Rotación':20} {'Perspectiva':20}\n")
        f.write("-" * 80 + "\n")
        
        for res in resultados:
            video_name = os.path.basename(res["video"])[:25]
            placa = res["placa_esperada"]
            rot_info = f"{res['max_sim_rot']:.1f}% ({res['max_params_rot'][0]}°, {res['max_params_rot'][1]}%)"
            persp_info = f"{res['max_sim_persp']:.1f}% ({res['max_params_persp'][0]}°, {res['max_params_persp'][1]}%)"
            f.write(f"{video_name:30} {placa:10} {rot_info:20} {persp_info:20}\n")
