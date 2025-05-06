## archivo: QA_mode.py
#!/usr/bin/env python
"""
QA_mode.py - Modo de validación automática de detección de placas en PortonAI
Este script recorre todos los videos almacenados en 'videosQA/',
los procesa usando el motor principal (main.py) y compara las
placas detectadas con las patentes esperadas definidas en 'patentes_QA.json'.
Además, guarda los resultados en la carpeta 'resultados/'.
"""

import os
import subprocess
import json
import sys
import time
import re
from datetime import datetime
from collections import defaultdict

# Añadir directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Importaciones para análisis avanzado ---
import numpy as np
import logging

# Configurar el manejo de logging cuando QA_ANALISIS_AVANZADO está activo
try:
    from config import (
        QA_ANALISIS_AVANZADO,
        QA_ANGULOS_VARIACION,
        QA_ESCALAS_VARIACION
    )
    
    # Si el análisis avanzado está activado, configurar el logger para filtrar mensajes OCR-MULTIESCALA
    if QA_ANALISIS_AVANZADO:
        # Crear un filtro personalizado para eliminar los mensajes OCR-MULTIESCALA
        class OcrMultiescalaFilter(logging.Filter):
            def filter(self, record):
                # Si el mensaje contiene [OCR-MULTIESCALA], no lo muestra en consola
                return "[OCR-MULTIESCALA]" not in record.getMessage()
                
        # Aplicar el filtro a todos los handlers de tipo StreamHandler (consola)
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.addFilter(OcrMultiescalaFilter())
        
        # Crear un handler específico para archivo si queremos guardar esos mensajes
        file_handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), "ocr_multiescala.log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(file_handler)
        
except ImportError:
    print("[WARN] No se pudo importar configuración de análisis avanzado, usando valores por defecto.")
    QA_ANALISIS_AVANZADO = False
    QA_ANGULOS_VARIACION = ((-5, 10), 0.5)
    QA_ESCALAS_VARIACION = (50, 150, 5)

# Importar análisis avanzado condicionalmente
if QA_ANALISIS_AVANZADO:
    try:
        from QA.analisis_avanzado import (
            generar_mapa_calor,
            generar_mapa_calor_agregado,
        )
        # Importar el procesador OCR para análisis avanzado
        from models import ModelManager
        from utils.ocr import OCRProcessor
    except ImportError as e:
        print(f"[ERROR] No se pudieron importar módulos necesarios para análisis avanzado: {e}")
        print("[WARN] Desactivando análisis avanzado")
        QA_ANALISIS_AVANZADO = False

# --- Configuración de rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videosQA")
PATENTES_JSON_PATH = os.path.join(BASE_DIR, "patentes_QA.json")
RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados")

# --- Función para colorear patente caracter por caracter ---
def colorear_patente(detectada, esperada):
    COLOR_RESET = "\033[0m"
    COLOR_OK = "\033[32m"    # Verde
    COLOR_FAIL = "\033[31m"  # Rojo
    resultado = ""

    for i in range(max(len(detectada), len(esperada))):
        char_detectada = detectada[i] if i < len(detectada) else ""
        char_esperada = esperada[i] if i < len(esperada) else ""

        if char_detectada == char_esperada:
            resultado += f"{COLOR_OK}{char_detectada}{COLOR_RESET}"
        else:
            resultado += f"{COLOR_FAIL}{char_detectada}{COLOR_RESET}"

    return resultado

# --- Añadir modo análisis de consenso OCR para casos fallidos ---
def extract_ocr_details_from_logs(stderr_output):
    """
    Extrae detalles del OCR multiescala desde los logs cuando una detección falla
    """
    ocr_details = []
    in_multi_scale_section = False
    current_scale = None
    
    # Buscar líneas con información OCR multiescala
    for line in stderr_output.splitlines():
        # Detectar inicio de procesamiento multiescala
        if "[OCR-MULTIESCALA] Procesando escala" in line:
            match = re.search(r"escala (\d+)%", line)
            if match:
                current_scale = match.group(1)
                in_multi_scale_section = True
        
        # Capturar resultado de cada escala
        elif in_multi_scale_section and "[OCR-MULTIESCALA] Escala" in line:
            match = re.search(r"Escala (\d+)%: '([^']+)' válido=(\w+) conf=([0-9.]+)", line)
            if match:
                scale = match.group(1)
                text = match.group(2)
                valid = match.group(3) == "True"
                conf = float(match.group(4))
                ocr_details.append({
                    "escala": scale + "%",
                    "texto": text,
                    "valido": valid,
                    "confianza": conf
                })
        
        # Detectar fin del proceso multiescala
        elif in_multi_scale_section and "[OCR-MULTIESCALA] Terminado" in line:
            in_multi_scale_section = False
    
    return ocr_details

# --- Asegurar carpetas críticas ---
if not os.path.exists(VIDEOS_DIR):
    print(f"[ERROR] No se encontró carpeta de videos: {VIDEOS_DIR}")
    sys.exit(1)

if not os.path.exists(RESULTADOS_DIR):
    os.makedirs(RESULTADOS_DIR)

# --- Inicializar JSON de patentes si falta o vacío ---
if not os.path.exists(PATENTES_JSON_PATH) or os.path.getsize(PATENTES_JSON_PATH) == 0:
    print(f"[WARN] No se encontró {PATENTES_JSON_PATH} o está vacío. Creando...")
    with open(PATENTES_JSON_PATH, "w") as f:
        json.dump({}, f, indent=4)

# --- Cargar patentes esperadas y normalizar formato antiguo/dañado ---
with open(PATENTES_JSON_PATH, "r") as f:
    mapping = json.load(f)

normalized = False
for key, val in list(mapping.items()):
    # Lista antigua → dict con claves completas
    if isinstance(val, list):
        mapping[key] = {
            "patentes": val,
            "direccion": "desconocida",
            "marcha": "desconocida",
            "iluminacion":"desconocida"
        }
        normalized = True
        print(f"[QA] Corrigiendo formato antiguo (lista) para {key}")

    # Dict incompleto → añadir campos que falten
    elif isinstance(val, dict):
        entry = val.copy()
        changed = False
        if "patentes" not in entry or not isinstance(entry["patentes"], list):
            entry["patentes"] = entry.get("patentes", []) if isinstance(entry.get("patentes"), list) else []
            changed = True
        if "direccion" not in entry:
            entry["direccion"] = "desconocida"
            changed = True
        if "marcha" not in entry:
            entry["marcha"] = "desconocida"
            changed = True
        if "iluminacion" not in entry:
            entry["iluminacion"] = "desconocida"
            changed = True
        if changed:
            mapping[key] = entry
            normalized = True
            print(f"[QA] Añadiendo campos faltantes para {key}")

# Si hubo transformaciones, reescribimos el JSON
if normalized:
    with open(PATENTES_JSON_PATH, "w") as f:
        json.dump(mapping, f, indent=4)
    print(f"[QA] Se guardó {PATENTES_JSON_PATH} con formato normalizado.")
# --- Añadir videos faltantes si es necesario ---
updated = False
for fname in sorted(os.listdir(VIDEOS_DIR)):
    if fname.lower().endswith(".dav") and fname not in mapping:
        mapping[fname] = {
            "patentes": ["TBC"],
            "direccion": "desconocida",
            "marcha": "desconocida",
            "iluminacion":"desconocida"
        }
        updated = True
        print(f"[QA] Añadido {fname} al JSON.")

if updated:
    with open(PATENTES_JSON_PATH, "w") as f:
        json.dump(mapping, f, indent=4)
    print(f"[QA] Se actualizó {PATENTES_JSON_PATH}")

# --- Inicio procesamiento ---
start_time = time.time()
results = {}

print("\n=== Iniciando procesamiento de videos QA ===\n")

# Mostrar información sobre los parámetros de corrección de ROI que se utilizarán
print("-" * 70)
print(f"💡 PARÁMETROS DE CORRECCIÓN DE ROI:")
print(f"✓ Ángulo de rotación: {ROI_ANGULO_ROTACION:.1f}°")
print(f"✓ Factor de escala: {ROI_ESCALA_FACTOR:.2f} ({ROI_ESCALA_FACTOR*100:.0f}%)")
print(f"✓ Aplicar corrección: {'Activado' if ROI_APLICAR_CORRECCION else 'Desactivado'}")
print("-" * 70)

# Inicializar modelos OCR si se activa el análisis avanzado
if QA_ANALISIS_AVANZADO:
    # Calcular rangos y totales para mensajes informativos
    rango_angulos = QA_ANGULOS_VARIACION[0]
    paso_angulo = QA_ANGULOS_VARIACION[1]
    min_escala, max_escala, paso_escala = QA_ESCALAS_VARIACION
    
    n_angulos = len(np.arange(rango_angulos[0], rango_angulos[1] + paso_angulo, paso_angulo))
    n_escalas = len(np.arange(min_escala, max_escala + paso_escala, paso_escala))
    
    total_iteraciones = n_angulos * n_escalas
    
    print("=" * 80)
    print(f"ANÁLISIS AVANZADO ACTIVADO")
    print(f"- Rango de ángulos: desde {rango_angulos[0]}° hasta {rango_angulos[1]}° con paso de {paso_angulo}° ({n_angulos} iteraciones)")
    print(f"- Rango de escalas: desde {min_escala}% hasta {max_escala}% con paso de {paso_escala}% ({n_escalas} iteraciones)")
    print(f"- Total de etapas de análisis por video: {n_angulos} × {n_escalas} = {total_iteraciones}")
    print("=" * 80)
    print()
    
    print("Modo de análisis avanzado activado. Inicializando modelos...")
    manager = ModelManager()
    model_ocr = manager.get_ocr_model()
    ocr_processor = OCRProcessor(model_ocr, manager.get_ocr_names())
    resultados_analisis_avanzado = []
    
    # Crear directorio para mapas de calor
    mapas_calor_dir = os.path.join(RESULTADOS_DIR, f"mapas_calor_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(mapas_calor_dir, exist_ok=True)

for fname in sorted(os.listdir(VIDEOS_DIR)):
    if not fname.lower().endswith(".dav"):
        continue

    video_path = os.path.join(VIDEOS_DIR, fname)
    expected_info = mapping.get(fname, {})
    expected_plates = expected_info.get("patentes", ["TBC"])
    direccion = expected_info.get("direccion", "desconocida")
    marcha = expected_info.get("marcha", "desconocida")
    iluminacion = expected_info.get("iluminacion", "desconocida")

    # Convertimos la lista en cadena simple para impresión
    expected_str = ", ".join(expected_plates)

    # Obtener nombre corto del archivo (últimos 12 caracteres sin extensión)
    short_fname = fname[-16:-4] if len(fname) > 16 else fname[:-4]

    # Línea de estado en curso (antes de procesar)
    print(f"[QA] {short_fname} | Esperada: {expected_str} | Detectadas:        | ⏳ | Área:        px² | X:      | Tiempo:       | Res:         | Progreso: [          ] 0%", end="\r")
    sys.stdout.flush()

    try:
        process = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "..", "main.py"),
             "--video_path", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        output = process.stdout.strip()
        stderr_output = process.stderr.strip()
        detected_plates = []
        roi_area = None
        roi_x = None
        process_time = None
        resolution = None

        for line in output.splitlines():
            if line.strip().startswith("[PLACA]"):
                parts = line.replace("[PLACA]", "").split("|")
                plate_text = parts[0].strip()
                detected_plates.append(plate_text)

                for part in parts:
                    if "Área:" in part:
                        roi_area = int(part.replace("Área:", "").replace("px²", "").strip())
                    if "X inicial:" in part:
                        roi_x = int(part.replace("X inicial:", "").strip())
                    if "Tiempo detección a OCR:" in part:
                        process_time = int(part.replace("Tiempo detección a OCR:", "").replace("ms", "").strip())
                    if "Res:" in part:
                        resolution = part.replace("Res:", "").strip()

        # Comparar resultados
        success = [p for p in expected_plates if p in detected_plates]
        failed = [p for p in expected_plates if p not in detected_plates]
        false_positives = [p for p in detected_plates if p not in expected_plates]
        rate = (len(success) / len(expected_plates) * 100) if expected_plates else 0.0

        # Análisis detallado de multiescala solo para casos fallidos
        ocr_analysis = None
        if failed and detected_plates:  # Si falló la detección pero detectó algo
            ocr_analysis = extract_ocr_details_from_logs(stderr_output)

        # Formatear área y X para mostrar siempre ancho fijo
        roi_area = f"{roi_area:>6}" if roi_area is not None else "  N/A "
        roi_x    = f"{roi_x:>4}"    if roi_x    is not None else " N/A"

        # Formatear tiempo de procesamiento
        process_time_str = f"{process_time:>5} ms" if process_time is not None else "  N/A  "

        # Formatear resolución
        resolution_str = f"{resolution:>9}" if resolution is not None else "  N/A     "

        # Símbolo de estado
        if success:
            simbolo = "🟢"
        elif not detected_plates:
            simbolo = "⚫"
        else:
            simbolo = "🔴"

        detected_str = colorear_patente(detected_plates[0], expected_plates[0]) if detected_plates else "".ljust(len(expected_str))

        # Mostrar resultado final de este video
        print(" " * 150, end="\r")  # Limpiar línea anterior
        print(f"[QA] {short_fname} | Esperada: {expected_str} | Detectadas: {detected_str} | {simbolo} | Área: {roi_area} px² | X: {roi_x} | Tiempo: {process_time_str} | Res: {resolution_str} | Progreso: [          ] 0%")
        sys.stdout.flush()

        results[fname] = {
            "expected": expected_plates,
            "direccion": direccion,
            "marcha": marcha,
            "iluminacion": iluminacion,
            "detected": detected_plates,
            "success": success,
            "failed": failed,
            "success_rate": rate,
            "roi_area": roi_area.strip() if isinstance(roi_area, str) else roi_area,
            "roi_x": roi_x.strip() if isinstance(roi_x, str) else roi_x,
            "process_time": process_time,
            "resolution": resolution,
            "ocr_analysis": ocr_analysis if ocr_analysis else None,
            "has_match_in_scales": any(entry["texto"] == expected_plates[0] for entry in (ocr_analysis or []))
        }

        # Análisis avanzado si está activado y tenemos placa esperada válida
        if QA_ANALISIS_AVANZADO and expected_plates and expected_plates[0] != "TBC":
            # Configuramos una variable global para la línea de estado actual
            global current_status_line
            current_status_line = f"[QA] {short_fname} | Esperada: {expected_str} | Detectadas: {detected_str} | {simbolo} | " \
                                 f"Área: {roi_area} px² | X: {roi_x} | Tiempo: {process_time_str} | " \
                                 f"Res: {resolution_str}"
            
            # Iniciar con la barra de progreso en cero
            # Limpiar la línea anterior con espacios suficientes para cubrir cualquier línea anterior
            print(" " * 150, end="\r")
            sys.stdout.flush()
            
            # Usar caracteres de escape ANSI para limpiar la línea actual completamente
            print("\033[K", end="\r")
            sys.stdout.flush()
            
            # Inicializar la barra de progreso en 0%
            bar = ' ' * 10
            print(f"{current_status_line} | Progreso: [{bar}] 0% (0/0)", end="\r")
            sys.stdout.flush()
            
            # Supresión total de los mensajes de salida
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            devnull = open(os.devnull, 'w')
            
            # Obtener parámetros de variación para calcular el total de iteraciones
            rango_angulos = QA_ANGULOS_VARIACION[0]
            paso_angulo = QA_ANGULOS_VARIACION[1]
            min_escala, max_escala, paso_escala = QA_ESCALAS_VARIACION
            
            # Calcular el total de iteraciones (ángulos × escalas)
            n_angulos = len(np.arange(rango_angulos[0], rango_angulos[1] + paso_angulo, paso_angulo))
            n_escalas = len(np.arange(min_escala, max_escala + paso_escala, paso_escala))
            total_iter = n_angulos * n_escalas
            
            # Definir un callback que actualiza la barra de progreso con el total correcto
            def progress_callback(current_iter, total_iter):
                # Restaurar stdout solo para imprimir la actualización
                sys.stdout = old_stdout
                
                # Calcular el porcentaje basado en el progreso total (ángulos × escalas)
                percent = int(current_iter / total_iter * 100)
                bar_length = 10
                filled_len = int(bar_length * current_iter // total_iter)
                bar = '█' * filled_len + ' ' * (bar_length - filled_len)
                
                # Limpiar la línea completamente antes de mostrar el progreso actualizado
                print("\033[K", end="\r")
                
                # Mostrar el número actual de la etapa y el total
                print(f"{current_status_line} | Progreso: [{bar}] {percent}% ({current_iter}/{total_iter})", end="\r")
                sys.stdout.flush()
                
                # Volver a redirigir stdout a /dev/null
                sys.stdout = devnull
            
            try:
                # Redirigir la salida estándar y de error a /dev/null
                sys.stdout = devnull
                sys.stderr = devnull
                
                # Establecer el callback en el procesador OCR
                ocr_processor.set_progress_callback(progress_callback)
                
                # Generar mapa de calor en silencio
                resultado_analisis = generar_mapa_calor(
                    video_path,
                    expected_plates[0],
                    ocr_processor,
                    rango_angulos,
                    paso_angulo,
                    min_escala,
                    max_escala,
                    paso_escala,
                    mapas_calor_dir,
                    fname
                )
                
                # Almacenar resultado si existe
                if resultado_analisis:
                    resultados_analisis_avanzado.append(resultado_analisis)
            
            finally:
                # Restaurar stdout y stderr y cerrar el archivo nulo
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                devnull.close()
            
            # Al finalizar, actualizar la línea de progreso al 100% (exitoso) o quedarse en 0% (fallido)
            if resultado_analisis:
                # Limpiar la línea completamente antes de mostrar el resultado final
                sys.stdout = old_stdout
                print("\033[K", end="\r") 
                sys.stdout.flush()
                
                # Mostrar completado
                bar = '█' * 10
                print(f"{current_status_line} | Progreso: [{bar}] 100% ({total_iter}/{total_iter})")
            else:
                # Limpiar la línea completamente antes de mostrar el resultado final
                sys.stdout = old_stdout
                print("\033[K", end="\r") 
                sys.stdout.flush()
                
                # Mostrar error
                bar = ' ' * 10
                print(f"{current_status_line} | Progreso: [{bar}] 0% (0/{total_iter})")

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Fallo procesando {fname}: {e}")
        results[fname] = {
            "expected": expected_plates,
            "direccion": direccion,
            "marcha": marcha,
            "iluminacion": iluminacion,
            "detected": [],
            "success": [],
            "failed": expected_plates,
            "success_rate": 0.0,
            "roi_area": None,
            "roi_x": None,
            "process_time": None,
            "resolution": None
        }

# --- Resumen global ---
total_expected = sum(len(d["expected"]) for d in results.values())
total_detected = sum(len(d["success"]) for d in results.values())
global_rate = (total_detected / total_expected * 100) if total_expected else 0.0

# Cálculo tiempo promedio
times = [d["process_time"] for d in results.values() if d["process_time"] is not None]
average_time_ms = (sum(times) / len(times)) if times else 0

print(f"\n=== Resumen Global ===")
print(f"Total placas esperadas: {total_expected}")
print(f"Total placas detectadas: {total_detected}")
print(f"Tasa de éxito global: {global_rate:.2f}%")
print(f"Tiempo promedio detección a OCR: {average_time_ms:.2f} ms")

# --- Estadísticas detalladas ---
# Estadísticas por patente
print("\n=== Estadísticas Detalladas por Patente ===")
plate_stats = defaultdict(lambda: {'expected': 0, 'success': 0, 'times': []})
for info in results.values():
    for plate in info['expected']:
        plate_stats[plate]['expected'] += 1
        if plate in info['success']:
            plate_stats[plate]['success'] += 1
        if info['process_time'] is not None:
            plate_stats[plate]['times'].append(info['process_time'])

print(f"{'Patente':10} {'Total':5} {'Detectado':9} {'Tasa%':6} {'TiempoAvg(ms)':14}")
for plate, stats in plate_stats.items():
    tot  = stats['expected']
    succ = stats['success']
    rate = (succ / tot * 100) if tot else 0.0
    avgt = (sum(stats['times']) / len(stats['times'])) if stats['times'] else 0.0
    print(f"{plate:10} {tot:5} {succ:9} {rate:6.2f}% {avgt:14.2f}")

# Función de ayuda para cálculo por campo
def print_group_stats(field, title):
    print(f"\n=== Estadísticas por {title} ===")
    grp = defaultdict(lambda: {'expected': 0, 'success': 0, 'times': []})
    for info in results.values():
        key = info.get(field, 'desconocida')
        grp[key]['expected'] += len(info['expected'])
        grp[key]['success']   += len(info['success'])
        if info['process_time'] is not None:
            grp[key]['times'].append(info['process_time'])

    print(f"{title:15} {'Total':5} {'Detectado':9} {'Tasa%':6} {'TiempoAvg(ms)':14}")
    for key, stats in grp.items():
        tot  = stats['expected']
        succ = stats['success']
        rate = (succ / tot * 100) if tot else 0.0
        avgt = (sum(stats['times']) / len(stats['times'])) if stats['times'] else 0.0
        print(f"{key:15} {tot:5} {succ:9} {rate:6.2f}% {avgt:14.2f}")

# Estadísticas por dirección, marcha e iluminación
print_group_stats('direccion',   'Dirección')
print_group_stats('marcha',      'Marcha')
print_group_stats('iluminacion', 'Iluminación')

# Estadísticas por resolución
print("\n=== Estadísticas por Resolución ===")
res_stats = defaultdict(lambda: {'expected': 0, 'success': 0, 'times': [], 'areas': []})
for info in results.values():
    if info.get('resolution'):
        res = info['resolution']
        res_stats[res]['expected'] += len(info['expected'])
        res_stats[res]['success'] += len(info['success'])
        if info.get('process_time') is not None:
            res_stats[res]['times'].append(info['process_time'])
        if info.get('roi_area') is not None and not isinstance(info['roi_area'], str):
            res_stats[res]['areas'].append(info['roi_area'])

print(f"{'Resolución':12} {'Total':5} {'Detectado':9} {'Tasa%':6} {'TiempoAvg(ms)':14} {'ÁreaAvg(px²)':12}")
for res, stats in res_stats.items():
    tot  = stats['expected']
    succ = stats['success']
    rate = (succ / tot * 100) if tot else 0.0
    avgt = (sum(stats['times']) / len(stats['times'])) if stats['times'] else 0.0
    avga = (sum(stats['areas']) / len(stats['areas'])) if stats['areas'] else 0.0
    print(f"{res:12} {tot:5} {succ:9} {rate:6.2f}% {avgt:14.2f} {avga:12.2f}")

# --- Guardar resultados en JSON ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"QA_{timestamp}_{total_detected}-{len(results)}_{int(average_time_ms)}ms.json"
output_path = os.path.join(RESULTADOS_DIR, output_filename)

with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

# Crear informe detallado para análisis de consenso
consensus_analysis_file = os.path.join(RESULTADOS_DIR, f"consensus_analysis_{timestamp}.txt")
with open(consensus_analysis_file, "w") as f:
    f.write("=== ANÁLISIS DETALLADO DE CONSENSO OCR ===\n\n")
    
    # Analizar primero los casos donde la escala tenía el texto correcto pero el consenso falló
    missed_opportunities = [
        (fname, result) for fname, result in results.items()
        if result["failed"] and result["detected"] and 
           result["ocr_analysis"] and 
           any(entry["texto"] == result["expected"][0] for entry in result["ocr_analysis"])
    ]
    
    if missed_opportunities:
        f.write("### CASOS DONDE ALGUNA ESCALA DETECTÓ LA PLACA CORRECTA PERO EL CONSENSO FALLÓ ###\n\n")
        for fname, result in missed_opportunities:
            short_name = fname[-16:-4] if len(fname) > 16 else fname[:-4]
            f.write(f"Archivo: {short_name} (Completo: {fname})\n")
            f.write(f"Esperado: {', '.join(result['expected'])}\n")
            f.write(f"Detectado por consenso: {', '.join(result['detected'])}\n")
            f.write("Resultados multiescala:\n")
            f.write(f"{'Escala':8} {'Texto':15} {'Válido':8} {'Confianza':10} {'Coincide':10}\n")
            
            for entry in result["ocr_analysis"]:
                valid_symbol = "Sí" if entry["valido"] else "No"
                matches_expected = entry["texto"] == result["expected"][0]
                match_symbol = "SÍ" if matches_expected else ""
                if matches_expected:
                    f.write(f"{entry['escala']:8} {entry['texto']:15} {valid_symbol:8} {entry['confianza']:10.2f} {match_symbol:10} *** CORRECTA IGNORADA ***\n")
                else:
                    f.write(f"{entry['escala']:8} {entry['texto']:15} {valid_symbol:8} {entry['confianza']:10.2f} {match_symbol:10}\n")
            f.write("\n" + "-"*50 + "\n\n")
    
    # Luego mostrar todos los demás casos fallidos donde ninguna escala tenía el texto correcto
    normal_failures = [
        (fname, result) for fname, result in results.items()
        if result["failed"] and result["detected"] and 
           result["ocr_analysis"] and 
           not any(entry["texto"] == result["expected"][0] for entry in result["ocr_analysis"])
    ]
    
    if normal_failures:
        f.write("### OTROS CASOS FALLIDOS (NINGUNA ESCALA DETECTÓ LA PLACA CORRECTA) ###\n\n")
        for fname, result in normal_failures:
            short_name = fname[-16:-4] if len(fname) > 16 else fname[:-4]
            f.write(f"Archivo: {short_name} (Completo: {fname})\n")
            f.write(f"Esperado: {', '.join(result['expected'])}\n")
            f.write(f"Detectado: {', '.join(result['detected'])}\n")
            f.write("Resultados multiescala:\n")
            f.write(f"{'Escala':8} {'Texto':15} {'Válido':8} {'Confianza':10}\n")
            
            for entry in result["ocr_analysis"]:
                valid_symbol = "Sí" if entry["valido"] else "No"
                f.write(f"{entry['escala']:8} {entry['texto']:15} {valid_symbol:8} {entry['confianza']:10.2f}\n")
            f.write("\n" + "-"*50 + "\n\n")

# Añadir estadísticas de oportunidades perdidas
missed_opportunities_count = sum(1 for fname, r in results.items() 
                                if mapping.get(fname, {}).get("incluir_qa", True) and
                                   r.get("has_match_in_scales", False) and r["failed"])

# Determinar total de casos fallidos que fueron incluidos en QA
total_failed_cases = len([r for fname, r in results.items() 
                         if mapping.get(fname, {}).get("incluir_qa", True) and
                            r["failed"] and r["detected"]])

if missed_opportunities_count > 0:
    print(f"\n=== Análisis de oportunidades perdidas ===")
    print(f"Detecciones donde alguna escala tenía el texto correcto pero el consenso falló: {missed_opportunities_count}")
    if total_failed_cases > 0:
        print(f"Esto representa un {missed_opportunities_count/total_failed_cases*100:.2f}% de los casos fallidos")
    print(f"Ver detalles completos en {consensus_analysis_file}")

print(f"\n[QA] Resultados completos guardados en {output_path}")
print(f"[QA] Análisis detallado de consenso guardado en {consensus_analysis_file}")

# Generar mapa de calor agregado si hay suficientes resultados
if QA_ANALISIS_AVANZADO and resultados_analisis_avanzado:
    print(f"\n=== Generando mapa de calor agregado con {len(resultados_analisis_avanzado)} videos ===")
    generar_mapa_calor_agregado(resultados_analisis_avanzado, mapas_calor_dir)
    print(f"[QA-AVANZADO] Mapas de calor y parámetros óptimos guardados en {mapas_calor_dir}")
