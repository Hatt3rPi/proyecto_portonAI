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
from datetime import datetime

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

    # Línea de estado en curso (antes de procesar)
    print(f"[QA] {fname} | Esperada: {expected_str} | Detectadas:        | ⏳ | Área:        px² | X:      | Tiempo:       | Res:         ", end="\r")
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
        rate = (len(success) / len(expected_plates) * 100) if expected_plates else 0.0

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
        print(f"[QA] {fname} | Esperada: {expected_str} | Detectadas: {detected_str} | {simbolo} | Área: {roi_area} px² | X: {roi_x} | Tiempo: {process_time_str} | Res: {resolution_str}")
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
            "resolution": resolution
        }

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
from collections import defaultdict

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

print(f"\n[QA] Resultados completos guardados en {output_path}")
