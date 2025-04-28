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

# --- Cargar patentes esperadas ---
with open(PATENTES_JSON_PATH, "r") as f:
    mapping = json.load(f)
    for k, v in mapping.items():
        if isinstance(v, list):
            mapping[k] = {
                "patentes": v,
                "direccion": "desconocida",
                "marcha": "desconocida"
            }
            print(f"[QA] Corrigiendo formato antiguo para {k}")

# --- Añadir videos faltantes si es necesario ---
updated = False
for fname in sorted(os.listdir(VIDEOS_DIR)):
    if fname.lower().endswith(".dav") and fname not in mapping:
        mapping[fname] = {
            "patentes": ["TBC"],
            "direccion": "desconocida",
            "marcha": "desconocida"
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

    print(f"[QA] {fname} | Esperada: {expected_plates} | Detectadas:        | ⏳ | Área:        px² | X:     ", end="\r")
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

        # Comparar resultados
        success = [p for p in expected_plates if p in detected_plates]
        failed = [p for p in expected_plates if p not in detected_plates]
        rate = (len(success) / len(expected_plates) * 100) if expected_plates else 0.0
        # Formato de área y X fijo en longitud
        if roi_area is not None:
            roi_area = f"{roi_area:>6}"  # 6 caracteres, alineado a la derecha
        else:
            roi_area = "  N/A "

        if roi_x is not None:
            roi_x = f"{roi_x:>4}"  # 4 caracteres, alineado a la derecha
        else:
            roi_x = "N/A"

        # Definir símbolo
        if success:
            simbolo = "🟢"
        elif not detected_plates:
            simbolo = "⚫"
        else:
            simbolo = "🔴"

        detected_str = colorear_patente(detected_plates[0], expected_plates[0]) if detected_plates else "".ljust(6)

        # Mostrar resultado
        print(" " * 150, end="\r")
        print(f"[QA] {fname} | Esperada: {expected_plates} | Detectadas: {detected_str} | {simbolo} | Área: {roi_area or ' N/A '} px² | X: {roi_x or ' N/A'}")
        sys.stdout.flush()

        results[fname] = {
            "expected": expected_plates,
            "direccion": direccion,
            "marcha": marcha,
            "detected": detected_plates,
            "success": success,
            "failed": failed,
            "success_rate": rate,
            "roi_area": roi_area,
            "roi_x": roi_x
        }

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Fallo procesando {fname}: {e}")
        results[fname] = {
            "expected": expected_plates,
            "direccion": direccion,
            "marcha": marcha,
            "detected": [],
            "success": [],
            "failed": expected_plates,
            "success_rate": 0.0,
            "roi_area": None,
            "roi_x": None
        }

# --- Resumen global ---
total_expected = sum(len(d["expected"]) for d in results.values())
total_detected = sum(len(d["success"]) for d in results.values())
global_rate = (total_detected / total_expected * 100) if total_expected else 0.0

print(f"\n=== Resumen Global ===")
print(f"Total placas esperadas: {total_expected}")
print(f"Total placas detectadas: {total_detected}")
print(f"Tasa de éxito global: {global_rate:.2f}%")

# --- Guardar resultados en JSON ---
end_time = time.time()
duration_seconds = int(end_time - start_time)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"QA_{timestamp}_{total_detected}-{len(results)}_{duration_seconds}s.json"
output_path = os.path.join(RESULTADOS_DIR, output_filename)

with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"\n[QA] Resultados completos guardados en {output_path}")
