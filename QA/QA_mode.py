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

# --- Configuración de rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videosQA")
PATENTES_JSON_PATH = os.path.join(BASE_DIR, "patentes_QA.json")
RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados")

# --- Asegurar existencia de carpetas críticas ---
if not os.path.exists(VIDEOS_DIR):
    print(f"[ERROR] No se encontró carpeta de videos: {VIDEOS_DIR}")
    sys.exit(1)

if not os.path.exists(RESULTADOS_DIR):
    os.makedirs(RESULTADOS_DIR)

# --- Inicializar el JSON de patentes si no existe o está vacío ---
if not os.path.exists(PATENTES_JSON_PATH) or os.path.getsize(PATENTES_JSON_PATH) == 0:
    print(f"[WARN] No se encontró {PATENTES_JSON_PATH} o está vacío, creando uno nuevo...")
    with open(PATENTES_JSON_PATH, "w") as f:
        json.dump({}, f, indent=4)

# --- Cargar patentes esperadas ---
with open(PATENTES_JSON_PATH, "r") as f:
    mapping = json.load(f)

# --- Revisión inicial: añadir videos faltantes con valores por defecto ---
updated = False
for fname in sorted(os.listdir(VIDEOS_DIR)):
    if not fname.lower().endswith(".dav"):
        continue
    if fname not in mapping:
        mapping[fname] = {
            "patentes": ["TBC"],
            "direccion": "desconocida",
            "marcha": "desconocida"
        }
        updated = True
        print(f"[QA] Añadido {fname} al JSON con valores por defecto.")

# Guardar el JSON si se añadieron entradas nuevas
if updated:
    with open(PATENTES_JSON_PATH, "w") as f:
        json.dump(mapping, f, indent=4)
    print(f"[QA] Se actualizó {PATENTES_JSON_PATH}")

# --- Procesamiento de videos ---
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

    print(f"[QA] Procesando {fname}... (Esperadas: {expected_plates})")

    try:
        # Llamar al main.py pasando el video como parámetro
        process = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "..", "main.py"),
             "--video_path", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        # Extraer las placas detectadas desde stdout
        output = process.stdout.strip()
        detected_plates = [line.strip() for line in output.splitlines() if line.strip()]

        # Comparación de resultados
        success = [p for p in expected_plates if p in detected_plates]
        failed = [p for p in expected_plates if p not in detected_plates]
        rate = (len(success) / len(expected_plates) * 100) if expected_plates else 0.0

        # Mostrar resultados en tiempo real
        print(f"[QA] {fname} | Detectadas: {detected_plates} | Éxito: {len(success)}/{len(expected_plates)} ({rate:.2f}%)")

        # Guardar resultados
        results[fname] = {
            "expected": expected_plates,
            "direccion": direccion,
            "marcha": marcha,
            "detected": detected_plates,
            "success": success,
            "failed": failed,
            "success_rate": rate
        }

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Fallo procesando {fname}: {e}")
        results[fname] = {
            "expected": expected_plates,
            "direccion": direccion,
            "marcha": marcha,
            "detected": [],
            "success": [],
            "failed": expected_plates,
            "success_rate": 0.0
        }

# --- Resumen por video ---
print("\n=== Resultados por video ===")
for video, data in results.items():
    print(f"{video}: {len(data['success'])}/{len(data['expected'])} placas detectadas"
          f" ({data['success_rate']:.2f}%)")
    if data["failed"]:
        print(f"   → No detectadas: {', '.join(data['failed'])}")

# --- Métrica global ---
total_expected = sum(len(d["expected"]) for d in results.values())
total_detected = sum(len(d["success"]) for d in results.values())
global_rate = (total_detected / total_expected * 100) if total_expected else 0.0

print(f"\n=== Resumen Global ===")
print(f"Total placas esperadas: {total_expected}")
print(f"Total placas detectadas: {total_detected}")
print(f"Tasa de éxito global: {global_rate:.2f}%")

# --- Guardar resultados en JSON ---
output_path = os.path.join(RESULTADOS_DIR, "resultado_QA.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"\n[QA] Resultados completos guardados en {output_path}")
