# utils/__init__.py

"""
Utilidades para el sistema PortonAI.

Este módulo inicializa todos los utilitarios requeridos por el sistema
principal, incluyendo funciones de preprocesamiento, manipulación de 
imágenes, OCR, tracking de placas y análisis avanzado para QA.
"""

# Importaciones básicas
import os
import sys
import time
import logging

# Importación de configuración necesaria para utils
try:
    from config import QA_ANALISIS_AVANZADO
except ImportError:
    QA_ANALISIS_AVANZADO = False

# Importación de utilidades de supresión de salidas de error
from .suppression import (
    suppress_c_stderr,
    open_stream_with_suppressed_stderr
)

# Importación de utilidades de API
from .api import (
    send_backend,
    send_plate_async
)

# Importación de utilidades para procesamiento de imágenes
from .image_processing import (
    resize_for_inference,
    preprocess_frame,
    load_calibration_params,
    is_frame_valid,
    calculate_roi_for_coverage
)

# Importación de utilidades para OCR
from .ocr import (
    is_valid_plate,
    apply_consensus_voting,
    consensus_by_positions,
    final_consensus,
    OCRProcessor,
    process_ocr_result_detailed
)

# Importación de utilidades para tracking híbrido
from .plate_tracker import (
    PlateTrackerManager,
    PlateTrackInstance,
    calculate_iou,
    is_in_exclusion_zone,
    filter_overlapping_by_size
)

# Importación de utilidades para snapshot
from .snapshot import (
    SnapshotManager
)

# Exponer variable de configuración para los módulos que la necesiten
__all__ = [
    'QA_ANALISIS_AVANZADO',
    
    'suppress_c_stderr', 'open_stream_with_suppressed_stderr',
    
    'send_backend', 'send_plate_async',
    
    'resize_for_inference', 'preprocess_frame', 'load_calibration_params',
    'is_frame_valid', 'calculate_roi_for_coverage',
    
    'is_valid_plate', 'apply_consensus_voting', 'consensus_by_positions',
    'final_consensus', 'OCRProcessor', 'process_ocr_result_detailed',
    
    'PlateTrackerManager', 'PlateTrackInstance', 'calculate_iou',
    'is_in_exclusion_zone', 'filter_overlapping_by_size',
    
    'SnapshotManager'
]
