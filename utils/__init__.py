"""
Inicialización del paquete utils.
Este archivo permite que Python reconozca el directorio como un paquete.
"""

# Importaciones comunes para hacer más fácil el acceso a los submodulos
from utils.image_processing import (
    resize_for_inference, preprocess_frame, correct_plate_orientation,
    process_image, load_calibration_params, save_calibration_params,
    is_frame_valid, calculate_roi_for_coverage
)

from utils.tracking import (
    compute_iou, find_vehicle_type_for_plate, 
    update_plate_area_history, compute_smoothed_rate_from_history,
    predict_time_to_threshold
)

from utils.suppression import (
    open_stream_with_suppressed_stderr, suppress_c_stderr
)

from utils.snapshot import SnapshotManager, fetch_hd_snapshot

from utils.ocr import OCRProcessor

from utils.api import send_backend, send_plate_async

from utils.consensus import (
    apply_consensus_voting, consensus_by_positions, final_consensus
)
