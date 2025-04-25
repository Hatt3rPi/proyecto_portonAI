# utils/__init__.py

"""
utils package for PortonAI.

Aquí centralizamos las utilidades de:
  - supresión de logs
  - procesamiento de imagen
  - tracking
  - snapshots HD
  - comunicación con APIs externas
  - OCR y estrategias de consenso
"""

# — Suppression utils —
from .suppression import (
    suppress_c_stderr,
    open_stream_with_suppressed_stderr,
)

# — Image processing utils —
from .image_processing import (
    resize_for_inference,
    preprocess_frame,
    correct_plate_orientation,
    process_image,
    load_calibration_params,
    save_calibration_params,
    is_frame_valid,
    calculate_roi_for_coverage,
)

# — Tracking utils —
from .tracking import (
    compute_iou,
    find_vehicle_type_for_plate,
    update_plate_area_history,
    compute_smoothed_rate_from_history,
    predict_time_to_threshold,
)

# — Snapshot utils —
from .snapshot import (
    SnapshotManager,
    fetch_hd_snapshot,
)

# — API utils —
from .api import (
    send_backend,
    send_plate_async,
)

# — OCR & Consensus utils —
from .ocr import (
    OCRProcessor,
    apply_consensus_voting,
    consensus_by_positions,
    final_consensus,
)

__all__ = [
    # suppression
    "suppress_c_stderr",
    "open_stream_with_suppressed_stderr",
    # image_processing
    "resize_for_inference",
    "preprocess_frame",
    "correct_plate_orientation",
    "process_image",
    "load_calibration_params",
    "save_calibration_params",
    "is_frame_valid",
    "calculate_roi_for_coverage",
    # tracking
    "compute_iou",
    "find_vehicle_type_for_plate",
    "update_plate_area_history",
    "compute_smoothed_rate_from_history",
    "predict_time_to_threshold",
    # snapshot
    "SnapshotManager",
    "fetch_hd_snapshot",
    # api
    "send_backend",
    "send_plate_async",
    # ocr & consensus
    "OCRProcessor",
    "apply_consensus_voting",
    "consensus_by_positions",
    "final_consensus",
]
