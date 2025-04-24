import os
import sys
import contextlib
import cv2
import numpy as np

# ---------------------------------------------------------------------
# FUNCIÓN PARA SUPRIMIR ERRORES DE DECODIFICACIÓN
# ---------------------------------------------------------------------
# Configuración para suprimir los mensajes de error de H264
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'
os.environ['AV_LOG_QUIET'] = '1'

@contextlib.contextmanager
def suppress_c_stderr():
    """
    Suprime mensajes de error en stderr durante la ejecución del bloque.
    Útil para eliminar mensajes de error de decodificación H264.
    """
    fd_stderr = sys.stderr.fileno()
    saved = os.dup(fd_stderr)
    devnull = os.open(os.devnull, os.O_RDWR)
    try:
        os.dup2(devnull, fd_stderr)
        yield
    finally:
        os.dup2(saved, fd_stderr)
        os.close(devnull)
        os.close(saved)

# Redirección permanente de stderr para FFmpeg
if hasattr(cv2, 'setLogLevel'):
    cv2.setLogLevel(0)  # Silencia los logs de OpenCV

def open_stream_with_suppressed_stderr(source):
    """
    Abre un cv2.VideoCapture silenciosamente (sin los logs H264 en stderr).
    
    Args:
        source: URL o path del video a abrir
        
    Returns:
        cv2.VideoCapture: El objeto capturador de video
    """
    with suppress_c_stderr():
        # Usar parámetros adicionales para reducir mensajes de error
        cap = cv2.VideoCapture(source, apiPreference=cv2.CAP_FFMPEG)
        if cap.isOpened():
            # Configurar opciones adicionales para minimizar mensajes de error
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Aumentar buffer
    return cap

"""
Algoritmos de supresión de detecciones redundantes
"""

def iou(box1, box2):
    """Calcula la intersección sobre unión (IoU) entre dos cajas delimitadoras"""
    # Convertir formato [x, y, w, h] a [x1, y1, x2, y2]
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
    
    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
    
    # Calcular área de intersección
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calcular áreas de cada caja
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    
    # Calcular IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    
    return iou

def non_max_suppression(boxes, scores, threshold=0.5):
    """Aplicar supresión no máxima a las detecciones"""
    if len(boxes) == 0:
        return []
    
    # Convertir a formato numpy para facilitar operaciones
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Ordenar por score
    idxs = np.argsort(scores)
    
    pick = []
    while len(idxs) > 0:
        # Agregar el índice con mayor score a la lista de resultados
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # Calcular IoU con las demás cajas
        overlap = np.array([iou(boxes[i], boxes[idxs[x]]) for x in range(last)])
        
        # Eliminar detectiones con IoU alto (similares a la seleccionada)
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))
    
    return boxes[pick].tolist()
