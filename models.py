#!/usr/bin/env python
"""
Carga y gestión de modelos de IA para PortonAI
"""

import logging
from ultralytics import YOLO
from config import MODELO_OBJETOS, MODELO_PATENTE, MODELO_OCR

class ModelManager:
    """
    Clase para manejar la carga y acceso a los modelos YOLO
    """
    _instance = None
    
    def __new__(cls):
        """Implementación de patrón Singleton para evitar múltiples instancias de modelos"""
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        logging.info("Inicializando modelos YOLO...")
        try:
            self.modelo_objetos = YOLO(MODELO_OBJETOS, task="detect", verbose=False)
            logging.info("Modelo de objetos cargado correctamente")
        except Exception as e:
            logging.error(f"Error al cargar modelo de objetos: {e}")
            self.modelo_objetos = None
            
        try:
            self.modelo_patentes = YOLO(MODELO_PATENTE, task="detect", verbose=False)
            logging.info("Modelo de patentes cargado correctamente")
        except Exception as e:
            logging.error(f"Error al cargar modelo de patentes: {e}")
            self.modelo_patentes = None
            
        try:
            self.modelo_ocr = YOLO(MODELO_OCR, task="detect", verbose=False)
            logging.info("Modelo OCR cargado correctamente")
        except Exception as e:
            logging.error(f"Error al cargar modelo OCR: {e}")
            self.modelo_ocr = None
            
        self._initialized = True
    
    def get_object_model(self):
        return self.modelo_objetos
        
    def get_plate_model(self):
        return self.modelo_patentes
        
    def get_ocr_model(self):
        return self.modelo_ocr
        
    def get_ocr_names(self):
        """Devuelve el mapeo de clases OCR"""
        if self.modelo_ocr:
            return self.modelo_ocr.names
        return []
