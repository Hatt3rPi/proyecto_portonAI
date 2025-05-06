#!/usr/bin/env python3
"""
Módulo de mejoramiento de imágenes para PortonAI.
Utiliza RealESRGAN para mejorar la resolución y calidad de las imágenes de placas
cuando el OCR estándar falla.
"""

import os
import sys
import logging
import numpy as np
import cv2
import torch
import time
from typing import Tuple, Optional, Union, Any

# Directorio base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuración de rutas para RealESRGAN
REALESRGAN_DIR = os.path.join(os.path.dirname(BASE_DIR), "Real-ESRGAN")
REALESRGAN_MODEL_PATH = os.path.join(REALESRGAN_DIR, "weights")

# Agregar RealESRGAN al path si está disponible
if os.path.exists(REALESRGAN_DIR):
    sys.path.append(REALESRGAN_DIR)

# Variable global para el modelo RealESRGAN
realesrgan_model = None
realesrgan_available = False

# Función para inicializar el modelo RealESRGAN
def initialize_realesrgan() -> bool:
    """
    Inicializa el modelo RealESRGAN para mejoramiento de imágenes.
    
    Returns:
        bool: True si se inicializó correctamente, False en caso contrario
    """
    global realesrgan_model, realesrgan_available
    
    if realesrgan_model is not None:
        return True
    
    try:
        # Verificar que el directorio de RealESRGAN existe
        if not os.path.exists(REALESRGAN_DIR):
            logging.warning(f"El directorio de RealESRGAN no existe: {REALESRGAN_DIR}")
            
            # Intentar descargar e instalar RealESRGAN
            logging.info("Intentando descargar e instalar RealESRGAN...")
            
            # Crear el directorio para RealESRGAN
            os.makedirs(REALESRGAN_DIR, exist_ok=True)
            
            # Método 1: Clonar el repositorio desde GitHub
            import subprocess
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/xinntao/Real-ESRGAN.git", REALESRGAN_DIR],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Instalar dependencias
                subprocess.run(
                    ["pip", "install", "-r", os.path.join(REALESRGAN_DIR, "requirements.txt")],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Instalar basicsr y facexlib
                subprocess.run(
                    ["pip", "install", "basicsr", "facexlib", "gfpgan"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Instalar el paquete en modo desarrollo
                subprocess.run(
                    ["pip", "install", "-e", REALESRGAN_DIR],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                logging.info("RealESRGAN instalado correctamente desde GitHub")
                
            except subprocess.CalledProcessError:
                logging.error("Error al clonar el repositorio de RealESRGAN")
                
                # Método 2: Intentar instalar directamente desde pip
                try:
                    subprocess.run(
                        ["pip", "install", "realesrgan"],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    logging.info("RealESRGAN instalado correctamente desde pip")
                    
                except subprocess.CalledProcessError:
                    logging.error("Error al instalar RealESRGAN desde pip")
                    return False
        
        # Asegurarnos de que existe el directorio para los pesos
        os.makedirs(REALESRGAN_MODEL_PATH, exist_ok=True)
        
        # Importar RealESRGAN
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            # Definir el modelo
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            
            # Ruta al modelo pre-entrenado
            model_path = os.path.join(REALESRGAN_MODEL_PATH, "RealESRGAN_x4plus.pth")
            
            # Si no existe el modelo, intentar descargarlo
            if not os.path.exists(model_path):
                logging.info("Descargando modelo RealESRGAN_x4plus.pth...")
                
                # Método 1: Usar wget para descargar
                try:
                    subprocess.run(
                        ["wget", "-P", REALESRGAN_MODEL_PATH, 
                         "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    logging.info("Modelo descargado correctamente")
                    
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logging.warning("Error al descargar con wget, intentando con requests...")
                    
                    # Método 2: Usar requests para descargar
                    import requests
                    try:
                        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
                        response = requests.get(url, stream=True)
                        if response.status_code == 200:
                            with open(model_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            logging.info("Modelo descargado correctamente con requests")
                        else:
                            logging.error(f"Error al descargar modelo: HTTP {response.status_code}")
                            return False
                            
                    except Exception as e:
                        logging.error(f"Error al descargar modelo con requests: {e}")
                        return False
            
            # Verificar si se descargó el modelo correctamente
            if not os.path.exists(model_path):
                logging.error("No se pudo descargar el modelo")
                return False
            
            # Crear el objeto RealESRGANer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Inicializar el potenciador
            realesrgan_model = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=0,  # 0 para procesar la imagen completa sin dividirla en tiles
                tile_pad=10,
                pre_pad=0,
                half=True,  # Usar half precision para acelerar
                device=device
            )
            
            logging.info(f"RealESRGAN inicializado correctamente en dispositivo: {device}")
            realesrgan_available = True
            return True
            
        except ImportError as e:
            logging.error(f"Error al importar módulos de RealESRGAN: {e}")
            return False
            
    except Exception as e:
        logging.error(f"Error al inicializar RealESRGAN: {e}")
        return False

# Función principal para mejorar la imagen de una placa
def enhance_plate_image(
    plate_img: np.ndarray, 
    outscale: float = 2.0, 
    return_np_array: bool = True
) -> Tuple[np.ndarray, bool]:
    """
    Mejora la calidad de la imagen de una placa utilizando RealESRGAN.
    
    Args:
        plate_img: Imagen de la placa en formato NumPy
        outscale: Factor de escala de salida (por defecto 2x)
        return_np_array: Si es True, devuelve un array NumPy; si es False, devuelve tensor PyTorch
        
    Returns:
        Tupla con (imagen_mejorada, éxito)
    """
    global realesrgan_model, realesrgan_available
    
    # Si no tenemos el modelo inicializado, intentar inicializarlo
    if realesrgan_model is None:
        if not initialize_realesrgan():
            logging.error("No se pudo inicializar RealESRGAN")
            return plate_img.copy(), False
    
    # Verificar que la imagen no está vacía
    if plate_img is None or plate_img.size == 0:
        logging.error("La imagen de entrada está vacía")
        return plate_img.copy() if plate_img is not None else np.array([]), False
    
    try:
        # Obtener dimensiones de la imagen
        h, w = plate_img.shape[:2]
        
        # Redimensionar si la imagen es muy grande
        img = plate_img.copy()
        if h > 150 or w > 600:
            # Resize a un tamaño más manejable
            scale_factor = min(150 / h, 600 / w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            img = cv2.resize(img, (new_w, new_h))
            logging.info(f"Imagen redimensionada a {new_w}x{new_h} antes de mejorar")
        
        # Verificamos si la imagen es en escala de grises y la convertimos a BGR si es necesario
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        start_time = time.time()
        
        # Convertir de BGR a RGB para el modelo
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convertir a tensor
        img_tensor = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1))).float()
        img_tensor = img_tensor.unsqueeze(0) / 255.0
        
        # Pasar al dispositivo correcto
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        
        # Si estamos en half precision
        if realesrgan_model.half:
            img_tensor = img_tensor.half()
        
        # Procesar con el modelo
        with torch.no_grad():
            realesrgan_model.pre_process(img_tensor)
            if realesrgan_model.tile_size > 0:
                realesrgan_model.tile_process()
            else:
                realesrgan_model.process()
            enhanced_tensor = realesrgan_model.post_process()
        
        # Convertir de tensor a numpy
        enhanced_img = enhanced_tensor.squeeze().float().cpu().clamp_(0, 1).numpy()
        enhanced_img = np.transpose(enhanced_img, (1, 2, 0))
        enhanced_img = (enhanced_img * 255.0).round().astype(np.uint8)
        
        # Convertir de RGB a BGR
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
        
        end_time = time.time()
        process_time = (end_time - start_time) * 1000  # milisegundos
        
        logging.info(f"Imagen mejorada exitosamente - Tiempo: {process_time:.2f}ms")
        
        return enhanced_img, True
        
    except Exception as e:
        logging.error(f"Error al mejorar imagen con RealESRGAN: {e}")
        return plate_img.copy(), False


class ImageEnhancer:
    """
    Clase para mejorar imágenes de placas utilizando RealESRGAN.
    """
    
    def __init__(self):
        """
        Inicializa el mejorador de imágenes.
        """
        self.upsampler = None
        self.initialized = False
        self.init_model()
    
    def init_model(self) -> bool:
        """
        Inicializa el modelo RealESRGAN.
        
        Returns:
            bool: True si se inicializó correctamente, False en caso contrario
        """
        try:
            if initialize_realesrgan():
                self.upsampler = realesrgan_model
                self.initialized = True
                return True
            return False
        except Exception as e:
            logging.error(f"Error al inicializar ImageEnhancer: {e}")
            return False
    
    def enhance_image(self, img: np.ndarray, outscale: float = 2.0) -> Tuple[np.ndarray, bool]:
        """
        Mejora la calidad de una imagen utilizando RealESRGAN.
        
        Args:
            img: Imagen en formato numpy array
            outscale: Factor de escala (por defecto 2x)
            
        Returns:
            Tupla con (imagen mejorada, éxito)
        """
        if not self.initialized and not self.init_model():
            logging.error("No se pudo inicializar el modelo RealESRGAN")
            return img.copy(), False
        
        # Verificar que la imagen no está vacía
        if img is None or img.size == 0:
            logging.error("La imagen de entrada está vacía")
            return img.copy() if img is not None else np.array([]), False
        
        try:
            # Obtener dimensiones de la imagen
            h, w = img.shape[:2]
            
            # Redimensionar si la imagen es muy grande
            if h > 150 or w > 600:
                # Resize a un tamaño más manejable
                scale_factor = min(150 / h, 600 / w)
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                img = cv2.resize(img, (new_w, new_h))
                logging.info(f"Imagen redimensionada a {new_w}x{new_h} antes de mejorar")
            
            # Verificamos si la imagen es en escala de grises y la convertimos a BGR si es necesario
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            start_time = time.time()
            
            # Aplicar mejora con RealESRGAN
            img_enhanced, _ = enhance_plate_image(img, outscale=outscale)
            
            end_time = time.time()
            process_time = (end_time - start_time) * 1000  # milisegundos
            
            logging.info(f"Imagen mejorada exitosamente - Tiempo: {process_time:.2f}ms")
            
            return img_enhanced, True
            
        except Exception as e:
            logging.error(f"Error al mejorar imagen con RealESRGAN: {e}")
            return img, False


# Función de conveniencia para mejorar imágenes sin necesidad de instanciar la clase
_enhancer = None

def get_enhancer():
    """
    Obtiene una instancia singleton del mejorador de imágenes.
    
    Returns:
        ImageEnhancer: Instancia del mejorador de imágenes
    """
    global _enhancer
    if _enhancer is None:
        _enhancer = ImageEnhancer()
    return _enhancer