"""
Script para usar la API de OpenAI para reconocimiento de patentes
"""

import os
import sys
import argparse
import requests
import json
import cv2
import base64
from PIL import Image
import io

# Añadir el directorio raíz al path para poder importar los módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.image_processing import resize_image, preprocess_for_plate_detection
from utils.snapshot import save_plate_snapshot

def encode_image_to_base64(image_path):
    """Convierte una imagen a base64 para enviar a la API"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def encode_image_from_array(image):
    """Convierte un array de imagen a base64 para enviar a la API"""
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        return None
    return base64.b64encode(encoded_image).decode('utf-8')

def analyze_plate_with_openai(image_path_or_array, api_key):
    """Analiza una imagen de patente usando GPT-4 Vision"""
    # Determinar si es una ruta o un array
    if isinstance(image_path_or_array, str):
        base64_image = encode_image_to_base64(image_path_or_array)
    else:
        base64_image = encode_image_from_array(image_path_or_array)
    
    if base64_image is None:
        return None
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Esta imagen contiene una patente de vehículo. Por favor, identifica el número de la patente. Responde SOLO con el texto de la patente, sin explicaciones adicionales."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    try:
        result = response.json()
        plate_text = result['choices'][0]['message']['content'].strip()
        return plate_text
    except Exception as e:
        print(f"Error al procesar respuesta de OpenAI: {e}")
        return None

def read_plate_openai(frame, bbox=None, api_key=None):
    """
    Función compatible con el resto del código que utiliza OpenAI para leer la patente
    
    Args:
        frame: Imagen completa del frame
        bbox: Coordenadas de la patente (x1, y1, x2, y2)
        api_key: API Key de OpenAI (opcional, si no se proporciona se lee de variables de entorno)
    
    Returns:
        Texto de la patente o None si no se pudo reconocer
    """
    # Obtener API key de argumentos o variable de entorno
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: Se requiere API Key de OpenAI")
        return None
        
    try:
        # Recortar la región de la patente si se proporciona bbox
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            plate_img = frame[y1:y2, x1:x2]
        else:
            plate_img = frame
            
        # Analizar directamente la imagen como array
        plate_text = analyze_plate_with_openai(plate_img, api_key)
        
        return plate_text
    except Exception as e:
        print(f"Error en read_plate_openai: {e}")
        return None

def process_plate_image(plate_img, use_openai=False, api_key=None):
    """
    Procesa una imagen de patente, usando OpenAI si se especifica
    
    Args:
        plate_img: Imagen de la patente
        use_openai: Si es True, usa OpenAI para reconocer la patente
        api_key: API Key de OpenAI (opcional)
    
    Returns:
        Diccionario con el resultado del OCR
    """
    if use_openai:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        plate_text = analyze_plate_with_openai(plate_img, api_key)
        return {"ocr_text": plate_text, "average_conf": 100.0 if plate_text else 0.0}
    else:
        # Método alternativo de OCR si no se usa OpenAI
        # Este punto requeriría integración con OCRProcessor
        return {"ocr_text": "", "average_conf": 0.0}

def main():
    parser = argparse.ArgumentParser(description='Reconocimiento de patentes usando OpenAI')
    parser.add_argument('--image', required=True, help='Ruta a la imagen con la patente')
    parser.add_argument('--api-key', help='API Key de OpenAI')
    
    args = parser.parse_args()
    
    # Obtener API key de argumentos o variable de entorno
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: Se requiere API Key de OpenAI")
        return
    
    # Analizar imagen
    plate_text = analyze_plate_with_openai(args.image, api_key)
    
    if plate_text:
        print(f"Patente detectada: {plate_text}")
        
        # Guardar resultado
        image = cv2.imread(args.image)
        save_plate_snapshot(image, plate_text, 1.0, "resultados_openai")
    else:
        print("No se pudo reconocer la patente")

if __name__ == "__main__":
    main()
