#!/usr/bin/env python
"""
Script para actualizar manualmente los parámetros de ROI en config.py
basándose en los resultados del análisis QA.
"""

import os
import sys
import re
import argparse
import glob
from datetime import datetime

# Añadir directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import ROI_ANGULO_ROTACION, ROI_ESCALA_FACTOR
    CONFIG_ACTUAL = {
        'angulo': ROI_ANGULO_ROTACION,
        'escala': ROI_ESCALA_FACTOR
    }
except ImportError:
    CONFIG_ACTUAL = {
        'angulo': 0.0,
        'escala': 1.0
    }

def encontrar_ultimos_parametros():
    """Busca y retorna los parámetros óptimos del análisis QA más reciente."""
    resultados_dir = os.path.join(os.path.dirname(__file__), "resultados")
    carpetas = [d for d in glob.glob(os.path.join(resultados_dir, "mapas_calor_*")) 
               if os.path.isdir(d)]
    
    if not carpetas:
        print("No se encontraron carpetas de resultados QA")
        return None
    
    # Ordenar por fecha (más reciente primero)
    carpetas.sort(reverse=True)
    ultima_carpeta = carpetas[0]
    
    # Buscar archivos de parámetros óptimos
    archivos = glob.glob(os.path.join(ultima_carpeta, "parametros_optimos_*.txt"))
    if not archivos:
        print(f"No se encontraron archivos de parámetros en {os.path.basename(ultima_carpeta)}")
        return None
    
    # Ordenar por fecha (más reciente primero)
    archivos.sort(reverse=True)
    ultimo_archivo = archivos[0]
    
    # Extraer parámetros óptimos
    angulo = None
    escala = None
    metodo = None
    
    with open(ultimo_archivo, 'r') as f:
        contenido = f.read()
        
        # Buscar la recomendación
        match_recomendacion = re.search(r"Parámetros recomendados: Ángulo=([-+]?[0-9]*\.?[0-9]+)°, Escala=([-+]?[0-9]*\.?[0-9]+)%", contenido)
        if match_recomendacion:
            angulo = float(match_recomendacion.group(1))
            escala = float(match_recomendacion.group(2)) / 100.0  # Convertir a factor decimal
            
            # Determinar el método
            if "Recomendación: Usar corrección con rotación simple" in contenido:
                metodo = "rotación simple"
            elif "Recomendación: Usar corrección con perspectiva" in contenido:
                metodo = "perspectiva"
    
    if angulo is not None and escala is not None:
        return {
            'archivo': ultimo_archivo,
            'angulo': angulo,
            'escala': escala,
            'metodo': metodo
        }
    
    print(f"No se pudieron extraer los parámetros óptimos de {os.path.basename(ultimo_archivo)}")
    return None

def actualizar_config(angulo, escala):
    """Actualiza los parámetros en el archivo config.py."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.py")
    
    if not os.path.exists(config_path):
        print(f"No se encontró el archivo config.py en {config_path}")
        return False
    
    try:
        # Leer el archivo config.py
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Actualizar los valores de configuración
        new_content = re.sub(r'ROI_ANGULO_ROTACION\s*=\s*[-+]?[0-9]*\.?[0-9]+', 
                           f'ROI_ANGULO_ROTACION = {angulo:.1f}', 
                           config_content)
        new_content = re.sub(r'ROI_ESCALA_FACTOR\s*=\s*[-+]?[0-9]*\.?[0-9]+', 
                           f'ROI_ESCALA_FACTOR = {escala:.1f}', 
                           new_content)
        new_content = re.sub(r'ROI_APLICAR_CORRECCION\s*=\s*\w+', 
                           f'ROI_APLICAR_CORRECCION = True', 
                           new_content)
        
        # Guardar el archivo actualizado
        with open(config_path, 'w') as f:
            f.write(new_content)
        
        print(f"Configuración actualizada exitosamente en {config_path}")
        return True
    
    except Exception as e:
        print(f"Error al actualizar config.py: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Actualiza parámetros óptimos en config.py")
    parser.add_argument("--angulo", type=float, help="Ángulo de rotación en grados")
    parser.add_argument("--escala", type=float, help="Factor de escala (0.1 a 1.5)")
    parser.add_argument("--auto", action="store_true", help="Usar parámetros óptimos del último análisis QA")
    
    args = parser.parse_args()
    
    if args.auto:
        # Buscar los parámetros óptimos del último análisis
        parametros = encontrar_ultimos_parametros()
        if parametros:
            print(f"Parámetros óptimos encontrados en {os.path.basename(parametros['archivo'])}")
            print(f"  - Método: {parametros['metodo']}")
            print(f"  - Ángulo: {parametros['angulo']:.1f}°")
            print(f"  - Escala: {parametros['escala']*100:.1f}%")
            
            print(f"\nConfiguración actual:")
            print(f"  - Ángulo: {CONFIG_ACTUAL['angulo']:.1f}°")
            print(f"  - Escala: {CONFIG_ACTUAL['escala']*100:.1f}%")
            
            if CONFIG_ACTUAL['angulo'] == parametros['angulo'] and CONFIG_ACTUAL['escala'] == parametros['escala']:
                print("\nLa configuración actual ya tiene los valores óptimos.")
                return
            
            confirmar = input("\n¿Desea actualizar la configuración con estos valores? (s/n): ")
            if confirmar.lower() == 's':
                actualizar_config(parametros['angulo'], parametros['escala'])
                
                # Registrar la actualización
                log_dir = os.path.join(os.path.dirname(__file__), "resultados")
                with open(os.path.join(log_dir, "actualizaciones_config.txt"), "a") as f:
                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Actualización manual de parámetros\n")
                    f.write(f"  - Ángulo: {CONFIG_ACTUAL['angulo']:.1f}° → {parametros['angulo']:.1f}°\n")
                    f.write(f"  - Escala: {CONFIG_ACTUAL['escala']*100:.1f}% → {parametros['escala']*100:.1f}%\n\n")
        else:
            print("No se pudieron determinar los parámetros óptimos automáticamente.")
    
    elif args.angulo is not None and args.escala is not None:
        # Validar parámetros
        if args.escala <= 0 or args.escala > 2:
            print("Error: El factor de escala debe estar entre 0.1 y 2.0")
            return
        
        if abs(args.angulo) > 45:
            print("Error: El ángulo debe estar entre -45 y 45 grados")
            return
        
        # Confirmar cambio
        print(f"Configuración actual:")
        print(f"  - Ángulo: {CONFIG_ACTUAL['angulo']:.1f}°")
        print(f"  - Escala: {CONFIG_ACTUAL['escala']*100:.1f}%")
        
        print(f"\nNueva configuración:")
        print(f"  - Ángulo: {args.angulo:.1f}°")
        print(f"  - Escala: {args.escala*100:.1f}%")
        
        confirmar = input("\n¿Desea aplicar estos cambios? (s/n): ")
        if confirmar.lower() == 's':
            actualizar_config(args.angulo, args.escala)
            
            # Registrar la actualización
            log_dir = os.path.join(os.path.dirname(__file__), "resultados")
            with open(os.path.join(log_dir, "actualizaciones_config.txt"), "a") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Actualización manual de parámetros\n")
                f.write(f"  - Ángulo: {CONFIG_ACTUAL['angulo']:.1f}° → {args.angulo:.1f}°\n")
                f.write(f"  - Escala: {CONFIG_ACTUAL['escala']*100:.1f}% → {args.escala*100:.1f}%\n\n")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
