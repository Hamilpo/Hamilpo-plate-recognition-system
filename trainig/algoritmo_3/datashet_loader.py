"""
Módulo para carga y manejo de datasets
"""

import os
from typing import List, Tuple

def cargar_dataset(ruta: str) -> List[Tuple[str, str]]:
    """
    Carga un dataset de imágenes y etiquetas
    
    Args:
        ruta: Ruta al directorio del dataset
        
    Returns:
        Lista de tuplas (ruta_imagen, etiqueta)
    """
    
    # TODO: Implementar carga real del dataset
    datos = []
    
    if os.path.exists(ruta):
        # Simular carga de datos
        for archivo in os.listdir(ruta):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extraer etiqueta del nombre del archivo
                etiqueta = os.path.splitext(archivo)[0]
                datos.append((os.path.join(ruta, archivo), etiqueta))
    
    return datos

def validar_dataset(datos: List[Tuple[str, str]]) -> bool:
    """
    Valida la estructura y formato del dataset
    
    Args:
        datos: Lista de datos a validar
        
    Returns:
        True si el dataset es válido
    """
    
    # TODO: Implementar validación real
    if not datos:
        return False
    
    # Verificar que las imágenes existan
    for ruta_imagen, _ in datos:
        if not os.path.exists(ruta_imagen):
            return False
    
    return True