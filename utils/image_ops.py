"""
Utilidades para operaciones con imágenes
"""

import cv2
import numpy as np
from PIL import Image
import io

def redimensionar_imagen(imagen: np.ndarray, ancho: int, alto: int) -> np.ndarray:
    """
    Redimensiona una imagen a las dimensiones especificadas
    
    Args:
        imagen: Imagen a redimensionar
        ancho: Nuevo ancho
        alto: Nuevo alto
        
    Returns:
        Imagen redimensionada
    """
    
    return cv2.resize(imagen, (ancho, alto))

def convertir_a_escala_grises(imagen: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen a escala de grises
    
    Args:
        imagen: Imagen color
        
    Returns:
        Imagen en escala de grises
    """
    
    if len(imagen.shape) == 3:
        return cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    return imagen

def normalizar_imagen(imagen: np.ndarray) -> np.ndarray:
    """
    Normaliza los valores de píxeles al rango [0, 1]
    
    Args:
        imagen: Imagen a normalizar
        
    Returns:
        Imagen normalizada
    """
    
    return imagen.astype(np.float32) / 255.0