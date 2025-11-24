"""
Utilidades para el Algoritmo 3
"""

import json
import pickle
from typing import Any

def cargar_modelo(ruta_modelo: str) -> Any:
    """
    Carga un modelo entrenado desde archivo
    
    Args:
        ruta_modelo: Ruta al archivo del modelo
        
    Returns:
        Modelo cargado
    """
    
    # TODO: Implementar carga real del modelo
    try:
        with open(ruta_modelo, 'rb') as f:
            modelo = pickle.load(f)
        return modelo
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return None

def guardar_modelo(modelo: Any, ruta_modelo: str) -> bool:
    """
    Guarda un modelo entrenado en archivo
    
    Args:
        modelo: Modelo a guardar
        ruta_modelo: Ruta donde guardar el modelo
        
    Returns:
        True si se guardó correctamente
    """
    
    # TODO: Implementar guardado real del modelo
    try:
        with open(ruta_modelo, 'wb') as f:
            pickle.dump(modelo, f)
        return True
    except Exception as e:
        print(f"Error guardando modelo: {e}")
        return False

def cargar_configuracion(ruta_config: str) -> dict:
    """
    Carga configuración desde archivo JSON
    
    Args:
        ruta_config: Ruta al archivo de configuración
        
    Returns:
        Diccionario con configuración
    """
    
    try:
        with open(ruta_config, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error cargando configuración: {e}")
        return {}