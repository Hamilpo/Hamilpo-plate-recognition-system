"""
Utilidades para operaciones con archivos
"""

import os
import json
import pickle
from typing import Any

def crear_directorio(ruta: str) -> bool:
    """
    Crea un directorio si no existe
    
    Args:
        ruta: Ruta del directorio a crear
        
    Returns:
        True si se creó o ya existe
    """
    
    try:
        os.makedirs(ruta, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creando directorio {ruta}: {e}")
        return False

def guardar_json(datos: Any, ruta: str) -> bool:
    """
    Guarda datos en formato JSON
    
    Args:
        datos: Datos a guardar
        ruta: Ruta del archivo
        
    Returns:
        True si se guardó correctamente
    """
    
    try:
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error guardando JSON en {ruta}: {e}")
        return False

def cargar_json(ruta: str) -> Any:
    """
    Carga datos desde archivo JSON
    
    Args:
        ruta: Ruta del archivo
        
    Returns:
        Datos cargados o None si hay error
    """
    
    try:
        with open(ruta, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error cargando JSON desde {ruta}: {e}")
        return None