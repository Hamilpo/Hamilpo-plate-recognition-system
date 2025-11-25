"""
Utilidades para operaciones con archivos - ADAPTADO al árbol oficial
"""

import os
import json
import pickle
from typing import Any
from pathlib import Path


def crear_directorio(ruta: str) -> bool:
    """
    Crea un directorio si no existe
    
    Args:
        ruta: Ruta del directorio a crear
        
    Returns:
        True si se creó o ya existe
    """
    
    try:
        Path(ruta).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creando directorio {ruta}: {e}")
        return False


def obtener_ruta_proyecto() -> Path:
    """
    Obtiene la ruta raíz del proyecto
    
    Returns:
        Path de la raíz del proyecto
    """
    return Path(__file__).resolve().parents[1]


def obtener_ruta_modelos() -> Path:
    """
    Obtiene la ruta de modelos del proyecto
    
    Returns:
        Path de la carpeta de modelos
    """
    return obtener_ruta_proyecto() / "data" / "models"


def obtener_ruta_procesados() -> Path:
    """
    Obtiene la ruta de archivos procesados
    
    Returns:
        Path de la carpeta de procesados
    """
    return obtener_ruta_proyecto() / "data" / "processed"


def obtener_ruta_resultados() -> Path:
    """
    Obtiene la ruta de resultados
    
    Returns:
        Path de la carpeta de resultados
    """
    return obtener_ruta_proyecto() / "data" / "results"


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
        # Asegurar que el directorio existe
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        
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


def guardar_pickle(datos: Any, ruta: str) -> bool:
    """
    Guarda datos en formato pickle
    
    Args:
        datos: Datos a guardar
        ruta: Ruta del archivo
        
    Returns:
        True si se guardó correctamente
    """
    
    try:
        # Asegurar que el directorio existe
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        
        with open(ruta, 'wb') as f:
            pickle.dump(datos, f)
        return True
    except Exception as e:
        print(f"Error guardando pickle en {ruta}: {e}")
        return False


def cargar_pickle(ruta: str) -> Any:
    """
    Carga datos desde archivo pickle
    
    Args:
        ruta: Ruta del archivo
        
    Returns:
        Datos cargados o None si hay error
    """
    
    try:
        with open(ruta, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error cargando pickle desde {ruta}: {e}")
        return None