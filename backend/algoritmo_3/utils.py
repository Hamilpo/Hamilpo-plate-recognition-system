"""
Utilidades específicas para el Algoritmo 3
TODO: Funciones auxiliares movidas desde los scripts originales
Mantiene EXACTAMENTE la misma funcionalidad original
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List, Optional
import os
import json


def redimensionar_imagen_1280x720(imagen: Image.Image) -> Image.Image:
    """
    Redimensiona imagen a 1280x720 - TODO: Movido EXACTO desde convertir_imagen_actual_1280x720
    """
    return imagen.resize((1280, 720), Image.BILINEAR)


def convertir_a_escala_grises(imagen: Image.Image) -> Image.Image:
    """
    Convierte imagen a escala de grises - TODO: Movido EXACTO desde ambos scripts
    """
    from PIL import ImageOps
    return ImageOps.grayscale(imagen)


def numpy_a_pil(imagen_np: np.ndarray, modo: str = 'L') -> Image.Image:
    """
    Convierte array numpy a imagen PIL - TODO: Función auxiliar nueva
    """
    if imagen_np.ndim == 2:
        return Image.fromarray(imagen_np.astype(np.uint8), mode=modo)
    else:
        return Image.fromarray(imagen_np.astype(np.uint8), mode='RGB')


def pil_a_numpy(imagen_pil: Image.Image) -> np.ndarray:
    """
    Convierte imagen PIL a array numpy - TODO: Función auxiliar nueva
    """
    return np.array(imagen_pil)


def obtener_recorte_objeto(labels: np.ndarray, etiqueta: int, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """
    Obtiene recorte de un objeto etiquetado - TODO: Adaptado EXACTO desde mostrar_recorte_etiqueta
    """
    x1, y1, x2, y2 = bbox
    H, W = labels.shape

    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    
    if x2 < x1 or y2 < y1:
        return Image.new('L', (1, 1), color=255)

    mask_lab = (labels == etiqueta).astype(np.uint8)
    if mask_lab.sum() == 0:
        return Image.new('L', (1, 1), color=255)

    mask_crop = mask_lab[y1:y2+1, x1:x2+1] * 255
    img_crop = 255 - mask_crop  # Carácter negro sobre fondo blanco - EXACTO al original

    return Image.fromarray(img_crop.astype(np.uint8), mode="L")


def redimensionar_para_visualizacion(imagen: Image.Image, max_size: int = 300) -> Image.Image:
    """
    Redimensiona imagen para visualización - TODO: Adaptado EXACTO desde _mostrar_imagen
    """
    w, h = imagen.size
    if w > 0 and h > 0:
        escala = min(max_size / w, max_size / h, 1.0)
    else:
        escala = 1.0

    new_w = max(1, int(w * escala))
    new_h = max(1, int(h * escala))
    return imagen.resize((new_w, new_h), Image.NEAREST)


def crear_fuente(tamano: int = 14) -> ImageFont.ImageFont:
    """
    Crea fuente para dibujar texto - TODO: Movido EXACTO desde ambos scripts
    """
    try:
        return ImageFont.truetype("arial.ttf", tamano)
    except Exception:
        return ImageFont.load_default()


def guardar_configuracion(config: dict, ruta: str) -> bool:
    """
    Guarda configuración en JSON - TODO: Función auxiliar nueva
    """
    try:
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error guardando configuración: {e}")
        return False


def cargar_configuracion(ruta: str) -> dict:
    """
    Carga configuración desde JSON - TODO: Función auxiliar nueva
    """
    try:
        with open(ruta, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error cargando configuración: {e}")
        return {}


def validar_caracter(caracter: str) -> bool:
    """
    Valida que el carácter sea válido (A-Z, 0-9) - TODO: Movido EXACTO desde clasificar_objeto
    """
    caracter = caracter.strip().upper()
    return len(caracter) == 1 and (caracter.isalpha() or caracter.isdigit())