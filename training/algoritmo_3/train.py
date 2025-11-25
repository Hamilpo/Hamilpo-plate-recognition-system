"""
Script de entrenamiento para el Algoritmo 3
TODO: Adaptado desde la lógica de ENTRENAMIENTO_PLACAS_V3.py
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any
import numpy as np

# Agregar ruta para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.algoritmo_3.procesamiento import Algoritmo3
from backend.algoritmo_3.utils import validar_caracter


def entrenar_desde_directorio(directorio_entrada: str, db_salida: str = None) -> Dict[str, Any]:
    """
    Entrena el modelo procesando imágenes de un directorio
    TODO: Adaptado desde la lógica de guardar_en_base_datos
    """
    
    print(f"🔍 Iniciando entrenamiento desde directorio: {directorio_entrada}")
    
    # Inicializar algoritmo
    algoritmo = Algoritmo3(db_salida)
    
    # Buscar archivos de imagen
    extensiones_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    archivos_imagen = []
    
    for archivo in os.listdir(directorio_entrada):
        if archivo.lower().endswith(extensiones_validas):
            archivos_imagen.append(os.path.join(directorio_entrada, archivo))
    
    if not archivos_imagen:
        print("❌ No se encontraron imágenes en el directorio")
        return {"exito": False, "error": "No hay imágenes"}
    
    print(f"📁 Encontradas {len(archivos_imagen)} imágenes")
    
    resultados = {
        "exito": True,
        "imagenes_procesadas": 0,
        "muestras_agregadas": 0,
        "errores": []
    }
    
    for i, ruta_imagen in enumerate(archivos_imagen, 1):
        print(f"🔄 Procesando imagen {i}/{len(archivos_imagen)}: {os.path.basename(ruta_imagen)}")
        
        try:
            # Cargar imagen
            if not algoritmo.cargar_imagen(ruta_imagen):
                resultados["errores"].append(f"Error cargando {ruta_imagen}")
                continue
            
            # Procesamiento automático
            umbral_otsu = algoritmo.calcular_umbral_otsu(algoritmo.img_gray_np)
            algoritmo.aplicar_umbral(umbral_otsu)
            algoritmo.aplicar_morfologia("Cerramiento", "Disco", 3)
            algoritmo.etiquetar_objetos()
            
            if not algoritmo.props:
                print(f"  ⚠️ No se detectaron objetos en {os.path.basename(ruta_imagen)}")
                continue
            
            # Para entrenamiento, necesitaríamos las etiquetas reales
            # Esto es un placeholder - en un caso real necesitaríamos un mapeo de etiquetas
            print(f"  📊 Detección: {len(algoritmo.props)} objetos")
            
            # Aquí iría la lógica para obtener las etiquetas reales
            # Por ahora, simulamos que no agregamos muestras sin etiquetas reales
            print("  ℹ️ Nota: Se necesita mapeo manual de etiquetas para entrenamiento supervisado")
            
            resultados["imagenes_procesadas"] += 1
            
        except Exception as e:
            error_msg = f"Error procesando {ruta_imagen}: {str(e)}"
            print(f"  ❌ {error_msg}")
            resultados["errores"].append(error_msg)
    
    print(f"✅ Procesamiento completado:")
    print(f"   - Imágenes procesadas: {resultados['imagenes_procesadas']}")
    print(f"   - Errores: {len(resultados['errores'])}")
    print(f"   - Muestras en BD: {len(algoritmo.base_datos)}")
    
    return resultados


def exportar_modelo(ruta_db: str, ruta_salida: str) -> bool:
    """
    Exporta la base de datos como modelo entrenado
    TODO: Función nueva para exportación
    """
    try:
        algoritmo = Algoritmo3(ruta_db)
        
        modelo = {
            "base_datos": algoritmo.base_datos,
            "estadisticas": {
                "total_muestras": len(algoritmo.base_datos),
                "caracteres_unicos": len(set(item['caracter'] for item in algoritmo.base_datos)),
                "rango_areas": {
                    "min": min(item['area'] for item in algoritmo.base_datos) if algoritmo.base_datos else 0,
                    "max": max(item['area'] for item in algoritmo.base_datos) if algoritmo.base_datos else 0
                }
            },
            "parametros": {
                "area_min": algoritmo.area_min,
                "area_max": algoritmo.area_max,
                "metodo_morfologia": "Cerramiento (Disco r=3)"
            }
        }
        
        with open(ruta_salida, 'w', encoding='utf-8') as f:
            json.dump(modelo, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Modelo exportado a: {ruta_salida}")
        print(f"   - Muestras: {modelo['estadisticas']['total_muestras']}")
        print(f"   - Caracteres únicos: {modelo['estadisticas']['caracteres_unicos']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error exportando modelo: {e}")
        return False


def estadisticas_base_datos(ruta_db: str) -> Dict[str, Any]:
    """
    Muestra estadísticas de la base de datos
    TODO: Función nueva para análisis
    """
    algoritmo = Algoritmo3(ruta_db)
    
    if not algoritmo.base_datos:
        return {"error": "Base de datos vacía"}
    
    caracteres = [item['caracter'] for item in algoritmo.base_datos]
    areas = [item['area'] for item in algoritmo.base_datos]
    
    stats = {
        "total_muestras": len(algoritmo.base_datos),
        "caracteres_unicos": len(set(caracteres)),
        "distribucion_caracteres": {car: caracteres.count(car) for car in set(caracteres)},
        "estadisticas_areas": {
            "min": min(areas),
            "max": max(areas),
            "promedio": np.mean(areas),
            "desviacion": np.std(areas)
        },
        "ruta_base_datos": ruta_db
    }
    
    return stats


def main():
    """Función principal del script de entrenamiento"""
    
    parser = argparse.ArgumentParser(description='Entrenamiento Algoritmo 3 - Reconocimiento de Patentes')
    
    subparsers = parser.add_subparsers(dest='comando', help='Comandos disponibles')
    
    # Comando entrenar
    parser_entrenar = subparsers.add_parser('entrenar', help='Entrenar desde directorio de imágenes')
    parser_entrenar.add_argument('--directorio', '-d', required=True, 
                               help='Directorio con imágenes de entrenamiento')
    parser_entrenar.add_argument('--salida', '-s', 
                               help='Ruta de salida para base de datos')
    
    # Comando exportar
    parser_exportar = subparsers.add_parser('exportar', help='Exportar modelo entrenado')
    parser_exportar.add_argument('--entrada', '-i', required=True,
                               help='Base de datos de entrada')
    parser_exportar.add_argument('--salida', '-o', required=True,
                               help='Archivo de salida para modelo')
    
    # Comando estadisticas
    parser_stats = subparsers.add_parser('estadisticas', help='Mostrar estadísticas de base de datos')
    parser_stats.add_argument('--base-datos', '-b', required=True,
                            help='Base de datos a analizar')
    
    args = parser.parse_args()
    
    if not args.comando:
        parser.print_help()
        return
    
    if args.comando == 'entrenar':
        print("🚀 Iniciando proceso de entrenamiento...")
        resultado = entrenar_desde_directorio(args.directorio, args.salida)
        
        if resultado["exito"]:
            print("✅ Entrenamiento completado exitosamente")
        else:
            print("❌ Entrenamiento completado con errores")
    
    elif args.comando == 'exportar':
        print("📦 Exportando modelo...")
        if exportar_modelo(args.entrada, args.salida):
            print("✅ Exportación completada")
        else:
            print("❌ Error en exportación")
    
    elif args.comando == 'estadisticas':
        print("📊 Calculando estadísticas...")
        stats = estadisticas_base_datos(args.base_datos)
        
        if "error" in stats:
            print(f"❌ {stats['error']}")
        else:
            print(f"📈 Estadísticas de la base de datos:")
            print(f"   - Muestras totales: {stats['total_muestras']}")
            print(f"   - Caracteres únicos: {stats['caracteres_unicos']}")
            print(f"   - Distribución de caracteres:")
            for char, count in stats['distribucion_caracteres'].items():
                print(f"     '{char}': {count} muestras")
            print(f"   - Áreas: {stats['estadisticas_areas']['min']}-{stats['estadisticas_areas']['max']} px")
            print(f"   - Ruta: {stats['ruta_base_datos']}")


if __name__ == "__main__":
    main()