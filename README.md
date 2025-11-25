# Algoritmo 3 - Reconocimiento de Patentes Vehiculares

## 📋 Descripción
Sistema completo para el reconocimiento automático de patentes vehiculares utilizando procesamiento de imágenes, morfología matemática y clasificación KNN.

## 🏗️ Estructura de Archivos

### Archivos Creados/Modificados

1. **`backend/algoritmo_3/procesamiento.py`**
   - Clase principal `Algoritmo3` con toda la lógica de procesamiento
   - Funciones movidas desde ambos scripts originales
   - Mantiene exactamente el mismo comportamiento algorítmico

2. **`backend/algoritmo_3/utils.py`**
   - Funciones auxiliares para operaciones con imágenes
   - Utilidades de conversión y validación

3. **`frontend/algoritmo_3.py`**
   - Interfaz Streamlit moderna y profesional
   - 4 pestañas: Cargar/Procesar, Entrenar/Clasificar, Predecir, Configuración
   - Comunicación completa con el backend

4. **`training/algoritmo_3/train.py`**
   - Script CLI para entrenamiento por lotes
   - Comandos: entrenar, exportar, estadísticas

## 🚀 Ejecución

### Interfaz Streamlit
```bash
streamlit run frontend/algoritmo_3.py