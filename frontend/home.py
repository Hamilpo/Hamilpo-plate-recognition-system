"""
Página de inicio de la aplicación
"""

import streamlit as st

def show_home():
    """Muestra la página de inicio"""
    
    st.title("🚗 Sistema de Reconocimiento de Patentes Vehiculares")
    
    st.markdown("""
    ## Bienvenido al sistema
    
    Esta aplicación permite el reconocimiento automático de patentes vehiculares
    utilizando algoritmos avanzados de visión artificial y machine learning.
    
    ### Características principales:
    - 📷 Procesamiento de imágenes en tiempo real
    - 🔍 Detección y segmentación de caracteres
    - 🧠 Algoritmos de machine learning (KNN)
    - 📊 Resultados precisos y confiables
    - 💾 Base de datos entrenable
    """)
    
    # Métricas de ejemplo
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Algoritmos Disponibles", "1", "Algoritmo 3")
    
    with col2:
        st.metric("Módulos Implementados", "4", "Completo")
    
    with col3:
        st.metric("Tecnologías", "Streamlit + OpenCV", "Python")
    
    st.markdown("""
    ### Módulos disponibles:
    - **Algoritmo 3 - Reconocimiento**: Sistema completo de reconocimiento de patentes
      - Segmentación por umbral
      - Operaciones morfológicas
      - Extracción de características
      - Clasificación KNN
      - Entrenamiento interactivo
    """)
    
    # Quick start guide
    with st.expander("🚀 Guía Rápida de Inicio"):
        st.markdown("""
        1. **Navega a "Algoritmo 3 - Reconocimiento"**
        2. **Carga una imagen de patente** en la pestaña "Cargar y Procesar"
        3. **Aplica umbral y operaciones morfológicas**
        4. **Etiqueta los caracteres** en la pestaña "Entrenar/Clasificar"
        5. **Guarda en la base de datos** para entrenar el modelo
        6. **Predice automáticamente** en la pestaña "Predecir"
        """)
    
    # Información técnica
    with st.expander("🔧 Información Técnica"):
        st.markdown("""
        **Tecnologías utilizadas:**
        - Frontend: Streamlit
        - Procesamiento: OpenCV, NumPy, PIL
        - Machine Learning: Scikit-learn (KNN)
        - Almacenamiento: CSV para base de datos
        
        **Características extraídas:**
        - Área, perímetro, circularidad
        - Relación de aspecto
        - Momentos invariantes de Hu (7 características)
        - Centroides y bounding boxes
        """)
