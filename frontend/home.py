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
    utilizando algoritmos avanzados de visión artificial.
    
    ### Características principales:
    - 📷 Procesamiento de imágenes en tiempo real
    - 🔍 Detección y segmentación de caracteres
    - 🧠 Algoritmos de machine learning
    - 📊 Resultados precisos y confiables
    
    ### Módulos disponibles:
    - **Algoritmo 3**: Sistema completo de reconocimiento
    """)
    
    # Métricas de ejemplo
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Precisión", "95%", "2%")
    
    with col2:
        st.metric("Imágenes Procesadas", "1,247", "12")
    
    with col3:
        st.metric("Tiempo Promedio", "0.8s", "-0.1s")