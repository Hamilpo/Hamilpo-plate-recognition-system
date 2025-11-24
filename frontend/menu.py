"""
Módulo para la configuración del menú lateral
"""

import streamlit as st

def setup_sidebar():
    """Configura y muestra el menú lateral"""
    
    st.sidebar.title("🚗 Reconocimiento de Patentes")
    st.sidebar.markdown("---")
    
    # Menú de navegación
    page = st.sidebar.radio(
        "Navegación",
        ["Inicio", "Algoritmo 3", "Documentación"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Sistema de reconocimiento de patentes vehiculares "
        "utilizando algoritmos de visión artificial"
    )
    
    return page