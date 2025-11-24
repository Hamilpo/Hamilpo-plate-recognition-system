"""
Archivo principal de la aplicación Streamlit
Coordinador de módulos y punto de entrada de la app
"""

import streamlit as st

# Importar módulos del frontend
from frontend.menu import setup_sidebar
from frontend.home import show_home
from frontend.algoritmo_3 import show_algoritmo_3

def main():
    """Función principal de la aplicación"""
    
    # Configurar página
    st.set_page_config(
        page_title="Sistema Reconocimiento de Patentes",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Configurar sidebar y obtener página seleccionada
    selected_page = setup_sidebar()
    
    # Navegación entre páginas
    if selected_page == "Inicio":
        show_home()
    elif selected_page == "Algoritmo 3":
        show_algoritmo_3()
    elif selected_page == "Documentación":
        st.title("📚 Documentación")
        st.info("TODO: Agregar documentación completa del proyecto")

if __name__ == "__main__":
    main()