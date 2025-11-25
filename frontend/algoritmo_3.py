"""
Interfaz Streamlit para el Algoritmo 3 - Reconocimiento de Patentes
TODO: Adaptado desde ENTRENAMIENTO_PLACAS_V3.py y PREDICCION_PLACAS_V2.py
Mantiene TODAS las funcionalidades del GUI original en interfaz moderna
"""

import streamlit as st
import tempfile
import os
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Agregar path para imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Importar backend
try:
    from backend.algoritmo_3.procesamiento import Algoritmo3
    from backend.algoritmo_3.utils import redimensionar_para_visualizacion, validar_caracter
except ImportError as e:
    st.error(f"Error importando backend: {e}")
    # Fallbacks para desarrollo
    class Algoritmo3:
        def __init__(self): 
            self.base_datos = []
            self.area_min = 3000
            self.area_max = 50000
            self.db_file = ""
        def get_info_imagen(self): return {"tiene_imagen": False, "tamano_bd": 0}
        def cargar_imagen(self, x): return True
        def calcular_umbral_otsu(self, x): return 128
        def aplicar_umbral(self, x): return True
        def aplicar_morfologia(self, op, forma, radio): return True
        def etiquetar_objetos(self, min_a, max_a): return True
        def clasificar_objeto(self, e, c): return True
        def guardar_en_base_datos(self): return 0
        def predecir_caracteres(self, use_knn, k): return []
        def limpiar_etiquetas(self): pass
    def redimensionar_para_visualizacion(img, size=300): return img
    def validar_caracter(c): return True


def show_algoritmo_3():
    """Interfaz principal del Algoritmo 3 - TODAS las funcionalidades del original"""
    
    st.title("🔍 Algoritmo 3 - Reconocimiento de Patentes Vehiculares")
    
    # Inicializar estado de sesión
    if 'algoritmo3' not in st.session_state:
        st.session_state.algoritmo3 = Algoritmo3()
        st.session_state.etiquetas_clasificadas = {}
        st.session_state.resultados_prediccion = []
        st.session_state.procesamiento_completo = False
        st.session_state.umbral_actual = 128

    algo3 = st.session_state.algoritmo3
    
    # Pestañas para TODAS las funcionalidades del original
    tab1, tab2, tab3, tab4 = st.tabs([
        "📷 Cargar y Procesar", 
        "🧠 Entrenar/Clasificar", 
        "🔮 Predecir", 
        "⚙️ Configuración"
    ])
    
    with tab1:
        show_cargar_procesar(algo3)
    
    with tab2:
        show_entrenar_clasificar(algo3)
    
    with tab3:
        show_predecir(algo3)
    
    with tab4:
        show_configuracion(algo3)


def show_cargar_procesar(algo3: Algoritmo3):
    """Interfaz para carga y procesamiento - TODOS los controles del original"""
    
    st.header("1. Cargar y Preprocesar Imagen")
    
    # Subir imagen - Mismo funcionamiento del original
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de patente vehicular",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
        key="algo3_upload"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_file is not None:
            # Cargar y mostrar imagen original
            imagen = Image.open(uploaded_file)
            st.image(imagen, caption="Imagen Original", use_column_width=True)
            
            if st.button("🔄 Cargar Imagen", type="primary"):
                with st.spinner("Cargando imagen..."):
                    temp_path = f"{tempfile.gettempdir()}/temp_image.png"
                    imagen.save(temp_path)
                    if algo3.cargar_imagen(temp_path):
                        st.success("✅ Imagen cargada correctamente")
                        st.session_state.procesamiento_completo = False
                        # Calcular Otsu automáticamente como en el original
                        if algo3.img_gray_np is not None:
                            umbral_otsu = algo3.calcular_umbral_otsu(algo3.img_gray_np)
                            st.session_state.umbral_otsu = umbral_otsu
                            st.session_state.umbral_actual = umbral_otsu
                    else:
                        st.error("❌ Error al cargar la imagen")
    
    with col2:
        # Información de la imagen - Acceso seguro
        info = algo3.get_info_imagen()
        if info.get("tiene_imagen", False):
            st.metric("Ancho", f"{info.get('ancho', 0)} px")
            st.metric("Alto", f"{info.get('alto', 0)} px")
            st.metric("Base de Datos", f"{info.get('tamano_bd', 0)} muestras")
            
            if st.button("📐 Convertir a 1280x720"):
                if algo3.img_original_pil:
                    algo3.img_original_pil = algo3.img_original_pil.resize((1280, 720), Image.BILINEAR)
                    algo3.img_gray_pil = ImageOps.grayscale(algo3.img_original_pil)
                    algo3.img_gray_np = np.array(algo3.img_gray_pil)
                    st.success("✅ Imagen convertida a 1280x720")
        else:
            st.info("ℹ️ Sube una imagen para ver información")
    
    if not info.get("tiene_imagen", False):
        st.info("ℹ️ Sube una imagen para comenzar el procesamiento")
        return
    
    st.header("2. Segmentación por Umbral")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        umbral = st.slider("Umbral de binarización", 0, 255, 
                          st.session_state.umbral_actual, key="umbral_slider")
        st.session_state.umbral_actual = umbral
    
    with col2:
        if st.button("🎯 Calcular Otsu", use_container_width=True):
            if algo3.img_gray_np is not None:
                umbral_otsu = algo3.calcular_umbral_otsu(algo3.img_gray_np)
                st.session_state.umbral_otsu = umbral_otsu
                st.session_state.umbral_actual = umbral_otsu
                st.success(f"Umbral Otsu: {umbral_otsu}")
    
    with col3:
        if 'umbral_otsu' in st.session_state:
            if st.button(f"🔧 Aplicar Otsu ({st.session_state.umbral_otsu})", use_container_width=True):
                st.session_state.umbral_actual = st.session_state.umbral_otsu
                st.rerun()
    
    if st.button("🔄 Aplicar Umbral", type="primary"):
        with st.spinner("Aplicando umbral..."):
            if algo3.aplicar_umbral(st.session_state.umbral_actual):
                st.success("✅ Umbral aplicado correctamente")
            else:
                st.error("❌ Error al aplicar umbral")
    
    # Mostrar imágenes de segmentación - TODAS las vistas del original
    if algo3.mask_01 is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            original_img = algo3.get_imagen_original()
            if original_img:
                st.image(original_img, caption="Original", use_column_width=True)
        with col2:
            gray_img = algo3.get_imagen_gris()
            if gray_img:
                st.image(gray_img, caption="Escala Grises", use_column_width=True)
        with col3:
            mask_img = algo3.get_mascara_binaria()
            if mask_img:
                st.image(mask_img, caption="Máscara Binaria", use_column_width=True)
        with col4:
            mask_inv_img = algo3.get_mascara_invertida()
            if mask_inv_img:
                st.image(mask_inv_img, caption="Máscara Invertida", use_column_width=True)
    
    st.header("3. Operaciones Morfológicas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        operacion = st.selectbox(
            "Operación",
            ["Cerramiento", "Apertura", "Erosión", "Dilatación"],
            index=0
        )
    
    with col2:
        forma = st.selectbox(
            "Forma del elemento",
            ["Disco", "Cuadrado", "Cruz"],
            index=0
        )
    
    with col3:
        radio = st.slider("Radio", 1, 10, 3, key="morph_radio")
    
    if st.button("⚡ Aplicar Morfología", type="primary"):
        with st.spinner("Aplicando operación morfológica..."):
            if algo3.aplicar_morfologia(operacion, forma, radio):
                st.success("✅ Operación morfológica aplicada")
                
                # Mostrar resultados morfológicos - Mismas vistas del original
                col1, col2 = st.columns(2)
                with col1:
                    mask_inv_img = algo3.get_mascara_invertida()
                    if mask_inv_img:
                        st.image(mask_inv_img, 
                               caption="Entrada (Máscara Invertida)", use_column_width=True)
                with col2:
                    morph_img = algo3.get_resultado_morfologia()
                    if morph_img:
                        st.image(morph_img, 
                               caption="Resultado Morfológico", use_column_width=True)
            else:
                st.error("❌ Error en operación morfológica")


def show_entrenar_clasificar(algo3: Algoritmo3):
    """Interfaz para entrenamiento y clasificación manual - TODAS las funciones del original"""
    
    st.header("🧠 Entrenamiento y Clasificación Manual")
    
    info = algo3.get_info_imagen()
    if not info.get("tiene_imagen", False):
        st.info("ℹ️ Primero carga y procesa una imagen en la pestaña 'Cargar y Procesar'")
        return
    
    st.subheader("Filtrado por Área")
    
    col1, col2 = st.columns(2)
    
    with col1:
        area_min = st.slider("Área mínima", 0, 10000, algo3.area_min, key="area_min")
    
    with col2:
        area_max = st.slider("Área máxima", 0, 100000, algo3.area_max, key="area_max")
    
    if st.button("🏷️ Etiquetar Objetos", type="primary"):
        with st.spinner("Etiquetando objetos..."):
            if algo3.etiquetar_objetos(area_min, area_max):
                num_objetos = len(algo3.props)
                st.success(f"✅ {num_objetos} objetos etiquetados")
                st.session_state.procesamiento_completo = True
                
                # TODO: Los objetos ya vienen ordenados de izquierda a derecha desde el backend
                st.info("ℹ️ Objetos ordenados automáticamente de izquierda a derecha")
            else:
                st.error("❌ Error en etiquetado")
    
    if not st.session_state.procesamiento_completo:
        return
    
    # Mostrar imagen etiquetada - Misma funcionalidad del original
    if algo3.labels is not None and algo3.props:
        try:
            imagen_etiquetada = algo3.generar_imagen_coloreada()
            st.image(imagen_etiquetada, caption="Objetos Etiquetados (Ordenados de Izquierda a Derecha)", use_column_width=True)
        except Exception as e:
            st.error(f"Error generando imagen etiquetada: {e}")
    
    st.subheader("Clasificación Manual")
    
    if not algo3.props:
        st.warning("⚠️ No hay objetos para clasificar")
        return
    
    # Tabla de objetos detectados - Misma información del original
    st.write("**Objetos Detectados (Ordenados de Izquierda a Derecha):**")
    
    datos_tabla = []
    for prop in algo3.props:
        etiqueta = prop["label"]
        clase = st.session_state.etiquetas_clasificadas.get(etiqueta, "")
        datos_tabla.append({
            "Etiqueta": etiqueta,
            "Área": prop["area"],
            "Centroide": f"({prop['centroid_x']:.1f}, {prop['centroid_y']:.1f})",
            "BBox": f"({prop['bbox'][0]},{prop['bbox'][1]})-({prop['bbox'][2]},{prop['bbox'][3]})",
            "Clase": clase
        })
    
    if datos_tabla:
        df = pd.DataFrame(datos_tabla)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Interfaz de clasificación - Mismos controles del original
    st.write("**Clasificar Objeto:**")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        # TODO: Los objetos ya están ordenados de izquierda a derecha desde el backend
        etiqueta_seleccionada = st.selectbox(
            "Etiqueta del objeto (Orden Izq→Der)",
            options=[p["label"] for p in algo3.props],
            key="etiqueta_select"
        )
    
    with col2:
        caracter = st.text_input("Carácter (A-Z, 0-9)", max_chars=1, key="caracter_input")
    
    with col3:
        st.write("")  # Espaciado
        if st.button("🎯 Asignar Clase", use_container_width=True):
            if validar_caracter(caracter):
                if algo3.clasificar_objeto(etiqueta_seleccionada, caracter.upper()):
                    st.session_state.etiquetas_clasificadas[etiqueta_seleccionada] = caracter.upper()
                    st.success(f"✅ Objeto {etiqueta_seleccionada} clasificado como '{caracter.upper()}'")
                    st.rerun()
                else:
                    st.error("❌ Error al clasificar objeto")
            else:
                st.error("❌ Carácter inválido. Use A-Z o 0-9.")
    
    # Guardar en base de datos - Misma funcionalidad del original
    if st.session_state.etiquetas_clasificadas:
        st.write("**Guardar Clasificaciones:**")
        st.info(f"📊 {len(st.session_state.etiquetas_clasificadas)} objetos clasificados en esta sesión")
        
        if st.button("💾 Guardar en Base de Datos", type="primary"):
            # Transferir clasificaciones de sesión al algoritmo
            algo3.clasificaciones = st.session_state.etiquetas_clasificadas.copy()
            
            guardados = algo3.guardar_en_base_datos()
            if guardados > 0:
                st.success(f"✅ {guardados} muestras guardadas en la base de datos")
                st.session_state.etiquetas_clasificadas = {}
                st.rerun()
            else:
                st.error("❌ Error al guardar en base de datos")


def show_predecir(algo3: Algoritmo3):
    """Interfaz para predicción automática - TODAS las funciones del original"""
    
    st.header("🔮 Predicción Automática")
    
    info = algo3.get_info_imagen()
    if not info.get("tiene_imagen", False) or not st.session_state.procesamiento_completo:
        st.info("ℹ️ Primero carga, procesa y etiqueta una imagen en las pestañas anteriores")
        return
    
    st.subheader("Configuración del Clasificador")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_knn = st.checkbox("Usar K-Vecinos (KNN)", value=True, key="use_knn")
        k_vecinos = st.slider("Número de vecinos (k)", 1, 20, 5, key="k_vecinos")
    
    with col2:
        # Acceso seguro
        tamano_bd = info.get('tamano_bd', 0)
        st.metric("Muestras en BD", tamano_bd)
        if tamano_bd == 0:
            st.warning("⚠️ Base de datos vacía. Entrene primero el sistema.")
    
    if st.button("🔍 Predecir Caracteres", type="primary"):
        with st.spinner("Prediciendo caracteres..."):
            # TODO: Los resultados ya vienen ordenados de izquierda a derecha desde el backend
            resultados = algo3.predecir_caracteres(use_knn, k_vecinos)
            st.session_state.resultados_prediccion = resultados
            
            if resultados:
                st.success(f"✅ {len(resultados)} caracteres predictos (Ordenados de Izquierda a Derecha)")
            else:
                st.warning("⚠️ No se pudieron predecir caracteres")
    
    if not st.session_state.resultados_prediccion:
        return
    
    # Mostrar resultados de predicción - Misma información del original
    st.subheader("Resultados de Predicción")
    
    # Imagen con predicciones
    try:
        imagen_predicciones = algo3.generar_imagen_predicciones(st.session_state.resultados_prediccion)
        st.image(imagen_predicciones, caption="Predicciones (Ordenadas de Izquierda a Derecha)", use_column_width=True)
    except Exception as e:
        st.error(f"Error generando imagen de predicciones: {e}")
    
    # Tabla de resultados - Mismos datos del original
    datos_prediccion = []
    placa_reconstruida = ""
    
    for i, resultado in enumerate(st.session_state.resultados_prediccion, 1):
        char_pred = resultado.get("char_pred", "?")
        distancia = resultado.get("dist", 0.0)
        bbox = resultado["bbox"]
        
        datos_prediccion.append({
            "#": i,
            "Área": resultado["area"],
            "BBox": f"({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})",
            "Carácter": char_pred,
            "Distancia": f"{distancia:.2f}" if distancia < float('inf') else "∞"
        })
        
        placa_reconstruida += char_pred
    
    if datos_prediccion:
        df_prediccion = pd.DataFrame(datos_prediccion)
        st.dataframe(df_prediccion, use_container_width=True, hide_index=True)
    
    # Mostrar placa reconstruida - Misma funcionalidad del original
    st.subheader("🪪 Placa Reconstruida")
    if placa_reconstruida:
        st.markdown(f"<h1 style='text-align: center; color: #0068c9;'>{placa_reconstruida}</h1>", 
                    unsafe_allow_html=True)
        st.info("ℹ️ Placa reconstruida en orden natural de lectura (izquierda a derecha)")
    else:
        st.info("ℹ️ No se pudo reconstruir la placa")


def show_configuracion(algo3: Algoritmo3):
    """Interfaz de configuración - TODAS las opciones del original"""
    
    st.header("⚙️ Configuración del Sistema")
    
    st.subheader("Base de Datos")
    
    info = algo3.get_info_imagen()
    # Acceso seguro
    st.metric("Muestras en Base de Datos", info.get('tamano_bd', 0))
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Recargar Base de Datos", use_container_width=True):
            algo3.cargar_base_datos()
            st.success("✅ Base de datos recargada")
    
    with col2:
        if st.button("🗑️ Limpiar Etiquetas", use_container_width=True):
            algo3.limpiar_etiquetas()
            st.session_state.etiquetas_clasificadas = {}
            st.session_state.resultados_prediccion = []
            st.session_state.procesamiento_completo = False
            st.success("✅ Etiquetas limpiadas")
    
    st.subheader("Configuración de Área")
    
    col1, col2 = st.columns(2)
    
    with col1:
        nuevo_area_min = st.number_input("Área mínima", 0, 100000, algo3.area_min)
    
    with col2:
        nuevo_area_max = st.number_input("Área máxima", 0, 100000, algo3.area_max)
    
    if st.button("💾 Guardar Configuración de Área"):
        algo3.area_min = nuevo_area_min
        algo3.area_max = nuevo_area_max
        st.success("✅ Configuración de área guardada")
    
    st.subheader("Información del Sistema")
    
    # Información completa del sistema
    st.json({
        "base_datos_ruta": str(algo3.db_file),
        "area_minima": algo3.area_min,
        "area_maxima": algo3.area_max,
        "imagen_cargada": info.get("tiene_imagen", False),
        "objetos_etiquetados": info.get("num_objetos", 0),
        "funcionalidad_orden": "Objetos ordenados automáticamente de izquierda a derecha"
    })


# Función principal para uso standalone
if __name__ == "__main__":
    show_algoritmo_3()