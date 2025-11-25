"""
Módulo principal del Algoritmo 3 - Procesamiento de reconocimiento de patentes
TODO: Código movido desde ENTRENAMIENTO_PLACAS_V3.py y PREDICCION_PLACAS_V2.py
"""

import numpy as np
import os
import csv
import random
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image, ImageOps, ImageDraw, ImageFont
import threading

try:
    from scipy import ndimage
    from scipy.signal import convolve2d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class Algoritmo3:
    """Clase principal que encapsula toda la lógica del Algoritmo 3"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.img_original_pil = None
        self.img_gray_pil = None
        self.img_gray_np = None
        self.mask_01 = None
        self.mask_inv_01 = None
        self.morph_result = None
        
        self.labels = None
        self.num_labels = 0
        self.props = []
        self.colored_label_image_pil = None
        
        # Configuración por defecto
        self.area_min = 3000
        self.area_max = 50000
        
        # Base de datos
        if db_path is None:
            base_dir = os.path.join(
                os.path.expanduser("~"),
                "Documents",
                "INTELIGENCIA ARTIFICIAL", 
                "CODIGO_PLACAS"
            )
            os.makedirs(base_dir, exist_ok=True)
            self.db_file = os.path.join(base_dir, "base_de_datos_entrenada.csv")
        else:
            self.db_file = db_path
            
        self.clasificaciones = {}
        self.base_datos = []
        self.X = None
        self.y = None
        self.feat_mean = None
        self.feat_std = None
        
        self.cargar_base_datos()
        
        try:
            self.font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            self.font = ImageFont.load_default()

    # ============ BASE DE DATOS Y KNN ============

    def cargar_base_datos(self) -> List[Dict]:
        """Carga la base de datos desde CSV - TODO: Movido desde ENTRENAMIENTO_PLACAS_V3.py"""
        data = []
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            # Convertir todos los campos necesarios
                            processed_row = {
                                'caracter': row.get('caracter', '').strip(),
                                'area': int(row.get('area', 0)),
                                'num_pixeles': int(row.get('num_pixeles', 0)),
                                'perimetro': int(row.get('perimetro', 0)),
                                'circularidad': float(row.get('circularidad', 0.0)),
                                'bbox_xmin': int(row.get('bbox_xmin', 0)),
                                'bbox_ymin': int(row.get('bbox_ymin', 0)),
                                'bbox_xmax': int(row.get('bbox_xmax', 0)),
                                'bbox_ymax': int(row.get('bbox_ymax', 0)),
                                'bbox_width': int(row.get('bbox_width', 0)),
                                'bbox_height': int(row.get('bbox_height', 0)),
                                'aspect_ratio': float(row.get('aspect_ratio', 0.0)),
                                'centroid_x': float(row.get('centroid_x', 0.0)),
                                'centroid_y': float(row.get('centroid_y', 0.0)),
                                'centroid_local_x': float(row.get('centroid_local_x', 0.0)),
                                'centroid_local_y': float(row.get('centroid_local_y', 0.0)),
                                'hu1': float(row.get('hu1', 0.0)),
                                'hu2': float(row.get('hu2', 0.0)),
                                'hu3': float(row.get('hu3', 0.0)),
                                'hu4': float(row.get('hu4', 0.0)),
                                'hu5': float(row.get('hu5', 0.0)),
                                'hu6': float(row.get('hu6', 0.0)),
                                'hu7': float(row.get('hu7', 0.0)),
                            }
                            data.append(processed_row)
                        except Exception as e:
                            continue
            except Exception as e:
                print(f"Error cargando base de datos: {e}")
                
        self.base_datos = data
        self._preparar_datos_knn()
        return data

    def _preparar_datos_knn(self):
        """Prepara los datos para KNN - TODO: Movido desde PREDICCION_PLACAS_V2.py"""
        if not self.base_datos:
            self.X = None
            self.y = None
            return
            
        X, y = [], []
        for item in self.base_datos:
            try:
                car = item['caracter']
                if not car:
                    continue
                feat = [
                    float(item['area']),
                    float(item['perimetro']),
                    float(item['circularidad']),
                    float(item['aspect_ratio']),
                    float(item['hu1']),
                    float(item['hu2']),
                    float(item['hu3']),
                    float(item['hu4']),
                    float(item['hu5']),
                    float(item['hu6']),
                    float(item['hu7']),
                ]
                X.append(feat)
                y.append(car)
            except Exception:
                continue
                
        if X:
            self.X = np.array(X, dtype=np.float64)
            self.y = np.array(y)
            self.feat_mean = self.X.mean(axis=0)
            self.feat_std = self.X.std(axis=0)
            self.feat_std[self.feat_std == 0] = 1.0
            self.X = (self.X - self.feat_mean) / self.feat_std
        else:
            self.X = None
            self.y = None

    def guardar_base_datos(self) -> bool:
        """Guarda la base de datos en CSV - TODO: Movido desde ENTRENAMIENTO_PLACAS_V3.py"""
        try:
            fieldnames = [
                "caracter", "area", "num_pixeles", "perimetro", "circularidad",
                "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", 
                "bbox_width", "bbox_height", "aspect_ratio",
                "centroid_x", "centroid_y", "centroid_local_x", "centroid_local_y",
                "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7"
            ]
            
            with open(self.db_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for item in self.base_datos:
                    writer.writerow(item)
            return True
        except Exception as e:
            print(f"Error guardando base de datos: {e}")
            return False

    def predecir_knn(self, feat_vec: List[float], k: int = 5) -> Tuple[str, float]:
        """Predice usando KNN - TODO: Movido desde PREDICCION_PLACAS_V2.py"""
        if self.X is None or self.y is None:
            return "?", float("inf")
            
        try:
            k = int(k)
            if k < 1:
                k = 1
        except Exception:
            k = 5

        fv = np.array(feat_vec, dtype=np.float64)
        fv = (fv - self.feat_mean) / self.feat_std

        # Distancia Cityblock (L1)
        dists = np.sum(np.abs(self.X - fv), axis=1)
        idx = np.argsort(dists)
        k = min(k, len(idx))
        idx_k = idx[:k]
        d_k = dists[idx_k]
        y_k = self.y[idx_k]

        best_char = "?"
        best_score = float("inf")
        for c in set(y_k):
            mask = (y_k == c)
            score = d_k[mask].mean()
            if score < best_score:
                best_score = score
                best_char = c
                
        return best_char, float(best_score)

    # ============ PROCESAMIENTO DE IMÁGENES ============

    def cargar_imagen(self, file_path: str) -> bool:
        """Carga una imagen para procesamiento"""
        try:
            img = Image.open(file_path)
            self.img_original_pil = img.convert("RGB")
            self.img_gray_pil = ImageOps.grayscale(self.img_original_pil)
            self.img_gray_np = np.array(self.img_gray_pil)
            self.clasificaciones = {}
            return True
        except Exception as e:
            print(f"Error cargando imagen: {e}")
            return False

    def aplicar_umbral(self, umbral: int) -> bool:
        """Aplica umbral a la imagen - TODO: Lógica movida desde actualizar_vistas_completas"""
        if self.img_gray_np is None:
            return False
            
        self.mask_01 = (self.img_gray_np > umbral).astype(np.uint8)
        self.mask_inv_01 = 1 - self.mask_01
        return True

    def calcular_umbral_otsu(self, gray_np: np.ndarray) -> int:
        """Calcula umbral óptimo usando Otsu - TODO: Movido desde ambos scripts"""
        hist, _ = np.histogram(gray_np.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        total = gray_np.size
        if total == 0:
            return 128
            
        sum_total = np.dot(np.arange(256), hist)
        sumB = 0.0
        wB = 0.0
        var_max = 0.0
        threshold = 128
        
        for i in range(256):
            wB += hist[i]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += i * hist[i]
            mB = sumB / wB
            mF = (sum_total - sumB) / wF
            var_between = wB * wF * (mB - mF) ** 2
            if var_between > var_max:
                var_max = var_between
                threshold = i
                
        return threshold

    # ============ OPERACIONES MORFOLÓGICAS ============

    def aplicar_morfologia(self, operacion: str = "Cerramiento", forma: str = "Disco", radio: int = 3) -> bool:
        """Aplica operaciones morfológicas - TODO: Movido desde _aplicar_morfologia_thread"""
        if self.mask_inv_01 is None:
            return False
            
        try:
            bin_img = self.mask_inv_01.astype(np.uint8)
            kernel = self.crear_elemento_estructurante(forma, radio)

            if operacion == "Erosión":
                result = self.erode(bin_img, kernel)
            elif operacion == "Dilatación":
                result = self.dilate(bin_img, kernel)
            elif operacion == "Apertura":
                result = self.dilate(self.erode(bin_img, kernel), kernel)
            elif operacion == "Cerramiento":
                result = self.erode(self.dilate(bin_img, kernel), kernel)
            else:
                result = bin_img.copy()

            self.morph_result = result
            return True
        except Exception as e:
            print(f"Error en morfología: {e}")
            return False

    @staticmethod
    def crear_elemento_estructurante(forma: str, radio: int) -> np.ndarray:
        """Crea elemento estructurante - TODO: Movido desde ambos scripts"""
        r = max(1, int(radio))
        size = 2 * r + 1
        if forma == "Cuadrado":
            kernel = np.ones((size, size), dtype=np.uint8)
        elif forma == "Cruz":
            kernel = np.zeros((size, size), dtype=np.uint8)
            kernel[r, :] = 1
            kernel[:, r] = 1
        elif forma == "Disco":
            y, x = np.ogrid[-r:r+1, -r:r+1]
            mask = x * x + y * y <= r * r
            kernel = mask.astype(np.uint8)
        else:
            kernel = np.ones((size, size), dtype=np.uint8)
        return kernel

    @staticmethod
    def dilate(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Dilatación morfológica - TODO: Movido desde ambos scripts"""
        binary = (binary > 0).astype(np.uint8)
        kh, kw = kernel.shape
        rh, rw = kh // 2, kw // 2
        H, W = binary.shape
        
        if SCIPY_AVAILABLE:
            try:
                result = convolve2d(binary, kernel, mode='same', boundary='fill', fillvalue=0)
                return (result > 0).astype(np.uint8)
            except Exception:
                pass
                
        # Fallback manual
        padded = np.pad(binary, ((rh, rh), (rw, rw)), mode="constant", constant_values=0)
        result = np.zeros((H, W), dtype=np.uint8)
        ys, xs = np.where(kernel > 0)
        for yk, xk in zip(ys, xs):
            region = padded[yk:yk+H, xk:xk+W]
            result = np.maximum(result, region)
        return result

    @staticmethod
    def erode(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Erosión morfológica - TODO: Movido desde ambos scripts"""
        binary = (binary > 0).astype(np.uint8)
        kh, kw = kernel.shape
        rh, rw = kh // 2, kw // 2
        H, W = binary.shape
        
        if SCIPY_AVAILABLE:
            try:
                kernel_sum = np.sum(kernel)
                result = convolve2d(binary, kernel, mode='same', boundary='fill', fillvalue=1)
                return (result == kernel_sum).astype(np.uint8)
            except Exception:
                pass
                
        # Fallback manual
        padded = np.pad(binary, ((rh, rh), (rw, rw)), mode="constant", constant_values=1)
        result = np.ones((H, W), dtype=np.uint8)
        ys, xs = np.where(kernel > 0)
        for yk, xk in zip(ys, xs):
            region = padded[yk:yk+H, xk:xk+W]
            result = np.minimum(result, region)
        return result

    # ============ ETIQUETADO Y CARACTERÍSTICAS ============

    def etiquetar_objetos(self, area_min: Optional[int] = None, area_max: Optional[int] = None) -> bool:
        """Etiqueta objetos en la imagen - TODO: Movido desde _etiquetar_objetos_thread"""
        if self.morph_result is not None:
            binary = (self.morph_result > 0).astype(np.uint8)
        elif self.mask_inv_01 is not None:
            binary = (self.mask_inv_01 > 0).astype(np.uint8)
        else:
            return False

        if area_min is None:
            area_min = self.area_min
        if area_max is None:
            area_max = self.area_max

        try:
            labels, num = self.connected_components_labeling_fast(binary)
            all_props = self.calcular_propiedades(labels, num)
            
            # Filtrar por área y relación de aspecto
            props = []
            for p in all_props:
                x1, y1, x2, y2 = p["bbox"]
                ancho = x2 - x1 + 1
                alto = y2 - y1 + 1
                aspect = ancho / max(1, alto)

                if (area_min <= p["area"] <= area_max) and (0.1 < aspect < 0.9):
                    props.append(p)

            # Reetiquetar
            label_map = {0: 0}
            new_label = 0
            for p in props:
                old_label = p["label"]
                new_label += 1
                label_map[old_label] = new_label
                p["label"] = new_label

            labels_filtrado = np.zeros_like(labels)
            for old_label, new_label in label_map.items():
                if old_label > 0:
                    labels_filtrado[labels == old_label] = new_label

            self.labels = labels_filtrado
            self.props = props
            self.num_labels = len(props)
            
            return True
        except Exception as e:
            print(f"Error en etiquetado: {e}")
            return False

    def connected_components_labeling_fast(self, binary: np.ndarray) -> Tuple[np.ndarray, int]:
        """Componentes conectados con scipy o fallback - TODO: Movido desde ambos scripts"""
        if SCIPY_AVAILABLE:
            try:
                labeled, num = ndimage.label(binary)
                return labeled, num
            except Exception:
                pass
        return self.connected_components_labeling(binary)

    @staticmethod
    def connected_components_labeling(binary: np.ndarray, connectivity: int = 8) -> Tuple[np.ndarray, int]:
        """Componentes conectados manual - TODO: Movido desde ambos scripts"""
        H, W = binary.shape
        labels = np.zeros((H, W), dtype=np.int32)
        current_label = 0
        
        if connectivity == 8:
            neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
            
        visited = np.zeros_like(binary, dtype=bool)
        
        for y in range(H):
            for x in range(W):
                if binary[y, x] and not visited[y, x]:
                    current_label += 1
                    stack = [(y, x)]
                    visited[y, x] = True
                    labels[y, x] = current_label
                    
                    while stack:
                        cy, cx = stack.pop()
                        for dy, dx in neighbors:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                if binary[ny, nx] and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    labels[ny, nx] = current_label
                                    stack.append((ny, nx))
                                    
        return labels, current_label

    @staticmethod
    def calcular_propiedades(labels: np.ndarray, num: int) -> List[Dict]:
        """Calcula propiedades de objetos etiquetados - TODO: Movido desde ambos scripts"""
        props = []
        if num == 0:
            return props
            
        for lab in range(1, num + 1):
            mask = labels == lab
            if not np.any(mask):
                continue
                
            ys, xs = np.where(mask)
            area = len(ys)
            if area == 0:
                continue
                
            centroid_x = float(xs.mean())
            centroid_y = float(ys.mean())
            bbox = (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys)))
            
            props.append({
                "label": lab,
                "area": int(area),
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "bbox": bbox
            })
            
        return props

    def extraer_medidas_objeto(self, etiqueta: int, prop: Dict) -> Tuple[int, int, float, float, float, List[float]]:
        """Extrae medidas detalladas de un objeto - TODO: Movido desde ambos scripts"""
        if self.labels is None:
            return 0, 0, 0.0, 0.0, 0.0, [0.0] * 7

        x1, y1, x2, y2 = prop["bbox"]
        x2i, y2i = x2 + 1, y2 + 1

        mask_lab = (self.labels == etiqueta).astype(np.uint8)
        mask_crop = mask_lab[y1:y2i, x1:x2i]

        num_pixeles = int(mask_crop.sum())
        if num_pixeles == 0:
            return 0, 0, 0.0, 0.0, 0.0, [0.0] * 7

        ys, xs = np.where(mask_crop > 0)
        centroid_local_x = float(xs.mean())
        centroid_local_y = float(ys.mean())

        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded = self.erode(mask_crop, kernel)
        contorno = mask_crop - eroded
        perimetro = int(np.count_nonzero(contorno))

        circularidad = 0.0
        if perimetro > 0:
            circularidad = 4.0 * np.pi * num_pixeles / float(perimetro ** 2)

        hu = self.calcular_momentos_hu(mask_crop, centroid_local_x, centroid_local_y, num_pixeles)
        return num_pixeles, perimetro, circularidad, centroid_local_x, centroid_local_y, hu

    @staticmethod
    def calcular_momentos_hu(mask_crop: np.ndarray, cx: float, cy: float, m00: int) -> List[float]:
        """Calcula momentos de Hu - TODO: Movido desde ambos scripts"""
        mask = mask_crop > 0
        if m00 <= 0 or not np.any(mask):
            return [0.0] * 7

        ys, xs = np.where(mask)
        x_shift = xs - cx
        y_shift = ys - cy

        mu20 = np.sum(x_shift ** 2)
        mu02 = np.sum(y_shift ** 2)
        mu11 = np.sum(x_shift * y_shift)
        mu30 = np.sum(x_shift ** 3)
        mu03 = np.sum(y_shift ** 3)
        mu21 = np.sum((x_shift ** 2) * y_shift)
        mu12 = np.sum(x_shift * (y_shift ** 2))

        def eta(mu, p, q):
            return mu / (m00 ** (1.0 + (p + q) / 2.0))

        eta20 = eta(mu20, 2, 0)
        eta02 = eta(mu02, 0, 2)
        eta11 = eta(mu11, 1, 1)
        eta30 = eta(mu30, 3, 0)
        eta03 = eta(mu03, 0, 3)
        eta21 = eta(mu21, 2, 1)
        eta12 = eta(mu12, 1, 2)

        phi1 = eta20 + eta02
        phi2 = (eta20 - eta02) ** 2 + 4.0 * (eta11 ** 2)
        phi3 = (eta30 - 3.0 * eta12) ** 2 + (3.0 * eta21 - eta03) ** 2
        phi4 = (eta30 + eta12) ** 2 + (eta21 + eta03) ** 2
        phi5 = (eta30 - 3.0 * eta12) * (eta30 + eta12) * (
            (eta30 + eta12) ** 2 - 3.0 * (eta21 + eta03) ** 2
        ) + (3.0 * eta21 - eta03) * (eta21 + eta03) * (
            3.0 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2
        )
        phi6 = (eta20 - eta02) * (
            (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2
        ) + 4.0 * eta11 * (eta30 + eta12) * (eta21 + eta03)
        phi7 = (3.0 * eta21 - eta03) * (eta30 + eta12) * (
            (eta30 + eta12) ** 2 - 3.0 * (eta21 + eta03) ** 2
        ) - (eta30 - 3.0 * eta12) * (eta21 + eta03) * (
            3.0 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2
        )

        return [float(phi1), float(phi2), float(phi3), float(phi4), 
                float(phi5), float(phi6), float(phi7)]

    # ============ CLASIFICACIÓN Y GUARDADO ============

    def clasificar_objeto(self, etiqueta: int, caracter: str) -> bool:
        """Clasifica un objeto con un carácter - TODO: Movido desde clasificar_objeto"""
        if not caracter or len(caracter) != 1 or not (caracter.isalpha() or caracter.isdigit()):
            return False
            
        self.clasificaciones[etiqueta] = caracter.upper()
        return True

    def guardar_en_base_datos(self) -> int:
        """Guarda clasificaciones en la base de datos - TODO: Movido desde guardar_en_base_datos"""
        if not self.clasificaciones or not self.props:
            return 0

        guardados = 0
        for etiqueta, caracter in self.clasificaciones.items():
            prop = next((p for p in self.props if p["label"] == etiqueta), None)
            if prop is None:
                continue

            x1, y1, x2, y2 = prop["bbox"]
            bbox_width = x2 - x1 + 1
            bbox_height = y2 - y1 + 1
            aspect_ratio = bbox_width / float(max(1, bbox_height))

            num_pixeles, perimetro, circularidad, cx_local, cy_local, hu = \
                self.extraer_medidas_objeto(etiqueta, prop)

            muestra = {
                "caracter": caracter,
                "area": prop["area"],
                "num_pixeles": num_pixeles if num_pixeles > 0 else prop["area"],
                "perimetro": perimetro,
                "circularidad": circularidad,
                "bbox_xmin": x1,
                "bbox_ymin": y1,
                "bbox_xmax": x2,
                "bbox_ymax": y2,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "aspect_ratio": aspect_ratio,
                "centroid_x": prop["centroid_x"],
                "centroid_y": prop["centroid_y"],
                "centroid_local_x": cx_local,
                "centroid_local_y": cy_local,
                "hu1": hu[0], "hu2": hu[1], "hu3": hu[2], "hu4": hu[3],
                "hu5": hu[4], "hu6": hu[5], "hu7": hu[6],
            }

            self.base_datos.append(muestra)
            guardados += 1

        if guardados > 0 and self.guardar_base_datos():
            self.clasificaciones = {}
            self._preparar_datos_knn()
            
        return guardados

    # ============ GENERACIÓN DE IMÁGENES ============

    def generar_imagen_coloreada(self, labels: Optional[np.ndarray] = None, 
                               props: Optional[List[Dict]] = None) -> Image.Image:
        """Genera imagen coloreada con etiquetas - TODO: Movido desde generar_imagen_coloreada"""
        if labels is None:
            labels = self.labels
        if props is None:
            props = self.props
        if labels is None:
            return Image.new('RGB', (100, 100), color='white')

        H, W = labels.shape
        colored = np.zeros((H, W, 3), dtype=np.uint8)
        random.seed(42)
        color_map = {0: (0, 0, 0)}
        
        for p in props:
            lab = p["label"]
            color_map[lab] = (
                random.randint(30, 255),
                random.randint(30, 255),
                random.randint(30, 255),
            )

        for lab, col in color_map.items():
            if lab == 0:
                continue
            mask = labels == lab
            colored[mask] = col

        pil_img = Image.fromarray(colored.astype(np.uint8))
        
        if len(props) < 100:
            draw = ImageDraw.Draw(pil_img)
            for p in props:
                lab = p["label"]
                cx = p["centroid_x"]
                cy = p["centroid_y"]
                text = str(lab)
                try:
                    bbox = draw.textbbox((0, 0), text, font=self.font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                except Exception:
                    text_w, text_h = (8 * len(text), 12)
                    
                tx = int(round(cx - text_w / 2))
                ty = int(round(cy - text_h / 2))
                draw.text((tx + 1, ty + 1), text, fill=(0, 0, 0), font=self.font)
                draw.text((tx, ty), text, fill=(255, 255, 255), font=self.font)

        return pil_img

    def generar_imagen_predicciones(self, resultados: List[Dict]) -> Image.Image:
        """Genera imagen con predicciones - TODO: Adaptado desde PREDICCION_PLACAS_V2.py"""
        if self.labels is None:
            return Image.new('RGB', (100, 100), color='white')

        H, W = self.labels.shape
        colored = np.zeros((H, W, 3), dtype=np.uint8)
        random.seed(42)
        color_map = {0: (0, 0, 0)}
        
        for p in resultados:
            lab = p["label"]
            color_map[lab] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255),
            )

        for lab, col in color_map.items():
            if lab == 0:
                continue
            mask = self.labels == lab
            colored[mask] = col

        pil_img = Image.fromarray(colored.astype(np.uint8))
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        for p in resultados:
            cx = p["centroid_x"]
            cy = p["centroid_y"]
            ch = p.get("char_pred", "?")
            text = ch
            
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
            except Exception:
                w, h = (10 * len(text), 16)
                
            tx = int(round(cx - w / 2))
            ty = int(round(cy - h / 2))
            draw.text((tx + 1, ty + 1), text, fill=(0, 0, 0), font=font)
            draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

        return pil_img

    # ============ PREDICCIÓN AUTOMÁTICA ============

    def predecir_caracteres(self, use_knn: bool = True, k: int = 5) -> List[Dict]:
        """Predice caracteres automáticamente - TODO: Adaptado desde PREDICCION_PLACAS_V2.py"""
        if self.labels is None or not self.props:
            return []

        resultados = []
        for p in self.props:
            num_pix, perim, circ, cx_local, cy_local, hu = \
                self.extraer_medidas_objeto(p["label"], p)
                
            x1, y1, x2, y2 = p["bbox"]
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            aspect_ratio = w / float(max(1, h))

            feat_vec = [
                float(p["area"]),
                float(perim),
                float(circ),
                float(aspect_ratio),
                float(hu[0]), float(hu[1]), float(hu[2]), float(hu[3]),
                float(hu[4]), float(hu[5]), float(hu[6]),
            ]

            if use_knn and self.X is not None:
                char_pred, dist = self.predecir_knn(feat_vec, k)
            else:
                char_pred, dist = "?", float("inf")

            p["char_pred"] = char_pred
            p["dist"] = dist
            resultados.append(p)

        return resultados

    # ============ UTILIDADES ============

    def limpiar_etiquetas(self):
        """Limpia el estado de etiquetado"""
        self.labels = None
        self.num_labels = 0
        self.props = []
        self.colored_label_image_pil = None
        self.clasificaciones = {}

    def get_info_imagen(self) -> Dict[str, Any]:
        """Obtiene información de la imagen actual"""
        if self.img_original_pil is None:
            return {"ancho": 0, "alto": 0, "tiene_imagen": False}
            
        ancho, alto = self.img_original_pil.size
        return {
            "ancho": ancho,
            "alto": alto, 
            "tiene_imagen": True,
            "num_objetos": self.num_labels,
            "tamano_bd": len(self.base_datos)
        }

    def get_imagen_original(self) -> Optional[Image.Image]:
        return self.img_original_pil

    def get_imagen_gris(self) -> Optional[Image.Image]:
        return self.img_gray_pil

    def get_mascara_binaria(self) -> Optional[Image.Image]:
        if self.mask_01 is not None:
            return Image.fromarray((self.mask_01 * 255).astype(np.uint8))
        return None

    def get_mascara_invertida(self) -> Optional[Image.Image]:
        if self.mask_inv_01 is not None:
            return Image.fromarray((self.mask_inv_01 * 255).astype(np.uint8))
        return None

    def get_resultado_morfologia(self) -> Optional[Image.Image]:
        if self.morph_result is not None:
            return Image.fromarray((self.morph_result * 255).astype(np.uint8))
        return None


# Funciones de conveniencia para mantener compatibilidad
def crear_elemento_estructurante(forma: str, radio: int) -> np.ndarray:
    """Wrapper para mantener compatibilidad - TODO: Movido desde ambos scripts"""
    return Algoritmo3.crear_elemento_estructurante(forma, radio)

def dilate(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Wrapper para mantener compatibilidad - TODO: Movido desde ambos scripts"""
    return Algoritmo3.dilate(binary, kernel)

def erode(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Wrapper para mantener compatibilidad - TODO: Movido desde ambos scripts"""
    return Algoritmo3.erode(binary, kernel)

def calcular_umbral_otsu(gray_np: np.ndarray) -> int:
    """Wrapper para mantener compatibilidad - TODO: Movido desde ambos scripts"""
    return Algoritmo3.calcular_umbral_otsu(gray_np)

def connected_components_labeling(binary: np.ndarray, connectivity: int = 8) -> Tuple[np.ndarray, int]:
    """Wrapper para mantener compatibilidad - TODO: Movido desde ambos scripts"""
    return Algoritmo3.connected_components_labeling(binary, connectivity)