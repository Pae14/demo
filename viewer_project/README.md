# Tactile Defect Inspection System: Advanced Metrology & AI Analysis (3D + 2D)

Este proyecto presenta una solución integral de **inspección visual-táctil 3D y 2D** diseñada para la detección, segmentación, metrología de profundidad y clasificación del estado de reparabilidad de defectos superficiales en entornos industriales. Utilizando el sensor biomimético **GelSight Mini**, la aplicación trasciende la visión convencional al capturar tanto la textura microscópica como la topografía 3D (nube de puntos) de las superficies.

La herramienta integra algoritmos de **Deep Learning** (YOLOv12, YOLOv26, RF-DETR, ResNet18 y CNN) con técnicas de **Visión Artificial clásica** (GLCM de Haralick), **Metrología 3D** (estimación de profundidad respecto a plano local ajustado, volumen de pérdida de material) y **Descriptores PCL 3D avanzados** (PFH y SHOT) para determinar con precisión matemática si un componente dañado puede ser reparado (*REPAIRABLE*) o debe desecharse (*NON-REPAIRABLE*).

---

## Arquitectura del Sistema (File Structure)

El software sigue un diseño modular y desacoplado para garantizar la mantenibilidad y el rendimiento en tiempo real en `/home/aurova/Desktop/paula/demo_lfs/viewer_project`:

| Archivo / Carpeta | Función |
| :--- | :--- |
| **`main.py`** | **Orquestador Central**. Ciclo de vida de la aplicación, navegación entre pestañas (Offline / Online ROS 2) y GUI en PyQt6 con soporte 3D y RViz. |
| **`workers.py`** | **Capa de Concurrencia e Integración**. Hilos de adquisición (`GSThread` para cámara local, `ROS2Thread` para topic de imagen `/gs_mini_img` y nube `/gs_mini_pcd`), sincronización temporal de nubes e inferencia IA en GPU/CPU (`DEVICE`). |
| **`utils_vision.py`** | **Motor Analítico 2D y 3D**. Métricas dimensionales, descriptores texturales GLCM, ajuste de planos por mínimos cuadrados sobre anillo perimetral, cálculo de profundidad P95, media, máxima y volumen. |
| **`pointcloud_export.py`** | **Gestor de Nubes 3D y Descriptores**. Exportación sincronizada de PCD, cálculo de normales de superficie, extracción de descriptores PFH (125 bins) y SHOT (352 bins) e integración de predicción de reparabilidad. |
| **`repairability.py`** | **Pipeline de Clasificación Supervisada**. Extracción y preparación de vectores de características (geometría, descriptores 3D o combinación de ambos) e inferencia con modelos entrenados (`repairability.joblib`). |
| **`train_repairability.py`** | **Herramienta de Entrenamiento**. Creación de plantillas CSV (`exports/labels.csv`), entrenamiento y validación cruzada agrupada por pieza (`GroupKFold`) evaluando F1-score y matrices de confusión. |
| **`segment_metrics.py`** | **Procesamiento de Métricas por Lotes**. Segmentación automática de datasets completos, extracción de métricas 2D + 3D por máscara y exportación a CSV. |
| **`build_excel_3d.py`** | **Consolidación de Datos**. Agrupación de métricas 3D, descriptores y datos de inspección en tablas estructuradas para análisis estadístico. |
| **`rgb_to_depth.py`** | **Reconstrucción de Profundidad Offline**. Conversión de pares de imágenes a mapas de profundidad y nubes de puntos `.npy`. |
| **`pcl_descriptor_export/`** | **Módulo C++ Nativo PCL**. Exportador de descriptores PFH/SHOT acelerado por la biblioteca PCL compilado dentro del contenedor Docker. |
| **`ui_components.py`** | **Framework de UI**. Estética *Glassmorphism*, paneles de visualización, controles de inferencia, botones animados y tarjetas de métricas. |
| **`config.py`** | **Fuente de Verdad**. Rutas de modelos, calibración GelSight (mm/px), umbrales de confianza, tópicos ROS y límites de reparabilidad 3D. |
| **`models/`** | Checkpoints entrenados de YOLO12, YOLO26, RF-DETR, ResNet18 y CNN. |
| **`exports/`** | Salida de informes CSV, nubes PCD exportadas, descriptores PFH/SHOT y modelos `repairability.joblib`. |

---

## Concurrencia y Rendimiento (Multi-threading)

Para garantizar un rendimiento industrial de alta tasa de refresco sin latencia ni bloqueos de interfaz:

* **UI Thread (PyQt6)**: Gestión del renderizado gráfico, visualización interactiva y eventos a 60 FPS.
* **Acquisition Thread (`ROS2Thread`)**: Subscripción no bloqueante con QoS `BEST_EFFORT` a `/gs_mini_img` y `/gs_mini_pcd`. Implementa una cola de sincronización temporal exacta: empareja cada imagen con la nube de puntos que comparte el mismo timestamp de captura.
* **Inference Thread (`InferenceWorker`)**: Inferencia asíncrona desacoplada de la visualización. El regulador *Inference Interval* permite modular la carga de la GPU para mantener fluidez total.
* **Point Cloud Processing Worker**: Ajuste de plano local y cálculo de volumen ejecutados con operaciones vectorizadas en NumPy y SciPy.

---

## Integración con ROS 2 (Robot Operating System)

La aplicación actúa como un nodo ROS 2 nativo bajo **ROS 2 Humble**:

* **`/gs_mini_img` (`sensor_msgs/msg/Image`)**: Flujo de imagen RGB táctil a 320×240 (decodificado mediante `cv_bridge`).
* **`/gs_mini_pcd` (`sensor_msgs/msg/PointCloud2`)**: Nube de puntos 3D organizada pixel a pixel (altura Z en metros).
* **Control de Vista (`Output View`)**: Alterna en tiempo real entre:
  1. *Segmentación*: Máscara superpuesta sobre el frame óptico.
  2. *Color Depth*: Mapa de elevación pseudocolor TURBO escalado dinámicamente según percentiles 2% y 98% de la coordenada Z.
  3. *Combined*: Fusión alfa de la segmentación del defecto sobre el mapa de elevación 3D.
* **Soporte RViz2**: Configuración incluida en `rviz/gs_pointcloud.rviz` en el frame `gelsightmini_camera_frame`.

---

## Desglose Exhaustivo de Métricas de Reparabilidad (2D + 3D)

La evaluación de integridad estructural y viabilidad de reparación combina tres familias de descriptores matemáticos complementarios:

### 1. Métricas Dimensionales y Morfológicas 2D

Calculadas sobre la máscara binaria segmentada del defecto, proyectada a unidades métricas reales mediante el campo de visión del GelSight Mini ($18.67\text{ mm} \times 14.00\text{ mm}$ para $320 \times 240$ px, i.e., $A_{px} \approx 0.00341\text{ mm}^2/\text{px}$):

* **Área ($mm^2$)**: Cuantifica la extensión bidimensional del daño sobre la superficie:

```math
\text{Área} = N_{\text{pixeles}} \times \left( \frac{\text{FOV}_W \times \text{FOV}_H}{W \times H} \right)
```

* **Perímetro ($mm$)**: Longitud a lo largo del contorno exterior continuo de la máscara:

```math
\text{Perímetro} = P_{\text{pixeles}} \times \sqrt{\frac{\text{FOV}_W \times \text{FOV}_H}{W \times H}}
```

* **Excentricidad ($e$)**: Relación de aspecto de la elipse equivalente ($a$: semieje mayor, $b$: semieje menor):

```math
e = \sqrt{1 - \frac{b^2}{a^2}}
```

  * $e \to 0$: Defecto circular o isotrópico (típico de agujeros o picaduras por impacto).
  * $e \to 1$: Defecto lineal o alargado (típico de grietas longitudinales o fisuras por tracción).

* **Compacidad / Circularidad ($C$)**: Mide la regularidad del contorno respecto a un círculo perfecto ($C = 1$):

```math
C = \frac{4\pi \cdot \text{Área}}{\text{Perímetro}^2}
```

* **Extensión de Ejes por SVD (Longitud y Anchura en $mm$)**:
  Mediante descomposición en valores singulares (SVD) de las coordenadas $(x, y)$ centradas del defecto, se determina el vector director de mayor varianza:
  * **Longitud (`major_axis_length_mm`)**: Extensión máxima en el eje principal.
  * **Anchura (`minor_axis_length_mm`)**: Extensión transversal máxima.

---

### 2. Métricas Texturales de Haralick (GLCM)

Extraídas sobre la matriz de coocurrencia de niveles de gris (GLCM) normalizada $P(i, j)$ calculada en la ROI del defecto en 4 direcciones espaciales ($\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$ a distancia $d = 1$):

* **Energía GLCM (Uniformidad / Angular Second Moment)**:

```math
\text{Energía} = \sum_{i} \sum_{j} P(i, j)^2
```

  * *Interpretación*: Mide la homogeneidad del patrón superficial. Valores altos indican textura lisa y suave (material sano). Valores bajos indican alta heterogeneidad (rugosidad severa, micro-desgarros por fricción o pérdida de recubrimiento).

* **Entropía GLCM**:

```math
\text{Entropía} = - \sum_{i} \sum_{j} P(i, j) \log_2(P(i, j) + \epsilon)
```

  * *Interpretación*: Mide el grado de desorden e impredecibilidad espacial en la reflectancia. Superficies muy dañadas o fracturadas exhiben entropías elevadas ($> 8.5\text{--}9.2$).

* **Homogeneidad GLCM (Inverse Difference Moment)**:

```math
\text{Homogeneidad} = \sum_{i} \sum_{j} \frac{P(i, j)}{1 + |i - j|}
```

  * *Interpretación*: Evalúa la cercanía de la distribución respecto a la diagonal principal de la GLCM. Disminuye ante transiciones bruscas o bordes afilados.

---

### 3. Métricas Topográficas 3D y de Profundidad GelSight

Calculadas a partir de la nube de puntos 3D organizada $(X, Y, Z)$ en metros:

* **Ajuste de Plano Local de Referencia**:
  Para aislar la curvatura intrínseca de la pieza del defecto en sí, se dilata la máscara con un kernel de $13 \times 13$ px y se resta la máscara original para formar un **anillo perimetral de piel sana circundante**:

```math
\text{Anillo} = (\text{dilate}(\text{mask}) \setminus \text{mask}) \cap \text{puntos\_válidos}
```

  Se ajusta un plano $Z = c_0 X + c_1 Y + c_2$ sobre dicho anillo por mínimos cuadrados (`np.linalg.lstsq`). Para cada punto del defecto, la profundidad relativa local en milímetros es:

```math
\text{depth}(x, y) = \max\left(Z_{\text{plano}}(x, y) - Z(x, y),\, 0\right) \times 1000.0
```

* **Profundidad Percentil 95 (`depth_p95_mm`)**:
  Percentil 95 de la profundidad relativa. Se prefiere al máximo absoluto para ser robusto frente a ruido puntual de sensor o reflejos ópticos especulares de la membrana del GelSight.
* **Profundidad Media (`depth_mean_mm`)**:
  Promedio de profundidad sobre toda la huella del defecto.
* **Profundidad Máxima (`depth_max_mm`)**:
  Profundidad puntual extrema alcanzada en la cavidad.
* **Volumen del Defecto (`depth_volume_mm3`)**:
  Volumen total estimado de la cavidad o pérdida de material:

```math
\text{Volumen} = \sum_{(x, y) \in \text{defecto}} \text{depth}(x, y) \times \Delta x \times \Delta y
```

  donde $\Delta x$ y $\Delta y$ son los espaciamientos espaciales medios entre columnas y filas de la nube de puntos en milímetros.

---

### 4. Descriptores 3D Locales de Superficie (PFH y SHOT)

Para capturar la geometría diferencial microscópica en nubes PCD exportadas (mediante el exportador nativo PCL o Open3D):

* **PFH (Point Feature Histograms - 125 bins)**:
  Para cada punto con al menos 12 vecinos dentro de un radio de búsqueda $r = 3 \times \text{espaciamiento mediano}$, se analizan todos los pares de normales de superficie $(\mathbf{n}_s, \mathbf{n}_t)$ calculando los cuatro ángulos característicos de Darboux $(\alpha, \phi, \theta, d)$. El histograma resultante de 125 bins describe con precisión invariante a traslación/rotación la curvatura y concavidad local.
* **SHOT (Signature of Histograms of OrienTations - 352 bins)**:
  Crea una cuadrícula esférica local 3D (32 divisiones espaciales: 8 azimutales, 2 de elevación y 2 radiales) y calcula histogramas de coseno del ángulo entre normales (11 bins por división $\to 32 \times 11 = 352$ bins). Permite discriminar defectos planos de fracturas agudas o arrancamiento de fibras.

---

### 5. Matriz de Decisión y Umbrales de Reparabilidad

El sistema evalúa el dictamen de reparación mediante reglas estadísticas de negocio validadas empíricamente:

| Clase de Defecto | Familia de Métrica | Parámetros Clave | Condición **NON-REPAIRABLE** | Justificación Técnica de Fallo |
| :--- | :--- | :--- | :--- | :--- |
| **Material Loss - Holes**<br>*(Pérdida de material)* | Geometría 2D | Área ($mm^2$)<br>Perímetro ($mm$) | **Área** > 20.5 mm² **O**<br>**Perímetro** > 18.6 mm | Pérdida de sección resistente que excede la capacidad de relleno o aporte sin deformación residual. |
| | Profundidad 3D (P90) | Área ($mm^2$)<br>P95 ($mm$)<br>Volumen ($mm^3$) | **Área** > 11.20 mm² **O**<br>**P95** > 0.178 mm (178 µm) **O**<br>**Volumen** > 1.297 mm³ | Cavidad profunda que concentra tensiones triaxiales y promueve inicio de grietas bajo fatiga. |
| **Crack**<br>*(Grietas y Fisuras)* | Geometría 2D y Textura | Área ($mm^2$)<br>Perímetro ($mm$)<br>Excentricidad ($e$)<br>Entropía GLCM | **Área** > 6.67 mm² **O**<br>**Perímetro** > 14.13 mm **O**<br>**e** < 0.98 *(ramificada)* **O**<br>**Entropía** > 9.28 | Fisuras con extensión excesiva, ramificadas (baja excentricidad) o con bordes de rotura caóticos (alta entropía). |
| | Profundidad 3D (P90) | Longitud ($mm$)<br>P95 ($mm$)<br>Volumen ($mm^3$) | **Longitud** > 5.73 mm **O**<br>**P95** > 0.069 mm (69 µm) **O**<br>**Volumen** > 0.121 mm³ | Fisura profunda que penetra capas estructurales internas no rectificables mediante pulido superficial. |
| **Scratch - Abrasion**<br>*(Desgaste por Abrasión)* | Textura 2D | Energía GLCM<br>Entropía GLCM | **Energía** < 0.12 **O**<br>**Entropía** > 8.46 | Desgaste superficial agresivo con rotura completa del patrón de acabado base. |
| | Profundidad 3D (P90) | Área ($mm^2$)<br>P95 ($mm$)<br>Volumen ($mm^3$) | **Área** > 45.93 mm² **O**<br>**P95** > 0.071 mm (71 µm) **O**<br>**Volumen** > 1.214 mm³ | Zona erosionada profunda que requeriría un decapado mayor a la tolerancia geométrica admisible. |
| **Scratch - Friction**<br>*(Fricción / Arrancamiento)* | Textura 2D | Energía GLCM<br>Entropía GLCM | **Energía** < 0.06 **O**<br>**Entropía** > 8.75 | Adherencia o microsoldadura en frío con deformación plástica severa de la capa exterior. |
| | Profundidad 3D (P90) | Área ($mm^2$)<br>P95 ($mm$)<br>Volumen ($mm^3$) | **Área** > 47.93 mm² **O**<br>**P95** > 0.085 mm (85 µm) **O**<br>**Volumen** > 2.024 mm³ | Surcos de frotamiento que exceden el umbral admisible de reacondicionamiento mecánico. |

---

## Clasificación Supervisada de Reparabilidad con Machine Learning

Además de las reglas heurísticas, el sistema incorpora un pipeline de clasificación supervisada basado en Random Forest / Gradient Boosting:

1. **Exportación de Ejemplares**:
   En la interfaz gráfica, al pulsar **`Export PCD + PFH/SHOT`**, se genera en `exports/pointclouds/<captura>/`:
   * Archivo `.pcd` con la nube 3D del defecto.
   * Archivo `.csv` con los 125 valores PFH y 352 valores SHOT por punto.
   * `manifest.json` y `depth_segmentation_overlay.png`.
2. **Generación de Plantilla de Etiquetas**:
   ```bash
   docker compose run --rm viewer python3 train_repairability.py --init-labels
   ```
   Crea `exports/labels.csv` con columnas `capture`, `file_stem`, `class_name`, `piece` y `repairable` ($1$ o $0$).
3. **Entrenamiento y Validación Cruzada**:
   ```bash
   docker compose run --rm viewer python3 train_repairability.py --use both
   ```
   Compara tres conjuntos de características (`geometry`, `descriptors` y `both`) utilizando validación cruzada por grupos (`GroupKFold` sobre `piece`), evitando el sobreajuste entre inspecciones de una misma pieza.
4. **Inferencia en Tiempo Real**:
   El modelo entrenado se guarda en `exports/repairability.joblib`. Al detectarse un defecto, la interfaz muestra automáticamente en la barra de estado:
   `ML: Holes: REPAIRABLE (82%)` o `ML: Crack: NON-REPAIRABLE (91%)`.

---

## Despliegue con Docker y GPU

La imagen está construida sobre **Ubuntu 22.04 + ROS 2 Humble** con aceleración CUDA para GPUs NVIDIA:

### 1. Construir la Imagen
```bash
cd /home/aurova/Desktop/paula/demo_lfs/viewer_project
docker compose build
```

### 2. Ejecutar la Interfaz Gráfica
Permitir acceso al servidor X11 y arrancar el contenedor:
```bash
xhost +local:root
docker compose up viewer
```

### 3. Publicar el Sensor GelSight Mini en ROS 2 (3D + 2D)
En otra terminal, arrancar el nodo de adquisición y cálculo de profundidad:
```bash
cd /home/aurova/Desktop/paula/demo_lfs/viewer_project
docker compose run --rm viewer ros2 run vb_tactile_sensors gs_image_depth
```

* El nodo publicará la imagen en `/gs_mini_img` y la nube de puntos 3D en `/gs_mini_pcd`.
* En la interfaz, ve a la pestaña **Online**, selecciona **ROS 2 Topic** y suscríbete a `/gs_mini_img`.

### 4. Visualización en RViz2 (Opcional)
Para inspeccionar la nube de puntos 3D de forma interactiva en RViz2:
```bash
docker compose up --build rviz
```

---

## Verificación de Hardware y Diagnóstico

Para verificar que PyTorch detecta la GPU dentro del contenedor:
```bash
docker compose run --rm viewer python3 -c "import torch; print('CUDA disponible:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Salida esperada con GPU NVIDIA:
```text
CUDA disponible: True | GPU: NVIDIA GeForce RTX 3060
```
*(Si no hay GPU NVIDIA, el sistema conmuta automáticamente a CPU sin fallos).*
