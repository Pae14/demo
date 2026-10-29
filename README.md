# Tactile Defect Inspection System: Advanced Metrology & AI Analysis 

Este proyecto presenta una solución integral de **inspección visual-táctil** diseñada para la detección, segmentación y clasificación de defectos superficiales en entornos industriales. Utilizando el sensor biomimético **GelSight Mini**, la aplicación captura la topografía de las superficies permitiendo un análisis de metrología dimensional y textural de alta precisión.

La herramienta integra algoritmos de **Deep Learning** (YOLOv12, YOLOv26, RF-DETR, ResNet18 y CNN) con técnicas de **Visión Artificial clásica** (GLCM de Haralick) y reglas de negocio basadas en estudios estadísticos para determinar de forma autónoma la integridad de los componentes inspeccionados.

---

## Arquitectura del Sistema (File Structure)

El software sigue un diseño modular y desacoplado para garantizar la mantenibilidad y el rendimiento en tiempo real:

| Archivo / Carpeta | Función |
| --- | --- |
| **`main.py`** | **Orquestador Central**. Ciclo de vida de la aplicación, navegación entre pestañas (Offline/Online) y lógica general de la GUI en PyQt6. |
| **`workers.py`** | **Capa de Concurrencia e Integración**. Hilos de adquisición (`GSThread` para cámara local, `ROS2Thread` para topic ROS 2), cargadores asíncronos y ejecución de inferencia en GPU/CPU (`DEVICE`). |
| **`utils_vision.py`** | **Motor Analítico**. Métricas dimensionales (área, perímetro, excentricidad), descriptores texturales GLCM y reglas heurísticas de reparabilidad. |
| **`ui_components.py`** | **Framework de UI**. Estética *Glassmorphism*, paneles de visualización, botones animados y tarjetas de métricas. |
| **`config.py`** | **Fuente de Verdad**. Rutas de modelos, calibración GelSight (mm/px), umbrales de confianza, cámara e intervalo de inferencia. |
| **`export_metrics.py`** | **Herramienta de Reporting**. Script para procesamiento por lotes y generación de informes CSV estadísticos. |
| **`models/`** | Directorio donde residen los pesos y checkpoints entrenados. |
| **`exports/`** | Carpeta de salida para capturas de pantalla, reportes y métricas. |
---

## Concurrencia y Rendimiento (Multi-threading)

Para garantizar un rendimiento de grado industrial y una experiencia de usuario fluida, la aplicación implementa una arquitectura **multihilo asíncrona**:

- **UI Thread**: Renderizado de GUI y animaciones sin bloqueos.
- **Acquisition Thread (`ROS2Thread`)**: Nodo de ROS 2 dedicado a la escucha y conversión de imágenes mediante `cv_bridge`.
- **Inference Thread**: Ejecución paralela de modelos de IA, permitiendo el análisis de un frame mientras se recibe el siguiente.
- **Async Loaders**: Carga de modelos pesados en segundo plano durante el inicio de la app.

---

## Modelos y Gestión de Pesos (`models/`)

La aplicación requiere los siguientes pesos preentrenados dentro de la carpeta `models/`:

| Modelo / Tarea | Archivo en el Host | Tamaño aprox. | Descripción |
| --- | --- | --- | --- |
| **YOLO12n-seg** | `models/yolo12/best.pt` | ~5.8 MB | Segmentación de instancias ligera y de alta velocidad. |
| **YOLO26n-seg** | `models/yolo26/best.pt` | ~6.3 MB | Segmentación de instancias de defectología táctil. |
| **RF-DETR** | `models/checkpoint_best_ema.pt` | ~389 MB | Detector Transformer de alta precisión para defectos complejos. |
| **ResNet18** | `models/checkpoint_epoch_49.pth.tar` | ~129 MB | Clasificador convolucional multiclase de defectos superficiales. |
| **CNN Keras** | `models/best_model.weights.h5` | ~191 MB | Modelo convolucional alternativo entrenado en TensorFlow/Keras. |

### Comprobación de integridad de los pesos (Git LFS)

Si el proyecto ha sido clonado mediante Git, verifique que los archivos pesados sean archivos binarios reales y no punteros de texto generados por Git LFS:

```bash
# Comprobar si algún archivo contiene punteros LFS sin descargar:
grep -rn "version https://git-lfs" models/
```

- Si el comando anterior **no devuelve nada**, los pesos están completos y listos para usarse.
- Si devuelve coincidencias, ejecute:
  ```bash
  git lfs install
  git lfs pull
  ```
- O verifique los tamaños con:
  ```bash
  ls -lh models/ models/yolo12/ models/yolo26/
  ```

---

## Despliegue con Docker (Recomendado)

El despliegue con Docker aísla todas las dependencias del sistema operativo, proveyendo **ROS 2 Humble**, compatibilidad con **GPU NVIDIA (CUDA 12.1)** y las librerías gráficas de PyQt6.

### 1. Requisitos Previos en el Host

- **Docker Engine** y **Docker Compose v2** instalados.
- Para aceleración por hardware: **NVIDIA Driver** y **NVIDIA Container Toolkit** instalados.
- Sesión gráfica **X11** activa para desplegar la ventana de la aplicación.
- Habilitar permisos de visualización X11:
  ```bash
  xhost +si:localuser:root
  ```

### 2. Compilación de la Imagen

Desde la raíz del proyecto (`/home/aurova/Desktop/paula/viewer_project`):

```bash
docker compose build
```

### 3. Verificación de la GPU (NVIDIA CUDA)

Para comprobar que el contenedor detecta la tarjeta gráfica NVIDIA y la versión de PyTorch con CUDA 12.1:

```bash
docker compose run --rm viewer python3 -c "import torch; print('CUDA disponible:', torch.cuda.is_available(), '| Dispositivo:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
*(Deberá mostrar `CUDA disponible: True | Dispositivo: NVIDIA GeForce RTX 3060`)*

### 4. Arranque de la Interfaz Gráfica

```bash
xhost +si:localuser:root
docker compose up --remove-orphans
```

> **Nota sobre dispositivos de cámara**: Si utiliza una cámara local USB física que esté en un puerto diferente de `/dev/video0`, puede indicarlo al arrancar:
> ```bash
> CAMERA_DEVICE=/dev/video2 docker compose up --remove-orphans
> ```

> **Nota para ordenadores sin tarjeta NVIDIA (Modo solo CPU)**:
> Si se ejecuta en un equipo que no dispone de GPU NVIDIA, comenta las líneas `deploy:` en `docker-compose.yml` para que Docker no solicite el driver de NVIDIA. La aplicación e inferencia funcionarán en CPU de forma automática.

---

## Conexión mediante Topic de ROS 2

El contenedor incluye el paquete **`vb_tactile_sensors`** precompilado en `/ros2_ws` y está configurado con `network_mode: host`, lo que permite que el middleware DDS de ROS 2 comparta el mismo espacio de red que el host y otros contenedores.

### 1. Iniciar el Publicador del Sensor (con Docker)

Desde una terminal en `/home/aurova/Desktop/paula/viewer_project`:

```bash
docker compose run --rm viewer ros2 run vb_tactile_sensors gs_image_depth
```

> **Nodos ejecutables disponibles en `vb_tactile_sensors`**:
> - `gs_image_depth`: Publica imagen (`/gs_mini_img`), imagen de profundidad (`/gs_depth_image`) y nube 3D (`/gs_mini_pcd`) del GelSight Mini con aceleración GPU (CUDA).
> - `gs_image`: Publica únicamente la imagen del GelSight Mini en `/gs_mini_img`.
> - `digit_image`: Publica la imagen del sensor DIGIT en `/digit_img`.
> - `digit_image_depth`: Publica imagen y mapa de profundidad para el sensor DIGIT.

### 2. Iniciar el Visor (GUI en otra terminal)

En una segunda terminal, arranca la interfaz gráfica:

```bash
cd /home/aurova/Desktop/paula/viewer_project
xhost +si:localuser:root
docker compose up --remove-orphans
```

### 3. Conectar la Interfaz al Topic

1. En la aplicación, navega a la pestaña **Online**.
2. En el menú desplegable **Source Selection**, selecciona **ROS 2 Topic**.
3. El campo **Topic Name** vendrá por defecto con `/gs_mini_img` (puedes cambiarlo si usas `/digit_img` o cualquier otro topic).
4. Pulsa **▶ Start**.
5. El indicador pasará a **⬤ Streaming** (verde) al recibir los frames y procesará la inferencia en tiempo real en la GPU NVIDIA.

### 4. Diagnóstico de Topics

Para inspeccionar o depurar los topics activos desde el entorno ROS 2:

```bash
# Listar los ejecutables del paquete sensor
docker compose run --rm viewer ros2 pkg executables vb_tactile_sensors

# Listar topics activos
docker compose run --rm viewer ros2 topic list

# Comprobar frecuencia de publicación (Hz)
docker compose run --rm viewer ros2 topic hz /gs_mini_img
```

---

## Ejecución Nativa en el Host (sin Docker)

Si dispone de un entorno Ubuntu con **ROS 2 Humble** y drivers de NVIDIA instalados directamente en el sistema:

1. **Crear y activar entorno virtual**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Instalar PyTorch con soporte CUDA**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Instalar dependencias del proyecto**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Instalar dependencias de ROS 2 para Python**:
   ```bash
   sudo apt update
   sudo apt install ros-humble-cv-bridge ros-humble-sensor-msgs ros-humble-std-msgs
   ```

5. **Compilar el paquete sensor en el host (opcional)**:
   ```bash
   colcon build --packages-select vb_tactile_sensors --symlink-install
   source install/setup.bash
   ```

6. **Iniciar la aplicación**:
   ```bash
   source /opt/ros/humble/setup.bash
   python3 main.py
   ```

---

## 🛠️ Métricas de Metrología y Lógica de Reparabilidad

El sistema incorpora un motor analítico multidimensional que evalúa el defecto superficial combinando **geometría 2D**, **textura microscópica (GLCM de Haralick)** y **topografía de profundidad 3D** (reconstruida mediante sensor táctil fotométrico GelSight Mini). 

A partir de estas métricas, el sistema clasifica de forma autónoma cada defecto como **REPAIRABLE** (reparable mediante rectificado, pulido o aporte) o **NON-REPAIRABLE** (daño estructural crítico o que supera las tolerancias admisibles).

---

### 1. Calibración del Sensor Táctil GelSight Mini

La conversión entre el espacio de imagen (píxeles) y las dimensiones métricas reales ($mm$) se basa en el campo de visión calibrado (*Field of View*) de la membrana elastomérica:

- **Ancho del sensor ($W_{mm}$)**: $18.6\text{ mm}$
- **Alto del sensor ($H_{mm}$)**: $14.3\text{ mm}$
- **Área por píxel**: $A_{px} = \frac{18.6 \times 14.3}{\text{ancho}_{px} \times \text{alto}_{px}}\text{ mm}^2/\text{px}$
- **Resolución espacial**: $\Delta l = \sqrt{A_{px}}\text{ mm/px}$

---

### 2. Métricas Texturales (GLCM de Haralick)

Sobre la región de interés (ROI) en escala de grises correspondiente a la máscara binaria del defecto, se calcula la **Matriz de Co-ocurrencia de Niveles de Gris (GLCM)** para una distancia inter-píxel de $d=1$ promediando en 4 orientaciones espaciales ($\theta = \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$):

$$\text{GLCM}(i, j) = \sum_{x} \sum_{y} \begin{cases} 1 & \text{si } I(x, y) = i \text{ y } I(x+\Delta x, y+\Delta y) = j \\ 0 & \text{en otro caso} \end{cases}$$

A partir de la matriz normalizada de probabilidades $P(i, j)$, se extraen los siguientes descriptores texturales:

* **Energía GLCM (Uniformidad / Segundo Momento Angular - ASM)**:
  $$\text{Energía} = \sum_{i} \sum_{j} P(i, j)^2$$
  * *Interpretación física*: Mide la homogeneidad y regularidad de las micro-texturas. Superficies pulidas o no dañadas exhiben valores altos de energía. Cuando existen surcos de fricción severa o desgarros por abrasión, la dispersión de intensidades es caótica, provocando una **caída drástica** de la energía ($\le 0.04 - 0.12$).
* **Entropía GLCM**:
  $$\text{Entropía} = -\sum_{i} \sum_{j} P(i, j) \log_2(P(i, j) + \epsilon)$$
  * *Interpretación física*: Mide la aleatoriedad, desorden y complejidad de la textura superficial. Un defecto con microrugosidades agudas, fracturas o desprendimiento granular produce un incremento notable de la entropía ($> 8.46 - 9.28$).
* **Homogeneidad GLCM (Momento de Diferencia Inversa - IDM)**:
  $$\text{Homogeneidad} = \sum_{i} \sum_{j} \frac{P(i, j)}{1 + (i - j)^2}$$
  * *Interpretación física*: Evalúa la suavidad de las transiciones de nivel de gris locales. Disminuye significativamente en transiciones bruscas o bordes afilados dentro del defecto.

---

### 3. Métricas Geométricas y Dimensionales (2D)

Calculadas sobre las regiones conexas de la segmentación (`skimage.measure.regionprops`):

* **Área ($mm^2$)**: Superficie total proyectada de la huella del defecto.
* **Perímetro ($mm$)**: Longitud del contorno periférico exterior del defecto.
* **Excentricidad ($e \in [0, 1]$)**:
  $$e = \sqrt{1 - \left(\frac{b}{a}\right)^2}$$
  Relación de aspecto elíptica ($a$: semieje mayor, $b$: semieje menor).
  * $e \to 0$: Geometría circular o regular (típica de picaduras o pérdida de material puntual - *Holes*).
  * $e \to 1$: Geometría marcadamente alargada o lineal (típica de grietas - *Cracks* con $e \ge 0.98$). Una grieta con excentricidad reducida ($e < 0.98$) indica **ramificación multidireccional crítica**.
* **Compacidad**: $C = \frac{4\pi \cdot \text{Área}}{\text{Perímetro}^2} \in (0, 1]$.
* **Longitud del Eje Mayor ($mm$) y Eje Menor ($mm$)**: Extensión longitudinal y transversal máxima de la elipse equivalente.

---

### 4. Métricas Topográficas y de Profundidad 3D (PointCloud)

A partir de la nube de puntos organizada `sensor_msgs/PointCloud2` obtenida mediante la reconstrucción fotométrica 3D del sensor GelSight Mini:

1. **Ajuste del Plano de Referencia Local**:
   Para aislar la curvatura de la pieza inspeccionada, se calcula un anillo dilatado de 13 píxeles alrededor de la máscara y se excluyen los puntos del defecto. Sobre los puntos sanos circundantes se ajusta un plano por mínimos cuadrados:
   $$z_{\text{plano}}(x, y) = c_0 x + c_1 y + c_2$$
2. **Cálculo de Altura Relativa y Profundidad ($mm$)**:
   $$\text{Profundidad}(x, y) = \max\left(z_{\text{plano}}(x, y) - z(x, y),\, 0\right) \times 1000\text{ mm}$$

Descriptores 3D extraídos por defecto:

* **Profundidad Media (`depth_mean_mm`)**: Promedio de penetración de la hendidura o cavidad.
* **Profundidad Percentil 95 (`depth_p95_mm`)**: Profundidad en el percentil 95 de los puntos del defecto. Es el descriptor más robusto y fiable frente a ruidos o picos espurios del sensor para evaluar la penetración máxima real.
* **Profundidad Máxima (`depth_max_mm`)**: Profundidad máxima puntual alcanzada en la cavidad.
* **Volumen del Defecto (`depth_volume_mm3`)**: Volumen total estimado de pérdida de material o deformación ($mm^3$):
  $$\text{Volumen} = \sum_{(x,y) \in \text{defecto}} \text{Profundidad}(x, y) \times A_{px}$$

---

### 5. Criterios y Umbrales de Reparabilidad por Defecto

Un defecto se dictamina como **NON-REPAIRABLE** si se superan los límites de tolerancia dimensional, textural o volumétrica establecidos a partir de los estudios estadísticos del dataset de defectología industrial:

| Clase de Defecto | Tipo de Métrica | Parámetro Evaluado | Condición de No Reparabilidad (**NON-REPAIRABLE**) | Justificación Técnica |
| :--- | :--- | :--- | :--- | :--- |
| **Material Loss - Holes**<br>*(Pérdida de material / Agujeros)* | Geometría 2D | Área ($mm^2$)<br>Perímetro ($mm$) | $\text{Área} > 20.5\text{ mm}^2$ **O** $\text{Perímetro} > 18.6\text{ mm}$ | Supera el límite de material mecanizable o relleno por aporte de soldadura sin comprometer la resistencia estructural. |
| | Profundidad 3D *(P90)* | Área ($mm^2$)<br>Profundidad P95 ($mm$)<br>Volumen ($mm^3$) | $\text{Área} > 11.20\text{ mm}^2$ **O**<br>$\text{P95} > 0.178\text{ mm}$ ($178\,\mu m$) **O**<br>$\text{Volumen} > 1.297\text{ mm}^3$ | Cavidades profundas que inducen concentración de tensiones severa y riesgo de colapso por fatiga. |
| **Crack**<br>*(Grietas / Fisuras)* | Geometría 2D y Textura | Área ($mm^2$)<br>Perímetro ($mm$)<br>Excentricidad ($e$)<br>Entropía GLCM | $\text{Área} > 6.67\text{ mm}^2$ **O**<br>$\text{Perímetro} > 14.13\text{ mm}$ **O**<br>$e < 0.98$ *(ramificación)* **O**<br>$\text{Entropía} > 9.28$ | Fisuras con extensión excesiva, ramificadas (baja excentricidad) o con bordes desestructurados (alta entropía). |
| | Profundidad 3D *(P90)* | Longitud ($mm$)<br>Profundidad P95 ($mm$)<br>Volumen ($mm^3$) | $\text{Longitud} > 5.73\text{ mm}$ **O**<br>$\text{P95} > 0.069\text{ mm}$ ($69\,\mu m$) **O**<br>$\text{Volumen} > 0.121\text{ mm}^3$ | Grietas profundas que penetran la capa de tratamiento superficial e impiden el reacondicionamiento por rectificado. |
| **Scratch - Abrasion**<br>*(Rayaduras por Abrasión)* | Textura 2D | Energía GLCM<br>Entropía GLCM | $\text{Energía} < 0.12$ **O**<br>$\text{Entropía} > 8.46$ | Desgaste superficial severo con pérdida total de uniformidad microscópica en el área de contacto. |
| | Profundidad 3D *(P90)* | Área ($mm^2$)<br>Profundidad P95 ($mm$)<br>Volumen ($mm^3$) | $\text{Área} > 45.93\text{ mm}^2$ **O**<br>$\text{P95} > 0.071\text{ mm}$ ($71\,\mu m$) **O**<br>$\text{Volumen} > 1.214\text{ mm}^3$ | Abrasión extendida que requiere un desbaste más profundo que la tolerancia dimensional de la pieza. |
| **Scratch - Friction**<br>*(Rayaduras por Fricción)* | Textura 2D | Energía GLCM<br>Entropía GLCM | $\text{Energía} < 0.06$ **O**<br>$\text{Entropía} > 8.75$ | Desgarro por microsoldadura en frío o frotamiento con desorden textural irreversible. |
| | Profundidad 3D *(P90)* | Área ($mm^2$)<br>Profundidad P95 ($mm$)<br>Volumen ($mm^3$) | $\text{Área} > 47.93\text{ mm}^2$ **O**<br>$\text{P95} > 0.085\text{ mm}$ ($85\,\mu m$) **O**<br>$\text{Volumen} > 2.024\text{ mm}^3$ | Surcos profundos de fricción que superan el espesor admisible de pulido. |

> **Visualización en la Interfaz**:
> Todos estos descriptores se calculan dinámicamente en tiempo real para cada defecto detectado por las redes neuronales y se muestran desglosados en el panel lateral de **Detections** y en el reporte generado mediante `export_metrics.py`. Si alguna métrica sobrepasa los umbrales de no reparabilidad, el estado visual del visor cambia automáticamente a **NON-REPAIRABLE** (rojo) junto con la notificación del parámetro infractor.

