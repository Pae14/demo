# Tactile Defect Inspection System: Advanced Metrology & AI Analysis 

Este proyecto presenta una solución integral de **inspección visual-táctil** diseñada para la detección, segmentación y clasificación de defectos superficiales en entornos industriales. Utilizando el sensor biomimético **GelSight Mini**, la aplicación trasciende la visión convencional al capturar la topografía microscópica de las superficies, permitiendo un análisis de metrología dimensional y textural de alta precisión.

La herramienta integra segmentación con **YOLO12, YOLO26, RF-DETR y DINOv3**, metrología sobre la profundidad del GelSight, descriptores PFH/SHOT y clasificación supervisada de reparabilidad. Las reglas geométricas y texturales existentes se mantienen como análisis heurístico independiente.

---

## Arquitectura del Sistema (File Structure)

Los scripts están agrupados por función. Los archivos de ejecución de la interfaz permanecen en la raíz de `viewer_project` para conservar los comandos existentes de Docker; las herramientas auxiliares no se ejecutan al iniciar la aplicación.

| Grupo | Archivos | Función |
| --- | --- | --- |
| Aplicación | `main.py`, `workers.py`, `ui_components.py`, `config.py` | Interfaz PyQt6, flujos de inferencia/ROS 2, componentes visuales y configuración. |
| Visión | `utils_vision.py` | Métricas de máscara, textura GLCM y reglas heurísticas existentes. |
| Nube y clasificación | `pointcloud_export.py`, `repairability.py` | Exportación PCD/PFH/SHOT, vista de normales, preparación de características y predicción supervisada. |
| Herramientas manuales | `train_repairability.py`, `export_metrics.py` | Entrenamiento del clasificador y generación offline de informes CSV, respectivamente. Se ejecutan solo cuando se solicitan. |
| Exportador PCL nativo | `pcl_descriptor_export/main.cpp` | Calcula PFH/SHOT por punto; se compila dentro de la imagen Docker. |
| Pruebas | `tests/` | Pruebas de sincronización ROS/nube, exportación PCL y características del clasificador. |

No se han eliminado los publicadores ni utilidades opcionales del paquete ROS `vb_tactile_sensors`: sus nodos están registrados en `setup.py`, aunque no todos sean parte del flujo principal del visor GelSight.

---

## Concurrencia y Rendimiento (Multi-threading)

Para garantizar un rendimiento de grado industrial y una experiencia de usuario fluida, la aplicación implementa una arquitectura **multihilo asíncrona**:

- **UI Thread**: Renderizado de GUI y animaciones sin bloqueos.
- **Acquisition Thread (`ROS2Thread`)**: Nodo de ROS 2 dedicado a la escucha y conversión de imágenes mediante `cv_bridge`.
- **Inference Thread**: Ejecución paralela de modelos de IA, permitiendo el análisis de un frame mientras se recibe el siguiente.
- **Async Loaders**: Carga de modelos pesados en segundo plano durante el inicio de la app.

---

## Integración con ROS 2 (Robot Operating System)

La aplicación actúa como un **nodo nativo de ROS 2**, ideal para entornos con brazos robóticos (MoveIt) y simuladores (Gazebo).

- **Suscripción Dinámica**: Configuración del topic en tiempo real desde la interfaz gráfica.
- **Gestión de QoS (Best Effort)**: Optimizado para telemetría robótica y drones, garantizando compatibilidad y baja latencia.
- **Robustez**: Soporte nativo para frecuencias de imagen bajas (hasta **1 Hz**) sin afectar la estabilidad de la UI.

---

## Lógica de Reparabilidad (Umbrales Estadísticos)

El sistema evalúa automáticamente si un defecto es **NON-REPAIRABLE** basándose en métricas dimensionales y texturales:

- **Hole**: Área > 20.5 mm² y Perímetro > 18.6 mm.
- **Crack**: Área > 6.67 mm², Perímetro > 14.13 mm, Excentricidad < 0.98 o Entropía > 9.28.
- **Abrasion**: Energía < 0.12 o Entropía > 8.46.
- **Friction**: Energía < 0.06 o Entropía > 8.75.

Estas reglas son heurísticas de la vista online y no sustituyen las etiquetas de resultado real usadas para entrenar y evaluar el clasificador supervisado.

La tarjeta de métricas de la interfaz también destaca indicadores específicos por defecto: área, P95/máxima profundidad y volumen para agujeros, longitud/anchura/área/P95 para grietas, y área/profundidad/volumen para abrasión y fricción. La longitud y anchura de una grieta son extensiones geométricas estimadas desde su máscara. Estos valores 3D son descriptivos: no se aplican como umbral de reparabilidad hasta validarlos con muestras etiquetadas por resultado de reparación.

---

## Despliegue con Docker

La imagen está basada en ROS 2 Humble sobre Ubuntu 22.04 y Python 3.10. Instala PyTorch/cu121, TensorFlow, PyQt6, OpenCV, Ultralytics, RF-DETR, Open3D, la biblioteca nativa PCL y las dependencias ROS (`cv_bridge`, `sensor_msgs`). PyTorch usa CPU si no se expone una GPU NVIDIA. Los pesos y datos se montan desde el host y no se copian en la imagen.

### Preparar el host

- Linux con Docker Engine y Docker Compose v2.
- Para la GUI, una sesión gráfica X11 activa. Compose monta el socket y el archivo de autorización `XAUTHORITY` en solo lectura.
- Para el sensor, acceso a `/dev/bus/usb`; la cámara local se pasa como dispositivo `/dev/video*`.
- Para GPU NVIDIA, driver compatible y NVIDIA Container Toolkit instalado y configurado con Docker.

Desde la raíz `demo/`:

```bash
cd viewer_project
mkdir -p data/test2/images data/test2/labels exports
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
docker compose build
docker compose up
```

`docker compose up` abre únicamente la interfaz. RViz2 es opcional y se inicia aparte cuando se solicita.

El contenedor usa la cookie X11 del usuario, montada en solo lectura; no es necesario ejecutar `xhost`.

Si la cámara no es `/dev/video0`, indica el dispositivo antes de arrancar:

```bash
CAMERA_DEVICE=/dev/video2 docker compose up
```

Para habilitar CUDA, usa el override opcional (no lo uses en equipos sin NVIDIA):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Comprueba la disponibilidad de GPU dentro del contenedor con:

```bash
docker compose run --rm viewer python3 -c "import torch; print(torch.cuda.is_available())"
```

### Pesos, datos y exportaciones

Coloca los archivos en estas rutas del host; `config.py` espera exactamente estas rutas:

| Uso | Ruta |
| --- | --- |
| Segmentación YOLO12 | `models/yolo12/best.pt` |
| Segmentación YOLO26 | `models/yolo26/best.pt` |
| RF-DETR | `models/checkpoint_best_ema.pt` |
| Clasificación ResNet | `models/checkpoint_epoch_49.pth.tar` |
| Clasificación CNN | `models/best_model.weights.h5` |
| Imágenes offline | `data/test2/images/` |
| Etiquetas Ground Truth YOLO | `data/test2/labels/` |
| Capturas e informes | `exports/` |

Los modelos se montan de solo lectura en `/app/models` y el dataset en `/app/data`; las salidas van a `exports/`. El `.dockerignore` excluye pesos, datos y artefactos de compilación del contexto de build.

### GUI y nodos ROS 2

El servicio predeterminado ejecuta la interfaz. Para publicar imagen y profundidad/nube del GelSight Mini, en otra terminal ejecuta:

```bash
cd viewer_project
docker compose run --rm viewer ros2 run vb_tactile_sensors gs_image_depth
```

La captura del GelSight usa FFmpeg para decodificar MJPEG y reducir el frame a 320×240 antes de entregarlo a ROS; si FFmpeg no está instalado, usa OpenCV como alternativa. Un hilo de captura continuo lee la cámara y el nodo publica cada frame nuevo en `/gs_mini_img`, sin bloquear el timer de ROS esperando a la cámara. Un segundo worker limita la reconstrucción de profundidad a 5 Hz y publica el `PointCloud2` organizado por píxel en `/gs_mini_pcd` con el timestamp y frame de la imagen procesada. En esta máquina el driver anuncia 25 FPS y la prueba directa de FFmpeg obtuvo aproximadamente 19 FPS; la frecuencia final depende del sensor y carga de CPU. En la pestaña Online selecciona `ROS 2 Topic` y usa `/gs_mini_img`. Los publicadores de imagen alternativos se ejecutan con `ros2 run vb_tactile_sensors gs_image` o `ros2 run vb_tactile_sensors digit_image`. `digit_image_depth` requiere el repositorio externo `digit-depth` y sus pesos, que no se incluyen en esta imagen.

La máscara segmentada solo se superpone y se exporta con la nube cuyo timestamp sea exactamente igual al de la imagen y cuyo frame y resolución sean compatibles. La interfaz mantiene en caché mensajes recientes para emparejar aunque la nube llegue después de la imagen; si no existe pareja exacta, muestra la segmentación sin profundidad y no reutiliza una nube atrasada. La imagen RGB se muestra a la frecuencia recibida; el control `Inference interval` limita la frecuencia de segmentación para no saturar la CPU/GPU, y las parejas imagen/nube sincronizadas más recientes tienen prioridad. La interfaz calcula profundidad media, P95, máxima y volumen estimado respecto a un plano local ajustado con los píxeles vecinos. `MAX_REPAIR_DEPTH_MM` en `config.py` permite añadir el P95 a la regla de reparabilidad; permanece en `None` hasta calibrar un límite válido para el material y proceso de reparación.

En modo Online, el selector `Output view` alterna entre la máscara de segmentación, la proyección coloreada de la altura Z y la combinación de profundidad con la máscara segmentada. RViz conserva la vista 3D interactiva en una ventana separada.

El botón `Export PCD + PFH/SHOT` guarda una captura manual del siguiente frame con imagen y nube sincronizadas. Para cada región segmentada con al menos 12 puntos 3D válidos escribe un archivo PCD, un CSV con 125 valores PFH y 352 valores SHOT por punto, un manifiesto JSON y `depth_segmentation_overlay.png`. Los archivos se guardan en `exports/pointclouds/<captura>/`; el manifiesto indica las regiones omitidas si no tienen suficientes puntos. El radio de vecindad se estima como tres veces el espaciamiento mediano entre píxeles vecinos. La interfaz muestra la ruta de la exportación en la barra de estado.

#### Entrenar la clasificación de reparabilidad

El clasificador supervisado estima si una instancia es reparable (`1`) o no reparable (`0`) a partir de las características geométricas, PFH y SHOT. No genera una etiqueta fiable automáticamente: primero hay que etiquetar ejemplos usando el resultado real de la reparación o un criterio de inspección acordado. No uses como etiquetas las predicciones de las reglas geométricas si el objetivo es comprobar si PFH/SHOT mejoran esas reglas.

1. Captura y exporta ejemplos representativos reparables y no reparables. Para una primera prueba se puede empezar con una sola clase de defecto, pero se necesitan las dos etiquetas de reparabilidad.
2. Crea o actualiza la plantilla CSV:

   ```bash
   docker compose run --rm viewer python3 train_repairability.py --init-labels
   ```

   Esto crea `exports/labels.csv`, con una fila por instancia exportada. Las columnas son `capture`, `file_stem`, `class_name`, `piece` y `repairable`. Al volver a generar la plantilla se conservan las etiquetas existentes que coincidan en `capture` y `file_stem`.
3. Edita `exports/labels.csv`: escribe `1` o `0` en `repairable`. Usa el mismo valor de `piece` para todas las capturas de una misma pieza física; eso permite agruparlas al evaluar el modelo y reduce la fuga entre entrenamiento y validación. Las filas sin etiqueta y las instancias de clases que aún no quieras usar pueden dejarse sin etiquetar.
4. Entrena y compara los conjuntos de características:

   ```bash
   docker compose run --rm viewer python3 train_repairability.py
   ```

   El script compara `geometry`, `descriptors` y `both` con validación cruzada agrupada por `piece`, e imprime la matriz de confusión y métricas, incluido el F1 de la clase no reparable. Requiere al menos 10 instancias etiquetadas, ambas etiquetas y al menos dos grupos `piece`. Para elegir manualmente el conjunto usado al guardar:

   ```bash
   docker compose run --rm viewer python3 train_repairability.py --use both
   ```

   El modelo queda en `exports/repairability.joblib`. Las métricas con conjuntos pequeños son inestables y no demuestran que una variante sea mejor; valida con más piezas que no se usaron para entrenar.
5. Reconstruye la imagen después de cambios de código y reinicia la interfaz:

   ```bash
   docker compose build viewer
   docker compose up viewer
   ```

   En la interfaz, pulsa `Export PCD + PFH/SHOT`. Si encuentra el modelo entrenado, la barra de estado añade la predicción y la probabilidad, por ejemplo `ML: Holes: REPAIRABLE (71%)`. La misma predicción se añade a cada instancia del `manifest.json` como `repairable` y `repairable_probability`. La clasificación se calcula al exportar, no en cada frame; sin `exports/repairability.joblib`, el PCD y los descriptores se exportan como siempre, pero no se añade predicción.

RViz usa `rviz/gs_pointcloud.rviz`, con el frame `gelsightmini_camera_frame` y el display PointCloud2 conectado a `/gs_mini_pcd`. Para abrir RViz aparte, ejecuta `docker compose up --build rviz`; no necesita que el sensor esté conectado para abrir la ventana.

Para abrir una shell ROS en el mismo entorno:

```bash
docker compose run --rm viewer bash
ros2 topic list
```

El contenedor usa `network_mode: host` para DDS. Mantén el mismo `ROS_DOMAIN_ID` que los demás nodos. Para usar otro dominio, por ejemplo:

```bash
ROS_DOMAIN_ID=7 docker compose up
```

### Estado de integración

- La inferencia seleccionable cubre YOLO12, YOLO26, RF-DETR y DINOv3. Las métricas de máscara incluyen área, perímetro, excentricidad, compacidad y GLCM; con ROS2 también se presentan métricas de profundidad y volumen calculadas sobre la zona segmentada.
- El worker ROS2 recibe `sensor_msgs/Image` y `/gs_mini_pcd`; solo usa la nube si está sincronizada y alineada. RViz2 muestra la nube en una ventana separada.
- DINOv3 usa el checkpoint `models/best.pt`, que incluye el backbone y la cabeza de segmentación.
- La construcción de la imagen instala dependencias e incluye el paquete ROS, pero no puede probar la cámara, el acceso gráfico, CUDA ni la compatibilidad de pesos sin el hardware y los checkpoints correspondientes.

> **Nota académica**: los umbrales de reparabilidad son heurísticos y requieren validación para cada tejido, calibración y protocolo de inspección.
