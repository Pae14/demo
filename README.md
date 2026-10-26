

# Tactile Defect Inspection System: Advanced Metrology & AI Analysis

Este proyecto presenta una solución integral de **inspección visual-táctil** diseñada para la detección, segmentación y clasificación de defectos superficiales en entornos industriales. Utilizando el sensor biomimético **GelSight Mini**, la aplicación trasciende la visión convencional al capturar la topografía microscópica de las superficies, permitiendo un análisis de metrología dimensional y textural de alta precisión.

La herramienta integra algoritmos de **Deep Learning** (YOLOv12, YOLOv26, RF-DETR) con técnicas de **Visión Artificial clásica** (GLCM de Haralick) y reglas basadas en estudios estadísticos para determinar de forma autónoma la integridad de los componentes inspeccionados.

---

## Arquitectura del Sistema (File Structure)

El software sigue un diseño modular y desacoplado para garantizar la mantenibilidad y el rendimiento en tiempo real:

1. **`main.py`**: **Orquestador Central**. Gestiona el ciclo de vida de la aplicación, la navegación entre pestañas (Offline/Online) y la interfaz principal PyQt6.
2. **`workers.py`**: **Capa de Concurrencia e Integración**. Implementa el procesamiento multihilo (`QThread`) y el nodo de **ROS 2** para la suscripción a sensores externos.
3. **`utils_vision.py`**: **Motor Analítico**. Contiene los algoritmos de metrología, los descriptores de textura GLCM y la lógica de decisión de reparabilidad.
4. **`ui_components.py`**: **Framework de UI**. Define la estética _Glassmorphism_, los widgets personalizados y el sistema de animaciones.
5. **`config.py`**: **Fuente de Verdad**. Centraliza constantes de calibración (mm/px), rutas globales de modelos y parámetros de red.
6. **`export_metrics.py`**: **Herramienta de Reporting**. Script para el procesamiento por lotes y generación de informes CSV estadísticos.

---

## Concurrencia y Rendimiento (Multi-threading)

Para garantizar un rendimiento de grado industrial y una experiencia de usuario fluida, la aplicación implementa una arquitectura **multihilo asíncrona**:

- **UI Thread**: Renderizado de GUI y animaciones sin bloqueos.
- **Acquisition Thread (****`ROS2Thread`****)**: Nodo de ROS 2 dedicado a la escucha y conversión de imágenes mediante `cv_bridge`.
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

- **Hole**: Área \> 20.5 mm² o Perímetro \> 18.6 mm.
- **Crack**: Área \> 6.67 mm², Perímetro \> 14.13 mm, Excentricidad \< 0.98 o Entropía \> 9.28.
- **Abrasion**: Energía \< 0.12 o Entropía \> 8.46.
- **Friction**: Energía \< 0.06 o Entropía \> 8.75.

---

## Despliegue con Docker

La imagen está basada en ROS 2 Humble sobre Ubuntu 22.04 y Python 3.10. Instala PyTorch/cu121, TensorFlow, PyQt6, OpenCV, Ultralytics, RF-DETR, Open3D y las dependencias ROS (`cv_bridge`, `sensor_msgs`). PyTorch usa CPU si no se expone una GPU NVIDIA. Los pesos y datos se montan desde el host y no se copian en la imagen.

### Preparar el host

- Linux con Docker Engine y Docker Compose v2.
- Para la GUI, una sesión gráfica X11 activa. El compose monta el socket X11 y la pantalla indicada por `DISPLAY`.
- Para el sensor, acceso a `/dev/bus/usb`; la cámara local se pasa como dispositivo `/dev/video*`.
- Para GPU NVIDIA, driver compatible y NVIDIA Container Toolkit instalado y configurado con Docker.

Desde la raíz `demo/`:

```
cd viewer_project
mkdir -p data/test2/images data/test2/labels exports
xhost +si:localuser:root
docker compose build
docker compose up
```

La interfaz y RViz2 se abren en el escritorio local. Sin sensor, RViz muestra la cuadrícula y deja el display `/gs_mini_pcd` esperando datos. Al terminar, revoca el permiso X11:

```
xhost -si:localuser:root
```

Si la cámara no es `/dev/video0`, indica el dispositivo antes de arrancar:

```
CAMERA_DEVICE=/dev/video2 docker compose up
```

Para habilitar CUDA, usa el override opcional (no lo uses en equipos sin NVIDIA):

```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Comprueba la disponibilidad de GPU dentro del contenedor con:

```
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

```
cd viewer_project
docker compose run --rm viewer ros2 run vb_tactile_sensors gs_image_depth
```

El nodo publica la imagen en `/gs_mini_img` y `PointCloud2` organizado por píxel en `/gs_mini_pcd`, ambos con el mismo timestamp y frame. En la pestaña Online selecciona `ROS 2 Topic` y usa `/gs_mini_img`. Los publicadores de imagen alternativos se ejecutan con `ros2 run vb_tactile_sensors gs_image` o `ros2 run vb_tactile_sensors digit_image`. `digit_image_depth` requiere el repositorio externo `digit-depth` y sus pesos, que no se incluyen en esta imagen.

Cuando imagen y nube coinciden en timestamp, frame y resolución, la máscara segmentada se aplica a los puntos correspondientes. La interfaz calcula profundidad media, P95, máxima y volumen estimado respecto a un plano local ajustado con los píxeles vecinos. `MAX_REPAIR_DEPTH_MM` en `config.py` permite añadir el P95 a la regla de reparabilidad; permanece en `None` hasta calibrar un límite válido para el material y proceso de reparación.

En modo Online, el selector `Output view` alterna entre la máscara de segmentación y una proyección coloreada de la altura Z del cloud sincronizado. RViz conserva la vista 3D interactiva en una ventana separada.

RViz usa `rviz/gs_pointcloud.rviz`, con el frame `gelsightmini_camera_frame` y el display PointCloud2 conectado a `/gs_mini_pcd`. Para abrir únicamente RViz, ejecuta `docker compose up --build rviz`; no necesita que el sensor esté conectado para abrir la ventana.

Para abrir una shell ROS en el mismo entorno:

```
docker compose run --rm viewer bash
ros2 topic list
```

El contenedor usa `network_mode: host` para DDS. Mantén el mismo `ROS_DOMAIN_ID` que los demás nodos. Para usar otro dominio, por ejemplo:

```
ROS_DOMAIN_ID=7 docker compose up
```

### Estado de integración

- La inferencia seleccionable cubre YOLO12, YOLO26, RF-DETR y DINOv3. Las métricas de máscara incluyen área, perímetro, excentricidad, compacidad y GLCM; con ROS2 también se presentan métricas de profundidad y volumen calculadas sobre la zona segmentada.
- El worker ROS2 recibe `sensor_msgs/Image` y `/gs_mini_pcd`; solo usa la nube si está sincronizada y alineada. RViz2 muestra la nube en una ventana separada.
- DINOv3 usa el checkpoint `models/best.pt`, que incluye el backbone y la cabeza de segmentación.
- La construcción de la imagen instala dependencias e incluye el paquete ROS, pero no puede probar la cámara, el acceso gráfico, CUDA ni la compatibilidad de pesos sin el hardware y los checkpoints correspondientes.

> **Nota académica**: los umbrales de reparabilidad son heurísticos y requieren validación para cada tejido, calibración y protocolo de inspección.

---

# Instalación del repositorio y recuperación de pesos mediante Git LFS

Esta sección documenta los pasos adicionales realizados para preparar el entorno, descargar el repositorio y diagnosticar los archivos de pesos de los modelos.

## 1\. Instalación de Git LFS

Durante el diagnóstico se detectó que varios archivos de modelos contenían punteros de Git LFS en lugar de los pesos reales.

Para instalar Git LFS en Ubuntu:

```
sudo apt update
sudo apt install git-lfs
git lfs install
```

Comprobar la instalación:

```
git lfs version
```

En el entorno utilizado se obtuvo:

```
git-lfs/2.9.2 (GitHub; linux amd64; go 1.13.5)
```

La actualización de los índices de paquetes de Ubuntu mostró advertencias relacionadas con las claves GPG de los repositorios Intel RealSense y ROS. Estas advertencias no estaban directamente relacionadas con el error de carga de RF-DETR.

## 2\. Descarga del repositorio

La copia original se descargó como ZIP desde GitHub. Por ello, la carpeta original no contenía un directorio `.git` y no permitía ejecutar directamente `git lfs pull`.

Se identificó la siguiente URL como posible origen del proyecto:

```
https://github.com/Pae14/demo.git
```

Para mantener intacta la copia original, se intentó clonar el repositorio en un directorio independiente:

```
cd ~/Escritorio/remain_project

git clone https://github.com/Pae14/demo.git demo_lfs
```

Si la clonación termina correctamente:

```
cd ~/Escritorio/remain_project/demo_lfs

git lfs pull
```

Estos comandos deben ejecutarse únicamente si el repositorio es accesible y la clonación se ha completado correctamente.

**Estado:** queda pendiente confirmar que la URL sea accesible y que el repositorio contenga los pesos necesarios. Si GitHub devuelve un error 404 o de permisos, será necesario verificar la dirección o los permisos de acceso.

## 3\. Diagnóstico de los archivos de pesos

Los archivos de modelos se encuentran en:

```
viewer_project/models/
```

Las rutas configuradas en `config.py` son:

| Modelo | Ruta |
| --- | --- |
| YOLO12 | `models/yolo12/best.pt` |
| YOLO26 | `models/yolo26/best.pt` |
| RF-DETR | `models/checkpoint_best_ema.pt` |
| DINOv3 | `models/best.pt` |
| ResNet | `models/checkpoint_epoch_49.pth.tar` |
| CNN | `models/best_model.weights.h5` |

Para comprobar los archivos:

```
cd ~/Escritorio/remain_project/demo-main\(1\)/demo-main/viewer_project

ls -lh models/
ls -lh models/yolo12/
ls -lh models/yolo26/
```

Para identificar los archivos que todavía contienen punteros de Git LFS:

```
grep -RIl \
  --exclude-dir=.git \
  "git-lfs.github.com/spec/v1" \
  models/
```

Durante el diagnóstico se encontraron cuatro archivos de aproximadamente 134 bytes:

```
best_model.weights.h5
best.pt
checkpoint_best_ema.pt
checkpoint_epoch_49.pth.tar
```

Al inspeccionar su contenido, se encontró el encabezado:

```
version https://git-lfs.github.com/spec/v1
oid sha256:871802e1f1d133572035662018df93390e84b3f7dacfb649504c0fc0f9c88fb8
size 345231921
```

Este contenido es un puntero de Git LFS, no los pesos binarios del modelo. El tamaño declarado para ese archivo es de 345.231.921 bytes.

Para comprobar un archivo concreto:

```
head -n 4 models/checkpoint_best_ema.pt
```

Si aparece el encabezado de Git LFS, el archivo real todavía no se ha recuperado.

## 4\. Recuperación de los pesos

Una vez confirmado que el repositorio original es accesible, descargar los objetos LFS mediante:

```
cd ~/Escritorio/remain_project/demo_lfs

git lfs pull
```

Después, verificar que los archivos ya no contienen punteros:

```
grep -RIl \
  --exclude-dir=.git \
  "git-lfs.github.com/spec/v1" \
  viewer_project/models/
```

Si el comando no devuelve resultados, no se han encontrado punteros con ese contenido en los archivos inspeccionados.

Comprobar el tamaño de los modelos:

```
ls -lh viewer_project/models/
ls -lh viewer_project/models/yolo12/
ls -lh viewer_project/models/yolo26/
```

No se deben intercambiar los archivos entre modelos ni descargar checkpoints aleatorios. Cada archivo debe corresponder al modelo y a la tarea para la que fue entrenado.

Si el repositorio no contiene los pesos o no es accesible, será necesario recuperar cada archivo desde su fuente original.

## 5\. Reconstrucción y arranque de Docker

Después de recuperar los pesos, volver al directorio del proyecto:

```
cd ~/Escritorio/remain_project/demo_lfs/viewer_project
```

Construir la imagen:

```
docker compose build
```

Iniciar la aplicación:

```
docker compose up
```

Para comprobar el estado de los servicios:

```
docker compose ps
```

Consultar los registros del visor:

```
docker compose logs --tail=100 viewer
```

Si se necesita entrar en un contenedor en ejecución:

```
docker ps
docker exec -it ID_O_NOMBRE_DEL_CONTENEDOR bash
```

Sustituir `ID_O_NOMBRE_DEL_CONTENEDOR` por el identificador o nombre que aparezca en `docker ps`.

Para salir:

```
exit
```

Los comandos `docker compose` deben ejecutarse normalmente desde el host Ubuntu, dentro del directorio que contiene `docker-compose.yml`.

## 6\. Ejecución del nodo ROS 2 del GelSight Mini

Para iniciar el publicador de imagen y profundidad desde el host Ubuntu:

```
cd ~/Escritorio/remain_project/demo_lfs/viewer_project

docker compose run --rm viewer \
  ros2 run vb_tactile_sensors gs_image_depth
```

En otra terminal, comprobar los topics:

```
cd ~/Escritorio/remain_project/demo_lfs/viewer_project

docker compose run --rm viewer ros2 topic list
```

Los topics esperados son:

```
/gs_mini_img
/gs_mini_pcd
```

`/gs_mini_img` corresponde a la imagen del sensor y `/gs_mini_pcd` a la nube de puntos 3D.

Su disponibilidad depende de que el nodo se haya iniciado correctamente y de que el sensor esté accesible.

## 7\. Estado final de la instalación

El diagnóstico ha permitido confirmar que:

- Git LFS está instalado en Ubuntu.
- La copia original descargada como ZIP no contiene un repositorio Git en su directorio raíz.
- Varios archivos de pesos contienen punteros de Git LFS en lugar de los modelos reales.
- El error `invalid load key, 'v'` de RF-DETR es compatible con el intento de cargar uno de esos punteros como checkpoint.
- Se ha identificado una posible URL del repositorio original, pero todavía debe verificarse su accesibilidad.
- Se han documentado los comandos para clonar el repositorio, descargar los objetos LFS, comprobar los pesos y ejecutar Docker y ROS 2.

La instalación no debe considerarse completada hasta confirmar que los pesos reales se han recuperado, que los modelos se cargan correctamente y que la aplicación arranca sin errores de inicialización.
