# ros2-lidar

Sistema de detección y fusión de obstáculos usando LiDAR y cámaras en ROS2.

## Requisitos del Sistema

**Distribución ROS2:** Este proyecto está desarrollado y probado con **ROS2 Jazzy Jalisco**. No se recomienda usar otras distribuciones para su ejecución.

**Nota sobre entornos:** Se asume que tienes los `source` de ROS2 configurados en tu `.bashrc`. Si no es así, ejecuta manualmente:
```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

## Instalación

### 1. Entorno Virtual de Python (Opcional pero Recomendado)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Dependencias de Python
El archivo `requirements.txt` está ubicado en `src/`. Instalar con:
```bash
pip3 install -r src/requirements.txt
```

### 3. Compilación
```bash
colcon build --symlink-install
```

## Preparación del Rosbag

**Requisito previo:** Necesitas el archivo rosbag (`.db3`, ~36GB) para ejecutar el sistema. Este archivo no está incluido en el repositorio por limitaciones de tamaño y derechos de propiedad.

### Ubicación del Rosbag
- **WSL/VM:** Monta el directorio en `/mnt/` (ej: `/mnt/c/Users/tu_usuario/Downloads/`)
- **Linux nativo:** Usa cualquier directorio de tu preferencia

## Arquitectura del Sistema

### Flujo de Datos

```
                         ROSBAG2 PLAYBACK
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
    /my_camera/pylon_ros2_camera_node    /ouster/points
         /image_raw (Image)              (PointCloud2)
                │                               │
                ▼                               ▼
    ┌─────────────────────┐         ┌─────────────────────┐
    │ Image Detector      │         │ LiDAR Detector      │
    │ (YOLOv8)            │         │ (DBSCAN Clustering) │
    └─────────────────────┘         └─────────────────────┘
                │                               │
                ▼                               ▼
    /camera/obstacles                /lidar/obstacles
    (ImageObstacleArray)            (PointCloudObstacleArray)
                │                               │
                └───────────────┬───────────────┘
                                ▼
                    ┌───────────────────────┐
                    │   Fusion Node         │
                    │ (ApproxTimeSynchron.) │
                    │ Hungarian/KDTree      │
                    └───────────────────────┘
                                │
                                ▼
                        /fused/obstacles
                      (FusedObstacleArray)
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Data Snapshot Service │
                    │ (Buffer histórico)    │
                    └───────────────────────┘
                                │
                                ▼
                    /get_fused_snapshot (Servicio)
```

### Componentes Principales

**1. Image Obstacle Detector** (`image_obstacle_detector`)
- **Entrada:** `/my_camera/pylon_ros2_camera_node/image_raw` (sensor_msgs/Image)
- **Salida:** `/camera/obstacles` (ImageObstacleArray), `/camera/obstacles/overlay` (Image)
- **Algoritmo:** YOLOv8 con clases COCO filtradas (person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign)
- **Preprocesamiento:** Resize a 640x640 con padding, normalización automática

**2. LiDAR Obstacle Detector** (`lidar_object_detector`)
- **Entrada:** `/ouster/points` (sensor_msgs/PointCloud2)
- **Salida:** `/lidar/obstacles` (PointCloudObstacleArray), `/lidar/obstacles/overlay` (MarkerArray)
- **Algoritmo:** DBSCAN clustering
- **Filtros aplicados:**
  - Ground filtering: z > -2.0m
  - FOV: [-90°, 90°]
  - Max distance: configurable (default 5.5m)
  - Downsampling: cada N puntos (configurable)

**3. Obstacle Fusion Node** (`fusion_data`)
- **Entradas:** `/camera/obstacles` + `/lidar/obstacles` (sincronizados con ApproximateTimeSynchronizer)
- **Salida:** `/fused/obstacles` (FusedObstacleArray), `/fused/obstacles/overlay` (MarkerArray)
- **Estrategia de asociación:**
  - ≤40 obstáculos: Hungarian Algorithm (asignación óptima global)
  - >40 obstáculos: KDTree Nearest Neighbor (eficiencia)
- **Fusión de centroides:** Ponderación por num_points (LiDAR) y confidence (imagen)
- **Tracking:** Reasignación de IDs por proximidad espacial entre frames

**4. Data Snapshot Server** (`data_snapshot_server`)
- **Suscripciones:** `/ouster/points`, `/lidar/obstacles`, `/camera/obstacles`
- **Servicio:** `/get_fused_snapshot` (GetFusedSnapshot.srv)
- **Funcionalidad:** Mantiene buffers históricos ilimitados, permite consultas por ventanas temporales

### Detalles de Implementación

**Sincronización Temporal:**
- ApproximateTimeSynchronizer con tolerancia configurable (default: 0.5s)
- Timestamps preservados del header original de los sensores

**Asociación de Datos:**
- Umbral de distancia configurable (default: 1.2m)
- Hungarian algorithm garantiza asignación óptima global en escenas pequeñas
- KDTree optimiza rendimiento en escenas densas

**Tracking Persistente:**
- IDs se mantienen entre frames si distancia euclidiana < reid_distance (default: 0.6m)
- Tracks expiran tras track_max_age sin detecciones (default: 0.8s)

**Limitaciones Conocidas:**
- La proyección de fusión no es precisa debido a falta de calibración extrínseca cámara-LiDAR en el rosbag
- No hay compensación de movimiento del vehículo entre sensores

## Configuración de Parámetros

### Detector de Imágenes

**Parámetros disponibles:**
```bash
# Umbral de confianza para detecciones (0.0-1.0)
ros2 param set /image_obstacle_detector confidence_threshold 0.7

# Área mínima de bounding box en píxeles
ros2 param set /image_obstacle_detector min_area 1000

# Procesar 1 de cada N frames (reducir carga)
ros2 param set /image_obstacle_detector frame_skip 2

# Publicar imagen con bounding boxes
ros2 param set /image_obstacle_detector publish_overlay true

# Ruta al modelo YOLOv8
ros2 param set /image_obstacle_detector model_path "ruta/al/modelo.pt"
```

**Valores por defecto:**
- confidence_threshold: 0.5
- min_area: 500
- frame_skip: 1
- publish_overlay: true
- model_path: "src/yolov8n.pt"

### Detector LiDAR

**Parámetros disponibles:**
```bash
# Radio máximo para agrupar puntos en un cluster (metros)
ros2 param set /lidar_object_detector eps 0.55

# Número mínimo de puntos para formar un cluster
ros2 param set /lidar_object_detector min_points 45

# Factor de downsampling (procesar 1 de cada N puntos)
ros2 param set /lidar_object_detector downsampling_factor 5

# Distancia máxima de detección (metros)
ros2 param set /lidar_object_detector max_distance 5.5
```

**Valores por defecto:**
- eps: 0.55
- min_points: 45
- downsampling_factor: 5
- max_distance: 5.5

### Nodo de Fusión

**Parámetros disponibles:**
```bash
# Distancia máxima para asociar detecciones LiDAR-Imagen (metros)
ros2 param set /obstacle_fusion_node association_distance 1.2

# Tolerancia temporal para sincronización de mensajes (segundos)
ros2 param set /obstacle_fusion_node time_tolerance 0.5

# Umbral para cambiar de Hungarian a KDTree
ros2 param set /obstacle_fusion_node max_hungarian_size 40

# Tiempo máximo sin detección antes de eliminar track (segundos)
ros2 param set /obstacle_fusion_node track_max_age 0.8

# Distancia máxima para reasignar ID a track existente (metros)
ros2 param set /obstacle_fusion_node reid_distance 0.6
```

**Valores por defecto:**
- association_distance: 1.2
- time_tolerance: 0.5
- max_hungarian_size: 40
- track_max_age: 0.8
- reid_distance: 0.6

## Ejecución

### Opción 1: Nodos Independientes

Abre 5 terminales para ejecutar cada componente:

```bash
# Terminal 1: Detector de imágenes (YOLO)
ros2 run lidar_detection_pkg image_obstacle_detector

# Terminal 2: Detector de obstáculos LiDAR
ros2 run lidar_detection_pkg lidar_object_detector

# Terminal 3: Fusión de datos (proyección imagen + LiDAR)
ros2 run lidar_detection_pkg fusion_data

# Terminal 4: Servicio de snapshot (opcional)
ros2 run lidar_detection_pkg data_snapshot_server

# Terminal 5: Reproducción del rosbag (ajusta la ruta)
ros2 bag play /ruta/a/tu/rosbag2_2025_02_27-13_08_14_0-001.db3
```

**Nota:** La proyección de fusión no es precisa debido a la falta de datos de posicionamiento exacto de las fuentes en el rosbag.

### Opción 2: Usando Launch File

```bash
# Terminal 1: Launch todos los nodos
ros2 launch lidar_detection_pkg lidar_image.launch.py

# Terminal 2: Reproducción del rosbag
ros2 bag play /ruta/a/tu/rosbag2_2025_02_27-13_08_14_0-001.db3
```

## Visualización con RViz

### Displays a añadir:

#### Visualizar Imagen con Detecciones (YOLO)
| Configuración | Valor |
|---------------|-------|
| **Tipo** | Image |
| **Tópico** | `/camera/obstacles/overlay` |
| **Descripción** | Muestra la imagen con bounding boxes de YOLOv8 |

#### Visualizar Detecciones LiDAR

**a) Nube de puntos cruda**

| Configuración | Valor |
|---------------|-------|
| **Tipo** | PointCloud2 |
| **Tópico** | `/ouster/points` |
| **Style** | Points |
| **Size** | 0.05 |
| **Color Transformer** | Intensity o AxisColor |

**b) Bounding boxes de clusters**

| Configuración | Valor |
|---------------|-------|
| **Tipo** | MarkerArray |
| **Tópico** | `/lidar/obstacles/overlay` |
| **Descripción** | Cubos rojos representando clusters detectados |

#### Visualizar Obstáculos Fusionados

| Configuración | Valor |
|---------------|-------|
| **Tipo** | MarkerArray |
| **Tópico** | `/fused/obstacles/overlay` |
| **Código de colores** | 🟢 Verde: LiDAR + Imagen<br>🔴 Rojo: Solo LiDAR<br>🔵 Azul: Solo Imagen |

### Resultados de Detección

**Detección de Imagen (YOLO):**
![Image Detection](/img/lidar-bounding-boxes.jpg)

**Detección LiDAR con Visualización:**
![LiDAR Detection](/img/image-bounding-boxes.png)

## Data Snapshot Service

### Descripción
El nodo `data_snapshot_server` proporciona un servicio de sincronización temporal que permite recuperar datos históricos de múltiples fuentes:
- Detecciones de imagen (YOLO)
- Detecciones de LiDAR (clustering DBSCAN)
- Nubes de puntos crudas del sensor

Los datos se almacenan en buffers ilimitados y pueden consultarse mediante ventanas temporales para análisis posteriores o depuración.

### Uso del Servicio

**1. Iniciar el servidor:**
```bash
ros2 run lidar_detection_pkg data_snapshot_server
```

**2. Iniciar los nodos de detección y el rosbag** (ver secciones anteriores)

**3. Realizar una consulta:**
```bash
# Consultar datos del último segundo (ajusta los timestamps según tus datos)
ros2 service call /get_fused_snapshot lidar_interfaces/srv/GetFusedSnapshot \
  "{start_time: {sec: 1740658100, nanosec: 0}, end_time: {sec: 1740658101, nanosec: 0}}"
```

**Respuesta del servicio:**
El servicio devuelve todas las detecciones y datos de sensores que ocurrieron dentro de la ventana temporal especificada, permitiendo análisis sincronizado de múltiples fuentes.

## Troubleshooting

- **Error de tópicos no encontrados:** Verifica que el rosbag se esté reproduciendo correctamente con `ros2 topic list`
- **Detector no publica:** Revisa los parámetros de confidence threshold o clustering
- **Proyección imprecisa:** Normal debido a falta de calibración precisa en los datos del rosbag
- **RViz no muestra markers:** Verifica que el Fixed Frame coincida con el frame_id de los mensajes (usa `ros2 topic echo /lidar/obstacles/overlay`)
- **Baja tasa de FPS:** Aumenta `frame_skip` en el detector de imágenes o `downsampling_factor` en LiDAR

## Notas Adicionales

- Los timestamps deben estar en formato de tiempo Unix (segundos y nanosegundos)
- La fusión de datos es aproximada debido a limitaciones en la calibración espacial
- Para mejor rendimiento, ajusta `frame_skip` en el detector de imágenes según tu hardware
- Los parámetros pueden modificarse en tiempo real sin reiniciar los nodos