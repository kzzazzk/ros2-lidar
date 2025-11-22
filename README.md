# ros2-lidar
## Instalación de Dependencias

### Cargar entorno de ROS

Cargar underlay:
```Bash

source /opt/ros/jazzy/setup.bash
```

### Dependencias del Sistema (rosdep)

Instalar dependencias de ROS y del sistema.

- Para Ubuntu estándar:
```Bash

rosdep install -i --from-path src --rosdistro jazzy -y
```

- Para Linux Mint:
```Bash

rosdep install -i --from-path src --rosdistro jazzy -y --os=ubuntu:noble
```
- Para Windows:
```Bash

    rosdep install -i --from-path src --rosdistro jazzy -y
```

### Dependencias de Python

El paquete lidar_detection_pkg requiere librerías específicas (torch, ultralytics, opencv-python, etc.). Para instalarlas automáticamente basándonos en la configuración del setup.py:
```Bash

# Instalar el paquete en modo editable para resolver el bloque 'install_requires'
pip3 install -e src/lidar_detection_pkg/
```

## Compilación (Build)

Compilar el espacio de trabajo con colcon. Usamos --symlink-install para facilitar el desarrollo (los cambios en scripts de Python se reflejan sin recompilar).
```Bash

colcon build --symlink-install
```
Si la compilación es exitosa, carga el overlay:
```Bash

source install/setup.bash
```
## Ejecución

Se requiere ejecutar múltiples procesos. Se recomienda usar Terminator o Tmux, o abrir 4 terminales distintas.

Importante: Hacer source install/setup.bash en cada terminal nueva.

### Terminal 1: Detector de Imágenes (YOLO/CV)

```Bash

ros2 run lidar_detection_pkg image_obstacle_detector
```
Nota: Si el tópico del rosbag difiere del esperado (/my_camera/...), usa remapping: --ros-args -r /my_camera/pylon_ros2_camera_node/image_raw:=/TOPICO_REAL

### Terminal 2: Detector de Objetos Lidar

```Bash

ros2 run lidar_detection_pkg lidar_obstacle_detector
```
### Terminal 3: Reproducción de Datos (Rosbag)

Ajustar la ruta según la ubicación del archivo .db3:
```Bash

cd ~/Downloads/
# Asegúrate de que el db3 esté dentro de una carpeta con su metadata si es necesario, o reprodúcelo directamente:
ros2 bag play rosbag2_2025_02_27-13_08_14_0-001.db3
```
## Visualización

Para visualizar los resultados (Bounding boxes, nubes de puntos y marcadores):
Abre RViz:
```Bash

ros2 run rviz2 rviz2
```

Configurar el Fixed Frame (Global Options) a un frame existente (ej. velodyne, map o base_link).
Añadir los siguientes displays (Botón "Add"):

- Image: Selecciona el tópico de la cámara.

- MarkerArray: Selecciona el tópico de visualización de obstáculos del Lidar.

- PointCloud2: (Opcional) Para ver la nube de puntos cruda del rosbag.

- Topics personalizados: Busca los tópicos de tipo ImageObstacleArray si tienes un plugin de visualización, o inspecciónalos vía terminal con ros2 topic echo.
