import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from collections import deque
import threading
import numpy as np

# Mensajes estándar
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

# Mensajes custom
from lidar_interfaces.msg import (
    ImageObstacleArray,
    PointCloudObstacleArray,
    PointCloudObstacle
)
from lidar_interfaces.srv import GetFusedSnapshot


class DataSnapshotServer(Node):
    """
    Nodo tipo Servicio que mantiene un buffer histórico de sensores y detecciones.
    Permite consultar el estado del entorno en un intervalo de tiempo pasado.
    """

    def __init__(self):
        super().__init__('data_snapshot_server')

        # --- Configuración ---
        # Tamaño del buffer en segundos (aprox)
        self.buffer_duration_sec = 600.0

        # Buffers thread-safe para almacenar datos (msg, receive_time_nanos)
        self._lidar_buffer = deque()
        self._image_buffer = deque()
        self._cloud_buffer = deque()
        self._buffer_lock = threading.Lock()

        # QoS para sensores (Best Effort es común para Lidar/Cámaras)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # --- Suscripciones ---
        self.create_subscription(
            PointCloud2,
            '/ouster/points',
            self._callback_cloud,
            sensor_qos
        )

        self.create_subscription(
            PointCloudObstacleArray,  # ¡Importante! Debe ser un Array
            '/obstacles',
            self._callback_lidar,
            reliable_qos
        )

        self.create_subscription(
            ImageObstacleArray,
            '/image_obstacles',
            self._callback_image,
            reliable_qos
        )

        # --- Servicio ---
        self.srv = self.create_service(
            GetFusedSnapshot,
            'get_fused_snapshot',
            self._handle_snapshot_request
        )

        self.get_logger().info("Data Snapshot Server iniciado. Esperando datos...")

    # --- Callbacks de Suscripción ---

    def _callback_cloud(self, msg: PointCloud2):
        self.get_logger().debug(f"CLOUD: Recibido timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        self._add_to_buffer(self._cloud_buffer, msg)

    def _callback_lidar(self, msg: PointCloudObstacleArray):
        self.get_logger().debug(f"LIDAR: Recibido timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec} con {len(msg.obstacles)} obstáculos.")
        self._add_to_buffer(self._lidar_buffer, msg)

    def _callback_image(self, msg: ImageObstacleArray):
        self.get_logger().debug(
            f"IMAGEN: Recibido timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec} con {len(msg.obstacles)} obstáculos.")
        self._add_to_buffer(self._image_buffer, msg)

    def _add_to_buffer(self, buffer: deque, msg):
        """
        Agrega mensaje al buffer sin realizar limpieza de datos antiguos.
        (La lógica de borrado ha sido eliminada por requerimiento explícito).
        """
        with self._buffer_lock:
            # Usamos el timestamp del header para la lógica de negocio
            # Si el header es 0, usamos el reloj del sistema (fallback)
            msg_time_ns = Time.from_msg(msg.header.stamp).nanoseconds
            if msg_time_ns == 0:
                msg_time_ns = self.get_clock().now().nanoseconds

            # 1. Añadir el nuevo mensaje al final del buffer
            buffer.append((msg_time_ns, msg))

            # 2. Lógica de Limpieza Original (DESACTIVADA)
            # La limpieza se ha eliminado para que el buffer nunca borre datos.
            # current_time = self.get_clock().now().nanoseconds
            # cutoff_time = current_time - (self.buffer_duration_sec * 1e9)
            # while buffer and buffer[0][0] < cutoff_time:
            #     buffer.popleft() # <-- Línea de borrado eliminada

            # 3. Log de Diagnóstico
            if len(buffer) % 10 == 0 or len(buffer) < 5 and len(buffer) > 0:
                self.get_logger().info(
                    f"BUFFER {msg.__class__.__name__}: Tamaño actual {len(buffer)}. Último tiempo: {msg_time_ns // 10 ** 9}")
    # --- Lógica del Servicio ---

    def _handle_snapshot_request(self, request, response):
        start_ns = Time.from_msg(request.start_time).nanoseconds
        end_ns = Time.from_msg(request.end_time).nanoseconds

        # Validación básica
        if start_ns > end_ns:
            response.success = False
            response.message = "Start time cannot be after end time."
            return response

        # Punto medio del intervalo solicitado para buscar la mejor coincidencia
        target_time = (start_ns + end_ns) / 2

        with self._buffer_lock:
            cloud_match = self._find_closest_msg(self._cloud_buffer, target_time, start_ns, end_ns)
            lidar_match = self._find_closest_msg(self._lidar_buffer, target_time, start_ns, end_ns)
            image_match = self._find_closest_msg(self._image_buffer, target_time, start_ns, end_ns)

        if cloud_match is None:
            response.success = False
            response.message = "No PointCloud found in the requested interval."
            return response

        response.success = True
        response.pointcloud = cloud_match
        response.message = "Snapshot retrieved."

        # Es aceptable devolver arrays vacíos si no hubo detección, pero el PointCloud es obligatorio
        if lidar_match:
            response.lidar_obstacles = lidar_match
        else:
            response.lidar_obstacles = PointCloudObstacleArray()  # Vacío
            response.lidar_obstacles.header.stamp = response.pointcloud.header.stamp
            response.lidar_obstacles.header.frame_id = response.pointcloud.header.frame_id

        if image_match:
            response.image_obstacles = image_match
        else:
            response.image_obstacles = ImageObstacleArray()  # Vacío
            response.image_obstacles.header.stamp = response.pointcloud.header.stamp

        return response

    def _find_closest_msg(self, buffer, target_time, start_limit, end_limit):
        """
        Busca el mensaje más cercano al target_time dentro de los límites.
        Complejidad O(N), pero N es pequeño (buffer limitado).
        """
        closest_msg = None
        min_diff = float('inf')

        for timestamp, msg in buffer:
            # Filtro estricto de ventana
            if timestamp < start_limit or timestamp > end_limit:
                continue

            diff = abs(timestamp - target_time)
            if diff < min_diff:
                min_diff = diff
                closest_msg = msg

        return closest_msg


def main(args=None):
    rclpy.init(args=args)
    node = DataSnapshotServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()