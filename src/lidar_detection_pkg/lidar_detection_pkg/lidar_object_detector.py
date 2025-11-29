#!/usr/bin/env python3
import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2
from sklearn.cluster import DBSCAN
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from lidar_interfaces.msg import PointCloudObstacle, PointCloudObstacleArray


class LidarObstacleDetector(Node):
    """
    Procesa nubes de puntos LiDAR para detectar obstáculos mediante clustering DBSCAN.
    Publica una lista de obstáculos detectados y marcadores de visualización.
    """

    DEFAULT_EPS = 0.55
    DEFAULT_MIN_POINTS = 45
    DEFAULT_DOWNSAMPLING = 5
    DEFAULT_MAX_DISTANCE = 5.5
    GROUND_THRESHOLD = -2.0
    FOV_MIN = -np.pi / 2
    FOV_MAX = np.pi / 2
    MARKER_LIFETIME_SEC = 1
    MIN_DIMENSION = 0.001

    def __init__(self):
        """Inicializa el nodo, parámetros y configuración de ROS 2."""
        super().__init__("lidar_obstacle_detector")

        # Declaration and loading of the parameters
        self._declare_parameters()
        self._load_parameters()

        # Setup of the communications
        self._setup_publishers()
        self._setup_subscriptions()

        # Register parameter callback
        self.add_on_set_parameters_callback(self._parameter_callback)

        self.get_logger().info("LidarObstacleDetector node started.")

    def _declare_parameters(self):
        """Declara los parámetros del nodo con sus valores por defecto."""
        self.declare_parameter("eps", self.DEFAULT_EPS)
        self.declare_parameter("min_points", self.DEFAULT_MIN_POINTS)
        self.declare_parameter("downsampling_factor", self.DEFAULT_DOWNSAMPLING)
        self.declare_parameter("max_distance", self.DEFAULT_MAX_DISTANCE)

    def _load_parameters(self):
        """Carga los valores actuales de los parámetros en variables de instancia."""
        self.eps = self.get_parameter("eps").get_parameter_value().double_value
        self.min_points = (
            self.get_parameter("min_points").get_parameter_value().integer_value
        )
        self.downsampling_factor = (
            self.get_parameter("downsampling_factor")
            .get_parameter_value()
            .integer_value
        )
        self.max_distance = (
            self.get_parameter("max_distance").get_parameter_value().double_value
        )

    def _setup_publishers(self):
        """Configura los publishers del nodo."""
        self._obstacle_publisher = self.create_publisher(
            PointCloudObstacleArray, "/lidar/obstacles", 10
        )
        self._overlay_publisher = self.create_publisher(
            MarkerArray, "/lidar/obstacles/overlay", 10
        )

    def _setup_subscriptions(self):
        """Configura las suscripciones del nodo."""
        self._pointcloud_sub = self.create_subscription(
            PointCloud2, "/ouster/points", self._pointcloud_callback, 10
        )

    def _parameter_callback(self, params: list[Parameter]) -> SetParametersResult:
        """
        Callback para la reconfiguración dinámica de parámetros.

        Args:
            params: Lista de parámetros modificados.

        Returns:
            SetParametersResult indicando éxito o fallo.
        """
        for param in params:
            if (param.name == "eps") and (param.value >= 0):
                self.eps = float(param.value)
            elif (param.name == "min_points") and (param.value >= 1):
                self.min_points = int(param.value)
            elif (param.name == "downsampling_factor") and (param.value >= 1):
                self.downsampling_factor = int(param.value)
            elif (param.name == "max_distance") and (param.value > 0):
                self.max_distance = float(param.value)
            else:
                return SetParametersResult(
                    successful=False,
                    reason=f"Invalid value for parameter '{param.name}': {param.value}",
                )
        return SetParametersResult(successful=True)

    def _pointcloud_callback(self, msg: PointCloud2) -> None:
        """
        Callback principal que procesa la nube de puntos entrante.

        Flow: Convert -> Preprocess -> Cluster -> Publish
        """
        points = self._convert_pointcloud_to_array(msg)
        if points is None or points.shape[0] == 0:
            return

        points = self._preprocess_points(points)
        # Check if points exist after preprocessing
        if points is None or points.shape[0] < max(2, self.min_points):
            # Si no hay puntos tras el filtrado, publicamos arrays vacíos para mantener vivo el sistema
            self._publish_empty_state(msg.header)
            return

        labels = self._perform_clustering(points)

        frame_id = msg.header.frame_id if msg.header.frame_id else "os_sensor"
        timestamp = msg.header.stamp

        self._publish_obstacles_and_markers(points, labels, frame_id, timestamp)

    def _convert_pointcloud_to_array(self, msg: PointCloud2) -> np.ndarray | None:
        """
        Convierte un mensaje PointCloud2 a un array de NumPy (N, 3).
        Intenta usar una conversión vectorizada eficiente primero.
        """
        try:
            points_iter = pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )
            # Optimizacion: fromiter es mas rápido que list comprehension
            flat = np.fromiter(
                (coord for p in points_iter for coord in p), dtype=np.float32
            )

            if flat.size == 0:
                return None

            points = flat.reshape((-1, 3))
            return points

        except (ValueError, RuntimeError) as e:
            self.get_logger().warn(f"Efficient conversion failed: {e}. Using fallback.")
            # Fallback
            try:
                points_iter = pc2.read_points(
                    msg, field_names=("x", "y", "z"), skip_nans=True
                )
                points = np.asarray(
                    [[p[0], p[1], p[2]] for p in points_iter], dtype=np.float32
                )
                return points if points.size > 0 else None
            except Exception as e:
                self.get_logger().error(f"PointCloud conversion fatal error: {e}")
                return None

    def _preprocess_points(self, points: np.ndarray) -> np.ndarray | None:
        """
        Filtra puntos por altura (suelo), distancia y FOV, y aplica downsampling.
        """
        # Ground filtering mask
        height_mask = points[:, 2] > self.GROUND_THRESHOLD

        # Angles and distances mask (vectorized)
        distances = np.linalg.norm(points[:, :2], axis=1)
        angles = np.arctan2(points[:, 1], points[:, 0])

        range_mask = distances < self.max_distance
        fov_mask = (angles >= self.FOV_MIN) & (angles <= self.FOV_MAX)

        final_mask = height_mask & range_mask & fov_mask
        points = points[final_mask]

        # Downsampling
        if self.downsampling_factor > 1:
            points = points[:: self.downsampling_factor]

        return points if points.shape[0] > 0 else None

    def _perform_clustering(self, points: np.ndarray) -> np.ndarray:
        """Ejecuta DBSCAN sobre los puntos procesados."""
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_points, n_jobs=-1).fit(
            points
        )
        return clustering.labels_

    def _publish_empty_state(self, header: Header):
        """Publica arrays vacíos cuando no se detectan obstáculos."""
        # Empty Obstacle Array
        obs_msg = PointCloudObstacleArray()
        obs_msg.header = header
        self._obstacle_publisher.publish(obs_msg)

        # Empty Marker Array
        self._overlay_publisher.publish(MarkerArray())

    def _publish_obstacles_and_markers(
        self, points: np.ndarray, labels: np.ndarray, frame_id: str, timestamp: Time
    ) -> None:
        """
        Genera los mensajes de obstáculos y marcadores y los publica.
        Agrupa todos los obstáculos en un solo mensaje PointCloudObstacleArray.
        """
        unique_labels = np.unique(labels[labels != -1])

        # Contenedores para el batch
        obstacle_list = []
        marker_array = MarkerArray()

        header = Header()
        header.frame_id = frame_id
        self._logger.info(f"Frame ID for publishing: {frame_id}")
        header.stamp = timestamp

        # Si no hay clusters, publicamos vacio
        if unique_labels.size == 0:
            self._publish_empty_state(header)
            return

        for obstacle_id, label in enumerate(unique_labels):
            cluster_mask = labels == label
            cluster = points[cluster_mask]

            centroid, dimensions = self._compute_cluster_properties(cluster)

            # 1. Crear mensaje de obstáculo
            obstacle_msg = self._create_obstacle_message(
                obstacle_id, centroid, dimensions, np.sum(cluster_mask), header
            )
            obstacle_list.append(obstacle_msg)

            # 2. Crear marcador visual
            marker = self._create_visualization_marker(obstacle_msg, obstacle_id, header)
            marker_array.markers.append(marker)

        # --- PUBLICACIÓN ---

        # Publicar Array de Obstáculos para el Servicio/Fusión
        array_msg = PointCloudObstacleArray()
        array_msg.header = header
        array_msg.obstacles = obstacle_list
        self._obstacle_publisher.publish(array_msg)

        # Publicar Marcadores para RViz
        self._overlay_publisher.publish(marker_array)

        self.get_logger().info(
            f"Published {len(obstacle_list)} obstacles",
            throttle_duration_sec=1.0,
        )

    def _compute_cluster_properties(
        self,
        cluster: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calcula centroide y dimensiones (AABB) de un cluster."""
        centroid = np.mean(cluster, axis=0)
        min_bounds = np.min(cluster, axis=0)
        max_bounds = np.max(cluster, axis=0)
        dimensions = max_bounds - min_bounds
        return centroid, dimensions

    def _create_obstacle_message(
        self,
        obstacle_id: int,
        centroid: np.ndarray,
        dimensions: np.ndarray,
        num_points: int,
        header: Header,
    ) -> PointCloudObstacle:
        """Instancia un mensaje PointCloudObstacle simple."""
        msg_out = PointCloudObstacle()
        # msg_out.header = header
        msg_out.id = obstacle_id
        msg_out.centroid = Point(
            x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2])
        )
        # Evitamos dimensiones 0 para no romper visualizaciones
        msg_out.width = float(max(dimensions[0], self.MIN_DIMENSION))
        msg_out.depth = float(
            max(dimensions[1], self.MIN_DIMENSION)
        )  # Depth suele ser Y en coords locales de objeto o bounding box
        msg_out.height = float(max(dimensions[2], self.MIN_DIMENSION))
        msg_out.num_points = int(num_points)
        return msg_out

    def _create_visualization_marker(
        self, obstacle_msg: PointCloudObstacle, obstacle_id: int, header: Header = None
    ) -> Marker:
        """Crea un Marker cúbico para RViz basado en el obstáculo."""
        marker = Marker()
        if header is not None:
            marker.header = header
        marker.ns = "obstacles"
        marker.id = obstacle_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position = obstacle_msg.centroid
        marker.pose.orientation.w = 1.0

        marker.scale.x = obstacle_msg.width
        marker.scale.y = obstacle_msg.depth
        marker.scale.z = obstacle_msg.height

        marker.color.a = 0.5
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        marker.lifetime.sec = self.MARKER_LIFETIME_SEC
        marker.lifetime.nanosec = 0

        return marker


def main(args=None):
    rclpy.init(args=args)
    node = LidarObstacleDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
