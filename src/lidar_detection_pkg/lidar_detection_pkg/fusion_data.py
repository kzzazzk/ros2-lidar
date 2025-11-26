import time
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from scipy.optimize import linear_sum_assignment, linprog
from scipy.spatial import KDTree
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from lidar_interfaces.msg import (FusedObstacle, FusedObstacleArray,
                                  ImageObstacle, ImageObstacleArray,
                                  PointCloudObstacle, PointCloudObstacleArray)


class ObstacleFusion(Node):
    """
    Nodo de fusión eficiente entre LiDAR e imagen.

    Estrategia
    1. sincroniza mensajes con ApproximateTimeSynchronizer
    2. extrae centroides y usa KDTree para búsqueda rápida de vecinos
    3. si los tamaños son pequeños y la escena puede requerir asignación global, aplica Hungarian
    4. fusiona centroides ponderando por num_points y confidence
    5. mantiene un tracker simple para IDs persistentes y evitar reseteos por ciclo
    """

    DEFAULT_ASSOCIATION_DISTANCE = 1.2
    DEFAULT_TIME_TOLERANCE = 0.1
    DEFAULT_MAX_HUNGARIAN_SIZE = 40
    DEFAULT_TRACK_MAX_AGE = 0.8
    DEFAULT_REID_DISTANCE = 0.6

    MARKER_LIFETIME_SEC = 1
    MIN_DIMENSION = 0.001

    def __init__(self):
        super().__init__("obstacle_fusion_node")

        # Declaration and loading of the parameters
        self._declare_parameters()
        self._load_parameters()

        # Setup of communications
        self._setup_publishers()
        self._setup_subscriptions()

        # callback parámetros dinámicos
        self.add_on_set_parameters_callback(self._parameter_callback)
        self._create_synchronizer()
        
        # tracker simple para mantener ids persistentes por corto tiempo
        self._last_fused_id = 0
        self._tracks = dict()  # id -> {"centroid": np.array, "last_seen": float}

        self.get_logger().info("ObstacleFusion started.")

    def _declare_parameters(self):
        self.declare_parameter("association_distance", float(self.DEFAULT_ASSOCIATION_DISTANCE))
        self.declare_parameter("time_tolerance", self.DEFAULT_TIME_TOLERANCE)
        self.declare_parameter("max_hungarian_size", self.DEFAULT_MAX_HUNGARIAN_SIZE)
        self.declare_parameter("track_max_age", self.DEFAULT_TRACK_MAX_AGE)
        self.declare_parameter("reid_distance", self.DEFAULT_REID_DISTANCE)

    def _load_parameters(self):
        self.association_distance = (self.get_parameter("association_distance").get_parameter_value().double_value)
        self.time_tolerance = (self.get_parameter("time_tolerance").get_parameter_value().double_value)
        self.max_hungarian_size = (self.get_parameter("max_hungarian_size").get_parameter_value().integer_value)
        self.track_max_age = (self.get_parameter("track_max_age").get_parameter_value().double_value)
        self.reid_distance = (self.get_parameter("reid_distance").get_parameter_value().double_value)

    def _setup_publishers(self):
        self._fused_pub = self.create_publisher(FusedObstacleArray, "/fused_obstacles", 10)
        self._marker_pub = self.create_publisher(MarkerArray, "/fused_obstacle_markers", 10)

    def _setup_subscriptions(self):
        self._lidar_sub = Subscriber(self, PointCloudObstacleArray, "/lidar_obstacles")
        self._image_sub = Subscriber(self, ImageObstacleArray, "/image_obstacles")

    def _create_synchronizer(self):
        """
        Crea un ApproximateTimeSynchronizer con los subscribers actuales.
        Si ya existía, se reemplaza.
        """
        # Si existía, no hay API directa para destruirlo, reasignamos y GC lo limpiará
        self._sync = ApproximateTimeSynchronizer(
            [self._lidar_sub, self._image_sub],
            queue_size=20,
            slop=self.time_tolerance,
            allow_headerless=False,
        )
        self._sync.registerCallback(self._fusion_callback)

    def _parameter_callback(self, params: List[Parameter]) -> SetParametersResult:
        """
        Actualiza parámetros y recrea sincronizer si cambia time_tolerance.
        """
        recreate_sync = False
        for param in params:
            if (param.name == "association_distance") and (param.value >= 0.0):
                self.association_distance = float(param.value)
            elif (param.name == "time_tolerance") and (param.value >= 0.0):
                self.time_tolerance = float(param.value)
                recreate_sync = True
            elif (param.name == "max_hungarian_size") and (param.value >= 1):
                self.max_hungarian_size = int(param.value)
            elif (param.name == "track_max_age") and (param.value >= 0.0):
                self.track_max_age = float(param.value)
            elif (param.name == "reid_distance") and (param.value >= 0.0):
                self.reid_distance = float(param.value)
            else:
                return SetParametersResult(
                    successful=False, reason=f"Invalid value for parameter '{param.name}': {param.value}"
                )
        if recreate_sync:
            # recrear synchronizer con nuevo slop
            self._create_synchronizer()
        return SetParametersResult(successful=True)

    def _fusion_callback(
        self, lidar_msg: PointCloudObstacleArray, image_msg: ImageObstacleArray
    ) -> None:
        """
        Callback principal. Flujo:
        1. Validación de Headers
        2. Extracción de centroides y arrays (para realizar cálculos más rápido)
        3. Se decide la estrategia de asociación (Hungarian / KDTree Nearest Neighbor)
        4. Creación del FusedObstacle con ponderación
        5. Asignación de IDs persistentes con tracker
        6. Publicación de resultados y markers
        """
        # Se elige el header más reciente y el frame_id de lidar si existe
        header = Header()
        header.frame_id = (
            lidar_msg.header.frame_id
            if lidar_msg.header.frame_id
            else image_msg.header.frame_id
        )
        header.stamp = (
            lidar_msg.header.stamp
            if (lidar_msg.header.stamp.sec, lidar_msg.header.stamp.nanosec)
            >= (image_msg.header.stamp.sec, image_msg.header.stamp.nanosec)
            else image_msg.header.stamp
        )

        # Si no hay ningún obstáculo publica vacío
        if len(lidar_msg.obstacles) == 0 and len(image_msg.obstacles) == 0:
            self._publish_empty_state(header)
            return

        lidar_obs = list(lidar_msg.obstacles)
        image_obs = list(image_msg.obstacles)

        lidar_centroids = (
            np.array([[o.centroid.x, o.centroid.y, o.centroid.z] for o in lidar_obs])
            if lidar_obs
            else np.empty((0, 3))
        )
        image_centroids = (
            np.array([[o.centroid.x, o.centroid.y, o.centroid.z] for o in image_obs])
            if image_obs
            else np.empty((0, 3))
        )

        associations: List[Tuple[int, int, float]] = list()

        # Estrategia de asociación
        n_l = lidar_centroids.shape[0]
        n_i = image_centroids.shape[0]

        if n_l > 0 and n_i > 0:
            # Tamaños pequeños y matching global: Hungarian
            if max(n_l, n_i) <= self.max_hungarian_size:
                dmat = self._pairwise_distance_matrix(lidar_centroids, image_centroids)
                row_ind, col_ind = linear_sum_assignment(dmat)
                for r, c in zip(row_ind, col_ind):
                    dist = float(dmat[r, c])
                    if dist <= self.association_distance:
                        associations.append((int(r), int(c), dist))
            # Si no: KDTree NN (salen menos asociaciones pero es más rápido y escalable)
            else:
                tree = KDTree(image_centroids)
                dists, idxs = tree.query(lidar_centroids, k=1, n_jobs=-1)
                used_image = set()
                for li, (dist, ii) in enumerate(zip(dists, idxs)):
                    if dist <= self.association_distance and int(ii) not in used_image:
                        associations.append((int(li), int(ii), float(dist)))
                        used_image.add(int(ii))
                

        # Se prepara el mensaje
        fused_list: List[FusedObstacle] = list()
        matched_lidar = set([a[0] for a in associations])
        matched_image = set([a[1] for a in associations])

        fused_temp: List[Tuple[np.ndarray, FusedObstacle]] = list()
        
        # Fusionados
        for lidar_idx, image_idx, dist in associations:
            fmsg = self._fuse_pair(lidar_obs[lidar_idx], image_obs[image_idx], dist)
            fused_temp.append((np.array([fmsg.centroid.x, fmsg.centroid.y, fmsg.centroid.z], dtype=float), fmsg))

        # Solo Lidar
        for i, lob in enumerate(lidar_obs):
            if i in matched_lidar:
                continue
            fmsg = self._fuse_lidar_only(lob)
            fused_temp.append((np.array([fmsg.centroid.x, fmsg.centroid.y, fmsg.centroid.z], dtype=float), fmsg))

        # Solo imagen
        for j, iob in enumerate(image_obs):
            if j in matched_image:
                continue
            fmsg = self._fuse_image_only(iob)
            fused_temp.append((np.array([fmsg.centroid.x, fmsg.centroid.y, fmsg.centroid.z], dtype=float), fmsg))

        current_time = (
            self.get_clock().now().seconds_nanoseconds()[0]
            + self.get_clock().now().seconds_nanoseconds()[1] * 1e-9
        )
        fused_list = list()
        for centroid_np, fmsg in fused_temp:
            assigned_id = self._assign_or_create_track(centroid_np, current_time)
            fmsg.id = assigned_id
            fused_list.append(fmsg)
            self._tracks[assigned_id] = {
                "centroid": centroid_np.copy(),
                "last_seen": current_time,
            }

        # Se limpian tracks viejos
        self._prune_tracks(current_time)

        # Publicación de resultados
        out_msg = FusedObstacleArray()
        out_msg.header = header
        out_msg.obstacles = fused_list
        self._fused_pub.publish(out_msg)

        # Markers
        marker_array = MarkerArray()
        for obs in fused_list:
            marker_array.markers.append(self._create_marker(obs, header))
        self._marker_pub.publish(marker_array)

        self.get_logger().info(
            f"Published {len(fused_list)} fused obstacles "
            f"(Lidar only: {sum(1 for o in fused_list if o.has_lidar_data and not o.has_image_data)}, "
            f"Image only: {sum(1 for o in fused_list if o.has_image_data and not o.has_lidar_data)}, "
            f"Fused: {sum(1 for o in fused_list if o.has_lidar_data and o.has_image_data)})",
            throttle_duration_sec=1.0,
        )

    def _pairwise_distance_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matriz de distancias euclideas entre a y b."""
        # vectorized
        diff = a[:, None, :] - b[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        return np.sqrt(d2)

    def _fuse_pair(
        self, lidar_obs: PointCloudObstacle, image_obs: ImageObstacle, distance: float
    ) -> FusedObstacle:
        fused = FusedObstacle()
        fused.has_lidar_data = True
        fused.has_image_data = True
        fused.association_distance = float(distance)

        # Se obtienen valores seguros
        ln = getattr(lidar_obs, "num_points", 0)
        ci = float(getattr(image_obs, "confidence", 0.0))

        # Pesos: si no hay num_points se usa 1 y si no hay confidence se usa 1
        w_l = float(max(ln, 1))
        w_i = float(max(ci, 0.001))

        total = w_l + w_i
        w_l /= total
        w_i /= total

        c_l = np.array(
            [lidar_obs.centroid.x, lidar_obs.centroid.y, lidar_obs.centroid.z],
            dtype=float,
        )
        c_i = np.array(
            [image_obs.centroid.x, image_obs.centroid.y, image_obs.centroid.z],
            dtype=float,
        )
        centroid = w_l * c_l + w_i * c_i
        fused.centroid = Point(
            x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2])
        )

        # Se prefiere Lidar para dimensiones y num_points
        fused.width = float(max(getattr(lidar_obs, "width", 0.0), self.MIN_DIMENSION))
        fused.height = float(max(getattr(lidar_obs, "height", 0.0), self.MIN_DIMENSION))
        fused.depth = float(max(getattr(lidar_obs, "depth", 0.0), self.MIN_DIMENSION))
        fused.num_points = int(getattr(lidar_obs, "num_points", 0))

        fused.class_id = int(getattr(image_obs, "class_id", -1))
        fused.class_name = str(getattr(image_obs, "class_name", "unknown"))
        fused.confidence = float(getattr(image_obs, "confidence", 0.0))

        return fused

    def _fuse_lidar_only(self, lidar_obs: PointCloudObstacle) -> FusedObstacle:
        fused = FusedObstacle()
        fused.has_lidar_data = True
        fused.has_image_data = False
        fused.association_distance = -1.0

        fused.centroid = Point(
            x=float(lidar_obs.centroid.x),
            y=float(lidar_obs.centroid.y),
            z=float(lidar_obs.centroid.z),
        )
        fused.width = float(max(getattr(lidar_obs, "width", 0.0), self.MIN_DIMENSION))
        fused.height = float(max(getattr(lidar_obs, "height", 0.0), self.MIN_DIMENSION))
        fused.depth = float(max(getattr(lidar_obs, "depth", 0.0), self.MIN_DIMENSION))
        fused.num_points = int(getattr(lidar_obs, "num_points", 0))
        fused.class_id = -1
        fused.class_name = "unknown"
        fused.confidence = 0.0
        return fused

    def _fuse_image_only(self, image_obs: ImageObstacle) -> FusedObstacle:
        fused = FusedObstacle()
        fused.has_lidar_data = False
        fused.has_image_data = True
        fused.association_distance = -1.0

        fused.centroid = Point(
            x=float(image_obs.centroid.x),
            y=float(image_obs.centroid.y),
            z=float(image_obs.centroid.z),
        )
        fused.width = float(max(getattr(image_obs, "width", 0.0), self.MIN_DIMENSION))
        fused.height = float(max(getattr(image_obs, "height", 0.0), self.MIN_DIMENSION))
        fused.depth = float(self.MIN_DIMENSION)
        fused.num_points = 0
        fused.class_id = int(getattr(image_obs, "class_id", -1))
        fused.class_name = str(getattr(image_obs, "class_name", "unknown"))
        fused.confidence = float(getattr(image_obs, "confidence", 0.0))
        return fused

    def _assign_or_create_track(self, centroid: np.ndarray, now_ts: float) -> int:
        """
        Reasigna a un track existente si está cerca, sino crea uno nuevo.
        Esto mantiene ids persistentes por corto tiempo sin coste elevado.
        """
        if not self._tracks:
            self._last_fused_id += 1
            return self._last_fused_id

        # buscar track más cercano
        track_ids = list(self._tracks.keys())
        centroids = np.array([self._tracks[tid]["centroid"] for tid in track_ids])
        diffs = centroids - centroid
        dists = np.linalg.norm(diffs, axis=1)
        min_idx = int(np.argmin(dists))
        if dists[min_idx] <= self.reid_distance:
            return track_ids[min_idx]
        # sino nuevo id
        self._last_fused_id += 1
        return self._last_fused_id

    def _prune_tracks(self, now_ts: float) -> None:
        """Elimina tracks que no se han visto en track_max_age segundos."""
        to_delete = list()
        for tid, info in self._tracks.items():
            if now_ts - info["last_seen"] > self.track_max_age:
                to_delete.append(tid)
        for tid in to_delete:
            del self._tracks[tid]

    def _publish_empty_state(self, header: Header):
        msg = FusedObstacleArray()
        msg.header = header
        self._fused_pub.publish(msg)
        self._marker_pub.publish(MarkerArray())

    def _create_marker(self, obs: FusedObstacle, header: Header) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = "fused_obstacles"
        marker.id = int(obs.id)
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position = obs.centroid
        marker.pose.orientation.w = 1.0

        marker.scale.x = max(obs.width, self.MIN_DIMENSION)
        marker.scale.y = max(obs.depth, self.MIN_DIMENSION)
        marker.scale.z = max(obs.height, self.MIN_DIMENSION)

        marker.color.a = 0.6
        if obs.has_lidar_data and obs.has_image_data:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        elif obs.has_lidar_data:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        else:
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0

        marker.lifetime.sec = self.MARKER_LIFETIME_SEC
        marker.lifetime.nanosec = 0
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleFusion()

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
