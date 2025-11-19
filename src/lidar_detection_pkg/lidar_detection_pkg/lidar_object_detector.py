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

from lidar_interfaces.msg import PointCloudObstacle


# TODO: Add docstrings to all the methods and the class constructor
class LidarObstacleDetector(Node):

    DEFAULT_EPS = 0.55
    DEFAULT_MIN_POINTS = 45
    DEFAULT_DOWNSAMPLING = 5
    DEFAULT_MAX_DISTANCE = 5.5
    GROUND_THRESHOLD = -2.0  # TODO: convert to parameter ¿?
    FOV_MIN = -np.pi / 2  # TODO: convert to parameter ¿?
    FOV_MAX = np.pi / 2  # TODO: convert to parameter ¿?
    MARKER_LIFETIME_SEC = 1
    MIN_DIMENSION = 0.001

    def __init__(self):
        super().__init__("lidar_obstacle_detector")

        # Declaration and loading of the parameters
        self._declare_parameter()
        self._load_parameters()

        # Setup of the communications (publishers first, then subscriptions per ROS2 convention)
        self._setup_publishers()
        self._setup_subscriptions()

        # Register parameter callback
        self.add_on_set_parameters_callback(self._parameter_callback)

        self.get_logger().info("LidarObstacleDetector node started.")

    def _declare_parameter(self):
        self.declare_parameter("eps", self.DEFAULT_EPS)
        self.declare_parameter("min_points", self.DEFAULT_MIN_POINTS)
        self.declare_parameter("downsampling_factor", self.DEFAULT_DOWNSAMPLING)
        self.declare_parameter("max_distance", self.DEFAULT_MAX_DISTANCE)

    def _load_parameters(self):
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
        self._obstacle_pub = self.create_publisher(
            PointCloudObstacle, "/obstacles", 10
        )  # TODO: Change to send a list of `PointCloudObstacle` to the fusion node
        self._marker_pub = self.create_publisher(MarkerArray, "/obstacle_markers", 10)

    def _setup_subscriptions(self):
        self._pointcloud_sub = self.create_subscription(
            PointCloud2, "/ouster/points", self._pointcloud_callback, 10
        )

    def _parameter_callback(self, params: list[Parameter]) -> SetParametersResult:
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
        points = self._convert_pointcloud_to_array(msg)
        if points is None or points.shape[0] == 0:
            return

        points = self._preprocess_points(points)
        if points is None or points.shape[0] < max(2, self.min_points):
            return

        labels = self._perform_clustering(points)

        frame_id = msg.header.frame_id if msg.header.frame_id else "os_sensor"
        timestamp = msg.header.stamp

        self._publish_obstacles_and_markers(points, labels, frame_id, timestamp)

    def _convert_pointcloud_to_array(self, msg: PointCloud2) -> np.ndarray | None:
        try:
            points_iter = pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )
            flat = np.fromiter(
                (coord for p in points_iter for coord in p), dtype=np.float32
            )

            if flat.size == 0:
                return None

            points = flat.reshape((-1, 3))
            return points

        except (ValueError, RuntimeError) as e:
            self.get_logger().warn(
                f"Failed to convert point cloud efficiently: {e}. Using fallback method."
            )

            # Fallback to safer but slower conversion
            try:
                points_iter = pc2.read_points(
                    msg, field_names=("x", "y", "z"), skip_nans=True
                )
                points = np.asarray(
                    [[p[0], p[1], p[2]] for p in points_iter], dtype=np.float32
                )
                return points if points.size > 0 else None
            except Exception as e:
                self.get_logger().error(f"PointCloud conversion failed: {e}")
                return None

    def _preprocess_points(self, points: np.ndarray) -> np.ndarray | None:
        # Ground filtering mask
        height_mask = points[:, 2] > self.GROUND_THRESHOLD

        # Angles and distances mask
        distances = np.linalg.norm(points[:, :2], axis=1)
        angles = np.arctan2(points[:, 1], points[:, 0])
        range_mask = distances < self.max_distance
        fov_mask = (angles >= self.FOV_MIN) & (angles <= self.FOV_MAX)

        final_mask = height_mask & range_mask & fov_mask
        points = points[final_mask]

        # Downsampling
        if self.downsampling_factor > 1:
            points = points[:: self.downsampling_factor]
            if points.shape[0] == 0:
                return None

        return points if points.shape[0] > 0 else None

    def _perform_clustering(self, points: np.ndarray) -> np.ndarray:
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_points, n_jobs=-1).fit(
            points
        )
        return clustering.labels_

    def _publish_obstacles_and_markers(
        self, points: np.ndarray, labels: np.ndarray, frame_id: str, timestamp: Time
    ) -> None:
        unique_labels = np.unique(labels[labels != -1])
        if unique_labels.size == 0:
            self._marker_pub.publish(MarkerArray())

        marker_array = MarkerArray()

        header = Header()
        header.frame_id = frame_id
        header.stamp = timestamp

        for obstacle_id, label in enumerate(unique_labels):
            cluster_mask = labels == label
            cluster = points[cluster_mask]

            centroid, dimensions = self._compute_cluster_properties(cluster)

            obstacle_msg = self._create_obstacle_message(
                obstacle_id, centroid, dimensions, np.sum(cluster_mask), header
            )
            marker = self._create_visualization_marker(obstacle_msg, obstacle_id)
            marker_array.markers.append(marker)

        self._marker_pub.publish(marker_array)

        if unique_labels.size > 0:
            self.get_logger().info(
                f"Published {unique_labels.size} obstacles and markers",
                throttle_duration_sec=1.0,
            )

    def _compute_cluster_properties(
        self,
        cluster: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        msg_out = PointCloudObstacle()
        msg_out.header = header
        msg_out.id = obstacle_id
        msg_out.centroid = Point(
            x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2])
        )
        msg_out.width = float(max(dimensions[0], self.MIN_DIMENSION))
        msg_out.height = float(max(dimensions[2], self.MIN_DIMENSION))
        msg_out.depth = float(max(dimensions[1], self.MIN_DIMENSION))
        msg_out.num_points = int(num_points)
        return msg_out

    def _create_visualization_marker(
        self, obstacle_msg: PointCloudObstacle, obstacle_id: int
    ) -> Marker:
        marker = Marker()
        marker.header = obstacle_msg.header
        marker.ns = "obstacles"
        marker.id = obstacle_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Position and orientation
        marker.pose.position = obstacle_msg.centroid
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Dimensions
        marker.scale.x = obstacle_msg.width
        marker.scale.y = obstacle_msg.depth
        marker.scale.z = obstacle_msg.height

        # Color (semi-transparent red)
        marker.color.a = 0.5
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        # Lifetime
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
    node.destroy_node()
    # rclpy.shutdown()  # <------------------- Falla si lo dejo puesto


if __name__ == "__main__":
    main()
