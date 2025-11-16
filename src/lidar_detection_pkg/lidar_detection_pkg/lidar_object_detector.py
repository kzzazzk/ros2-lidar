#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from lidar_interfaces.msg import PointCloudObstacle
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker, MarkerArray

class LidarObstacleDetector(Node):
    def __init__(self):
        super().__init__('lidar_obstacle_detector')
        self.subscription = self.create_subscription(
            PointCloud2, '/ouster/points', self.callback, 10)

        self.publisher = self.create_publisher(PointCloudObstacle, '/obstacles', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/obstacle_markers', 10)

        self.declare_parameter('eps', 0.55)
        self.declare_parameter('min_points', 45)
        self.get_logger().info('LidarObstacleDetector node started.')

    def callback(self, msg: PointCloud2):
        # --- convert PointCloud2 → Nx3 float array robustly ---
        points_iter = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        # Use fromiter for memory efficiency, then reshape
        flat = np.fromiter((coord for p in points_iter for coord in p), dtype=np.float32)
        if flat.size == 0:
            return
        try:
            points = flat.reshape((-1, 3))
        except ValueError:
            # fallback safe conversion
            points_iter = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            points = np.array([[p[0], p[1], p[2]] for p in points_iter], dtype=np.float32)
            if points.size == 0:
                return

        # Downsample for speed
        points = points[::5]
        if points.shape[0] == 0:
            return

        # Basic ground filter (adjust threshold to your sensor)
        points = points[points[:, 2] > -2]

        MAX_DISTANCE = 5.5

        distances = np.linalg.norm(points[:, :2], axis=1)
        mask_range = distances < MAX_DISTANCE

        angles = np.arctan2(points[:, 1], points[:, 0])
        mask_fov = (angles >= -np.pi/2) & (angles <= np.pi/2)

        points = points[mask_range & mask_fov]

        if points.shape[0] == 0:
            return

        # Clustering parameters
        eps = float(self.get_parameter('eps').value)
        min_pts = int(self.get_parameter('min_points').value)

        # If too few points, skip clustering
        if points.shape[0] < max(2, min_pts):
            return

        clustering = DBSCAN(eps=eps, min_samples=min_pts).fit(points)
        labels = clustering.labels_
        unique_labels = np.unique(labels)

        # Frame: reuse incoming frame_id, but fallback to a sensible default
        frame_id = msg.header.frame_id if msg.header.frame_id else "os_sensor"

        id_counter = 0
        marker_array = MarkerArray()

        for label in unique_labels:
            if label == -1:
                continue  # noise

            cluster = points[labels == label]
            if cluster.size == 0:
                continue

            centroid = np.mean(cluster, axis=0)
            min_bounds = np.min(cluster, axis=0)
            max_bounds = np.max(cluster, axis=0)
            dims = max_bounds - min_bounds

            # Publish Obstacle message
            msg_out = PointCloudObstacle()
            msg_out.header = Header()
            msg_out.header.stamp = self.get_clock().now().to_msg()
            msg_out.header.frame_id = frame_id
            msg_out.id = id_counter
            msg_out.centroid = Point(x=float(centroid[0]),
                                     y=float(centroid[1]),
                                     z=float(centroid[2]))
            msg_out.width = float(max(dims[0], 0.001))
            msg_out.height = float(max(dims[2], 0.001))
            msg_out.depth = float(max(dims[1], 0.001))
            msg_out.num_points = int(len(cluster))
            self.publisher.publish(msg_out)

            # Build visualization marker
            marker = Marker()
            marker.header = msg_out.header
            marker.ns = "obstacles"
            marker.id = id_counter
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position = msg_out.centroid
            # Ensure a valid orientation (identity quaternion)
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = msg_out.width
            marker.scale.y = msg_out.depth
            marker.scale.z = msg_out.height
            marker.color.a = 0.5
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            # Lifetime so old markers disappear automatically
            marker.lifetime.sec = 1
            marker.lifetime.nanosec = 0

            marker_array.markers.append(marker)
            id_counter += 1

        # publish markers (empty array if none found)
        self.marker_pub.publish(marker_array)
        self.get_logger().info(f"Published {id_counter} obstacles and markers")

def main(args=None):
    rclpy.init(args=args)
    node = LidarObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
