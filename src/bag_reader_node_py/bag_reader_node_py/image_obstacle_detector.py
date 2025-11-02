#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from lidar_interfaces.msg import Obstacle
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
from importlib.metadata import version as pkg_version
import torch


class ImageObstacleDetector(Node):
    def __init__(self):
        super().__init__('image_obstacle_detector')
        self.subscription = self.create_subscription(
            Image, '/my_camera/pylon_ros2_camera_node/image_raw', self.callback, 10)

        self.publisher = self.create_publisher(Obstacle, '/obstacles', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/obstacle_markers', 10)
        self.image_overlay_pub = self.create_publisher(Image, '/camera/image_with_markers', 10)

        self.bridge = CvBridge()
        self.get_logger().info('ImageObstacleDetector node started.')

        # --- Detect Ultralytics version ---
        try:
            version = pkg_version("ultralytics")
        except Exception:
            version = "0.0.0"

        self.get_logger().info(f"Detected Ultralytics version: {version}")
        major_version = int(version.split('.')[0])

        # --- Try YOLOv8 ---
        try:
            if major_version >= 8:
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')
                self.model_type = "v8"
                self.get_logger().info("Loaded YOLOv8 model successfully.")
            else:
                raise ImportError
        except Exception:
            # --- Try YOLOv5 package ---
            from yolov5 import YOLOv5
            self.model = YOLOv5("yolov5s.pt", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.model_type = "v5"
            self.get_logger().info("Loaded YOLOv5 model (package) successfully.")

    def callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        frame_id = msg.header.frame_id if msg.header.frame_id else "camera"
        display = cv_image.copy()

        # --- Run YOLO inference ---
        if self.model_type == "v8":
            results = self.model(cv_image)
            detections = results[0].boxes
            boxes = [box.xyxy[0].tolist() for box in detections]
        else:
            # YOLOv5 or torch.hub
            results = self.model(cv_image)
            boxes = results.xyxy[0].cpu().numpy()

        id_counter = 0
        marker_array = MarkerArray()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            w = x2 - x1
            h = y2 - y1
            if w * h < 50:
                continue

            cx = x1 + w / 2
            cy = y1 + h / 2

            msg_out = Obstacle()
            msg_out.header = Header()
            msg_out.header.stamp = self.get_clock().now().to_msg()
            msg_out.header.frame_id = frame_id
            msg_out.id = id_counter
            msg_out.centroid = Point(x=float(cx), y=float(cy), z=0.0)
            msg_out.width = float(w)
            msg_out.height = float(h)
            msg_out.depth = 0.01
            msg_out.num_points = w * h
            self.publisher.publish(msg_out)

            marker = Marker()
            marker.header = msg_out.header
            marker.ns = "obstacles"
            marker.id = id_counter
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position = msg_out.centroid
            marker.pose.orientation.w = 1.0
            marker.scale.x = msg_out.width
            marker.scale.y = msg_out.depth
            marker.scale.z = msg_out.height
            marker.color.a = 0.5
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.lifetime.sec = 1
            marker_array.markers.append(marker)

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, str(id_counter), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            id_counter += 1

        self.marker_pub.publish(marker_array)

        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(display, encoding='bgr8')
            overlay_msg.header.stamp = self.get_clock().now().to_msg()
            overlay_msg.header.frame_id = frame_id
            self.image_overlay_pub.publish(overlay_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish overlay image: {e}")

        self.get_logger().info(f"Published {id_counter} obstacles.")


def main(args=None):
    rclpy.init(args=args)
    node = ImageObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
