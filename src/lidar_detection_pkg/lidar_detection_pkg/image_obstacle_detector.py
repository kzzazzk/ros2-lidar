#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from lidar_interfaces.msg import Obstacle
from cv_bridge import CvBridge
import cv2
from importlib.metadata import version as pkg_version
import torch


class ImageObstacleDetector(Node):
    def __init__(self):
        super().__init__('image_obstacle_detector')
        
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('min_area', 500)
        
        self.subscription = self.create_subscription(
            Image, '/my_camera/pylon_ros2_camera_node/image_raw', self.callback, 10)
        
        self.publisher = self.create_publisher(Obstacle, '/obstacles', 10)
        self.image_overlay_pub = self.create_publisher(Image, '/camera/image_with_markers', 10)
        
        self.bridge = CvBridge()
        
        self.get_logger().info('ImageObstacleDetector node started.')
        
        try:
            version = pkg_version("ultralytics")
        except Exception:
            version = "0.0.0"
        
        self.get_logger().info(f"Detected Ultralytics version: {version}")
        major_version = int(version.split('.')[0])
        
        try:
            if major_version >= 8:
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')
                self.model_type = "v8"
                self.get_logger().info("Loaded YOLOv8 model successfully.")
            else:
                raise ImportError
        except Exception:
            try:
                from yolov5 import YOLOv5
                self.model = YOLOv5("yolov5s.pt", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                self.model_type = "v5"
                self.get_logger().info("Loaded YOLOv5 model successfully.")
            except Exception as e:
                self.get_logger().error(f"Failed to load YOLO model: {e}")
                raise
        
        self.get_logger().info(f"Confidence threshold: {self.get_parameter('confidence_threshold').value}")
        self.get_logger().info(f"Minimum area: {self.get_parameter('min_area').value}")

    def callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return
        
        frame_id = msg.header.frame_id if msg.header.frame_id else "camera"
        display = cv_image.copy()
        
        detections = self.run_inference(cv_image)
        
        id_counter = 0
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            confidence = det['confidence']
            class_id = det['class_id']
            class_name = det['class_name']
            
            if confidence < self.get_parameter('confidence_threshold').value:
                continue
            
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            if area < self.get_parameter('min_area').value:
                continue
            
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            
            msg_out = Obstacle()
            msg_out.header = Header()
            msg_out.header.stamp = self.get_clock().now().to_msg()
            msg_out.header.frame_id = frame_id
            msg_out.id = id_counter
            msg_out.centroid = Point(x=float(cx), y=float(cy), z=0.0)
            msg_out.width = float(w)
            msg_out.height = float(h)
            msg_out.depth = 0.0
            msg_out.num_points = int(area)
            self.publisher.publish(msg_out)
            
            color = self.get_color_for_class(class_id)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name} {confidence:.2f}"
            
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(display, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(display, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.putText(display, f"ID:{id_counter}", (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            id_counter += 1
        
        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(display, encoding='bgr8')
            overlay_msg.header.stamp = msg.header.stamp
            overlay_msg.header.frame_id = frame_id
            self.image_overlay_pub.publish(overlay_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish overlay image: {e}")
        
        if id_counter > 0:
            self.get_logger().info(f"Published {id_counter} obstacles.", throttle_duration_sec=1.0)
    
    def run_inference(self, cv_image):
        detections = []
        
        if self.model_type == "v8":
            results = self.model(cv_image, verbose=False)
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                detections.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        else:
            results = self.model.predict(cv_image)
            predictions = results.pred[0].cpu().numpy()
            names = results.names
            
            for pred in predictions:
                x1, y1, x2, y2, confidence, class_id = pred[:6]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                class_id = int(class_id)
                class_name = names[class_id]
                
                detections.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': float(confidence),
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        return detections
    
    def get_color_for_class(self, class_id):
        """Return a consistent color for each class."""
        colors = {
            0: (255, 0, 0),      # person - red
            1: (0, 255, 255),    # bicycle - cyan
            2: (0, 255, 0),      # car - green
            3: (255, 0, 255),    # motorcycle - magenta
            5: (255, 255, 0),    # bus - yellow
            7: (0, 128, 255),    # truck - orange
            9: (0, 0, 255),      # traffic light - blue
            11: (128, 0, 128),   # stop sign - purple
        }
        return colors.get(class_id, (0, 255, 0))  # default green


def main(args=None):
    rclpy.init(args=args)
    node = ImageObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()