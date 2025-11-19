#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from torch import cuda
from ultralytics import YOLO

from lidar_interfaces.msg import ImageObstacle, ImageObstacleArray


class ImageObstacleDetector(Node):

    DEFAULT_CONFIDENCE: float = 0.5
    DEFAULT_MIN_AREA: int = 500
    DEFAULT_PUBLISH_OVERLAY: bool = True
    DEFAULT_MODEL_PATH: str = "yolov8n.pt"
    DEFAULT_FRAME_SKIP: int = 1
    DEFAULT_LOG_STATS_INTERVAL: int = 100
    YOLO_TARGET_SIZE: int = 640

    # OpenCV uses BGR format
    DEFAULT_COLOR: tuple[int, int, int] = (128, 128, 128)  # gray
    COCO_INFO: dict[int, tuple[str, tuple[int, int, int]]] = {
        0: ("person", (255, 0, 0)),  # red
        1: ("bicycle", (0, 255, 255)),  # cyan
        2: ("car", (0, 255, 0)),  # green
        3: ("motorcycle", (255, 0, 255)),  # magenta
        5: ("bus", (255, 255, 0)),  # yellow
        7: ("truck", (0, 128, 255)),  # orange
        9: ("traffic light", (0, 0, 255)),  # blue
        11: ("stop sign", (128, 0, 128)),  # purple
    }
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 2
    BOX_THICKNESS = 2

    def __init__(self):
        super().__init__("image_obstacle_detector")

        self._declare_parameters()
        self._load_parameters()
        self._setup_publishers()
        self._setup_subscriptions()

        self._bridge = CvBridge()
        self._load_model()

        self._frame_count = 0
        self._processed_count = 0
        self._total_detections = 0
        self._inference_times = []

        self.add_on_set_parameters_callback(self._parameter_callback)

        self._use_cuda = cuda.is_available()
        self.get_logger().info(
            f"ImageObstacleDetector started - "
            f"Confidence: {self.confidence_threshold}, "
            f"Min area: {self.min_area}, "
            f"CUDA: {self._use_cuda}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("confidence_threshold", self.DEFAULT_CONFIDENCE)
        self.declare_parameter("min_area", self.DEFAULT_MIN_AREA)
        self.declare_parameter("model_path", self.DEFAULT_MODEL_PATH)
        self.declare_parameter("publish_overlay", self.DEFAULT_PUBLISH_OVERLAY)
        self.declare_parameter("frame_skip", self.DEFAULT_FRAME_SKIP)
        self.declare_parameter("log_stats_interval", self.DEFAULT_LOG_STATS_INTERVAL)

    def _load_parameters(self) -> None:
        self.confidence_threshold = (
            self.get_parameter("confidence_threshold")
            .get_parameter_value()
            .double_value
        )
        self.min_area = (
            self.get_parameter("min_area").get_parameter_value().integer_value
        )
        self.model_path = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        self.publish_overlay = (
            self.get_parameter("publish_overlay").get_parameter_value().bool_value
        )
        self.frame_skip = (
            self.get_parameter("frame_skip").get_parameter_value().integer_value
        )
        self.log_stats_interval = (
            self.get_parameter("log_stats_interval").get_parameter_value().integer_value
        )

    def _setup_publishers(self) -> None:
        self._obstacle_publisher = self.create_publisher(
            ImageObstacleArray, "/image/obstacles", 10
        )
        self._overlay_publisher = self.create_publisher(
            Image, "/image/obstacles/overlay", 10
        )

    def _setup_subscriptions(self) -> None:
        self._image_subscription = self.create_subscription(
            Image,
            "/my_camera/pylon_ros2_camera_node/image_raw",
            self._image_callback,
            10,
        )

    def _parameter_callback(self, params: list[Parameter]) -> SetParametersResult:
        for param in params:
            if param.name == "confidence_threshold" and 0.0 <= param.value <= 1.0:
                self.confidence_threshold = float(param.value)
                self.get_logger().info(f"Updated confidence_threshold: {param.value}")
            elif param.name == "min_area" and param.value >= 0:
                self.min_area = int(param.value)
                self.get_logger().info(f"Updated min_area: {param.value}")
            elif param.name == "model_path" and isinstance(param.value, str):
                old_path = self.model_path
                self.model_path = str(param.value)
                try:
                    self._load_model()
                    self.get_logger().info(f"Updated model_path: {param.value}")
                except Exception as e:
                    self.model_path = old_path
                    return SetParametersResult(
                        successful=False, reason=f"Failed to load model: {e}"
                    )
            elif param.name == "publish_overlay" and isinstance(param.value, bool):
                self.publish_overlay = bool(param.value)
                self.get_logger().info(f"Updated publish_overlay: {param.value}")
            elif param.name == "frame_skip" and param.value >= 1:
                self.frame_skip = int(param.value)
                self.get_logger().info(f"Updated frame_skip: {param.value}")
            elif param.name == "log_stats_interval" and param.value >= 0:
                self.log_stats_interval = int(param.value)
                self.get_logger().info(f"Updated log_stats_interval: {param.value}")
            else:
                return SetParametersResult(
                    successful=False,
                    reason=f"Invalid value for parameter '{param.name}': {param.value}",
                )
        return SetParametersResult(successful=True)

    def _load_model(self) -> None:
        try:
            self.model = YOLO(self.model_path)
            self.class_names = self.model.names
            self.get_logger().info(f"Loaded YOLOv8 model: {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            raise

    def _image_callback(self, msg: Image) -> None:
        self._frame_count += 1

        if self._frame_count % self.frame_skip != 0:
            return

        self._processed_count += 1

        cv_image = self._ros_to_cv2(msg)
        if cv_image is None:
            return

        preprocessed, scale, pad_w, pad_h = self._preprocess_image(cv_image)
        if preprocessed is None:
            return

        start_time = self.get_clock().now()
        results = self._run_inference(preprocessed)
        inference_time = (self.get_clock().now() - start_time).nanoseconds / 1e6
        self._inference_times.append(inference_time)

        detections = self._extract_detections(
            results, scale, pad_w, pad_h, cv_image.shape
        )

        if detections:
            self._publish_obstacles(detections, msg.header)

        if self.publish_overlay:
            overlay = self._create_overlay(cv_image, detections)
            self._publish_overlay(overlay, msg.header)

        if (
            self.log_stats_interval > 0
            and self._processed_count % self.log_stats_interval == 0
        ):
            self._log_statistics()

    def _ros_to_cv2(self, msg: Image) -> np.ndarray | None:
        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            if cv_image.ndim == 2:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            elif cv_image.ndim != 3 or cv_image.shape[2] != 3:
                self.get_logger().error(f"Unexpected image format: {cv_image.shape}")
                return None

            if cv_image.shape[0] == 0 or cv_image.shape[1] == 0:
                self.get_logger().error("Received empty image")
                return None

            return cv_image
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS image: {e}")
            return None

    def _preprocess_image(
        self, image: np.ndarray
    ) -> tuple[np.ndarray | None, float, int, int]:
        try:
            h, w = image.shape[:2]
            target = self.YOLO_TARGET_SIZE

            scale = target / max(h, w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            pad_w = (target - new_w) // 2
            pad_h = (target - new_h) // 2

            padded = cv2.copyMakeBorder(
                resized,
                top=pad_h,
                bottom=target - new_h - pad_h,
                left=pad_w,
                right=target - new_w - pad_w,
                borderType=cv2.BORDER_CONSTANT,
                value=[114, 114, 114],
            )

            return padded, scale, pad_w, pad_h
        except Exception as e:
            self.get_logger().error(f"Failed to preprocess image: {e}")
            return None, 1.0, 0, 0

    def _run_inference(self, image: np.ndarray):
        try:
            results = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                imgsz=self.YOLO_TARGET_SIZE,
                half=True if self._use_cuda else False,
                device="cuda" if self._use_cuda else "cpu",
                classes=list(self.COCO_INFO.keys()),
                augment=False,
                verbose=False,
            )
            return results
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return None

    def _extract_detections(
        self, results, scale: float, pad_w: int, pad_h: int, orig_shape: tuple
    ) -> list[dict]:
        if not results or len(results) == 0:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections = []
        orig_h, orig_w = orig_shape[:2]

        for box in boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue

            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Transform to original image coordinates
            x1 = (x1 - pad_w) / scale
            x2 = (x2 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            y2 = (y2 - pad_h) / scale

            # Clamp and ensure valid box
            x1 = max(0.0, min(x1, orig_w - 1))
            x2 = max(0.0, min(x2, orig_w - 1))
            y1 = max(0.0, min(y1, orig_h - 1))
            y2 = max(0.0, min(y2, orig_h - 1))

            # Ensure x1 < x2 and y1 < y2
            if x1 >= x2 or y1 >= y2:
                continue

            width = x2 - x1
            height = y2 - y1
            area = width * height

            if area < self.min_area:
                continue

            cx = x1 + width / 2.0
            cy = y1 + height / 2.0

            class_name = self.COCO_INFO.get(cls, ("unknown", self.DEFAULT_COLOR))[0]

            detections.append(
                {
                    "confidence": conf,
                    "class_id": cls,
                    "class_name": class_name,
                    "centroid": (cx, cy),
                    "width": width,
                    "height": height,
                    "area": area,
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                }
            )

        return detections

    def _publish_obstacles(self, detections: list[dict], header: Header) -> None:
        array_msg = ImageObstacleArray()
        array_msg.header = header

        for i, det in enumerate(detections):
            obs = ImageObstacle()
            obs.id = i
            obs.class_id = det["class_id"]
            obs.class_name = det["class_name"]
            obs.confidence = det["confidence"]
            obs.centroid.x = float(det["centroid"][0])
            obs.centroid.y = float(det["centroid"][1])
            obs.centroid.z = 0.0
            obs.width = float(det["width"])
            obs.height = float(det["height"])
            obs.area = float(det["area"])

            array_msg.obstacles.append(obs)

        self._obstacle_publisher.publish(array_msg)
        self._total_detections += len(detections)

    def _create_overlay(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        overlay = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            class_id = det["class_id"]
            conf = det["confidence"]

            color = self.COCO_INFO.get(class_id, ("", self.DEFAULT_COLOR))[1]

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, self.BOX_THICKNESS)

            label = f"{det['class_name']} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS
            )

            # Background for text
            cv2.rectangle(
                overlay,
                (x1, y1 - label_h - 10),
                (x1 + label_w + 5, y1),
                color,
                -1,
            )

            cv2.putText(
                overlay,
                label,
                (x1 + 2, y1 - 5),
                self.FONT,
                self.FONT_SCALE,
                (0, 0, 0),  # Black text
                self.FONT_THICKNESS,
            )

        return overlay

    def _publish_overlay(self, overlay: np.ndarray, header: Header) -> None:
        try:
            overlay_msg = self._bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            overlay_msg.header = header
            self._overlay_publisher.publish(overlay_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish overlay: {e}")

    def _log_statistics(self) -> None:
        if len(self._inference_times) == 0:
            return

        avg_inference = sum(self._inference_times) / len(self._inference_times)
        avg_detections = (
            self._total_detections / self._processed_count
            if self._processed_count > 0
            else 0
        )

        self.get_logger().info(
            f"Processed {self._processed_count}/{self._frame_count} frames | "
            f"Detections: {self._total_detections} "
            f"({avg_detections:.2f}/frame) | "
            f"Avg inference: {avg_inference:.2f}ms"
        )

    def get_stats(self) -> dict:
        return {
            "total_frames": self._frame_count,
            "processed_frames": self._processed_count,
            "total_detections": self._total_detections,
            "avg_detections_per_frame": (
                self._total_detections / max(self._processed_count, 1)
            ),
            "avg_inference_time": (
                sum(self._inference_times) / len(self._inference_times)
                if self._inference_times
                else 0.0
            ),
        }


def main(args=None):
    rclpy.init(args=args)
    node = ImageObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
