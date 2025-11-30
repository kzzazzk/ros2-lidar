from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="lidar_detection_pkg",
                executable="lidar_object_detector",
                name="lidar_object_detector",
                output="screen",
            ),
            Node(
                package="lidar_detection_pkg",
                executable="image_obstacle_detector",
                name="image_obstacle_detector",
                output="screen",
            ),
            Node(
                package="lidar_detection_pkg",
                executable="fusion_data",
                name="fusion_data",
                output="screen",
            ),
            Node(
                package="lidar_detection_pkg",
                executable="data_snapshot_server",
                name="data_snapshot_server",
                output="screen",
            ),
        ]
    )
