cmake_minimum_required(VERSION 3.8)
project(lidar_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
# Necesario para builtin_interfaces/Time usado en el .srv
find_package(builtin_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)

# Definimos los mensajes y el servicio
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PointCloudObstacle.msg"
  "msg/ImageObstacle.msg"
  "msg/ImageObstacleArray.msg"
  "msg/PointCloudObstacleArray.msg"
  # SERVICIO
  "srv/GetFusedSnapshot.srv"
  DEPENDENCIES geometry_msgs builtin_interfaces sensor_msgs
)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()