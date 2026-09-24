cmake_minimum_required(VERSION 3.16)

find_package(yaml-cpp REQUIRED)

add_subdirectory(serial)

# 创建目标 io
add_library(io STATIC 
    hikrobot/hikrobot.cpp    
    mindvision/mindvision.cpp  
    usbcamera/usbcamera.cpp  
    camera.cpp
    cboard.cpp
    dm_imu/dm_imu.cpp
    gimbal/gimbal.cpp
    ovgimbal.cpp
)

# hikrobot
target_include_directories(io PUBLIC hikrobot/include)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(io PUBLIC hikrobot/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(io PUBLIC hikrobot/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

# mindvision
target_include_directories(io PUBLIC mindvision/include)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(io PUBLIC mindvision/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(io PUBLIC mindvision/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

target_link_libraries(io MvCameraControl MVSDK usb-1.0 yaml-cpp serial)

################## ROS2 Humble ##################
find_package(fastcdr REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rm_interfaces REQUIRED)
find_package(rosidl_typesupport_cpp QUIET)
find_package(sp_msgs QUIET)

if(rosidl_typesupport_cpp_FOUND AND sp_msgs_FOUND)
    add_library(io_ros2_nav STATIC
        ros2/publish2nav.cpp
        ros2/ros2.cpp
        ros2/subscribe2nav.cpp
    )
    ament_target_dependencies(io_ros2_nav rclcpp std_msgs sp_msgs)
    message(STATUS "Enabled io_ros2_nav target.")
else()
    message(STATUS "sp_msgs or rosidl_typesupport_cpp not found, skipping io_ros2_nav target.")
endif()

add_library(io_ros2_gimbal STATIC
    ros2/ros2_gimbal.cpp
)
ament_target_dependencies(io_ros2_gimbal rclcpp std_msgs rm_interfaces)
message(STATUS "Enabled io_ros2_gimbal target.")
