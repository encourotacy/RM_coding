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

################## 检查 ROS 环境 ##################
find_package(fastcdr QUIET)
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(rosidl_typesupport_cpp QUIET)
find_package(sp_msgs QUIET)
find_package(rm_interfaces QUIET)

set(_ros2_core_ready TRUE)
if(NOT ament_cmake_FOUND OR NOT rclcpp_FOUND OR NOT std_msgs_FOUND)
    set(_ros2_core_ready FALSE)
endif()

if(_ros2_core_ready)
    message(STATUS "ROS2 core environment found, enabling optional ROS2 IO targets.")
else()
    message(STATUS "ROS2 core environment not found, skipping optional ROS2 IO targets.")
endif()

if(_ros2_core_ready)
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
endif()

set(_sentry_special_missing_deps "")
# [[cyclonedds]] typed messages need rm_interfaces instead of fastcdr
if(NOT rm_interfaces_FOUND)
    list(APPEND _sentry_special_missing_deps "rm_interfaces")
endif()
if(NOT ament_cmake_FOUND)
    list(APPEND _sentry_special_missing_deps "ament_cmake")
endif()
if(NOT rclcpp_FOUND)
    list(APPEND _sentry_special_missing_deps "rclcpp")
endif()
if(NOT std_msgs_FOUND)
    list(APPEND _sentry_special_missing_deps "std_msgs")
endif()

list(JOIN _sentry_special_missing_deps ", " _sentry_special_missing_deps_str)

if(SENTRY_SPECIAL_MODE STREQUAL "OFF")
    message(STATUS "SENTRY_SPECIAL_MODE=OFF, skipping io_ros2_gimbal target.")
elseif(SENTRY_SPECIAL_MODE STREQUAL "ON")
    if(_sentry_special_missing_deps)
        message(FATAL_ERROR "SENTRY_SPECIAL_MODE=ON but io_ros2_gimbal dependencies are missing: ${_sentry_special_missing_deps_str}")
    endif()
    add_library(io_ros2_gimbal STATIC
        ros2/ros2_gimbal.cpp
    )
    ament_target_dependencies(io_ros2_gimbal rclcpp std_msgs rm_interfaces)
    # [[cyclonedds]] fastcdr no longer needed for typed messages
    # target_link_libraries(io_ros2_gimbal fastcdr)
    message(STATUS "Enabled io_ros2_gimbal target (SENTRY_SPECIAL_MODE=ON).")
else()
    if(_sentry_special_missing_deps)
        message(STATUS "SENTRY_SPECIAL_MODE=AUTO, skipping io_ros2_gimbal target because dependencies are missing: ${_sentry_special_missing_deps_str}")
    else()
        add_library(io_ros2_gimbal STATIC
            ros2/ros2_gimbal.cpp
        )
        ament_target_dependencies(io_ros2_gimbal rclcpp std_msgs rm_interfaces)
        # [[cyclonedds]] fastcdr no longer needed for typed messages
        # target_link_libraries(io_ros2_gimbal fastcdr)
        message(STATUS "Enabled io_ros2_gimbal target (SENTRY_SPECIAL_MODE=AUTO).")
    endif()
endif()
