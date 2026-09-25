cmake_minimum_required(VERSION 3.16.3)

project(sp_vision)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fastcdr QUIET)
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake/")
find_package(OpenVINO REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

set(SENTRY_SPECIAL_MODE "AUTO" CACHE STRING "Build sentry-specific ROS2 gimbal and ovsentry targets: AUTO, ON, or OFF")
set_property(CACHE SENTRY_SPECIAL_MODE PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${SENTRY_SPECIAL_MODE}" SENTRY_SPECIAL_MODE)
set(SENTRY_SPECIAL_MODE "${SENTRY_SPECIAL_MODE}" CACHE STRING "Build sentry-specific ROS2 gimbal and ovsentry targets: AUTO, ON, or OFF" FORCE)

set(_sentry_special_mode_values AUTO ON OFF)
if(NOT SENTRY_SPECIAL_MODE IN_LIST _sentry_special_mode_values)
    message(FATAL_ERROR "Invalid SENTRY_SPECIAL_MODE='${SENTRY_SPECIAL_MODE}'. Expected one of: AUTO, ON, OFF.")
endif()

add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks/auto_aim)
add_subdirectory(tasks/auto_buff)
add_subdirectory(tasks/omniperception)

# 把同一套依赖的源文件收成一组，目标名仍是源文件名。
function(sp_add_executables)
    cmake_parse_arguments(ARG "" "" "GLOBS;LIBS" ${ARGN})
    file(GLOB _sources CONFIGURE_DEPENDS ${ARG_GLOBS})
    foreach(_source ${_sources})
        get_filename_component(_target "${_source}" NAME_WE)
        add_executable(${_target} ${_source})
        target_link_libraries(${_target} ${ARG_LIBS})
    endforeach()
endfunction()

set(SP_OPENCV_FMT ${OpenCV_LIBS} fmt::fmt)

####################src###################
sp_add_executables(
    GLOBS
        src/standard.cpp
        src/mt_standard.cpp
        src/standard_mpc.cpp
        src/auto_aim_debug_mpc.cpp
        src/mt_auto_aim_debug.cpp
        src/uav.cpp
    LIBS ${SP_OPENCV_FMT} yaml-cpp auto_aim auto_buff tools io
)
sp_add_executables(
    GLOBS src/auto_buff_debug.cpp src/auto_buff_debug_mpc.cpp
    LIBS ${SP_OPENCV_FMT} yaml-cpp auto_buff tools io ${CERES_LIBRARIES}
)
sp_add_executables(
    GLOBS src/uav_debug.cpp src/ovgimbal_*.cpp src/ovsentry_mpc_replay.cpp
    LIBS ${SP_OPENCV_FMT} yaml-cpp auto_aim tools io
)

##################calibration################
sp_add_executables(
    GLOBS calibration/capture.cpp
    LIBS ${SP_OPENCV_FMT} tools io
)
sp_add_executables(
    GLOBS calibration/capture_chessboard.cpp
    LIBS ${SP_OPENCV_FMT} yaml-cpp tools io
)
sp_add_executables(
    GLOBS calibration/calibrate_*.cpp
    LIBS ${SP_OPENCV_FMT} yaml-cpp tools
)
sp_add_executables(
    GLOBS calibration/split_video.cpp
    LIBS ${SP_OPENCV_FMT} tools
)

##################tests##################
sp_add_executables(
    GLOBS
        tests/auto_aim_test.cpp
        tests/camera_detect_test.cpp
        tests/usbcamera_detect_test.cpp
        tests/handeye_test.cpp
        tests/minimum_vision_system.cpp
        tests/gimbal_test.cpp
        tests/planner_test.cpp
        tests/planner_test_offline.cpp
        tests/outpost_auto_aim_logic_test.cpp
        tests/normal_auto_aim_v2_logic_test.cpp
        tests/armor_type_logic_test.cpp
        tests/sentry_mpc_takeover_test.cpp
        tests/planner_sentry_safety_test.cpp
    LIBS ${SP_OPENCV_FMT} yaml-cpp tools io auto_aim
)
sp_add_executables(
    GLOBS tests/auto_buff_test.cpp
    LIBS ${SP_OPENCV_FMT} yaml-cpp auto_buff tools io
)
sp_add_executables(
    GLOBS
        tests/camera_test.cpp
        tests/cboard_test.cpp
        tests/fire_test.cpp
        tests/multi_usbcamera_test.cpp
    LIBS ${SP_OPENCV_FMT} tools io
)
sp_add_executables(
    GLOBS tests/camera_thread_test.cpp
    LIBS ${SP_OPENCV_FMT} auto_aim tools io
)
sp_add_executables(
    GLOBS tests/detector_video_test.cpp
    LIBS ${SP_OPENCV_FMT} yaml-cpp tools auto_aim
)
sp_add_executables(
    GLOBS
        tests/gimbal_response_test.cpp
        tests/usbcamera_test.cpp
        tests/dm_test.cpp
        tests/camera_passthrough_test.cpp
    LIBS ${SP_OPENCV_FMT} yaml-cpp tools io
)
sp_add_executables(
    GLOBS
        tests/pattern_geometry_test.cpp
        tests/gimbal_axis_order_test.cpp
        tests/shooter_logic_test.cpp
        tests/sentry_yaw_control_test.cpp
        tests/sentry_mpc_transform_test.cpp
    LIBS ${SP_OPENCV_FMT} tools
)
sp_add_executables(
    GLOBS tests/ros2_gimbal_big_yaw_lookup_test.cpp
    LIBS ${SP_OPENCV_FMT} tools
)

add_executable(sentry_camera_axis_test tests/sentry_camera_axis_test.cpp)
target_link_libraries(sentry_camera_axis_test Eigen3::Eigen yaml-cpp)

# 单独编译全向逻辑，避免和 omniperception 对象库重复链接同一份源文件。
add_executable(
    ovsentry_omni_logic_test
    tests/ovsentry_omni_logic_test.cpp
    tasks/omniperception/ovsentry_omni_logic.cpp
)
target_link_libraries(ovsentry_omni_logic_test ${SP_OPENCV_FMT} tools)

if(BUILD_TESTING)
    add_test(NAME gimbal_axis_order_test COMMAND gimbal_axis_order_test)
    add_test(NAME shooter_logic_test COMMAND shooter_logic_test)
    add_test(NAME sentry_yaw_control_test COMMAND sentry_yaw_control_test)
    add_test(NAME sentry_mpc_transform_test COMMAND sentry_mpc_transform_test)
    add_test(NAME sentry_mpc_takeover_test COMMAND sentry_mpc_takeover_test)
    add_test(NAME ros2_gimbal_big_yaw_lookup_test COMMAND ros2_gimbal_big_yaw_lookup_test)
    add_test(NAME camera_passthrough_test COMMAND camera_passthrough_test)
    add_test(
        NAME sentry_camera_axis_test
        COMMAND sentry_camera_axis_test ${CMAKE_CURRENT_SOURCE_DIR}/configs/sentry.yaml
    )
    add_test(
        NAME planner_sentry_safety_test
        COMMAND planner_sentry_safety_test ${CMAKE_CURRENT_SOURCE_DIR}/configs/sentry.yaml
    )
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/tests/local/local_tests.cmake OPTIONAL)

# 只有在 io 子目录成功生成对应的可选 ROS2 库时，才编译相关入口。
if(TARGET io_ros2_nav)
    sp_add_executables(
        GLOBS src/sentry*.cpp
        LIBS ${SP_OPENCV_FMT} yaml-cpp auto_aim omniperception tools io io_ros2_nav
    )
    sp_add_executables(
        GLOBS tests/publish_test.cpp
        LIBS ${SP_OPENCV_FMT} tools io auto_aim io_ros2_nav
    )
    sp_add_executables(
        GLOBS tests/subscribe_test.cpp tests/topic_loop_test.cpp
        LIBS ${SP_OPENCV_FMT} tools io io_ros2_nav
    )
else()
    message(STATUS "io_ros2_nav target unavailable, skipping sentry and ROS2 nav tests.")
endif()

if(TARGET io_ros2_gimbal)
    sp_add_executables(
        GLOBS src/ovsentry_auto_aim_test.cpp src/ovsentry_mpc.cpp
        LIBS ${SP_OPENCV_FMT} yaml-cpp auto_aim tools io io_ros2_gimbal
    )
    sp_add_executables(
        GLOBS src/ovsentry_omni.cpp
        LIBS ${SP_OPENCV_FMT} yaml-cpp auto_aim omniperception tools io io_ros2_gimbal
    )
    sp_add_executables(
        GLOBS src/ovsentry_omni_mpc.cpp src/ovsentry_omni_mpc_debug.cpp
        LIBS ${SP_OPENCV_FMT} yaml-cpp auto_aim auto_buff omniperception tools io io_ros2_gimbal nlohmann_json::nlohmann_json
    )
else()
    if(SENTRY_SPECIAL_MODE STREQUAL "OFF")
        message(STATUS "SENTRY_SPECIAL_MODE=OFF, skipping ovsentry_* targets.")
    else()
        message(STATUS "io_ros2_gimbal target unavailable under SENTRY_SPECIAL_MODE=${SENTRY_SPECIAL_MODE}, skipping ovsentry_* targets.")
    endif()
endif()
