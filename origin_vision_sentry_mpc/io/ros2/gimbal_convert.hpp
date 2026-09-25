#ifndef IO__GIMBAL_CONVERT_HPP
#define IO__GIMBAL_CONVERT_HPP

#include "io/gimbal/gimbal.hpp"
#include "io/ros2/ros2_gimbal.hpp"

namespace io
{
inline GimbalState to_gimbal_state(const ROS2GimbalState & state)
{
  return {
    static_cast<float>(state.yaw), static_cast<float>(state.yaw_vel),
    static_cast<float>(state.pitch), static_cast<float>(state.pitch_vel),
    static_cast<float>(state.bullet_speed), 0};
}

}  // namespace io

#endif  // IO__GIMBAL_CONVERT_HPP
