#ifndef IO__ROS2_GIMBAL_STATE_HPP
#define IO__ROS2_GIMBAL_STATE_HPP

namespace io
{

struct ROS2GimbalState
{
  double yaw = 0.0;
  double yaw_vel = 0.0;
  double pitch = 0.0;
  double pitch_vel = 0.0;
  double bullet_speed = 0.0;
  double big_yaw = 0.0;
};

}  // namespace io

#endif  // IO__ROS2_GIMBAL_STATE_HPP
