#ifndef AUTO_AIM__SENTRY_MPC_TRANSFORM_HPP
#define AUTO_AIM__SENTRY_MPC_TRANSFORM_HPP

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>

#include "tools/math_tools.hpp"

namespace auto_aim::sentry_mpc_transform
{
struct JointTrajectory
{
  double yaw = 0.0;
  double pitch = 0.0;
  double yaw_vel = 0.0;
  double pitch_vel = 0.0;
  double yaw_acc = 0.0;
  double pitch_acc = 0.0;
};

struct YawMotion
{
  double angle = 0.0;
  double velocity = 0.0;
  double acceleration = 0.0;
};

// The electrical small-yaw controller uses a world-referenced yaw target, while pitch is the
// physical outer joint. Keep those two command semantics explicit instead of treating small yaw
// as the inner joint angle.
struct WorldSmallYawCommand
{
  double small_yaw = 0.0;
  double pitch = 0.0;
};

inline Eigen::Vector2d world_direction_to_big_yaw_local(
  const Eigen::Vector3d & world_direction, double big_yaw,
  tools::GimbalAxisOrder local_axis_order)
{
  const Eigen::Vector3d local_direction =
    Eigen::AngleAxisd(-big_yaw, Eigen::Vector3d::UnitZ()) * world_direction;
  return tools::gimbal_command_from_direction(local_direction, local_axis_order);
}

inline Eigen::Vector2d world_yaw_elevation_to_big_yaw_local(
  double world_yaw, double elevation, double big_yaw,
  tools::GimbalAxisOrder local_axis_order)
{
  const double cos_elevation = std::cos(elevation);
  const Eigen::Vector3d world_direction{
    cos_elevation * std::cos(world_yaw), cos_elevation * std::sin(world_yaw),
    std::sin(elevation)};
  return world_direction_to_big_yaw_local(world_direction, big_yaw, local_axis_order);
}

inline WorldSmallYawCommand world_direction_to_world_small_yaw_command(
  const Eigen::Vector3d & world_direction, double big_yaw,
  tools::GimbalAxisOrder local_axis_order)
{
  const Eigen::Vector2d local_command =
    world_direction_to_big_yaw_local(world_direction, big_yaw, local_axis_order);
  return {std::atan2(world_direction.y(), world_direction.x()), local_command[1]};
}

inline WorldSmallYawCommand world_yaw_elevation_to_world_small_yaw_command(
  double world_yaw, double elevation, double big_yaw,
  tools::GimbalAxisOrder local_axis_order)
{
  const Eigen::Vector2d local_command =
    world_yaw_elevation_to_big_yaw_local(
      world_yaw, elevation, big_yaw, local_axis_order);
  return {world_yaw, local_command[1]};
}

inline YawMotion center_yaw_motion(double x, double y, double vx, double vy)
{
  YawMotion motion;
  motion.angle = std::atan2(y, x);

  const double radius_sq = x * x + y * y;
  if (radius_sq < 1e-8) return motion;

  const double cross = x * vy - y * vx;
  const double radial_dot = x * vx + y * vy;
  motion.velocity = cross / radius_sq;
  motion.acceleration = -2.0 * radial_dot * cross / (radius_sq * radius_sq);
  return motion;
}

inline Eigen::Vector2d local_command_at(
  const JointTrajectory & world_plan, const YawMotion & big_yaw, double time,
  tools::GimbalAxisOrder axis_order)
{
  const double time_sq = time * time;
  const double world_yaw =
    world_plan.yaw + world_plan.yaw_vel * time + 0.5 * world_plan.yaw_acc * time_sq;
  const double world_pitch =
    world_plan.pitch + world_plan.pitch_vel * time + 0.5 * world_plan.pitch_acc * time_sq;
  const double big_yaw_angle =
    big_yaw.angle + big_yaw.velocity * time + 0.5 * big_yaw.acceleration * time_sq;

  const Eigen::Vector3d world_direction =
    tools::gimbal_direction_from_command(world_yaw, world_pitch, axis_order);
  return world_direction_to_big_yaw_local(world_direction, big_yaw_angle, axis_order);
}

inline JointTrajectory to_big_yaw_local(
  const JointTrajectory & world_plan, const YawMotion & big_yaw,
  tools::GimbalAxisOrder axis_order, double dt = 0.01)
{
  dt = std::max(dt, 1e-4);
  const Eigen::Vector2d p0 = local_command_at(world_plan, big_yaw, 0.0, axis_order);
  Eigen::Vector2d p1 = local_command_at(world_plan, big_yaw, dt, axis_order);
  Eigen::Vector2d p2 = local_command_at(world_plan, big_yaw, 2.0 * dt, axis_order);

  // 在 +/-pi 分支附近保持关节轨迹连续，避免生成瞬时大速度。
  p1[0] = p0[0] + tools::limit_rad(p1[0] - p0[0]);
  p2[0] = p1[0] + tools::limit_rad(p2[0] - p1[0]);
  p1[1] = p0[1] + tools::limit_rad(p1[1] - p0[1]);
  p2[1] = p1[1] + tools::limit_rad(p2[1] - p1[1]);

  JointTrajectory local;
  local.yaw = p0[0];
  local.pitch = p0[1];
  local.yaw_vel = (-3.0 * p0[0] + 4.0 * p1[0] - p2[0]) / (2.0 * dt);
  local.pitch_vel = (-3.0 * p0[1] + 4.0 * p1[1] - p2[1]) / (2.0 * dt);
  local.yaw_acc = (p2[0] - 2.0 * p1[0] + p0[0]) / (dt * dt);
  local.pitch_acc = (p2[1] - 2.0 * p1[1] + p0[1]) / (dt * dt);
  return local;
}
}  // namespace auto_aim::sentry_mpc_transform

#endif  // AUTO_AIM__SENTRY_MPC_TRANSFORM_HPP
