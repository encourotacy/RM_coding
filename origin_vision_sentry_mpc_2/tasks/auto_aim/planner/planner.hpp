#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <Eigen/Dense>
#include <list>
#include <optional>

#include "tasks/auto_aim/target.hpp"
#include "tools/math_tools.hpp"
#include "tinympc/tiny_api.hpp"

namespace auto_aim
{
constexpr double DT = 0.01;
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

using Trajectory = Eigen::Matrix<double, 4, HORIZON>;  // yaw, yaw_vel, pitch, pitch_vel

struct Plan
{
  bool control = false;
  bool fire = false;
  float target_yaw = 0.0F;
  float target_pitch = 0.0F;
  float yaw = 0.0F;
  float yaw_vel = 0.0F;
  float yaw_acc = 0.0F;
  float pitch = 0.0F;
  float pitch_vel = 0.0F;
  float pitch_acc = 0.0F;
};

struct SentryPlan
{
  // yaw is the world-referenced small-yaw command; pitch is the physical outer-pitch joint.
  Plan world_small_yaw_plan{};
  double big_yaw = 0.0;
};

class Planner
{
public:
  Eigen::Vector4d debug_xyza;
  Planner(const std::string & config_path);

  Plan plan(Target target, double bullet_speed);
  Plan plan(std::optional<Target> target, double bullet_speed);
  SentryPlan plan_sentry_world(
    Target target, double bullet_speed, std::optional<int> preferred_armor_id = std::nullopt,
    bool aim_center = false);
  SentryPlan plan_sentry_world(
    std::optional<Target> target, double bullet_speed,
    std::optional<int> preferred_armor_id = std::nullopt, bool aim_center = false);

private:
  double yaw_offset_;
  double pitch_offset_;
  double fire_thresh_;
  double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;
  double outpost_prediction_offset_s_;
  tools::GimbalAxisOrder gimbal_axis_order_;

  TinySolver * yaw_solver_;
  TinySolver * pitch_solver_;

  void setup_yaw_solver(const std::string & config_path);
  void setup_pitch_solver(const std::string & config_path);

  Plan plan_impl(
    Target target, double bullet_speed, bool sentry_world, double * sentry_big_yaw,
    std::optional<int> preferred_armor_id, bool aim_center);
  Eigen::Matrix<double, 2, 1> aim(
    const Target & target, double bullet_speed, bool sentry_world,
    std::optional<int> preferred_armor_id, bool aim_center);
  Eigen::Vector4d select_aim_point(
    const Target & target, std::optional<int> preferred_armor_id, bool aim_center) const;
  Eigen::Vector4d select_armor(
    const Target & target, std::optional<int> preferred_armor_id) const;
  Trajectory get_trajectory(
    Target & target, double yaw0, double bullet_speed, bool sentry_world,
    std::optional<int> preferred_armor_id, bool aim_center);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__PLANNER_HPP
