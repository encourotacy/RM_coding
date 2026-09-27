#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <Eigen/Dense>
#include <list>
#include <optional>
#include <vector>

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
  // 高转速直接允许开火；前哨站还要过锁定和相位窗口。mpc_fire 是距离阈值的结果。
  bool allow_fire(const Target & target, bool mpc_fire, bool aim_valid, double armor_yaw);
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
  bool is_multiple_thresh_ = false;
  std::vector<double> planner_judge_distance_;
  std::vector<double> planner_fire_thresh_;
  double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;
  double outpost_prediction_offset_s_;
  bool high_spin_force_fire_enabled_ = false;
  bool high_spin_force_fire_active_ = false;
  double high_spin_force_fire_enter_speed_ = 8.0;
  double high_spin_force_fire_exit_speed_ = 6.0;
  bool outpost_fire_require_locked_ = true;
  bool outpost_fire_window_enabled_ = false;
  double outpost_fire_enter_angle_ = 0.0;
  double outpost_fire_exit_angle_ = 0.0;
  tools::GimbalAxisOrder gimbal_axis_order_;

  TinySolver * yaw_solver_;
  TinySolver * pitch_solver_;

  void setup_yaw_solver(const std::string & config_path);
  void setup_pitch_solver(const std::string & config_path);
  double fire_thresh_for(double distance) const;
  void update_high_spin(bool enabled_for_target, double angular_speed);
  bool outpost_phase_in_window(const Target & target, double armor_yaw) const;

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
