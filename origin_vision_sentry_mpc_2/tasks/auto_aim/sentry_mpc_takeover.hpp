#ifndef AUTO_AIM__SENTRY_MPC_TAKEOVER_HPP
#define AUTO_AIM__SENTRY_MPC_TAKEOVER_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>

#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/sentry_yaw_control.hpp"

namespace auto_aim
{

struct SentryMpcSetpoint
{
  io::Command command{false, false, 0.0, 0.0};
  double yaw_vel = 0.0;
  double pitch_vel = 0.0;
  double yaw_acc = 0.0;
  double pitch_acc = 0.0;
  double takeover_alpha = 0.0;
  bool fire_ready = false;
};

// MPC第一次接管时从云台实测关节位置平滑过渡到规划轨迹，
// 避免把规划器虚拟的历史轨迹中点作为第一条阶跃指令直接下发。
// 后续坐标语义保持为：
// big_yaw=连续世界角，small_yaw=世界系小yaw目标，pitch=物理外pitch关节。
class SentryMpcTakeover
{
public:
  explicit SentryMpcTakeover(
    double transition_time_s,
    double yaw_acceleration_scale = std::numeric_limits<double>::infinity(),
    double pitch_acceleration_scale = std::numeric_limits<double>::infinity())
  : base_transition_time_s_(
      std::isfinite(transition_time_s) ? std::max(0.0, transition_time_s) : 0.20),
    transition_time_s_(base_transition_time_s_),
    yaw_acceleration_scale_(sanitize_acceleration(yaw_acceleration_scale)),
    pitch_acceleration_scale_(sanitize_acceleration(pitch_acceleration_scale))
  {
  }

  void reset() { active_ = false; }

  SentryMpcSetpoint update(
    const Plan & plan, double center_world_yaw, double current_small_yaw,
    double current_small_yaw_vel, double current_pitch, double current_pitch_vel,
    double current_big_yaw, std::chrono::steady_clock::time_point now)
  {
    SentryMpcSetpoint output;
    if (!valid(
          plan, center_world_yaw, current_small_yaw, current_small_yaw_vel, current_pitch,
          current_pitch_vel, current_big_yaw)) {
      reset();
      return output;
    }

    if (!active_) {
      active_ = true;
      started_at_ = now;
      start_small_yaw_ = current_small_yaw;
      start_small_yaw_vel_ = current_small_yaw_vel;
      start_pitch_ = current_pitch;
      start_pitch_vel_ = current_pitch_vel;
      start_big_yaw_ = current_big_yaw;
      last_desired_small_yaw_ =
        sentry_yaw_control::nearest_continuous_yaw(plan.yaw, current_small_yaw);
      last_desired_big_yaw_ =
        sentry_yaw_control::nearest_continuous_yaw(center_world_yaw, current_big_yaw);
      // 以MPC已有的轴加速度量级估算首次接管时长：角差越大，接管越慢。
      // 这不是新的场景门限，也不按目标类别分支。
      transition_time_s_ = std::max(
        {base_transition_time_s_,
         required_transition_time(
           last_desired_small_yaw_ - start_small_yaw_, plan.yaw_vel - start_small_yaw_vel_,
           yaw_acceleration_scale_),
         required_transition_time(
           last_desired_big_yaw_ - start_big_yaw_, 0.0, yaw_acceleration_scale_),
         required_transition_time(
           plan.pitch - start_pitch_, plan.pitch_vel - start_pitch_vel_,
           pitch_acceleration_scale_)});
    }

    const double elapsed =
      std::max(0.0, std::chrono::duration<double>(now - started_at_).count());
    const Blend blend = make_blend(elapsed);

    // 每帧相对上一条连续目标解包，
    // 避免长时跟踪跨越相对初始角±pi时跳2pi。
    last_desired_small_yaw_ =
      sentry_yaw_control::nearest_continuous_yaw(plan.yaw, last_desired_small_yaw_);
    last_desired_big_yaw_ = sentry_yaw_control::nearest_continuous_yaw(
      center_world_yaw, last_desired_big_yaw_);

    const AxisSetpoint small_yaw = blend_axis(
      start_small_yaw_, start_small_yaw_vel_, last_desired_small_yaw_, plan.yaw_vel,
      plan.yaw_acc, elapsed, blend);
    const AxisSetpoint pitch = blend_axis(
      start_pitch_, start_pitch_vel_, plan.pitch, plan.pitch_vel, plan.pitch_acc, elapsed,
      blend);
    const AxisSetpoint big_yaw =
      blend_axis(start_big_yaw_, 0.0, last_desired_big_yaw_, 0.0, 0.0, elapsed, blend);

    output.command = {true, false, small_yaw.position, pitch.position};
    // 这里的大小yaw都已经相对上一条连续目标展开。
    // 不能再相对可能包装到[-pi,pi]的反馈做第二次nearest，
    // 否则181°会被重新写回-179°。
    output.command.small_yaw = small_yaw.position;
    output.command.big_yaw = big_yaw.position;
    output.command.has_target_yaw = true;
    output.yaw_vel = small_yaw.velocity;
    output.pitch_vel = pitch.velocity;
    output.yaw_acc = small_yaw.acceleration;
    output.pitch_acc = pitch.acceleration;
    output.takeover_alpha = blend.position;
    output.fire_ready = blend.complete;
    return output;
  }

private:
  struct Blend
  {
    double position = 0.0;
    double velocity = 0.0;
    double acceleration = 0.0;
    bool complete = false;
  };

  struct AxisSetpoint
  {
    double position = 0.0;
    double velocity = 0.0;
    double acceleration = 0.0;
  };

  Blend make_blend(double elapsed) const
  {
    if (transition_time_s_ <= 1e-6) return {1.0, 0.0, 0.0, true};

    const double u = std::clamp(elapsed / transition_time_s_, 0.0, 1.0);
    const double u2 = u * u;
    const double u3 = u2 * u;
    const double u4 = u3 * u;
    const double u5 = u4 * u;
    // smootherstep在接管区间两端的一、二阶导数均为0。
    const double position = 6.0 * u5 - 15.0 * u4 + 10.0 * u3;
    const double velocity =
      (30.0 * u4 - 60.0 * u3 + 30.0 * u2) / transition_time_s_;
    const double acceleration =
      (120.0 * u3 - 180.0 * u2 + 60.0 * u) /
      (transition_time_s_ * transition_time_s_);
    return {position, velocity, acceleration, u >= 1.0};
  }

  static AxisSetpoint blend_axis(
    double start_position, double start_velocity, double desired_position,
    double desired_velocity, double desired_acceleration, double elapsed, const Blend & blend)
  {
    const double start_reference = start_position + start_velocity * elapsed;
    const double delta_position = desired_position - start_reference;

    AxisSetpoint output;
    output.position = start_reference + blend.position * delta_position;
    output.velocity =
      (1.0 - blend.position) * start_velocity + blend.position * desired_velocity +
      blend.velocity * delta_position;
    output.acceleration =
      blend.position * desired_acceleration +
      2.0 * blend.velocity * (desired_velocity - start_velocity) +
      blend.acceleration * delta_position;
    return output;
  }

  static double sanitize_acceleration(double acceleration_scale)
  {
    return std::isfinite(acceleration_scale) && acceleration_scale > 0.0
             ? acceleration_scale
             : std::numeric_limits<double>::infinity();
  }

  static double required_transition_time(
    double position_delta, double velocity_delta, double acceleration_scale)
  {
    if (!std::isfinite(acceleration_scale)) return 0.0;
    // smootherstep的归一化峰值：max|s''|=10/sqrt(3)，max|2s'|=3.75。
    // 将位置项和速度差项合并估算；计划自身的acc已由MPC约束，因此这里不把
    // 估算结果宣传为最终setpoint的硬加速度限幅。
    constexpr double peak_position_acceleration = 5.773502691896258;
    constexpr double peak_velocity_acceleration = 3.75;
    const double linear = peak_velocity_acceleration * std::abs(velocity_delta);
    const double constant = peak_position_acceleration * std::abs(position_delta);
    return (linear + std::sqrt(linear * linear + 4.0 * acceleration_scale * constant)) /
           (2.0 * acceleration_scale);
  }

  static bool valid(
    const Plan & plan, double center_world_yaw, double current_small_yaw,
    double current_small_yaw_vel, double current_pitch, double current_pitch_vel,
    double current_big_yaw)
  {
    return plan.control && std::isfinite(center_world_yaw) && std::isfinite(plan.yaw) &&
           std::isfinite(plan.pitch) && std::isfinite(plan.yaw_vel) &&
           std::isfinite(plan.pitch_vel) && std::isfinite(plan.yaw_acc) &&
           std::isfinite(plan.pitch_acc) && std::isfinite(current_small_yaw) &&
           std::isfinite(current_small_yaw_vel) && std::isfinite(current_pitch) &&
           std::isfinite(current_pitch_vel) && std::isfinite(current_big_yaw);
  }

  double base_transition_time_s_ = 0.0;
  double transition_time_s_ = 0.0;
  double yaw_acceleration_scale_ = std::numeric_limits<double>::infinity();
  double pitch_acceleration_scale_ = std::numeric_limits<double>::infinity();
  bool active_ = false;
  std::chrono::steady_clock::time_point started_at_{};
  double start_small_yaw_ = 0.0;
  double start_small_yaw_vel_ = 0.0;
  double start_pitch_ = 0.0;
  double start_pitch_vel_ = 0.0;
  double start_big_yaw_ = 0.0;
  double last_desired_small_yaw_ = 0.0;
  double last_desired_big_yaw_ = 0.0;
};

template <typename Gimbal>
void dispatch_sentry_mpc(Gimbal & gimbal, const SentryMpcSetpoint & setpoint)
{
  const auto & command = setpoint.command;
  const double big_yaw = command.has_target_yaw ? command.big_yaw : command.yaw;
  const double small_yaw = command.has_target_yaw ? command.small_yaw : command.yaw;
  gimbal.send_mpc(
    command.control, command.shoot, big_yaw, small_yaw, command.pitch, setpoint.yaw_vel,
    setpoint.pitch_vel, setpoint.yaw_acc, setpoint.pitch_acc,
    static_cast<uint8_t>(command.armor_id), command.vx, command.vy, command.horizon_distance);
}

}  // namespace auto_aim

#endif  // AUTO_AIM__SENTRY_MPC_TAKEOVER_HPP
