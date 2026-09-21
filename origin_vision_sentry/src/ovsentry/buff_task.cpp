#include "buff_task.hpp"

#include "auto_aim_helpers.hpp"
#include "io/gimbal/gimbal.hpp"
#include "io/ros2/ros2_gimbal.hpp"
#include "tools/math_tools.hpp"

namespace ovsentry
{
namespace
{

io::GimbalState make_buff_gimbal_state(const io::ROS2GimbalState & state)
{
  return {
    static_cast<float>(state.yaw), static_cast<float>(state.yaw_vel),
    static_cast<float>(state.pitch), static_cast<float>(state.pitch_vel),
    static_cast<float>(state.bullet_speed), 0};
}

}  // namespace

BuffTask::BuffTask(SentryRuntime & rt)
: rt_(rt),
  detector_(std::make_unique<auto_buff::Buff_Detector>(rt.cfg().config_path)),
  solver_(std::make_unique<auto_buff::Solver>(rt.cfg().config_path)),
  aimer_(std::make_unique<auto_buff::Aimer>(rt.cfg().config_path))
{
}

void BuffTask::reset_hold() { hold_command_.reset(); }

void BuffTask::run()
{
  auto & f = rt_.frame();
  f.left_img.release();
  f.right_img.release();
  f.back_img.release();
  f.tracker_state = "small_buff";
  f.armors.clear();
  f.targets.clear();
  if (!detector_ || !solver_ || !aimer_) {
    rt_.send_idle();
    return;
  }

  solver_->set_R_gimbal2world(f.q);
  const auto t_buff0 = std::chrono::steady_clock::now();
  f.buff_power_rune = detector_->detect(f.main_img);
  const auto t_buff1 = std::chrono::steady_clock::now();
  f.buff_detect_ms = tools::delta_time(t_buff1, t_buff0) * 1e3;
  solver_->solve(f.buff_power_rune);
  target_.get_target(f.buff_power_rune, f.main_timestamp);
  f.buff_target_ready = !target_.is_unsolve();

  auto buff_target_copy = target_;
  f.buff_plan =
    aimer_->mpc_aim(buff_target_copy, f.main_timestamp, make_buff_gimbal_state(f.gimbal_state), true);

  f.command.control = f.buff_plan.control;
  f.command.shoot = f.buff_plan.fire;
  f.command.yaw = tools::limit_rad(f.buff_plan.yaw);
  f.command.pitch = f.buff_plan.pitch;
  f.command.big_yaw = nearest_continuous_yaw_rad(f.command.yaw, f.gimbal_state.big_yaw);
  f.command.small_yaw = f.command.yaw;
  f.command.has_target_yaw = f.command.control;

  if (f.command.control) {
    hold_command_ = f.command;
    last_control_at_ = f.now;
  } else if (
    hold_command_.has_value() && f.now - last_control_at_ <= rt_.cfg().buff_lost_cmd_hold) {
    f.command = hold_command_.value();
    f.command.shoot = false;
    f.buff_command_held = true;
  }

  rt_.gimbal().send_mpc(
    f.command.control, f.command.shoot, f.command.big_yaw, f.command.small_yaw, f.command.pitch,
    f.buff_command_held ? 0.0 : f.buff_plan.yaw_vel,
    f.buff_command_held ? 0.0 : f.buff_plan.pitch_vel,
    f.buff_command_held ? 0.0 : f.buff_plan.yaw_acc,
    f.buff_command_held ? 0.0 : f.buff_plan.pitch_acc, 0, 0.0, 0.0, 0.0);
}

}  // namespace ovsentry
