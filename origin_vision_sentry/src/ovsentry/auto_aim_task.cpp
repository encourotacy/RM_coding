#include "auto_aim_task.hpp"

#include "auto_aim_helpers.hpp"
#include "io/ros2/ros2_gimbal.hpp"

namespace ovsentry
{

AutoAimTask::AutoAimTask(SentryRuntime & rt)
: rt_(rt),
  yolo_(std::make_unique<auto_aim::YOLO>(rt.cfg().config_path, yolo_debug_, "auto_aim_device")),
  tracker_(rt.cfg().config_path, rt.solver()),
  aimer_(rt.cfg().config_path),
  shooter_(rt.cfg().config_path),
  planner_(rt.cfg().config_path)
{
}

void AutoAimTask::detect_and_track()
{
  auto & f = rt_.frame();
  f.t0 = std::chrono::steady_clock::now();
  f.armors.clear();
  f.targets.clear();
  f.tracker_state = "idle";
  f.armors = yolo_->detect(f.main_img, f.frame_count);
  rt_.decider().armor_filter(f.armors);
  rt_.decider().set_priority(f.armors);
  apply_armor_target_mask(f.armors, f.armor_target_mask);
  f.targets = tracker_.track(f.armors, f.main_timestamp);
  f.tracker_state = tracker_.state();
  f.t1 = std::chrono::steady_clock::now();
}

void AutoAimTask::aim_and_send()
{
  auto & f = rt_.frame();
  f.left_img.release();
  f.right_img.release();
  f.back_img.release();

  const bool armor_acquiring = f.tracker_state == "detecting";
  if (armor_acquiring) {
    f.command.control = true;
    f.command.shoot = false;
    f.command.yaw = f.gimbal_state.yaw;
    f.command.pitch = -f.gimbal_state.pitch;
    f.command.big_yaw = f.gimbal_state.big_yaw;
    f.command.small_yaw = f.gimbal_state.yaw;
    f.command.has_target_yaw = true;
  } else {
    f.command = aimer_.aim(f.targets, f.main_timestamp, rt_.gimbal().bullet_speed(), aimer_to_now_);
    if (f.command.control && !f.targets.empty()) {
      apply_sentry_tracking_yaws(f.command, f.targets.front(), f.gimbal_state.big_yaw);
    }
    f.command.shoot =
      shooter_.shoot(f.command, aimer_, f.targets, f.ypr, f.tracker_state == "tracking");
    fill_nav_target_info(f.command, f.targets);
  }

  const bool outpost_convergence = !f.targets.empty() &&
                                   f.targets.front().name == auto_aim::ArmorName::outpost &&
                                   (!f.targets.front().convergened() || f.targets.front().diverged());
  const bool static_outpost_direct =
    !f.targets.empty() && f.targets.front().outpost_static_direct_active();
  if (!armor_acquiring && outpost_convergence && !static_outpost_direct) {
    f.command = io::Command{false, false, 0.0, 0.0};
  }

  const bool unlocked_outpost = !f.targets.empty() && is_unlocked_outpost_target(f.targets.front());
  double small_yaw_vel = 0.0;
  double pitch_vel = 0.0;
  double small_yaw_acc = 0.0;
  double pitch_acc = 0.0;
  if (f.command.control && !armor_acquiring && !f.targets.empty() && !unlocked_outpost) {
    const auto mpc_plan = planner_.plan(f.targets.front(), rt_.gimbal().bullet_speed());
    if (mpc_plan.control) {
      small_yaw_vel = mpc_plan.yaw_vel;
      pitch_vel = mpc_plan.pitch_vel;
      small_yaw_acc = mpc_plan.yaw_acc;
      pitch_acc = mpc_plan.pitch_acc;
    }
  }

  const double big_yaw = f.command.has_target_yaw ? f.command.big_yaw : f.command.yaw;
  const double small_yaw = f.command.has_target_yaw ? f.command.small_yaw : f.command.yaw;
  rt_.gimbal().send_mpc(
    f.command.control, f.command.shoot, big_yaw, small_yaw, f.command.pitch, small_yaw_vel,
    pitch_vel, small_yaw_acc, pitch_acc, static_cast<uint8_t>(f.command.armor_id), f.command.vx,
    f.command.vy, f.command.horizon_distance);
}

}  // namespace ovsentry
