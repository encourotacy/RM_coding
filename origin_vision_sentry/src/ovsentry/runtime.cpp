#include "runtime.hpp"

#include <cmath>
#include <exception>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include "auto_aim_task.hpp"
#include "buff_task.hpp"
#include "io/cboard.hpp"
#include "io/ros2/ros2_gimbal.hpp"
#include "overlay.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace ovsentry
{

void FrameState::reset_outputs()
{
  best_omni_result.reset();
  omni_target_abs_yaw_deg.reset();
  omni_candidate_abs_yaw_deg.reset();
  omni_candidate_base_big_yaw_deg.reset();
  omni_candidate_age_ms.reset();
  omni_candidate_delta_deg.reset();
  omni_target_error_deg.reset();
  omni_selected_confidence.reset();
  omni_selected_priority.reset();
  omni_selected_slot.reset();
  omni_hold_applied = false;
  omni_retarget_blocked = false;
  omni_retarget_cd_active = false;
  omni_same_target_continuation = false;
  omni_target_reached = false;
  omni_cmd_timeout_active = false;
  omni_cmd_timed_out = false;
  omni_block_reason = "none";
  omni_cmd_elapsed_ms = 0.0;
  omni_retarget_remaining_ms = 0.0;
  command = io::Command{false, false, 0.0, 0.0};
  buff_power_rune.reset();
  buff_plan = auto_aim::Plan{false, false, 0, 0, 0, 0, 0, 0, 0, 0};
  buff_detect_ms = 0.0;
  buff_target_ready = false;
  buff_command_held = false;
}

std::optional<RuntimeConfig> prepare_config(int argc, char ** argv, AppMode mode)
{
  auto config = parse_runtime_config(argc, argv);
  if (!config) return std::nullopt;

  config->mode = mode;
  tools::logger()->info(
    "[OVSentry{}] inference devices: auto_aim={} omni={}", app_mode_name(mode),
    config->auto_aim_device, config->omni_device);
  return config;
}

SentryRuntime::~SentryRuntime() = default;

SentryRuntime::SentryRuntime(RuntimeConfig cfg, const char * window_title, bool omni_mosaic)
: cfg_(std::move(cfg)),
  window_title_(window_title),
  omni_mosaic_(omni_mosaic),
  recorder_(30),
  gimbal_(std::make_unique<io::ROS2Gimbal>(cfg_.config_path)),
  armor_ignore_subscriber_(cfg_.auto_aim_ignore_topic, cfg_.auto_aim_ignore_msg_type),
  auto_aim_camera_(std::make_unique<io::Camera>(cfg_.config_path)),
  solver_(cfg_.config_path),
  decider_(cfg_.config_path)
{
}

bool SentryRuntime::read_main_frame()
{
  try {
    auto_aim_camera_->read(frame_.main_img, frame_.main_timestamp);
    if (frame_.main_img.empty()) return false;
  } catch (const std::exception & e) {
    tools::logger()->error(
      "[OVSentry{}] main camera read failed: {}", app_mode_name(cfg_.mode), e.what());
    return false;
  }
  return true;
}

void SentryRuntime::update_sensors()
{
  frame_.frame_count++;
  frame_.q = gimbal_->imu_at_image(frame_.main_timestamp);
  solver_.set_R_gimbal2world(frame_.q);
  frame_.gimbal_state = gimbal_->state();
  frame_.ypr = tools::eulers(solver_.R_gimbal2world(), 2, 1, 0);
  frame_.armor_target_mask = read_nav_armor_target_mask(armor_ignore_subscriber_);
  frame_.t0 = frame_.t1 = std::chrono::steady_clock::now();
}

void SentryRuntime::reset_outputs()
{
  frame_.reset_outputs();
  frame_.now = std::chrono::steady_clock::now();
}

bool SentryRuntime::buff_requested() const { return buff_request_subscriber_.requested(); }

void SentryRuntime::send_idle()
{
  gimbal_->send_mpc(false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0);
}

bool SentryRuntime::finish_frame(
  bool small_buff, bool omni, AutoAimTask * auto_aim, BuffTask * buff)
{
  publish_telemetry(small_buff, omni, auto_aim, buff);
  if (!cfg_.display) return true;
  return render_display(small_buff, omni, auto_aim, buff);
}

void SentryRuntime::publish_telemetry(
  bool small_buff, bool omni, AutoAimTask * auto_aim, BuffTask * buff)
{
  const auto & f = frame_;
  nlohmann::json data;
  data["app_mode"] = app_mode_name(cfg_.mode);
  data["mode"] = small_buff ? 2 : (omni ? 1 : 0);
  data["gimbal_mode"] = io::MODES[static_cast<int>(gimbal_->mode())];
  data["armor_num"] = f.armors.size();
  data["tracker_state"] = f.tracker_state;
  data["armor_acquiring"] = (!small_buff && f.tracker_state == "detecting") ? 1 : 0;
  data["gimbal_yaw"] = f.ypr[0] * 57.3;
  data["gimbal_small_yaw"] = f.gimbal_state.yaw * 57.3;
  data["gimbal_big_yaw"] = f.gimbal_state.big_yaw * 57.3;
  data["bullet_speed"] = gimbal_->bullet_speed();
  data["mpc_control"] = f.command.control ? 1 : 0;
  data["mpc_fire"] = f.command.shoot ? 1 : 0;
  data["mpc_yaw"] = (f.command.has_target_yaw ? f.command.small_yaw : f.command.yaw) * 57.3;
  data["mpc_pitch"] = f.command.pitch * 57.3;
  data["target_armor_id"] = static_cast<int>(f.command.armor_id);
  data["target_vx"] = f.command.vx;
  data["target_vy"] = f.command.vy;
  data["horizon_distance"] = f.command.horizon_distance;
  data["aim_source"] =
    (!small_buff && !omni && f.command.control && auto_aim) ? auto_aim->aimer().debug_aim_point.source
                                                           : -1;
  data["aim_armor_id"] =
    (!small_buff && !omni && f.command.control && auto_aim) ? auto_aim->aimer().debug_aim_point.armor_id
                                                           : -1;
  if (small_buff) {
    data["buff_detected"] = f.buff_power_rune.has_value() ? 1 : 0;
    data["buff_target_ready"] = f.buff_target_ready ? 1 : 0;
    data["buff_command_held"] = f.buff_command_held ? 1 : 0;
    data["buff_detect_ms"] = f.buff_detect_ms;
    data["buff_yaw"] = f.buff_plan.yaw * 57.3;
    data["buff_pitch"] = f.buff_plan.pitch * 57.3;
    data["buff_yaw_vel"] = f.buff_plan.yaw_vel * 57.3;
    data["buff_pitch_vel"] = f.buff_plan.pitch_vel * 57.3;
    if (f.buff_power_rune.has_value()) {
      data["buff_r_yaw"] = f.buff_power_rune->ypd_in_world[0] * 57.3;
      data["buff_r_pitch"] = f.buff_power_rune->ypd_in_world[1] * 57.3;
      data["buff_r_distance"] = f.buff_power_rune->ypd_in_world[2];
      data["buff_roll"] = f.buff_power_rune->ypr_in_world[2] * 57.3;
    }
    if (f.buff_target_ready && buff) {
      const auto x = buff->target().ekf_x();
      data["buff_target_roll"] = x[5] * 57.3;
      data["buff_target_spd"] = x[6] * 57.3;
    }
  }
  if (!f.targets.empty()) {
    const auto & target = f.targets.front();
    data["target_name"] = auto_aim::ARMOR_NAMES[target.name];
    data["outpost_layer_locked"] = target.outpost_layer_locked() ? 1 : 0;
    data["outpost_preview_ready"] = target.outpost_unlocked_prediction_ready() ? 1 : 0;
    const auto & ekf_data = target.ekf().data;
    if (ekf_data.count("init_preview_ready")) {
      data["init_preview_ready"] = ekf_data.at("init_preview_ready");
    }
    if (ekf_data.count("init_omega_margin")) {
      data["init_omega_margin"] = ekf_data.at("init_omega_margin");
    }
    if (ekf_data.count("init_margin")) {
      data["init_margin"] = ekf_data.at("init_margin");
    }
    if (auto_aim && target.name == auto_aim::ArmorName::outpost &&
        auto_aim->aimer().debug_aim_point.valid) {
      const auto x = target.ekf_x();
      const double center_yaw = std::atan2(x[2], x[0]);
      data["outpost_aim_phase_deg"] =
        std::abs(tools::limit_rad(auto_aim->aimer().debug_aim_point.xyza[3] - center_yaw)) * 57.3;
    }
  }
  data["omni_yaw_hold"] = f.omni_hold_applied ? 1 : 0;
  data["omni_target_reached"] = f.omni_target_reached ? 1 : 0;
  data["omni_cmd_timeout_active"] = f.omni_cmd_timeout_active ? 1 : 0;
  data["omni_cmd_timed_out"] = f.omni_cmd_timed_out ? 1 : 0;
  data["omni_cmd_elapsed_ms"] = f.omni_cmd_elapsed_ms;
  data["omni_retarget_cd_active"] = f.omni_retarget_cd_active ? 1 : 0;
  data["omni_retarget_blocked"] = f.omni_retarget_blocked ? 1 : 0;
  data["omni_retarget_remaining_ms"] = f.omni_retarget_remaining_ms;
  data["omni_same_target_continuation"] = f.omni_same_target_continuation ? 1 : 0;
  data["omni_block_reason"] = f.omni_block_reason;
  if (f.omni_target_abs_yaw_deg.has_value()) data["omni_target_yaw"] = f.omni_target_abs_yaw_deg.value();
  if (f.omni_candidate_abs_yaw_deg.has_value()) {
    data["omni_candidate_abs_yaw"] = f.omni_candidate_abs_yaw_deg.value();
  }
  if (f.omni_candidate_base_big_yaw_deg.has_value()) {
    data["omni_candidate_base_big_yaw"] = f.omni_candidate_base_big_yaw_deg.value();
  }
  if (f.omni_candidate_age_ms.has_value()) data["omni_candidate_age_ms"] = f.omni_candidate_age_ms.value();
  if (f.omni_candidate_delta_deg.has_value()) {
    data["omni_candidate_delta_deg"] = f.omni_candidate_delta_deg.value();
  }
  if (f.omni_target_error_deg.has_value()) data["omni_target_error_deg"] = f.omni_target_error_deg.value();
  if (f.omni_selected_confidence.has_value()) {
    data["omni_selected_confidence"] = f.omni_selected_confidence.value();
  }
  if (f.omni_selected_priority.has_value()) {
    data["omni_selected_priority"] = f.omni_selected_priority.value();
  }
  if (f.omni_selected_slot.has_value()) data["omni_selected_slot"] = f.omni_selected_slot.value();
  data["yolo_time"] = tools::delta_time(f.t1, f.t0) * 1e3;
  plotter_.plot(data);
}

bool SentryRuntime::render_display(
  bool small_buff, bool omni, AutoAimTask * auto_aim, BuffTask * buff)
{
  auto & f = frame_;
  if (small_buff && buff && buff->solver()) {
    draw_small_buff_overlay(
      f.main_img, f.buff_power_rune, buff->target(), *buff->solver(), f.buff_plan);
  } else if (auto_aim) {
    draw_auto_aim_overlay(f.main_img, f.targets, auto_aim->aimer(), solver_);
    tools::draw_text(
      f.main_img, fmt::format("[{}] mode={}", f.tracker_state, omni ? "OMNI" : "MPC"), {10, 30},
      {255, 255, 255}, 0.8, 2);
    tools::draw_text(
      f.main_img,
      fmt::format(
        "mpc yaw={:.2f} pitch={:.2f} fire={}",
        (f.command.has_target_yaw ? f.command.small_yaw : f.command.yaw) * 57.3,
        f.command.pitch * 57.3, f.command.shoot ? 1 : 0),
      {10, 60}, {154, 50, 205}, 0.8, 2);
  }
  if (f.omni_target_abs_yaw_deg.has_value()) {
    tools::draw_text(
      f.main_img, fmt::format("omni target yaw={:.2f}", f.omni_target_abs_yaw_deg.value()), {10, 90},
      {0, 255, 255}, 0.8, 2);
  }
  if (f.omni_retarget_cd_active) {
    tools::draw_text(
      f.main_img, fmt::format("omni retarget cd {:.0f}ms", f.omni_retarget_remaining_ms), {10, 120},
      f.omni_retarget_blocked ? cv::Scalar(0, 180, 255) : cv::Scalar(255, 220, 0), 0.8, 2);
  }
  if (f.omni_cmd_timeout_active || f.omni_cmd_timed_out) {
    tools::draw_text(
      f.main_img,
      fmt::format(
        "omni cmd timeout {:.0f}/{:.0f}ms hit={}", f.omni_cmd_elapsed_ms,
        cfg_.omni_command_timeout_s * 1e3, f.omni_cmd_timed_out ? 1 : 0),
      {10, 150}, f.omni_cmd_timed_out ? cv::Scalar(0, 120, 255) : cv::Scalar(180, 255, 180), 0.8, 2);
  }
  if (f.omni_target_error_deg.has_value()) {
    tools::draw_text(
      f.main_img,
      fmt::format(
        "omni target err={:.1f} reached={}", f.omni_target_error_deg.value(),
        f.omni_target_reached ? 1 : 0),
      {10, 180}, {180, 255, 180}, 0.8, 2);
  }

  if (!omni_mosaic_) {
    cv::imshow(window_title_, resize_for_view(f.main_img));
    return cv::waitKey(1) != 'q';
  }

  cv::Mat left_show =
    f.left_img.empty() ? cv::Mat::zeros(f.main_img.size(), f.main_img.type()) : f.left_img.clone();
  cv::Mat right_show =
    f.right_img.empty() ? cv::Mat::zeros(f.main_img.size(), f.main_img.type()) : f.right_img.clone();
  cv::Mat back_show =
    f.back_img.empty() ? cv::Mat::zeros(f.main_img.size(), f.main_img.type()) : f.back_img.clone();

  if (omni && f.best_omni_result.has_value()) {
    const auto & best = f.best_omni_result.value();
    if (best.cam.spec.slot == omniperception::OmniCameraSlot::left) {
      draw_omni_overlay(left_show, best);
    } else if (best.cam.spec.slot == omniperception::OmniCameraSlot::right) {
      draw_omni_overlay(right_show, best);
    } else {
      draw_omni_overlay(back_show, best);
    }
  }

  cv::Mat main_small = resize_for_view(f.main_img);
  cv::Mat left_small = resize_for_view(left_show);
  cv::Mat right_small = resize_for_view(right_show);
  cv::Mat back_small = resize_for_view(back_show);
  cv::Mat top_row, bottom_row, canvas;
  cv::hconcat(main_small, left_small, top_row);
  cv::hconcat(right_small, back_small, bottom_row);
  cv::vconcat(top_row, bottom_row, canvas);
  cv::imshow(window_title_, canvas);
  return cv::waitKey(1) != 'q';
}

void SentryRuntime::shutdown() { send_idle(); }

}  // namespace ovsentry
