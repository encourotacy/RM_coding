#include "app.hpp"

#include <algorithm>
#include <exception>
#include <vector>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include "auto_aim_helpers.hpp"
#include "io/gimbal/gimbal.hpp"
#include "io/ros2/ros2_gimbal.hpp"
#include "overlay.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
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

int run_ovsentry_app(int argc, char ** argv, AppMode mode)
{
  auto config = parse_runtime_config(argc, argv);
  if (!config) return 0;

  config->mode = mode;
  tools::logger()->info(
    "[OVSentry{}] inference devices: auto_aim={} omni={}", app_mode_name(mode),
    config->auto_aim_device, config->omni_device);

  OVSentryOmniMpc app(std::move(*config));
  return app.run();
}

OVSentryOmniMpc::~OVSentryOmniMpc() = default;

OVSentryOmniMpc::OVSentryOmniMpc(RuntimeConfig cfg)
: cfg_(std::move(cfg)),
  recorder_(30),
  gimbal_(std::make_unique<io::ROS2Gimbal>(cfg_.config_path)),
  armor_ignore_subscriber_(cfg_.auto_aim_ignore_topic, cfg_.auto_aim_ignore_msg_type),
  auto_aim_camera_(std::make_unique<io::Camera>(cfg_.config_path)),
  solver_(cfg_.config_path),
  tracker_(cfg_.config_path, solver_),
  aimer_(cfg_.config_path),
  shooter_(cfg_.config_path),
  planner_(cfg_.config_path),
  decider_(cfg_.config_path)
{
  init_mode_modules();
}

void OVSentryOmniMpc::init_mode_modules()
{
  if (uses_auto_aim_detect(cfg_.mode)) {
    yolo_auto_ = std::make_unique<auto_aim::YOLO>(cfg_.config_path, yolo_debug_, "auto_aim_device");
  }
  if (uses_buff(cfg_.mode)) {
    buff_detector_ = std::make_unique<auto_buff::Buff_Detector>(cfg_.config_path);
    buff_solver_ = std::make_unique<auto_buff::Solver>(cfg_.config_path);
    buff_aimer_ = std::make_unique<auto_buff::Aimer>(cfg_.config_path);
  }
  if (uses_omni(cfg_.mode)) {
    yolo_omni_left_ = std::make_unique<auto_aim::YOLO>(cfg_.config_path, yolo_debug_, "omni_device");
    yolo_omni_right_ = std::make_unique<auto_aim::YOLO>(cfg_.config_path, yolo_debug_, "omni_device");
    yolo_omni_back_ = std::make_unique<auto_aim::YOLO>(cfg_.config_path, yolo_debug_, "omni_device");
    cam_left_ = std::make_unique<io::USBCamera>(cfg_.left_cam.dev_name, cfg_.config_path);
    cam_right_ = std::make_unique<io::USBCamera>(cfg_.right_cam.dev_name, cfg_.config_path);
    cam_back_ = std::make_unique<io::USBCamera>(cfg_.back_cam.dev_name, cfg_.config_path);
    cam_left_->device_name = cfg_.left_cam.spec.label;
    cam_right_->device_name = cfg_.right_cam.spec.label;
    cam_back_->device_name = cfg_.back_cam.spec.label;
  }
}

const char * OVSentryOmniMpc::window_title() const
{
  switch (cfg_.mode) {
    case AppMode::AutoAim:
      return "ovsentry_auto_aim";
    case AppMode::Buff:
      return "ovsentry_buff";
    case AppMode::Omni:
      return "ovsentry_omni";
    case AppMode::AutoSwitch:
      return "ovsentry_mpc";
  }
  return "ovsentry_mpc";
}

void OVSentryOmniMpc::update_mode_flags()
{
  switch (cfg_.mode) {
    case AppMode::Buff:
      small_buff_mode_ = true;
      omni_mode_ = false;
      break;
    case AppMode::Omni:
      small_buff_mode_ = false;
      omni_mode_ = true;
      break;
    case AppMode::AutoAim:
      small_buff_mode_ = false;
      omni_mode_ = false;
      break;
    case AppMode::AutoSwitch:
    default:
      small_buff_mode_ = buff_request_subscriber_.requested();
      omni_mode_ = false;
      break;
  }
}

int OVSentryOmniMpc::run()
{
  while (!exiter_.exit()) {
    if (!read_main_frame()) continue;

    frame_count_++;
    q_ = gimbal_->imu_at_image(main_timestamp_);
    solver_.set_R_gimbal2world(q_);
    gimbal_state_ = gimbal_->state();
    ypr_ = tools::eulers(solver_.R_gimbal2world(), 2, 1, 0);

    update_mode_flags();
    detect_and_track();
    if (cfg_.mode == AppMode::AutoSwitch) {
      omni_mode_ = !small_buff_mode_ && tracker_state_ == "lost";
    }

    reset_frame_outputs();
    now_ = std::chrono::steady_clock::now();
    update_omni_session_on_mode_change();

    if (small_buff_mode_) {
      run_small_buff();
    } else if (omni_mode_) {
      run_omni();
    } else {
      run_auto_aim_mpc();
    }

    publish_telemetry();
    prev_omni_mode_ = omni_mode_;
    if (cfg_.display && !render_display()) break;
  }

  shutdown();
  return 0;
}

bool OVSentryOmniMpc::read_main_frame()
{
  try {
    auto_aim_camera_->read(main_img_, main_timestamp_);
    if (main_img_.empty()) return false;
  } catch (const std::exception & e) {
    tools::logger()->error(
      "[OVSentry{}] main camera read failed: {}", app_mode_name(cfg_.mode), e.what());
    return false;
  }
  return true;
}

void OVSentryOmniMpc::detect_and_track()
{
  t0_ = std::chrono::steady_clock::now();
  armor_target_mask_ = read_nav_armor_target_mask(armor_ignore_subscriber_);
  armors_.clear();
  targets_.clear();
  tracker_state_ = small_buff_mode_ ? "small_buff" : "idle";
  if (!small_buff_mode_ && yolo_auto_) {
    armors_ = yolo_auto_->detect(main_img_, frame_count_);
    decider_.armor_filter(armors_);
    decider_.set_priority(armors_);
    apply_armor_target_mask(armors_, armor_target_mask_);
    targets_ = tracker_.track(armors_, main_timestamp_);
    tracker_state_ = tracker_.state();
  }
  t1_ = std::chrono::steady_clock::now();
}

void OVSentryOmniMpc::reset_frame_outputs()
{
  best_omni_result_.reset();
  omni_target_abs_yaw_deg_.reset();
  omni_candidate_abs_yaw_deg_.reset();
  omni_candidate_base_big_yaw_deg_.reset();
  omni_candidate_age_ms_.reset();
  omni_candidate_delta_deg_.reset();
  omni_target_error_deg_.reset();
  omni_selected_confidence_.reset();
  omni_selected_priority_.reset();
  omni_selected_slot_.reset();
  omni_hold_applied_ = false;
  omni_retarget_blocked_ = false;
  omni_retarget_cd_active_ = false;
  omni_same_target_continuation_ = false;
  omni_target_reached_ = false;
  omni_cmd_timeout_active_ = false;
  omni_cmd_timed_out_ = false;
  omni_block_reason_ = "none";
  omni_cmd_elapsed_ms_ = 0.0;
  omni_retarget_remaining_ms_ = 0.0;
  command_ = io::Command{false, false, 0.0, 0.0};
  buff_power_rune_.reset();
  buff_plan_ = auto_aim::Plan{false, false, 0, 0, 0, 0, 0, 0, 0, 0};
  buff_detect_ms_ = 0.0;
  buff_target_ready_ = false;
  buff_command_held_ = false;
}

void OVSentryOmniMpc::clear_omni_timeout_session()
{
  active_omni_timeout_target_.reset();
  active_omni_timeout_started_at_ = std::chrono::steady_clock::time_point{};
  active_omni_timeout_running_ = false;
}

void OVSentryOmniMpc::clear_omni_redirect_state()
{
  omni_hold_command_.reset();
  session_accepted_omni_target_.reset();
  cooldown_anchor_omni_target_.reset();
  omni_retarget_cooldown_deadline_ = std::chrono::steady_clock::time_point{};
  clear_omni_timeout_session();
}

void OVSentryOmniMpc::update_omni_session_on_mode_change()
{
  if (cooldown_anchor_omni_target_.has_value() && now_ >= omni_retarget_cooldown_deadline_) {
    cooldown_anchor_omni_target_.reset();
    omni_retarget_cooldown_deadline_ = std::chrono::steady_clock::time_point{};
  }

  if (omni_mode_ && !prev_omni_mode_) {
    clear_omni_redirect_state();
  } else if (!omni_mode_ && prev_omni_mode_) {
    clear_omni_redirect_state();
  }
}

void OVSentryOmniMpc::run_small_buff()
{
  left_img_.release();
  right_img_.release();
  back_img_.release();
  clear_omni_redirect_state();
  if (!buff_detector_ || !buff_solver_ || !buff_aimer_) {
    gimbal_->send_mpc(false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0);
    return;
  }

  buff_solver_->set_R_gimbal2world(q_);
  const auto t_buff0 = std::chrono::steady_clock::now();
  buff_power_rune_ = buff_detector_->detect(main_img_);
  const auto t_buff1 = std::chrono::steady_clock::now();
  buff_detect_ms_ = tools::delta_time(t_buff1, t_buff0) * 1e3;
  buff_solver_->solve(buff_power_rune_);
  buff_small_target_.get_target(buff_power_rune_, main_timestamp_);
  buff_target_ready_ = !buff_small_target_.is_unsolve();

  auto buff_target_copy = buff_small_target_;
  buff_plan_ = buff_aimer_->mpc_aim(
    buff_target_copy, main_timestamp_, make_buff_gimbal_state(gimbal_state_), true);

  command_.control = buff_plan_.control;
  command_.shoot = buff_plan_.fire;
  command_.yaw = tools::limit_rad(buff_plan_.yaw);
  command_.pitch = buff_plan_.pitch;
  command_.big_yaw = nearest_continuous_yaw_rad(command_.yaw, gimbal_state_.big_yaw);
  command_.small_yaw = command_.yaw;
  command_.has_target_yaw = command_.control;

  if (command_.control) {
    buff_hold_command_ = command_;
    buff_last_control_at_ = now_;
  } else if (
    buff_hold_command_.has_value() && now_ - buff_last_control_at_ <= cfg_.buff_lost_cmd_hold) {
    command_ = buff_hold_command_.value();
    command_.shoot = false;
    buff_command_held_ = true;
  }

  gimbal_->send_mpc(
    command_.control, command_.shoot, command_.big_yaw, command_.small_yaw, command_.pitch,
    buff_command_held_ ? 0.0 : buff_plan_.yaw_vel, buff_command_held_ ? 0.0 : buff_plan_.pitch_vel,
    buff_command_held_ ? 0.0 : buff_plan_.yaw_acc, buff_command_held_ ? 0.0 : buff_plan_.pitch_acc, 0,
    0.0, 0.0, 0.0);
}

OmniCandidateFrame OVSentryOmniMpc::read_omni_frame(
  io::USBCamera & camera, cv::Mat & img, std::chrono::steady_clock::time_point & ts,
  const OmniCamConfig & cam_cfg)
{
  OmniCandidateFrame frame;
  frame.result.cam = cam_cfg;
  const bool ok = camera.read_with_timeout(img, ts, cfg_.omni_read_timeout);
  if (!ok || img.empty()) {
    img.release();
    return frame;
  }
  frame.timestamp = ts;
  frame.base_big_yaw_rad = gimbal_->big_yaw_at_image(ts);
  frame.has_base_big_yaw = true;
  return frame;
}

void OVSentryOmniMpc::finalize_omni_frame(
  OmniCandidateFrame & frame, const OmniCamConfig & cam_cfg, double infer_ms)
{
  frame.result.infer_ms = infer_ms;
  decider_.armor_filter(frame.result.armors);
  decider_.set_priority(frame.result.armors);
  apply_armor_target_mask(frame.result.armors, armor_target_mask_);
  frame.result.top_armor = pick_top_armor(frame.result.armors);
  if (frame.result.top_armor.has_value()) {
    auto [dyaw, dpitch] = calc_delta_angle_deg(frame.result.top_armor.value(), cam_cfg);
    frame.result.delta_yaw_deg = dyaw;
    frame.result.delta_pitch_deg = dpitch;
    frame.candidate = build_omni_candidate(frame.result, frame.timestamp, frame.base_big_yaw_rad);
  }
}

void OVSentryOmniMpc::run_omni()
{
  buff_hold_command_.reset();
  if (!cam_left_ || !cam_right_ || !cam_back_ || !yolo_omni_left_ || !yolo_omni_right_ ||
      !yolo_omni_back_) {
    gimbal_->send_mpc(false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0);
    return;
  }

  auto left_frame = read_omni_frame(*cam_left_, left_img_, ts_left_, cfg_.left_cam);
  auto right_frame = read_omni_frame(*cam_right_, right_img_, ts_right_, cfg_.right_cam);
  auto back_frame = read_omni_frame(*cam_back_, back_img_, ts_back_, cfg_.back_cam);

  auto t_omni0 = std::chrono::steady_clock::now();
  if (left_frame.has_base_big_yaw && !left_img_.empty()) {
    left_frame.result.armors = yolo_omni_left_->detect(left_img_, frame_count_);
  }
  auto t_omni1 = std::chrono::steady_clock::now();
  if (right_frame.has_base_big_yaw && !right_img_.empty()) {
    right_frame.result.armors = yolo_omni_right_->detect(right_img_, frame_count_);
  }
  auto t_omni2 = std::chrono::steady_clock::now();
  if (back_frame.has_base_big_yaw && !back_img_.empty()) {
    back_frame.result.armors = yolo_omni_back_->detect(back_img_, frame_count_);
  }
  auto t_omni3 = std::chrono::steady_clock::now();

  finalize_omni_frame(left_frame, cfg_.left_cam, tools::delta_time(t_omni1, t_omni0) * 1e3);
  finalize_omni_frame(right_frame, cfg_.right_cam, tools::delta_time(t_omni2, t_omni1) * 1e3);
  finalize_omni_frame(back_frame, cfg_.back_cam, tools::delta_time(t_omni3, t_omni2) * 1e3);

  omni_retarget_cd_active_ =
    cooldown_anchor_omni_target_.has_value() && now_ < omni_retarget_cooldown_deadline_;
  if (omni_retarget_cd_active_) {
    omni_retarget_remaining_ms_ =
      std::chrono::duration<double, std::milli>(omni_retarget_cooldown_deadline_ - now_).count();
  }
  const auto reference_omni_target = omniperception::select_omni_retarget_reference_target(
    session_accepted_omni_target_, cooldown_anchor_omni_target_, omni_retarget_cd_active_);

  std::vector<OmniCandidateFrame> candidate_frames;
  if (left_frame.candidate.has_value()) candidate_frames.push_back(left_frame);
  if (right_frame.candidate.has_value()) candidate_frames.push_back(right_frame);
  if (back_frame.candidate.has_value()) candidate_frames.push_back(back_frame);

  std::vector<omniperception::OmniCandidate> candidates;
  candidates.reserve(candidate_frames.size());
  for (const auto & frame : candidate_frames) candidates.push_back(frame.candidate.value());

  const auto selected_candidate = omniperception::select_omni_candidate(
    candidates, reference_omni_target, gimbal_state_.big_yaw, cfg_.omni_retarget_min_delta_deg);

  if (selected_candidate.has_value()) {
    omni_candidate_abs_yaw_deg_ = selected_candidate->abs_yaw_rad * 57.3;
    omni_candidate_base_big_yaw_deg_ = selected_candidate->base_big_yaw_rad * 57.3;
    omni_candidate_age_ms_ = tools::delta_time(now_, selected_candidate->timestamp) * 1e3;
    omni_selected_confidence_ = selected_candidate->confidence;
    omni_selected_priority_ = static_cast<int>(selected_candidate->priority);
    omni_selected_slot_ = slot_name(selected_candidate->slot);

    const auto selected_frame = std::find_if(
      candidate_frames.begin(), candidate_frames.end(), [&](const OmniCandidateFrame & frame) {
        return same_candidate_frame(frame, selected_candidate.value());
      });
    if (selected_frame != candidate_frames.end()) best_omni_result_ = selected_frame->result;

    const auto decision = omniperception::evaluate_omni_retarget(
      selected_candidate.value(), reference_omni_target, gimbal_state_.big_yaw,
      omni_retarget_cd_active_, cfg_.omni_retarget_min_delta_deg);
    omni_candidate_delta_deg_ = decision.candidate_delta_deg;
    omni_same_target_continuation_ = decision.same_target_continuation;

    if (decision.accept) {
      command_ = selected_candidate->command;
      omni_hold_command_ = command_;
      const auto accepted_target = make_accepted_omni_target(selected_candidate.value());
      session_accepted_omni_target_ = accepted_target;
      if (
        !active_omni_timeout_running_ || !active_omni_timeout_target_.has_value() ||
        !same_omni_target_continuation(
          active_omni_timeout_target_.value(), accepted_target, cfg_.omni_retarget_min_delta_deg)) {
        active_omni_timeout_started_at_ = now_;
        active_omni_timeout_running_ = true;
      }
      active_omni_timeout_target_ = accepted_target;
      omni_target_abs_yaw_deg_ = command_.big_yaw * 57.3;
      if (omniperception::should_start_omni_retarget_cooldown(
            decision, cfg_.omni_retarget_min_delta_deg)) {
        cooldown_anchor_omni_target_ = accepted_target;
        omni_retarget_cooldown_deadline_ = now_ + cfg_.omni_retarget_cooldown;
        omni_retarget_cd_active_ = true;
        omni_retarget_remaining_ms_ = cfg_.omni_retarget_cooldown_s * 1e3;
      }
    } else if (reference_omni_target.has_value()) {
      command_ = reference_omni_target->command;
      omni_hold_command_ = command_;
      omni_target_abs_yaw_deg_ = command_.big_yaw * 57.3;
      omni_retarget_blocked_ = true;
      omni_block_reason_ = decision.block_reason;
    }
  } else if (omni_hold_command_.has_value()) {
    const double target_error_deg =
      angular_distance_deg(omni_hold_command_->big_yaw, gimbal_state_.big_yaw);
    if (target_error_deg > cfg_.omni_hold_release_tolerance_deg) {
      command_ = omni_hold_command_.value();
      omni_target_abs_yaw_deg_ = command_.big_yaw * 57.3;
      omni_hold_applied_ = true;
    } else {
      omni_hold_command_.reset();
    }
  } else {
    omni_hold_command_.reset();
    clear_omni_timeout_session();
  }

  if (command_.control && command_.has_target_yaw) {
    if (active_omni_timeout_running_ && active_omni_timeout_target_.has_value()) {
      omni_cmd_timeout_active_ = true;
      omni_cmd_elapsed_ms_ =
        std::chrono::duration<double, std::milli>(now_ - active_omni_timeout_started_at_).count();
    }

    const double target_error_deg = angular_distance_deg(command_.big_yaw, gimbal_state_.big_yaw);
    if (target_error_deg > cfg_.omni_hold_release_tolerance_deg) {
      if (
        active_omni_timeout_running_ &&
        (now_ - active_omni_timeout_started_at_) > cfg_.omni_command_timeout) {
        tools::logger()->warn(
          "[OVSentry{}] omni command timed out after {:.0f}ms without reaching target yaw",
          app_mode_name(cfg_.mode),
          omni_cmd_elapsed_ms_);
        command_ = io::Command{false, false, 0.0, 0.0};
        omni_target_abs_yaw_deg_.reset();
        omni_hold_applied_ = false;
        omni_cmd_timed_out_ = true;
        omni_cmd_timeout_active_ = false;
        clear_omni_redirect_state();
      }
    } else {
      clear_omni_timeout_session();
    }
  } else {
    clear_omni_timeout_session();
  }

  if (command_.control && command_.has_target_yaw) {
    omni_target_error_deg_ = angular_distance_deg(command_.big_yaw, gimbal_state_.big_yaw);
    omni_target_reached_ = omni_target_error_deg_.value() <= cfg_.omni_hold_release_tolerance_deg;
    if (omni_hold_command_.has_value() && omni_target_reached_) {
      omni_hold_command_.reset();
    }
  }

  const double omni_big_yaw = command_.has_target_yaw ? command_.big_yaw : command_.yaw;
  const double omni_small_yaw = command_.has_target_yaw ? command_.small_yaw : command_.yaw;
  gimbal_->send_mpc(
    command_.control, command_.shoot, omni_big_yaw, omni_small_yaw, command_.pitch, 0.0, 0.0, 0.0,
    0.0, static_cast<uint8_t>(command_.armor_id), 0.0, 0.0, 0.0);
}

void OVSentryOmniMpc::run_auto_aim_mpc()
{
  left_img_.release();
  right_img_.release();
  back_img_.release();
  clear_omni_redirect_state();
  buff_hold_command_.reset();

  const bool armor_acquiring = tracker_state_ == "detecting";
  if (armor_acquiring) {
    command_.control = true;
    command_.shoot = false;
    command_.yaw = gimbal_state_.yaw;
    command_.pitch = -gimbal_state_.pitch;
    command_.big_yaw = gimbal_state_.big_yaw;
    command_.small_yaw = gimbal_state_.yaw;
    command_.has_target_yaw = true;
  } else {
    command_ = aimer_.aim(targets_, main_timestamp_, gimbal_->bullet_speed(), aimer_to_now_);
    if (command_.control && !targets_.empty()) {
      apply_sentry_tracking_yaws(command_, targets_.front(), gimbal_state_.big_yaw);
    }
    command_.shoot = shooter_.shoot(command_, aimer_, targets_, ypr_, tracker_state_ == "tracking");
    fill_nav_target_info(command_, targets_);
  }

  const bool outpost_convergence = !targets_.empty() &&
                                   targets_.front().name == auto_aim::ArmorName::outpost &&
                                   (!targets_.front().convergened() || targets_.front().diverged());
  const bool static_outpost_direct =
    !targets_.empty() && targets_.front().outpost_static_direct_active();
  if (!armor_acquiring && outpost_convergence && !static_outpost_direct) {
    command_ = io::Command{false, false, 0.0, 0.0};
  }

  const bool unlocked_outpost = !targets_.empty() && is_unlocked_outpost_target(targets_.front());
  double small_yaw_vel = 0.0;
  double pitch_vel = 0.0;
  double small_yaw_acc = 0.0;
  double pitch_acc = 0.0;
  if (command_.control && !armor_acquiring && !targets_.empty() && !unlocked_outpost) {
    const auto mpc_plan = planner_.plan(targets_.front(), gimbal_->bullet_speed());
    if (mpc_plan.control) {
      small_yaw_vel = mpc_plan.yaw_vel;
      pitch_vel = mpc_plan.pitch_vel;
      small_yaw_acc = mpc_plan.yaw_acc;
      pitch_acc = mpc_plan.pitch_acc;
    }
  }

  const double big_yaw = command_.has_target_yaw ? command_.big_yaw : command_.yaw;
  const double small_yaw = command_.has_target_yaw ? command_.small_yaw : command_.yaw;
  gimbal_->send_mpc(
    command_.control, command_.shoot, big_yaw, small_yaw, command_.pitch, small_yaw_vel, pitch_vel,
    small_yaw_acc, pitch_acc, static_cast<uint8_t>(command_.armor_id), command_.vx, command_.vy,
    command_.horizon_distance);
}

void OVSentryOmniMpc::publish_telemetry()
{
  nlohmann::json data;
  data["app_mode"] = app_mode_name(cfg_.mode);
  data["mode"] = small_buff_mode_ ? 2 : (omni_mode_ ? 1 : 0);
  data["gimbal_mode"] = io::MODES[static_cast<int>(gimbal_->mode())];
  data["armor_num"] = armors_.size();
  data["tracker_state"] = tracker_state_;
  data["armor_acquiring"] = (!small_buff_mode_ && tracker_state_ == "detecting") ? 1 : 0;
  data["gimbal_yaw"] = ypr_[0] * 57.3;
  data["gimbal_small_yaw"] = gimbal_state_.yaw * 57.3;
  data["gimbal_big_yaw"] = gimbal_state_.big_yaw * 57.3;
  data["bullet_speed"] = gimbal_->bullet_speed();
  data["mpc_control"] = command_.control ? 1 : 0;
  data["mpc_fire"] = command_.shoot ? 1 : 0;
  data["mpc_yaw"] = (command_.has_target_yaw ? command_.small_yaw : command_.yaw) * 57.3;
  data["mpc_pitch"] = command_.pitch * 57.3;
  data["target_armor_id"] = static_cast<int>(command_.armor_id);
  data["target_vx"] = command_.vx;
  data["target_vy"] = command_.vy;
  data["horizon_distance"] = command_.horizon_distance;
  data["aim_source"] =
    (!small_buff_mode_ && !omni_mode_ && command_.control) ? aimer_.debug_aim_point.source : -1;
  data["aim_armor_id"] =
    (!small_buff_mode_ && !omni_mode_ && command_.control) ? aimer_.debug_aim_point.armor_id : -1;
  if (small_buff_mode_) {
    data["buff_detected"] = buff_power_rune_.has_value() ? 1 : 0;
    data["buff_target_ready"] = buff_target_ready_ ? 1 : 0;
    data["buff_command_held"] = buff_command_held_ ? 1 : 0;
    data["buff_detect_ms"] = buff_detect_ms_;
    data["buff_yaw"] = buff_plan_.yaw * 57.3;
    data["buff_pitch"] = buff_plan_.pitch * 57.3;
    data["buff_yaw_vel"] = buff_plan_.yaw_vel * 57.3;
    data["buff_pitch_vel"] = buff_plan_.pitch_vel * 57.3;
    if (buff_power_rune_.has_value()) {
      data["buff_r_yaw"] = buff_power_rune_->ypd_in_world[0] * 57.3;
      data["buff_r_pitch"] = buff_power_rune_->ypd_in_world[1] * 57.3;
      data["buff_r_distance"] = buff_power_rune_->ypd_in_world[2];
      data["buff_roll"] = buff_power_rune_->ypr_in_world[2] * 57.3;
    }
    if (buff_target_ready_) {
      const auto x = buff_small_target_.ekf_x();
      data["buff_target_roll"] = x[5] * 57.3;
      data["buff_target_spd"] = x[6] * 57.3;
    }
  }
  if (!targets_.empty()) {
    const auto & target = targets_.front();
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
    if (target.name == auto_aim::ArmorName::outpost && aimer_.debug_aim_point.valid) {
      const auto x = target.ekf_x();
      const double center_yaw = std::atan2(x[2], x[0]);
      data["outpost_aim_phase_deg"] =
        std::abs(tools::limit_rad(aimer_.debug_aim_point.xyza[3] - center_yaw)) * 57.3;
    }
  }
  data["omni_yaw_hold"] = omni_hold_applied_ ? 1 : 0;
  data["omni_target_reached"] = omni_target_reached_ ? 1 : 0;
  data["omni_cmd_timeout_active"] = omni_cmd_timeout_active_ ? 1 : 0;
  data["omni_cmd_timed_out"] = omni_cmd_timed_out_ ? 1 : 0;
  data["omni_cmd_elapsed_ms"] = omni_cmd_elapsed_ms_;
  data["omni_retarget_cd_active"] = omni_retarget_cd_active_ ? 1 : 0;
  data["omni_retarget_blocked"] = omni_retarget_blocked_ ? 1 : 0;
  data["omni_retarget_remaining_ms"] = omni_retarget_remaining_ms_;
  data["omni_same_target_continuation"] = omni_same_target_continuation_ ? 1 : 0;
  data["omni_block_reason"] = omni_block_reason_;
  if (omni_target_abs_yaw_deg_.has_value()) data["omni_target_yaw"] = omni_target_abs_yaw_deg_.value();
  if (omni_candidate_abs_yaw_deg_.has_value()) {
    data["omni_candidate_abs_yaw"] = omni_candidate_abs_yaw_deg_.value();
  }
  if (omni_candidate_base_big_yaw_deg_.has_value()) {
    data["omni_candidate_base_big_yaw"] = omni_candidate_base_big_yaw_deg_.value();
  }
  if (omni_candidate_age_ms_.has_value()) data["omni_candidate_age_ms"] = omni_candidate_age_ms_.value();
  if (omni_candidate_delta_deg_.has_value()) {
    data["omni_candidate_delta_deg"] = omni_candidate_delta_deg_.value();
  }
  if (omni_target_error_deg_.has_value()) data["omni_target_error_deg"] = omni_target_error_deg_.value();
  if (omni_selected_confidence_.has_value()) {
    data["omni_selected_confidence"] = omni_selected_confidence_.value();
  }
  if (omni_selected_priority_.has_value()) {
    data["omni_selected_priority"] = omni_selected_priority_.value();
  }
  if (omni_selected_slot_.has_value()) data["omni_selected_slot"] = omni_selected_slot_.value();
  data["yolo_time"] = tools::delta_time(t1_, t0_) * 1e3;
  plotter_.plot(data);
}

bool OVSentryOmniMpc::render_display()
{
  if (small_buff_mode_ && buff_solver_) {
    draw_small_buff_overlay(
      main_img_, buff_power_rune_, buff_small_target_, *buff_solver_, buff_plan_);
  } else {
    draw_auto_aim_overlay(main_img_, targets_, aimer_, solver_);
    tools::draw_text(
      main_img_, fmt::format("[{}] mode={}", tracker_state_, omni_mode_ ? "OMNI" : "MPC"), {10, 30},
      {255, 255, 255}, 0.8, 2);
    tools::draw_text(
      main_img_,
      fmt::format(
        "mpc yaw={:.2f} pitch={:.2f} fire={}",
        (command_.has_target_yaw ? command_.small_yaw : command_.yaw) * 57.3, command_.pitch * 57.3,
        command_.shoot ? 1 : 0),
      {10, 60}, {154, 50, 205}, 0.8, 2);
  }
  if (omni_target_abs_yaw_deg_.has_value()) {
    tools::draw_text(
      main_img_, fmt::format("omni target yaw={:.2f}", omni_target_abs_yaw_deg_.value()), {10, 90},
      {0, 255, 255}, 0.8, 2);
  }
  if (omni_retarget_cd_active_) {
    tools::draw_text(
      main_img_, fmt::format("omni retarget cd {:.0f}ms", omni_retarget_remaining_ms_), {10, 120},
      omni_retarget_blocked_ ? cv::Scalar(0, 180, 255) : cv::Scalar(255, 220, 0), 0.8, 2);
  }
  if (omni_cmd_timeout_active_ || omni_cmd_timed_out_) {
    tools::draw_text(
      main_img_,
      fmt::format(
        "omni cmd timeout {:.0f}/{:.0f}ms hit={}", omni_cmd_elapsed_ms_,
        cfg_.omni_command_timeout_s * 1e3, omni_cmd_timed_out_ ? 1 : 0),
      {10, 150}, omni_cmd_timed_out_ ? cv::Scalar(0, 120, 255) : cv::Scalar(180, 255, 180), 0.8, 2);
  }
  if (omni_target_error_deg_.has_value()) {
    tools::draw_text(
      main_img_,
      fmt::format(
        "omni target err={:.1f} reached={}", omni_target_error_deg_.value(),
        omni_target_reached_ ? 1 : 0),
      {10, 180}, {180, 255, 180}, 0.8, 2);
  }

  const char * title = window_title();
  if (!uses_omni(cfg_.mode)) {
    cv::imshow(title, resize_for_view(main_img_));
    return cv::waitKey(1) != 'q';
  }

  cv::Mat left_show =
    left_img_.empty() ? cv::Mat::zeros(main_img_.size(), main_img_.type()) : left_img_.clone();
  cv::Mat right_show =
    right_img_.empty() ? cv::Mat::zeros(main_img_.size(), main_img_.type()) : right_img_.clone();
  cv::Mat back_show =
    back_img_.empty() ? cv::Mat::zeros(main_img_.size(), main_img_.type()) : back_img_.clone();

  if (omni_mode_ && best_omni_result_.has_value()) {
    const auto & best = best_omni_result_.value();
    if (best.cam.spec.slot == omniperception::OmniCameraSlot::left) {
      draw_omni_overlay(left_show, best);
    } else if (best.cam.spec.slot == omniperception::OmniCameraSlot::right) {
      draw_omni_overlay(right_show, best);
    } else {
      draw_omni_overlay(back_show, best);
    }
  }

  cv::Mat main_small = resize_for_view(main_img_);
  cv::Mat left_small = resize_for_view(left_show);
  cv::Mat right_small = resize_for_view(right_show);
  cv::Mat back_small = resize_for_view(back_show);
  cv::Mat top_row, bottom_row, canvas;
  cv::hconcat(main_small, left_small, top_row);
  cv::hconcat(right_small, back_small, bottom_row);
  cv::vconcat(top_row, bottom_row, canvas);
  cv::imshow(title, canvas);
  return cv::waitKey(1) != 'q';
}

void OVSentryOmniMpc::shutdown()
{
  gimbal_->send_mpc(false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0);
}

}  // namespace ovsentry
