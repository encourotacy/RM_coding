#include "omni_task.hpp"

#include <algorithm>
#include <vector>

#include "auto_aim_helpers.hpp"
#include "io/ros2/ros2_gimbal.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace ovsentry
{

OmniTask::OmniTask(SentryRuntime & rt)
: rt_(rt),
  yolo_left_(std::make_unique<auto_aim::YOLO>(rt.cfg().config_path, yolo_debug_, "omni_device")),
  yolo_right_(std::make_unique<auto_aim::YOLO>(rt.cfg().config_path, yolo_debug_, "omni_device")),
  yolo_back_(std::make_unique<auto_aim::YOLO>(rt.cfg().config_path, yolo_debug_, "omni_device")),
  cam_left_(std::make_unique<io::USBCamera>(rt.cfg().left_cam.dev_name, rt.cfg().config_path)),
  cam_right_(std::make_unique<io::USBCamera>(rt.cfg().right_cam.dev_name, rt.cfg().config_path)),
  cam_back_(std::make_unique<io::USBCamera>(rt.cfg().back_cam.dev_name, rt.cfg().config_path))
{
  cam_left_->device_name = rt.cfg().left_cam.spec.label;
  cam_right_->device_name = rt.cfg().right_cam.spec.label;
  cam_back_->device_name = rt.cfg().back_cam.spec.label;
}

void OmniTask::clear_timeout_session()
{
  timeout_target_.reset();
  timeout_started_at_ = std::chrono::steady_clock::time_point{};
  timeout_running_ = false;
}

void OmniTask::clear_redirect_state()
{
  hold_command_.reset();
  session_accepted_target_.reset();
  cooldown_anchor_target_.reset();
  retarget_cooldown_deadline_ = std::chrono::steady_clock::time_point{};
  clear_timeout_session();
}

void OmniTask::reset() { clear_redirect_state(); }

void OmniTask::prepare(bool omni_mode)
{
  auto & f = rt_.frame();
  if (cooldown_anchor_target_.has_value() && f.now >= retarget_cooldown_deadline_) {
    cooldown_anchor_target_.reset();
    retarget_cooldown_deadline_ = std::chrono::steady_clock::time_point{};
  }

  if (omni_mode != prev_omni_mode_) clear_redirect_state();
  prev_omni_mode_ = omni_mode;
}

OmniCandidateFrame OmniTask::read_frame(
  io::USBCamera & camera, cv::Mat & img, std::chrono::steady_clock::time_point & ts,
  const OmniCamConfig & cam_cfg)
{
  OmniCandidateFrame frame;
  frame.result.cam = cam_cfg;
  const bool ok = camera.read_with_timeout(img, ts, rt_.cfg().omni_read_timeout);
  if (!ok || img.empty()) {
    img.release();
    return frame;
  }
  frame.timestamp = ts;
  frame.base_big_yaw_rad = rt_.gimbal().big_yaw_at_image(ts);
  frame.has_base_big_yaw = true;
  return frame;
}

void OmniTask::finalize_frame(
  OmniCandidateFrame & frame, const OmniCamConfig & cam_cfg, double infer_ms)
{
  auto & f = rt_.frame();
  frame.result.infer_ms = infer_ms;
  rt_.decider().armor_filter(frame.result.armors);
  rt_.decider().set_priority(frame.result.armors);
  apply_armor_target_mask(frame.result.armors, f.armor_target_mask);
  frame.result.top_armor = pick_top_armor(frame.result.armors);
  if (frame.result.top_armor.has_value()) {
    auto [dyaw, dpitch] = calc_delta_angle_deg(frame.result.top_armor.value(), cam_cfg);
    frame.result.delta_yaw_deg = dyaw;
    frame.result.delta_pitch_deg = dpitch;
    frame.candidate = build_omni_candidate(frame.result, frame.timestamp, frame.base_big_yaw_rad);
  }
}

void OmniTask::run()
{
  auto & f = rt_.frame();
  const auto & cfg = rt_.cfg();
  if (!cam_left_ || !cam_right_ || !cam_back_ || !yolo_left_ || !yolo_right_ || !yolo_back_) {
    rt_.send_idle();
    return;
  }

  auto left_frame = read_frame(*cam_left_, f.left_img, f.ts_left, cfg.left_cam);
  auto right_frame = read_frame(*cam_right_, f.right_img, f.ts_right, cfg.right_cam);
  auto back_frame = read_frame(*cam_back_, f.back_img, f.ts_back, cfg.back_cam);

  auto t_omni0 = std::chrono::steady_clock::now();
  if (left_frame.has_base_big_yaw && !f.left_img.empty()) {
    left_frame.result.armors = yolo_left_->detect(f.left_img, f.frame_count);
  }
  auto t_omni1 = std::chrono::steady_clock::now();
  if (right_frame.has_base_big_yaw && !f.right_img.empty()) {
    right_frame.result.armors = yolo_right_->detect(f.right_img, f.frame_count);
  }
  auto t_omni2 = std::chrono::steady_clock::now();
  if (back_frame.has_base_big_yaw && !f.back_img.empty()) {
    back_frame.result.armors = yolo_back_->detect(f.back_img, f.frame_count);
  }
  auto t_omni3 = std::chrono::steady_clock::now();

  finalize_frame(left_frame, cfg.left_cam, tools::delta_time(t_omni1, t_omni0) * 1e3);
  finalize_frame(right_frame, cfg.right_cam, tools::delta_time(t_omni2, t_omni1) * 1e3);
  finalize_frame(back_frame, cfg.back_cam, tools::delta_time(t_omni3, t_omni2) * 1e3);

  f.omni_retarget_cd_active =
    cooldown_anchor_target_.has_value() && f.now < retarget_cooldown_deadline_;
  if (f.omni_retarget_cd_active) {
    f.omni_retarget_remaining_ms =
      std::chrono::duration<double, std::milli>(retarget_cooldown_deadline_ - f.now).count();
  }
  const auto reference_omni_target = omniperception::select_omni_retarget_reference_target(
    session_accepted_target_, cooldown_anchor_target_, f.omni_retarget_cd_active);

  std::vector<OmniCandidateFrame> candidate_frames;
  if (left_frame.candidate.has_value()) candidate_frames.push_back(left_frame);
  if (right_frame.candidate.has_value()) candidate_frames.push_back(right_frame);
  if (back_frame.candidate.has_value()) candidate_frames.push_back(back_frame);

  std::vector<omniperception::OmniCandidate> candidates;
  candidates.reserve(candidate_frames.size());
  for (const auto & frame : candidate_frames) candidates.push_back(frame.candidate.value());

  const auto selected_candidate = omniperception::select_omni_candidate(
    candidates, reference_omni_target, f.gimbal_state.big_yaw, cfg.omni_retarget_min_delta_deg);

  if (selected_candidate.has_value()) {
    f.omni_candidate_abs_yaw_deg = selected_candidate->abs_yaw_rad * 57.3;
    f.omni_candidate_base_big_yaw_deg = selected_candidate->base_big_yaw_rad * 57.3;
    f.omni_candidate_age_ms = tools::delta_time(f.now, selected_candidate->timestamp) * 1e3;
    f.omni_selected_confidence = selected_candidate->confidence;
    f.omni_selected_priority = static_cast<int>(selected_candidate->priority);
    f.omni_selected_slot = slot_name(selected_candidate->slot);

    const auto selected_frame = std::find_if(
      candidate_frames.begin(), candidate_frames.end(), [&](const OmniCandidateFrame & frame) {
        return same_candidate_frame(frame, selected_candidate.value());
      });
    if (selected_frame != candidate_frames.end()) f.best_omni_result = selected_frame->result;

    const auto decision = omniperception::evaluate_omni_retarget(
      selected_candidate.value(), reference_omni_target, f.gimbal_state.big_yaw,
      f.omni_retarget_cd_active, cfg.omni_retarget_min_delta_deg);
    f.omni_candidate_delta_deg = decision.candidate_delta_deg;
    f.omni_same_target_continuation = decision.same_target_continuation;

    if (decision.accept) {
      f.command = selected_candidate->command;
      hold_command_ = f.command;
      const auto accepted_target = make_accepted_omni_target(selected_candidate.value());
      session_accepted_target_ = accepted_target;
      if (
        !timeout_running_ || !timeout_target_.has_value() ||
        !same_omni_target_continuation(
          timeout_target_.value(), accepted_target, cfg.omni_retarget_min_delta_deg)) {
        timeout_started_at_ = f.now;
        timeout_running_ = true;
      }
      timeout_target_ = accepted_target;
      f.omni_target_abs_yaw_deg = f.command.big_yaw * 57.3;
      if (omniperception::should_start_omni_retarget_cooldown(
            decision, cfg.omni_retarget_min_delta_deg)) {
        cooldown_anchor_target_ = accepted_target;
        retarget_cooldown_deadline_ = f.now + cfg.omni_retarget_cooldown;
        f.omni_retarget_cd_active = true;
        f.omni_retarget_remaining_ms = cfg.omni_retarget_cooldown_s * 1e3;
      }
    } else if (reference_omni_target.has_value()) {
      f.command = reference_omni_target->command;
      hold_command_ = f.command;
      f.omni_target_abs_yaw_deg = f.command.big_yaw * 57.3;
      f.omni_retarget_blocked = true;
      f.omni_block_reason = decision.block_reason;
    }
  } else if (hold_command_.has_value()) {
    const double target_error_deg =
      angular_distance_deg(hold_command_->big_yaw, f.gimbal_state.big_yaw);
    if (target_error_deg > cfg.omni_hold_release_tolerance_deg) {
      f.command = hold_command_.value();
      f.omni_target_abs_yaw_deg = f.command.big_yaw * 57.3;
      f.omni_hold_applied = true;
    } else {
      hold_command_.reset();
    }
  } else {
    hold_command_.reset();
    clear_timeout_session();
  }

  if (f.command.control && f.command.has_target_yaw) {
    if (timeout_running_ && timeout_target_.has_value()) {
      f.omni_cmd_timeout_active = true;
      f.omni_cmd_elapsed_ms =
        std::chrono::duration<double, std::milli>(f.now - timeout_started_at_).count();
    }

    const double target_error_deg = angular_distance_deg(f.command.big_yaw, f.gimbal_state.big_yaw);
    if (target_error_deg > cfg.omni_hold_release_tolerance_deg) {
      if (timeout_running_ && (f.now - timeout_started_at_) > cfg.omni_command_timeout) {
        tools::logger()->warn(
          "[OVSentry{}] omni command timed out after {:.0f}ms without reaching target yaw",
          app_mode_name(cfg.mode), f.omni_cmd_elapsed_ms);
        f.command = io::Command{false, false, 0.0, 0.0};
        f.omni_target_abs_yaw_deg.reset();
        f.omni_hold_applied = false;
        f.omni_cmd_timed_out = true;
        f.omni_cmd_timeout_active = false;
        clear_redirect_state();
      }
    } else {
      clear_timeout_session();
    }
  } else {
    clear_timeout_session();
  }

  if (f.command.control && f.command.has_target_yaw) {
    f.omni_target_error_deg = angular_distance_deg(f.command.big_yaw, f.gimbal_state.big_yaw);
    f.omni_target_reached = f.omni_target_error_deg.value() <= cfg.omni_hold_release_tolerance_deg;
    if (hold_command_.has_value() && f.omni_target_reached) {
      hold_command_.reset();
    }
  }

  const double omni_big_yaw = f.command.has_target_yaw ? f.command.big_yaw : f.command.yaw;
  const double omni_small_yaw = f.command.has_target_yaw ? f.command.small_yaw : f.command.yaw;
  rt_.gimbal().send_mpc(
    f.command.control, f.command.shoot, omni_big_yaw, omni_small_yaw, f.command.pitch, 0.0, 0.0,
    0.0, 0.0, static_cast<uint8_t>(f.command.armor_id), 0.0, 0.0, 0.0);
}

}  // namespace ovsentry
