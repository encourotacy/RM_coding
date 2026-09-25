#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/ros2/sentry_request.hpp"
#include "io/ros2/gimbal_convert.hpp"
#include "io/ros2/ros2_gimbal.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/sentry_command.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tasks/omniperception/ovsentry_omni_logic.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/yaml.hpp"
#include "tools/recorder.hpp"

namespace
{
struct OmniDebugFrame
{
  omniperception::OmniCandidateFrame frame;
  double delta_pitch_deg = 0.0;
  double infer_ms = 0.0;
};

cv::Scalar omni_slot_color(omniperception::OmniCameraSlot slot)
{
  switch (slot) {
    case omniperception::OmniCameraSlot::left:
      return {0, 255, 0};
    case omniperception::OmniCameraSlot::right:
      return {0, 255, 255};
    case omniperception::OmniCameraSlot::back:
      return {255, 200, 0};
    default:
      return {255, 255, 255};
  }
}

std::string slot_name(omniperception::OmniCameraSlot slot)
{
  switch (slot) {
    case omniperception::OmniCameraSlot::left:
      return "LEFT";
    case omniperception::OmniCameraSlot::right:
      return "RIGHT";
    case omniperception::OmniCameraSlot::back:
      return "BACK";
    case omniperception::OmniCameraSlot::extra:
      return "EXTRA";
    default:
      return "UNKNOWN";
  }
}

bool same_candidate_frame(
  const omniperception::OmniCandidateFrame & frame, const omniperception::OmniCandidate & candidate)
{
  if (!frame.candidate.has_value()) return false;
  return frame.candidate->slot == candidate.slot &&
         frame.candidate->armor_name == candidate.armor_name &&
         frame.candidate->timestamp == candidate.timestamp;
}

void draw_omni_overlay(
  cv::Mat & img, const omniperception::OmniInferenceResult & result, double delta_pitch_deg,
  double infer_ms)
{
  const auto color = omni_slot_color(result.cam.spec.slot);
  tools::draw_text(
    img,
    fmt::format("{} ({}) {:.1f}ms", slot_name(result.cam.spec.slot), result.cam.dev_name, infer_ms),
    {10, 30}, color, 0.7, 2);

  if (!result.top_armor.has_value()) {
    tools::draw_text(img, "no target", {10, 60}, {120, 120, 120}, 0.7, 2);
    return;
  }

  const auto & armor = result.top_armor.value();
  tools::draw_points(img, armor.points, color, 2);
  tools::draw_text(
    img,
    fmt::format(
      "{} pri={} conf={:.2f}", auto_aim::ARMOR_NAMES[armor.name],
      static_cast<int>(armor.priority), armor.confidence),
    {10, 60}, color, 0.7, 2);
  tools::draw_text(
    img, fmt::format("delta yaw={:.1f} pitch={:.1f}", result.delta_yaw_deg, delta_pitch_deg),
    {10, 90}, color, 0.7, 2);
}

void draw_auto_aim_overlay(
  cv::Mat & img, const std::list<auto_aim::Target> & targets, const auto_aim::Aimer & aimer,
  const auto_aim::Solver & solver)
{
  if (targets.empty()) return;

  const auto & target = targets.front();
  for (const auto & xyza : target.armor_xyza_list()) {
    const auto image_points =
      solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
    tools::draw_points(img, image_points, {0, 255, 0});
  }

  const auto & aim_point = aimer.debug_aim_point;
  const auto aim_image_points =
    solver.reproject_armor(aim_point.xyza.head(3), aim_point.xyza[3], target.armor_type, target.name);
  tools::draw_points(img, aim_image_points, aim_point.valid ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));
}

void draw_small_buff_overlay(
  cv::Mat & img, std::optional<auto_buff::PowerRune> & power_rune,
  const auto_buff::SmallTarget & target, const auto_buff::Solver & solver,
  const auto_aim::Plan & plan)
{
  tools::draw_text(img, "SMALL_BUFF", {10, 30}, {0, 255, 255}, 0.8, 2);
  tools::draw_text(
    img,
    fmt::format(
      "buff yaw={:.2f} pitch={:.2f} fire={}", plan.yaw * 57.3, plan.pitch * 57.3,
      plan.fire ? 1 : 0),
    {10, 60}, {0, 255, 255}, 0.8, 2);

  if (!power_rune.has_value()) {
    tools::draw_text(img, "no buff", {10, 90}, {120, 120, 120}, 0.8, 2);
    return;
  }

  auto & rune = power_rune.value();
  tools::draw_points(img, rune.target().points, {0, 255, 0}, 2);
  tools::draw_point(img, rune.target().center, {0, 0, 255}, 4);
  tools::draw_point(img, rune.r_center, {255, 0, 255}, 4);

  if (target.is_unsolve()) return;
  const auto image_points =
    solver.reproject_buff(target.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.0)),
                          target.ekf_x()[4], target.ekf_x()[5]);
  tools::draw_points(img, image_points, {255, 0, 0}, 2);
}

cv::Mat resize_for_view(const cv::Mat & img)
{
  cv::Mat resized;
  cv::resize(img, resized, {640, 360});
  return resized;
}
}  // namespace

const std::string keys =
  "{help h usage ? |                         | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml    | 位置参数，yaml配置文件路径 }"
  "{left           | __yaml__                | 左前相机设备名(相对/dev)，默认读yaml.omni_left_path }"
  "{right          | __yaml__                | 右前相机设备名(相对/dev)，默认读yaml.omni_right_path }"
  "{back           | __yaml__                | 正后相机设备名(相对/dev)，默认读yaml.omni_back_path }"
  "{left_yaw       | 60                      | 左前相机中心yaw角(deg) }"
  "{right_yaw      | -60                     | 右前相机中心yaw角(deg) }"
  "{back_yaw       | 180                     | 正后相机中心yaw角(deg) }"
  "{fov_h          | 120                     | USB相机水平视场角(deg) }"
  "{fov_v          | 67                      | USB相机垂直视场角(deg) }"
  "{no-display     |                         | 关闭画面显示 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  auto yaml = tools::load(config_path);
  auto read_infer_device = [&](const std::string & key) {
      if (yaml[key]) return yaml[key].as<std::string>();
      if (yaml["device"]) return yaml["device"].as<std::string>();
      return std::string("UNKNOWN");
    };
  auto read_cam_path = [&](const std::string & cli_key, const std::string & yaml_key,
                           const std::string & fallback) {
      const auto cli_value = cli.get<std::string>(cli_key);
      if (!cli_value.empty() && cli_value != "__yaml__") return io::normalize_dev_name(cli_value);
      if (yaml[yaml_key]) return io::normalize_dev_name(yaml[yaml_key].as<std::string>());
      return io::normalize_dev_name(fallback);
    };
  auto read_cli_or_yaml_double = [&](const std::string & cli_key, const std::string & yaml_key,
                                     double fallback) {
      if (cli.has(cli_key)) return cli.get<double>(cli_key);
      if (yaml[yaml_key]) return yaml[yaml_key].as<double>();
      return fallback;
    };

  const std::string auto_aim_device = read_infer_device("auto_aim_device");
  const std::string omni_device = read_infer_device("omni_device");
  const double omni_retarget_cooldown_s =
    yaml["omni_retarget_cooldown_s"] ? yaml["omni_retarget_cooldown_s"].as<double>() : 2.5;
  const double omni_hold_release_tolerance_deg =
    yaml["omni_hold_release_tolerance_deg"] ? yaml["omni_hold_release_tolerance_deg"].as<double>() : 3.0;
  const double omni_retarget_min_delta_deg =
    yaml["omni_retarget_min_delta_deg"] ? yaml["omni_retarget_min_delta_deg"].as<double>() : 20.0;
  const double omni_command_timeout_s =
    yaml["omni_command_timeout_s"] ? yaml["omni_command_timeout_s"].as<double>() : 0.5;
  const double buff_lost_cmd_hold_s =
    yaml["buff_lost_cmd_hold_s"] ? yaml["buff_lost_cmd_hold_s"].as<double>() : 0.3;
  const auto omni_read_timeout = std::chrono::milliseconds(
    std::max(1, yaml["omni_camera_read_timeout_ms"] ? yaml["omni_camera_read_timeout_ms"].as<int>() : 10));
  const auto omni_retarget_cooldown = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(omni_retarget_cooldown_s));
  const auto omni_command_timeout = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(omni_command_timeout_s));
  const auto buff_lost_cmd_hold = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(std::max(0.0, buff_lost_cmd_hold_s)));
  const std::string auto_aim_ignore_topic = yaml["auto_aim_ignore_topic"]
                                              ? yaml["auto_aim_ignore_topic"].as<std::string>()
                                              : "/request_auto_aim_ignore";
  const std::string auto_aim_ignore_msg_type =
    yaml["auto_aim_ignore_msg_type"] ? yaml["auto_aim_ignore_msg_type"].as<std::string>()
                                     : "rm_interfaces/msg/RequestAutoAimIgnore";

  const double omni_fov_h_deg = read_cli_or_yaml_double("fov_h", "omni_fov_h_deg", 120.0);
  const double omni_fov_v_deg = read_cli_or_yaml_double("fov_v", "omni_fov_v_deg", 67.0);
  const omniperception::OmniCamConfig left_cam_cfg{
    {omniperception::OmniCameraSlot::left, "left", read_cam_path("left", "omni_left_path", "video0"),
     read_cli_or_yaml_double("left_yaw", "omni_left_yaw_deg", 60.0), omni_fov_h_deg, omni_fov_v_deg},
    read_cam_path("left", "omni_left_path", "video0")};
  const omniperception::OmniCamConfig right_cam_cfg{
    {omniperception::OmniCameraSlot::right, "right", read_cam_path("right", "omni_right_path", "video2"),
     read_cli_or_yaml_double("right_yaw", "omni_right_yaw_deg", -60.0), omni_fov_h_deg, omni_fov_v_deg},
    read_cam_path("right", "omni_right_path", "video2")};
  const omniperception::OmniCamConfig back_cam_cfg{
    {omniperception::OmniCameraSlot::back, "back", read_cam_path("back", "omni_back_path", "video4"),
     read_cli_or_yaml_double("back_yaw", "omni_back_yaw_deg", 180.0), omni_fov_h_deg, omni_fov_v_deg},
    read_cam_path("back", "omni_back_path", "video4")};

  tools::logger()->info(
    "[OVSentryOmniMPC] inference devices: auto_aim={} omni={}", auto_aim_device, omni_device);

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder(30);
  const bool display = !cli.has("no-display");
  constexpr bool yolo_debug = false;

  auto gimbal = std::make_unique<io::ROS2Gimbal>(config_path);
  io::ArmorIgnoreSubscriber armor_ignore_subscriber(auto_aim_ignore_topic, auto_aim_ignore_msg_type);
  io::BuffRequestSubscriber buff_request_subscriber;
  auto auto_aim_camera = std::make_unique<io::Camera>(config_path);

  auto_aim::YOLO yolo_auto(config_path, yolo_debug, "auto_aim_device");
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  auto_aim::Planner planner(config_path);
  omniperception::Decider decider(config_path);
  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto_buff::SmallTarget buff_small_target;
  auto_buff::Aimer buff_aimer(config_path);
  constexpr bool aimer_to_now = true;

  auto yolo_omni_left = std::make_unique<auto_aim::YOLO>(config_path, yolo_debug, "omni_device");
  auto yolo_omni_right = std::make_unique<auto_aim::YOLO>(config_path, yolo_debug, "omni_device");
  auto yolo_omni_back = std::make_unique<auto_aim::YOLO>(config_path, yolo_debug, "omni_device");

  io::USBCamera cam_left(left_cam_cfg.dev_name, config_path);
  io::USBCamera cam_right(right_cam_cfg.dev_name, config_path);
  io::USBCamera cam_back(back_cam_cfg.dev_name, config_path);
  cam_left.device_name = left_cam_cfg.spec.label;
  cam_right.device_name = right_cam_cfg.spec.label;
  cam_back.device_name = back_cam_cfg.spec.label;

  cv::Mat main_img, left_img, right_img, back_img;
  std::chrono::steady_clock::time_point main_timestamp, ts_left, ts_right, ts_back;
  std::optional<io::Command> omni_hold_command;
  std::optional<omniperception::AcceptedOmniTarget> session_accepted_omni_target;
  std::optional<omniperception::AcceptedOmniTarget> cooldown_anchor_omni_target;
  std::optional<omniperception::AcceptedOmniTarget> active_omni_timeout_target;
  std::chrono::steady_clock::time_point omni_retarget_cooldown_deadline{};
  std::chrono::steady_clock::time_point active_omni_timeout_started_at{};
  std::optional<io::Command> buff_hold_command;
  std::chrono::steady_clock::time_point buff_last_control_at{};
  bool active_omni_timeout_running = false;
  bool prev_omni_mode = false;
  int frame_count = 0;

  while (!exiter.exit()) {
    try {
      auto_aim_camera->read(main_img, main_timestamp);
      if (main_img.empty()) continue;
    } catch (const std::exception & e) {
      tools::logger()->error("[OVSentryOmniMPC] main camera read failed: {}", e.what());
      continue;
    }

    frame_count++;
    Eigen::Quaterniond q = gimbal->imu_at_image(main_timestamp);
    // recorder.record(main_img,q, main_timestamp);
    solver.set_R_gimbal2world(q);
    const auto gimbal_state = gimbal->state();
    const bool small_buff_mode = buff_request_subscriber.requested();
    
    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    auto t0 = std::chrono::steady_clock::now();
    const auto armor_ignore_list = armor_ignore_subscriber.ignored();
    const auto_aim::ArmorTargetMask armor_target_mask{
      armor_ignore_list.enabled, armor_ignore_list.ignored_ids};
    std::list<auto_aim::Armor> armors;
    std::list<auto_aim::Target> targets;
    std::string tracker_state = small_buff_mode ? "small_buff" : "idle";
    if (!small_buff_mode) {
      armors = yolo_auto.detect(main_img, frame_count);
      decider.armor_filter(armors);
      decider.set_priority(armors);
      auto_aim::apply_armor_target_mask(armors, armor_target_mask);
      targets = tracker.track(armors, main_timestamp);
      tracker_state = tracker.state();
    }
    auto t1 = std::chrono::steady_clock::now();
    const bool omni_mode = !small_buff_mode && tracker_state == "lost";

    std::optional<omniperception::OmniInferenceResult> best_omni_result;
    double best_delta_pitch_deg = 0.0;
    double best_infer_ms = 0.0;
    std::optional<double> omni_target_abs_yaw_deg;
    std::optional<double> omni_candidate_abs_yaw_deg;
    std::optional<double> omni_candidate_base_big_yaw_deg;
    std::optional<double> omni_candidate_age_ms;
    std::optional<double> omni_candidate_delta_deg;
    std::optional<double> omni_target_error_deg;
    std::optional<double> omni_selected_confidence;
    std::optional<int> omni_selected_priority;
    std::optional<std::string> omni_selected_slot;
    bool omni_hold_applied = false;
    bool omni_retarget_blocked = false;
    bool omni_retarget_cd_active = false;
    bool omni_same_target_continuation = false;
    bool omni_target_reached = false;
    bool omni_cmd_timeout_active = false;
    bool omni_cmd_timed_out = false;
    std::string omni_block_reason = "none";
    double omni_cmd_elapsed_ms = 0.0;
    double omni_retarget_remaining_ms = 0.0;
    const auto now = std::chrono::steady_clock::now();
    io::Command command{false, false, 0.0, 0.0};
    std::optional<auto_buff::PowerRune> buff_power_rune;
    auto_aim::Plan buff_plan{false, false, 0, 0, 0, 0, 0, 0, 0, 0};
    double buff_detect_ms = 0.0;
    bool buff_target_ready = false;
    bool buff_command_held = false;

    auto clear_omni_timeout_session = [&]() {
        active_omni_timeout_target.reset();
        active_omni_timeout_started_at = std::chrono::steady_clock::time_point{};
        active_omni_timeout_running = false;
      };

    auto clear_omni_redirect_state = [&]() {
        omni_hold_command.reset();
        session_accepted_omni_target.reset();
        cooldown_anchor_omni_target.reset();
        omni_retarget_cooldown_deadline = std::chrono::steady_clock::time_point{};
        clear_omni_timeout_session();
      };

    if (cooldown_anchor_omni_target.has_value() && now >= omni_retarget_cooldown_deadline) {
      cooldown_anchor_omni_target.reset();
      omni_retarget_cooldown_deadline = std::chrono::steady_clock::time_point{};
    }

    if (omni_mode && !prev_omni_mode) {
      clear_omni_redirect_state();
    } else if (!omni_mode && prev_omni_mode) {
      clear_omni_redirect_state();
    }

    if (small_buff_mode) {
      left_img.release();
      right_img.release();
      back_img.release();
      clear_omni_redirect_state();

      buff_solver.set_R_gimbal2world(q);
      const auto t_buff0 = std::chrono::steady_clock::now();
      buff_power_rune = buff_detector.detect(main_img);
      const auto t_buff1 = std::chrono::steady_clock::now();
      buff_detect_ms = tools::delta_time(t_buff1, t_buff0) * 1e3;
      buff_solver.solve(buff_power_rune);
      buff_small_target.get_target(buff_power_rune, main_timestamp);
      buff_target_ready = !buff_small_target.is_unsolve();

      auto buff_target_copy = buff_small_target;
      buff_plan =
        buff_aimer.mpc_aim(buff_target_copy, main_timestamp, io::to_gimbal_state(gimbal_state), true);

      command.control = buff_plan.control;
      command.shoot = buff_plan.fire;
      command.yaw = tools::limit_rad(buff_plan.yaw);
      command.pitch = buff_plan.pitch;
      command.big_yaw = auto_aim::nearest_continuous_yaw_rad(command.yaw, gimbal_state.big_yaw);
      command.small_yaw = command.yaw;
      command.has_target_yaw = command.control;

      if (command.control) {
        buff_hold_command = command;
        buff_last_control_at = now;
      } else if (
        buff_hold_command.has_value() &&
        now - buff_last_control_at <= buff_lost_cmd_hold) {
        // Keep navigation from reclaiming the gimbal for a brief detector/planner dropout.
        command = buff_hold_command.value();
        command.shoot = false;
        buff_command_held = true;
      }

      gimbal->send_mpc(
        command.control, command.shoot, command.big_yaw, command.small_yaw, command.pitch,
        buff_command_held ? 0.0 : buff_plan.yaw_vel,
        buff_command_held ? 0.0 : buff_plan.pitch_vel,
        buff_command_held ? 0.0 : buff_plan.yaw_acc,
        buff_command_held ? 0.0 : buff_plan.pitch_acc, 0, 0.0,
        0.0, 0.0);
    } else if (omni_mode) {
      buff_hold_command.reset();
      auto read_omni_frame = [&](io::USBCamera & camera, cv::Mat & img,
                                 std::chrono::steady_clock::time_point & ts,
                                 const omniperception::OmniCamConfig & cam_cfg) {
          OmniDebugFrame debug_frame;
          auto & frame = debug_frame.frame;
          frame.result.cam = cam_cfg;
          const bool ok = camera.read_with_timeout(img, ts, omni_read_timeout);
          if (!ok || img.empty()) {
            img.release();
            return debug_frame;
          }
          frame.timestamp = ts;
          frame.base_big_yaw_rad = gimbal->big_yaw_at_image(ts);
          frame.has_base_big_yaw = true;
          return debug_frame;
        };

      auto left_frame = read_omni_frame(cam_left, left_img, ts_left, left_cam_cfg);
      auto right_frame = read_omni_frame(cam_right, right_img, ts_right, right_cam_cfg);
      auto back_frame = read_omni_frame(cam_back, back_img, ts_back, back_cam_cfg);

      auto t_omni0 = std::chrono::steady_clock::now();
      if (left_frame.frame.has_base_big_yaw && !left_img.empty()) {
        left_frame.frame.result.armors = yolo_omni_left->detect(left_img, frame_count);
      }
      auto t_omni1 = std::chrono::steady_clock::now();
      if (right_frame.frame.has_base_big_yaw && !right_img.empty()) {
        right_frame.frame.result.armors = yolo_omni_right->detect(right_img, frame_count);
      }
      auto t_omni2 = std::chrono::steady_clock::now();
      if (back_frame.frame.has_base_big_yaw && !back_img.empty()) {
        back_frame.frame.result.armors = yolo_omni_back->detect(back_img, frame_count);
      }
      auto t_omni3 = std::chrono::steady_clock::now();

      auto finalize_frame = [&](OmniDebugFrame & debug_frame,
                                const omniperception::OmniCamConfig & cam_cfg, double infer_ms) {
          debug_frame.infer_ms = infer_ms;
          auto & frame = debug_frame.frame;
          decider.armor_filter(frame.result.armors);
          decider.set_priority(frame.result.armors);
          auto_aim::apply_armor_target_mask(frame.result.armors, armor_target_mask);
          debug_frame.delta_pitch_deg = omniperception::fill_omni_candidate(frame, cam_cfg.spec);
        };

      finalize_frame(left_frame, left_cam_cfg, tools::delta_time(t_omni1, t_omni0) * 1e3);
      finalize_frame(right_frame, right_cam_cfg, tools::delta_time(t_omni2, t_omni1) * 1e3);
      finalize_frame(back_frame, back_cam_cfg, tools::delta_time(t_omni3, t_omni2) * 1e3);

      omni_retarget_cd_active =
        cooldown_anchor_omni_target.has_value() && now < omni_retarget_cooldown_deadline;
      if (omni_retarget_cd_active) {
        omni_retarget_remaining_ms =
          std::chrono::duration<double, std::milli>(omni_retarget_cooldown_deadline - now).count();
      }
      const auto reference_omni_target = omniperception::select_omni_retarget_reference_target(
        session_accepted_omni_target, cooldown_anchor_omni_target, omni_retarget_cd_active);

      std::vector<OmniDebugFrame> candidate_frames;
      if (left_frame.frame.candidate.has_value()) candidate_frames.push_back(left_frame);
      if (right_frame.frame.candidate.has_value()) candidate_frames.push_back(right_frame);
      if (back_frame.frame.candidate.has_value()) candidate_frames.push_back(back_frame);

      std::vector<omniperception::OmniCandidate> candidates;
      candidates.reserve(candidate_frames.size());
      for (const auto & frame : candidate_frames) candidates.push_back(frame.frame.candidate.value());

      const auto selected_candidate = omniperception::select_omni_candidate(
        candidates, reference_omni_target, gimbal_state.big_yaw, omni_retarget_min_delta_deg);

      if (selected_candidate.has_value()) {
        omni_candidate_abs_yaw_deg = selected_candidate->abs_yaw_rad * 57.3;
        omni_candidate_base_big_yaw_deg = selected_candidate->base_big_yaw_rad * 57.3;
        omni_candidate_age_ms = tools::delta_time(now, selected_candidate->timestamp) * 1e3;
        omni_selected_confidence = selected_candidate->confidence;
        omni_selected_priority = static_cast<int>(selected_candidate->priority);
        omni_selected_slot = slot_name(selected_candidate->slot);

        const auto selected_frame = std::find_if(
          candidate_frames.begin(), candidate_frames.end(), [&](const OmniDebugFrame & frame) {
            return same_candidate_frame(frame.frame, selected_candidate.value());
          });
        if (selected_frame != candidate_frames.end()) {
          best_omni_result = selected_frame->frame.result;
          best_delta_pitch_deg = selected_frame->delta_pitch_deg;
          best_infer_ms = selected_frame->infer_ms;
        }

        const auto decision = omniperception::evaluate_omni_retarget(
          selected_candidate.value(), reference_omni_target, gimbal_state.big_yaw,
          omni_retarget_cd_active, omni_retarget_min_delta_deg);
        omni_candidate_delta_deg = decision.candidate_delta_deg;
        omni_same_target_continuation = decision.same_target_continuation;

        if (decision.accept) {
          command = selected_candidate->command;
          omni_hold_command = command;
          const auto accepted_target =
            omniperception::make_accepted_omni_target(selected_candidate.value());
          session_accepted_omni_target = accepted_target;
          if (
            !active_omni_timeout_running || !active_omni_timeout_target.has_value() ||
            !omniperception::same_omni_target_continuation(
              active_omni_timeout_target.value(), accepted_target, omni_retarget_min_delta_deg)) {
            active_omni_timeout_started_at = now;
            active_omni_timeout_running = true;
          }
          active_omni_timeout_target = accepted_target;
          omni_target_abs_yaw_deg = command.big_yaw * 57.3;
          if (omniperception::should_start_omni_retarget_cooldown(
                decision, omni_retarget_min_delta_deg)) {
            cooldown_anchor_omni_target = accepted_target;
            omni_retarget_cooldown_deadline = now + omni_retarget_cooldown;
            omni_retarget_cd_active = true;
            omni_retarget_remaining_ms = omni_retarget_cooldown_s * 1e3;
          }
        } else if (reference_omni_target.has_value()) {
          command = reference_omni_target->command;
          omni_hold_command = command;
          omni_target_abs_yaw_deg = command.big_yaw * 57.3;
          omni_retarget_blocked = true;
          omni_block_reason = decision.block_reason;
        }
      } else if (omni_hold_command.has_value()) {
        const double target_error_deg =
          omniperception::angular_distance_deg(omni_hold_command->big_yaw, gimbal_state.big_yaw);
        if (target_error_deg > omni_hold_release_tolerance_deg) {
          command = omni_hold_command.value();
          omni_target_abs_yaw_deg = command.big_yaw * 57.3;
          omni_hold_applied = true;
        } else {
          omni_hold_command.reset();
        }
      } else {
        omni_hold_command.reset();
        clear_omni_timeout_session();
      }

      if (command.control && command.has_target_yaw) {
        if (active_omni_timeout_running && active_omni_timeout_target.has_value()) {
          omni_cmd_timeout_active = true;
          omni_cmd_elapsed_ms =
            std::chrono::duration<double, std::milli>(now - active_omni_timeout_started_at).count();
        }

        const double target_error_deg =
          omniperception::angular_distance_deg(command.big_yaw, gimbal_state.big_yaw);
        if (target_error_deg > omni_hold_release_tolerance_deg) {
          if (
            active_omni_timeout_running &&
            (now - active_omni_timeout_started_at) > omni_command_timeout) {
            tools::logger()->warn(
              "[OVSentryOmniMPC] omni command timed out after {:.0f}ms without reaching target yaw",
              omni_cmd_elapsed_ms);
            command = io::Command{false, false, 0.0, 0.0};
            omni_target_abs_yaw_deg.reset();
            omni_hold_applied = false;
            omni_cmd_timed_out = true;
            omni_cmd_timeout_active = false;
            clear_omni_redirect_state();
          }
        } else {
          clear_omni_timeout_session();
        }
      } else {
        clear_omni_timeout_session();
      }

      if (command.control && command.has_target_yaw) {
        omni_target_error_deg =
          omniperception::angular_distance_deg(command.big_yaw, gimbal_state.big_yaw);
        omni_target_reached = omni_target_error_deg.value() <= omni_hold_release_tolerance_deg;
        if (omni_hold_command.has_value() && omni_target_reached) {
          omni_hold_command.reset();
        }
      }

      const double omni_big_yaw = command.has_target_yaw ? command.big_yaw : command.yaw;
      const double omni_small_yaw = command.has_target_yaw ? command.small_yaw : command.yaw;
      gimbal->send_mpc(
        command.control, command.shoot, omni_big_yaw, omni_small_yaw, command.pitch,
        0.0, 0.0, 0.0, 0.0, static_cast<uint8_t>(command.armor_id), 0.0, 0.0, 0.0);
    } else {
      left_img.release();
      right_img.release();
      back_img.release();
      clear_omni_redirect_state();
      buff_hold_command.reset();

      const bool armor_acquiring = tracker_state == "detecting";
      if (armor_acquiring) {
        // Do not steer from a just-initialized EKF. Hold control so navigation cannot patrol away
        // while the tracker collects min_detect_count_ observations.
        command.control = true;
        command.shoot = false;
        command.yaw = gimbal_state.yaw;
        command.pitch = -gimbal_state.pitch;
        command.big_yaw = gimbal_state.big_yaw;
        command.small_yaw = gimbal_state.yaw;
        command.has_target_yaw = true;
      } else {
        command = aimer.aim(targets, main_timestamp, gimbal->bullet_speed(), aimer_to_now);
        if (command.control && !targets.empty()) {
          auto_aim::apply_sentry_tracking_yaws(command, targets.front(), gimbal_state.big_yaw);
        }
        command.shoot = shooter.shoot(command, aimer, targets, ypr, tracker_state == "tracking");
        auto_aim::fill_nav_target_info(command, targets);
      }

      const bool outpost_convergence = 
      !targets.empty() && targets.front().name == auto_aim::ArmorName::outpost &&
      (!targets.front().convergened() || targets.front().diverged());
      const bool static_outpost_direct =
        !targets.empty() && targets.front().outpost_static_direct_active();
      if (!armor_acquiring && outpost_convergence && !static_outpost_direct) {
        command = io::Command{false, false, 0.0, 0.0};
      }
      
      const bool unlocked_outpost =
        !targets.empty() && auto_aim::is_unlocked_outpost_target(targets.front());
      double small_yaw_vel = 0.0;
      double pitch_vel = 0.0;
      double small_yaw_acc = 0.0;
      double pitch_acc = 0.0;
      if (command.control && !armor_acquiring && !targets.empty() && !unlocked_outpost) {
        const auto mpc_plan = planner.plan(targets.front(), gimbal->bullet_speed());
        if (mpc_plan.control) {
          small_yaw_vel = mpc_plan.yaw_vel;
          pitch_vel = mpc_plan.pitch_vel;
          small_yaw_acc = mpc_plan.yaw_acc;
          pitch_acc = mpc_plan.pitch_acc;
        }
      }

      const double big_yaw = command.has_target_yaw ? command.big_yaw : command.yaw;
      const double small_yaw = command.has_target_yaw ? command.small_yaw : command.yaw;
      gimbal->send_mpc(
        command.control, command.shoot, big_yaw, small_yaw, command.pitch, small_yaw_vel,
        pitch_vel, small_yaw_acc, pitch_acc, static_cast<uint8_t>(command.armor_id), command.vx,
        command.vy, command.horizon_distance);
    }

    nlohmann::json data;
    data["mode"] = small_buff_mode ? 2 : (omni_mode ? 1 : 0);
    data["gimbal_mode"] = io::MODES[static_cast<int>(gimbal->mode())];
    data["armor_num"] = armors.size();
    data["tracker_state"] = tracker_state;
    data["armor_acquiring"] = (!small_buff_mode && tracker_state == "detecting") ? 1 : 0;
    data["gimbal_yaw"] = ypr[0] * 57.3;
    data["gimbal_small_yaw"] = gimbal_state.yaw * 57.3;
    data["gimbal_big_yaw"] = gimbal_state.big_yaw * 57.3;
    data["bullet_speed"] = gimbal->bullet_speed();
    data["mpc_control"] = command.control ? 1 : 0;
    data["mpc_fire"] = command.shoot ? 1 : 0;
    data["mpc_yaw"] = (command.has_target_yaw ? command.small_yaw : command.yaw) * 57.3;
    data["mpc_pitch"] = command.pitch * 57.3;
    data["target_armor_id"] = static_cast<int>(command.armor_id);
    data["target_vx"] = command.vx;
    data["target_vy"] = command.vy;
    data["horizon_distance"] = command.horizon_distance;
    data["aim_source"] = (!small_buff_mode && !omni_mode && command.control) ? aimer.debug_aim_point.source : -1;
    data["aim_armor_id"] =
      (!small_buff_mode && !omni_mode && command.control) ? aimer.debug_aim_point.armor_id : -1;
    if (small_buff_mode) {
      data["buff_detected"] = buff_power_rune.has_value() ? 1 : 0;
      data["buff_target_ready"] = buff_target_ready ? 1 : 0;
      data["buff_command_held"] = buff_command_held ? 1 : 0;
      data["buff_detect_ms"] = buff_detect_ms;
      data["buff_yaw"] = buff_plan.yaw * 57.3;
      data["buff_pitch"] = buff_plan.pitch * 57.3;
      data["buff_yaw_vel"] = buff_plan.yaw_vel * 57.3;
      data["buff_pitch_vel"] = buff_plan.pitch_vel * 57.3;
      if (buff_power_rune.has_value()) {
        data["buff_r_yaw"] = buff_power_rune->ypd_in_world[0] * 57.3;
        data["buff_r_pitch"] = buff_power_rune->ypd_in_world[1] * 57.3;
        data["buff_r_distance"] = buff_power_rune->ypd_in_world[2];
        data["buff_roll"] = buff_power_rune->ypr_in_world[2] * 57.3;
      }
      if (buff_target_ready) {
        const auto x = buff_small_target.ekf_x();
        data["buff_target_roll"] = x[5] * 57.3;
        data["buff_target_spd"] = x[6] * 57.3;
      }
    }
    if (!targets.empty()) {
      const auto & target = targets.front();
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
      if (target.name == auto_aim::ArmorName::outpost && aimer.debug_aim_point.valid) {
        const auto x = target.ekf_x();
        const double center_yaw = std::atan2(x[2], x[0]);
        data["outpost_aim_phase_deg"] =
          std::abs(tools::limit_rad(aimer.debug_aim_point.xyza[3] - center_yaw)) * 57.3;
      }
    }
    data["omni_yaw_hold"] = omni_hold_applied ? 1 : 0;
    data["omni_target_reached"] = omni_target_reached ? 1 : 0;
    data["omni_cmd_timeout_active"] = omni_cmd_timeout_active ? 1 : 0;
    data["omni_cmd_timed_out"] = omni_cmd_timed_out ? 1 : 0;
    data["omni_cmd_elapsed_ms"] = omni_cmd_elapsed_ms;
    data["omni_retarget_cd_active"] = omni_retarget_cd_active ? 1 : 0;
    data["omni_retarget_blocked"] = omni_retarget_blocked ? 1 : 0;
    data["omni_retarget_remaining_ms"] = omni_retarget_remaining_ms;
    data["omni_same_target_continuation"] = omni_same_target_continuation ? 1 : 0;
    data["omni_block_reason"] = omni_block_reason;
    if (omni_target_abs_yaw_deg.has_value()) data["omni_target_yaw"] = omni_target_abs_yaw_deg.value();
    if (omni_candidate_abs_yaw_deg.has_value()) data["omni_candidate_abs_yaw"] = omni_candidate_abs_yaw_deg.value();
    if (omni_candidate_base_big_yaw_deg.has_value()) {
      data["omni_candidate_base_big_yaw"] = omni_candidate_base_big_yaw_deg.value();
    }
    if (omni_candidate_age_ms.has_value()) data["omni_candidate_age_ms"] = omni_candidate_age_ms.value();
    if (omni_candidate_delta_deg.has_value()) data["omni_candidate_delta_deg"] = omni_candidate_delta_deg.value();
    if (omni_target_error_deg.has_value()) data["omni_target_error_deg"] = omni_target_error_deg.value();
    if (omni_selected_confidence.has_value()) data["omni_selected_confidence"] = omni_selected_confidence.value();
    if (omni_selected_priority.has_value()) data["omni_selected_priority"] = omni_selected_priority.value();
    if (omni_selected_slot.has_value()) data["omni_selected_slot"] = omni_selected_slot.value();
    data["yolo_time"] = tools::delta_time(t1, t0) * 1e3;
    plotter.plot(data);

    prev_omni_mode = omni_mode;
    if (!display) continue;

    if (small_buff_mode) {
      draw_small_buff_overlay(main_img, buff_power_rune, buff_small_target, buff_solver, buff_plan);
    } else {
      draw_auto_aim_overlay(main_img, targets, aimer, solver);
      tools::draw_text(main_img, fmt::format("[{}] mode={}", tracker_state, omni_mode ? "OMNI" : "MPC"),
        {10, 30}, {255, 255, 255}, 0.8, 2);
      tools::draw_text(main_img,
        fmt::format("mpc yaw={:.2f} pitch={:.2f} fire={}",
          (command.has_target_yaw ? command.small_yaw : command.yaw) * 57.3,
          command.pitch * 57.3, command.shoot ? 1 : 0),
        {10, 60}, {154, 50, 205}, 0.8, 2);
    }
    if (omni_target_abs_yaw_deg.has_value()) {
      tools::draw_text(main_img, fmt::format("omni target yaw={:.2f}", omni_target_abs_yaw_deg.value()),
        {10, 90}, {0, 255, 255}, 0.8, 2);
    }
    if (omni_retarget_cd_active) {
      tools::draw_text(
        main_img, fmt::format("omni retarget cd {:.0f}ms", omni_retarget_remaining_ms),
        {10, 120}, omni_retarget_blocked ? cv::Scalar(0, 180, 255) : cv::Scalar(255, 220, 0), 0.8,
        2);
    }
    if (omni_cmd_timeout_active || omni_cmd_timed_out) {
      tools::draw_text(
        main_img,
        fmt::format(
          "omni cmd timeout {:.0f}/{:.0f}ms hit={}", omni_cmd_elapsed_ms,
          omni_command_timeout_s * 1e3, omni_cmd_timed_out ? 1 : 0),
        {10, 150}, omni_cmd_timed_out ? cv::Scalar(0, 120, 255) : cv::Scalar(180, 255, 180), 0.8,
        2);
    }
    if (omni_target_error_deg.has_value()) {
      tools::draw_text(
        main_img,
        fmt::format(
          "omni target err={:.1f} reached={}", omni_target_error_deg.value(),
          omni_target_reached ? 1 : 0),
        {10, 180}, {180, 255, 180}, 0.8, 2);
    }

    cv::Mat left_show = left_img.empty() ? cv::Mat::zeros(main_img.size(), main_img.type()) : left_img.clone();
    cv::Mat right_show = right_img.empty() ? cv::Mat::zeros(main_img.size(), main_img.type()) : right_img.clone();
    cv::Mat back_show = back_img.empty() ? cv::Mat::zeros(main_img.size(), main_img.type()) : back_img.clone();

    if (omni_mode && best_omni_result.has_value()) {
      const auto & best = best_omni_result.value();
      if (best.cam.spec.slot == omniperception::OmniCameraSlot::left) {
        draw_omni_overlay(left_show, best, best_delta_pitch_deg, best_infer_ms);
      } else if (best.cam.spec.slot == omniperception::OmniCameraSlot::right) {
        draw_omni_overlay(right_show, best, best_delta_pitch_deg, best_infer_ms);
      } else {
        draw_omni_overlay(back_show, best, best_delta_pitch_deg, best_infer_ms);
      }
    }

    cv::Mat main_small = resize_for_view(main_img);
    cv::Mat left_small = resize_for_view(left_show);
    cv::Mat right_small = resize_for_view(right_show);
    cv::Mat back_small = resize_for_view(back_show);
    cv::Mat top_row, bottom_row, canvas;
    cv::hconcat(main_small, left_small, top_row);
    cv::hconcat(right_small, back_small, bottom_row);
    cv::vconcat(top_row, bottom_row, canvas);
    cv::imshow("ovsentry_omni_mpc", canvas);
    if (cv::waitKey(1) == 'q') break;
  }

  gimbal->send_mpc(false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0);

  return 0;
}
