#include <algorithm>
#include <chrono>
#include <cmath>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/ros2/ros2_gimbal.hpp"
#include "io/ros2/sentry_request.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/sentry_command.hpp"
#include "tasks/auto_aim/sentry_mpc_safety.hpp"
#include "tasks/auto_aim/sentry_mpc_takeover.hpp"
#include "tasks/auto_aim/sentry_mpc_transform.hpp"
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
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/recorder.hpp"
#include "tools/yaml.hpp"

namespace
{
bool is_buff_mode(io::Mode mode) { return mode == io::small_buff || mode == io::big_buff; }

const char * buff_mode_name(io::Mode mode)
{
  if (mode == io::small_buff) return "small_buff";
  if (mode == io::big_buff) return "big_buff";
  return "none";
}

void apply_world_direction_target(
  io::Command & command, const Eigen::Vector3d & world_direction, double big_yaw_rad,
  double current_small_yaw_rad, tools::GimbalAxisOrder gimbal_axis_order)
{
  const auto sentry_command =
    auto_aim::sentry_mpc_transform::world_direction_to_world_small_yaw_command(
      world_direction, big_yaw_rad, gimbal_axis_order);
  command.yaw = auto_aim::nearest_continuous_yaw_rad(sentry_command.small_yaw, current_small_yaw_rad);
  command.pitch = sentry_command.pitch;
  command.big_yaw = big_yaw_rad;
  command.small_yaw = command.yaw;
  command.has_target_yaw = true;
}

bool gimbal_state_is_finite(const io::ROS2GimbalState & state)
{
  return std::isfinite(state.yaw) && std::isfinite(state.yaw_vel) && std::isfinite(state.pitch) &&
         std::isfinite(state.pitch_vel) && std::isfinite(state.bullet_speed) &&
         std::isfinite(state.big_yaw);
}

}  // namespace

const std::string keys =
  "{help h usage ? |                         | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml    | 位置参数，yaml配置文件路径 }"
  "{left           | __yaml__                | 左前相机设备名(相对/dev)，默认读yaml.omni_left_path "
  "}"
  "{right          | __yaml__                | "
  "右前相机设备名(相对/dev)，默认读yaml.omni_right_path }"
  "{back           | __yaml__                | 正后相机设备名(相对/dev)，默认读yaml.omni_back_path "
  "}"
  "{left_yaw       |                         | 左前相机中心yaw角(deg) }"
  "{right_yaw      |                         | 右前相机中心yaw角(deg) }"
  "{back_yaw       |                         | 正后相机中心yaw角(deg) }"
  "{fov_h          |                         | USB相机水平视场角(deg) }"
  "{fov_v          |                         | USB相机垂直视场角(deg) }";

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
  auto read_cam_path =
    [&](const std::string & cli_key, const std::string & yaml_key, const std::string & fallback) {
      const auto cli_value = cli.get<std::string>(cli_key);
      if (!cli_value.empty() && cli_value != "__yaml__") return io::normalize_dev_name(cli_value);
      if (yaml[yaml_key]) return io::normalize_dev_name(yaml[yaml_key].as<std::string>());
      return io::normalize_dev_name(fallback);
    };
  auto read_cli_or_yaml_double =
    [&](const std::string & cli_key, const std::string & yaml_key, double fallback) {
      if (cli.has(cli_key)) return cli.get<double>(cli_key);
      if (yaml[yaml_key]) return yaml[yaml_key].as<double>();
      return fallback;
    };
  const auto read_or = [&](const char * key, double fallback) {
    return yaml[key] ? yaml[key].as<double>() : fallback;
  };
  const auto read_or_int = [&](const char * key, int fallback) {
    return yaml[key] ? yaml[key].as<int>() : fallback;
  };
  const auto read_or_string = [&](const char * key, const char * fallback) {
    return yaml[key] ? yaml[key].as<std::string>() : std::string(fallback);
  };
  const auto seconds = [](double value) {
    return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(value));
  };
  const auto read_non_negative = [&](const char * key, double fallback) {
    const double value = read_or(key, fallback);
    return std::isfinite(value) ? std::max(0.0, value) : fallback;
  };
  const auto read_if_finite = [&](const char * key, double fallback, auto accept) {
    const double value = read_or(key, fallback);
    return std::isfinite(value) && accept(value) ? value : fallback;
  };

  const tools::GimbalAxisOrder gimbal_axis_order =
    yaml["gimbal_axis_order"]
      ? tools::parse_gimbal_axis_order(yaml["gimbal_axis_order"].as<std::string>())
      : tools::GimbalAxisOrder::yaw_pitch;
  const std::string auto_aim_device = read_infer_device("auto_aim_device");
  const std::string omni_device = read_infer_device("omni_device");
  const double omni_retarget_cooldown_s = read_or("omni_retarget_cooldown_s", 2.5);
  const double omni_hold_release_tolerance_deg = read_or("omni_hold_release_tolerance_deg", 3.0);
  const double omni_retarget_min_delta_deg = read_or("omni_retarget_min_delta_deg", 20.0);
  const double omni_command_timeout_s = read_or("omni_command_timeout_s", 0.5);
  const double main_lost_cmd_hold_s = read_non_negative("main_lost_cmd_hold_s", 0.25);
  const auto omni_read_timeout =
    std::chrono::milliseconds(std::max(1, read_or_int("omni_camera_read_timeout_ms", 10)));
  const auto omni_retarget_cooldown = seconds(omni_retarget_cooldown_s);
  const auto omni_command_timeout = seconds(omni_command_timeout_s);
  const auto main_lost_cmd_hold_duration = seconds(main_lost_cmd_hold_s);
  const std::string auto_aim_ignore_topic =
    read_or_string("auto_aim_ignore_topic", "/request_auto_aim_ignore");
  const std::string auto_aim_ignore_msg_type =
    read_or_string("auto_aim_ignore_msg_type", "rm_interfaces/msg/RequestAutoAimIgnore");
  const double takeover_time_s = read_or("mpc_takeover_time_s", 0.20);
  const double status_timeout_s = read_if_finite(
    "mpc_gimbal_status_timeout_s", 0.20, [](double value) { return value >= 0.0; });
  const double max_yaw_acc =
    read_if_finite("max_yaw_acc", 50.0, [](double value) { return value > 0.0; });
  const double max_pitch_acc =
    read_if_finite("max_pitch_acc", 100.0, [](double value) { return value > 0.0; });
  auto_aim::SentryMpcSafetyLimits safety_limits;
  safety_limits.min_pitch = read_or("mpc_pitch_min_deg", -60.0) / 57.3;
  safety_limits.max_pitch = read_or("mpc_pitch_max_deg", 30.0) / 57.3;
  const auto status_timeout = seconds(status_timeout_s);

  const double omni_fov_h_deg = read_cli_or_yaml_double("fov_h", "omni_fov_h_deg", 120.0);
  const double omni_fov_v_deg = read_cli_or_yaml_double("fov_v", "omni_fov_v_deg", 67.0);
  const auto make_omni_cam =
    [&](omniperception::OmniCameraSlot slot, const char * label, const char * cli_key,
        const char * yaml_path_key, const char * fallback_dev, const char * yaw_cli,
        const char * yaw_yaml, double yaw_fallback) {
      const auto dev = read_cam_path(cli_key, yaml_path_key, fallback_dev);
      return omniperception::OmniCamConfig{
        {slot, label, dev, read_cli_or_yaml_double(yaw_cli, yaw_yaml, yaw_fallback), omni_fov_h_deg,
         omni_fov_v_deg},
        dev};
    };
  const auto left_cam_cfg = make_omni_cam(
    omniperception::OmniCameraSlot::left, "left", "left", "omni_left_path", "video0", "left_yaw",
    "omni_left_yaw_deg", 60.0);
  const auto right_cam_cfg = make_omni_cam(
    omniperception::OmniCameraSlot::right, "right", "right", "omni_right_path", "video2",
    "right_yaw", "omni_right_yaw_deg", -60.0);
  const auto back_cam_cfg = make_omni_cam(
    omniperception::OmniCameraSlot::back, "back", "back", "omni_back_path", "video4", "back_yaw",
    "omni_back_yaw_deg", 180.0);

  tools::logger()->info(
    "[OVSentryOmniMPC] inference devices: auto_aim={} omni={}", auto_aim_device, omni_device);
  tools::logger()->info(
    "[OVSentryOmniMPC] omni center yaw(deg): left={:.1f} right={:.1f} back={:.1f}",
    left_cam_cfg.spec.center_yaw_deg, right_cam_cfg.spec.center_yaw_deg,
    back_cam_cfg.spec.center_yaw_deg);

  tools::Exiter exiter;
  tools::Recorder recorder(30);
  constexpr bool yolo_debug = false;

  auto gimbal = std::make_unique<io::ROS2Gimbal>(config_path);
  io::ArmorIgnoreSubscriber armor_ignore_subscriber(auto_aim_ignore_topic, auto_aim_ignore_msg_type);
  auto auto_aim_camera = std::make_unique<io::Camera>(config_path);

  auto_aim::YOLO yolo_auto(config_path, yolo_debug, "auto_aim_device");
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  auto_aim::Planner planner(config_path);
  auto_aim::SentryMpcTakeover takeover(takeover_time_s, max_yaw_acc, max_pitch_acc);
  auto_aim::SentryMpcSafetyGate safety_gate(safety_limits);
  omniperception::Decider decider(config_path);
  constexpr bool aimer_to_now = true;

  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto_buff::SmallTarget buff_small_target;
  auto_buff::BigTarget buff_big_target;
  auto_buff::Aimer buff_aimer(config_path);

  auto yolo_omni_left = std::make_unique<auto_aim::YOLO>(config_path, yolo_debug, "omni_device");
  auto yolo_omni_right = std::make_unique<auto_aim::YOLO>(config_path, yolo_debug, "omni_device");
  std::unique_ptr<auto_aim::YOLO> yolo_omni_back;
  if (!back_cam_cfg.dev_name.empty()) {
    yolo_omni_back = std::make_unique<auto_aim::YOLO>(config_path, yolo_debug, "omni_device");
  }

  io::USBCamera cam_left(left_cam_cfg.dev_name, config_path);
  io::USBCamera cam_right(right_cam_cfg.dev_name, config_path);
  std::unique_ptr<io::USBCamera> cam_back;
  if (!back_cam_cfg.dev_name.empty()) {
    cam_back = std::make_unique<io::USBCamera>(back_cam_cfg.dev_name, config_path);
  }
  cam_left.device_name = left_cam_cfg.spec.label;
  cam_right.device_name = right_cam_cfg.spec.label;
  if (cam_back) cam_back->device_name = back_cam_cfg.spec.label;

  cv::Mat main_img, left_img, right_img, back_img;
  std::chrono::steady_clock::time_point main_timestamp, ts_left, ts_right, ts_back;
  std::optional<io::Command> omni_hold_command;
  std::optional<omniperception::AcceptedOmniTarget> session_accepted_omni_target;
  std::optional<omniperception::AcceptedOmniTarget> cooldown_anchor_omni_target;
  std::optional<omniperception::AcceptedOmniTarget> active_omni_timeout_target;
  std::optional<auto_aim::SentryMpcSetpoint> main_lost_hold_setpoint;
  std::chrono::steady_clock::time_point omni_retarget_cooldown_deadline{};
  std::chrono::steady_clock::time_point active_omni_timeout_started_at{};
  std::chrono::steady_clock::time_point main_lost_cmd_hold_deadline{};
  bool active_omni_timeout_running = false;
  bool main_lost_cmd_hold_running = false;
  bool main_camera_tracker_active = false;
  bool prev_omni_mode = false;
  bool status_warning_active = false;
  int frame_count = 0;

  tools::logger()->info(
    "[OVSentryOmniMPC] world-frame dual-yaw MPC enabled; takeover={:.0f}ms, "
    "pitch_limit=[{:.1f},{:.1f}]deg",
    takeover_time_s * 1e3, safety_limits.min_pitch * 57.3, safety_limits.max_pitch * 57.3);

  const auto clear_omni_timeout_session = [&]() {
    active_omni_timeout_target.reset();
    active_omni_timeout_started_at = std::chrono::steady_clock::time_point{};
    active_omni_timeout_running = false;
  };
  const auto clear_omni_redirect_state = [&]() {
    omni_hold_command.reset();
    session_accepted_omni_target.reset();
    cooldown_anchor_omni_target.reset();
    omni_retarget_cooldown_deadline = std::chrono::steady_clock::time_point{};
    clear_omni_timeout_session();
  };
  const auto clear_main_lost_cmd_hold = [&]() {
    main_lost_hold_setpoint.reset();
    main_lost_cmd_hold_deadline = std::chrono::steady_clock::time_point{};
    main_lost_cmd_hold_running = false;
  };
  const auto make_gimbal_hold_setpoint = [](const auto & state) {
    auto_aim::SentryMpcSetpoint setpoint;
    setpoint.command = io::Command{true, false, state.yaw, state.pitch};
    setpoint.command.small_yaw = state.yaw;
    setpoint.command.big_yaw = state.big_yaw;
    setpoint.command.has_target_yaw = true;
    return setpoint;
  };

  while (!exiter.exit()) {
    try {
      auto_aim_camera->read(main_img, main_timestamp);
      if (main_img.empty()) {
        tools::logger()->warn("[OVSentryOmniMPC] empty main camera frame, skipping");
        gimbal->release_control();
        takeover.clear_target();
        clear_omni_redirect_state();
        clear_main_lost_cmd_hold();
        main_camera_tracker_active = false;
        continue;
      }
    } catch (const std::exception & e) {
      tools::logger()->error("[OVSentryOmniMPC] main camera read failed: {}", e.what());
      gimbal->release_control();
      takeover.clear_target();
      clear_omni_redirect_state();
      clear_main_lost_cmd_hold();
      main_camera_tracker_active = false;
      continue;
    }

    frame_count++;
    bool gimbal_status_fresh = gimbal->status_is_fresh(status_timeout);
    if (!gimbal_status_fresh) {
      if (!status_warning_active) {
        tools::logger()->warn(
          "[OVSentryOmniMPC] gimbal status missing or stale; control remains disabled");
        status_warning_active = true;
      }
      gimbal->release_control();
      takeover.clear_target();
      clear_omni_redirect_state();
      clear_main_lost_cmd_hold();
      main_camera_tracker_active = false;
      continue;
    }
    if (status_warning_active) {
      tools::logger()->info("[OVSentryOmniMPC] fresh gimbal status restored");
      status_warning_active = false;
    }

    const auto q_at_image = gimbal->try_imu_at_image(main_timestamp, status_timeout);
    const auto initial_gimbal_state = gimbal->state();
    recorder.record(main_img, q_at_image.value_or(Eigen::Quaterniond::Identity()), main_timestamp);
    if (
      !q_at_image.has_value() || !q_at_image->coeffs().allFinite() || q_at_image->norm() < 1e-6 ||
      !gimbal_state_is_finite(initial_gimbal_state)) {
      tools::logger()->warn("[OVSentryOmniMPC] invalid gimbal state; control remains disabled");
      gimbal->release_control();
      takeover.clear_target();
      clear_omni_redirect_state();
      clear_main_lost_cmd_hold();
      main_camera_tracker_active = false;
      continue;
    }

    const Eigen::Quaterniond q = q_at_image->normalized();
    solver.set_R_gimbal2world(q);
    buff_solver.set_R_gimbal2world(q);
    auto gimbal_state = initial_gimbal_state;
    const auto gimbal_mode = gimbal->mode();
    const bool buff_mode = is_buff_mode(gimbal_mode);
    std::list<auto_aim::Armor> armors;
    std::list<auto_aim::Target> targets;
    std::string tracker_state = buff_mode ? buff_mode_name(gimbal_mode) : "idle";
    bool omni_mode = false;
    const auto armor_ignore_list = armor_ignore_subscriber.ignored();
    const auto_aim::ArmorTargetMask armor_target_mask{
      armor_ignore_list.enabled, armor_ignore_list.ignored_ids};
    if (!buff_mode) {
      armors = yolo_auto.detect(main_img, frame_count);
      decider.armor_filter(armors);
      decider.set_priority(armors);
      auto_aim::apply_armor_target_mask(armors, armor_target_mask);
      targets = tracker.track(armors, main_timestamp);
      tracker_state = tracker.state();
      omni_mode = tracker_state == "lost";
      if (tracker_state != "lost") main_camera_tracker_active = true;
    }

    gimbal_status_fresh = gimbal->status_is_fresh(status_timeout);
    gimbal_state = gimbal->state();
    if (!gimbal_status_fresh || !gimbal_state_is_finite(gimbal_state)) {
      tools::logger()->warn(
        "[OVSentryOmniMPC] gimbal status became stale during detection; command suppressed");
      gimbal->release_control();
      takeover.clear_target();
      clear_omni_redirect_state();
      clear_main_lost_cmd_hold();
      main_camera_tracker_active = false;
      status_warning_active = true;
      continue;
    }

    bool omni_retarget_cd_active = false;
    bool main_lost_cmd_hold_applied = false;
    const auto now = std::chrono::steady_clock::now();
    io::Command command{false, false, 0.0, 0.0};
    bool tracker_control_ready = false;
    bool aim_point_ready = false;
    bool high_spin_center_aim_active = false;
    std::optional<auto_buff::PowerRune> buff_power_runes;

    if (cooldown_anchor_omni_target.has_value() && now >= omni_retarget_cooldown_deadline) {
      cooldown_anchor_omni_target.reset();
      omni_retarget_cooldown_deadline = std::chrono::steady_clock::time_point{};
    }

    if (omni_mode && !prev_omni_mode) {
      clear_omni_redirect_state();
      if (
        main_camera_tracker_active &&
        main_lost_cmd_hold_duration > std::chrono::steady_clock::duration::zero()) {
        main_lost_hold_setpoint = make_gimbal_hold_setpoint(gimbal_state);
        main_lost_cmd_hold_deadline = now + main_lost_cmd_hold_duration;
        main_lost_cmd_hold_running = true;
      } else {
        clear_main_lost_cmd_hold();
      }
    } else if (!omni_mode && prev_omni_mode) {
      clear_omni_redirect_state();
      clear_main_lost_cmd_hold();
    }

    if (buff_mode) {
      takeover.clear_target();
      clear_main_lost_cmd_hold();
      main_camera_tracker_active = false;
      left_img.release();
      right_img.release();
      back_img.release();
      clear_omni_redirect_state();

      buff_power_runes = buff_detector.detect(main_img);
      buff_solver.solve(buff_power_runes);

      if (gimbal_mode == io::small_buff) {
        buff_small_target.get_target(buff_power_runes, main_timestamp);
        if (!buff_small_target.is_unsolve()) {
          command =
            buff_aimer.aim(buff_small_target, main_timestamp, gimbal->bullet_speed(), aimer_to_now);
        }
      } else {
        buff_big_target.get_target(buff_power_runes, main_timestamp);
        if (!buff_big_target.is_unsolve()) {
          command =
            buff_aimer.aim(buff_big_target, main_timestamp, gimbal->bullet_speed(), aimer_to_now);
        }
      }

      if (command.control) {
        // BuffAimer returns a command in the configured mechanical axis order. Recover its world
        // direction once, then emit a world-referenced small yaw plus the physical pitch joint.
        const Eigen::Vector3d world_direction =
          tools::gimbal_direction_from_command(command.yaw, command.pitch, gimbal_axis_order);
        const double world_yaw = std::atan2(world_direction.y(), world_direction.x());
        const double continuous_big_yaw =
          auto_aim::nearest_continuous_yaw_rad(world_yaw, gimbal_state.big_yaw);
        apply_world_direction_target(
          command, world_direction, continuous_big_yaw, gimbal_state.yaw, gimbal_axis_order);
      }

      auto_aim::SentryMpcSetpoint buff_setpoint;
      buff_setpoint.command = command;
      auto_aim::dispatch_sentry_mpc(*gimbal, buff_setpoint);

    } else if (omni_mode) {
      takeover.clear_target();
      auto read_omni_frame = [&](
                               io::USBCamera & camera, cv::Mat & img,
                               std::chrono::steady_clock::time_point & ts,
                               const omniperception::OmniCamConfig & cam_cfg) {
        omniperception::OmniCandidateFrame frame;
        frame.result.cam = cam_cfg;
        const bool ok = camera.read_with_timeout(img, ts, omni_read_timeout);
        if (!ok || img.empty()) {
          img.release();
          return frame;
        }
        frame.timestamp = ts;
        frame.base_big_yaw_rad = gimbal->big_yaw_at_image(ts);
        frame.has_base_big_yaw = true;
        return frame;
      };

      auto left_frame = read_omni_frame(cam_left, left_img, ts_left, left_cam_cfg);
      auto right_frame = read_omni_frame(cam_right, right_img, ts_right, right_cam_cfg);
      omniperception::OmniCandidateFrame back_frame;
      if (cam_back) {
        back_frame = read_omni_frame(*cam_back, back_img, ts_back, back_cam_cfg);
      } else {
        back_img.release();
      }

      if (left_frame.has_base_big_yaw && !left_img.empty()) {
        left_frame.result.armors = yolo_omni_left->detect(left_img, frame_count);
      }
      if (right_frame.has_base_big_yaw && !right_img.empty()) {
        right_frame.result.armors = yolo_omni_right->detect(right_img, frame_count);
      }
      if (yolo_omni_back && back_frame.has_base_big_yaw && !back_img.empty()) {
        back_frame.result.armors = yolo_omni_back->detect(back_img, frame_count);
      }

      auto finalize_frame =
        [&](omniperception::OmniCandidateFrame & frame, const omniperception::OmniCamConfig & cam_cfg) {
          decider.armor_filter(frame.result.armors);
          decider.set_priority(frame.result.armors);
          auto_aim::apply_armor_target_mask(frame.result.armors, armor_target_mask);
          omniperception::fill_omni_candidate(frame, cam_cfg.spec, gimbal_axis_order);
        };

      finalize_frame(left_frame, left_cam_cfg);
      finalize_frame(right_frame, right_cam_cfg);
      finalize_frame(back_frame, back_cam_cfg);

      omni_retarget_cd_active =
        cooldown_anchor_omni_target.has_value() && now < omni_retarget_cooldown_deadline;
      const auto reference_omni_target = omniperception::select_omni_retarget_reference_target(
        session_accepted_omni_target, cooldown_anchor_omni_target, omni_retarget_cd_active);

      std::vector<omniperception::OmniCandidateFrame> candidate_frames;
      if (left_frame.candidate.has_value()) candidate_frames.push_back(left_frame);
      if (right_frame.candidate.has_value()) candidate_frames.push_back(right_frame);
      if (back_frame.candidate.has_value()) candidate_frames.push_back(back_frame);

      std::vector<omniperception::OmniCandidate> candidates;
      candidates.reserve(candidate_frames.size());
      for (const auto & frame : candidate_frames) candidates.push_back(frame.candidate.value());

      const auto selected_candidate = omniperception::select_omni_candidate(
        candidates, reference_omni_target, gimbal_state.big_yaw, omni_retarget_min_delta_deg);

      if (selected_candidate.has_value()) {
        const auto decision = omniperception::evaluate_omni_retarget(
          selected_candidate.value(), reference_omni_target, gimbal_state.big_yaw,
          omni_retarget_cd_active, omni_retarget_min_delta_deg);

        if (decision.accept) {
          command = selected_candidate->command;
          omni_hold_command = command;
          const auto accepted_target = omniperception::make_accepted_omni_target(selected_candidate.value());
          session_accepted_omni_target = accepted_target;
          if (
            !active_omni_timeout_running || !active_omni_timeout_target.has_value() ||
            !omniperception::same_omni_target_continuation(
              active_omni_timeout_target.value(), accepted_target, omni_retarget_min_delta_deg)) {
            active_omni_timeout_started_at = now;
            active_omni_timeout_running = true;
          }
          active_omni_timeout_target = accepted_target;
          if (omniperception::should_start_omni_retarget_cooldown(
                decision, omni_retarget_min_delta_deg)) {
            cooldown_anchor_omni_target = accepted_target;
            omni_retarget_cooldown_deadline = now + omni_retarget_cooldown;
            omni_retarget_cd_active = true;
          }
        } else if (reference_omni_target.has_value()) {
          command = reference_omni_target->command;
          omni_hold_command = command;
        }
      } else if (omni_hold_command.has_value()) {
        const double target_error_deg = omniperception::angular_distance_deg(omni_hold_command->big_yaw, gimbal_state.big_yaw);
        if (target_error_deg > omni_hold_release_tolerance_deg) {
          command = omni_hold_command.value();
        } else {
          omni_hold_command.reset();
        }
      }

      if (command.control && command.has_target_yaw) {
        const double target_error_deg = omniperception::angular_distance_deg(command.big_yaw, gimbal_state.big_yaw);
        if (target_error_deg > omni_hold_release_tolerance_deg) {
          if (
            active_omni_timeout_running &&
            (now - active_omni_timeout_started_at) > omni_command_timeout) {
            const double omni_cmd_elapsed_ms =
              std::chrono::duration<double, std::milli>(now - active_omni_timeout_started_at).count();
            tools::logger()->warn(
              "[OVSentryOmniMPC] omni command timed out after {:.0f}ms without reaching target yaw",
              omni_cmd_elapsed_ms);
            command = io::Command{false, false, 0.0, 0.0};
            clear_omni_redirect_state();
          }
        } else {
          clear_omni_timeout_session();
          if (omni_hold_command.has_value()) omni_hold_command.reset();
        }
      } else {
        clear_omni_timeout_session();
      }

      if (main_lost_cmd_hold_running && main_lost_hold_setpoint.has_value()) {
        if (now < main_lost_cmd_hold_deadline) {
          auto hold_setpoint = main_lost_hold_setpoint.value();
          hold_setpoint.command.shoot = false;
          hold_setpoint.fire_ready = false;
          command = hold_setpoint.command;
          auto_aim::dispatch_sentry_mpc(*gimbal, hold_setpoint);
          main_lost_cmd_hold_applied = true;
        } else {
          clear_main_lost_cmd_hold();
          main_camera_tracker_active = false;
        }
      }

      if (!main_lost_cmd_hold_applied) {
        auto_aim::SentryMpcSetpoint omni_setpoint;
        omni_setpoint.command = command;
        if (omni_setpoint.command.has_target_yaw) {
          omni_setpoint.command.small_yaw = auto_aim::nearest_continuous_yaw_rad(
            omni_setpoint.command.small_yaw, gimbal_state.yaw);
        }
        auto_aim::dispatch_sentry_mpc(*gimbal, omni_setpoint);
      }
    } else {
      left_img.release();
      right_img.release();
      back_img.release();
      clear_omni_redirect_state();

      // Aimer keeps armor-selection debug state; MPC provides all commanded motion.
      tracker_control_ready = tracker_state == "tracking";
      shooter.update_high_spin_modes(targets);
      high_spin_center_aim_active = shooter.high_spin_center_aim_active();
      (void)aimer.aim(targets, main_timestamp, gimbal->bullet_speed(), aimer_to_now);

      auto_aim::Plan mpc_plan{false};
      auto_aim::SentryMpcSetpoint setpoint;
      std::optional<int> control_armor_id;
      if (
        !high_spin_center_aim_active && !targets.empty() && tracker_control_ready &&
        aimer.debug_aim_point.valid) {
        control_armor_id = aimer.debug_aim_point.armor_id;
      }
      aim_point_ready = high_spin_center_aim_active || control_armor_id.has_value();

      if (targets.empty() || !tracker_control_ready || !aim_point_ready) {
        takeover.clear_target();
      } else {
        const auto & target = targets.front();
        takeover.begin_target(target.name, target.armor_type, high_spin_center_aim_active);

        const auto sentry_plan = planner.plan_sentry_world(
          std::optional<auto_aim::Target>{target}, gimbal->bullet_speed(),
          control_armor_id, high_spin_center_aim_active);
        mpc_plan = sentry_plan.world_small_yaw_plan;
        if (safety_gate.plan_is_safe(mpc_plan, gimbal_state.pitch)) {
          setpoint = takeover.update(
            mpc_plan, sentry_plan.big_yaw, gimbal_state.yaw, gimbal_state.yaw_vel,
            gimbal_state.pitch, gimbal_state.pitch_vel, gimbal_state.big_yaw, now);
          if (!safety_gate.setpoint_is_safe(
                setpoint.command, setpoint.yaw_vel, setpoint.pitch_vel, setpoint.yaw_acc,
                setpoint.pitch_acc)) {
            setpoint = {};
            takeover.reset();
          }
        } else {
          takeover.reset();
        }
      }

      const auto command_gimbal_state = gimbal->state();
      if (
        !gimbal->status_is_fresh(status_timeout) || !gimbal_state_is_finite(command_gimbal_state)) {
        tools::logger()->warn(
          "[OVSentryOmniMPC] gimbal status became stale during planning; command suppressed");
        gimbal->release_control();
        takeover.clear_target();
        clear_main_lost_cmd_hold();
        main_camera_tracker_active = false;
        status_warning_active = true;
        continue;
      }

      // Never release control to electrical cruise while the main tracker is acquiring,
      // reconnecting, or waiting for a safe MPC trajectory. This hold is intentionally
      // position-only and never authorizes shooting.
      if (!setpoint.command.control && tracker_state != "lost") {
        setpoint = make_gimbal_hold_setpoint(command_gimbal_state);
      }

      command = setpoint.command;
      const Eigen::Vector3d motor_ypr{command_gimbal_state.yaw, command_gimbal_state.pitch, 0.0};
      const bool shooter_ready =
        shooter.shoot(command, aimer, targets, motor_ypr, tracker_control_ready);
      const bool high_spin_force_fire = shooter.high_spin_force_fire_active();
      const bool high_spin_fire_ready = high_spin_force_fire && mpc_plan.control;
      const bool normal_fire_ready = setpoint.fire_ready && mpc_plan.fire && shooter_ready;
      command.shoot =
        command.control && tracker_control_ready && (high_spin_fire_ready || normal_fire_ready);

      auto_aim::fill_nav_target_info(command, targets);

      setpoint.command = command;
      auto_aim::dispatch_sentry_mpc(*gimbal, setpoint);
    }

    prev_omni_mode = omni_mode;
  }

  return 0;
}
