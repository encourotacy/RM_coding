#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
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
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/yaml.hpp"

namespace
{
struct TargetSession
{
  auto_aim::ArmorName name;
  auto_aim::ArmorType armor_type;
  bool aim_center = false;

  bool operator==(const TargetSession & rhs) const
  {
    return name == rhs.name && armor_type == rhs.armor_type && aim_center == rhs.aim_center;
  }
};

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

cv::Scalar slot_color(omniperception::OmniCameraSlot slot)
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

bool is_buff_mode(io::Mode mode) { return mode == io::small_buff || mode == io::big_buff; }

const char * gimbal_mode_name(io::Mode mode)
{
  switch (mode) {
    case io::idle:
      return "idle";
    case io::auto_aim:
      return "auto_aim";
    case io::small_buff:
      return "small_buff";
    case io::big_buff:
      return "big_buff";
    case io::outpost:
      return "outpost";
    default:
      return "unknown";
  }
}

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

void disable_gimbal(io::ROS2Gimbal & gimbal)
{
  gimbal.send_mpc(false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0);
}

class GimbalSafeStop
{
public:
  explicit GimbalSafeStop(io::ROS2Gimbal & gimbal) : gimbal_(gimbal) {}
  ~GimbalSafeStop() { disable_gimbal(gimbal_); }

private:
  io::ROS2Gimbal & gimbal_;
};

void draw_omni_overlay(cv::Mat & img, const omniperception::OmniInferenceResult & result)
{
  const auto color = slot_color(result.cam.spec.slot);
  tools::draw_text(
    img,
    fmt::format(
      "{} ({}) {:.1f}ms", slot_name(result.cam.spec.slot), result.cam.dev_name, result.infer_ms),
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
      "{} pri={} conf={:.2f}", auto_aim::ARMOR_NAMES[armor.name], static_cast<int>(armor.priority),
      armor.confidence),
    {10, 60}, color, 0.7, 2);
  tools::draw_text(
    img, fmt::format("delta yaw={:.1f} pitch={:.1f}", result.delta_yaw_deg, result.delta_pitch_deg),
    {10, 90}, color, 0.7, 2);
}

void draw_auto_aim_overlay(
  cv::Mat & img, const std::list<auto_aim::Armor> & armors,
  const std::list<auto_aim::Target> & targets, const auto_aim::Aimer & aimer,
  const auto_aim::Solver & solver)
{
  const cv::Scalar kDetectionColor{0, 255, 255};  // yellow
  const cv::Scalar kOutpostDetectionColor{0, 255, 0};  // green
  const cv::Scalar kTrackerColor{0, 255, 0};      // green
  const cv::Scalar kOutpostTrackerColor{0, 0, 255};  // red
  const cv::Scalar kAimColor{0, 0, 255};          // red

  int detection_index = 0;
  for (const auto & armor : armors) {
    ++detection_index;
    if (armor.points.empty()) continue;
    const cv::Scalar color =
      armor.name == auto_aim::ArmorName::outpost ? kOutpostDetectionColor : kDetectionColor;
    tools::draw_points(img, armor.points, color, 3);
    tools::draw_text(
      img,
      fmt::format(
        "D{} {} {:.2f}", detection_index, auto_aim::ARMOR_NAMES[armor.name], armor.confidence),
      armor.points.front() + cv::Point2f{0.0f, -8.0f}, color, 0.55, 2);
  }

  if (targets.empty()) {
    const bool detected_outpost = std::any_of(
      armors.begin(), armors.end(), [](const auto_aim::Armor & armor) {
        return armor.name == auto_aim::ArmorName::outpost;
      });
    tools::draw_text(
      img,
      detected_outpost ? "green=detector  red=tracker | tracker: waiting"
                       : "yellow=detector  green=tracker  red=aim | tracker: waiting",
      {10, std::max(25, img.rows - 20)}, {220, 220, 220}, 0.6, 2);
    return;
  }

  const auto & target = targets.front();
  const bool is_outpost = target.name == auto_aim::ArmorName::outpost;
  int tracker_armor_index = 0;
  for (const auto & xyza : target.armor_xyza_list()) {
    const auto image_points =
      solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
    ++tracker_armor_index;
    if (image_points.empty()) continue;
    const cv::Scalar color = is_outpost ? kOutpostTrackerColor : kTrackerColor;
    tools::draw_points(img, image_points, color, 3);
    tools::draw_text(
      img, fmt::format("T{}", tracker_armor_index),
      image_points.front() + cv::Point2f{0.0f, 18.0f}, color, 0.55, 2);
  }

  if (!is_outpost && aimer.debug_aim_point.valid) {
    const auto & aim_xyza = aimer.debug_aim_point.xyza;
    const auto image_points =
      solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
    tools::draw_points(img, image_points, kAimColor, 3);
  }

  const std::string tracker_status =
    is_outpost ? (target.outpost_layer_locked() ? "locked" : "initializing") : "tracking";
  const std::string legend =
    is_outpost
      ? fmt::format(
          "green=detector({})  red=tracker({}) | {} {}", armors.size(), tracker_armor_index,
          auto_aim::ARMOR_NAMES[target.name], tracker_status)
      : fmt::format(
          "yellow=detector({})  green=tracker({})  red=aim | {} {}", armors.size(),
          tracker_armor_index, auto_aim::ARMOR_NAMES[target.name], tracker_status);
  tools::draw_text(
    img, legend, {10, std::max(25, img.rows - 20)}, {220, 220, 220}, 0.6, 2);
}

cv::Mat resize_for_view(const cv::Mat & img)
{
  constexpr int view_width = 640;
  constexpr int view_height = 360;
  const double scale = std::min(
    static_cast<double>(view_width) / img.cols, static_cast<double>(view_height) / img.rows);

  cv::Mat resized;
  cv::resize(img, resized, {}, scale, scale);

  cv::Mat canvas = cv::Mat::zeros(view_height, view_width, img.type());
  const int x = (view_width - resized.cols) / 2;
  const int y = (view_height - resized.rows) / 2;
  resized.copyTo(canvas(cv::Rect{x, y, resized.cols, resized.rows}));
  return canvas;
}

bool same_candidate_frame(
  const omniperception::OmniCandidateFrame & frame, const omniperception::OmniCandidate & candidate)
{
  if (!frame.candidate.has_value()) return false;
  return frame.candidate->slot == candidate.slot &&
         frame.candidate->armor_name == candidate.armor_name &&
         frame.candidate->timestamp == candidate.timestamp;
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
  "{fov_v          |                         | USB相机垂直视场角(deg) }"
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

  const tools::GimbalAxisOrder gimbal_axis_order =
    yaml["gimbal_axis_order"]
      ? tools::parse_gimbal_axis_order(yaml["gimbal_axis_order"].as<std::string>())
      : tools::GimbalAxisOrder::yaw_pitch;
  const std::string auto_aim_device = read_infer_device("auto_aim_device");
  const std::string omni_device = read_infer_device("omni_device");
  const double omni_retarget_cooldown_s =
    yaml["omni_retarget_cooldown_s"] ? yaml["omni_retarget_cooldown_s"].as<double>() : 2.5;
  const double omni_hold_release_tolerance_deg =
    yaml["omni_hold_release_tolerance_deg"] ? yaml["omni_hold_release_tolerance_deg"].as<double>()
                                            : 3.0;
  const double omni_retarget_min_delta_deg =
    yaml["omni_retarget_min_delta_deg"] ? yaml["omni_retarget_min_delta_deg"].as<double>() : 20.0;
  const double omni_command_timeout_s =
    yaml["omni_command_timeout_s"] ? yaml["omni_command_timeout_s"].as<double>() : 0.5;
  const double configured_main_lost_cmd_hold_s = read_or("main_lost_cmd_hold_s", 0.25);
  const double main_lost_cmd_hold_s =
    std::isfinite(configured_main_lost_cmd_hold_s)
      ? std::max(0.0, configured_main_lost_cmd_hold_s)
      : 0.25;
  const auto omni_read_timeout = std::chrono::milliseconds(std::max(
    1, yaml["omni_camera_read_timeout_ms"] ? yaml["omni_camera_read_timeout_ms"].as<int>() : 10));
  const auto omni_retarget_cooldown =
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(omni_retarget_cooldown_s));
  const auto omni_command_timeout = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(omni_command_timeout_s));
  const auto main_lost_cmd_hold_duration =
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(main_lost_cmd_hold_s));
  const std::string auto_aim_ignore_topic = yaml["auto_aim_ignore_topic"]
                                              ? yaml["auto_aim_ignore_topic"].as<std::string>()
                                              : "/request_auto_aim_ignore";
  const std::string auto_aim_ignore_msg_type =
    yaml["auto_aim_ignore_msg_type"] ? yaml["auto_aim_ignore_msg_type"].as<std::string>()
                                     : "rm_interfaces/msg/RequestAutoAimIgnore";
  const double takeover_time_s = read_or("mpc_takeover_time_s", 0.20);
  const double configured_status_timeout_s = read_or("mpc_gimbal_status_timeout_s", 0.20);
  const double status_timeout_s =
    std::isfinite(configured_status_timeout_s) && configured_status_timeout_s >= 0.0
      ? configured_status_timeout_s
      : 0.20;
  const double configured_max_yaw_acc = read_or("max_yaw_acc", 50.0);
  const double configured_max_pitch_acc = read_or("max_pitch_acc", 100.0);
  const double max_yaw_acc = std::isfinite(configured_max_yaw_acc) && configured_max_yaw_acc > 0.0
                               ? configured_max_yaw_acc
                               : 50.0;
  const double max_pitch_acc =
    std::isfinite(configured_max_pitch_acc) && configured_max_pitch_acc > 0.0
      ? configured_max_pitch_acc
      : 100.0;
  auto_aim::SentryMpcSafetyLimits safety_limits;
  safety_limits.min_pitch = read_or("mpc_pitch_min_deg", -60.0) / 57.3;
  safety_limits.max_pitch = read_or("mpc_pitch_max_deg", 30.0) / 57.3;
  const auto status_timeout = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(status_timeout_s));

  const double omni_fov_h_deg = read_cli_or_yaml_double("fov_h", "omni_fov_h_deg", 120.0);
  const double omni_fov_v_deg = read_cli_or_yaml_double("fov_v", "omni_fov_v_deg", 67.0);
  const omniperception::OmniCamConfig left_cam_cfg{
    {omniperception::OmniCameraSlot::left, "left",
     read_cam_path("left", "omni_left_path", "video0"),
     read_cli_or_yaml_double("left_yaw", "omni_left_yaw_deg", 60.0), omni_fov_h_deg,
     omni_fov_v_deg},
    read_cam_path("left", "omni_left_path", "video0")};
  const omniperception::OmniCamConfig right_cam_cfg{
    {omniperception::OmniCameraSlot::right, "right",
     read_cam_path("right", "omni_right_path", "video2"),
     read_cli_or_yaml_double("right_yaw", "omni_right_yaw_deg", -60.0), omni_fov_h_deg,
     omni_fov_v_deg},
    read_cam_path("right", "omni_right_path", "video2")};
  const omniperception::OmniCamConfig back_cam_cfg{
    {omniperception::OmniCameraSlot::back, "back",
     read_cam_path("back", "omni_back_path", "video4"),
     read_cli_or_yaml_double("back_yaw", "omni_back_yaw_deg", 180.0), omni_fov_h_deg,
     omni_fov_v_deg},
    read_cam_path("back", "omni_back_path", "video4")};

  tools::logger()->info(
    "[OVSentryOmniMPC] inference devices: auto_aim={} omni={}", auto_aim_device, omni_device);
  tools::logger()->info(
    "[OVSentryOmniMPC] omni center yaw(deg): left={:.1f} right={:.1f} back={:.1f}",
    left_cam_cfg.spec.center_yaw_deg, right_cam_cfg.spec.center_yaw_deg,
    back_cam_cfg.spec.center_yaw_deg);

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder(30);
  const bool display = !cli.has("no-display");
  constexpr bool yolo_debug = false;

  auto gimbal = std::make_unique<io::ROS2Gimbal>(config_path);
  GimbalSafeStop safe_stop(*gimbal);
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
  std::optional<TargetSession> active_target_session;
  bool status_warning_active = false;
  int frame_count = 0;

  tools::logger()->info(
    "[OVSentryOmniMPC] world-frame dual-yaw MPC enabled; takeover={:.0f}ms, "
    "pitch_limit=[{:.1f},{:.1f}]deg",
    takeover_time_s * 1e3, safety_limits.min_pitch * 57.3, safety_limits.max_pitch * 57.3);

  const auto reset_control_session = [&]() {
    takeover.reset();
    active_target_session.reset();
  };
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
        disable_gimbal(*gimbal);
        reset_control_session();
        clear_omni_redirect_state();
        clear_main_lost_cmd_hold();
        main_camera_tracker_active = false;
        continue;
      }
    } catch (const std::exception & e) {
      tools::logger()->error("[OVSentryOmniMPC] main camera read failed: {}", e.what());
      disable_gimbal(*gimbal);
      reset_control_session();
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
      disable_gimbal(*gimbal);
      reset_control_session();
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
      disable_gimbal(*gimbal);
      reset_control_session();
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
    const Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    auto t0 = std::chrono::steady_clock::now();
    auto t1 = t0;
    std::list<auto_aim::Armor> armors;
    std::list<auto_aim::Target> targets;
    std::string tracker_state = buff_mode ? buff_mode_name(gimbal_mode) : "idle";
    bool omni_mode = false;
    const auto armor_ignore_list = armor_ignore_subscriber.ignored();
    const auto_aim::ArmorTargetMask armor_target_mask{
      armor_ignore_list.enabled, armor_ignore_list.ignored_ids};
    if (!buff_mode) {
      t0 = std::chrono::steady_clock::now();
      armors = yolo_auto.detect(main_img, frame_count);
      t1 = std::chrono::steady_clock::now();
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
      disable_gimbal(*gimbal);
      reset_control_session();
      clear_omni_redirect_state();
      clear_main_lost_cmd_hold();
      main_camera_tracker_active = false;
      status_warning_active = true;
      continue;
    }

    std::optional<omniperception::OmniInferenceResult> best_omni_result;
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
    bool main_tracker_hold_applied = false;
    bool main_lost_cmd_hold_applied = false;
    std::string omni_block_reason = "none";
    double omni_cmd_elapsed_ms = 0.0;
    double omni_retarget_remaining_ms = 0.0;
    double main_lost_cmd_hold_remaining_ms = 0.0;
    const auto now = std::chrono::steady_clock::now();
    io::Command command{false, false, 0.0, 0.0};
    std::optional<auto_aim::Plan> main_mpc_plan;
    std::optional<auto_aim::SentryMpcSetpoint> main_mpc_setpoint;
    bool tracker_control_ready = false;
    bool aim_point_ready = false;
    bool high_spin_center_aim_active = false;
    std::optional<auto_buff::PowerRune> buff_power_runes;
    bool buff_target_solved = false;
    double buff_detect_time_ms = 0.0;
    double buff_solve_time_ms = 0.0;
    double buff_aim_time_ms = 0.0;

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
      reset_control_session();
      clear_main_lost_cmd_hold();
      main_camera_tracker_active = false;
      left_img.release();
      right_img.release();
      back_img.release();
      clear_omni_redirect_state();

      const auto buff_detect_start = std::chrono::steady_clock::now();
      buff_power_runes = buff_detector.detect(main_img);
      const auto buff_detect_end = std::chrono::steady_clock::now();

      const auto buff_solve_start = std::chrono::steady_clock::now();
      buff_solver.solve(buff_power_runes);
      const auto buff_solve_end = std::chrono::steady_clock::now();

      const auto buff_aim_start = std::chrono::steady_clock::now();
      if (gimbal_mode == io::small_buff) {
        buff_small_target.get_target(buff_power_runes, main_timestamp);
        if (!buff_small_target.is_unsolve()) {
          buff_target_solved = true;
          command =
            buff_aimer.aim(buff_small_target, main_timestamp, gimbal->bullet_speed(), aimer_to_now);
        }
      } else {
        buff_big_target.get_target(buff_power_runes, main_timestamp);
        if (!buff_big_target.is_unsolve()) {
          buff_target_solved = true;
          command =
            buff_aimer.aim(buff_big_target, main_timestamp, gimbal->bullet_speed(), aimer_to_now);
        }
      }
      const auto buff_aim_end = std::chrono::steady_clock::now();

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

      const double buff_big_yaw = command.has_target_yaw ? command.big_yaw : command.yaw;
      const double buff_small_yaw = command.has_target_yaw ? command.small_yaw : command.yaw;

      gimbal->send_mpc(
        command.control, command.shoot, buff_big_yaw, buff_small_yaw, command.pitch, 0.0, 0.0, 0.0,
        0.0, 0, 0.0, 0.0, 0.0);

      buff_detect_time_ms = tools::delta_time(buff_detect_end, buff_detect_start) * 1e3;
      buff_solve_time_ms = tools::delta_time(buff_solve_end, buff_solve_start) * 1e3;
      buff_aim_time_ms = tools::delta_time(buff_aim_end, buff_aim_start) * 1e3;
    } else if (omni_mode) {
      reset_control_session();
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
      back_frame.result.cam = back_cam_cfg;
      if (cam_back) {
        back_frame = read_omni_frame(*cam_back, back_img, ts_back, back_cam_cfg);
      } else {
        back_img.release();
      }

      auto t_omni0 = std::chrono::steady_clock::now();
      if (left_frame.has_base_big_yaw && !left_img.empty()) {
        left_frame.result.armors = yolo_omni_left->detect(left_img, frame_count);
      }
      auto t_omni1 = std::chrono::steady_clock::now();
      if (right_frame.has_base_big_yaw && !right_img.empty()) {
        right_frame.result.armors = yolo_omni_right->detect(right_img, frame_count);
      }
      auto t_omni2 = std::chrono::steady_clock::now();
      if (yolo_omni_back && back_frame.has_base_big_yaw && !back_img.empty()) {
        back_frame.result.armors = yolo_omni_back->detect(back_img, frame_count);
      }
      auto t_omni3 = std::chrono::steady_clock::now();

      auto finalize_frame =
        [&](omniperception::OmniCandidateFrame & frame, const omniperception::OmniCamConfig & cam_cfg, double infer_ms) {
          frame.result.infer_ms = infer_ms;
          decider.armor_filter(frame.result.armors);
          decider.set_priority(frame.result.armors);
          auto_aim::apply_armor_target_mask(frame.result.armors, armor_target_mask);
          omniperception::fill_omni_candidate(frame, cam_cfg.spec, gimbal_axis_order);
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
        omni_candidate_abs_yaw_deg = selected_candidate->abs_yaw_rad * 57.3;
        omni_candidate_base_big_yaw_deg = selected_candidate->base_big_yaw_rad * 57.3;
        omni_candidate_age_ms = tools::delta_time(now, selected_candidate->timestamp) * 1e3;
        omni_selected_confidence = selected_candidate->confidence;
        omni_selected_priority = static_cast<int>(selected_candidate->priority);
        omni_selected_slot = slot_name(selected_candidate->slot);

        const auto selected_frame = std::find_if(
          candidate_frames.begin(), candidate_frames.end(), [&](const omniperception::OmniCandidateFrame & frame) {
            return same_candidate_frame(frame, selected_candidate.value());
          });
        if (selected_frame != candidate_frames.end()) best_omni_result = selected_frame->result;

        const auto decision = omniperception::evaluate_omni_retarget(
          selected_candidate.value(), reference_omni_target, gimbal_state.big_yaw,
          omni_retarget_cd_active, omni_retarget_min_delta_deg);
        omni_candidate_delta_deg = decision.candidate_delta_deg;
        omni_same_target_continuation = decision.same_target_continuation;

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
        const double target_error_deg = omniperception::angular_distance_deg(omni_hold_command->big_yaw, gimbal_state.big_yaw);
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

        const double target_error_deg = omniperception::angular_distance_deg(command.big_yaw, gimbal_state.big_yaw);
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
        omni_target_error_deg = omniperception::angular_distance_deg(command.big_yaw, gimbal_state.big_yaw);
        omni_target_reached = omni_target_error_deg.value() <= omni_hold_release_tolerance_deg;
        if (omni_hold_command.has_value() && omni_target_reached) {
          omni_hold_command.reset();
        }
      }

      if (main_lost_cmd_hold_running && main_lost_hold_setpoint.has_value()) {
        if (now < main_lost_cmd_hold_deadline) {
          auto hold_setpoint = main_lost_hold_setpoint.value();
          hold_setpoint.command.shoot = false;
          hold_setpoint.fire_ready = false;
          command = hold_setpoint.command;
          main_mpc_setpoint = hold_setpoint;
          auto_aim::dispatch_sentry_mpc(*gimbal, hold_setpoint);
          main_lost_cmd_hold_applied = true;
          main_lost_cmd_hold_remaining_ms =
            std::chrono::duration<double, std::milli>(main_lost_cmd_hold_deadline - now).count();
        } else {
          clear_main_lost_cmd_hold();
          main_camera_tracker_active = false;
        }
      }

      if (!main_lost_cmd_hold_applied) {
        const double omni_big_yaw = command.has_target_yaw ? command.big_yaw : command.yaw;
        const double omni_small_yaw =
          command.has_target_yaw ? auto_aim::nearest_continuous_yaw_rad(command.small_yaw, gimbal_state.yaw)
                                 : command.yaw;
        gimbal->send_mpc(
          command.control, command.shoot, omni_big_yaw, omni_small_yaw, command.pitch, 0.0, 0.0,
          0.0, 0.0, static_cast<uint8_t>(command.armor_id), 0.0, 0.0, 0.0);
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
        reset_control_session();
      } else {
        const auto & target = targets.front();
        const TargetSession session{
          target.name, target.armor_type, high_spin_center_aim_active};
        if (!active_target_session.has_value() || !(active_target_session.value() == session)) {
          takeover.reset();
          active_target_session = session;
        }

        const auto sentry_plan = planner.plan_sentry_world(
          std::optional<auto_aim::Target>{target}, gimbal->bullet_speed(),
          control_armor_id, high_spin_center_aim_active);
        mpc_plan = sentry_plan.world_small_yaw_plan;
        main_mpc_plan = mpc_plan;
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
        disable_gimbal(*gimbal);
        reset_control_session();
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
        main_tracker_hold_applied = true;
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
      main_mpc_setpoint = setpoint;
      auto_aim::dispatch_sentry_mpc(*gimbal, setpoint);
    }

    nlohmann::json data;
    data["mode"] = buff_mode ? 2 : (omni_mode ? 1 : 0);
    data["gimbal_mode"] = gimbal_mode_name(gimbal_mode);
    data["buff_mode"] = buff_mode ? 1 : 0;
    data["armor_num"] = armors.size();
    data["tracker_state"] = tracker_state;
    data["gimbal_status_fresh"] = gimbal_status_fresh ? 1 : 0;
    data["tracker_control_ready"] = tracker_control_ready ? 1 : 0;
    data["aim_point_ready"] = aim_point_ready ? 1 : 0;
    data["main_tracker_hold"] = main_tracker_hold_applied ? 1 : 0;
    data["main_lost_cmd_hold"] = main_lost_cmd_hold_applied ? 1 : 0;
    data["main_lost_cmd_hold_remaining_ms"] = main_lost_cmd_hold_remaining_ms;
    data["high_spin_force_fire_active"] =
      (!buff_mode && !omni_mode && shooter.high_spin_force_fire_active()) ? 1 : 0;
    data["high_spin_force_fire_enabled"] = shooter.high_spin_force_fire_enabled() ? 1 : 0;
    data["high_spin_force_fire_enter_speed"] = shooter.high_spin_force_fire_enter_speed();
    data["high_spin_force_fire_exit_speed"] = shooter.high_spin_force_fire_exit_speed();
    data["high_spin_center_aim_active"] = high_spin_center_aim_active ? 1 : 0;
    data["high_spin_center_aim_enabled"] = shooter.high_spin_center_aim_enabled() ? 1 : 0;
    data["high_spin_center_aim_enter_speed"] = shooter.high_spin_center_aim_enter_speed();
    data["high_spin_center_aim_exit_speed"] = shooter.high_spin_center_aim_exit_speed();
    data["mpc_setpoint_fire_ready"] =
      main_mpc_setpoint.has_value() && main_mpc_setpoint->fire_ready ? 1 : 0;
    data["gimbal_yaw"] = ypr[0] * 57.3;
    data["gimbal_small_yaw"] = gimbal_state.yaw * 57.3;
    data["gimbal_big_yaw"] = gimbal_state.big_yaw * 57.3;
    data["bullet_speed"] = gimbal->bullet_speed();
    data["mpc_control"] = command.control ? 1 : 0;
    data["mpc_fire"] = command.shoot ? 1 : 0;
    data["mpc_yaw"] = (command.has_target_yaw ? command.small_yaw : command.yaw) * 57.3;
    data["mpc_pitch"] = command.pitch * 57.3;
    if (main_mpc_plan.has_value()) {
      data["mpc_target_yaw"] = main_mpc_plan->target_yaw * 57.3;
      data["mpc_target_pitch"] = main_mpc_plan->target_pitch * 57.3;
      data["mpc_plan_fire"] = main_mpc_plan->fire ? 1 : 0;
    }
    if (main_mpc_setpoint.has_value()) {
      data["mpc_yaw_vel"] = main_mpc_setpoint->yaw_vel * 57.3;
      data["mpc_pitch_vel"] = main_mpc_setpoint->pitch_vel * 57.3;
      data["mpc_yaw_acc"] = main_mpc_setpoint->yaw_acc * 57.3;
      data["mpc_pitch_acc"] = main_mpc_setpoint->pitch_acc * 57.3;
      data["mpc_takeover_alpha"] = main_mpc_setpoint->takeover_alpha;
    }
    data["target_armor_id"] = static_cast<int>(command.armor_id);
    data["target_vx"] = command.vx;
    data["target_vy"] = command.vy;
    data["horizon_distance"] = command.horizon_distance;
    data["aim_source"] =
      (!omni_mode && !buff_mode && command.control) ? aimer.debug_aim_point.source : -1;
    data["aim_armor_id"] =
      (!omni_mode && !buff_mode && command.control) ? aimer.debug_aim_point.armor_id : -1;
    if (buff_mode) {
      data["buff_energy"] = gimbal_mode == io::small_buff ? "small" : "big";
      data["buff_has_target"] = buff_power_runes.has_value() ? 1 : 0;
      data["buff_target_solved"] = buff_target_solved ? 1 : 0;
      data["buff_detect_time"] = buff_detect_time_ms;
      data["buff_solve_time"] = buff_solve_time_ms;
      data["buff_aim_time"] = buff_aim_time_ms;
      if (buff_power_runes.has_value()) {
        const auto & p = buff_power_runes.value();
        data["buff_R_yaw"] = p.ypd_in_world[0];
        data["buff_R_pitch"] = p.ypd_in_world[1];
        data["buff_R_dis"] = p.ypd_in_world[2];
        data["buff_yaw"] = p.ypr_in_world[0] * 57.3;
        data["buff_pitch"] = p.ypr_in_world[1] * 57.3;
        data["buff_roll"] = p.ypr_in_world[2] * 57.3;
      }

      const auto_buff::Target & buff_target = gimbal_mode == io::small_buff
                                                ? static_cast<const auto_buff::Target &>(
                                                    buff_small_target)
                                                : static_cast<const auto_buff::Target &>(
                                                    buff_big_target);
      if (!buff_target.is_unsolve()) {
        const auto x = buff_target.ekf_x();
        data["buff_target_yaw"] = x[4] * 57.3;
        data["buff_target_angle"] = x[5] * 57.3;
        data["buff_target_spd"] = x[6] * 57.3;
      }
    }
    if (!targets.empty()) {
      const auto & target = targets.front();
      data["target_name"] = auto_aim::ARMOR_NAMES[target.name];
      data["target_is_outpost"] = target.name == auto_aim::ArmorName::outpost ? 1 : 0;
      data["target_angular_speed"] = std::abs(target.ekf_x()[7]);
      data["target_angular_speed_deg_s"] = std::abs(target.ekf_x()[7]) * 57.3;
      const auto & ekf_data = target.ekf().data;
      for (const auto * key : {
             "residual_yaw", "residual_pitch", "residual_distance", "residual_angle", "nis",
             "recent_nis_failures"}) {
        const auto iter = ekf_data.find(key);
        if (iter != ekf_data.end()) data["tracker_" + std::string(key)] = iter->second;
      }
      data["outpost_layer_locked"] = target.outpost_layer_locked() ? 1 : 0;
      data["outpost_preview_ready"] = target.outpost_unlocked_prediction_ready() ? 1 : 0;
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

    if (buff_mode) {
      if (buff_power_runes.has_value()) {
        auto & p = buff_power_runes.value();
        for (size_t i = 0; i < std::min<size_t>(4, p.target().points.size()); ++i) {
          tools::draw_point(main_img, p.target().points[i]);
        }
        tools::draw_point(main_img, p.target().center, {0, 0, 255}, 3);
        tools::draw_point(main_img, p.r_center, {0, 255, 255}, 3);
      }
      tools::draw_text(
        main_img,
        fmt::format(
          "buff {} target={} solved={}", gimbal_mode == io::small_buff ? "small" : "big",
          buff_power_runes.has_value() ? 1 : 0, buff_target_solved ? 1 : 0),
        {10, 90}, {180, 255, 180}, 0.8, 2);
    } else {
      draw_auto_aim_overlay(main_img, armors, targets, aimer, solver);
    }
    const std::string mode_label = buff_mode ? buff_mode_name(gimbal_mode)
                                             : (omni_mode ? "OMNI" : "MPC");
    tools::draw_text(main_img, fmt::format("[{}] mode={}", tracker_state, mode_label),
      {10, 30}, {255, 255, 255}, 0.8, 2);
    tools::draw_text(main_img,
      fmt::format("mpc yaw={:.2f} pitch={:.2f} fire={}",
        (command.has_target_yaw ? command.small_yaw : command.yaw) * 57.3,
        command.pitch * 57.3, command.shoot ? 1 : 0),
      {10, 60}, {154, 50, 205}, 0.8, 2);
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

    cv::Mat left_show =
      left_img.empty() ? cv::Mat::zeros(main_img.size(), main_img.type()) : left_img.clone();
    cv::Mat right_show =
      right_img.empty() ? cv::Mat::zeros(main_img.size(), main_img.type()) : right_img.clone();
    cv::Mat back_show =
      back_img.empty() ? cv::Mat::zeros(main_img.size(), main_img.type()) : back_img.clone();

    if (omni_mode && best_omni_result.has_value()) {
      const auto & best = best_omni_result.value();
      if (best.cam.spec.slot == omniperception::OmniCameraSlot::left) {
        draw_omni_overlay(left_show, best);
      } else if (best.cam.spec.slot == omniperception::OmniCameraSlot::right) {
        draw_omni_overlay(right_show, best);
      } else {
        draw_omni_overlay(back_show, best);
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
