#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/ros2/ros2_gimbal.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
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
struct OmniCamConfig
{
  omniperception::CameraSpec spec;
  std::string dev_name;
  cv::Scalar color;
};

struct OmniInferenceResult
{
  OmniCamConfig cam;
  std::list<auto_aim::Armor> armors;
  std::optional<auto_aim::Armor> top_armor;
  double delta_yaw_deg = 0.0;
  double delta_pitch_deg = 0.0;
  double infer_ms = 0.0;
};

struct OmniCandidateFrame
{
  OmniInferenceResult result;
  std::chrono::steady_clock::time_point timestamp{};
  double base_big_yaw_rad = 0.0;
  bool has_base_big_yaw = false;
  std::optional<omniperception::OmniCandidate> candidate;
};

std::string normalize_dev_name(const std::string & dev)
{
  if (dev.rfind("/dev/", 0) == 0) return dev.substr(5);
  return dev;
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

bool better_armor(const auto_aim::Armor & lhs, const auto_aim::Armor & rhs)
{
  if (lhs.priority != rhs.priority) return lhs.priority < rhs.priority;
  return lhs.confidence > rhs.confidence;
}

std::optional<auto_aim::Armor> pick_top_armor(const std::list<auto_aim::Armor> & armors)
{
  if (armors.empty()) return std::nullopt;

  auto best_it = armors.begin();
  for (auto it = std::next(armors.begin()); it != armors.end(); ++it) {
    if (better_armor(*it, *best_it)) best_it = it;
  }
  return *best_it;
}

std::pair<double, double> calc_delta_angle_deg(
  const auto_aim::Armor & armor, const OmniCamConfig & cam)
{
  const double delta_yaw =
    cam.spec.center_yaw_deg + (0.5 - armor.center_norm.x) * cam.spec.fov_h_deg;
  const double delta_pitch = (armor.center_norm.y - 0.5) * cam.spec.fov_v_deg;
  return {delta_yaw, delta_pitch};
}

double angular_distance_deg(double lhs_rad, double rhs_rad)
{
  return std::abs(tools::limit_rad(lhs_rad - rhs_rad)) * 57.3;
}

uint8_t armor_name_to_nav_id(auto_aim::ArmorName name)
{
  switch (name) {
    case auto_aim::ArmorName::one:
      return 1;   // 英雄
    case auto_aim::ArmorName::two:
      return 2;   // 工程
    case auto_aim::ArmorName::three:
      return 3;   // 3号步兵
    case auto_aim::ArmorName::four:
      return 4;   // 4号步兵
    case auto_aim::ArmorName::five:
      return 5;   // 5号步兵
    // 当前 ArmorName 枚举中没有无人机/飞镖；如果后续新增对应枚举，分别映射为 6 / 8。
    case auto_aim::ArmorName::sentry:
      return 7;   // 哨兵
    case auto_aim::ArmorName::base:
      return 9;   // 基地
    case auto_aim::ArmorName::outpost:
      return 10;  // 前哨站
    default:
      return 0;   // 无目标 / 未知
  }
}

void fill_nav_target_info(io::Command & command, const std::list<auto_aim::Target> & targets)
{
  if (targets.empty()) {
    command.armor_id = 0;
    command.vx = 0.0;
    command.vy = 0.0;
    return;
  }

  const auto & target = targets.front();
  const auto x = target.ekf_x();
  command.armor_id = armor_name_to_nav_id(target.name);
  command.vx = x[1];
  command.vy = x[3];
}

double get_horizon_distance(const std::list<auto_aim::Target> & targets)
{
  if (targets.empty()) return 0.0;
  const auto & ekf_x = targets.front().ekf_x();
  return std::sqrt(ekf_x[0] * ekf_x[0] + ekf_x[2] * ekf_x[2]);
}

void draw_omni_overlay(cv::Mat & img, const OmniInferenceResult & result)
{
  tools::draw_text(
    img,
    fmt::format("{} ({}) {:.1f}ms", slot_name(result.cam.spec.slot), result.cam.dev_name, result.infer_ms),
    {10, 30}, result.cam.color, 0.7, 2);

  if (!result.top_armor.has_value()) {
    tools::draw_text(img, "no target", {10, 60}, {120, 120, 120}, 0.7, 2);
    return;
  }

  const auto & armor = result.top_armor.value();
  tools::draw_points(img, armor.points, result.cam.color, 2);
  tools::draw_text(
    img,
    fmt::format(
      "{} pri={} conf={:.2f}", auto_aim::ARMOR_NAMES[armor.name], static_cast<int>(armor.priority),
      armor.confidence),
    {10, 60}, result.cam.color, 0.7, 2);
  tools::draw_text(
    img, fmt::format("delta yaw={:.1f} pitch={:.1f}", result.delta_yaw_deg, result.delta_pitch_deg),
    {10, 90}, result.cam.color, 0.7, 2);
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

cv::Mat resize_for_view(const cv::Mat & img)
{
  cv::Mat resized;
  cv::resize(img, resized, {640, 360});
  return resized;
}

void apply_abs_yaw_target(io::Command & command, double abs_yaw_rad)
{
  command.control = true;
  command.yaw = tools::limit_rad(abs_yaw_rad);
  command.big_yaw = abs_yaw_rad;
  command.small_yaw = command.yaw;
  command.has_target_yaw = true;
}

double nearest_continuous_yaw_rad(double wrapped_yaw_rad, double reference_yaw_rad)
{
  return reference_yaw_rad + tools::limit_rad(wrapped_yaw_rad - reference_yaw_rad);
}

double target_center_big_yaw_rad(const auto_aim::Target & target, double current_big_yaw_rad)
{
  const auto & ekf_x = target.ekf_x();
  const double wrapped_center_yaw = std::atan2(ekf_x[2], ekf_x[0]);
  return nearest_continuous_yaw_rad(wrapped_center_yaw, current_big_yaw_rad);
}

void apply_sentry_tracking_yaws(
  io::Command & command, const auto_aim::Target & target, double current_big_yaw_rad)
{
  if (!command.control) return;
  command.small_yaw = command.yaw;
  command.big_yaw = target_center_big_yaw_rad(target, current_big_yaw_rad);
  command.has_target_yaw = true;
}

std::optional<omniperception::OmniCandidate> build_omni_candidate(
  const OmniInferenceResult & result, std::chrono::steady_clock::time_point timestamp,
  double base_big_yaw_rad)
{
  if (!result.top_armor.has_value()) return std::nullopt;

  const auto & armor = result.top_armor.value();
  omniperception::OmniCandidate candidate;
  candidate.slot = result.cam.spec.slot;
  candidate.armor_name = armor.name;
  candidate.priority = armor.priority;
  candidate.confidence = armor.confidence;
  candidate.timestamp = timestamp;
  candidate.base_big_yaw_rad = base_big_yaw_rad;
  candidate.abs_yaw_rad = base_big_yaw_rad + result.delta_yaw_deg / 57.3;
  apply_abs_yaw_target(candidate.command, candidate.abs_yaw_rad);
  candidate.command.pitch = 0.26;
  return candidate;
}

omniperception::AcceptedOmniTarget make_accepted_omni_target(
  const omniperception::OmniCandidate & candidate)
{
  omniperception::AcceptedOmniTarget accepted_target;
  accepted_target.slot = candidate.slot;
  accepted_target.armor_name = candidate.armor_name;
  accepted_target.priority = candidate.priority;
  accepted_target.confidence = candidate.confidence;
  accepted_target.timestamp = candidate.timestamp;
  accepted_target.base_big_yaw_rad = candidate.base_big_yaw_rad;
  accepted_target.abs_yaw_rad = candidate.abs_yaw_rad;
  accepted_target.command = candidate.command;
  return accepted_target;
}

bool same_candidate_frame(
  const OmniCandidateFrame & frame, const omniperception::OmniCandidate & candidate)
{
  if (!frame.candidate.has_value()) return false;
  return frame.candidate->slot == candidate.slot &&
         frame.candidate->armor_name == candidate.armor_name &&
         frame.candidate->timestamp == candidate.timestamp;
}

bool same_omni_target_continuation(
  const omniperception::AcceptedOmniTarget & lhs, const omniperception::AcceptedOmniTarget & rhs,
  double retarget_min_delta_deg)
{
  if (lhs.slot != rhs.slot) return false;
  if (lhs.armor_name != rhs.armor_name) return false;
  return angular_distance_deg(lhs.abs_yaw_rad, rhs.abs_yaw_rad) < retarget_min_delta_deg;
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
      if (!cli_value.empty() && cli_value != "__yaml__") return normalize_dev_name(cli_value);
      if (yaml[yaml_key]) return normalize_dev_name(yaml[yaml_key].as<std::string>());
      return normalize_dev_name(fallback);
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
  const double omni_retarget_min_delta_deg =
    yaml["omni_retarget_min_delta_deg"] ? yaml["omni_retarget_min_delta_deg"].as<double>() : 20.0;
  const double omni_hold_release_tolerance_deg =
    yaml["omni_hold_release_tolerance_deg"] ? yaml["omni_hold_release_tolerance_deg"].as<double>() : 3.0;
  const double omni_command_timeout_s =
    yaml["omni_command_timeout_s"] ? yaml["omni_command_timeout_s"].as<double>() : 0.5;
  const double main_lost_cmd_hold_s =
    yaml["main_lost_cmd_hold_s"] ? yaml["main_lost_cmd_hold_s"].as<double>() : 0.0;
  const auto omni_read_timeout = std::chrono::milliseconds(
    std::max(1, yaml["omni_camera_read_timeout_ms"] ? yaml["omni_camera_read_timeout_ms"].as<int>() : 10));
  const auto omni_retarget_cooldown = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(omni_retarget_cooldown_s));
  const auto omni_command_timeout = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(omni_command_timeout_s));
  const auto main_lost_cmd_hold_duration =
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(main_lost_cmd_hold_s));

  tools::logger()->info(
    "[OVSentryOmni] inference devices: auto_aim={} omni={}", auto_aim_device, omni_device);

  const double omni_fov_h_deg = read_cli_or_yaml_double("fov_h", "omni_fov_h_deg", 120.0);
  const double omni_fov_v_deg = read_cli_or_yaml_double("fov_v", "omni_fov_v_deg", 67.0);
  const OmniCamConfig left_cam_cfg{
    {omniperception::OmniCameraSlot::left, "left",
     read_cam_path("left", "omni_left_path", "video0"),
     read_cli_or_yaml_double("left_yaw", "omni_left_yaw_deg", 60.0),
     omni_fov_h_deg, omni_fov_v_deg},
    read_cam_path("left", "omni_left_path", "video0"), {0, 255, 0}};
  const OmniCamConfig right_cam_cfg{
    {omniperception::OmniCameraSlot::right, "right",
     read_cam_path("right", "omni_right_path", "video2"),
     read_cli_or_yaml_double("right_yaw", "omni_right_yaw_deg", -60.0),
     omni_fov_h_deg, omni_fov_v_deg},
    read_cam_path("right", "omni_right_path", "video2"), {0, 255, 255}};
  const OmniCamConfig back_cam_cfg{
    {omniperception::OmniCameraSlot::back, "back",
     read_cam_path("back", "omni_back_path", "video4"),
     read_cli_or_yaml_double("back_yaw", "omni_back_yaw_deg", 180.0),
     omni_fov_h_deg, omni_fov_v_deg},
    read_cam_path("back", "omni_back_path", "video4"), {255, 200, 0}};

  tools::logger()->info(
    "[OVSentryOmni] omni cams: /dev/{} /dev/{} /dev/{}", left_cam_cfg.dev_name,
    right_cam_cfg.dev_name, back_cam_cfg.dev_name);

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;
  const bool display = !cli.has("no-display");
  constexpr bool aimer_to_now = true;

  std::unique_ptr<io::ROS2Gimbal> gimbal;
  std::unique_ptr<io::Camera> auto_aim_camera;
  try {
    gimbal = std::make_unique<io::ROS2Gimbal>(config_path);
    auto_aim_camera = std::make_unique<io::Camera>(config_path);
  } catch (const std::exception & e) {
    tools::logger()->error("[OVSentryOmni] gimbal/camera init failed: {}", e.what());
    return 1;
  }

  auto_aim::YOLO yolo_auto(config_path, false, "auto_aim_device");
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  omniperception::Decider decider(config_path);

  std::unique_ptr<auto_aim::YOLO> yolo_omni_left;
  std::unique_ptr<auto_aim::YOLO> yolo_omni_right;
  std::unique_ptr<auto_aim::YOLO> yolo_omni_back;
  try {
    yolo_omni_left = std::make_unique<auto_aim::YOLO>(config_path, false, "omni_device");
    yolo_omni_right = std::make_unique<auto_aim::YOLO>(config_path, false, "omni_device");
    yolo_omni_back = std::make_unique<auto_aim::YOLO>(config_path, false, "omni_device");
  } catch (const std::exception & e) {
    tools::logger()->error("[OVSentryOmni] omni model init failed: {}", e.what());
    tools::logger()->error(
      "[OVSentryOmni] 请检查 {} 中 omni_device 是否可用（当前={}）。", config_path, omni_device);
    return 1;
  }

  io::USBCamera cam_left(left_cam_cfg.dev_name, config_path);
  io::USBCamera cam_right(right_cam_cfg.dev_name, config_path);
  io::USBCamera cam_back(back_cam_cfg.dev_name, config_path);
  cam_left.device_name = left_cam_cfg.spec.label;
  cam_right.device_name = right_cam_cfg.spec.label;
  cam_back.device_name = back_cam_cfg.spec.label;

  cv::Mat main_img;
  cv::Mat left_img, right_img, back_img;
  std::chrono::steady_clock::time_point main_timestamp;
  std::chrono::steady_clock::time_point ts_left, ts_right, ts_back;
  std::optional<io::Command> omni_hold_command;
  std::optional<io::Command> last_main_tracking_command;
  std::optional<omniperception::AcceptedOmniTarget> session_accepted_omni_target;
  std::optional<omniperception::AcceptedOmniTarget> cooldown_anchor_omni_target;
  std::optional<omniperception::AcceptedOmniTarget> active_omni_timeout_target;
  std::chrono::steady_clock::time_point omni_retarget_cooldown_deadline{};
  std::chrono::steady_clock::time_point active_omni_timeout_started_at{};
  std::chrono::steady_clock::time_point main_lost_cmd_hold_deadline{};
  bool active_omni_timeout_running = false;
  bool main_lost_cmd_hold_running = false;
  bool prev_omni_mode = false;
  int frame_count = 0;

  while (!exiter.exit()) {
    try {
      auto_aim_camera->read(main_img, main_timestamp);
      if (main_img.empty()) {
        tools::logger()->warn("[OVSentryOmni] 主相机空图像，跳过此帧");
        continue;
      }
    } catch (const std::exception & e) {
      tools::logger()->error("[OVSentryOmni] 主相机读取失败: {}", e.what());
      continue;
    }

    frame_count++;

    Eigen::Quaterniond q = gimbal->imu_at_image(main_timestamp);
    solver.set_R_gimbal2world(q);
    //recorder.record(main_img, q, main_timestamp);

    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
    const auto gimbal_state = gimbal->state();
    double omni_base_yaw_rad = ypr[0];

    auto t0 = std::chrono::steady_clock::now();
    auto armors = yolo_auto.detect(main_img, frame_count);
    auto t1 = std::chrono::steady_clock::now();

    decider.armor_filter(armors);
    decider.set_priority(armors);

    auto targets = tracker.track(armors, main_timestamp);

    io::Command command{false, false, 0, 0};
    bool omni_hold_applied = false;
    const std::string tracker_state = tracker.state();
    const bool omni_mode = tracker_state == "lost";

    std::optional<double> omni_target_abs_yaw_deg;
    std::optional<OmniInferenceResult> best_omni_result;
    std::optional<double> omni_candidate_abs_yaw_deg;
    std::optional<double> omni_candidate_base_big_yaw_deg;
    std::optional<double> omni_candidate_age_ms;
    std::optional<double> omni_candidate_delta_deg;
    std::optional<double> omni_target_error_deg;
    std::optional<double> omni_selected_confidence;
    std::optional<int> omni_selected_priority;
    std::optional<std::string> omni_selected_slot;
    bool omni_retarget_blocked = false;
    bool omni_retarget_cd_active = false;
    bool omni_same_target_continuation = false;
    bool omni_target_reached = false;
    bool omni_cmd_timeout_active = false;
    bool omni_cmd_timed_out = false;
    bool main_lost_cmd_hold_applied = false;
    std::string omni_block_reason = "none";
    double omni_cmd_elapsed_ms = 0.0;
    double omni_retarget_remaining_ms = 0.0;
    double main_lost_cmd_hold_remaining_ms = 0.0;
    const auto now = std::chrono::steady_clock::now();

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
      if (main_lost_cmd_hold_duration > std::chrono::steady_clock::duration::zero() &&
          last_main_tracking_command.has_value()) {
        main_lost_cmd_hold_deadline = now + main_lost_cmd_hold_duration;
        main_lost_cmd_hold_running = true;
      } else {
        main_lost_cmd_hold_running = false;
      }
    } else if (!omni_mode && prev_omni_mode) {
      clear_omni_redirect_state();
      main_lost_cmd_hold_running = false;
      main_lost_cmd_hold_deadline = std::chrono::steady_clock::time_point{};
    }

    if (omni_mode) {
      omni_base_yaw_rad = gimbal_state.big_yaw;

      auto read_omni_frame = [&](io::USBCamera & camera, cv::Mat & img,
                                 std::chrono::steady_clock::time_point & ts,
                                 const OmniCamConfig & cam_cfg) {
          OmniCandidateFrame frame;
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
      auto back_frame = read_omni_frame(cam_back, back_img, ts_back, back_cam_cfg);

      auto t_omni0 = std::chrono::steady_clock::now();
      if (left_frame.has_base_big_yaw && !left_img.empty()) {
        left_frame.result.armors = yolo_omni_left->detect(left_img, frame_count);
      }
      auto t_omni1 = std::chrono::steady_clock::now();
      if (right_frame.has_base_big_yaw && !right_img.empty()) {
        right_frame.result.armors = yolo_omni_right->detect(right_img, frame_count);
      }
      auto t_omni2 = std::chrono::steady_clock::now();
      if (back_frame.has_base_big_yaw && !back_img.empty()) {
        back_frame.result.armors = yolo_omni_back->detect(back_img, frame_count);
      }
      auto t_omni3 = std::chrono::steady_clock::now();

      auto finalize_frame = [&](OmniCandidateFrame & frame, const OmniCamConfig & cam_cfg,
                                double infer_ms) {
          frame.result.infer_ms = infer_ms;
          decider.armor_filter(frame.result.armors);
          decider.set_priority(frame.result.armors);
          frame.result.top_armor = pick_top_armor(frame.result.armors);
          if (frame.result.top_armor.has_value()) {
            auto [dyaw, dpitch] = calc_delta_angle_deg(frame.result.top_armor.value(), cam_cfg);
            frame.result.delta_yaw_deg = dyaw;
            frame.result.delta_pitch_deg = dpitch;
            frame.candidate =
              build_omni_candidate(frame.result, frame.timestamp, frame.base_big_yaw_rad);
          }
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

      std::vector<OmniCandidateFrame> candidate_frames;
      if (left_frame.candidate.has_value()) candidate_frames.push_back(left_frame);
      if (right_frame.candidate.has_value()) candidate_frames.push_back(right_frame);
      if (back_frame.candidate.has_value()) candidate_frames.push_back(back_frame);

      std::vector<omniperception::OmniCandidate> candidates;
      candidates.reserve(candidate_frames.size());
      for (const auto & frame : candidate_frames) {
        candidates.push_back(frame.candidate.value());
      }

      const auto selected_candidate = omniperception::select_omni_candidate(
        candidates, reference_omni_target, gimbal_state.big_yaw, omni_retarget_min_delta_deg);
      if (selected_candidate.has_value()) {
        omni_candidate_abs_yaw_deg = selected_candidate->abs_yaw_rad * 57.3;
        omni_candidate_base_big_yaw_deg = selected_candidate->base_big_yaw_rad * 57.3;
        omni_candidate_age_ms = tools::delta_time(now, selected_candidate->timestamp) * 1e3;
        omni_selected_confidence = selected_candidate->confidence;
        omni_selected_priority = static_cast<int>(selected_candidate->priority);
        omni_selected_slot = slot_name(selected_candidate->slot);
        omni_base_yaw_rad = selected_candidate->base_big_yaw_rad;

        const auto selected_frame = std::find_if(
          candidate_frames.begin(), candidate_frames.end(),
          [&](const OmniCandidateFrame & frame) {
            return same_candidate_frame(frame, selected_candidate.value());
          });
        if (selected_frame != candidate_frames.end()) {
          best_omni_result = selected_frame->result;
        }

        const auto decision = omniperception::evaluate_omni_retarget(
          selected_candidate.value(), reference_omni_target, gimbal_state.big_yaw,
          omni_retarget_cd_active, omni_retarget_min_delta_deg);
        omni_candidate_delta_deg = decision.candidate_delta_deg;
        omni_same_target_continuation = decision.same_target_continuation;

        if (decision.accept) {
          command = selected_candidate->command;
          omni_hold_command = command;
          const auto accepted_target = make_accepted_omni_target(selected_candidate.value());
          session_accepted_omni_target = accepted_target;
          if (
            !active_omni_timeout_running || !active_omni_timeout_target.has_value() ||
            !same_omni_target_continuation(
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
          angular_distance_deg(omni_hold_command->big_yaw, gimbal_state.big_yaw);
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

        const double target_error_deg = angular_distance_deg(command.big_yaw, gimbal_state.big_yaw);
        if (target_error_deg > omni_hold_release_tolerance_deg) {
          if (
            active_omni_timeout_running &&
            (now - active_omni_timeout_started_at) > omni_command_timeout) {
            tools::logger()->warn(
              "[OVSentryOmni] omni command timed out after {:.0f}ms without reaching target yaw",
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

      if (!command.control && main_lost_cmd_hold_running && last_main_tracking_command.has_value()) {
        if (now < main_lost_cmd_hold_deadline) {
          command = last_main_tracking_command.value();
          main_lost_cmd_hold_applied = true;
          main_lost_cmd_hold_remaining_ms =
            std::chrono::duration<double, std::milli>(main_lost_cmd_hold_deadline - now).count();
        } else {
          main_lost_cmd_hold_running = false;
          main_lost_cmd_hold_deadline = std::chrono::steady_clock::time_point{};
          last_main_tracking_command.reset();
        }
      }
    } else {
      left_img.release();
      right_img.release();
      back_img.release();
      clear_omni_redirect_state();
      command = aimer.aim(targets, main_timestamp, gimbal->bullet_speed(), aimer_to_now);
      if (tracker_state == "tracking" && command.control && !targets.empty()) {
        apply_sentry_tracking_yaws(command, targets.front(), gimbal_state.big_yaw);
      }
      if (command.control) {
        last_main_tracking_command = command;
      }
      main_lost_cmd_hold_running = false;
      main_lost_cmd_hold_deadline = std::chrono::steady_clock::time_point{};
    }

    command.shoot = shooter.shoot(command, aimer, targets, ypr, tracker_state == "tracking");
    command.horizon_distance = command.control ? get_horizon_distance(targets) : 0.0;
    fill_nav_target_info(command, targets);
    if (omni_mode && command.control && command.has_target_yaw) {
      omni_target_error_deg = angular_distance_deg(command.big_yaw, gimbal_state.big_yaw);
      omni_target_reached = omni_target_error_deg.value() <= omni_hold_release_tolerance_deg;
      if (omni_hold_command.has_value() && omni_target_reached) {
        omni_hold_command.reset();
      }
    }
    gimbal->send(command);

    const double yolo_time = tools::delta_time(t1, t0) * 1e3;

    nlohmann::json data;
    data["mode"] = omni_mode ? 1 : 0;
    data["armor_num"] = armors.size();
    data["tracker_state"] = tracker_state;
    data["gimbal_yaw"] = ypr[0] * 57.3;
    data["gimbal_small_yaw"] = gimbal_state.yaw * 57.3;
    data["gimbal_pitch"] = ypr[1] * 57.3;
    data["gimbal_big_yaw"] = gimbal_state.big_yaw * 57.3;
    data["omni_base_yaw"] = omni_base_yaw_rad * 57.3;
    data["bullet_speed"] = gimbal->bullet_speed();
    data["cmd_control"] = command.control ? 1 : 0;
    data["cmd_shoot"] = command.shoot ? 1 : 0;
    data["cmd_yaw"] = command.yaw * 57.3;
    data["cmd_pitch"] = command.pitch * 57.3;
    if (command.has_target_yaw) {
      data["cmd_big_yaw"] = command.big_yaw * 57.3;
      data["cmd_small_yaw"] = command.small_yaw * 57.3;
    }
    if (!targets.empty()) {
      const auto & target = targets.front();
      const Eigen::VectorXd x = target.ekf_x();
      data["ekf_x"] = x[0];
      data["ekf_vx"] = x[1];
      data["ekf_y"] = x[2];
      data["ekf_vy"] = x[3];
      data["ekf_z"] = x[4];
      data["ekf_vz"] = x[5];
      data["ekf_a"] = x[6] * 57.3;
      data["ekf_w"] = x[7];
      data["last_id"] = target.last_id;
      data["aim_id"] = aimer.debug_aim_point.armor_id;
      data["aim_source"] = aimer.debug_aim_point.source;
      for (const auto & key :
           {"outpost_selected_id", "outpost_layer_residual", "outpost_phase_residual",
            "outpost_center_speed", "outpost_layer_locked", "init_best_score", "init_margin",
            "init_distinct_layers", "init_z_vz", "init_z_max_residual"}) {
        const auto iter = target.ekf().data.find(key);
        if (iter != target.ekf().data.end()) data[key] = iter->second;
      }
    }
    data["horizon_distance"] = command.horizon_distance;
    data["target_armor_id"] = static_cast<int>(command.armor_id);
    data["target_vx"] = command.vx;
    data["target_vy"] = command.vy;
    if (omni_target_abs_yaw_deg.has_value()) data["omni_target_yaw"] = omni_target_abs_yaw_deg.value();
    if (omni_candidate_abs_yaw_deg.has_value()) data["omni_candidate_abs_yaw"] = omni_candidate_abs_yaw_deg.value();
    if (omni_candidate_base_big_yaw_deg.has_value()) {
      data["omni_candidate_base_big_yaw"] = omni_candidate_base_big_yaw_deg.value();
    }
    if (omni_candidate_age_ms.has_value()) data["omni_candidate_age_ms"] = omni_candidate_age_ms.value();
    data["omni_yaw_hold"] = omni_hold_applied ? 1 : 0;
    data["omni_target_reached"] = omni_target_reached ? 1 : 0;
    data["omni_cmd_timeout_active"] = omni_cmd_timeout_active ? 1 : 0;
    data["omni_cmd_timed_out"] = omni_cmd_timed_out ? 1 : 0;
    data["omni_cmd_elapsed_ms"] = omni_cmd_elapsed_ms;
    data["omni_retarget_cd_active"] = omni_retarget_cd_active ? 1 : 0;
    data["omni_retarget_blocked"] = omni_retarget_blocked ? 1 : 0;
    data["omni_retarget_remaining_ms"] = omni_retarget_remaining_ms;
    data["main_lost_cmd_hold_active"] = main_lost_cmd_hold_applied ? 1 : 0;
    data["main_lost_cmd_hold_remaining_ms"] = main_lost_cmd_hold_remaining_ms;
    data["omni_same_target_continuation"] = omni_same_target_continuation ? 1 : 0;
    data["omni_block_reason"] = omni_block_reason;
    if (omni_candidate_delta_deg.has_value()) data["omni_candidate_delta_deg"] = omni_candidate_delta_deg.value();
    if (omni_target_error_deg.has_value()) data["omni_target_error_deg"] = omni_target_error_deg.value();
    if (omni_selected_confidence.has_value()) data["omni_selected_confidence"] = omni_selected_confidence.value();
    if (omni_selected_priority.has_value()) data["omni_selected_priority"] = omni_selected_priority.value();
    if (omni_selected_slot.has_value()) data["omni_selected_slot"] = omni_selected_slot.value();
    data["yolo_time"] = yolo_time;
    plotter.plot(data);

    prev_omni_mode = omni_mode;
    if (!display) continue;

    draw_auto_aim_overlay(main_img, targets, aimer, solver);

    tools::draw_text(main_img, fmt::format("[{}]", tracker_state), {10, 30}, {255, 255, 255}, 0.8, 2);
    tools::draw_text(
      main_img, fmt::format("mode={}", omni_mode ? "OMNI" : "AUTO_AIM"), {10, 60},
      omni_mode ? cv::Scalar(0, 220, 255) : cv::Scalar(0, 255, 0), 0.8, 2);
    tools::draw_text(
      main_img,
      fmt::format(
        "cmd yaw={:.2f} pitch={:.2f} shoot={}", command.yaw * 57.3, command.pitch * 57.3,
        command.shoot ? 1 : 0),
      {10, 90}, {154, 50, 205}, 0.8, 2);
    tools::draw_text(
      main_img, fmt::format("omni base yaw={:.2f}", omni_base_yaw_rad * 57.3), {10, 120},
      cv::Scalar(0, 220, 255), 0.8, 2);
    if (omni_target_abs_yaw_deg.has_value()) {
      tools::draw_text(
        main_img, fmt::format("omni target yaw={:.2f} deg", omni_target_abs_yaw_deg.value()),
        {10, 150}, {0, 255, 255}, 0.8, 2);
    }
    if (omni_hold_applied) {
      tools::draw_text(
        main_img, fmt::format("omni yaw hold err<={:.1f} deg", omni_hold_release_tolerance_deg), {10, 180},
        {0, 180, 255}, 0.8, 2);
    }
    if (omni_retarget_cd_active) {
      tools::draw_text(
        main_img, fmt::format("omni retarget cd {:.0f}ms", omni_retarget_remaining_ms), {10, 210},
        omni_retarget_blocked ? cv::Scalar(0, 180, 255) : cv::Scalar(255, 220, 0), 0.8, 2);
    }
    if (omni_cmd_timeout_active || omni_cmd_timed_out) {
      tools::draw_text(
        main_img,
        fmt::format(
          "omni cmd timeout {:.0f}/{:.0f}ms hit={}", omni_cmd_elapsed_ms,
          omni_command_timeout_s * 1e3, omni_cmd_timed_out ? 1 : 0),
        {10, 240}, omni_cmd_timed_out ? cv::Scalar(0, 120, 255) : cv::Scalar(180, 255, 180), 0.8,
        2);
    }
    if (omni_candidate_delta_deg.has_value()) {
      tools::draw_text(
        main_img, fmt::format("omni candidate delta={:.1f} deg", omni_candidate_delta_deg.value()),
        {10, 270}, {255, 255, 0}, 0.8, 2);
    }
    if (omni_selected_slot.has_value()) {
      tools::draw_text(
        main_img,
        fmt::format(
          "slot={} pri={} conf={:.2f}", omni_selected_slot.value(),
          omni_selected_priority.value_or(0), omni_selected_confidence.value_or(0.0)),
        {10, 300}, {180, 255, 180}, 0.8, 2);
    }
    tools::draw_text(
      main_img, fmt::format("block={} same={}", omni_block_reason, omni_same_target_continuation ? 1 : 0),
      {10, 330}, {180, 220, 255}, 0.8, 2);
    if (omni_target_error_deg.has_value()) {
      tools::draw_text(
        main_img,
        fmt::format(
          "omni target err={:.1f} reached={}", omni_target_error_deg.value(), omni_target_reached ? 1 : 0),
        {10, 360}, {180, 255, 180}, 0.8, 2);
    }

    cv::Mat left_show = left_img.empty() ? cv::Mat::zeros(main_img.size(), main_img.type()) : left_img.clone();
    cv::Mat right_show =
      right_img.empty() ? cv::Mat::zeros(main_img.size(), main_img.type()) : right_img.clone();
    cv::Mat back_show = back_img.empty() ? cv::Mat::zeros(main_img.size(), main_img.type()) : back_img.clone();

    tools::draw_text(main_img, "MAIN (AUTO AIM)", {10, 390}, {0, 255, 0}, 0.8, 2);
    tools::draw_text(
      left_show, fmt::format("{} ({:.0f} deg)", slot_name(left_cam_cfg.spec.slot), left_cam_cfg.spec.center_yaw_deg),
      {10, 30}, left_cam_cfg.color, 0.8, 2);
    tools::draw_text(
      right_show,
      fmt::format("{} ({:.0f} deg)", slot_name(right_cam_cfg.spec.slot), right_cam_cfg.spec.center_yaw_deg),
      {10, 30}, right_cam_cfg.color, 0.8, 2);
    tools::draw_text(
      back_show, fmt::format("{} ({:.0f} deg)", slot_name(back_cam_cfg.spec.slot), back_cam_cfg.spec.center_yaw_deg),
      {10, 30}, back_cam_cfg.color, 0.8, 2);

    if (!omni_mode) {
      tools::draw_text(left_show, "inactive", {10, 60}, {120, 120, 120}, 0.7, 2);
      tools::draw_text(right_show, "inactive", {10, 60}, {120, 120, 120}, 0.7, 2);
      tools::draw_text(back_show, "inactive", {10, 60}, {120, 120, 120}, 0.7, 2);
    } else if (best_omni_result.has_value()) {
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

    cv::imshow("ovsentry_omni", canvas);
    if (cv::waitKey(1) == 'q') break;
  }

  return 0;
}
