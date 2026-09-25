#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/ros2/ros2_gimbal.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tasks/omniperception/ovsentry_omni_logic.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/thread_safe_queue.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace
{
struct OmniCamConfig
{
  omniperception::CameraSpec spec;
  std::string dev_name;
};

struct OmniFrame
{
  std::list<auto_aim::Armor> armors;
  std::optional<auto_aim::Armor> top_armor;
  std::chrono::steady_clock::time_point timestamp{};
  double base_big_yaw_rad = 0.0;
  std::optional<omniperception::OmniCandidate> candidate;
};

std::string normalize_dev_name(const std::string & dev)
{
  if (dev.rfind("/dev/", 0) == 0) return dev.substr(5);
  return dev;
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
  return {
    cam.spec.center_yaw_deg + (0.5 - armor.center_norm.x) * cam.spec.fov_h_deg,
    (armor.center_norm.y - 0.5) * cam.spec.fov_v_deg};
}

double nearest_continuous_yaw_rad(double wrapped_yaw_rad, double reference_yaw_rad)
{
  return reference_yaw_rad + tools::limit_rad(wrapped_yaw_rad - reference_yaw_rad);
}

double target_center_big_yaw_rad(const auto_aim::Target & target, double current_big_yaw_rad)
{
  const auto & x = target.ekf_x();
  return nearest_continuous_yaw_rad(std::atan2(x[2], x[0]), current_big_yaw_rad);
}

void apply_sentry_tracking_yaws(
  io::Command & command, const auto_aim::Target & target, double current_big_yaw_rad)
{
  if (!command.control) return;
  command.small_yaw = command.yaw;
  command.big_yaw = target_center_big_yaw_rad(target, current_big_yaw_rad);
  command.has_target_yaw = true;
}

void apply_abs_yaw_target(io::Command & command, double abs_yaw_rad)
{
  command.control = true;
  command.yaw = tools::limit_rad(abs_yaw_rad);
  command.big_yaw = abs_yaw_rad;
  command.small_yaw = command.yaw;
  command.has_target_yaw = true;
}

double horizon_distance(const auto_aim::Target & target)
{
  const auto & x = target.ekf_x();
  return std::sqrt(x[0] * x[0] + x[2] * x[2]);
}

std::optional<omniperception::OmniCandidate> make_candidate(
  const OmniFrame & frame, const OmniCamConfig & cam)
{
  if (!frame.top_armor.has_value()) return std::nullopt;
  const auto & armor = frame.top_armor.value();
  const auto [delta_yaw_deg, delta_pitch_deg] = calc_delta_angle_deg(armor, cam);

  omniperception::OmniCandidate candidate;
  candidate.slot = cam.spec.slot;
  candidate.armor_name = armor.name;
  candidate.priority = armor.priority;
  candidate.confidence = armor.confidence;
  candidate.timestamp = frame.timestamp;
  candidate.base_big_yaw_rad = frame.base_big_yaw_rad;
  candidate.abs_yaw_rad = frame.base_big_yaw_rad + delta_yaw_deg / 57.3;
  apply_abs_yaw_target(candidate.command, candidate.abs_yaw_rad);
  candidate.command.pitch = 0.26 + delta_pitch_deg / 57.3;
  return candidate;
}

OmniFrame detect_omni_frame(
  io::USBCamera & camera, auto_aim::YOLO & yolo, io::ROS2Gimbal & gimbal,
  omniperception::Decider & decider, cv::Mat & img, std::chrono::milliseconds timeout,
  const OmniCamConfig & cam_cfg, int frame_count)
{
  OmniFrame frame;
  const bool ok = camera.read_with_timeout(img, frame.timestamp, timeout);
  if (!ok || img.empty()) return frame;

  frame.base_big_yaw_rad = gimbal.big_yaw_at_image(frame.timestamp);
  frame.armors = yolo.detect(img, frame_count);
  decider.armor_filter(frame.armors);
  decider.set_priority(frame.armors);
  frame.top_armor = pick_top_armor(frame.armors);
  frame.candidate = make_candidate(frame, cam_cfg);
  return frame;
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
  "{fov_v          | 67                      | USB相机垂直视场角(deg) }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  auto yaml = tools::load(config_path);
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

  const double omni_retarget_min_delta_deg =
    yaml["omni_retarget_min_delta_deg"] ? yaml["omni_retarget_min_delta_deg"].as<double>() : 20.0;
  const double omni_hold_release_tolerance_deg =
    yaml["omni_hold_release_tolerance_deg"] ? yaml["omni_hold_release_tolerance_deg"].as<double>() : 3.0;
  const auto omni_read_timeout = std::chrono::milliseconds(
    std::max(1, yaml["omni_camera_read_timeout_ms"] ? yaml["omni_camera_read_timeout_ms"].as<int>() : 10));
  const double omni_fov_h_deg = read_cli_or_yaml_double("fov_h", "omni_fov_h_deg", 120.0);
  const double omni_fov_v_deg = read_cli_or_yaml_double("fov_v", "omni_fov_v_deg", 67.0);

  const OmniCamConfig left_cam_cfg{
    {omniperception::OmniCameraSlot::left, "left", read_cam_path("left", "omni_left_path", "video0"),
     read_cli_or_yaml_double("left_yaw", "omni_left_yaw_deg", 60.0), omni_fov_h_deg, omni_fov_v_deg},
    read_cam_path("left", "omni_left_path", "video0")};
  const OmniCamConfig right_cam_cfg{
    {omniperception::OmniCameraSlot::right, "right", read_cam_path("right", "omni_right_path", "video2"),
     read_cli_or_yaml_double("right_yaw", "omni_right_yaw_deg", -60.0), omni_fov_h_deg, omni_fov_v_deg},
    read_cam_path("right", "omni_right_path", "video2")};
  const OmniCamConfig back_cam_cfg{
    {omniperception::OmniCameraSlot::back, "back", read_cam_path("back", "omni_back_path", "video4"),
     read_cli_or_yaml_double("back_yaw", "omni_back_yaw_deg", 180.0), omni_fov_h_deg, omni_fov_v_deg},
    read_cam_path("back", "omni_back_path", "video4")};

  tools::Exiter exiter;
  auto gimbal = std::make_unique<io::ROS2Gimbal>(config_path);
  auto auto_aim_camera = std::make_unique<io::Camera>(config_path);

  auto_aim::YOLO yolo_auto(config_path, false, "auto_aim_device");
  auto_aim::YOLO yolo_omni_left(config_path, false, "omni_device");
  auto_aim::YOLO yolo_omni_right(config_path, false, "omni_device");
  auto_aim::YOLO yolo_omni_back(config_path, false, "omni_device");
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);
  omniperception::Decider decider(config_path);

  io::USBCamera cam_left(left_cam_cfg.dev_name, config_path);
  io::USBCamera cam_right(right_cam_cfg.dev_name, config_path);
  io::USBCamera cam_back(back_cam_cfg.dev_name, config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::atomic<bool> quit{false};
  std::atomic<bool> mpc_enabled{false};
  std::atomic<uint64_t> mpc_generation{0};
  std::mutex gimbal_send_mutex;

  std::thread mpc_thread([&]() {
    while (!quit.load()) {
      if (!mpc_enabled.load()) {
        std::this_thread::sleep_for(10ms);
        continue;
      }

      const auto local_generation = mpc_generation.load();
      auto target = target_queue.front();
      auto gs = gimbal->state();
      auto plan = planner.plan(target, gs.bullet_speed);

      if (!mpc_enabled.load() || local_generation != mpc_generation.load()) {
        continue;
      }

      double big_yaw = plan.yaw;
      double small_yaw = plan.yaw;
      if (target.has_value() && plan.control) {
        big_yaw = target_center_big_yaw_rad(target.value(), gs.big_yaw);
        small_yaw = plan.yaw;
      }

      std::lock_guard<std::mutex> lk(gimbal_send_mutex);
      if (!mpc_enabled.load() || local_generation != mpc_generation.load()) {
        continue;
      }
      gimbal->send_mpc(
        plan.control, plan.fire, big_yaw, small_yaw, plan.pitch, plan.yaw_vel, plan.pitch_vel,
        plan.yaw_acc, plan.pitch_acc);
      std::this_thread::sleep_for(10ms);
    }
  });

  cv::Mat main_img, left_img, right_img, back_img;
  std::chrono::steady_clock::time_point main_timestamp;
  std::optional<io::Command> omni_hold_command;
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
    solver.set_R_gimbal2world(gimbal->imu_at_image(main_timestamp));

    auto armors = yolo_auto.detect(main_img, frame_count);
    decider.armor_filter(armors);
    decider.set_priority(armors);
    auto targets = tracker.track(armors, main_timestamp);
    const std::string tracker_state = tracker.state();
    const bool omni_mode = tracker_state == "lost";
    const bool enable_mpc = !omni_mode && !targets.empty();

    if (enable_mpc) {
      target_queue.push(targets.front());
      if (!mpc_enabled.exchange(true)) {
        mpc_generation.fetch_add(1);
      }
      // 进入主相机 MPC 跟踪后，清掉旧的全向保持目标，避免下次丢失时复用过期 yaw。
      omni_hold_command.reset();
      left_img.release();
      right_img.release();
      back_img.release();
      continue;
    }

    target_queue.push(std::nullopt);
    if (mpc_enabled.exchange(false)) {
      mpc_generation.fetch_add(1);
    }

    const auto gimbal_state = gimbal->state();
    auto left_frame = detect_omni_frame(
      cam_left, yolo_omni_left, *gimbal, decider, left_img, omni_read_timeout, left_cam_cfg, frame_count);
    auto right_frame = detect_omni_frame(
      cam_right, yolo_omni_right, *gimbal, decider, right_img, omni_read_timeout, right_cam_cfg, frame_count);
    auto back_frame = detect_omni_frame(
      cam_back, yolo_omni_back, *gimbal, decider, back_img, omni_read_timeout, back_cam_cfg, frame_count);

    std::vector<omniperception::OmniCandidate> candidates;
    if (left_frame.candidate.has_value()) candidates.push_back(left_frame.candidate.value());
    if (right_frame.candidate.has_value()) candidates.push_back(right_frame.candidate.value());
    if (back_frame.candidate.has_value()) candidates.push_back(back_frame.candidate.value());

    io::Command omni_command{false, false, 0.0, 0.0};
    std::optional<omniperception::AcceptedOmniTarget> no_reference_target;
    const auto selected_candidate = omniperception::select_omni_candidate(
      candidates, no_reference_target, gimbal_state.big_yaw, omni_retarget_min_delta_deg);

    if (selected_candidate.has_value()) {
      omni_command = selected_candidate->command;
      omni_hold_command = omni_command;
    } else if (omni_hold_command.has_value()) {
      const double error_deg =
        std::abs(tools::limit_rad(omni_hold_command->big_yaw - gimbal_state.big_yaw)) * 57.3;
      if (error_deg > omni_hold_release_tolerance_deg) {
        omni_command = omni_hold_command.value();
      } else {
        omni_hold_command.reset();
      }
    }

    std::lock_guard<std::mutex> lk(gimbal_send_mutex);
    gimbal->send(omni_command);
  }

  quit.store(true);
  if (mpc_thread.joinable()) mpc_thread.join();
  {
    std::lock_guard<std::mutex> lk(gimbal_send_mutex);
    gimbal->send_mpc(false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  }

  return 0;
}
