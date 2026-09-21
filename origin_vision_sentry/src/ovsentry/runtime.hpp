#ifndef OVSENTRY__RUNTIME_HPP
#define OVSENTRY__RUNTIME_HPP

#include <chrono>
#include <list>
#include <memory>
#include <optional>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "io/camera.hpp"
#include "io/command.hpp"
#include "io/ros2/ros2_gimbal_state.hpp"
#include "subscribers.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tools/exiter.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "types.hpp"

namespace io
{
class ROS2Gimbal;
}

namespace ovsentry
{

class AutoAimTask;
class BuffTask;

struct FrameState
{
  cv::Mat main_img, left_img, right_img, back_img;
  std::chrono::steady_clock::time_point main_timestamp{}, ts_left{}, ts_right{}, ts_back{};
  std::chrono::steady_clock::time_point t0{}, t1{}, now{};
  Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
  io::ROS2GimbalState gimbal_state{};
  Eigen::Vector3d ypr{Eigen::Vector3d::Zero()};
  ArmorTargetMask armor_target_mask{};
  std::list<auto_aim::Armor> armors;
  std::list<auto_aim::Target> targets;
  std::string tracker_state = "idle";
  int frame_count = 0;
  io::Command command{false, false, 0.0, 0.0};

  std::optional<OmniInferenceResult> best_omni_result;
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

  std::optional<auto_buff::PowerRune> buff_power_rune;
  auto_aim::Plan buff_plan{false, false, 0, 0, 0, 0, 0, 0, 0, 0};
  double buff_detect_ms = 0.0;
  bool buff_target_ready = false;
  bool buff_command_held = false;

  void reset_outputs();
};

class SentryRuntime
{
public:
  SentryRuntime(RuntimeConfig cfg, const char * window_title, bool omni_mosaic);
  ~SentryRuntime();

  const RuntimeConfig & cfg() const { return cfg_; }
  FrameState & frame() { return frame_; }
  const FrameState & frame() const { return frame_; }

  bool exit() const { return exiter_.exit(); }
  bool read_main_frame();
  void update_sensors();
  void reset_outputs();
  bool buff_requested() const;

  io::ROS2Gimbal & gimbal() { return *gimbal_; }
  auto_aim::Solver & solver() { return solver_; }
  omniperception::Decider & decider() { return decider_; }

  void send_idle();
  bool finish_frame(bool small_buff, bool omni, AutoAimTask * auto_aim, BuffTask * buff);
  void shutdown();

private:
  void publish_telemetry(bool small_buff, bool omni, AutoAimTask * auto_aim, BuffTask * buff);
  bool render_display(bool small_buff, bool omni, AutoAimTask * auto_aim, BuffTask * buff);

  RuntimeConfig cfg_;
  const char * window_title_;
  bool omni_mosaic_;
  tools::Exiter exiter_;
  tools::Plotter plotter_;
  tools::Recorder recorder_;
  std::unique_ptr<io::ROS2Gimbal> gimbal_;
  ArmorIgnoreSubscriber armor_ignore_subscriber_;
  BuffRequestSubscriber buff_request_subscriber_;
  std::unique_ptr<io::Camera> auto_aim_camera_;
  auto_aim::Solver solver_;
  omniperception::Decider decider_;
  FrameState frame_;
};

std::optional<RuntimeConfig> prepare_config(int argc, char ** argv, AppMode mode);

}  // namespace ovsentry

#endif  // OVSENTRY__RUNTIME_HPP
