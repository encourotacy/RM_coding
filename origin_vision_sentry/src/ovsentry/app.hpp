#ifndef OVSENTRY__APP_HPP
#define OVSENTRY__APP_HPP

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
#include "io/usbcamera/usbcamera.hpp"
#include "subscribers.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/target.hpp"
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
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "types.hpp"

namespace io
{
class ROS2Gimbal;
}

namespace ovsentry
{

int run_ovsentry_app(int argc, char ** argv, AppMode mode);

class OVSentryOmniMpc
{
public:
  explicit OVSentryOmniMpc(RuntimeConfig cfg);
  ~OVSentryOmniMpc();
  int run();

private:
  void init_mode_modules();
  void update_mode_flags();
  bool read_main_frame();
  void detect_and_track();
  void reset_frame_outputs();
  void update_omni_session_on_mode_change();
  void run_small_buff();
  void run_omni();
  void run_auto_aim_mpc();
  void publish_telemetry();
  bool render_display();
  void shutdown();
  const char * window_title() const;

  OmniCandidateFrame read_omni_frame(
    io::USBCamera & camera, cv::Mat & img, std::chrono::steady_clock::time_point & ts,
    const OmniCamConfig & cam_cfg);
  void finalize_omni_frame(
    OmniCandidateFrame & frame, const OmniCamConfig & cam_cfg, double infer_ms);
  void clear_omni_timeout_session();
  void clear_omni_redirect_state();

  RuntimeConfig cfg_;
  tools::Exiter exiter_;
  tools::Plotter plotter_;
  tools::Recorder recorder_;
  std::unique_ptr<io::ROS2Gimbal> gimbal_;
  ArmorIgnoreSubscriber armor_ignore_subscriber_;
  BuffRequestSubscriber buff_request_subscriber_;
  std::unique_ptr<io::Camera> auto_aim_camera_;
  std::unique_ptr<auto_aim::YOLO> yolo_auto_;
  auto_aim::Solver solver_;
  auto_aim::Tracker tracker_;
  auto_aim::Aimer aimer_;
  auto_aim::Shooter shooter_;
  auto_aim::Planner planner_;
  omniperception::Decider decider_;
  std::unique_ptr<auto_buff::Buff_Detector> buff_detector_;
  std::unique_ptr<auto_buff::Solver> buff_solver_;
  auto_buff::SmallTarget buff_small_target_;
  std::unique_ptr<auto_buff::Aimer> buff_aimer_;
  std::unique_ptr<auto_aim::YOLO> yolo_omni_left_;
  std::unique_ptr<auto_aim::YOLO> yolo_omni_right_;
  std::unique_ptr<auto_aim::YOLO> yolo_omni_back_;
  std::unique_ptr<io::USBCamera> cam_left_;
  std::unique_ptr<io::USBCamera> cam_right_;
  std::unique_ptr<io::USBCamera> cam_back_;

  cv::Mat main_img_, left_img_, right_img_, back_img_;
  std::chrono::steady_clock::time_point main_timestamp_, ts_left_, ts_right_, ts_back_;
  std::optional<io::Command> omni_hold_command_;
  std::optional<omniperception::AcceptedOmniTarget> session_accepted_omni_target_;
  std::optional<omniperception::AcceptedOmniTarget> cooldown_anchor_omni_target_;
  std::optional<omniperception::AcceptedOmniTarget> active_omni_timeout_target_;
  std::chrono::steady_clock::time_point omni_retarget_cooldown_deadline_{};
  std::chrono::steady_clock::time_point active_omni_timeout_started_at_{};
  std::optional<io::Command> buff_hold_command_;
  std::chrono::steady_clock::time_point buff_last_control_at_{};
  bool active_omni_timeout_running_ = false;
  bool prev_omni_mode_ = false;
  int frame_count_ = 0;

  Eigen::Quaterniond q_{Eigen::Quaterniond::Identity()};
  io::ROS2GimbalState gimbal_state_{};
  bool small_buff_mode_ = false;
  bool omni_mode_ = false;
  Eigen::Vector3d ypr_{Eigen::Vector3d::Zero()};
  ArmorTargetMask armor_target_mask_;
  std::list<auto_aim::Armor> armors_;
  std::list<auto_aim::Target> targets_;
  std::string tracker_state_ = "idle";
  std::chrono::steady_clock::time_point t0_{};
  std::chrono::steady_clock::time_point t1_{};
  std::chrono::steady_clock::time_point now_{};

  std::optional<OmniInferenceResult> best_omni_result_;
  std::optional<double> omni_target_abs_yaw_deg_;
  std::optional<double> omni_candidate_abs_yaw_deg_;
  std::optional<double> omni_candidate_base_big_yaw_deg_;
  std::optional<double> omni_candidate_age_ms_;
  std::optional<double> omni_candidate_delta_deg_;
  std::optional<double> omni_target_error_deg_;
  std::optional<double> omni_selected_confidence_;
  std::optional<int> omni_selected_priority_;
  std::optional<std::string> omni_selected_slot_;
  bool omni_hold_applied_ = false;
  bool omni_retarget_blocked_ = false;
  bool omni_retarget_cd_active_ = false;
  bool omni_same_target_continuation_ = false;
  bool omni_target_reached_ = false;
  bool omni_cmd_timeout_active_ = false;
  bool omni_cmd_timed_out_ = false;
  std::string omni_block_reason_ = "none";
  double omni_cmd_elapsed_ms_ = 0.0;
  double omni_retarget_remaining_ms_ = 0.0;
  io::Command command_{false, false, 0.0, 0.0};
  std::optional<auto_buff::PowerRune> buff_power_rune_;
  auto_aim::Plan buff_plan_{false, false, 0, 0, 0, 0, 0, 0, 0, 0};
  double buff_detect_ms_ = 0.0;
  bool buff_target_ready_ = false;
  bool buff_command_held_ = false;

  static constexpr bool yolo_debug_ = false;
  static constexpr bool aimer_to_now_ = true;
};

}  // namespace ovsentry

#endif  // OVSENTRY__APP_HPP
