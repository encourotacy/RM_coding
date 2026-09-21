#ifndef OVSENTRY__OMNI_TASK_HPP
#define OVSENTRY__OMNI_TASK_HPP

#include <chrono>
#include <memory>
#include <optional>

#include "io/command.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "runtime.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/omniperception/ovsentry_omni_logic.hpp"
#include "types.hpp"

namespace ovsentry
{

class OmniTask
{
public:
  explicit OmniTask(SentryRuntime & rt);

  void prepare(bool omni_mode);
  void run();
  void reset();

private:
  OmniCandidateFrame read_frame(
    io::USBCamera & camera, cv::Mat & img, std::chrono::steady_clock::time_point & ts,
    const OmniCamConfig & cam_cfg);
  void finalize_frame(OmniCandidateFrame & frame, const OmniCamConfig & cam_cfg, double infer_ms);
  void clear_timeout_session();
  void clear_redirect_state();

  SentryRuntime & rt_;
  std::unique_ptr<auto_aim::YOLO> yolo_left_;
  std::unique_ptr<auto_aim::YOLO> yolo_right_;
  std::unique_ptr<auto_aim::YOLO> yolo_back_;
  std::unique_ptr<io::USBCamera> cam_left_;
  std::unique_ptr<io::USBCamera> cam_right_;
  std::unique_ptr<io::USBCamera> cam_back_;

  std::optional<io::Command> hold_command_;
  std::optional<omniperception::AcceptedOmniTarget> session_accepted_target_;
  std::optional<omniperception::AcceptedOmniTarget> cooldown_anchor_target_;
  std::optional<omniperception::AcceptedOmniTarget> timeout_target_;
  std::chrono::steady_clock::time_point retarget_cooldown_deadline_{};
  std::chrono::steady_clock::time_point timeout_started_at_{};
  bool timeout_running_ = false;
  bool prev_omni_mode_ = false;

  static constexpr bool yolo_debug_ = false;
};

}  // namespace ovsentry

#endif  // OVSENTRY__OMNI_TASK_HPP
