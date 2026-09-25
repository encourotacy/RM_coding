#ifndef OMNIPERCEPTION__DETECTION_HPP
#define OMNIPERCEPTION__DETECTION_HPP

#include <chrono>
#include <list>
#include <string>

#include "tasks/auto_aim/armor.hpp"

namespace omniperception
{
enum class OmniCameraSlot
{
  unknown = 0,
  left,
  right,
  back,
  extra
};

struct CameraSpec
{
  OmniCameraSlot slot = OmniCameraSlot::unknown;
  std::string label;
  std::string dev_path;
  double center_yaw_deg = 0.0;
  double fov_h_deg = 0.0;
  double fov_v_deg = 0.0;
};

//一个识别结果可能包含多个armor,需要排序和过滤。armors, timestamp, delta_yaw, delta_pitch
struct DetectionResult
{
  std::list<auto_aim::Armor> armors;
  std::chrono::steady_clock::time_point timestamp{};
  double delta_yaw = 0.0;    //rad
  double delta_pitch = 0.0;  //rad
  double abs_yaw_rad = 0.0;
  OmniCameraSlot slot = OmniCameraSlot::unknown;
  std::string camera_label;
  double base_yaw_rad = 0.0;
  bool has_base_yaw = false;
  bool has_abs_yaw = false;
  double infer_ms = 0.0;
};
}  // namespace omniperception

#endif
