#include "decider.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace omniperception
{
namespace
{
using PriorityMap = std::unordered_map<auto_aim::ArmorName, auto_aim::ArmorPriority>;

double angular_distance_deg(double lhs_rad, double rhs_rad)
{
  return std::abs(tools::limit_rad(lhs_rad - rhs_rad)) * 57.3;
}

const PriorityMap & priority_map_for_mode(int mode)
{
  static const PriorityMap mode1 = {
    {auto_aim::ArmorName::one, auto_aim::ArmorPriority::first},
    {auto_aim::ArmorName::two, auto_aim::ArmorPriority::forth},
    {auto_aim::ArmorName::three, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::four, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::five, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::sentry, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::outpost, auto_aim::ArmorPriority::fifth},
    {auto_aim::ArmorName::base, auto_aim::ArmorPriority::fifth},
    {auto_aim::ArmorName::not_armor, auto_aim::ArmorPriority::fifth}};
  static const PriorityMap mode2 = {
    {auto_aim::ArmorName::two, auto_aim::ArmorPriority::first},
    {auto_aim::ArmorName::one, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::three, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::four, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::five, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::sentry, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::outpost, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::base, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::not_armor, auto_aim::ArmorPriority::third}};
  return mode == MODE_ONE ? mode1 : mode2;
}

bool armor_filter_with_config(
  std::list<auto_aim::Armor> & armors, auto_aim::Color enemy_color,
  const std::vector<auto_aim::ArmorName> & invincible_armor)
{
  if (armors.empty()) return true;
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color; });
  armors.remove_if([&](const auto_aim::Armor & a) { return a.name == auto_aim::ArmorName::five; });
  //armors.remove_if([&](const auto_aim::Armor & a) { return a.name == auto_aim::ArmorName::outpost; });
  armors.remove_if([&](const auto_aim::Armor & a) {
    return std::find(invincible_armor.begin(), invincible_armor.end(), a.name) != invincible_armor.end();
  });
  return armors.empty();
}

void set_priority_with_mode(std::list<auto_aim::Armor> & armors, int mode)
{
  if (armors.empty()) return;
  const PriorityMap & priority_map = priority_map_for_mode(mode);
  for (auto & armor : armors) {
    armor.priority = priority_map.at(armor.name);
  }
}

void prepare_detection_queue(
  std::vector<DetectionResult> & detection_queue, auto_aim::Color enemy_color, int mode,
  const std::vector<auto_aim::ArmorName> & invincible_armor)
{
  if (detection_queue.empty()) return;

  for (auto & dr : detection_queue) {
    armor_filter_with_config(dr.armors, enemy_color, invincible_armor);
    set_priority_with_mode(dr.armors, mode);
    dr.armors.sort(
      [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });
  }

  detection_queue.erase(
    std::remove_if(
      detection_queue.begin(), detection_queue.end(),
      [](const DetectionResult & dr) { return dr.armors.empty(); }),
    detection_queue.end());
}

double reference_yaw_delta_deg(const DetectionResult & dr, const SelectionContext & context)
{
  if (!context.has_reference_abs_yaw || !dr.has_abs_yaw) return 1e9;
  return angular_distance_deg(dr.abs_yaw_rad, context.reference_abs_yaw_rad);
}

double current_base_yaw_delta_deg(const DetectionResult & dr, const SelectionContext & context)
{
  if (!context.has_current_abs_yaw || !dr.has_base_yaw) return 0.0;
  return angular_distance_deg(dr.base_yaw_rad, context.current_abs_yaw_rad);
}
}  // namespace

Decider::Decider(const std::string & config_path) : detector_(config_path), count_(0)
{
  auto yaml = YAML::LoadFile(config_path);
  img_width_ = yaml["image_width"].as<double>();
  img_height_ = yaml["image_height"].as<double>();
  fov_h_ = yaml["fov_h"].as<double>();
  fov_v_ = yaml["fov_v"].as<double>();
  new_fov_h_ = yaml["new_fov_h"].as<double>();
  new_fov_v_ = yaml["new_fov_v"].as<double>();
  enemy_color_ =
    (yaml["enemy_color"].as<std::string>() == "red") ? auto_aim::Color::red : auto_aim::Color::blue;
  mode_ = yaml["mode"].as<double>();
}

io::Command Decider::decide(
  auto_aim::YOLO & yolo, const Eigen::Vector3d & gimbal_pos, io::USBCamera & usbcam1,
  io::USBCamera & usbcam2, io::Camera & back_camera)
{
  Eigen::Vector2d delta_angle;
  io::USBCamera * cams[] = {&usbcam1, &usbcam2};

  cv::Mat usb_img;
  std::chrono::steady_clock::time_point timestamp;
  if (count_ < 0 || count_ > 2) {
    throw std::runtime_error("count_ out of valid range [0,2]");
  }
  if (count_ == 2) {
    back_camera.read(usb_img, timestamp);
  } else {
    cams[count_]->read(usb_img, timestamp);
  }
  auto armors = yolo.detect(usb_img);
  auto empty = armor_filter(armors);

  if (!empty) {
    if (count_ == 2) {
      delta_angle = this->delta_angle(armors, "back");
    } else {
      delta_angle = this->delta_angle(armors, cams[count_]->device_name);
    }

    tools::logger()->debug(
      "[{} camera] delta yaw:{:.2f},target pitch:{:.2f},armor number:{},armor name:{}",
      (count_ == 2 ? "back" : cams[count_]->device_name), delta_angle[0], delta_angle[1],
      armors.size(), auto_aim::ARMOR_NAMES[armors.front().name]);

    count_ = (count_ + 1) % 3;

    return io::Command{
      true, false, tools::limit_rad(gimbal_pos[0] + delta_angle[0] / 57.3),
      tools::limit_rad(delta_angle[1] / 57.3)};
  }

  count_ = (count_ + 1) % 3;
  // 如果没有找到目标，返回默认命令
  return io::Command{false, false, 0, 0};
}

io::Command Decider::decide(
  auto_aim::YOLO & yolo, const Eigen::Vector3d & gimbal_pos, io::Camera & back_cammera)
{
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  back_cammera.read(img, timestamp);
  auto armors = yolo.detect(img);
  auto empty = armor_filter(armors);

  if (!empty) {
    auto delta_angle = this->delta_angle(armors, "back");
    tools::logger()->debug(
      "[back camera] delta yaw:{:.2f},target pitch:{:.2f},armor number:{},armor name:{}",
      delta_angle[0], delta_angle[1], armors.size(), auto_aim::ARMOR_NAMES[armors.front().name]);

    return io::Command{
      true, false, tools::limit_rad(gimbal_pos[0] + delta_angle[0] / 57.3),
      tools::limit_rad(delta_angle[1] / 57.3)};
  }

  return io::Command{false, false, 0, 0};
}

io::Command Decider::decide(const std::vector<DetectionResult> & detection_queue)
{
  if (detection_queue.empty()) {
    return io::Command{false, false, 0, 0};
  }

  DetectionResult dr = detection_queue.front();
  if (dr.armors.empty()) return io::Command{false, false, 0, 0};
  tools::logger()->info(
    "omniperceptron find {},delta yaw is {:.4f}", auto_aim::ARMOR_NAMES[dr.armors.front().name],
    dr.delta_yaw * 57.3);

  return io::Command{true, false, dr.delta_yaw, dr.delta_pitch};
};

Eigen::Vector2d Decider::delta_angle(
  const std::list<auto_aim::Armor> & armors, const std::string & camera)
{
  if (camera == "left") {
    return delta_angle(
      armors, CameraSpec{OmniCameraSlot::left, "left", "", 62.0, new_fov_h_, new_fov_v_});
  }

  if (camera == "right") {
    return delta_angle(
      armors, CameraSpec{OmniCameraSlot::right, "right", "", -62.0, new_fov_h_, new_fov_v_});
  }

  return delta_angle(armors, CameraSpec{OmniCameraSlot::back, camera, "", 170.0, 54.2, 44.5});
}

Eigen::Vector2d Decider::delta_angle(
  const std::list<auto_aim::Armor> & armors, const CameraSpec & camera_spec)
{
  Eigen::Vector2d delta_angle;
  delta_angle[0] =
    camera_spec.center_yaw_deg + (0.5 - armors.front().center_norm.x) * camera_spec.fov_h_deg;
  delta_angle[1] = armors.front().center_norm.y * camera_spec.fov_v_deg - camera_spec.fov_v_deg / 2;
  return delta_angle;
}

bool Decider::armor_filter(std::list<auto_aim::Armor> & armors)
{
  return armor_filter_with_config(armors, enemy_color_, invincible_armor_);
}

void Decider::set_priority(std::list<auto_aim::Armor> & armors)
{
  set_priority_with_mode(armors, mode_);
}

void Decider::sort(std::vector<DetectionResult> & detection_queue)
{
  prepare_detection_queue(detection_queue, enemy_color_, mode_, invincible_armor_);
  if (detection_queue.empty()) return;

  // 根据优先级对 DetectionResult 进行排序
  std::sort(
    detection_queue.begin(), detection_queue.end(),
    [](const DetectionResult & a, const DetectionResult & b) {
      return a.armors.front().priority < b.armors.front().priority;
    });
}

void sort_for_ovsentry_omni(
  std::vector<DetectionResult> & detection_queue, const SelectionContext & context,
  auto_aim::Color enemy_color, int mode,
  const std::vector<auto_aim::ArmorName> & invincible_armor)
{
  prepare_detection_queue(detection_queue, enemy_color, mode, invincible_armor);
  if (detection_queue.empty()) return;

  detection_queue.erase(
    std::remove_if(
      detection_queue.begin(), detection_queue.end(),
      [&](DetectionResult & dr) {
        if (dr.has_base_yaw && !dr.has_abs_yaw) {
          dr.abs_yaw_rad = dr.base_yaw_rad + dr.delta_yaw;
          dr.has_abs_yaw = true;
        }

        if (context.stale_ms > 0.0) {
          const double age_ms = tools::delta_time(context.now_timestamp, dr.timestamp) * 1e3;
          if (age_ms > context.stale_ms) return true;
        }

        if (context.max_base_yaw_delta_deg > 0.0 && dr.has_base_yaw && context.has_current_abs_yaw) {
          const double base_yaw_delta_deg = current_base_yaw_delta_deg(dr, context);
          if (base_yaw_delta_deg > context.max_base_yaw_delta_deg) return true;
        }

        return false;
      }),
    detection_queue.end());
  if (detection_queue.empty()) return;

  std::sort(
    detection_queue.begin(), detection_queue.end(),
    [&](const DetectionResult & a, const DetectionResult & b) {
      const auto & armor_a = a.armors.front();
      const auto & armor_b = b.armors.front();
      if (armor_a.priority != armor_b.priority) return armor_a.priority < armor_b.priority;

      const double angle_delta_a = reference_yaw_delta_deg(a, context);
      const double angle_delta_b = reference_yaw_delta_deg(b, context);
      if (std::abs(angle_delta_a - angle_delta_b) > 1e-6) return angle_delta_a < angle_delta_b;

      if (context.preferred_slot != OmniCameraSlot::unknown && a.slot != b.slot) {
        const bool prefer_a = a.slot == context.preferred_slot;
        const bool prefer_b = b.slot == context.preferred_slot;
        if (prefer_a != prefer_b) return prefer_a;
      }

      if (a.timestamp != b.timestamp) return a.timestamp > b.timestamp;
      if (std::abs(a.infer_ms - b.infer_ms) > 1e-6) return a.infer_ms < b.infer_ms;
      if (a.slot != b.slot) return static_cast<int>(a.slot) < static_cast<int>(b.slot);
      return a.camera_label < b.camera_label;
    });
}

void Decider::sort_for_ovsentry_omni(
  std::vector<DetectionResult> & detection_queue, const SelectionContext & context)
{
  ::omniperception::sort_for_ovsentry_omni(detection_queue, context, enemy_color_, mode_, invincible_armor_);
}

Eigen::Vector4d Decider::get_target_info(
  const std::list<auto_aim::Armor> & armors, const std::list<auto_aim::Target> & targets)
{
  if (armors.empty() || targets.empty()) return Eigen::Vector4d::Zero();

  auto target = targets.front();

  for (const auto & armor : armors) {
    if (armor.name == target.name) {
      return Eigen::Vector4d{
        armor.xyz_in_gimbal[0], armor.xyz_in_gimbal[1], 1,
        static_cast<double>(armor.name) + 1};  //避免歧义+1(详见通信协议)
    }
  }

  return Eigen::Vector4d::Zero();
}

void Decider::get_invincible_armor(const std::vector<int8_t> & invincible_enemy_ids)
{
  invincible_armor_.clear();

  if (invincible_enemy_ids.empty()) return;

  for (const auto & id : invincible_enemy_ids) {
    tools::logger()->info("invincible armor id: {}", id);
    invincible_armor_.push_back(auto_aim::ArmorName(id - 1));
  }
}

void Decider::get_auto_aim_target(
  std::list<auto_aim::Armor> & armors, const std::vector<int8_t> & auto_aim_target)
{
  if (auto_aim_target.empty()) return;

  std::vector<auto_aim::ArmorName> auto_aim_targets;

  for (const auto & target : auto_aim_target) {
    if (target <= 0 || static_cast<size_t>(target) > auto_aim::ARMOR_NAMES.size()) {
      tools::logger()->warn("Received invalid auto_aim target value: {}", int(target));
      continue;
    }
    auto_aim_targets.push_back(static_cast<auto_aim::ArmorName>(target - 1));
    tools::logger()->info("nav send auto_aim target is {}", auto_aim::ARMOR_NAMES[target - 1]);
  }

  if (auto_aim_targets.empty()) return;

  armors.remove_if([&](const auto_aim::Armor & a) {
    return std::find(auto_aim_targets.begin(), auto_aim_targets.end(), a.name) ==
           auto_aim_targets.end();
  });
}

}  // namespace omniperception
