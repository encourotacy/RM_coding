#include "overlay.hpp"

#include <fmt/core.h>

#include "auto_aim_helpers.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tools/img_tools.hpp"

namespace ovsentry
{

void draw_omni_overlay(cv::Mat & img, const OmniInferenceResult & result)
{
  tools::draw_text(
    img,
    fmt::format(
      "{} ({}) {:.1f}ms", slot_name(result.cam.spec.slot), result.cam.dev_name, result.infer_ms),
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
  const auto aim_image_points = solver.reproject_armor(
    aim_point.xyza.head(3), aim_point.xyza[3], target.armor_type, target.name);
  tools::draw_points(
    img, aim_image_points, aim_point.valid ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));
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
      "buff yaw={:.2f} pitch={:.2f} fire={}", plan.yaw * 57.3, plan.pitch * 57.3, plan.fire ? 1 : 0),
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
  const auto image_points = solver.reproject_buff(
    target.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.0)), target.ekf_x()[4], target.ekf_x()[5]);
  tools::draw_points(img, image_points, {255, 0, 0}, 2);
}

cv::Mat resize_for_view(const cv::Mat & img)
{
  cv::Mat resized;
  cv::resize(img, resized, {640, 360});
  return resized;
}

}  // namespace ovsentry
