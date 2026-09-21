#ifndef OVSENTRY__OVERLAY_HPP
#define OVSENTRY__OVERLAY_HPP

#include <list>
#include <optional>

#include <opencv2/opencv.hpp>

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "types.hpp"

namespace ovsentry
{

void draw_omni_overlay(cv::Mat & img, const OmniInferenceResult & result);
void draw_auto_aim_overlay(
  cv::Mat & img, const std::list<auto_aim::Target> & targets, const auto_aim::Aimer & aimer,
  const auto_aim::Solver & solver);
void draw_small_buff_overlay(
  cv::Mat & img, std::optional<auto_buff::PowerRune> & power_rune,
  const auto_buff::SmallTarget & target, const auto_buff::Solver & solver,
  const auto_aim::Plan & plan);
cv::Mat resize_for_view(const cv::Mat & img);

}  // namespace ovsentry

#endif  // OVSENTRY__OVERLAY_HPP
