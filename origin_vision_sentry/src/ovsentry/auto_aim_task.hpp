#ifndef OVSENTRY__AUTO_AIM_TASK_HPP
#define OVSENTRY__AUTO_AIM_TASK_HPP

#include <memory>

#include "runtime.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace ovsentry
{

class AutoAimTask
{
public:
  explicit AutoAimTask(SentryRuntime & rt);

  void detect_and_track();
  void aim_and_send();
  const auto_aim::Aimer & aimer() const { return aimer_; }

private:
  SentryRuntime & rt_;
  std::unique_ptr<auto_aim::YOLO> yolo_;
  auto_aim::Tracker tracker_;
  auto_aim::Aimer aimer_;
  auto_aim::Shooter shooter_;
  auto_aim::Planner planner_;

  static constexpr bool yolo_debug_ = false;
  static constexpr bool aimer_to_now_ = true;
};

}  // namespace ovsentry

#endif  // OVSENTRY__AUTO_AIM_TASK_HPP
