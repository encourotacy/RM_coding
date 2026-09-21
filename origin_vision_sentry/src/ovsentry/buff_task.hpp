#ifndef OVSENTRY__BUFF_TASK_HPP
#define OVSENTRY__BUFF_TASK_HPP

#include <chrono>
#include <memory>
#include <optional>

#include "io/command.hpp"
#include "runtime.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"

namespace ovsentry
{

class BuffTask
{
public:
  explicit BuffTask(SentryRuntime & rt);

  void run();
  void reset_hold();
  const auto_buff::SmallTarget & target() const { return target_; }
  const auto_buff::Solver * solver() const { return solver_.get(); }

private:
  SentryRuntime & rt_;
  std::unique_ptr<auto_buff::Buff_Detector> detector_;
  std::unique_ptr<auto_buff::Solver> solver_;
  auto_buff::SmallTarget target_;
  std::unique_ptr<auto_buff::Aimer> aimer_;
  std::optional<io::Command> hold_command_;
  std::chrono::steady_clock::time_point last_control_at_{};
};

}  // namespace ovsentry

#endif  // OVSENTRY__BUFF_TASK_HPP
