#include <cmath>
#include <iostream>

#include "tasks/auto_aim/planner/planner.hpp"

namespace
{
auto_aim::Target make_target()
{
  auto_aim::Target target(3.0, 0.5, 0.2, 0.1);
  target.name = auto_aim::ArmorName::three;
  target.armor_type = auto_aim::ArmorType::small;
  target.priority = auto_aim::ArmorPriority::fifth;
  target.jumped = false;
  target.last_id = 0;
  return target;
}

bool finite(const auto_aim::Plan & plan)
{
  return std::isfinite(plan.target_yaw) && std::isfinite(plan.target_pitch) &&
         std::isfinite(plan.yaw) && std::isfinite(plan.yaw_vel) &&
         std::isfinite(plan.yaw_acc) && std::isfinite(plan.pitch) &&
         std::isfinite(plan.pitch_vel) && std::isfinite(plan.pitch_acc);
}
}  // namespace

int main(int argc, char * argv[])
{
  if (argc != 2) {
    std::cerr << "usage: planner_sentry_safety_test <sentry.yaml>" << std::endl;
    return 1;
  }

  auto_aim::Planner planner(argv[1]);
  const auto nominal = planner.plan_sentry_world(make_target(), 22.0, 0);
  if (!nominal.world_small_yaw_plan.control || !finite(nominal.world_small_yaw_plan) ||
      !std::isfinite(nominal.big_yaw)) {
    std::cerr << "nominal sentry MPC plan did not converge to a finite command" << std::endl;
    return 1;
  }

  const auto invalid_id = planner.plan_sentry_world(make_target(), 22.0, 99);
  if (invalid_id.world_small_yaw_plan.control) {
    std::cerr << "invalid preferred armor id did not fail closed" << std::endl;
    return 1;
  }

  return 0;
}
