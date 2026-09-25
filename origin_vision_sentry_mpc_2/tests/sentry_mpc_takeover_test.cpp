#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

#include "tasks/auto_aim/sentry_mpc_safety.hpp"
#include "tasks/auto_aim/sentry_mpc_takeover.hpp"

namespace
{
bool expect(bool condition, const std::string & message)
{
  if (condition) return true;
  std::cerr << message << std::endl;
  return false;
}

bool expect_near(double actual, double expected, double tolerance, const std::string & message)
{
  if (std::abs(actual - expected) <= tolerance) return true;
  std::cerr << message << ", actual=" << actual << ", expected=" << expected << std::endl;
  return false;
}

struct FakeGimbal
{
  int calls = 0;
  bool control = false;
  bool fire = false;
  double big_yaw = 0.0;
  double small_yaw = 0.0;
  double pitch = 0.0;
  double yaw_vel = 0.0;
  double pitch_vel = 0.0;
  double yaw_acc = 0.0;
  double pitch_acc = 0.0;
  uint8_t armor_id = 0;
  double vx = 0.0;
  double vy = 0.0;
  double distance = 0.0;

  void send_mpc(
    bool control_value, bool fire_value, double big_yaw_value, double small_yaw_value,
    double pitch_value, double yaw_vel_value, double pitch_vel_value, double yaw_acc_value,
    double pitch_acc_value, uint8_t armor_id_value, double vx_value, double vy_value,
    double distance_value)
  {
    calls++;
    control = control_value;
    fire = fire_value;
    big_yaw = big_yaw_value;
    small_yaw = small_yaw_value;
    pitch = pitch_value;
    yaw_vel = yaw_vel_value;
    pitch_vel = pitch_vel_value;
    yaw_acc = yaw_acc_value;
    pitch_acc = pitch_acc_value;
    armor_id = armor_id_value;
    vx = vx_value;
    vy = vy_value;
    distance = distance_value;
  }
};
}  // namespace

int main()
{
  constexpr double pi = 3.14159265358979323846;
  constexpr double deg = pi / 180.0;
  const auto t0 = std::chrono::steady_clock::time_point{};

  auto_aim::Plan plan;
  plan.control = true;
  plan.yaw = -179.0 * deg;
  plan.pitch = -10.0 * deg;
  plan.yaw_vel = 1.1;
  plan.pitch_vel = 2.2;
  plan.yaw_acc = 3.3;
  plan.pitch_acc = 4.4;

  auto_aim::SentryMpcTakeover immediate(0.0);
  auto output = immediate.update(
    plan, -178.0 * deg, 179.0 * deg, 0.0, 6.0 * deg, 0.0, 178.0 * deg, t0);
  if (!expect(output.command.control, "valid MPC plan must enable control")) return 1;
  if (!expect(output.fire_ready, "zero-duration takeover must be immediately fire-ready")) return 1;
  if (!expect_near(output.command.small_yaw, 181.0 * deg, 1e-6, "small yaw unwrap failed")) {
    return 1;
  }
  if (!expect_near(output.command.big_yaw, 182.0 * deg, 1e-6, "big yaw unwrap failed")) {
    return 1;
  }
  if (!expect_near(output.command.pitch, plan.pitch, 1e-6, "physical pitch was changed")) {
    return 1;
  }

  output.command.shoot = true;
  output.command.armor_id = 7;
  output.command.vx = 5.5;
  output.command.vy = 6.6;
  output.command.horizon_distance = 7.7;
  FakeGimbal fake;
  auto_aim::dispatch_sentry_mpc(fake, output);
  if (!expect(fake.calls == 1 && fake.control && fake.fire, "dispatch did not preserve flags")) {
    return 1;
  }
  if (!expect_near(fake.yaw_vel, 1.1, 1e-6, "yaw velocity was dropped")) return 1;
  if (!expect_near(fake.pitch_vel, 2.2, 1e-6, "pitch velocity was dropped")) return 1;
  if (!expect_near(fake.yaw_acc, 3.3, 1e-6, "yaw acceleration was dropped")) return 1;
  if (!expect_near(fake.pitch_acc, 4.4, 1e-6, "pitch acceleration was dropped")) return 1;
  if (!expect(
        fake.armor_id == 7 && std::abs(fake.vx - 5.5) < 1e-9 &&
          std::abs(fake.vy - 6.6) < 1e-9 && std::abs(fake.distance - 7.7) < 1e-9,
        "target metadata was dropped")) {
    return 1;
  }

  auto_aim::SentryMpcTakeover smooth(0.2);
  auto smooth_plan = plan;
  smooth_plan.yaw_vel = 0.0;
  smooth_plan.pitch_vel = 0.0;
  smooth_plan.yaw_acc = 0.0;
  smooth_plan.pitch_acc = 0.0;
  constexpr double measured_yaw_vel = 0.4;
  constexpr double measured_pitch_vel = -0.2;
  const auto start = smooth.update(
    smooth_plan, -178.0 * deg, 10.0 * deg, measured_yaw_vel, 6.0 * deg,
    measured_pitch_vel, 20.0 * deg, t0);
  if (!expect_near(
        start.command.small_yaw, 10.0 * deg, 1e-9,
        "takeover must start at measured yaw")) {
    return 1;
  }
  if (!expect_near(start.command.pitch, 6.0 * deg, 1e-9, "takeover must start at measured pitch")) {
    return 1;
  }
  if (!expect_near(
        start.yaw_vel, measured_yaw_vel, 1e-12,
        "takeover must start with measured yaw velocity")) {
    return 1;
  }
  if (!expect_near(
        start.pitch_vel, measured_pitch_vel, 1e-12,
        "takeover must start with measured pitch velocity")) {
    return 1;
  }

  const auto before_middle = smooth.update(
    smooth_plan, -178.0 * deg, 10.0 * deg, measured_yaw_vel, 6.0 * deg,
    measured_pitch_vel, 20.0 * deg, t0 + std::chrono::milliseconds(99));
  const auto middle = smooth.update(
    smooth_plan, -178.0 * deg, 10.0 * deg, measured_yaw_vel, 6.0 * deg,
    measured_pitch_vel, 20.0 * deg, t0 + std::chrono::milliseconds(100));
  const auto after_middle = smooth.update(
    smooth_plan, -178.0 * deg, 10.0 * deg, measured_yaw_vel, 6.0 * deg,
    measured_pitch_vel, 20.0 * deg, t0 + std::chrono::milliseconds(101));
  if (!expect_near(middle.takeover_alpha, 0.5, 1e-12, "takeover timing mismatch")) return 1;
  const double numerical_pitch_vel =
    (after_middle.command.pitch - before_middle.command.pitch) / 0.002;
  const double numerical_pitch_acc =
    (after_middle.pitch_vel - before_middle.pitch_vel) / 0.002;
  if (!expect_near(
        middle.pitch_vel, numerical_pitch_vel, 2e-3,
        "takeover pitch velocity is not the derivative of position")) {
    return 1;
  }
  if (!expect_near(
        middle.pitch_acc, numerical_pitch_acc, 5e-2,
        "takeover pitch acceleration is not the derivative of velocity")) {
    return 1;
  }
  if (!expect(!middle.fire_ready, "takeover must block fire before completion")) return 1;

  const auto finish = smooth.update(
    plan, -178.0 * deg, 10.0 * deg, measured_yaw_vel, 6.0 * deg,
    measured_pitch_vel, 20.0 * deg, t0 + std::chrono::milliseconds(200));
  if (!expect(finish.fire_ready, "completed takeover must allow fire gating")) return 1;
  if (!expect_near(finish.pitch_acc, 4.4, 1e-6, "completed takeover lost dynamics")) return 1;

  auto_aim::SentryMpcTakeover wrap_tracking(0.0);
  plan.yaw = 179.0 * deg;
  auto wrapped = wrap_tracking.update(
    plan, 179.0 * deg, 178.0 * deg, 0.0, 0.0, 0.0, 178.0 * deg, t0);
  if (!expect_near(wrapped.command.small_yaw, 179.0 * deg, 1e-6, "initial yaw unwrap failed")) {
    return 1;
  }
  plan.yaw = -179.0 * deg;
  wrapped = wrap_tracking.update(
    plan, -179.0 * deg, -179.0 * deg, 0.0, 0.0, 0.0, -179.0 * deg,
    t0 + std::chrono::milliseconds(10));
  if (!expect_near(
        wrapped.command.small_yaw, 181.0 * deg, 1e-6,
        "long-running small yaw must remain continuous across pi")) {
    return 1;
  }
  if (!expect_near(
        wrapped.command.big_yaw, 181.0 * deg, 1e-6,
        "wrapped big-yaw feedback must not change the continuous output branch")) {
    return 1;
  }
  plan.yaw = -178.0 * deg;
  wrapped = wrap_tracking.update(
    plan, -178.0 * deg, -178.0 * deg, 0.0, 0.0, 0.0, -178.0 * deg,
    t0 + std::chrono::milliseconds(20));
  if (!expect_near(
        wrapped.command.small_yaw, 182.0 * deg, 1e-6,
        "continuous yaw reference must advance beyond the initial branch")) {
    return 1;
  }
  if (!expect_near(
        wrapped.command.big_yaw, 182.0 * deg, 1e-6,
        "continuous big yaw must advance beyond the initial branch")) {
    return 1;
  }

  plan.yaw = std::numeric_limits<float>::quiet_NaN();
  const auto invalid = smooth.update(
    plan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, t0 + std::chrono::milliseconds(210));
  if (!expect(
        !invalid.command.control && !invalid.fire_ready,
        "non-finite plan must disable control")) {
    return 1;
  }

  auto_aim::SentryMpcSafetyLimits limits;
  limits.min_pitch = -60.0 * deg;
  limits.max_pitch = 30.0 * deg;
  auto_aim::SentryMpcSafetyGate safety(limits);
  auto_aim::Plan safe_plan;
  safe_plan.control = true;
  if (!expect(safety.plan_is_safe(safe_plan, 0.0), "nominal pitch plan was rejected")) {
    return 1;
  }

  // 大仰角对所有目标使用同一物理范围，不再按目标类型或计划-实测角差设门限。
  safe_plan.pitch = -52.0 * deg;
  safe_plan.target_pitch = -52.0 * deg;
  if (!expect(
        safety.plan_is_safe(safe_plan, -12.0 * deg),
        "mechanically valid large-angle plan was rejected")) {
    return 1;
  }

  auto_aim::SentryMpcTakeover large_angle_takeover(0.2, 50.0, 100.0);
  auto_aim::Plan large_angle_plan = safe_plan;
  large_angle_plan.yaw = 0.0;
  large_angle_plan.yaw_vel = 0.0;
  large_angle_plan.yaw_acc = 0.0;
  large_angle_plan.pitch_vel = 0.0;
  large_angle_plan.pitch_acc = 0.0;
  const auto large_start = large_angle_takeover.update(
    large_angle_plan, 0.0, 0.0, 0.0, -12.0 * deg, 0.0, 0.0, t0);
  const auto large_middle = large_angle_takeover.update(
    large_angle_plan, 0.0, 0.0, 0.0, -12.0 * deg, 0.0, 0.0,
    t0 + std::chrono::milliseconds(100));
  const auto large_finish = large_angle_takeover.update(
    large_angle_plan, 0.0, 0.0, 0.0, -12.0 * deg, 0.0, 0.0,
    t0 + std::chrono::milliseconds(210));
  if (!expect_near(
        large_start.command.pitch, -12.0 * deg, 1e-9,
        "large-angle takeover did not start at measured pitch")) {
    return 1;
  }
  if (!expect(
        large_middle.command.pitch < -12.0 * deg &&
          large_middle.command.pitch > -52.0 * deg,
        "large-angle takeover midpoint is not between measured and planned pitch")) {
    return 1;
  }
  if (!expect_near(
        large_finish.command.pitch, -52.0 * deg, 1e-6,
        "large-angle takeover did not finish at the planned pitch")) {
    return 1;
  }
  if (!expect(
        !large_start.fire_ready && !large_middle.fire_ready && large_finish.fire_ready,
        "large-angle takeover fire gate timing is incorrect")) {
    return 1;
  }
  if (!expect(
        safety.setpoint_is_safe(
          large_start.command, large_start.yaw_vel, large_start.pitch_vel,
          large_start.yaw_acc, large_start.pitch_acc) &&
          safety.setpoint_is_safe(
            large_middle.command, large_middle.yaw_vel, large_middle.pitch_vel,
            large_middle.yaw_acc, large_middle.pitch_acc) &&
          safety.setpoint_is_safe(
            large_finish.command, large_finish.yaw_vel, large_finish.pitch_vel,
            large_finish.yaw_acc, large_finish.pitch_acc),
        "large-angle takeover left the unified mechanical range")) {
    return 1;
  }

  // 固定0.2s只作为最短接管时间；极大角差按MPC已有加速度量级自动延长。
  auto_aim::SentryMpcTakeover adaptive_pitch_takeover(0.2, 50.0, 100.0);
  auto_aim::Plan extreme_pitch_plan = large_angle_plan;
  extreme_pitch_plan.pitch = -60.0 * deg;
  extreme_pitch_plan.target_pitch = -60.0 * deg;
  (void)adaptive_pitch_takeover.update(
    extreme_pitch_plan, 0.0, 0.0, 0.0, 30.0 * deg, 0.0, 0.0, t0);
  const auto adaptive_pitch_peak = adaptive_pitch_takeover.update(
    extreme_pitch_plan, 0.0, 0.0, 0.0, 30.0 * deg, 0.0, 0.0,
    t0 + std::chrono::milliseconds(64));
  const auto adaptive_pitch_at_base_time = adaptive_pitch_takeover.update(
    extreme_pitch_plan, 0.0, 0.0, 0.0, 30.0 * deg, 0.0, 0.0,
    t0 + std::chrono::milliseconds(200));
  const auto adaptive_pitch_finish = adaptive_pitch_takeover.update(
    extreme_pitch_plan, 0.0, 0.0, 0.0, 30.0 * deg, 0.0, 0.0,
    t0 + std::chrono::milliseconds(310));
  if (!expect(
        std::abs(adaptive_pitch_peak.pitch_acc) <= 100.0 + 1e-3,
        "zero-dynamics adaptive pitch example exceeded its acceleration scale")) {
    return 1;
  }
  if (!expect(
        !adaptive_pitch_at_base_time.fire_ready && adaptive_pitch_finish.fire_ready,
        "extreme pitch takeover was not extended beyond the base duration")) {
    return 1;
  }
  if (!expect_near(
        adaptive_pitch_finish.command.pitch, -60.0 * deg, 1e-6,
        "adaptive pitch takeover did not reach the target")) {
    return 1;
  }

  auto_aim::SentryMpcTakeover adaptive_yaw_takeover(0.2, 50.0, 100.0);
  auto_aim::Plan extreme_yaw_plan = large_angle_plan;
  extreme_yaw_plan.pitch = 0.0;
  extreme_yaw_plan.target_pitch = 0.0;
  extreme_yaw_plan.yaw = pi;
  (void)adaptive_yaw_takeover.update(
    extreme_yaw_plan, pi, 0.0, 0.0, 0.0, 0.0, 0.0, t0);
  const auto adaptive_yaw_at_base_time = adaptive_yaw_takeover.update(
    extreme_yaw_plan, pi, 0.0, 0.0, 0.0, 0.0, 0.0,
    t0 + std::chrono::milliseconds(200));
  const auto adaptive_yaw_finish = adaptive_yaw_takeover.update(
    extreme_yaw_plan, pi, 0.0, 0.0, 0.0, 0.0, 0.0,
    t0 + std::chrono::milliseconds(610));
  if (!expect(
        !adaptive_yaw_at_base_time.fire_ready && adaptive_yaw_finish.fire_ready,
        "extreme yaw takeover was not extended beyond the base duration")) {
    return 1;
  }

  safe_plan.pitch = -60.0 * deg;
  safe_plan.target_pitch = -60.0 * deg;
  if (!expect(
        safety.plan_is_safe(safe_plan, -60.0 * deg),
        "negative mechanical pitch boundary was rejected")) {
    return 1;
  }

  safe_plan.pitch = 30.0 * deg;
  safe_plan.target_pitch = 30.0 * deg;
  if (!expect(
        safety.plan_is_safe(safe_plan, 30.0 * deg),
        "positive mechanical pitch boundary was rejected")) {
    return 1;
  }

  safe_plan.pitch = -52.0 * deg;
  safe_plan.target_pitch = -61.0 * deg;
  if (!expect(
        !safety.plan_is_safe(safe_plan, -12.0 * deg),
        "target pitch escaped the negative mechanical limit")) {
    return 1;
  }

  safe_plan.target_pitch = 0.0;
  safe_plan.pitch = 31.0 * deg;
  if (!expect(
        !safety.plan_is_safe(safe_plan, 0.0),
        "planned pitch escaped the positive mechanical limit")) {
    return 1;
  }

  safe_plan.pitch = 0.0;
  if (!expect(
        !safety.plan_is_safe(safe_plan, -61.0 * deg),
        "measured pitch outside the mechanical range was accepted")) {
    return 1;
  }

  safe_plan.target_pitch = 0.0;
  safe_plan.pitch_acc = std::numeric_limits<float>::quiet_NaN();
  if (!expect(
        !safety.plan_is_safe(safe_plan, 0.0),
        "non-finite plan was accepted")) {
    return 1;
  }

  io::Command unsafe_setpoint{true, false, 0.0, -61.0 * deg};
  if (!expect(
        !safety.setpoint_is_safe(unsafe_setpoint, 0.0, 0.0, 0.0, 0.0),
        "takeover output escaped the mechanical pitch limit")) {
    return 1;
  }
  io::Command nominal_setpoint{true, false, 0.0, 0.0};
  if (!expect(
        safety.setpoint_is_safe(nominal_setpoint, 1.0, -1.0, 1.0, -1.0),
        "finite nominal setpoint was rejected")) {
    return 1;
  }
  if (!expect(
        !safety.setpoint_is_safe(
          nominal_setpoint, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0),
        "non-finite setpoint was accepted")) {
    return 1;
  }

  return 0;
}
