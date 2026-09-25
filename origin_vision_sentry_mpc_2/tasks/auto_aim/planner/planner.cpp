#include "planner.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"
#include "tasks/auto_aim/sentry_mpc_transform.hpp"

using namespace std::chrono_literals;

namespace auto_aim
{
Planner::Planner(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  gimbal_axis_order_ = tools::GimbalAxisOrder::yaw_pitch;
  if (yaml["gimbal_axis_order"]) {
    gimbal_axis_order_ = tools::parse_gimbal_axis_order(yaml["gimbal_axis_order"].as<std::string>());
  }
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");
  outpost_prediction_offset_s_ = yaml["outpost_prediction_offset_s"].as<double>(0.0);
  if (!std::isfinite(outpost_prediction_offset_s_)) outpost_prediction_offset_s_ = 0.0;

  setup_yaw_solver(config_path);
  setup_pitch_solver(config_path);
}

Plan Planner::plan(Target target, double bullet_speed)
{
  return plan_impl(std::move(target), bullet_speed, false, nullptr, std::nullopt, false);
}

SentryPlan Planner::plan_sentry_world(
  Target target, double bullet_speed, std::optional<int> preferred_armor_id, bool aim_center)
{
  SentryPlan sentry_plan;
  sentry_plan.world_small_yaw_plan =
    plan_impl(
      std::move(target), bullet_speed, true, &sentry_plan.big_yaw, preferred_armor_id,
      aim_center);
  return sentry_plan;
}

SentryPlan Planner::plan_sentry_world(
  std::optional<Target> target, double bullet_speed, std::optional<int> preferred_armor_id,
  bool aim_center)
{
  if (!target.has_value()) return {};

  try {
    const auto x = target->ekf_x();
    if (x.size() < 8 || !x.allFinite()) throw std::runtime_error("Invalid target state");
    double delay_time =
      std::abs(x[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;
    if (target->name == ArmorName::outpost) {
      delay_time = std::max(0.0, delay_time + outpost_prediction_offset_s_);
    }
    if (!std::isfinite(delay_time)) throw std::runtime_error("Invalid processing delay");
    const auto future =
      std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6));
    target->predict(future);
    return plan_sentry_world(
      std::move(target.value()), bullet_speed, preferred_armor_id, aim_center);
  } catch (const std::exception & e) {
    tools::logger()->warn("[Planner] sentry prediction rejected: {}", e.what());
    return {};
  }
}

Plan Planner::plan_impl(
  Target target, double bullet_speed, bool sentry_world, double * sentry_big_yaw,
  std::optional<int> preferred_armor_id, bool aim_center)
{
  // 0. Check bullet speed
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  // 1. Predict fly_time and get trajectory. Invalid/empty armor candidates must never escape to
  // an entry point that could keep sending the previous command.
  double yaw0;
  Trajectory traj;
  try {
    const auto selected_aim_point = select_aim_point(target, preferred_armor_id, aim_center);
    const Eigen::Vector3d xyz = selected_aim_point.head<3>();
    const double min_dist = xyz.head<2>().norm();
    auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
    if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");
    target.predict(bullet_traj.fly_time);

    if (sentry_world && sentry_big_yaw != nullptr) {
      const auto & x = target.ekf_x();
      *sentry_big_yaw = std::atan2(x[2], x[0]);
    }

    // 2. Get trajectory
    yaw0 = aim(target, bullet_speed, sentry_world, preferred_armor_id, aim_center)(0);
    traj =
      get_trajectory(target, yaw0, bullet_speed, sentry_world, preferred_armor_id, aim_center);
    if (!std::isfinite(yaw0) || !traj.allFinite()) {
      throw std::runtime_error("Non-finite target trajectory");
    }
  } catch (const std::exception & e) {
    tools::logger()->warn("[Planner] target rejected at {:.2f}m/s: {}", bullet_speed, e.what());
    return {false};
  }

  // 3. Solve yaw
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  if (tiny_set_x0(yaw_solver_, x0) != 0) return {false};

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  const int yaw_solve_status = tiny_solve(yaw_solver_);

  // 4. Solve pitch
  x0 << traj(2, 0), traj(3, 0);
  if (tiny_set_x0(pitch_solver_, x0) != 0) return {false};

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  const int pitch_solve_status = tiny_solve(pitch_solver_);
  const bool solvers_converged = yaw_solve_status == 0 && pitch_solve_status == 0;
  if (!solvers_converged) {
    tools::logger()->debug(
      "[Planner] MPC reached iteration limit (yaw={}, pitch={}); fire disabled",
      yaw_solve_status, pitch_solve_status);
  }

  if (
    !yaw_solver_->work->x.allFinite() || !yaw_solver_->work->u.allFinite() ||
    !pitch_solver_->work->x.allFinite() || !pitch_solver_->work->u.allFinite()) {
    tools::logger()->warn("[Planner] MPC produced non-finite output");
    return {false};
  }

  Plan plan;

  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.target_pitch = traj(2, HALF_HORIZON);

  plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);

  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

  if (
    !std::isfinite(plan.target_yaw) || !std::isfinite(plan.target_pitch) ||
    !std::isfinite(plan.yaw) || !std::isfinite(plan.yaw_vel) ||
    !std::isfinite(plan.yaw_acc) || !std::isfinite(plan.pitch) ||
    !std::isfinite(plan.pitch_vel) || !std::isfinite(plan.pitch_acc)) {
    tools::logger()->warn("[Planner] MPC setpoint contains non-finite values");
    return {false};
  }

  plan.control = true;

  auto shoot_offset_ = 2;
  plan.fire =
    solvers_converged &&
    std::hypot(
      traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
      traj(2, HALF_HORIZON + shoot_offset_) -
        pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;
  return plan;
}

Plan Planner::plan(std::optional<Target> target, double bullet_speed)
{
  if (!target.has_value()) return {false};

  try {
    const auto x = target->ekf_x();
    if (x.size() < 8 || !x.allFinite()) throw std::runtime_error("Invalid target state");
    const double delay_time =
      std::abs(x[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;
    if (!std::isfinite(delay_time)) throw std::runtime_error("Invalid processing delay");
    const auto future =
      std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6));
    target->predict(future);
    return plan(std::move(target.value()), bullet_speed);
  } catch (const std::exception & e) {
    tools::logger()->warn("[Planner] prediction rejected: {}", e.what());
    return {false};
  }
}

void Planner::setup_yaw_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_yaw_acc = tools::read<double>(yaml, "max_yaw_acc");
  auto Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");
  auto R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");
  const int max_iter =
    std::clamp(yaml["mpc_max_iter"] ? yaml["mpc_max_iter"].as<int>() : 20, 1, 100);

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  yaw_solver_->settings->max_iter = max_iter;
}

void Planner::setup_pitch_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_pitch_acc = tools::read<double>(yaml, "max_pitch_acc");
  auto Q_pitch = tools::read<std::vector<double>>(yaml, "Q_pitch");
  auto R_pitch = tools::read<std::vector<double>>(yaml, "R_pitch");
  const int max_iter =
    std::clamp(yaml["mpc_max_iter"] ? yaml["mpc_max_iter"].as<int>() : 20, 1, 100);

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
  tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

  pitch_solver_->settings->max_iter = max_iter;
}

Eigen::Matrix<double, 2, 1> Planner::aim(
  const Target & target, double bullet_speed, bool sentry_world,
  std::optional<int> preferred_armor_id, bool aim_center)
{
  const Eigen::Vector4d selected_aim_point =
    select_aim_point(target, preferred_armor_id, aim_center);
  const Eigen::Vector3d xyz = selected_aim_point.head<3>();
  const double yaw = selected_aim_point[3];
  const double min_dist = xyz.head<2>().norm();
  debug_xyza = Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);

  auto azim = std::atan2(xyz.y(), xyz.x());
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  const double world_yaw = azim + yaw_offset_;
  const double elevation = bullet_traj.pitch + pitch_offset_;
  if (sentry_world) {
    const auto & x = target.ekf_x();
    const double big_yaw = std::atan2(x[2], x[0]);
    const auto command =
      sentry_mpc_transform::world_yaw_elevation_to_world_small_yaw_command(
        world_yaw, elevation, big_yaw, gimbal_axis_order_);
    return {command.small_yaw, command.pitch};
  }

  return tools::gimbal_command_from_yaw_elevation(world_yaw, elevation, gimbal_axis_order_);
}

Eigen::Vector4d Planner::select_aim_point(
  const Target & target, std::optional<int> preferred_armor_id, bool aim_center) const
{
  if (!aim_center) return select_armor(target, preferred_armor_id);

  const auto x = target.ekf_x();
  if (x.size() < 8 || !x.allFinite()) {
    throw std::runtime_error("Invalid target center state");
  }

  // x[4] is the lower armor-plane height. For four-armor targets x[10] is the alternating
  // height difference, so their midpoint is a better vertical vehicle center.
  double center_z = x[4];
  if (x.size() > 10 && std::isfinite(x[10])) center_z += 0.5 * x[10];
  return {x[0], x[2], center_z, x[6]};
}

Eigen::Vector4d Planner::select_armor(
  const Target & target, std::optional<int> preferred_armor_id) const
{
  const auto armor_xyza_list = target.armor_xyza_list();
  if (armor_xyza_list.empty()) throw std::runtime_error("Target has no armor candidate!");

  if (preferred_armor_id.has_value()) {
    if (
      preferred_armor_id.value() < 0 ||
      preferred_armor_id.value() >= static_cast<int>(armor_xyza_list.size())) {
      throw std::out_of_range("Preferred armor id is outside the predicted armor list");
    }
    return armor_xyza_list[preferred_armor_id.value()];
  }

  return *std::min_element(
    armor_xyza_list.begin(), armor_xyza_list.end(),
    [](const Eigen::Vector4d & lhs, const Eigen::Vector4d & rhs) {
      return lhs.head<2>().squaredNorm() < rhs.head<2>().squaredNorm();
    });
}

Trajectory Planner::get_trajectory(
  Target & target, double yaw0, double bullet_speed, bool sentry_world,
  std::optional<int> preferred_armor_id, bool aim_center)
{
  Trajectory traj;

  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target, bullet_speed, sentry_world, preferred_armor_id, aim_center);

  target.predict(DT);  // [0] = -HALF_HORIZON * DT -> [HHALF_HORIZON] = 0
  auto yaw_pitch = aim(target, bullet_speed, sentry_world, preferred_armor_id, aim_center);

  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
    auto yaw_pitch_next = aim(target, bullet_speed, sentry_world, preferred_armor_id, aim_center);

    auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  return traj;
}

}  // namespace auto_aim
