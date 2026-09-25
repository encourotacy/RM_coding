#include "outpost_target.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
constexpr double OUTPOST_RULE_SPEED = 0.8 * CV_PI;
constexpr double OUTPOST_XY_PROCESS_NOISE = 640.0;
constexpr double OUTPOST_Z_PROCESS_NOISE = 40.0;
constexpr double OUTPOST_YAW_PROCESS_NOISE = 4.0;
constexpr std::size_t INIT_LOCK_MIN_OBSERVATIONS = 2;
constexpr double INIT_LOCK_MAX_SCORE = 35.0;
constexpr double INIT_LOCK_MIN_MARGIN = 0.0;
constexpr double INIT_Z_RESIDUAL_GATE = 0.05;
constexpr double INIT_Z_MAX_RESIDUAL = 0.12;
constexpr double INIT_Z_VELOCITY_GATE = 1.2;
constexpr double INIT_Z_MAX_VELOCITY = 5.0;
constexpr double PREVIEW_MAX_SCORE = 45.0;
constexpr double PREVIEW_MAX_Z_RESIDUAL = 0.15;
constexpr double PREVIEW_MAX_CENTER_SPEED = 8.0;
constexpr double PREVIEW_MAX_AGE = 0.35;
constexpr double PREVIEW_MIN_OMEGA_MARGIN = 0.0;

inline double angle_offset(int id)
{
  return -static_cast<double>(id) * 2.0 * CV_PI / 3.0;
}

double normalized_square(double value, double gate)
{
  const double safe_gate = gate <= 1e-6 ? 1e-6 : gate;
  const double normalized = value / safe_gate;
  return normalized * normalized;
}

struct LineFitResult
{
  double intercept = 0.0;
  double slope = 0.0;
  double residual_score = 0.0;
  double max_abs_residual = 0.0;
};

LineFitResult fit_line(const std::vector<double> & times, const std::vector<double> & values)
{
  LineFitResult result;
  if (times.empty() || times.size() != values.size()) return result;

  const double inv_n = 1.0 / static_cast<double>(times.size());
  const double mean_t = std::accumulate(times.begin(), times.end(), 0.0) * inv_n;
  const double mean_v = std::accumulate(values.begin(), values.end(), 0.0) * inv_n;

  double variance_t = 0.0;
  double covariance_tv = 0.0;
  for (std::size_t i = 0; i < times.size(); ++i) {
    const double dt = times[i] - mean_t;
    variance_t += dt * dt;
    covariance_tv += dt * (values[i] - mean_v);
  }

  result.slope = variance_t > 1e-9 ? covariance_tv / variance_t : 0.0;
  result.intercept = mean_v - result.slope * mean_t;

  for (std::size_t i = 0; i < times.size(); ++i) {
    const double residual = values[i] - (result.intercept + result.slope * times[i]);
    result.max_abs_residual = std::max(result.max_abs_residual, std::abs(residual));
    result.residual_score += normalized_square(residual, INIT_Z_RESIDUAL_GATE);
  }
  return result;
}
}  // namespace

OutpostTarget::OutpostTarget(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius,
  const Eigen::VectorXd & P0_dig, bool static_direct_enabled, double static_speed_threshold,
  int static_motion_confirm_frames)
: t_(t),
  static_direct_enabled_(static_direct_enabled),
  static_speed_threshold_(std::max(0.0, static_speed_threshold)),
  static_motion_confirm_frames_(std::max(1, static_motion_confirm_frames))
{
  const auto & xyz = armor.xyz_in_world;
  const auto & ypr = armor.ypr_in_world;
  const double center_x = xyz[0] + radius * std::cos(ypr[0]);
  const double center_y = xyz[1] + radius * std::sin(ypr[0]);
  const double center_z = xyz[2];

  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, radius, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };
  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);

  layer_locked_ = false;
  last_id_ = -1;
  record_observed_armor(armor);
  const Observation initial_observation{t, xyz, center_from_armor(armor), ypr[0]};
  static_motion_observations_.push_back(initial_observation);
  init_observations_.push_back(initial_observation);
  ekf_.data["outpost_layer_locked"] = 0.0;
  ekf_.data["init_best_score"] = std::numeric_limits<double>::infinity();
  ekf_.data["init_margin"] = 0.0;
  ekf_.data["init_distinct_layers"] = 1.0;
  ekf_.data["init_z_vz"] = 0.0;
  ekf_.data["init_z_max_residual"] = 0.0;
}

void OutpostTarget::predict(std::chrono::steady_clock::time_point t)
{
  const double dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void OutpostTarget::predict(double dt)
{
  if (dt > 0.0) last_observed_age_s_ += dt;

  // clang-format off
  Eigen::MatrixXd F{
    {1, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 1, dt, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 1, dt, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 1, dt, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
  };
  // clang-format on

  const double a = dt * dt * dt * dt / 4.0;
  const double b = dt * dt * dt / 2.0;
  const double c = dt * dt;
  // clang-format off
  Eigen::MatrixXd Q{
    {a * OUTPOST_XY_PROCESS_NOISE, b * OUTPOST_XY_PROCESS_NOISE, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {b * OUTPOST_XY_PROCESS_NOISE, c * OUTPOST_XY_PROCESS_NOISE, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, a * OUTPOST_XY_PROCESS_NOISE, b * OUTPOST_XY_PROCESS_NOISE, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, b * OUTPOST_XY_PROCESS_NOISE, c * OUTPOST_XY_PROCESS_NOISE, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, a * OUTPOST_Z_PROCESS_NOISE, b * OUTPOST_Z_PROCESS_NOISE, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, b * OUTPOST_Z_PROCESS_NOISE, c * OUTPOST_Z_PROCESS_NOISE, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, a * OUTPOST_YAW_PROCESS_NOISE, b * OUTPOST_YAW_PROCESS_NOISE, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, b * OUTPOST_YAW_PROCESS_NOISE, c * OUTPOST_YAW_PROCESS_NOISE, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
  };
  // clang-format on

  auto f = [&](const Eigen::VectorXd & x) {
    Eigen::VectorXd prior = F * x;
    prior[6] = tools::limit_rad(prior[6]);
    return prior;
  };
  if (std::abs(ekf_.x[7]) > 3.4) {
    ekf_.x[7] = ekf_.x[7] > 0 ? OUTPOST_RULE_SPEED : -OUTPOST_RULE_SPEED;
  }
  ekf_.predict(F, Q, f);
}

void OutpostTarget::update(const Armor & armor, std::optional<int> forced_id)
{
  update_static_direct_state(armor);
  if (static_direct_active_) {
    layer_locked_ = false;
    preview_ready_ = false;
    init_observations_.clear();
    last_id_ = -1;
    jumped_ = false;
    switched_ = false;
    ekf_.x[7] = 0.0;
    ekf_.data["outpost_layer_locked"] = 0.0;
    ekf_.data["outpost_selected_id"] = -1.0;
    record_observed_armor(armor);
    update_count_++;
    return;
  }

  if (!layer_locked_) {
    observe_unlocked(armor);
    update_count_++;
    return;
  }

  const auto predictions = armor_xyza_list();
  int id = forced_id.has_value() && *forced_id >= 0 && *forced_id < ARMOR_NUM
             ? *forced_id
             : match_id(armor, predictions, -1, last_layer_);

  last_layer_ = id;
  if (id != 0) jumped_ = true;
  switched_ = id != last_id_;
  if (switched_) switch_count_++;
  last_id_ = id;
  update_count_++;

  const Eigen::Vector3d predicted_xyz = armor_xyz(ekf_.x, id);
  const double predicted_angle = tools::limit_rad(ekf_.x[6] + angle_offset(id));
  ekf_.data["outpost_layer_locked"] = 1.0;
  ekf_.data["outpost_selected_id"] = id;
  ekf_.data["outpost_layer_residual"] = armor.xyz_in_world[2] - predicted_xyz[2];
  ekf_.data["outpost_phase_residual"] =
    tools::limit_rad(armor.ypr_in_world[0] - predicted_angle);
  ekf_.data["outpost_center_speed"] = std::hypot(ekf_.x[1], ekf_.x[3]);

  update_ypda(armor, id);
}

void OutpostTarget::update_ypda(const Armor & armor, int id)
{
  Eigen::MatrixXd H = measurement_jacobian(ekf_.x, id);
  const double center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  const double delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig{{
    0.02, 0.02, std::log(std::abs(delta_angle) + 1.0) + 3.0,
    std::log(std::abs(armor.ypd_in_world[2]) + 1.0) / 200.0 + 0.5}};
  Eigen::MatrixXd R = R_dig.asDiagonal();

  auto h = [&](const Eigen::VectorXd & x) {
    const Eigen::VectorXd xyz = armor_xyz(x, id);
    const Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    const double angle = tools::limit_rad(x[6] + angle_offset(id));
    return Eigen::Vector4d{ypd[0], ypd[1], ypd[2], angle};
  };
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  const auto & ypd = armor.ypd_in_world;
  const auto & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};
  ekf_.update(z, H, R, h, z_subtract);
  record_observed_armor(armor);
}

int OutpostTarget::match_id(
  const Armor & armor, const std::vector<Eigen::Vector4d> & xyza_list,
  int measured_layer, int previous_layer) const
{
  if (xyza_list.empty()) return 0;
  auto clamp_layer = [](int layer) { return layer >= 0 && layer < 3 ? layer : -1; };
  measured_layer = clamp_layer(measured_layer);
  previous_layer = clamp_layer(previous_layer);
  const double z_gate = std::max(0.06, OUTPOST_LAYER_SPACING * 0.8);

  int best_id = 0;
  double best_score = std::numeric_limits<double>::infinity();
  for (int idx = 0; idx < ARMOR_NUM; ++idx) {
    const auto & xyza = xyza_list[idx];
    const Eigen::Vector3d predicted_ypd = tools::xyz2ypd(xyza.head(3));
    const double yaw_err = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3]));
    const double bearing_err =
      std::abs(tools::limit_rad(armor.ypd_in_world[0] - predicted_ypd[0]));
    const double pitch_err =
      std::abs(tools::limit_rad(armor.ypd_in_world[1] - predicted_ypd[1]));
    const double dist_err = std::abs(armor.ypd_in_world[2] - predicted_ypd[2]);
    const double z_err = std::abs(armor.xyz_in_world[2] - xyza[2]);

    double score =
      normalized_square(yaw_err, 0.45) +
      0.7 * normalized_square(bearing_err, 0.40) +
      0.4 * normalized_square(pitch_err, 0.30) +
      0.25 * normalized_square(dist_err, 0.60) +
      0.8 * normalized_square(z_err, z_gate);

    if (measured_layer >= 0) {
      const int diff = std::abs(idx - measured_layer);
      if (diff == 0) score *= 0.45;
      else if (diff == 1) score += 1.0;
      else score += 3.0;
    } else if (previous_layer >= 0 && std::abs(idx - previous_layer) >= 2) {
      score += 0.15;
    }
    if (score < best_score) {
      best_score = score;
      best_id = idx;
    }
  }
  return best_id;
}

Eigen::VectorXd OutpostTarget::ekf_x() const { return ekf_.x; }
const tools::ExtendedKalmanFilter & OutpostTarget::ekf() const { return ekf_; }
bool OutpostTarget::has_last_observed_armor() const { return has_last_observed_armor_; }
Eigen::Vector4d OutpostTarget::last_observed_armor_xyza() const { return last_observed_armor_xyza_; }
double OutpostTarget::last_observed_age() const { return last_observed_age_s_; }
bool OutpostTarget::layer_locked() const { return layer_locked_; }
bool OutpostTarget::static_direct_active() const { return static_direct_active_; }
bool OutpostTarget::unlocked_prediction_ready() const
{
  return !layer_locked_ && preview_ready_ && last_observed_age_s_ <= PREVIEW_MAX_AGE;
}
int OutpostTarget::last_id() const { return last_id_; }
bool OutpostTarget::jumped() const { return jumped_; }

void OutpostTarget::set_association_debug(
  int best_id, const std::array<double, 3> & scores, double best_score,
  const std::string & reject_reason)
{
  ekf_.data["assoc_best_id"] = static_cast<double>(best_id);
  ekf_.data["assoc_best_score"] = best_score;
  ekf_.data["assoc_score_0"] = scores[0];
  ekf_.data["assoc_score_1"] = scores[1];
  ekf_.data["assoc_score_2"] = scores[2];
  ekf_.data["assoc_reject_reason"] = reject_reason.empty() ? 0.0 : 1.0;
}

void OutpostTarget::set_layer_correction_debug(
  int raw_id, int height_id, double raw_z_residual, double best_z_residual,
  double z_improvement, int count, bool pending, bool applied)
{
  ekf_.data["outpost_layer_raw_id"] = static_cast<double>(raw_id);
  ekf_.data["outpost_layer_height_id"] = static_cast<double>(height_id);
  ekf_.data["outpost_layer_raw_z_residual"] = raw_z_residual;
  ekf_.data["outpost_layer_best_z_residual"] = best_z_residual;
  ekf_.data["outpost_layer_z_improvement"] = z_improvement;
  ekf_.data["outpost_layer_correction_count"] = static_cast<double>(count);
  ekf_.data["outpost_layer_correction_pending"] = pending ? 1.0 : 0.0;
  ekf_.data["outpost_layer_correction_applied"] = applied ? 1.0 : 0.0;
}

std::vector<Eigen::Vector4d> OutpostTarget::armor_xyza_list() const
{
  if (!layer_locked_) {
    if (!has_last_observed_armor_) return {};
    return {predicted_unlocked_xyza()};
  }
  std::vector<Eigen::Vector4d> result;
  result.reserve(ARMOR_NUM);
  for (int id = 0; id < ARMOR_NUM; ++id) {
    const double angle = tools::limit_rad(ekf_.x[6] + angle_offset(id));
    const Eigen::Vector3d xyz = armor_xyz(ekf_.x, id);
    result.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return result;
}

bool OutpostTarget::diverged() const
{
  const bool r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  const bool l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;
  if (r_ok && l_ok) return false;
  tools::logger()->debug("[OutpostTarget] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

bool OutpostTarget::converged()
{
  if (layer_locked_ && update_count_ > 8 && !diverged()) converged_ = true;
  return converged_;
}

void OutpostTarget::record_observed_armor(const Armor & armor)
{
  last_observed_armor_xyza_ << armor.xyz_in_world[0], armor.xyz_in_world[1],
    armor.xyz_in_world[2], armor.ypr_in_world[0];
  has_last_observed_armor_ = true;
  last_observed_age_s_ = 0.0;
}

Eigen::Vector3d OutpostTarget::armor_xyz(const Eigen::VectorXd & x, int id) const
{
  const double angle = tools::limit_rad(x[6] + angle_offset(id));
  return {
    x[0] - x[8] * std::cos(angle),
    x[2] - x[8] * std::sin(angle),
    x[4] + OUTPOST_LAYER_SPACING * static_cast<double>(id)};
}

Eigen::MatrixXd OutpostTarget::measurement_jacobian(const Eigen::VectorXd & x, int id) const
{
  const double angle = tools::limit_rad(x[6] + angle_offset(id));
  const double radius = x[8];
  const double dx_da = radius * std::sin(angle);
  const double dy_da = -radius * std::cos(angle);
  const double dx_dr = -std::cos(angle);
  const double dy_dr = -std::sin(angle);

  // clang-format off
  Eigen::MatrixXd H_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, 0, 0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, 0, 0},
    {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0}
  };
  // clang-format on
  const Eigen::MatrixXd H_ypd = tools::xyz2ypd_jacobian(armor_xyz(x, id));
  // clang-format off
  Eigen::MatrixXd H_ypda{
    {H_ypd(0, 0), H_ypd(0, 1), H_ypd(0, 2), 0},
    {H_ypd(1, 0), H_ypd(1, 1), H_ypd(1, 2), 0},
    {H_ypd(2, 0), H_ypd(2, 1), H_ypd(2, 2), 0},
    {0, 0, 0, 1}
  };
  // clang-format on
  return H_ypda * H_xyza;
}

Eigen::Vector3d OutpostTarget::center_from_armor(const Armor & armor) const
{
  return {
    armor.xyz_in_world[0] + ekf_.x[8] * std::cos(armor.ypr_in_world[0]),
    armor.xyz_in_world[1] + ekf_.x[8] * std::sin(armor.ypr_in_world[0]),
    armor.xyz_in_world[2]};
}

Eigen::Vector4d OutpostTarget::predicted_unlocked_xyza() const
{
  if (!preview_ready_ || preview_layer_ < 0) return last_observed_armor_xyza_;
  const double dt = std::max(0.0, tools::delta_time(t_, preview_t_));
  Eigen::Vector3d center = preview_center_;
  center[0] += preview_center_vxy_[0] * dt;
  center[1] += preview_center_vxy_[1] * dt;
  const double base_z = preview_base_z_ + preview_base_vz_ * dt;
  const double theta = tools::limit_rad(preview_theta_ + preview_omega_ * dt);
  const double angle = tools::limit_rad(theta + angle_offset(preview_layer_));
  const double z = base_z + OUTPOST_LAYER_SPACING * static_cast<double>(preview_layer_);
  return {
    center[0] - ekf_.x[8] * std::cos(angle),
    center[1] - ekf_.x[8] * std::sin(angle), z, angle};
}

void OutpostTarget::observe_unlocked(const Armor & armor)
{
  record_observed_armor(armor);
  init_observations_.push_back(
    Observation{t_, armor.xyz_in_world, center_from_armor(armor), armor.ypr_in_world[0]});
  if (init_observations_.size() > INIT_CACHE_LIMIT) init_observations_.pop_front();
  if (!try_lock_layers()) {
    last_id_ = -1;
    jumped_ = false;
    ekf_.data["outpost_layer_locked"] = 0.0;
    ekf_.data["outpost_selected_id"] = -1.0;
    ekf_.data["outpost_layer_residual"] = 0.0;
    ekf_.data["outpost_phase_residual"] = 0.0;
    ekf_.data["outpost_center_speed"] = 0.0;
  }
}

void OutpostTarget::update_static_direct_state(const Armor & armor)
{
  if (!static_direct_enabled_) return;

  static_motion_observations_.push_back(
    Observation{t_, armor.xyz_in_world, center_from_armor(armor), armor.ypr_in_world[0]});
  while (
    static_motion_observations_.size() > 2 &&
    tools::delta_time(
      static_motion_observations_.back().t, static_motion_observations_.front().t) > 0.16)
  {
    static_motion_observations_.pop_front();
  }

  if (static_motion_observations_.size() < 2) return;
  const double dt = tools::delta_time(
    static_motion_observations_.back().t, static_motion_observations_.front().t);
  if (dt < 0.10) return;

  const double yaw_rate = std::abs(tools::limit_rad(
    static_motion_observations_.back().yaw - static_motion_observations_.front().yaw)) / dt;
  const bool should_use_direct = yaw_rate <= static_speed_threshold_;
  const int desired_mode = should_use_direct ? 1 : 0;
  const int current_mode = static_direct_active_ ? 1 : 0;
  if (desired_mode == current_mode) {
    static_mode_candidate_ = -1;
    static_mode_count_ = 0;
    return;
  }

  if (static_mode_candidate_ == desired_mode) {
    static_mode_count_++;
  } else {
    static_mode_candidate_ = desired_mode;
    static_mode_count_ = 1;
  }
  if (static_mode_count_ < static_motion_confirm_frames_) return;

  static_direct_active_ = should_use_direct;
  static_mode_candidate_ = -1;
  static_mode_count_ = 0;
  layer_locked_ = false;
  preview_ready_ = false;
  init_observations_.clear();
  last_id_ = -1;
  jumped_ = false;
  if (static_direct_active_) {
    ekf_.x[7] = 0.0;
    ekf_.data["outpost_layer_locked"] = 0.0;
  }
}

bool OutpostTarget::try_lock_layers()
{
  if (init_observations_.size() < 2) {
    preview_ready_ = false;
    ekf_.data["init_best_score"] = std::numeric_limits<double>::infinity();
    ekf_.data["init_margin"] = 0.0;
    ekf_.data["init_omega_margin"] = 0.0;
    ekf_.data["init_distinct_layers"] = 1.0;
    ekf_.data["init_z_vz"] = 0.0;
    ekf_.data["init_z_max_residual"] = 0.0;
    ekf_.data["init_preview_ready"] = 0.0;
    return false;
  }

  struct Hypothesis
  {
    double score = 0.0;
    double omega = 0.0;
    std::vector<int> layers;
    std::vector<double> base_zs;
    std::vector<double> times;
    double base_z0 = 0.0;
    double base_vz = 0.0;
    double max_z_residual = 0.0;
    int distinct_layers = 0;
  };

  constexpr double step = 2.0 * CV_PI / 3.0;
  const auto & first = init_observations_.front();
  std::vector<Hypothesis> hypotheses;
  hypotheses.reserve(6);

  for (int first_layer = 0; first_layer < ARMOR_NUM; ++first_layer) {
    for (double omega : {OUTPOST_RULE_SPEED, -OUTPOST_RULE_SPEED}) {
      const double theta0 = tools::limit_rad(first.yaw + static_cast<double>(first_layer) * step);
      Hypothesis hypothesis;
      hypothesis.omega = omega;
      Eigen::Vector3d previous_center = Eigen::Vector3d::Zero();
      std::chrono::steady_clock::time_point previous_t{};
      bool has_previous = false;
      std::array<bool, 3> layer_seen{false, false, false};

      for (const auto & obs : init_observations_) {
        const double dt = tools::delta_time(obs.t, first.t);
        const double theta = tools::limit_rad(theta0 + omega * dt);
        int best_layer = 0;
        double best_phase_error = std::numeric_limits<double>::infinity();
        for (int layer = 0; layer < ARMOR_NUM; ++layer) {
          const double predicted_yaw = tools::limit_rad(theta - static_cast<double>(layer) * step);
          const double phase_error = std::abs(tools::limit_rad(obs.yaw - predicted_yaw));
          if (phase_error < best_phase_error) {
            best_phase_error = phase_error;
            best_layer = layer;
          }
        }

        const double base_z = obs.xyz[2] - OUTPOST_LAYER_SPACING * best_layer;
        hypothesis.layers.push_back(best_layer);
        hypothesis.base_zs.push_back(base_z);
        hypothesis.times.push_back(dt);
        layer_seen[best_layer] = true;
        hypothesis.score += normalized_square(best_phase_error, 0.35);
        if (has_previous) {
          const double dt_step = std::max(1e-3, tools::delta_time(obs.t, previous_t));
          const double center_step = (obs.center.head<2>() - previous_center.head<2>()).norm();
          hypothesis.score +=
            0.25 * normalized_square(center_step, 0.45 * dt_step + 0.12);
        }
        previous_center = obs.center;
        previous_t = obs.t;
        has_previous = true;
      }

      hypothesis.distinct_layers = static_cast<int>(layer_seen[0]) +
                                   static_cast<int>(layer_seen[1]) +
                                   static_cast<int>(layer_seen[2]);
      const LineFitResult z_fit = fit_line(hypothesis.times, hypothesis.base_zs);
      hypothesis.base_z0 = z_fit.intercept;
      hypothesis.base_vz = z_fit.slope;
      hypothesis.max_z_residual = z_fit.max_abs_residual;
      hypothesis.score += 1.8 * z_fit.residual_score;
      hypothesis.score += 0.45 * normalized_square(hypothesis.base_vz, INIT_Z_VELOCITY_GATE);
      if (std::abs(hypothesis.base_vz) > INIT_Z_MAX_VELOCITY) {
        hypothesis.score += 12.0 * normalized_square(
          std::abs(hypothesis.base_vz) - INIT_Z_MAX_VELOCITY, INIT_Z_VELOCITY_GATE);
      }
      hypotheses.push_back(std::move(hypothesis));
    }
  }

  std::sort(hypotheses.begin(), hypotheses.end(), [](const auto & a, const auto & b) {
    return a.score < b.score;
  });
  const Hypothesis & best = hypotheses.front();
  const double second_score = hypotheses.size() > 1 ? hypotheses[1].score : best.score;
  const double margin = second_score - best.score;
  double opposite_omega_score = std::numeric_limits<double>::infinity();
  for (const auto & hypothesis : hypotheses) {
    if (hypothesis.omega * best.omega < 0.0) {
      opposite_omega_score = std::min(opposite_omega_score, hypothesis.score);
    }
  }
  const double omega_margin = opposite_omega_score - best.score;
  ekf_.data["init_best_score"] = best.score;
  ekf_.data["init_margin"] = margin;
  ekf_.data["init_omega_margin"] = omega_margin;
  ekf_.data["init_distinct_layers"] = static_cast<double>(best.distinct_layers);
  ekf_.data["init_z_vz"] = best.base_vz;
  ekf_.data["init_z_max_residual"] = best.max_z_residual;
  ekf_.data["init_preview_ready"] = 0.0;

  const auto & last_obs = init_observations_.back();
  const int layer = best.layers.back();
  const double base_z = best.base_z0 + best.base_vz * best.times.back();
  const double theta = tools::limit_rad(last_obs.yaw + static_cast<double>(layer) * step);

  preview_ready_ = false;
  if (best.score <= PREVIEW_MAX_SCORE && omega_margin >= PREVIEW_MIN_OMEGA_MARGIN &&
      best.max_z_residual <= PREVIEW_MAX_Z_RESIDUAL &&
      std::abs(best.base_vz) <= INIT_Z_MAX_VELOCITY) {
    const auto & prev_obs = init_observations_[init_observations_.size() - 2];
    const double dt = std::max(1e-3, tools::delta_time(last_obs.t, prev_obs.t));
    preview_layer_ = layer;
    preview_omega_ = best.omega;
    preview_theta_ = theta;
    preview_base_z_ = base_z;
    preview_base_vz_ = best.base_vz;
    preview_center_ = last_obs.center;
    preview_center_vxy_[0] = std::clamp(
      (last_obs.center[0] - prev_obs.center[0]) / dt,
      -PREVIEW_MAX_CENTER_SPEED, PREVIEW_MAX_CENTER_SPEED);
    preview_center_vxy_[1] = std::clamp(
      (last_obs.center[1] - prev_obs.center[1]) / dt,
      -PREVIEW_MAX_CENTER_SPEED, PREVIEW_MAX_CENTER_SPEED);
    preview_t_ = last_obs.t;
    preview_ready_ = true;
    ekf_.data["init_preview_ready"] = 1.0;
  }

  if (init_observations_.size() < INIT_LOCK_MIN_OBSERVATIONS ||
      best.distinct_layers < 2 || best.score > INIT_LOCK_MAX_SCORE ||
      margin < INIT_LOCK_MIN_MARGIN || best.max_z_residual > INIT_Z_MAX_RESIDUAL ||
      std::abs(best.base_vz) > INIT_Z_MAX_VELOCITY) {
    return false;
  }

  ekf_.x[0] = last_obs.center[0];
  ekf_.x[2] = last_obs.center[1];
  ekf_.x[4] = base_z;
  ekf_.x[5] = best.base_vz;
  ekf_.x[6] = theta;
  ekf_.x[7] = best.omega;
  if (init_observations_.size() >= 2) {
    const auto & prev_obs = init_observations_[init_observations_.size() - 2];
    const double dt = std::max(1e-3, tools::delta_time(last_obs.t, prev_obs.t));
    ekf_.x[1] = std::clamp((last_obs.center[0] - prev_obs.center[0]) / dt, -8.0, 8.0);
    ekf_.x[3] = std::clamp((last_obs.center[1] - prev_obs.center[1]) / dt, -8.0, 8.0);
  }

  layer_locked_ = true;
  last_layer_ = layer;
  last_id_ = layer;
  jumped_ = layer != 0;
  ekf_.data["outpost_layer_locked"] = 1.0;
  ekf_.data["outpost_selected_id"] = static_cast<double>(layer);
  ekf_.data["outpost_layer_residual"] = 0.0;
  ekf_.data["outpost_phase_residual"] = 0.0;
  ekf_.data["outpost_center_speed"] = std::hypot(ekf_.x[1], ekf_.x[3]);
  return true;
}

}  // namespace auto_aim
