#include "target.hpp"

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
constexpr double NORMAL_TRANSLATION_PROCESS_NOISE = 100.0;
constexpr double NORMAL_YAW_PROCESS_NOISE = 400.0;
constexpr double OUTPOST_XY_PROCESS_NOISE = 100.0;
constexpr double OUTPOST_Z_PROCESS_NOISE = 10.0;
constexpr double OUTPOST_YAW_PROCESS_NOISE = 4.0;
constexpr double OUTPOST_INIT_Z_RESIDUAL_GATE = 0.035;
constexpr double OUTPOST_INIT_Z_MAX_RESIDUAL = 0.075;
constexpr double OUTPOST_INIT_Z_VELOCITY_GATE = 1.2;
constexpr double OUTPOST_INIT_Z_MAX_VELOCITY = 3.5;
constexpr double OUTPOST_PREVIEW_MAX_SCORE = 28.0;
constexpr double OUTPOST_PREVIEW_MAX_Z_RESIDUAL = 0.10;
constexpr double OUTPOST_PREVIEW_MAX_CENTER_SPEED = 8.0;
constexpr double OUTPOST_PREVIEW_MAX_AGE = 0.35;
constexpr double OUTPOST_PREVIEW_MIN_OMEGA_MARGIN = 0.02;
constexpr double OUTPOST_LOCK_MIN_OMEGA_MARGIN = 0.05;
constexpr int OUTPOST_INIT_MAX_OUTLIER_FRAMES = 1;

double armor_angle_offset(int id, int armor_num, ArmorName name)
{
  const double step = 2 * CV_PI / armor_num;
  const int signed_id = (name == ArmorName::outpost && armor_num == 3) ? -id : id;
  return static_cast<double>(signed_id) * step;
}

double normalized_square(double value, double gate)
{
  const double safe_gate = (gate <= 1e-6) ? 1e-6 : gate;
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
    result.residual_score += normalized_square(residual, OUTPOST_INIT_Z_RESIDUAL_GATE);
  }

  return result;
}
}  // namespace

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0)
{
  const auto radius_value = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  const auto center_x = xyz[0] + radius_value * std::cos(ypr[0]);
  const auto center_y = xyz[1] + radius_value * std::sin(ypr[0]);
  const auto center_z = xyz[2];

  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, radius_value, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);

  if (name == ArmorName::outpost && armor_num_ == 3) {
    outpost_layer_locked_ = false;
    last_id = -1;
    record_observed_armor(armor);
    outpost_init_observations_.push_back(
      OutpostObservation{t, xyz, outpost_center_from_armor(armor), ypr[0]});
    ekf_.data["outpost_layer_locked"] = 0.0;
    ekf_.data["init_best_score"] = std::numeric_limits<double>::infinity();
    ekf_.data["init_margin"] = 0.0;
    ekf_.data["init_distinct_layers"] = 1.0;
    ekf_.data["init_z_vz"] = 0.0;
    ekf_.data["init_z_max_residual"] = 0.0;
  }
}

Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  const auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  if (dt > 0.0) {
    last_observed_age_s_ += dt;
  }

  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
  };
  // clang-format on

  double q_xy = NORMAL_TRANSLATION_PROCESS_NOISE;
  double q_z = NORMAL_TRANSLATION_PROCESS_NOISE;
  double q_yaw = NORMAL_YAW_PROCESS_NOISE;
  if (name == ArmorName::outpost) {
    q_xy = OUTPOST_XY_PROCESS_NOISE;
    q_z = OUTPOST_Z_PROCESS_NOISE;
    q_yaw = OUTPOST_YAW_PROCESS_NOISE;
  }
  const auto a = dt * dt * dt * dt / 4;
  const auto b = dt * dt * dt / 2;
  const auto c = dt * dt;
  // clang-format off
  Eigen::MatrixXd Q{
    {a * q_xy, b * q_xy,        0,        0,       0,       0,        0,        0, 0, 0, 0},
    {b * q_xy, c * q_xy,        0,        0,       0,       0,        0,        0, 0, 0, 0},
    {        0,        0, a * q_xy, b * q_xy,       0,       0,        0,        0, 0, 0, 0},
    {        0,        0, b * q_xy, c * q_xy,       0,       0,        0,        0, 0, 0, 0},
    {        0,        0,        0,        0, a * q_z, b * q_z,        0,        0, 0, 0, 0},
    {        0,        0,        0,        0, b * q_z, c * q_z,        0,        0, 0, 0, 0},
    {        0,        0,        0,        0,       0,       0, a * q_yaw, b * q_yaw, 0, 0, 0},
    {        0,        0,        0,        0,       0,       0, b * q_yaw, c * q_yaw, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0}
  };
  // clang-format on

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  // The outpost rotates at the rule-defined angular speed. Once its layer and direction have
  // been identified, keep the EKF's yaw-rate state on that model instead of letting individual
  // PnP yaw updates slowly distort the long-horizon prediction.
  if (is_outpost_model() && outpost_layer_locked_) {
    ekf_.x[7] = ekf_.x[7] >= 0.0 ? OUTPOST_RULE_SPEED : -OUTPOST_RULE_SPEED;
  }

  ekf_.predict(F, Q, f);
}

void Target::update(const Armor & armor, std::optional<int> forced_id)
{
  if (is_outpost_model() && !outpost_layer_locked_) {
    observe_unlocked_outpost(armor);
    update_count_++;
    return;
  }

  int measured_outpost_layer = -1;
  int previous_outpost_layer = -1;
  if (name == ArmorName::outpost) {
    previous_outpost_layer = outpost_last_layer_;
  }

  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();
  int id = -1;
  if (forced_id.has_value() && forced_id.value() >= 0 && forced_id.value() < armor_num_) {
    id = forced_id.value();
  } else {
    id = measured_outpost_layer >= 0
           ? measured_outpost_layer
           : match_armor_id(armor, xyza_list, -1, previous_outpost_layer);
  }

  if (name == ArmorName::outpost) {
    outpost_last_layer_ = id;
  }

  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;

  if (name == ArmorName::outpost) {
    const Eigen::Vector3d predicted_xyz = h_armor_xyz(ekf_.x, id);
    const double predicted_angle =
      tools::limit_rad(ekf_.x[6] + armor_angle_offset(id, armor_num_, name));
    ekf_.data["outpost_layer_locked"] = 1.0;
    ekf_.data["outpost_selected_id"] = id;
    ekf_.data["outpost_layer_residual"] = armor.xyz_in_world[2] - predicted_xyz[2];
    ekf_.data["outpost_phase_residual"] =
      tools::limit_rad(armor.ypr_in_world[0] - predicted_angle);
    ekf_.data["outpost_center_speed"] = std::hypot(ekf_.x[1], ekf_.x[3]);
  }

  update_ypda(armor, id);
}

void Target::update_ypda(const Armor & armor, int id)
{
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  const auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  const auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);

  Eigen::VectorXd R_dig;
  if (name == ArmorName::outpost) {
    R_dig = Eigen::VectorXd{
      {0.02, 0.02, std::log(std::abs(delta_angle) + 1) + 3,
       std::log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 0.5}};
  } else {
    R_dig = Eigen::VectorXd{
      {4e-3, 4e-3, std::log(std::abs(delta_angle) + 1) + 1,
       std::log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};
  }

  Eigen::MatrixXd R = R_dig.asDiagonal();

  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    const auto angle = tools::limit_rad(x[6] + armor_angle_offset(id, armor_num_, name));
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};

  ekf_.update(z, H, R, h, z_subtract);
  record_observed_armor(armor);
}

int Target::match_armor_id(
  const Armor & armor, const std::vector<Eigen::Vector4d> & xyza_list,
  int measured_outpost_layer, int previous_outpost_layer)
{
  if (name == ArmorName::outpost && armor_num_ == 3) {
    return match_outpost_armor_id(armor, xyza_list, measured_outpost_layer, previous_outpost_layer);
  }
  return match_default_armor_id(armor, xyza_list);
}

int Target::match_default_armor_id(
  const Armor & armor, const std::vector<Eigen::Vector4d> & xyza_list)
{
  if (xyza_list.empty()) return 0;

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  xyza_i_list.reserve(xyza_list.size());
  for (int i = 0; i < armor_num_; ++i) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      const Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      const Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  int best_id = 0;
  double min_angle_error = std::numeric_limits<double>::infinity();
  const int candidates = std::min<int>(3, xyza_i_list.size());
  for (int i = 0; i < candidates; ++i) {
    const auto & xyza = xyza_i_list[i].first;
    const Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    const double angle_error =
      std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
      std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

    if (angle_error < min_angle_error) {
      best_id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }

  return best_id;
}

int Target::match_outpost_armor_id(
  const Armor & armor, const std::vector<Eigen::Vector4d> & xyza_list,
  int measured_outpost_layer, int previous_outpost_layer)
{
  if (xyza_list.empty()) return 0;

  auto clamp_layer = [&](int layer) -> int {
    if (layer < 0 || layer >= armor_num_) return -1;
    return layer;
  };

  measured_outpost_layer = clamp_layer(measured_outpost_layer);
  previous_outpost_layer = clamp_layer(previous_outpost_layer);

  const double spacing = OUTPOST_LAYER_SPACING;
  const double z_gate = std::max(0.06, spacing * 0.8);

  int best_id = 0;
  double best_score = std::numeric_limits<double>::infinity();

  for (int idx = 0; idx < armor_num_; ++idx) {
    const auto & xyza = xyza_list[idx];
    const Eigen::Vector3d predicted_ypd = tools::xyz2ypd(xyza.head(3));

    const double yaw_err = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3]));
    const double bearing_err = std::abs(tools::limit_rad(armor.ypd_in_world[0] - predicted_ypd[0]));
    const double pitch_err = std::abs(tools::limit_rad(armor.ypd_in_world[1] - predicted_ypd[1]));
    const double dist_err = std::abs(armor.ypd_in_world[2] - predicted_ypd[2]);
    const double z_err = std::abs(armor.xyz_in_world[2] - xyza[2]);

    double score =
      1.0 * normalized_square(yaw_err, 0.45) +
      0.7 * normalized_square(bearing_err, 0.40) +
      0.4 * normalized_square(pitch_err, 0.30) +
      0.25 * normalized_square(dist_err, 0.60) +
      0.8 * normalized_square(z_err, z_gate);

    if (measured_outpost_layer >= 0) {
      const int diff = std::abs(idx - measured_outpost_layer);
      if (diff == 0) {
        score *= 0.45;
      } else if (diff == 1) {
        score += 1.0;
      } else {
        score += 3.0;
      }
    } else if (previous_outpost_layer >= 0) {
      const int diff_prev = std::abs(idx - previous_outpost_layer);
      if (diff_prev >= 2) score += 0.15;
    }

    if (score < best_score) {
      best_score = score;
      best_id = idx;
    }
  }

  return best_id;
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

bool Target::has_last_observed_armor() const { return has_last_observed_armor_; }

Eigen::Vector4d Target::last_observed_armor_xyza() const { return last_observed_armor_xyza_; }

double Target::last_observed_age() const { return last_observed_age_s_; }

void Target::clear_timing_debug_pair()
{
  has_timing_debug_pair_ = false;
  timing_debug_detection_points_.clear();
}

void Target::set_timing_debug_pair(const Armor & armor, const Eigen::Vector4d & predicted_xyza)
{
  timing_debug_detection_points_ = armor.points;
  timing_debug_prediction_xyza_ = predicted_xyza;
  has_timing_debug_pair_ = true;
}

bool Target::has_timing_debug_pair() const { return has_timing_debug_pair_; }

const std::vector<cv::Point2f> & Target::timing_debug_detection_points() const
{
  return timing_debug_detection_points_;
}

Eigen::Vector4d Target::timing_debug_prediction_xyza() const
{
  return timing_debug_prediction_xyza_;
}

bool Target::outpost_layer_locked() const { return !is_outpost_model() || outpost_layer_locked_; }

bool Target::outpost_unlocked_prediction_ready() const
{
  return is_outpost_model() && !outpost_layer_locked_ && outpost_preview_ready_ &&
         last_observed_age_s_ <= OUTPOST_PREVIEW_MAX_AGE;
}

void Target::set_outpost_association_debug(
  int best_id, const std::array<double, 3> & scores, double best_score,
  const std::string & reject_reason)
{
  if (!is_outpost_model()) return;
  ekf_.data["assoc_best_id"] = static_cast<double>(best_id);
  ekf_.data["assoc_best_score"] = best_score;
  ekf_.data["assoc_score_0"] = scores[0];
  ekf_.data["assoc_score_1"] = scores[1];
  ekf_.data["assoc_score_2"] = scores[2];
  ekf_.data["assoc_reject_reason"] = reject_reason.empty() ? 0.0 : 1.0;
}

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  if (is_outpost_model() && !outpost_layer_locked_) {
    if (outpost_unlocked_prediction_ready()) return {predicted_unlocked_outpost_xyza()};
    if (!has_last_observed_armor_) return {};
    return {last_observed_armor_xyza_};
  }

  std::vector<Eigen::Vector4d> xyza_list;
  for (int i = 0; i < armor_num_; i++) {
    const auto angle = tools::limit_rad(ekf_.x[6] + armor_angle_offset(i, armor_num_, name));
    const Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return xyza_list;
}

bool Target::diverged() const
{
  const auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  const auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

void Target::record_observed_armor(const Armor & armor)
{
  last_observed_armor_xyza_ << armor.xyz_in_world[0], armor.xyz_in_world[1], armor.xyz_in_world[2],
    armor.ypr_in_world[0];
  has_last_observed_armor_ = true;
  last_observed_age_s_ = 0.0;
}

bool Target::convergened()
{
  if (name != ArmorName::outpost && update_count_ > 3 && !diverged()) {
    is_converged_ = true;
  }

  if (is_outpost_model() && outpost_layer_locked_ && update_count_ > 8 && !diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  const auto angle = tools::limit_rad(x[6] + armor_angle_offset(id, armor_num_, name));
  const auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
  const auto is_outpost = (name == ArmorName::outpost) && (armor_num_ == 3);

  const auto radius = use_l_h ? x[8] + x[9] : x[8];
  const auto armor_x = x[0] - radius * std::cos(angle);
  const auto armor_y = x[2] - radius * std::sin(angle);

  auto armor_z = x[4];
  if (use_l_h) {
    armor_z += x[10];
  } else if (is_outpost) {
    armor_z += OUTPOST_LAYER_SPACING * static_cast<double>(id);
  }

  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  const auto angle = tools::limit_rad(x[6] + armor_angle_offset(id, armor_num_, name));
  const auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  const auto radius = use_l_h ? x[8] + x[9] : x[8];
  const auto dx_da = radius * std::sin(angle);
  const auto dy_da = -radius * std::cos(angle);

  const auto dx_dr = -std::cos(angle);
  const auto dy_dr = -std::sin(angle);
  const auto dx_dl = use_l_h ? -std::cos(angle) : 0.0;
  const auto dy_dl = use_l_h ? -std::sin(angle) : 0.0;
  const auto dz_dh = use_l_h ? 1.0 : 0.0;

  // clang-format off
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
  };
  // clang-format on

  const Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  const Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  // clang-format off
  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };
  // clang-format on

  return H_armor_ypda * H_armor_xyza;
}

bool Target::checkinit() { return isinit; }

bool Target::is_outpost_model() const
{
  return name == ArmorName::outpost && armor_num_ == 3;
}

Eigen::Vector3d Target::outpost_center_from_armor(const Armor & armor) const
{
  return {
    armor.xyz_in_world[0] + ekf_.x[8] * std::cos(armor.ypr_in_world[0]),
    armor.xyz_in_world[1] + ekf_.x[8] * std::sin(armor.ypr_in_world[0]),
    armor.xyz_in_world[2]};
}

Eigen::Vector4d Target::predicted_unlocked_outpost_xyza() const
{
  if (!outpost_preview_ready_ || outpost_preview_layer_ < 0) return last_observed_armor_xyza_;

  const double dt = std::max(0.0, tools::delta_time(t_, outpost_preview_t_));
  Eigen::Vector3d center = outpost_preview_center_;
  center[0] += outpost_preview_center_vxy_[0] * dt;
  center[1] += outpost_preview_center_vxy_[1] * dt;

  const double base_z = outpost_preview_base_z_ + outpost_preview_base_vz_ * dt;
  const double theta = tools::limit_rad(outpost_preview_theta_ + outpost_preview_omega_ * dt);
  const double angle =
    tools::limit_rad(theta + armor_angle_offset(outpost_preview_layer_, armor_num_, name));
  const double radius = ekf_.x[8];
  const double z = base_z + OUTPOST_LAYER_SPACING * static_cast<double>(outpost_preview_layer_);

  return {center[0] - radius * std::cos(angle), center[1] - radius * std::sin(angle), z, angle};
}

void Target::observe_unlocked_outpost(const Armor & armor)
{
  record_observed_armor(armor);

  outpost_init_observations_.push_back(
    OutpostObservation{t_, armor.xyz_in_world, outpost_center_from_armor(armor), armor.ypr_in_world[0]});
  if (outpost_init_observations_.size() > OUTPOST_INIT_CACHE_LIMIT) {
    outpost_init_observations_.pop_front();
  }

  if (!try_lock_outpost_layers()) {
    last_id = -1;
    jumped = false;
    ekf_.data["outpost_layer_locked"] = 0.0;
    ekf_.data["outpost_selected_id"] = -1.0;
    ekf_.data["outpost_layer_residual"] = 0.0;
    ekf_.data["outpost_phase_residual"] = 0.0;
    ekf_.data["outpost_center_speed"] = 0.0;
  }
}

bool Target::try_lock_outpost_layers()
{
  if (outpost_init_observations_.size() < 2) {
    outpost_preview_ready_ = false;
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
    bool alive = true;
    double score = 0.0;
    double omega = 0.0;
    std::vector<int> layers;
    std::vector<double> base_zs;
    std::vector<double> times;
    double base_z0 = 0.0;
    double base_vz = 0.0;
    double max_z_residual = 0.0;
    int distinct_layers = 0;
    int outlier_count = 0;
    int reject_reason = 0;
    std::size_t last_valid_index = 0;
    double base_z_min = -std::numeric_limits<double>::infinity();
    double base_z_max = std::numeric_limits<double>::infinity();
  };

  constexpr double step = 2.0 * CV_PI / 3.0;
  const double spacing = OUTPOST_LAYER_SPACING;
  const auto & first = outpost_init_observations_.front();

  std::vector<Hypothesis> hypotheses;
  hypotheses.reserve(6);

  const auto phase_gate_from_distance = [](double distance) {
    return std::clamp(0.24 + 0.012 * distance, 0.26, 0.42);
  };
  const auto z_gate_from_distance = [](double distance) {
    return std::clamp(0.035 + 0.008 * distance, 0.06, 0.18);
  };

  for (int first_layer = 0; first_layer < armor_num_; ++first_layer) {
    for (double omega : {OUTPOST_RULE_SPEED, -OUTPOST_RULE_SPEED}) {
      const double theta0 = tools::limit_rad(first.yaw + static_cast<double>(first_layer) * step);
      Hypothesis hypothesis;
      hypothesis.omega = omega;
      hypothesis.layers.reserve(outpost_init_observations_.size());
      hypothesis.base_zs.reserve(outpost_init_observations_.size());
      hypothesis.times.reserve(outpost_init_observations_.size());

      Eigen::Vector3d previous_center = Eigen::Vector3d::Zero();
      std::chrono::steady_clock::time_point previous_t{};
      bool has_previous = false;
      std::array<bool, 3> layer_seen{false, false, false};

      for (std::size_t obs_index = 0; obs_index < outpost_init_observations_.size(); ++obs_index) {
        const auto & obs = outpost_init_observations_[obs_index];
        const double dt = tools::delta_time(obs.t, first.t);
        const double theta = tools::limit_rad(theta0 + omega * dt);

        int best_layer = 0;
        double best_phase_error = std::numeric_limits<double>::infinity();
        for (int layer = 0; layer < armor_num_; ++layer) {
          const double predicted_yaw =
            tools::limit_rad(theta - static_cast<double>(layer) * step);
          const double phase_error = std::abs(tools::limit_rad(obs.yaw - predicted_yaw));
          if (phase_error < best_phase_error) {
            best_phase_error = phase_error;
            best_layer = layer;
          }
        }

        const double distance = obs.xyz.norm();
        if (best_phase_error > phase_gate_from_distance(distance)) {
          hypothesis.outlier_count++;
          hypothesis.reject_reason = 1;
          hypothesis.alive =
            hypothesis.outlier_count <= OUTPOST_INIT_MAX_OUTLIER_FRAMES;
          if (!hypothesis.alive) break;
          continue;
        }

        const double base_z = obs.xyz[2] - spacing * static_cast<double>(best_layer);
        const double z_gate = z_gate_from_distance(distance);
        const double proposed_base_z_min = std::max(hypothesis.base_z_min, base_z - z_gate);
        const double proposed_base_z_max = std::min(hypothesis.base_z_max, base_z + z_gate);
        if (proposed_base_z_min > proposed_base_z_max) {
          hypothesis.outlier_count++;
          hypothesis.reject_reason = 2;
          hypothesis.alive =
            hypothesis.outlier_count <= OUTPOST_INIT_MAX_OUTLIER_FRAMES;
          if (!hypothesis.alive) break;
          continue;
        }
        hypothesis.base_z_min = proposed_base_z_min;
        hypothesis.base_z_max = proposed_base_z_max;
        hypothesis.last_valid_index = obs_index;
        hypothesis.layers.push_back(best_layer);
        hypothesis.base_zs.push_back(base_z);
        hypothesis.times.push_back(dt);
        layer_seen[best_layer] = true;

        hypothesis.score += 1.0 * normalized_square(best_phase_error, 0.35);
        if (has_previous) {
          const double dt_step = std::max(1e-3, tools::delta_time(obs.t, previous_t));
          const double center_step = (obs.center.head<2>() - previous_center.head<2>()).norm();
          hypothesis.score += 0.25 * normalized_square(center_step, 0.45 * dt_step + 0.12);
        }

        previous_center = obs.center;
        previous_t = obs.t;
        has_previous = true;
      }

      if (hypothesis.layers.empty()) hypothesis.alive = false;
      hypothesis.distinct_layers =
        static_cast<int>(layer_seen[0]) + static_cast<int>(layer_seen[1]) +
        static_cast<int>(layer_seen[2]);
      if (hypothesis.alive) {
        const LineFitResult z_fit = fit_line(hypothesis.times, hypothesis.base_zs);
        hypothesis.base_z0 = z_fit.intercept;
        hypothesis.base_vz = z_fit.slope;
        hypothesis.max_z_residual = z_fit.max_abs_residual;
        hypothesis.score += 1.8 * z_fit.residual_score;
        hypothesis.score +=
          0.45 * normalized_square(hypothesis.base_vz, OUTPOST_INIT_Z_VELOCITY_GATE);
        if (std::abs(hypothesis.base_vz) > OUTPOST_INIT_Z_MAX_VELOCITY) {
          hypothesis.score +=
            12.0 *
            normalized_square(
              std::abs(hypothesis.base_vz) - OUTPOST_INIT_Z_MAX_VELOCITY,
              OUTPOST_INIT_Z_VELOCITY_GATE);
        }
      }
      hypotheses.push_back(std::move(hypothesis));
    }
  }

  int alive_count = 0;
  for (std::size_t i = 0; i < hypotheses.size(); ++i) {
    const auto & hypothesis = hypotheses[i];
    alive_count += hypothesis.alive ? 1 : 0;
    ekf_.data["init_hypothesis_" + std::to_string(i) + "_alive"] =
      hypothesis.alive ? 1.0 : 0.0;
    ekf_.data["init_hypothesis_" + std::to_string(i) + "_reason"] =
      static_cast<double>(hypothesis.reject_reason);
  }
  ekf_.data["init_alive_hypotheses"] = static_cast<double>(alive_count);

  if (alive_count == 0) {
    const OutpostObservation latest = outpost_init_observations_.back();
    outpost_init_observations_.clear();
    outpost_init_observations_.push_back(latest);
    outpost_preview_ready_ = false;
    ekf_.data["init_preview_ready"] = 0.0;
    ekf_.data["init_restarted"] = 1.0;
    return false;
  }
  ekf_.data["init_restarted"] = 0.0;

  hypotheses.erase(
    std::remove_if(
      hypotheses.begin(), hypotheses.end(),
      [](const Hypothesis & hypothesis) { return !hypothesis.alive; }),
    hypotheses.end());
  std::sort(
    hypotheses.begin(), hypotheses.end(),
    [](const Hypothesis & lhs, const Hypothesis & rhs) { return lhs.score < rhs.score; });

  const Hypothesis & best = hypotheses.front();
  const double second_score =
    hypotheses.size() > 1 ? hypotheses[1].score : std::numeric_limits<double>::infinity();
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

  const auto & last_obs = outpost_init_observations_[best.last_valid_index];
  const int layer = best.layers.back();
  const double base_z = best.base_z0 + best.base_vz * best.times.back();
  const double theta = tools::limit_rad(last_obs.yaw + static_cast<double>(layer) * step);

  outpost_preview_ready_ = false;
  if (
    best.score <= OUTPOST_PREVIEW_MAX_SCORE &&
    omega_margin >= OUTPOST_PREVIEW_MIN_OMEGA_MARGIN &&
    best.max_z_residual <= OUTPOST_PREVIEW_MAX_Z_RESIDUAL &&
    std::abs(best.base_vz) <= OUTPOST_INIT_Z_MAX_VELOCITY &&
    best.last_valid_index + 1 == outpost_init_observations_.size()) {
    const auto & prev_obs = outpost_init_observations_[best.last_valid_index - 1];
    const double dt = std::max(1e-3, tools::delta_time(last_obs.t, prev_obs.t));

    outpost_preview_layer_ = layer;
    outpost_preview_omega_ = best.omega;
    outpost_preview_theta_ = theta;
    outpost_preview_base_z_ = base_z;
    outpost_preview_base_vz_ = best.base_vz;
    outpost_preview_center_ = last_obs.center;
    const double center_vx = std::clamp(
      (last_obs.center[0] - prev_obs.center[0]) / dt, -OUTPOST_PREVIEW_MAX_CENTER_SPEED,
      OUTPOST_PREVIEW_MAX_CENTER_SPEED);
    const double center_vy = std::clamp(
      (last_obs.center[1] - prev_obs.center[1]) / dt, -OUTPOST_PREVIEW_MAX_CENTER_SPEED,
      OUTPOST_PREVIEW_MAX_CENTER_SPEED);
    outpost_preview_center_vxy_ << center_vx, center_vy;
    outpost_preview_t_ = last_obs.t;
    outpost_preview_ready_ = true;
    ekf_.data["init_preview_ready"] = 1.0;
  }

  if (outpost_init_observations_.size() < 3 || best.distinct_layers < 2 || best.score > 18.0 ||
      margin < 0.05 || omega_margin < OUTPOST_LOCK_MIN_OMEGA_MARGIN ||
      best.max_z_residual > OUTPOST_INIT_Z_MAX_RESIDUAL ||
      std::abs(best.base_vz) > OUTPOST_INIT_Z_MAX_VELOCITY ||
      best.last_valid_index + 1 != outpost_init_observations_.size()) {
    return false;
  }

  ekf_.x[0] = last_obs.center[0];
  ekf_.x[2] = last_obs.center[1];
  ekf_.x[4] = base_z;
  ekf_.x[5] = best.base_vz;
  ekf_.x[6] = theta;
  ekf_.x[7] = best.omega;

  if (best.last_valid_index >= 1) {
    const auto & prev_obs = outpost_init_observations_[best.last_valid_index - 1];
    const double dt = std::max(1e-3, tools::delta_time(last_obs.t, prev_obs.t));
    ekf_.x[1] = std::clamp((last_obs.center[0] - prev_obs.center[0]) / dt, -8.0, 8.0);
    ekf_.x[3] = std::clamp((last_obs.center[1] - prev_obs.center[1]) / dt, -8.0, 8.0);
  }

  outpost_layer_locked_ = true;
  outpost_last_layer_ = layer;
  last_id = layer;
  jumped = layer != 0;

  ekf_.data["outpost_layer_locked"] = 1.0;
  ekf_.data["outpost_selected_id"] = static_cast<double>(layer);
  ekf_.data["outpost_layer_residual"] = 0.0;
  ekf_.data["outpost_phase_residual"] = 0.0;
  ekf_.data["outpost_center_speed"] = std::hypot(ekf_.x[1], ekf_.x[3]);
  return true;
}

}  // namespace auto_aim
