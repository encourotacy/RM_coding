#include "target.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
constexpr double NORMAL_TRANSLATION_PROCESS_NOISE = 100.0;
constexpr double NORMAL_YAW_PROCESS_NOISE = 400.0;

double armor_angle_offset(int id, int armor_num)
{
  return static_cast<double>(id) * 2.0 * CV_PI / static_cast<double>(armor_num);
}
}  // namespace

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig, bool outpost_static_direct_enabled,
  double outpost_static_speed_threshold, int outpost_static_motion_confirm_frames)
: name(armor.name),
  armor_type(armor.type),
  priority(armor.priority),
  jumped(false),
  last_id(0),
  armor_num_(armor_num),
  t_(t)
{
  if (name == ArmorName::outpost && armor_num == 3) {
    outpost_target_.emplace(
      armor, t, radius, P0_dig, outpost_static_direct_enabled, outpost_static_speed_threshold,
      outpost_static_motion_confirm_frames);
    last_id = outpost_target_->last_id();
    return;
  }

  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
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
}

Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  if (outpost_target_.has_value()) {
    outpost_target_->predict(t);
    return;
  }

  const double dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  if (outpost_target_.has_value()) {
    outpost_target_->predict(dt);
    return;
  }

  if (dt > 0.0) last_observed_age_s_ += dt;

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

  const double a = dt * dt * dt * dt / 4.0;
  const double b = dt * dt * dt / 2.0;
  const double c = dt * dt;
  // clang-format off
  Eigen::MatrixXd Q{
    {a * NORMAL_TRANSLATION_PROCESS_NOISE, b * NORMAL_TRANSLATION_PROCESS_NOISE, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {b * NORMAL_TRANSLATION_PROCESS_NOISE, c * NORMAL_TRANSLATION_PROCESS_NOISE, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, a * NORMAL_TRANSLATION_PROCESS_NOISE, b * NORMAL_TRANSLATION_PROCESS_NOISE, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, b * NORMAL_TRANSLATION_PROCESS_NOISE, c * NORMAL_TRANSLATION_PROCESS_NOISE, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, a * NORMAL_TRANSLATION_PROCESS_NOISE, b * NORMAL_TRANSLATION_PROCESS_NOISE, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, b * NORMAL_TRANSLATION_PROCESS_NOISE, c * NORMAL_TRANSLATION_PROCESS_NOISE, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, a * NORMAL_YAW_PROCESS_NOISE, b * NORMAL_YAW_PROCESS_NOISE, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, b * NORMAL_YAW_PROCESS_NOISE, c * NORMAL_YAW_PROCESS_NOISE, 0, 0, 0},
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
  ekf_.predict(F, Q, f);
}

void Target::update(const Armor & armor, std::optional<int> forced_id)
{
  if (outpost_target_.has_value()) {
    outpost_target_->update(armor, forced_id);
    last_id = outpost_target_->last_id();
    jumped = outpost_target_->jumped();
    return;
  }

  const std::vector<Eigen::Vector4d> xyza_list = armor_xyza_list();
  int id = 0;
  if (forced_id.has_value() && forced_id.value() >= 0 && forced_id.value() < armor_num_) {
    id = forced_id.value();
  } else {
    id = match_armor_id(armor, xyza_list);
  }

  if (id != 0) jumped = true;
  is_switch_ = id != last_id;
  if (is_switch_) switch_count_++;
  last_id = id;
  update_count_++;
  update_ypda(armor, id);
}

void Target::update_ypda(const Armor & armor, int id)
{
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  const double center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  const double delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);

  Eigen::VectorXd R_dig{
    {4e-3, 4e-3, std::log(std::abs(delta_angle) + 1.0) + 1.0,
     std::log(std::abs(armor.ypd_in_world[2]) + 1.0) / 200.0 + 9e-2}};
  Eigen::MatrixXd R = R_dig.asDiagonal();

  auto h = [&](const Eigen::VectorXd & x) {
    const Eigen::VectorXd xyz = h_armor_xyz(x, id);
    const Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    const double angle = tools::limit_rad(x[6] + armor_angle_offset(id, armor_num_));
    return Eigen::Vector4d{ypd[0], ypd[1], ypd[2], angle};
  };

  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) {
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
  const Armor & armor, const std::vector<Eigen::Vector4d> & xyza_list) const
{
  if (xyza_list.empty()) return 0;

  std::vector<std::pair<Eigen::Vector4d, int>> candidates;
  candidates.reserve(xyza_list.size());
  for (int i = 0; i < armor_num_; ++i) candidates.push_back({xyza_list[i], i});

  std::sort(
    candidates.begin(), candidates.end(),
    [](const auto & a, const auto & b) {
      return tools::xyz2ypd(a.first.head(3))[2] < tools::xyz2ypd(b.first.head(3))[2];
    });

  int best_id = 0;
  double min_angle_error = std::numeric_limits<double>::infinity();
  const int count = std::min<int>(3, candidates.size());
  for (int i = 0; i < count; ++i) {
    const Eigen::Vector4d & xyza = candidates[i].first;
    const Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    const double error =
      std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
      std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));
    if (error < min_angle_error) {
      min_angle_error = error;
      best_id = candidates[i].second;
    }
  }
  return best_id;
}

Eigen::VectorXd Target::ekf_x() const
{
  return outpost_target_.has_value() ? outpost_target_->ekf_x() : ekf_.x;
}

const tools::ExtendedKalmanFilter & Target::ekf() const
{
  return outpost_target_.has_value() ? outpost_target_->ekf() : ekf_;
}

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  if (outpost_target_.has_value()) return outpost_target_->armor_xyza_list();

  std::vector<Eigen::Vector4d> xyza_list;
  xyza_list.reserve(armor_num_);
  for (int i = 0; i < armor_num_; ++i) {
    const double angle = tools::limit_rad(ekf_.x[6] + armor_angle_offset(i, armor_num_));
    const Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return xyza_list;
}

bool Target::has_last_observed_armor() const
{
  return outpost_target_.has_value() ? outpost_target_->has_last_observed_armor()
                                     : has_last_observed_armor_;
}

Eigen::Vector4d Target::last_observed_armor_xyza() const
{
  return outpost_target_.has_value() ? outpost_target_->last_observed_armor_xyza()
                                     : last_observed_armor_xyza_;
}

double Target::last_observed_age() const
{
  return outpost_target_.has_value() ? outpost_target_->last_observed_age()
                                     : last_observed_age_s_;
}

bool Target::outpost_layer_locked() const
{
  return !outpost_target_.has_value() || outpost_target_->layer_locked();
}

bool Target::outpost_static_direct_active() const
{
  return outpost_target_.has_value() && outpost_target_->static_direct_active();
}

bool Target::outpost_unlocked_prediction_ready() const
{
  return outpost_target_.has_value() && !outpost_target_->layer_locked() &&
         outpost_target_->unlocked_prediction_ready();
}

void Target::set_outpost_association_debug(
  int best_id, const std::array<double, 3> & scores, double best_score,
  const std::string & reject_reason)
{
  if (outpost_target_.has_value()) {
    outpost_target_->set_association_debug(best_id, scores, best_score, reject_reason);
  }
}

void Target::set_outpost_layer_correction_debug(
  int raw_id, int height_id, double raw_z_residual, double best_z_residual,
  double z_improvement, int count, bool pending, bool applied)
{
  if (outpost_target_.has_value()) {
    outpost_target_->set_layer_correction_debug(
      raw_id, height_id, raw_z_residual, best_z_residual, z_improvement, count, pending, applied);
  }
}

bool Target::diverged() const
{
  if (outpost_target_.has_value()) return outpost_target_->diverged();

  const bool r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  const bool l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;
  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

void Target::record_observed_armor(const Armor & armor)
{
  last_observed_armor_xyza_ << armor.xyz_in_world[0], armor.xyz_in_world[1],
    armor.xyz_in_world[2], armor.ypr_in_world[0];
  has_last_observed_armor_ = true;
  last_observed_age_s_ = 0.0;
}

bool Target::convergened()
{
  if (outpost_target_.has_value()) return outpost_target_->converged();
  if (update_count_ > 3 && !diverged()) is_converged_ = true;
  return is_converged_;
}

Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  const double angle = tools::limit_rad(x[6] + armor_angle_offset(id, armor_num_));
  const bool use_l_h = armor_num_ == 4 && (id == 1 || id == 3);
  const double radius = use_l_h ? x[8] + x[9] : x[8];

  const double armor_x = x[0] - radius * std::cos(angle);
  const double armor_y = x[2] - radius * std::sin(angle);
  const double armor_z = x[4] + (use_l_h ? x[10] : 0.0);
  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  const double angle = tools::limit_rad(x[6] + armor_angle_offset(id, armor_num_));
  const bool use_l_h = armor_num_ == 4 && (id == 1 || id == 3);
  const double radius = use_l_h ? x[8] + x[9] : x[8];

  const double dx_da = radius * std::sin(angle);
  const double dy_da = -radius * std::cos(angle);
  const double dx_dr = -std::cos(angle);
  const double dy_dr = -std::sin(angle);
  const double dx_dl = use_l_h ? -std::cos(angle) : 0.0;
  const double dy_dl = use_l_h ? -std::sin(angle) : 0.0;
  const double dz_dh = use_l_h ? 1.0 : 0.0;

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

}  // namespace auto_aim
