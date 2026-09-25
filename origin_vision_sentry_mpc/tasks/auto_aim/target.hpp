#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>

#include <array>
#include <chrono>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "armor.hpp"
#include "outpost_target.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

class Target
{
public:
  ArmorName name;
  ArmorType armor_type;
  ArmorPriority priority;
  bool jumped;
  int last_id;  // debug only

  Target() = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig, bool outpost_static_direct_enabled = false,
    double outpost_static_speed_threshold = 0.15,
    int outpost_static_motion_confirm_frames = 3);
  Target(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  void update(const Armor & armor, std::optional<int> forced_id = std::nullopt);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;
  bool has_last_observed_armor() const;
  Eigen::Vector4d last_observed_armor_xyza() const;
  double last_observed_age() const;
  bool outpost_layer_locked() const;
  bool outpost_static_direct_active() const;
  bool outpost_unlocked_prediction_ready() const;
  void set_outpost_association_debug(
    int best_id, const std::array<double, 3> & scores, double best_score,
    const std::string & reject_reason);
  void set_outpost_layer_correction_debug(
    int raw_id, int height_id, double raw_z_residual, double best_z_residual,
    double z_improvement, int count, bool pending, bool applied);

  bool diverged() const;
  bool convergened();

  bool isinit = false;
  bool checkinit();

private:
  int armor_num_ = 0;
  int switch_count_ = 0;
  int update_count_ = 0;

  bool is_switch_ = false;
  bool is_converged_ = false;

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_{};
  std::optional<OutpostTarget> outpost_target_;

  bool has_last_observed_armor_ = false;
  Eigen::Vector4d last_observed_armor_xyza_{Eigen::Vector4d::Zero()};
  double last_observed_age_s_ = std::numeric_limits<double>::infinity();

  void update_ypda(const Armor & armor, int id);
  void record_observed_armor(const Armor & armor);

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;

  int match_armor_id(
    const Armor & armor, const std::vector<Eigen::Vector4d> & xyza_list) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP
