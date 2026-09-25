#ifndef AUTO_AIM__OUTPOST_TARGET_HPP
#define AUTO_AIM__OUTPOST_TARGET_HPP

#include <Eigen/Dense>

#include <array>
#include <chrono>
#include <cstddef>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

constexpr double OUTPOST_LAYER_SPACING = 0.1;

// Behavior-preserving extraction of the original three-plate outpost model.
// The numerical model and update cadence intentionally match the original
// Target implementation: one selected observation updates the EKF once/frame.
class OutpostTarget
{
public:
  OutpostTarget() = default;
  OutpostTarget(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius,
    const Eigen::VectorXd & P0_dig, bool static_direct_enabled = false,
    double static_speed_threshold = 0.15, int static_motion_confirm_frames = 3);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  void update(const Armor & armor, std::optional<int> forced_id = std::nullopt);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  bool has_last_observed_armor() const;
  Eigen::Vector4d last_observed_armor_xyza() const;
  double last_observed_age() const;
  bool layer_locked() const;
  bool unlocked_prediction_ready() const;
  bool static_direct_active() const;

  void set_association_debug(
    int best_id, const std::array<double, 3> & scores, double best_score,
    const std::string & reject_reason);
  void set_layer_correction_debug(
    int raw_id, int height_id, double raw_z_residual, double best_z_residual,
    double z_improvement, int count, bool pending, bool applied);

  bool diverged() const;
  bool converged();
  int last_id() const;
  bool jumped() const;

private:
  static constexpr int ARMOR_NUM = 3;
  static constexpr std::size_t INIT_CACHE_LIMIT = 12;

  struct Observation
  {
    std::chrono::steady_clock::time_point t;
    Eigen::Vector3d xyz;
    Eigen::Vector3d center;
    double yaw = 0.0;
  };

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_{};

  int update_count_ = 0;
  int switch_count_ = 0;
  int last_id_ = -1;
  int last_layer_ = 0;
  bool switched_ = false;
  bool converged_ = false;
  bool jumped_ = false;

  bool has_last_observed_armor_ = false;
  Eigen::Vector4d last_observed_armor_xyza_{Eigen::Vector4d::Zero()};
  double last_observed_age_s_ = std::numeric_limits<double>::infinity();

  bool layer_locked_ = false;
  bool static_direct_enabled_ = false;
  double static_speed_threshold_ = 0.15;
  int static_motion_confirm_frames_ = 3;
  bool static_direct_active_ = false;
  int static_mode_candidate_ = -1;
  int static_mode_count_ = 0;
  std::deque<Observation> static_motion_observations_;
  std::deque<Observation> init_observations_;
  bool preview_ready_ = false;
  int preview_layer_ = -1;
  double preview_omega_ = 0.0;
  double preview_theta_ = 0.0;
  double preview_base_z_ = 0.0;
  double preview_base_vz_ = 0.0;
  Eigen::Vector3d preview_center_{Eigen::Vector3d::Zero()};
  Eigen::Vector2d preview_center_vxy_{Eigen::Vector2d::Zero()};
  std::chrono::steady_clock::time_point preview_t_{};

  void update_ypda(const Armor & armor, int id);
  void record_observed_armor(const Armor & armor);
  Eigen::Vector3d center_from_armor(const Armor & armor) const;
  Eigen::Vector4d predicted_unlocked_xyza() const;
  void observe_unlocked(const Armor & armor);
  void update_static_direct_state(const Armor & armor);
  bool try_lock_layers();

  Eigen::Vector3d armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd measurement_jacobian(const Eigen::VectorXd & x, int id) const;
  int match_id(
    const Armor & armor, const std::vector<Eigen::Vector4d> & xyza_list,
    int measured_layer, int previous_layer) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OUTPOST_TARGET_HPP
