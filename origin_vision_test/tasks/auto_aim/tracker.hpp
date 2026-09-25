#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <optional>
#include <string>
#include <map>
#include <vector>

#include "armor.hpp"
#include "solver.hpp"
#include "target.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{
struct OmniSwitchConstraint
{
  bool enabled = false;
  ArmorName armor_name = ArmorName::not_armor;
  ArmorPriority priority = ArmorPriority::fifth;
  double abs_yaw_rad = 0.0;
  bool has_abs_yaw = false;
  double match_deg = 0.0;
};

std::optional<double> omni_switch_match_delta_deg(
  ArmorName armor_name, ArmorPriority priority, const Eigen::Vector3d & xyz_in_world,
  const OmniSwitchConstraint & constraint);

class Tracker
{
public:
  Tracker(const std::string & config_path, Solver & solver);

  std::string state() const;

  std::list<Target> track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
    bool use_enemy_color = true);

  std::tuple<omniperception::DetectionResult, std::list<Target>> track(
    const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
    std::chrono::steady_clock::time_point t, bool use_enemy_color = true,
    const std::optional<OmniSwitchConstraint> & switch_constraint = std::nullopt);

private:
  Solver & solver_;
  Color enemy_color_;
  int min_detect_count_;
  int max_temp_lost_count_;
  int detect_count_;
  int temp_lost_count_;
  int outpost_max_temp_lost_count_;
  int normal_temp_lost_count_;
  std::string state_, pre_state_;
  Target target_;
  std::chrono::steady_clock::time_point last_timestamp_;
  ArmorPriority omni_target_priority_;
  std::map<ArmorName, ArmorPriority> armor_priority_;

  struct OutpostLayerCorrectionDecision
  {
    int layer_id = -1;
    bool defer_update = false;
    bool corrected = false;
  };

  bool outpost_layer_correction_enabled_;
  int outpost_layer_correction_frames_;
  int outpost_layer_switch_frames_;
  double outpost_layer_correction_z_gate_;
  bool outpost_static_direct_enabled_;
  double outpost_static_speed_threshold_;
  int outpost_static_motion_confirm_frames_;
  int outpost_layer_correction_candidate_ = -1;
  int outpost_layer_correction_count_ = 0;
  int outpost_layer_switch_candidate_ = -1;
  int outpost_layer_switch_count_ = 0;

  void apply_priority(std::list<Armor> & armors) const;

  void sort_armors(std::list<Armor> & armors) const;

  void state_machine(bool found);

  void handle_large_dt(double dt);

  bool set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  bool update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  OutpostLayerCorrectionDecision decide_outpost_layer_correction(
    const Armor & armor, int raw_id,
    const std::vector<Eigen::Vector4d> & predicted_armors);
  void reset_outpost_layer_correction();
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRACKER_HPP
