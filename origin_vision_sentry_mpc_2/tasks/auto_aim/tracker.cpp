#include "tracker.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
constexpr double NORMAL_ASSOCIATION_MAX_SCORE = 40.0;

double normalized_square(double value, double gate)
{
  const double safe_gate = gate <= 1e-6 ? 1e-6 : gate;
  const double normalized = value / safe_gate;
  return normalized * normalized;
}

std::string normalize_priority_key(std::string key)
{
  std::string normalized;
  normalized.reserve(key.size());
  for (unsigned char ch : key) {
    if (ch == '_' || ch == '-' || ch == ' ') continue;
    normalized.push_back(static_cast<char>(std::tolower(ch)));
  }
  return normalized;
}

std::optional<ArmorName> armor_name_from_priority_key(const std::string & raw_key)
{
  const auto key = normalize_priority_key(raw_key);

  if (key == "1" || key == "one" || key == "hero" || raw_key == "英雄") {
    return ArmorName::one;
  }
  if (key == "2" || key == "two" || key == "engineer" || raw_key == "工程") {
    return ArmorName::two;
  }
  if (key == "3" || key == "three" || key == "infantry3" || raw_key == "步兵3") {
    return ArmorName::three;
  }
  if (key == "4" || key == "four" || key == "infantry4" || raw_key == "步兵4") {
    return ArmorName::four;
  }
  if (key == "5" || key == "five" || key == "infantry5" || raw_key == "步兵5") {
    return ArmorName::five;
  }
  if (key == "sentry" || raw_key == "哨兵") {
    return ArmorName::sentry;
  }
  if (key == "outpost" || raw_key == "前哨站") {
    return ArmorName::outpost;
  }
  if (key == "base" || raw_key == "基地") {
    return ArmorName::base;
  }
  if (key == "notarmor" || raw_key == "非装甲") {
    return ArmorName::not_armor;
  }

  return std::nullopt;
}

ArmorPriority armor_priority_from_yaml_value(const YAML::Node & node)
{
  const int priority = std::max(1, node.as<int>());
  return static_cast<ArmorPriority>(priority);
}
}  // namespace

std::optional<double> omni_switch_match_delta_deg(
  ArmorName armor_name, ArmorPriority priority, const Eigen::Vector3d & xyz_in_world,
  const OmniSwitchConstraint & constraint)
{
  if (!constraint.enabled || !constraint.has_abs_yaw) return std::nullopt;
  if (armor_name != constraint.armor_name || priority != constraint.priority) return std::nullopt;
  const double armor_abs_yaw = std::atan2(xyz_in_world[1], xyz_in_world[0]);
  return std::abs(tools::limit_rad(armor_abs_yaw - constraint.abs_yaw_rad)) * 57.3;
}

Tracker::Tracker(const std::string & config_path, Solver & solver)
: solver_{solver},
  detect_count_(0),
  temp_lost_count_(0),
  state_{"lost"},
  pre_state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now()),
  omni_target_priority_{ArmorPriority::fifth},
  armor_priority_{
    {ArmorName::one, ArmorPriority::fifth},     {ArmorName::two, ArmorPriority::fifth},
    {ArmorName::three, ArmorPriority::fifth},   {ArmorName::four, ArmorPriority::fifth},
    {ArmorName::five, ArmorPriority::fifth},    {ArmorName::sentry, ArmorPriority::fifth},
    {ArmorName::outpost, ArmorPriority::fifth}, {ArmorName::base, ArmorPriority::fifth},
    {ArmorName::not_armor, ArmorPriority::fifth}}
{
  auto yaml = YAML::LoadFile(config_path);
  enemy_color_ = (yaml["enemy_color"].as<std::string>() == "red") ? Color::red : Color::blue;
  min_detect_count_ = yaml["min_detect_count"].as<int>();
  max_temp_lost_count_ = yaml["max_temp_lost_count"].as<int>();
  outpost_max_temp_lost_count_ = yaml["outpost_max_temp_lost_count"].as<int>();
  normal_temp_lost_count_ = max_temp_lost_count_;

  YAML::Node priority_yaml;
  bool has_priority_yaml = false;
  if (yaml["armor_priority"].IsDefined()) {
    priority_yaml = yaml["armor_priority"];
    has_priority_yaml = true;
  } else if (yaml["target_priority"].IsDefined()) {
    priority_yaml = yaml["target_priority"];
    has_priority_yaml = true;
  } else if (yaml["fire_priority"].IsDefined()) {
    priority_yaml = yaml["fire_priority"];
    has_priority_yaml = true;
  }

  if (has_priority_yaml) {
    if (!priority_yaml.IsMap()) {
      tools::logger()->warn("[Tracker] armor_priority should be a YAML map.");
    } else {
      for (const auto & item : priority_yaml) {
        const auto key = item.first.as<std::string>();
        const auto armor_name = armor_name_from_priority_key(key);
        if (!armor_name.has_value()) {
          tools::logger()->warn("[Tracker] Unknown armor priority key: {}", key);
          continue;
        }
        armor_priority_[armor_name.value()] = armor_priority_from_yaml_value(item.second);
      }
    }
  }
}

std::string Tracker::state() const { return state_; }

std::list<Target> Tracker::track(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  handle_large_dt(dt);
  // 过滤掉非我方装甲板
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });

  // 过滤前哨站顶部装甲板
  // armors.remove_if([this](const auto_aim::Armor & a) {
  //   return a.name == ArmorName::outpost &&
  //          solver_.oupost_reprojection_error(a, 27.5 * CV_PI / 180.0) <
  //            solver_.oupost_reprojection_error(a, -15 * CV_PI / 180.0);
  // });

  sort_armors(armors);

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  }

  // 此时主相机画面中出现了优先级更高的装甲板，切换目标
  else if (state_ == "tracking" && !armors.empty() && armors.front().priority < target_.priority) {
    found = set_target(armors, t);
    tools::logger()->debug("auto_aim switch target to {}", ARMOR_NAMES[armors.front().name]);
  }

  else {
    found = update_target(armors, t);
  }

  state_machine(found);

  // 发散检测
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {};
  }

  // 收敛效果检测：
  if (
    std::accumulate(
      target_.ekf().recent_nis_failures.begin(), target_.ekf().recent_nis_failures.end(), 0) >=
    (0.4 * target_.ekf().window_size)) {
    tools::logger()->debug("[Target] Bad Converge Found!");
    state_ = "lost";
    return {};
  }

  if (state_ == "lost") return {};

  std::list<Target> targets = {target_};
  return targets;
}

std::tuple<omniperception::DetectionResult, std::list<Target>> Tracker::track(
  const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
  std::chrono::steady_clock::time_point t, bool use_enemy_color,
  const std::optional<OmniSwitchConstraint> & switch_constraint)
{
  omniperception::DetectionResult switch_target;
  switch_target.timestamp = t;
  omniperception::DetectionResult temp_target;
  temp_target.timestamp = t;
  if (!detection_queue.empty()) {
    temp_target = detection_queue.front();
    sort_armors(temp_target.armors);
  }

  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  handle_large_dt(dt);

  sort_armors(armors);

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  }

  // 此时主相机画面中出现了优先级更高的装甲板，切换目标
  else if (state_ == "tracking" && !armors.empty() && armors.front().priority < target_.priority) {
    found = set_target(armors, t);
    tools::logger()->debug("auto_aim switch target to {}", ARMOR_NAMES[armors.front().name]);
  }

  // 此时全向感知相机画面中出现了优先级更高的装甲板，切换目标
  else if (
    state_ == "tracking" && !temp_target.armors.empty() &&
    temp_target.armors.front().priority < target_.priority && target_.convergened()) {
    state_ = "switching";
    switch_target = temp_target;
    switch_target.timestamp = t;
    omni_target_priority_ = temp_target.armors.front().priority;
    found = false;
    tools::logger()->debug("omniperception find higher priority target");
  }

  else if (state_ == "switching") {
    if (switch_constraint.has_value() && switch_constraint->enabled && switch_constraint->has_abs_yaw) {
      found = false;
      for (auto armor : armors) {
        if (
          armor.name != switch_constraint->armor_name ||
          armor.priority != switch_constraint->priority) {
          continue;
        }
        solver_.solve(armor);
        const auto delta_deg = omni_switch_match_delta_deg(
          armor.name, armor.priority, armor.xyz_in_world, switch_constraint.value());
        if (delta_deg.has_value() && delta_deg.value() <= switch_constraint->match_deg) {
          found = true;
          break;
        }
      }
    } else {
      found = !armors.empty() && armors.front().priority == omni_target_priority_;
    }
  }

  else if (state_ == "detecting" && pre_state_ == "switching") {
    found = set_target(armors, t);
  }

  else {
    found = update_target(armors, t);
  }

  pre_state_ = state_;
  // 更新状态机
  state_machine(found);

  // 发散检测
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {switch_target, {}};  // 返回switch_target和空的targets
  }

  if (state_ == "lost") return {switch_target, {}};  // 返回switch_target和空的targets

  std::list<Target> targets = {target_};
  return {switch_target, targets};
}

void Tracker::apply_priority(std::list<Armor> & armors) const
{
  for (auto & armor : armors) {
    auto priority_iter = armor_priority_.find(armor.name);
    armor.priority =
      priority_iter == armor_priority_.end() ? ArmorPriority::fifth : priority_iter->second;
  }
}

void Tracker::sort_armors(std::list<Armor> & armors) const
{
  apply_priority(armors);

  // 先按画面中心距离排序；std::list::sort 是稳定排序，后面同优先级时会保留这个顺序。
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO: 可改成从相机配置读取
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 按开火优先级排序：数字越小，优先级越高；1 最高。
  armors.sort([](const Armor & a, const Armor & b) { return a.priority < b.priority; });
}

void Tracker::state_machine(bool found)
{
  if (state_ == "lost") {
    if (!found) return;

    state_ = "detecting";
    detect_count_ = 1;
  }

  else if (state_ == "detecting") {
    if (found) {
      detect_count_++;
      if (detect_count_ >= min_detect_count_) state_ = "tracking";
    } else {
      detect_count_ = 0;
      state_ = "lost";
    }
  }

  else if (state_ == "tracking") {
    if (found) return;

    temp_lost_count_ = 1;
    state_ = "temp_lost";
  }

  else if (state_ == "switching") {
    if (found) {
      state_ = "detecting";
    } else {
      temp_lost_count_++;
      if (temp_lost_count_ > 200) state_ = "lost";
    }
  }

  else if (state_ == "temp_lost") {
    if (found) {
      state_ = "tracking";
    } else {
      temp_lost_count_++;
      if (target_.name == ArmorName::outpost)
        //前哨站的temp_lost_count需要设置的大一些
        max_temp_lost_count_ = outpost_max_temp_lost_count_;
      else
        max_temp_lost_count_ = normal_temp_lost_count_;

      if (temp_lost_count_ > max_temp_lost_count_) state_ = "lost";
    }
  }
}

void Tracker::handle_large_dt(double dt)
{
  // 时间间隔过长，说明可能发生了相机离线。前哨站模型已经带有更长的 temp_lost
  // 保持窗口，单次时间戳跳变不应清空层级语义；后续由关联和状态机决定是否丢失。
  if (state_ == "lost" || dt <= 0.1) return;

  tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
  if (target_.name == ArmorName::outpost) return;

  state_ = "lost";
}

bool Tracker::set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  if (armors.empty()) return false;

  auto & armor = armors.front();
  solver_.solve(armor);

  if (armor.name == ArmorName::outpost) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 25, 81, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.2765, 3, P0_dig);
  } else if (armor.name == ArmorName::base) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.3205, 3, P0_dig);
  } else {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 4, P0_dig);
  }

  return true;
}

bool Tracker::update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  target_.predict(t);
  target_.clear_timing_debug_pair();

  if (target_.name != ArmorName::outpost) {
    // A frame can contain more than one armor with the same number and type. Updating the
    // filter with every one of them mixes different robots into one target, which is especially
    // visible once the gimbal moves and produces a large jump in the world-frame yaw command.
    // Associate one observation against the predicted armor set before updating the EKF.
    const std::vector<Eigen::Vector4d> predicted_armors = target_.armor_xyza_list();
    Armor * best_armor = nullptr;
    int best_id = -1;
    double best_score = std::numeric_limits<double>::infinity();

    for (auto & armor : armors) {
      if (armor.name != target_.name || armor.type != target_.armor_type) continue;
      solver_.solve(armor);

      for (int id = 0; id < static_cast<int>(predicted_armors.size()); ++id) {
        const Eigen::Vector4d & xyza = predicted_armors[id];
        const Eigen::Vector3d predicted_ypd = tools::xyz2ypd(xyza.head(3));
        const double yaw_err = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3]));
        const double bearing_err =
          std::abs(tools::limit_rad(armor.ypd_in_world[0] - predicted_ypd[0]));
        const double pitch_err =
          std::abs(tools::limit_rad(armor.ypd_in_world[1] - predicted_ypd[1]));
        const double dist_err = std::abs(armor.ypd_in_world[2] - predicted_ypd[2]);
        const double z_err = std::abs(armor.xyz_in_world[2] - xyza[2]);

        double score =
          1.0 * normalized_square(yaw_err, 0.45) +
          0.7 * normalized_square(bearing_err, 0.40) +
          0.4 * normalized_square(pitch_err, 0.30) +
          0.25 * normalized_square(dist_err, 0.60) +
          0.8 * normalized_square(z_err, 0.08);
        if (id == target_.last_id) score *= 0.95;

        if (score < best_score) {
          best_score = score;
          best_armor = &armor;
          best_id = id;
        }
      }
    }

    if (best_armor == nullptr || best_score > NORMAL_ASSOCIATION_MAX_SCORE) {
      tools::logger()->debug(
        "[Tracker] normal armor association rejected: candidates={}, score={:.2f}",
        armors.size(), best_score);
      return false;
    }

    target_.set_timing_debug_pair(*best_armor, predicted_armors[best_id]);
    target_.update(*best_armor, best_id);
    return true;
  }

  std::vector<Armor *> candidates;
  candidates.reserve(armors.size());
  for (auto & armor : armors) {
    if (armor.name != target_.name || armor.type != target_.armor_type) continue;
    candidates.push_back(&armor);
  }
  if (candidates.empty()) return false;

  const std::vector<Eigen::Vector4d> predicted_armors = target_.armor_xyza_list();
  const int predicted_count = static_cast<int>(predicted_armors.size());
  auto normalized_square = [](double value, double gate) {
    const double safe_gate = gate <= 1e-6 ? 1e-6 : gate;
    const double normalized = value / safe_gate;
    return normalized * normalized;
  };

  Armor * best_armor = nullptr;
  double best_score = std::numeric_limits<double>::infinity();
  int best_id = -1;
  std::array<double, 3> best_scores{
    std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity()};

  for (auto * armor_ptr : candidates) {
    auto & armor = *armor_ptr;
    solver_.solve(armor);

    for (int id = 0; id < std::max(1, predicted_count); ++id) {
      double score = 0.0;
      if (predicted_count > 0) {
        const Eigen::Vector4d & xyza = predicted_armors[id];
        const Eigen::Vector3d predicted_ypd = tools::xyz2ypd(xyza.head(3));
        const double yaw_err = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3]));
        const double bearing_err =
          std::abs(tools::limit_rad(armor.ypd_in_world[0] - predicted_ypd[0]));
        const double pitch_err =
          std::abs(tools::limit_rad(armor.ypd_in_world[1] - predicted_ypd[1]));
        const double dist_err = std::abs(armor.ypd_in_world[2] - predicted_ypd[2]);
        const double z_err = std::abs(armor.xyz_in_world[2] - xyza[2]);
        // Fixed-armor PnP height error rises with range. Preserve the established near-range
        // 8cm gate, then relax only the outpost association at longer range.
        const double z_gate = std::clamp(0.035 + 0.008 * armor.ypd_in_world[2], 0.08, 0.18);

        score =
          1.0 * normalized_square(yaw_err, 0.45) +
          0.7 * normalized_square(bearing_err, 0.40) +
          0.4 * normalized_square(pitch_err, 0.30) +
          0.25 * normalized_square(dist_err, 0.60) +
          0.8 * normalized_square(z_err, z_gate);

        if (id == target_.last_id) score *= 0.95;
      }

      if (score < best_score) {
        best_score = score;
        best_armor = &armor;
        best_id = id;
      }
      if (id >= 0 && id < static_cast<int>(best_scores.size())) {
        best_scores[id] = std::min(best_scores[id], score);
      }
    }
  }

  if (best_armor == nullptr) return false;

  const double reject_score = target_.outpost_layer_locked() ? 28.0 : 80.0;
  if (best_score > reject_score) {
    target_.set_outpost_association_debug(best_id, best_scores, best_score, "score_gate");
    return false;
  }

  target_.set_outpost_association_debug(best_id, best_scores, best_score, "");
  target_.update(*best_armor, best_id);
  return true;
}

}  // namespace auto_aim
