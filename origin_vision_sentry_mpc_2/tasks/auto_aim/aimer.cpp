#include "aimer.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_aim
{
Aimer::Aimer(const std::string & config_path)
: left_yaw_offset_(std::nullopt),
  right_yaw_offset_(std::nullopt),
  gimbal_axis_order_(tools::GimbalAxisOrder::yaw_pitch)
{
  auto yaml = YAML::LoadFile(config_path);
  if (yaml["gimbal_axis_order"].IsDefined()) {
    gimbal_axis_order_ = tools::parse_gimbal_axis_order(yaml["gimbal_axis_order"].as<std::string>());
  }
  yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;
  pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;
  comming_angle_ = yaml["comming_angle"].as<double>() / 57.3;
  leaving_angle_ = yaml["leaving_angle"].as<double>() / 57.3;
  resistance_k_ = yaml["resistance_k"].as<double>(0.01);
  high_speed_delay_time_ = yaml["high_speed_delay_time"].as<double>();
  low_speed_delay_time_ = yaml["low_speed_delay_time"].as<double>();
  outpost_prediction_offset_s_ = yaml["outpost_prediction_offset_s"].as<double>(0.0);
  if (!std::isfinite(outpost_prediction_offset_s_)) outpost_prediction_offset_s_ = 0.0;
  decision_speed_ = yaml["decision_speed"].as<double>();
  if (yaml["left_yaw_offset"].IsDefined() && yaml["right_yaw_offset"].IsDefined()) {
    left_yaw_offset_ = yaml["left_yaw_offset"].as<double>() / 57.3;
    right_yaw_offset_ = yaml["right_yaw_offset"].as<double>() / 57.3;
    tools::logger()->info("[Aimer] successfully loading shootmode");
  }
}

io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  bool to_now)
{
  return aim_with_yaw_offset(targets, timestamp, bullet_speed, yaw_offset_, to_now);
}

io::Command Aimer::aim_with_yaw_offset(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  double yaw_offset, bool to_now)
{
  if (targets.empty()) return {false, false, 0, 0};
  auto target = targets.front();

  double delay_time =
    std::abs(target.ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;
  if (target.name == ArmorName::outpost) {
    delay_time = std::max(0.0, delay_time + outpost_prediction_offset_s_);
  }

  if (bullet_speed < 14) bullet_speed = 23;

  auto future = timestamp;
  double processing_delay = 0.005;
  if (to_now) {
    processing_delay = std::max(0.0, tools::delta_time(std::chrono::steady_clock::now(), timestamp));
  }
  const double predict_delay = processing_delay + delay_time;
  future += std::chrono::microseconds(static_cast<int64_t>(predict_delay * 1e6));
  target.predict(future);

  auto aim_point0 = choose_aim_point(target);
  debug_aim_point = aim_point0;
  if (!aim_point0.valid) {
    return {false, false, 0, 0};
  }

  const Eigen::Vector3d xyz0 = aim_point0.xyza.head(3);
  const auto d0 = std::sqrt(xyz0[0] * xyz0[0] + xyz0[1] * xyz0[1]);
  tools::Trajectory trajectory0(bullet_speed, d0, xyz0[2], resistance_k_);
  if (trajectory0.unsolvable) {
    tools::logger()->debug(
      "[Aimer] Unsolvable trajectory0: {:.2f} {:.2f} {:.2f}", bullet_speed, d0, xyz0[2]);
    debug_aim_point.valid = false;
    return {false, false, 0, 0};
  }

  if (target.name == ArmorName::outpost && !target.outpost_layer_locked() &&
      !target.outpost_unlocked_prediction_ready()) {
    const auto yaw_pitch = tools::gimbal_command_from_yaw_elevation(
      std::atan2(xyz0.y(), xyz0.x()) + yaw_offset, trajectory0.pitch + pitch_offset_,
      gimbal_axis_order_);
    return {true, false, yaw_pitch[0], yaw_pitch[1]};
  }

  double prev_fly_time = trajectory0.fly_time;
  tools::Trajectory current_traj = trajectory0;
  std::vector<Target> iteration_target(10, target);

  for (int iter = 0; iter < 10; ++iter) {
    const auto predict_time =
      future + std::chrono::microseconds(static_cast<int64_t>(prev_fly_time * 1e6));
    iteration_target[iter].predict(predict_time);

    auto aim_point = choose_aim_point(iteration_target[iter]);
    debug_aim_point = aim_point;
    if (!aim_point.valid) {
      return {false, false, 0, 0};
    }

    const Eigen::Vector3d xyz = aim_point.xyza.head(3);
    const double d = std::sqrt(xyz.x() * xyz.x() + xyz.y() * xyz.y());
    current_traj = tools::Trajectory(bullet_speed, d, xyz.z(), resistance_k_);

    if (current_traj.unsolvable) {
      tools::logger()->debug(
        "[Aimer] Unsolvable trajectory in iter {}: speed={:.2f}, d={:.2f}, z={:.2f}", iter + 1,
        bullet_speed, d, xyz.z());
      debug_aim_point.valid = false;
      return {false, false, 0, 0};
    }

    if (std::abs(current_traj.fly_time - prev_fly_time) < 0.001) {
      break;
    }
    prev_fly_time = current_traj.fly_time;
  }

  const Eigen::Vector3d final_xyz = debug_aim_point.xyza.head(3);
  const auto yaw_pitch = tools::gimbal_command_from_yaw_elevation(
    std::atan2(final_xyz.y(), final_xyz.x()) + yaw_offset, current_traj.pitch + pitch_offset_,
    gimbal_axis_order_);
  return {true, false, yaw_pitch[0], yaw_pitch[1]};
}

io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  io::ShootMode shoot_mode, bool to_now)
{
  double yaw_offset;
  if (shoot_mode == io::left_shoot && left_yaw_offset_.has_value()) {
    yaw_offset = left_yaw_offset_.value();
  } else if (shoot_mode == io::right_shoot && right_yaw_offset_.has_value()) {
    yaw_offset = right_yaw_offset_.value();
  } else {
    yaw_offset = yaw_offset_;
  }

  return aim_with_yaw_offset(targets, timestamp, bullet_speed, yaw_offset, to_now);
}

AimPoint Aimer::choose_aim_point(const Target & target)
{
  const Eigen::VectorXd ekf_x = target.ekf_x();
  const std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
  const auto armor_num = armor_xyza_list.size();

  constexpr int SRC_INVALID = 0;
  constexpr int SRC_SINGLE_FIXED = 1;
  constexpr int SRC_OBSERVED_DIRECT = 2;
  constexpr int SRC_DIRECT_LAST_ID = 3;
  constexpr int SRC_DIRECT_MIN_SWING = 4;
  constexpr int SRC_DIRECT_OBS_MATCH = 5;
  constexpr int SRC_INDIRECT = 6;
  constexpr int SRC_DIRECT_LOCKED = 7;

  auto make_point = [&](bool valid, const Eigen::Vector4d & xyza, int armor_id, int source) {
    AimPoint point;
    point.valid = valid;
    point.xyza = xyza;
    point.armor_id = armor_id;
    point.source = source;
    return point;
  };

  if (armor_num == 0) {
    lock_id_ = -1;
    return make_point(false, Eigen::Vector4d::Zero(), -1, SRC_INVALID);
  }
  if (target.name == ArmorName::outpost && !target.outpost_layer_locked()) {
    lock_id_ = -1;
    const double max_age = target.outpost_unlocked_prediction_ready() ? 0.35 : 0.14;
    if (!target.has_last_observed_armor() || target.last_observed_age() > max_age) {
      return make_point(false, Eigen::Vector4d::Zero(), -1, SRC_INVALID);
    }
    return make_point(true, armor_xyza_list[0], 0, SRC_OBSERVED_DIRECT);
  }
  if (armor_num == 1) {
    lock_id_ = -1;
    return make_point(true, armor_xyza_list[0], 0, SRC_SINGLE_FIXED);
  }

  if (target.name != ArmorName::outpost && !target.jumped) {
    lock_id_ = -1;
    return make_point(true, armor_xyza_list[0], 0, SRC_SINGLE_FIXED);
  }

  const auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);
  const auto vyaw = ekf_x[7];

  auto choose_min_swing_id = [&](const std::vector<int> & ids) -> int {
    int best_id = ids.front();
    double best_cost = std::numeric_limits<double>::infinity();
    for (int id : ids) {
      const double cost = std::abs(tools::limit_rad(armor_xyza_list[id][3] - center_yaw));
      if (cost < best_cost) {
        best_cost = cost;
        best_id = id;
      }
    }
    return best_id;
  };

  std::vector<double> delta_angle_list;
  for (size_t i = 0; i < armor_num; ++i) {
    delta_angle_list.emplace_back(tools::limit_rad(armor_xyza_list[i][3] - center_yaw));
  }

  if (target.name != ArmorName::outpost) {
    std::vector<int> id_list;
    for (size_t i = 0; i < armor_num; ++i) {
      if (std::abs(delta_angle_list[i]) > 60.0 / 57.3) continue;
      id_list.push_back(static_cast<int>(i));
    }

    if (id_list.empty()) {
      lock_id_ = -1;
      return make_point(false, armor_xyza_list[0], -1, SRC_INVALID);
    }

    if (id_list.size() > 1) {
      const int id0 = id_list[0];
      const int id1 = id_list[1];
      if (lock_id_ != id0 && lock_id_ != id1) {
        lock_id_ = (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1])) ? id0 : id1;
      }
      return make_point(true, armor_xyza_list[lock_id_], lock_id_, SRC_DIRECT_LOCKED);
    }

    lock_id_ = -1;
    return make_point(true, armor_xyza_list[id_list[0]], id_list[0], SRC_DIRECT_MIN_SWING);
  }

  auto contains_id = [&](const std::vector<int> & ids, int id) -> bool {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
  };
  auto closest_id_to_obs = [&](const Eigen::Vector4d & obs_xyza) -> int {
    int best_id = -1;
    double best_dist = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < armor_num; ++i) {
      const double dist = (armor_xyza_list[i].head(3) - obs_xyza.head(3)).norm();
      if (dist < best_dist) {
        best_dist = dist;
        best_id = static_cast<int>(i);
      }
    }
    return best_id;
  };

  double coming_angle = comming_angle_;
  double leaving_angle = leaving_angle_;
  if (target.name == ArmorName::outpost) {
    coming_angle = 70.0 / 57.3;
    leaving_angle = 30.0 / 57.3;
  }

  const bool obs_fresh =
    target.has_last_observed_armor() && target.last_observed_age() <= 0.14;
  const Eigen::Vector4d obs_xyza =
    obs_fresh ? target.last_observed_armor_xyza() : Eigen::Vector4d::Zero();
  const int obs_match_id = obs_fresh ? closest_id_to_obs(obs_xyza) : -1;
  const auto abs_vyaw = std::abs(vyaw);

  std::vector<int> direct_ids;
  for (size_t i = 0; i < armor_num; ++i) {
    if (std::abs(delta_angle_list[i]) > coming_angle) continue;
    if (vyaw > 0 && delta_angle_list[i] < leaving_angle) direct_ids.push_back(static_cast<int>(i));
    if (vyaw < 0 && delta_angle_list[i] > -leaving_angle) direct_ids.push_back(static_cast<int>(i));
    if (abs_vyaw < 1e-3) direct_ids.push_back(static_cast<int>(i));
  }

  if (!direct_ids.empty()) {
    if (
      lock_id_ >= 0 && lock_id_ < static_cast<int>(armor_num) && contains_id(direct_ids, lock_id_) &&
      std::abs(delta_angle_list[lock_id_]) < coming_angle - 5.0 / 57.3)
    {
      return make_point(true, armor_xyza_list[lock_id_], lock_id_, SRC_DIRECT_LOCKED);
    }

    int chosen = -1;
    int source = SRC_DIRECT_MIN_SWING;
    if (abs_vyaw < 1.2 && obs_match_id >= 0 && contains_id(direct_ids, obs_match_id)) {
      chosen = obs_match_id;
      source = SRC_DIRECT_OBS_MATCH;
    } else if (
      abs_vyaw < 1.2 && target.last_id >= 0 && target.last_id < static_cast<int>(armor_num) &&
      contains_id(direct_ids, target.last_id))
    {
      chosen = target.last_id;
      source = SRC_DIRECT_LAST_ID;
    } else {
      chosen = choose_min_swing_id(direct_ids);
    }

    lock_id_ = chosen;
    return make_point(true, armor_xyza_list[chosen], chosen, source);
  }

  if (abs_vyaw < 1e-3) {
    lock_id_ = -1;
    return make_point(false, armor_xyza_list[0], -1, SRC_INVALID);
  }

  const double wait_angle = (vyaw > 0) ? -coming_angle : coming_angle;
  const double max_out_angle = leaving_angle;

  int indirect_id = -1;
  double min_armor_to_wait = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < armor_num; ++i) {
    const double delta = delta_angle_list[i];
    const double armor_to_wait =
      tools::limit_rad(((vyaw > 0) ? (wait_angle - delta) : (delta - wait_angle)) - CV_PI +
                       max_out_angle) +
      CV_PI - max_out_angle;
    if (armor_to_wait < min_armor_to_wait) {
      min_armor_to_wait = armor_to_wait;
      indirect_id = static_cast<int>(i);
    }
  }

  if (indirect_id < 0) {
    lock_id_ = -1;
    return make_point(false, armor_xyza_list[0], -1, SRC_INVALID);
  }

  double emerge_time = min_armor_to_wait / (abs_vyaw + 1e-3);
  emerge_time = std::clamp(emerge_time, 0.0, 0.35);

  Target future_target = target;
  future_target.predict(emerge_time);
  const auto future_xyza_list = future_target.armor_xyza_list();
  if (indirect_id >= static_cast<int>(future_xyza_list.size())) {
    lock_id_ = -1;
    return make_point(false, armor_xyza_list[0], -1, SRC_INVALID);
  }

  lock_id_ = indirect_id;
  return make_point(true, future_xyza_list[indirect_id], indirect_id, SRC_INDIRECT);
}

}  // namespace auto_aim
