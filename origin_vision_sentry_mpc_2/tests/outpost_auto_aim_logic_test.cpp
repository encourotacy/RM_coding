#include <chrono>
#include <array>
#include <cmath>
#include <iostream>
#include <list>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <yaml-cpp/yaml.h>

#define private public
#include "tasks/auto_aim/target.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/aimer.hpp"
#undef private

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace
{
constexpr double kFixedArmorWidth = 129e-3;
constexpr double kLightbarLength = 56e-3;

bool expect(bool condition, const std::string & message)
{
  if (condition) return true;
  std::cerr << message << std::endl;
  return false;
}

bool expect_near(double actual, double expected, double eps, const std::string & message)
{
  if (std::abs(actual - expected) <= eps) return true;
  std::cerr << message << ", actual=" << actual << ", expected=" << expected << std::endl;
  return false;
}

std::vector<cv::Point3f> fixed_armor_points()
{
  return {
    {0.0f, static_cast<float>(kFixedArmorWidth / 2), static_cast<float>(kLightbarLength / 2)},
    {0.0f, static_cast<float>(-kFixedArmorWidth / 2), static_cast<float>(kLightbarLength / 2)},
    {0.0f, static_cast<float>(-kFixedArmorWidth / 2), static_cast<float>(-kLightbarLength / 2)},
    {0.0f, static_cast<float>(kFixedArmorWidth / 2), static_cast<float>(-kLightbarLength / 2)},
  };
}

int num_id_for(auto_aim::ArmorName name)
{
  switch (name) {
    case auto_aim::ArmorName::one:
      return 1;
    case auto_aim::ArmorName::two:
      return 2;
    case auto_aim::ArmorName::three:
      return 3;
    case auto_aim::ArmorName::four:
      return 4;
    case auto_aim::ArmorName::five:
      return 5;
    case auto_aim::ArmorName::outpost:
      return 6;
    case auto_aim::ArmorName::base:
      return 7;
    default:
      return 6;
  }
}

auto_aim::Armor make_world_armor(
  auto_aim::ArmorName name, auto_aim::ArmorType type, const Eigen::Vector3d & xyz, double yaw)
{
  const std::vector<cv::Point2f> dummy_points{{0.0f, 0.0f}, {10.0f, 0.0f}, {10.0f, 10.0f}, {0.0f, 10.0f}};
  auto_aim::Armor armor(
    0, num_id_for(name), 1.0f, cv::Rect(0, 0, 10, 10), dummy_points);
  armor.name = name;
  armor.type = type;
  armor.color = auto_aim::Color::blue;
  armor.priority = auto_aim::ArmorPriority::first;
  armor.confidence = 1.0;
  armor.center = {720.0f, 540.0f};
  armor.center_norm = {0.5f, 0.5f};
  armor.xyz_in_world = xyz;
  armor.ypr_in_world = {yaw, 0.0, 0.0};
  armor.ypd_in_world = tools::xyz2ypd(xyz);
  return armor;
}

auto_aim::Armor make_outpost_layer_armor(
  const Eigen::Vector3d & center_base, double theta, int layer)
{
  const double radius = 0.2765;
  const double angle = tools::limit_rad(theta - static_cast<double>(layer) * 2 * CV_PI / 3);
  const Eigen::Vector3d xyz{
    center_base[0] - radius * std::cos(angle),
    center_base[1] - radius * std::sin(angle),
    center_base[2] + 0.1 * static_cast<double>(layer)};
  return make_world_armor(
    auto_aim::ArmorName::outpost, auto_aim::ArmorType::base_outpost, xyz, angle);
}

std::vector<cv::Point2f> project_fixed_armor(
  const std::string & config_path, const Eigen::Vector3d & xyz_in_world, double yaw,
  auto_aim::ArmorName name)
{
  const auto yaml = YAML::LoadFile(config_path);
  const auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  const auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  const auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  const auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();

  const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_camera2gimbal(R_camera2gimbal_data.data());
  const Eigen::Vector3d t_camera2gimbal(t_camera2gimbal_data.data());
  const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  const Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());

  cv::Mat camera_matrix_cv;
  cv::Mat distort_coeffs_cv;
  cv::eigen2cv(camera_matrix, camera_matrix_cv);
  cv::eigen2cv(distort_coeffs, distort_coeffs_cv);

  const auto sin_yaw = std::sin(yaw);
  const auto cos_yaw = std::cos(yaw);
  const auto pitch = (name == auto_aim::ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
  const auto sin_pitch = std::sin(pitch);
  const auto cos_pitch = std::cos(pitch);

  // clang-format off
  const Eigen::Matrix3d R_armor2world{
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };
  // clang-format on

  const Eigen::Matrix3d R_armor2camera = R_camera2gimbal.transpose() * R_armor2world;
  const Eigen::Vector3d t_armor2camera =
    R_camera2gimbal.transpose() * (xyz_in_world - t_camera2gimbal);

  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(R_armor2camera, R_armor2camera_cv);
  cv::Vec3d rvec;
  cv::Rodrigues(R_armor2camera_cv, rvec);
  cv::Vec3d tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  std::vector<cv::Point2f> image_points;
  cv::projectPoints(
    fixed_armor_points(), rvec, tvec, camera_matrix_cv, distort_coeffs_cv, image_points);
  return image_points;
}

auto_aim::Armor make_detection_outpost_armor(
  const std::string & config_path, const Eigen::Vector3d & xyz, double yaw)
{
  auto armor =
    make_world_armor(auto_aim::ArmorName::outpost, auto_aim::ArmorType::base_outpost, xyz, yaw);
  armor.points = project_fixed_armor(config_path, xyz, yaw, auto_aim::ArmorName::outpost);
  armor.center = (armor.points[0] + armor.points[1] + armor.points[2] + armor.points[3]) * 0.25f;
  return armor;
}

}  // namespace

int main()
{
  const std::string config_path = "configs/sentry.yaml";
  const auto now = std::chrono::steady_clock::now();

  {
    const tools::Trajectory no_resistance(27.0, 3.0, 0.0, 0.0);
    const tools::Trajectory with_resistance(27.0, 3.0, 0.0, 0.023);
    if (!expect(!no_resistance.unsolvable, "no-resistance trajectory should be solvable")) return 1;
    if (!expect(!with_resistance.unsolvable, "resistance trajectory should be solvable")) return 1;
    if (!expect(
          with_resistance.fly_time > no_resistance.fly_time,
          "air resistance should increase fly time")) {
      return 1;
    }
    if (!expect(
          with_resistance.pitch > no_resistance.pitch,
          "air resistance should require a larger pitch")) {
      return 1;
    }
  }

  {
    auto_aim::Solver solver(config_path);
    solver.set_R_gimbal2world(Eigen::Quaterniond::Identity());

    const Eigen::Vector3d expected_xyz(3.0, 0.2, 0.8);
    const double expected_yaw = 0.15;
    auto image_points =
      project_fixed_armor(config_path, expected_xyz, expected_yaw, auto_aim::ArmorName::outpost);
    auto armor =
      make_world_armor(auto_aim::ArmorName::outpost, auto_aim::ArmorType::base_outpost, expected_xyz, expected_yaw);
    armor.points = image_points;
    armor.center =
      (image_points[0] + image_points[1] + image_points[2] + image_points[3]) * 0.25f;

    solver.solve(armor);

    if (!expect_near(armor.xyz_in_world[0], expected_xyz[0], 0.03, "solver x mismatch")) return 1;
    if (!expect_near(armor.xyz_in_world[1], expected_xyz[1], 0.03, "solver y mismatch")) return 1;
    if (!expect_near(armor.xyz_in_world[2], expected_xyz[2], 0.03, "solver z mismatch")) return 1;
    if (!expect_near(
          tools::limit_rad(armor.ypr_in_world[0] - expected_yaw), 0.0, 0.03,
          "solver yaw mismatch")) {
      return 1;
    }
  }

  {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 25, 81, 0.4, 100, 1e-4, 0, 0}};
    const Eigen::Vector3d center_base(2.20, 0.10, 0.70);
    const double theta = 0.40;

    for (int layer = 0; layer < 3; ++layer) {
      auto first_observed = make_outpost_layer_armor(center_base, theta, layer);
      auto_aim::Target target(first_observed, now, 0.2765, 3, P0_dig);
      if (!expect(
            !target.outpost_layer_locked(),
            "outpost target should not lock from a single observation at layer " +
              std::to_string(layer))) {
        return 1;
      }
      if (!expect(
            target.armor_xyza_list().size() == 1,
            "single-observation outpost should expose only observed armor at layer " +
              std::to_string(layer))) {
        return 1;
      }
    }
  }

  {
    Eigen::VectorXd P0_dig = Eigen::VectorXd::Zero(11);
    auto visible_armor = make_outpost_layer_armor({2.20, 0.10, 0.70}, 0.40, 0);
    auto_aim::Target target(visible_armor, now, 0.2765, 3, P0_dig);
    target.predict(0.05);
    if (!expect(
          target.ekf_.P(0, 0) > target.ekf_.P(4, 4) * 2.0,
          "outpost moving-platform xy process noise should be tuned separately from z")) {
      return 1;
    }
    if (!expect_near(
          target.ekf_.P(0, 0), target.ekf_.P(2, 2), 1e-12,
          "outpost x/y process noise should stay symmetric")) {
      return 1;
    }
  }

  {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 25, 81, 0.4, 100, 1e-4, 0, 0}};
    const Eigen::Vector3d center_base0(2.20, 0.10, 0.70);
    const Eigen::Vector3d center_base1(2.16, 0.13, 0.74);
    const Eigen::Vector3d center_base2(2.12, 0.16, 0.78);
    const double theta0 = 0.40;
    const double theta1 = theta0 + 0.05;
    const double theta2 = theta1 + 0.05;
    auto first_observed_layer_2 = make_outpost_layer_armor(center_base0, theta0, 2);
    auto second_observed_layer_1 = make_outpost_layer_armor(center_base1, theta1, 1);
    auto third_observed_layer_0 = make_outpost_layer_armor(center_base2, theta2, 0);
    auto_aim::Target target(first_observed_layer_2, now, 0.2765, 3, P0_dig);

    if (!expect(
          !target.outpost_layer_locked(),
          "outpost target should not lock layer semantics from the first observation")) {
      return 1;
    }
    if (!expect(
          target.armor_xyza_list().size() == 1,
          "unlocked outpost target should expose only the observed armor")) {
      return 1;
    }

    target.predict(now + std::chrono::milliseconds(20));
    target.update(second_observed_layer_1);
    if (!expect(
          !target.outpost_layer_locked(),
          "outpost target should keep layer semantics unlocked while hypotheses are ambiguous")) {
      return 1;
    }

    target.predict(now + std::chrono::milliseconds(40));
    target.update(third_observed_layer_0);

    if (!expect(target.outpost_layer_locked(), "outpost target should lock after distinct layers")) {
      return 1;
    }
    if (!expect(
          target.last_id == 0,
          "outpost moving-platform init should recover layer order after seeing multiple layers, actual=" +
            std::to_string(target.last_id))) {
      return 1;
    }
    if (!expect_near(
          target.ekf_x()[4], third_observed_layer_0.xyz_in_world[2], 0.08,
          "outpost base z should be updated from the matched layer")) {
      return 1;
    }
  }

  {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 25, 81, 0.4, 100, 1e-4, 0, 0}};
    constexpr double dt = 0.05;
    constexpr double base_vz = 0.4;
    const Eigen::Vector3d center_base0(2.20, 0.10, 0.70);
    const double theta0 = 0.40;

    auto moving_observation = [&](const std::array<int, 4> & layers, int index, double z_noise) {
      const double time = dt * static_cast<double>(index);
      Eigen::Vector3d center_base{
        center_base0[0] - 0.015 * static_cast<double>(index),
        center_base0[1] + 0.010 * static_cast<double>(index),
        center_base0[2] + base_vz * time + z_noise};
      return make_outpost_layer_armor(
        center_base, theta0 + 0.8 * CV_PI * time, layers[index]);
    };

    const std::array<int, 4> ambiguous_layers{2, 2, 2, 1};
    auto_aim::Target ambiguous_target(
      moving_observation(ambiguous_layers, 0, 0.0), now, 0.2765, 3, P0_dig);
    for (int i = 1; i < static_cast<int>(ambiguous_layers.size()); ++i) {
      ambiguous_target.predict(now + std::chrono::microseconds(static_cast<int64_t>(dt * i * 1e6)));
      ambiguous_target.update(moving_observation(ambiguous_layers, i, 0.0));
    }
    if (!expect(
          !ambiguous_target.outpost_layer_locked(),
          "outpost init should stay unlocked when only two adjacent layers are observed")) {
      return 1;
    }

    const std::array<int, 5> layers{2, 2, 2, 1, 0};
    auto moving_observation_5 = [&](int index, double z_noise) {
      const double time = dt * static_cast<double>(index);
      Eigen::Vector3d center_base{
        center_base0[0] - 0.015 * static_cast<double>(index),
        center_base0[1] + 0.010 * static_cast<double>(index),
        center_base0[2] + base_vz * time + z_noise};
      return make_outpost_layer_armor(
        center_base, theta0 + 0.8 * CV_PI * time, layers[index]);
    };

    auto_aim::Target target(moving_observation_5(0, 0.0), now, 0.2765, 3, P0_dig);
    for (int i = 1; i < static_cast<int>(layers.size()); ++i) {
      target.predict(now + std::chrono::microseconds(static_cast<int64_t>(dt * i * 1e6)));
      target.update(moving_observation_5(i, i == 4 ? 0.03 : 0.0));
    }

    if (!expect(
          target.outpost_layer_locked(),
          "outpost target should lock while platform z moves smoothly, score=" +
            std::to_string(target.ekf_.data["init_best_score"]) + ", margin=" +
            std::to_string(target.ekf_.data["init_margin"]) + ", distinct=" +
            std::to_string(target.ekf_.data["init_distinct_layers"]))) {
      return 1;
    }
    if (!expect_near(
          target.ekf_x()[5], base_vz, 0.25,
          "outpost init should estimate z velocity from a smooth multi-frame trend")) {
      return 1;
    }
  }

  {
    auto_aim::Solver solver(config_path);
    solver.set_R_gimbal2world(Eigen::Quaterniond::Identity());
    auto_aim::Tracker tracker(config_path, solver);

    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 25, 81, 0.4, 100, 1e-4, 0, 0}};
    const Eigen::Vector3d center_base(2.20, 0.10, 0.70);
    const double theta = 0.40;
    auto first = make_outpost_layer_armor(center_base, theta, 2);
    auto second = make_outpost_layer_armor(center_base, theta + 0.05, 1);
    auto third = make_outpost_layer_armor(center_base, theta + 0.10, 0);
    tracker.target_ = auto_aim::Target(first, now, 0.2765, 3, P0_dig);
    tracker.target_.predict(now + std::chrono::milliseconds(20));
    tracker.target_.update(second);
    tracker.target_.predict(now + std::chrono::milliseconds(40));
    tracker.target_.update(third);
    if (!expect(tracker.target_.outpost_layer_locked(), "tracker fixture should be layer locked")) {
      return 1;
    }
    tracker.target_.last_id = 2;

    const auto predicted = tracker.target_.armor_xyza_list();
    std::list<auto_aim::Armor> armors;
    armors.push_back(make_detection_outpost_armor(config_path, predicted[0].head(3), predicted[0][3]));
    armors.push_back(make_detection_outpost_armor(config_path, predicted[2].head(3), predicted[2][3]));

    const bool found = tracker.update_target(armors, now + std::chrono::milliseconds(10));
    if (!expect(found, "outpost tracker should find a candidate from multiple detections")) return 1;
    if (!expect(
          tracker.target_.last_id == 0,
          "outpost tracker candidate selection should not be biased to previous last_id, actual=" +
            std::to_string(tracker.target_.last_id))) {
      return 1;
    }
  }

  {
    auto_aim::Solver solver(config_path);
    solver.set_R_gimbal2world(Eigen::Quaterniond::Identity());
    auto_aim::Tracker tracker(config_path, solver);

    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 25, 81, 0.4, 100, 1e-4, 0, 0}};
    const Eigen::Vector3d center_base(2.20, 0.10, 0.70);
    auto first = make_outpost_layer_armor(center_base, 0.40, 2);
    auto second = make_outpost_layer_armor(center_base, 0.45, 1);
    auto third = make_outpost_layer_armor(center_base, 0.50, 0);
    tracker.target_ = auto_aim::Target(first, now, 0.2765, 3, P0_dig);
    tracker.target_.predict(now + std::chrono::milliseconds(20));
    tracker.target_.update(second);
    tracker.target_.predict(now + std::chrono::milliseconds(40));
    tracker.target_.update(third);
    if (!expect(
          tracker.target_.outpost_layer_locked(),
          "tracker fixture should be layer locked before large-dt test")) {
      return 1;
    }

    tracker.state_ = "tracking";
    tracker.last_timestamp_ = now + std::chrono::milliseconds(40);
    std::list<auto_aim::Armor> no_armors;
    const auto targets = tracker.track(no_armors, now + std::chrono::milliseconds(153));

    if (!expect(
          tracker.state() == "temp_lost",
          "locked outpost should enter temp_lost instead of lost on a single large dt")) {
      return 1;
    }
    if (!expect(!targets.empty(), "large-dt outpost temp_lost should keep the target model")) {
      return 1;
    }
    if (!expect(
          targets.front().outpost_layer_locked(),
          "large-dt outpost temp_lost should preserve layer semantics")) {
      return 1;
    }
  }

  {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 25, 81, 0.4, 100, 1e-4, 0, 0}};
    const Eigen::Vector3d center_base(2.0, 0.0, 1.0);
    auto first = make_outpost_layer_armor(center_base, 0.0, 2);
    auto second = make_outpost_layer_armor(center_base, 0.05, 1);
    auto third = make_outpost_layer_armor(center_base, 0.10, 0);
    auto_aim::Target target(first, now, 0.2765, 3, P0_dig);
    target.predict(now + std::chrono::milliseconds(20));
    target.update(second);
    target.predict(now + std::chrono::milliseconds(40));
    target.update(third);
    if (!expect(target.outpost_layer_locked(), "target should be locked before checking full geometry")) {
      return 1;
    }

    const auto initial_xyza = target.armor_xyza_list();
    if (!expect_near(
          initial_xyza[1][2] - initial_xyza[0][2], 0.1, 1e-6,
          "outpost layer spacing between layer 0 and 1 should be 0.1m")) {
      return 1;
    }
    if (!expect_near(
          initial_xyza[2][2] - initial_xyza[0][2], 0.2, 1e-6,
          "outpost layer spacing between layer 0 and 2 should be 0.2m")) {
      return 1;
    }
    if (!expect_near(
          tools::limit_rad(initial_xyza[0][3] - initial_xyza[1][3]), 2 * CV_PI / 3, 1e-6,
          "outpost low layer should be 120deg after middle layer")) {
      return 1;
    }
    if (!expect_near(
          tools::limit_rad(initial_xyza[2][3] - initial_xyza[1][3]), -2 * CV_PI / 3, 1e-6,
          "outpost high layer should be 120deg before middle layer")) {
      return 1;
    }

    auto observed_layer_2 = make_world_armor(
      auto_aim::ArmorName::outpost, auto_aim::ArmorType::base_outpost, initial_xyza[2].head(3),
      initial_xyza[2][3]);
    target.update(observed_layer_2, 2);

    if (!expect(target.last_id == 2, "target should match the observed outpost layer")) return 1;
    if (!expect(target.has_last_observed_armor(), "target should cache last observed armor")) return 1;
    if (!expect_near(
          target.last_observed_armor_xyza()[2], initial_xyza[2][2], 1e-6,
          "cached observed armor z mismatch")) {
      return 1;
    }
  }

  {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 25, 81, 0.4, 100, 1e-4, 0, 0}};
    const Eigen::Vector3d center_base(2.20, 0.10, 0.70);
    auto first = make_outpost_layer_armor(center_base, 0.40, 2);
    auto second = make_outpost_layer_armor(center_base, 0.45, 1);
    auto third = make_outpost_layer_armor(center_base, 0.50, 0);
    auto_aim::Target target(first, now, 0.2765, 3, P0_dig);
    target.predict(now + std::chrono::milliseconds(20));
    target.update(second);
    target.predict(now + std::chrono::milliseconds(40));
    target.update(third);
    if (!expect(target.outpost_layer_locked(), "target should be locked before z outlier test")) {
      return 1;
    }

    const double base_before = target.ekf_x()[4];
    target.ekf_.P = Eigen::MatrixXd::Identity(11, 11) * 1e-6;
    const auto predicted = target.armor_xyza_list();
    auto noisy_layer_0 = make_world_armor(
      auto_aim::ArmorName::outpost, auto_aim::ArmorType::base_outpost,
      predicted[0].head(3) + Eigen::Vector3d(0.0, 0.0, 0.20), predicted[0][3]);

    target.update(noisy_layer_0, 0);

    if (!expect(
          std::abs(target.ekf_x()[4] - base_before) < 0.05,
          "locked outpost base z should be estimated by EKF, not manually overwritten")) {
      return 1;
    }
  }

  {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 25, 81, 0.4, 100, 1e-4, 0, 0}};
    auto visible_armor =
      make_world_armor(auto_aim::ArmorName::outpost, auto_aim::ArmorType::base_outpost, {2.0, 0.0, 1.0}, 0.0);
    auto_aim::Target target(visible_armor, now, 0.2765, 3, P0_dig);
    if (!expect(
          !target.outpost_layer_locked(),
          "single visible outpost armor should remain unlocked before aiming")) {
      return 1;
    }

    auto_aim::Aimer aimer(config_path);
    std::list<auto_aim::Target> targets{target};
    const auto command = aimer.aim(targets, now, 27.0, false);

    if (!expect(command.control, "aimer should output a control command for visible outpost")) {
      return 1;
    }
    if (!expect(aimer.debug_aim_point.valid, "aimer debug aim point should be valid")) return 1;
    if (!expect(aimer.debug_aim_point.armor_id == 0, "aimer should directly aim at the visible armor")) {
      return 1;
    }
  }

  {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 25, 81, 0.4, 100, 1e-4, 0, 0}};
    const Eigen::Vector3d center_base(2.20, 0.10, 0.70);
    const double theta0 = 0.30;
    const double dt = 0.02;
    auto first = make_outpost_layer_armor(center_base, theta0, 1);
    auto second =
      make_outpost_layer_armor(center_base, theta0 + 0.8 * CV_PI * dt, 1);
    auto_aim::Target target(first, now, 0.2765, 3, P0_dig);
    const auto second_t = now + std::chrono::milliseconds(20);
    target.predict(second_t);
    target.update(second);

    if (!expect(
          !target.outpost_layer_locked(),
          "same-layer two-frame outpost should still wait for layer lock")) {
      return 1;
    }
    if (!expect(
          target.outpost_unlocked_prediction_ready(),
          "same-layer two-frame outpost should expose a short-horizon prediction preview")) {
      return 1;
    }

    auto_aim::Aimer aimer(config_path);
    std::list<auto_aim::Target> targets{target};
    const auto command = aimer.aim(targets, second_t, 27.0, false);

    if (!expect(command.control, "aimer should keep control while using unlocked outpost preview")) {
      return 1;
    }
    if (!expect(aimer.debug_aim_point.valid, "preview aim point should be valid")) return 1;

    const double lead_angle =
      tools::limit_rad(aimer.debug_aim_point.xyza[3] - second.ypr_in_world[0]);
    if (!expect(
          lead_angle > 0.12 && lead_angle < 0.8,
          "unlocked outpost preview should lead the last observation before full convergence")) {
      return 1;
    }
  }

  return 0;
}
