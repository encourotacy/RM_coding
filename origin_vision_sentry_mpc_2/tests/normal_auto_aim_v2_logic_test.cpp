#include <chrono>
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

#include "tasks/auto_aim/solver.hpp"
#include "tools/math_tools.hpp"

namespace
{
constexpr double kLightbarLength = 56e-3;
constexpr double kSmallArmorWidth = 135e-3;

bool expect(bool condition, const std::string & message)
{
  if (condition) return true;
  std::cerr << message << std::endl;
  return false;
}

auto_aim::Armor make_world_armor(
  auto_aim::ArmorName name, auto_aim::ArmorType type, const Eigen::Vector3d & xyz, double yaw)
{
  const std::vector<cv::Point2f> dummy_points{
    {0.0f, 0.0f}, {10.0f, 0.0f}, {10.0f, 10.0f}, {0.0f, 10.0f}};
  auto_aim::Armor armor(0, 1, 1.0f, cv::Rect(0, 0, 10, 10), dummy_points);
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

std::vector<cv::Point2f> project_small_armor(
  const std::string & config_path, const Eigen::Vector3d & xyz_in_world, double yaw)
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
  const auto pitch = 15.0 * CV_PI / 180.0;
  const auto sin_pitch = std::sin(pitch);
  const auto cos_pitch = std::cos(pitch);

  const Eigen::Matrix3d R_armor2world{
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch, cos_yaw, sin_yaw * sin_pitch},
    {-sin_pitch, 0.0, cos_pitch}};

  const Eigen::Matrix3d R_armor2camera = R_camera2gimbal.transpose() * R_armor2world;
  const Eigen::Vector3d t_armor2camera =
    R_camera2gimbal.transpose() * (xyz_in_world - t_camera2gimbal);

  const std::vector<cv::Point3f> object_points{
    {0.0f, static_cast<float>(kSmallArmorWidth / 2), static_cast<float>(kLightbarLength / 2)},
    {0.0f, static_cast<float>(-kSmallArmorWidth / 2), static_cast<float>(kLightbarLength / 2)},
    {0.0f, static_cast<float>(-kSmallArmorWidth / 2), static_cast<float>(-kLightbarLength / 2)},
    {0.0f, static_cast<float>(kSmallArmorWidth / 2), static_cast<float>(-kLightbarLength / 2)},
  };

  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(R_armor2camera, R_armor2camera_cv);
  cv::Vec3d rvec;
  cv::Rodrigues(R_armor2camera_cv, rvec);
  cv::Vec3d tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  std::vector<cv::Point2f> image_points;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix_cv, distort_coeffs_cv, image_points);
  return image_points;
}

auto_aim::Armor make_detection_armor(
  const std::string & config_path, auto_aim::ArmorName name, const Eigen::Vector3d & xyz, double yaw)
{
  auto armor = make_world_armor(name, auto_aim::ArmorType::small, xyz, yaw);
  armor.points = project_small_armor(config_path, xyz, yaw);
  armor.center = (armor.points[0] + armor.points[1] + armor.points[2] + armor.points[3]) * 0.25f;
  return armor;
}

Eigen::VectorXd normal_p0()
{
  return Eigen::VectorXd{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
}

}  // namespace

int main()
{
  const std::string config_path = "configs/sentry.yaml";
  const auto now = std::chrono::steady_clock::now();

  {
    auto armor = make_world_armor(auto_aim::ArmorName::one, auto_aim::ArmorType::small, {2.0, 0.0, 0.5}, 0.0);
    auto_aim::Target target(armor, now, 0.2, 4, normal_p0());

    auto_aim::Aimer aimer(config_path);
    const auto aim_point = aimer.choose_aim_point(target);

    if (!expect(aim_point.valid, "non-jumped normal target should produce a valid aim point")) return 1;
    if (!expect(aim_point.armor_id == 0, "non-jumped normal target should keep aiming at armor 0")) {
      return 1;
    }
  }

  {
    const double yaw = 25.0 / 57.3;
    const Eigen::Vector3d center(2.0, 0.0, 0.5);
    const Eigen::Vector3d xyz(center[0] - 0.2 * std::cos(yaw), center[1] - 0.2 * std::sin(yaw), center[2]);
    auto armor = make_world_armor(auto_aim::ArmorName::one, auto_aim::ArmorType::small, xyz, yaw);
    auto_aim::Target target(armor, now, 0.2, 4, normal_p0());
    target.jumped = true;
    target.ekf_.x[7] = 12.0;

    auto_aim::Aimer aimer(config_path);
    const auto aim_point = aimer.choose_aim_point(target);

    if (!expect(aim_point.valid, "high-vyaw normal target should still choose a direct armor under v2 logic")) {
      return 1;
    }
    if (!expect(
          aim_point.armor_id == 0,
          "high-vyaw normal target should keep the only 60-degree visible armor, actual armor_id=" +
            std::to_string(aim_point.armor_id))) {
      return 1;
    }
  }

  {
    const double yaw = 45.0 / 57.3;
    const Eigen::Vector3d center(2.0, 0.0, 0.5);
    const Eigen::Vector3d xyz(center[0] - 0.2 * std::cos(yaw), center[1] - 0.2 * std::sin(yaw), center[2]);
    auto armor = make_world_armor(auto_aim::ArmorName::one, auto_aim::ArmorType::small, xyz, yaw);
    auto_aim::Target target(armor, now, 0.2, 4, normal_p0());
    target.jumped = true;

    auto_aim::Aimer aimer(config_path);
    const auto first_point = aimer.choose_aim_point(target);
    if (!expect(first_point.valid, "locked normal target should produce a valid aim point")) return 1;
    const int first_id = first_point.armor_id;
    if (!expect(first_id == 0 || first_id == 3, "lock mode should select one of the two visible armors")) {
      return 1;
    }

    target.ekf_.x[6] = 40.0 / 57.3;
    const auto second_point = aimer.choose_aim_point(target);
    if (!expect(second_point.valid, "locked normal target should continue producing a valid aim point")) return 1;
    if (!expect(second_point.armor_id == first_id, "lock mode should keep the previous locked armor")) {
      return 1;
    }
  }

  {
    const double yaw = 45.0 / 57.3;
    const Eigen::Vector3d center(2.0, 0.0, 0.5);
    const Eigen::Vector3d xyz(center[0] - 0.2 * std::cos(yaw), center[1] - 0.2 * std::sin(yaw), center[2]);
    auto armor = make_world_armor(auto_aim::ArmorName::one, auto_aim::ArmorType::small, xyz, yaw);
    auto_aim::Target target(armor, now, 0.2, 4, normal_p0());
    target.last_id = 3;
    target.update_count_ = 5;

    const auto predicted = target.armor_xyza_list();
    auto observed = make_world_armor(
      auto_aim::ArmorName::one, auto_aim::ArmorType::small, predicted[0].head(3), predicted[0][3]);
    const int matched_id = target.match_default_armor_id(observed, predicted);
    if (!expect(matched_id == 0, "normal armor matching should not be held back by previous last_id bias")) {
      return 1;
    }
  }

  {
    auto_aim::Solver solver(config_path);
    solver.set_R_gimbal2world(Eigen::Quaterniond::Identity());
    auto_aim::Tracker tracker(config_path, solver);

    auto initial_armor =
      make_world_armor(auto_aim::ArmorName::one, auto_aim::ArmorType::small, {2.0, 0.0, 0.5}, 0.0);
    tracker.target_ = auto_aim::Target(initial_armor, now, 0.2, 4, normal_p0());
    tracker.target_.update_count_ = 0;

    const auto predicted = tracker.target_.armor_xyza_list();
    std::list<auto_aim::Armor> armors;
    armors.push_back(make_detection_armor(
      config_path, auto_aim::ArmorName::one, predicted[0].head(3), predicted[0][3]));
    armors.push_back(make_detection_armor(
      config_path, auto_aim::ArmorName::one, predicted[3].head(3), predicted[3][3]));

    const bool found = tracker.update_target(armors, now + std::chrono::milliseconds(10));
    if (!expect(found, "tracker should find normal armor candidates")) return 1;
    if (!expect(
          tracker.target_.update_count_ == 2,
          "normal tracker update should consume all matching candidates like v2")) {
      return 1;
    }
  }

  return 0;
}
