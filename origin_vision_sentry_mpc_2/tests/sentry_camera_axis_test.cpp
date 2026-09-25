#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace
{
bool expect(bool condition, const std::string & message)
{
  if (condition) return true;
  std::cerr << message << std::endl;
  return false;
}
}  // namespace

int main(int argc, char * argv[])
{
  if (argc != 2) {
    std::cerr << "usage: sentry_camera_axis_test <sentry.yaml>" << std::endl;
    return 1;
  }

  const auto yaml = YAML::LoadFile(argv[1]);
  const int width = yaml["image_width"].as<int>();
  const int height = yaml["image_height"].as<int>();
  const auto k_data = yaml["camera_matrix"].as<std::vector<double>>();
  const auto r_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  if (!expect(k_data.size() == 9 && r_data.size() == 9, "invalid calibration matrix size")) {
    return 1;
  }

  const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K(k_data.data());
  const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R(r_data.data());
  if (!expect(
        K.allFinite() && K(0, 0) > 0.0 && K(1, 1) > 0.0,
        "camera intrinsics must be finite with positive focal lengths")) {
    return 1;
  }
  if (!expect(
        K(0, 2) >= 0.0 && K(0, 2) < width && K(1, 2) >= 0.0 && K(1, 2) < height,
        "principal point must lie inside the raw sensor image")) {
    return 1;
  }
  if (!expect(
        (R * R.transpose() - Eigen::Matrix3d::Identity()).norm() < 1e-6 &&
          std::abs(R.determinant() - 1.0) < 1e-6,
        "R_camera2gimbal must be a proper rotation")) {
    return 1;
  }

  const auto ray_in_gimbal = [&](double u, double v) {
    const Eigen::Vector3d ray_camera{
      (u - K(0, 2)) / K(0, 0), (v - K(1, 2)) / K(1, 1), 1.0};
    return (R * ray_camera.normalized()).normalized();
  };
  const auto yaw = [](const Eigen::Vector3d & ray) { return std::atan2(ray.y(), ray.x()); };
  const auto elevation = [](const Eigen::Vector3d & ray) {
    return std::atan2(ray.z(), std::hypot(ray.x(), ray.y()));
  };

  const double cx = K(0, 2);
  const double cy = K(1, 2);
  const Eigen::Vector3d center = ray_in_gimbal(cx, cy);
  const Eigen::Vector3d top = ray_in_gimbal(cx, cy - 200.0);
  const Eigen::Vector3d bottom = ray_in_gimbal(cx, cy + 200.0);
  const Eigen::Vector3d left = ray_in_gimbal(cx - 200.0, cy);
  const Eigen::Vector3d right = ray_in_gimbal(cx + 200.0, cy);

  if (!expect(center.x() > 0.9, "raw optical axis must point forward in the gimbal frame")) {
    return 1;
  }
  if (!expect(
        elevation(top) > elevation(center) && elevation(center) > elevation(bottom),
        "raw pixel y direction is inconsistent: top must map above bottom")) {
    return 1;
  }
  if (!expect(
        yaw(left) > yaw(center) && yaw(center) > yaw(right),
        "raw pixel x direction is inconsistent: left must map left of right")) {
    return 1;
  }

  return 0;
}
