#include "math_tools.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <opencv2/core.hpp>  // CV_PI
#include <stdexcept>

namespace tools
{
GimbalAxisOrder parse_gimbal_axis_order(const std::string & value)
{
  std::string normalized;
  normalized.reserve(value.size());
  for (unsigned char ch : value) {
    if (std::isalnum(ch)) normalized.push_back(static_cast<char>(std::tolower(ch)));
  }

  if (
    normalized == "yawpitch" || normalized == "yawthenpitch" ||
    normalized == "yawbeforepitch") {
    return GimbalAxisOrder::yaw_pitch;
  }
  if (
    normalized == "pitchyaw" || normalized == "pitchthenyaw" ||
    normalized == "pitchbeforeyaw") {
    return GimbalAxisOrder::pitch_yaw;
  }

  throw std::runtime_error("Unsupported gimbal_axis_order: " + value);
}

const char * gimbal_axis_order_name(GimbalAxisOrder order)
{
  switch (order) {
    case GimbalAxisOrder::yaw_pitch:
      return "yaw_pitch";
    case GimbalAxisOrder::pitch_yaw:
      return "pitch_yaw";
  }
  return "yaw_pitch";
}

double limit_rad(double angle)
{
  while (angle > CV_PI) angle -= 2 * CV_PI;
  while (angle <= -CV_PI) angle += 2 * CV_PI;
  return angle;
}

Eigen::Vector3d eulers(Eigen::Quaterniond q, int axis0, int axis1, int axis2, bool extrinsic)
{
  if (!extrinsic) std::swap(axis0, axis2);

  auto i = axis0, j = axis1, k = axis2;
  auto is_proper = (i == k);
  if (is_proper) k = 3 - i - j;
  auto sign = (i - j) * (j - k) * (k - i) / 2;

  double a, b, c, d;
  Eigen::Vector4d xyzw = q.coeffs();
  if (is_proper) {
    a = xyzw[3];
    b = xyzw[i];
    c = xyzw[j];
    d = xyzw[k] * sign;
  } else {
    a = xyzw[3] - xyzw[j];
    b = xyzw[i] + xyzw[k] * sign;
    c = xyzw[j] + xyzw[3];
    d = xyzw[k] * sign - xyzw[i];
  }

  Eigen::Vector3d eulers;
  auto n2 = a * a + b * b + c * c + d * d;
  eulers[1] = std::acos(2 * (a * a + b * b) / n2 - 1);

  auto half_sum = std::atan2(b, a);
  auto half_diff = std::atan2(-d, c);

  auto eps = 1e-7;
  auto safe1 = std::abs(eulers[1]) >= eps;
  auto safe2 = std::abs(eulers[1] - CV_PI) >= eps;
  auto safe = safe1 && safe2;
  if (safe) {
    eulers[0] = half_sum + half_diff;
    eulers[2] = half_sum - half_diff;
  } else {
    if (!extrinsic) {
      eulers[0] = 0;
      if (!safe1) eulers[2] = 2 * half_sum;
      if (!safe2) eulers[2] = -2 * half_diff;
    } else {
      eulers[2] = 0;
      if (!safe1) eulers[0] = 2 * half_sum;
      if (!safe2) eulers[0] = 2 * half_diff;
    }
  }

  for (int i = 0; i < 3; i++) eulers[i] = limit_rad(eulers[i]);

  if (!is_proper) {
    eulers[2] *= sign;
    eulers[1] -= CV_PI / 2;
  }

  if (!extrinsic) std::swap(eulers[0], eulers[2]);

  return eulers;
}

Eigen::Vector3d eulers(Eigen::Matrix3d R, int axis0, int axis1, int axis2, bool extrinsic)
{
  Eigen::Quaterniond q(R);
  return eulers(q, axis0, axis1, axis2, extrinsic);
}

Eigen::Matrix3d rotation_matrix(const Eigen::Vector3d & ypr)
{
  double roll = ypr[2];
  double pitch = ypr[1];
  double yaw = ypr[0];
  double cos_yaw = cos(yaw);
  double sin_yaw = sin(yaw);
  double cos_pitch = cos(pitch);
  double sin_pitch = sin(pitch);
  double cos_roll = cos(roll);
  double sin_roll = sin(roll);
  // clang-format off
    Eigen::Matrix3d R{
      {cos_yaw * cos_pitch, cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll, cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll},
      {sin_yaw * cos_pitch, sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll, sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll},
      {         -sin_pitch,                                cos_pitch * sin_roll,                                cos_pitch * cos_roll}
    };
  // clang-format on
  return R;
}

Eigen::Vector3d xyz2ypd(const Eigen::Vector3d & xyz)
{
  auto x = xyz[0], y = xyz[1], z = xyz[2];
  auto yaw = std::atan2(y, x);
  auto pitch = std::atan2(z, std::sqrt(x * x + y * y));
  auto distance = std::sqrt(x * x + y * y + z * z);
  return {yaw, pitch, distance};
}

Eigen::MatrixXd xyz2ypd_jacobian(const Eigen::Vector3d & xyz)
{
  auto x = xyz[0], y = xyz[1], z = xyz[2];

  auto dyaw_dx = -y / (x * x + y * y);
  auto dyaw_dy = x / (x * x + y * y);
  auto dyaw_dz = 0.0;

  auto dpitch_dx = -(x * z) / ((z * z / (x * x + y * y) + 1) * std::pow((x * x + y * y), 1.5));
  auto dpitch_dy = -(y * z) / ((z * z / (x * x + y * y) + 1) * std::pow((x * x + y * y), 1.5));
  auto dpitch_dz = 1 / ((z * z / (x * x + y * y) + 1) * std::pow((x * x + y * y), 0.5));

  auto ddistance_dx = x / std::pow((x * x + y * y + z * z), 0.5);
  auto ddistance_dy = y / std::pow((x * x + y * y + z * z), 0.5);
  auto ddistance_dz = z / std::pow((x * x + y * y + z * z), 0.5);

  // clang-format off
  Eigen::MatrixXd J{
    {dyaw_dx, dyaw_dy, dyaw_dz},
    {dpitch_dx, dpitch_dy, dpitch_dz},
    {ddistance_dx, ddistance_dy, ddistance_dz}
  };
  // clang-format on

  return J;
}

Eigen::Vector3d ypd2xyz(const Eigen::Vector3d & ypd)
{
  auto yaw = ypd[0], pitch = ypd[1], distance = ypd[2];
  auto x = distance * std::cos(pitch) * std::cos(yaw);
  auto y = distance * std::cos(pitch) * std::sin(yaw);
  auto z = distance * std::sin(pitch);
  return {x, y, z};
}

Eigen::MatrixXd ypd2xyz_jacobian(const Eigen::Vector3d & ypd)
{
  auto yaw = ypd[0], pitch = ypd[1], distance = ypd[2];
  double cos_yaw = std::cos(yaw);
  double sin_yaw = std::sin(yaw);
  double cos_pitch = std::cos(pitch);
  double sin_pitch = std::sin(pitch);

  auto dx_dyaw = distance * cos_pitch * -sin_yaw;
  auto dy_dyaw = distance * cos_pitch * cos_yaw;
  auto dz_dyaw = 0.0;

  auto dx_dpitch = distance * -sin_pitch * cos_yaw;
  auto dy_dpitch = distance * -sin_pitch * sin_yaw;
  auto dz_dpitch = distance * cos_pitch;

  auto dx_ddistance = cos_pitch * cos_yaw;
  auto dy_ddistance = cos_pitch * sin_yaw;
  auto dz_ddistance = sin_pitch;

  // clang-format off
  Eigen::MatrixXd J{
    {dx_dyaw, dx_dpitch, dx_ddistance},
    {dy_dyaw, dy_dpitch, dy_ddistance},
    {dz_dyaw, dz_dpitch, dz_ddistance}
  };
  // clang-format on

  return J;
}

Eigen::Vector3d gimbal_direction_from_command(
  double yaw, double pitch, GimbalAxisOrder order)
{
  const double cos_yaw = std::cos(yaw);
  const double sin_yaw = std::sin(yaw);
  const double cos_pitch = std::cos(pitch);
  const double sin_pitch = std::sin(pitch);

  switch (order) {
    case GimbalAxisOrder::yaw_pitch:
      return {cos_pitch * cos_yaw, cos_pitch * sin_yaw, -sin_pitch};
    case GimbalAxisOrder::pitch_yaw:
      return {cos_pitch * cos_yaw, sin_yaw, -sin_pitch * cos_yaw};
  }

  return {cos_pitch * cos_yaw, cos_pitch * sin_yaw, -sin_pitch};
}

Eigen::Vector2d gimbal_command_from_direction(
  const Eigen::Vector3d & direction, GimbalAxisOrder order)
{
  if (direction.squaredNorm() < 1e-12) return {0.0, 0.0};

  const Eigen::Vector3d unit = direction.normalized();

  if (order == GimbalAxisOrder::yaw_pitch) {
    return {
      std::atan2(unit.y(), unit.x()),
      std::atan2(-unit.z(), std::hypot(unit.x(), unit.y()))};
  }

  const double sin_yaw = std::clamp(unit.y(), -1.0, 1.0);
  const double yaw_front = std::asin(sin_yaw);
  const double yaw_back = sin_yaw >= 0.0 ? CV_PI - yaw_front : -CV_PI - yaw_front;
  const double bearing = std::atan2(unit.y(), unit.x());

  auto candidate = [&](double yaw) {
    const double cos_yaw = std::cos(yaw);
    double pitch = 0.0;
    if (std::abs(cos_yaw) > 1e-9) {
      pitch = std::atan2(-unit.z() / cos_yaw, unit.x() / cos_yaw);
    }
    return Eigen::Vector2d{limit_rad(yaw), limit_rad(pitch)};
  };

  const Eigen::Vector2d front = candidate(yaw_front);
  const Eigen::Vector2d back = candidate(yaw_back);
  const double front_cost = std::abs(front[1]) + 1e-3 * std::abs(limit_rad(front[0] - bearing));
  const double back_cost = std::abs(back[1]) + 1e-3 * std::abs(limit_rad(back[0] - bearing));
  return front_cost <= back_cost ? front : back;
}

Eigen::Vector2d gimbal_command_from_yaw_elevation(
  double yaw, double elevation, GimbalAxisOrder order)
{
  const double cos_elevation = std::cos(elevation);
  const Eigen::Vector3d direction{
    cos_elevation * std::cos(yaw), cos_elevation * std::sin(yaw), std::sin(elevation)};
  return gimbal_command_from_direction(direction, order);
}

double delta_time(
  const std::chrono::steady_clock::time_point & a, const std::chrono::steady_clock::time_point & b)
{
  std::chrono::duration<double> c = a - b;
  return c.count();
}

double get_abs_angle(const Eigen::Vector2d & vec1, const Eigen::Vector2d & vec2)
{
  if (vec1.norm() == 0. || vec2.norm() == 0.) {
    return 0.;
  }
  return std::acos(vec1.dot(vec2) / (vec1.norm() * vec2.norm()));
}

double limit_min_max(double input, double min, double max)
{
  if (input > max)
    return max;
  else if (input < min)
    return min;
  return input;
}
}  // namespace tools
