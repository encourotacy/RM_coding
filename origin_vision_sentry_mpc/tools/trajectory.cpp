#include "trajectory.hpp"

#include <cmath>

namespace tools
{
constexpr double g = 9.7833;

namespace
{
double resistance_residual(double theta, double v0, double d, double h, double k)
{
  const double cos_theta = std::cos(theta);
  const double sin_theta = std::sin(theta);
  const double ratio = k * d / (v0 * cos_theta);

  if (ratio >= 0.999) return 1e6;

  const double term1 = (k * v0 * sin_theta + g) * d / (k * v0 * cos_theta);
  const double term2 = g / (k * k) * std::log(1.0 - ratio);
  return term1 + term2 - h;
}

double resistance_fly_time(double theta, double v0, double d, double k)
{
  const double ratio = k * d / (v0 * std::cos(theta));
  if (ratio >= 0.999) return 1e6;
  return -std::log(1.0 - ratio) / k;
}
}  // namespace

Trajectory::Trajectory(double v0, double d, double h, double k)
{
  if (k < 1e-6) {
    const auto a = g * d * d / (2 * v0 * v0);
    const auto b = -d;
    const auto c = a + h;
    const auto delta = b * b - 4 * a * c;

    if (delta < 0) {
      unsolvable = true;
      return;
    }

    unsolvable = false;
    const auto tan_pitch_1 = (-b + std::sqrt(delta)) / (2 * a);
    const auto tan_pitch_2 = (-b - std::sqrt(delta)) / (2 * a);
    const auto pitch_1 = std::atan(tan_pitch_1);
    const auto pitch_2 = std::atan(tan_pitch_2);
    const auto t_1 = d / (v0 * std::cos(pitch_1));
    const auto t_2 = d / (v0 * std::cos(pitch_2));

    pitch = (t_1 < t_2) ? pitch_1 : pitch_2;
    fly_time = (t_1 < t_2) ? t_1 : t_2;
    return;
  }

  double theta0;
  {
    const double a = g * d * d / (2 * v0 * v0);
    const double b = -d;
    const double c = a + h;
    const double delta = b * b - 4 * a * c;
    if (delta < 0) {
      theta0 = std::atan2(h, d);
    } else {
      const double tan_pitch_1 = (-b + std::sqrt(delta)) / (2 * a);
      const double tan_pitch_2 = (-b - std::sqrt(delta)) / (2 * a);
      const double pitch_1 = std::atan(tan_pitch_1);
      const double pitch_2 = std::atan(tan_pitch_2);
      const double t_1 = d / (v0 * std::cos(pitch_1));
      const double t_2 = d / (v0 * std::cos(pitch_2));
      theta0 = (t_1 < t_2) ? pitch_1 : pitch_2;
    }
  }

  double theta = theta0;
  constexpr double eps = 1e-8;
  constexpr int max_iter = 25;
  bool solved = false;

  for (int i = 0; i < max_iter; ++i) {
    const double residual = resistance_residual(theta, v0, d, h, k);
    if (std::abs(residual) < 1e-6) {
      solved = true;
      break;
    }

    const double residual_plus = resistance_residual(theta + eps, v0, d, h, k);
    const double derivative = (residual_plus - residual) / eps;
    if (std::abs(derivative) < 1e-12) break;

    theta -= residual / derivative;
    theta = std::max(-0.785, std::min(1.309, theta));
  }

  if (!solved) {
    const double residual = resistance_residual(theta, v0, d, h, k);
    solved = std::abs(residual) < 0.01;
  }

  if (!solved) {
    unsolvable = true;
    return;
  }

  unsolvable = false;
  pitch = theta;
  fly_time = resistance_fly_time(theta, v0, d, k);
}

}  // namespace tools
