#include <cmath>
#include <iostream>

#include "tasks/auto_aim/pattern_geometry.hpp"

namespace
{
bool expect_near(double actual, double expected, double eps, const char * message)
{
  if (std::abs(actual - expected) <= eps) return true;
  std::cerr << message << ", actual=" << actual << ", expected=" << expected << std::endl;
  return false;
}
}  // namespace

int main()
{
  using auto_aim::ArmorName;

  constexpr double eps = 1e-9;
  const double fixed_ratio = auto_aim::pattern_expand_ratio_for(ArmorName::outpost);
  const double base_ratio = auto_aim::pattern_expand_ratio_for(ArmorName::base);
  const double default_ratio = auto_aim::pattern_expand_ratio_for(ArmorName::three);

  if (!expect_near(
        auto_aim::pattern_height_mm_for(ArmorName::outpost), 100.0, eps,
        "outpost pattern height should be 100mm")) {
    return 1;
  }
  if (!expect_near(
        auto_aim::pattern_height_mm_for(ArmorName::base), 100.0, eps,
        "base pattern height should be 100mm")) {
    return 1;
  }
  if (!expect_near(
        auto_aim::pattern_height_mm_for(ArmorName::three), 126.0, eps,
        "regular armor pattern height should remain 126mm")) {
    return 1;
  }

  if (!expect_near(
        fixed_ratio, 0.5 * 100.0 / 56.0, eps, "outpost pattern expand ratio mismatch")) {
    return 1;
  }
  if (!expect_near(
        base_ratio, 0.5 * 100.0 / 56.0, eps, "base pattern expand ratio mismatch")) {
    return 1;
  }
  if (!expect_near(
        default_ratio, 0.5 * 126.0 / 56.0, eps, "regular armor pattern expand ratio mismatch")) {
    return 1;
  }

  if (!(fixed_ratio < default_ratio && base_ratio < default_ratio)) {
    std::cerr << "fixed armor ROI should be tighter than regular armor ROI" << std::endl;
    return 1;
  }

  return 0;
}
