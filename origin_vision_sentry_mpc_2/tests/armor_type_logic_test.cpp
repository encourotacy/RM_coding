#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "tasks/auto_aim/armor.hpp"

namespace
{
bool expect(bool condition, const std::string & message)
{
  if (condition) return true;
  std::cerr << message << std::endl;
  return false;
}

auto_aim::Armor make_yolov5_armor(int num_id)
{
  std::vector<cv::Point2f> points{
    {0.0f, 0.0f}, {10.0f, 0.0f}, {10.0f, 8.0f}, {0.0f, 8.0f}};
  return auto_aim::Armor(0, num_id, 1.0f, cv::Rect(0, 0, 10, 8), points);
}
}  // namespace

int main()
{
  using auto_aim::ArmorName;
  using auto_aim::ArmorType;

  if (!expect(
        auto_aim::default_type_for_name(ArmorName::outpost) == ArmorType::base_outpost,
        "outpost should default to base_outpost type")) {
    return 1;
  }
  if (!expect(
        auto_aim::default_type_for_name(ArmorName::base) == ArmorType::base_outpost,
        "base should default to base_outpost type")) {
    return 1;
  }
  if (!expect(
        auto_aim::default_type_for_name(ArmorName::one) == ArmorType::big,
        "hero should default to big type")) {
    return 1;
  }
  if (!expect(
        auto_aim::default_type_for_name(ArmorName::two) == ArmorType::small,
        "engineer should default to small type")) {
    return 1;
  }
  if (!expect(
        auto_aim::default_type_for_name(ArmorName::sentry) == ArmorType::small,
        "sentry should default to small type")) {
    return 1;
  }

  if (!expect(
        !auto_aim::is_valid_type_for_name(ArmorName::outpost, ArmorType::small),
        "outpost small type should be invalid")) {
    return 1;
  }
  if (!expect(
        !auto_aim::is_valid_type_for_name(ArmorName::base, ArmorType::big),
        "base big type should be invalid")) {
    return 1;
  }
  if (!expect(
        auto_aim::is_valid_type_for_name(ArmorName::three, ArmorType::small),
        "infantry small type should be valid")) {
    return 1;
  }
  if (!expect(
        auto_aim::is_valid_type_for_name(ArmorName::three, ArmorType::big),
        "infantry big type should be valid")) {
    return 1;
  }

  const auto outpost = make_yolov5_armor(6);
  if (!expect(outpost.name == ArmorName::outpost, "YOLOV5 num_id=6 should map to outpost")) {
    return 1;
  }
  if (!expect(
        outpost.type == ArmorType::base_outpost,
        "YOLOV5 outpost should use base_outpost type")) {
    return 1;
  }

  const auto base = make_yolov5_armor(7);
  if (!expect(base.name == ArmorName::base, "YOLOV5 num_id=7 should map to base")) {
    return 1;
  }
  if (!expect(
        base.type == ArmorType::base_outpost, "YOLOV5 base should use base_outpost type")) {
    return 1;
  }

  return 0;
}
