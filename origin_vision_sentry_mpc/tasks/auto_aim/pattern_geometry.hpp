#ifndef AUTO_AIM__PATTERN_GEOMETRY_HPP
#define AUTO_AIM__PATTERN_GEOMETRY_HPP

#include "armor.hpp"

namespace auto_aim
{

constexpr double LIGHTBAR_LENGTH_MM = 56.0;
constexpr double DEFAULT_PATTERN_HEIGHT_MM = 126.0;
constexpr double FIXED_PATTERN_HEIGHT_MM = 100.0;

inline double pattern_height_mm_for(ArmorName name)
{
  return (name == ArmorName::outpost || name == ArmorName::base) ? FIXED_PATTERN_HEIGHT_MM
                                                                  : DEFAULT_PATTERN_HEIGHT_MM;
}

// This helper only affects pattern ROI extraction, not PnP object points.
inline double pattern_expand_ratio_for(ArmorName name)
{
  return 0.5 * pattern_height_mm_for(name) / LIGHTBAR_LENGTH_MM;
}

}  // namespace auto_aim

#endif  // AUTO_AIM__PATTERN_GEOMETRY_HPP
