#ifndef OVSENTRY__CONFIG_HPP
#define OVSENTRY__CONFIG_HPP

#include <chrono>
#include <optional>
#include <string>

#include "types.hpp"

namespace ovsentry
{

enum class AppMode { AutoSwitch, AutoAim, Buff, Omni };

inline const char * app_mode_name(AppMode mode)
{
  switch (mode) {
    case AppMode::AutoAim:
      return "AutoAim";
    case AppMode::Buff:
      return "Buff";
    case AppMode::Omni:
      return "Omni";
    case AppMode::AutoSwitch:
      return "MPC";
  }
  return "MPC";
}

struct RuntimeConfig
{
  AppMode mode = AppMode::AutoSwitch;
  std::string config_path;
  std::string auto_aim_device;
  std::string omni_device;
  std::string auto_aim_ignore_topic;
  std::string auto_aim_ignore_msg_type;
  double omni_retarget_cooldown_s = 2.5;
  double omni_hold_release_tolerance_deg = 3.0;
  double omni_retarget_min_delta_deg = 20.0;
  double omni_command_timeout_s = 0.5;
  double buff_lost_cmd_hold_s = 0.3;
  std::chrono::milliseconds omni_read_timeout{10};
  std::chrono::steady_clock::duration omni_retarget_cooldown{};
  std::chrono::steady_clock::duration omni_command_timeout{};
  std::chrono::steady_clock::duration buff_lost_cmd_hold{};
  OmniCamConfig left_cam;
  OmniCamConfig right_cam;
  OmniCamConfig back_cam;
  bool display = true;
};

std::optional<RuntimeConfig> parse_runtime_config(int argc, char ** argv);

}  // namespace ovsentry

#endif  // OVSENTRY__CONFIG_HPP
