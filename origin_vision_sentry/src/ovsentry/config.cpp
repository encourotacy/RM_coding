#include "config.hpp"

#include <algorithm>
#include <string>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "tasks/omniperception/detection.hpp"
#include "tools/yaml.hpp"

namespace ovsentry
{
namespace
{

const std::string keys =
  "{help h usage ? |                         | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml    | 位置参数，yaml配置文件路径 }"
  "{left           | __yaml__                | 左前相机设备名(相对/dev)，默认读yaml.omni_left_path }"
  "{right          | __yaml__                | 右前相机设备名(相对/dev)，默认读yaml.omni_right_path }"
  "{back           | __yaml__                | 正后相机设备名(相对/dev)，默认读yaml.omni_back_path }"
  "{left_yaw       | 60                      | 左前相机中心yaw角(deg) }"
  "{right_yaw      | -60                     | 右前相机中心yaw角(deg) }"
  "{back_yaw       | 180                     | 正后相机中心yaw角(deg) }"
  "{fov_h          | 120                     | USB相机水平视场角(deg) }"
  "{fov_v          | 67                      | USB相机垂直视场角(deg) }"
  "{no-display     |                         | 关闭画面显示 }";

std::string normalize_dev_name(const std::string & dev)
{
  if (dev.rfind("/dev/", 0) == 0) return dev.substr(5);
  return dev;
}

}  // namespace

std::optional<RuntimeConfig> parse_runtime_config(int argc, char ** argv)
{
  cv::CommandLineParser cli(argc, argv, keys);
  RuntimeConfig cfg;
  cfg.config_path = cli.get<std::string>(0);
  if (cli.has("help") || cfg.config_path.empty()) {
    cli.printMessage();
    return std::nullopt;
  }

  auto yaml = tools::load(cfg.config_path);
  auto read_infer_device = [&](const std::string & key) {
    if (yaml[key]) return yaml[key].as<std::string>();
    if (yaml["device"]) return yaml["device"].as<std::string>();
    return std::string("UNKNOWN");
  };
  auto read_cam_path = [&](const std::string & cli_key, const std::string & yaml_key,
                           const std::string & fallback) {
    const auto cli_value = cli.get<std::string>(cli_key);
    if (!cli_value.empty() && cli_value != "__yaml__") return normalize_dev_name(cli_value);
    if (yaml[yaml_key]) return normalize_dev_name(yaml[yaml_key].as<std::string>());
    return normalize_dev_name(fallback);
  };
  auto read_cli_or_yaml_double = [&](const std::string & cli_key, const std::string & yaml_key,
                                     double fallback) {
    if (cli.has(cli_key)) return cli.get<double>(cli_key);
    if (yaml[yaml_key]) return yaml[yaml_key].as<double>();
    return fallback;
  };

  cfg.auto_aim_device = read_infer_device("auto_aim_device");
  cfg.omni_device = read_infer_device("omni_device");
  cfg.omni_retarget_cooldown_s =
    yaml["omni_retarget_cooldown_s"] ? yaml["omni_retarget_cooldown_s"].as<double>() : 2.5;
  cfg.omni_hold_release_tolerance_deg =
    yaml["omni_hold_release_tolerance_deg"] ? yaml["omni_hold_release_tolerance_deg"].as<double>()
                                            : 3.0;
  cfg.omni_retarget_min_delta_deg =
    yaml["omni_retarget_min_delta_deg"] ? yaml["omni_retarget_min_delta_deg"].as<double>() : 20.0;
  cfg.omni_command_timeout_s =
    yaml["omni_command_timeout_s"] ? yaml["omni_command_timeout_s"].as<double>() : 0.5;
  cfg.buff_lost_cmd_hold_s =
    yaml["buff_lost_cmd_hold_s"] ? yaml["buff_lost_cmd_hold_s"].as<double>() : 0.3;
  cfg.omni_read_timeout = std::chrono::milliseconds(std::max(
    1, yaml["omni_camera_read_timeout_ms"] ? yaml["omni_camera_read_timeout_ms"].as<int>() : 10));
  cfg.omni_retarget_cooldown = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(cfg.omni_retarget_cooldown_s));
  cfg.omni_command_timeout = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(cfg.omni_command_timeout_s));
  cfg.buff_lost_cmd_hold = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(std::max(0.0, cfg.buff_lost_cmd_hold_s)));
  cfg.auto_aim_ignore_topic = yaml["auto_aim_ignore_topic"]
                                ? yaml["auto_aim_ignore_topic"].as<std::string>()
                                : "/request_auto_aim_ignore";
  cfg.auto_aim_ignore_msg_type =
    yaml["auto_aim_ignore_msg_type"] ? yaml["auto_aim_ignore_msg_type"].as<std::string>()
                                     : "rm_interfaces/msg/RequestAutoAimIgnore";

  const double omni_fov_h_deg = read_cli_or_yaml_double("fov_h", "omni_fov_h_deg", 120.0);
  const double omni_fov_v_deg = read_cli_or_yaml_double("fov_v", "omni_fov_v_deg", 67.0);
  cfg.left_cam = OmniCamConfig{
    {omniperception::OmniCameraSlot::left, "left", read_cam_path("left", "omni_left_path", "video0"),
     read_cli_or_yaml_double("left_yaw", "omni_left_yaw_deg", 60.0), omni_fov_h_deg, omni_fov_v_deg},
    read_cam_path("left", "omni_left_path", "video0"),
    {0, 255, 0}};
  cfg.right_cam = OmniCamConfig{
    {omniperception::OmniCameraSlot::right, "right",
     read_cam_path("right", "omni_right_path", "video2"),
     read_cli_or_yaml_double("right_yaw", "omni_right_yaw_deg", -60.0), omni_fov_h_deg,
     omni_fov_v_deg},
    read_cam_path("right", "omni_right_path", "video2"),
    {0, 255, 255}};
  cfg.back_cam = OmniCamConfig{
    {omniperception::OmniCameraSlot::back, "back", read_cam_path("back", "omni_back_path", "video4"),
     read_cli_or_yaml_double("back_yaw", "omni_back_yaw_deg", 180.0), omni_fov_h_deg, omni_fov_v_deg},
    read_cam_path("back", "omni_back_path", "video4"),
    {255, 200, 0}};
  cfg.display = !cli.has("no-display");
  return cfg;
}

}  // namespace ovsentry
