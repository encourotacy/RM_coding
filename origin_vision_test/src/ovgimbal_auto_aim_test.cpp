#include <fmt/core.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/ovgimbal.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

const std::string keys =
  "{help h usage ? |                           | 输出命令行参数说明}"
  "{@config-path   | configs/standard4.yaml   | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  io::OVGimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  constexpr bool aimer_to_now = true;

  tools::logger()->info(
    "[OVGimbal] Minimal chain enabled: OVGimbal + Camera + YOLO + Tracker + Aimer + Shooter");
  tools::logger()->info("[OVGimbal] timing config: offset={:.2f}ms", gimbal.offset_ms());

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  int frame_count = 0;

  while (!exiter.exit()) {
    try {
      camera.read(img, timestamp);
      if (img.empty()) {
        tools::logger()->warn("读取到空图像，跳过此帧");
        continue;
      }
    } catch (const std::exception & e) {
      tools::logger()->error("读取摄像机图像失败: {}", e.what());
      continue;
    }

    frame_count++;

    Eigen::Quaterniond q = gimbal.imu_at_image(timestamp);
    solver.set_R_gimbal2world(q);
    recorder.record(img, q, timestamp);

    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
    double roll = ypr[2] * 57.3;
    double pitch = ypr[1] * 57.3;
    double yaw = ypr[0] * 57.3;

    auto yolo_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);
    auto yolo_end = std::chrono::steady_clock::now();

    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, timestamp);
    auto tracker_end = std::chrono::steady_clock::now();

    auto aimer_start = std::chrono::steady_clock::now();
    auto command = aimer.aim(targets, timestamp, gimbal.bullet_speed(), aimer_to_now);
    auto aimer_end = std::chrono::steady_clock::now();

    command.shoot = shooter.shoot(command, aimer, targets, ypr, tracker.state() == "tracking");
    gimbal.send(command);

    double yolo_time = tools::delta_time(yolo_end, yolo_start) * 1e3;
    double tracker_time = tools::delta_time(tracker_end, tracker_start) * 1e3;
    double aimer_time = tools::delta_time(aimer_end, aimer_start) * 1e3;
    double total_time = tools::delta_time(aimer_end, yolo_start) * 1e3;

    tools::draw_text(img, fmt::format("[{}]", tracker.state()), {10, 30}, {255, 255, 255});
    tools::draw_text(
      img,
      fmt::format(
        "Cmd: yaw={:.2f}, pitch={:.2f}, shoot={}", command.yaw * 57.3, command.pitch * 57.3,
        command.shoot),
      {10, 60}, {154, 50, 205});
    tools::draw_text(
      img, fmt::format("Attitude: R={:.1f}, P={:.1f}, Y={:.1f}", roll, pitch, yaw), {10, 90},
      {255, 255, 255});
    tools::draw_text(
      img, fmt::format("Time: YOLO={:.1f}ms, Total={:.1f}ms", yolo_time, total_time), {10, 120},
      {0, 255, 255});

    nlohmann::json data;
    data["armor_num"] = armors.size();
    if (!armors.empty()) {
      auto min_x = 1e10;
      auto & armor = armors.front();
      for (auto & a : armors) {
        if (a.center.x < min_x) {
          min_x = a.center.x;
          armor = a;
        }
      }
      solver.solve(armor);
      data["armor_x"] = armor.xyz_in_world[0];
      data["armor_y"] = armor.xyz_in_world[1];
      data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
      data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
    }

    if (!targets.empty()) {
      auto target = targets.front();
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      auto aim_point = aimer.debug_aim_point;
      Eigen::Vector4d aim_xyza = aim_point.xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      if (aim_point.valid)
        tools::draw_points(img, image_points, {0, 0, 255});
      else
        tools::draw_points(img, image_points, {255, 0, 0});

      Eigen::VectorXd x = target.ekf_x();
      data["x"] = x[0];
      data["vx"] = x[1];
      data["y"] = x[2];
      data["vy"] = x[3];
      data["z"] = x[4];
      data["vz"] = x[5];
      data["a"] = x[6] * 57.3;
      data["w"] = x[7];
      data["r"] = x[8];
      data["l"] = x[9];
      data["h"] = x[10];
      data["last_id"] = target.last_id;
    }

    data["gimbal_roll"] = roll;
    data["gimbal_pitch"] = pitch;
    data["gimbal_yaw"] = yaw;
    data["bullet_speed"] = gimbal.bullet_speed();
    if (command.control) {
      data["cmd_yaw"] = command.yaw * 57.3;
      data["cmd_pitch"] = command.pitch * 57.3;
      data["cmd_shoot"] = command.shoot;
    }

    data["yolo_time"] = yolo_time;
    data["tracker_time"] = tracker_time;
    data["aimer_time"] = aimer_time;
    data["total_time"] = total_time;

    plotter.plot(data);

    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("OVGimbal Auto Aim Test", img);
    auto key = cv::waitKey(30);
    if (key == 'q') break;
  }

  return 0;
}
