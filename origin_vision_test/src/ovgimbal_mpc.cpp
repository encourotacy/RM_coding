#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <thread>

#include "io/camera.hpp"
#include "io/ovgimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_safe_queue.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                           | 输出命令行参数说明}"
  "{@config-path   | configs/standard4.yaml   | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  tools::Plotter plotter;

  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  io::OVGimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::atomic<bool> quit = false;
  std::atomic<float> plan_yaw = 0.0f;
  std::atomic<float> plan_pitch = 0.0f;
  std::atomic<float> plan_yaw_vel = 0.0f;
  std::atomic<float> plan_pitch_vel = 0.0f;
  std::atomic<float> plan_yaw_acc = 0.0f;
  std::atomic<float> plan_pitch_acc = 0.0f;
  std::atomic<bool> plan_fire = false;

  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();

    while (!quit) {
      auto target = target_queue.front();
      auto plan = planner.plan(target, gimbal.bullet_speed());

      plan_fire.store(plan.fire);
      plan_yaw.store(plan.yaw);
      plan_pitch.store(plan.pitch);
      plan_yaw_vel.store(plan.yaw_vel);
      plan_pitch_vel.store(plan.pitch_vel);
      plan_yaw_acc.store(plan.yaw_acc);
      plan_pitch_acc.store(plan.pitch_acc);

      gimbal.send(io::Command{plan.control, plan.fire, plan.yaw, plan.pitch});

      auto gs = gimbal.state();
      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
      data["gimbal_yaw"] = gs.yaw;
      data["gimbal_pitch"] = gs.pitch;
      data["gimbal_yaw_vel"] = gs.yaw_vel;
      data["gimbal_pitch_vel"] = gs.pitch_vel;
      data["target_yaw"] = plan.target_yaw;
      data["target_pitch"] = plan.target_pitch;
      data["plan_yaw"] = plan.yaw;
      data["plan_yaw_vel"] = plan.yaw_vel;
      data["plan_yaw_acc"] = plan.yaw_acc;
      data["plan_pitch"] = plan.pitch;
      data["plan_pitch_vel"] = plan.pitch_vel;
      data["plan_pitch_acc"] = plan.pitch_acc;
      data["fire"] = plan.fire ? 1 : 0;
      data["fired"] = plan.fire ? 1 : 0;

      if (target.has_value()) {
        data["target_z"] = target->ekf_x()[4];
        data["target_vz"] = target->ekf_x()[5];
        data["w"] = target->ekf_x()[7];
      } else {
        data["w"] = 0.0;
      }

      plotter.plot(data);
      std::this_thread::sleep_for(10ms);
    }
  });

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  int frame_count = 0;

  while (!exiter.exit()) {
    camera.read(img, t);
    frame_count++;

    auto q = gimbal.imu_at_image(t);
    solver.set_R_gimbal2world(q);
    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
    auto gs = gimbal.state();

    auto armors = yolo.detect(img, frame_count);
    auto targets = tracker.track(armors, t);
    if (!targets.empty())
      target_queue.push(targets.front());
    else
      target_queue.push(std::nullopt);

    if (!targets.empty()) {
      auto target = targets.front();
      for (const Eigen::Vector4d & xyza : target.armor_xyza_list()) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      Eigen::Vector4d aim_xyza = planner.debug_xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      tools::draw_points(img, image_points, {0, 0, 255});
    }

    tools::draw_text(
      img,
      fmt::format(
        "Yaw/Pitch: {:.2f}/{:.2f} deg | Vel: {:.2f}/{:.2f} deg/s", ypr[0] * 57.3, ypr[1] * 57.3,
        gs.yaw_vel * 57.3, gs.pitch_vel * 57.3),
      {10, 30}, {255, 255, 255});
    tools::draw_text(
      img,
      fmt::format(
        "MPC yaw/pitch: {:.2f}/{:.2f} deg | fire={}", plan_yaw.load() * 57.3,
        plan_pitch.load() * 57.3, plan_fire.load() ? 1 : 0),
      {10, 60}, {154, 50, 205});
    tools::draw_text(
      img, fmt::format("Tracker={}", tracker.state()), {10, 90}, {0, 255, 0});
    tools::draw_text(
      img,
      fmt::format(
        "MPC vel: {:.2f}/{:.2f} deg/s | acc: {:.2f}/{:.2f} deg/s^2",
        plan_yaw_vel.load() * 57.3, plan_pitch_vel.load() * 57.3, plan_yaw_acc.load() * 57.3,
        plan_pitch_acc.load() * 57.3),
      {10, 120}, {0, 255, 255});

    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("ovgimbal_mpc", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(io::Command{false, false, 0.0, 0.0});

  return 0;
}
