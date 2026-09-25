#include "perceptron.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>

#include <yaml-cpp/yaml.h>

#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace omniperception
{
Perceptron::Perceptron(
  io::USBCamera * usbcam1, io::USBCamera * usbcam2, io::USBCamera * usbcam3,
  io::USBCamera * usbcam4, const std::string & config_path)
: Perceptron(
    std::vector<WorkerConfig>{
      WorkerConfig{usbcam1, "left", std::nullopt, "", {}},
      WorkerConfig{usbcam2, "right", std::nullopt, "", {}},
      WorkerConfig{usbcam3, "back", std::nullopt, "", {}},
      WorkerConfig{usbcam4, "back", std::nullopt, "", {}}},
    config_path)
{}

Perceptron::Perceptron(const std::vector<WorkerConfig> & workers, const std::string & config_path)
: detection_queue_(10), worker_configs_(workers), decider_(config_path), stop_flag_(false)
{
  auto yaml = YAML::LoadFile(config_path);
  const auto read_timeout_ms =
    yaml["omni_camera_read_timeout_ms"] ? yaml["omni_camera_read_timeout_ms"].as<int>() : 10;
  read_timeout_ = std::chrono::milliseconds(std::max(read_timeout_ms, 1));
  debug_snapshots_.resize(worker_configs_.size());
  yolo_workers_.reserve(worker_configs_.size());

  for (size_t i = 0; i < worker_configs_.size(); ++i) {
    const auto & worker = worker_configs_[i];
    if (worker.infer_device_key.empty()) {
      yolo_workers_.push_back(std::make_shared<auto_aim::YOLO>(config_path, false));
    } else {
      yolo_workers_.push_back(
        std::make_shared<auto_aim::YOLO>(config_path, false, worker.infer_device_key));
    }

    if (worker.camera_spec.has_value()) {
      debug_snapshots_[i].spec = worker.camera_spec.value();
    } else {
      debug_snapshots_[i].spec.label = worker.camera_label;
    }
  }

  std::this_thread::sleep_for(std::chrono::seconds(2));
  for (size_t i = 0; i < worker_configs_.size(); ++i) {
    threads_.emplace_back([this, i] { parallel_infer(worker_configs_[i], yolo_workers_[i]); });
  }

  tools::logger()->info("Perceptron initialized with {} workers.", worker_configs_.size());
}

Perceptron::~Perceptron()
{
  {
    std::unique_lock<std::mutex> lock(mutex_);
    stop_flag_ = true;  // 设置退出标志
  }
  condition_.notify_all();  // 唤醒所有等待的线程

  // 等待线程结束
  for (auto & t : threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
  tools::logger()->info("Perceptron destructed.");
}

std::vector<DetectionResult> Perceptron::get_detection_queue()
{
  std::vector<DetectionResult> result;
  DetectionResult temp;

  // 注意：这里的 pop 不阻塞（假设队列为空时会报错或忽略）
  while (!detection_queue_.empty()) {
    detection_queue_.pop(temp);
    result.push_back(std::move(temp));
  }

  return result;
}

std::vector<Perceptron::DebugSnapshot> Perceptron::debug_snapshots() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return debug_snapshots_;
}

// 将并行推理逻辑移动到类成员函数
void Perceptron::parallel_infer(
  const WorkerConfig & worker, std::shared_ptr<auto_aim::YOLO> & yolov8_parallel)
{
  auto * cam = worker.camera;
  if (!cam) {
    tools::logger()->error("Camera pointer is null!");
    return;
  }
  auto update_snapshot = [&](const cv::Mat & image, std::chrono::steady_clock::time_point timestamp,
                             const std::optional<auto_aim::Armor> & top_armor, double delta_yaw_deg,
                             double delta_pitch_deg, double infer_ms, double base_yaw_rad,
                             double abs_yaw_rad, bool has_base_yaw, bool has_abs_yaw, bool has_detection,
                             bool camera_online, int timeout_count) {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = std::find_if(
        worker_configs_.begin(), worker_configs_.end(),
        [&](const auto & candidate) { return candidate.camera == worker.camera; });
      if (it == worker_configs_.end()) return;

      const size_t index = static_cast<size_t>(std::distance(worker_configs_.begin(), it));
      auto & snapshot = debug_snapshots_[index];
      if (worker.camera_spec.has_value()) {
        snapshot.spec = worker.camera_spec.value();
      } else {
        snapshot.spec.label = worker.camera_label;
      }
      snapshot.image = image.empty() ? cv::Mat() : image.clone();
      snapshot.timestamp = timestamp;
      snapshot.top_armor = top_armor;
      snapshot.delta_yaw_deg = delta_yaw_deg;
      snapshot.delta_pitch_deg = delta_pitch_deg;
      snapshot.infer_ms = infer_ms;
      snapshot.base_yaw_rad = base_yaw_rad;
      snapshot.abs_yaw_rad = abs_yaw_rad;
      snapshot.has_base_yaw = has_base_yaw;
      snapshot.has_abs_yaw = has_abs_yaw;
      snapshot.has_detection = has_detection;
      snapshot.camera_online = camera_online;
      snapshot.consecutive_timeout_count = timeout_count;
    };

  int consecutive_timeout_count = 0;
  while (true) {
    cv::Mat usb_img;
    std::chrono::steady_clock::time_point ts{};

    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (stop_flag_) break;  // 检查是否需要退出
    }

    bool frame_ready = false;
    try {
      frame_ready = cam->read_with_timeout(usb_img, ts, read_timeout_);
    } catch (const std::exception & e) {
      consecutive_timeout_count++;
      const double base_yaw_rad = worker.base_yaw_provider ? worker.base_yaw_provider() : 0.0;
      tools::logger()->warn(
        "[Perceptron:{}] read exception: {}", worker.camera_label.empty() ? cam->device_name : worker.camera_label,
        e.what());
      update_snapshot({}, std::chrono::steady_clock::now(), std::nullopt, 0.0, 0.0, 0.0, base_yaw_rad,
                      0.0, static_cast<bool>(worker.base_yaw_provider), false, false, false,
                      consecutive_timeout_count);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }

    if (!frame_ready || usb_img.empty()) {
      consecutive_timeout_count++;
      const double base_yaw_rad = worker.base_yaw_provider ? worker.base_yaw_provider() : 0.0;
      update_snapshot({}, std::chrono::steady_clock::now(), std::nullopt, 0.0, 0.0, 0.0, base_yaw_rad,
                      0.0, static_cast<bool>(worker.base_yaw_provider), false, false, false,
                      consecutive_timeout_count);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }

    consecutive_timeout_count = 0;
    const double base_yaw_rad = worker.base_yaw_provider ? worker.base_yaw_provider() : 0.0;
    const bool has_base_yaw = static_cast<bool>(worker.base_yaw_provider);

    try {
      const auto infer_begin = std::chrono::steady_clock::now();
      auto armors = yolov8_parallel->detect(usb_img);
      const auto infer_end = std::chrono::steady_clock::now();
      const double infer_ms = tools::delta_time(infer_end, infer_begin) * 1e3;
      auto debug_armors = armors;
      decider_.armor_filter(debug_armors);
      decider_.set_priority(debug_armors);
      debug_armors.sort(
        [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });
      std::optional<auto_aim::Armor> top_armor;
      if (!debug_armors.empty()) top_armor = debug_armors.front();

      double delta_yaw_deg = 0.0;
      double delta_pitch_deg = 0.0;
      if (!armors.empty()) {
        Eigen::Vector2d delta_angle;
        if (worker.camera_spec.has_value()) {
          delta_angle = decider_.delta_angle(armors, worker.camera_spec.value());
        } else {
          const std::string camera_label =
            worker.camera_label.empty() ? cam->device_name : worker.camera_label;
          delta_angle = decider_.delta_angle(armors, camera_label);
        }
        delta_yaw_deg = delta_angle[0];
        delta_pitch_deg = delta_angle[1];

        DetectionResult dr;
        dr.armors = std::move(armors);
        dr.timestamp = ts;
        dr.delta_yaw = delta_yaw_deg / 57.3;
        dr.delta_pitch = delta_pitch_deg / 57.3;
        dr.slot = worker.camera_spec.has_value() ? worker.camera_spec->slot : OmniCameraSlot::unknown;
        dr.camera_label = worker.camera_spec.has_value() ? worker.camera_spec->label : worker.camera_label;
        dr.base_yaw_rad = base_yaw_rad;
        dr.has_base_yaw = has_base_yaw;
        if (dr.has_base_yaw) {
          dr.abs_yaw_rad = dr.base_yaw_rad + dr.delta_yaw;
          dr.has_abs_yaw = true;
        }
        dr.infer_ms = infer_ms;
        detection_queue_.push(dr);  // 推入线程安全队列
      }

      const double abs_yaw_rad =
        (has_base_yaw && top_armor.has_value()) ? (base_yaw_rad + delta_yaw_deg / 57.3) : 0.0;
      update_snapshot(usb_img, ts, top_armor, delta_yaw_deg, delta_pitch_deg, infer_ms, base_yaw_rad,
                      abs_yaw_rad, has_base_yaw, has_base_yaw && top_armor.has_value(),
                      top_armor.has_value(), true, consecutive_timeout_count);
    } catch (const std::exception & e) {
      tools::logger()->warn(
        "[Perceptron:{}] infer exception: {}",
        worker.camera_label.empty() ? cam->device_name : worker.camera_label, e.what());
      update_snapshot(usb_img, ts, std::nullopt, 0.0, 0.0, 0.0, base_yaw_rad, 0.0, has_base_yaw,
                      false, false, true, consecutive_timeout_count);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
}

}  // namespace omniperception
