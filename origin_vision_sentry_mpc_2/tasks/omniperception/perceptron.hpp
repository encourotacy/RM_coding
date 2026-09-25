#ifndef OMNIPERCEPTION__PERCEPTRON_HPP
#define OMNIPERCEPTION__PERCEPTRON_HPP

#include <chrono>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "decider.hpp"
#include "detection.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tools/thread_pool.hpp"
#include "tools/thread_safe_queue.hpp"

namespace omniperception
{

class Perceptron
{
public:
  struct WorkerConfig
  {
    io::USBCamera * camera = nullptr;
    std::string camera_label;
    std::optional<CameraSpec> camera_spec;
    std::string infer_device_key;
    std::function<double()> base_yaw_provider;
  };

  struct DebugSnapshot
  {
    CameraSpec spec;
    cv::Mat image;
    std::chrono::steady_clock::time_point timestamp{};
    std::optional<auto_aim::Armor> top_armor;
    double delta_yaw_deg = 0.0;
    double delta_pitch_deg = 0.0;
    double infer_ms = 0.0;
    double base_yaw_rad = 0.0;
    double abs_yaw_rad = 0.0;
    bool has_base_yaw = false;
    bool has_abs_yaw = false;
    bool has_detection = false;
    bool camera_online = false;
    int consecutive_timeout_count = 0;
  };

  Perceptron(
    io::USBCamera * usbcma1, io::USBCamera * usbcam2, io::USBCamera * usbcam3,
    io::USBCamera * usbcam4, const std::string & config_path);
  Perceptron(const std::vector<WorkerConfig> & workers, const std::string & config_path);

  ~Perceptron();

  std::vector<DetectionResult> get_detection_queue();
  std::vector<DebugSnapshot> debug_snapshots() const;

  void parallel_infer(const WorkerConfig & worker, std::shared_ptr<auto_aim::YOLO> & yolo_parallel);

private:
  std::vector<std::thread> threads_;
  tools::ThreadSafeQueue<DetectionResult> detection_queue_;
  std::vector<WorkerConfig> worker_configs_;
  std::vector<DebugSnapshot> debug_snapshots_;

  std::vector<std::shared_ptr<auto_aim::YOLO>> yolo_workers_;

  Decider decider_;
  bool stop_flag_;
  std::chrono::milliseconds read_timeout_{10};
  mutable std::mutex mutex_;
  std::condition_variable condition_;
};

}  // namespace omniperception
#endif
