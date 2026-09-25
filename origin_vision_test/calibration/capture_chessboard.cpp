#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/ovgimbal.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ?  |                                   | 输出命令行参数说明}"
  "{@config-path c  | configs/calibration_chessboard.yaml | yaml配置文件路径 }"
  "{o output-folder | assets/chessboard_calib          | 输出文件夹路径   }";

namespace
{
void write_q(const std::string & q_path, const Eigen::Quaterniond & q)
{
  std::ofstream q_file(q_path);
  Eigen::Vector4d xyzw = q.coeffs();
  q_file << fmt::format("{} {} {} {}", xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
}

bool detect_chessboard(
  const cv::Mat & img, const cv::Size & pattern_size, double preview_scale,
  std::vector<cv::Point2f> & corners)
{
  cv::Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

  cv::Mat search_gray = gray;
  if (preview_scale < 0.999) {
    cv::resize(gray, search_gray, {}, preview_scale, preview_scale, cv::INTER_AREA);
  }

  const int flags =
    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
  std::vector<cv::Point2f> search_corners;
  bool found = cv::findChessboardCorners(search_gray, pattern_size, search_corners, flags);
  if (!found) {
    corners.clear();
    return false;
  }

  corners = search_corners;
  if (preview_scale < 0.999) {
    for (auto & corner : corners) {
      corner.x /= preview_scale;
      corner.y /= preview_scale;
    }
  }

  cv::cornerSubPix(
    gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
    cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01));
  return true;
}
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const std::string config_path = cli.get<std::string>(0);
  const std::string output_folder = cli.get<std::string>("o");
  std::filesystem::create_directories(output_folder);

  auto yaml = YAML::LoadFile(config_path);
  const int pattern_cols = yaml["pattern_cols"].as<int>();
  const int pattern_rows = yaml["pattern_rows"].as<int>();
  const double square_size_mm = yaml["square_size_mm"].as<double>();
  const double preview_scale = yaml["preview_scale"].as<double>(0.5);
  const int detect_interval = yaml["detect_interval"].as<int>(2);
  const cv::Size pattern_size(pattern_cols, pattern_rows);

  io::OVGimbal gimbal(config_path);
  io::Camera camera(config_path);
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  int count = 0;
  int frame_id = 0;
  bool found = false;
  std::vector<cv::Point2f> corners;

  tools::logger()->info(
    "标定板参数: {}x{} 内角点, 方格大小 {}mm", pattern_cols, pattern_rows, square_size_mm);

  while (true) {
    camera.read(img, timestamp);
    frame_id++;

    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    const bool imu_ready = gimbal.has_imu();
    if (imu_ready) {
      q = gimbal.imu_at_image(timestamp);
    }

    if (frame_id % std::max(1, detect_interval) == 0 || corners.empty()) {
      found = detect_chessboard(img, pattern_size, preview_scale, corners);
    }

    cv::Mat preview = img.clone();
    cv::drawChessboardCorners(preview, pattern_size, corners, found);
    Eigen::Vector3d zyx = tools::eulers(q, 2, 1, 0) * 57.3;
    tools::draw_text(preview, fmt::format("Z {:.2f}", zyx[0]), {40, 40}, {0, 0, 255});
    tools::draw_text(preview, fmt::format("Y {:.2f}", zyx[1]), {40, 80}, {0, 0, 255});
    tools::draw_text(preview, fmt::format("X {:.2f}", zyx[2]), {40, 120}, {0, 0, 255});
    tools::draw_text(
      preview, imu_ready ? "IMU ready" : "IMU waiting", {40, 160},
      imu_ready ? cv::Scalar(0, 180, 0) : cv::Scalar(0, 0, 255));
    tools::draw_text(
      preview, found ? "Chessboard found" : "Chessboard lost", {40, 200},
      found ? cv::Scalar(0, 180, 0) : cv::Scalar(0, 0, 255));
    tools::draw_text(preview, fmt::format("Saved {}", count), {40, 240}, {255, 128, 0});
    tools::draw_text(
      preview, fmt::format("Detect every {} frame", std::max(1, detect_interval)), {40, 280},
      {255, 128, 0});
    cv::resize(preview, preview, {}, preview_scale, preview_scale, cv::INTER_AREA);

    cv::imshow("Press s to save, q to quit", preview);
    const int key = cv::waitKey(1);
    if (key == 'q') break;
    if (key != 's') continue;

    if (!found) {
      tools::logger()->warn("未检测到棋盘格，忽略本次保存");
      continue;
    }
    if (!imu_ready) {
      tools::logger()->warn("IMU 尚未就绪，忽略本次保存");
      continue;
    }

    ++count;
    const std::string img_path = fmt::format("{}/{}.jpg", output_folder, count);
    const std::string q_path = fmt::format("{}/{}.txt", output_folder, count);
    cv::imwrite(img_path, img);
    write_q(q_path, q);
    tools::logger()->info("[{}] Saved in {}", count, output_folder);
  }

  tools::logger()->warn("注意四元数输出顺序为wxyz");
  return 0;
}
