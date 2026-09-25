#include <fmt/core.h>
#include <fmt/ranges.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <utility>

#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ? |                                  | 输出命令行参数说明}"
  "{config-path c  | configs/calibration_chessboard.yaml | yaml配置文件路径 }"
  "{@input-folder  | assets/chessboard_calib          | 输入文件夹路径   }";

std::vector<cv::Point3f> chessboard_corners_3d(
  const cv::Size & pattern_size, const float square_size)
{
  std::vector<cv::Point3f> corners_3d;

  for (int i = 0; i < pattern_size.height; i++)
    for (int j = 0; j < pattern_size.width; j++)
      corners_3d.push_back({j * square_size, i * square_size, 0});

  return corners_3d;
}

bool read_q(const std::string & q_path, Eigen::Quaterniond & q)
{
  std::ifstream q_file(q_path);
  double w, x, y, z;
  if (!(q_file >> w >> x >> y >> z)) return false;

  q = Eigen::Quaterniond(w, x, y, z);
  if (!q.coeffs().allFinite() || q.norm() < 1e-9) return false;
  q.normalize();
  return true;
}

void load(
  const std::string & input_folder, const std::string & config_path,
  std::vector<Eigen::Matrix3d> & R_imubody2imuabs_list, std::vector<cv::Mat> & rvecs,
  std::vector<cv::Mat> & tvecs)
{
  auto yaml = YAML::LoadFile(config_path);
  auto pattern_cols = yaml["pattern_cols"].as<int>();
  auto pattern_rows = yaml["pattern_rows"].as<int>();
  auto square_size_mm = yaml["square_size_mm"].as<double>();
  auto preview_scale = yaml["preview_scale"].as<double>(0.5);
  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();

  cv::Size pattern_size(pattern_cols, pattern_rows);
  cv::Matx33d camera_matrix(camera_matrix_data.data());
  cv::Mat distort_coeffs(distort_coeffs_data);

  fmt::print("标定板参数: {}x{} 内角点, 方格大小 {}mm\n", pattern_cols, pattern_rows, square_size_mm);
  fmt::print("开始加载标定数据...\n\n");

  int success_count = 0;
  for (int i = 1; true; i++) {
    auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
    auto q_path = fmt::format("{}/{}.txt", input_folder, i);
    auto img = cv::imread(img_path);
    if (img.empty()) break;

    Eigen::Quaterniond q;
    if (!read_q(q_path, q)) {
      fmt::print("[failure] {} - 四元数文件无效，跳过\n", q_path);
      continue;
    }

    Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
    Eigen::Vector3d ypr = tools::eulers(R_imubody2imuabs, 2, 1, 0) * 180.0 / CV_PI;

    auto drawing = img.clone();
    tools::draw_text(drawing, fmt::format("yaw   {:.2f}", ypr[0]), {40, 40}, {0, 0, 255});
    tools::draw_text(drawing, fmt::format("pitch {:.2f}", ypr[1]), {40, 80}, {0, 0, 255});
    tools::draw_text(drawing, fmt::format("roll  {:.2f}", ypr[2]), {40, 120}, {0, 0, 255});

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat search_gray = gray;
    if (preview_scale < 0.999) {
      cv::resize(gray, search_gray, {}, preview_scale, preview_scale, cv::INTER_AREA);
    }

    std::vector<cv::Point2f> corners_2d;
    std::vector<cv::Point2f> search_corners;
    int flags =
      cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
    auto success = cv::findChessboardCorners(search_gray, pattern_size, search_corners, flags);

    if (success) {
      corners_2d = search_corners;
      if (preview_scale < 0.999) {
        for (auto & corner : corners_2d) {
          corner.x /= preview_scale;
          corner.y /= preview_scale;
        }
      }

      cv::cornerSubPix(
        gray, corners_2d, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
    }

    cv::drawChessboardCorners(drawing, pattern_size, corners_2d, success);
    cv::resize(drawing, drawing, {}, 0.5, 0.5);
    cv::imshow("Press any key to continue", drawing);
    cv::waitKey(0);

    fmt::print(
      "[{}] {} - {}\n", success ? "success" : "failure", img_path,
      success ? "已添加到标定数据" : "跳过");
    if (!success) continue;

    success_count++;

    cv::Mat rvec, tvec;
    auto corners_3d = chessboard_corners_3d(pattern_size, square_size_mm);
    cv::solvePnP(
      corners_3d, corners_2d, camera_matrix, distort_coeffs, rvec, tvec, false,
      cv::SOLVEPNP_ITERATIVE);

    R_imubody2imuabs_list.emplace_back(R_imubody2imuabs);
    rvecs.emplace_back(rvec);
    tvecs.emplace_back(tvec);
  }

  fmt::print("\n成功加载 {} 组标定数据\n", success_count);
  if (success_count < 10) {
    fmt::print("警告: 标定数据量较少（建议至少15-20组），可能影响标定精度\n");
  }
}

struct HandeyeResult
{
  Eigen::Matrix3d R_gimbal2imubody;
  cv::Mat R_camera2gimbal;
  cv::Mat t_camera2gimbal;
  Eigen::Vector3d ypr;
  double offset_angle_deg;
};

std::vector<Eigen::Matrix3d> axis_mapping_candidates()
{
  std::vector<Eigen::Matrix3d> candidates;
  std::array<int, 3> permutation{0, 1, 2};

  do {
    for (int sx : {-1, 1}) {
      for (int sy : {-1, 1}) {
        for (int sz : {-1, 1}) {
          Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
          const std::array<int, 3> signs{sx, sy, sz};
          for (int row = 0; row < 3; row++) R(row, permutation[row]) = signs[row];

          // Only proper rotations are allowed. Reflections have determinant -1.
          if (R.determinant() > 0.5) candidates.emplace_back(R);
        }
      }
    }
  } while (std::next_permutation(permutation.begin(), permutation.end()));

  return candidates;
}

std::vector<double> matrix_row_major_data(const Eigen::Matrix3d & R)
{
  std::vector<double> data;
  data.reserve(9);
  for (int row = 0; row < 3; row++)
    for (int col = 0; col < 3; col++) data.emplace_back(R(row, col));
  return data;
}

HandeyeResult calibrate_with_axis_mapping(
  const Eigen::Matrix3d & R_gimbal2imubody,
  const std::vector<Eigen::Matrix3d> & R_imubody2imuabs_list,
  const std::vector<cv::Mat> & rvecs, const std::vector<cv::Mat> & tvecs)
{
  std::vector<cv::Mat> R_gimbal2world_list;
  std::vector<cv::Mat> t_gimbal2world_list;
  R_gimbal2world_list.reserve(R_imubody2imuabs_list.size());
  t_gimbal2world_list.reserve(R_imubody2imuabs_list.size());

  for (const auto & R_imubody2imuabs : R_imubody2imuabs_list) {
    Eigen::Matrix3d R_gimbal2world =
      R_gimbal2imubody.transpose() * R_imubody2imuabs * R_gimbal2imubody;
    cv::Mat R_gimbal2world_cv;
    cv::eigen2cv(R_gimbal2world, R_gimbal2world_cv);
    R_gimbal2world_list.emplace_back(R_gimbal2world_cv);
    t_gimbal2world_list.emplace_back(cv::Mat::zeros(3, 1, CV_64F));
  }

  HandeyeResult result;
  result.R_gimbal2imubody = R_gimbal2imubody;
  cv::calibrateHandEye(
    R_gimbal2world_list, t_gimbal2world_list, rvecs, tvecs, result.R_camera2gimbal,
    result.t_camera2gimbal);
  result.t_camera2gimbal /= 1e3;

  Eigen::Matrix3d R_camera2gimbal_eigen;
  cv::cv2eigen(result.R_camera2gimbal, R_camera2gimbal_eigen);
  const Eigen::Matrix3d R_gimbal2ideal{{0, -1, 0}, {0, 0, -1}, {1, 0, 0}};
  const Eigen::Matrix3d R_camera2ideal = R_gimbal2ideal * R_camera2gimbal_eigen;
  result.ypr = tools::eulers(R_camera2ideal, 1, 0, 2) * 180.0 / CV_PI;

  // Use the coordinate-independent SO(3) angle as the total installation-offset score.
  const double cos_angle = std::clamp((R_camera2ideal.trace() - 1.0) / 2.0, -1.0, 1.0);
  result.offset_angle_deg = std::acos(cos_angle) * 180.0 / CV_PI;
  return result;
}

void print_yaml(
  const std::vector<double> & R_gimbal2imubody_data, const cv::Mat & R_camera2gimbal,
  const cv::Mat & t_camera2gimbal, const Eigen::Vector3d & ypr)
{
  YAML::Emitter result;
  std::vector<double> R_camera2gimbal_data(
    R_camera2gimbal.begin<double>(), R_camera2gimbal.end<double>());
  std::vector<double> t_camera2gimbal_data(
    t_camera2gimbal.begin<double>(), t_camera2gimbal.end<double>());

  result << YAML::BeginMap;
  result << YAML::Key << "R_gimbal2imubody";
  result << YAML::Value << YAML::Flow << R_gimbal2imubody_data;
  result << YAML::Newline;
  result << YAML::Newline;
  result << YAML::Comment(fmt::format(
    "相机同理想情况的偏角: yaw{:.2f} pitch{:.2f} roll{:.2f} degree", ypr[0], ypr[1], ypr[2]));
  result << YAML::Key << "R_camera2gimbal";
  result << YAML::Value << YAML::Flow << R_camera2gimbal_data;
  result << YAML::Key << "t_camera2gimbal";
  result << YAML::Value << YAML::Flow << t_camera2gimbal_data;
  result << YAML::Newline;
  result << YAML::EndMap;

  fmt::print("\n==================== 标定结果 ====================\n");
  fmt::print("{}\n", result.c_str());
  fmt::print("==================================================\n\n");
  fmt::print("请将上述结果复制到你的机器人配置文件中（如 configs/standard4.yaml）\n");
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto input_folder = cli.get<std::string>(0);
  auto config_path = cli.get<std::string>("config-path");

  fmt::print("\n=== 棋盘格手眼标定程序 ===\n\n");

  std::vector<Eigen::Matrix3d> R_imubody2imuabs_list;
  std::vector<cv::Mat> rvecs, tvecs;
  load(input_folder, config_path, R_imubody2imuabs_list, rvecs, tvecs);

  if (R_imubody2imuabs_list.empty()) {
    fmt::print("错误: 没有有效的标定数据\n");
    return -1;
  }

  const auto candidates = axis_mapping_candidates();
  std::vector<HandeyeResult> results;
  results.reserve(candidates.size());

  fmt::print("\n开始遍历全部 {} 个合法轴映射...\n", candidates.size());
  for (std::size_t i = 0; i < candidates.size(); i++) {
    try {
      auto result =
        calibrate_with_axis_mapping(candidates[i], R_imubody2imuabs_list, rvecs, tvecs);
      if (!std::isfinite(result.offset_angle_deg) || !cv::checkRange(result.R_camera2gimbal) ||
          !cv::checkRange(result.t_camera2gimbal)) {
        fmt::print("[{:02}] 数值无效，已跳过\n", i + 1);
        continue;
      }

      const auto mapping_data = matrix_row_major_data(result.R_gimbal2imubody);
      fmt::print(
        "[{:02}] 总偏角={:7.3f}°, yaw={:7.2f}°, pitch={:7.2f}°, roll={:7.2f}°, "
        "R=[{}]\n",
        i + 1, result.offset_angle_deg, result.ypr[0], result.ypr[1], result.ypr[2],
        fmt::join(mapping_data, ", "));
      results.emplace_back(std::move(result));
    } catch (const cv::Exception & e) {
      fmt::print("[{:02}] OpenCV 标定失败，已跳过: {}\n", i + 1, e.what());
    }
  }

  if (results.empty()) {
    fmt::print("错误: 所有轴映射均标定失败\n");
    return -1;
  }

  const auto best = std::min_element(
    results.begin(), results.end(), [](const HandeyeResult & a, const HandeyeResult & b) {
      return a.offset_angle_deg < b.offset_angle_deg;
    });

  const auto R_gimbal2imubody_data = matrix_row_major_data(best->R_gimbal2imubody);
  fmt::print(
    "\n最优轴映射: 总偏角={:.3f}°, yaw={:.2f}°, pitch={:.2f}°, roll={:.2f}°\n",
    best->offset_angle_deg, best->ypr[0], best->ypr[1], best->ypr[2]);
  print_yaml(
    R_gimbal2imubody_data, best->R_camera2gimbal, best->t_camera2gimbal, best->ypr);

  fmt::print("提示: 相机偏角表示相机安装的理想程度\n");
  fmt::print("      如果偏角较大（>5度），建议调整相机安装位置\n");

  return 0;
}