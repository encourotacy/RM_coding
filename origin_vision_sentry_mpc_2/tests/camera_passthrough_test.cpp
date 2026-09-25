#include <chrono>
#include <iostream>
#include <memory>

#include "io/camera.hpp"

namespace
{
class FakeCamera : public io::CameraBase
{
public:
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override
  {
    img = (cv::Mat_<uint8_t>(2, 3) << 1, 2, 3, 4, 5, 6);
    timestamp = expected_timestamp;
  }

  const std::chrono::steady_clock::time_point expected_timestamp{
    std::chrono::milliseconds(1234)};
};
}  // namespace

int main()
{
  auto backend = std::make_unique<FakeCamera>();
  const auto expected_timestamp = backend->expected_timestamp;
  io::Camera camera(std::move(backend));

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  camera.read(img, timestamp);

  if (timestamp != expected_timestamp) {
    std::cerr << "camera timestamp was changed" << std::endl;
    return 1;
  }
  if (img.rows != 2 || img.cols != 3) {
    std::cerr << "camera image dimensions were changed" << std::endl;
    return 1;
  }

  const cv::Mat expected = (cv::Mat_<uint8_t>(2, 3) << 1, 2, 3, 4, 5, 6);
  if (cv::countNonZero(img != expected) != 0) {
    std::cerr << "camera image was rotated, flipped, or otherwise modified" << std::endl;
    return 1;
  }
  return 0;
}
