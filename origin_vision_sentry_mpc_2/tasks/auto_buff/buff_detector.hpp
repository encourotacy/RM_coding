#ifndef AUTO_BUFF__TRACK_HPP
#define AUTO_BUFF__TRACK_HPP

#include <yaml-cpp/yaml.h>

#include <deque>
#include <memory>
#include <optional>
#include <vector>

#include "buff_type.hpp"
#include "tools/img_tools.hpp"
#include "yolo11_buff.hpp"
const int LOSE_MAX = 20;  // 丢失的阙值
namespace auto_buff
{
class Target;

class Buff_Detector
{
public:
  explicit Buff_Detector(const std::string & config);
  Buff_Detector(const std::string & config, const std::string & model_key);
  ~Buff_Detector();

  void setEnergyType(EnergyType type);
  void setTarget(std::shared_ptr<Target> target_ptr) { target_ = std::move(target_ptr); }
  EnergyType getEnergyType() const { return energy_type_; }

  std::optional<PowerRune> detect_24(cv::Mat & bgr_img);

  std::optional<PowerRune> detect(cv::Mat & bgr_img);

std::optional<PowerRune> detect_debug(cv::Mat & bgr_img, cv::Point2f v);

private:
  void handle_img(const cv::Mat & bgr_img, cv::Mat & dilated_img);

  cv::Point2f get_r_center(std::vector<FanBlade> & fanblades, cv::Mat & bgr_img);

  void handle_lose();

  void fill_board_metadata(PowerRune & rune, const std::vector<YOLO11_BUFF::Object> & results) const;

  YOLO11_BUFF MODE_;
  Track_status status_;
  int lose_;  // 丢失的次数
  double lastlen_;
  std::optional<PowerRune> last_powerrune_ = std::nullopt;
  EnergyType energy_type_ = EnergyType::SMALL;
  std::shared_ptr<Target> target_;
};
}  // namespace auto_buff
#endif  // DETECTOR_HPP