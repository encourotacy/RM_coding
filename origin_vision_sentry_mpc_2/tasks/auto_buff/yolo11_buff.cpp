#include "yolo11_buff.hpp"

#include <optional>
#include <stdexcept>

const double ConfidenceThreshold = 0.7f;
const double BigBuffConfidenceThreshold = 0.40f;
const double BigBuffIouThreshold = 0.60f;

namespace
{
struct DetectionLayout
{
  int channels = 0;
  int candidates = 0;
  bool channel_first = true;
  int keypoint_stride = 2;
  float confidence_threshold = static_cast<float>(ConfidenceThreshold);
};

struct DecodedBox
{
  cv::Rect rect;
  float confidence = 0.0F;
  std::vector<cv::Point2f> keypoints;
};

std::optional<DetectionLayout> detect_layout(const ov::Shape & shape, bool & logged)
{
  if (shape.size() != 3 || shape[0] != 1) return std::nullopt;

  DetectionLayout layout;
  if (shape[1] == 17 || shape[1] == 23) {
    layout.channels = static_cast<int>(shape[1]);
    layout.candidates = static_cast<int>(shape[2]);
    layout.channel_first = true;
  } else if (shape[2] == 17 || shape[2] == 23) {
    layout.channels = static_cast<int>(shape[2]);
    layout.candidates = static_cast<int>(shape[1]);
    layout.channel_first = false;
  } else {
    return std::nullopt;
  }

  if (layout.channels == 23) {
    layout.keypoint_stride = 3;
    layout.confidence_threshold = static_cast<float>(BigBuffConfidenceThreshold);
  }

  if (!logged) {
    tools::logger()->info(
      "[YOLO11_BUFF] output channels={} candidates={} format={}", layout.channels, layout.candidates,
      layout.keypoint_stride == 3 ? "xy_score" : "xy");
    logged = true;
  }
  return layout;
}

float output_at(const float * data, const DetectionLayout & layout, int channel, int candidate)
{
  if (layout.channel_first) return data[channel * layout.candidates + candidate];
  return data[candidate * layout.channels + channel];
}

std::vector<DecodedBox> decode_boxes(
  const ov::Tensor & output, float factor, int num_points, bool & logged)
{
  const auto layout = detect_layout(output.get_shape(), logged);
  if (!layout.has_value()) {
    tools::logger()->warn("[YOLO11_BUFF] 无法解析大符/小符模型输出");
    return {};
  }

  const float * data = output.data<const float>();
  std::vector<DecodedBox> boxes;
  for (int i = 0; i < layout->candidates; ++i) {
    const float score = output_at(data, *layout, 4, i);
    if (score <= layout->confidence_threshold) continue;

    DecodedBox box;
    const float cx = output_at(data, *layout, 0, i);
    const float cy = output_at(data, *layout, 1, i);
    const float ow = output_at(data, *layout, 2, i);
    const float oh = output_at(data, *layout, 3, i);
    box.rect.x = static_cast<int>((cx - 0.5F * ow) * factor);
    box.rect.y = static_cast<int>((cy - 0.5F * oh) * factor);
    box.rect.width = static_cast<int>(ow * factor);
    box.rect.height = static_cast<int>(oh * factor);
    box.confidence = score;
    box.keypoints.reserve(num_points);
    for (int j = 0; j < num_points; ++j) {
      const int base = 5 + j * layout->keypoint_stride;
      const float x = output_at(data, *layout, base, i) * factor;
      const float y = output_at(data, *layout, base + 1, i) * factor;
      box.keypoints.emplace_back(x, y);
    }
    boxes.push_back(std::move(box));
  }
  return boxes;
}
}  // namespace

namespace auto_buff
{
YOLO11_BUFF::YOLO11_BUFF(const std::string & config) : YOLO11_BUFF(config, "model") {}

YOLO11_BUFF::YOLO11_BUFF(const std::string & config, const std::string & model_key)
{
  auto yaml = YAML::LoadFile(config);
  std::string model_path;
  if (yaml[model_key]) {
    model_path = yaml[model_key].as<std::string>();
  } else if (yaml["model"]) {
    model_path = yaml["model"].as<std::string>();
    tools::logger()->warn(
      "[YOLO11_BUFF] {} 未配置，回退使用 model={}", model_key, model_path);
  } else {
    throw std::runtime_error("[YOLO11_BUFF] yaml 缺少 model 配置");
  }

  model = core.read_model(model_path);
  compiled_model = core.compile_model(model, "CPU");
  infer_request = compiled_model.create_infer_request();
  input_tensor = infer_request.get_input_tensor();
  input_tensor.set_shape({1, 3, 640, 640});
  tools::logger()->info("[YOLO11_BUFF] loaded {}={}", model_key, model_path);
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_multicandidateboxes(cv::Mat & image)
{
  const int64 start = cv::getTickCount();
  if (image.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return {};
  }

  const float factor = fill_tensor_data_image(input_tensor, image);
  infer_request.infer();
  const auto decoded = decode_boxes(
    infer_request.get_output_tensor(), factor, NUM_POINTS, output_layout_logged_);

  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  boxes.reserve(decoded.size());
  confidences.reserve(decoded.size());
  for (const auto & box : decoded) {
    boxes.push_back(box.rect);
    confidences.push_back(box.confidence);
  }

  std::vector<int> indexes;
  if (!boxes.empty()) {
    cv::dnn::NMSBoxes(
      boxes, confidences, static_cast<float>(BigBuffConfidenceThreshold),
      static_cast<float>(BigBuffIouThreshold), indexes);
  }

  std::vector<Object> object_result;
  for (const int index : indexes) {
    Object obj;
    obj.rect = decoded[index].rect;
    obj.prob = decoded[index].confidence;
    obj.kpt = decoded[index].keypoints;
    object_result.push_back(obj);

    cv::rectangle(image, obj.rect, cv::Scalar(255, 255, 255), 1, 8);
    const std::string label = "buff:" + std::to_string(obj.prob).substr(0, 4);
    const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
    const cv::Rect textBox(obj.rect.tl().x, obj.rect.tl().y - 15, textSize.width, textSize.height + 5);
    cv::rectangle(image, textBox, cv::Scalar(0, 255, 255), cv::FILLED);
    cv::putText(
      image, label, cv::Point(obj.rect.tl().x, obj.rect.tl().y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
      cv::Scalar(0, 0, 0));
    for (int i = 0; i < NUM_POINTS && i < static_cast<int>(obj.kpt.size()); ++i) {
      cv::circle(image, obj.kpt[i], 2, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
    }
  }

  const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  cv::putText(
    image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0,
    cv::Scalar(255, 0, 0), 2, 8);
  return object_result;
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_onecandidatebox(cv::Mat & image)
{
  const int64 start = cv::getTickCount();
  if (image.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return {};
  }

  const float factor = fill_tensor_data_image(input_tensor, image);
  infer_request.infer();
  const auto decoded = decode_boxes(
    infer_request.get_output_tensor(), factor, NUM_POINTS, output_layout_logged_);

  int best_index = -1;
  float max_confidence = 0.0F;
  for (size_t i = 0; i < decoded.size(); ++i) {
    if (decoded[i].confidence > max_confidence) {
      max_confidence = decoded[i].confidence;
      best_index = static_cast<int>(i);
    }
  }

  std::vector<Object> object_result;
  if (best_index >= 0) {
    Object obj;
    obj.rect = decoded[best_index].rect;
    obj.prob = decoded[best_index].confidence;
    obj.kpt = decoded[best_index].keypoints;
    object_result.push_back(obj);

    if (max_confidence < 0.7) save(std::to_string(start), image);
    cv::rectangle(image, obj.rect, cv::Scalar(255, 255, 255), 1, 8);
    const std::string label = "buff:" + std::to_string(max_confidence).substr(0, 4);
    const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
    const cv::Rect textBox(obj.rect.tl().x, obj.rect.tl().y - 15, textSize.width, textSize.height + 5);
    cv::rectangle(image, textBox, cv::Scalar(0, 255, 255), cv::FILLED);
    cv::putText(
      image, label, cv::Point(obj.rect.tl().x, obj.rect.tl().y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
      cv::Scalar(0, 0, 0));
    for (int i = 0; i < NUM_POINTS && i < static_cast<int>(obj.kpt.size()); ++i) {
      cv::circle(image, obj.kpt[i], 2, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
      cv::putText(
        image, std::to_string(i + 1), obj.kpt[i] + cv::Point2f(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
    }
  }

  const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  cv::putText(
    image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0,
    cv::Scalar(255, 0, 0), 2, 8);
  return object_result;
}

void YOLO11_BUFF::convert(
  const cv::Mat & input, cv::Mat & output, const bool normalize, const bool BGR2RGB) const
{
  input.convertTo(output, CV_32F);
  if (normalize) output = output / 255.0;  // 归一化到[0, 1]
  if (BGR2RGB) cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
}

float YOLO11_BUFF::fill_tensor_data_image(ov::Tensor & input_tensor, const cv::Mat & input_image) const
{
  /// letterbox变换: 不改变宽高比(aspect ratio), 将input_image缩放并放置到blob_image左上角
  const ov::Shape tensor_shape = input_tensor.get_shape();
  const size_t num_channels = tensor_shape[1];
  const size_t height = tensor_shape[2];
  const size_t width = tensor_shape[3];
  // 缩放因子
  const float scale = std::min(height / float(input_image.rows), width / float(input_image.cols));
  const cv::Matx23f matrix{
    scale, 0.0, 0.0, 0.0, scale, 0.0,
  };
  cv::Mat blob_image;
  // 下面根据scale范围进行数据转换, 这只是为了提高一点速度(主要是提高了交换通道的速度)
  // 如果不在意这点速度提升的可以固定一种做法(两个if分支随便一个都可以)
  if (scale < 1.0f) {
    // 要缩小, 那么先缩小再交换通道
    cv::warpAffine(input_image, blob_image, matrix, cv::Size(width, height));
    convert(blob_image, blob_image, true, true);
  } else {
    // 要放大, 那么先交换通道再放大
    convert(input_image, blob_image, true, true);
    cv::warpAffine(blob_image, blob_image, matrix, cv::Size(width, height));
  }

  /// 将图像数据填入input_tensor
  float * const input_tensor_data = input_tensor.data<float>();
  // 原有图片数据为 HWC格式，模型输入节点要求的为 CHW 格式
  for (size_t c = 0; c < num_channels; c++) {
    for (size_t h = 0; h < height; h++) {
      for (size_t w = 0; w < width; w++) {
        input_tensor_data[c * width * height + h * width + w] =
          blob_image.at<cv::Vec<float, 3>>(h, w)[c];
      }
    }
  }
  return 1 / scale;
}

void YOLO11_BUFF::printInputAndOutputsInfo(const ov::Model & network)
{
  std::cout << "model name: " << network.get_friendly_name() << std::endl;

  const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
  for (const ov::Output<const ov::Node> & input : inputs) {
    std::cout << "    inputs" << std::endl;

    const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
    std::cout << "        input name: " << name << std::endl;

    const ov::element::Type type = input.get_element_type();
    std::cout << "        input type: " << type << std::endl;

    const ov::Shape shape = input.get_shape();
    std::cout << "        input shape: " << shape << std::endl;
  }

  const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
  for (const ov::Output<const ov::Node> & output : outputs) {
    std::cout << "    outputs" << std::endl;

    const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
    std::cout << "        output name: " << name << std::endl;

    const ov::element::Type type = output.get_element_type();
    std::cout << "        output type: " << type << std::endl;

    const ov::Shape shape = output.get_shape();
    std::cout << "        output shape: " << shape << std::endl;
  }
}

void YOLO11_BUFF::save(const std::string & programName, const cv::Mat & image)
{
  const std::filesystem::path saveDir = "../result/";
  if (!std::filesystem::exists(saveDir)) {
    std::filesystem::create_directories(saveDir);
  }
  const std::filesystem::path savePath = saveDir / (programName + ".jpg");
  cv::imwrite(savePath.string(), image);
}
}  // namespace auto_buff