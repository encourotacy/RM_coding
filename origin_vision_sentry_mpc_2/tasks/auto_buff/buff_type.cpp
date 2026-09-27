#include "buff_type.hpp"

#include <algorithm>
#include <limits>

#include "tools/logger.hpp"
namespace auto_buff
{
FanBlade::FanBlade(
  const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t)
: center(keypoints_center), type(t)
{
  points.insert(points.end(), kpt.begin(), kpt.end());
}

FanBlade::FanBlade(FanBlade_type t) : type(t)
{
  if (t != _unlight) exit(-1);
}

PowerRune::PowerRune(
  std::vector<FanBlade> & ts, const cv::Point2f center, std::optional<PowerRune> last_powerrune)
: r_center(center),
  light_num(static_cast<int>(ts.size())),
  board_hit_rings(ts.size(), 0),
  board_confidences(ts.size(), 0.0),
  board_tracked_frames(ts.size(), 1),
  board_areas(ts.size(), 0.0)
{
  if (light_num == 0) {
    tools::logger()->debug("[PowerRune] 识别出错!");
    unsolvable_ = true;
    return;
  }

  for (auto & blade : ts) {
    blade.type = _light;
    blade.angle = atan_angle(blade.center);
  }

  std::sort(ts.begin(), ts.end(), [](const FanBlade & a, const FanBlade & b) {
    return a.angle < b.angle;
  });
  fanblades = ts;
  activated_arms = light_num;

  if (last_powerrune.has_value() && !last_powerrune->fanblades.empty()) {
    const cv::Point2f last_selected_center = last_powerrune->target().center;
    float min_distance = std::numeric_limits<float>::max();
    int best_index = 0;
    for (size_t i = 0; i < fanblades.size(); ++i) {
      const float distance = cv::norm(fanblades[i].center - last_selected_center);
      if (distance < min_distance) {
        min_distance = distance;
        best_index = static_cast<int>(i);
      }
    }
    selected_board_index = best_index;
  } else {
    selected_board_index = 0;
  }
}

std::vector<int> PowerRune::getLitBoardIndices() const
{
  std::vector<int> indices;
  for (size_t i = 0; i < fanblades.size(); ++i) {
    if (isBoardLit(static_cast<int>(i))) indices.push_back(static_cast<int>(i));
  }
  return indices;
}

bool PowerRune::isBoardLit(int index) const
{
  if (index < 0 || index >= static_cast<int>(fanblades.size())) return false;
  return fanblades[index].type == _light || fanblades[index].type == _target;
}

void PowerRune::setBoardHitRing(int index, int ring)
{
  if (index < 0) return;
  if (index >= static_cast<int>(board_hit_rings.size())) board_hit_rings.resize(index + 1, 0);
  board_hit_rings[index] = ring;
}

int PowerRune::getHitRingForBoard(int index) const
{
  if (index < 0 || index >= static_cast<int>(board_hit_rings.size())) return 0;
  return board_hit_rings[index];
}

void PowerRune::setSelectedBoardIndex(int index)
{
  if (index < 0) {
    selected_board_index = -1;
    return;
  }
  if (isBoardLit(index)) selected_board_index = index;
}

void PowerRune::setCandidateBoardIndex(int index)
{
  if (index < 0) {
    candidate_board_index = -1;
    return;
  }
  if (isBoardLit(index)) candidate_board_index = index;
}

int PowerRune::resolved_selected_index() const
{
  if (isBoardLit(selected_board_index)) return selected_board_index;
  for (size_t i = 0; i < fanblades.size(); ++i) {
    if (isBoardLit(static_cast<int>(i))) return static_cast<int>(i);
  }
  return 0;
}

double PowerRune::atan_angle(cv::Point2f point) const
{
  auto v = point - r_center;
  auto angle = std::atan2(v.y, v.x);
  return angle >= 0 ? angle : angle + CV_2PI;
}

void DoubleBoardController::reset_tracking()
{
  active_board_lost_frames_ = 0;
  last_active_center_ = {0.0F, 0.0F};
  last_candidate_center_ = {0.0F, 0.0F};
  state_.reset();
}

void DoubleBoardController::start_tracking(const PowerRune & rune)
{
  const auto lit_indices = rune.getLitBoardIndices();
  state_.active_board_index = pick_best_board(rune, lit_indices);
  active_board_lost_frames_ = 0;
  if (
    state_.active_board_index >= 0 &&
    state_.active_board_index < static_cast<int>(rune.fanblades.size())) {
    last_active_center_ = rune.fanblades[state_.active_board_index].center;
  }
  update_candidate_tracking(rune, state_.active_board_index);
}

int DoubleBoardController::pick_best_board(
  const PowerRune & rune, const std::vector<int> & candidates) const
{
  if (candidates.empty()) return -1;

  double best_cost = std::numeric_limits<double>::max();
  int best_index = -1;
  const double center_x = std::max(1.0, static_cast<double>(image_center_.x));
  const double center_y = std::max(1.0, static_cast<double>(image_center_.y));

  for (int index : candidates) {
    if (!rune.isBoardLit(index)) continue;

    const auto & board = rune.fanblades[index];
    const double yaw_offset_norm = std::clamp(
      static_cast<double>(std::abs(board.center.x - image_center_.x) / center_x), 0.0, 1.0);
    const double pitch_offset_norm = std::clamp(
      static_cast<double>(std::abs(board.center.y - image_center_.y) / center_y), 0.0, 1.0);
    const double confidence = index < static_cast<int>(rune.board_confidences.size())
                                ? rune.board_confidences[index]
                                : 0.0;
    const int tracked_frames = index < static_cast<int>(rune.board_tracked_frames.size())
                                 ? rune.board_tracked_frames[index]
                                 : 1;
    const double confidence_penalty = 1.0 - std::clamp(confidence, 0.0, 1.0);
    const double stability_penalty =
      1.0 - std::clamp(tracked_frames / static_cast<double>(MAX_TRACK_FRAMES), 0.0, 1.0);
    const double cost = 0.60 * yaw_offset_norm + 0.20 * pitch_offset_norm +
                        0.10 * confidence_penalty + 0.10 * stability_penalty;

    bool replace = best_index < 0 || cost < best_cost - 1e-6;
    if (!replace && std::abs(cost - best_cost) <= 1e-6) {
      const double best_confidence = best_index < static_cast<int>(rune.board_confidences.size())
                                       ? rune.board_confidences[best_index]
                                       : 0.0;
      const int best_tracked_frames = best_index < static_cast<int>(rune.board_tracked_frames.size())
                                        ? rune.board_tracked_frames[best_index]
                                        : 1;
      replace = confidence > best_confidence + 1e-6 ||
                (std::abs(confidence - best_confidence) <= 1e-6 && tracked_frames > best_tracked_frames) ||
                (std::abs(confidence - best_confidence) <= 1e-6 && tracked_frames == best_tracked_frames &&
                 index < best_index);
    }
    if (replace) {
      best_cost = cost;
      best_index = index;
    }
  }
  return best_index;
}

int DoubleBoardController::rematch_board(
  const PowerRune & rune, const cv::Point2f & last_center, int exclude_index) const
{
  const auto lit_indices = rune.getLitBoardIndices();
  if (lit_indices.empty()) return -1;

  float radius = 0.0F;
  for (int index : lit_indices) radius += cv::norm(rune.fanblades[index].center - rune.r_center);
  radius /= static_cast<float>(lit_indices.size());

  const float rematch_threshold = std::max(40.0F, radius * 0.45F);
  float min_distance = std::numeric_limits<float>::max();
  int rematched = -1;
  for (int index : lit_indices) {
    if (index == exclude_index) continue;
    const float distance = cv::norm(rune.fanblades[index].center - last_center);
    if (distance < min_distance) {
      min_distance = distance;
      rematched = index;
    }
  }
  return (min_distance <= rematch_threshold) ? rematched : -1;
}

void DoubleBoardController::update_candidate_tracking(const PowerRune & rune, int exclude_index)
{
  int next_candidate = -1;
  if (state_.candidate_board_index >= 0) {
    next_candidate = rematch_board(rune, last_candidate_center_, exclude_index);
  }
  if (next_candidate < 0) {
    auto lit_indices = rune.getLitBoardIndices();
    lit_indices.erase(
      std::remove(lit_indices.begin(), lit_indices.end(), exclude_index), lit_indices.end());
    next_candidate = pick_best_board(rune, lit_indices);
  }

  state_.candidate_board_index = next_candidate;
  if (next_candidate >= 0 && next_candidate < static_cast<int>(rune.fanblades.size())) {
    last_candidate_center_ = rune.fanblades[next_candidate].center;
  } else {
    last_candidate_center_ = {0.0F, 0.0F};
  }
}

void DoubleBoardController::apply_selection(PowerRune & rune) const
{
  rune.setSelectedBoardIndex(-1);
  rune.setCandidateBoardIndex(-1);
  rune.setSelectedBoardIndex(state_.active_board_index);
  rune.setCandidateBoardIndex(state_.candidate_board_index);
  rune.hit_ring =
    rune.selectedBoardIndex() >= 0 ? rune.getHitRingForBoard(rune.selectedBoardIndex()) : 0;
}

void DoubleBoardController::updateSelection(
  PowerRune & rune, const std::chrono::steady_clock::time_point & timestamp)
{
  (void)timestamp;

  if (state_.active_board_index < 0) {
    if (rune.getLitBoardIndices().empty()) {
      reset_tracking();
    } else {
      start_tracking(rune);
    }
    apply_selection(rune);
    return;
  }

  const int rematched_active = rematch_board(rune, last_active_center_, -1);
  if (rematched_active >= 0) {
    state_.active_board_index = rematched_active;
    last_active_center_ = rune.fanblades[rematched_active].center;
    active_board_lost_frames_ = 0;
    update_candidate_tracking(rune, state_.active_board_index);
    apply_selection(rune);
    return;
  }

  active_board_lost_frames_++;
  update_candidate_tracking(rune, -1);
  if (active_board_lost_frames_ >= REMATCH_LOST_FRAMES) {
    if (
      state_.candidate_board_index >= 0 &&
      state_.candidate_board_index < static_cast<int>(rune.fanblades.size()) &&
      rune.isBoardLit(state_.candidate_board_index)) {
      state_.active_board_index = state_.candidate_board_index;
      last_active_center_ = rune.fanblades[state_.active_board_index].center;
      active_board_lost_frames_ = 0;
      update_candidate_tracking(rune, state_.active_board_index);
    } else {
      reset_tracking();
    }
  }
  apply_selection(rune);
}
}  // namespace auto_buff
