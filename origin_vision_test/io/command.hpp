#ifndef IO__COMMAND_HPP
#define IO__COMMAND_HPP

namespace io
{
struct Command
{
  bool control;
  bool shoot;
  double yaw;
  double pitch;
  double horizon_distance = 0;  // Legacy CAN path uses horizontal distance.
  int armor_id = 0;
  double vx = 0;
  double vy = 0;
  double big_yaw = 0;
  double small_yaw = 0;
  bool has_target_yaw = false;
};

}  // namespace io

#endif  // IO__COMMAND_HPP
