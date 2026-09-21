#include "ovsentry/auto_aim_task.hpp"
#include "ovsentry/buff_task.hpp"
#include "ovsentry/omni_task.hpp"
#include "ovsentry/runtime.hpp"

int main(int argc, char * argv[])
{
  auto config = ovsentry::prepare_config(argc, argv, ovsentry::AppMode::AutoSwitch);
  if (!config) return 0;

  ovsentry::SentryRuntime runtime(std::move(*config), "ovsentry_mpc", true);
  ovsentry::AutoAimTask auto_aim(runtime);
  ovsentry::BuffTask buff(runtime);
  ovsentry::OmniTask omni(runtime);

  while (!runtime.exit()) {
    if (!runtime.read_main_frame()) continue;
    runtime.update_sensors();

    const bool small_buff = runtime.buff_requested();
    if (!small_buff) auto_aim.detect_and_track();
    const bool omni_mode = !small_buff && runtime.frame().tracker_state == "lost";

    runtime.reset_outputs();
    omni.prepare(omni_mode);

    if (small_buff) {
      omni.reset();
      buff.run();
    } else if (omni_mode) {
      buff.reset_hold();
      omni.run();
    } else {
      omni.reset();
      buff.reset_hold();
      auto_aim.aim_and_send();
    }

    if (!runtime.finish_frame(small_buff, omni_mode, &auto_aim, &buff)) break;
  }

  runtime.shutdown();
  return 0;
}
