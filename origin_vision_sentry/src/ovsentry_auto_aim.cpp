#include "ovsentry/auto_aim_task.hpp"
#include "ovsentry/runtime.hpp"

int main(int argc, char * argv[])
{
  auto config = ovsentry::prepare_config(argc, argv, ovsentry::AppMode::AutoAim);
  if (!config) return 0;

  ovsentry::SentryRuntime runtime(std::move(*config), "ovsentry_auto_aim", false);
  ovsentry::AutoAimTask auto_aim(runtime);

  while (!runtime.exit()) {
    if (!runtime.read_main_frame()) continue;
    runtime.update_sensors();
    auto_aim.detect_and_track();
    runtime.reset_outputs();
    auto_aim.aim_and_send();
    if (!runtime.finish_frame(false, false, &auto_aim, nullptr)) break;
  }

  runtime.shutdown();
  return 0;
}
