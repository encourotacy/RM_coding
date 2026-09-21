#include "ovsentry/omni_task.hpp"
#include "ovsentry/runtime.hpp"

int main(int argc, char * argv[])
{
  auto config = ovsentry::prepare_config(argc, argv, ovsentry::AppMode::Omni);
  if (!config) return 0;

  ovsentry::SentryRuntime runtime(std::move(*config), "ovsentry_omni", true);
  ovsentry::OmniTask omni(runtime);

  while (!runtime.exit()) {
    if (!runtime.read_main_frame()) continue;
    runtime.update_sensors();
    runtime.reset_outputs();
    omni.prepare(true);
    omni.run();
    if (!runtime.finish_frame(false, true, nullptr, nullptr)) break;
  }

  runtime.shutdown();
  return 0;
}
