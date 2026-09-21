#include "ovsentry/buff_task.hpp"
#include "ovsentry/runtime.hpp"

int main(int argc, char * argv[])
{
  auto config = ovsentry::prepare_config(argc, argv, ovsentry::AppMode::Buff);
  if (!config) return 0;

  ovsentry::SentryRuntime runtime(std::move(*config), "ovsentry_buff", false);
  ovsentry::BuffTask buff(runtime);

  while (!runtime.exit()) {
    if (!runtime.read_main_frame()) continue;
    runtime.update_sensors();
    runtime.reset_outputs();
    buff.run();
    if (!runtime.finish_frame(true, false, nullptr, &buff)) break;
  }

  runtime.shutdown();
  return 0;
}
