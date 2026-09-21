#include "src/ovsentry/app.hpp"
#include "src/ovsentry/config.hpp"
#include "tools/logger.hpp"

int main(int argc, char * argv[])
{
  auto config = ovsentry::parse_runtime_config(argc, argv);
  if (!config) return 0;

  tools::logger()->info(
    "[OVSentryOmniMPC] inference devices: auto_aim={} omni={}", config->auto_aim_device,
    config->omni_device);

  ovsentry::OVSentryOmniMpc app(std::move(*config));
  return app.run();
}
