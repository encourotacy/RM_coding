#include "ovsentry/app.hpp"

int main(int argc, char * argv[])
{
  return ovsentry::run_ovsentry_app(argc, argv, ovsentry::AppMode::Omni);
}
