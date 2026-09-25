#ifndef IO__SENTRY_REQUEST_HPP
#define IO__SENTRY_REQUEST_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace io
{
struct ArmorIgnoreList
{
  bool enabled = false;
  std::vector<uint8_t> ignored_ids;
};

class ArmorIgnoreSubscriber
{
public:
  ArmorIgnoreSubscriber(
    const std::string & topic = "/request_auto_aim_ignore",
    const std::string & msg_type = "rm_interfaces/msg/RequestAutoAimIgnore");
  ~ArmorIgnoreSubscriber();

  ArmorIgnoreList ignored() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class BuffRequestSubscriber
{
public:
  explicit BuffRequestSubscriber(const std::string & topic = "/request_buff");
  ~BuffRequestSubscriber();

  bool requested() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace io

#endif  // IO__SENTRY_REQUEST_HPP
