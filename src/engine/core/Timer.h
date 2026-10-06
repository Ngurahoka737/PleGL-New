#pragma once

#include <chrono>

namespace plegl {

class Timer {
 public:
  Timer() : start_(Clock::now()) {}
  void reset() { start_ = Clock::now(); }
  double ms() const { return std::chrono::duration<double, std::milli>(Clock::now() - start_).count(); }
  double us() const { return std::chrono::duration<double, std::micro>(Clock::now() - start_).count(); }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

}  // namespace plegl
