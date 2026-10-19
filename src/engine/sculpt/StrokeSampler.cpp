#include "sculpt/StrokeSampler.h"

#include <algorithm>
#include <cmath>

namespace plegl {

void StrokeSampler::begin(const StrokeSample& s, float spacing, std::vector<StrokeSample>& out) {
  setSpacing(spacing);
  last_ = s;
  carried_ = 0.0f;
  out.push_back(s);
}

void StrokeSampler::setSpacing(float spacing) { spacing_ = std::max(spacing, 0.5f); }

void StrokeSampler::moveTo(const StrokeSample& s, std::vector<StrokeSample>& out) {
  const float dx = s.x - last_.x, dy = s.y - last_.y;
  const float len = std::sqrt(dx * dx + dy * dy);
  if (len <= 0.0f) {
    last_.pressure = s.pressure;
    return;
  }
  // First dab lies `spacing - carried` along this segment, then every `spacing`.
  float t = spacing_ - carried_;
  while (t <= len) {
    const float a = t / len;
    out.push_back({last_.x + dx * a, last_.y + dy * a, last_.pressure + (s.pressure - last_.pressure) * a});
    t += spacing_;
  }
  carried_ = len - (t - spacing_);
  last_ = s;
}

}  // namespace plegl
