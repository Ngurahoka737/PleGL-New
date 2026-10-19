#pragma once

#include <vector>

namespace plegl {

struct StrokeSample {
  float x = 0.0f;  // Screen position in pixels.
  float y = 0.0f;
  float pressure = 1.0f;
};

// Turns irregular input events into evenly spaced dab positions along the path, interpolating
// pressure. Spacing is in the same units as the samples (pixels), so dab density does not
// depend on how fast the mouse or pen moves or how often the OS reports it.
class StrokeSampler {
 public:
  // Starts a stroke; the first point always produces a dab.
  void begin(const StrokeSample& s, float spacing, std::vector<StrokeSample>& out);
  // Adds an input point and appends the dabs it completes.
  void moveTo(const StrokeSample& s, std::vector<StrokeSample>& out);
  void setSpacing(float spacing);

 private:
  StrokeSample last_;
  float spacing_ = 1.0f;
  float carried_ = 0.0f;  // Distance travelled since the last emitted dab.
};

}  // namespace plegl
