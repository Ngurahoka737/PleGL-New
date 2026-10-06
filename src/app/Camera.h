#pragma once

#include "core/Types.h"

namespace plegl {

// Turntable camera: yaw around world +Y, pitch around the camera's right axis, looking at
// `target` from `distance`. Orbiting can pivot around any point (for example the surface under
// the cursor) by rotating both eye and target around it.
class Camera {
 public:
  Vec3 target{0.0f};
  float distance = 5.0f;
  float yaw = 0.6f;     // Radians.
  float pitch = 0.35f;  // Radians, clamped just short of the poles.
  float fovY = 0.70f;   // Radians (about 40 degrees).

  void setViewport(int width, int height) {
    width_ = width > 0 ? width : 1;
    height_ = height > 0 ? height : 1;
  }
  int viewportWidth() const { return width_; }
  int viewportHeight() const { return height_; }

  Vec3 position() const;
  Vec3 forward() const;
  Vec3 right() const;
  Vec3 up() const;
  Mat4 view() const;
  Mat4 projection() const;

  // Drag deltas are in viewport pixels.
  void orbit(float dx, float dy, const Vec3& pivot);
  void orbit(float dx, float dy) { orbit(dx, dy, target); }
  void pan(float dx, float dy);
  void dolly(float dy);         // Drag zoom (Alt + right mouse).
  void zoomSteps(float steps);  // Mouse wheel; positive zooms in.
  void frame(const Aabb& bounds);

  // World-space ray through a viewport pixel (origin top-left).
  Ray rayThroughPixel(float x, float y) const;
  // Approximate world size of one pixel at the given point.
  float worldPerPixel(const Vec3& at) const;

 private:
  int width_ = 1;
  int height_ = 1;
};

}  // namespace plegl
