#include "Camera.h"

#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

namespace plegl {
namespace {
constexpr float kMaxPitch = 1.5533f;  // 89 degrees.
}

Vec3 Camera::forward() const {
  // Looking from eye to target.
  return -Vec3{std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw)};
}

Vec3 Camera::position() const { return target - forward() * distance; }

Vec3 Camera::right() const { return glm::normalize(glm::cross(forward(), Vec3{0.0f, 1.0f, 0.0f})); }

Vec3 Camera::up() const { return glm::cross(right(), forward()); }

Mat4 Camera::view() const { return glm::lookAt(position(), target, Vec3{0.0f, 1.0f, 0.0f}); }

Mat4 Camera::projection() const {
  const float nearZ = std::max(distance * 0.002f, 1e-4f);
  const float farZ = distance * 200.0f + 100.0f;
  return glm::perspective(fovY, static_cast<float>(width_) / static_cast<float>(height_), nearZ, farZ);
}

void Camera::orbit(float dx, float dy, const Vec3& pivot) {
  const float speed = 0.008f;
  const float dYaw = -dx * speed;
  const float newPitch = std::clamp(pitch + dy * speed, -kMaxPitch, kMaxPitch);
  const float dPitch = newPitch - pitch;

  // Rotate eye and target around the pivot by the same rotation the view direction undergoes.
  const Quat qYaw = glm::angleAxis(dYaw, Vec3{0.0f, 1.0f, 0.0f});
  const Quat qPitch = glm::angleAxis(-dPitch, right());
  const Quat q = qYaw * qPitch;
  const Vec3 eye = pivot + q * (position() - pivot);
  target = pivot + q * (target - pivot);
  yaw += dYaw;
  pitch = newPitch;
  distance = glm::length(target - eye);
}

void Camera::pan(float dx, float dy) {
  const float scale = worldPerPixel(target);
  target += (-dx * right() + dy * up()) * scale;
}

void Camera::dolly(float dy) { distance = std::max(distance * std::exp(dy * 0.005f), 1e-3f); }

void Camera::zoomSteps(float steps) { distance = std::max(distance * std::pow(0.88f, steps), 1e-3f); }

void Camera::frame(const Aabb& bounds) {
  if (!bounds.valid()) return;
  target = bounds.center();
  const float radius = std::max(glm::length(bounds.extent()) * 0.5f, 1e-3f);
  distance = radius / std::sin(fovY * 0.5f) * 1.1f;
}

Ray Camera::rayThroughPixel(float x, float y) const {
  const float ndcX = 2.0f * x / static_cast<float>(width_) - 1.0f;
  const float ndcY = 1.0f - 2.0f * y / static_cast<float>(height_);
  const float tanHalf = std::tan(fovY * 0.5f);
  const float aspect = static_cast<float>(width_) / static_cast<float>(height_);
  Ray r;
  r.origin = position();
  r.dir = glm::normalize(forward() + right() * (ndcX * tanHalf * aspect) + up() * (ndcY * tanHalf));
  return r;
}

float Camera::worldPerPixel(const Vec3& at) const {
  const float depth = std::max(glm::dot(at - position(), forward()), 1e-4f);
  return 2.0f * depth * std::tan(fovY * 0.5f) / static_cast<float>(height_);
}

}  // namespace plegl
