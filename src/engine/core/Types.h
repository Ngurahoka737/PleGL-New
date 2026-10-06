#pragma once

#include <cstdint>
#include <limits>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace plegl {

using Index = std::int32_t;
inline constexpr Index kInvalid = -1;

using Vec2 = glm::vec2;
using Vec3 = glm::vec3;
using Vec4 = glm::vec4;
using Mat3 = glm::mat3;
using Mat4 = glm::mat4;
using Quat = glm::quat;

struct Aabb {
  Vec3 min{std::numeric_limits<float>::max()};
  Vec3 max{-std::numeric_limits<float>::max()};

  void expand(const Vec3& p) {
    min = glm::min(min, p);
    max = glm::max(max, p);
  }
  void expand(const Aabb& b) {
    min = glm::min(min, b.min);
    max = glm::max(max, b.max);
  }
  bool valid() const { return min.x <= max.x && min.y <= max.y && min.z <= max.z; }
  Vec3 center() const { return (min + max) * 0.5f; }
  Vec3 extent() const { return max - min; }
};

struct Ray {
  Vec3 origin{0.0f};
  Vec3 dir{0.0f, 0.0f, -1.0f};  // Need not be normalized; hit t is in units of dir.
};

}  // namespace plegl
