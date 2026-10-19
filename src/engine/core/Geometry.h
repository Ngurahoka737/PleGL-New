#pragma once

#include <cmath>

#include "core/Types.h"

namespace plegl {

// Closest point to p on triangle abc (Ericson, Real-Time Collision Detection 5.1.5).
inline Vec3 closestPointOnTriangle(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
  const Vec3 ab = b - a, ac = c - a, ap = p - a;
  const float d1 = glm::dot(ab, ap), d2 = glm::dot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) return a;
  const Vec3 bp = p - b;
  const float d3 = glm::dot(ab, bp), d4 = glm::dot(ac, bp);
  if (d3 >= 0.0f && d4 <= d3) return b;
  const float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) return a + ab * (d1 / (d1 - d3));
  const Vec3 cp = p - c;
  const float d5 = glm::dot(ab, cp), d6 = glm::dot(ac, cp);
  if (d6 >= 0.0f && d5 <= d6) return c;
  const float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) return a + ac * (d2 / (d2 - d6));
  const float va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
  const float denom = 1.0f / (va + vb + vc);
  return a + ab * (vb * denom) + ac * (vc * denom);
}

// Barycentric coordinates (u, v, w) of p with respect to triangle abc, so p = u a + v b + w c for
// p in the triangle's plane. Degenerate triangles give (1, 0, 0).
inline Vec3 barycentric(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
  const Vec3 v0 = b - a, v1 = c - a, v2 = p - a;
  const float d00 = glm::dot(v0, v0), d01 = glm::dot(v0, v1), d11 = glm::dot(v1, v1);
  const float d20 = glm::dot(v2, v0), d21 = glm::dot(v2, v1);
  const float denom = d00 * d11 - d01 * d01;
  if (!(std::abs(denom) > 1e-30f)) return {1.0f, 0.0f, 0.0f};
  const float v = (d11 * d20 - d01 * d21) / denom;
  const float w = (d00 * d21 - d01 * d20) / denom;
  return {1.0f - v - w, v, w};
}

// Squared distance from p to the box (0 inside).
inline float distanceSq(const Aabb& b, const Vec3& p) {
  const Vec3 d = glm::max(glm::max(b.min - p, p - b.max), Vec3{0.0f});
  return glm::dot(d, d);
}

}  // namespace plegl
