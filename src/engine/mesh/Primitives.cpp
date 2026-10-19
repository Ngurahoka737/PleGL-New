#include "mesh/Primitives.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <unordered_map>

namespace plegl {
namespace {

constexpr float kPi = std::numbers::pi_v<float>;

struct Soup {
  std::vector<Vec3> positions;
  std::vector<Index> indices;
  std::vector<Index> sizes;

  void tri(Index a, Index b, Index c) {
    indices.insert(indices.end(), {a, b, c});
    sizes.push_back(3);
  }
  void quad(Index a, Index b, Index c, Index d) {
    indices.insert(indices.end(), {a, b, c, d});
    sizes.push_back(4);
  }
  Mesh build() { return buildMesh(std::move(positions), indices, sizes); }
};

// Six grid faces of a cube in [-1, 1]^3, welded. `warp` maps a grid coordinate in [-1, 1].
Soup cubeSoup(int res, float (*warp)(float)) {
  res = std::max(res, 1);
  Soup s;
  const Vec3 normals[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  const Vec3 ups[6] = {{0, 1, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 1}, {0, 1, 0}, {0, 1, 0}};
  for (int face = 0; face < 6; ++face) {
    const Vec3 n = normals[face];
    const Vec3 u = ups[face];
    const Vec3 v = glm::cross(n, u);  // u x v == n, so (u, v) grid order winds outward.
    const Index base = static_cast<Index>(s.positions.size());
    for (int j = 0; j <= res; ++j) {
      for (int i = 0; i <= res; ++i) {
        const float a = warp(-1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(res));
        const float b = warp(-1.0f + 2.0f * static_cast<float>(j) / static_cast<float>(res));
        s.positions.push_back(n + a * u + b * v);
      }
    }
    const Index row = res + 1;
    for (int j = 0; j < res; ++j) {
      for (int i = 0; i < res; ++i) {
        const Index p = base + j * row + i;
        s.quad(p, p + 1, p + row + 1, p + row);
      }
    }
  }
  weldVertices(s.positions, s.indices, 1e-4f / static_cast<float>(res));
  return s;
}

float identity(float x) { return x; }
float equalAngle(float x) { return std::tan(x * kPi * 0.25f); }

}  // namespace

Mesh makeUvSphere(int segments, int rings, float radius) {
  segments = std::max(segments, 3);
  rings = std::max(rings, 2);
  Soup s;
  s.positions.push_back({0.0f, radius, 0.0f});
  for (int r = 1; r < rings; ++r) {
    const float theta = kPi * static_cast<float>(r) / static_cast<float>(rings);
    const float y = std::cos(theta) * radius;
    const float rr = std::sin(theta) * radius;
    for (int i = 0; i < segments; ++i) {
      const float phi = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(segments);
      s.positions.push_back({rr * std::cos(phi), y, -rr * std::sin(phi)});
    }
  }
  const Index south = static_cast<Index>(s.positions.size());
  s.positions.push_back({0.0f, -radius, 0.0f});

  auto ring = [&](int r, int i) { return 1 + (r - 1) * segments + (i % segments); };
  for (int i = 0; i < segments; ++i) s.tri(0, ring(1, i), ring(1, i + 1));
  for (int r = 1; r < rings - 1; ++r) {
    for (int i = 0; i < segments; ++i) s.quad(ring(r, i), ring(r + 1, i), ring(r + 1, i + 1), ring(r, i + 1));
  }
  for (int i = 0; i < segments; ++i) s.tri(south, ring(rings - 1, i + 1), ring(rings - 1, i));
  return s.build();
}

Mesh makeIcosphere(int subdivisions, float radius) {
  const float t = (1.0f + std::sqrt(5.0f)) * 0.5f;
  std::vector<Vec3> pos = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
                           {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
  std::vector<Index> tris = {0, 11, 5, 0, 5,  1,  0, 1, 7, 0, 7,  10, 0, 10, 11, 1, 5, 9, 5, 11,
                             4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3,  9,  4, 3,  4,  2, 3, 2, 6, 3,
                             6, 8,  3,  8, 9,  4, 9, 5, 2, 4, 11, 6,  2, 10, 8,  6, 7, 9, 8, 1};
  for (Vec3& p : pos) p = glm::normalize(p);

  for (int level = 0; level < std::max(subdivisions, 0); ++level) {
    std::unordered_map<std::uint64_t, Index> midpoints;
    midpoints.reserve(tris.size());
    auto mid = [&](Index a, Index b) {
      const std::uint64_t key = (static_cast<std::uint64_t>(std::min(a, b)) << 32) | static_cast<std::uint32_t>(std::max(a, b));
      auto [it, inserted] = midpoints.try_emplace(key, static_cast<Index>(pos.size()));
      if (inserted) pos.push_back(glm::normalize(pos[a] + pos[b]));
      return it->second;
    };
    std::vector<Index> next;
    next.reserve(tris.size() * 4);
    for (std::size_t i = 0; i < tris.size(); i += 3) {
      const Index a = tris[i], b = tris[i + 1], c = tris[i + 2];
      const Index ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
      next.insert(next.end(), {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca});
    }
    tris = std::move(next);
  }
  for (Vec3& p : pos) p *= radius;
  std::vector<Index> sizes(tris.size() / 3, 3);
  return buildMesh(std::move(pos), tris, sizes);
}

Mesh makeCube(int resolution, float size) {
  Soup s = cubeSoup(resolution, identity);
  for (Vec3& p : s.positions) p *= size * 0.5f;
  return s.build();
}

Mesh makeQuadSphere(int resolution, float radius) {
  Soup s = cubeSoup(resolution, equalAngle);
  for (Vec3& p : s.positions) p = glm::normalize(p) * radius;
  return s.build();
}

Mesh makePlane(int resolution, float size) {
  resolution = std::max(resolution, 1);
  Soup s;
  const Index row = resolution + 1;
  for (int j = 0; j <= resolution; ++j) {
    for (int i = 0; i <= resolution; ++i) {
      const float x = (-0.5f + static_cast<float>(i) / static_cast<float>(resolution)) * size;
      const float z = (-0.5f + static_cast<float>(j) / static_cast<float>(resolution)) * size;
      s.positions.push_back({x, 0.0f, z});
    }
  }
  for (int j = 0; j < resolution; ++j) {
    for (int i = 0; i < resolution; ++i) {
      const Index p = j * row + i;
      s.quad(p, p + row, p + row + 1, p + 1);  // Counter-clockwise seen from +Y.
    }
  }
  return s.build();
}

}  // namespace plegl
