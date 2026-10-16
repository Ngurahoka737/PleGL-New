#pragma once

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <numbers>
#include <vector>

#include "mesh/Mesh.h"
#include "scene/Scene.h"

namespace plegl::test {

inline void requireValid(const Mesh& m) {
  const ValidationResult r = validate(m);
  INFO(r.message);
  REQUIRE(r.ok);
}

// For a closed, outward-wound mesh centred at the origin, every face normal points away from it.
inline void requireOutward(const Mesh& m) {
  for (Index f = 0; f < m.faceCount(); ++f) {
    const float d = glm::dot(m.faceAreaNormal(f), m.faceCentroid(f));
    if (d <= 0.0f) {
      INFO("face " << f << " points inward");
      REQUIRE(d > 0.0f);
    }
  }
}

inline Index eulerCharacteristic(const Mesh& m) { return m.vertexCount() - m.edgeCount() + m.faceCount(); }

// Nearest surface point of a closed object around the origin in the given direction, hit from
// outside along that direction.
inline Vec3 surfacePoint(const SceneObject& obj, Vec3 dir, float from = 3.0f) {
  dir = glm::normalize(dir);
  RayHit hit;
  const bool ok = obj.bvh.raycast(obj.mesh, Ray{dir * from, -dir}, hit);
  REQUIRE(ok);
  return hit.position;
}

// Distance from the origin to that surface point.
inline float radiusAlong(const SceneObject& obj, Vec3 dir) { return glm::length(surfacePoint(obj, dir)); }

inline void requireNormalsMatchFullRecompute(const Mesh& m, float tolerance = 1e-4f) {
  Mesh fresh = m;
  fresh.computeNormals();
  for (Index v = 0; v < m.vertexCount(); ++v) {
    const float d = glm::length(fresh.normals[v] - m.normals[v]);
    if (d > tolerance) {
      INFO("vertex " << v << " normal is stale by " << d);
      REQUIRE(d <= tolerance);
    }
  }
}

// Every normal equal, bit for bit, to what a full recompute gives.
inline void requireNormalsBitExact(const Mesh& m) {
  Mesh fresh = m;
  fresh.computeNormals();
  for (Index v = 0; v < m.vertexCount(); ++v) {
    if (std::memcmp(&fresh.normals[v], &m.normals[v], sizeof(Vec3)) != 0) {
      INFO("vertex " << v << " normal differs from a full recompute");
      REQUIRE(false);
    }
  }
}

inline void requireBoundsContainPositions(const SceneObject& obj) {
  for (const BvhLeaf& leaf : obj.bvh.leaves()) {
    for (Index f = leaf.faceBegin; f < leaf.faceEnd; ++f) {
      if (obj.mesh.faceHe[f] == kInvalid) continue;
      obj.mesh.forEachFaceVertex(f, [&](Index v) {
        const Vec3& p = obj.mesh.positions[v];
        const bool inside = glm::all(glm::greaterThanEqual(p, leaf.bounds.min)) &&
                            glm::all(glm::lessThanEqual(p, leaf.bounds.max));
        if (!inside) {
          INFO("vertex " << v << " lies outside its leaf bounds");
          REQUIRE(inside);
        }
      });
    }
  }
}

// FNV-1a over the bits of a float array.
inline std::uint64_t hashBits(const std::vector<Vec3>& values) {
  std::uint64_t h = 1469598103934665603ull;
  const auto* b = reinterpret_cast<const unsigned char*>(values.data());
  for (std::size_t i = 0; i < values.size() * sizeof(Vec3); ++i) {
    h ^= b[i];
    h *= 1099511628211ull;
  }
  return h;
}

// Two n x n quad grids (edge 0.2) in the z = 0 plane that share only the corner at the origin: one
// non-manifold vertex with two fans.
inline Mesh makeBowtie(int n) {
  std::vector<Vec3> p{{0, 0, 0}};
  std::vector<Index> idx, sizes;
  for (float s : {1.0f, -1.0f}) {
    std::vector<Index> id(static_cast<std::size_t>((n + 1) * (n + 1)), 0);
    for (int j = 0; j <= n; ++j)
      for (int i = 0; i <= n; ++i) {
        if (i == 0 && j == 0) continue;  // The shared corner.
        id[static_cast<std::size_t>(j * (n + 1) + i)] = static_cast<Index>(p.size());
        p.push_back({s * static_cast<float>(i) * 0.2f, s * static_cast<float>(j) * 0.2f, 0.0f});
      }
    auto at = [&](int i, int j) { return id[static_cast<std::size_t>(j * (n + 1) + i)]; };
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        idx.insert(idx.end(), {at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)});
        sizes.push_back(4);
      }
  }
  BuildReport report;
  Mesh m = buildMesh(std::move(p), idx, sizes, &report);
  REQUIRE(report.nonManifoldVertices == 1);
  m.computeNormals();
  return m;
}

// A closed tube along +Y from y = 0 to `length`: `segments` quads around, `rings` quads along, and
// a triangle fan closing each end.
inline Mesh makeTube(int segments, int rings, float radius, float length) {
  std::vector<Vec3> p;
  std::vector<Index> idx, sizes;
  for (int j = 0; j <= rings; ++j)
    for (int i = 0; i < segments; ++i) {
      const double a = 2.0 * std::numbers::pi * i / segments;
      p.push_back({radius * static_cast<float>(std::cos(a)), length * static_cast<float>(j) / static_cast<float>(rings),
                   radius * static_cast<float>(std::sin(a))});
    }
  auto at = [&](int i, int j) { return static_cast<Index>(j * segments + (i % segments)); };
  for (int j = 0; j < rings; ++j)
    for (int i = 0; i < segments; ++i) {
      idx.insert(idx.end(), {at(i, j), at(i, j + 1), at(i + 1, j + 1), at(i + 1, j)});
      sizes.push_back(4);
    }
  const Index bottom = static_cast<Index>(p.size());
  p.push_back({0.0f, 0.0f, 0.0f});
  const Index top = static_cast<Index>(p.size());
  p.push_back({0.0f, length, 0.0f});
  for (int i = 0; i < segments; ++i) {
    idx.insert(idx.end(), {bottom, at(i, 0), at(i + 1, 0)});
    idx.insert(idx.end(), {top, at(i + 1, rings), at(i, rings)});
    sizes.push_back(3);
    sizes.push_back(3);
  }
  Mesh m = buildMesh(std::move(p), idx, sizes);
  m.computeNormals();
  return m;
}

}  // namespace plegl::test
