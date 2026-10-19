#include "sculpt/Brush.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "core/Parallel.h"
#include "sculpt/Neighbours.h"

namespace plegl {

namespace {
// Displacement per dab at full strength, as a fraction of the radius. With the default spacing
// (a dab every 10% of the radius) a stroke at 50% strength raises the surface by roughly a
// quarter of the radius, which matches the feel of common sculpting tools.
constexpr float kDrawScale = 0.04f;
// Clay plane height above the area centre, as a fraction of the radius, and the share of the gap
// to that plane closed per dab at full strength.
constexpr float kClayOffset = 0.06f;
constexpr float kClayRate = 0.5f;
constexpr float kInflateScale = 0.03f;
// Share of the distance to the plane removed per dab at full strength.
constexpr float kFlattenRate = 0.35f;
// Crease: groove depth per dab like Draw, plus a pull toward the centre in the tangent plane.
constexpr float kCreaseDepth = 0.03f;
constexpr float kCreasePinch = 0.15f;

// Share of each vertex's movement that the mask lets through.
inline float unmasked(const float* mask, Index v) { return mask ? 1.0f - mask[v] : 1.0f; }

// Runs `displace(v, p, weight)` for every vertex owned by the dab's leaves that lies inside the
// sphere, in parallel per leaf, and adds the returned offset clamped to kMaxDabMove * radius.
// `weight` is falloff times strength times the unmasked share. Only for brushes that read
// nothing but their own vertex.
template <typename F>
void displaceInside(BrushContext& ctx, F&& displace) {
  const Dab& d = ctx.dab;
  Mesh& m = ctx.mesh;
  const float* mask = m.mask.empty() ? nullptr : m.mask.data();
  const float r2 = d.radius * d.radius;
  const float invR = 1.0f / d.radius;
  const float maxMove = d.radius * kMaxDabMove;
  const float maxMove2 = maxMove * maxMove;
  parallelFor(0, ctx.leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const BvhLeaf& leaf = ctx.bvh.leaves()[ctx.leaves[i]];
      for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
        const Vec3& p = m.positions[v];
        const Vec3 delta = p - d.center;
        const float dist2 = glm::dot(delta, delta);
        if (dist2 >= r2) continue;
        const float w = d.strength * falloffWeight(d.falloff, std::sqrt(dist2) * invR) * unmasked(mask, v);
        if (w <= 0.0f) continue;
        Vec3 move = displace(v, p, w);
        const float len2 = glm::dot(move, move);
        if (len2 > maxMove2) move *= maxMove / std::sqrt(len2);
        m.positions[v] += move;
      }
    }
  });
}
}  // namespace

void DrawBrush::apply(BrushContext& ctx) const {
  const Dab& d = ctx.dab;
  Mesh& m = ctx.mesh;
  const float r2 = d.radius * d.radius;
  const float invR = 1.0f / d.radius;
  const Vec3 offset = d.areaNormal * (d.radius * d.strength * kDrawScale * (d.invert ? -1.0f : 1.0f));
  const float* mask = m.mask.empty() ? nullptr : m.mask.data();
  parallelFor(0, ctx.leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const BvhLeaf& leaf = ctx.bvh.leaves()[ctx.leaves[i]];
      for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
        const Vec3 delta = m.positions[v] - d.center;
        const float dist2 = glm::dot(delta, delta);
        if (dist2 >= r2) continue;
        const float w = falloffWeight(d.falloff, std::sqrt(dist2) * invR) * unmasked(mask, v);
        if (w <= 0.0f) continue;
        m.positions[v] += offset * w;
      }
    }
  });
}

void SmoothBrush::apply(BrushContext& ctx) const {
  const Dab& d = ctx.dab;
  Mesh& m = ctx.mesh;
  const float r2 = d.radius * d.radius;
  const float invR = 1.0f / d.radius;
  const float* mask = m.mask.empty() ? nullptr : m.mask.data();
  // Smoothing reads neighbours, so compute every new position first and write them afterwards.
  std::vector<std::vector<std::pair<Index, Vec3>>> results(ctx.leaves.size());
  parallelFor(0, ctx.leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const BvhLeaf& leaf = ctx.bvh.leaves()[ctx.leaves[i]];
      auto& out = results[i];
      for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
        const Vec3& p = m.positions[v];
        const Vec3 delta = p - d.center;
        const float dist2 = glm::dot(delta, delta);
        if (dist2 >= r2) continue;
        const float w = d.strength * falloffWeight(d.falloff, std::sqrt(dist2) * invR) * unmasked(mask, v);
        if (w <= 0.0f) continue;
        neighbourMean<Vec3>(
            m, v, [&](Index u) { return m.positions[u]; },
            [&](const Vec3& target) { out.emplace_back(v, p + (target - p) * w); });
      }
    }
  });
  parallelFor(0, results.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i)
      for (const auto& [v, p] : results[i]) m.positions[v] = p;
  });
}

void ClayBrush::apply(BrushContext& ctx) const {
  const Dab& d = ctx.dab;
  const float sign = d.invert ? -1.0f : 1.0f;
  const Vec3 n = d.areaNormal * sign;  // Work in a frame where "up" is the build direction.
  const Vec3 planePoint = d.areaCenter + n * (d.radius * kClayOffset);
  displaceInside(ctx, [&](Index, const Vec3& p, float w) {
    const float below = glm::dot(planePoint - p, n);  // > 0 when the vertex is under the plane.
    return below > 0.0f ? n * (below * kClayRate * w) : Vec3{0.0f};
  });
}

void InflateBrush::apply(BrushContext& ctx) const {
  const Dab& d = ctx.dab;
  const float amount = d.radius * kInflateScale * (d.invert ? -1.0f : 1.0f);
  const Mesh& m = ctx.mesh;
  displaceInside(ctx, [&](Index v, const Vec3&, float w) { return m.normals[v] * (amount * w); });
}

void FlattenBrush::apply(BrushContext& ctx) const {
  const Dab& d = ctx.dab;
  const Vec3 n = d.areaNormal;
  const float rate = kFlattenRate * (d.invert ? -1.0f : 1.0f);
  displaceInside(ctx, [&](Index, const Vec3& p, float w) {
    return n * (-glm::dot(p - d.areaCenter, n) * rate * w);
  });
}

void CreaseBrush::apply(BrushContext& ctx) const {
  const Dab& d = ctx.dab;
  const Vec3 n = d.areaNormal;
  const Vec3 push = n * (d.radius * kCreaseDepth * (d.invert ? 1.0f : -1.0f));
  displaceInside(ctx, [&](Index, const Vec3& p, float w) {
    Vec3 toCenter = d.center - p;
    toCenter -= n * glm::dot(toCenter, n);  // Pinch only along the surface.
    return (push + toCenter * kCreasePinch) * w;
  });
}

void MaskBrush::apply(BrushContext& ctx) const {
  const Dab& d = ctx.dab;
  Mesh& m = ctx.mesh;
  const float r2 = d.radius * d.radius;
  const float invR = 1.0f / d.radius;
  const float target = d.invert ? 0.0f : 1.0f;
  parallelFor(0, ctx.leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const BvhLeaf& leaf = ctx.bvh.leaves()[ctx.leaves[i]];
      for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
        const Vec3 delta = m.positions[v] - d.center;
        const float dist2 = glm::dot(delta, delta);
        if (dist2 >= r2) continue;
        const float w = d.strength * falloffWeight(d.falloff, std::sqrt(dist2) * invR);
        if (w <= 0.0f) continue;
        float& value = m.mask[v];
        value = w >= 1.0f ? target : std::clamp(value + (target - value) * w, 0.0f, 1.0f);
      }
    }
  });
}

void MaskSmoothBrush::apply(BrushContext& ctx) const {
  const Dab& d = ctx.dab;
  Mesh& m = ctx.mesh;
  const float r2 = d.radius * d.radius;
  const float invR = 1.0f / d.radius;
  std::vector<std::vector<std::pair<Index, float>>> results(ctx.leaves.size());
  parallelFor(0, ctx.leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const BvhLeaf& leaf = ctx.bvh.leaves()[ctx.leaves[i]];
      auto& out = results[i];
      for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
        const Vec3 delta = m.positions[v] - d.center;
        const float dist2 = glm::dot(delta, delta);
        if (dist2 >= r2) continue;
        const float w = d.strength * falloffWeight(d.falloff, std::sqrt(dist2) * invR);
        if (w <= 0.0f) continue;
        neighbourMeanAll<float>(
            m, v, [&](Index u) { return m.mask[u]; },
            [&](float target) { out.emplace_back(v, std::clamp(m.mask[v] + (target - m.mask[v]) * w, 0.0f, 1.0f)); });
      }
    }
  });
  parallelFor(0, results.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i)
      for (const auto& [v, value] : results[i]) m.mask[v] = value;
  });
}

}  // namespace plegl
