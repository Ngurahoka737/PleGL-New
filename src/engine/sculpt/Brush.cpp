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

// Share of each vertex's movement that the mask and the face set filter let through.
inline float unmasked(const BrushContext& ctx, const float* mask, Index v) {
  if (ctx.filter.faceSets && !ctx.filter.allows(ctx.mesh, v)) return 0.0f;
  return mask ? 1.0f - mask[v] : 1.0f;
}

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
        const float w = d.strength * falloffWeight(d.falloff, std::sqrt(dist2) * invR) * unmasked(ctx, mask, v);
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
        const float w = falloffWeight(d.falloff, std::sqrt(dist2) * invR) * unmasked(ctx, mask, v);
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
        const float w = d.strength * falloffWeight(d.falloff, std::sqrt(dist2) * invR) * unmasked(ctx, mask, v);
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

namespace {
// Adds `delta` to a layer offset and returns the vertex's new composite position. The composite
// moves by strength * delta, which is clamped to kMaxDabMove * radius like every brush.
Vec3 moveLayer(const LayerTarget& t, Index v, Vec3 delta, float maxMove) {
  Vec3& o = (*t.offset)[v];
  const float move = std::abs(t.strength) * glm::length(delta);
  if (move > maxMove) delta *= maxMove / move;
  o.x = o.x + delta.x;
  o.y = o.y + delta.y;
  o.z = o.z + delta.z;
  return composeVertex(*t.stack, v);
}
}  // namespace

void EraseLayerBrush::apply(BrushContext& ctx) const {
  if (!ctx.layer || !ctx.layer->offset) return;
  const LayerTarget& t = *ctx.layer;
  const Dab& d = ctx.dab;
  Mesh& m = ctx.mesh;
  const float r2 = d.radius * d.radius;
  const float invR = 1.0f / d.radius;
  const float maxMove = d.radius * kMaxDabMove;
  const float* mask = m.mask.empty() ? nullptr : m.mask.data();
  // Each vertex reads only its own offset, so leaves can be written in place in parallel.
  parallelFor(0, ctx.leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const BvhLeaf& leaf = ctx.bvh.leaves()[ctx.leaves[i]];
      for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
        const Vec3 delta = m.positions[v] - d.center;
        const float dist2 = glm::dot(delta, delta);
        if (dist2 >= r2) continue;
        const float w = d.strength * falloffWeight(d.falloff, std::sqrt(dist2) * invR) * unmasked(ctx, mask, v);
        if (w <= 0.0f) continue;
        const Vec3& o = (*t.offset)[v];
        if (isZero(o)) continue;
        m.positions[v] = moveLayer(t, v, Vec3{-(w * o.x), -(w * o.y), -(w * o.z)}, maxMove);
      }
    }
  });
}

void LayerSmoothBrush::apply(BrushContext& ctx) const {
  if (!ctx.layer || !ctx.layer->offset) return;
  const LayerTarget& t = *ctx.layer;
  const std::vector<Vec3>& offset = *t.offset;
  const Dab& d = ctx.dab;
  Mesh& m = ctx.mesh;
  const float r2 = d.radius * d.radius;
  const float invR = 1.0f / d.radius;
  const float maxMove = d.radius * kMaxDabMove;
  const float* mask = m.mask.empty() ? nullptr : m.mask.data();
  // Like SmoothBrush: every new offset is computed from the old ones first, then written.
  std::vector<std::vector<std::pair<Index, Vec3>>> results(ctx.leaves.size());
  parallelFor(0, ctx.leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const BvhLeaf& leaf = ctx.bvh.leaves()[ctx.leaves[i]];
      auto& out = results[i];
      for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
        const Vec3 delta = m.positions[v] - d.center;
        const float dist2 = glm::dot(delta, delta);
        if (dist2 >= r2) continue;
        const float w = d.strength * falloffWeight(d.falloff, std::sqrt(dist2) * invR) * unmasked(ctx, mask, v);
        if (w <= 0.0f) continue;
        neighbourMean<Vec3>(
            m, v, [&](Index u) { return offset[u]; },
            [&](const Vec3& mean) {
              const Vec3 change = (mean - offset[v]) * w;
              if (!isZero(change)) out.emplace_back(v, change);
            });
      }
    }
  });
  parallelFor(0, results.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i)
      for (const auto& [v, change] : results[i]) m.positions[v] = moveLayer(t, v, change, maxMove);
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
        if (w <= 0.0f || !ctx.filter.allows(m, v)) continue;
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
        if (w <= 0.0f || !ctx.filter.allows(m, v)) continue;
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

void FaceSetBrush::apply(BrushContext& ctx) const {
  const Dab& d = ctx.dab;
  Mesh& m = ctx.mesh;
  const float r2 = d.radius * d.radius;
  const float invR = 1.0f / d.radius;
  const float minWeight = 1.0f - d.strength;
  const float* mask = m.mask.empty() ? nullptr : m.mask.data();
  const std::int32_t paint = ctx.paintFaceSet;
  // Every face belongs to exactly one leaf, so leaves can be painted in parallel.
  parallelFor(0, ctx.leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const BvhLeaf& leaf = ctx.bvh.leaves()[ctx.leaves[i]];
      for (Index f = leaf.faceBegin; f < leaf.faceEnd; ++f) {
        std::int32_t& value = m.faceSets[f];
        if (value == paint || value < 0 || m.faceHe[f] == kInvalid || !ctx.filter.allowsFace(f)) continue;
        Vec3 c{0.0f};
        float masked = 0.0f;
        int n = 0;
        m.forEachFaceVertex(f, [&](Index v) {
          c += m.positions[v];
          if (mask) masked += mask[v];
          ++n;
        });
        c /= static_cast<float>(n);
        const Vec3 delta = c - d.center;
        const float dist2 = glm::dot(delta, delta);
        if (dist2 >= r2) continue;
        const float w = falloffWeight(d.falloff, std::sqrt(dist2) * invR);
        if (w <= 0.0f || w < minWeight || masked >= 0.5f * static_cast<float>(n)) continue;
        value = paint;
      }
    }
  });
}

}  // namespace plegl
