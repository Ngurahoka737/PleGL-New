#include "sculpt/Brush.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <tuple>
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

// A move computed before any is written, for brushes that read neighbours.
struct Pending {
  Index v;
  float f;  // Raw falloff, for symmetric shares.
  Vec3 move;
};

// Changes a mask value by `delta` (toward a target in [0, 1]) where symmetric copies are active.
void writeMask(BrushContext& ctx, Index v, float f, float delta) {
  const std::int64_t s = ctx.shares->slotOf(v);
  delta *= ctx.shares->share(s, f);
  if (ctx.shares->deferred(s)) {
    ctx.shares->pendingScalar[s] += delta;
  } else {
    ctx.mesh.mask[v] = std::clamp(ctx.mesh.mask[v] + delta, 0.0f, 1.0f);
  }
}

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
        const float f = falloffWeight(d.falloff, std::sqrt(dist2) * invR);
        const float w = d.strength * f * unmasked(ctx, mask, v);
        if (w <= 0.0f) continue;
        Vec3 move = displace(v, p, w);
        const float len2 = glm::dot(move, move);
        if (len2 > maxMove2) move *= maxMove / std::sqrt(len2);
        writeMove(ctx, v, f, move);
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
        const float f = falloffWeight(d.falloff, std::sqrt(dist2) * invR);
        const float w = f * unmasked(ctx, mask, v);
        if (w <= 0.0f) continue;
        writeMove(ctx, v, f, offset * w);
      }
    }
  });
}

void SmoothBrush::apply(BrushContext& ctx) const {
  const Dab& d = ctx.dab;
  Mesh& m = ctx.mesh;
  const float r2 = d.radius * d.radius;
  const float invR = 1.0f / d.radius;
  // Clamped like every brush: on a coarse mesh the neighbour mean can lie farther than the reach
  // the sculptor recomputes normals in.
  const float maxMove = d.radius * kMaxDabMove;
  const float maxMove2 = maxMove * maxMove;
  const float* mask = m.mask.empty() ? nullptr : m.mask.data();
  // Smoothing reads neighbours, so compute every move first and write them afterwards.
  std::vector<std::vector<Pending>> results(ctx.leaves.size());
  parallelFor(0, ctx.leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const BvhLeaf& leaf = ctx.bvh.leaves()[ctx.leaves[i]];
      auto& out = results[i];
      for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
        const Vec3& p = m.positions[v];
        const Vec3 delta = p - d.center;
        const float dist2 = glm::dot(delta, delta);
        if (dist2 >= r2) continue;
        const float f = falloffWeight(d.falloff, std::sqrt(dist2) * invR);
        const float w = d.strength * f * unmasked(ctx, mask, v);
        if (w <= 0.0f) continue;
        neighbourMean<Vec3>(
            m, v, [&](Index u) { return m.positions[u]; },
            [&](const Vec3& target) {
              Vec3 move = (target - p) * w;
              const float len2 = glm::dot(move, move);
              if (len2 > maxMove2) move *= maxMove / std::sqrt(len2);
              out.push_back({v, f, move});
            });
      }
    }
  });
  parallelFor(0, results.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i)
      for (const Pending& r : results[i]) writeMove(ctx, r.v, r.f, r.move);
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

void SnakeHookBrush::apply(BrushContext& ctx) const {
  const Dab& d = ctx.dab;
  Mesh& m = ctx.mesh;
  const float pullLen = glm::length(d.pull);
  if (!(pullLen > 0.0f)) return;
  const float r2 = d.radius * d.radius;
  const float invR = 1.0f / d.radius;
  const float maxMove = d.radius * kMaxDabMove;
  const float maxMove2 = maxMove * maxMove;
  // Pinch draws vertices toward the pull axis by a share that grows with the step, so a pull split
  // into more sub-dabs narrows the same.
  const bool pinching = d.pinch > 0.0f && pullLen >= 1e-9f * d.radius;
  const Vec3 axis = d.pull / pullLen;
  const float pinchRate = pinching ? d.pinch * kSnakePinch * pullLen / (kSnakeStep * d.radius) : 0.0f;
  const float* mask = m.mask.empty() ? nullptr : m.mask.data();
  parallelFor(0, ctx.leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const BvhLeaf& leaf = ctx.bvh.leaves()[ctx.leaves[i]];
      for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
        const Vec3& p = m.positions[v];
        const Vec3 delta = p - d.center;
        const float dist2 = glm::dot(delta, delta);
        if (dist2 >= r2) continue;
        const float f = falloffWeight(d.falloff, std::sqrt(dist2) * invR);
        const float w = f * unmasked(ctx, mask, v);
        if (w <= 0.0f) continue;
        Vec3 move = d.pull * w;
        if (pinching) {
          const Vec3 toCenter = -delta;
          move += (toCenter - axis * glm::dot(axis, toCenter)) * (pinchRate * w);
        }
        const float len2 = glm::dot(move, move);
        if (len2 > maxMove2) move *= maxMove / std::sqrt(len2);
        writeMove(ctx, v, f, move);
      }
    }
  });
}

namespace {
// Adds `delta` to a layer offset and returns the vertex's new composite position.
Vec3 addToLayer(const LayerTarget& t, Index v, const Vec3& delta) {
  Vec3& o = (*t.offset)[v];
  o.x = o.x + delta.x;
  o.y = o.y + delta.y;
  o.z = o.z + delta.z;
  return composeVertex(*t.stack, v);
}

// Changes a layer offset by `delta` and sets the vertex to its composite. The composite moves by
// strength * delta, which is clamped to kMaxDabMove * radius like every brush, then scaled by the
// vertex's symmetric share (and deferred where copies overlap).
void moveLayer(BrushContext& ctx, Index v, float f, Vec3 delta, float maxMove) {
  const LayerTarget& t = *ctx.layer;
  const float move = std::abs(t.strength) * glm::length(delta);
  if (move > maxMove) delta *= maxMove / move;
  if (ctx.shares) {
    const std::int64_t s = ctx.shares->slotOf(v);
    delta *= ctx.shares->share(s, f);
    if (ctx.shares->deferred(s)) {
      ctx.shares->pending[s] += delta;
      return;
    }
  }
  ctx.mesh.positions[v] = addToLayer(t, v, delta);
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
        const float f = falloffWeight(d.falloff, std::sqrt(dist2) * invR);
        const float w = d.strength * f * unmasked(ctx, mask, v);
        if (w <= 0.0f) continue;
        const Vec3& o = (*t.offset)[v];
        if (isZero(o)) continue;
        moveLayer(ctx, v, f, Vec3{-(w * o.x), -(w * o.y), -(w * o.z)}, maxMove);
      }
    }
  });
}

void LayerSmoothBrush::apply(BrushContext& ctx) const {
  if (!ctx.layer || !ctx.layer->offset) return;
  const std::vector<Vec3>& offset = *ctx.layer->offset;
  const Dab& d = ctx.dab;
  Mesh& m = ctx.mesh;
  const float r2 = d.radius * d.radius;
  const float invR = 1.0f / d.radius;
  const float maxMove = d.radius * kMaxDabMove;
  const float* mask = m.mask.empty() ? nullptr : m.mask.data();
  // Like SmoothBrush: every new offset is computed from the old ones first, then written.
  std::vector<std::vector<Pending>> results(ctx.leaves.size());
  parallelFor(0, ctx.leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const BvhLeaf& leaf = ctx.bvh.leaves()[ctx.leaves[i]];
      auto& out = results[i];
      for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
        const Vec3 delta = m.positions[v] - d.center;
        const float dist2 = glm::dot(delta, delta);
        if (dist2 >= r2) continue;
        const float f = falloffWeight(d.falloff, std::sqrt(dist2) * invR);
        const float w = d.strength * f * unmasked(ctx, mask, v);
        if (w <= 0.0f) continue;
        neighbourMean<Vec3>(
            m, v, [&](Index u) { return offset[u]; },
            [&](const Vec3& mean) {
              const Vec3 change = (mean - offset[v]) * w;
              if (!isZero(change)) out.push_back({v, f, change});
            });
      }
    }
  });
  parallelFor(0, results.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i)
      for (const Pending& r : results[i]) moveLayer(ctx, r.v, r.f, r.move, maxMove);
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
        const float f = falloffWeight(d.falloff, std::sqrt(dist2) * invR);
        const float w = d.strength * f;
        if (w <= 0.0f || !ctx.filter.allows(m, v)) continue;
        float& value = m.mask[v];
        if (!ctx.shares) {
          value = w >= 1.0f ? target : std::clamp(value + (target - value) * w, 0.0f, 1.0f);
        } else {
          writeMask(ctx, v, f, (target - value) * std::min(w, 1.0f));
        }
      }
    }
  });
}

void MaskSmoothBrush::apply(BrushContext& ctx) const {
  const Dab& d = ctx.dab;
  Mesh& m = ctx.mesh;
  const float r2 = d.radius * d.radius;
  const float invR = 1.0f / d.radius;
  std::vector<std::vector<std::tuple<Index, float, float>>> results(ctx.leaves.size());
  parallelFor(0, ctx.leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const BvhLeaf& leaf = ctx.bvh.leaves()[ctx.leaves[i]];
      auto& out = results[i];
      for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
        const Vec3 delta = m.positions[v] - d.center;
        const float dist2 = glm::dot(delta, delta);
        if (dist2 >= r2) continue;
        const float f = falloffWeight(d.falloff, std::sqrt(dist2) * invR);
        const float w = d.strength * f;
        if (w <= 0.0f || !ctx.filter.allows(m, v)) continue;
        neighbourMeanAll<float>(
            m, v, [&](Index u) { return m.mask[u]; },
            [&](float target) { out.emplace_back(v, f, std::clamp(m.mask[v] + (target - m.mask[v]) * w, 0.0f, 1.0f)); });
      }
    }
  });
  parallelFor(0, results.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i)
      for (const auto& [v, f, value] : results[i]) {
        if (!ctx.shares) {
          m.mask[v] = value;
        } else {
          writeMask(ctx, v, f, value - m.mask[v]);
        }
      }
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
        if ((value == paint && !ctx.faceClaim) || value < 0 || m.faceHe[f] == kInvalid || !ctx.filter.allowsFace(f))
          continue;
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
        // With several copies the first one to reach a face paints it; later ones leave it.
        if (ctx.faceClaim && std::atomic_ref<std::uint32_t>(ctx.faceClaim[f]).exchange(
                                 ctx.claimStamp, std::memory_order_relaxed) == ctx.claimStamp)
          continue;
        value = paint;
      }
    }
  });
}

}  // namespace plegl
