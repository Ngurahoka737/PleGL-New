#include "sculpt/Brush.h"

#include <cmath>
#include <vector>

#include "core/Parallel.h"

namespace plegl {

namespace {
// Displacement per dab at full strength, as a fraction of the radius. With the default spacing
// (a dab every 10% of the radius) a stroke at 50% strength raises the surface by roughly a
// quarter of the radius, which matches the feel of common sculpting tools.
constexpr float kDrawScale = 0.04f;
}  // namespace

void DrawBrush::apply(BrushContext& ctx) const {
  const Dab& d = ctx.dab;
  Mesh& m = ctx.mesh;
  const float r2 = d.radius * d.radius;
  const float invR = 1.0f / d.radius;
  const Vec3 offset = d.areaNormal * (d.radius * d.strength * kDrawScale * (d.invert ? -1.0f : 1.0f));
  parallelFor(0, ctx.leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const BvhLeaf& leaf = ctx.bvh.leaves()[ctx.leaves[i]];
      for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
        const Vec3 delta = m.positions[v] - d.center;
        const float dist2 = glm::dot(delta, delta);
        if (dist2 >= r2) continue;
        m.positions[v] += offset * falloffWeight(d.falloff, std::sqrt(dist2) * invR);
      }
    }
  });
}

void SmoothBrush::apply(BrushContext& ctx) const {
  const Dab& d = ctx.dab;
  Mesh& m = ctx.mesh;
  const float r2 = d.radius * d.radius;
  const float invR = 1.0f / d.radius;
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
        if (dist2 >= r2 || m.vertHe[v] == kInvalid) continue;

        Vec3 sum{0.0f}, borderSum{0.0f};
        int count = 0, borderCount = 0;
        m.forEachOutgoing(v, [&](Index h) {
          const Vec3& q = m.positions[m.heTarget(h)];
          sum += q;
          ++count;
          if (m.heTwin[h] == kInvalid) {
            borderSum += q;
            ++borderCount;
          }
          const Index prev = m.hePrev(h);
          if (m.heTwin[prev] == kInvalid) {  // Incoming border edge: its start is a neighbour too.
            borderSum += m.positions[m.heVert[prev]];
            ++borderCount;
          }
        });
        const Vec3 target = borderCount > 0 ? borderSum / static_cast<float>(borderCount)
                                            : sum / static_cast<float>(count);
        const float w = d.strength * falloffWeight(d.falloff, std::sqrt(dist2) * invR);
        out.emplace_back(v, p + (target - p) * w);
      }
    }
  });
  parallelFor(0, results.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i)
      for (const auto& [v, p] : results[i]) m.positions[v] = p;
  });
}

}  // namespace plegl
