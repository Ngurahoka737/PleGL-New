#include "sculpt/MaskOps.h"

#include <algorithm>

#include "core/Parallel.h"
#include "sculpt/Neighbours.h"

namespace plegl {

const char* maskOpName(MaskOp op) {
  switch (op) {
    case MaskOp::Invert: return "Invert Mask";
    case MaskOp::Clear: return "Clear Mask";
    case MaskOp::Fill: return "Fill Mask";
    case MaskOp::Blur: return "Blur Mask";
    case MaskOp::Sharpen: return "Sharpen Mask";
  }
  return "Mask";
}

namespace {
// Runs fn(v) for every vertex owned by a BVH leaf, in parallel per leaf. Isolated vertices belong
// to no leaf, so per-leaf undo could not restore them; they are not drawn either, so mask ops
// leave them alone.
template <typename Fn>
void forEachLeafVertex(const Bvh& bvh, Fn&& fn) {
  const auto& leaves = bvh.leaves();
  parallelFor(0, leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i)
      for (Index v = leaves[i].vertBegin; v < leaves[i].vertEnd; ++v) fn(v);
  });
}

// dst[v] = mean of src over the neighbours of v (src[v] for vertices without neighbours).
void neighbourMeans(const Mesh& m, const Bvh& bvh, const std::vector<float>& src, std::vector<float>& dst) {
  forEachLeafVertex(bvh, [&](Index v) {
    dst[v] = src[v];
    neighbourMeanAll<float>(m, v, [&](Index u) { return src[u]; }, [&](float mean) { dst[v] = mean; });
  });
}
}  // namespace

std::optional<SculptUndo> applyMaskOp(SceneObject& object, MaskOp op, int iterations) {
  Mesh& m = object.mesh;
  const Bvh& bvh = object.bvh;
  if (m.mask.empty() && op != MaskOp::Invert && op != MaskOp::Fill) return std::nullopt;  // Already all zero.
  m.ensureMask();
  const std::vector<float> old = m.mask;
  iterations = std::max(iterations, 1);

  switch (op) {
    case MaskOp::Invert: forEachLeafVertex(bvh, [&](Index v) { m.mask[v] = 1.0f - old[v]; }); break;
    case MaskOp::Clear: forEachLeafVertex(bvh, [&](Index v) { m.mask[v] = 0.0f; }); break;
    case MaskOp::Fill: forEachLeafVertex(bvh, [&](Index v) { m.mask[v] = 1.0f; }); break;
    case MaskOp::Blur:
    case MaskOp::Sharpen: {
      std::vector<float> src = old, mean(old.size());
      for (int it = 0; it < iterations; ++it) {
        neighbourMeans(m, bvh, src, mean);
        if (op == MaskOp::Blur) {
          forEachLeafVertex(bvh, [&](Index v) { m.mask[v] = 0.5f * (src[v] + mean[v]); });
        } else {
          forEachLeafVertex(bvh, [&](Index v) { m.mask[v] = std::clamp(2.0f * src[v] - mean[v], 0.0f, 1.0f); });
        }
        src = m.mask;
      }
      break;
    }
  }

  // Hidden parts keep their mask: only vertices of a visible face change.
  if (m.anyHidden()) {
    forEachLeafVertex(bvh, [&](Index v) {
      if (!m.vertexVisible(v)) m.mask[v] = old[v];
    });
  }

  // Record only the leaves whose values changed.
  const auto& leaves = bvh.leaves();
  std::vector<char> changed(leaves.size(), 0);
  parallelFor(0, leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const BvhLeaf& l = leaves[i];
      changed[i] = !std::equal(old.begin() + l.vertBegin, old.begin() + l.vertEnd, m.mask.begin() + l.vertBegin);
    }
  });
  SculptUndo entry;
  entry.label = maskOpName(op);
  entry.objectId = object.id;
  entry.topologyVersion = object.topologyVersion;
  for (std::size_t i = 0; i < leaves.size(); ++i) {
    if (!changed[i]) continue;
    const BvhLeaf& l = leaves[i];
    LeafState before, after;
    before.leaf = after.leaf = static_cast<Index>(i);
    before.mask.assign(old.begin() + l.vertBegin, old.begin() + l.vertEnd);
    after.mask.assign(m.mask.begin() + l.vertBegin, m.mask.begin() + l.vertEnd);
    entry.before.push_back(std::move(before));
    entry.after.push_back(std::move(after));
  }
  if (entry.before.empty()) return std::nullopt;
  if (entry.before.size() > kMaskDirtyAllLeaves) {
    object.markMaskDirtyAll();
  } else {
    for (const LeafState& s : entry.before) object.markMaskDirty(s.leaf);
  }
  return entry;
}

}  // namespace plegl
