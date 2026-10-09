#include "sculpt/FaceSetOps.h"

#include <algorithm>

#include "core/Parallel.h"

namespace plegl {

const char* faceSetOpName(FaceSetOp op) {
  switch (op) {
    case FaceSetOp::FromMask: return "Face Set from Mask";
    case FaceSetOp::FromLooseParts: return "Face Sets from Loose Parts";
    case FaceSetOp::Clear: return "Clear Face Sets";
    case FaceSetOp::Hide: return "Hide Face Set";
    case FaceSetOp::Isolate: return "Isolate Face Set";
    case FaceSetOp::RevealAll: return "Reveal All";
    case FaceSetOp::InvertVisibility: return "Invert Visibility";
  }
  return "Face Sets";
}

namespace {

// Runs fn(f) for every face owned by a BVH leaf, in parallel per leaf.
template <typename Fn>
void forEachLeafFace(const Bvh& bvh, Fn&& fn) {
  const auto& leaves = bvh.leaves();
  parallelFor(0, leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i)
      for (Index f = leaves[i].faceBegin; f < leaves[i].faceEnd; ++f) fn(f);
  });
}

// Builds the undo entry for the leaves whose face sets differ from `old`, and marks them for
// upload: colours always, index data where a face was hidden or shown.
std::optional<SculptUndo> recordFaceSets(SceneObject& object, const std::vector<std::int32_t>& old, const char* label) {
  const Mesh& m = object.mesh;
  const auto& leaves = object.bvh.leaves();
  // 0: unchanged, 1: face sets changed, 2: visibility changed as well.
  std::vector<char> changed(leaves.size(), 0);
  parallelFor(0, leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const BvhLeaf& l = leaves[i];
      char c = 0;
      for (Index f = l.faceBegin; f < l.faceEnd && c < 2; ++f) {
        if (old[f] == m.faceSets[f]) continue;
        c = (old[f] < 0) != (m.faceSets[f] < 0) ? 2 : 1;
      }
      changed[i] = c;
    }
  });
  SculptUndo entry;
  entry.label = label;
  entry.objectId = object.id;
  entry.topologyVersion = object.topologyVersion;
  for (std::size_t i = 0; i < leaves.size(); ++i) {
    if (!changed[i]) continue;
    const BvhLeaf& l = leaves[i];
    LeafState before, after;
    before.leaf = after.leaf = static_cast<Index>(i);
    before.faceSets.assign(old.begin() + l.faceBegin, old.begin() + l.faceEnd);
    after.faceSets.assign(m.faceSets.begin() + l.faceBegin, m.faceSets.begin() + l.faceEnd);
    entry.before.push_back(std::move(before));
    entry.after.push_back(std::move(after));
    if (changed[i] == 2) object.markVisibilityDirty(static_cast<Index>(i));
  }
  if (entry.before.empty()) return std::nullopt;
  if (entry.before.size() > kMaskDirtyAllLeaves) {
    object.markFaceSetDirtyAll();
  } else {
    for (const LeafState& s : entry.before) object.markFaceSetDirty(s.leaf);
  }
  return entry;
}

// Gives every connected part of the mesh (faces joined across edges) its own set, starting at
// `first`, keeping each face's visibility.
void assignLooseParts(Mesh& m, const Bvh& bvh, std::int32_t first) {
  const Index nf = m.faceCount();
  std::vector<std::int32_t> part(static_cast<std::size_t>(nf), 0);
  std::vector<Index> stack;
  std::int32_t next = first;
  const auto& leaves = bvh.leaves();
  for (const BvhLeaf& leaf : leaves) {
    for (Index seed = leaf.faceBegin; seed < leaf.faceEnd; ++seed) {
      if (part[seed] != 0 || m.faceHe[seed] == kInvalid) continue;
      const std::int32_t id = next++;
      part[seed] = id;
      stack.assign(1, seed);
      while (!stack.empty()) {
        const Index f = stack.back();
        stack.pop_back();
        const Index start = m.faceHe[f];
        Index h = start;
        do {
          const Index t = m.heTwin[h];
          if (t != kInvalid) {
            const Index g = m.heFace[t];
            if (part[g] == 0) {
              part[g] = id;
              stack.push_back(g);
            }
          }
          h = m.heNext[h];
        } while (h != start);
      }
    }
  }
  forEachLeafFace(bvh, [&](Index f) {
    if (part[f] != 0) m.faceSets[f] = m.faceSets[f] < 0 ? -part[f] : part[f];
  });
}

}  // namespace

std::optional<SculptUndo> applyFaceSetOp(SceneObject& object, FaceSetOp op, std::int32_t faceSet) {
  Mesh& m = object.mesh;
  const Bvh& bvh = object.bvh;
  const bool any = !m.faceSets.empty();
  // Without face sets every face is in the default set and visible.
  if (!any && (op == FaceSetOp::Clear || op == FaceSetOp::RevealAll)) return std::nullopt;
  if (op == FaceSetOp::FromMask && (!m.anyMasked() || object.newFaceSetId() == 0)) return std::nullopt;
  if ((op == FaceSetOp::Hide || op == FaceSetOp::Isolate) && faceSet <= 0) return std::nullopt;
  m.ensureFaceSets();
  const std::vector<std::int32_t> old = m.faceSets;
  std::vector<std::int32_t>& sets = m.faceSets;

  switch (op) {
    case FaceSetOp::FromMask: {
      const std::int32_t id = object.newFaceSetId();
      forEachLeafFace(bvh, [&](Index f) {
        if (sets[f] < 0) return;
        bool all = true;
        m.forEachFaceVertex(f, [&](Index v) { all &= m.mask[v] >= 0.5f; });
        if (all) sets[f] = id;
      });
      break;
    }
    case FaceSetOp::FromLooseParts:
      assignLooseParts(m, bvh, kDefaultFaceSet + 1);  // Every part gets a colour.
      break;
    case FaceSetOp::Clear:
      forEachLeafFace(bvh, [&](Index f) { sets[f] = sets[f] < 0 ? -kDefaultFaceSet : kDefaultFaceSet; });
      break;
    case FaceSetOp::Hide:
      forEachLeafFace(bvh, [&](Index f) {
        if (sets[f] == faceSet) sets[f] = -faceSet;
      });
      break;
    case FaceSetOp::Isolate: {
      // Already isolated (that set is all that shows, and it shows): show everything instead.
      bool isolated = true, shown = false;
      for (Index f = 0; f < m.faceCount() && isolated; ++f) {
        if (m.faceHe[f] == kInvalid) continue;
        const bool visible = sets[f] > 0;
        const bool inSet = faceSetId(sets[f]) == faceSet;
        isolated = visible == inSet;
        shown |= visible;
      }
      if (isolated && shown) {
        forEachLeafFace(bvh, [&](Index f) { sets[f] = faceSetId(sets[f]); });
      } else {
        forEachLeafFace(bvh, [&](Index f) {
          const std::int32_t id = faceSetId(sets[f]);
          sets[f] = id == faceSet ? id : -id;
        });
      }
      break;
    }
    case FaceSetOp::RevealAll:
      forEachLeafFace(bvh, [&](Index f) { sets[f] = faceSetId(sets[f]); });
      break;
    case FaceSetOp::InvertVisibility:
      forEachLeafFace(bvh, [&](Index f) { sets[f] = -sets[f]; });
      break;
  }
  return recordFaceSets(object, old, faceSetOpName(op));
}

std::optional<SculptUndo> maskFaceSet(SceneObject& object, std::int32_t faceSet) {
  Mesh& m = object.mesh;
  const Bvh& bvh = object.bvh;
  if (faceSet <= 0) return std::nullopt;
  if (m.faceSets.empty() && faceSet != kDefaultFaceSet) return std::nullopt;
  m.ensureMask();
  const std::vector<float> old = m.mask;
  // A vertex can be shared by faces of several leaves; each leaf writes only its own vertices.
  const auto& leaves = bvh.leaves();
  parallelFor(0, leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      for (Index v = leaves[i].vertBegin; v < leaves[i].vertEnd; ++v) {
        if (m.mask[v] >= 1.0f) continue;
        bool inSet = false;
        m.forEachOutgoing(v, [&](Index h) { inSet |= m.faceSetValue(m.heFace[h]) == faceSet; });
        if (inSet) m.mask[v] = 1.0f;
      }
    }
  });

  std::vector<char> changed(leaves.size(), 0);
  parallelFor(0, leaves.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const BvhLeaf& l = leaves[i];
      changed[i] = !std::equal(old.begin() + l.vertBegin, old.begin() + l.vertEnd, m.mask.begin() + l.vertBegin);
    }
  });
  SculptUndo entry;
  entry.label = "Mask Face Set";
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
