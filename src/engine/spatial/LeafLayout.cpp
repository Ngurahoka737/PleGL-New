#include "spatial/LeafLayout.h"

#include <algorithm>
#include <numeric>
#include <string>

#include "core/Parallel.h"

namespace plegl {
namespace {

// Map values for live region vertices while a consolidation is being planned.
constexpr Index kRegionUnused = -2;  // Not used by any region face (so far).
constexpr Index kRegionUsed = -3;    // Used by a region face, not numbered yet.
// Set on the vertex and half-edge map values of rebuilt elements, so copying an untouched leaf can
// tell which of its references lead into the region without a leaf lookup. Meshes stay far below
// 2^30 elements. Clearing the bit keeps kInvalid and the values above negative.
constexpr Index kRegionBit = Index{1} << 30;
inline Index strip(Index v) { return v & ~kRegionBit; }

Index faceSpan(const BvhLeaf& l) { return l.faceEnd - l.faceBegin; }
Index vertSpan(const BvhLeaf& l) { return l.vertEnd - l.vertBegin; }
Index heSpan(const BvhLeaf& l) { return l.heEnd - l.heBegin; }

bool sameSizes(const BvhLeaf& a, const BvhLeaf& b) {
  return faceSpan(a) == faceSpan(b) && vertSpan(a) == vertSpan(b) && heSpan(a) == heSpan(b);
}

template <class T>
void copySlice(const std::vector<T>& src, Index begin, Index end, std::vector<T>& out) {
  out.assign(src.begin() + begin, src.begin() + end);
}

template <class T>
std::size_t sliceBytes(const std::vector<T>& v) {
  return v.size() * sizeof(T);
}

// Sizes `w` to n elements, reserving at least the capacity of `like` so the arrays that ping-pong
// between a mesh and the workspace keep the mesh's headroom. Arrays far larger than that (left by
// a bigger object sharing the workspace) are given up, or the smaller mesh would keep them.
template <class T>
void fit(std::vector<T>& w, const std::vector<T>& like, Index n) {
  const std::size_t want = std::max(like.capacity(), static_cast<std::size_t>(n));
  if (w.capacity() < want || w.capacity() > 2 * want + 65536) {
    std::vector<T>().swap(w);  // Reallocate without copying contents that are about to be overwritten.
    w.reserve(want);
  }
  w.resize(static_cast<std::size_t>(n));
}

void fitMesh(const Mesh& like, Mesh& w, Index nv, Index nf, Index nh, bool normals, bool mask, bool faceSets) {
  fit(w.positions, like.positions, nv);
  fit(w.vertHe, like.vertHe, nv);
  if (normals) {
    fit(w.normals, like.normals, nv);
  } else {
    w.normals.clear();
  }
  if (mask) {
    fit(w.mask, like.mask, nv);
  } else {
    w.mask.clear();
  }
  fit(w.faceHe, like.faceHe, nf);
  if (faceSets) {
    fit(w.faceSets, like.faceSets, nf);
  } else {
    w.faceSets.clear();
  }
  fit(w.heNext, like.heNext, nh);
  fit(w.heTwin, like.heTwin, nh);
  fit(w.heVert, like.heVert, nh);
  fit(w.heFace, like.heFace, nh);
}

// Where references are sent while copying one source leaf into the new layout.
struct CopyTarget {
  const Index* vmap;
  const Index* fmap;
  const Index* hmap;
  const Vec3* normalsIn;  // nullptr when the source mesh has no normals.
  const float* maskIn;    // nullptr: the source has no mask, write zeros.
  const std::int32_t* faceSetsIn;  // nullptr: the source has no face sets, write the default.
  Mesh* out;
  bool normalsOut;
  bool maskOut;
  bool faceSetsOut;
};

// Where a consolidation's appended and isolated elements start.
struct RefContext {
  Index isoBegin, tailVert, tailHe;
};

struct LeafCopy {
  bool bad = false;  // An element or reference that cannot be mapped (a missed claim).
  std::vector<std::pair<Index, Index>> vertRefs, heRefs;
};

// Copies one source leaf to its mapped place. kRegion: the leaf may hold removed elements, which
// are skipped. kRecord: the leaf is untouched; it must hold no removed element and no reference to
// an appended or isolated element, and references into the region are recorded as (old, new) pairs.
template <bool kRegion, bool kRecord>
void copyLeaf(const Mesh& m, const BvhLeaf& src, const CopyTarget& t, const RefContext* ctx, LeafCopy& res) {
  Mesh& w = *t.out;
  for (Index f = src.faceBegin; f < src.faceEnd; ++f) {
    if (m.faceHe[f] == kInvalid) {
      if (!kRegion) res.bad = true;
      continue;
    }
    const Index nf = t.fmap[f];
    const Index he = strip(t.hmap[m.faceHe[f]]);
    if (nf < 0 || he < 0) {
      res.bad = true;
      continue;
    }
    w.faceHe[nf] = he;
    if (t.faceSetsOut) w.faceSets[nf] = t.faceSetsIn ? t.faceSetsIn[f] : kDefaultFaceSet;
  }
  for (Index h = src.heBegin; h < src.heEnd; ++h) {
    if (m.heFace[h] == kInvalid) {
      if (!kRegion) res.bad = true;
      continue;
    }
    const Index nh = strip(t.hmap[h]);
    const Index next = strip(t.hmap[m.heNext[h]]);
    const Index face = t.fmap[m.heFace[h]];
    const Index u = m.heVert[h];
    const Index rawVert = t.vmap[u];
    const Index twinOld = m.heTwin[h];
    const Index rawTwin = twinOld == kInvalid ? kInvalid : t.hmap[twinOld];
    if (nh < 0 || next < 0 || face < 0 || rawVert < 0 || (twinOld != kInvalid && rawTwin < 0)) {
      res.bad = true;
      continue;
    }
    if constexpr (kRecord) {
      if (twinOld != kInvalid && (rawTwin & kRegionBit)) {
        if (twinOld >= ctx->tailHe) {
          res.bad = true;
        } else {
          res.heRefs.emplace_back(twinOld, strip(rawTwin));
        }
      }
      if (rawVert & kRegionBit) {
        if (u >= ctx->tailVert) {
          res.bad = true;
        } else {
          res.vertRefs.emplace_back(u, strip(rawVert));
        }
      } else if (u >= ctx->isoBegin) {
        res.bad = true;  // An isolated vertex: no face can use it.
      }
    }
    w.heNext[nh] = next;
    w.heFace[nh] = face;
    w.heVert[nh] = strip(rawVert);
    w.heTwin[nh] = twinOld == kInvalid ? kInvalid : strip(rawTwin);
  }
  for (Index v = src.vertBegin; v < src.vertEnd; ++v) {
    const Index e = m.vertHe[v];
    if (e == kInvalid) {
      if (!kRegion) res.bad = true;
      continue;
    }
    const Index nv = strip(t.vmap[v]);
    const Index rawE = t.hmap[e];
    const Index ne = strip(rawE);
    if (nv < 0 || ne < 0) {
      res.bad = true;
      continue;
    }
    if constexpr (kRecord) {
      if (rawE & kRegionBit) {
        if (e >= ctx->tailHe) {
          res.bad = true;
        } else {
          res.heRefs.emplace_back(e, ne);
        }
      }
    }
    w.positions[nv] = m.positions[v];
    if (t.normalsOut) w.normals[nv] = t.normalsIn ? t.normalsIn[v] : Vec3{0.0f};
    if (t.maskOut) w.mask[nv] = t.maskIn ? t.maskIn[v] : 0.0f;
    w.vertHe[nv] = ne;
  }
}

// Isolated vertices: no faces, so only attributes move.
void copyIsolated(const Mesh& m, Index begin, Index end, Index to, const CopyTarget& t) {
  Mesh& w = *t.out;
  for (Index v = begin; v < end; ++v, ++to) {
    w.positions[to] = m.positions[v];
    if (t.normalsOut) w.normals[to] = t.normalsIn ? t.normalsIn[v] : Vec3{0.0f};
    if (t.maskOut) w.mask[to] = t.maskIn ? t.maskIn[v] : 0.0f;
    w.vertHe[to] = kInvalid;
  }
}

void fillShift(std::vector<Index>& map, Index begin, Index end, Index to) {
  for (Index i = begin; i < end; ++i) map[i] = i - begin + to;
}

void fillInvalid(std::vector<Index>& map, Index begin, Index end) {
  std::fill(map.begin() + begin, map.begin() + end, kInvalid);
}

void mergeRefs(std::vector<LeafCopy>& copies, LayoutDelta& delta) {
  for (LeafCopy& c : copies) {
    delta.vertRefs.insert(delta.vertRefs.end(), c.vertRefs.begin(), c.vertRefs.end());
    delta.heRefs.insert(delta.heRefs.end(), c.heRefs.begin(), c.heRefs.end());
  }
  for (auto* refs : {&delta.vertRefs, &delta.heRefs}) {
    std::sort(refs->begin(), refs->end());
    refs->erase(std::unique(refs->begin(), refs->end()), refs->end());
  }
}

// One rebuilt leaf of a consolidation.
struct RegionSlot {
  Index leaf = kInvalid;
  Index groupBegin = 0, groupEnd = 0;  // Range of the partitioned region face order.
  std::vector<Index> orphans;
};

struct LayoutPlan {
  std::vector<BvhLeaf> leaves;  // New leaves; region leaves still need bounds.
  std::vector<char> region;     // Per new leaf.
  std::vector<RegionSlot> slots;
  Index vertexCount = 0, faceCount = 0, halfEdgeCount = 0;
  Index isoBegin = 0;  // First isolated vertex.
  Index regionFaces = 0;
};

// Numbers every element of the new layout: fills ws.vmap/fmap/hmap (in-stroke index -> new index,
// kInvalid for removed elements) and the new leaf ranges.
LayoutPlan planConsolidation(const Mesh& m, const Bvh& bvh, const std::vector<char>& claimed, LayoutWorkspace& ws) {
  const auto old = bvh.leaves();
  const Index T = bvh.firstTailLeaf();
  const Index L = static_cast<Index>(old.size());
  const Index tv = bvh.tailStartVertex();
  const Index isoBegin = T > 0 ? old[T - 1].vertEnd : 0;
  std::vector<Index>& vmap = ws.vmap;
  std::vector<Index>& fmap = ws.fmap;
  std::vector<Index>& hmap = ws.hmap;
  vmap.resize(static_cast<std::size_t>(m.vertexCount()));
  fmap.resize(static_cast<std::size_t>(m.faceCount()));
  hmap.resize(static_cast<std::size_t>(m.halfEdgeCount()));

  // Source ranges of the region: claimed old leaves and every tail leaf.
  std::vector<Index> sources;
  for (Index l = 0; l < T; ++l)
    if (claimed[l]) sources.push_back(l);
  for (Index l = T; l < L; ++l) sources.push_back(l);

  // Untouched vertices get a placeholder for now; live region vertices are marked.
  parallelFor(0, static_cast<std::size_t>(T), 4, [&](std::size_t b, std::size_t e) {
    for (std::size_t l = b; l < e; ++l)
      if (!claimed[l]) fillInvalid(vmap, old[l].vertBegin, old[l].vertEnd);
  });
  fillInvalid(vmap, isoBegin, tv);
  for (Index l : sources) {
    const BvhLeaf& s = old[l];
    fillInvalid(fmap, s.faceBegin, s.faceEnd);
    fillInvalid(hmap, s.heBegin, s.heEnd);
    for (Index v = s.vertBegin; v < s.vertEnd; ++v) vmap[v] = m.vertHe[v] != kInvalid ? kRegionUnused : kInvalid;
  }

  // Live region faces, and which region vertices they use.
  std::vector<Index> faces;
  for (Index l : sources) {
    for (Index f = old[l].faceBegin; f < old[l].faceEnd; ++f) {
      if (m.faceHe[f] == kInvalid) continue;
      faces.push_back(f);
      m.forEachFaceVertex(f, [&](Index v) {
        if (vmap[v] == kRegionUnused) vmap[v] = kRegionUsed;
      });
    }
  }
  std::vector<Vec3> centroids(faces.size());
  parallelFor(0, faces.size(), 4096, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) centroids[i] = m.faceCentroid(faces[i]);
  });
  std::vector<Index> order(faces.size());
  std::iota(order.begin(), order.end(), 0);
  const std::vector<Index> groups = partitionFaces(centroids, order, bvh.maxLeafFaces());
  const Index groupCount = static_cast<Index>(groups.size()) - 1;

  // Leaf indices: claimed first, then empty leaves, then new ones at the end.
  std::vector<Index> slotLeaves;
  for (Index l = 0; l < T; ++l)
    if (claimed[l]) slotLeaves.push_back(l);
  for (Index l = 0; l < T && static_cast<Index>(slotLeaves.size()) < groupCount; ++l)
    if (!claimed[l] && old[l].empty()) slotLeaves.push_back(l);
  Index leafCount = T;
  while (static_cast<Index>(slotLeaves.size()) < groupCount) slotLeaves.push_back(leafCount++);

  // Orphans: live region vertices that no region face uses any more (only untouched faces do).
  std::vector<Index> orphans;
  for (Index l : sources)
    for (Index v = old[l].vertBegin; v < old[l].vertEnd; ++v)
      if (vmap[v] == kRegionUnused) orphans.push_back(v);
  if (!orphans.empty() && slotLeaves.empty()) slotLeaves.push_back(leafCount++);

  LayoutPlan plan;
  plan.slots.resize(slotLeaves.size());
  std::vector<Vec3> groupCenter(slotLeaves.size(), Vec3{0.0f});
  for (std::size_t s = 0; s < slotLeaves.size(); ++s) {
    RegionSlot& slot = plan.slots[s];
    slot.leaf = slotLeaves[s];
    if (static_cast<Index>(s) < groupCount) {
      slot.groupBegin = groups[s];
      slot.groupEnd = groups[s + 1];
      Aabb b;
      for (Index i = slot.groupBegin; i < slot.groupEnd; ++i) b.expand(centroids[order[i]]);
      groupCenter[s] = b.center();
    }
  }
  for (Index v : orphans) {
    std::size_t best = 0;
    float bestD = std::numeric_limits<float>::infinity();
    for (std::size_t s = 0; s < plan.slots.size(); ++s) {
      if (plan.slots[s].groupBegin == plan.slots[s].groupEnd) continue;
      const Vec3 d = groupCenter[s] - m.positions[v];
      const float d2 = glm::dot(d, d);
      if (d2 < bestD) {
        bestD = d2;
        best = s;
      }
    }
    plan.slots[best].orphans.push_back(v);
  }
  std::sort(plan.slots.begin(), plan.slots.end(),
            [](const RegionSlot& a, const RegionSlot& b) { return a.leaf < b.leaf; });

  // Walk the new leaves in order, numbering region elements and placing untouched leaves.
  plan.leaves.resize(static_cast<std::size_t>(leafCount));
  plan.region.assign(static_cast<std::size_t>(leafCount), 0);
  for (const RegionSlot& slot : plan.slots) plan.region[slot.leaf] = 1;
  Index f = 0, h = 0, v = 0;
  std::size_t nextSlot = 0;
  for (Index l = 0; l < leafCount; ++l) {
    BvhLeaf& out = plan.leaves[l];
    if (!plan.region[l]) {
      const BvhLeaf& o = old[l];
      out = o;
      out.flags &= static_cast<std::uint8_t>(~kLeafMayHaveDead);
      out.faceBegin = f;
      out.heBegin = h;
      out.vertBegin = v;
      f += faceSpan(o);
      h += heSpan(o);
      v += vertSpan(o);
      out.faceEnd = f;
      out.heEnd = h;
      out.vertEnd = v;
      continue;
    }
    const RegionSlot& slot = plan.slots[nextSlot++];
    out = BvhLeaf{};
    out.faceBegin = f;
    out.heBegin = h;
    out.vertBegin = v;
    for (Index i = slot.groupBegin; i < slot.groupEnd; ++i) {
      const Index face = faces[order[i]];
      fmap[face] = f++;
      const Index start = m.faceHe[face];
      Index e = start;
      do {
        hmap[e] = (h++) | kRegionBit;
        const Index u = m.heVert[e];
        if (vmap[u] == kRegionUsed) vmap[u] = (v++) | kRegionBit;
        e = m.heNext[e];
      } while (e != start);
    }
    for (Index u : slot.orphans) vmap[u] = (v++) | kRegionBit;
    out.faceEnd = f;
    out.heEnd = h;
    out.vertEnd = v;
    if (!slot.orphans.empty()) out.flags = kLeafOwnsOrphans;
  }
  const Index isoNew = v;

  // Untouched leaves and isolated vertices only shift.
  parallelFor(0, static_cast<std::size_t>(T), 4, [&](std::size_t b, std::size_t e) {
    for (std::size_t l = b; l < e; ++l) {
      if (plan.region[l]) continue;
      const BvhLeaf& o = old[l];
      const BvhLeaf& n = plan.leaves[l];
      fillShift(fmap, o.faceBegin, o.faceEnd, n.faceBegin);
      fillShift(hmap, o.heBegin, o.heEnd, n.heBegin);
      fillShift(vmap, o.vertBegin, o.vertEnd, n.vertBegin);
    }
  });
  fillShift(vmap, isoBegin, tv, isoNew);

  plan.faceCount = f;
  plan.halfEdgeCount = h;
  plan.isoBegin = isoNew;
  plan.vertexCount = isoNew + (tv - isoBegin);
  plan.regionFaces = static_cast<Index>(faces.size());
  return plan;
}

// Copies the in-stroke mesh into ws.mesh following the plan. Returns false on a missed claim.
bool copyConsolidation(const Mesh& m, const Bvh& bvh, const std::vector<char>& claimed, const LayoutPlan& plan,
                       LayoutWorkspace& ws, LayoutDelta& delta) {
  const auto old = bvh.leaves();
  const Index T = bvh.firstTailLeaf();
  const Index L = static_cast<Index>(old.size());
  const Index isoBegin = T > 0 ? old[T - 1].vertEnd : 0;
  const bool normals = m.normals.size() == m.positions.size();
  const bool mask = !m.mask.empty();
  const bool faceSets = !m.faceSets.empty();
  fitMesh(m, ws.mesh, plan.vertexCount, plan.faceCount, plan.halfEdgeCount, normals, mask, faceSets);

  const CopyTarget target{ws.vmap.data(),
                          ws.fmap.data(),
                          ws.hmap.data(),
                          normals ? m.normals.data() : nullptr,
                          mask ? m.mask.data() : nullptr,
                          faceSets ? m.faceSets.data() : nullptr,
                          &ws.mesh,
                          normals,
                          mask,
                          faceSets};
  const RefContext ctx{isoBegin, bvh.tailStartVertex(), bvh.tailStartHalfEdge()};
  std::vector<LeafCopy> copies(static_cast<std::size_t>(L));
  parallelFor(0, static_cast<std::size_t>(L), 2, [&](std::size_t b, std::size_t e) {
    for (std::size_t l = b; l < e; ++l) {
      const bool untouched = static_cast<Index>(l) < T && !claimed[l];
      if (untouched) {
        copyLeaf<false, true>(m, old[l], target, &ctx, copies[l]);
      } else {
        copyLeaf<true, false>(m, old[l], target, nullptr, copies[l]);
      }
    }
  });
  copyIsolated(m, isoBegin, bvh.tailStartVertex(), plan.isoBegin, target);
  for (const LeafCopy& c : copies)
    if (c.bad) return false;
  mergeRefs(copies, delta);
  return true;
}

Aabb regionBounds(const Mesh& m, const BvhLeaf& leaf) {
  Aabb b;
  for (Index e = leaf.heBegin; e < leaf.heEnd; ++e) b.expand(m.positions[m.heVert[e]]);
  if (leaf.flags & kLeafOwnsOrphans)
    for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) b.expand(m.positions[v]);
  return b;
}

}  // namespace

// --- Leaf snapshots ----------------------------------------------------------------------------

std::size_t TopoLeafState::bytes() const {
  return sliceBytes(positions) + sliceBytes(normals) + sliceBytes(mask) + sliceBytes(faceSets) + sliceBytes(vertHe) +
         sliceBytes(faceHe) + sliceBytes(heNext) + sliceBytes(heTwin) + sliceBytes(heVert) + sliceBytes(heFace);
}

TopoLeafState captureLeaf(const Mesh& m, const Bvh& bvh, Index leaf) {
  TopoLeafState s;
  s.leaf = leaf;
  s.ranges = bvh.leaves()[leaf];
  const BvhLeaf& r = s.ranges;
  copySlice(m.positions, r.vertBegin, r.vertEnd, s.positions);
  if (m.normals.size() == m.positions.size()) copySlice(m.normals, r.vertBegin, r.vertEnd, s.normals);
  if (!m.mask.empty()) copySlice(m.mask, r.vertBegin, r.vertEnd, s.mask);
  if (!m.faceSets.empty()) copySlice(m.faceSets, r.faceBegin, r.faceEnd, s.faceSets);
  copySlice(m.vertHe, r.vertBegin, r.vertEnd, s.vertHe);
  copySlice(m.faceHe, r.faceBegin, r.faceEnd, s.faceHe);
  copySlice(m.heNext, r.heBegin, r.heEnd, s.heNext);
  copySlice(m.heTwin, r.heBegin, r.heEnd, s.heTwin);
  copySlice(m.heVert, r.heBegin, r.heEnd, s.heVert);
  copySlice(m.heFace, r.heBegin, r.heEnd, s.heFace);
  return s;
}

// --- Claims ------------------------------------------------------------------------------------

void ClaimRecorder::begin(const Mesh& mesh, Bvh& bvh) {
  mesh_ = &mesh;
  bvh_ = &bvh;
  tailFace_ = bvh.tailStartFace();
  tailVert_ = bvh.tailStartVertex();
  tailHe_ = bvh.tailStartHalfEdge();
  slot_.assign(static_cast<std::size_t>(bvh.firstTailLeaf()), -1);
  claims_.clear();
  order_.clear();
  std::fill(std::begin(lastLeaf_), std::end(lastLeaf_), kInvalid);
}

Index ClaimRecorder::findLeaf(ElementKind kind, Index index) {
  const auto leaves = bvh_->leaves();
  const int k = static_cast<int>(kind);
  auto inside = [&](const BvhLeaf& l) {
    switch (kind) {
      case ElementKind::Vertex:
        return index >= l.vertBegin && index < l.vertEnd;
      case ElementKind::Face:
        return index >= l.faceBegin && index < l.faceEnd;
      case ElementKind::HalfEdge:
        return index >= l.heBegin && index < l.heEnd;
    }
    return false;
  };
  if (lastLeaf_[k] != kInvalid && inside(leaves[lastLeaf_[k]])) return lastLeaf_[k];
  Index leaf = kInvalid;
  switch (kind) {
    case ElementKind::Vertex:
      leaf = bvh_->leafOfVertex(index);
      break;
    case ElementKind::Face:
      leaf = bvh_->leafOfFace(index);
      break;
    case ElementKind::HalfEdge:
      leaf = bvh_->leafOfHalfEdge(index);
      break;
  }
  lastLeaf_[k] = leaf;
  return leaf;
}

void ClaimRecorder::beforeWrite(ElementKind kind, Index index) {
  const Index tail = kind == ElementKind::Vertex ? tailVert_ : kind == ElementKind::Face ? tailFace_ : tailHe_;
  if (index < 0 || index >= tail) return;  // Appended this stroke: owned by a tail leaf.
  const Index leaf = findLeaf(kind, index);
  if (leaf != kInvalid) claimLeaf(leaf);
}

void ClaimRecorder::claimLeaf(Index leaf) {
  if (leaf < 0 || leaf >= static_cast<Index>(slot_.size()) || slot_[leaf] >= 0) return;
  slot_[leaf] = static_cast<std::int32_t>(claims_.size());
  claims_.push_back(captureLeaf(*mesh_, *bvh_, leaf));
  order_.push_back(leaf);
  bvh_->addLeafFlags(leaf, kLeafMayHaveDead);
}

std::vector<TopoLeafState> ClaimRecorder::takeClaims() {
  std::vector<TopoLeafState> out = std::move(claims_);
  std::sort(out.begin(), out.end(), [](const TopoLeafState& a, const TopoLeafState& b) { return a.leaf < b.leaf; });
  claims_.clear();
  order_.clear();
  slot_.clear();
  mesh_ = nullptr;
  bvh_ = nullptr;
  return out;
}

// --- Consolidation -----------------------------------------------------------------------------

std::size_t LayoutWorkspace::bytes() const {
  return sliceBytes(mesh.positions) + sliceBytes(mesh.normals) + sliceBytes(mesh.mask) + sliceBytes(mesh.faceSets) +
         sliceBytes(mesh.vertHe) + sliceBytes(mesh.faceHe) + sliceBytes(mesh.heNext) + sliceBytes(mesh.heTwin) +
         sliceBytes(mesh.heVert) + sliceBytes(mesh.heFace) + sliceBytes(vmap) + sliceBytes(fmap) + sliceBytes(hmap);
}

std::size_t LayoutDelta::bytes() const {
  return regionLeaves.size() * sizeof(Index) + (vertRefs.size() + heRefs.size()) * sizeof(std::pair<Index, Index>);
}

ConsolidateResult consolidate(Mesh& mesh, Bvh& bvh, std::span<const Index> claimedLeaves, LayoutWorkspace& ws) {
  ConsolidateResult result;
  if (!bvh.dynamic()) return result;
  result.beforeVertexCount = bvh.tailStartVertex();
  result.beforeFaceCount = bvh.tailStartFace();
  result.beforeHalfEdgeCount = bvh.tailStartHalfEdge();
  const Index T = bvh.firstTailLeaf();
  const bool appended = mesh.faceCount() != bvh.tailStartFace() || mesh.vertexCount() != bvh.tailStartVertex() ||
                        mesh.halfEdgeCount() != bvh.tailStartHalfEdge();
  if (claimedLeaves.empty() && !appended) {
    bvh.endDynamic();
    return result;
  }
  bvh.growTail(mesh);  // In case the caller did not after its last edit.

  std::vector<char> claimed(static_cast<std::size_t>(T), 0);
  for (Index l : claimedLeaves)
    if (l >= 0 && l < T) claimed[l] = 1;

  LayoutPlan plan = planConsolidation(mesh, bvh, claimed, ws);
  if (!copyConsolidation(mesh, bvh, claimed, plan, ws, result.delta)) {
    // An unclaimed leaf changed. Rebuild every leaf: the mesh stays valid, undo is lost.
    result.missedClaim = true;
    result.delta = {};
    claimed.assign(claimed.size(), 1);
    plan = planConsolidation(mesh, bvh, claimed, ws);
    // With every leaf rebuilt only a reference to a removed element could fail, which
    // validateLive() rules out.
    copyConsolidation(mesh, bvh, claimed, plan, ws, result.delta);
    result.delta = {};
  }
  std::swap(mesh, ws.mesh);

  for (const RegionSlot& slot : plan.slots) result.delta.regionLeaves.push_back(slot.leaf);
  parallelFor(0, plan.slots.size(), 2, [&](std::size_t b, std::size_t e) {
    for (std::size_t s = b; s < e; ++s) {
      BvhLeaf& leaf = plan.leaves[plan.slots[s].leaf];
      leaf.bounds = regionBounds(mesh, leaf);
    }
  });
  result.regionFaces = plan.regionFaces;
  bvh.setLeaves(std::move(plan.leaves));
  return result;
}

// --- Undo sides --------------------------------------------------------------------------------

std::size_t LayoutSide::bytes() const {
  std::size_t n = bvh ? bvh->memoryBytes() : 0;
  for (const TopoLeafState& s : region) n += s.bytes();
  for (const LeafState& s : posOnly)
    n += sliceBytes(s.positions) + sliceBytes(s.normals) + sliceBytes(s.mask) + sliceBytes(s.faceSets);
  return n;
}

LayoutSide beforeSide(std::shared_ptr<const Bvh> beforeBvh, std::vector<TopoLeafState> claims,
                      std::vector<LeafState> posOnly, const ConsolidateResult& result, std::uint64_t version) {
  LayoutSide side;
  side.topologyVersion = version;
  const auto leaves = beforeBvh->leaves();
  const Index L = static_cast<Index>(leaves.size());
  std::size_t c = 0;
  for (Index l : result.delta.regionLeaves) {
    if (l >= L) break;  // Appended by the consolidation.
    TopoLeafState s;
    if (c < claims.size() && claims[c].leaf == l) {
      s = std::move(claims[c++]);
    } else {
      s.leaf = l;  // An empty leaf the consolidation reused.
    }
    s.ranges = leaves[l];
    side.region.push_back(std::move(s));
  }
  side.bvh = std::move(beforeBvh);
  side.posOnly = std::move(posOnly);
  side.vertexCount = result.beforeVertexCount;
  side.faceCount = result.beforeFaceCount;
  side.halfEdgeCount = result.beforeHalfEdgeCount;
  return side;
}

LayoutSide captureSide(const Mesh& mesh, const Bvh& bvh, const LayoutDelta& delta,
                       std::span<const Index> posOnlyLeaves, std::uint64_t version) {
  LayoutSide side;
  side.topologyVersion = version;
  side.bvh = std::make_shared<const Bvh>(bvh);
  const auto leaves = bvh.leaves();
  for (Index l : delta.regionLeaves)
    if (l < static_cast<Index>(leaves.size())) side.region.push_back(captureLeaf(mesh, bvh, l));
  const bool normals = mesh.normals.size() == mesh.positions.size();
  for (Index l : posOnlyLeaves) {
    const BvhLeaf& r = leaves[l];
    LeafState s;
    s.leaf = l;
    copySlice(mesh.positions, r.vertBegin, r.vertEnd, s.positions);
    if (normals) copySlice(mesh.normals, r.vertBegin, r.vertEnd, s.normals);
    side.posOnly.push_back(std::move(s));
  }
  side.vertexCount = mesh.vertexCount();
  side.faceCount = mesh.faceCount();
  side.halfEdgeCount = mesh.halfEdgeCount();
  return side;
}

bool relayout(Mesh& mesh, Bvh& bvh, const LayoutSide& to, const LayoutDelta& delta, bool toAfter,
              LayoutWorkspace& ws) {
  if (!to.bvh || bvh.dynamic()) return false;
  const auto from = bvh.leaves();
  const auto dst = to.bvh->leaves();
  const Index Lf = static_cast<Index>(from.size()), Lt = static_cast<Index>(dst.size());
  const Index L = std::max(Lf, Lt);
  const Index nv = mesh.vertexCount(), nf = mesh.faceCount(), nh = mesh.halfEdgeCount();

  // The untouched leaves must exist on both sides with the same sizes.
  std::vector<char> region(static_cast<std::size_t>(L), 0);
  for (Index l : delta.regionLeaves) {
    if (l < 0 || l >= L) return false;
    region[l] = 1;
  }
  for (Index l = 0; l < L; ++l)
    if (!region[l] && (l >= Lf || l >= Lt || !sameSizes(from[l], dst[l]))) return false;
  if ((Lf ? from[Lf - 1].faceEnd : 0) != nf || (Lf ? from[Lf - 1].heEnd : 0) != nh) return false;
  if ((Lt ? dst[Lt - 1].faceEnd : 0) != to.faceCount || (Lt ? dst[Lt - 1].heEnd : 0) != to.halfEdgeCount)
    return false;
  const Index isoFrom = Lf ? from[Lf - 1].vertEnd : 0;
  const Index isoTo = Lt ? dst[Lt - 1].vertEnd : 0;
  if (nv - isoFrom != to.vertexCount - isoTo) return false;

  // Every region leaf of `to` has a slice that fills its ranges.
  const bool normals = mesh.normals.size() == mesh.positions.size();
  std::vector<const TopoLeafState*> slices(static_cast<std::size_t>(Lt), nullptr);
  for (const TopoLeafState& s : to.region) {
    if (s.leaf < 0 || s.leaf >= Lt || !region[s.leaf]) return false;
    slices[s.leaf] = &s;
  }
  bool mask = !mesh.mask.empty();
  bool faceSets = !mesh.faceSets.empty();
  for (Index l = 0; l < Lt; ++l) {
    if (!region[l]) continue;
    const TopoLeafState* s = slices[l];
    const BvhLeaf& r = dst[l];
    const auto verts = static_cast<std::size_t>(vertSpan(r)), hes = static_cast<std::size_t>(heSpan(r));
    if (!s || s->faceHe.size() != static_cast<std::size_t>(faceSpan(r)) || s->positions.size() != verts ||
        s->vertHe.size() != verts || s->heNext.size() != hes || s->heTwin.size() != hes || s->heVert.size() != hes ||
        s->heFace.size() != hes || (!s->mask.empty() && s->mask.size() != verts) ||
        (!s->faceSets.empty() && s->faceSets.size() != s->faceHe.size()) || (normals && s->normals.size() != verts))
      return false;
    mask |= !s->mask.empty();
    faceSets |= !s->faceSets.empty();
  }
  for (const LeafState& s : to.posOnly) {
    if (s.leaf < 0 || s.leaf >= Lt || region[s.leaf]) return false;
    const auto verts = static_cast<std::size_t>(vertSpan(dst[s.leaf]));
    if ((!s.positions.empty() && s.positions.size() != verts) || (!s.normals.empty() && s.normals.size() != verts) ||
        (!s.mask.empty() && s.mask.size() != verts) ||
        (!s.faceSets.empty() && s.faceSets.size() != static_cast<std::size_t>(faceSpan(dst[s.leaf]))))
      return false;
    mask |= !s.mask.empty();
    faceSets |= !s.faceSets.empty();
  }

  // Maps from the current numbering: untouched elements shift, region elements are only reached
  // through the delta's pairs.
  std::vector<Index>& vmap = ws.vmap;
  std::vector<Index>& fmap = ws.fmap;
  std::vector<Index>& hmap = ws.hmap;
  vmap.resize(nv);
  fmap.resize(nf);
  hmap.resize(nh);
  parallelFor(0, static_cast<std::size_t>(Lf), 4, [&](std::size_t b, std::size_t e) {
    for (std::size_t l = b; l < e; ++l) {
      const BvhLeaf& o = from[l];
      if (region[l]) {
        fillInvalid(fmap, o.faceBegin, o.faceEnd);
        fillInvalid(hmap, o.heBegin, o.heEnd);
        fillInvalid(vmap, o.vertBegin, o.vertEnd);
      } else {
        fillShift(fmap, o.faceBegin, o.faceEnd, dst[l].faceBegin);
        fillShift(hmap, o.heBegin, o.heEnd, dst[l].heBegin);
        fillShift(vmap, o.vertBegin, o.vertEnd, dst[l].vertBegin);
      }
    }
  });
  fillShift(vmap, isoFrom, nv, isoTo);
  auto inRegion = [&](Index leaf) { return leaf != kInvalid && region[leaf]; };
  for (const auto& [b, a] : delta.vertRefs) {
    const Index src = toAfter ? b : a, out = toAfter ? a : b;
    if (src < 0 || src >= nv || out < 0 || out >= to.vertexCount || !inRegion(bvh.leafOfVertex(src))) return false;
    vmap[src] = out;
  }
  for (const auto& [b, a] : delta.heRefs) {
    const Index src = toAfter ? b : a, out = toAfter ? a : b;
    if (src < 0 || src >= nh || out < 0 || out >= to.halfEdgeCount || !inRegion(bvh.leafOfHalfEdge(src))) return false;
    hmap[src] = out;
  }

  fitMesh(mesh, ws.mesh, to.vertexCount, to.faceCount, to.halfEdgeCount, normals, mask, faceSets);
  const CopyTarget target{vmap.data(),
                          fmap.data(),
                          hmap.data(),
                          normals ? mesh.normals.data() : nullptr,
                          mesh.mask.empty() ? nullptr : mesh.mask.data(),
                          mesh.faceSets.empty() ? nullptr : mesh.faceSets.data(),
                          &ws.mesh,
                          normals,
                          mask,
                          faceSets};
  std::vector<LeafCopy> copies(static_cast<std::size_t>(Lf));
  parallelFor(0, static_cast<std::size_t>(Lf), 2, [&](std::size_t b, std::size_t e) {
    for (std::size_t l = b; l < e; ++l)
      if (!region[l]) copyLeaf<false, false>(mesh, from[l], target, nullptr, copies[l]);
  });
  for (const LeafCopy& c : copies)
    if (c.bad) return false;
  copyIsolated(mesh, isoFrom, nv, isoTo, target);

  Mesh& w = ws.mesh;
  parallelFor(0, to.region.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const TopoLeafState& s = to.region[i];
      const BvhLeaf& r = dst[s.leaf];
      std::copy(s.positions.begin(), s.positions.end(), w.positions.begin() + r.vertBegin);
      if (normals) std::copy(s.normals.begin(), s.normals.end(), w.normals.begin() + r.vertBegin);
      if (mask) {
        if (s.mask.empty()) {
          std::fill(w.mask.begin() + r.vertBegin, w.mask.begin() + r.vertEnd, 0.0f);
        } else {
          std::copy(s.mask.begin(), s.mask.end(), w.mask.begin() + r.vertBegin);
        }
      }
      if (faceSets) {
        if (s.faceSets.empty()) {
          std::fill(w.faceSets.begin() + r.faceBegin, w.faceSets.begin() + r.faceEnd, kDefaultFaceSet);
        } else {
          std::copy(s.faceSets.begin(), s.faceSets.end(), w.faceSets.begin() + r.faceBegin);
        }
      }
      std::copy(s.vertHe.begin(), s.vertHe.end(), w.vertHe.begin() + r.vertBegin);
      std::copy(s.faceHe.begin(), s.faceHe.end(), w.faceHe.begin() + r.faceBegin);
      std::copy(s.heNext.begin(), s.heNext.end(), w.heNext.begin() + r.heBegin);
      std::copy(s.heTwin.begin(), s.heTwin.end(), w.heTwin.begin() + r.heBegin);
      std::copy(s.heVert.begin(), s.heVert.end(), w.heVert.begin() + r.heBegin);
      std::copy(s.heFace.begin(), s.heFace.end(), w.heFace.begin() + r.heBegin);
    }
  });
  for (const LeafState& s : to.posOnly) {
    const Index begin = dst[s.leaf].vertBegin;
    std::copy(s.positions.begin(), s.positions.end(), w.positions.begin() + begin);
    if (normals) std::copy(s.normals.begin(), s.normals.end(), w.normals.begin() + begin);
    if (mask) std::copy(s.mask.begin(), s.mask.end(), w.mask.begin() + begin);
    if (faceSets) std::copy(s.faceSets.begin(), s.faceSets.end(), w.faceSets.begin() + dst[s.leaf].faceBegin);
  }

  std::swap(mesh, ws.mesh);
  bvh = *to.bvh;
  bvh.endDynamic();
  return true;
}

// --- Validation --------------------------------------------------------------------------------

ValidationResult validateDynamic(const Mesh& m, const Bvh& bvh) {
  auto fail = [](std::string msg) { return ValidationResult{false, std::move(msg)}; };
  if (!bvh.dynamic()) return fail("BVH is not in dynamic mode");
  const ValidationResult live = validateLive(m);
  if (!live.ok) return live;
  const auto leaves = bvh.leaves();
  const Index T = bvh.firstTailLeaf();
  const Index L = static_cast<Index>(leaves.size());
  const Index tf = bvh.tailStartFace(), tv = bvh.tailStartVertex(), th = bvh.tailStartHalfEdge();
  auto contains = [](const Aabb& b, const Vec3& p) {
    return b.valid() && glm::all(glm::lessThanEqual(b.min, p)) && glm::all(glm::greaterThanEqual(b.max, p));
  };

  Index f = 0, h = 0, v = 0;
  for (Index l = 0; l < L; ++l) {
    const BvhLeaf& leaf = leaves[l];
    const std::string at = " at leaf " + std::to_string(l);
    if (l == T) {
      if (f != tf || h != th || v > tv) return fail("old leaves do not end at the tail start");
      f = tf;
      h = th;
      v = tv;
    }
    if (leaf.faceBegin != f || leaf.heBegin != h || leaf.vertBegin != v) return fail("ranges not contiguous" + at);
    if (leaf.faceEnd < leaf.faceBegin || leaf.heEnd < leaf.heBegin || leaf.vertEnd < leaf.vertBegin)
      return fail("negative range" + at);
    f = leaf.faceEnd;
    h = leaf.heEnd;
    v = leaf.vertEnd;
    const bool mayHaveDead = (leaf.flags & kLeafMayHaveDead) != 0;
    const bool ownsAll = (leaf.flags & (kLeafMayHaveDead | kLeafOwnsOrphans)) != 0;
    for (Index face = leaf.faceBegin; face < leaf.faceEnd; ++face) {
      if (m.faceHe[face] == kInvalid) {
        if (!mayHaveDead) return fail("removed face in an unflagged leaf" + at);
        continue;
      }
      bool inside = true;
      m.forEachFaceVertex(face, [&](Index u) { inside &= contains(leaf.bounds, m.positions[u]); });
      if (!inside) return fail("leaf bounds miss a face" + at);
    }
    for (Index e = leaf.heBegin; e < leaf.heEnd; ++e)
      if (m.heFace[e] == kInvalid && !mayHaveDead) return fail("removed half-edge in an unflagged leaf" + at);
    for (Index u = leaf.vertBegin; u < leaf.vertEnd; ++u) {
      if (m.vertHe[u] == kInvalid) {
        if (!mayHaveDead) return fail("removed vertex in an unflagged leaf" + at);
        if (m.positions[u] != kDeadPosition) return fail("removed vertex not at kDeadPosition" + at);
        continue;
      }
      if (ownsAll && !contains(leaf.bounds, m.positions[u])) return fail("leaf bounds miss an owned vertex" + at);
    }
  }
  if (L == T && (f != tf || h != th || v > tv)) return fail("old leaves do not end at the tail start");
  if (L == T && (m.faceCount() != tf || m.halfEdgeCount() != th || m.vertexCount() != tv))
    return fail("appended elements without a tail leaf");
  if (L > T && (f != m.faceCount() || h != m.halfEdgeCount() || v != m.vertexCount()))
    return fail("tail leaves do not cover the appended elements");
  const Index isoBegin = T > 0 ? leaves[T - 1].vertEnd : 0;
  for (Index u = isoBegin; u < tv; ++u)
    if (m.vertHe[u] != kInvalid) return fail("isolated block vertex with faces: " + std::to_string(u));

  // Tree nodes still contain their children.
  const auto nodes = bvh.nodes();
  auto nested = [](const Aabb& outer, const Aabb& inner) {
    return !inner.valid() || (glm::all(glm::lessThanEqual(outer.min, inner.min)) &&
                              glm::all(glm::greaterThanEqual(outer.max, inner.max)));
  };
  for (const BvhNode& node : nodes) {
    if (node.isLeaf()) {
      if (!nested(node.bounds, leaves[node.leaf].bounds)) return fail("leaf node bounds too small");
    } else if (!nested(node.bounds, nodes[node.left].bounds) || !nested(node.bounds, nodes[node.right].bounds)) {
      return fail("node bounds too small");
    }
  }
  return {};
}

}  // namespace plegl
