#include "multires/Propagate.h"

#include <cstring>

#include "core/Parallel.h"
#include "multires/Subdivide.h"

namespace plegl {

namespace {

bool sameBits(const Vec3& a, const Vec3& b) { return std::memcmp(&a, &b, sizeof(Vec3)) == 0; }

// One level as a sync sees it: the source level reads old values from the reference and new ones
// from the live mesh; every other level reads old values from its parked mesh and new ones from
// the workspace overlay where this sync already changed them.
struct Access {
  const Mesh* mesh = nullptr;
  const MultiresLevel* meta = nullptr;
  const LevelReference* ref = nullptr;  // Source level only.
  SyncWorkspace::Level* ov = nullptr;   // Other levels only.
  bool source = false;

  const Vec3& posOld(Index v) const { return source ? ref->positions[v] : mesh->positions[v]; }
  const Vec3& posNew(Index v) const {
    if (source) return mesh->positions[v];
    return ov->pos.stamp.has(v) ? ov->pos.value[v] : mesh->positions[v];
  }
  float maskOld(Index v) const {
    const std::vector<float>& m = source ? ref->mask : mesh->mask;
    return m.empty() ? 0.0f : m[v];
  }
  float maskNew(Index v) const {
    if (!source && ov->mask.stamp.has(v)) return ov->mask.value[v];
    return mesh->mask.empty() ? 0.0f : mesh->mask[v];
  }
  std::int32_t setOld(Index f) const {
    const std::vector<std::int32_t>& s = source ? ref->faceSets : mesh->faceSets;
    return s.empty() ? kDefaultFaceSet : s[f];
  }
  std::int32_t setNew(Index f) const {
    if (!source && ov->sets.stamp.has(f)) return ov->sets.value[f];
    return mesh->faceSets.empty() ? kDefaultFaceSet : mesh->faceSets[f];
  }
  void writePos(Index v, const Vec3& p) {
    ov->pos.value[v] = p;
    if (ov->pos.stamp.insert(v)) ov->pos.changed.push_back(v);
  }
  void writeMask(Index v, float m) {
    ov->mask.value[v] = m;
    if (ov->mask.stamp.insert(v)) ov->mask.changed.push_back(v);
  }
  void writeSet(Index f, std::int32_t s) {
    ov->sets.value[f] = s;
    if (ov->sets.stamp.insert(f)) ov->sets.changed.push_back(f);
  }
};

template <class Fn>
void forEachAround(const Access& a, Index v, Fn&& fn) {
  forEachFaceAround(*a.mesh, *a.meta, v, fn);
}

// Subdivided position of fine vertex i from coarse positions X.
template <class Pos>
Vec3 subdividedPosition(const Mesh& coarse, const MultiresLevel& coarseMeta, std::uint32_t parent, Pos&& X) {
  const auto FP = [&](Index f) { return ccFacePoint(coarse, f, X); };
  const Index p = parentIndex(parent);
  switch (parentKind(parent)) {
    case kParentVertex: return ccVertexPoint(coarse, p, coarseMeta.rule[p], X, FP);
    case kParentEdge: return ccEdgePoint(coarse, p, X, FP);
    default: return FP(p);
  }
}

// Splits [0, n) into fixed chunks, runs fn(begin, end, out) per chunk in parallel and returns the
// chunks' outputs concatenated in order, so the result does not depend on scheduling.
template <class Fn>
std::vector<Index> collectParallel(std::size_t n, Fn&& fn) {
  constexpr std::size_t kChunk = 32768;
  const std::size_t chunks = (n + kChunk - 1) / kChunk;
  std::vector<std::vector<Index>> parts(chunks);
  parallelFor(0, chunks, 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t c = b; c < e; ++c) fn(c * kChunk, std::min(n, (c + 1) * kChunk), parts[c]);
  });
  std::vector<Index> out;
  for (auto& p : parts) out.insert(out.end(), p.begin(), p.end());
  return out;
}

// --- Up: coarse level C (k-1) to fine level F (k) ------------------------------------------

void upPositions(const Access& C, Access& F, const SubdivisionLinks& L, const std::vector<Index>& changed,
                 SyncWorkspace& ws, bool limit) {
  const Mesh& cm = *C.mesh;
  const Mesh& fm = *F.mesh;
  const Index fineCount = fm.vertexCount();
  std::vector<Index>& region = ws.region;    // Fine vertices whose value may change.
  std::vector<Index>& support = ws.support;  // Fine vertices whose S the region reads.
  region.clear();
  support.clear();
  ws.fine.begin(static_cast<std::size_t>(fineCount));
  ws.fineAll.begin(static_cast<std::size_t>(fineCount));
  if (limit) {
    // Rings of coarse faces: Φ0 touches a changed vertex, Φ1 and Φ2 add one vertex ring each. S
    // changes only on the children of Φ0; a frame also reads its fine one-ring, so frames change
    // only on the children of Φ1, and reading them needs S on the children of Φ2.
    std::vector<Index>& ring = ws.ring;
    ring.clear();
    ws.faces.begin(cm.faceHe.size());
    ws.verts.begin(cm.positions.size());
    for (Index c : changed) {
      ws.verts.insert(c);
      forEachAround(C, c, [&](Index, Index f) {
        if (ws.faces.insert(f)) ring.push_back(f);
      });
    }
    std::size_t ringEnd[3] = {ring.size(), 0, 0};
    std::size_t begin = 0;
    for (int r = 1; r <= 2; ++r) {
      const std::size_t end = ring.size();
      for (std::size_t i = begin; i < end; ++i) {
        cm.forEachFaceVertex(ring[i], [&](Index w) {
          if (!ws.verts.insert(w)) return;
          forEachAround(C, w, [&](Index, Index f) {
            if (ws.faces.insert(f)) ring.push_back(f);
          });
        });
      }
      ringEnd[r] = ring.size();
      begin = end;
    }
    auto add = [&](Index i, bool inRegion) {
      if (ws.fineAll.insert(i)) support.push_back(i);
      if (inRegion && ws.fine.insert(i)) region.push_back(i);
    };
    for (std::size_t k = 0; k < ringEnd[2]; ++k) {
      const Index f = ring[k];
      const bool inRegion = k < ringEnd[1];
      add(L.faceChild[f], inRegion);
      const Index start = cm.faceHe[f];
      Index h = start;
      do {
        add(L.vertexChild[cm.heVert[h]], inRegion);
        add(L.edgeChild[h], inRegion);
        h = cm.heNext[h];
      } while (h != start);
    }
    // Isolated vertices have no faces; their children follow them directly.
    for (Index c : changed)
      if (cm.vertHe[c] == kInvalid) add(L.vertexChild[c], true);
  } else {
    for (Index i = 0; i < fineCount; ++i) {
      ws.fine.insert(i);
      ws.fineAll.insert(i);
      region.push_back(i);
      support.push_back(i);
    }
  }
  std::sort(region.begin(), region.end());

  if (ws.sOld.size() < static_cast<std::size_t>(fineCount)) {
    ws.sOld.resize(static_cast<std::size_t>(fineCount));
    ws.sNew.resize(static_cast<std::size_t>(fineCount));
  }
  const auto Xold = [&](Index v) -> const Vec3& { return C.posOld(v); };
  const auto Xnew = [&](Index v) -> const Vec3& { return C.posNew(v); };
  parallelFor(0, support.size(), 2048, [&](std::size_t b, std::size_t e) {
    for (std::size_t k = b; k < e; ++k) {
      const Index i = support[k];
      ws.sOld[i] = subdividedPosition(cm, *C.meta, L.parent[i], Xold);
      ws.sNew[i] = subdividedPosition(cm, *C.meta, L.parent[i], Xnew);
    }
  });

  ws.scratch.resize(region.size());
  ws.flags.assign(region.size(), 0);
  const auto Sold = [&](Index j) -> const Vec3& { return ws.sOld[j]; };
  const auto Snew = [&](Index j) -> const Vec3& { return ws.sNew[j]; };
  parallelFor(0, region.size(), 1024, [&](std::size_t b, std::size_t e) {
    for (std::size_t k = b; k < e; ++k) {
      const Index i = region[k];
      const Vec3& so = ws.sOld[i];
      const Vec3& sn = ws.sNew[i];
      const Vec3& pOld = F.posOld(i);
      const DetailFrame fo = detailFrame(fm, i, Sold);
      const DetailFrame fn = detailFrame(fm, i, Snew);
      const bool sameS = sameBits(so, sn), sameFrame = fo == fn;
      if (sameS && sameFrame) continue;  // Nothing this vertex depends on changed: keep its bits.
      Vec3 p;
      if (sameFrame || fo.degenerate || fn.degenerate) {
        p = pOld + (sn - so);
      } else {
        const Vec3 d = pOld - so;
        p = sn + fn.t * glm::dot(fo.t, d) + fn.b * glm::dot(fo.b, d) + fn.n * glm::dot(fo.n, d);
      }
      if (sameBits(p, pOld)) continue;
      ws.scratch[k] = p;
      ws.flags[k] = 1;
    }
  });
  for (std::size_t k = 0; k < region.size(); ++k)
    if (ws.flags[k]) F.writePos(region[k], ws.scratch[k]);
}

void upMask(const Access& C, Access& F, const SubdivisionLinks& L, const std::vector<Index>& changed,
            SyncWorkspace& ws) {
  const Mesh& cm = *C.mesh;
  std::vector<Index>& region = ws.region;
  region.clear();
  ws.fine.begin(F.mesh->positions.size());
  auto add = [&](Index i) {
    if (ws.fine.insert(i)) region.push_back(i);
  };
  for (Index c : changed) {
    add(L.vertexChild[c]);
    forEachAround(C, c, [&](Index h, Index f) {
      add(L.faceChild[f]);
      add(L.edgeChild[h]);
      add(L.edgeChild[cm.hePrev(h)]);
    });
  }
  std::sort(region.begin(), region.end());
  const auto change = [&](Index v) { return C.maskNew(v) - C.maskOld(v); };
  for (Index i : region) {
    const std::uint32_t parent = L.parent[i];
    const Index p = parentIndex(parent);
    const float old = F.maskOld(i);
    float value;
    if (parentKind(parent) == kParentVertex) {
      value = C.maskNew(p);
    } else if (parentKind(parent) == kParentEdge) {
      value = std::clamp(old + (change(cm.heVert[p]) + change(cm.heTarget(p))) * 0.5f, 0.0f, 1.0f);
    } else {
      float sum = 0.0f;
      Index n = 0;
      cm.forEachFaceVertex(p, [&](Index v) {
        sum += change(v);
        ++n;
      });
      value = std::clamp(old + sum / static_cast<float>(n), 0.0f, 1.0f);
    }
    if (value != old) F.writeMask(i, value);
  }
}

void upSets(const Access& C, Access& F, const SubdivisionLinks& L, const std::vector<Index>& changed) {
  const Mesh& cm = *C.mesh;
  for (Index f : changed) {
    const std::int32_t o = C.setOld(f), n = C.setNew(f);
    const bool idChanged = faceSetId(o) != faceSetId(n);
    const bool hiddenChanged = faceSetHidden(o) != faceSetHidden(n);
    const Index start = cm.faceHe[f];
    Index h = start;
    do {
      const Index q = L.childFace[h];
      const std::int32_t child = F.setOld(q);
      const std::int32_t id = idChanged ? faceSetId(n) : faceSetId(child);
      const bool hidden = hiddenChanged ? faceSetHidden(n) : faceSetHidden(child);
      const std::int32_t value = hidden ? -id : id;
      if (value != child) F.writeSet(q, value);
      h = cm.heNext[h];
    } while (h != start);
  }
}

// --- Down: fine level F (k+1) to coarse level C (k) ----------------------------------------

void downPositions(const Access& F, Access& C, const SubdivisionLinks& L, const std::vector<Index>& changed,
                   SyncWorkspace& ws) {
  const Mesh& cm = *C.mesh;
  const std::size_t fineCount = F.mesh->positions.size();
  if (ws.delta.size() < fineCount) ws.delta.resize(fineCount);
  ws.fine.begin(fineCount);
  for (Index j : changed) {
    ws.delta[j] = F.posNew(j) - F.posOld(j);
    ws.fine.insert(j);
  }
  std::vector<Index>& targets = ws.targets;
  targets.clear();
  ws.verts.begin(cm.positions.size());
  auto mark = [&](Index v) {
    if (ws.verts.insert(v)) targets.push_back(v);
  };
  for (Index j : changed) {
    const std::uint32_t parent = L.parent[j];
    const Index p = parentIndex(parent);
    if (parentKind(parent) == kParentVertex) {
      mark(p);
    } else if (parentKind(parent) == kParentEdge) {
      mark(cm.heVert[p]);
      mark(cm.heTarget(p));
    } else {
      cm.forEachFaceVertex(p, mark);
    }
  }
  std::sort(targets.begin(), targets.end());
  const auto D = [&](Index i) { return ws.fine.has(i) ? ws.delta[i] : Vec3{0.0f}; };
  ws.scratch.resize(targets.size());
  ws.flags.assign(targets.size(), 0);
  parallelFor(0, targets.size(), 1024, [&](std::size_t b, std::size_t e) {
    for (std::size_t k = b; k < e; ++k) {
      const Index v = targets[k];
      const VertexRule rule = C.meta->rule[v];
      Vec3 d;
      if (rule == VertexRule::Pinned) {
        d = D(L.vertexChild[v]);
      } else if (rule == VertexRule::Boundary) {
        Index ha = kInvalid, hb = kInvalid;  // Outgoing and incoming open half-edges.
        cm.forEachOutgoing(v, [&](Index h) {
          if (cm.heTwin[h] == kInvalid) ha = h;
          const Index prev = cm.hePrev(h);
          if (cm.heTwin[prev] == kInvalid) hb = prev;
        });
        d = (D(L.vertexChild[v]) * 2.0f + (D(L.edgeChild[ha]) + D(L.edgeChild[hb]))) * 0.25f;
      } else {
        Vec3 sumE{0.0f}, sumF{0.0f};
        int n = 0;
        cm.forEachOutgoing(v, [&](Index h) {
          sumE += D(L.edgeChild[h]);
          sumF += D(L.faceChild[cm.heFace[h]]);
          ++n;
        });
        d = (D(L.vertexChild[v]) * 4.0f + sumE * 2.0f + sumF) / static_cast<float>(4 + 3 * n);
      }
      if (d.x == 0.0f && d.y == 0.0f && d.z == 0.0f) continue;
      const Vec3& old = C.posOld(v);
      const Vec3 p = old + d;
      if (sameBits(p, old)) continue;
      ws.scratch[k] = p;
      ws.flags[k] = 1;
    }
  });
  for (std::size_t k = 0; k < targets.size(); ++k)
    if (ws.flags[k]) C.writePos(targets[k], ws.scratch[k]);
}

void downMask(const Access& F, Access& C, const SubdivisionLinks& L, const std::vector<Index>& changed) {
  for (Index j : changed) {
    const std::uint32_t parent = L.parent[j];
    if (parentKind(parent) != kParentVertex) continue;
    const Index v = parentIndex(parent);
    const float value = F.maskNew(j);
    if (value != C.maskOld(v)) C.writeMask(v, value);
  }
}

void downSets(const Access& F, Access& C, const SubdivisionLinks& L, const std::vector<Index>& changed,
              SyncWorkspace& ws) {
  const Mesh& cm = *C.mesh;
  std::vector<Index>& targets = ws.targets;
  targets.clear();
  ws.faces.begin(cm.faceHe.size());
  for (Index q : changed) {
    const Index f = cm.heFace[L.parentHalfEdge[q]];
    if (ws.faces.insert(f)) targets.push_back(f);
  }
  std::sort(targets.begin(), targets.end());
  std::vector<std::int32_t> values;
  for (Index f : targets) {
    values.clear();
    bool allHidden = true;
    const Index start = cm.faceHe[f];
    Index h = start;
    do {
      const std::int32_t s = F.setNew(L.childFace[h]);
      values.push_back(s);
      allHidden &= faceSetHidden(s);
      h = cm.heNext[h];
    } while (h != start);
    // Most common id; ties go to the earliest child in corner order.
    std::int32_t best = faceSetId(values[0]);
    int bestCount = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
      const std::int32_t id = faceSetId(values[i]);
      int count = 0;
      for (std::int32_t s : values) count += faceSetId(s) == id ? 1 : 0;
      if (count > bestCount) {
        best = id;
        bestCount = count;
      }
    }
    const std::int32_t value = allHidden ? -best : best;
    if (value != C.setOld(f)) C.writeSet(f, value);
  }
}

LevelDelta collectDelta(int level, const Access& a) {
  LevelDelta d;
  d.level = level;
  const Mesh& m = *a.mesh;
  auto sorted = [](std::vector<Index> v) {
    std::sort(v.begin(), v.end());
    return v;
  };
  d.posIndex = sorted(a.ov->pos.changed);
  for (Index v : d.posIndex) {
    d.posBefore.push_back(m.positions[v]);
    d.posAfter.push_back(a.ov->pos.value[v]);
  }
  d.maskIndex = sorted(a.ov->mask.changed);
  for (Index v : d.maskIndex) {
    d.maskBefore.push_back(a.maskOld(v));
    d.maskAfter.push_back(a.ov->mask.value[v]);
  }
  d.setIndex = sorted(a.ov->sets.changed);
  for (Index f : d.setIndex) {
    d.setBefore.push_back(a.setOld(f));
    d.setAfter.push_back(a.ov->sets.value[f]);
  }
  d.maskCreated = m.mask.empty() && !d.maskIndex.empty();
  d.setsCreated = m.faceSets.empty() && !d.setIndex.empty();
  return d;
}

}  // namespace

std::vector<std::pair<Index, Index>> findNonManifoldFans(const Mesh& m) {
  std::vector<Index> outCount(m.positions.size(), 0);
  for (Index h = 0; h < m.halfEdgeCount(); ++h) ++outCount[m.heVert[h]];
  std::vector<std::uint8_t> nonManifold(m.positions.size(), 0);
  bool any = false;
  for (Index v = 0; v < m.vertexCount(); ++v) {
    if (m.vertHe[v] == kInvalid) continue;
    Index visited = 0;
    m.forEachOutgoing(v, [&](Index) { ++visited; });
    if (visited != outCount[v]) nonManifold[v] = 1, any = true;
  }
  std::vector<std::pair<Index, Index>> out;
  if (!any) return out;
  for (Index h = 0; h < m.halfEdgeCount(); ++h)
    if (nonManifold[m.heVert[h]]) out.emplace_back(m.heVert[h], h);
  std::sort(out.begin(), out.end());
  return out;
}

SyncDelta diffActive(const Multires& stack, const Mesh& live) {
  SyncDelta out;
  out.sourceLevel = stack.active;
  LevelDelta d;
  d.level = stack.active;
  const LevelReference& ref = stack.reference;
  d.posIndex = collectParallel(live.positions.size(), [&](std::size_t b, std::size_t e, std::vector<Index>& o) {
    for (std::size_t v = b; v < e; ++v)
      if (!sameBits(live.positions[v], ref.positions[v])) o.push_back(static_cast<Index>(v));
  });
  for (Index v : d.posIndex) {
    d.posBefore.push_back(ref.positions[v]);
    d.posAfter.push_back(live.positions[v]);
  }
  if (!live.mask.empty() || !ref.mask.empty()) {
    const auto at = [](const std::vector<float>& m, std::size_t v) { return m.empty() ? 0.0f : m[v]; };
    d.maskIndex = collectParallel(live.positions.size(), [&](std::size_t b, std::size_t e, std::vector<Index>& o) {
      for (std::size_t v = b; v < e; ++v)
        if (at(live.mask, v) != at(ref.mask, v)) o.push_back(static_cast<Index>(v));
    });
    for (Index v : d.maskIndex) {
      d.maskBefore.push_back(at(ref.mask, static_cast<std::size_t>(v)));
      d.maskAfter.push_back(at(live.mask, static_cast<std::size_t>(v)));
    }
    d.maskCreated = ref.mask.empty() && !d.maskIndex.empty();
  }
  if (!live.faceSets.empty() || !ref.faceSets.empty()) {
    const auto at = [](const std::vector<std::int32_t>& s, std::size_t f) { return s.empty() ? kDefaultFaceSet : s[f]; };
    d.setIndex = collectParallel(live.faceHe.size(), [&](std::size_t b, std::size_t e, std::vector<Index>& o) {
      for (std::size_t f = b; f < e; ++f)
        if (at(live.faceSets, f) != at(ref.faceSets, f)) o.push_back(static_cast<Index>(f));
    });
    for (Index f : d.setIndex) {
      d.setBefore.push_back(at(ref.faceSets, static_cast<std::size_t>(f)));
      d.setAfter.push_back(at(live.faceSets, static_cast<std::size_t>(f)));
    }
    d.setsCreated = ref.faceSets.empty() && !d.setIndex.empty();
  }
  if (!d.empty()) out.levels.push_back(std::move(d));
  return out;
}

SyncDelta computeSync(const Multires& stack, const Mesh& live, SyncDelta diff, SyncWorkspace& ws, bool limitRegion) {
  if (diff.empty()) return diff;
  const int a = stack.active;
  const int count = stack.levelCount();
  ws.levels.resize(static_cast<std::size_t>(count));
  std::vector<Access> acc(static_cast<std::size_t>(count));
  for (int k = 0; k < count; ++k) {
    Access& x = acc[k];
    x.meta = &stack.levels[k];
    if (k == a) {
      x.mesh = &live;
      x.ref = &stack.reference;
      x.source = true;
    } else {
      x.mesh = &stack.levels[k].mesh;
      x.ov = &ws.levels[k];
      x.ov->pos.begin(x.mesh->positions.size());
      x.ov->mask.begin(x.mesh->positions.size());
      x.ov->sets.begin(x.mesh->faceHe.size());
    }
  }
  const LevelDelta& src = diff.levels[0];
  auto sortedChanged = [](const std::vector<Index>& v) {
    std::vector<Index> s = v;
    std::sort(s.begin(), s.end());
    return s;
  };

  std::vector<Index> pos = src.posIndex, mask = src.maskIndex, sets = src.setIndex;
  for (int k = a + 1; k < count; ++k) {
    const SubdivisionLinks& L = stack.levels[k].links;
    if (!pos.empty()) upPositions(acc[k - 1], acc[k], L, pos, ws, limitRegion);
    if (!mask.empty()) upMask(acc[k - 1], acc[k], L, mask, ws);
    if (!sets.empty()) upSets(acc[k - 1], acc[k], L, sets);
    pos = sortedChanged(acc[k].ov->pos.changed);
    mask = sortedChanged(acc[k].ov->mask.changed);
    sets = sortedChanged(acc[k].ov->sets.changed);
    if (pos.empty() && mask.empty() && sets.empty()) break;
  }
  pos = src.posIndex, mask = src.maskIndex, sets = src.setIndex;
  for (int k = a - 1; k >= 0; --k) {
    const SubdivisionLinks& L = stack.levels[k + 1].links;
    if (!pos.empty()) downPositions(acc[k + 1], acc[k], L, pos, ws);
    if (!mask.empty()) downMask(acc[k + 1], acc[k], L, mask);
    if (!sets.empty()) downSets(acc[k + 1], acc[k], L, sets, ws);
    pos = sortedChanged(acc[k].ov->pos.changed);
    mask = sortedChanged(acc[k].ov->mask.changed);
    sets = sortedChanged(acc[k].ov->sets.changed);
    if (pos.empty() && mask.empty() && sets.empty()) break;
  }
  for (int k = 0; k < count; ++k) {
    if (k == a) continue;
    LevelDelta d = collectDelta(k, acc[k]);
    if (!d.empty()) diff.levels.push_back(std::move(d));
  }
  return diff;
}

void applyDelta(Multires& stack, const SyncDelta& delta, bool after, SyncWorkspace& ws) {
  for (const LevelDelta& d : delta.levels) {
    const auto& pos = after ? d.posAfter : d.posBefore;
    const auto& mask = after ? d.maskAfter : d.maskBefore;
    const auto& sets = after ? d.setAfter : d.setBefore;
    if (d.level == delta.sourceLevel) {
      LevelReference& ref = stack.reference;
      for (std::size_t i = 0; i < d.posIndex.size(); ++i) ref.positions[d.posIndex[i]] = pos[i];
      if (!d.maskIndex.empty() && ref.mask.empty()) ref.mask.assign(ref.positions.size(), 0.0f);
      for (std::size_t i = 0; i < d.maskIndex.size(); ++i) ref.mask[d.maskIndex[i]] = mask[i];
      if (!after && d.maskCreated) ref.mask.clear();
      // The canonical map has one entry per face of the level, also while it is active.
      if (!d.setIndex.empty() && ref.faceSets.empty())
        ref.faceSets.assign(stack.levels[static_cast<std::size_t>(d.level)].canon.face.size(), kDefaultFaceSet);
      for (std::size_t i = 0; i < d.setIndex.size(); ++i) ref.faceSets[d.setIndex[i]] = sets[i];
      if (!after && d.setsCreated) ref.faceSets.clear();
      continue;
    }
    MultiresLevel& level = stack.levels[static_cast<std::size_t>(d.level)];
    Mesh& m = level.mesh;
    if (!d.posIndex.empty()) {
      for (std::size_t i = 0; i < d.posIndex.size(); ++i) m.positions[d.posIndex[i]] = pos[i];
      // Normals change on every corner of every face around a moved vertex.
      ws.verts.begin(m.positions.size());
      std::vector<Index>& verts = ws.targets;
      verts.clear();
      for (Index v : d.posIndex) {
        if (ws.verts.insert(v)) verts.push_back(v);
        forEachFaceAround(m, level, v, [&](Index, Index f) {
          m.forEachFaceVertex(f, [&](Index w) {
            if (ws.verts.insert(w)) verts.push_back(w);
          });
        });
      }
      if (m.normals.size() != m.positions.size()) m.normals.resize(m.positions.size());
      parallelFor(0, verts.size(), 2048, [&](std::size_t b, std::size_t e) {
        for (std::size_t k = b; k < e; ++k) m.normals[verts[k]] = m.vertexNormal(verts[k]);
      });
      level.boundsStale = true;
    }
    if (!d.maskIndex.empty()) {
      m.ensureMask();
      for (std::size_t i = 0; i < d.maskIndex.size(); ++i) m.mask[d.maskIndex[i]] = mask[i];
      if (!after && d.maskCreated) m.mask.clear();
    }
    if (!d.setIndex.empty()) {
      m.ensureFaceSets();
      for (std::size_t i = 0; i < d.setIndex.size(); ++i) m.faceSets[d.setIndex[i]] = sets[i];
      if (!after && d.setsCreated) m.faceSets.clear();
    }
  }
}

std::uint64_t syncWork(const Multires& stack, const SyncDelta& diff) {
  if (diff.levels.empty()) return 0;
  const std::uint64_t changed = diff.levels[0].posIndex.size() + diff.levels[0].maskIndex.size();
  std::uint64_t factor = 0, f = 1;
  for (int k = stack.active + 1; k < stack.levelCount(); ++k) {
    f *= 4;
    factor += f;
  }
  // Going down touches at most the changed vertices again on every lower level.
  return changed * (factor + static_cast<std::uint64_t>(stack.active) + 1);
}

}  // namespace plegl
