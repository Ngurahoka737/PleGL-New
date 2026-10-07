#include "sculpt/Dyntopo.h"

#include <algorithm>
#include <cmath>

#include "core/Parallel.h"

namespace plegl {
namespace {

// A fan longer than this means the mesh is not in a state the session trusts.
constexpr int kMaxFan = 64;
// Largest valence a collapse may leave behind.
constexpr int kMaxValence = 12;

float dist2(const Vec3& a, const Vec3& b) {
  const Vec3 d = a - b;
  return glm::dot(d, d);
}

// Squared distance from p to the segment a-b.
float segmentDist2(const Vec3& p, const Vec3& a, const Vec3& b) {
  const Vec3 ab = b - a;
  const float len2 = glm::dot(ab, ab);
  const float t = len2 > 0.0f ? std::clamp(glm::dot(p - a, ab) / len2, 0.0f, 1.0f) : 0.0f;
  return dist2(a + ab * t, p);
}

// Calls fn(h) for every outgoing half-edge of v, like Mesh::forEachOutgoing, but returns false
// after kMaxFan edges instead of trusting the mesh to close the fan.
template <class Fn>
bool walkFan(const Mesh& m, Index v, Fn&& fn) {
  const Index start = m.vertHe[v];
  if (start == kInvalid) return true;
  int n = 0;
  Index h = start;
  for (;;) {
    if (++n > kMaxFan) return false;
    fn(h);
    const Index t = m.heTwin[m.hePrev(h)];
    if (t == kInvalid) break;
    h = t;
    if (h == start) return true;
  }
  h = start;
  for (;;) {
    const Index t = m.heTwin[h];
    if (t == kInvalid) return true;
    h = m.heNext[t];
    if (++n > kMaxFan) return false;
    fn(h);
  }
}

// Area normal of face f (sum of corner cross products) with vertices a and b moved to p.
Vec3 movedNormal(const Mesh& m, Index f, Index a, Index b, const Vec3& p) {
  Vec3 n{0.0f};
  const Index start = m.faceHe[f];
  Index h = start;
  do {
    const Index u = m.heVert[h], w = m.heVert[m.heNext[h]];
    const Vec3& pu = (u == a || u == b) ? p : m.positions[u];
    const Vec3& pw = (w == a || w == b) ? p : m.positions[w];
    n += glm::cross(pu, pw);
    h = m.heNext[h];
  } while (h != start);
  return n;
}

bool faceHasBoth(const Mesh& m, Index f, Index a, Index b) {
  bool hasA = false, hasB = false;
  m.forEachFaceVertex(f, [&](Index v) {
    hasA |= v == a;
    hasB |= v == b;
  });
  return hasA && hasB;
}

void sortUnique(std::vector<Index>& v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
}

}  // namespace

float DyntopoSession::effectiveDetail(float detail, float radius, float meshDiagonal) {
  return std::max({detail, radius / 20.0f, 1e-6f * meshDiagonal});
}

DyntopoSession::DyntopoSession(SceneObject& object, const DyntopoOptions& options)
    : obj_(object), m_(object.mesh), bvh_(object.bvh), opt_(options), ed_(object.mesh) {
  beforeBvh_ = std::make_shared<const Bvh>(bvh_);
  beforeVersion_ = object.topologyVersion;
  const float diag = glm::length(bvh_.bounds().extent());
  diagonal_ = std::isfinite(diag) && diag > 0.0f ? diag : 1.0f;

  // Room for the stroke's new elements, so no edit reallocates inside a dab. Reserving only when
  // the spare capacity runs low keeps a growing mesh from reallocating on every stroke.
  const Index nv = m_.vertexCount(), nf = m_.faceCount(), nh = m_.halfEdgeCount();
  const Index spare = std::max<Index>(65536, nv / 4);
  if (!m_.hasHeadroom(spare / 2, 2 * spare, 5 * spare)) m_.reserveHeadroom(nv + spare, nf + 4 * spare, nh + 10 * spare);

  // Non-manifold vertices (fans that do not cover every outgoing edge) are locked. A mesh found
  // clean is remembered by version, so the check runs once per topology.
  if (object.manifoldCheckedVersion != object.topologyVersion) {
    std::vector<Index> fan(static_cast<std::size_t>(nv), 0);
    parallelFor(0, fan.size(), 8192, [&](std::size_t b, std::size_t e) {
      for (std::size_t v = b; v < e; ++v) {
        Index n = 0;
        if (!walkFan(m_, static_cast<Index>(v), [&](Index) { ++n; })) n = -1;
        fan[v] = n;
      }
    });
    long long total = 0;
    bool walksOk = true;
    for (Index n : fan) {
      total += n;
      walksOk &= n >= 0;
    }
    if (walksOk && total == nh) {
      object.manifoldCheckedVersion = object.topologyVersion;
    } else {
      std::vector<Index> out(static_cast<std::size_t>(nv), 0);
      for (Index h = 0; h < nh; ++h) ++out[m_.heVert[h]];
      lock_.assign(static_cast<std::size_t>(nv), 0);
      for (Index v = 0; v < nv; ++v) {
        if (fan[v] != out[v]) {
          lock_[v] = 1;
          ++lockedCount_;
        }
      }
    }
  }

  bvh_.beginDynamic(m_);
  rec_.begin(m_, bvh_);
  observer_.session = this;
  ed_.setObserver(&observer_);
}

bool DyntopoSession::changedTopology() const {
  return !rec_.empty() || m_.faceCount() != bvh_.tailStartFace() || m_.vertexCount() != bvh_.tailStartVertex() ||
         m_.halfEdgeCount() != bvh_.tailStartHalfEdge();
}

void DyntopoSession::Observer::beforeWrite(ElementKind kind, Index index) {
  session->rec_.beforeWrite(kind, index);
  if (kind == ElementKind::Face) {
    session->markFace(index);
  } else if (kind == ElementKind::HalfEdge) {
    const Index f = session->m_.heFace[index];
    if (f != kInvalid) session->markFace(f);
  }
}

void DyntopoSession::markFace(Index f) {
  if (faceLeaf_ != kInvalid) {
    const BvhLeaf& cached = bvh_.leaves()[faceLeaf_];
    if (f >= cached.faceBegin && f < cached.faceEnd) return;
  }
  // Faces appended by the edit in progress have no leaf yet; growing the tail marks them.
  const Index l = bvh_.leafOfFace(f);
  if (l == kInvalid) return;
  faceLeaf_ = l;
  pass_.topoDirtyLeaves.push_back(l);
}

bool DyntopoSession::frozen(Index f, int& size) const {
  size = 0;
  const Index start = m_.faceHe[f];
  Index h = start;
  do {
    if (++size > 4 || locked(m_.heVert[h])) return true;
    h = m_.heNext[h];
  } while (h != start);
  return size < 3;
}

bool DyntopoSession::pickDiagonal(Index f, Index& ha, Index& hb) const {
  Index e[4];
  e[0] = m_.faceHe[f];
  for (int i = 1; i < 4; ++i) e[i] = m_.heNext[e[i - 1]];
  Index v[4];
  Vec3 p[4];
  for (int i = 0; i < 4; ++i) {
    v[i] = m_.heVert[e[i]];
    p[i] = m_.positions[v[i]];
  }
  const Vec3 n = m_.faceAreaNormal(f);
  // Diagonal from corner i to i + 2: both triangles must face the quad's way, and the edge must
  // not exist already.
  auto valid = [&](int i) {
    const Vec3& p0 = p[i];
    const Vec3& p1 = p[(i + 1) % 4];
    const Vec3& p2 = p[(i + 2) % 4];
    const Vec3& p3 = p[(i + 3) % 4];
    return glm::dot(glm::cross(p1 - p0, p2 - p0), n) > 0.0f && glm::dot(glm::cross(p2 - p0, p3 - p0), n) > 0.0f &&
           !ed_.connected(v[i], v[(i + 2) % 4]);
  };
  const bool ok0 = valid(0), ok1 = valid(1);
  if (!ok0 && !ok1) return false;
  const bool use0 = ok0 && (!ok1 || dist2(p[0], p[2]) <= dist2(p[1], p[3]));
  ha = use0 ? e[0] : e[1];
  hb = use0 ? e[2] : e[3];
  return true;
}

const DyntopoPass& DyntopoSession::pass(const Vec3& center, float radius, const DabTopology& topo) {
  pass_.splits = pass_.collapses = pass_.edits = 0;
  pass_.changedVerts.clear();
  pass_.refitLeaves.clear();
  pass_.topoDirtyLeaves.clear();
  if (faulted_ || topo.detail <= 0.0f || radius <= 0.0f) return pass_;
  passTimer_.reset();
  opsSinceCheck_ = 0;
  faceLeaf_ = kInvalid;
  leavesBefore_ = bvh_.leaves().size();
  openTailBefore_ = leavesBefore_ > static_cast<std::size_t>(bvh_.firstTailLeaf()) ? bvh_.leaves().back() : BvhLeaf{};

  const float d = effectiveDetail(topo.detail, radius, diagonal_);
  const float lmax2 = d * d;
  const float lmin2 = 0.16f * lmax2;  // Collapse below 0.4 D: split halves (>= 0.5 D) never qualify.
  leaves_.clear();
  bvh_.querySphere(center, radius, leaves_);
  gather(center, radius, lmax2, lmin2, topo.hintFace);
  if (opt_.refine != DyntopoRefine::CollapseOnly) splitAll(center, radius, lmax2);
  if (opt_.refine != DyntopoRefine::SplitOnly) {
    std::sort(collapses_.begin(), collapses_.end(), [](const Edge& x, const Edge& y) { return x.len2 < y.len2; });
    for (const Edge& e : collapses_) {
      if (faulted_ || pass_.collapses >= opt_.maxCollapses || overBudget()) break;
      collapseOne(e, lmax2, lmin2);
    }
  }
  finishPass();
  return pass_;
}

bool DyntopoSession::overBudget() {
  if (opt_.timeBudgetMs <= 0.0 || ++opsSinceCheck_ < 32) return false;
  opsSinceCheck_ = 0;
  return passTimer_.ms() > opt_.timeBudgetMs;
}

void DyntopoSession::gather(const Vec3& c, float r, float lmax2, float lmin2, Index hintFace) {
  const float r2 = r * r;
  perLeafSplits_.resize(leaves_.size());
  perLeafCollapses_.resize(leaves_.size());
  // Each edge once (an interior one from its lower half-edge). An edge that meets the sphere lies
  // inside the bounds of both its faces' leaves, so both are queried.
  parallelFor(0, leaves_.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      std::vector<Edge>& sp = perLeafSplits_[i];
      std::vector<Edge>& co = perLeafCollapses_[i];
      sp.clear();
      co.clear();
      const BvhLeaf leaf = bvh_.leaves()[leaves_[i]];
      for (Index f = leaf.faceBegin; f < leaf.faceEnd; ++f) {
        const Index h0 = m_.faceHe[f];
        if (h0 == kInvalid) continue;
        Index h = h0;
        int guard = 0;
        do {
          const Index t = m_.heTwin[h];
          const Index next = m_.heNext[h];
          if (t == kInvalid || h < t) {
            const Index va = m_.heVert[h], vb = m_.heVert[next];
            const Vec3& pa = m_.positions[va];
            const Vec3& pb = m_.positions[vb];
            const float len2 = dist2(pa, pb);
            if (len2 > lmax2) {
              if (segmentDist2(c, pa, pb) <= r2 && !locked(va) && !locked(vb)) sp.push_back({len2, h, va, vb});
            } else if (len2 < lmin2 && t != kInvalid) {  // Border vertices never merge.
              if (dist2((pa + pb) * 0.5f, c) <= r2 && !locked(va) && !locked(vb)) co.push_back({len2, h, va, vb});
            }
          }
          h = next;
        } while (h != h0 && ++guard < kMaxFan);
      }
    }
  });
  splits_.clear();
  collapses_.clear();
  for (std::size_t i = 0; i < leaves_.size(); ++i) {
    splits_.insert(splits_.end(), perLeafSplits_[i].begin(), perLeafSplits_[i].end());
    collapses_.insert(collapses_.end(), perLeafCollapses_[i].begin(), perLeafCollapses_[i].end());
  }
  // A face larger than the brush: split its long edges even if none of them reaches the sphere.
  if (hintFace >= 0 && hintFace < m_.faceCount() && m_.faceHe[hintFace] != kInvalid) {
    const Index h0 = m_.faceHe[hintFace];
    Index h = h0;
    int guard = 0;
    do {
      const Index va = m_.heVert[h], vb = m_.heTarget(h);
      const float len2 = dist2(m_.positions[va], m_.positions[vb]);
      if (len2 > lmax2 && !locked(va) && !locked(vb)) splits_.push_back({len2, h, va, vb});
      h = m_.heNext[h];
    } while (h != h0 && ++guard < kMaxFan);
  }
}

void DyntopoSession::splitAll(const Vec3& c, float r, float lmax2) {
  if (outOfRoom_) return;
  const float r2 = r * r;
  auto shorter = [](const Edge& x, const Edge& y) { return x.len2 < y.len2; };
  auto push = [&](const Edge& e) {
    splits_.push_back(e);
    std::push_heap(splits_.begin(), splits_.end(), shorter);
  };
  std::make_heap(splits_.begin(), splits_.end(), shorter);
  // Every split pops an edge, and so does every deferral to a longer edge; a quad that refuses
  // both diagonals could otherwise defer the same edge forever.
  std::size_t pops = 0;
  const std::size_t maxPops = splits_.size() + 16 * static_cast<std::size_t>(std::max(opt_.maxSplits, 1));
  while (!splits_.empty() && pass_.splits < opt_.maxSplits && !faulted_ && ++pops <= maxPops) {
    if (overBudget()) break;
    std::pop_heap(splits_.begin(), splits_.end(), shorter);
    const Edge e = splits_.back();
    splits_.pop_back();
    // Earlier edits may have changed or removed this edge.
    const Index h = e.h;
    if (m_.heFace[h] == kInvalid || m_.heVert[h] != e.a || m_.heTarget(h) != e.b) continue;
    const Index t = m_.heTwin[h];  // kInvalid on an open border: there is only one face to cut.
    if ((t != kInvalid && m_.heFace[t] == m_.heFace[h]) || dist2(m_.positions[e.a], m_.positions[e.b]) <= lmax2)
      continue;
    int sizeF = 0, sizeG = 0;
    if (frozen(m_.heFace[h], sizeF) || (t != kInvalid && frozen(m_.heFace[t], sizeG))) continue;
    if (!m_.hasHeadroom(1, 4, 10)) {
      outOfRoom_ = true;
      break;
    }
    // Quads on either side are cut into triangles first; check both before writing anything.
    Index fa = kInvalid, fb = kInvalid, ga = kInvalid, gb = kInvalid;
    if (sizeF == 4 && !pickDiagonal(m_.heFace[h], fa, fb)) continue;
    if (sizeG == 4 && !pickDiagonal(m_.heFace[t], ga, gb)) continue;
    bool cut = false;
    for (const auto& [x, y] : {std::pair{fa, fb}, std::pair{ga, gb}}) {
      if (x == kInvalid) continue;
      const Index f = m_.heFace[x];
      m_.forEachFaceVertex(f, [&](Index v) { pass_.changedVerts.push_back(v); });
      if (ed_.splitFace(x, y) == kInvalid) continue;
      ++pass_.edits;
      cut = true;
      // The new diagonal is an edge like any other: split it too if it is long and in reach.
      const Index d = m_.halfEdgeCount() - 2;
      const Index u = m_.heVert[d], w = m_.heTarget(d);
      const float len2 = dist2(m_.positions[u], m_.positions[w]);
      if (len2 > lmax2 && segmentDist2(c, m_.positions[u], m_.positions[w]) <= r2) push({len2, d, u, w});
    }
    if (cut) bvh_.growTail(m_);

    // Longest-edge bisection: a triangle is only cut across its longest edge. Cutting a shorter
    // one next to a long edge the brush does not reach makes ever thinner slivers that never get
    // shorter. So a longer edge of either triangle goes first, even outside the brush, and this
    // edge comes back after it. A triangle whose longest edge may not be cut (it borders a frozen
    // face) is left whole, and this edge with it.
    bool deferred = false, blocked = false;
    for (Index s : {h, t}) {
      if (s == kInvalid) continue;
      Index longest = kInvalid;
      float longest2 = e.len2 * 1.0001f;
      for (Index o : {m_.heNext[s], m_.hePrev(s)}) {
        const float len2 = dist2(m_.positions[m_.heVert[o]], m_.positions[m_.heTarget(o)]);
        if (len2 > longest2) {
          longest = o;
          longest2 = len2;
        }
      }
      if (longest == kInvalid) continue;
      const Index lt = m_.heTwin[longest];
      int size = 0;
      if (lt != kInvalid && (m_.heFace[lt] == m_.heFace[longest] || frozen(m_.heFace[lt], size))) {
        blocked = true;
        continue;
      }
      push({longest2, longest, m_.heVert[longest], m_.heTarget(longest)});
      deferred = true;
    }
    if (blocked) continue;
    if (deferred) {
      push(e);
      continue;
    }

    const Index v = ed_.splitEdge(h, 0.5f);
    ++pass_.edits;
    // h: a -> v and t: b -> v. Fan both faces from v to their opposite corner.
    for (Index s : {h, t}) {
      if (s == kInvalid) continue;
      const Index from = m_.heNext[s];  // Starts at v.
      const Index to = m_.hePrev(s);    // Starts at the corner before s.
      if (m_.heNext[from] != to && ed_.splitFace(from, to) != kInvalid) ++pass_.edits;
    }
    bvh_.growTail(m_);
    ++pass_.splits;
    ++totalSplits_;

    // Normals change around v, and the edges at v that are still long get split in turn.
    const Vec3 pv = m_.positions[v];
    pass_.changedVerts.push_back(v);
    const bool ok = walkFan(m_, v, [&](Index o) {
      const Index w = m_.heTarget(o);
      pass_.changedVerts.push_back(w);
      const float len2 = dist2(pv, m_.positions[w]);
      if (len2 > lmax2 && segmentDist2(c, pv, m_.positions[w]) <= r2 && !locked(w)) {
        splits_.push_back({len2, o, v, w});
        std::push_heap(splits_.begin(), splits_.end(), shorter);
      }
    });
    if (!ok) faulted_ = true;
  }
}

bool DyntopoSession::collapseOne(const Edge& e, float lmax2, float lmin2) {
  const Index h = e.h;
  if (m_.heFace[h] == kInvalid || m_.heVert[h] != e.a || m_.heTarget(h) != e.b) return false;
  const Index t = m_.heTwin[h];
  const Index a = e.a, b = e.b;
  const Vec3 pa = m_.positions[a], pb = m_.positions[b];
  if (t == kInvalid || dist2(pa, pb) >= lmin2) return false;

  // Every face around both ends must be editable.
  ring_.clear();
  bool ok = true;
  int valA = 0, valB = 0;
  ok &= walkFan(m_, a, [&](Index o) {
    ring_.push_back(m_.heFace[o]);
    ++valA;
  });
  ok &= walkFan(m_, b, [&](Index o) {
    ring_.push_back(m_.heFace[o]);
    ++valB;
  });
  if (!ok) {
    faulted_ = true;
    return false;
  }
  sortUnique(ring_);
  int apexes = 0;
  for (Index f : ring_) {
    int size = 0;
    if (frozen(f, size)) return false;
    if (size == 3 && faceHasBoth(m_, f, a, b)) ++apexes;
  }
  if (valA + valB - 2 - apexes > kMaxValence) return false;

  // The more masked end keeps its place; otherwise both meet in the middle.
  const float ma = m_.mask.empty() ? 0.0f : m_.mask[a];
  const float mb = m_.mask.empty() ? 0.0f : m_.mask[b];
  Index keep = a, gone = b;
  Vec3 p = (pa + pb) * 0.5f;
  if (ma > mb) {
    p = pa;
  } else if (mb > ma) {
    keep = b;
    gone = a;
    p = pb;
  }

  // No edge at the merged vertex may come out long enough to be split again.
  for (Index v : {a, b}) {
    bool shortEnough = true;
    walkFan(m_, v, [&](Index o) {
      const Index w = m_.heTarget(o);
      if (w != a && w != b && dist2(p, m_.positions[w]) > lmax2) shortEnough = false;
    });
    if (!shortEnough) return false;
  }
  // No face around the edge may flip or collapse to nothing. Triangles on the edge disappear.
  const Vec3 fallback = m_.normals.empty() ? Vec3{0.0f} : m_.normals[a] + m_.normals[b];
  for (Index f : ring_) {
    const bool both = faceHasBoth(m_, f, a, b);
    int size = 0;
    frozen(f, size);
    if (both && size == 3) continue;
    const Vec3 n0 = movedNormal(m_, f, kInvalid, kInvalid, p);
    const Vec3 n1 = movedNormal(m_, f, a, b, p);
    const float l0 = glm::length(n0), l1 = glm::length(n1);
    if (l1 <= 1e-4f * lmax2) return false;
    if (l0 > 1e-12f) {
      if (glm::dot(n0, n1) <= 0.2f * l0 * l1) return false;
    } else if (!(glm::dot(n1, fallback) > 0.0f)) {
      return false;
    }
  }

  if (!ed_.collapseEdge(keep == a ? h : t, p)) return false;
  m_.positions[gone] = kDeadPosition;  // Its leaf was claimed by the collapse.
  ++pass_.edits;
  ++pass_.collapses;
  ++totalCollapses_;
  touchRing(keep, true);
  return true;
}

void DyntopoSession::touchRing(Index v, bool refit) {
  pass_.changedVerts.push_back(v);
  if (!walkFan(m_, v, [&](Index o) {
        pass_.changedVerts.push_back(m_.heTarget(o));
        if (refit) pass_.refitLeaves.push_back(bvh_.leafOfFace(m_.heFace[o]));
      }))
    faulted_ = true;
}

void DyntopoSession::finishPass() {
  // Tail leaves that grew or appeared during the pass.
  const auto leaves = bvh_.leaves();
  const std::size_t firstTail = static_cast<std::size_t>(bvh_.firstTailLeaf());
  const std::size_t from = std::max(firstTail, leavesBefore_ > 0 ? leavesBefore_ - 1 : 0);
  for (std::size_t l = from; l < leaves.size(); ++l) {
    const BvhLeaf& leaf = leaves[l];
    const bool grew = l >= leavesBefore_ || leaf.faceEnd != openTailBefore_.faceEnd ||
                      leaf.vertEnd != openTailBefore_.vertEnd || leaf.heEnd != openTailBefore_.heEnd;
    if (!grew) continue;
    pass_.topoDirtyLeaves.push_back(static_cast<Index>(l));
  }
  sortUnique(pass_.topoDirtyLeaves);
  // Bounds change where faces changed and around vertices that moved.
  pass_.refitLeaves.insert(pass_.refitLeaves.end(), pass_.topoDirtyLeaves.begin(), pass_.topoDirtyLeaves.end());
  pass_.refitLeaves.erase(std::remove(pass_.refitLeaves.begin(), pass_.refitLeaves.end(), kInvalid),
                          pass_.refitLeaves.end());
  sortUnique(pass_.refitLeaves);
  pass_.changedVerts.erase(std::remove_if(pass_.changedVerts.begin(), pass_.changedVerts.end(),
                                          [&](Index v) { return m_.vertHe[v] == kInvalid; }),
                           pass_.changedVerts.end());
  sortUnique(pass_.changedVerts);
}

}  // namespace plegl
