#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "multires/Multires.h"

namespace plegl {

// Propagation between subdivision levels (see multires/Multires.h for the model).
//
// A sync takes the pending edits of the active level a (diffActive) and computes what they mean
// for every other level (computeSync), as sparse value changes that applyDelta writes:
//
//  - Up (levels a+1 .. top): a fine vertex sits at S + F·d, where S is its subdivision of the
//    coarser level, F a frame built from S around it (detailFrame) and d its detail. When the
//    coarser level moves, S and F move and d stays, so sculpted detail rides along rigidly. Only
//    the region an edit can reach is evaluated, and a vertex whose S and frame did not change
//    bitwise keeps its exact value.
//  - Down (levels a-1 .. 0): the change of every fine vertex is restricted to the coarse vertices
//    with a tent filter, which keeps constants (a translation arrives exactly) and removes the
//    checkerboard pattern (pores do not alias into the coarse shape). Changes are filtered, not
//    values, so each level keeps its own sculpting.
//  - Sculpt layers on a parked level: the fold acts on the base and the offsets, and positions
//    are recomposed from them. Up, a translation moves the base; a rotation of the frame moves the
//    base like a position and turns every offset with the surface, so layer detail rides along.
//    Down, the change goes into the base. The active level's layers are never touched.
//  - Mask: up by assignment at vertex children and interpolated changes elsewhere, down by
//    injection (the coarse value is its vertex child's), so 1.0 stays exact everywhere.
//  - Face sets: the id and the hidden flag travel separately. Up, children take whichever of the
//    two changed on their parent; down, a parent takes the most common id of its children (ties
//    go to the earliest child in corner order) and is hidden only when all its children are.
//
// Every kernel gathers in a fixed order, so results do not depend on thread count or live order.

// The local frame detail is carried in at fine vertex i, computed from subdivided positions S:
// n from the area normals of the faces around i, t from the direction to the end of vertHe[i]
// projected off n (an axis when that is too short), b = n x t.
struct DetailFrame {
  Vec3 t{0.0f}, b{0.0f}, n{0.0f};
  bool degenerate = true;  // The faces around i have no area: no frame.
  bool operator==(const DetailFrame&) const = default;
};

template <class Pos>
Vec3 newellNormal(const Mesh& m, Index f, Pos&& P) {
  // Same operations as Mesh::faceAreaNormal, on other positions.
  Vec3 n{0.0f};
  const Index start = m.faceHe[f];
  Index h = start;
  do {
    const Vec3& a = P(m.heVert[h]);
    const Vec3& b = P(m.heVert[m.heNext[h]]);
    n.x += (a.y - b.y) * (a.z + b.z);
    n.y += (a.z - b.z) * (a.x + b.x);
    n.z += (a.x - b.x) * (a.y + b.y);
    h = m.heNext[h];
  } while (h != start);
  return n;
}

template <class Pos>
DetailFrame detailFrame(const Mesh& fine, Index i, Pos&& S) {
  DetailFrame fr;
  if (fine.vertHe[i] == kInvalid) return fr;
  Vec3 sum{0.0f};
  fine.forEachOutgoing(i, [&](Index h) { sum += newellNormal(fine, fine.heFace[h], S); });
  const float nn = glm::dot(sum, sum);
  if (!(nn > 1e-30f)) return fr;
  fr.n = sum / std::sqrt(nn);
  const Vec3 e = S(fine.heTarget(fine.vertHe[i])) - S(i);
  Vec3 t0 = e - fr.n * glm::dot(fr.n, e);
  const float ee = glm::dot(e, e);
  if (!(ee > 1e-30f) || !(glm::dot(t0, t0) > 1e-12f * ee)) {
    const Vec3 a = glm::abs(fr.n);
    const Vec3 axis = (a.x <= a.y && a.x <= a.z) ? Vec3{1, 0, 0} : (a.y <= a.z ? Vec3{0, 1, 0} : Vec3{0, 0, 1});
    t0 = axis - fr.n * glm::dot(fr.n, axis);
  }
  fr.t = t0 / std::sqrt(glm::dot(t0, t0));
  fr.b = glm::cross(fr.n, fr.t);
  fr.degenerate = false;
  return fr;
}

// Generation marks over an index range: begin() starts an empty set in O(1).
struct SyncStamps {
  std::vector<std::uint32_t> mark;
  std::uint32_t current = 0;
  void begin(std::size_t size) {
    if (mark.size() < size) mark.resize(size, 0);
    if (++current == 0) {
      std::fill(mark.begin(), mark.end(), 0u);
      current = 1;
    }
  }
  bool insert(Index i) {
    std::uint32_t& m = mark[static_cast<std::size_t>(i)];
    if (m == current) return false;
    m = current;
    return true;
  }
  bool has(Index i) const { return mark[static_cast<std::size_t>(i)] == current; }
};

// Scratch memory for syncs, kept between them (grows to about 70 bytes per vertex of all levels;
// replace it with a fresh one to give that back).
struct SyncWorkspace {
  template <class T>
  struct Overlay {
    std::vector<T> value;
    SyncStamps stamp;
    std::vector<Index> changed;
    void begin(std::size_t size) {
      if (value.size() < size) value.resize(size);
      stamp.begin(size);
      changed.clear();
    }
  };
  struct Level {
    Overlay<Vec3> pos;
    Overlay<float> mask;
    Overlay<std::int32_t> sets;
    // Levels with sculpt layers: the new base and offsets where they changed.
    Overlay<Vec3> base;
    std::vector<Overlay<Vec3>> offs;
  };
  std::vector<Level> levels;
  SyncStamps faces, verts, fine, fineAll;
  std::vector<Index> ring, region, support, targets;
  std::vector<Vec3> sOld, sNew, delta, scratch, scratchBase;
  std::vector<std::vector<Vec3>> scratchOffs;
  std::vector<std::uint8_t> flags;
};

// Calls fn(h, f) for every face f around vertex v, with h the half-edge of f leaving v. Unlike
// Mesh::forEachOutgoing this also reaches the other fans of a non-manifold vertex.
template <class Fn>
void forEachFaceAround(const Mesh& m, const MultiresLevel& level, Index v, Fn&& fn) {
  if (m.vertHe[v] == kInvalid) return;
  if (level.rule[v] == VertexRule::Pinned && !level.nonManifoldFaces.empty()) {
    const auto range = std::equal_range(level.nonManifoldFaces.begin(), level.nonManifoldFaces.end(),
                                        std::pair<Index, Index>{v, kInvalid},
                                        [](const auto& x, const auto& y) { return x.first < y.first; });
    if (range.first != range.second) {
      for (auto it = range.first; it != range.second; ++it) fn(it->second, m.heFace[it->second]);
      return;
    }
  }
  m.forEachOutgoing(v, [&](Index h) { fn(h, m.heFace[h]); });
}

// For every non-manifold vertex, all of its outgoing half-edges, sorted (MultiresLevel::nonManifoldFaces).
std::vector<std::pair<Index, Index>> findNonManifoldFans(const Mesh& mesh);

// Pending edits of the active level: every value where `live` differs bitwise from the reference.
SyncDelta diffActive(const Multires& stack, const Mesh& live);

// Carries `diff` (from diffActive) to every other level. Reads the stack and `live` and writes
// only the workspace and the result, so it can run on a worker while the stack stays untouched.
// limitRegion = false evaluates whole levels instead of the region the edits can reach (tests
// use it to check that the region is exact).
SyncDelta computeSync(const Multires& stack, const Mesh& live, SyncDelta diff, SyncWorkspace& ws,
                      bool limitRegion = true);

// Writes a sync's after (or before) values: entries of the source level into the reference, the
// others into the parked levels, whose normals are recomputed around the changed vertices and
// whose bounds are marked stale. Never touches the live mesh. The source level must be active.
void applyDelta(Multires& stack, const SyncDelta& delta, bool after, SyncWorkspace& ws);

// Rough number of vertex updates computeSync will make, for choosing a worker over the UI thread.
std::uint64_t syncWork(const Multires& stack, const SyncDelta& diff);

}  // namespace plegl
