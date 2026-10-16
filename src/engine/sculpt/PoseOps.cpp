#include "sculpt/PoseOps.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <tuple>
#include <unordered_map>

#include "core/Parallel.h"
#include "core/Timer.h"

namespace plegl {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// Union-find over small dense ids; the lower root always stays the root, so results do not
// depend on the order of unions.
struct DisjointSet {
  std::vector<int> parent;
  explicit DisjointSet(int n = 0) : parent(static_cast<std::size_t>(n)) {
    for (int i = 0; i < n; ++i) parent[static_cast<std::size_t>(i)] = i;
  }
  int add() {
    parent.push_back(static_cast<int>(parent.size()));
    return static_cast<int>(parent.size()) - 1;
  }
  int find(int a) {
    while (parent[static_cast<std::size_t>(a)] != a) {
      parent[static_cast<std::size_t>(a)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(a)])];
      a = parent[static_cast<std::size_t>(a)];
    }
    return a;
  }
  int unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b) return a;
    if (b < a) std::swap(a, b);
    parent[static_cast<std::size_t>(b)] = a;
    return a;
  }
};

// Calls fn(u) for every vertex sharing an edge with v, in every fan of a non-manifold vertex.
// A neighbour may come twice.
template <class Fn>
void forEachNeighbour(const Mesh& m, std::span<const std::pair<Index, Index>> fans, Index v, Fn&& fn) {
  const auto visit = [&](Index h) {
    fn(m.heTarget(h));
    const Index prev = m.hePrev(h);
    if (m.heTwin[prev] == kInvalid) fn(m.heVert[prev]);  // Incoming border edge.
  };
  if (!fans.empty()) {
    const auto range = std::equal_range(fans.begin(), fans.end(), std::pair<Index, Index>{v, kInvalid},
                                        [](const auto& x, const auto& y) { return x.first < y.first; });
    if (range.first != range.second) {
      for (auto it = range.first; it != range.second; ++it)
        if (m.heFace[it->second] != kInvalid) visit(it->second);
      return;
    }
  }
  m.forEachOutgoing(v, visit);
}

template <class Fn>
void forEachFaceAround(const Mesh& m, std::span<const std::pair<Index, Index>> fans, Index v, Fn&& fn) {
  if (!fans.empty()) {
    const auto range = std::equal_range(fans.begin(), fans.end(), std::pair<Index, Index>{v, kInvalid},
                                        [](const auto& x, const auto& y) { return x.first < y.first; });
    if (range.first != range.second) {
      for (auto it = range.first; it != range.second; ++it)
        if (m.heFace[it->second] != kInvalid) fn(m.heFace[it->second]);
      return;
    }
  }
  m.forEachOutgoing(v, [&](Index h) {
    if (m.heFace[h] != kInvalid) fn(m.heFace[h]);
  });
}

// Calls fn(u) for every other corner of every face around v: the straight line across a face is
// a much closer measure of distance along the surface than a path of edges (a quad's diagonal is
// 1.41 edges, not 2). A corner may come several times.
template <class Fn>
void forEachFaceCorner(const Mesh& m, std::span<const std::pair<Index, Index>> fans, Index v, Fn&& fn) {
  forEachFaceAround(m, fans, v, [&](Index f) {
    m.forEachFaceVertex(f, [&](Index u) {
      if (u != v) fn(u);
    });
  });
}

float smoothStep(float t) {
  t = std::clamp(t, 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

// x turned by phi about the unit axis k.
Vec3 rodrigues(const Vec3& x, const Vec3& k, float phi) {
  const float c = std::cos(phi);
  const float s = std::sin(phi);
  return x * c + glm::cross(k, x) * s + k * (glm::dot(k, x) * (1.0f - c));
}

Mat3 rodriguesMatrix(const Vec3& k, float phi) {
  return Mat3{rodrigues({1.0f, 0.0f, 0.0f}, k, phi), rodrigues({0.0f, 1.0f, 0.0f}, k, phi),
              rodrigues({0.0f, 0.0f, 1.0f}, k, phi)};
}

class PoseBuilder {
 public:
  PoseBuilder(const Mesh& m, const Bvh& bvh, std::span<const std::pair<Index, Index>> fans, const SymmetryGroup& group,
              std::span<const FaceSetFilter> filters, const Vec3& click, float radius, const PoseOptions& options,
              const PoseLimits& limits)
      : m_(m), bvh_(bvh), fans_(fans), group_(group), filters_(filters), click_(click), radius_(radius),
        options_(options), limits_(limits) {}

  PoseBuild run();

 private:
  struct Seed {
    float d;
    int op;
    Index v;
    bool operator<(const Seed& o) const { return std::tie(d, op, v) < std::tie(o.d, o.op, o.v); }
  };
  // An edge from a region vertex u to an outside vertex v, cut where the joint crosses it.
  struct Crossing {
    Index u, v;
    Vec3 q;
    float len, t;
  };
  struct Segment {
    int crossing;
    Vec3 a, b;
  };
  struct RigWork {
    CopyCluster cluster;
    std::vector<Index> region;
    std::vector<Crossing> crossings;
    std::vector<int> loopOf;     // Per crossing.
    int loops = 0;
    std::vector<Segment> segments;
    std::vector<char> rootLoop;  // Per loop.
    std::vector<Index> distal;   // Vertices beyond the other loops: they turn rigidly.
  };

  float maskAt(Index v) const { return m_.mask.empty() ? 0.0f : m_.mask[v]; }
  bool live(Index v) const { return m_.vertHe[v] != kInvalid; }
  // May v join op's region?
  bool allowed(Index v, int op) const {
    if (!live(v)) return false;
    if (!filters_.empty() && !filters_[static_cast<std::size_t>(op)].allows(m_, v)) return false;
    return options_.mode != PoseMode::Mask || maskAt(v) < 0.5f;
  }
  // May v move with a rig whose copies are `coset`?
  bool movable(Index v, OpMask coset) const {
    if (!live(v) || (options_.mode == PoseMode::Mask && maskAt(v) >= 0.5f)) return false;
    if (filters_.empty()) return true;
    for (int k = 0; k < group_.size(); ++k)
      if ((coset >> k & 1) && filters_[static_cast<std::size_t>(k)].allows(m_, v)) return true;
    return false;
  }
  bool overTime() {
    if (limits_.timeCapMs <= 0.0 || timer_.ms() <= limits_.timeCapMs) return false;
    out_.failure = PoseFailure::Aborted;
    return true;
  }
  bool seed(std::vector<Seed>& seeds);
  bool flood(std::vector<Seed>& seeds);
  Crossing crossing(Index u, Index v) const;
  void findLoops(RigWork& rig, int index);
  bool race(RigWork& rig, int index, std::uint64_t& touchedRigs);
  Vec3 pivotOf(const RigWork& rig, bool& loose);
  void weigh(const RigWork& rig, PoseRig& out);
  void finish();

  const Mesh& m_;
  const Bvh& bvh_;
  std::span<const std::pair<Index, Index>> fans_;
  const SymmetryGroup& group_;
  std::span<const FaceSetFilter> filters_;
  Vec3 click_;
  float radius_;
  PoseOptions options_;
  PoseLimits limits_;
  Timer timer_;
  PoseBuild out_;

  // Per vertex.
  std::vector<float> dist_;           // Topology mode: distance along the surface (tentative outside).
  std::vector<std::int8_t> label_;    // The op whose region reached the vertex first, -1 none.
  std::vector<std::uint8_t> inRegion_;
  std::vector<std::int16_t> rigOf_;   // Rig of a region vertex, -1 outside.
  std::vector<std::uint32_t> stamp_;  // Generic per-vertex stamp, with the values below.
  std::vector<std::int32_t> owner_;
  std::uint32_t stampValue_ = 0;
  std::vector<std::uint32_t> faceStamp_;
  std::uint32_t faceStampValue_ = 0;
  std::vector<std::uint8_t> claimed_;  // Moves with an earlier rig.
  std::vector<Index> region_;          // In the order the flood reached them.

  std::uint32_t nextStamp() {
    if (++stampValue_ == 0) {
      std::fill(stamp_.begin(), stamp_.end(), 0);
      stampValue_ = 1;
    }
    return stampValue_;
  }
  std::uint32_t nextFaceStamp() {
    if (++faceStampValue_ == 0) {
      std::fill(faceStamp_.begin(), faceStamp_.end(), 0);
      faceStampValue_ = 1;
    }
    return faceStampValue_;
  }
};

bool PoseBuilder::seed(std::vector<Seed>& seeds) {
  for (int k = 0; k < group_.size(); ++k) {
    const Vec3 c = k == 0 ? click_ : transform(group_.op(k).L, click_);
    Bvh::ClosestHit hit;
    if (!bvh_.closestPoint(m_, c, radius_, hit, true)) {
      out_.missingOps |= OpMask{1} << k;
      if (k == 0) {
        out_.failure = PoseFailure::NoSurface;
        return false;
      }
      continue;
    }
    if (k == 0) {
      Index nearest = kInvalid;
      float best = kInf;
      m_.forEachFaceVertex(hit.face, [&](Index v) {
        const float d = glm::length(m_.positions[v] - c);
        if (d < best) {
          best = d;
          nearest = v;
        }
      });
      const float limit = options_.mode == PoseMode::Mask ? 0.5f : 0.999f;
      if (nearest == kInvalid || maskAt(nearest) >= limit) {
        out_.failure = PoseFailure::Masked;
        return false;
      }
    }
    m_.forEachFaceVertex(hit.face, [&](Index v) {
      if (allowed(v, k)) seeds.push_back({glm::length(m_.positions[v] - c), k, v});
    });
  }
  if (std::none_of(seeds.begin(), seeds.end(), [](const Seed& s) { return s.op == 0; })) {
    out_.failure = PoseFailure::Masked;  // Every corner under the press is held back.
    return false;
  }
  std::sort(seeds.begin(), seeds.end());
  return true;
}

bool PoseBuilder::flood(std::vector<Seed>& seeds) {
  std::size_t work = 0;
  if (options_.mode == PoseMode::Topology) {
    // Multi-source Dijkstra over edge lengths, ties by (distance, op, vertex), up to one radius.
    using Entry = std::tuple<float, int, Index>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
    for (const Seed& s : seeds) {
      if (s.d < dist_[s.v] || (s.d == dist_[s.v] && s.op < label_[s.v])) {
        dist_[s.v] = s.d;
        label_[s.v] = static_cast<std::int8_t>(s.op);
        queue.emplace(s.d, s.op, s.v);
      }
    }
    while (!queue.empty()) {
      const auto [d, k, v] = queue.top();
      queue.pop();
      if (inRegion_[v] || d != dist_[v] || k != label_[v]) continue;  // Stale.
      if (d > radius_) break;
      inRegion_[v] = 1;
      region_.push_back(v);
      if ((++work & 255) == 0 && overTime()) return false;
      const Vec3 p = m_.positions[v];
      forEachFaceCorner(m_, fans_, v, [&](Index w) {
        if (inRegion_[w] || !allowed(w, k)) return;
        const float nd = d + glm::length(m_.positions[w] - p);
        if (nd < dist_[w] || (nd == dist_[w] && k < label_[w])) {
          dist_[w] = nd;
          label_[w] = static_cast<std::int8_t>(k);
          queue.emplace(nd, k, w);
        }
      });
    }
  } else {
    // Breadth first over the unmasked vertices: the whole unmasked part the press is on.
    for (const Seed& s : seeds) {
      if (inRegion_[s.v]) continue;
      inRegion_[s.v] = 1;
      label_[s.v] = static_cast<std::int8_t>(s.op);
      region_.push_back(s.v);
    }
    for (std::size_t i = 0; i < region_.size(); ++i) {
      const Index v = region_[i];
      const int k = label_[v];
      if ((++work & 255) == 0 && overTime()) return false;
      forEachNeighbour(m_, fans_, v, [&](Index w) {
        if (inRegion_[w] || !allowed(w, k)) return;
        inRegion_[w] = 1;
        label_[w] = static_cast<std::int8_t>(k);
        region_.push_back(w);
      });
    }
  }
  return true;
}

PoseBuilder::Crossing PoseBuilder::crossing(Index u, Index v) const {
  const Vec3 pu = m_.positions[u];
  const Vec3 pv = m_.positions[v];
  const float len = glm::length(pv - pu);
  float t = 0.5f;
  if (options_.mode == PoseMode::Topology) {
    // Where the distance along the surface passes the radius; an outside vertex the flood never
    // reached counts as one edge farther.
    const float du = dist_[u];
    float dv = dist_[v];
    if (!std::isfinite(dv) || !(dv > du)) dv = du + len;
    if (dv > du) t = std::clamp((radius_ - du) / (dv - du), 0.0f, 1.0f);
  } else {
    // Where the mask passes 0.5.
    const float mu = maskAt(u);
    const float mv = maskAt(v);
    if (mu < 0.5f && mv >= 0.5f && mv > mu) t = (0.5f - mu) / (mv - mu);
  }
  return {u, v, glm::mix(pu, pv, t), len, t};
}

void PoseBuilder::findLoops(RigWork& rig, int index) {
  const auto inRig = [&](Index v) { return rigOf_[v] == index; };
  const std::uint32_t fs = nextFaceStamp();
  std::unordered_map<Index, int> byEdge;  // The half-edge leaving the region -> crossing.
  DisjointSet loops;
  struct Cut {
    int crossing;
    bool exit;  // From the region outward along the face's boundary.
  };
  std::vector<Cut> cuts;
  for (Index v : rig.region) {
    forEachFaceAround(m_, fans_, v, [&](Index f) {
      if (faceStamp_[f] == fs) return;
      faceStamp_[f] = fs;
      cuts.clear();
      const Index h0 = m_.faceHe[f];
      Index h = h0;
      do {
        const Index a = m_.heVert[h];
        const Index b = m_.heTarget(h);
        const bool ia = inRig(a);
        if (ia != inRig(b)) {
          const Index key = ia || m_.heTwin[h] == kInvalid ? h : m_.heTwin[h];
          const auto [it, inserted] = byEdge.try_emplace(key, static_cast<int>(rig.crossings.size()));
          if (inserted) {
            rig.crossings.push_back(ia ? crossing(a, b) : crossing(b, a));
            loops.add();
          }
          cuts.push_back({it->second, ia});
        }
        h = m_.heNext[h];
      } while (h != h0);
      for (std::size_t i = 1; i < cuts.size(); ++i) loops.unite(cuts[0].crossing, cuts[i].crossing);
      // The region's corners of this face lie between a cut entering it and the next leaving it.
      for (std::size_t i = 0; i < cuts.size(); ++i) {
        if (cuts[i].exit) continue;
        for (std::size_t j = 1; j < cuts.size(); ++j) {
          const Cut& next = cuts[(i + j) % cuts.size()];
          if (!next.exit) continue;
          const Vec3 a = rig.crossings[static_cast<std::size_t>(cuts[i].crossing)].q;
          const Vec3 b = rig.crossings[static_cast<std::size_t>(next.crossing)].q;
          rig.segments.push_back({cuts[i].crossing, a, b});
          break;
        }
      }
    });
  }
  // Loops numbered in the order their first crossing was found.
  rig.loopOf.assign(rig.crossings.size(), -1);
  std::vector<int> numberOfRoot(rig.crossings.size(), -1);
  rig.loops = 0;
  for (std::size_t c = 0; c < rig.crossings.size(); ++c) {
    int& n = numberOfRoot[static_cast<std::size_t>(loops.find(static_cast<int>(c)))];
    if (n < 0) n = rig.loops++;
    rig.loopOf[c] = n;
  }
  rig.rootLoop.assign(static_cast<std::size_t>(rig.loops), 1);
}

bool PoseBuilder::race(RigWork& rig, int index, std::uint64_t& touchedRigs) {
  // Every loop sends a front into the rest of the mesh, one ring per round, in loop order. Fronts
  // that meet border the same part. Once a single front still grows, its loop borders the
  // largest part, the root; every other front has walked its whole part, which turns rigidly.
  const int n = rig.loops;
  if (n < 2) return true;
  const std::uint32_t s = nextStamp();
  DisjointSet classes(n);
  std::vector<std::vector<Index>> front(static_cast<std::size_t>(n));
  std::vector<std::size_t> count(static_cast<std::size_t>(n), 0);
  std::vector<std::uint64_t> touched(static_cast<std::size_t>(n), 0);
  std::vector<std::pair<int, int>> meets;
  std::vector<Index> visitedList;
  const std::size_t cap = limits_.raceVisitCap > 0 ? limits_.raceVisitCap
                                                   : std::max<std::size_t>(1'000'000, 2 * region_.size());
  bool capped = false;
  const auto visit = [&](Index w, int c) {
    stamp_[w] = s;
    owner_[w] = c;
    visitedList.push_back(w);
    ++count[static_cast<std::size_t>(c)];
    ++out_.visited;
    if (rigOf_[w] >= 0) touched[static_cast<std::size_t>(c)] |= std::uint64_t{1} << rigOf_[w];
    if (visitedList.size() >= cap) capped = true;
  };
  const auto merge = [&] {
    for (const auto& [a, b] : meets) {
      const int ra = classes.find(a);
      const int rb = classes.find(b);
      if (ra == rb) continue;
      const int root = classes.unite(ra, rb);
      const int other = root == ra ? rb : ra;
      const std::size_t r = static_cast<std::size_t>(root), o = static_cast<std::size_t>(other);
      count[r] += count[o];
      touched[r] |= touched[o];
      front[r].insert(front[r].end(), front[o].begin(), front[o].end());
      front[o].clear();
    }
    meets.clear();
  };
  for (std::size_t c = 0; c < rig.crossings.size(); ++c) {
    const Index v = rig.crossings[c].v;
    const int loop = rig.loopOf[c];
    if (!live(v)) continue;
    if (stamp_[v] != s) {
      if (capped) break;
      visit(v, loop);
      front[static_cast<std::size_t>(loop)].push_back(v);
    } else if (owner_[v] != loop) {
      meets.emplace_back(owner_[v], loop);
    }
  }
  merge();
  std::vector<Index> next;
  for (;;) {
    int growing = 0;
    for (int c = 0; c < n; ++c) growing += classes.find(c) == c && !front[static_cast<std::size_t>(c)].empty();
    if (growing <= 1 || capped) break;
    for (int c = 0; c < n && !capped; ++c) {
      if (classes.find(c) != c) continue;
      next.clear();
      for (Index v : front[static_cast<std::size_t>(c)]) {
        forEachNeighbour(m_, fans_, v, [&](Index w) {
          if (capped || rigOf_[w] == index || !live(w)) return;
          if (stamp_[w] == s) {
            if (classes.find(owner_[w]) != c) meets.emplace_back(owner_[w], c);
            return;
          }
          visit(w, c);
          next.push_back(w);
        });
        if (capped) break;
      }
      front[static_cast<std::size_t>(c)].swap(next);
      if (overTime()) return false;
    }
    merge();
  }
  // The one front still growing, else (all done at once, or capped) the one that visited most.
  int root = -1;
  int growing = 0;
  for (int c = 0; c < n; ++c) {
    if (classes.find(c) != c || front[static_cast<std::size_t>(c)].empty()) continue;
    ++growing;
    root = c;
  }
  if (growing != 1) {
    root = -1;
    for (int c = 0; c < n; ++c) {
      if (classes.find(c) != c) continue;
      if (root < 0 || count[static_cast<std::size_t>(c)] > count[static_cast<std::size_t>(root)]) root = c;
    }
  }
  for (int l = 0; l < n; ++l) rig.rootLoop[static_cast<std::size_t>(l)] = classes.find(l) == root;
  for (Index v : visitedList)
    if (classes.find(owner_[v]) != root) rig.distal.push_back(v);
  for (int c = 0; c < n; ++c)
    if (classes.find(c) == c && c != root) touchedRigs |= touched[static_cast<std::size_t>(c)];
  return true;
}

Vec3 PoseBuilder::pivotOf(const RigWork& rig, bool& loose) {
  loose = rig.crossings.empty();
  Vec3 pivot{0.0f};
  if (loose) {
    // A whole loose part: its area centroid.
    const std::uint32_t fs = nextFaceStamp();
    const int index = rigOf_[rig.region.front()];
    double area = 0.0;
    glm::dvec3 sum{0.0};
    for (Index v : rig.region) {
      forEachFaceAround(m_, fans_, v, [&](Index f) {
        if (faceStamp_[f] == fs) return;
        faceStamp_[f] = fs;
        bool inside = true;
        m_.forEachFaceVertex(f, [&](Index u) { inside &= rigOf_[u] == index; });
        if (!inside) return;
        const double a = glm::length(m_.faceAreaNormal(f));
        area += a;
        sum += glm::dvec3(m_.faceCentroid(f)) * a;
      });
    }
    if (area > 0.0) {
      pivot = Vec3(sum / area);
    } else {
      glm::dvec3 mean{0.0};
      for (Index v : rig.region) mean += glm::dvec3(m_.positions[v]);
      pivot = Vec3(mean / static_cast<double>(rig.region.size()));
    }
  } else {
    // The root loop's length-weighted centre: uneven vertex spacing does not pull it aside.
    double length = 0.0;
    glm::dvec3 sum{0.0};
    for (const Segment& seg : rig.segments) {
      if (!rig.rootLoop[static_cast<std::size_t>(rig.loopOf[static_cast<std::size_t>(seg.crossing)])]) continue;
      const double l = glm::length(seg.b - seg.a);
      length += l;
      sum += glm::dvec3(seg.a + seg.b) * (0.5 * l);
    }
    if (length > 0.0) {
      pivot = Vec3(sum / length);
    } else {
      glm::dvec3 mean{0.0};
      int count = 0;
      for (std::size_t c = 0; c < rig.crossings.size(); ++c) {
        if (!rig.rootLoop[static_cast<std::size_t>(rig.loopOf[c])]) continue;
        mean += glm::dvec3(rig.crossings[c].q);
        ++count;
      }
      pivot = Vec3(mean / static_cast<double>(std::max(count, 1)));
    }
  }
  // Merged copies turn about a joint on the plane or axis they share.
  if (rig.cluster.merged()) pivot = transform(rig.cluster.localProjector, pivot);
  return pivot;
}

void PoseBuilder::weigh(const RigWork& rig, PoseRig& out) {
  // The moving side: the region and the parts beyond its other loops.
  const std::uint32_t side = nextStamp();
  for (Index v : rig.region) stamp_[v] = side;
  for (Index v : rig.distal) stamp_[v] = side;
  const auto moving = [&](Index v) { return stamp_[v] == side; };
  // Distance to the root loop on both sides, up to half the band.
  const float band = std::clamp(options_.softness, 0.0f, 1.0f) * radius_;
  const float half = 0.5f * band;
  std::unordered_map<Index, float> distance;
  if (!out.loose && band > 0.0f) {
    using Entry = std::pair<float, Index>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
    // Vertices that may not move hold the band back: a joint along a locked border or the mask
    // does not bend the far side.
    const auto offer = [&](Index v, float d) {
      if (!(d < half) || claimed_[v] || !movable(v, rig.cluster.members)) return;
      auto [it, inserted] = distance.try_emplace(v, d);
      if (!inserted && !(d < it->second)) return;
      it->second = d;
      queue.emplace(d, v);
    };
    for (std::size_t c = 0; c < rig.crossings.size(); ++c) {
      if (!rig.rootLoop[static_cast<std::size_t>(rig.loopOf[c])]) continue;
      const Crossing& x = rig.crossings[c];
      offer(x.u, x.t * x.len);
      offer(x.v, (1.0f - x.t) * x.len);
    }
    while (!queue.empty()) {
      const auto [d, v] = queue.top();
      queue.pop();
      if (d != distance[v]) continue;
      const bool vSide = moving(v);
      const Vec3 p = m_.positions[v];
      forEachFaceCorner(m_, fans_, v, [&](Index w) {
        if (moving(w) != vSide) return;  // The band grows on each side separately.
        offer(w, d + glm::length(m_.positions[w] - p));
      });
    }
  }
  std::vector<std::pair<Index, float>> weights;
  const auto add = [&](Index v, float w) {
    if (claimed_[v] || !movable(v, rig.cluster.members)) return;
    w *= 1.0f - maskAt(v);
    if (w > 0.0f) weights.emplace_back(v, w);
  };
  const auto movingWeight = [&](Index v) {
    if (band <= 0.0f || out.loose) return 1.0f;
    const auto it = distance.find(v);
    return it == distance.end() ? 1.0f : smoothStep((it->second + half) / band);
  };
  for (Index v : rig.region) add(v, movingWeight(v));
  for (Index v : rig.distal) add(v, movingWeight(v));
  for (const auto& [v, d] : distance)
    if (!moving(v)) add(v, smoothStep((half - d) / band));
  std::sort(weights.begin(), weights.end());
  weights.erase(std::unique(weights.begin(), weights.end(),
                            [](const auto& a, const auto& b) { return a.first == b.first; }),
                weights.end());
  out.verts.reserve(weights.size());
  out.weight.reserve(weights.size());
  for (const auto& [v, w] : weights) {
    claimed_[v] = 1;
    out.verts.push_back(v);
    out.weight.push_back(std::min(w, 1.0f));
    out.start.push_back(m_.positions[v]);
    out.startNormal.push_back(m_.normals[v]);
  }
  // The root loop for the overlay, thinned to the point budget.
  std::vector<const Segment*> root;
  for (const Segment& seg : rig.segments)
    if (rig.rootLoop[static_cast<std::size_t>(rig.loopOf[static_cast<std::size_t>(seg.crossing)])]) root.push_back(&seg);
  const std::size_t step = std::max<std::size_t>(1, (2 * root.size() + kPoseLoopPoints - 1) / kPoseLoopPoints);
  for (std::size_t i = 0; i < root.size(); i += step) {
    out.rootLoop.push_back(root[i]->a);
    out.rootLoop.push_back(root[i]->b);
  }
}

void PoseBuilder::finish() {
  // Which normals change and how: a face whose corners all turn fully with one rig turns rigidly;
  // a vertex whose faces all do keeps its normal turned; any other vertex next to a moving one is
  // recomputed.
  const Index nv = m_.vertexCount();
  std::vector<float> w(static_cast<std::size_t>(nv), 0.0f);
  std::vector<std::int16_t> rigOfMove(static_cast<std::size_t>(nv), -1);
  for (std::size_t r = 0; r < out_.rigs.size(); ++r) {
    const PoseRig& rig = out_.rigs[r];
    for (std::size_t i = 0; i < rig.verts.size(); ++i) {
      w[rig.verts[i]] = rig.weight[i];
      rigOfMove[rig.verts[i]] = static_cast<std::int16_t>(r);
    }
  }
  constexpr std::uint8_t kStatic = 0, kRigid = 1, kBent = 2;
  const std::uint32_t fs = nextFaceStamp();
  std::vector<Index> faces;
  std::unordered_map<Index, std::uint8_t> faceClass;
  for (const PoseRig& rig : out_.rigs) {
    for (Index v : rig.verts) {
      forEachFaceAround(m_, fans_, v, [&](Index f) {
        if (faceStamp_[f] == fs) return;
        faceStamp_[f] = fs;
        faces.push_back(f);
        bool rigid = true;
        const std::int16_t first = rigOfMove[m_.heVert[m_.faceHe[f]]];
        m_.forEachFaceVertex(f, [&](Index u) { rigid &= w[u] == 1.0f && rigOfMove[u] == first; });
        faceClass[f] = rigid ? kRigid : kBent;
      });
    }
  }
  const std::uint32_t vs = nextStamp();
  std::vector<Index> bent;
  for (Index f : faces) {
    m_.forEachFaceVertex(f, [&](Index u) {
      if (stamp_[u] == vs) return;
      stamp_[u] = vs;
      bool allRigid = true, anyBent = false, anyRigid = false, anyStatic = false;
      forEachFaceAround(m_, fans_, u, [&](Index g) {
        const auto it = faceClass.find(g);
        const std::uint8_t c = it == faceClass.end() ? kStatic : it->second;
        allRigid &= c == kRigid;
        anyBent |= c == kBent;
        anyRigid |= c == kRigid;
        anyStatic |= c == kStatic;
      });
      if (allRigid && rigOfMove[u] >= 0) {
        PoseRig& rig = out_.rigs[static_cast<std::size_t>(rigOfMove[u])];
        const auto it = std::lower_bound(rig.verts.begin(), rig.verts.end(), u);
        rig.rigidSlots.push_back(static_cast<std::uint32_t>(it - rig.verts.begin()));
      } else if (anyBent || (anyRigid && anyStatic)) {
        bent.push_back(u);
      }
    });
  }
  std::sort(bent.begin(), bent.end());
  out_.bentNormals = std::move(bent);
  for (PoseRig& rig : out_.rigs) std::sort(rig.rigidSlots.begin(), rig.rigidSlots.end());
  for (Index f : faces) out_.refitLeaves.push_back(bvh_.leafOfFace(f));
  for (const PoseRig& rig : out_.rigs)
    for (Index v : rig.verts) out_.snapshotLeaves.push_back(bvh_.leafOfVertex(v));
  for (Index v : out_.bentNormals) out_.snapshotLeaves.push_back(bvh_.leafOfVertex(v));
  for (std::vector<Index>* list : {&out_.refitLeaves, &out_.snapshotLeaves}) {
    list->erase(std::remove(list->begin(), list->end(), kInvalid), list->end());
    std::sort(list->begin(), list->end());
    list->erase(std::unique(list->begin(), list->end()), list->end());
  }
}

PoseBuild PoseBuilder::run() {
  const std::size_t nv = static_cast<std::size_t>(m_.vertexCount());
  if (nv == 0 || m_.faceCount() == 0 || !(radius_ > 0.0f)) {
    out_.failure = PoseFailure::NoSurface;
    return std::move(out_);
  }
  dist_.assign(nv, kInf);
  label_.assign(nv, -1);
  inRegion_.assign(nv, 0);
  rigOf_.assign(nv, -1);
  stamp_.assign(nv, 0);
  owner_.assign(nv, 0);
  claimed_.assign(nv, 0);
  faceStamp_.assign(static_cast<std::size_t>(m_.faceCount()), 0);

  std::vector<Seed> seeds;
  if (!seed(seeds) || !flood(seeds)) {
    out_.buildMs = timer_.ms();
    return std::move(out_);
  }

  // Copies whose regions touch turn as one: the subgroup they generate with op 0.
  DisjointSet ops(group_.size());
  for (Index v : region_)
    forEachNeighbour(m_, fans_, v, [&](Index u) {
      if (inRegion_[u] && label_[u] != label_[v]) ops.unite(label_[u], label_[v]);
    });
  // Copies of the press that nearly coincide (on a plane or axis) are one part too, even where
  // op 0 reached every vertex first.
  CopyClusters pressed;
  clusterCopies(group_, click_, kCopyMergeFraction * radius_, pressed);
  OpMask together = pressed[0].subgroup;
  for (int k = 0; k < group_.size(); ++k)
    if (ops.find(k) == ops.find(0)) together |= OpMask{1} << k;
  OpMask subgroup = group_.closure(together);

  std::vector<RigWork> rigs;
  for (int attempt = 0; attempt <= group_.size(); ++attempt) {
    CopyClusters clusters;
    const int n = clustersOf(group_, subgroup, click_, clusters);
    std::array<std::int16_t, kMaxSymmetryCopies> rigOfOp{};
    rigs.assign(static_cast<std::size_t>(n), RigWork{});
    for (int i = 0; i < n; ++i) {
      rigs[static_cast<std::size_t>(i)].cluster = clusters[static_cast<std::size_t>(i)];
      for (int k = 0; k < group_.size(); ++k)
        if (clusters[static_cast<std::size_t>(i)].members >> k & 1) rigOfOp[static_cast<std::size_t>(k)] = static_cast<std::int16_t>(i);
    }
    for (Index v : region_) {
      rigOf_[v] = rigOfOp[static_cast<std::size_t>(label_[v])];
      rigs[static_cast<std::size_t>(rigOf_[v])].region.push_back(v);
    }
    // A part beyond one rig's joint that reaches another rig's region would turn twice: those
    // rigs merge, and the search runs again.
    OpMask grow = 0;
    for (int i = 0; i < n; ++i) {
      RigWork& rig = rigs[static_cast<std::size_t>(i)];
      if (rig.region.empty()) continue;
      findLoops(rig, i);
      std::uint64_t touched = 0;
      if (!race(rig, i, touched)) {
        out_.buildMs = timer_.ms();
        return std::move(out_);
      }
      for (int j = 0; j < n; ++j)
        if (j != i && (touched >> j & 1))
          grow |= OpMask{1} << group_.mul(group_.inverse(rig.cluster.op), rigs[static_cast<std::size_t>(j)].cluster.op);
    }
    if (!grow || attempt == group_.size()) break;
    subgroup = group_.closure(subgroup | grow);
  }

  for (std::size_t i = 0; i < rigs.size(); ++i) {
    const RigWork& work = rigs[i];
    if (work.region.empty()) continue;
    PoseRig rig;
    rig.cluster = work.cluster;
    rig.pivot = pivotOf(work, rig.loose);
    if (out_.rigs.empty() && glm::length(click_ - rig.pivot) < 0.1f * radius_) {
      out_.failure = PoseFailure::NoJoint;
      out_.buildMs = timer_.ms();
      return std::move(out_);
    }
    weigh(work, rig);
    if (!rig.verts.empty() || out_.rigs.empty()) out_.rigs.push_back(std::move(rig));
  }
  finish();
  out_.buildMs = timer_.ms();
  return std::move(out_);
}

}  // namespace

PoseBuild buildPose(const Mesh& mesh, const Bvh& bvh, std::span<const std::pair<Index, Index>> fans,
                    const SymmetryGroup& group, std::span<const FaceSetFilter> filterPerOp, const Vec3& click,
                    float radius, const PoseOptions& options, const PoseLimits& limits) {
  PoseBuilder builder(mesh, bvh, fans, group, filterPerOp, click, radius, options, limits);
  return builder.run();
}

Vec3 poseRotation(const Vec3& from, const Vec3& pivot, const Vec3& target, const Vec3& viewNormal) {
  const Vec3 a = from - pivot;
  const Vec3 b = target - pivot;
  const float la = glm::length(a);
  const float lb = glm::length(b);
  if (!(la > 1e-12f) || !(lb > 1e-12f)) return Vec3{0.0f};
  const Vec3 ua = a / la;
  const Vec3 ub = b / lb;
  const Vec3 c = glm::cross(ua, ub);
  const float s = glm::length(c);
  const float d = glm::dot(ua, ub);
  Vec3 axis;
  if (s < 1e-6f) {
    if (d >= 0.0f) return Vec3{0.0f};
    // A half turn: about the axis across the view.
    axis = glm::cross(ua, viewNormal);
    if (glm::length(axis) < 1e-6f) axis = glm::cross(ua, std::abs(ua.x) < 0.9f ? Vec3{1, 0, 0} : Vec3{0, 1, 0});
    axis = glm::normalize(axis);
  } else {
    axis = c / s;
  }
  const float angle = std::atan2(s, d);
  if (angle < 1e-6f) return Vec3{0.0f};
  return axis * angle;
}

void applyPoseRig(Mesh& mesh, const PoseRig& rig, const Vec3& rotation) {
  const float angle = glm::length(rotation);
  if (!(angle > 1e-12f)) {
    for (std::size_t i = 0; i < rig.verts.size(); ++i) mesh.positions[rig.verts[i]] = rig.start[i];
    for (std::uint32_t slot : rig.rigidSlots) mesh.normals[rig.verts[slot]] = rig.startNormal[slot];
    return;
  }
  const Vec3 axis = rotation / angle;
  const Mat3 R = rodriguesMatrix(axis, angle);
  const Vec3 o = rig.pivot;
  parallelFor(0, rig.verts.size(), 4096, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const float w = rig.weight[i];
      const Vec3 x = rig.start[i] - o;
      mesh.positions[rig.verts[i]] = o + (w == 1.0f ? R * x : rodrigues(x, axis, angle * w));
    }
  });
  for (std::uint32_t slot : rig.rigidSlots) {
    const Vec3 n = R * rig.startNormal[slot];
    const float len = glm::length(n);
    mesh.normals[rig.verts[slot]] = len > 0.0f ? n / len : rig.startNormal[slot];
  }
}

}  // namespace plegl
