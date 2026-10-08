#include "remesh/QuadRemesh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "core/Geometry.h"
#include "core/Parallel.h"
#include "core/Timer.h"
#include "mesh/MeshEdit.h"
#include "spatial/Bvh.h"

namespace plegl {

namespace {

constexpr float kSliverDiagonal = 0.35f;  // Collapse quads with a diagonal shorter than this * h.
constexpr float kDiamondDiagonal = 1.5f;  // Valence-driven collapses only on quads up to this size.
// Corners may reach 180 degrees, since an edge rotation on a regular grid always makes one
// straight corner that relaxation then rounds out. Only clearly reflex corners are refused.
constexpr float kMinCornerSin = -0.1f;
constexpr float kRelaxRate = 0.6f;
constexpr float kProjectDistance = 1.0f;  // Search radius for projection, in target edges.
constexpr int kGatherRings = 3;           // How far drift passes look for other irregular vertices.

int irregularity(int valence) { return (valence - 4) * (valence - 4); }

// True if no corner of the quad p0..p3 folds over when seen along n.
bool quadOk(const std::array<Vec3, 4>& p, const Vec3& n) {
  for (int i = 0; i < 4; ++i) {
    const Vec3 u = p[(i + 1) & 3] - p[i], w = p[(i + 3) & 3] - p[i];
    const float lu = glm::length(u), lw = glm::length(w);
    if (lu <= 0.0f || lw <= 0.0f) return false;
    if (glm::dot(glm::cross(u, w), n) < kMinCornerSin * lu * lw) return false;
  }
  return true;
}

std::uint32_t hashMix(std::uint32_t i, std::uint32_t seed) {
  std::uint32_t r = (i + 1u) * 2654435761u ^ seed * 0x9E3779B9u;
  r ^= r >> 15;
  r *= 0x2C1B3C6Du;
  r ^= r >> 12;
  return r;
}

// Topology passes over a quad mesh, scored by valence irregularity sum((valence - 4)^2).
//
// Improve passes only take moves that lower the score. They get stuck with scattered 3-5 pairs
// (the quad mesh version of a lattice dislocation): moving a pair one step costs nothing, and
// only two pairs that meet can cancel. Drift passes therefore also take moves that keep the
// score, preferring ones that bring irregular vertices closer to other irregular vertices, so
// pairs wander towards each other and the next improve pass cancels them.
class Optimizer {
 public:
  enum class Mode { Improve, Drift };

  Optimizer(Mesh& mesh, float h) : ed_(mesh), m_(mesh), h_(h), val_(mesh.vertexCount()) {
    parallelFor(0, val_.size(), 4096, [&](std::size_t b, std::size_t e) {
      for (std::size_t v = b; v < e; ++v) val_[v] = m_.valence(static_cast<Index>(v));
    });
    stamp_.assign(val_.size(), 0);
  }

  // Collapses slivers, and quads whose diagonal collapse lowers (or in drift mode keeps, while
  // gathering irregular vertices) the irregularity of their corners.
  int collapsePass(Mode mode, std::uint32_t seed = 0) {
    int done = 0;
    const Index nf = m_.faceCount();
    for (Index f = 0; f < nf; ++f) {
      if (!ed_.faceAlive(f)) continue;
      std::array<Index, 4> e{};
      e[0] = m_.faceHe[f];
      for (int i = 1; i < 4; ++i) e[i] = m_.heNext[e[i - 1]];
      if (m_.heNext[e[3]] != e[0]) continue;
      std::array<Index, 4> v{};
      std::array<int, 4> val{};
      for (int i = 0; i < 4; ++i) {
        v[i] = m_.heVert[e[i]];
        val[i] = val_[v[i]];
      }
      const float diag[2] = {glm::length(m_.positions[v[0]] - m_.positions[v[2]]),
                             glm::length(m_.positions[v[1]] - m_.positions[v[3]])};
      // Score before and after collapsing the diagonal starting at corner c.
      auto score = [&](int c, int& before, int& after) {
        const int a = val[c], b = val[c + 1], o = val[c + 2], d = val[(c + 3) & 3];
        before = irregularity(a) + irregularity(o) + irregularity(b) + irregularity(d);
        after = irregularity(a + o - 2) + irregularity(b - 1) + irregularity(d - 1);
      };
      int k = -1;
      if (std::min(diag[0], diag[1]) < kSliverDiagonal * h_) {
        k = diag[0] < diag[1] ? 0 : 1;
      } else {
        int bestGain = 0;
        for (int c = 0; c < 2; ++c) {
          if (diag[c] > kDiamondDiagonal * h_) continue;
          int before = 0, after = 0;
          score(c, before, after);
          if (before - after > bestGain) {
            bestGain = before - after;
            k = c;
          }
        }
        if (k < 0 && mode == Mode::Drift) {
          const std::uint32_t r = hashMix(f, seed);
          const int c = int(r & 1u);
          int before = 0, after = 0;
          score(c, before, after);
          if (before > 0 && before == after && diag[c] <= kDiamondDiagonal * h_) {
            const Index skip[6] = {v[0], v[1], v[2], v[3], v[0], v[0]};
            int sb = 0, sa = 0;
            for (int i = 0; i < 4; ++i)
              if (val[i] != 4) sb += nearbyIrregular(v[i], skip);
            if (val[c] + val[c + 2] - 2 != 4) sa += nearbyIrregular(v[c], skip);
            if (val[c + 1] - 1 != 4) sa += nearbyIrregular(v[c + 1], skip);
            if (val[(c + 3) & 3] - 1 != 4) sa += nearbyIrregular(v[(c + 3) & 3], skip);
            if (sa > sb || (sa == sb && (r & 12u) == 0)) k = c;
          }
        }
      }
      if (k >= 0 && ed_.collapseDiagonal(e[k])) {
        val_[v[k]] = val[k] + val[k + 2] - 2;
        --val_[v[k + 1]];
        --val_[v[(k + 3) & 3]];
        ++done;
      }
    }
    return done;
  }

  // Rotates edges between two quads when that lowers (or in drift mode keeps, while gathering
  // irregular vertices) the irregularity of the six vertices around them.
  int rotatePass(Mode mode, std::uint32_t seed = 0) {
    int done = 0;
    const Index nh = m_.halfEdgeCount();
    for (Index h = 0; h < nh; ++h) {
      if (!ed_.halfEdgeAlive(h)) continue;
      const Index t = m_.heTwin[h];
      if (t == kInvalid || t < h) continue;
      const Index hn = m_.heNext[h], hnn = m_.heNext[hn], hp = m_.heNext[hnn];
      const Index tn = m_.heNext[t], tnn = m_.heNext[tn], tp = m_.heNext[tnn];
      if (m_.heNext[hp] != h || m_.heNext[tp] != t) continue;  // Both faces must be quads.
      // F = [a, b, c, e], G = [b, a, d, g]. Rotating once gives the edge d-c, twice e-g.
      const Index a = m_.heVert[h], b = m_.heVert[t], c = m_.heVert[hnn], e = m_.heVert[hp];
      const Index d = m_.heVert[tnn], g = m_.heVert[tp];
      const int va = val_[a], vb = val_[b], vc = val_[c], ve = val_[e], vd = val_[d], vg = val_[g];
      if (va <= 3 || vb <= 3) continue;
      const int base = irregularity(va) + irregularity(vb) + irregularity(vc) + irregularity(vd) + irregularity(ve) +
                       irregularity(vg);
      const int lowered = irregularity(va - 1) + irregularity(vb - 1);
      const int once = lowered + irregularity(vc + 1) + irregularity(vd + 1) + irregularity(ve) + irregularity(vg);
      const int twice = lowered + irregularity(vc) + irregularity(vd) + irregularity(ve + 1) + irregularity(vg + 1);
      int first = 0, second = 0;  // Configurations to try, in order: 1 = once, 2 = twice.
      if (once < base || twice < base) {
        first = once <= twice ? 1 : 2;
        second = first == 1 ? (twice < base ? 2 : 0) : (once < base ? 1 : 0);
      } else {
        if (mode != Mode::Drift || base == 0 || std::min(once, twice) > base) continue;
        // Score-neutral: take it only if it gathers irregular vertices (or now and then on a tie).
        const std::uint32_t r = hashMix(h, seed);
        const bool useOnce = once == base && (twice != base || (r & 2u));
        const Index six[6] = {a, b, c, d, e, g};
        const int delta[6] = {-1, -1, useOnce, useOnce, !useOnce, !useOnce};
        int sb = 0, sa = 0;
        for (int i = 0; i < 6; ++i) {
          const int v0 = val_[six[i]], v1 = v0 + delta[i];
          if (v0 == 4 && v1 == 4) continue;
          const int n = nearbyIrregular(six[i], six);
          if (v0 != 4) sb += n;
          if (v1 != 4) sa += n;
        }
        if (sa < sb || (sa == sb && (r & 12u) != 0)) continue;
        first = useOnce ? 1 : 2;
      }

      const auto& p = m_.positions;
      const Vec3 n = m_.faceAreaNormal(m_.heFace[h]) + m_.faceAreaNormal(m_.heFace[t]);
      auto valid = [&](int option) {
        if (option == 1) return quadOk({p[d], p[c], p[e], p[a]}, n) && quadOk({p[c], p[d], p[g], p[b]}, n);
        return quadOk({p[g], p[e], p[a], p[d]}, n) && quadOk({p[e], p[g], p[b], p[c]}, n) && !ed_.connected(e, g);
      };
      int steps = 0;
      if (valid(first)) {
        steps = first;
      } else if (second != 0 && valid(second)) {
        steps = second;
      }
      if (steps == 0 || !ed_.rotateEdge(h)) continue;
      // The second step cannot be refused: c and d gained an edge and e-g was checked above.
      if (steps == 2) ed_.rotateEdge(h);
      --val_[a];
      --val_[b];
      ++val_[steps == 1 ? c : e];
      ++val_[steps == 1 ? d : g];
      ++done;
    }
    return done;
  }

  void compact() { ed_.compact(); }

 private:
  // Irregular vertices within kGatherRings rings of v, not counting `skip`.
  int nearbyIrregular(Index v, const Index (&skip)[6]) {
    ++tick_;
    queue_.clear();
    queue_.push_back(v);
    stamp_[v] = tick_;
    for (const Index s : skip) stamp_[s] = tick_;
    int count = 0;
    std::size_t begin = 0;
    for (int ring = 0; ring < kGatherRings; ++ring) {
      const std::size_t end = queue_.size();
      for (std::size_t i = begin; i < end; ++i) {
        m_.forEachOutgoing(queue_[i], [&](Index he) {
          const Index w = m_.heTarget(he);
          if (stamp_[w] == tick_) return;
          stamp_[w] = tick_;
          count += val_[w] != 4;
          queue_.push_back(w);
        });
      }
      begin = end;
    }
    return count;
  }

  MeshEditor ed_;
  Mesh& m_;
  float h_;
  std::vector<int> val_;  // Valence of every vertex, kept up to date by the passes.
  std::vector<std::uint32_t> stamp_;
  std::uint32_t tick_ = 0;
  std::vector<Index> queue_;
};

// Moves vertices towards the average of their neighbours along the surface. The last step also
// projects them onto the reference, so the shape does not shrink or drift. (Tangential steps
// barely leave the surface, and projection is most of the cost.)
void relax(Mesh& m, const Mesh& ref, const Bvh& refBvh, float h, int iterations) {
  std::vector<Vec3> next(m.positions.size());
  for (int it = 0; it < iterations; ++it) {
    m.computeNormals();
    parallelFor(0, m.positions.size(), 2048, [&](std::size_t b, std::size_t e) {
      for (std::size_t i = b; i < e; ++i) {
        const Index v = static_cast<Index>(i);
        const Vec3 p = m.positions[v], n = m.normals[v];
        Vec3 sum{0.0f};
        int count = 0;
        m.forEachOutgoing(v, [&](Index he) {
          sum += m.positions[m.heTarget(he)];
          ++count;
        });
        Vec3 d = count ? sum / float(count) - p : Vec3{0.0f};
        d -= n * glm::dot(d, n);
        Vec3 q = p + kRelaxRate * d;
        Bvh::ClosestHit hit;
        if (it + 1 == iterations && refBvh.closestPoint(ref, q, kProjectDistance * h, hit) &&
            glm::dot(hit.faceNormal, n) > 0.0f) {
          q = hit.position;
        }
        next[v] = q;
      }
    });
    m.positions.swap(next);
  }
  m.computeNormals();
}

// Splits every quad into four: new vertices at edge midpoints and face centres. Existing
// vertices keep their valence and every new vertex has valence 4.
Mesh subdivideQuads(const Mesh& m) {
  std::vector<Vec3> pos(m.positions);
  std::vector<Index> edgeVert(m.halfEdgeCount(), kInvalid);
  for (Index h = 0; h < m.halfEdgeCount(); ++h) {
    if (edgeVert[h] != kInvalid) continue;
    const Index v = static_cast<Index>(pos.size());
    pos.push_back((m.positions[m.heVert[h]] + m.positions[m.heTarget(h)]) * 0.5f);
    edgeVert[h] = v;
    if (m.heTwin[h] != kInvalid) edgeVert[m.heTwin[h]] = v;
  }
  std::vector<Index> idx, sizes;
  idx.reserve(m.halfEdgeCount() * 4);
  sizes.reserve(m.halfEdgeCount());
  for (Index f = 0; f < m.faceCount(); ++f) {
    const Index c = static_cast<Index>(pos.size());
    pos.push_back(m.faceCentroid(f));
    const Index h0 = m.faceHe[f];
    Index h = h0;
    do {
      const Index next = m.heNext[h];
      idx.insert(idx.end(), {m.heVert[next], edgeVert[next], c, edgeVert[h]});
      sizes.push_back(4);
      h = next;
    } while (h != h0);
  }
  return buildMesh(std::move(pos), idx, sizes);
}

}  // namespace

std::optional<Mesh> quadRemesh(const Mesh& input, const QuadRemeshParams& params, QuadRemeshStats* stats,
                               std::string* error) {
  QuadRemeshStats local;
  QuadRemeshStats& st = stats ? *stats : local;
  const bool optimize = params.rounds > 0;
  const int subdivisions = optimize ? std::clamp(params.subdivisions, 0, 3) : 0;
  float h = params.targetEdge * float(1 << subdivisions);  // Edge length of the current mesh.
  std::optional<Mesh> out = voxelRemesh(input, {h, params.maxResolution}, &st.voxel, error);
  if (!out) return out;
  const bool masked = input.anyMasked();
  const bool faceSets = input.hasFaceSetData();
  if (!optimize) {  // Plain voxel remesh.
    st.raw = st.optimized = measureQuality(*out);
    if (masked || faceSets) {
      Mesh ref = input;
      Bvh refBvh;
      refBvh.build(ref, {.maxLeafFaces = 8});
      if (masked) transferMask(ref, refBvh, *out);
      if (faceSets) transferFaceSets(ref, refBvh, *out);
    }
    return out;
  }

  Timer t;
  Mesh ref = input;
  Bvh refBvh;
  refBvh.build(ref, {.maxLeafFaces = 8});  // Small leaves: millions of point queries follow.
  const Mesh* errRef = params.measureError ? &ref : nullptr;
  const Bvh* errBvh = params.measureError ? &refBvh : nullptr;
  double statsMs = 0.0;
  {
    Timer s;
    st.raw = measureQuality(*out, errRef, errBvh);
    statsMs = s.ms();
  }
  using Mode = Optimizer::Mode;
  for (int round = 0; round < params.rounds; ++round) {
    if (params.optimizeValence) {
      Optimizer opt(*out, h);
      st.collapsed += opt.collapsePass(Mode::Improve);
      st.rotated += opt.rotatePass(Mode::Improve);
      for (int k = 0; k < params.driftPasses; ++k) {
        const auto seed = static_cast<std::uint32_t>(round * 1000 + k);
        st.rotated += opt.rotatePass(Mode::Drift, seed);
        st.collapsed += opt.collapsePass(Mode::Drift, seed);
        st.rotated += opt.rotatePass(Mode::Improve);
      }
      opt.compact();
    }
    relax(*out, ref, refBvh, h, params.relaxIterations);
  }
  for (int i = 0; i < subdivisions; ++i) {
    *out = subdivideQuads(*out);
    h *= 0.5f;
    relax(*out, ref, refBvh, h, params.relaxIterations);
  }
  if (masked) transferMask(ref, refBvh, *out);
  if (faceSets) transferFaceSets(ref, refBvh, *out);
  st.optimizeMs = t.ms() - statsMs;
  st.optimized = measureQuality(*out, errRef, errBvh);
  return out;
}

void transferMask(const Mesh& source, const Bvh& sourceBvh, Mesh& target) {
  if (!source.anyMasked()) {
    target.mask.clear();
    return;
  }
  target.mask.assign(target.positions.size(), 0.0f);
  parallelFor(0, target.positions.size(), 2048, [&](std::size_t b, std::size_t e) {
    for (std::size_t v = b; v < e; ++v) {
      Bvh::ClosestHit hit;
      if (!sourceBvh.closestPoint(source, target.positions[v], std::numeric_limits<float>::infinity(), hit)) continue;
      const Index* c = hit.corners;
      const Vec3 w = barycentric(hit.position, source.positions[c[0]], source.positions[c[1]], source.positions[c[2]]);
      const float value = w.x * source.mask[c[0]] + w.y * source.mask[c[1]] + w.z * source.mask[c[2]];
      target.mask[v] = std::clamp(value, 0.0f, 1.0f);
    }
  });
}

void transferFaceSets(const Mesh& source, const Bvh& sourceBvh, Mesh& target) {
  if (!source.hasFaceSetData()) {
    target.faceSets.clear();
    return;
  }
  target.faceSets.assign(target.faceHe.size(), kDefaultFaceSet);
  parallelFor(0, target.faceHe.size(), 1024, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const Index f = static_cast<Index>(i);
      if (target.faceHe[f] == kInvalid) continue;
      const Vec3 c = target.faceCentroid(f);
      const Vec3 an = target.faceAreaNormal(f);
      constexpr float kInf = std::numeric_limits<float>::infinity();
      Bvh::ClosestHit hit;
      if (!sourceBvh.closestPoint(source, c, kInf, hit)) continue;
      if (glm::dot(hit.faceNormal, an) <= 0.0f) {
        // Probably the far side of a thin part. Ask again from a point just outside this face,
        // which is closer to the side the face belongs to.
        const float len = glm::length(an);
        Bvh::ClosestHit outside;
        if (len > 1e-20f && sourceBvh.closestPoint(source, c + an * (0.5f * std::sqrt(0.5f * len) / len), kInf,
                                                   outside) &&
            glm::dot(outside.faceNormal, an) > 0.0f)
          hit = outside;
      }
      target.faceSets[f] = source.faceSets[hit.face];
    }
  });
}

MeshQuality measureQuality(const Mesh& m, const Mesh* reference, const Bvh* referenceBvh) {
  MeshQuality q;
  if (m.faceCount() == 0) return q;
  Index quads = 0, val4 = 0;
  for (Index f = 0; f < m.faceCount(); ++f) quads += m.faceSize(f) == 4;
  for (Index v = 0; v < m.vertexCount(); ++v) val4 += m.valence(v) == 4;
  q.quadRatio = double(quads) / m.faceCount();
  q.valence4Ratio = m.vertexCount() ? double(val4) / m.vertexCount() : 0.0;

  double sum = 0.0, sumSq = 0.0;
  std::size_t edges = 0;
  for (Index h = 0; h < m.halfEdgeCount(); ++h) {
    const Index t = m.heTwin[h];
    if (t != kInvalid && t < h) continue;
    const double len = glm::length(m.positions[m.heTarget(h)] - m.positions[m.heVert[h]]);
    sum += len;
    sumSq += len * len;
    ++edges;
  }
  const double mean = sum / double(edges);
  q.meanEdge = mean;
  q.edgeLengthCv = mean > 0.0 ? std::sqrt(std::max(0.0, sumSq / double(edges) - mean * mean)) / mean : 0.0;

  if (reference && referenceBvh && mean > 0.0) {
    const float maxDist = float(10.0 * mean);
    std::vector<float> err(m.faceCount());
    parallelFor(0, err.size(), 1024, [&](std::size_t b, std::size_t e) {
      for (std::size_t f = b; f < e; ++f) {
        Bvh::ClosestHit hit;
        const Vec3 c = m.faceCentroid(static_cast<Index>(f));
        err[f] = referenceBvh->closestPoint(*reference, c, maxDist, hit) ? std::sqrt(hit.distSq) : maxDist;
      }
    });
    double total = 0.0, worst = 0.0;
    for (float e : err) {
      total += e;
      worst = std::max(worst, double(e));
    }
    q.meanError = total / double(err.size()) / mean;
    q.maxError = worst / mean;
  }
  return q;
}

}  // namespace plegl
