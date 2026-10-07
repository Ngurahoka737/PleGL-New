#include "remesh/QuadRemesh.h"

#include <algorithm>
#include <array>
#include <cmath>

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

class Optimizer {
 public:
  Optimizer(Mesh& mesh, float h) : ed_(mesh), m_(mesh), h_(h), val_(mesh.vertexCount()) {
    parallelFor(0, val_.size(), 4096, [&](std::size_t b, std::size_t e) {
      for (std::size_t v = b; v < e; ++v) val_[v] = m_.valence(static_cast<Index>(v));
    });
  }

  // Removes slivers and quads whose collapse makes the surrounding valences more regular.
  int collapsePass() {
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
      const float d0 = glm::length(m_.positions[v[0]] - m_.positions[v[2]]);
      const float d1 = glm::length(m_.positions[v[1]] - m_.positions[v[3]]);
      int k = -1;
      if (std::min(d0, d1) < kSliverDiagonal * h_) {
        k = d0 < d1 ? 0 : 1;
      } else {
        int bestGain = 0;
        for (int c = 0; c < 2; ++c) {
          if ((c == 0 ? d0 : d1) > kDiamondDiagonal * h_) continue;
          const int a = val[c], b = val[c + 1], o = val[c + 2], d = val[(c + 3) & 3];
          const int before = irregularity(a) + irregularity(o) + irregularity(b) + irregularity(d);
          const int after = irregularity(a + o - 2) + irregularity(b - 1) + irregularity(d - 1);
          if (before - after > bestGain) {
            bestGain = before - after;
            k = c;
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

  // Rotates edges between two quads when that lowers the irregularity of the six vertices
  // around them.
  int rotatePass() {
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
      const int common = irregularity(va) + irregularity(vb);
      const int base = common + irregularity(vc) + irregularity(vd) + irregularity(ve) + irregularity(vg);
      const int lowered = irregularity(va - 1) + irregularity(vb - 1);
      const int once = lowered + irregularity(vc + 1) + irregularity(vd + 1) + irregularity(ve) + irregularity(vg);
      const int twice = lowered + irregularity(vc) + irregularity(vd) + irregularity(ve + 1) + irregularity(vg + 1);
      const int best = std::min(once, twice);
      if (best >= base) continue;

      const auto& p = m_.positions;
      const Vec3 n = m_.faceAreaNormal(m_.heFace[h]) + m_.faceAreaNormal(m_.heFace[t]);
      int steps = 0;
      if (once == best && quadOk({p[d], p[c], p[e], p[a]}, n) && quadOk({p[c], p[d], p[g], p[b]}, n)) {
        steps = 1;
      } else if (twice < base && quadOk({p[g], p[e], p[a], p[d]}, n) && quadOk({p[e], p[g], p[b], p[c]}, n) &&
                 !ed_.connected(e, g)) {
        steps = 2;
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
  MeshEditor ed_;
  Mesh& m_;
  float h_;
  std::vector<int> val_;  // Valence of every vertex, kept up to date by the passes.
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

}  // namespace

std::optional<Mesh> quadRemesh(const Mesh& input, const QuadRemeshParams& params, QuadRemeshStats* stats,
                               std::string* error) {
  QuadRemeshStats local;
  QuadRemeshStats& st = stats ? *stats : local;
  std::optional<Mesh> out = voxelRemesh(input, {params.targetEdge, params.maxResolution}, &st.voxel, error);
  if (!out) return out;
  if (params.rounds <= 0) {  // Plain voxel remesh.
    st.raw = st.optimized = measureQuality(*out);
    return out;
  }

  Timer t;
  Mesh ref = input;
  Bvh refBvh;
  refBvh.build(ref, {.maxLeafFaces = 8});  // Small leaves: millions of point queries follow.
  const float h = params.targetEdge;
  const Mesh* errRef = params.measureError ? &ref : nullptr;
  const Bvh* errBvh = params.measureError ? &refBvh : nullptr;
  double statsMs = 0.0;
  {
    Timer s;
    st.raw = measureQuality(*out, errRef, errBvh);
    statsMs = s.ms();
  }
  for (int round = 0; round < params.rounds; ++round) {
    if (params.optimizeValence) {
      Optimizer opt(*out, h);
      st.collapsed += opt.collapsePass();
      st.rotated += opt.rotatePass();
      opt.compact();
    }
    relax(*out, ref, refBvh, h, params.relaxIterations);
  }
  st.optimizeMs = t.ms() - statsMs;
  st.optimized = measureQuality(*out, errRef, errBvh);
  return out;
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
