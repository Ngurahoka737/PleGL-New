#include "remesh/VoxelGrid.h"

#include <algorithm>
#include <cmath>

#include "core/Parallel.h"

namespace plegl {

namespace {

using P2 = glm::vec2;

// Twice the signed area of (p0, p1, q): positive when q is left of p0 -> p1.
inline float edgeFn(const P2& p0, const P2& p1, const P2& q) {
  return (p1.x - p0.x) * (q.y - p0.y) - (p1.y - p0.y) * (q.x - p0.x);
}

// Squared distance from p to triangle abc (Ericson, Real-Time Collision Detection 5.1.5).
float pointTriangleDistSq(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
  const Vec3 ab = b - a, ac = c - a, ap = p - a;
  const float d1 = glm::dot(ab, ap), d2 = glm::dot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) return glm::dot(ap, ap);
  const Vec3 bp = p - b;
  const float d3 = glm::dot(ab, bp), d4 = glm::dot(ac, bp);
  if (d3 >= 0.0f && d4 <= d3) return glm::dot(bp, bp);
  const float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    const Vec3 q = a + ab * (d1 / (d1 - d3));
    return glm::dot(p - q, p - q);
  }
  const Vec3 cp = p - c;
  const float d5 = glm::dot(ab, cp), d6 = glm::dot(ac, cp);
  if (d6 >= 0.0f && d5 <= d6) return glm::dot(cp, cp);
  const float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    const Vec3 q = a + ac * (d2 / (d2 - d6));
    return glm::dot(p - q, p - q);
  }
  const float va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
    const Vec3 q = b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    return glm::dot(p - q, p - q);
  }
  const float denom = 1.0f / (va + vb + vc);
  const Vec3 q = a + ab * (vb * denom) + ac * (vc * denom);
  return glm::dot(p - q, p - q);
}

struct Crossing {
  std::uint32_t row;
  float t;     // Position along the ray axis, in voxels.
  int w;       // +1 or -1 depending on which way the surface faces.
};

}  // namespace

bool VoxelGrid::build(const Mesh& mesh, const Params& params, std::string* error) {
  auto fail = [&](std::string msg) {
    if (error) *error = std::move(msg);
    return false;
  };
  if (mesh.faceCount() == 0) return fail("The mesh has no faces.");
  if (!(params.voxelSize > 0.0f)) return fail("Voxel size must be positive.");

  Aabb bounds;
  for (const Vec3& p : mesh.positions) bounds.expand(p);
  h_ = params.voxelSize;
  // Pad so the outermost nodes are always outside and farther than the band from the surface.
  const int pad = static_cast<int>(std::ceil(kBand)) + 1;
  for (int a = 0; a < 3; ++a) {
    const double cells = std::ceil(static_cast<double>(bounds.extent()[a]) / h_);
    const double nodes = cells + 1.0 + 2.0 * pad;
    if (nodes > params.maxResolution) {
      return fail("Voxel size is too small for this mesh: it needs " + std::to_string(static_cast<long long>(nodes)) +
                  " voxels across, the limit is " + std::to_string(params.maxResolution) + ".");
    }
    n_[a] = static_cast<int>(nodes);
  }
  origin_ = bounds.min - Vec3(static_cast<float>(pad) * h_);

  // Vertices in grid units, computed once so shared vertices give bit-identical edge tests.
  std::vector<Vec3> g(mesh.positions.size());
  for (std::size_t i = 0; i < g.size(); ++i) g[i] = (mesh.positions[i] - origin_) / h_;
  std::vector<std::array<Index, 3>> tris;
  tris.reserve(static_cast<std::size_t>(mesh.faceCount()) * 2);
  std::vector<Index> fv;
  for (Index f = 0; f < mesh.faceCount(); ++f) {
    fv.clear();
    mesh.forEachFaceVertex(f, [&](Index v) { fv.push_back(v); });
    for (std::size_t i = 1; i + 1 < fv.size(); ++i) tris.push_back({fv[0], fv[i], fv[i + 1]});
  }

  computeSigns(g, tris);
  computeDistances(g, tris);
  return true;
}

void VoxelGrid::computeSigns(const std::vector<Vec3>& g, const std::vector<std::array<Index, 3>>& tris) {
  votes_.assign(static_cast<std::size_t>(n_[0]) * n_[1] * n_[2], 0);
  constexpr std::size_t kChunk = 2048;
  const std::size_t chunks = (tris.size() + kChunk - 1) / kChunk;

  for (int a = 0; a < 3; ++a) {
    const int u = (a + 1) % 3, v = (a + 2) % 3;  // (u, v, a) is right-handed.
    const int nu = n_[u], nv = n_[v], na = n_[a];

    // 1. Every triangle reports where it crosses each grid row parallel to axis a.
    std::vector<std::vector<Crossing>> perChunk(chunks);
    parallelFor(0, chunks, 1, [&](std::size_t cb, std::size_t ce) {
      for (std::size_t c = cb; c < ce; ++c) {
        auto& out = perChunk[c];
        const std::size_t end = std::min(tris.size(), (c + 1) * kChunk);
        for (std::size_t ti = c * kChunk; ti < end; ++ti) {
          const auto& t = tris[ti];
          const P2 p[3] = {{g[t[0]][u], g[t[0]][v]}, {g[t[1]][u], g[t[1]][v]}, {g[t[2]][u], g[t[2]][v]}};
          const float area2 = edgeFn(p[0], p[1], p[2]);
          if (area2 == 0.0f) continue;  // Parallel to the rays: never crossed.
          const int u0 = std::max(0, static_cast<int>(std::ceil(std::min({p[0].x, p[1].x, p[2].x}))));
          const int u1 = std::min(nu - 1, static_cast<int>(std::floor(std::max({p[0].x, p[1].x, p[2].x}))));
          const int v0 = std::max(0, static_cast<int>(std::ceil(std::min({p[0].y, p[1].y, p[2].y}))));
          const int v1 = std::min(nv - 1, static_cast<int>(std::floor(std::max({p[0].y, p[1].y, p[2].y}))));
          if (u0 > u1 || v0 > v1) continue;
          // Edge tests use the edge's endpoints in index order, so the two triangles sharing an
          // edge evaluate the exact same expression. A row exactly on the edge then belongs to
          // exactly one of them: the one whose interior lies on the positive side.
          struct EdgeTest {
            P2 lo, hi;
            float side;  // Sign of the opposite vertex.
          } edges[3];
          bool degenerate = false;
          for (int e = 0; e < 3; ++e) {
            int i0 = e, i1 = (e + 1) % 3;
            if (t[i1] < t[i0]) std::swap(i0, i1);
            edges[e].lo = p[i0];
            edges[e].hi = p[i1];
            edges[e].side = edgeFn(p[i0], p[i1], p[(e + 2) % 3]);
            degenerate |= edges[e].side == 0.0f;
          }
          if (degenerate) continue;
          const int w = area2 > 0.0f ? 1 : -1;
          const float invArea = 1.0f / area2;
          for (int jv = v0; jv <= v1; ++jv) {
            for (int ju = u0; ju <= u1; ++ju) {
              const P2 q{static_cast<float>(ju), static_cast<float>(jv)};
              bool in = true;
              for (const EdgeTest& et : edges) {
                const float e = edgeFn(et.lo, et.hi, q);
                if (!(e * et.side > 0.0f || (e == 0.0f && et.side > 0.0f))) {
                  in = false;
                  break;
                }
              }
              if (!in) continue;
              const float l0 = edgeFn(p[1], p[2], q) * invArea;
              const float l1 = edgeFn(p[2], p[0], q) * invArea;
              const float l2 = 1.0f - l0 - l1;
              const float ta = l0 * g[t[0]][a] + l1 * g[t[1]][a] + l2 * g[t[2]][a];
              out.push_back({static_cast<std::uint32_t>(ju + nu * jv), ta, w});
            }
          }
        }
      }
    });

    // 2. Group crossings by row (counting sort).
    const std::size_t rows = static_cast<std::size_t>(nu) * nv;
    std::vector<std::uint32_t> start(rows + 1, 0);
    for (const auto& list : perChunk)
      for (const Crossing& c : list) ++start[c.row + 1];
    for (std::size_t r = 0; r < rows; ++r) start[r + 1] += start[r];
    std::vector<Crossing> sorted(start[rows]);
    {
      std::vector<std::uint32_t> fill(start.begin(), start.end() - 1);
      for (const auto& list : perChunk)
        for (const Crossing& c : list) sorted[fill[c.row]++] = c;
    }
    perChunk.clear();

    // 3. Walk each row: a node is inside along this axis when the winding number before it is
    // non-zero. Each node is written by exactly one row, so rows run in parallel.
    parallelFor(0, rows, 256, [&](std::size_t rb, std::size_t re) {
      for (std::size_t r = rb; r < re; ++r) {
        const std::uint32_t b = start[r], e = start[r + 1];
        if (b == e) continue;
        std::sort(sorted.begin() + b, sorted.begin() + e, [](const Crossing& x, const Crossing& y) { return x.t < y.t; });
        int c[3];
        c[u] = static_cast<int>(r % nu);
        c[v] = static_cast<int>(r / nu);
        int winding = 0;
        std::uint32_t k = b;
        for (int i = 0; i < na; ++i) {
          while (k < e && sorted[k].t < static_cast<float>(i)) winding += sorted[k++].w;
          if (k == e && winding == 0) break;
          if (winding != 0) {
            c[a] = i;
            ++votes_[nodeIndex(c[0], c[1], c[2])];
          }
        }
      }
    });
  }
}

void VoxelGrid::computeDistances(const std::vector<Vec3>& g, const std::vector<std::array<Index, 3>>& tris) {
  for (int a = 0; a < 3; ++a) b_[a] = (n_[a] + kBlock - 1) / kBlock;
  blockTable_.assign(static_cast<std::size_t>(b_[0]) * b_[1] * b_[2], -1);
  blockCoords_.clear();

  // Node range each triangle can reach within the band.
  auto nodeRange = [&](const std::array<Index, 3>& t, int lo[3], int hi[3]) {
    for (int a = 0; a < 3; ++a) {
      const float mn = std::min({g[t[0]][a], g[t[1]][a], g[t[2]][a]}) - kBand;
      const float mx = std::max({g[t[0]][a], g[t[1]][a], g[t[2]][a]}) + kBand;
      lo[a] = std::max(0, static_cast<int>(std::ceil(mn)));
      hi[a] = std::min(n_[a] - 1, static_cast<int>(std::floor(mx)));
    }
  };

  // Mark and allocate the blocks the band touches, and bucket triangles by block layer in z.
  std::vector<std::vector<std::uint32_t>> layers(static_cast<std::size_t>(b_[2]));
  for (std::size_t ti = 0; ti < tris.size(); ++ti) {
    int lo[3], hi[3];
    nodeRange(tris[ti], lo, hi);
    if (lo[0] > hi[0] || lo[1] > hi[1] || lo[2] > hi[2]) continue;
    for (int bz = lo[2] / kBlock; bz <= hi[2] / kBlock; ++bz) {
      layers[bz].push_back(static_cast<std::uint32_t>(ti));
      for (int by = lo[1] / kBlock; by <= hi[1] / kBlock; ++by)
        for (int bx = lo[0] / kBlock; bx <= hi[0] / kBlock; ++bx)
          blockTable_[bx + static_cast<std::size_t>(b_[0]) * (by + static_cast<std::size_t>(b_[1]) * bz)] = 0;
    }
  }
  int slots = 0;
  for (int bz = 0; bz < b_[2]; ++bz)
    for (int by = 0; by < b_[1]; ++by)
      for (int bx = 0; bx < b_[0]; ++bx) {
        int& s = blockTable_[bx + static_cast<std::size_t>(b_[0]) * (by + static_cast<std::size_t>(b_[1]) * bz)];
        if (s < 0) continue;
        s = slots++;
        blockCoords_.push_back({bx, by, bz});
      }
  constexpr int kBlockNodes = kBlock * kBlock * kBlock;
  distSq_.assign(static_cast<std::size_t>(slots) * kBlockNodes, kBand * kBand);

  // Each layer of blocks is written by one task only.
  parallelFor(0, layers.size(), 1, [&](std::size_t lb, std::size_t le) {
    for (std::size_t bz = lb; bz < le; ++bz) {
      for (std::uint32_t ti : layers[bz]) {
        const auto& t = tris[ti];
        const Vec3 &A = g[t[0]], &B = g[t[1]], &C = g[t[2]];
        Vec3 n = glm::cross(B - A, C - A);
        const float nl = glm::length(n);
        const bool hasNormal = nl > 1e-12f;
        if (hasNormal) n /= nl;
        int lo[3], hi[3];
        nodeRange(t, lo, hi);
        const int k0 = std::max(lo[2], static_cast<int>(bz) * kBlock);
        const int k1 = std::min(hi[2], static_cast<int>(bz) * kBlock + kBlock - 1);
        for (int k = k0; k <= k1; ++k)
          for (int j = lo[1]; j <= hi[1]; ++j)
            for (int i = lo[0]; i <= hi[0]; ++i) {
              const Vec3 p(static_cast<float>(i), static_cast<float>(j), static_cast<float>(k));
              if (hasNormal && std::abs(glm::dot(p - A, n)) >= kBand) continue;  // Cheap plane reject.
              const int slot = slotOf(i, j, k);
              float& d = distSq_[static_cast<std::size_t>(slot) * kBlockNodes + (i % kBlock) +
                                 kBlock * ((j % kBlock) + kBlock * (k % kBlock))];
              const float dd = pointTriangleDistSq(p, A, B, C);
              if (dd < d) d = dd;
            }
      }
    }
  });
}

float VoxelGrid::distance(int i, int j, int k) const {
  const int slot = slotOf(i, j, k);
  float mag = kBand;
  if (slot >= 0) {
    constexpr int kBlockNodes = kBlock * kBlock * kBlock;
    mag = std::sqrt(distSq_[static_cast<std::size_t>(slot) * kBlockNodes + (i % kBlock) +
                            kBlock * ((j % kBlock) + kBlock * (k % kBlock))]);
  }
  return inside(i, j, k) ? -mag : mag;
}

std::size_t VoxelGrid::memoryBytes() const {
  return votes_.size() + blockTable_.size() * sizeof(int) + distSq_.size() * sizeof(float) +
         blockCoords_.size() * sizeof(blockCoords_[0]);
}

}  // namespace plegl
