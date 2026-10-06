#include "remesh/SurfaceNets.h"

#include <algorithm>
#include <array>
#include <cstdint>

#include "core/Parallel.h"

namespace plegl {

namespace {

// Cell corners are numbered x + 2y + 4z. Cell edges are numbered a * 4 + o[u] + 2 * o[v] for an
// edge along axis a, where u = a + 1 and v = a + 2 (mod 3) and o are the edge's fixed offsets.
int edgeId(int a, const int off[3]) { return a * 4 + off[(a + 1) % 3] + 2 * off[(a + 2) % 3]; }
int cornerId(const int off[3]) { return off[0] + 2 * off[1] + 4 * off[2]; }

struct Tables {
  int edgeCorner[12][2];  // Corner at the low and high end of each edge.
  int faceCorner[6][4];   // Corners of each face in cyclic order.
  int faceEdge[6][4];     // faceEdge[f][k] joins faceCorner[f][k] and faceCorner[f][k + 1].

  Tables() {
    for (int a = 0; a < 3; ++a)
      for (int b0 = 0; b0 < 2; ++b0)
        for (int b1 = 0; b1 < 2; ++b1) {
          int off[3];
          off[(a + 1) % 3] = b0;
          off[(a + 2) % 3] = b1;
          off[a] = 0;
          const int e = edgeId(a, off);
          edgeCorner[e][0] = cornerId(off);
          off[a] = 1;
          edgeCorner[e][1] = cornerId(off);
        }
    for (int f = 0; f < 3; ++f)
      for (int s = 0; s < 2; ++s) {
        const int face = f * 2 + s;
        const int p = (f + 1) % 3, q = (f + 2) % 3;
        const int cyc[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
        for (int k = 0; k < 4; ++k) {
          int off[3];
          off[f] = s;
          off[p] = cyc[k][0];
          off[q] = cyc[k][1];
          faceCorner[face][k] = cornerId(off);
          // Edge to the next corner: along p for k = 0, 2 and along q for k = 1, 3, sitting at
          // the lower of the two coordinates on that axis.
          const int axis = (k % 2 == 0) ? p : q;
          int eoff[3] = {off[0], off[1], off[2]};
          eoff[axis] = 0;
          if (axis == q) eoff[p] = cyc[k][0];  // k = 1: p = 1; k = 3: p = 0.
          if (axis == p) eoff[q] = cyc[k][1];  // k = 0: q = 0; k = 2: q = 1.
          faceEdge[face][k] = edgeId(axis, eoff);
        }
      }
  }
};

const Tables& tables() {
  static const Tables t;
  return t;
}

struct CellInfo {
  std::uint32_t firstVertex;  // Within the block's vertex list.
  std::uint8_t component[12]; // Patch of each crossing edge, 0xFF if the edge is not crossed.
};

struct BlockData {
  std::vector<int> cellIndex;  // Per cell of the block: index into cells, or -1.
  std::vector<CellInfo> cells;
  std::vector<Vec3> vertices;
  std::vector<Index> quads;
};

int findRoot(int* parent, int x) {
  while (parent[x] != x) x = parent[x] = parent[parent[x]];
  return x;
}

}  // namespace

SurfaceNetsMesh extractSurfaceNets(const VoxelGrid& grid) {
  const Tables& T = tables();
  constexpr int B = VoxelGrid::kBlock;
  const auto& blocks = grid.blocks();
  std::vector<BlockData> data(blocks.size());
  const int n[3] = {grid.nx(), grid.ny(), grid.nz()};
  const Vec3 origin = grid.origin();
  const float h = grid.voxelSize();

  // Pass 1: one vertex per surface patch in every cell the surface crosses.
  parallelFor(0, blocks.size(), 4, [&](std::size_t sb, std::size_t se) {
    for (std::size_t slot = sb; slot < se; ++slot) {
      BlockData& bd = data[slot];
      bd.cellIndex.assign(B * B * B, -1);
      const auto& bc = blocks[slot];
      for (int lz = 0; lz < B; ++lz)
        for (int ly = 0; ly < B; ++ly)
          for (int lx = 0; lx < B; ++lx) {
            const int ci = bc[0] * B + lx, cj = bc[1] * B + ly, ck = bc[2] * B + lz;
            if (ci + 1 >= n[0] || cj + 1 >= n[1] || ck + 1 >= n[2]) continue;
            bool in[8];
            int insideCount = 0;
            for (int c = 0; c < 8; ++c) {
              in[c] = grid.inside(ci + (c & 1), cj + ((c >> 1) & 1), ck + ((c >> 2) & 1));
              insideCount += in[c];
            }
            if (insideCount == 0 || insideCount == 8) continue;
            float d[8];
            for (int c = 0; c < 8; ++c) d[c] = grid.distance(ci + (c & 1), cj + ((c >> 1) & 1), ck + ((c >> 2) & 1));

            int parent[12];
            for (int e = 0; e < 12; ++e) parent[e] = e;
            auto crossed = [&](int e) { return in[T.edgeCorner[e][0]] != in[T.edgeCorner[e][1]]; };
            auto unite = [&](int x, int y) { parent[findRoot(parent, x)] = findRoot(parent, y); };
            for (int f = 0; f < 6; ++f) {
              int crossing[4], count = 0;
              for (int k = 0; k < 4; ++k)
                if (crossed(T.faceEdge[f][k])) crossing[count++] = k;
              if (count == 2) {
                unite(T.faceEdge[f][crossing[0]], T.faceEdge[f][crossing[1]]);
              } else if (count == 4) {
                // Saddle face. Sum the corners in a fixed (global index) order so the neighbour
                // sharing this face gets the bit-identical value and the same pairing.
                std::array<std::pair<std::size_t, float>, 4> corners;
                for (int k = 0; k < 4; ++k) {
                  const int c = T.faceCorner[f][k];
                  const std::size_t gi = static_cast<std::size_t>(ci + (c & 1)) +
                                         static_cast<std::size_t>(n[0]) *
                                             (static_cast<std::size_t>(cj + ((c >> 1) & 1)) +
                                              static_cast<std::size_t>(n[1]) * (ck + ((c >> 2) & 1)));
                  corners[k] = {gi, d[c]};
                }
                std::sort(corners.begin(), corners.end());
                const float centre = ((corners[0].second + corners[1].second) + corners[2].second) + corners[3].second;
                // Inside centre: the inside corners connect through it, so the curves cut off
                // the two outside corners. Pair the two edges around each corner of that kind.
                const bool cutOutside = centre < 0.0f;
                for (int k = 0; k < 4; ++k) {
                  const bool cornerInside = in[T.faceCorner[f][k]];
                  if (cornerInside == cutOutside) continue;
                  unite(T.faceEdge[f][(k + 3) % 4], T.faceEdge[f][k]);
                }
              }
            }

            CellInfo info;
            info.firstVertex = static_cast<std::uint32_t>(bd.vertices.size());
            std::fill(std::begin(info.component), std::end(info.component), std::uint8_t{0xFF});
            int rootLabel[12];
            std::fill(std::begin(rootLabel), std::end(rootLabel), -1);
            Vec3 sum[12];
            int cnt[12] = {};
            int labels = 0;
            for (int e = 0; e < 12; ++e) {
              if (!crossed(e)) continue;
              const int r = findRoot(parent, e);
              if (rootLabel[r] < 0) {
                rootLabel[r] = labels;
                sum[labels] = Vec3{0.0f};
                ++labels;
              }
              const int l = rootLabel[r];
              info.component[e] = static_cast<std::uint8_t>(l);
              const int c0 = T.edgeCorner[e][0], c1 = T.edgeCorner[e][1];
              const float denom = d[c0] - d[c1];
              const float t = std::clamp(denom != 0.0f ? d[c0] / denom : 0.5f, 0.0f, 1.0f);
              const Vec3 p0(static_cast<float>(c0 & 1), static_cast<float>((c0 >> 1) & 1), static_cast<float>((c0 >> 2) & 1));
              const Vec3 p1(static_cast<float>(c1 & 1), static_cast<float>((c1 >> 1) & 1), static_cast<float>((c1 >> 2) & 1));
              sum[l] += p0 + (p1 - p0) * t;
              ++cnt[l];
            }
            const Vec3 base(static_cast<float>(ci), static_cast<float>(cj), static_cast<float>(ck));
            for (int l = 0; l < labels; ++l)
              bd.vertices.push_back(origin + (base + sum[l] / static_cast<float>(cnt[l])) * h);
            bd.cellIndex[lx + B * (ly + B * lz)] = static_cast<int>(bd.cells.size());
            bd.cells.push_back(info);
          }
    }
  });

  std::vector<Index> vertexBase(blocks.size() + 1, 0);
  for (std::size_t s = 0; s < blocks.size(); ++s)
    vertexBase[s + 1] = vertexBase[s] + static_cast<Index>(data[s].vertices.size());

  // Vertex of the patch that edge (axis a, starting at node) belongs to in the given cell.
  auto cellVertex = [&](int ci, int cj, int ck, int a, const int node[3]) -> Index {
    if (ci < 0 || cj < 0 || ck < 0) return kInvalid;
    const int slot = grid.slotOf(ci, cj, ck);
    if (slot < 0) return kInvalid;
    const BlockData& bd = data[slot];
    const int idx = bd.cellIndex[(ci % B) + B * ((cj % B) + B * (ck % B))];
    if (idx < 0) return kInvalid;
    const int cell[3] = {ci, cj, ck};
    int off[3];
    for (int x = 0; x < 3; ++x) off[x] = node[x] - cell[x];
    off[a] = 0;
    const std::uint8_t comp = bd.cells[idx].component[edgeId(a, off)];
    if (comp == 0xFF) return kInvalid;
    return vertexBase[slot] + static_cast<Index>(bd.cells[idx].firstVertex + comp);
  };

  // Pass 2: one quad per crossed grid edge, joining the patches of its four cells.
  parallelFor(0, blocks.size(), 4, [&](std::size_t sb, std::size_t se) {
    for (std::size_t slot = sb; slot < se; ++slot) {
      BlockData& bd = data[slot];
      const auto& bc = blocks[slot];
      for (int lz = 0; lz < B; ++lz)
        for (int ly = 0; ly < B; ++ly)
          for (int lx = 0; lx < B; ++lx) {
            const int node[3] = {bc[0] * B + lx, bc[1] * B + ly, bc[2] * B + lz};
            if (node[0] >= n[0] || node[1] >= n[1] || node[2] >= n[2]) continue;
            const bool in0 = grid.inside(node[0], node[1], node[2]);
            for (int a = 0; a < 3; ++a) {
              int next[3] = {node[0], node[1], node[2]};
              if (++next[a] >= n[a]) continue;
              if (grid.inside(next[0], next[1], next[2]) == in0) continue;
              const int u = (a + 1) % 3, v = (a + 2) % 3;
              // Cells around the edge, counter-clockwise in (u, v): the quad then faces +a.
              const int ring[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
              Index q[4];
              bool ok = true;
              for (int r = 0; r < 4 && ok; ++r) {
                int cell[3] = {node[0], node[1], node[2]};
                cell[u] -= ring[r][0];
                cell[v] -= ring[r][1];
                q[r] = cellVertex(cell[0], cell[1], cell[2], a, node);
                ok = q[r] != kInvalid;
              }
              if (!ok) continue;
              if (in0) {  // Inside to outside along +a: the surface faces +a.
                bd.quads.insert(bd.quads.end(), {q[0], q[1], q[2], q[3]});
              } else {
                bd.quads.insert(bd.quads.end(), {q[3], q[2], q[1], q[0]});
              }
            }
          }
    }
  });

  SurfaceNetsMesh out;
  out.positions.resize(static_cast<std::size_t>(vertexBase.back()));
  std::size_t quadCount = 0;
  for (const BlockData& bd : data) quadCount += bd.quads.size();
  out.quads.reserve(quadCount);
  for (std::size_t s = 0; s < blocks.size(); ++s) {
    std::copy(data[s].vertices.begin(), data[s].vertices.end(), out.positions.begin() + vertexBase[s]);
    out.quads.insert(out.quads.end(), data[s].quads.begin(), data[s].quads.end());
  }
  return out;
}

}  // namespace plegl
