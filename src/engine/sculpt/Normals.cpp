#include "sculpt/Normals.h"

#include <algorithm>
#include <atomic>

#include "core/Parallel.h"

namespace plegl {

void recomputeVertexNormals(Mesh& m, std::span<const Index> verts, NormalWorkspace& ws, bool allowWhole) {
  if (verts.empty()) return;
  if (allowWhole && verts.size() * 2 > m.positions.size()) {
    m.computeNormals(ws.faceN);
    return;
  }
  const std::size_t nf = m.faceHe.size();
  if (ws.faceN.size() < nf) ws.faceN.resize(nf);
  if (ws.faceClaim.size() < nf) ws.faceClaim.resize(nf, 0);
  if (++ws.stamp == 0) {
    std::fill(ws.faceClaim.begin(), ws.faceClaim.end(), 0);
    ws.stamp = 1;
  }
  const std::uint32_t stamp = ws.stamp;
  // Pass 1: whoever claims a face first writes its normal. Values are only read after the join.
  parallelFor(0, verts.size(), 2048, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      m.forEachOutgoing(verts[i], [&](Index h) {
        const Index f = m.heFace[h];
        if (std::atomic_ref<std::uint32_t>(ws.faceClaim[f]).exchange(stamp, std::memory_order_relaxed) != stamp)
          ws.faceN[f] = m.faceAreaNormal(f);
      });
    }
  });
  // Pass 2: the sums in Mesh::vertexNormal's order.
  parallelFor(0, verts.size(), 2048, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const Index v = verts[i];
      Vec3 n{0.0f};
      m.forEachOutgoing(v, [&](Index h) { n += ws.faceN[m.heFace[h]]; });
      m.normals[v] = safeNormalize(n);
    }
  });
}

}  // namespace plegl
