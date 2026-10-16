#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "mesh/Mesh.h"

namespace plegl {

// Scratch for recomputeVertexNormals: one face normal and one claim stamp per face.
struct NormalWorkspace {
  std::vector<std::uint32_t> faceClaim;
  std::uint32_t stamp = 0;
  std::vector<Vec3> faceN;
  std::size_t bytes() const { return faceClaim.capacity() * sizeof(std::uint32_t) + faceN.capacity() * sizeof(Vec3); }
};

// Recomputes the normals of `verts` (no repeats) bit for bit as Mesh::vertexNormal does. Each
// incident face normal is computed once: the first vertex to reach a face claims it and writes
// its normal, then each vertex sums its faces in Mesh::vertexNormal's order. With `allowWhole`,
// sets covering more than half the mesh recompute every normal instead (Mesh::computeNormals);
// pass false while the mesh may hold dead faces.
void recomputeVertexNormals(Mesh& m, std::span<const Index> verts, NormalWorkspace& ws, bool allowWhole);

}  // namespace plegl
