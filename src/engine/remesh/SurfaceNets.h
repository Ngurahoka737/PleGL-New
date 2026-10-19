#pragma once

#include <vector>

#include "remesh/VoxelGrid.h"

namespace plegl {

// Extracts the zero surface of a VoxelGrid as an all-quad mesh (Surface Nets): one vertex per
// surface patch inside each voxel cell, one quad per grid edge the surface crosses. Quads face
// outward.
//
// Plain Surface Nets puts one vertex in a cell even when two sheets of surface pass through it,
// which makes non-manifold edges where the sheets nearly touch. Here each cell gets one vertex
// per connected patch instead: crossing edges are grouped through the cell faces they share,
// and ambiguous faces (four crossings) are resolved from the face centre value. Both cells of a
// face resolve it identically, so every quad edge is shared by exactly two quads.
struct SurfaceNetsMesh {
  std::vector<Vec3> positions;
  std::vector<Index> quads;  // Four indices per quad.
};

SurfaceNetsMesh extractSurfaceNets(const VoxelGrid& grid);

}  // namespace plegl
