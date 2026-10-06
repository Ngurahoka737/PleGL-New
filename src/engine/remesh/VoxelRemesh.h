#pragma once

#include <optional>
#include <string>

#include "mesh/Mesh.h"

namespace plegl {

struct VoxelRemeshParams {
  float voxelSize = 0.01f;  // Target edge length of the output, in mesh units.
  int maxResolution = 512;  // Refuse grids larger than this along any axis.
};

struct VoxelRemeshStats {
  int resolution[3] = {0, 0, 0};
  double gridMs = 0.0;     // Sign and narrow-band distance.
  double extractMs = 0.0;  // Surface Nets.
  double buildMs = 0.0;    // Half-edge topology.
  std::size_t gridBytes = 0;
  BuildReport report;
};

// Rebuilds a mesh from its volume: the input is turned into a signed distance field on a grid
// and the zero surface is extracted as quads. The result does not depend on the input topology,
// so stretched faces, uneven density, overlapping parts and self-intersections all come out as
// one even, all-quad surface. Holes in open meshes are mostly outvoted; badly open meshes can
// lose parts.
//
// Returns nothing (and sets `error`) when the mesh is empty, the grid would be too large, or no
// closed volume was found.
std::optional<Mesh> voxelRemesh(const Mesh& input, const VoxelRemeshParams& params,
                                VoxelRemeshStats* stats = nullptr, std::string* error = nullptr);

// Signed volume enclosed by a closed mesh (positive when faces point outward).
double meshVolume(const Mesh& mesh);

}  // namespace plegl
