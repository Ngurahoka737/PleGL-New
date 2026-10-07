#pragma once

#include <optional>
#include <string>

#include "remesh/VoxelRemesh.h"

namespace plegl {

class Bvh;

struct QuadRemeshParams {
  float targetEdge = 0.01f;  // Target edge length of the output, in mesh units.
  int maxResolution = 512;   // Refuse voxel grids larger than this along any axis.
  int rounds = 3;            // Optimisation rounds: valence pass, then relaxation. 0 = plain voxel remesh.
  int relaxIterations = 3;   // Relaxation steps per round.
  bool optimizeValence = true;
  bool measureError = false;  // Also measure surface error in the stats (costs a pass of point queries).
};

// Shape of a quad mesh, and optionally how far it sits from a reference surface.
struct MeshQuality {
  double quadRatio = 0.0;      // Faces with 4 corners.
  double valence4Ratio = 0.0;  // Vertices with 4 edges.
  double meanEdge = 0.0;       // Mean edge length, in mesh units.
  double edgeLengthCv = 0.0;   // Standard deviation / mean of edge lengths. Lower is more even.
  double meanError = 0.0;      // Mean distance from face centres to the reference, in edges.
  double maxError = 0.0;       // Largest such distance, in edges.
};

struct QuadRemeshStats {
  VoxelRemeshStats voxel;
  MeshQuality raw;          // Straight out of the voxel remesh.
  MeshQuality optimized;    // After optimisation.
  int collapsed = 0;        // Quads removed by diagonal collapse.
  int rotated = 0;          // Edge rotations.
  double optimizeMs = 0.0;  // Everything after the voxel remesh, excluding the quality stats.
};

// Voxel remesh followed by quad optimisation:
//  1. voxelRemesh() gives a closed all-quad surface with uniform size.
//  2. Sliver quads, and quads whose diagonal collapse lowers valence irregularity (the 3-x-3-x
//     "diamond" pattern Surface Nets leaves at curved spots), are collapsed.
//  3. Edges are rotated wherever that brings the six vertices around them closer to valence 4.
//  4. Vertices are relaxed along the surface and projected back onto the input mesh, which
//     evens out edge lengths and removes the voxel staircase.
// The result stays closed, manifold and all quads.
std::optional<Mesh> quadRemesh(const Mesh& input, const QuadRemeshParams& params, QuadRemeshStats* stats = nullptr,
                               std::string* error = nullptr);

// Measures `mesh`. Errors are measured only when `reference` and its `referenceBvh` are given.
MeshQuality measureQuality(const Mesh& mesh, const Mesh* reference = nullptr, const Bvh* referenceBvh = nullptr);

}  // namespace plegl
