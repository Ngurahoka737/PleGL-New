#pragma once

#include <optional>
#include <string>

#include "remesh/VoxelRemesh.h"

namespace plegl {

class Bvh;

struct QuadRemeshParams {
  float targetEdge = 0.01f;  // Target edge length of the output, in mesh units.
  int maxResolution = 512;   // Refuse voxel grids larger than this along any axis.
  // The quad layout is built at targetEdge * 2^subdivisions and then split into four per level.
  // Subdividing keeps every irregular vertex but adds only valence-4 ones, so each level cuts
  // the share of irregular vertices by about four. Volume details thinner than the coarse edge
  // can be lost by the voxel step; the surface itself is still fitted at the target edge.
  int subdivisions = 1;
  int rounds = 3;           // Optimisation rounds: valence passes, then relaxation. 0 = plain voxel remesh.
  int driftPasses = 5;      // Per round: passes that move leftover 3-5 pairs together so they cancel.
  int relaxIterations = 3;  // Relaxation steps per round and after each subdivision.
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
//  1. voxelRemesh() at the coarse edge (see `subdivisions`) gives a closed all-quad surface.
//  2. Sliver quads, and quads whose diagonal collapse lowers valence irregularity (the 3-x-3-x
//     "diamond" pattern Surface Nets leaves at curved spots), are collapsed.
//  3. Edges are rotated wherever that brings the six vertices around them closer to valence 4.
//  4. Drift passes move the leftover 3-5 pairs towards each other so later passes cancel them.
//  5. Vertices are relaxed along the surface and projected back onto the input mesh, which
//     evens out edge lengths and removes the voxel staircase.
//  6. The mesh is subdivided down to the target edge and relaxed again.
//  7. If the input has a mask, it is transferred to the result by closest point.
// The result stays closed, manifold and all quads.
std::optional<Mesh> quadRemesh(const Mesh& input, const QuadRemeshParams& params, QuadRemeshStats* stats = nullptr,
                               std::string* error = nullptr);

// Measures `mesh`. Errors are measured only when `reference` and its `referenceBvh` are given.
MeshQuality measureQuality(const Mesh& mesh, const Mesh* reference = nullptr, const Bvh* referenceBvh = nullptr);

// Gives every vertex of `target` the mask value of the closest point on `source` (interpolated
// across that triangle). Leaves target.mask empty when nothing in `source` is masked. quadRemesh
// calls this itself, so a remesh keeps the mask.
void transferMask(const Mesh& source, const Bvh& sourceBvh, Mesh& target);

}  // namespace plegl
