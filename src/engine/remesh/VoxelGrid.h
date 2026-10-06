#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "mesh/Mesh.h"

namespace plegl {

// A signed distance field sampled on the nodes of a regular grid, built from any polygon mesh.
//
// The sign (inside or outside) is exact at every node: rays along each grid axis count the
// signed crossings of the surface (a winding number), and a node is inside when at least two of
// the three axes say so. Overlapping shells and self-intersections therefore become their
// union, and a few holes in an open mesh are outvoted instead of leaking.
//
// The distance magnitude is only stored in a narrow band around the surface, in 8x8x8 blocks,
// so memory follows the surface area rather than the volume. Outside the band the field reads
// as +/- band.
class VoxelGrid {
 public:
  struct Params {
    float voxelSize = 0.01f;   // Grid spacing in mesh units.
    int maxResolution = 512;   // Largest node count allowed along any axis.
  };

  static constexpr int kBlock = 8;
  static constexpr float kBand = 2.0f;  // Narrow band half-width, in voxels.

  // Returns false (with `error`) if the mesh is empty or the grid would be too large.
  bool build(const Mesh& mesh, const Params& params, std::string* error = nullptr);

  // Node counts along each axis and the world position of node (0, 0, 0).
  int nx() const { return n_[0]; }
  int ny() const { return n_[1]; }
  int nz() const { return n_[2]; }
  Vec3 origin() const { return origin_; }
  float voxelSize() const { return h_; }
  Vec3 nodePosition(int i, int j, int k) const { return origin_ + Vec3(i, j, k) * h_; }

  bool inside(int i, int j, int k) const { return votes_[nodeIndex(i, j, k)] >= 2; }
  // Signed distance in voxels (negative inside), clamped to +/- kBand away from the surface.
  float distance(int i, int j, int k) const;
  // True if the block holding node (i, j, k) stores distances (the node is near the surface).
  bool nearSurface(int i, int j, int k) const { return slotOf(i, j, k) >= 0; }

  // Slot of the block holding node (i, j, k), or -1 if that block is not allocated. Slots index
  // blocks(), so callers can keep their own per-block data.
  int slotOf(int i, int j, int k) const {
    return blockTable_[static_cast<std::size_t>(i / kBlock) +
                       static_cast<std::size_t>(b_[0]) * (j / kBlock + static_cast<std::size_t>(b_[1]) * (k / kBlock))];
  }

  // Allocated blocks, as block coordinates, in slot order; every node closer than kBand voxels
  // to the surface lies in one of them.
  const std::vector<std::array<int, 3>>& blocks() const { return blockCoords_; }
  std::size_t memoryBytes() const;

 private:
  std::size_t nodeIndex(int i, int j, int k) const {
    return static_cast<std::size_t>(i) + static_cast<std::size_t>(n_[0]) * (static_cast<std::size_t>(j) +
                                                                          static_cast<std::size_t>(n_[1]) * k);
  }
  void computeSigns(const std::vector<Vec3>& g, const std::vector<std::array<Index, 3>>& tris);
  void computeDistances(const std::vector<Vec3>& g, const std::vector<std::array<Index, 3>>& tris);

  int n_[3] = {0, 0, 0};
  int b_[3] = {0, 0, 0};
  Vec3 origin_{0.0f};
  float h_ = 1.0f;
  std::vector<std::uint8_t> votes_;      // Per node: how many axes say "inside" (0..3).
  std::vector<int> blockTable_;          // Per block: slot in distSq_, or -1.
  std::vector<std::array<int, 3>> blockCoords_;
  std::vector<float> distSq_;            // kBlock^3 squared distances (voxels^2) per slot.
};

}  // namespace plegl
