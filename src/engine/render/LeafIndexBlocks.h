#pragma once

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "mesh/Mesh.h"
#include "spatial/Bvh.h"

namespace plegl {

// Draw indices of one BVH leaf, appended to `out`. Both skip faces and half-edges removed by a
// dynamic topology stroke, so they work on a mesh in the middle of one, and hidden faces.
//
// Triangles: every visible face of the leaf, fan-triangulated from its first corner.
void appendLeafTriangles(const Mesh& mesh, const BvhLeaf& leaf, std::vector<std::uint32_t>& out);
// The face set value of every triangle appendLeafTriangles() emits, in the same order.
void appendLeafTriangleFaceSets(const Mesh& mesh, const BvhLeaf& leaf, std::vector<std::int32_t>& out);
// Edges: every half-edge of the leaf with a visible face that is the lower of its pair, on an
// open border, or next to a hidden face, as a line. Quads draw as quads. Every edge of the mesh
// with a visible face appears in exactly one leaf.
void appendLeafEdges(const Mesh& mesh, const BvhLeaf& leaf, std::vector<std::uint32_t>& out);

enum class LeafIndexKind : std::uint8_t { Triangles, Edges };

void appendLeafIndices(LeafIndexKind kind, const Mesh& mesh, const BvhLeaf& leaf, std::vector<std::uint32_t>& out);

class LeafIndexBlocks;

// Every leaf's indices back to back in leaf order, generated in parallel, with `blocks` packed to
// match. On a compact mesh this is the same list a single walk over all faces (or half-edges)
// gives, since leaves own contiguous ranges.
std::vector<std::uint32_t> buildLeafIndices(LeafIndexKind kind, const Mesh& mesh, const Bvh& bvh,
                                            LeafIndexBlocks& blocks);

// The face set of every triangle slot (index / 3) of a triangle buffer laid out by `blocks`, which
// must match the mesh (as buildLeafIndices() left them, or kept current with place()). Slots no
// leaf uses hold kDefaultFaceSet.
std::vector<std::int32_t> buildTriangleFaceSets(const Mesh& mesh, const Bvh& bvh, const LeafIndexBlocks& blocks);

// Where each leaf's indices live inside one index buffer. After pack() the blocks sit back to
// back in leaf order, exactly like one plain upload of the whole mesh. While a dynamic topology
// stroke runs, place() updates single leaves: a block that still fits stays where it is, a block
// that outgrows its room moves to a free range or to the end, with slack so a growing leaf does
// not move on every dab. The buffer itself is the caller's; this class only does the arithmetic.
class LeafIndexBlocks {
 public:
  // Every block starts at, and reserves room in, multiples of `granularity` indices. Triangle
  // blocks use 3, so index / 3 numbers a triangle slot that stays put while its block does.
  LeafIndexBlocks() = default;
  explicit LeafIndexBlocks(std::uint32_t granularity) : granularity_(granularity) {}

  struct Block {
    std::uint32_t first = 0;     // In indices, not bytes.
    std::uint32_t capacity = 0;  // Room reserved for the leaf.
    std::uint32_t count = 0;     // Indices in use.
  };

  // One block per leaf, sized exactly (rounded up to the granularity), in leaf order.
  void pack(std::span<const std::uint32_t> counts);
  // Gives `leaf` room for `count` indices; leaves past the current count are added empty.
  // Returns true if the block moved (or is new), false if it kept its place. Either way the
  // caller writes the leaf's whole index list at block(leaf).first.
  bool place(Index leaf, std::uint32_t count);
  // Drops every leaf from `leafCount` on and frees their room.
  void truncate(std::size_t leafCount);

  std::size_t leafCount() const { return blocks_.size(); }
  const Block& block(Index leaf) const { return blocks_[static_cast<std::size_t>(leaf)]; }
  // One past the furthest index any block may use: the size the buffer must have.
  std::uint32_t size() const { return end_; }
  // Indices in use, summed over all blocks.
  std::uint64_t used() const;

  // The used part of every block as (first, count) runs in buffer order, with touching runs
  // merged: after pack() that is a single run. For one multi-draw call.
  void runs(std::vector<std::uint32_t>& first, std::vector<std::uint32_t>& count) const;

  // Blocks lie inside size() and overlap neither each other nor the free list. For tests.
  bool consistent() const;

 private:
  void release(std::uint32_t first, std::uint32_t size);
  std::uint32_t allocate(std::uint32_t size);

  std::vector<Block> blocks_;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> free_;  // (first, size), sorted, coalesced.
  std::uint32_t end_ = 0;
  std::uint32_t granularity_ = 1;
};

}  // namespace plegl
