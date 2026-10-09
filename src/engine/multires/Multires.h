#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "mesh/Mesh.h"
#include "spatial/Bvh.h"

namespace plegl {

// Multiresolution: an object can carry a stack of Catmull-Clark subdivision levels. Level 0 is the
// base mesh; level k+1 subdivides level k once, so every face of level k becomes one quad per
// corner. The artist sculpts on one level at a time (the active level); the others follow when
// the active level changes.
//
// Every level is a complete Mesh with absolute positions, normals, mask and face sets, in its own
// BVH order. A level's BVH is built once, when the level is created or loaded, and afterwards only
// refit, so its leaf layout and topology version stay the same for the level's whole life. That
// keeps sculpt undo entries (matched by object id and topology version, copied by leaf range)
// valid on every level. The active level lives in SceneObject::mesh and bvh, so brushes, picking,
// the renderer and undo need no changes; the other levels are parked in the stack.
//
// Edits on the active level are found by comparing it bitwise with `reference`, a copy of the
// active level taken at its last sync. A sync runs only when the active level changes (or a
// command needs every level current): changes travel up to finer levels with the detail carried
// in a local frame, and down to coarser levels through a smoothing restriction. Values no edit
// can affect are never rewritten, so stepping between levels without editing changes nothing.

inline constexpr Index kMaxMultiresFaces = 8'388'608;  // Per level.
inline constexpr int kMaxMultiresLevels = 8;           // L0 to L7.

// How Catmull-Clark moves a vertex (see classifyVertices).
enum class VertexRule : std::uint8_t {
  Smooth,    // Interior vertex: the usual Catmull-Clark vertex rule.
  Boundary,  // On an open border: the cubic B-spline rule along the border.
  Pinned,    // Isolated, non-manifold or a corner: never moves.
};

// Live index -> canonical index. The canonical numbering does not depend on BVH order: it comes
// from level 0's polygon list and the subdivision rules alone, so it is what files store. On
// level k+1, vertex children come first (canonical id of their parent vertex), then edge children
// (number of level-k vertices plus the rank of the edge's owner half-edge in canonical half-edge
// order), then face children (vertices plus edges plus the parent face's canonical id). The child
// quad of parent half-edge h has the canonical id of h and starts at its vertex child.
//
// The canonical half-edges of a face are numbered from faceStart in corner order from its live
// faceHe (Mesh::reorder keeps that start corner).
struct CanonicalMap {
  std::vector<Index> vert;
  std::vector<Index> face;
  // Level 0 only: first canonical half-edge of every canonical face plus the total (size F + 1).
  // Empty on levels made of quads only, where face c starts at 4c.
  std::vector<Index> faceStart;
  Index firstHalfEdge(Index canonicalFace) const {
    return faceStart.empty() ? 4 * canonicalFace : faceStart[static_cast<std::size_t>(canonicalFace)];
  }
};

// Live indices connecting level k-1 (coarse) with level k (fine). Both layouts are frozen for the
// levels' lifetime, so these stay valid.
struct SubdivisionLinks {
  std::vector<Index> vertexChild;     // Per coarse vertex.
  std::vector<Index> edgeChild;       // Per coarse half-edge; both halves of an edge agree.
  std::vector<Index> faceChild;       // Per coarse face.
  std::vector<Index> childFace;       // Per coarse half-edge: the fine quad at its start corner.
  std::vector<std::uint32_t> parent;  // Per fine vertex: kind << 30 | coarse index (see below).
  std::vector<Index> parentHalfEdge;  // Per fine face: the coarse half-edge it was made from.
  std::size_t bytes() const;
  void clear() { *this = SubdivisionLinks{}; }
};
// Kinds in SubdivisionLinks::parent: a vertex child (index of the coarse vertex), an edge child
// (index of a coarse half-edge of the edge) or a face child (index of the coarse face).
inline constexpr std::uint32_t kParentVertex = 0, kParentEdge = 1, kParentFace = 2;
inline std::uint32_t parentKind(std::uint32_t p) { return p >> 30; }
inline Index parentIndex(std::uint32_t p) { return static_cast<Index>(p & 0x3FFFFFFFu); }
inline std::uint32_t makeParent(std::uint32_t kind, Index index) {
  return kind << 30 | static_cast<std::uint32_t>(index);
}

struct MultiresLevel {
  Mesh mesh;  // Empty while the level is active (the object holds it).
  Bvh bvh;
  std::uint64_t version = 0;     // This level's topology version, fixed for its life.
  std::uint64_t layoutHash = 0;  // layoutHash(bvh) when the level was created.
  bool boundsStale = false;      // Parked positions changed: refit when the level becomes active.
  std::vector<VertexRule> rule;  // Per live vertex.
  CanonicalMap canon;
  SubdivisionLinks links;        // To the level below; empty on level 0.
  std::size_t bytes() const;
};

// The active level as it was at its last sync, in its live order. An empty channel means the
// level had no mask (all 0) or no face sets (all default).
struct LevelReference {
  std::vector<Vec3> positions;
  std::vector<float> mask;
  std::vector<std::int32_t> faceSets;
  std::size_t bytes() const;
};

// Sparse value changes on one level, in that level's live indices, ascending.
struct LevelDelta {
  int level = 0;
  std::vector<Index> posIndex;
  std::vector<Vec3> posBefore, posAfter;
  std::vector<Index> maskIndex;
  std::vector<float> maskBefore, maskAfter;
  std::vector<Index> setIndex;
  std::vector<std::int32_t> setBefore, setAfter;
  bool maskCreated = false;  // The level had no mask before (it is dropped again on `before`).
  bool setsCreated = false;  // Likewise for face sets.
  bool empty() const { return posIndex.empty() && maskIndex.empty() && setIndex.empty(); }
  std::size_t bytes() const;
};

// Everything one sync changes. Entries of `sourceLevel` (the active level when it was computed)
// are reference values: applying the sync moves the reference to the live values, undoing it moves
// the reference back, so the edits are pending again. Entries of other levels are parked values.
struct SyncDelta {
  int sourceLevel = -1;
  std::vector<LevelDelta> levels;
  bool empty() const;
  std::size_t bytes() const;
};

struct Multires {
  std::vector<MultiresLevel> levels;  // [0] is the base. A stack always has at least 2 levels.
  int active = 0;
  LevelReference reference;
  // At least every face set id synced into any level, so a new id is unique on all levels.
  std::int32_t faceSetIdBound = kDefaultFaceSet;
  std::uint64_t serial = 0;  // Bumped by every change to the stack, so background work can tell.

  int levelCount() const { return static_cast<int>(levels.size()); }
  int top() const { return levelCount() - 1; }
  // Deep copy. With freshVersions every level gets a new topology version (for a duplicate
  // object); without, the copy is interchangeable with the original (for undo).
  std::unique_ptr<Multires> clone(bool freshVersions) const;
  std::size_t bytes() const;
};

// FNV-1a over the leaf ranges: changes whenever a level's BVH is rebuilt with another layout.
std::uint64_t layoutHash(const Bvh& bvh);

// Canonical map of a base mesh whose live order is its canonical order (level 0 of a new stack).
CanonicalMap identityCanonicalMap(const Mesh& mesh);
// Canonical id of every live half-edge (faces walked from faceHe).
std::vector<Index> canonicalHalfEdges(const Mesh& mesh, const CanonicalMap& canon);
// The level's polygons in canonical face order: the size of every face and its corners as
// canonical vertex ids, from the start corner.
void canonicalPolygons(const Mesh& mesh, const CanonicalMap& canon, std::vector<std::uint32_t>& sizes,
                       std::vector<std::uint32_t>& corners);
// Inverse of a live -> canonical map (canonical -> live).
std::vector<Index> invertMap(const std::vector<Index>& map);

}  // namespace plegl
