#pragma once

#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "mesh/Mesh.h"

namespace plegl {

// Leaf flags.
inline constexpr std::uint8_t kLeafOwnsOrphans = 1;  // Owns vertices none of its faces use.
inline constexpr std::uint8_t kLeafMayHaveDead = 2;  // Holds removed elements (dynamic topology only).

// A BVH leaf is the engine's unit of work. Each leaf owns a contiguous range of faces, of their
// half-edges and of vertices, so raycasts, brush queries, partial GPU uploads, undo snapshots and
// parallel jobs can all be expressed per leaf. Leaves are sorted by these ranges; a leaf can be
// empty (all ranges empty), in which case it has no node in the tree.
struct BvhLeaf {
  Index faceBegin = 0;
  Index faceEnd = 0;
  Index vertBegin = 0;  // Vertices first used by this leaf's faces.
  Index vertEnd = 0;
  Index heBegin = 0;    // Half-edges of the leaf's faces, laid out face by face.
  Index heEnd = 0;
  std::uint8_t flags = 0;
  Aabb bounds;          // Bounds of every face in the leaf (may include vertices owned elsewhere).
  bool empty() const { return faceBegin == faceEnd && vertBegin == vertEnd; }
};

struct BvhNode {
  Aabb bounds;
  Index left = kInvalid;   // Child node indices, kInvalid for leaves.
  Index right = kInvalid;
  Index leaf = kInvalid;   // Index into leaves() for leaf nodes.
  bool isLeaf() const { return leaf != kInvalid; }
};

struct RayHit {
  float t = std::numeric_limits<float>::infinity();  // In units of the ray direction.
  Index face = kInvalid;
  Index leaf = kInvalid;
  Vec3 position{0.0f};
  Vec3 faceNormal{0.0f};    // Geometric normal of the hit triangle, normalized.
  Vec3 smoothNormal{0.0f};  // Interpolated vertex normal, normalized.
};

class Bvh {
 public:
  struct Params {
    int maxLeafFaces = 1024;
  };

  // Builds the hierarchy and REORDERS the mesh (faces, half-edges and vertices) so that every
  // leaf owns contiguous index ranges. Any external index into the mesh is invalidated.
  void build(Mesh& mesh, const Params& params);
  void build(Mesh& mesh) { build(mesh, Params{}); }

  // Nearest hit along the ray (both triangle sides), or false. `visibleOnly` skips hidden faces
  // (see Mesh::faceSets): picking and brushes pass it, remeshing does not.
  bool raycast(const Mesh& mesh, const Ray& ray, RayHit& hit, float tMax = std::numeric_limits<float>::infinity(),
               bool visibleOnly = false) const;

  // Nearest surface point to p within maxDist. Returns false if there is none.
  struct ClosestHit {
    Vec3 position{0.0f};
    Index face = kInvalid;
    float distSq = 0.0f;
    Vec3 faceNormal{0.0f};  // Normalized geometric normal of the closest face.
    // Corners of the closest triangle (faces are fanned from their first half-edge), for
    // interpolating vertex attributes at `position`.
    Index corners[3] = {kInvalid, kInvalid, kInvalid};
  };
  bool closestPoint(const Mesh& mesh, const Vec3& p, float maxDist, ClosestHit& out, bool visibleOnly = false) const;

  // Appends every leaf whose bounds intersect the sphere.
  void querySphere(const Vec3& center, float radius, std::vector<Index>& outLeaves) const;

  // Recomputes all bounds after vertices moved (topology unchanged).
  void refit(const Mesh& mesh);
  // Recomputes bounds for the given leaves and their ancestors only.
  void refitLeaves(const Mesh& mesh, std::span<const Index> leaves);

  // Leaf owning vertex v, or kInvalid for isolated vertices.
  Index leafOfVertex(Index v) const;
  // Leaf owning face f, or kInvalid.
  Index leafOfFace(Index f) const;
  // Leaf owning half-edge h, or kInvalid.
  Index leafOfHalfEdge(Index h) const;

  // --- Dynamic topology -------------------------------------------------------------------
  // During a dynamic topology stroke the mesh keeps removed elements in place and appends new
  // ones. The existing leaves keep their ranges; appended elements are owned by "tail" leaves
  // added after them, which have no tree node and are scanned linearly by every query.
  void beginDynamic(const Mesh& mesh);
  // Extends the open tail leaf over everything appended since, closing it at kTailLeafFaces.
  void growTail(const Mesh& mesh);
  void endDynamic() { dynamic_ = false; }
  bool dynamic() const { return dynamic_; }
  bool isTail(Index leaf) const { return dynamic_ && leaf >= firstTailLeaf_; }
  Index firstTailLeaf() const { return dynamic_ ? firstTailLeaf_ : static_cast<Index>(leaves_.size()); }
  Index tailStartFace() const { return tailStartFace_; }
  Index tailStartVertex() const { return tailStartVert_; }
  Index tailStartHalfEdge() const { return tailStartHe_; }
  void addLeafFlags(Index leaf, std::uint8_t flags) { leaves_[leaf].flags |= flags; }
  static constexpr Index kTailLeafFaces = 1020;

  // Replaces the leaves (sorted by range, empty ones allowed) and builds a new tree over them.
  // Leaf indices are kept: this is how dynamic topology folds edits back into the hierarchy.
  void setLeaves(std::vector<BvhLeaf> leaves);
  // Rebuilds the tree over the current leaves (median split on leaf centres), keeping leaf indices.
  void rebuildTree();

  std::span<const BvhLeaf> leaves() const { return leaves_; }
  // Leaf size the hierarchy was built with; dynamic topology keeps new leaves within it.
  Index maxLeafFaces() const { return maxLeafFaces_; }
  std::size_t memoryBytes() const;
  std::span<const BvhNode> nodes() const { return nodes_; }
  bool empty() const { return nodes_.empty(); }
  Aabb bounds() const { return nodes_.empty() ? Aabb{} : nodes_[0].bounds; }

 private:
  Aabb leafBounds(const Mesh& mesh, const BvhLeaf& leaf) const;
  template <typename Fn>
  void forEachTailLeaf(Fn&& fn) const {
    if (!dynamic_) return;
    for (Index l = firstTailLeaf_; l < static_cast<Index>(leaves_.size()); ++l) fn(l);
  }

  std::vector<BvhNode> nodes_;    // Depth-first preorder: children always follow their parent.
  std::vector<BvhLeaf> leaves_;
  std::vector<Index> leafNode_;   // leaf index -> node index (kInvalid for empty and tail leaves)
  std::vector<Index> parent_;     // node index -> parent node index
  Index maxLeafFaces_ = 1024;
  bool dynamic_ = false;
  Index firstTailLeaf_ = 0;
  Index tailStartFace_ = 0, tailStartVert_ = 0, tailStartHe_ = 0;
};

// Splits `order` (face indices) into groups of at most maxLeafFaces faces by recursive median
// splits on the longest axis of the face centroids, as Bvh::build does. Reorders `order` and
// returns the group boundaries (first entry 0, last entry order.size()).
std::vector<Index> partitionFaces(std::span<const Vec3> centroids, std::vector<Index>& order, Index maxLeafFaces);

// Checks the at-rest layout: a valid mesh without removed elements whose leaves partition faces,
// half-edges (laid out face by face) and face-using vertices into contiguous ranges in leaf order,
// with isolated vertices last; every owned vertex used by a face of its leaf (unless the leaf owns
// orphans); bounds that contain their leaves; and a tree that reaches every non-empty leaf once.
ValidationResult validateLayout(const Mesh& mesh, const Bvh& bvh);

}  // namespace plegl
