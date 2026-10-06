#pragma once

#include <limits>
#include <span>
#include <vector>

#include "mesh/Mesh.h"

namespace plegl {

// A BVH leaf is the engine's unit of work. Each leaf owns a contiguous range of faces and a
// contiguous range of vertices, so raycasts, brush queries, partial GPU uploads, undo snapshots
// and parallel jobs can all be expressed per leaf.
struct BvhLeaf {
  Index faceBegin = 0;
  Index faceEnd = 0;
  Index vertBegin = 0;  // Vertices first used by this leaf's faces.
  Index vertEnd = 0;
  Aabb bounds;          // Bounds of every face in the leaf (may include vertices owned elsewhere).
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

  // Nearest hit along the ray (both triangle sides), or false.
  bool raycast(const Mesh& mesh, const Ray& ray, RayHit& hit,
               float tMax = std::numeric_limits<float>::infinity()) const;

  // Appends every leaf whose bounds intersect the sphere.
  void querySphere(const Vec3& center, float radius, std::vector<Index>& outLeaves) const;

  // Recomputes all bounds after vertices moved (topology unchanged).
  void refit(const Mesh& mesh);
  // Recomputes bounds for the given leaves and their ancestors only.
  void refitLeaves(const Mesh& mesh, std::span<const Index> leaves);

  // Leaf owning vertex v, or kInvalid for isolated vertices.
  Index leafOfVertex(Index v) const;

  std::span<const BvhLeaf> leaves() const { return leaves_; }
  std::span<const BvhNode> nodes() const { return nodes_; }
  bool empty() const { return nodes_.empty(); }
  Aabb bounds() const { return nodes_.empty() ? Aabb{} : nodes_[0].bounds; }

 private:
  Aabb leafBounds(const Mesh& mesh, const BvhLeaf& leaf) const;

  std::vector<BvhNode> nodes_;    // Depth-first preorder: children always follow their parent.
  std::vector<BvhLeaf> leaves_;
  std::vector<Index> leafNode_;   // leaf index -> node index
  std::vector<Index> parent_;     // node index -> parent node index
};

}  // namespace plegl
