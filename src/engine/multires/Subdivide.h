#pragma once

#include <optional>
#include <span>
#include <string>
#include <vector>

#include "multires/Multires.h"

namespace plegl {

// Catmull-Clark rule of every vertex of a base mesh:
//  - Pinned: isolated, non-manifold (its faces form more than one fan), the corner of an open fan
//    with one face, or a closed fan with fewer than 3 faces.
//  - Boundary: an open fan with at least 2 faces (non-manifold edges are open after buildMesh, so
//    they act as borders).
//  - Smooth: everything else.
// Subdivision keeps these classes, so this also gives the rules of every finer level.
std::vector<VertexRule> classifyVertices(const Mesh& mesh);

struct SubdivideOptions {
  Index maxFaces = kMaxMultiresFaces;  // Refuse when the new level would have more faces.
  int maxLeafFaces = 1024;             // BVH leaf size of the new level.
  // Build the new level's BVH (which reorders it). Without, the result keeps the provisional
  // numbering below and has no BVH; for tests.
  bool buildBvh = true;
};

struct SubdivisionResult {
  Mesh mesh;
  Bvh bvh;
  CanonicalMap canon;
  SubdivisionLinks links;
  std::vector<VertexRule> rule;
};

// One Catmull-Clark step. The topology is closed-form: the child quad of parent half-edge h has
// the corners [vertex child of h's vertex, edge child of h, face child of h's face, edge child of
// prev(h)]; twins and vertHe follow from the parent's. Positions use the stencils below, mask is
// interpolated (vertex children copy, edge children average 2, face children average the corners)
// and face sets are inherited (hidden faces stay hidden). Every value is gathered by exactly one
// task in a fixed order, so the result is the same bits at any thread count, and the same bits
// for any live order of the parent with the same canonical structure.
//
// `parentCanon` is the parent's canonical map (identityCanonicalMap for a new base); `parentRule`
// its vertex rules. The parent must be compact (no removed elements). Returns nothing and sets
// `error` when the mesh is empty or the result would exceed options.maxFaces.
std::optional<SubdivisionResult> subdivide(const Mesh& parent, const CanonicalMap& parentCanon,
                                           std::span<const VertexRule> parentRule,
                                           const SubdivideOptions& options = {}, std::string* error = nullptr);

// --- Stencils -------------------------------------------------------------------------------
// Shared by subdivide() and by propagation, which evaluates them on other coarse positions.
// `X(v)` returns the coarse position of vertex v; `FP(f)` the face point of face f (ccFacePoint
// or a cached copy of it).

template <class Pos>
Vec3 ccFacePoint(const Mesh& m, Index f, Pos&& X) {
  // Same operations in the same order as Mesh::faceCentroid.
  Vec3 c{0.0f};
  Index n = 0;
  m.forEachFaceVertex(f, [&](Index v) {
    c += X(v);
    ++n;
  });
  return c / static_cast<float>(n);
}

// Edge point of the edge of half-edge h. The sum is grouped so both half-edges give the same bits.
template <class Pos, class FacePoint>
Vec3 ccEdgePoint(const Mesh& m, Index h, Pos&& X, FacePoint&& FP) {
  const Vec3 ab = X(m.heVert[h]) + X(m.heTarget(h));
  const Index t = m.heTwin[h];
  if (t == kInvalid) return ab * 0.5f;
  return (ab + (FP(m.heFace[h]) + FP(m.heFace[t]))) * 0.25f;
}

// The two border neighbours of a Boundary vertex: the end of its outgoing open half-edge and the
// start of its incoming one.
inline void borderNeighbours(const Mesh& m, Index v, Index& a, Index& b) {
  a = b = v;
  m.forEachOutgoing(v, [&](Index h) {
    if (m.heTwin[h] == kInvalid) a = m.heTarget(h);
    const Index prev = m.hePrev(h);
    if (m.heTwin[prev] == kInvalid) b = m.heVert[prev];
  });
}

template <class Pos, class FacePoint>
Vec3 ccVertexPoint(const Mesh& m, Index v, VertexRule rule, Pos&& X, FacePoint&& FP) {
  if (rule == VertexRule::Pinned) return X(v);
  if (rule == VertexRule::Boundary) {
    Index a, b;
    borderNeighbours(m, v, a, b);
    return X(v) * 0.75f + (X(a) + X(b)) * 0.125f;
  }
  Vec3 sumU{0.0f}, sumF{0.0f};
  int n = 0;
  m.forEachOutgoing(v, [&](Index h) {
    sumU += X(m.heTarget(h));
    sumF += FP(m.heFace[h]);
    ++n;
  });
  const float fn = static_cast<float>(n);
  return X(v) * ((fn - 2.0f) / fn) + (sumU + sumF) * (1.0f / (fn * fn));
}

}  // namespace plegl
