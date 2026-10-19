#pragma once

#include <vector>

#include "mesh/Mesh.h"

namespace plegl {

// Local topology edits on a half-edge mesh: the building blocks for remeshing and, later,
// dynamic topology.
//
// Edits never move other elements, so indices stay valid while editing. Removed elements are
// only marked dead; compact() drops them and renumbers everything once editing is done. Until
// then the mesh must not be handed to code that walks all faces or half-edges (BVH, renderer,
// validate()).
//
// Every edit checks that the result stays manifold and returns false (changing nothing) when it
// would not. Edits are refused on open (boundary) edges and at boundary vertices; remesh output
// is always closed. Geometric quality (fold-overs, angles) is the caller's decision.
class MeshEditor {
 public:
  explicit MeshEditor(Mesh& mesh);

  Mesh& mesh() { return m_; }
  bool faceAlive(Index f) const { return m_.faceHe[f] != kInvalid; }
  bool vertexAlive(Index v) const { return !deadVertex_[v]; }
  bool halfEdgeAlive(Index h) const { return m_.heFace[h] != kInvalid; }

  // Number of edges at vertex v (closed fans only).
  int valence(Index v) const;
  // True if an edge joins u and w.
  bool connected(Index u, Index w) const;

  // Rotates the edge of h one step forward inside its two faces: in face [a, b, c, ...] and twin
  // face [b, a, d, ...] the edge a-b becomes d-c. Face sizes stay the same, so on two triangles
  // this is the classic edge flip and on two quads it is the quad edge rotation. a and b lose
  // one edge each, c and d gain one. Refused if c-d already exists or a or b would drop below
  // valence 3.
  bool rotateEdge(Index h);

  // Inserts a vertex on the edge of h at parameter t (0 = start, 1 = end). Both faces gain a
  // vertex (a triangle becomes a quad). Returns the new vertex.
  Index splitEdge(Index h, float t = 0.5f);

  // Splits face f by a new edge between the start vertices of ha and hb (both in f, not
  // neighbours). Returns the new face, or kInvalid if refused (the edge already exists).
  Index splitFace(Index ha, Index hb);

  // Merges the end vertex of h into its start vertex, placed at `position`. Triangles on the
  // edge disappear; larger faces lose a corner. Refused when it would join two parts of the
  // surface (link condition), leave a vertex with fewer than 3 edges, or touch the boundary.
  bool collapseEdge(Index h, const Vec3& position);

  // Collapses the quad holding h by merging its corners heVert[h] and the one opposite it, at
  // their midpoint. The quad disappears and every other face stays a quad, which makes this the
  // quad mesh counterpart of an edge collapse. Refused for non-quads and under the same
  // conditions as collapseEdge().
  bool collapseDiagonal(Index h);

  // Drops dead elements, renumbers, and recomputes normals.
  void compact();

 private:
  void killHalfEdge(Index h);
  void collectOutgoing(Index v, std::vector<Index>& out) const;
  bool closedFan(Index v) const;

  Mesh& m_;
  std::vector<char> deadVertex_;
  mutable std::vector<Index> scratchA_, scratchB_;
};

}  // namespace plegl
