#pragma once

#include <span>
#include <string>
#include <vector>

#include "core/Types.h"

namespace plegl {

// Polygon mesh stored as an index-based half-edge structure with struct-of-arrays attributes.
//
// Topology arrays hold 32-bit indices only, so a mesh can be copied, serialized or handed to
// another thread cheaply. Hot sculpting paths only touch `positions` and `normals`.
//
// Conventions:
//  - Half-edge h starts at heVert[h] and ends at heVert[heNext[h]].
//  - heTwin[h] == kInvalid marks an open (boundary) edge; there are no explicit boundary loops.
//  - Faces may be triangles, quads or n-gons, wound counter-clockwise when seen from outside.
//  - The half-edges of face f are contiguous in faceHe order only after reorder(); always walk
//    them through heNext.
class Mesh {
 public:
  // Vertex attributes.
  std::vector<Vec3> positions;
  std::vector<Vec3> normals;

  // Topology.
  std::vector<Index> vertHe;  // One outgoing half-edge per vertex (kInvalid if isolated).
  std::vector<Index> faceHe;  // One half-edge per face.
  std::vector<Index> heNext;
  std::vector<Index> heTwin;
  std::vector<Index> heVert;
  std::vector<Index> heFace;

  Index vertexCount() const { return static_cast<Index>(positions.size()); }
  Index faceCount() const { return static_cast<Index>(faceHe.size()); }
  Index halfEdgeCount() const { return static_cast<Index>(heNext.size()); }

  Index heTarget(Index h) const { return heVert[heNext[h]]; }
  Index hePrev(Index h) const;
  Index faceSize(Index f) const;
  bool isBoundaryVertex(Index v) const;
  int valence(Index v) const;

  // Calls fn(h) for every outgoing half-edge of v. Each incident face is visited exactly once.
  template <class Fn>
  void forEachOutgoing(Index v, Fn&& fn) const;

  // Calls fn(vertex) for every vertex of face f, in winding order.
  template <class Fn>
  void forEachFaceVertex(Index f, Fn&& fn) const {
    const Index start = faceHe[f];
    Index h = start;
    do {
      fn(heVert[h]);
      h = heNext[h];
    } while (h != start);
  }

  // Unnormalized face normal (Newell's method); its length is twice the polygon area.
  Vec3 faceAreaNormal(Index f) const;
  Vec3 faceCentroid(Index f) const;
  Aabb bounds() const;

  // Recomputes area-weighted vertex normals for every vertex (parallel).
  void computeNormals();
  // Recomputes normals for a contiguous range of vertices only.
  void computeNormals(Index vertBegin, Index vertEnd);

  // Number of unique undirected edges (boundary edges counted once).
  Index edgeCount() const;

  // Reorders faces into `faceOrder` (a permutation of all face indices) and renumbers vertices
  // by first use in that order, so faces that are adjacent in the new order also own adjacent
  // vertices. Half-edges are laid out face by face. Returns, for each new face index f, the
  // number of distinct vertices first used by faces before f (size faceCount() + 1).
  std::vector<Index> reorder(std::span<const Index> faceOrder);

  void clear();
};

struct BuildReport {
  Index nonManifoldEdges = 0;  // Directed edges used by more than one face.
  Index degenerateFaces = 0;   // Faces with fewer than 3 distinct vertices (skipped).
  Index isolatedVertices = 0;  // Vertices referenced by no face.
  Index nonManifoldVertices = 0;  // Vertices whose faces do not form a single fan.
  bool ok() const { return nonManifoldEdges == 0 && degenerateFaces == 0; }
};

// Builds half-edge topology from a polygon soup. `faceSizes[i]` vertices of face i are read in
// order from `faceIndices`. Normals are computed. Twins are matched through per-vertex
// adjacency, so the cost is linear in the number of half-edges.
Mesh buildMesh(std::vector<Vec3> positions, std::span<const Index> faceIndices,
               std::span<const Index> faceSizes, BuildReport* report = nullptr);

// Merges vertices closer than `epsilon` and rebuilds faces accordingly. Used by primitives that
// are generated face-by-face and by importers of unwelded files.
void weldVertices(std::vector<Vec3>& positions, std::vector<Index>& faceIndices, float epsilon);

struct ValidationResult {
  bool ok = true;
  std::string message;  // First problem found, empty when ok.
};

// Checks every topological invariant. Intended for debug builds and tests.
ValidationResult validate(const Mesh& mesh);

// ---------------------------------------------------------------------------------------------

template <class Fn>
void Mesh::forEachOutgoing(Index v, Fn&& fn) const {
  const Index start = vertHe[v];
  if (start == kInvalid) return;
  // Walk one way around the vertex: twin(prev(h)) is the next outgoing edge.
  Index h = start;
  for (;;) {
    fn(h);
    const Index t = heTwin[hePrev(h)];
    if (t == kInvalid) break;  // Hit the boundary; walk the other way below.
    h = t;
    if (h == start) return;    // Closed fan, done.
  }
  // Open fan: walk the other way from start: next(twin(h)).
  h = start;
  for (;;) {
    const Index t = heTwin[h];
    if (t == kInvalid) return;
    h = heNext[t];
    fn(h);
  }
}

}  // namespace plegl
