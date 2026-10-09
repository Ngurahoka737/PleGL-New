#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/Types.h"

namespace plegl {

// Position written to vertices removed during a dynamic topology stroke. It is finite, so brush
// distance tests (dist2 >= r2) reject it without creating inf or NaN, but far from any surface.
inline constexpr Vec3 kDeadPosition{1e18f, 1e18f, 1e18f};

// Face sets: every face belongs to one numbered group, drawn in its own colour and used to limit
// brushes and to hide parts of the mesh. A stored value v names set |v|; a negative value marks
// the face hidden. 0 is never stored. Faces of a mesh without the array are in kDefaultFaceSet
// and visible; that set is drawn without colour. Ids stay at or below kMaxFaceSetId, so a new id
// (the largest plus one) never overflows and INT32_MAX stays free for sentinels.
inline constexpr std::int32_t kDefaultFaceSet = 1;
inline constexpr std::int32_t kMaxFaceSetId = INT32_MAX - 1;
inline std::int32_t faceSetId(std::int32_t value) { return value < 0 ? -value : value; }
inline bool faceSetHidden(std::int32_t value) { return value < 0; }
// True for a value that may be stored: a set id in [1, kMaxFaceSetId], negated when hidden.
inline bool validFaceSetValue(std::int32_t value) {
  return value != 0 && value >= -kMaxFaceSetId && value <= kMaxFaceSetId;
}

// Old index of every face and vertex after Mesh::reorder() (new -> old).
struct ReorderMap {
  std::vector<Index> faceOld;
  std::vector<Index> vertOld;
};

// Polygon mesh stored as an index-based half-edge structure with struct-of-arrays attributes.
//
// Topology arrays hold 32-bit indices only, so a mesh can be copied, serialized or handed to
// another thread cheaply. Hot sculpting paths only touch `positions`, `normals` and `mask`;
// `faceSets` is read only when a stroke is limited to a face set or part of the mesh is hidden.
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
  // Sculpt mask in [0, 1]; 1 means fully protected from brushes. Empty means nothing is masked,
  // otherwise it has one value per vertex. Topology builders leave it empty.
  std::vector<float> mask;
  // Face attribute: the face set value of every face (see kDefaultFaceSet), or empty when every
  // face is in the default set and visible. Topology builders leave it empty.
  std::vector<std::int32_t> faceSets;

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

  // Allocates the mask (all zero) if it is empty.
  void ensureMask() {
    if (mask.empty()) mask.assign(positions.size(), 0.0f);
  }
  // True if any vertex has a non-zero mask value.
  bool anyMasked() const;

  // Allocates the face sets (all kDefaultFaceSet, visible) if they are empty.
  void ensureFaceSets() {
    if (faceSets.empty()) faceSets.assign(faceHe.size(), kDefaultFaceSet);
  }
  // Stored value of face f (negative when hidden).
  std::int32_t faceSetValue(Index f) const { return faceSets.empty() ? kDefaultFaceSet : faceSets[f]; }
  bool faceHidden(Index f) const { return !faceSets.empty() && faceSets[f] < 0; }
  // True if any live face is in a set other than the default one.
  bool anyFaceSet() const;
  // True if any live face is hidden.
  bool anyHidden() const;
  // True if any live face has a value other than kDefaultFaceSet (a set or hidden), that is, if
  // the face sets carry information that must be kept.
  bool hasFaceSetData() const;
  // Largest face set id in use (kDefaultFaceSet when there are no face sets).
  std::int32_t maxFaceSetId() const;
  // An id no face uses (the largest in use plus one), or 0 when the largest is kMaxFaceSetId.
  std::int32_t newFaceSetId() const;
  // True if at least one face around v is visible. Vertices without faces count as hidden.
  bool vertexVisible(Index v) const;

  // Reserves capacity for at least this many vertices, faces and half-edges in every array, so
  // appending up to that point never reallocates (dynamic topology edits run inside a dab).
  void reserveHeadroom(Index vertices, Index faces, Index halfEdges);
  // True if `vertices`, `faces` and `halfEdges` more elements fit without reallocating.
  bool hasHeadroom(Index vertices, Index faces, Index halfEdges) const;

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

  // Area-weighted normal of vertex v: the sum of faceAreaNormal() over its faces in
  // forEachOutgoing() order, normalized ((0, 0, 1) when the sum is zero). Every normal the engine
  // writes comes from this one function, so normals are a pure function of positions and
  // topology and recomputing them always reproduces the same bits.
  Vec3 vertexNormal(Index v) const;
  // Recomputes area-weighted vertex normals for every vertex (parallel).
  void computeNormals();
  // Recomputes normals for a contiguous range of vertices only.
  void computeNormals(Index vertBegin, Index vertEnd);

  // Number of unique undirected edges (boundary edges counted once).
  Index edgeCount() const;

  // Reorders faces into `faceOrder` (a permutation of all face indices) and renumbers vertices
  // by first use in that order, so faces that are adjacent in the new order also own adjacent
  // vertices. Half-edges are laid out face by face, each face starting at its old faceHe, and
  // vertHe keeps pointing at the same half-edge, so face corner order and vertex fan order are
  // unchanged. Returns, for each new face index f, the number of distinct vertices first used by
  // faces before f (size faceCount() + 1). `map`, if given, receives the old index of every new
  // face and vertex.
  std::vector<Index> reorder(std::span<const Index> faceOrder, ReorderMap* map = nullptr);

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
// Like validate(), but skips elements removed by MeshEditor that compact() has not dropped yet:
// faces with faceHe == kInvalid, half-edges with heFace == kInvalid and vertices without faces.
// Live elements may only reference live elements.
ValidationResult validateLive(const Mesh& mesh);

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
