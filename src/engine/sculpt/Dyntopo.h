#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "core/Timer.h"
#include "mesh/MeshEdit.h"
#include "scene/Scene.h"
#include "spatial/LeafLayout.h"

namespace plegl {

enum class DyntopoRefine : std::uint8_t { SplitCollapse, SplitOnly, CollapseOnly };

struct DyntopoOptions {
  DyntopoRefine refine = DyntopoRefine::SplitCollapse;
  int maxSplits = 1024;       // Per pass.
  int maxCollapses = 1024;    // Per pass.
  double timeBudgetMs = 3.0;  // Per pass, on top of the counts. 0 turns it off (deterministic).
};

// Topology request for one dab.
struct DabTopology {
  float detail = 0.0f;        // Target edge length in object space. 0 leaves topology alone.
  Index hintFace = kInvalid;  // Face under the cursor: its long edges are split even when the
                              // brush is too small to reach any of them.
};

// What one topology pass changed.
struct DyntopoPass {
  int splits = 0;
  int collapses = 0;
  int edits = 0;                       // Every topology edit, quad triangulations included.
  std::vector<Index> changedVerts;     // Live vertices whose normal may have changed (sorted).
  std::vector<Index> refitLeaves;      // Leaves whose bounds may have changed (sorted).
  std::vector<Index> topoDirtyLeaves;  // Leaves whose triangles or edges changed, for GPU index data (sorted).
  bool changed() const { return edits > 0; }
};

// One dynamic topology stroke on one object. Creating it reserves headroom, finds the vertices it
// must not touch and puts the BVH into dynamic mode. Each pass then splits edges longer than the
// detail size D and collapses edges shorter than 0.4 D inside the brush sphere, editing in place
// (removed elements stay as tombstones, new ones go to BVH tail leaves) and recording the leaves it
// changes for undo. Sculptor::endStroke folds everything back into a compact layout.
//
// Splits follow longest-edge bisection: a triangle is only ever cut across its longest edge, so
// repeated refinement cannot produce slivers. Untouchable: vertices with mask >= 0.5, non-manifold
// vertices, every face that has one of those as a corner, and faces with more than 4 corners.
// Open borders keep their shape: border edges are split at their midpoint, but border vertices
// never merge. Quads are cut into triangles only where an edit needs it, along their shorter
// diagonal that does not fold.
class DyntopoSession {
 public:
  DyntopoSession(SceneObject& object, const DyntopoOptions& options);
  DyntopoSession(const DyntopoSession&) = delete;
  DyntopoSession& operator=(const DyntopoSession&) = delete;

  // Effective detail: never finer than radius / 20, so one dab cannot explode the face count.
  static float effectiveDetail(float detail, float radius, float meshDiagonal);

  const DyntopoPass& pass(const Vec3& center, float radius, const DabTopology& topo);

  ClaimRecorder& recorder() { return rec_; }
  const ClaimRecorder& recorder() const { return rec_; }
  // True once anything was claimed or appended.
  bool changedTopology() const;
  // Set when the session found the mesh in a state it does not trust; topology then stops for
  // the rest of the stroke while the brush keeps working.
  bool faulted() const { return faulted_; }
  // Set when the headroom reserved for the stroke ran out; splits stop for the rest of it.
  bool outOfRoom() const { return outOfRoom_; }
  Index lockedVertices() const { return lockedCount_; }
  int totalSplits() const { return totalSplits_; }
  int totalCollapses() const { return totalCollapses_; }

  // The layout the stroke started from, for undo.
  const std::shared_ptr<const Bvh>& beforeBvh() const { return beforeBvh_; }
  std::uint64_t beforeVersion() const { return beforeVersion_; }

 private:
  // Forwards writes to the claim recorder and notes which leaves' faces change.
  class Observer final : public EditObserver {
   public:
    void beforeWrite(ElementKind kind, Index index) override;
    DyntopoSession* session = nullptr;
  };

  struct Edge {
    float len2;
    Index h, a, b;
  };

  bool locked(Index v) const {
    return (v < static_cast<Index>(lock_.size()) && lock_[v]) || (!m_.mask.empty() && m_.mask[v] >= 0.5f);
  }
  bool frozen(Index f, int& size) const;
  bool pickDiagonal(Index f, Index& ha, Index& hb) const;
  void markFace(Index f);
  void markHalfEdge(Index h);
  void gather(const Vec3& c, float r, float lmax2, float lmin2, Index hintFace);
  void splitAll(const Vec3& c, float r, float lmax2);
  bool collapseOne(const Edge& e, float lmax2, float lmin2);
  void touchRing(Index v, bool refit);
  void finishPass();
  bool overBudget();

  SceneObject& obj_;
  Mesh& m_;
  Bvh& bvh_;
  DyntopoOptions opt_;
  ClaimRecorder rec_;
  Observer observer_;
  MeshEditor ed_;
  std::shared_ptr<const Bvh> beforeBvh_;
  std::uint64_t beforeVersion_ = 0;
  float diagonal_ = 1.0f;
  std::vector<char> lock_;  // Non-manifold vertices at stroke start.
  Index lockedCount_ = 0;
  bool faulted_ = false;
  bool outOfRoom_ = false;
  int totalSplits_ = 0, totalCollapses_ = 0;

  // Per pass.
  DyntopoPass pass_;
  std::vector<Index> leaves_;
  std::vector<std::vector<Edge>> perLeafSplits_, perLeafCollapses_;
  std::vector<Edge> splits_, collapses_;
  std::vector<Index> ring_;
  Index faceLeaf_ = kInvalid;  // Leaf of the last face marked, which is already in topoDirtyLeaves.
  Index heLeaf_ = kInvalid;    // The same for half-edges.
  std::size_t leavesBefore_ = 0;
  BvhLeaf openTailBefore_;
  Timer passTimer_;
  int opsSinceCheck_ = 0;
};

}  // namespace plegl
