#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "mesh/Mesh.h"
#include "mesh/MeshEdit.h"
#include "spatial/Bvh.h"

namespace plegl {

// Attribute data owned by one BVH leaf at one moment. Each channel is either empty (not recorded,
// so applying the state leaves it alone) or holds one value per vertex of the leaf (per face for
// face sets). Sculpt strokes record positions and normals; mask strokes and mask operations record
// the mask only; face set strokes and operations (hide and reveal too) the face sets only.
struct LeafState {
  Index leaf = kInvalid;
  std::vector<Vec3> positions;
  std::vector<Vec3> normals;
  std::vector<float> mask;
  std::vector<std::int32_t> faceSets;
};

// Everything one BVH leaf owns at one moment: its ranges and a raw slice of every mesh array over
// them, in the numbering of the layout they were taken from. Dynamic topology undo stores whole
// leaves this way, because an edit can rewire anything inside a leaf.
struct TopoLeafState {
  Index leaf = kInvalid;
  BvhLeaf ranges;
  std::vector<Vec3> positions;
  std::vector<Vec3> normals;  // Empty when the mesh had no normals.
  std::vector<float> mask;    // Empty when the mesh had no mask.
  std::vector<std::int32_t> faceSets;  // Empty when the mesh had no face sets.
  std::vector<Index> vertHe, faceHe, heNext, heTwin, heVert, heFace;
  std::size_t bytes() const;
};

TopoLeafState captureLeaf(const Mesh& mesh, const Bvh& bvh, Index leaf);

// Records which BVH leaves a dynamic topology stroke changes. Attached to the stroke's MeshEditor
// as its observer, it snapshots a whole leaf the first time an element the leaf owns is about to
// change, and flags the leaf as holding removed elements. Elements appended during the stroke
// belong to tail leaves and are never snapshotted.
class ClaimRecorder final : public EditObserver {
 public:
  // Starts recording a stroke. The BVH must be in dynamic mode; both must outlive the recording.
  void begin(const Mesh& mesh, Bvh& bvh);
  void beforeWrite(ElementKind kind, Index index) override;
  // Claims a leaf by hand, for writes made outside MeshEditor.
  void claimLeaf(Index leaf);

  bool claimed(Index leaf) const {
    return leaf >= 0 && leaf < static_cast<Index>(slot_.size()) && slot_[leaf] >= 0;
  }
  // The snapshot taken when the leaf was claimed, or nullptr. Its positions and normals may be
  // replaced by older ones, from a position-only snapshot taken earlier in the stroke.
  TopoLeafState* claim(Index leaf) { return claimed(leaf) ? &claims_[slot_[leaf]] : nullptr; }
  // Claimed leaves, in the order they were claimed.
  std::span<const Index> claimedLeaves() const { return order_; }
  bool empty() const { return order_.empty(); }
  // Moves the snapshots out, sorted by leaf, and stops recording.
  std::vector<TopoLeafState> takeClaims();

 private:
  Index findLeaf(ElementKind kind, Index index);

  const Mesh* mesh_ = nullptr;
  Bvh* bvh_ = nullptr;
  Index tailFace_ = 0, tailVert_ = 0, tailHe_ = 0;
  std::vector<std::int32_t> slot_;  // Old leaf -> index into claims_, or -1.
  std::vector<TopoLeafState> claims_;
  std::vector<Index> order_;
  Index lastLeaf_[3] = {kInvalid, kInvalid, kInvalid};  // Per element kind; writes cluster.
};

// Scratch memory for relayouts, kept between strokes so a relayout writes into memory that is
// already mapped: the workspace mesh and the object's mesh swap arrays every time, and both keep
// their capacity. One workspace can serve every object. Not thread safe.
struct LayoutWorkspace {
  Mesh mesh;
  std::vector<Index> vmap, fmap, hmap;
  std::size_t bytes() const;
};

// What one consolidation changed: the leaves it rebuilt (claimed, reused empty and appended
// leaves) and, as (before, after) index pairs, the elements of those leaves that untouched leaves
// reference. With the region slices of both layouts this is enough to switch the mesh between
// them in either direction.
struct LayoutDelta {
  std::vector<Index> regionLeaves;  // Sorted.
  std::vector<std::pair<Index, Index>> vertRefs;
  std::vector<std::pair<Index, Index>> heRefs;
  std::size_t bytes() const;
};

struct ConsolidateResult {
  LayoutDelta delta;
  // An unclaimed leaf had changed, so every leaf was rebuilt. The mesh and BVH are valid, but the
  // claims do not describe the stroke and must not be used for undo.
  bool missedClaim = false;
  // Mesh sizes when the stroke began.
  Index beforeVertexCount = 0, beforeFaceCount = 0, beforeHalfEdgeCount = 0;
  Index regionFaces = 0;  // Faces in the rebuilt leaves.
};

// Ends a dynamic topology stroke: drops removed elements and folds the claimed and tail leaves
// back into a compact layout. The region (claimed plus tail leaves) is split into leaves of at most
// bvh.maxLeafFaces() faces, which take the claimed leaf indices first, then empty leaves,
// then new indices at the end; untouched leaves keep their index, contents and bounds and only
// shift. The BVH leaves dynamic mode with a rebuilt tree. Cost: O(mesh) copying plus O(region).
ConsolidateResult consolidate(Mesh& mesh, Bvh& bvh, std::span<const Index> claimed, LayoutWorkspace& ws);

// One layout of an object's mesh as dynamic topology undo stores it: the whole BVH, slices of the
// region leaves and the position-only changes of untouched leaves.
struct LayoutSide {
  std::uint64_t topologyVersion = 0;
  std::shared_ptr<const Bvh> bvh;
  std::vector<TopoLeafState> region;  // Sorted by leaf: every region leaf this layout has.
  std::vector<LeafState> posOnly;     // Untouched leaves whose vertices moved during the stroke.
  Index vertexCount = 0, faceCount = 0, halfEdgeCount = 0;
  std::size_t bytes() const;
};

// The layout a stroke started from. `claims` are the recorder's snapshots (sorted by leaf, with
// positions and normals from before the stroke) and `beforeBvh` the BVH copied at stroke begin.
LayoutSide beforeSide(std::shared_ptr<const Bvh> beforeBvh, std::vector<TopoLeafState> claims,
                      std::vector<LeafState> posOnly, const ConsolidateResult& result, std::uint64_t version);

// The current layout, as the other side of `delta`. posOnlyLeaves lists the untouched leaves to
// record positions and normals for (those of the other side's posOnly).
LayoutSide captureSide(const Mesh& mesh, const Bvh& bvh, const LayoutDelta& delta,
                       std::span<const Index> posOnlyLeaves, std::uint64_t version);

// Switches the mesh from the other side of `delta` to `to`: untouched leaves are moved into place,
// their references into the region translated through the delta's pairs, and `to`'s region slices,
// BVH and position-only states are written verbatim. `toAfter` says which way the pairs are read.
// Returns false, changing nothing, when the mesh does not match the layout `to` expects.
bool relayout(Mesh& mesh, Bvh& bvh, const LayoutSide& to, const LayoutDelta& delta, bool toAfter,
              LayoutWorkspace& ws);

// Checks a mesh in the middle of a dynamic topology stroke: valid apart from removed elements,
// old leaves partitioning the stroke-begin ranges, tail leaves covering everything appended in
// order, removed elements only in flagged leaves, and bounds that contain every live face and
// owned vertex of the flagged leaves.
ValidationResult validateDynamic(const Mesh& mesh, const Bvh& bvh);

}  // namespace plegl
