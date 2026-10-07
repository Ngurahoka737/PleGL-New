#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "sculpt/Brush.h"
#include "sculpt/Dyntopo.h"
#include "sculpt/Undo.h"

namespace plegl {

struct StrokeOptions {
  float strength = 0.5f;
  Falloff falloff = Falloff::Smooth;
  bool invert = false;
  bool symmetryX = false;  // Mirror every dab across the object's local X = 0 plane.
  // Dynamic topology: dabs that pass a detail size split and collapse edges under the brush.
  // Ignored by grab and mask strokes.
  bool dyntopo = false;
  DyntopoOptions dyntopoOptions{};
};

struct DabTiming {
  double totalMs = 0.0;    // Whole dab: query, brush, normals, refit, bookkeeping.
  double brushMs = 0.0;    // Brush kernel only.
  double normalsMs = 0.0;
  int leaves = 0;
  int vertices = 0;        // Vertices whose normals were recomputed.
  double topologyMs = 0.0; // Dynamic topology pass, normals and refit after it.
  int splits = 0;
  int collapses = 0;
};

// What the last ended stroke did to topology.
struct StrokeTopologyStats {
  bool dyntopo = false;    // The stroke ran with dynamic topology and changed the mesh.
  int splits = 0;
  int collapses = 0;
  Index facesBefore = 0;
  Index facesAfter = 0;
  double consolidateMs = 0.0;
  bool lostUndo = false;   // Undo could not be recorded (a missed claim); history was cut.
  bool faulted = false;    // Topology stopped part way through the stroke.
  bool outOfRoom = false;  // Splits stopped because the stroke's headroom ran out.
  Index lockedVertices = 0;
};

// Runs strokes on one scene object. For every dab it queries the BVH, snapshots the touched
// leaves for undo, runs the brush, recomputes normals around the moved vertices, refits the BVH
// and marks GPU ranges dirty. Brushes only move positions. Mask brushes take a shorter path: the
// sculptor allocates the mask, snapshots only mask values and skips normals and refitting.
//
// With dynamic topology each dab first refines or coarsens the mesh under the brush (see
// DyntopoSession), then runs the brush as usual. Leaves the topology pass changed are recorded
// whole; other leaves keep the usual position snapshots. endStroke() compacts the mesh again.
class Sculptor {
 public:
  void beginStroke(SceneObject& object, const Brush& brush, const StrokeOptions& options, std::string label);
  bool active() const { return object_ != nullptr; }
  SceneObject* object() const { return object_; }

  // Applies one dab at a surface point in the object's local space. `radius` and `strength`
  // already include pen pressure. `topology` only matters for dynamic topology strokes. Returns
  // false if nothing was under the dab.
  bool dab(const Vec3& center, float radius, float strength, const DabTopology& topology = {});

  // Grab stroke: captures the vertices inside the sphere (and its X mirror when symmetry is on)
  // and then moves them rigidly with the cursor, weighted by falloff, until the stroke ends.
  // Returns false (and starts nothing) if no vertex is inside the sphere.
  bool beginGrab(SceneObject& object, const StrokeOptions& options, const Vec3& center, float radius,
                 std::string label);
  // Moves the captured vertices to their start position plus `offset` (local space).
  void grab(const Vec3& offset);

  // Ends the stroke. Returns the undo entry, or nothing if the stroke changed nothing. Mask
  // strokes keep only the leaves whose mask actually changed. A dynamic topology stroke is
  // compacted here and returns a DyntopoUndo.
  std::optional<StrokeUndo> endStroke();

  // Scratch memory for compacting dynamic topology strokes, shared with the undo stack. Without
  // one the sculptor makes its own on first use.
  void setLayoutWorkspace(std::shared_ptr<LayoutWorkspace> workspace) { workspace_ = std::move(workspace); }
  // The running stroke's topology session, or nullptr.
  const DyntopoSession* dyntopo() const { return dyntopo_.get(); }

  const DabTiming& lastDab() const { return lastDab_; }
  int dabCount() const { return dabCount_; }
  const StrokeTopologyStats& lastStrokeTopology() const { return lastStroke_; }

 private:
  void start(SceneObject& object, const StrokeOptions& options, std::string label);
  bool applyOne(const Dab& dab, const DabTopology& topology);
  void applyTopology(const Dab& dab, const DabTopology& topology);
  void mergeClaims();
  std::optional<StrokeUndo> endDyntopoStroke(SceneObject& obj, DyntopoSession& session);
  void recomputeNormals(std::span<const Index> verts);
  void snapshot(Index leaf);
  bool computeArea(Dab& dab, std::span<const Index> leaves) const;

  SceneObject* object_ = nullptr;
  const Brush* brush_ = nullptr;
  bool maskStroke_ = false;
  StrokeOptions options_;
  SculptUndo undo_;
  std::unordered_map<Index, std::size_t> snapshotIndex_;  // leaf -> index in undo_.before
  std::vector<std::uint32_t> vertexStamp_;
  std::uint32_t stamp_ = 0;
  std::vector<Index> leaves_, normalVerts_, dirtyLeaves_;
  // Grab state: captured vertices, their start positions and weights for the primary and the
  // mirrored sphere, plus the fixed set of vertices whose normals and leaves a grab touches.
  struct GrabVertex {
    Index v;
    Vec3 start;
    float weight;
    float mirrorWeight;
  };
  std::vector<GrabVertex> grabVerts_;
  std::vector<Index> grabNormalVerts_, grabRefitLeaves_, grabDirtyLeaves_;
  DabTiming lastDab_;
  int dabCount_ = 0;
  std::unique_ptr<DyntopoSession> dyntopo_;
  std::size_t mergedClaims_ = 0;  // Claims already merged with position snapshots.
  std::shared_ptr<LayoutWorkspace> workspace_;
  StrokeTopologyStats lastStroke_;
};

}  // namespace plegl
