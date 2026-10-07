#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "sculpt/Brush.h"
#include "sculpt/Undo.h"

namespace plegl {

struct StrokeOptions {
  float strength = 0.5f;
  Falloff falloff = Falloff::Smooth;
  bool invert = false;
  bool symmetryX = false;  // Mirror every dab across the object's local X = 0 plane.
};

struct DabTiming {
  double totalMs = 0.0;    // Whole dab: query, brush, normals, refit, bookkeeping.
  double brushMs = 0.0;    // Brush kernel only.
  double normalsMs = 0.0;
  int leaves = 0;
  int vertices = 0;        // Vertices whose normals were recomputed.
};

// Runs strokes on one scene object. For every dab it queries the BVH, snapshots the touched
// leaves for undo, runs the brush, recomputes normals around the moved vertices, refits the BVH
// and marks GPU ranges dirty. Brushes only move positions. Mask brushes take a shorter path: the
// sculptor allocates the mask, snapshots only mask values and skips normals and refitting.
class Sculptor {
 public:
  void beginStroke(SceneObject& object, const Brush& brush, const StrokeOptions& options, std::string label);
  bool active() const { return object_ != nullptr; }
  SceneObject* object() const { return object_; }

  // Applies one dab at a surface point in the object's local space. `radius` and `strength`
  // already include pen pressure. Returns false if nothing was under the dab.
  bool dab(const Vec3& center, float radius, float strength);

  // Grab stroke: captures the vertices inside the sphere (and its X mirror when symmetry is on)
  // and then moves them rigidly with the cursor, weighted by falloff, until the stroke ends.
  // Returns false (and starts nothing) if no vertex is inside the sphere.
  bool beginGrab(SceneObject& object, const StrokeOptions& options, const Vec3& center, float radius,
                 std::string label);
  // Moves the captured vertices to their start position plus `offset` (local space).
  void grab(const Vec3& offset);

  // Ends the stroke. Returns the undo entry, or nothing if the stroke changed nothing. Mask
  // strokes keep only the leaves whose mask actually changed.
  std::optional<SculptUndo> endStroke();

  const DabTiming& lastDab() const { return lastDab_; }
  int dabCount() const { return dabCount_; }

 private:
  void start(SceneObject& object, const StrokeOptions& options, std::string label);
  bool applyOne(const Dab& dab);
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
};

}  // namespace plegl
