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
// and marks GPU ranges dirty. Brushes only move positions.
class Sculptor {
 public:
  void beginStroke(SceneObject& object, const Brush& brush, const StrokeOptions& options, std::string label);
  bool active() const { return object_ != nullptr; }
  SceneObject* object() const { return object_; }

  // Applies one dab at a surface point in the object's local space. `radius` and `strength`
  // already include pen pressure. Returns false if nothing was under the dab.
  bool dab(const Vec3& center, float radius, float strength);

  // Ends the stroke. Returns the undo entry, or nothing if the stroke changed nothing.
  std::optional<SculptUndo> endStroke();

  const DabTiming& lastDab() const { return lastDab_; }
  int dabCount() const { return dabCount_; }

 private:
  bool applyOne(const Dab& dab);
  void snapshot(Index leaf);
  Vec3 areaNormal(const Vec3& center, float radius, std::span<const Index> leaves) const;

  SceneObject* object_ = nullptr;
  const Brush* brush_ = nullptr;
  StrokeOptions options_;
  SculptUndo undo_;
  std::unordered_map<Index, std::size_t> snapshotIndex_;  // leaf -> index in undo_.before
  std::vector<std::uint32_t> vertexStamp_;
  std::uint32_t stamp_ = 0;
  std::vector<Index> leaves_, normalVerts_, dirtyLeaves_;
  DabTiming lastDab_;
  int dabCount_ = 0;
};

}  // namespace plegl
