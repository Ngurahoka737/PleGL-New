#pragma once

#include <span>

#include "mesh/Mesh.h"
#include "spatial/Bvh.h"

namespace plegl {

enum class Falloff { Smooth, Sharp, Linear, Constant };

// Weight for a normalized distance t in [0, 1] from the dab centre (1 at the centre, 0 at the rim
// for every curve except Constant).
inline float falloffWeight(Falloff f, float t) {
  if (t >= 1.0f) return 0.0f;
  if (t <= 0.0f) return 1.0f;
  switch (f) {
    case Falloff::Smooth: return 1.0f - t * t * (3.0f - 2.0f * t);
    case Falloff::Sharp: return (1.0f - t) * (1.0f - t);
    case Falloff::Linear: return 1.0f - t;
    case Falloff::Constant: return 1.0f;
  }
  return 0.0f;
}

// One brush application, in the object's local space.
struct Dab {
  Vec3 center{0.0f};
  float radius = 0.1f;
  float strength = 0.5f;  // 0..1, pressure already applied.
  Falloff falloff = Falloff::Smooth;
  bool invert = false;
  Vec3 areaNormal{0.0f, 1.0f, 0.0f};  // Falloff-weighted mean normal under the dab.
  Vec3 areaCenter{0.0f};              // Falloff-weighted mean position under the dab.
};

// What a brush may touch: the mesh positions owned by `leaves`.
struct BrushContext {
  Mesh& mesh;
  const Bvh& bvh;
  std::span<const Index> leaves;  // Leaves whose bounds intersect the dab sphere.
  const Dab& dab;
};

// A brush moves vertices; it never changes topology, normals or bounds. The sculptor recomputes
// those, records undo and marks GPU ranges afterwards, so new brushes only implement apply().
//
// Contract: only vertices inside the dab sphere move, and none moves farther than kMaxDabMove
// radius in one dab, so none ends up beyond 1.25 radius from the centre (the sculptor finds
// stale normals with that reach). Grab moves vertices freely and has its own path in Sculptor.
class Brush {
 public:
  virtual ~Brush() = default;
  virtual const char* name() const = 0;
  // True if apply() reads Dab::areaNormal / Dab::areaCenter.
  virtual bool needsArea() const { return false; }
  virtual void apply(BrushContext& ctx) const = 0;
};

// Largest displacement a brush may apply to one vertex in one dab, as a fraction of the radius.
inline constexpr float kMaxDabMove = 0.2f;

// Pushes the surface out along the area normal (in when inverted).
class DrawBrush final : public Brush {
 public:
  const char* name() const override { return "Draw"; }
  bool needsArea() const override { return true; }
  void apply(BrushContext& ctx) const override;
};

// Moves vertices toward the average of their neighbours. Open-border vertices only average
// along the border, so open meshes do not shrink inward from their edges.
class SmoothBrush final : public Brush {
 public:
  const char* name() const override { return "Smooth"; }
  void apply(BrushContext& ctx) const override;
};

// Builds volume in layers: vertices below a plane slightly above the surface are pulled up to it,
// so strokes add material and flatten it at the same time (carves when inverted).
class ClayBrush final : public Brush {
 public:
  const char* name() const override { return "Clay"; }
  bool needsArea() const override { return true; }
  void apply(BrushContext& ctx) const override;
};

// Pushes every vertex along its own normal, so forms swell (shrink when inverted).
class InflateBrush final : public Brush {
 public:
  const char* name() const override { return "Inflate"; }
  void apply(BrushContext& ctx) const override;
};

// Pulls vertices onto the plane through the area under the brush (pushes away when inverted).
class FlattenBrush final : public Brush {
 public:
  const char* name() const override { return "Flatten"; }
  bool needsArea() const override { return true; }
  void apply(BrushContext& ctx) const override;
};

// Cuts a sharp groove: pushes in like Draw while pinching vertices toward the brush centre.
// Inverted it raises a sharp ridge.
class CreaseBrush final : public Brush {
 public:
  const char* name() const override { return "Crease"; }
  bool needsArea() const override { return true; }
  void apply(BrushContext& ctx) const override;
};

}  // namespace plegl
