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
// Contract: only vertices inside the dab sphere move, and none ends up farther than 1.25 radius
// from the centre (the sculptor finds stale normals with that reach). Brushes that move vertices
// farther per dab, such as a future Grab, must widen that reach.
class Brush {
 public:
  virtual ~Brush() = default;
  virtual const char* name() const = 0;
  virtual bool needsAreaNormal() const { return false; }
  virtual void apply(BrushContext& ctx) const = 0;
};

// Pushes the surface out along the area normal (in when inverted).
class DrawBrush final : public Brush {
 public:
  const char* name() const override { return "Draw"; }
  bool needsAreaNormal() const override { return true; }
  void apply(BrushContext& ctx) const override;
};

// Moves vertices toward the average of their neighbours. Open-border vertices only average
// along the border, so open meshes do not shrink inward from their edges.
class SmoothBrush final : public Brush {
 public:
  const char* name() const override { return "Smooth"; }
  void apply(BrushContext& ctx) const override;
};

}  // namespace plegl
