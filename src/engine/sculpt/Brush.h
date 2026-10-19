#pragma once

#include <cstdint>
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

// Face set limits on a stroke, on top of the mask. Off (the default) it costs nothing; on, it
// walks the faces around each vertex inside the dab.
struct FaceSetFilter {
  // Mesh::faceSets data while any limit below is on, otherwise nullptr. Refreshed every dab.
  const std::int32_t* faceSets = nullptr;
  bool skipHidden = false;      // Vertices with no visible face stay put.
  std::int32_t onlySet = 0;     // > 0: only vertices touching a visible face of this set move.
  bool lockBoundaries = false;  // Vertices where face sets meet stay put.

  // True if the brush may change vertex v.
  bool allows(const Mesh& m, Index v) const {
    if (!faceSets) return true;
    bool visible = false, inSet = false, mixed = false;
    std::int32_t first = 0;
    m.forEachOutgoing(v, [&](Index h) {
      const std::int32_t value = faceSets[m.heFace[h]];
      const std::int32_t id = faceSetId(value);
      visible |= value > 0;
      inSet |= value > 0 && id == onlySet;
      mixed |= first != 0 && id != first;
      if (first == 0) first = id;
    });
    if (first == 0) return false;  // No faces.
    return (!skipHidden || visible) && (onlySet <= 0 || inSet) && (!lockBoundaries || !mixed);
  }
  // True if a face set brush may repaint face f.
  bool allowsFace(Index f) const {
    return !faceSets || ((!skipHidden || faceSets[f] > 0) && (onlySet <= 0 || faceSets[f] == onlySet));
  }
};

// What a brush may touch: the mesh positions (or, for mask brushes, mask values, and for face set
// brushes, face sets) owned by `leaves`.
struct BrushContext {
  Mesh& mesh;
  const Bvh& bvh;
  std::span<const Index> leaves;  // Leaves whose bounds intersect the dab sphere.
  const Dab& dab;
  FaceSetFilter filter;
  std::int32_t paintFaceSet = kDefaultFaceSet;  // The set a face set brush paints.
};

// A brush moves vertices; it never changes topology, normals or bounds. The sculptor recomputes
// those, records undo and marks GPU ranges afterwards, so new brushes only implement apply().
// Moving brushes scale their effect by (1 - mask), so fully masked vertices never move, and leave
// vertices the face set filter refuses alone. Mask brushes (editsMask() true) change only
// Mesh::mask, which the sculptor allocates beforehand; face set brushes (editsFaceSets() true)
// only Mesh::faceSets, likewise allocated.
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
  // True if apply() writes Mesh::mask instead of positions.
  virtual bool editsMask() const { return false; }
  // True if apply() writes Mesh::faceSets instead of positions.
  virtual bool editsFaceSets() const { return false; }
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

// Paints the mask toward 1 (toward 0, erasing, when inverted). Each dab moves a value part of the
// way to its target, so overlapping dabs (for example at the symmetry plane) converge instead of
// overshooting.
class MaskBrush final : public Brush {
 public:
  const char* name() const override { return "Mask"; }
  bool editsMask() const override { return true; }
  void apply(BrushContext& ctx) const override;
};

// Smooths mask values toward the average of their neighbours, softening mask edges.
class MaskSmoothBrush final : public Brush {
 public:
  const char* name() const override { return "Smooth Mask"; }
  bool editsMask() const override { return true; }
  void apply(BrushContext& ctx) const override;
};

// Paints BrushContext::paintFaceSet onto the visible faces whose centre lies inside the dab, as
// far out as the falloff stays above 1 - strength (so a light touch paints a thinner line). Faces
// that are mostly masked keep their set.
class FaceSetBrush final : public Brush {
 public:
  const char* name() const override { return "Face Set"; }
  bool editsFaceSets() const override { return true; }
  void apply(BrushContext& ctx) const override;
};

}  // namespace plegl
