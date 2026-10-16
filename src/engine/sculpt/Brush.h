#pragma once

#include <cmath>
#include <cstdint>
#include <span>

#include "mesh/Mesh.h"
#include "sculpt/Symmetry.h"
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
  // Snake Hook only: how far this dab pulls the vertex at its centre, and how much the pulled part
  // narrows (0..1).
  Vec3 pull{0.0f};
  float pinch = 0.0f;
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

// The sculpt layer a layer brush (Brush::writesLayer() true) edits: its offsets, scaled by
// `strength` in the composite.
struct LayerTarget {
  LayerStack* stack = nullptr;
  std::vector<Vec3>* offset = nullptr;
  float strength = 1.0f;
};

// How symmetric copies of one dab share the vertices they overlap on. A vertex within reach of two
// or more copies has a slot; each copy scales its move there by f^p / sum of f^p over the copies
// (f: the copy's raw falloff), so the shares add up to 1 and copies never stack. Moves into
// deferred slots wait in `pending` until every copy has run, so no copy reads another's writes.
// Vertices without a slot (slotOf -1) take a share of 1 and are written at once.
struct ShareView {
  const std::uint32_t* stamp = nullptr;
  std::uint32_t current = 0;
  const std::uint32_t* slot = nullptr;
  const float* denom = nullptr;
  const std::uint8_t* count = nullptr;
  Vec3* pending = nullptr;         // Position moves, or layer offset changes for layer brushes.
  float* pendingScalar = nullptr;  // Mask changes.
  float exponent = kShareExponent;
  bool deferAll = false;           // Brushes that read neighbours defer every slot.

  std::int64_t slotOf(Index v) const { return stamp[v] == current ? static_cast<std::int64_t>(slot[v]) : -1; }
  float share(std::int64_t s, float f) const {
    if (s < 0) return 1.0f;
    const float d = denom[s];
    return d > 0.0f ? (exponent == 1.0f ? f : std::pow(f, exponent)) / d : 0.0f;
  }
  bool deferred(std::int64_t s) const { return s >= 0 && (deferAll || count[s] >= 2); }
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
  LayerTarget* layer = nullptr;                 // Set for layer brushes only.
  // Symmetric copies that overlap: how this copy shares vertices with the others (nullptr when
  // no copy overlaps another, which keeps the plain arithmetic).
  const ShareView* shares = nullptr;
  // Face set brushes with several copies: a face claimed by an earlier copy in this dab
  // (faceClaim[f] == claimStamp) keeps that copy's set.
  std::uint32_t* faceClaim = nullptr;
  std::uint32_t claimStamp = 0;
};

// Adds a position move (already clamped) for vertex v with raw falloff f, scaled by the vertex's
// share and deferred when copies overlap there.
inline void writeMove(BrushContext& ctx, Index v, float f, Vec3 move) {
  if (!ctx.shares) {
    ctx.mesh.positions[v] += move;
    return;
  }
  const std::int64_t s = ctx.shares->slotOf(v);
  move *= ctx.shares->share(s, f);
  if (ctx.shares->deferred(s)) {
    ctx.shares->pending[s] += move;
  } else {
    ctx.mesh.positions[v] += move;
  }
}

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
  // True if apply() writes the offsets of BrushContext::layer and sets each changed position to
  // its composite. Other brushes move positions only; on a mesh with sculpt layers the sculptor
  // commits that movement into the stroke's target when the stroke ends.
  virtual bool writesLayer() const { return false; }
  // True if apply() reads neighbouring vertices, so overlapping copies must defer every write.
  virtual bool readsNeighbours() const { return false; }
  virtual void apply(BrushContext& ctx) const = 0;
};

// Largest displacement a brush may apply to one vertex in one dab, as a fraction of the radius.
inline constexpr float kMaxDabMove = 0.2f;
// Snake Hook: the longest pull of one sub-dab (90% of kMaxDabMove, so the centre moves exactly
// that far) and the share of the distance to the pull axis a full step with pinch 1 closes.
inline constexpr float kSnakeStep = 0.18f;
inline constexpr float kSnakePinch = 0.1f;

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
  bool readsNeighbours() const override { return true; }
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

// Snake Hook: moves the vertices around the dab centre by Dab::pull, weighted by falloff and
// mask, and with Dab::pinch also draws them toward the line through the centre along the pull, so
// pulled parts narrow. Strength is not used: the sculptor scales the pull (see
// Sculptor::snakeTo), which runs this brush in sub-dabs glued to one vertex.
class SnakeHookBrush final : public Brush {
 public:
  const char* name() const override { return "Snake Hook"; }
  void apply(BrushContext& ctx) const override;
};

// Fades the target layer's offsets toward zero under the brush, so the surface returns to how it
// looks without that layer. Inverting does nothing different.
class EraseLayerBrush final : public Brush {
 public:
  const char* name() const override { return "Erase Layer"; }
  bool writesLayer() const override { return true; }
  void apply(BrushContext& ctx) const override;
};

// Smooth with "This layer only": smooths the target layer's offsets instead of the surface, so
// the layer's detail softens while everything under it stays. The sculptor runs it in place of
// SmoothBrush when StrokeOptions::smoothLayerOnly is set and the target is a layer.
class LayerSmoothBrush final : public Brush {
 public:
  const char* name() const override { return "Smooth"; }
  bool writesLayer() const override { return true; }
  bool readsNeighbours() const override { return true; }
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
  bool readsNeighbours() const override { return true; }
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
