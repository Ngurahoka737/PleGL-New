#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "core/Types.h"

namespace plegl {

// Sculpt layers: displacements stored on top of a base shape, each scaled by its own strength.
// What the artist sees (and what Mesh::positions holds) is the composite:
//
//   positions[i] = base[i] + sum over visible layers k of strength_k * offset_k[i]
//
// summed bottom to top in list order by composeFrom(), the only function that adds layers up.
// Layers have no blend modes, so the order never changes the shape; it is fixed only so the
// result is the same to the last bit every time. Offsets are in object (local) space.
//
// Strokes act on the composite as usual and are committed into the active layer (or the base)
// when they end; see Sculptor and sculpt/LayerOps.h.

inline constexpr int kMaxLayers = 16;                 // Per stack, created by the app.
inline constexpr int kMaxFileLayers = 255;            // Per stack, read from files.
inline constexpr float kMaxLayerStrength = 10.0f;     // |strength| <= 1000 %.
inline constexpr float kMinStrokeStrength = 0.05f;    // Strokes on a weaker layer are refused.
inline constexpr std::size_t kMaxLayerNameBytes = 63;
// Layer memory of one object over all its levels: Add and Duplicate are refused above this.
inline constexpr std::size_t kMaxObjectLayerBytes = std::size_t{1} << 30;
// The same when reading a project file, where layers made elsewhere may be larger.
inline constexpr std::size_t kMaxFileObjectLayerBytes = std::size_t{2} << 30;
// Strength drags recompute normals live up to this many affected vertices, on release above it.
inline constexpr std::size_t kLiveNormalsLimit = 600'000;

struct SculptLayer {
  std::uint32_t id = 0;      // >= 1, unique in its stack, never reused.
  std::string name;          // At most kMaxLayerNameBytes bytes of UTF-8.
  float strength = 1.0f;     // Finite, |strength| <= kMaxLayerStrength.
  bool visible = true;
  std::vector<Vec3> offset;  // One per vertex.
};

// The layers of one mesh. An empty stack is always exactly LayerStack{} (no base, epoch 0,
// nextId 1, active 0), so it carries no state.
struct LayerStack {
  std::vector<Vec3> base;         // The mesh without any layer; empty when there are no layers.
  std::vector<SculptLayer> list;  // Bottom to top: composition order.
  std::uint32_t active = 0;       // Stroke target: a layer id, or 0 for the base. Not history.
  std::uint32_t nextId = 1;
  // Set from nextTopologyVersion() when the stack is created, loaded or copied, so two stacks
  // never share a state key.
  std::uint64_t epoch = 0;

  bool empty() const { return list.empty(); }
  int indexOf(std::uint32_t id) const;  // -1 when absent.
  SculptLayer* find(std::uint32_t id);
  const SculptLayer* find(std::uint32_t id) const;
  // The array strokes on `id` write: the base for 0, a layer's offsets otherwise; nullptr when the
  // stack is empty or the layer is absent.
  std::vector<Vec3>* array(std::uint32_t id);
  const std::vector<Vec3>* array(std::uint32_t id) const;
  std::size_t bytes() const;
};

inline bool isZero(const Vec3& v) { return v.x == 0.0f && v.y == 0.0f && v.z == 0.0f; }  // -0 too.
inline bool sameBits(const Vec3& a, const Vec3& b) { return std::memcmp(&a, &b, sizeof(Vec3)) == 0; }

// Adds the visible layers onto `p`. offsetOf(k) returns layer k's offset at the vertex. Each
// product is its own statement and the engine is built without floating-point contraction, so the
// result does not depend on the compiler, the platform or the caller. A product of +-0 is never
// added: adding +0 to a -0 base would flip its sign bit, so a layer that is zero at a vertex is bit
// for bit the same as no layer there.
template <class OffsetAt>
Vec3 composeFrom(const LayerStack& s, Vec3 p, OffsetAt&& offsetOf) {
  for (std::size_t k = 0; k < s.list.size(); ++k) {
    const SculptLayer& l = s.list[k];
    if (!l.visible || l.strength == 0.0f) continue;
    const Vec3& o = offsetOf(k);
    const float x = l.strength * o.x;
    const float y = l.strength * o.y;
    const float z = l.strength * o.z;
    if (x != 0.0f) p.x = p.x + x;
    if (y != 0.0f) p.y = p.y + y;
    if (z != 0.0f) p.z = p.z + z;
  }
  return p;
}

// The composite at vertex i of a non-empty stack.
inline Vec3 composeVertex(const LayerStack& s, Index i) {
  const auto k = static_cast<std::size_t>(i);
  return composeFrom(s, s.base[k], [&](std::size_t l) -> const Vec3& { return s.list[l].offset[k]; });
}

// The composite of every vertex (parallel); `out` is resized to the base size.
void composeAll(const LayerStack& s, std::vector<Vec3>& out);

// Vertices where `offset` is not zero, ascending.
void layerSupport(const std::vector<Vec3>& offset, std::vector<Index>& out);

// Identifies everything that decides the composite apart from the arrays: the epoch and each
// layer's id, visibility and strength bits, in list order (FNV-1a). 0 for an empty stack, never 0
// otherwise. Stroke undo entries apply only under the key they were recorded with.
std::uint64_t layerStateKey(const LayerStack& s);

// Strength strokes on `id` are divided by: 1 for the base, 0 when the layer is absent.
float targetStrength(const LayerStack& s, std::uint32_t id);

// `name` trimmed and cut at kMaxLayerNameBytes on a UTF-8 boundary; "Layer <id>" when that leaves
// nothing.
std::string clampLayerName(std::string_view name, std::uint32_t id);

}  // namespace plegl
