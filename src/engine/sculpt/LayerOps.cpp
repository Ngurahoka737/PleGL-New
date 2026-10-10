#include "sculpt/LayerOps.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <utility>

#include "core/Parallel.h"
#include "core/Timer.h"
#include "sculpt/MaskOps.h"

namespace plegl {

namespace {
bool movesVertices(const Brush* brush) { return !brush || (!brush->editsMask() && !brush->editsFaceSets()); }
}  // namespace

std::string layerTargetRefusal(const SceneObject& object, const Brush* brush, std::uint32_t target) {
  if (!movesVertices(brush)) return {};
  const LayerStack& stack = object.mesh.layers;
  const bool erase = brush && dynamic_cast<const EraseLayerBrush*>(brush);
  if (stack.empty() || target == 0) {
    return erase ? "Erase Layer works on a layer. Pick a layer under Sculpt layers." : std::string{};
  }
  const SculptLayer* layer = stack.find(target);
  if (!layer) return "Pick a layer under Sculpt layers.";
  if (!layer->visible) return "Layer '" + layer->name + "' is hidden. Show it (L) to sculpt on it.";
  if (!(std::abs(layer->strength) >= kMinStrokeStrength)) {
    const long percent = std::lround(layer->strength * 100.0f);
    return "Layer '" + layer->name + "' is at " + std::to_string(percent) +
           " %. Raise its strength above 5 % to sculpt on it.";
  }
  return {};
}

std::string layerStrokeRefusal(const SceneObject& object, const Brush* brush, const StrokeOptions& options) {
  // Grab never runs dynamic topology, and subdivision levels never do either.
  if (brush && movesVertices(brush) && options.dyntopo && !object.multires && !object.mesh.layers.empty())
    return "Dynamic topology does not work on objects with sculpt layers. Turn it off (Ctrl+D), or use Layers > "
           "Apply All Layers.";
  return layerTargetRefusal(object, brush ? &strokeBrush(*brush, object, options) : nullptr, options.layerTarget);
}

const Brush& strokeBrush(const Brush& brush, const SceneObject& object, const StrokeOptions& options) {
  static const LayerSmoothBrush layerSmooth;
  if (options.smoothLayerOnly && options.layerTarget != 0 && object.mesh.layers.find(options.layerTarget) &&
      dynamic_cast<const SmoothBrush*>(&brush))
    return layerSmooth;
  return brush;
}

// ---- Recomposing -------------------------------------------------------------------------------

std::size_t LayerWorkspace::bytes() const {
  return (support.capacity() + changed.capacity() + ring.capacity() + leaves.capacity() + faces.capacity()) *
             sizeof(Index) +
         (next.capacity() + oldP.capacity() + faceN.capacity()) * sizeof(Vec3) +
         (faceMark.mark.capacity() + vertMark.mark.capacity()) * sizeof(std::uint32_t);
}

namespace {

// Writes positionOf(v) for every v in `verts` and collects the vertices whose bits changed.
template <class PositionOf>
void writePositions(Mesh& m, std::span<const Index> verts, LayerWorkspace& ws, PositionOf&& positionOf) {
  ws.next.resize(verts.size());
  parallelFor(0, verts.size(), 4096, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) ws.next[i] = positionOf(verts[i]);
  });
  ws.changed.clear();
  ws.oldP.clear();
  for (std::size_t i = 0; i < verts.size(); ++i) {
    Vec3& p = m.positions[verts[i]];
    if (sameBits(p, ws.next[i])) continue;
    ws.changed.push_back(verts[i]);
    ws.oldP.push_back(p);
    p = ws.next[i];
  }
}

// a = a united with b; both ascending without repeats.
void unite(std::vector<Index>& a, const std::vector<Index>& b, std::vector<Index>& scratch) {
  if (b.empty()) return;
  scratch.clear();
  std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(scratch));
  a.swap(scratch);
}

bool sameFloat(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

int activeLevel(const SceneObject& obj) { return obj.multires ? obj.multires->active : -1; }

LayerUndo beginEntry(const SceneObject& obj, LayerOp op, std::string label) {
  LayerUndo e;
  e.label = std::move(label);
  e.objectId = obj.id;
  e.topologyVersion = obj.topologyVersion;
  e.level = activeLevel(obj);
  e.op = op;
  e.before = layerSide(obj.mesh.layers);
  return e;
}

LayerUndo finishEntry(SceneObject& obj, LayerUndo& e) {
  e.after = layerSide(obj.mesh.layers);
  if (obj.multires) ++obj.multires->serial;
  return std::move(e);
}

bool fail(std::string* error, std::string text) {
  if (error) *error = std::move(text);
  return false;
}

std::nullopt_t refuse(std::string* error, std::string text) {
  fail(error, std::move(text));
  return std::nullopt;
}

std::string quoted(const SculptLayer& l) { return "'" + l.name + "'"; }

SculptLayer* findLayer(SceneObject& obj, std::uint32_t id, std::string* error) {
  SculptLayer* l = obj.mesh.layers.find(id);
  if (!l) fail(error, "Pick a layer under Sculpt layers.");
  return l;
}

// Room for `extra` more bytes of layers on this object.
bool roomFor(const SceneObject& obj, std::size_t extra, std::size_t limit, std::string* error) {
  if (static_cast<int>(obj.mesh.layers.list.size()) >= kMaxLayers)
    return fail(error, "An object (or subdivision level) can have at most " + std::to_string(kMaxLayers) +
                           " sculpt layers.");
  if (objectLayerBytes(obj) + extra > limit)
    return fail(error, "Not enough room for another layer: the sculpt layers of one object may use at most " +
                           std::to_string(limit >> 20) + " MB.");
  return true;
}

// Visible layers with a strength: the only ones the composite sees.
bool contributes(const SculptLayer& l) { return l.visible && l.strength != 0.0f; }

// Recomposes the live stack at `verts` with normals and bounds, as one-shot operations do.
void recomposeNow(SceneObject& obj, LayerWorkspace& ws) {
  const std::vector<Index> verts = std::move(ws.support);
  recomposeVertices(obj, verts, ws, NormalsMode::Now, true);
  ws.support = std::move(verts);
}

// Folds the strength-scaled offsets `o` into `t` per component, under the composeFrom rules.
void addScaled(Vec3& t, float s, const Vec3& o) {
  const float x = s * o.x;
  const float y = s * o.y;
  const float z = s * o.z;
  if (x != 0.0f) t.x = t.x + x;
  if (y != 0.0f) t.y = t.y + y;
  if (z != 0.0f) t.z = t.z + z;
}

}  // namespace

std::size_t recomposeVertices(SceneObject& object, std::span<const Index> verts, LayerWorkspace& ws, NormalsMode mode,
                              bool refit) {
  const LayerStack& stack = object.mesh.layers;
  // A base without layers is fine: its composite is the base (Delete recomposes that way).
  if (stack.base.empty() || verts.empty()) {
    ws.changed.clear();
    ws.oldP.clear();
    return 0;
  }
  writePositions(object.mesh, verts, ws, [&](Index v) { return composeVertex(stack, v); });
  finishPositions(object, ws.changed, ws, mode == NormalsMode::Now, refit);
  return ws.changed.size();
}

void finishPositions(SceneObject& object, std::span<const Index> changed, LayerWorkspace& ws, bool normals, bool refit) {
  if (changed.empty()) return;
  Mesh& m = object.mesh;
  Bvh& bvh = object.bvh;
  const std::size_t nv = m.positions.size();
  if (changed.size() * 2 > nv) {
    // Most of the mesh: whole-mesh passes give the same bits for less work.
    if (normals) m.computeNormals();
    if (refit) bvh.refit(m);
    object.markPositionsDirtyAll(normals);
    return;
  }
  // Vertices sharing a face with a changed one get new normals; the leaves of those faces new
  // bounds.
  ws.vertMark.begin(nv);
  ws.faceMark.begin(m.faceHe.size());
  ws.ring.clear();
  ws.leaves.clear();
  for (Index v : changed) {
    m.forEachOutgoing(v, [&](Index h) {
      const Index f = m.heFace[h];
      if (f == kInvalid || !ws.faceMark.insert(f)) return;
      if (refit) ws.leaves.push_back(bvh.leafOfFace(f));
      m.forEachFaceVertex(f, [&](Index u) {
        if (ws.vertMark.insert(u)) ws.ring.push_back(u);
      });
    });
  }
  std::sort(ws.ring.begin(), ws.ring.end());
  if (normals) leafNormals(m, ws.ring, ws);
  if (refit && !ws.leaves.empty()) {
    std::sort(ws.leaves.begin(), ws.leaves.end());
    ws.leaves.erase(std::unique(ws.leaves.begin(), ws.leaves.end()), ws.leaves.end());
    bvh.refitLeaves(m, ws.leaves);
  }
  // Owner leaves of the uploaded vertices, found once per run of the ascending list.
  const std::span<const Index> upload = normals ? std::span<const Index>(ws.ring) : changed;
  ws.leaves.clear();
  Index lastOwner = kInvalid;
  for (Index v : upload) {
    if (lastOwner != kInvalid) {
      const BvhLeaf& l = bvh.leaves()[lastOwner];
      if (v >= l.vertBegin && v < l.vertEnd) continue;
    }
    const Index owner = bvh.leafOfVertex(v);
    if (owner == kInvalid) continue;
    lastOwner = owner;
    ws.leaves.push_back(owner);
  }
  std::sort(ws.leaves.begin(), ws.leaves.end());
  ws.leaves.erase(std::unique(ws.leaves.begin(), ws.leaves.end()), ws.leaves.end());
  if (ws.leaves.size() > kPositionsDirtyAllLeaves) {
    object.markPositionsDirtyAll(normals);
  } else {
    for (Index l : ws.leaves) object.markLeafDirty(l);
  }
}

void leafNormals(Mesh& m, std::span<const Index> verts, LayerWorkspace& ws) {
  if (verts.size() * 2 > m.positions.size()) {
    m.computeNormals();
    return;
  }
  // Each face normal once, then the sums in Mesh::vertexNormal's order.
  ws.faceMark.begin(m.faceHe.size());
  ws.faces.clear();
  for (Index v : verts) {
    m.forEachOutgoing(v, [&](Index h) {
      const Index f = m.heFace[h];
      if (ws.faceMark.insert(f)) ws.faces.push_back(f);
    });
  }
  if (ws.faceN.size() < m.faceHe.size()) ws.faceN.resize(m.faceHe.size());
  parallelFor(0, ws.faces.size(), 2048, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) ws.faceN[ws.faces[i]] = m.faceAreaNormal(ws.faces[i]);
  });
  parallelFor(0, verts.size(), 2048, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const Index v = verts[i];
      Vec3 n{0.0f};
      m.forEachOutgoing(v, [&](Index h) { n += ws.faceN[m.heFace[h]]; });
      m.normals[v] = safeNormalize(n);
    }
  });
}

void rebaseReference(Multires& stack, std::span<const Index> changed, std::span<const Vec3> oldP, const Mesh& live,
                     LayerUndo* record) {
  std::vector<Vec3>& ref = stack.reference.positions;
  if (ref.size() != live.positions.size()) return;
  bool moved = false;
  for (std::size_t k = 0; k < changed.size(); ++k) {
    const Index v = changed[k];
    if (!sameBits(ref[v], oldP[k])) continue;  // A pending edit: it carries the rounding along.
    if (record) {
      record->refIndex.push_back(v);
      record->refBefore.push_back(ref[v]);
      record->refAfter.push_back(live.positions[v]);
    }
    ref[v] = live.positions[v];
    moved = true;
  }
  if (moved) ++stack.serial;
}

std::size_t objectLayerBytes(const SceneObject& object) {
  std::size_t n = object.mesh.layers.bytes();
  if (object.multires) {
    for (const MultiresLevel& level : object.multires->levels) n += level.mesh.layers.bytes();
  }
  return n;
}

LayerSide layerSide(const LayerStack& stack) {
  LayerSide side;
  side.hasStack = !stack.empty();
  side.epoch = stack.epoch;
  side.nextId = stack.nextId;
  side.active = stack.active;
  for (const SculptLayer& l : stack.list) side.layers.push_back({l.id, l.name, l.strength, l.visible});
  return side;
}

// ---- Operations --------------------------------------------------------------------------------

std::optional<LayerUndo> addLayer(SceneObject& object, LayerWorkspace&, std::string* error, std::size_t byteLimit) {
  if (error) error->clear();
  Mesh& m = object.mesh;
  LayerStack& s = m.layers;
  const std::size_t nv = m.positions.size();
  if (nv == 0) return refuse(error, "The mesh is empty.");
  if (!roomFor(object, nv * sizeof(Vec3) * (s.empty() ? 2 : 1), byteLimit, error)) return std::nullopt;
  if (s.empty()) {
    for (const Vec3& p : m.positions)
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
        return refuse(error, "The mesh has vertices with invalid coordinates.");
  }
  LayerUndo e = beginEntry(object, LayerOp::Add, "Add Layer");
  if (s.empty()) {
    s.base = m.positions;
    s.epoch = nextTopologyVersion();
    s.nextId = 1;
  }
  SculptLayer l;
  l.id = s.nextId++;
  l.name = "Layer " + std::to_string(l.id);
  l.offset.assign(nv, Vec3{0.0f});
  s.list.push_back(std::move(l));
  s.active = s.list.back().id;
  return finishEntry(object, e);
}

std::optional<LayerUndo> duplicateLayer(SceneObject& object, std::uint32_t id, LayerWorkspace&, std::string* error,
                                        std::size_t byteLimit) {
  if (error) error->clear();
  LayerStack& s = object.mesh.layers;
  const int k = s.indexOf(id);
  if (k < 0) return refuse(error, "Pick a layer under Sculpt layers.");
  if (!roomFor(object, object.mesh.positions.size() * sizeof(Vec3), byteLimit, error)) return std::nullopt;
  LayerUndo e = beginEntry(object, LayerOp::Duplicate, "Duplicate Layer");
  SculptLayer copy = s.list[static_cast<std::size_t>(k)];
  copy.id = s.nextId++;
  copy.name = clampLayerName(copy.name + " copy", copy.id);
  copy.visible = false;  // Visible, it would double the layer's effect.
  s.list.insert(s.list.begin() + k + 1, std::move(copy));
  return finishEntry(object, e);
}

std::optional<LayerUndo> deleteLayer(SceneObject& object, std::uint32_t id, LayerWorkspace& ws, std::string* error) {
  if (error) error->clear();
  LayerStack& s = object.mesh.layers;
  const int k = s.indexOf(id);
  if (k < 0) return refuse(error, "Pick a layer under Sculpt layers.");
  LayerUndo e = beginEntry(object, LayerOp::Delete, "Delete Layer");
  const auto at = static_cast<std::size_t>(k);
  const bool shows = contributes(s.list[at]);
  ws.support.clear();
  if (shows) layerSupport(s.list[at].offset, ws.support);
  e.held.push_back({id, {}, std::move(s.list[at].offset)});
  s.list.erase(s.list.begin() + k);
  if (s.active == id) s.active = at > 0 ? s.list[at - 1].id : (at < s.list.size() ? s.list[at].id : 0);
  if (shows) recomposeNow(object, ws);  // With no layer left this gives the base.
  if (s.list.empty()) {
    e.held.push_back({0, {}, std::move(s.base)});
    s = LayerStack{};
  }
  e.recompose = shows;
  if (shows) e.recomposeIds.push_back(id);
  return finishEntry(object, e);
}

std::optional<LayerUndo> renameLayer(SceneObject& object, std::uint32_t id, std::string_view name, std::string* error) {
  if (error) error->clear();
  SculptLayer* l = findLayer(object, id, error);
  if (!l) return std::nullopt;
  std::string clamped = clampLayerName(name, id);
  if (clamped == l->name) return std::nullopt;
  LayerUndo e = beginEntry(object, LayerOp::Rename, "Rename Layer");
  l->name = std::move(clamped);
  return finishEntry(object, e);
}

namespace {
// Sets the eyes of the layers `visible` says (indexed like the list) and recomposes where the
// changed layers reach.
std::optional<LayerUndo> setEyes(SceneObject& object, LayerOp op, std::string label, const std::vector<char>& visible,
                                 LayerWorkspace& ws) {
  LayerStack& s = object.mesh.layers;
  bool any = false;
  for (std::size_t k = 0; k < s.list.size(); ++k) any |= (s.list[k].visible != (visible[k] != 0));
  if (!any) return std::nullopt;
  LayerUndo e = beginEntry(object, op, std::move(label));
  std::vector<Index> support, scratch;
  ws.support.clear();
  for (std::size_t k = 0; k < s.list.size(); ++k) {
    SculptLayer& l = s.list[k];
    if (l.visible == (visible[k] != 0)) continue;
    l.visible = visible[k] != 0;
    if (l.strength == 0.0f) continue;
    e.recomposeIds.push_back(l.id);
    layerSupport(l.offset, support);
    unite(ws.support, support, scratch);
  }
  e.recompose = !e.recomposeIds.empty();
  recomposeNow(object, ws);
  return finishEntry(object, e);
}
}  // namespace

std::optional<LayerUndo> setLayerVisible(SceneObject& object, std::uint32_t id, bool visible, LayerWorkspace& ws,
                                         std::string* error) {
  if (error) error->clear();
  LayerStack& s = object.mesh.layers;
  const int k = s.indexOf(id);
  if (k < 0) return refuse(error, "Pick a layer under Sculpt layers.");
  std::vector<char> eyes;
  for (const SculptLayer& l : s.list) eyes.push_back(l.visible ? 1 : 0);
  eyes[static_cast<std::size_t>(k)] = visible ? 1 : 0;
  return setEyes(object, LayerOp::Visibility, visible ? "Show Layer" : "Hide Layer", eyes, ws);
}

std::optional<LayerUndo> soloLayer(SceneObject& object, std::uint32_t id, LayerWorkspace& ws, std::string* error) {
  if (error) error->clear();
  LayerStack& s = object.mesh.layers;
  const int k = s.indexOf(id);
  if (k < 0) return refuse(error, "Pick a layer under Sculpt layers.");
  bool alone = s.list[static_cast<std::size_t>(k)].visible;
  for (const SculptLayer& l : s.list) alone &= l.id == id || !l.visible;
  std::vector<char> eyes;
  for (const SculptLayer& l : s.list) eyes.push_back(alone || l.id == id ? 1 : 0);
  return setEyes(object, LayerOp::Solo, alone ? "Show All Layers" : "Solo Layer", eyes, ws);
}

std::optional<LayerUndo> setAllLayersVisible(SceneObject& object, bool visible, LayerWorkspace& ws,
                                             std::string* error) {
  if (error) error->clear();
  const std::vector<char> eyes(object.mesh.layers.list.size(), visible ? 1 : 0);
  return setEyes(object, visible ? LayerOp::ShowAll : LayerOp::HideAll,
                 visible ? "Show All Layers" : "Hide All Layers", eyes, ws);
}

namespace {
std::optional<LayerUndo> changeStrength(SceneObject& object, SculptLayer& l, float strength, LayerOp op,
                                        std::string label, LayerWorkspace& ws) {
  if (sameFloat(l.strength, strength)) return std::nullopt;
  LayerUndo e = beginEntry(object, op, std::move(label));
  const bool recompose = l.visible && (l.strength != 0.0f || strength != 0.0f);
  l.strength = strength;
  if (recompose) {
    layerSupport(l.offset, ws.support);
    recomposeNow(object, ws);
    e.recompose = true;
    e.recomposeIds.push_back(l.id);
  }
  return finishEntry(object, e);
}
}  // namespace

std::optional<LayerUndo> setLayerStrength(SceneObject& object, std::uint32_t id, float strength, LayerWorkspace& ws,
                                          std::string* error) {
  if (error) error->clear();
  SculptLayer* l = findLayer(object, id, error);
  if (!l) return std::nullopt;
  if (!std::isfinite(strength)) return refuse(error, "The strength must be a number.");
  strength = std::clamp(strength, -kMaxLayerStrength, kMaxLayerStrength);
  return changeStrength(object, *l, strength, LayerOp::Strength, "Layer Strength", ws);
}

std::optional<LayerUndo> invertLayer(SceneObject& object, std::uint32_t id, LayerWorkspace& ws, std::string* error) {
  if (error) error->clear();
  SculptLayer* l = findLayer(object, id, error);
  if (!l || l->strength == 0.0f) return std::nullopt;
  return changeStrength(object, *l, -l->strength, LayerOp::Invert, "Invert Layer", ws);
}

std::optional<LayerUndo> mergeLayerDown(SceneObject& object, std::uint32_t id, LayerWorkspace& ws, std::string* error) {
  if (error) error->clear();
  LayerStack& s = object.mesh.layers;
  const int k = s.indexOf(id);
  if (k < 0) return refuse(error, "Pick a layer under Sculpt layers.");
  if (k == 0) return refuse(error, "Nothing below to merge into. Use Apply.");
  SculptLayer& a = s.list[static_cast<std::size_t>(k)];
  SculptLayer& b = s.list[static_cast<std::size_t>(k - 1)];
  if (!a.visible || !b.visible) return refuse(error, "Show both layers to merge them.");
  LayerUndo e = beginEntry(object, LayerOp::MergeDown, "Merge Down");
  std::vector<Index> support, scratch;
  layerSupport(a.offset, ws.support);
  layerSupport(b.offset, support);
  unite(ws.support, support, scratch);
  // B' = sB * B + sA * A in a new array, so the old one can go into the entry whole.
  std::vector<Vec3> merged(b.offset.size(), Vec3{0.0f});
  const float sa = a.strength, sb = b.strength;
  parallelFor(0, ws.support.size(), 4096, [&](std::size_t lo, std::size_t hi) {
    for (std::size_t i = lo; i < hi; ++i) {
      const Index v = ws.support[i];
      Vec3 t{0.0f};
      addScaled(t, sb, b.offset[v]);
      addScaled(t, sa, a.offset[v]);
      merged[v] = t;
    }
  });
  const std::uint32_t bid = b.id;
  e.held.push_back({bid, {}, std::move(b.offset)});
  b.offset = std::move(merged);
  b.strength = 1.0f;
  e.held.push_back({id, {}, std::move(a.offset)});
  s.list.erase(s.list.begin() + k);
  s.active = bid;
  e.recompose = true;
  e.recomposeIds = {id, bid};
  recomposeNow(object, ws);
  if (object.multires) rebaseReference(*object.multires, ws.changed, ws.oldP, object.mesh, &e);
  return finishEntry(object, e);
}

std::optional<LayerUndo> applyAllLayers(SceneObject& object, LayerWorkspace&, std::string* error) {
  if (error) error->clear();
  LayerStack& s = object.mesh.layers;
  if (s.empty()) return std::nullopt;
  LayerUndo e = beginEntry(object, LayerOp::ApplyAll, "Apply All Layers");
  // The positions already are the composite: only the arrays go, into the entry.
  e.held.push_back({0, {}, std::move(s.base)});
  for (SculptLayer& l : s.list) e.held.push_back({l.id, {}, std::move(l.offset)});
  s = LayerStack{};
  return finishEntry(object, e);
}

std::optional<LayerUndo> applyLayer(SceneObject& object, std::uint32_t id, LayerWorkspace& ws, std::string* error) {
  if (error) error->clear();
  LayerStack& s = object.mesh.layers;
  const int k = s.indexOf(id);
  if (k < 0) return refuse(error, "Pick a layer under Sculpt layers.");
  SculptLayer& l = s.list[static_cast<std::size_t>(k)];
  if (!l.visible) return refuse(error, "Show the layer to apply it, or delete it.");
  if (s.list.size() == 1) {
    // base + s * L is exactly the composite of a stack holding only L: the same as Apply All.
    std::optional<LayerUndo> e = applyAllLayers(object, ws, error);
    if (e) {
      e->op = LayerOp::Apply;
      e->label = "Apply Layer";
    }
    return e;
  }
  LayerUndo e = beginEntry(object, LayerOp::Apply, "Apply Layer");
  layerSupport(l.offset, ws.support);
  HeldArray patch{0, ws.support, {}};
  patch.values.resize(ws.support.size());
  const float strength = l.strength;
  parallelFor(0, ws.support.size(), 4096, [&](std::size_t lo, std::size_t hi) {
    for (std::size_t i = lo; i < hi; ++i) {
      const Index v = ws.support[i];
      patch.values[i] = s.base[v];
      addScaled(s.base[v], strength, l.offset[v]);
    }
  });
  e.held.push_back(std::move(patch));
  e.held.push_back({id, {}, std::move(l.offset)});
  s.list.erase(s.list.begin() + k);
  s.active = 0;
  e.recompose = true;
  e.recomposeIds = {id};
  recomposeNow(object, ws);
  if (object.multires) rebaseReference(*object.multires, ws.changed, ws.oldP, object.mesh, &e);
  return finishEntry(object, e);
}

std::optional<SculptUndo> maskFromLayer(SceneObject& object, std::uint32_t id, std::string* error) {
  if (error) error->clear();
  SculptLayer* l = findLayer(object, id, error);
  if (!l) return std::nullopt;
  if (l->strength == 0.0f) return refuse(error, "Layer " + quoted(*l) + " is at 0 %.");
  Mesh& m = object.mesh;
  const std::size_t nv = m.positions.size();
  std::vector<float> length(nv);
  const float s = l->strength;
  parallelFor(0, nv, 16384, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) length[i] = glm::length(l->offset[i] * s);
  });
  const float most = nv > 0 ? *std::max_element(length.begin(), length.end()) : 0.0f;
  if (!(most > 0.0f) || !std::isfinite(most)) return refuse(error, "Layer " + quoted(*l) + " is empty.");
  m.ensureMask();
  std::vector<float> old = m.mask;
  parallelFor(0, nv, 16384, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) m.mask[i] = std::min(1.0f, length[i] / most);
  });
  return recordMaskEdit(object, old, "Mask from Layer");
}

bool selectLayer(SceneObject& object, std::uint32_t id) {
  LayerStack& s = object.mesh.layers;
  if (s.empty() || (id != 0 && !s.find(id))) return false;
  s.active = id;
  return true;
}

// ---- Strength drag -----------------------------------------------------------------------------

bool StrengthDrag::begin(SceneObject& object, std::uint32_t id, LayerWorkspace& ws, std::string* error,
                         std::size_t liveNormalsLimit) {
  if (error) error->clear();
  if (object_) end();
  SculptLayer* l = findLayer(object, id, error);
  if (!l) return false;
  object_ = &object;
  ws_ = &ws;
  id_ = id;
  liveNormalsLimit_ = liveNormalsLimit;
  start_ = l->strength;
  hasPending_ = skipNext_ = deferred_ = moved_ = false;
  entry_ = beginEntry(object, LayerOp::Strength, "Layer Strength");
  layerSupport(l->offset, support_);
  return true;
}

void StrengthDrag::apply(float strength) {
  SculptLayer* l = object_->mesh.layers.find(id_);
  if (!l || sameFloat(l->strength, strength)) return;
  l->strength = strength;
  if (!l->visible || support_.empty()) return;  // Hidden: nothing to show yet.
  Timer t;
  const NormalsMode mode = support_.size() <= liveNormalsLimit_ ? NormalsMode::Now : NormalsMode::Defer;
  // Nothing is picked while the pointer is on the panel, so bounds wait for the end.
  recomposeVertices(*object_, support_, *ws_, mode, false);
  deferred_ |= mode == NormalsMode::Defer;
  moved_ = true;
  skipNext_ = t.ms() > 12.0;
}

void StrengthDrag::update(float strength) {
  if (!object_ || !std::isfinite(strength)) return;
  strength = std::clamp(strength, -kMaxLayerStrength, kMaxLayerStrength);
  if (skipNext_) {
    pending_ = strength;
    hasPending_ = true;
    skipNext_ = false;
    return;
  }
  hasPending_ = false;
  apply(strength);
}

void StrengthDrag::finish() {
  if (hasPending_) apply(pending_);
  hasPending_ = false;
  if (moved_) finishPositions(*object_, support_, *ws_, deferred_, true);
}

std::optional<LayerUndo> StrengthDrag::end() {
  if (!object_) return std::nullopt;
  finish();
  SceneObject& obj = *object_;
  object_ = nullptr;
  const SculptLayer* l = obj.mesh.layers.find(id_);
  if (!l || sameFloat(l->strength, start_)) return std::nullopt;
  if (l->visible && !support_.empty()) {
    entry_.recompose = true;
    entry_.recomposeIds = {id_};
  }
  return finishEntry(obj, entry_);
}

void StrengthDrag::cancel() {
  if (!object_) return;
  hasPending_ = false;
  apply(start_);
  finish();
  object_ = nullptr;
}

// ---- Undo --------------------------------------------------------------------------------------

bool applyLayerUndo(Scene& scene, LayerUndo& e, bool redo, LayerWorkspace& ws) {
  SceneObject* obj = scene.find(e.objectId);
  if (!obj || obj->topologyVersion != e.topologyVersion || activeLevel(*obj) != e.level) return false;
  const LayerSide& from = redo ? e.before : e.after;
  const LayerSide& to = redo ? e.after : e.before;
  LayerStack& live = obj->mesh.layers;
  if (!layerSide(live).sameState(from)) return false;
  const std::size_t nv = obj->mesh.positions.size();

  // Every array the other side needs must be live or held, at the right size.
  auto heldWhole = [&](std::uint32_t id) -> HeldArray* {
    for (HeldArray& h : e.held)
      if (h.id == id && h.index.empty() && !h.values.empty()) return &h;
    return nullptr;
  };
  auto available = [&](std::uint32_t id) {
    const std::vector<Vec3>* a = live.array(id);
    if (a) return true;
    const HeldArray* h = heldWhole(id);
    return h && h->values.size() == nv;
  };
  if (to.hasStack) {
    if (!available(0)) return false;
    for (const LayerMeta& l : to.layers)
      if (!available(l.id)) return false;
  }

  // The pool: every live array, by id (0 for the base).
  std::vector<std::pair<std::uint32_t, std::vector<Vec3>>> pool;
  auto find = [&](std::uint32_t id) -> std::vector<Vec3>* {
    for (auto& [pid, a] : pool)
      if (pid == id) return &a;
    return nullptr;
  };
  auto inTo = [&](std::uint32_t id) {
    if (!to.hasStack) return false;
    if (id == 0) return true;
    return std::any_of(to.layers.begin(), to.layers.end(), [&](const LayerMeta& l) { return l.id == id; });
  };
  if (!live.empty()) {
    pool.emplace_back(0u, std::move(live.base));
    for (SculptLayer& l : live.list) pool.emplace_back(l.id, std::move(l.offset));
  }
  // Arrays coming back, and the other side's contents of arrays both sides have.
  for (HeldArray& h : e.held) {
    if (!h.index.empty() || h.values.empty()) continue;
    std::vector<Vec3>* a = find(h.id);
    if (!a) {
      if (inTo(h.id)) {
        pool.emplace_back(h.id, std::move(h.values));
        h.values.clear();
      }
    } else if (inTo(h.id)) {
      std::swap(*a, h.values);
    }
  }
  // Patches swap value by value, so applying one twice undoes it.
  for (HeldArray& h : e.held) {
    if (h.index.empty()) continue;
    std::vector<Vec3>* a = find(h.id);
    if (!a) continue;
    for (std::size_t k = 0; k < h.index.size(); ++k) std::swap((*a)[h.index[k]], h.values[k]);
  }

  // Where the composite can change: the supports of the listed layers in either version.
  std::vector<Index> verts, support, scratch;
  if (e.recompose) {
    for (std::uint32_t id : e.recomposeIds) {
      if (const std::vector<Vec3>* a = find(id)) {
        layerSupport(*a, support);
        unite(verts, support, scratch);
      }
      for (const HeldArray& h : e.held) {
        if (h.id != id || !h.index.empty() || h.values.empty()) continue;
        layerSupport(h.values, support);
        unite(verts, support, scratch);
      }
    }
    for (const HeldArray& h : e.held) {
      if (h.index.empty()) continue;
      support = h.index;
      unite(verts, support, scratch);
    }
  }

  // The other side's stack, from its settings and the pool.
  LayerStack next;
  if (to.hasStack) {
    next.epoch = to.epoch;
    next.nextId = to.nextId;
    next.active = to.active;
    std::vector<Vec3>* base = find(0);
    next.base = std::move(*base);
    pool.erase(std::find_if(pool.begin(), pool.end(), [](const auto& p) { return p.first == 0; }));
    for (const LayerMeta& meta : to.layers) {
      auto it = std::find_if(pool.begin(), pool.end(), [&](const auto& p) { return p.first == meta.id; });
      next.list.push_back({meta.id, meta.name, meta.strength, meta.visible, std::move(it->second)});
      pool.erase(it);
    }
  }
  // New positions: the composite, or the plain base when the other side has no stack.
  if (!verts.empty()) {
    if (to.hasStack) {
      writePositions(obj->mesh, verts, ws, [&](Index v) { return composeVertex(next, v); });
    } else if (const std::vector<Vec3>* base = find(0)) {
      writePositions(obj->mesh, verts, ws, [&](Index v) { return (*base)[v]; });
    }
  }
  // Arrays leaving go into the entry: into a free slot with their id, else a free one, else a new one.
  for (auto& [id, a] : pool) {
    HeldArray* slot = nullptr;
    for (HeldArray& h : e.held)
      if (h.index.empty() && h.values.empty() && h.id == id) slot = &h;
    if (!slot) {
      for (HeldArray& h : e.held)
        if (h.index.empty() && h.values.empty()) slot = &h;
    }
    if (!slot) slot = &e.held.emplace_back();
    slot->id = id;
    slot->values = std::move(a);
  }
  live = std::move(next);
  if (!verts.empty()) finishPositions(*obj, ws.changed, ws, true, true);

  if (obj->multires) {
    std::vector<Vec3>& ref = obj->multires->reference.positions;
    for (std::size_t k = 0; k < e.refIndex.size(); ++k) {
      const Index v = e.refIndex[k];
      if (static_cast<std::size_t>(v) < ref.size()) ref[v] = redo ? e.refAfter[k] : e.refBefore[k];
    }
    ++obj->multires->serial;
  }
  return true;
}

}  // namespace plegl
