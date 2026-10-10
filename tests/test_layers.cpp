#include <cmath>
#include <cstring>
#include <random>

#include "TestUtil.h"
#include "mesh/MeshEdit.h"
#include "mesh/Primitives.h"
#include "remesh/QuadRemesh.h"
#include "scene/Scene.h"
#include "sculpt/LayerOps.h"
#include "sculpt/Neighbours.h"
#include "sculpt/Sculptor.h"
#include "sculpt/Undo.h"
#include "spatial/LeafLayout.h"

using namespace plegl;

namespace {

float unit(std::mt19937& rng) { return static_cast<float>(rng() >> 8) * (1.0f / 16777216.0f); }

bool bitwise(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(Vec3)) == 0);
}

// Starts the stack (base = positions) when there is none and adds a layer with zero offsets on
// top, which becomes the target. Leaves positions as they are, like the Add operation.
SculptLayer& addLayer(Mesh& m, float strength = 1.0f) {
  LayerStack& s = m.layers;
  if (s.empty()) {
    s.base = m.positions;
    s.epoch = nextTopologyVersion();
  }
  SculptLayer l;
  l.id = s.nextId++;
  l.name = "Layer " + std::to_string(l.id);
  l.strength = strength;
  l.offset.assign(m.positions.size(), Vec3{0.0f});
  s.list.push_back(std::move(l));
  s.active = s.list.back().id;
  return s.list.back();
}

// Positions, normals and bounds from the stack again, as the layer operations leave them.
void recompose(SceneObject& obj) {
  composeAll(obj.mesh.layers, obj.mesh.positions);
  obj.mesh.computeNormals();
  obj.bvh.refit(obj.mesh);
}

// A smooth bump pattern of layer detail, along the base normals.
void paintPattern(SceneObject& obj, SculptLayer& layer, float amplitude) {
  const Mesh& m = obj.mesh;
  for (Index v = 0; v < m.vertexCount(); ++v) {
    const Vec3& p = m.layers.base[v];
    layer.offset[v] = m.normals[v] * (amplitude * std::sin(7.0f * p.x) * std::cos(5.0f * p.y));
  }
}

void requireLayers(const Mesh& m) {
  const ValidationResult r = validateLayers(m, true);
  INFO(r.message);
  REQUIRE(r.ok);
}

Vec3 surfacePoint(const SceneObject& obj, Vec3 dir) {
  dir = glm::normalize(dir);
  RayHit hit;
  const bool ok = obj.bvh.raycast(obj.mesh, Ray{dir * 4.0f, -dir}, hit);
  REQUIRE(ok);
  return hit.position;
}

std::vector<Vec3> path(const SceneObject& obj, Vec3 from, Vec3 to, int n) {
  std::vector<Vec3> dabs;
  for (int i = 0; i < n; ++i) {
    const float t = n > 1 ? static_cast<float>(i) / static_cast<float>(n - 1) : 0.0f;
    dabs.push_back(surfacePoint(obj, from + (to - from) * t));
  }
  return dabs;
}

std::optional<StrokeUndo> stroke(Sculptor& sculptor, SceneObject& obj, const Brush& brush, const StrokeOptions& o,
                                 const std::vector<Vec3>& dabs, float radius, float strength) {
  sculptor.beginStroke(obj, brush, o, brush.name());
  REQUIRE(sculptor.active());
  for (const Vec3& c : dabs) sculptor.dab(c, radius, strength);
  return sculptor.endStroke();
}

// Every leaf's bounds as a full refit gives them: picking must see the committed surface.
void requireFreshBounds(const SceneObject& obj) {
  Bvh fresh = obj.bvh;
  fresh.refit(obj.mesh);
  const auto a = obj.bvh.leaves(), b = fresh.leaves();
  REQUIRE(a.size() == b.size());
  std::size_t stale = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    stale += !(a[i].bounds.min == b[i].bounds.min && a[i].bounds.max == b[i].bounds.max);
  CHECK(stale == 0);
}

StrokeOptions onLayer(const Mesh& m) {
  StrokeOptions o;
  o.strength = 0.8f;
  o.layerTarget = m.layers.active;
  return o;
}

// Everything a stroke on a layered mesh changes, and its undo must bring back.
struct Snap {
  std::vector<Vec3> positions, normals, base;
  std::vector<std::vector<Vec3>> offsets;
  std::uint64_t key = 0;
};

Snap snap(const SceneObject& obj) {
  Snap s{obj.mesh.positions, obj.mesh.normals, obj.mesh.layers.base, {}, layerStateKey(obj.mesh.layers)};
  for (const SculptLayer& l : obj.mesh.layers.list) s.offsets.push_back(l.offset);
  return s;
}

void requireSnap(const SceneObject& obj, const Snap& s) {
  CHECK(bitwise(obj.mesh.positions, s.positions));
  CHECK(bitwise(obj.mesh.normals, s.normals));
  CHECK(bitwise(obj.mesh.layers.base, s.base));
  REQUIRE(obj.mesh.layers.list.size() == s.offsets.size());
  for (std::size_t k = 0; k < s.offsets.size(); ++k) CHECK(bitwise(obj.mesh.layers.list[k].offset, s.offsets[k]));
  CHECK(layerStateKey(obj.mesh.layers) == s.key);
}

}  // namespace

// ---- Core --------------------------------------------------------------------------------------

TEST_CASE("layers: a zero, hidden or silent layer leaves the composite bit for bit") {
  Mesh m = makeIcosphere(1);
  m.positions[0] = {-0.0f, 1.0f, -0.0f};
  addLayer(m);
  addLayer(m, 0.0f);
  addLayer(m, 2.0f);
  SculptLayer& b = m.layers.list[1];
  SculptLayer& c = m.layers.list[2];
  b.offset[0] = {1.0f, 1.0f, 1.0f};  // Strength 0.
  c.offset[0] = {2.0f, 2.0f, 2.0f};
  c.visible = false;
  const Vec3 p = composeVertex(m.layers, 0);
  CHECK(sameBits(p, m.layers.base[0]));
  CHECK(std::signbit(p.x));
  CHECK(std::signbit(p.z));
  c.visible = true;
  CHECK(composeVertex(m.layers, 0).y == doctest::Approx(5.0f));
}

TEST_CASE("layers: composeAll matches composeVertex bit for bit") {
  Mesh m = makeIcosphere(4);
  std::mt19937 rng(7);
  for (int k = 0; k < 3; ++k) {
    SculptLayer& l = addLayer(m, 0.3f + 0.7f * static_cast<float>(k));
    for (Vec3& o : l.offset) o = Vec3{unit(rng), unit(rng), unit(rng)} * 0.01f;
  }
  std::vector<Vec3> all;
  composeAll(m.layers, all);
  REQUIRE(all.size() == m.positions.size());
  for (Index v = 0; v < m.vertexCount(); ++v) REQUIRE(sameBits(all[v], composeVertex(m.layers, v)));
}

TEST_CASE("layers: the state key follows strength, visibility, ids and epoch only") {
  Mesh m = makeIcosphere(1);
  addLayer(m);
  addLayer(m, 0.5f);
  const std::uint64_t k0 = layerStateKey(m.layers);
  CHECK(k0 != 0);
  CHECK(layerStateKey(LayerStack{}) == 0);
  m.layers.list[0].name = "Renamed";
  m.layers.active = 0;
  CHECK(layerStateKey(m.layers) == k0);
  m.layers.list[1].strength = std::nextafter(0.5f, 1.0f);
  CHECK(layerStateKey(m.layers) != k0);
  m.layers.list[1].strength = 0.5f;
  m.layers.list[1].visible = false;
  CHECK(layerStateKey(m.layers) != k0);
  m.layers.list[1].visible = true;
  m.layers.list[1].id = 9;
  m.layers.nextId = 10;
  CHECK(layerStateKey(m.layers) != k0);
  m.layers.list[1].id = 2;
  ++m.layers.epoch;
  CHECK(layerStateKey(m.layers) != k0);
}

TEST_CASE("layers: reordering for the BVH moves every layer array with its vertex") {
  Mesh m = makeIcosphere(4);
  SculptLayer& l = addLayer(m, 0.5f);
  for (Index v = 0; v < m.vertexCount(); ++v) l.offset[v] = m.layers.base[v] * 0.25f;
  composeAll(m.layers, m.positions);
  Bvh bvh;
  bvh.build(m, Bvh::Params{64});
  for (Index v = 0; v < m.vertexCount(); ++v) REQUIRE(sameBits(m.layers.list[0].offset[v], m.layers.base[v] * 0.25f));
  m.computeNormals();
  requireLayers(m);
}

TEST_CASE("layers: copyWithoutLayers copies everything else") {
  Mesh m = makeIcosphere(2);
  m.ensureMask();
  m.ensureFaceSets();
  m.mask[3] = 0.5f;
  m.faceSets[2] = 7;
  addLayer(m);
  const Mesh c = copyWithoutLayers(m);
  CHECK(c.layers.empty());
  CHECK(c.layers.epoch == 0);
  CHECK(c.layers.base.empty());
  CHECK(bitwise(c.positions, m.positions));
  CHECK(bitwise(c.normals, m.normals));
  CHECK(c.mask == m.mask);
  CHECK(c.faceSets == m.faceSets);
  CHECK(c.vertHe == m.vertHe);
  CHECK(c.faceHe == m.faceHe);
  CHECK(c.heNext == m.heNext);
  CHECK(c.heTwin == m.heTwin);
  CHECK(c.heVert == m.heVert);
  CHECK(c.heFace == m.heFace);
}

TEST_CASE("layers: validateLayers catches broken stacks") {
  Mesh m = makeIcosphere(2);
  addLayer(m);
  requireLayers(m);
  auto broken = [&](auto&& edit) {
    Mesh c = m;
    edit(c);
    return !validateLayers(c, true).ok;
  };
  CHECK(broken([](Mesh& c) { c.layers.list[0].offset.pop_back(); }));
  CHECK(broken([](Mesh& c) { c.layers.base.pop_back(); }));
  CHECK(broken([](Mesh& c) { c.layers.list[0].offset[5].y = std::nanf(""); }));
  CHECK(broken([](Mesh& c) {
    c.layers.list.push_back(c.layers.list[0]);
    ++c.layers.nextId;
  }));
  CHECK(broken([](Mesh& c) { c.layers.list[0].id = c.layers.nextId; }));
  CHECK(broken([](Mesh& c) { c.layers.list[0].strength = 11.0f; }));
  CHECK(broken([](Mesh& c) { c.layers.list[0].name.clear(); }));
  CHECK(broken([](Mesh& c) { c.layers.active = 99; }));
  CHECK(broken([](Mesh& c) { c.layers.epoch = 0; }));
  CHECK(broken([](Mesh& c) { c.layers.list[0].offset[9] = {0.0f, 1e-3f, 0.0f}; }));  // Composite is stale.
  CHECK(broken([](Mesh& c) {
    c.layers.list.clear();  // Empty, but still carrying state.
  }));
  CHECK_FALSE(broken([](Mesh& c) { c.layers.list[0].strength = -10.0f; }));
}

TEST_CASE("layers: clampLayerName trims and cuts on a character boundary") {
  CHECK(clampLayerName("  Detail \t", 3) == "Detail");
  CHECK(clampLayerName("   ", 4) == "Layer 4");
  CHECK(clampLayerName("a\nb", 1) == "a b");
  std::string name(62, 'x');
  name += "\xC3\xA9\xC3\xA9";  // Two two-byte characters: only 63 bytes fit, so both go.
  const std::string cut = clampLayerName(name, 1);
  CHECK(cut == std::string(62, 'x'));
  CHECK(cut.size() <= kMaxLayerNameBytes);
}

// ---- Strokes -----------------------------------------------------------------------------------

TEST_CASE("layers: a stroke on a layer goes into that layer only") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeIcosphere(5));
  SculptLayer& below = addLayer(obj.mesh);
  paintPattern(obj, below, 0.02f);
  recompose(obj);
  addLayer(obj.mesh);
  const Snap before = snap(obj);
  Sculptor sculptor;
  DrawBrush draw;
  const float r = 0.25f;
  const auto dabs = path(obj, {1, 0.2f, 0.1f}, {0.6f, 0.8f, 0.2f}, 8);
  auto undo = stroke(sculptor, obj, draw, onLayer(obj.mesh), dabs, r, 0.9f);
  REQUIRE(undo);
  requireLayers(obj.mesh);
  requireFreshBounds(obj);
  CHECK(bitwise(obj.mesh.layers.base, before.base));
  CHECK(bitwise(obj.mesh.layers.list[0].offset, before.offsets[0]));
  std::size_t touched = 0;
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
    if (isZero(obj.mesh.layers.list[1].offset[v])) continue;
    ++touched;
    float nearest = 1e9f;
    for (const Vec3& c : dabs) nearest = std::min(nearest, glm::length(before.positions[v] - c));
    REQUIRE(nearest < r * 1.5f);
  }
  CHECK(touched > 50);
  const auto* s = std::get_if<SculptUndo>(&*undo);
  REQUIRE(s);
  CHECK(s->layerTarget == obj.mesh.layers.list[1].id);
  CHECK(s->layerKey == before.key);
  for (const LeafState& st : s->before) CHECK(st.layer.size() == st.positions.size());

  // Undo and redo are exact.
  const Snap after = snap(obj);
  UndoStack stack;
  stack.push(std::move(*undo));
  CHECK_FALSE(stack.undo(scene).empty());
  requireSnap(obj, before);
  CHECK_FALSE(stack.redo(scene).empty());
  requireSnap(obj, after);
  requireLayers(obj.mesh);
}

TEST_CASE("layers: strokes divide by the layer's strength") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeIcosphere(4));
  Sculptor sculptor;
  DrawBrush draw;
  SUBCASE("half strength stores twice the movement") {
    // The same stroke on a copy without layers: the surface must end up where the brush left it.
    Scene plainScene;
    SceneObject& plain = plainScene.add("P", makeIcosphere(4));  // Built the same way: same vertex order.
    REQUIRE(bitwise(plain.mesh.positions, obj.mesh.positions));
    Sculptor plainSculptor;
    StrokeOptions plainOptions;
    plainOptions.strength = 0.8f;
    const auto dabs = path(obj, {0, 1, 0}, {0.3f, 1, 0}, 4);
    REQUIRE(stroke(plainSculptor, plain, draw, plainOptions, dabs, 0.3f, 1.0f));

    addLayer(obj.mesh, 0.5f);
    const std::vector<Vec3> start = obj.mesh.positions;
    REQUIRE(stroke(sculptor, obj, draw, onLayer(obj.mesh), dabs, 0.3f, 1.0f));
    requireLayers(obj.mesh);
    requireFreshBounds(obj);
    float worst = 0.0f, offPlain = 0.0f, moved = 0.0f;
    for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
      const Vec3 d = obj.mesh.positions[v] - start[v];
      worst = std::max(worst, glm::length(obj.mesh.layers.list[0].offset[v] - d * 2.0f));
      offPlain = std::max(offPlain, glm::length(obj.mesh.positions[v] - plain.mesh.positions[v]));
      moved = std::max(moved, glm::length(d));
    }
    CHECK(worst <= 1e-6f);
    CHECK(moved > 1e-2f);
    CHECK(offPlain <= 1e-6f);
  }
  SUBCASE("a negative layer still moves the surface the way the brush pushes") {
    addLayer(obj.mesh, -1.0f);
    const Vec3 c = surfacePoint(obj, {0, 1, 0});
    REQUIRE(stroke(sculptor, obj, draw, onLayer(obj.mesh), {c}, 0.3f, 1.0f));
    requireLayers(obj.mesh);
    Index top = 0;
    for (Index v = 0; v < obj.mesh.vertexCount(); ++v)
      if (glm::length(obj.mesh.layers.base[v] - c) < glm::length(obj.mesh.layers.base[top] - c)) top = v;
    CHECK(obj.mesh.positions[top].y > obj.mesh.layers.base[top].y + 1e-3f);
    CHECK(obj.mesh.layers.list[0].offset[top].y < -1e-3f);
  }
}

TEST_CASE("layers: refused strokes say why and do not start") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeIcosphere(3));
  DrawBrush draw;
  EraseLayerBrush erase;
  MaskBrush mask;
  Sculptor sculptor;
  const Vec3 c = surfacePoint(obj, {0, 1, 0});
  StrokeOptions o;
  CHECK(layerStrokeRefusal(obj, &erase, o) == "Erase Layer works on a layer. Pick a layer under Sculpt layers.");
  CHECK(layerStrokeRefusal(obj, &draw, o).empty());
  SculptLayer& l = addLayer(obj.mesh);
  o.layerTarget = l.id;
  CHECK(layerStrokeRefusal(obj, &draw, o).empty());
  l.visible = false;
  CHECK(layerStrokeRefusal(obj, &draw, o) == "Layer 'Layer 1' is hidden. Show it (L) to sculpt on it.");
  CHECK(layerStrokeRefusal(obj, nullptr, o) == "Layer 'Layer 1' is hidden. Show it (L) to sculpt on it.");
  CHECK(layerStrokeRefusal(obj, &mask, o).empty());  // The mask is the whole mesh's.
  sculptor.beginStroke(obj, draw, o, "Draw");
  CHECK_FALSE(sculptor.active());
  CHECK_FALSE(sculptor.dab(c, 0.3f, 1.0f));
  CHECK_FALSE(sculptor.endStroke());
  CHECK_FALSE(sculptor.beginGrab(obj, o, c, 0.3f, "Grab"));
  l.visible = true;
  l.strength = 0.04f;
  CHECK(layerStrokeRefusal(obj, &draw, o) == "Layer 'Layer 1' is at 4 %. Raise its strength above 5 % to sculpt on it.");
  l.strength = -0.05f;
  CHECK(layerStrokeRefusal(obj, &draw, o).empty());
  o.layerTarget = 0;
  CHECK(layerStrokeRefusal(obj, &erase, o) == "Erase Layer works on a layer. Pick a layer under Sculpt layers.");
  o.layerTarget = 77;
  CHECK(layerStrokeRefusal(obj, &draw, o) == "Pick a layer under Sculpt layers.");
  CHECK(layerStrokeRefusal(obj, &mask, o).empty());
  o.layerTarget = l.id;
  o.dyntopo = true;
  CHECK(layerStrokeRefusal(obj, &draw, o) ==
        "Dynamic topology does not work on objects with sculpt layers. Turn it off (Ctrl+D), or use Layers > Apply All Layers.");
  CHECK(layerStrokeRefusal(obj, nullptr, o).empty());  // Grab never runs dynamic topology.
  CHECK(layerStrokeRefusal(obj, &mask, o).empty());
}

TEST_CASE("layers: on the base, strokes keep the brush's result exactly") {
  Scene scene;
  SceneObject& a = scene.add("A", makeIcosphere(4));
  SceneObject& b = scene.add("B", makeIcosphere(4));
  SculptLayer& far = addLayer(a.mesh);
  for (Index v = 0; v < a.mesh.vertexCount(); ++v)
    if (a.mesh.layers.base[v].x < -0.4f) far.offset[v] = a.mesh.normals[v] * 0.05f;
  recompose(a);
  a.mesh.layers.active = 0;
  const std::vector<Vec3> offsets = far.offset;
  Sculptor sculptor;
  ClayBrush clay;
  StrokeOptions o = onLayer(a.mesh);
  REQUIRE(o.layerTarget == 0);
  StrokeOptions plain;
  plain.strength = o.strength;
  const auto dabs = path(a, {1, 0, 0}, {0.8f, 0.5f, 0}, 6);
  REQUIRE(stroke(sculptor, a, clay, o, dabs, 0.3f, 1.0f));
  REQUIRE(stroke(sculptor, b, clay, plain, dabs, 0.3f, 1.0f));
  requireLayers(a.mesh);
  CHECK(bitwise(offsets, a.mesh.layers.list[0].offset));
  std::size_t compared = 0;
  for (Index v = 0; v < a.mesh.vertexCount(); ++v) {
    if (a.mesh.layers.base[v].x < 0.0f) continue;  // Far from the layer.
    REQUIRE(sameBits(a.mesh.positions[v], b.mesh.positions[v]));
    REQUIRE(sameBits(a.mesh.layers.base[v], b.mesh.positions[v]));
    ++compared;
  }
  CHECK(compared > 1000);
}

TEST_CASE("layers: undo restores every normal on a coarse mesh with small leaves") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeCube(4));
  obj.bvh.build(obj.mesh, Bvh::Params{2});
  obj.topologyVersion = nextTopologyVersion();
  SculptLayer& l = addLayer(obj.mesh, 0.7f);
  paintPattern(obj, l, 0.03f);
  recompose(obj);
  const Snap before = snap(obj);
  Sculptor sculptor;
  DrawBrush draw;
  InflateBrush inflate;
  UndoStack stack;
  for (const Brush* brush : {static_cast<const Brush*>(&draw), static_cast<const Brush*>(&inflate)}) {
    auto undo = stroke(sculptor, obj, *brush, onLayer(obj.mesh), path(obj, {1, 0.5f, 0.5f}, {1, -0.5f, 0.5f}, 3),
                       0.6f, 1.0f);  // Grid spacing 0.5: a few vertices per dab.
    REQUIRE(undo);
    requireLayers(obj.mesh);
    requireFreshBounds(obj);  // Strength 0.7 rounds, so the commit moves vertices of tiny leaves.
    stack.push(std::move(*undo));
  }
  const Snap after = snap(obj);
  CHECK_FALSE(stack.undo(scene).empty());
  CHECK_FALSE(stack.undo(scene).empty());
  requireSnap(obj, before);
  requireLayers(obj.mesh);
  CHECK_FALSE(stack.redo(scene).empty());
  CHECK_FALSE(stack.redo(scene).empty());
  requireSnap(obj, after);
}

TEST_CASE("layers: Erase Layer fades the layer and nothing else") {
  float strength = 1.0f;
  SUBCASE("full strength") {}
  SUBCASE("strong layer") { strength = 3.0f; }
  Scene scene;
  SceneObject& obj = scene.add("A", makeIcosphere(5));
  SculptLayer& l = addLayer(obj.mesh, strength);
  paintPattern(obj, l, 0.05f);
  recompose(obj);
  const Snap before = snap(obj);
  Sculptor sculptor;
  EraseLayerBrush erase;
  const float r = 0.3f;
  StrokeOptions o = onLayer(obj.mesh);
  o.invert = true;  // Has no effect on Erase.
  sculptor.beginStroke(obj, erase, o, "Erase Layer");
  REQUIRE(sculptor.active());
  for (const Vec3& c : path(obj, {0.2f, 1, 0}, {0.6f, 0.8f, 0.3f}, 6)) {
    const std::vector<Vec3> p0 = obj.mesh.positions;
    sculptor.dab(c, r, 1.0f);
    for (Index v = 0; v < obj.mesh.vertexCount(); ++v)
      REQUIRE(glm::length(obj.mesh.positions[v] - p0[v]) <= kMaxDabMove * r * 1.0001f);
  }
  auto undo = sculptor.endStroke();
  REQUIRE(undo);
  requireLayers(obj.mesh);
  CHECK(bitwise(obj.mesh.layers.base, before.base));
  std::size_t faded = 0;
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
    const float now = glm::length(obj.mesh.layers.list[0].offset[v]);
    const float was = glm::length(before.offsets[0][v]);
    REQUIRE(now <= was * 1.000001f);
    faded += now < was * 0.9f ? 1 : 0;
  }
  CHECK(faded > 50);
  UndoStack stack;
  stack.push(std::move(*undo));
  CHECK_FALSE(stack.undo(scene).empty());
  requireSnap(obj, before);
}

TEST_CASE("layers: Smooth with This layer only smooths the layer's offsets") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeIcosphere(5));
  SculptLayer& l = addLayer(obj.mesh, 0.8f);
  std::mt19937 rng(3);
  for (Vec3& o : l.offset) o = Vec3{unit(rng) - 0.5f, unit(rng) - 0.5f, unit(rng) - 0.5f} * 0.01f;
  recompose(obj);
  const Snap before = snap(obj);
  const Vec3 c = surfacePoint(obj, {0, 0, 1});
  const float r = 0.3f;
  auto roughness = [&](const std::vector<Vec3>& offset) {
    float sum = 0.0f;
    for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
      if (glm::length(before.positions[v] - c) > r * 0.5f) continue;
      neighbourMean<Vec3>(
          obj.mesh, v, [&](Index u) { return offset[u]; },
          [&](const Vec3& mean) { sum += glm::length(mean - offset[v]); });
    }
    return sum;
  };
  Sculptor sculptor;
  SmoothBrush smooth;
  StrokeOptions o = onLayer(obj.mesh);
  o.smoothLayerOnly = true;
  CHECK(&strokeBrush(smooth, obj, o) != &smooth);
  REQUIRE(stroke(sculptor, obj, smooth, o, {c, c, c}, r, 1.0f));
  requireLayers(obj.mesh);
  CHECK(bitwise(obj.mesh.layers.base, before.base));
  CHECK(roughness(obj.mesh.layers.list[0].offset) < 0.5f * roughness(before.offsets[0]));

  // On the base the option does nothing: plain smoothing, committed into the base.
  o.layerTarget = 0;
  CHECK(&strokeBrush(smooth, obj, o) == &smooth);
  const std::vector<Vec3> offsets = obj.mesh.layers.list[0].offset;
  const std::vector<Vec3> base = obj.mesh.layers.base;
  REQUIRE(stroke(sculptor, obj, smooth, o, {c, c}, r, 1.0f));
  requireLayers(obj.mesh);
  CHECK(bitwise(obj.mesh.layers.list[0].offset, offsets));
  CHECK_FALSE(bitwise(obj.mesh.layers.base, base));
}

TEST_CASE("layers: a layer edit that rounds away in the positions is still recorded") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeIcosphere(4));
  SculptLayer& l = addLayer(obj.mesh);
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) l.offset[v] = obj.mesh.layers.base[v] * 1e-9f;
  recompose(obj);
  const Snap before = snap(obj);
  REQUIRE(bitwise(before.positions, before.base));  // The layer is too faint to show.
  Sculptor sculptor;
  EraseLayerBrush erase;
  auto undo = stroke(sculptor, obj, erase, onLayer(obj.mesh), {surfacePoint(obj, {1, 0, 0})}, 0.3f, 1.0f);
  REQUIRE(undo);
  requireLayers(obj.mesh);
  CHECK(bitwise(obj.mesh.positions, before.positions));
  CHECK_FALSE(bitwise(obj.mesh.layers.list[0].offset, before.offsets[0]));
  UndoStack stack;
  stack.push(std::move(*undo));
  CHECK_FALSE(stack.undo(scene).empty());
  requireSnap(obj, before);
}

TEST_CASE("layers: grab with symmetry on a layer, undo and redo") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeIcosphere(4));
  SculptLayer& l = addLayer(obj.mesh, 0.6f);
  paintPattern(obj, l, 0.02f);
  recompose(obj);
  addLayer(obj.mesh, 1.5f);
  const Snap before = snap(obj);
  Sculptor sculptor;
  StrokeOptions o = onLayer(obj.mesh);
  o.symmetryX = true;
  REQUIRE(sculptor.beginGrab(obj, o, surfacePoint(obj, {0.6f, 0.8f, 0}), 0.35f, "Grab"));
  sculptor.grab({0.05f, 0.1f, 0.0f});
  sculptor.grab({0.1f, 0.2f, 0.05f});
  auto undo = sculptor.endStroke();
  REQUIRE(undo);
  requireLayers(obj.mesh);
  CHECK(bitwise(obj.mesh.layers.base, before.base));
  CHECK(bitwise(obj.mesh.layers.list[0].offset, before.offsets[0]));
  const Snap after = snap(obj);
  UndoStack stack;
  stack.push(std::move(*undo));
  CHECK_FALSE(stack.undo(scene).empty());
  requireSnap(obj, before);
  CHECK_FALSE(stack.redo(scene).empty());
  requireSnap(obj, after);
}

TEST_CASE("layers: stroke undo applies only under the layer state it was recorded in") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeIcosphere(4));
  addLayer(obj.mesh);
  Sculptor sculptor;
  DrawBrush draw;
  MaskBrush mask;
  UndoStack stack;
  auto undo = stroke(sculptor, obj, draw, onLayer(obj.mesh), {surfacePoint(obj, {0, 1, 0})}, 0.3f, 1.0f);
  REQUIRE(undo);
  stack.push(std::move(*undo));
  auto maskUndo = stroke(sculptor, obj, mask, StrokeOptions{}, {surfacePoint(obj, {1, 0, 0})}, 0.3f, 1.0f);
  REQUIRE(maskUndo);
  stack.push(std::move(*maskUndo));
  // A strength change behind history's back (the real operation records an entry).
  obj.mesh.layers.list[0].strength = 0.5f;
  recompose(obj);
  const Snap changed = snap(obj);
  const std::vector<float> maskNow = obj.mesh.mask;
  CHECK_FALSE(stack.undo(scene).empty());  // The mask does not depend on the layers.
  CHECK(obj.mesh.mask != maskNow);
  CHECK(stack.undo(scene).empty());  // The stroke is skipped.
  requireSnap(obj, changed);
  requireLayers(obj.mesh);
}

// ---- Guards ------------------------------------------------------------------------------------

TEST_CASE("layers: dynamic topology leaves layered meshes alone") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeIcosphere(3));
  addLayer(obj.mesh);
  const Index faces = obj.mesh.faceCount();
  const std::uint64_t version = obj.topologyVersion;
  StrokeOptions o = onLayer(obj.mesh);
  o.dyntopo = true;
  o.dyntopoOptions.timeBudgetMs = 0.0;
  DrawBrush draw;
  Sculptor sculptor;
  sculptor.beginStroke(obj, draw, o, "Draw");
  REQUIRE(sculptor.active());
  CHECK(sculptor.dyntopo() == nullptr);  // The sculptor turns it off by itself.
  const Vec3 c = surfacePoint(obj, {0, 1, 0});
  sculptor.dab(c, 0.4f, 1.0f, DabTopology{0.02f});
  auto undo = sculptor.endStroke();
  REQUIRE(undo);
  CHECK(std::holds_alternative<SculptUndo>(*undo));
  CHECK(obj.mesh.faceCount() == faces);
  CHECK(obj.topologyVersion == version);
  requireLayers(obj.mesh);

  {
    DyntopoSession session(obj, o.dyntopoOptions);
    CHECK(session.refused());
    const DyntopoPass& pass = session.pass(c, 0.4f, DabTopology{0.02f});
    CHECK_FALSE(pass.changed());
    CHECK_FALSE(session.changedTopology());
    CHECK_FALSE(obj.bvh.dynamic());
  }
  const Snap now = snap(obj);
  LayoutWorkspace ws;
  const ConsolidateResult result = consolidate(obj.mesh, obj.bvh, {}, ws);
  CHECK(result.refused);
  CHECK(ws.mesh.positions.empty());
  LayoutSide side;
  side.bvh = std::make_shared<const Bvh>(obj.bvh);
  CHECK_FALSE(relayout(obj.mesh, obj.bvh, side, LayoutDelta{}, false, ws));
  CHECK(ws.mesh.positions.empty());
  requireSnap(obj, now);
}

TEST_CASE("layers: dynamic topology on another object never touches a layered one") {
  Scene scene;
  SceneObject& a = scene.add("A", makeIcosphere(4));
  SceneObject& b = scene.add("B", makeIcosphere(3));
  SculptLayer& l = addLayer(a.mesh, 0.7f);
  paintPattern(a, l, 0.02f);
  recompose(a);
  auto workspace = std::make_shared<LayoutWorkspace>();
  UndoStack stack;
  stack.setLayoutWorkspace(workspace);
  Sculptor sculptor;
  sculptor.setLayoutWorkspace(workspace);
  DrawBrush draw;
  auto undo = stroke(sculptor, a, draw, onLayer(a.mesh), {surfacePoint(a, {0, 1, 0})}, 0.3f, 1.0f);
  REQUIRE(undo);
  stack.push(std::move(*undo));
  const Snap layered = snap(a);
  StrokeOptions dyn;
  dyn.dyntopo = true;
  dyn.dyntopoOptions.timeBudgetMs = 0.0;
  for (int i = 0; i < 3; ++i) {
    sculptor.beginStroke(b, draw, dyn, "Draw");
    for (const Vec3& c : path(b, {1, 0.1f * i, 0}, {0.5f, 0.8f, 0.1f * i}, 4)) {
      Bvh::ClosestHit hit;
      const Index face = b.bvh.closestPoint(b.mesh, c, 0.3f, hit) ? hit.face : kInvalid;
      sculptor.dab(c, 0.3f, 1.0f, DabTopology{0.03f, face});
    }
    auto topo = sculptor.endStroke();
    REQUIRE(topo);
    CHECK(std::holds_alternative<DyntopoUndo>(*topo));
    stack.push(std::move(*topo));
  }
  for (int round = 0; round < 2; ++round) {
    for (int i = 0; i < 3; ++i) CHECK_FALSE(stack.undo(scene).empty());
    for (int i = 0; i < 3; ++i) CHECK_FALSE(stack.redo(scene).empty());
  }
  requireSnap(a, layered);
  CHECK(b.mesh.layers.empty());
  CHECK(validateLayers(b.mesh, false).ok);
  CHECK(workspace->mesh.layers.empty());
  for (int i = 0; i < 3; ++i) CHECK_FALSE(stack.undo(scene).empty());
  CHECK_FALSE(stack.undo(scene).empty());  // A's stroke still applies.
  requireLayers(a.mesh);
}

TEST_CASE("layers: remeshing bakes the layers and undo brings them back") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeIcosphere(4));
  SculptLayer& l = addLayer(obj.mesh, 1.0f);
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) l.offset[v] = obj.mesh.normals[v] * 0.2f;  // Radius 1.2.
  recompose(obj);
  const Snap before = snap(obj);
  const std::uint64_t version = obj.topologyVersion;
  const Mesh input = copyWithoutLayers(obj.mesh);
  std::optional<Mesh> result = quadRemesh(input, QuadRemeshParams{.targetEdge = 0.08f, .rounds = 0});
  REQUIRE(result);
  CHECK(result->layers.empty());
  float meanRadius = 0.0f;
  for (const Vec3& p : result->positions) meanRadius += glm::length(p);
  meanRadius /= static_cast<float>(result->vertexCount());
  CHECK(meanRadius == doctest::Approx(1.2f).epsilon(0.03));

  // As App::requestRemesh installs it.
  TopologyUndo entry;
  entry.label = "Remesh";
  entry.objectId = obj.id;
  entry.before = std::make_shared<MeshState>(MeshState{std::move(obj.mesh), std::move(obj.bvh), obj.topologyVersion, nullptr});
  obj.mesh = std::move(*result);
  obj.bvh = Bvh{};
  obj.bvh.build(obj.mesh);
  obj.topologyVersion = nextTopologyVersion();
  entry.after = std::make_shared<MeshState>(MeshState{obj.mesh, obj.bvh, obj.topologyVersion, nullptr});
  UndoStack stack;
  stack.push(std::move(entry));
  CHECK_FALSE(stack.undo(scene).empty());
  CHECK(obj.topologyVersion == version);
  requireSnap(obj, before);
  requireLayers(obj.mesh);
  CHECK_FALSE(stack.redo(scene).empty());
  CHECK(obj.mesh.layers.empty());
}

TEST_CASE("layers: compact carries the layers with their vertices") {
  Mesh m = makeQuadSphere(6);
  SculptLayer& l = addLayer(m);
  for (Index v = 0; v < m.vertexCount(); ++v) l.offset[v] = Vec3{static_cast<float>(v), 0.0f, 0.0f};  // A tag.
  const std::vector<Vec3> base = m.layers.base;
  MeshEditor ed(m);
  int collapsed = 0;
  for (Index h = 0; h < m.halfEdgeCount() && collapsed < 20; h += 37)
    if (ed.halfEdgeAlive(h)) collapsed += ed.collapseDiagonal(h) ? 1 : 0;
  REQUIRE(collapsed > 0);
  ed.compact();
  test::requireValid(m);
  REQUIRE(m.layers.base.size() == m.positions.size());
  REQUIRE(m.layers.list[0].offset.size() == m.positions.size());
  Index last = -1;
  for (Index v = 0; v < m.vertexCount(); ++v) {
    const Index was = static_cast<Index>(m.layers.list[0].offset[v].x);
    CHECK(was > last);  // Survivors keep their order.
    REQUIRE(sameBits(m.layers.base[v], base[was]));
    last = was;
  }
}
