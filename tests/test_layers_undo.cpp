#include <cmath>
#include <cstring>
#include <functional>
#include <random>

#include "TestUtil.h"
#include "mesh/Primitives.h"
#include "multires/MultiresOps.h"
#include "scene/Scene.h"
#include "sculpt/LayerOps.h"
#include "sculpt/Sculptor.h"
#include "sculpt/Undo.h"

using namespace plegl;

namespace {

bool bitwise(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(Vec3)) == 0);
}

bool sameFloat(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

void requireLayers(const Mesh& m) {
  const ValidationResult r = validateLayers(m, true);
  INFO(r.message);
  REQUIRE(r.ok);
}

// Everything about an object a layer operation may change.
struct State {
  std::vector<Vec3> positions, normals;
  std::vector<float> mask;
  LayerStack layers;
  std::vector<BvhLeaf> leaves;
};

State capture(const SceneObject& obj) {
  return {obj.mesh.positions, obj.mesh.normals, obj.mesh.mask, obj.mesh.layers,
          {obj.bvh.leaves().begin(), obj.bvh.leaves().end()}};
}

void requireState(const SceneObject& obj, const State& s) {
  CHECK(bitwise(obj.mesh.positions, s.positions));
  CHECK(bitwise(obj.mesh.normals, s.normals));
  CHECK(obj.mesh.mask == s.mask);
  const LayerStack &a = obj.mesh.layers, &b = s.layers;
  CHECK(bitwise(a.base, b.base));
  CHECK(a.active == b.active);
  CHECK(a.nextId == b.nextId);
  CHECK(a.epoch == b.epoch);
  REQUIRE(a.list.size() == b.list.size());
  for (std::size_t k = 0; k < a.list.size(); ++k) {
    INFO("layer " << k);
    CHECK(a.list[k].id == b.list[k].id);
    CHECK(a.list[k].name == b.list[k].name);
    CHECK(sameFloat(a.list[k].strength, b.list[k].strength));
    CHECK(a.list[k].visible == b.list[k].visible);
    CHECK(bitwise(a.list[k].offset, b.list[k].offset));
  }
  // Bounds must follow the positions back, or picking misses the surface after undo.
  const auto leaves = obj.bvh.leaves();
  REQUIRE(leaves.size() == s.leaves.size());
  for (std::size_t i = 0; i < leaves.size(); ++i) {
    CHECK(leaves[i].bounds.min == s.leaves[i].bounds.min);
    CHECK(leaves[i].bounds.max == s.leaves[i].bounds.max);
  }
}

Vec3 surfacePoint(const SceneObject& obj, Vec3 dir) {
  dir = glm::normalize(dir);
  RayHit hit;
  const bool ok = obj.bvh.raycast(obj.mesh, Ray{dir * 4.0f, -dir}, hit);
  REQUIRE(ok);
  return hit.position;
}

// A stroke on the target layer, as the app runs it.
SculptUndo strokeOn(SceneObject& obj, Vec3 dir, float radius = 0.35f) {
  Sculptor sculptor;
  DrawBrush draw;
  StrokeOptions o;
  o.strength = 1.0f;
  o.layerTarget = obj.mesh.layers.active;
  sculptor.beginStroke(obj, draw, o, "Draw");
  REQUIRE(sculptor.active());
  for (int i = 0; i < 4; ++i) sculptor.dab(surfacePoint(obj, dir + Vec3{0.05f * i, 0, 0}), radius, 1.0f);
  auto undo = sculptor.endStroke();
  REQUIRE(undo);
  REQUIRE(std::holds_alternative<SculptUndo>(*undo));
  return std::get<SculptUndo>(std::move(*undo));
}

// An object with three layers of sculpted detail (strengths 1, 0.5, -0.8) and an undo stack that
// starts after the setup.
struct Fixture {
  Scene scene;
  SceneObject* obj = nullptr;
  LayerWorkspace ws;
  UndoStack stack;
  std::string error;

  explicit Fixture(int layers = 3) {
    obj = &scene.add("A", makeIcosphere(4));
    const Vec3 dirs[] = {{0, 1, 0}, {1, 0.3f, 0}, {0.2f, 0.8f, 0.5f}};
    const float strengths[] = {1.0f, 0.5f, -0.8f};
    for (int k = 0; k < layers; ++k) {
      REQUIRE(addLayer(*obj, ws, &error));
      strokeOn(*obj, dirs[k]);
      REQUIRE(setLayerStrength(*obj, obj->mesh.layers.active, strengths[k], ws, &error).has_value() ==
              (strengths[k] != 1.0f));
    }
    obj->clearDirty();
    requireLayers(obj->mesh);
  }
  std::uint32_t id(int k) const { return obj->mesh.layers.list[static_cast<std::size_t>(k)].id; }

  // Runs `op`, then checks undo, redo, undo and redo again restore each side bit for bit.
  void roundTrip(const std::function<std::optional<LayerUndo>()>& op) {
    const State before = capture(*obj);
    std::optional<LayerUndo> e = op();
    INFO(error);
    REQUIRE(e);
    requireLayers(obj->mesh);
    const State after = capture(*obj);
    const std::string label = e->label;
    stack.push(std::move(*e));
    for (int round = 0; round < 2; ++round) {
      INFO("round " << round << " of " << label);
      CHECK(stack.undo(scene) == label);
      requireState(*obj, before);
      requireLayers(obj->mesh);
      CHECK(stack.redo(scene) == label);
      requireState(*obj, after);
      requireLayers(obj->mesh);
    }
  }
};

}  // namespace

TEST_CASE("layer ops: Add keeps the shape and starts the stack") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeIcosphere(3));
  obj.mesh.positions[5].x = -0.0f;
  obj.mesh.computeNormals();
  LayerWorkspace ws;
  std::string error;
  const std::vector<Vec3> positions = obj.mesh.positions;
  auto e = addLayer(obj, ws, &error);
  REQUIRE(e);
  CHECK(bitwise(obj.mesh.positions, positions));
  CHECK(bitwise(obj.mesh.layers.base, positions));
  CHECK(obj.mesh.layers.epoch != 0);
  CHECK(layerStateKey(obj.mesh.layers) != 0);
  CHECK(obj.mesh.layers.active == 1);
  CHECK(obj.mesh.layers.list[0].name == "Layer 1");
  CHECK_FALSE(e->before.hasStack);
  CHECK(e->after.hasStack);
  REQUIRE(duplicateLayer(obj, 1, ws, &error));
  CHECK(bitwise(obj.mesh.positions, positions));
  CHECK(obj.mesh.layers.list.size() == 2);
  CHECK(obj.mesh.layers.list[1].name == "Layer 1 copy");
  CHECK_FALSE(obj.mesh.layers.list[1].visible);
  CHECK(obj.mesh.layers.active == 1);  // The target stays on the original.
  requireLayers(obj.mesh);
}

TEST_CASE("layer ops: every operation undoes and redoes bit for bit") {
  Fixture f;
  SceneObject& obj = *f.obj;
  SUBCASE("add") { f.roundTrip([&] { return addLayer(obj, f.ws, &f.error); }); }
  SUBCASE("duplicate") { f.roundTrip([&] { return duplicateLayer(obj, f.id(1), f.ws, &f.error); }); }
  SUBCASE("delete") {
    f.roundTrip([&] { return deleteLayer(obj, f.id(1), f.ws, &f.error); });
    CHECK(obj.mesh.layers.list.size() == 2);
  }
  SUBCASE("rename") {
    f.roundTrip([&] { return renameLayer(obj, f.id(0), "  Pores  ", &f.error); });
    CHECK(obj.mesh.layers.list[0].name == "Pores");
    CHECK_FALSE(renameLayer(obj, f.id(0), "Pores", &f.error));
    CHECK(f.error.empty());
  }
  SUBCASE("hide") { f.roundTrip([&] { return setLayerVisible(obj, f.id(2), false, f.ws, &f.error); }); }
  SUBCASE("strength") { f.roundTrip([&] { return setLayerStrength(obj, f.id(0), 0.25f, f.ws, &f.error); }); }
  SUBCASE("strength is clamped") {
    f.roundTrip([&] { return setLayerStrength(obj, f.id(0), 50.0f, f.ws, &f.error); });
    CHECK(obj.mesh.layers.list[0].strength == kMaxLayerStrength);
    CHECK_FALSE(setLayerStrength(obj, f.id(0), std::nanf(""), f.ws, &f.error));
    CHECK_FALSE(f.error.empty());
  }
  SUBCASE("solo, then solo again shows all") {
    f.roundTrip([&] { return soloLayer(obj, f.id(1), f.ws, &f.error); });
    CHECK(obj.mesh.layers.list[0].visible == false);
    CHECK(obj.mesh.layers.list[1].visible == true);
    f.stack.clear();  // Eye steps in a row would merge.
    f.roundTrip([&] { return soloLayer(obj, f.id(1), f.ws, &f.error); });
    for (const SculptLayer& l : obj.mesh.layers.list) CHECK(l.visible);
  }
  SUBCASE("hide all and show all") {
    f.roundTrip([&] { return setAllLayersVisible(obj, false, f.ws, &f.error); });
    CHECK(bitwise(obj.mesh.positions, obj.mesh.layers.base));
    f.stack.clear();
    f.roundTrip([&] { return setAllLayersVisible(obj, true, f.ws, &f.error); });
  }
  SUBCASE("invert") { f.roundTrip([&] { return invertLayer(obj, f.id(1), f.ws, &f.error); }); }
  SUBCASE("merge down") { f.roundTrip([&] { return mergeLayerDown(obj, f.id(1), f.ws, &f.error); }); }
  SUBCASE("apply") { f.roundTrip([&] { return applyLayer(obj, f.id(1), f.ws, &f.error); }); }
  SUBCASE("apply all") { f.roundTrip([&] { return applyAllLayers(obj, f.ws, &f.error); }); }
  SUBCASE("delete every layer, then undo all of it") {
    const State start = capture(obj);
    f.roundTrip([&] { return deleteLayer(obj, f.id(0), f.ws, &f.error); });
    f.roundTrip([&] { return deleteLayer(obj, f.id(1), f.ws, &f.error); });
    f.roundTrip([&] { return deleteLayer(obj, f.id(0), f.ws, &f.error); });
    CHECK(obj.mesh.layers.empty());
    CHECK(bitwise(obj.mesh.positions, start.layers.base));  // Back to the shape without layers.
    Mesh fresh = obj.mesh;
    fresh.computeNormals();
    CHECK(bitwise(obj.mesh.normals, fresh.normals));
    CHECK(validateLayers(obj.mesh, false).ok);  // Exactly LayerStack{}.
    for (int i = 0; i < 3; ++i) CHECK_FALSE(f.stack.undo(f.scene).empty());
    requireState(obj, start);
  }
  SUBCASE("apply every layer one by one") {
    const State start = capture(obj);
    f.roundTrip([&] { return applyLayer(obj, f.id(2), f.ws, &f.error); });
    f.roundTrip([&] { return applyLayer(obj, f.id(0), f.ws, &f.error); });
    f.roundTrip([&] { return applyLayer(obj, f.id(0), f.ws, &f.error); });
    CHECK(obj.mesh.layers.empty());
    float worst = 0.0f;
    for (Index v = 0; v < obj.mesh.vertexCount(); ++v)
      worst = std::max(worst, glm::length(obj.mesh.positions[v] - start.positions[v]));
    CHECK(worst <= 1e-6f);  // The shape stays, up to rounding.
    for (int i = 0; i < 3; ++i) CHECK_FALSE(f.stack.undo(f.scene).empty());
    requireState(obj, start);
  }
}

TEST_CASE("layer ops: refusals") {
  Fixture f(2);
  SceneObject& obj = *f.obj;
  CHECK_FALSE(mergeLayerDown(obj, f.id(0), f.ws, &f.error));
  CHECK(f.error == "Nothing below to merge into. Use Apply.");
  REQUIRE(setLayerVisible(obj, f.id(0), false, f.ws, &f.error));
  CHECK_FALSE(mergeLayerDown(obj, f.id(1), f.ws, &f.error));
  CHECK(f.error == "Show both layers to merge them.");
  CHECK_FALSE(applyLayer(obj, f.id(0), f.ws, &f.error));
  CHECK(f.error == "Show the layer to apply it, or delete it.");
  CHECK_FALSE(deleteLayer(obj, 99, f.ws, &f.error));
  CHECK(f.error == "Pick a layer under Sculpt layers.");
  CHECK_FALSE(setLayerVisible(obj, f.id(0), false, f.ws, &f.error));  // Already hidden: no change.
  CHECK(f.error.empty());
  for (int k = 2; k < kMaxLayers; ++k) REQUIRE(addLayer(obj, f.ws, &f.error));
  CHECK_FALSE(addLayer(obj, f.ws, &f.error));
  CHECK(f.error == "An object (or subdivision level) can have at most 16 sculpt layers.");
  CHECK_FALSE(duplicateLayer(obj, f.id(0), f.ws, &f.error));
  REQUIRE(deleteLayer(obj, f.id(5), f.ws, &f.error));
  const std::size_t used = objectLayerBytes(obj);
  CHECK_FALSE(addLayer(obj, f.ws, &f.error, used + 100));
  CHECK(f.error.find("Not enough room") == 0);
  CHECK(addLayer(obj, f.ws, &f.error, used + obj.mesh.positions.size() * sizeof(Vec3)));
  std::string name(70, 'n');
  REQUIRE(renameLayer(obj, f.id(0), name, &f.error));
  CHECK(obj.mesh.layers.list[0].name.size() == kMaxLayerNameBytes);
}

TEST_CASE("layer ops: merge down folds strengths in and changes the shape by rounding only") {
  Fixture f;
  SceneObject& obj = *f.obj;
  const LayerStack s = obj.mesh.layers;
  const std::vector<Vec3> positions = obj.mesh.positions;
  REQUIRE(mergeLayerDown(obj, f.id(2), f.ws, &f.error));
  const SculptLayer& b = obj.mesh.layers.list[1];
  CHECK(b.id == s.list[1].id);
  CHECK(b.strength == 1.0f);
  CHECK(obj.mesh.layers.active == b.id);
  float worst = 0.0f, worstShape = 0.0f;
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
    const Vec3 want = s.list[1].offset[v] * s.list[1].strength + s.list[2].offset[v] * s.list[2].strength;
    worst = std::max(worst, glm::length(b.offset[v] - want));
    worstShape = std::max(worstShape, glm::length(obj.mesh.positions[v] - positions[v]));
  }
  CHECK(worst <= 1e-7f);
  CHECK(worstShape <= 1e-6f);
  requireLayers(obj.mesh);
}

TEST_CASE("layer ops: apply all keeps every bit of the shape") {
  Fixture f;
  SceneObject& obj = *f.obj;
  REQUIRE(setLayerVisible(obj, f.id(1), false, f.ws, &f.error));
  const std::vector<Vec3> positions = obj.mesh.positions, normals = obj.mesh.normals;
  REQUIRE(applyAllLayers(obj, f.ws, &f.error));
  CHECK(obj.mesh.layers.empty());
  CHECK(bitwise(obj.mesh.positions, positions));
  CHECK(bitwise(obj.mesh.normals, normals));
  CHECK_FALSE(applyAllLayers(obj, f.ws, &f.error));
  CHECK(f.error.empty());
}

TEST_CASE("layer ops: a strength drag returns exactly where it started") {
  Fixture f;
  SceneObject& obj = *f.obj;
  const State start = capture(obj);
  for (std::size_t limit : {kLiveNormalsLimit, std::size_t{0}}) {
    INFO("live normals limit " << limit);
    StrengthDrag drag;
    REQUIRE(drag.begin(obj, f.id(0), f.ws, &f.error, limit));
    for (float s : {0.9f, 0.6f, 0.3f, 0.7f, 1.0f}) drag.update(s);
    CHECK_FALSE(drag.end());  // Back at 1: nothing to undo.
    requireState(obj, start);
  }
  // Live and deferred normals end with the same bits.
  State ends[2];
  for (int i = 0; i < 2; ++i) {
    StrengthDrag drag;
    REQUIRE(drag.begin(obj, f.id(0), f.ws, &f.error, i == 0 ? kLiveNormalsLimit : 0));
    for (float s : {0.8f, 0.5f, 0.3f}) drag.update(s);
    auto e = drag.end();
    REQUIRE(e);
    requireLayers(obj.mesh);
    ends[i] = capture(obj);
    f.stack.push(std::move(*e));
    CHECK(f.stack.undo(f.scene) == "Layer Strength");
    requireState(obj, start);
  }
  CHECK(bitwise(ends[0].positions, ends[1].positions));
  CHECK(bitwise(ends[0].normals, ends[1].normals));
  // Cancel puts everything back.
  StrengthDrag drag;
  REQUIRE(drag.begin(obj, f.id(1), f.ws, &f.error));
  drag.update(-2.0f);
  drag.cancel();
  CHECK_FALSE(drag.active());
  requireState(obj, start);
}

TEST_CASE("layer ops: eye toggles in a row merge into one step") {
  Fixture g;
  SceneObject& obj = *g.obj;
  const std::size_t n = g.stack.size();
  SUBCASE("two toggles leave nothing") {
    g.stack.push(*setLayerVisible(obj, g.id(0), false, g.ws, &g.error));
    CHECK(g.stack.size() == n + 1);
    g.stack.push(*setLayerVisible(obj, g.id(0), true, g.ws, &g.error));
    CHECK(g.stack.size() == n);
  }
  SUBCASE("three toggles leave one, undone in one step") {
    const State start = capture(obj);
    g.stack.push(*setLayerVisible(obj, g.id(0), false, g.ws, &g.error));
    g.stack.push(*soloLayer(obj, g.id(1), g.ws, &g.error));
    g.stack.push(*setAllLayersVisible(obj, false, g.ws, &g.error));
    CHECK(g.stack.size() == n + 1);
    CHECK(g.stack.undo(g.scene) == "Layer Visibility");
    requireState(obj, start);
    requireLayers(obj.mesh);
  }
  SUBCASE("a stroke in between keeps them apart") {
    g.stack.push(*setLayerVisible(obj, g.id(1), false, g.ws, &g.error));
    g.stack.push(strokeOn(obj, {0, 0, 1}));  // On layer 3.
    g.stack.push(*setLayerVisible(obj, g.id(1), true, g.ws, &g.error));
    CHECK(g.stack.size() == n + 3);
  }
}

TEST_CASE("layer ops: entries apply only to the state they left") {
  Fixture f;
  SceneObject& obj = *f.obj;
  auto e = renameLayer(obj, f.id(0), "Skin", &f.error);
  REQUIRE(e);
  f.stack.push(std::move(*e));
  obj.mesh.layers.list[0].name = "Changed behind history's back";
  const State now = capture(obj);
  CHECK(f.stack.undo(f.scene).empty());
  requireState(obj, now);
  // Strokes recorded under other layer settings are skipped too.
  Fixture g;
  g.stack.push(strokeOn(*g.obj, {0, 0, 1}));
  g.obj->mesh.layers.list[0].strength = 0.3f;  // No entry: the stroke's key no longer matches.
  CHECK(g.stack.undo(g.scene).empty());
}

TEST_CASE("layer ops: selecting a layer is not an edit") {
  Fixture f;
  SceneObject& obj = *f.obj;
  const std::uint64_t key = layerStateKey(obj.mesh.layers);
  CHECK(selectLayer(obj, f.id(0)));
  CHECK(obj.mesh.layers.active == f.id(0));
  CHECK(selectLayer(obj, 0));
  CHECK_FALSE(selectLayer(obj, 77));
  CHECK(obj.mesh.layers.active == 0);
  CHECK(layerStateKey(obj.mesh.layers) == key);
  // Undoing an entry restores the selection of its side, and checks nothing about the current one.
  auto e = setLayerStrength(obj, f.id(2), 0.1f, f.ws, &f.error);
  REQUIRE(e);
  f.stack.push(std::move(*e));
  CHECK(selectLayer(obj, f.id(1)));
  CHECK_FALSE(f.stack.undo(f.scene).empty());
}

TEST_CASE("layer ops: mask from layer") {
  Fixture f;
  SceneObject& obj = *f.obj;
  auto e = maskFromLayer(obj, f.id(0), &f.error);
  REQUIRE(e);
  float most = 0.0f;
  for (float m : obj.mesh.mask) {
    CHECK(m >= 0.0f);
    CHECK(m <= 1.0f);
    most = std::max(most, m);
  }
  CHECK(most == 1.0f);
  REQUIRE(addLayer(obj, f.ws, &f.error));
  CHECK_FALSE(maskFromLayer(obj, obj.mesh.layers.active, &f.error));
  CHECK(f.error == "Layer 'Layer 4' is empty.");
  UndoStack stack;
  stack.push(std::move(*e));
  CHECK(stack.undo(f.scene) == "Mask from Layer");
  for (float m : obj.mesh.mask) CHECK(m == 0.0f);
}

TEST_CASE("layer ops: duplicated objects get layer stacks of their own") {
  Fixture f;
  SceneObject* copy = f.scene.duplicate(f.obj->id);
  REQUIRE(copy);
  requireLayers(copy->mesh);
  CHECK(copy->mesh.layers.epoch != f.obj->mesh.layers.epoch);
  CHECK(layerStateKey(copy->mesh.layers) != layerStateKey(f.obj->mesh.layers));
  const State original = capture(*f.obj);
  REQUIRE(setLayerStrength(*copy, copy->mesh.layers.list[0].id, 0.2f, f.ws, &f.error));
  REQUIRE(deleteLayer(*copy, copy->mesh.layers.list[1].id, f.ws, &f.error));
  requireState(*f.obj, original);
}

namespace {

// Every level's values (or the object's, without levels), to compare whole objects bit for bit.
struct ObjectState {
  int active = -1;
  std::vector<State> levels;
};

ObjectState captureAll(const SceneObject& obj) {
  ObjectState s;
  if (!obj.multires) {
    s.levels.push_back(capture(obj));
    return s;
  }
  s.active = obj.multires->active;
  for (int k = 0; k < obj.multires->levelCount(); ++k) {
    const Mesh& m = k == s.active ? obj.mesh : obj.multires->levels[static_cast<std::size_t>(k)].mesh;
    s.levels.push_back({m.positions, m.normals, m.mask, m.layers, {}});
  }
  return s;
}

void requireSameObject(const ObjectState& a, const ObjectState& b) {
  CHECK(a.active == b.active);
  REQUIRE(a.levels.size() == b.levels.size());
  for (std::size_t k = 0; k < a.levels.size(); ++k) {
    INFO("level " << k);
    const State &x = a.levels[k], &y = b.levels[k];
    CHECK(bitwise(x.positions, y.positions));
    CHECK(bitwise(x.normals, y.normals));
    // An undone mask edit may leave an all-zero mask where there was none; both mean unmasked.
    auto expanded = [](const State& st) {
      return st.mask.empty() ? std::vector<float>(st.positions.size(), 0.0f) : st.mask;
    };
    CHECK(expanded(x) == expanded(y));
    CHECK(x.layers.list.size() == y.layers.list.size());
    CHECK(bitwise(x.layers.base, y.layers.base));
  }
}

}  // namespace

TEST_CASE("layer fuzz: strokes, operations, drags, level steps, undo and redo stay consistent") {
  for (int levels : {0, 3}) {
    INFO("levels " << levels);
    Scene scene;
    SyncWorkspace sws;
    LayerWorkspace ws;
    UndoStack stack;
    // About 20K vertices on the sculpted level either way.
    SceneObject& obj = scene.add("Fuzz", makeQuadSphere(levels ? 7 : 58));
    for (int i = 0; i < levels; ++i) REQUIRE(subdivideObject(obj, sws));
    const ObjectState initial = captureAll(obj);
    std::mt19937 rng(levels ? 4242u : 77u);
    auto pick = [&](int n) { return static_cast<int>(rng() % static_cast<unsigned>(n)); };
    auto uniform = [&](float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); };
    auto randomLayer = [&]() -> std::uint32_t {
      const LayerStack& s = obj.mesh.layers;
      return s.empty() ? 0 : s.list[static_cast<std::size_t>(pick(static_cast<int>(s.list.size())))].id;
    };
    std::string error;
    auto push = [&](auto entry) {
      if (entry) stack.push(std::move(*entry));
    };
    static DrawBrush draw;
    static SmoothBrush smooth;
    static EraseLayerBrush erase;
    int strokes = 0, ops = 0, undos = 0;
    for (int step = 0; step < 300; ++step) {
      INFO("step " << step);
      const int kind = pick(levels ? 16 : 15);
      const std::uint32_t id = randomLayer();
      switch (kind) {
        case 0:
        case 1:
        case 2: {  // A stroke on a random target with a random brush.
          if (pick(3) == 0) selectLayer(obj, pick(4) == 0 ? 0 : id);
          StrokeOptions o;
          o.layerTarget = obj.mesh.layers.active;
          o.smoothLayerOnly = pick(2) == 0;
          o.symmetryX = pick(2) == 0;
          const int b = pick(4);
          const Brush* brush = b == 0 ? nullptr : b == 1 ? static_cast<const Brush*>(&smooth)
                                     : b == 2 ? static_cast<const Brush*>(&erase) : &draw;
          if (!layerTargetRefusal(obj, brush, o.layerTarget).empty()) break;
          const Vec3 dir{uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)};
          if (glm::length(dir) < 0.1f) break;
          const Vec3 c = surfacePoint(obj, dir);
          Sculptor sculptor;
          if (brush) {
            sculptor.beginStroke(obj, *brush, o, brush->name());
            for (int i = 0; i < 3; ++i) sculptor.dab(c + Vec3{0.03f * i, 0, 0}, uniform(0.1f, 0.4f), uniform(0.2f, 1));
          } else {
            if (!sculptor.beginGrab(obj, o, c, uniform(0.2f, 0.5f), "Grab")) break;
            sculptor.grab(Vec3{uniform(-0.1f, 0.1f), uniform(-0.1f, 0.1f), uniform(-0.1f, 0.1f)});
          }
          auto entry = sculptor.endStroke();
          if (entry) stack.push(std::move(*entry));
          ++strokes;
          break;
        }
        case 3: push(addLayer(obj, ws, &error)); ++ops; break;
        case 4: if (id) push(duplicateLayer(obj, id, ws, &error)); ++ops; break;
        case 5: if (id) push(deleteLayer(obj, id, ws, &error)); ++ops; break;
        case 6: if (id) push(setLayerVisible(obj, id, pick(2) == 0, ws, &error)); ++ops; break;
        case 7: if (id) push(setLayerStrength(obj, id, uniform(-2, 2), ws, &error)); ++ops; break;
        case 8:
          if (id) push(pick(2) == 0 ? soloLayer(obj, id, ws, &error) : setAllLayersVisible(obj, pick(2) == 0, ws, &error));
          ++ops;
          break;
        case 9: if (id) push(pick(2) == 0 ? invertLayer(obj, id, ws, &error) : renameLayer(obj, id, "R", &error)); ++ops; break;
        case 10: if (id) push(mergeLayerDown(obj, id, ws, &error)); ++ops; break;
        case 11:
          if (id) push(pick(4) == 0 ? applyAllLayers(obj, ws, &error) : applyLayer(obj, id, ws, &error));
          ++ops;
          break;
        case 12:
          if (id) {
            if (pick(2) == 0) {
              push(maskFromLayer(obj, id, &error));
            } else {  // A drag, live or with normals left for the end.
              StrengthDrag drag;
              if (drag.begin(obj, id, ws, &error, pick(2) == 0 ? kLiveNormalsLimit : 10)) {
                for (int i = 0; i < 3; ++i) drag.update(uniform(-1.5f, 1.5f));
                if (pick(5) == 0)
                  drag.cancel();
                else
                  push(drag.end());
              }
            }
          }
          ++ops;
          break;
        case 13: stack.undo(scene); ++undos; break;
        case 14: stack.redo(scene); break;
        case 15: push(setActiveLevel(obj, pick(levels + 1), sws)); break;
      }
      {
        const ValidationResult r = validateLayers(obj.mesh, true);
        INFO(r.message);
        REQUIRE(r.ok);
      }
      if (obj.multires) {
        const ValidationResult r = validateMultires(obj);
        INFO(r.message);
        REQUIRE(r.ok);
      }
    }
    CHECK(strokes > 30);
    CHECK(ops > 100);
    CHECK(undos > 10);
    // Everything undoes back to the start, bit for bit.
    while (!stack.undo(scene).empty()) {
    }
    requireSameObject(captureAll(obj), initial);
    CHECK(obj.mesh.layers.empty());
  }
}
