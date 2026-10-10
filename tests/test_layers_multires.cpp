#include <cmath>
#include <cstring>
#include <limits>

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

const Mesh& levelMesh(const SceneObject& o, int k) {
  return k == o.multires->active ? o.mesh : o.multires->levels[static_cast<std::size_t>(k)].mesh;
}

void requireValidLevels(const SceneObject& o) {
  const ValidationResult r = validateMultires(o);
  INFO(r.message);
  REQUIRE(r.ok);
}

Vec3 surfacePoint(const SceneObject& obj, Vec3 dir) {
  dir = glm::normalize(dir);
  RayHit hit;
  REQUIRE(obj.bvh.raycast(obj.mesh, Ray{dir * 3.0f, -dir}, hit, std::numeric_limits<float>::infinity(), true));
  return hit.position;
}

// A Draw stroke on the active level, on its target layer (or the base when it has none).
void stroke(SceneObject& obj, Vec3 dir, float radius = 0.35f, float strength = 0.7f) {
  static DrawBrush draw;
  Sculptor sculptor;
  StrokeOptions o;
  o.layerTarget = obj.mesh.layers.active;
  sculptor.beginStroke(obj, draw, o, "Draw");
  REQUIRE(sculptor.active());
  const Vec3 c = surfacePoint(obj, dir);
  for (int i = 0; i < 3; ++i) sculptor.dab(c, radius, strength);
  REQUIRE(sculptor.endStroke());
}

// A new layer on the active level with a stroke on it.
void sculptLayer(SceneObject& obj, LayerWorkspace& lws, Vec3 dir, float strength = 1.0f) {
  std::string error;
  REQUIRE(addLayer(obj, lws, &error));
  stroke(obj, dir);
  if (strength != 1.0f) REQUIRE(setLayerStrength(obj, obj.mesh.layers.active, strength, lws, &error));
}

void subdivide(SceneObject& obj, SyncWorkspace& ws, int times) {
  for (int i = 0; i < times; ++i) {
    std::string error;
    auto u = subdivideObject(obj, ws, &error, 256);
    INFO(error);
    REQUIRE(u);
  }
}

SceneObject& addSphere(Scene& scene, const char* name) {
  SceneObject& obj = scene.add(name, makeQuadSphere(6));
  obj.bvh.build(obj.mesh, Bvh::Params{256});
  obj.topologyVersion = nextTopologyVersion();
  return obj;
}

// Every level's positions and layers.
struct Levels {
  std::vector<std::vector<Vec3>> pos, base;
  std::vector<std::vector<std::vector<Vec3>>> offs;
};

Levels capture(const SceneObject& o) {
  Levels s;
  for (int k = 0; k < o.multires->levelCount(); ++k) {
    const Mesh& m = levelMesh(o, k);
    s.pos.push_back(m.positions);
    s.base.push_back(m.layers.base);
    s.offs.emplace_back();
    for (const SculptLayer& l : m.layers.list) s.offs.back().push_back(l.offset);
  }
  return s;
}

void requireSame(const Levels& a, const Levels& b) {
  REQUIRE(a.pos.size() == b.pos.size());
  for (std::size_t k = 0; k < a.pos.size(); ++k) {
    INFO("level " << k);
    CHECK(bitwise(a.pos[k], b.pos[k]));
    CHECK(bitwise(a.base[k], b.base[k]));
    REQUIRE(a.offs[k].size() == b.offs[k].size());
    for (std::size_t l = 0; l < a.offs[k].size(); ++l) CHECK(bitwise(a.offs[k][l], b.offs[k][l]));
  }
}

float worstRelative(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
  REQUIRE(a.size() == b.size());
  float worst = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i)
    worst = std::max(worst, glm::length(a[i] - b[i]) / std::max(1.0f, glm::length(b[i])));
  return worst;
}

}  // namespace

TEST_CASE("layers and levels: subdividing keeps the layers on level 0 and starts the new level without") {
  Scene scene;
  SyncWorkspace ws;
  LayerWorkspace lws;
  SceneObject& a = addSphere(scene, "A");
  sculptLayer(a, lws, {0, 1, 0});
  sculptLayer(a, lws, {1, 0.2f, 0}, -0.5f);
  SceneObject& b = scene.add("B", copyWithoutLayers(a.mesh));
  b.bvh.build(b.mesh, Bvh::Params{256});
  b.topologyVersion = nextTopologyVersion();
  const LayerStack before = a.mesh.layers;
  subdivide(a, ws, 1);
  subdivide(b, ws, 1);
  requireValidLevels(a);
  const Mesh& base = a.multires->levels[0].mesh;
  REQUIRE(base.layers.list.size() == 2);
  CHECK(bitwise(base.layers.base, before.base));
  CHECK(bitwise(base.layers.list[1].offset, before.list[1].offset));
  CHECK(a.mesh.layers.empty());
  // The new level is the subdivided composite.
  CHECK(bitwise(a.mesh.positions, b.mesh.positions));
}

TEST_CASE("layers and levels: the up fold keeps layered levels composite and in step with plain ones") {
  Scene scene;
  SyncWorkspace ws;
  LayerWorkspace lws;
  // A keeps its top-level detail in layers, B the same detail in the base.
  SceneObject& a = addSphere(scene, "A");
  SceneObject& b = addSphere(scene, "B");
  subdivide(a, ws, 2);
  subdivide(b, ws, 2);
  const Vec3 dirs[] = {{0, 1, 0.2f}, {0.8f, 0.4f, 0}, {-0.3f, 0.9f, 0.3f}};
  for (const Vec3& d : dirs) {
    sculptLayer(a, lws, d);
    stroke(b, d);
  }
  REQUIRE(worstRelative(a.mesh.positions, b.mesh.positions) < 1e-5f);
  for (SceneObject* o : {&a, &b}) {
    REQUIRE(setActiveLevel(*o, 1, ws));
    stroke(*o, {0.1f, 1, 0.1f}, 0.6f, 0.9f);
    REQUIRE(setActiveLevel(*o, 2, ws));
  }
  requireValidLevels(a);
  REQUIRE(a.mesh.layers.list.size() == 3);
  CHECK(worstRelative(a.mesh.positions, b.mesh.positions) < 1e-5f);
  // The stroke on level 1 really reached level 2.
  CHECK(!bitwise(a.mesh.layers.base, a.multires->levels[0].mesh.positions));
}

TEST_CASE("layers and levels: layer detail turns with the surface") {
  Scene scene;
  SyncWorkspace ws;
  LayerWorkspace lws;
  SceneObject& obj = addSphere(scene, "A");
  subdivide(obj, ws, 2);
  sculptLayer(obj, lws, {0, 1, 0});
  sculptLayer(obj, lws, {1, 0, 0}, 0.5f);
  const LayerStack before = obj.mesh.layers;
  REQUIRE(setActiveLevel(obj, 1, ws));
  const Mat3 r = Mat3(glm::rotate(Mat4(1.0f), glm::radians(30.0f), Vec3{0, 0, 1}));
  for (Vec3& p : obj.mesh.positions) p = r * p;
  obj.mesh.computeNormals();
  REQUIRE(setActiveLevel(obj, 2, ws));
  requireValidLevels(obj);
  const LayerStack& after = obj.mesh.layers;
  REQUIRE(after.list.size() == before.list.size());
  std::size_t moved = 0;
  float worstDir = 0.0f, worstLen = 0.0f;
  for (std::size_t l = 0; l < after.list.size(); ++l) {
    for (std::size_t v = 0; v < after.base.size(); ++v) {
      const Vec3& o = before.list[l].offset[v];
      const Vec3& n = after.list[l].offset[v];
      if (isZero(o)) {
        CHECK(sameBits(n, o));  // Zero offsets stay exactly zero.
        continue;
      }
      ++moved;
      worstDir = std::max(worstDir, glm::length(n - r * o));
      worstLen = std::max(worstLen, std::abs(glm::length(n) - glm::length(o)));
    }
  }
  CHECK(moved > 0);
  CHECK(worstDir < 1e-4f);
  CHECK(worstLen < 1e-6f);
  // The base turned too: the composite is the turned shape.
  float worst = 0.0f;
  for (std::size_t v = 0; v < after.base.size(); ++v)
    worst = std::max(worst, glm::length(obj.mesh.positions[v] - r * composeVertex(before, static_cast<Index>(v))));
  CHECK(worst < 1e-4f);
}

TEST_CASE("layers and levels: edits from above move the base of a layered level and keep its layers") {
  Scene scene;
  SyncWorkspace ws;
  LayerWorkspace lws;
  SceneObject& a = addSphere(scene, "A");
  sculptLayer(a, lws, {0, 1, 0});
  sculptLayer(a, lws, {0.3f, 0.2f, 1}, 2.0f);
  // B starts as A's composite without layers, so both get the same level 1.
  SceneObject& b = scene.add("B", copyWithoutLayers(a.mesh));
  b.bvh.build(b.mesh, Bvh::Params{256});
  b.topologyVersion = nextTopologyVersion();
  subdivide(a, ws, 1);
  subdivide(b, ws, 1);
  REQUIRE(bitwise(a.mesh.positions, b.mesh.positions));
  const LayerStack before = a.multires->levels[0].mesh.layers;
  const std::vector<Vec3> plainBefore = b.multires->levels[0].mesh.positions;
  for (SceneObject* o : {&a, &b}) {
    stroke(*o, {0.2f, 1, 0}, 0.5f, 0.9f);
    REQUIRE(setActiveLevel(*o, 0, ws));
  }
  requireValidLevels(a);
  const LayerStack& after = a.mesh.layers;
  REQUIRE(after.list.size() == 2);
  for (std::size_t l = 0; l < 2; ++l) CHECK(bitwise(after.list[l].offset, before.list[l].offset));
  // The base moved by what the plain object moved.
  std::size_t moved = 0;
  float worst = 0.0f;
  for (std::size_t v = 0; v < after.base.size(); ++v) {
    const Vec3 d = b.mesh.positions[v] - plainBefore[v];
    if (!isZero(d)) ++moved;
    worst = std::max(worst, glm::length((after.base[v] - before.base[v]) - d));
  }
  CHECK(moved > 0);
  CHECK(worst < 1e-6f);
}

TEST_CASE("layers and levels: undoing a level step restores parked bases and offsets") {
  Scene scene;
  SyncWorkspace ws;
  LayerWorkspace lws;
  UndoStack undo;
  SceneObject& obj = addSphere(scene, "A");
  sculptLayer(obj, lws, {0, 1, 0});
  subdivide(obj, ws, 2);
  sculptLayer(obj, lws, {1, 0.2f, 0}, 0.5f);
  sculptLayer(obj, lws, {0, 0.3f, 1}, -1.0f);
  auto u = setActiveLevel(obj, 1, ws);
  REQUIRE(u);
  undo.push(std::move(*u));
  stroke(obj, {0.3f, 1, 0.2f}, 0.6f, 0.9f);
  const Levels edited = capture(obj);
  u = setActiveLevel(obj, 0, ws);  // Carries the stroke up into level 2 and down into level 0.
  REQUIRE(u);
  undo.push(std::move(*u));
  requireValidLevels(obj);
  const Levels synced = capture(obj);
  CHECK(!bitwise(synced.base[0], edited.base[0]));
  CHECK(!bitwise(synced.base[2], edited.base[2]));
  for (int round = 0; round < 2; ++round) {
    INFO("round " << round);
    REQUIRE(!undo.undo(scene).empty());
    requireSame(capture(obj), edited);
    requireValidLevels(obj);
    REQUIRE(!undo.redo(scene).empty());
    requireSame(capture(obj), synced);
    requireValidLevels(obj);
  }
}

TEST_CASE("layers and levels: Apply and Merge Down do not turn rounding into pending edits") {
  Scene scene;
  SyncWorkspace ws;
  LayerWorkspace lws;
  UndoStack undo;
  SceneObject& obj = addSphere(scene, "A");
  subdivide(obj, ws, 1);
  sculptLayer(obj, lws, {0, 1, 0}, 0.7f);
  sculptLayer(obj, lws, {0.2f, 1, 0.3f}, 0.3f);
  sculptLayer(obj, lws, {0.3f, 1, -0.2f}, -0.6f);  // All three overlap, so sums round.
  // Sync, so nothing is pending.
  REQUIRE(setActiveLevel(obj, 0, ws));
  REQUIRE(setActiveLevel(obj, 1, ws));
  REQUIRE(diffActive(*obj.multires, obj.mesh).empty());
  std::string error;
  SUBCASE("no pending edits") {
    const std::vector<Vec3> ref = obj.multires->reference.positions;
    // The middle layer: summed in a new order, so some composites round differently.
    auto e = applyLayer(obj, obj.mesh.layers.list[1].id, lws, &error);
    INFO(error);
    REQUIRE(e);
    undo.push(std::move(*e));
    CHECK(!bitwise(obj.mesh.positions, ref));
    CHECK(diffActive(*obj.multires, obj.mesh).empty());
    const std::vector<Vec3> applied = obj.mesh.positions;
    auto m = mergeLayerDown(obj, obj.mesh.layers.list[1].id, lws, &error);
    REQUIRE(m);
    undo.push(std::move(*m));
    CHECK(!bitwise(obj.mesh.positions, applied));
    CHECK(diffActive(*obj.multires, obj.mesh).empty());
    requireValidLevels(obj);
    CHECK(undo.undo(scene) == "Merge Down");
    CHECK(undo.undo(scene) == "Apply Layer");
    CHECK(bitwise(obj.multires->reference.positions, ref));
    CHECK(diffActive(*obj.multires, obj.mesh).empty());
  }
  SUBCASE("a pending edit keeps its reference") {
    // A pending edit on a vertex the first layer moves.
    const LayerStack& s = obj.mesh.layers;
    Index v = 0;
    while (isZero(s.list[1].offset[v])) ++v;
    obj.mesh.layers.base[v] += Vec3{0.01f, 0, 0};
    obj.mesh.positions[v] = composeVertex(obj.mesh.layers, v);
    obj.mesh.computeNormals();
    const Vec3 refV = obj.multires->reference.positions[v];
    const std::vector<Vec3> ref = obj.multires->reference.positions;
    auto e = applyLayer(obj, s.list[1].id, lws, &error);
    REQUIRE(e);
    CHECK(sameBits(obj.multires->reference.positions[v], refV));
    CHECK(!bitwise(obj.multires->reference.positions, ref));  // Others moved along.
    const SyncDelta d = diffActive(*obj.multires, obj.mesh);
    REQUIRE(d.levels.size() == 1);
    CHECK(d.levels[0].posIndex == std::vector<Index>{v});
  }
}

TEST_CASE("layers and levels: deleting levels keeps their layers through undo") {
  Scene scene;
  SyncWorkspace ws;
  LayerWorkspace lws;
  UndoStack undo;
  SceneObject& obj = addSphere(scene, "A");
  sculptLayer(obj, lws, {0, 1, 0});
  subdivide(obj, ws, 3);
  sculptLayer(obj, lws, {1, 0, 0}, 0.5f);
  REQUIRE(setActiveLevel(obj, 1, ws));
  sculptLayer(obj, lws, {0, 0, 1});
  const Levels start = capture(obj);
  SUBCASE("higher") {
    auto u = deleteHigherLevels(obj);
    REQUIRE(u);
    undo.push(std::move(*u));
    CHECK(obj.multires->levelCount() == 2);
    CHECK(obj.mesh.layers.list.size() == 1);
    requireValidLevels(obj);
    CHECK(undo.undo(scene) == "Delete Higher Levels");
    requireSame(capture(obj), start);
    requireValidLevels(obj);
    CHECK(undo.redo(scene) == "Delete Higher Levels");
    CHECK(obj.multires->levelCount() == 2);
  }
  SUBCASE("lower") {
    auto u = deleteLowerLevels(obj);
    REQUIRE(u);
    undo.push(std::move(*u));
    CHECK(obj.multires->levelCount() == 3);
    CHECK(obj.mesh.layers.list.size() == 1);
    requireValidLevels(obj);
    CHECK(undo.undo(scene) == "Delete Lower Levels");
    requireSame(capture(obj), start);
    requireValidLevels(obj);
    CHECK(undo.redo(scene) == "Delete Lower Levels");
    CHECK(obj.multires->levelCount() == 3);
  }
}
