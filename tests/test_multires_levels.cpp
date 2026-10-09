#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <set>

#include "TestUtil.h"
#include "mesh/Primitives.h"
#include "multires/MultiresOps.h"
#include "scene/Scene.h"
#include "sculpt/Sculptor.h"

using namespace plegl;

namespace {

bool sameBits(const Vec3& a, const Vec3& b) { return std::memcmp(&a, &b, sizeof(Vec3)) == 0; }

template <class T>
bool sameArray(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

const Mesh& levelMesh(const SceneObject& o, int k) {
  return k == o.multires->active ? o.mesh : o.multires->levels[static_cast<std::size_t>(k)].mesh;
}
Mesh& levelMesh(SceneObject& o, int k) {
  return k == o.multires->active ? o.mesh : o.multires->levels[static_cast<std::size_t>(k)].mesh;
}

// Every level's values, with empty channels expanded, so states compare regardless of which level
// is active.
struct Snapshot {
  struct Level {
    std::vector<Vec3> pos, nrm;
    std::vector<float> mask;
    std::vector<std::int32_t> sets;
    std::uint64_t version = 0, layout = 0;
  };
  std::vector<Level> levels;
};

Snapshot snapshot(const SceneObject& o) {
  Snapshot s;
  for (int k = 0; k < o.multires->levelCount(); ++k) {
    const Mesh& m = levelMesh(o, k);
    Snapshot::Level l;
    l.pos = m.positions;
    l.nrm = m.normals;
    l.mask = m.mask.empty() ? std::vector<float>(m.positions.size(), 0.0f) : m.mask;
    l.sets = m.faceSets.empty() ? std::vector<std::int32_t>(m.faceHe.size(), kDefaultFaceSet) : m.faceSets;
    l.version = o.multires->levels[static_cast<std::size_t>(k)].version;
    l.layout = layoutHash(k == o.multires->active ? o.bvh : o.multires->levels[static_cast<std::size_t>(k)].bvh);
    s.levels.push_back(std::move(l));
  }
  return s;
}

void requireSame(const Snapshot& a, const Snapshot& b) {
  REQUIRE(a.levels.size() == b.levels.size());
  for (std::size_t k = 0; k < a.levels.size(); ++k) {
    INFO("level " << k);
    CHECK(sameArray(a.levels[k].pos, b.levels[k].pos));
    CHECK(sameArray(a.levels[k].nrm, b.levels[k].nrm));
    CHECK(sameArray(a.levels[k].mask, b.levels[k].mask));
    CHECK(sameArray(a.levels[k].sets, b.levels[k].sets));
    CHECK(a.levels[k].version == b.levels[k].version);
    CHECK(a.levels[k].layout == b.levels[k].layout);
  }
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

std::optional<SculptUndo> stroke(SceneObject& obj, Vec3 dir, float radius, float strength = 0.6f,
                                 const StrokeOptions& opts = {}) {
  static DrawBrush draw;
  Sculptor sculptor;
  sculptor.beginStroke(obj, draw, opts, "Draw");
  const Vec3 c = surfacePoint(obj, dir);
  for (int i = 0; i < 3; ++i) sculptor.dab(c, radius, strength);
  auto undo = sculptor.endStroke();
  if (!undo) return std::nullopt;
  return std::get<SculptUndo>(std::move(*undo));
}

// A quad sphere with `levels` levels above it, active at the top.
SceneObject& addLeveled(Scene& scene, SyncWorkspace& ws, int levels, int resolution = 6, int leafFaces = 256) {
  Mesh m = makeQuadSphere(resolution);
  SceneObject& obj = scene.add("Head", std::move(m));
  if (leafFaces != 1024) {
    obj.bvh.build(obj.mesh, Bvh::Params{leafFaces});
    obj.topologyVersion = nextTopologyVersion();
  }
  for (int i = 0; i < levels; ++i) {
    std::string error;
    auto u = subdivideObject(obj, ws, &error, leafFaces);
    INFO(error);
    REQUIRE(u);
  }
  return obj;
}

void step(SceneObject& obj, int target, UndoStack& undo, SyncWorkspace& ws) {
  auto u = setActiveLevel(obj, target, ws);
  REQUIRE(u);
  undo.push(std::move(*u));
}

void moveVertices(SceneObject& obj, const std::function<bool(const Vec3&)>& select, const Vec3& offset) {
  for (Vec3& p : obj.mesh.positions)
    if (select(p)) p += offset;
  obj.mesh.computeNormals();
}

}  // namespace

TEST_CASE("levels: subdividing builds a valid stack and keeps the base version") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = scene.add("Head", makeQuadSphere(4));
  const std::uint64_t baseVersion = obj.topologyVersion;
  const Mesh base = obj.mesh;
  auto u = subdivideObject(obj, ws);
  REQUIRE(u);
  requireValidLevels(obj);
  CHECK(obj.multires->levelCount() == 2);
  CHECK(obj.multires->active == 1);
  CHECK(obj.multires->levels[0].version == baseVersion);
  CHECK(obj.topologyVersion != baseVersion);
  CHECK(sameArray(obj.multires->levels[0].mesh.positions, base.positions));
  CHECK(obj.mesh.faceCount() == 4 * base.faceCount());
  REQUIRE(subdivideObject(obj, ws));
  REQUIRE(subdivideObject(obj, ws));
  requireValidLevels(obj);
  CHECK(obj.multires->levelCount() == 4);
  CHECK(obj.mesh.faceCount() == 64 * base.faceCount());
  // Subdivide works from the top only.
  REQUIRE(setActiveLevel(obj, 1, ws));
  std::string error;
  CHECK_FALSE(prepareSubdivide(obj, &error));
  CHECK(!error.empty());
}

TEST_CASE("levels: steps without edits are bit-exact") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = addLeveled(scene, ws, 3);
  // Some mask and face sets on the top level, synced down by a first step.
  obj.mesh.ensureMask();
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) obj.mesh.mask[v] = obj.mesh.positions[v].y > 0.3f ? 1.0f : 0.0f;
  obj.mesh.ensureFaceSets();
  for (Index f = 0; f < obj.mesh.faceCount(); ++f) obj.mesh.faceSets[f] = obj.mesh.faceCentroid(f).x > 0.2f ? 3 : 1;
  UndoStack undo;
  step(obj, 0, undo, ws);
  step(obj, 3, undo, ws);
  requireValidLevels(obj);
  const Snapshot start = snapshot(obj);
  const std::size_t entries = undo.size();
  std::mt19937 rng(11);
  for (int i = 0; i < 100; ++i) {
    int target = static_cast<int>(rng() % 4);
    if (target == obj.multires->active) target = (target + 1) % 4;
    step(obj, target, undo, ws);
  }
  requireSame(snapshot(obj), start);
  // The steps merged into at most one entry.
  CHECK(undo.size() <= entries + 1);
  requireValidLevels(obj);
}

TEST_CASE("levels: consecutive level steps merge") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = addLeveled(scene, ws, 3);
  UndoStack undo;
  step(obj, 2, undo, ws);
  step(obj, 1, undo, ws);
  CHECK(undo.size() == 1);
  step(obj, 2, undo, ws);
  step(obj, 3, undo, ws);
  CHECK(undo.size() == 0);  // Back where it started: nothing to undo.
  step(obj, 2, undo, ws);
  auto s = stroke(obj, {0, 1, 0}, 0.4f);
  REQUIRE(s);
  undo.push(std::move(*s));
  step(obj, 1, undo, ws);
  step(obj, 0, undo, ws);
  CHECK(undo.size() == 3);  // Step, stroke, two steps merged into one.
  CHECK(undo.undo(scene) == "Switch to Level 0");
  CHECK(obj.multires->active == 2);
  CHECK(undo.undo(scene) == "Draw");
  CHECK(undo.undo(scene) == "Switch to Level 2");
  CHECK(obj.multires->active == 3);

  // Only a switch that starts where the last one ended merges: same level index but another
  // version (the stack changed without an entry) stays separate.
  UndoStack fresh;
  MultiresUndo a;
  a.objectId = obj.id;
  a.levelBefore = 1, a.levelAfter = 2, a.countBefore = a.countAfter = 4;
  a.versionBefore = 10, a.versionAfter = 11;
  MultiresUndo b = a;
  b.levelBefore = 2, b.levelAfter = 1, b.versionBefore = 99, b.versionAfter = 10;
  fresh.push(std::move(a));
  fresh.push(std::move(b));
  CHECK(fresh.size() == 2);
}

TEST_CASE("levels: a stroke undoes after leaving and returning to its level") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = addLeveled(scene, ws, 3);
  UndoStack undo;
  step(obj, 1, undo, ws);
  undo.clear();
  const Snapshot initial = snapshot(obj);
  auto s1 = stroke(obj, {0, 1, 0.3f}, 0.5f);
  REQUIRE(s1);
  undo.push(std::move(*s1));
  step(obj, 3, undo, ws);
  auto s3 = stroke(obj, {0.2f, 1, 0}, 0.3f);
  REQUIRE(s3);
  undo.push(std::move(*s3));
  step(obj, 1, undo, ws);
  requireValidLevels(obj);
  const Snapshot final = snapshot(obj);
  CHECK(undo.size() == 4);
  for (int i = 0; i < 4; ++i) CHECK(!undo.undo(scene).empty());
  CHECK(obj.multires->active == 1);
  requireSame(snapshot(obj), initial);
  CHECK(diffActive(*obj.multires, obj.mesh).empty());
  for (int i = 0; i < 4; ++i) CHECK(!undo.redo(scene).empty());
  requireSame(snapshot(obj), final);
  requireValidLevels(obj);
}

TEST_CASE("levels: undoing a level step brings the pending edits back") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = addLeveled(scene, ws, 3);
  UndoStack undo;
  step(obj, 2, undo, ws);
  const Snapshot before = snapshot(obj);
  auto s = stroke(obj, {0, 0, 1}, 0.4f);
  REQUIRE(s);
  undo.push(std::move(*s));
  const std::vector<Vec3> sculpted = obj.mesh.positions;
  CHECK(!diffActive(*obj.multires, obj.mesh).empty());
  step(obj, 1, undo, ws);
  // The edit reached the parked levels.
  CHECK_FALSE(sameArray(obj.multires->levels[3].mesh.positions, before.levels[3].pos));
  CHECK_FALSE(sameArray(obj.mesh.positions, before.levels[1].pos));
  CHECK(undo.undo(scene) == "Switch to Level 1");
  CHECK(obj.multires->active == 2);
  CHECK(sameArray(obj.mesh.positions, sculpted));
  for (int k : {0, 1, 3}) {
    INFO("level " << k);
    CHECK(sameArray(obj.multires->levels[static_cast<std::size_t>(k)].mesh.positions,
                    before.levels[static_cast<std::size_t>(k)].pos));
    CHECK(sameArray(obj.multires->levels[static_cast<std::size_t>(k)].mesh.normals,
                    before.levels[static_cast<std::size_t>(k)].nrm));
  }
  CHECK(!diffActive(*obj.multires, obj.mesh).empty());
  requireValidLevels(obj);
}

TEST_CASE("levels: a stroke entry never applies on another level") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = addLeveled(scene, ws, 3);
  UndoStack undo;
  step(obj, 2, undo, ws);
  auto s = stroke(obj, {0, 1, 0}, 0.4f);
  REQUIRE(s);
  CHECK(s->topologyVersion == obj.multires->levels[2].version);
  // Leave level 2 behind the undo stack's back: the entry must refuse to apply on level 3.
  REQUIRE(setActiveLevel(obj, 3, ws));
  const std::vector<Vec3> l3 = obj.mesh.positions;
  undo.clear();
  undo.push(std::move(*s));
  CHECK(undo.undo(scene).empty());
  CHECK(sameArray(obj.mesh.positions, l3));
}

TEST_CASE("levels: undoing subdivide restores the plain object and older history") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = scene.add("Head", makeQuadSphere(5));
  UndoStack undo;
  auto s = stroke(obj, {0, 1, 0}, 0.4f);
  REQUIRE(s);
  undo.push(std::move(*s));
  const Mesh sculpted = obj.mesh;
  const std::uint64_t version = obj.topologyVersion;
  auto u = subdivideObject(obj, ws);
  REQUIRE(u);
  const std::uint64_t level1 = obj.topologyVersion;
  const std::uint64_t level1Layout = obj.multires->levels[1].layoutHash;
  undo.push(std::move(*u));
  CHECK(undo.undo(scene) == "Subdivide");
  CHECK(!obj.multires);
  CHECK(obj.topologyVersion == version);
  CHECK(sameArray(obj.mesh.positions, sculpted.positions));
  CHECK(sameArray(obj.mesh.normals, sculpted.normals));
  CHECK(undo.undo(scene) == "Draw");
  CHECK(undo.redo(scene) == "Draw");
  CHECK(undo.redo(scene) == "Subdivide");
  REQUIRE(obj.multires);
  CHECK(obj.topologyVersion == level1);
  CHECK(layoutHash(obj.bvh) == level1Layout);
  requireValidLevels(obj);
  // A second subdivide with pending edits on level 1 folds them into level 0, and undo unfolds.
  auto s1 = stroke(obj, {1, 0, 0}, 0.4f);
  REQUIRE(s1);
  undo.push(std::move(*s1));
  const std::vector<Vec3> l0 = obj.multires->levels[0].mesh.positions;
  auto u2 = subdivideObject(obj, ws);
  REQUIRE(u2);
  undo.push(std::move(*u2));
  CHECK_FALSE(sameArray(obj.multires->levels[0].mesh.positions, l0));
  CHECK(undo.undo(scene) == "Subdivide");
  CHECK(sameArray(obj.multires->levels[0].mesh.positions, l0));
  CHECK(!diffActive(*obj.multires, obj.mesh).empty());
  requireValidLevels(obj);
}

TEST_CASE("levels: layouts and versions never change") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = addLeveled(scene, ws, 3);
  std::vector<std::uint64_t> layouts, versions;
  for (const MultiresLevel& l : obj.multires->levels) {
    layouts.push_back(l.layoutHash);
    versions.push_back(l.version);
  }
  UndoStack undo;
  for (int round = 0; round < 3; ++round) {
    for (int k : {1, 3, 0, 2}) {
      step(obj, k, undo, ws);
      auto s = stroke(obj, {std::cos(float(round + k)), 0.5f, std::sin(float(round + k))}, 0.6f);
      if (s) undo.push(std::move(*s));
    }
  }
  while (undo.canUndo()) undo.undo(scene);
  requireValidLevels(obj);
  for (std::size_t k = 0; k < layouts.size(); ++k) {
    CHECK(obj.multires->levels[k].layoutHash == layouts[k]);
    CHECK(obj.multires->levels[k].version == versions[k]);
  }
}

TEST_CASE("propagation: base edits carry detail rigidly") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = addLeveled(scene, ws, 3);
  // Detail on the top level.
  for (int i = 0; i < 6; ++i) REQUIRE(stroke(obj, {std::cos(float(i)), 0.3f, std::sin(float(i))}, 0.25f, 0.8f));
  REQUIRE(setActiveLevel(obj, 0, ws));
  const std::vector<Vec3> top = obj.multires->levels[3].mesh.positions;
  SUBCASE("translation") {
    const Vec3 t{0.3f, -0.2f, 0.5f};
    moveVertices(obj, [](const Vec3&) { return true; }, t);
    REQUIRE(setActiveLevel(obj, 3, ws));
    float worst = 0.0f;
    for (Index v = 0; v < obj.mesh.vertexCount(); ++v) worst = std::max(worst, glm::length(obj.mesh.positions[v] - (top[v] + t)));
    CHECK(worst < 1e-5f);
  }
  SUBCASE("rotation") {
    const Mat3 r = Mat3(glm::rotate(Mat4(1.0f), glm::radians(90.0f), Vec3{0, 1, 0}));
    for (Vec3& p : obj.mesh.positions) p = r * p;
    obj.mesh.computeNormals();
    REQUIRE(setActiveLevel(obj, 3, ws));
    float worst = 0.0f;
    for (Index v = 0; v < obj.mesh.vertexCount(); ++v) worst = std::max(worst, glm::length(obj.mesh.positions[v] - r * top[v]));
    CHECK(worst < 1e-4f);
  }
  requireValidLevels(obj);
}

TEST_CASE("propagation: the region evaluated equals a whole-level evaluation") {
  Scene scene;
  SyncWorkspace ws, wsFull;
  SceneObject& obj = addLeveled(scene, ws, 3);
  for (int i = 0; i < 4; ++i) REQUIRE(stroke(obj, {std::cos(float(i)), -0.2f, std::sin(float(i))}, 0.3f, 0.8f));
  for (int level : {1, 0, 2}) {
    REQUIRE(setActiveLevel(obj, level, ws));
    REQUIRE(stroke(obj, {0.1f, 1, 0.2f}, 0.35f, 0.7f));
    obj.mesh.ensureMask();
    obj.mesh.mask[5] = 1.0f;
    obj.mesh.ensureFaceSets();
    obj.mesh.faceSets[7] = -9;
    const Multires& s = *obj.multires;
    const SyncDelta limited = computeSync(s, obj.mesh, diffActive(s, obj.mesh), ws, true);
    const SyncDelta full = computeSync(s, obj.mesh, diffActive(s, obj.mesh), wsFull, false);
    REQUIRE(limited.levels.size() == full.levels.size());
    for (std::size_t i = 0; i < limited.levels.size(); ++i) {
      const LevelDelta &a = limited.levels[i], &b = full.levels[i];
      INFO("from level " << level << " to level " << a.level);
      CHECK(a.level == b.level);
      CHECK(a.posIndex == b.posIndex);
      CHECK(sameArray(a.posAfter, b.posAfter));
      CHECK(a.maskIndex == b.maskIndex);
      CHECK(a.maskAfter == b.maskAfter);
      CHECK(a.setIndex == b.setIndex);
      CHECK(a.setAfter == b.setAfter);
    }
    // Most of the finest level stays untouched by a local edit.
    for (const LevelDelta& d : limited.levels)
      if (d.level == 3) CHECK(d.posIndex.size() < obj.multires->levels[3].mesh.positions.size() / 2);
  }
}

TEST_CASE("propagation: the edited level keeps its sculpted values") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = addLeveled(scene, ws, 3);
  REQUIRE(setActiveLevel(obj, 1, ws));
  REQUIRE(stroke(obj, {0, 1, 0}, 0.5f, 0.9f));
  const std::vector<Vec3> sculpted = obj.mesh.positions;
  const std::vector<Vec3> normals = obj.mesh.normals;
  REQUIRE(setActiveLevel(obj, 3, ws));
  REQUIRE(setActiveLevel(obj, 1, ws));
  CHECK(sameArray(obj.mesh.positions, sculpted));
  CHECK(sameArray(obj.mesh.normals, normals));
}

TEST_CASE("propagation: restriction keeps constants and removes the checkerboard") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = scene.add("Plane", makePlane(6));
  REQUIRE(subdivideObject(obj, ws));
  REQUIRE(subdivideObject(obj, ws));
  REQUIRE(setActiveLevel(obj, 1, ws));
  const std::vector<Vec3> coarse = obj.mesh.positions;
  const std::vector<VertexRule> rule = obj.multires->levels[1].rule;
  REQUIRE(setActiveLevel(obj, 2, ws));
  SUBCASE("constant") {
    const Vec3 c{0.5f, -0.25f, 1.0f};
    for (Vec3& p : obj.mesh.positions) p += c;
    REQUIRE(setActiveLevel(obj, 1, ws));
    for (Index v = 0; v < obj.mesh.vertexCount(); ++v) CHECK(sameBits(obj.mesh.positions[v], coarse[v] + c));
  }
  SUBCASE("checkerboard") {
    const SubdivisionLinks& L = obj.multires->levels[2].links;
    const Vec3 c{0.0f, 0.25f, 0.0f};
    for (Index v = 0; v < obj.mesh.vertexCount(); ++v)
      obj.mesh.positions[v] += parentKind(L.parent[v]) == kParentEdge ? -c : c;
    REQUIRE(setActiveLevel(obj, 1, ws));
    int checked = 0;
    for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
      if (rule[v] == VertexRule::Pinned) continue;  // Corners follow their vertex child only.
      CHECK(sameBits(obj.mesh.positions[v], coarse[v]));
      ++checked;
    }
    CHECK(checked > 40);
  }
}

TEST_CASE("propagation: fine edits reach lower levels smoothed and only nearby") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = addLeveled(scene, ws, 3);
  std::vector<std::vector<Vec3>> before;
  for (int k = 0; k < 4; ++k) before.push_back(levelMesh(obj, k).positions);
  const Vec3 center = surfacePoint(obj, {0, 1, 0});
  REQUIRE(stroke(obj, {0, 1, 0}, 0.3f, 1.0f));
  float fineMax = 0.0f;
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v)
    fineMax = std::max(fineMax, glm::length(obj.mesh.positions[v] - before[3][v]));
  REQUIRE(fineMax > 0.0f);
  REQUIRE(setActiveLevel(obj, 0, ws));
  for (int k = 0; k < 3; ++k) {
    const Mesh& m = levelMesh(obj, k);
    int moved = 0;
    for (Index v = 0; v < m.vertexCount(); ++v) {
      const float d = glm::length(m.positions[v] - before[k][v]);
      CHECK(d <= fineMax * 1.0001f);
      if (d > 0.0f) {
        ++moved;
        CHECK(glm::length(before[k][v] - center) < 0.3f * 1.25f + 0.8f);
      }
    }
    CHECK(moved > 0);
  }
}

TEST_CASE("propagation: alternating edits do not drift") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = addLeveled(scene, ws, 2, 4);
  for (int i = 0; i < 3; ++i) REQUIRE(stroke(obj, {std::cos(float(i)), 0.4f, std::sin(float(i))}, 0.3f, 0.8f));
  REQUIRE(setActiveLevel(obj, 0, ws));
  const std::vector<Vec3> base = obj.mesh.positions;
  const std::vector<Vec3> top = obj.multires->levels[2].mesh.positions;
  const Vec3 t{0.0f, 0.15f, 0.05f};
  for (int round = 0; round < 50; ++round) {
    moveVertices(obj, [](const Vec3& p) { return p.x > 0.3f; }, t);
    REQUIRE(setActiveLevel(obj, 2, ws));
    REQUIRE(setActiveLevel(obj, 0, ws));
    obj.mesh.positions = base;
    obj.mesh.computeNormals();
    REQUIRE(setActiveLevel(obj, 2, ws));
    REQUIRE(setActiveLevel(obj, 0, ws));
  }
  const std::vector<Vec3>& now = obj.multires->levels[2].mesh.positions;
  float worst = 0.0f;
  int identical = 0;
  for (std::size_t v = 0; v < top.size(); ++v) {
    worst = std::max(worst, glm::length(now[v] - top[v]));
    identical += sameBits(now[v], top[v]) ? 1 : 0;
  }
  CHECK(worst < 1e-5f);
  CHECK(identical > static_cast<int>(top.size()) / 4);  // Far from the moved region nothing changed.
}

TEST_CASE("propagation: degenerate coarse faces give finite results") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = addLeveled(scene, ws, 2, 4);
  REQUIRE(stroke(obj, {0, 1, 0}, 0.3f));
  REQUIRE(setActiveLevel(obj, 0, ws));
  // Collapse a few faces onto points.
  for (Index f : {0, 5, 9}) {
    const Vec3 c = obj.mesh.faceCentroid(f);
    obj.mesh.forEachFaceVertex(f, [&](Index v) { obj.mesh.positions[v] = c; });
  }
  obj.mesh.computeNormals();
  REQUIRE(setActiveLevel(obj, 2, ws));
  for (const Vec3& p : obj.mesh.positions) REQUIRE((std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)));
  for (const Vec3& n : obj.mesh.normals) REQUIRE((std::isfinite(n.x) && std::isfinite(n.y) && std::isfinite(n.z)));
  requireValidLevels(obj);
}

TEST_CASE("propagation: mask travels both ways and 1.0 stays exact") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = addLeveled(scene, ws, 3, 4);
  REQUIRE(setActiveLevel(obj, 1, ws));
  obj.mesh.ensureMask();
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) obj.mesh.mask[v] = obj.mesh.positions[v].y > 0.2f ? 1.0f : 0.0f;
  const std::vector<float> l1 = obj.mesh.mask;
  REQUIRE(setActiveLevel(obj, 3, ws));
  // Up: vertex children copy, all-masked edges and faces stay exactly 1.
  const Mesh& l2 = obj.multires->levels[2].mesh;
  const SubdivisionLinks& L2 = obj.multires->levels[2].links;
  const Mesh& c1 = obj.multires->levels[1].mesh;
  for (Index v = 0; v < c1.vertexCount(); ++v) CHECK(l2.mask[L2.vertexChild[v]] == l1[v]);
  int halfway = 0;
  for (Index h = 0; h < c1.halfEdgeCount(); ++h) {
    const float a = l1[c1.heVert[h]], b = l1[c1.heTarget(h)];
    if (a == 1.0f && b == 1.0f) CHECK(l2.mask[L2.edgeChild[h]] == 1.0f);
    // The level had no mask before, so an edge child gets half its parents' change.
    if (a != b) {
      CHECK(l2.mask[L2.edgeChild[h]] == 0.5f);
      ++halfway;
    }
  }
  CHECK(halfway > 0);
  // Down: injection.
  for (int k = 0; k < 3; ++k) {
    const Mesh& coarse = levelMesh(obj, k);
    const Mesh& fine = levelMesh(obj, k + 1);
    const SubdivisionLinks& L = obj.multires->levels[static_cast<std::size_t>(k + 1)].links;
    REQUIRE(coarse.mask.size() == coarse.positions.size());
    for (Index v = 0; v < coarse.vertexCount(); ++v) CHECK(coarse.mask[v] == fine.mask[L.vertexChild[v]]);
  }
  // Mask that exists only on the top level (on edge and face children) is invisible below, but
  // Clear on another level still removes it.
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v)
    if (parentKind(obj.multires->levels[3].links.parent[v]) != kParentVertex && obj.mesh.positions[v].y < -0.5f)
      obj.mesh.mask[v] = 1.0f;
  REQUIRE(setActiveLevel(obj, 1, ws));
  UndoStack undo;
  auto clear = applyMaskOpAllLevels(obj, MaskOp::Clear, ws);
  REQUIRE(clear);
  undo.push(std::move(*clear));
  for (int k = 0; k < 4; ++k) {
    const Mesh& m = levelMesh(obj, k);
    for (float x : m.mask) REQUIRE(x == 0.0f);
  }
  CHECK(undo.undo(scene) == "Clear Mask");
  REQUIRE(setActiveLevel(obj, 3, ws));
  bool fineOnly = false;
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) fineOnly |= obj.mesh.positions[v].y < -0.5f && obj.mesh.mask[v] == 1.0f;
  CHECK(fineOnly);
  requireValidLevels(obj);
}

TEST_CASE("propagation: face sets travel both ways") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = addLeveled(scene, ws, 2, 4);
  REQUIRE(setActiveLevel(obj, 0, ws));
  obj.mesh.ensureFaceSets();
  obj.mesh.faceSets[3] = 5;
  obj.mesh.faceSets[4] = -1;  // Hidden.
  const Index f3 = 3, f4 = 4;
  REQUIRE(setActiveLevel(obj, 1, ws));
  const SubdivisionLinks& L = obj.multires->levels[1].links;
  const Mesh& base = obj.multires->levels[0].mesh;
  auto children = [&](Index f) {
    std::vector<Index> q;
    const Index start = base.faceHe[f];
    Index h = start;
    do {
      q.push_back(L.childFace[h]);
      h = base.heNext[h];
    } while (h != start);
    return q;
  };
  for (Index q : children(f3)) CHECK(obj.mesh.faceSets[q] == 5);
  for (Index q : children(f4)) CHECK(obj.mesh.faceSets[q] == -1);
  SUBCASE("one of four children does not move the parent; a tie goes to the first corner") {
    const std::vector<Index> q = children(f3);
    obj.mesh.faceSets[q[1]] = 7;
    REQUIRE(setActiveLevel(obj, 0, ws));
    CHECK(obj.mesh.faceSets[f3] == 5);
    REQUIRE(setActiveLevel(obj, 1, ws));
    obj.mesh.faceSets[q[0]] = 7;  // Now 7, 7, 5, 5: corner 0 has 7.
    REQUIRE(setActiveLevel(obj, 0, ws));
    CHECK(obj.mesh.faceSets[f3] == 7);
  }
  SUBCASE("a parent is hidden only when all its children are") {
    const std::vector<Index> q = children(f4);
    obj.mesh.faceSets[q[2]] = 1;  // Reveal one child.
    REQUIRE(setActiveLevel(obj, 0, ws));
    CHECK(obj.mesh.faceSets[f4] == 1);
  }
  SUBCASE("reveal all on the base reveals faces hidden only on the top") {
    REQUIRE(setActiveLevel(obj, 2, ws));
    obj.mesh.faceSets[10] = -faceSetId(obj.mesh.faceSets[10]);
    REQUIRE(faceSetHidden(obj.mesh.faceSets[10]));
    REQUIRE(setActiveLevel(obj, 0, ws));
    REQUIRE(applyFaceSetOpAllLevels(obj, FaceSetOp::RevealAll, ws));
    REQUIRE(setActiveLevel(obj, 2, ws));
    for (std::int32_t s : obj.mesh.faceSets) CHECK(s > 0);
  }
  SUBCASE("new ids are unique across levels") {
    REQUIRE(setActiveLevel(obj, 2, ws));
    obj.mesh.faceSets[20] = 40;  // One fine face: lost to the majority below.
    REQUIRE(setActiveLevel(obj, 0, ws));
    CHECK(obj.mesh.maxFaceSetId() < 40);
    CHECK(obj.newFaceSetId() == 41);
  }
  requireValidLevels(obj);
}

TEST_CASE("levels: deleting higher and lower levels undoes bit-exactly") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = addLeveled(scene, ws, 3);
  UndoStack undo;
  step(obj, 1, undo, ws);
  REQUIRE(stroke(obj, {0, 1, 0}, 0.4f));  // Pending: Delete does not sync.
  const Snapshot start = snapshot(obj);
  SUBCASE("higher") {
    auto u = deleteHigherLevels(obj);
    REQUIRE(u);
    undo.push(std::move(*u));
    CHECK(obj.multires->levelCount() == 2);
    requireValidLevels(obj);
    CHECK(undo.undo(scene) == "Delete Higher Levels");
    requireSame(snapshot(obj), start);
    CHECK(undo.redo(scene) == "Delete Higher Levels");
    CHECK(obj.multires->levelCount() == 2);
  }
  SUBCASE("lower") {
    auto u = deleteLowerLevels(obj);
    REQUIRE(u);
    undo.push(std::move(*u));
    CHECK(obj.multires->levelCount() == 3);
    CHECK(obj.multires->active == 0);
    requireValidLevels(obj);
    CHECK(undo.undo(scene) == "Delete Lower Levels");
    requireSame(snapshot(obj), start);
    requireValidLevels(obj);
    CHECK(undo.redo(scene) == "Delete Lower Levels");
    CHECK(obj.multires->active == 0);
  }
  SUBCASE("dissolve from the base and from the top") {
    step(obj, 0, undo, ws);
    const Snapshot atBase = snapshot(obj);
    auto u = deleteHigherLevels(obj);
    REQUIRE(u);
    undo.push(std::move(*u));
    CHECK(!obj.multires);
    CHECK(obj.topologyVersion == atBase.levels[0].version);
    CHECK(undo.undo(scene) == "Delete Higher Levels");
    requireSame(snapshot(obj), atBase);
    step(obj, 3, undo, ws);
    const Snapshot atTop = snapshot(obj);
    auto v = deleteLowerLevels(obj);
    REQUIRE(v);
    undo.push(std::move(*v));
    CHECK(!obj.multires);
    CHECK(obj.topologyVersion == atTop.levels[3].version);
    CHECK(undo.undo(scene) == "Delete Lower Levels");
    requireSame(snapshot(obj), atTop);
    requireValidLevels(obj);
  }
}

TEST_CASE("levels: remesh drops the levels and undo restores them with their versions") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = addLeveled(scene, ws, 2);
  UndoStack undo;
  step(obj, 1, undo, ws);
  auto s = stroke(obj, {0, 1, 0}, 0.4f);
  REQUIRE(s);
  undo.push(std::move(*s));
  const Snapshot before = snapshot(obj);
  // What the app's remesh does: the stack moves into the before state, unsynced.
  TopologyUndo entry;
  entry.label = "Remesh";
  entry.objectId = obj.id;
  entry.before = std::make_shared<MeshState>(MeshState{obj.mesh, obj.bvh, obj.topologyVersion, std::move(obj.multires)});
  obj.multires = nullptr;
  obj.mesh = makeQuadSphere(3);
  obj.rebuildSpatial();
  entry.after = std::make_shared<MeshState>(MeshState{obj.mesh, obj.bvh, obj.topologyVersion, nullptr});
  undo.push(std::move(entry));
  CHECK(undo.undo(scene) == "Remesh");
  REQUIRE(obj.multires);
  requireSame(snapshot(obj), before);
  CHECK(!diffActive(*obj.multires, obj.mesh).empty());  // The stroke is still pending.
  CHECK(undo.undo(scene) == "Draw");
  CHECK(diffActive(*obj.multires, obj.mesh).empty());
  CHECK(undo.redo(scene) == "Draw");
  CHECK(undo.redo(scene) == "Remesh");
  CHECK(!obj.multires);
  CHECK(undo.undo(scene) == "Remesh");
  step(obj, 2, undo, ws);
  requireValidLevels(obj);
}

TEST_CASE("levels: duplicates copy the levels with fresh versions") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = addLeveled(scene, ws, 2);
  REQUIRE(stroke(obj, {0, 1, 0}, 0.4f));
  SceneObject* copy = scene.duplicate(obj.id);
  REQUIRE(copy);
  REQUIRE(copy->multires);
  CHECK(copy->multires != obj.multires);
  for (int k = 0; k < 3; ++k) {
    CHECK(copy->multires->levels[static_cast<std::size_t>(k)].version !=
          obj.multires->levels[static_cast<std::size_t>(k)].version);
  }
  CHECK(copy->topologyVersion == copy->multires->levels[2].version);
  requireValidLevels(*copy);
  requireSame([&] {
    Snapshot s = snapshot(*copy);
    for (auto& l : s.levels) l.version = 0;
    return s;
  }(),
              [&] {
                Snapshot s = snapshot(obj);
                for (auto& l : s.levels) l.version = 0;
                return s;
              }());
  CHECK(!diffActive(*copy->multires, copy->mesh).empty());  // Pending edits came along.
  REQUIRE(setActiveLevel(*copy, 0, ws));
  requireValidLevels(*copy);
}

TEST_CASE("levels: dynamic topology is ignored on objects with levels") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = addLeveled(scene, ws, 1);
  const Index faces = obj.mesh.faceCount();
  StrokeOptions opts;
  opts.dyntopo = true;
  static DrawBrush draw;
  Sculptor sculptor;
  sculptor.beginStroke(obj, draw, opts, "Draw");
  CHECK(sculptor.dyntopo() == nullptr);
  const Vec3 c = surfacePoint(obj, {0, 1, 0});
  DabTopology topo;
  topo.detail = 0.01f;
  sculptor.dab(c, 0.3f, 0.5f, topo);
  auto undo = sculptor.endStroke();
  REQUIRE(undo);
  CHECK(std::holds_alternative<SculptUndo>(*undo));
  CHECK(obj.mesh.faceCount() == faces);
  requireValidLevels(obj);
}

TEST_CASE("levels: normals stay a pure function of positions after strokes") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = addLeveled(scene, ws, 2);
  REQUIRE(stroke(obj, {0, 1, 0}, 0.4f));
  REQUIRE(setActiveLevel(obj, 0, ws));
  for (int k = 0; k < 3; ++k) {
    Mesh m = levelMesh(obj, k);
    const std::vector<Vec3> stored = m.normals;
    m.computeNormals();
    INFO("level " << k);
    CHECK(sameArray(stored, m.normals));
  }
}
