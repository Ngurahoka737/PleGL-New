#include <algorithm>
#include <cmath>

#include "TestUtil.h"
#include "mesh/Primitives.h"
#include "scene/Scene.h"
#include "sculpt/LayerOps.h"
#include "sculpt/Sculptor.h"
#include "sculpt/Undo.h"

using namespace plegl;

namespace {

StrokeOptions snakeOptions(SymmetrySpec s = {}) {
  StrokeOptions o;
  o.strength = 1.0f;
  o.symmetryX = s.x;
  o.symmetryY = s.y;
  o.symmetryZ = s.z;
  o.radialAxis = s.radialAxis;
  o.radialCount = s.radialCount;
  return o;
}

// Moves the target in `ticks` even steps from the hook's position to `end`, one tick per step,
// then holds it there for `hold` more ticks.
void pullTo(Sculptor& s, const SceneObject& obj, const Vec3& end, float r, int ticks, int hold = 0,
            float gain = 1.0f, float detail = 0.0f) {
  const Vec3 start = obj.mesh.positions[s.snakeHook()];
  for (int i = 1; i <= ticks + hold; ++i) {
    const float t = std::min(1.0f, static_cast<float>(i) / static_cast<float>(ticks));
    DabTopology topo;
    topo.detail = detail;
    s.snakeTo(glm::mix(start, end, t), r, gain, topo);
  }
}

float maxMove(const Mesh& a, const Mesh& b) {
  float most = 0.0f;
  for (Index v = 0; v < a.vertexCount(); ++v) most = std::max(most, glm::length(a.positions[v] - b.positions[v]));
  return most;
}

// Mean distance from the line through `p` along +Y of the vertices with y in [y0, y1].
float crossSection(const Mesh& m, const Vec3& p, float y0, float y1, int& count) {
  float sum = 0.0f;
  count = 0;
  for (const Vec3& q : m.positions) {
    if (q.y < y0 || q.y > y1) continue;
    sum += glm::length(Vec2{q.x - p.x, q.z - p.z});
    ++count;
  }
  return count > 0 ? sum / static_cast<float>(count) : 0.0f;
}

}  // namespace

TEST_CASE("snake hook: the hook follows the cursor and pulls out a tentacle") {
  Scene scene;
  SceneObject& obj = scene.add("S", makeQuadSphere(32));
  const Mesh before = obj.mesh;
  SnakeHookBrush snake;
  Sculptor s;
  const float r = 0.15f;
  REQUIRE(s.beginSnake(obj, snake, snakeOptions(), test::surfacePoint(obj, {0.0f, 1.0f, 0.0f}), r, 0.0f, "Snake Hook"));
  const Index hook = s.snakeHook();
  REQUIRE(hook != kInvalid);
  const Vec3 end = obj.mesh.positions[hook] + Vec3{0.4f * r, 3.0f * r, -0.3f * r};
  pullTo(s, obj, end, r, 60);
  CHECK(s.snakeHook() == hook);
  CHECK(glm::length(obj.mesh.positions[hook] - end) < 0.01f * r);
  // Without dynamic topology the pulled part is a horn: the hook at its tip, the vertices around
  // it trailing behind.
  int horn = 0;
  for (const Vec3& p : obj.mesh.positions) horn += p.y > 1.0f + 0.5f * r;
  CHECK(horn >= 5);
  test::requireNormalsMatchFullRecompute(obj.mesh);
  test::requireBoundsContainPositions(obj);
  auto undo = s.endStroke();
  REQUIRE(undo);
  test::requireValid(obj.mesh);
  UndoStack stack;
  stack.push(std::move(*undo));
  REQUIRE_FALSE(stack.undo(scene).empty());
  CHECK(test::hashBits(obj.mesh.positions) == test::hashBits(before.positions));
  CHECK(test::hashBits(obj.mesh.normals) == test::hashBits(before.normals));
}

TEST_CASE("snake hook: a flick runs at most 32 sub-dabs of one step and the rest catches up") {
  Scene scene;
  SceneObject& obj = scene.add("S", makeQuadSphere(32));
  SnakeHookBrush snake;
  Sculptor s;
  const float r = 0.15f;
  REQUIRE(s.beginSnake(obj, snake, snakeOptions(), test::surfacePoint(obj, {0.1f, 1.0f, 0.0f}), r, 0.0f, "Snake Hook"));
  const Index hook = s.snakeHook();
  const Vec3 start = obj.mesh.positions[hook];
  const Vec3 end = start + glm::normalize(Vec3{0.3f, 1.0f, 0.2f}) * (8.0f * r);
  // A zero budget stops after one sub-dab: nothing moves farther than the brush contract allows.
  const Mesh before = obj.mesh;
  SnakeStep step = s.snakeTo(end, r, 1.0f, {}, 0.0);
  CHECK(step.subDabs == 1);
  CHECK(maxMove(before, obj.mesh) <= kMaxDabMove * r * 1.0001f);
  CHECK(glm::length(obj.mesh.positions[hook] - start) == doctest::Approx(kSnakeStep * r).epsilon(1e-3));
  // Without a budget the tick stops at 32 sub-dabs of one step each; the next tick finishes.
  step = s.snakeTo(end, r, 1.0f);
  CHECK(step.subDabs == kMaxSnakeSubDabs);
  CHECK(step.remaining == doctest::Approx((8.0f - 33.0f * kSnakeStep) * r).epsilon(0.01));
  step = s.snakeTo(end, r, 1.0f);
  CHECK(step.subDabs == static_cast<int>(std::ceil((8.0f - 33.0f * kSnakeStep) / kSnakeStep - 1e-3f)));
  CHECK(step.remaining < 0.01f * r);
  test::requireNormalsMatchFullRecompute(obj.mesh);
  REQUIRE(s.endStroke());
  test::requireValid(obj.mesh);
}

TEST_CASE("snake hook: the mask refuses a protected hook and steering pulls a half-masked one home") {
  const float r = 0.15f;
  SnakeHookBrush snake;
  {
    Scene scene;
    SceneObject& obj = scene.add("S", makeQuadSphere(24));
    obj.mesh.ensureMask();
    std::fill(obj.mesh.mask.begin(), obj.mesh.mask.end(), 1.0f);
    Sculptor s;
    CHECK_FALSE(s.beginSnake(obj, snake, snakeOptions(), test::surfacePoint(obj, {0, 1, 0}), r, 0.0f, "Snake Hook"));
    CHECK_FALSE(s.active());
    CHECK(s.snakeHook() == kInvalid);
  }
  {
    Scene scene;
    SceneObject& obj = scene.add("S", makeQuadSphere(24));
    obj.mesh.ensureMask();
    std::fill(obj.mesh.mask.begin(), obj.mesh.mask.end(), 0.5f);
    Sculptor s;
    REQUIRE(s.beginSnake(obj, snake, snakeOptions(), test::surfacePoint(obj, {0, 1, 0}), r, 0.0f, "Snake Hook"));
    const Index hook = s.snakeHook();
    const Vec3 end = obj.mesh.positions[hook] + Vec3{0.0f, 3.0f * r, 0.5f * r};
    pullTo(s, obj, end, r, 60, 5);
    CHECK(glm::length(obj.mesh.positions[hook] - end) < 0.01f * r);
    REQUIRE(s.endStroke());
  }
  {
    // Mostly masked: steering would ask for five times the pull, the cap allows three.
    Scene scene;
    SceneObject& obj = scene.add("S", makeQuadSphere(24));
    obj.mesh.ensureMask();
    std::fill(obj.mesh.mask.begin(), obj.mesh.mask.end(), 0.9f);
    Sculptor s;
    REQUIRE(s.beginSnake(obj, snake, snakeOptions(), test::surfacePoint(obj, {0, 1, 0}), r, 0.0f, "Snake Hook"));
    const Index hook = s.snakeHook();
    const Vec3 start = obj.mesh.positions[hook];
    const SnakeStep step = s.snakeTo(start + Vec3{0.0f, 0.5f * r, 0.0f}, r, 1.0f);
    CHECK(step.subDabs == static_cast<int>(std::ceil(3.0f * 0.5f / kSnakeStep)));
    CHECK(glm::length(obj.mesh.positions[hook] - start) == doctest::Approx(0.1f * 3.0f * 0.5f * r).epsilon(0.01));
    REQUIRE(s.endStroke());
  }
}

TEST_CASE("snake hook: a hook on the mirror plane stays on it and a hook near it is steered home") {
  const float r = 0.15f;
  SnakeHookBrush snake;
  {
    Scene scene;
    SceneObject& obj = scene.add("S", makeQuadSphere(32));
    Sculptor s;
    REQUIRE(s.beginSnake(obj, snake, snakeOptions({true, false, false}), test::surfacePoint(obj, {0, 1, 0}), r, 0.0f,
                         "Snake Hook"));
    const Index hook = s.snakeHook();
    REQUIRE(s.snakeHook(1) == kInvalid);  // One merged cluster.
    REQUIRE(obj.mesh.positions[hook].x == 0.0f);
    const Vec3 start = obj.mesh.positions[hook];
    // The pull across the plane is out of reach and costs no sub-dabs.
    const Vec3 delta{0.5f * r, 0.4f * r, 0.3f * r};
    const SnakeStep step = s.snakeTo(start + delta, r, 1.0f);
    CHECK(step.subDabs == static_cast<int>(std::ceil(glm::length(Vec2{delta.y, delta.z}) / (kSnakeStep * r))));
    for (int i = 1; i <= 30; ++i) {
      const float t = static_cast<float>(i) / 30.0f;
      s.snakeTo(start + Vec3{0.8f * r * std::sin(6.0f * t), 2.5f * r * t, r * t}, r, 1.0f);
      REQUIRE(obj.mesh.positions[hook].x == 0.0f);
      REQUIRE_FALSE(std::signbit(obj.mesh.positions[hook].x));
    }
    CHECK(obj.mesh.positions[hook].y > start.y + 2.4f * r);
    test::requireNormalsMatchFullRecompute(obj.mesh);
    REQUIRE(s.endStroke());
  }
  {
    // 0.3 r off the plane the mirror copy overlaps and takes part of the hook; steering makes up
    // for it in the same tick (unsteered the hook would fall about 17% short).
    Scene scene;
    SceneObject& obj = scene.add("S", makeQuadSphere(32));
    Sculptor s;
    REQUIRE(s.beginSnake(obj, snake, snakeOptions({true, false, false}),
                         test::surfacePoint(obj, {0.3f * r, 1.0f, 0.0f}), r, 0.0f, "Snake Hook"));
    const Index hook = s.snakeHook();
    REQUIRE(s.snakeHook(1) != kInvalid);
    const Vec3 start = obj.mesh.positions[hook];
    REQUIRE(start.x > 0.2f * r);
    const Vec3 delta{0.0f, 0.6f * r, 0.8f * r};
    s.snakeTo(start + delta, r, 1.0f);
    CHECK(glm::length(obj.mesh.positions[hook] - (start + delta)) <= 0.02f * glm::length(delta));
    // The mirror hook mirrors it.
    const Vec3 mirrored = obj.mesh.positions[s.snakeHook(1)];
    CHECK(glm::length(mirrored - Vec3{-1.0f, 1.0f, 1.0f} * obj.mesh.positions[hook]) < 1e-5f);
    REQUIRE(s.endStroke());
  }
}

TEST_CASE("snake hook: radial copies far apart each pull their own tentacle") {
  Scene scene;
  SceneObject& obj = scene.add("S", makeQuadSphere(32));
  SnakeHookBrush snake;
  Sculptor s;
  const float r = 0.15f;
  REQUIRE(s.beginSnake(obj, snake, snakeOptions({false, false, false, Axis::Y, 4}),
                       test::surfacePoint(obj, {0.8f, 0.5f, 0.2f}), r, 0.0f, "Snake Hook"));
  const Vec3 start = obj.mesh.positions[s.snakeHook()];
  pullTo(s, obj, start + glm::normalize(start) * (2.0f * r), r, 30);
  const SymmetryGroup& g = s.symmetry();
  REQUIRE(g.size() == 4);
  for (int k = 1; k < 4; ++k) {
    const Index h = s.snakeHook(k);
    REQUIRE(h != kInvalid);
    INFO("copy " << k);
    // Cluster k lives in the frame of op k.
    CHECK(glm::length(obj.mesh.positions[h] - transform(g.op(k).L, obj.mesh.positions[s.snakeHook()])) < 1e-4f);
  }
  REQUIRE(s.endStroke());
}

TEST_CASE("snake hook: dynamic topology keeps the hook and undoes exactly") {
  Scene scene;
  SceneObject& obj = scene.add("S", makeQuadSphere(16));
  const Mesh before = obj.mesh;
  SnakeHookBrush snake;
  Sculptor s;
  const float r = 0.2f;
  StrokeOptions o = snakeOptions();
  o.dyntopo = true;
  o.dyntopoOptions.timeBudgetMs = 0.0;
  REQUIRE(s.beginSnake(obj, snake, o, test::surfacePoint(obj, {0.1f, 1.0f, 0.1f}), r, 0.0f, "Snake Hook"));
  const Index hook = s.snakeHook();
  const Vec3 start = obj.mesh.positions[hook];
  const Vec3 end = start + Vec3{0.2f * r, 3.0f * r, 0.0f};
  for (int i = 1; i <= 40; ++i) {
    DabTopology topo;
    topo.detail = 0.04f;
    s.snakeTo(glm::mix(start, end, static_cast<float>(i) / 40.0f), r, 1.0f, topo);
    REQUIRE(s.snakeHook() == hook);
    REQUIRE(obj.mesh.vertHe[hook] != kInvalid);
  }
  CHECK(glm::length(obj.mesh.positions[hook] - end) < 0.01f * r);
  // New vertices fill the tentacle along its length.
  int filled = 0;
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v)
    filled += obj.mesh.vertHe[v] != kInvalid && obj.mesh.positions[v].y > start.y + 1.5f * r;
  CHECK(filled >= 50);
  test::requireBoundsContainPositions(obj);
  auto undo = s.endStroke();
  REQUIRE(undo);
  REQUIRE(s.lastStrokeTopology().dyntopo);
  CHECK(s.lastStrokeTopology().splits > 20);
  test::requireValid(obj.mesh);
  test::requireNormalsMatchFullRecompute(obj.mesh);
  UndoStack stack;
  stack.push(std::move(*undo));
  REQUIRE_FALSE(stack.undo(scene).empty());
  CHECK(obj.mesh.faceCount() == before.faceCount());
  CHECK(test::hashBits(obj.mesh.positions) == test::hashBits(before.positions));
}

TEST_CASE("snake hook: normals after every tick and bounds after every sub-dab") {
  Scene scene;
  SceneObject& obj = scene.add("S", makeQuadSphere(32));
  SnakeHookBrush snake;
  Sculptor s;
  const float r = 0.15f;
  REQUIRE(s.beginSnake(obj, snake, snakeOptions(), test::surfacePoint(obj, {0.2f, 1.0f, -0.1f}), r, 0.5f,
                       "Snake Hook"));
  const Vec3 start = obj.mesh.positions[s.snakeHook()];
  for (int i = 1; i <= 12; ++i) {
    const Vec3 target = start + Vec3{0.1f * r, 0.5f * r, 0.0f} * static_cast<float>(i);
    // A zero budget runs one sub-dab per tick.
    const SnakeStep step = s.snakeTo(target, r, 1.0f, {}, 0.0);
    REQUIRE(step.subDabs == 1);
    test::requireBoundsContainPositions(obj);
    test::requireNormalsMatchFullRecompute(obj.mesh);
  }
  REQUIRE(s.endStroke());
}

TEST_CASE("snake hook: a stroke on a sculpt layer commits into it and undoes") {
  Scene scene;
  SceneObject& obj = scene.add("S", makeQuadSphere(32));
  LayerWorkspace ws;
  std::string error;
  REQUIRE(addLayer(obj, ws, &error));
  const std::vector<Vec3> base = obj.mesh.layers.base;
  const std::vector<Vec3> offsets = obj.mesh.layers.list[0].offset;
  const std::vector<Vec3> positions = obj.mesh.positions;
  SnakeHookBrush snake;
  Sculptor s;
  const float r = 0.15f;
  StrokeOptions o = snakeOptions();
  o.layerTarget = obj.mesh.layers.active;
  REQUIRE(s.beginSnake(obj, snake, o, test::surfacePoint(obj, {0, 1, 0}), r, 0.0f, "Snake Hook"));
  const Index hook = s.snakeHook();
  pullTo(s, obj, obj.mesh.positions[hook] + Vec3{0.0f, 2.0f * r, 0.0f}, r, 20);
  auto undo = s.endStroke();
  REQUIRE(undo);
  const ValidationResult valid = validateLayers(obj.mesh, true);
  INFO(valid.message);
  CHECK(valid.ok);
  CHECK(test::hashBits(obj.mesh.layers.base) == test::hashBits(base));
  CHECK(obj.mesh.layers.list[0].offset[hook].y > 1.5f * r);
  UndoStack stack;
  stack.push(std::move(*undo));
  REQUIRE_FALSE(stack.undo(scene).empty());
  CHECK(test::hashBits(obj.mesh.layers.list[0].offset) == test::hashBits(offsets));
  CHECK(test::hashBits(obj.mesh.positions) == test::hashBits(positions));
}

TEST_CASE("snake hook: pinch narrows the pulled part") {
  const float r = 0.2f;
  float section[2] = {};
  for (int i = 0; i < 2; ++i) {
    Scene scene;
    SceneObject& obj = scene.add("S", makeQuadSphere(32));
    SnakeHookBrush snake;
    Sculptor s;
    StrokeOptions o = snakeOptions();
    o.dyntopo = true;
    o.dyntopoOptions.timeBudgetMs = 0.0;
    REQUIRE(s.beginSnake(obj, snake, o, test::surfacePoint(obj, {0, 1, 0}), r, i == 0 ? 0.0f : 1.0f, "Snake Hook"));
    const Vec3 start = obj.mesh.positions[s.snakeHook()];
    pullTo(s, obj, start + Vec3{0.0f, 2.5f * r, 0.0f}, r, 40, 0, 1.0f, 0.03f);
    REQUIRE(s.endStroke());
    test::requireValid(obj.mesh);
    int count = 0;
    section[i] = crossSection(obj.mesh, start, start.y + 1.0f * r, start.y + 1.5f * r, count);
    INFO("pinch " << i);
    REQUIRE(count >= 8);
  }
  CHECK(section[1] < 0.9f * section[0]);
}

TEST_CASE("snake hook: face set auto-masking keeps the other set still") {
  Scene scene;
  Mesh mesh = makeQuadSphere(32);
  mesh.ensureFaceSets();
  for (Index f = 0; f < mesh.faceCount(); ++f) mesh.faceSets[f] = mesh.faceCentroid(f).x > 0.0f ? 2 : 3;
  SceneObject& obj = scene.add("S", std::move(mesh));
  const Mesh before = obj.mesh;
  SnakeHookBrush snake;
  Sculptor s;
  const float r = 0.2f;
  StrokeOptions o = snakeOptions();
  o.faceSetAutoMask = true;
  REQUIRE(s.beginSnake(obj, snake, o, test::surfacePoint(obj, {0.3f * r, 1.0f, 0.0f}), r, 0.0f, "Snake Hook"));
  const Index hook = s.snakeHook();
  pullTo(s, obj, obj.mesh.positions[hook] + Vec3{0.0f, 2.0f * r, 0.0f}, r, 30);
  REQUIRE(s.endStroke());
  int moved = 0, still = 0;
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
    bool other = true;
    obj.mesh.forEachOutgoing(v, [&](Index h) { other &= obj.mesh.faceSets[obj.mesh.heFace[h]] == 3; });
    const bool changed = test::hashBits({obj.mesh.positions[v]}) != test::hashBits({before.positions[v]});
    if (other) {
      REQUIRE_FALSE(changed);
      ++still;
    } else {
      moved += changed;
    }
  }
  CHECK(moved > 10);
  CHECK(still > 100);
}

TEST_CASE("snake hook: the follow gain does not depend on the tick length") {
  for (float strength : {0.1f, 0.3f, 0.7f}) {
    const float g8 = snakeGain(strength, 8.0);
    const float g16 = snakeGain(strength, 16.0);
    CHECK(1.0f - (1.0f - g8) * (1.0f - g8) == doctest::Approx(g16).epsilon(1e-3));
    CHECK(snakeGain(strength, 1000.0 / 60.0) == doctest::Approx(strength).epsilon(1e-4));
    CHECK(snakeGain(strength, 500.0) == snakeGain(strength, 50.0));
    CHECK(snakeGain(strength, 0.0) == snakeGain(strength, 1.0));
  }
  CHECK(snakeGain(1.0f, 3.0) == 1.0f);
  CHECK(snakeGain(0.0f, 16.0) == 0.0f);
}

TEST_CASE("snake hook: a hook another copy's topology pass removes is glued to the nearest live vertex") {
  // Two mirror copies close together: the pass around one hook collapses edges at the other.
  Scene scene;
  SceneObject& obj = scene.add("S", makeQuadSphere(64));
  SnakeHookBrush snake;
  Sculptor s;
  const float r = 0.2f;
  StrokeOptions o = snakeOptions({true, false, false});
  o.dyntopo = true;
  o.dyntopoOptions.timeBudgetMs = 0.0;
  REQUIRE(s.beginSnake(obj, snake, o, test::surfacePoint(obj, {0.04f, 1.0f, 0.0f}), r, 0.0f, "Snake Hook"));
  REQUIRE(s.snakeHook(1) != kInvalid);
  const Vec3 start = obj.mesh.positions[s.snakeHook()];
  bool moved = false;
  for (int i = 1; i <= 10; ++i) {
    const Index before = s.snakeHook(1);
    DabTopology topo;
    topo.detail = 0.15f;
    s.snakeTo(start + Vec3{0.0f, 0.15f * r * static_cast<float>(i), 0.0f}, r, 1.0f, topo);
    REQUIRE(s.snakeHook() != kInvalid);
    REQUIRE(s.snakeHook(1) != kInvalid);
    REQUIRE(obj.mesh.vertHe[s.snakeHook(1)] != kInvalid);
    moved = moved || s.snakeHook(1) != before;
  }
  CHECK(moved);  // The case this test is about happened.
  // Both copies pulled their tentacle.
  const Vec3 p0 = obj.mesh.positions[s.snakeHook()], p1 = obj.mesh.positions[s.snakeHook(1)];
  CHECK(p0.y > start.y + 1.2f * r);
  CHECK(p1.y > start.y + 1.2f * r);
  CHECK(std::abs(p0.x + p1.x) < 0.1f * r);
  REQUIRE(s.endStroke());
  test::requireValid(obj.mesh);
}

TEST_CASE("snake hook: a hook pinned on the mirror plane settles when the mirror hook is another vertex") {
  // An uneven mesh at the seam: the copy across the plane hooks a different vertex, so the copies
  // disagree, but the pin holds the primary hook on the plane. The pull stays along the plane
  // and the stroke settles instead of spreading the surface sideways every tick.
  Scene scene;
  Mesh mesh = makeQuadSphere(16);
  const float r = 0.25f;
  Index seam = kInvalid;
  for (Index v = 0; v < mesh.vertexCount(); ++v)
    if (mesh.positions[v].x == 0.0f && mesh.positions[v].y > 0.0f &&
        (seam == kInvalid || mesh.positions[v].y > mesh.positions[seam].y))
      seam = v;
  REQUIRE(seam != kInvalid);
  const Vec3 s0 = mesh.positions[seam];
  const Vec3 press = s0 + Vec3{0.08f * r, 0.0f, 0.0f};  // Far enough off the plane not to merge.
  // The nearest vertex on the other side moves right under the mirrored press.
  Index across = kInvalid;
  for (Index v = 0; v < mesh.vertexCount(); ++v)
    if (mesh.positions[v].x < 0.0f &&
        (across == kInvalid ||
         glm::length(mesh.positions[v] - s0) < glm::length(mesh.positions[across] - s0)))
      across = v;
  REQUIRE(across != kInvalid);
  mesh.positions[across] = Vec3{-press.x, press.y, press.z};
  mesh.computeNormals();
  const Vec3 a0 = mesh.positions[across];
  SceneObject& obj = scene.add("S", std::move(mesh));
  // Adding the object reorders its vertices.
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
    if (obj.mesh.positions[v] == s0) seam = v;
    if (obj.mesh.positions[v] == a0) across = v;
  }
  SnakeHookBrush snake;
  Sculptor s;
  REQUIRE(s.beginSnake(obj, snake, snakeOptions({true, false, false}), press, r, 0.0f, "Snake Hook"));
  REQUIRE(s.snakeHook() == seam);
  REQUIRE(s.snakeHook(1) == across);
  const Vec3 target = s0 + Vec3{0.5f * r, 1.5f * r, 0.2f * r};
  SnakeStep step;
  int still = 0;
  for (int i = 0; i < 200 && still < 3; ++i) {
    step = s.snakeTo(target, r, 0.7f);
    still = step.subDabs == 0 ? still + 1 : 0;
  }
  CHECK(still == 3);
  const Vec3 hook = obj.mesh.positions[seam];
  CHECK(hook.x == 0.0f);
  CHECK(std::abs(hook.y - target.y) < 1e-4f * r);
  CHECK(std::abs(hook.z - target.z) < 1e-4f * r);
  REQUIRE(s.endStroke());
}

TEST_CASE("snake hook: a hook on the plane following an off-plane cursor at low gain settles") {
  // The pull along the plane shrinks to under a float step at the hook; it must count as none.
  for (const float gain : {0.2f, 0.4f}) {
    Scene scene;
    SceneObject& obj = scene.add("S", makeQuadSphere(16));
    SnakeHookBrush snake;
    Sculptor s;
    const float r = 0.15f;
    REQUIRE(s.beginSnake(obj, snake, snakeOptions({true, false, false}), test::surfacePoint(obj, {0, 1, 0}), r, 0.0f,
                         "Snake Hook"));
    const Vec3 start = obj.mesh.positions[s.snakeHook()];
    REQUIRE(start.x == 0.0f);
    const Vec3 target = start + Vec3{0.4f * r, 0.73f * r, 0.31f * r};
    int still = 0, ticks = 0;
    for (; ticks < 600 && still < 3; ++ticks) still = s.snakeTo(target, r, gain).subDabs == 0 ? still + 1 : 0;
    INFO("gain " << gain << " ticks " << ticks);
    CHECK(still == 3);
    REQUIRE(s.endStroke());
  }
}
