// Randomised strokes for the Phase 7 brushes: random symmetry specs, brushes, radii, masks, face
// sets and dynamic topology, each checked against the stroke contract (valid mesh, finite
// positions, normals and bounds up to date, mask in range, masked vertices still, plane vertices
// on their planes) and undone bit-exactly. PLEGL_FUZZ_CASES raises the number of cases.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>

#include "TestUtil.h"
#include "mesh/Primitives.h"
#include "scene/Scene.h"
#include "sculpt/Sculptor.h"
#include "sculpt/Undo.h"

using namespace plegl;

namespace {

int fuzzCases(int fallback) {
  if (const char* v = std::getenv("PLEGL_FUZZ_CASES")) return std::max(1, std::atoi(v));
  return fallback;
}

// PLEGL_FUZZ_FIRST starts at a given case, to rerun one that failed.
int fuzzFirst() {
  const char* v = std::getenv("PLEGL_FUZZ_FIRST");
  return v ? std::max(0, std::atoi(v)) : 0;
}

struct Rng {
  std::mt19937 gen;
  explicit Rng(unsigned seed) : gen(seed) {}
  float uniform(float a, float b) { return std::uniform_real_distribution<float>(a, b)(gen); }
  int below(int n) { return std::uniform_int_distribution<int>(0, n - 1)(gen); }
  bool coin(float p) { return uniform(0.0f, 1.0f) < p; }
  Vec3 direction() {
    Vec3 d{0.0f};
    while (glm::length(d) < 0.2f) d = {uniform(-1.0f, 1.0f), uniform(-1.0f, 1.0f), uniform(-1.0f, 1.0f)};
    return glm::normalize(d);
  }
};

SymmetrySpec randomSpec(Rng& r) {
  static const int kCounts[] = {1, 1, 1, 2, 3, 4, 5, 8, 12, 16, 32};
  SymmetrySpec s;
  s.x = r.coin(0.6f);
  s.y = r.coin(0.3f);
  s.z = r.coin(0.3f);
  s.radialAxis = static_cast<Axis>(r.below(3));
  s.radialCount = kCounts[r.below(11)];
  return sanitized(s);
}

StrokeOptions optionsFor(const SymmetrySpec& s, float strength) {
  StrokeOptions o;
  o.strength = strength;
  o.symmetryX = s.x;
  o.symmetryY = s.y;
  o.symmetryZ = s.z;
  o.radialAxis = s.radialAxis;
  o.radialCount = s.radialCount;
  return o;
}

// Random bumps, a soft mask blob with a fully masked core, and two face sets split by a plane.
void decorate(SceneObject& obj, Rng& r) {
  DrawBrush draw;
  Sculptor s;
  s.beginStroke(obj, draw, optionsFor({}, 0.8f), "pre");
  for (int i = 0; i < 4; ++i) s.dab(test::surfacePoint(obj, r.direction()), r.uniform(0.1f, 0.3f), 1.0f);
  s.endStroke();
  Mesh& m = obj.mesh;
  m.ensureMask();
  const Vec3 blob = r.direction();
  const float edge = r.uniform(0.3f, 0.9f);
  for (Index v = 0; v < m.vertexCount(); ++v) {
    const float d = glm::dot(glm::normalize(m.positions[v]), blob);
    m.mask[v] = std::clamp((d - edge) * 8.0f, 0.0f, 1.0f);
  }
  m.ensureFaceSets();
  const Vec3 split = r.direction();
  for (Index f = 0; f < m.faceCount(); ++f) m.faceSets[f] = glm::dot(m.faceCentroid(f), split) > 0.1f ? 2 : 1;
}

void requireSane(const SceneObject& obj) {
  test::requireValid(obj.mesh);
  for (const Vec3& p : obj.mesh.positions) REQUIRE((std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)));
  test::requireNormalsMatchFullRecompute(obj.mesh);
  test::requireBoundsContainPositions(obj);
  for (float m : obj.mesh.mask) REQUIRE((m >= 0.0f && m <= 1.0f));
}

bool same(const Vec3& a, const Vec3& b) { return std::memcmp(&a, &b, sizeof(Vec3)) == 0; }

void requireUndoRestores(Scene& scene, SceneObject& obj, std::optional<StrokeUndo> undo, const Mesh& before) {
  if (undo) {
    UndoStack stack;
    stack.push(std::move(*undo));
    REQUIRE_FALSE(stack.undo(scene).empty());
  }
  REQUIRE(obj.mesh.faceCount() == before.faceCount());
  CHECK(test::hashBits(obj.mesh.positions) == test::hashBits(before.positions));
  CHECK(test::hashBits(obj.mesh.normals) == test::hashBits(before.normals));
  CHECK(obj.mesh.mask == before.mask);
  CHECK(obj.mesh.faceSets == before.faceSets);
  test::requireBoundsContainPositions(obj);
}

}  // namespace

TEST_CASE("fuzz: random symmetry, brushes and strokes keep the contract and undo exactly") {
  DrawBrush draw;
  ClayBrush clay;
  SmoothBrush smooth;
  InflateBrush inflate;
  FlattenBrush flatten;
  CreaseBrush crease;
  MaskBrush mask;
  FaceSetBrush faceSet;
  SnakeHookBrush snake;
  const Brush* dabBrushes[] = {&draw, &clay, &smooth, &inflate, &flatten, &crease, &mask, &faceSet};
  enum Kind { kGrab = 8, kSnake = 9, kKinds = 10 };
  const int cases = fuzzCases(60);
  for (int c = fuzzFirst(); c < fuzzFirst() + cases; ++c) {
    Rng r(1000u + static_cast<unsigned>(c));
    const SymmetrySpec spec = randomSpec(r);
    const int kind = r.below(kKinds);
    const bool moves = kind < 6 || kind == kGrab || kind == kSnake;
    const bool dyntopo = (kind < 6 || kind == kSnake) && r.coin(0.35f);
    const float radius = r.uniform(0.08f, 0.45f);
    const float strength = r.uniform(0.2f, 1.0f);
    INFO("case " << c << ", kind " << kind << ", copies " << copyCount(spec) << ", dyntopo " << dyntopo
                 << ", radius " << radius);
    Scene scene;
    SceneObject& obj = scene.add("S", makeQuadSphere(20));
    decorate(obj, r);
    const Mesh before = obj.mesh;
    // The pin tolerance the stroke uses: from the bounds at its start.
    const float eps = kPinTolerance * glm::length(obj.bvh.bounds().extent());
    StrokeOptions o = optionsFor(spec, strength);
    o.dyntopo = dyntopo;
    o.dyntopoOptions.timeBudgetMs = 0.0;
    o.faceSetAutoMask = r.coin(0.2f);
    o.lockFaceSetBoundaries = r.coin(0.2f);
    o.invert = r.coin(0.3f);
    const Vec3 a = r.direction();
    const Vec3 b = glm::normalize(a + r.direction() * 0.6f);
    const float detail = radius / 6.0f;
    Sculptor s;
    if (kind < 8) {
      s.beginStroke(obj, *dabBrushes[kind], o, dabBrushes[kind]->name());
      REQUIRE(s.active());
      for (int i = 0; i < 12; ++i) {
        const Vec3 p = test::surfacePoint(obj, glm::normalize(glm::mix(a, b, static_cast<float>(i) / 11.0f)));
        DabTopology topo;
        if (dyntopo) {
          topo.detail = detail;
          Bvh::ClosestHit hit;
          if (obj.bvh.closestPoint(obj.mesh, p, radius, hit)) topo.hintFace = hit.face;
        }
        s.dab(p, radius, strength, topo);
      }
    } else if (kind == kGrab) {
      if (!s.beginGrab(obj, o, test::surfacePoint(obj, a), radius, "Grab")) continue;
      const Vec3 pull = r.direction() * r.uniform(0.05f, 0.4f);
      for (int i = 1; i <= 8; ++i) s.grab(pull * (static_cast<float>(i) / 8.0f));
    } else {
      const Vec3 start = test::surfacePoint(obj, a);
      if (!s.beginSnake(obj, snake, o, start, radius, r.uniform(0.0f, 1.0f), "Snake Hook")) continue;
      const Vec3 side = r.direction() * 0.3f;
      for (int i = 1; i <= 10; ++i) {
        DabTopology topo;
        if (dyntopo) topo.detail = detail;
        const float t = static_cast<float>(i) / 10.0f;
        s.snakeTo(start + (a + side * t) * (0.6f * t), radius, r.uniform(0.3f, 1.0f), topo);
      }
    }
    auto undo = s.endStroke();
    requireSane(obj);
    if (moves && !dyntopo) {
      // Fully masked vertices never move; vertices on a mirror plane stay on it (or untouched).
      for (Index v = 0; v < before.vertexCount(); ++v) {
        const Vec3& p0 = before.positions[v];
        const Vec3& p = obj.mesh.positions[v];
        if (before.mask[v] >= 1.0f && !same(p, p0)) {
          INFO("masked vertex " << v << " moved from (" << p0.x << ", " << p0.y << ", " << p0.z << ") to (" << p.x
                                << ", " << p.y << ", " << p.z << ")");
          REQUIRE(same(p, p0));
        }
        for (int axis = 0; axis < 3; ++axis) {
          const bool mirrored = axis == 0 ? spec.x : axis == 1 ? spec.y : spec.z;
          if (mirrored && std::abs(p0[axis]) <= eps && !same(p, p0)) {
            INFO("vertex " << v << " left plane " << axis << ": " << p[axis]);
            REQUIRE(p[axis] == 0.0f);
          }
        }
      }
    }
    requireUndoRestores(scene, obj, std::move(undo), before);
  }
}

TEST_CASE("fuzz: random poses keep the contract, hold the mask and undo exactly") {
  const int cases = fuzzCases(40);
  for (int c = fuzzFirst(); c < fuzzFirst() + cases; ++c) {
    Rng r(5000u + static_cast<unsigned>(c));
    const SymmetrySpec spec = randomSpec(r);
    const bool tube = r.coin(0.5f);
    PoseOptions pose;
    pose.mode = r.coin(0.3f) ? PoseMode::Mask : PoseMode::Topology;
    pose.softness = r.uniform(0.0f, 1.0f);
    const float radius = r.uniform(0.2f, 1.2f);
    INFO("case " << c << ", copies " << copyCount(spec) << ", tube " << tube << ", mask mode "
                 << (pose.mode == PoseMode::Mask) << ", softness " << pose.softness << ", radius " << radius);
    Scene scene;
    SceneObject& obj =
        scene.add("P", tube ? test::makeTube(16, 40, 0.1f, 2.0f) : makeQuadSphere(20));
    Mesh& m = obj.mesh;
    m.ensureMask();
    const Vec3 dir = r.direction();
    const float edge = r.uniform(-0.5f, 0.5f);
    const Vec3 centre = tube ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{0.0f};
    for (Index v = 0; v < m.vertexCount(); ++v) {
      const float d = glm::dot(m.positions[v] - centre, dir);
      m.mask[v] = std::clamp((edge - d) * 6.0f, 0.0f, 1.0f);
    }
    if (r.coin(0.3f)) {
      m.ensureFaceSets();
      for (Index f = 0; f < m.faceCount(); ++f) m.faceSets[f] = m.faceCentroid(f).x > 0.05f ? 3 : 1;
    }
    const Mesh before = m;
    StrokeOptions o = optionsFor(spec, 1.0f);
    o.faceSetAutoMask = r.coin(0.2f);
    o.lockFaceSetBoundaries = r.coin(0.2f);
    Vec3 click;
    if (tube) {
      const float y = r.uniform(0.1f, 1.9f), a = r.uniform(0.0f, 6.2832f);
      RayHit hit;
      const Vec3 out{std::cos(a), 0.0f, std::sin(a)};
      if (!obj.bvh.raycast(m, Ray{out + Vec3{0.0f, y, 0.0f}, -out}, hit)) continue;
      click = hit.position;
    } else {
      click = test::surfacePoint(obj, r.direction());
    }
    Sculptor s;
    if (!s.beginPose(obj, o, pose, click, radius, "Pose")) {
      REQUIRE_FALSE(s.active());
      CHECK(s.lastPoseFailure() != PoseFailure::None);
      CHECK(test::hashBits(m.positions) == test::hashBits(before.positions));
      continue;
    }
    const Vec3 pivot = s.poseBuild()->rigs[0].pivot;
    for (int i = 0; i < 3; ++i) {
      const Vec3 axis = r.direction();
      const float angle = r.uniform(-2.5f, 2.5f);
      const Vec3 target = pivot + glm::angleAxis(angle, axis) * (click - pivot);
      s.pose(target, r.direction());
      for (const Vec3& p : m.positions) REQUIRE((std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)));
    }
    auto undo = s.endStroke();
    test::requireValid(m);
    test::requireNormalsBitExact(m);
    test::requireBoundsContainPositions(obj);
    for (Index v = 0; v < m.vertexCount(); ++v)
      if (before.mask[v] >= 1.0f) REQUIRE(same(m.positions[v], before.positions[v]));
    requireUndoRestores(scene, obj, std::move(undo), before);
  }
}
