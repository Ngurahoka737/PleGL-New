#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <tuple>

#include "TestUtil.h"
#include "core/Parallel.h"
#include "mesh/Primitives.h"
#include "multires/MultiresOps.h"
#include "scene/Scene.h"
#include "sculpt/LayerOps.h"
#include "sculpt/Sculptor.h"
#include "sculpt/Symmetry.h"
#include "sculpt/Undo.h"

using namespace plegl;

namespace {

SymmetrySpec spec(bool x, bool y, bool z, Axis axis = Axis::Y, int count = 1) { return {x, y, z, axis, count}; }

float maxDiff(const Mat3& a, const Mat3& b) {
  float d = 0.0f;
  for (int c = 0; c < 3; ++c)
    for (int r = 0; r < 3; ++r) d = std::max(d, std::abs(a[c][r] - b[c][r]));
  return d;
}

Mat3 rotation(const Vec3& axis, float angle) { return glm::mat3_cast(glm::angleAxis(angle, glm::normalize(axis))); }

StrokeOptions withSymmetry(const SymmetrySpec& s, float strength = 0.5f) {
  StrokeOptions o;
  o.strength = strength;
  o.symmetryX = s.x;
  o.symmetryY = s.y;
  o.symmetryZ = s.z;
  o.radialAxis = s.radialAxis;
  o.radialCount = s.radialCount;
  return o;
}

// Index of the vertex at L * p for every vertex (within 1e-5), or kInvalid.
std::vector<Index> imageMap(const Mesh& m, const Mat3& L) {
  auto key = [](const Vec3& p) {
    return std::tuple{static_cast<long>(std::lround(p.x * 1e4f)), static_cast<long>(std::lround(p.y * 1e4f)),
                      static_cast<long>(std::lround(p.z * 1e4f))};
  };
  std::map<std::tuple<long, long, long>, std::vector<Index>> grid;
  for (Index v = 0; v < m.vertexCount(); ++v) grid[key(m.positions[v])].push_back(v);
  std::vector<Index> out(static_cast<std::size_t>(m.vertexCount()), kInvalid);
  for (Index v = 0; v < m.vertexCount(); ++v) {
    const Vec3 q = transform(L, m.positions[v]);
    const auto [kx, ky, kz] = key(q);
    for (long dx = -1; dx <= 1 && out[v] == kInvalid; ++dx)
      for (long dy = -1; dy <= 1 && out[v] == kInvalid; ++dy)
        for (long dz = -1; dz <= 1 && out[v] == kInvalid; ++dz) {
          const auto it = grid.find({kx + dx, ky + dy, kz + dz});
          if (it == grid.end()) continue;
          for (Index u : it->second)
            if (glm::length(m.positions[u] - q) < 1e-5f) out[v] = u;
        }
  }
  return out;
}

// Runs `n` dabs along a path between two directions on a closed object around the origin.
bool strokePath(Sculptor& s, SceneObject& obj, Vec3 a, Vec3 b, float r, int n, float strength = 1.0f,
                float detail = 0.0f) {
  bool any = false;
  for (int i = 0; i < n; ++i) {
    const float t = n == 1 ? 0.0f : static_cast<float>(i) / static_cast<float>(n - 1);
    DabTopology topo;
    topo.detail = detail;
    const Vec3 c = test::surfacePoint(obj, glm::mix(a, b, t));
    if (detail > 0.0f) {
      Bvh::ClosestHit hit;
      if (obj.bvh.closestPoint(obj.mesh, c, r, hit)) topo.hintFace = hit.face;
    }
    any |= s.dab(c, r, strength, topo);
  }
  return any;
}

struct Brushes {
  DrawBrush draw;
  ClayBrush clay;
  SmoothBrush smooth;
  InflateBrush inflate;
  FlattenBrush flatten;
  CreaseBrush crease;
  std::vector<const Brush*> moving() const { return {&draw, &clay, &smooth, &inflate, &flatten, &crease}; }
};

// Some bumps first, so Smooth and Flatten have something to work on.
void roughen(SceneObject& obj) {
  DrawBrush draw;
  Sculptor s;
  s.beginStroke(obj, draw, withSymmetry({}, 0.8f), "pre");
  for (Vec3 d : {Vec3{0.3f, 0.9f, 0.1f}, Vec3{-0.2f, 0.9f, -0.3f}, Vec3{0.05f, 1.0f, 0.4f}, Vec3{0.9f, 0.1f, 0.3f}})
    s.dab(test::surfacePoint(obj, d), 0.15f, 1.0f);
  s.endStroke();
}

std::vector<float> heights(const SceneObject& obj, const std::vector<Vec3>& dirs) {
  std::vector<float> out;
  for (const Vec3& d : dirs) out.push_back(test::radiusAlong(obj, d));
  return out;
}

}  // namespace

// ----- Group -----------------------------------------------------------------------------------------

TEST_CASE("symmetry: group sizes") {
  CHECK(copyCount(spec(false, false, false)) == 1);
  CHECK(copyCount(spec(true, false, false)) == 2);
  CHECK(copyCount(spec(true, true, false)) == 4);
  CHECK(copyCount(spec(true, true, true)) == 8);
  CHECK(copyCount(spec(false, false, false, Axis::Y, 6)) == 6);
  CHECK(copyCount(spec(true, false, false, Axis::Y, 6)) == 12);
  CHECK(copyCount(spec(true, false, true, Axis::Y, 6)) == 12);  // R_3 is SxSz.
  CHECK(copyCount(spec(true, false, true, Axis::Y, 5)) == 20);
  CHECK(copyCount(spec(true, false, false, Axis::Y, 32)) == 64);
  CHECK(copyCount(spec(true, true, false, Axis::Y, 32)) == 128);
  CHECK(maxRadialCount(spec(true, true, false, Axis::Y, 1)) == 16);
  CHECK(maxRadialCount(spec(false, false, false, Axis::Z, 1)) == 32);
}

TEST_CASE("symmetry: sanitized clamps the copies to the limit") {
  const SymmetrySpec s = sanitized(spec(true, true, false, Axis::Y, 32));
  CHECK(s.radialCount == 16);
  CHECK(copyCount(s) == 64);
  // An odd count with two perpendicular mirrors has twice the copies of an even one.
  const SymmetrySpec odd = sanitized(spec(true, false, true, Axis::Y, 31));
  CHECK(copyCount(odd) <= kMaxSymmetryCopies);
  CHECK(odd.radialCount == 30);
  CHECK(sanitized(spec(false, false, false, Axis::X, 0)).radialCount == 1);
  CHECK(sanitized(spec(false, false, false, Axis::X, 99)).radialCount == 32);
  SymmetrySpec bad = spec(true, false, false);
  bad.radialAxis = static_cast<Axis>(7);
  CHECK(sanitized(bad).radialAxis == Axis::Y);
}

TEST_CASE("symmetry: the ops form a group") {
  for (const SymmetrySpec& s : {spec(true, false, false), spec(true, true, true), spec(false, false, false, Axis::Y, 6),
                                spec(true, false, true, Axis::Y, 5), spec(true, false, false, Axis::Z, 32),
                                spec(false, true, false, Axis::X, 3)}) {
    const SymmetryGroup g(s);
    INFO("size " << g.size());
    REQUIRE(g.size() == copyCount(s));
    CHECK(maxDiff(g.op(0).L, Mat3{1.0f}) == 0.0f);
    for (int a = 0; a < g.size(); ++a) {
      CHECK(std::abs(g.op(a).det - glm::determinant(g.op(a).L)) < 1e-5f);
      CHECK(g.mul(a, g.inverse(a)) == 0);
      CHECK(g.mul(g.inverse(a), a) == 0);
      for (int b = 0; b < g.size(); ++b) {
        // Closure: every product is an op of the group.
        CHECK(maxDiff(g.op(a).L * g.op(b).L, g.op(g.mul(a, b)).L) < 1e-5f);
      }
    }
    CHECK(g.closure(0) == 1);
    CHECK(g.closure(g.all()) == g.all());
  }
  // Quarter turns are exact.
  const SymmetryGroup g4(spec(false, false, false, Axis::Y, 4));
  const Vec3 p{0.3f, 0.7f, -0.2f};
  CHECK(transform(g4.op(1).L, p) == Vec3{-0.2f, 0.7f, -0.3f});
  CHECK(transform(g4.op(2).L, p) == Vec3{-0.3f, 0.7f, 0.2f});
}

TEST_CASE("symmetry: rotations transform as axial vectors") {
  const SymmetryGroup g(spec(true, false, true, Axis::Y, 5));
  const Vec3 omega = glm::normalize(Vec3{0.3f, 0.8f, -0.5f});
  const float theta = 0.7f;
  const Mat3 R = rotation(omega, theta);
  for (int k = 0; k < g.size(); ++k) {
    const SymmetryOp& o = g.op(k);
    const Mat3 conjugated = o.L * R * glm::transpose(o.L);
    const Mat3 expected = rotation(o.L * omega * o.det, theta);
    CHECK(maxDiff(conjugated, expected) < 1e-5f);
  }
}

TEST_CASE("symmetry: projectors, stabilisers and the planes and axes topology keeps") {
  const SymmetryGroup x(spec(true, false, false));
  CHECK(maxDiff(x.fixProjector(1), Mat3{Vec3{0, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}}) == 0.0f);
  CHECK(x.mirrorPlanes() == 1);
  CHECK(x.rotationAxes() == 0);
  const SymmetryGroup radial(spec(false, false, false, Axis::Y, 3));
  CHECK(maxDiff(radial.fixProjector(1), Mat3{Vec3{0, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 0}}) == 0.0f);
  CHECK(radial.mirrorPlanes() == 0);
  CHECK(radial.rotationAxes() == 2);
  // X with a half turn around Y also mirrors across z = 0.
  const SymmetryGroup xr(spec(true, false, false, Axis::Y, 2));
  CHECK(xr.mirrorPlanes() == (1 | 4));
  CHECK(xr.stabilizer({0.0f, 0.5f, 0.0f}, 1e-6f) == xr.all());
  CHECK(xr.stabilizer({0.3f, 0.5f, 0.2f}, 1e-6f) == 1);
  CHECK(pinnedAxes(1, 2, {1e-7f, 0.4f, 0.5f}, 1e-6f) == 1);
  CHECK(pinnedAxes(0, 2, {1e-7f, 0.4f, -1e-7f}, 1e-6f) == (1 | 4));
  CHECK(pinnedAxes(0, 2, {1e-7f, 0.4f, 0.5f}, 1e-6f) == 0);
}

// ----- Clusters --------------------------------------------------------------------------------------

TEST_CASE("symmetry: copies near a plane merge onto it, farther ones stay apart") {
  const SymmetryGroup g(spec(true, false, false));
  CopyClusters out;
  const float r = 0.2f;
  REQUIRE(clusterCopies(g, {0.03f * r, 0.5f, 0.1f}, kCopyMergeFraction * r, out) == 1);
  CHECK(out[0].center.x == 0.0f);
  CHECK(out[0].center.y == 0.5f);
  CHECK(out[0].merged());
  CHECK(out[0].members == 3);
  // The merged dab's vectors lose the part across the plane.
  CHECK(transform(out[0].polar, {1.0f, 2.0f, 3.0f}) == Vec3{0.0f, 2.0f, 3.0f});
  REQUIRE(clusterCopies(g, {0.06f * r, 0.5f, 0.1f}, kCopyMergeFraction * r, out) == 2);
  CHECK(out[0].center == Vec3{0.06f * r, 0.5f, 0.1f});
  CHECK(out[1].center == Vec3{-0.06f * r, 0.5f, 0.1f});
  CHECK_FALSE(out[1].merged());
}

TEST_CASE("symmetry: clusters map to clusters under every op") {
  const SymmetryGroup g(spec(true, false, true, Axis::Y, 6));
  const float r = 0.3f;
  for (Vec3 c : {Vec3{0.5f, 0.7f, 0.2f}, Vec3{0.005f, 0.7f, 0.4f}, Vec3{0.004f, 0.9f, 0.003f}}) {
    CopyClusters base;
    const int n = clusterCopies(g, c, kCopyMergeFraction * r, base);
    for (int k = 0; k < g.size(); ++k) {
      CopyClusters image;
      REQUIRE(clusterCopies(g, transform(g.op(k).L, c), kCopyMergeFraction * r, image) == n);
      for (int i = 0; i < n; ++i) {
        const Vec3 mapped = transform(g.op(k).L, base[static_cast<std::size_t>(i)].center);
        float best = 1e9f;
        for (int j = 0; j < n; ++j) best = std::min(best, glm::length(image[static_cast<std::size_t>(j)].center - mapped));
        CHECK(best < 1e-5f);
      }
    }
  }
}

TEST_CASE("symmetry: radial 32 merges near its axis") {
  const SymmetryGroup g(spec(false, false, false, Axis::Y, 32));
  CopyClusters out;
  const float r = 1.0f;
  CHECK(clusterCopies(g, {0.25f, 0.5f, 0.0f}, kCopyMergeFraction * r, out) == 1);
  CHECK(std::abs(out[0].center.x) < 1e-6f);
  CHECK(std::abs(out[0].center.z) < 1e-6f);
  CHECK(clusterCopies(g, {0.26f, 0.5f, 0.0f}, kCopyMergeFraction * r, out) == 32);
}

TEST_CASE("symmetry: seam pins") {
  const SymmetryGroup x(spec(true, false, false));
  const SeamPin onPlane = seamPin(x, {1e-7f, 0.5f, 0.2f}, 1e-5f);
  CHECK(maxDiff(onPlane.P, Mat3{Vec3{0, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}}) == 0.0f);
  CHECK(onPlane.zeroAxes == 1);
  const Vec3 q = snapPinned(onPlane, {-0.0f, 0.5f, 0.2f}, {-3e-7f, 0.6f, 0.25f});
  CHECK(q.x == 0.0f);
  CHECK_FALSE(std::signbit(q.x));
  CHECK(q.y == 0.6f);
  CHECK(q.z == 0.25f);
  const SymmetryGroup radial(spec(false, false, false, Axis::Y, 4));
  const SeamPin onAxis = seamPin(radial, {1e-7f, 0.5f, -2e-7f}, 1e-5f);
  CHECK(maxDiff(onAxis.P, Mat3{Vec3{0, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 0}}) == 0.0f);
  CHECK(onAxis.zeroAxes == (1 | 4));
  const Vec3 a = snapPinned(onAxis, {1e-7f, 0.5f, -2e-7f}, {-1e-6f, 0.55f, 3e-6f});
  CHECK(a == Vec3{0.0f, 0.55f, 0.0f});
  CHECK_FALSE(std::signbit(a.x));
  CHECK_FALSE(std::signbit(a.z));
}

// ----- Strokes ---------------------------------------------------------------------------------------

TEST_CASE("symmetry: draw makes one equal bump per copy") {
  Scene scene;
  SceneObject& obj = scene.add("S", makeQuadSphere(48));
  DrawBrush draw;
  Sculptor s;
  s.beginStroke(obj, draw, withSymmetry(spec(true, true, true), 1.0f), "Draw");
  const Vec3 d{0.5f, 0.6f, 0.62f};
  const Vec3 c = test::surfacePoint(obj, d);
  for (int i = 0; i < 6; ++i) REQUIRE(s.dab(c, 0.25f, 1.0f));
  CHECK(s.lastDab().clusters == 8);
  REQUIRE(s.endStroke());
  std::vector<Vec3> dirs;
  for (float sx : {1.0f, -1.0f})
    for (float sy : {1.0f, -1.0f})
      for (float sz : {1.0f, -1.0f}) dirs.push_back({d.x * sx, d.y * sy, d.z * sz});
  const std::vector<float> h = heights(obj, dirs);
  CHECK(h[0] > 1.02f);
  for (float v : h) CHECK(v == doctest::Approx(h[0]).epsilon(0.002));
  test::requireNormalsMatchFullRecompute(obj.mesh);
  test::requireBoundsContainPositions(obj);
}

TEST_CASE("symmetry: radial draw makes one equal bump per copy around the axis") {
  Scene scene;
  // 48 segments around Y: every sixth of a turn maps the tube onto itself.
  SceneObject& obj = scene.add("T", test::makeTube(48, 24, 0.5f, 2.0f));
  DrawBrush draw;
  Sculptor s;
  s.beginStroke(obj, draw, withSymmetry(spec(false, false, false, Axis::Y, 6), 1.0f), "Draw");
  const float angle = 0.3f;
  auto at = [&](float a) {
    const Vec3 dir{std::cos(a), 0.0f, std::sin(a)};
    RayHit hit;
    REQUIRE(obj.bvh.raycast(obj.mesh, Ray{dir * 3.0f + Vec3{0, 1, 0}, -dir}, hit));
    return hit.position;
  };
  const Vec3 c = at(angle);
  for (int i = 0; i < 6; ++i) REQUIRE(s.dab(c, 0.25f, 1.0f));
  CHECK(s.lastDab().clusters == 6);
  REQUIRE(s.endStroke());
  std::vector<float> h;
  for (int k = 0; k < 6; ++k) {
    // Copy k sits at the turn by +k/6 (right-handed about +Y, so the angle in x-z decreases).
    const float a = angle - static_cast<float>(k) * 2.0f * 3.14159265f / 6.0f;
    const Vec3 p = at(a);
    h.push_back(std::sqrt(p.x * p.x + p.z * p.z));
  }
  CHECK(h[0] > 0.51f);
  for (float v : h) CHECK(v == doctest::Approx(h[0]).epsilon(0.002));
}

TEST_CASE("symmetry: overlapping copies share the seam instead of doubling it") {
  // A stroke along z, its X mirror 2 x0 away. Vertex heights (along the sphere's radius) are
  // compared with the same stroke alone.
  const float r = 0.3f;
  auto run = [&](float x0, bool symmetric) {
    Scene scene;
    SceneObject& obj = scene.add("S", makeQuadSphere(64));
    const Mesh before = obj.mesh;
    auto nearest = [&](Vec3 dir) {
      const Vec3 p = test::surfacePoint(obj, dir);
      Index best = 0;
      for (Index v = 1; v < before.vertexCount(); ++v)
        if (glm::length(before.positions[v] - p) < glm::length(before.positions[best] - p)) best = v;
      return best;
    };
    const Index centre = nearest({x0, 1.0f, 0.0f}), seam = nearest({0.0f, 1.0f, 0.0f});
    DrawBrush draw;
    Sculptor s;
    s.beginStroke(obj, draw, withSymmetry(spec(symmetric, false, false), 0.5f), "Draw");
    for (int i = 0; i < 60; ++i) {
      const float z = -0.3f + 0.6f * static_cast<float>(i) / 59.0f;
      s.dab(test::surfacePoint(obj, {x0, 1.0f, z}), r, 0.5f);
    }
    s.endStroke();
    auto height = [&](Index v) {
      return glm::dot(obj.mesh.positions[v] - before.positions[v], glm::normalize(before.positions[v]));
    };
    return std::pair{height(centre), height(seam)};
  };
  const float x0 = 0.3f * r;
  const auto [lonePeak, loneSeam] = run(x0, false);
  const auto [peak, seam] = run(x0, true);
  CHECK(seam <= loneSeam * 1.02f + 1e-4f);
  CHECK(peak >= 0.85f * lonePeak);
  // Just inside and just outside the merge distance the result hardly changes.
  const auto [inside, insideSeam] = run(0.049f * r, true);
  const auto [outside, outsideSeam] = run(0.051f * r, true);
  CHECK(std::abs(inside - outside) < 0.05f * inside);
  CHECK(std::abs(insideSeam - outsideSeam) < 0.05f * insideSeam);
}

TEST_CASE("symmetry: vertices on the plane stay exactly on it under every brush") {
  Brushes b;
  for (const Brush* brush : b.moving()) {
    INFO(brush->name());
    Scene scene;
    SceneObject& obj = scene.add("S", makeQuadSphere(40));
    roughen(obj);
    const Mesh before = obj.mesh;
    const float eps = kPinTolerance * glm::length(obj.bvh.bounds().extent());
    Sculptor s;
    s.beginStroke(obj, *brush, withSymmetry(spec(true, false, false), 0.8f), brush->name());
    // Zig-zag across the plane, 200 dabs.
    for (int i = 0; i < 200; ++i) {
      const float t = static_cast<float>(i) / 199.0f;
      const float x = 0.25f * std::sin(t * 25.0f);
      s.dab(test::surfacePoint(obj, {x, 1.0f, -0.5f + t}), 0.2f, 0.8f);
    }
    REQUIRE(s.endStroke());
    int pinned = 0;
    for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
      if (std::abs(before.positions[v].x) > eps || obj.mesh.positions[v] == before.positions[v]) continue;
      const float x = obj.mesh.positions[v].x;
      if (x != 0.0f || std::signbit(x)) {
        INFO("vertex " << v << " left the plane: x = " << x);
        REQUIRE(x == 0.0f);
        REQUIRE_FALSE(std::signbit(x));
      }
      ++pinned;
    }
    CHECK(pinned > 10);
    test::requireNormalsMatchFullRecompute(obj.mesh);
    test::requireBoundsContainPositions(obj);
  }
}

TEST_CASE("symmetry: dynamic topology keeps vertices on the plane and undoes exactly") {
  Scene scene;
  SceneObject& obj = scene.add("S", makeQuadSphere(24));
  const float eps = kPinTolerance * glm::length(obj.bvh.bounds().extent());
  const Mesh before = obj.mesh;
  std::set<Index> onPlane;
  for (Index v = 0; v < before.vertexCount(); ++v)
    if (std::abs(before.positions[v].x) <= eps) onPlane.insert(v);
  DrawBrush draw;
  Sculptor s;
  StrokeOptions o = withSymmetry(spec(true, false, false), 0.6f);
  o.dyntopo = true;
  o.dyntopoOptions.timeBudgetMs = 0.0;
  s.beginStroke(obj, draw, o, "Draw");
  for (int i = 0; i < 100; ++i) {
    const float t = static_cast<float>(i) / 99.0f;
    const float x = 0.2f * std::sin(t * 20.0f);
    DabTopology topo;
    topo.detail = 0.02f;
    const Vec3 c = test::surfacePoint(obj, {x, 1.0f, -0.4f + 0.8f * t});
    Bvh::ClosestHit hit;
    if (obj.bvh.closestPoint(obj.mesh, c, 0.15f, hit)) topo.hintFace = hit.face;
    s.dab(c, 0.15f, 0.6f, topo);
  }
  auto undo = s.endStroke();
  REQUIRE(undo);
  REQUIRE(s.lastStrokeTopology().dyntopo);
  CHECK(s.lastStrokeTopology().splits > 100);
  test::requireValid(obj.mesh);
  const Mesh& m = obj.mesh;
  // Every vertex near the plane is on it; vertices of the original mesh near the plane were
  // either left alone or pinned.
  int exact = 0;
  for (Index v = 0; v < m.vertexCount(); ++v) {
    const float x = m.positions[v].x;
    if (std::abs(x) > eps) continue;
    const bool original = std::any_of(onPlane.begin(), onPlane.end(), [&](Index u) {
      return std::memcmp(&before.positions[u], &m.positions[v], sizeof(Vec3)) == 0;
    });
    if (!original) {
      INFO("vertex " << v << " near the plane at x = " << x);
      REQUIRE(x == 0.0f);
      REQUIRE_FALSE(std::signbit(x));
      ++exact;
    }
  }
  CHECK(exact > 10);
  test::requireNormalsMatchFullRecompute(m);
  UndoStack stack;
  stack.push(std::move(*undo));
  REQUIRE_FALSE(stack.undo(scene).empty());
  CHECK(test::hashBits(obj.mesh.positions) == test::hashBits(before.positions));
  CHECK(obj.mesh.faceCount() == before.faceCount());
}

TEST_CASE("symmetry: radial copies near the axis keep the brush contract") {
  Brushes b;
  for (const Brush* brush : b.moving()) {
    INFO(brush->name());
    Scene scene;
    SceneObject& obj = scene.add("S", makeQuadSphere(48));
    roughen(obj);
    const Mesh before = obj.mesh;
    const float r = 0.3f;
    Sculptor s;
    s.beginStroke(obj, *brush, withSymmetry(spec(false, false, false, Axis::Y, 8), 1.0f), brush->name());
    REQUIRE(s.dab(test::surfacePoint(obj, {0.3f * r, 1.0f, 0.05f}), r, 1.0f));
    CHECK(s.lastDab().clusters == 8);
    REQUIRE(s.endStroke());
    float most = 0.0f;
    for (Index v = 0; v < obj.mesh.vertexCount(); ++v)
      most = std::max(most, glm::length(obj.mesh.positions[v] - before.positions[v]));
    CHECK(most <= r * kMaxDabMove * 1.0001f);
    test::requireNormalsMatchFullRecompute(obj.mesh);
    test::requireBoundsContainPositions(obj);
  }
}

TEST_CASE("symmetry: strokes with every kind of symmetry keep normals and bounds and undo exactly") {
  Brushes b;
  MaskBrush mask;
  FaceSetBrush faceSet;
  std::vector<const Brush*> all = b.moving();
  all.push_back(&mask);
  all.push_back(&faceSet);
  for (const SymmetrySpec& sym : {spec(true, true, false), spec(false, false, false, Axis::Y, 5),
                                  spec(true, false, true, Axis::Y, 4)}) {
    for (const Brush* brush : all) {
      INFO(brush->name() << " copies " << copyCount(sym));
      Scene scene;
      SceneObject& obj = scene.add("S", makeQuadSphere(32));
      roughen(obj);
      obj.mesh.ensureMask();
      obj.mesh.ensureFaceSets();
      const Mesh before = obj.mesh;
      Sculptor s;
      s.beginStroke(obj, *brush, withSymmetry(sym, 0.7f), brush->name());
      REQUIRE(strokePath(s, obj, {0.05f, 1.0f, -0.3f}, {0.6f, 0.6f, 0.4f}, 0.25f, 30, 0.8f));
      auto undo = s.endStroke();
      REQUIRE(undo);
      test::requireNormalsMatchFullRecompute(obj.mesh);
      test::requireBoundsContainPositions(obj);
      for (float m : obj.mesh.mask) REQUIRE((m >= 0.0f && m <= 1.0f));
      UndoStack stack;
      stack.push(std::move(*undo));
      REQUIRE_FALSE(stack.undo(scene).empty());
      CHECK(test::hashBits(obj.mesh.positions) == test::hashBits(before.positions));
      CHECK(test::hashBits(obj.mesh.normals) == test::hashBits(before.normals));
      CHECK(obj.mesh.mask == before.mask);
      CHECK(obj.mesh.faceSets == before.faceSets);
    }
  }
}

TEST_CASE("symmetry: grab with mirrors and radial copies") {
  for (const SymmetrySpec& sym : {spec(false, true, false), spec(false, false, false, Axis::Y, 4)}) {
    Scene scene;
    SceneObject& obj = scene.add("S", makeQuadSphere(40));
    const Mesh before = obj.mesh;
    const SymmetryGroup g(sym);
    const std::vector<Index> image = imageMap(obj.mesh, g.op(1).L);
    Sculptor s;
    const Vec3 c = test::surfacePoint(obj, {0.6f, 0.5f, 0.5f});
    REQUIRE(s.beginGrab(obj, withSymmetry(sym, 1.0f), c, 0.3f, "Grab"));
    const Vec3 offset{0.05f, 0.1f, -0.03f};
    s.grab(offset);
    REQUIRE(s.endStroke());
    // The vertex nearest the grab moved by the offset; its image moved by the op's image of it.
    Index nearest = 0;
    for (Index v = 1; v < obj.mesh.vertexCount(); ++v)
      if (glm::length(obj.mesh.positions[v] - c) < glm::length(obj.mesh.positions[nearest] - c)) nearest = v;
    const Vec3 moved = obj.mesh.positions[nearest] - before.positions[nearest];
    REQUIRE(image[nearest] != kInvalid);
    const Vec3 movedImage = obj.mesh.positions[image[nearest]] - before.positions[image[nearest]];
    CHECK(glm::length(movedImage - transform(g.op(1).L, moved)) < 1e-5f);
    CHECK(glm::length(moved) > 0.05f);
    test::requireNormalsMatchFullRecompute(obj.mesh);
    test::requireBoundsContainPositions(obj);
  }
}

TEST_CASE("symmetry: grab on the plane keeps the seam on it and does not double the pull") {
  Scene scene;
  SceneObject& obj = scene.add("S", makeQuadSphere(40));
  const Mesh before = obj.mesh;
  const float eps = kPinTolerance * glm::length(obj.bvh.bounds().extent());
  Sculptor s;
  const Vec3 c = test::surfacePoint(obj, {0.0f, 1.0f, 0.0f});
  REQUIRE(s.beginGrab(obj, withSymmetry(spec(true, false, false), 1.0f), c, 0.3f, "Grab"));
  s.grab({0.1f, 0.2f, 0.0f});
  REQUIRE(s.endStroke());
  CHECK(s.lastDab().clusters == 1);
  float most = 0.0f;
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
    const Vec3 move = obj.mesh.positions[v] - before.positions[v];
    most = std::max(most, move.y);
    if (std::abs(before.positions[v].x) <= eps && move != Vec3{0.0f}) {
      CHECK(obj.mesh.positions[v].x == 0.0f);
      CHECK_FALSE(std::signbit(obj.mesh.positions[v].x));
    }
  }
  CHECK(most <= 0.2f + 1e-5f);
  CHECK(most > 0.15f);
}

TEST_CASE("symmetry: grab at a pinched vertex on the axis") {
  Scene scene;
  // The two grids of the bowtie map onto each other by a half turn around z.
  SceneObject& obj = scene.add("Bowtie", test::makeBowtie(4));
  Sculptor s;
  REQUIRE(s.beginGrab(obj, withSymmetry(spec(false, false, false, Axis::Z, 2), 1.0f), {0.0f, 0.0f, 0.0f}, 0.1f,
                      "Grab"));
  s.grab({0.05f, 0.0f, 0.15f});  // The in-plane part leaves the axis and is dropped.
  REQUIRE(s.endStroke());
  CHECK(obj.mesh.positions[0] == Vec3{0.0f, 0.0f, 0.15f});
  test::requireNormalsBitExact(obj.mesh);
  test::requireBoundsContainPositions(obj);
}

TEST_CASE("symmetry: overlapping mask copies stay in range and symmetric") {
  Scene scene;
  SceneObject& obj = scene.add("S", makeQuadSphere(40));
  const std::vector<Index> mirror = imageMap(obj.mesh, SymmetryGroup(spec(true, false, false)).op(1).L);
  MaskBrush mask;
  MaskSmoothBrush smooth;
  for (const Brush* brush : std::vector<const Brush*>{&mask, &smooth, &mask}) {
    Sculptor s;
    s.beginStroke(obj, *brush, withSymmetry(spec(true, false, false), 0.6f), brush->name());
    REQUIRE(strokePath(s, obj, {0.06f, 1.0f, -0.3f}, {0.08f, 1.0f, 0.3f}, 0.3f, 40, 0.6f));
    s.endStroke();
  }
  int masked = 0;
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
    const float m = obj.mesh.mask[v];
    REQUIRE((m >= 0.0f && m <= 1.0f));
    masked += m > 0.0f;
    if (mirror[v] != kInvalid) CHECK(std::abs(m - obj.mesh.mask[mirror[v]]) < 1e-5f);
  }
  CHECK(masked > 50);
}

TEST_CASE("symmetry: face set copies paint one new set, continue their own, first copy wins") {
  Scene scene;
  Mesh mesh = makeQuadSphere(32);
  mesh.ensureFaceSets();
  // A band around z = 0 is set 4; elsewhere x > 0 is set 2 and x < 0 set 3, so mirror copies
  // continue different sets.
  for (Index f = 0; f < mesh.faceCount(); ++f) {
    const Vec3 c = mesh.faceCentroid(f);
    mesh.faceSets[f] = std::abs(c.z) < 0.15f ? 4 : c.x > 0.0f ? 2 : 3;
  }
  SceneObject& obj = scene.add("S", std::move(mesh));
  FaceSetBrush brush;
  {
    Sculptor s;
    s.beginStroke(obj, brush, withSymmetry(spec(true, false, false, Axis::Y, 2), 1.0f), "Face Set");  // x and z mirrors
    REQUIRE(s.dab(test::surfacePoint(obj, {0.5f, 0.7f, 0.5f}), 0.25f, 1.0f));
    const std::int32_t fresh = s.paintFaceSet();
    REQUIRE(s.endStroke());
    CHECK(fresh > 4);
    int perQuadrant[4] = {};
    for (Index f = 0; f < obj.mesh.faceCount(); ++f) {
      if (obj.mesh.faceSets[f] != fresh) continue;
      const Vec3 c = obj.mesh.faceCentroid(f);
      ++perQuadrant[(c.x > 0.0f ? 1 : 0) + (c.z > 0.0f ? 2 : 0)];
    }
    CHECK(perQuadrant[3] > 3);
    for (int q : perQuadrant) CHECK(q == perQuadrant[3]);
  }
  {
    // Extend: each copy continues the set under its own first dab into the band.
    StrokeOptions o = withSymmetry(spec(true, false, false), 1.0f);
    o.extendFaceSet = true;
    Sculptor s;
    s.beginStroke(obj, brush, o, "Face Set");
    REQUIRE(s.dab(test::surfacePoint(obj, {0.6f, 0.5f, 0.25f}), 0.25f, 1.0f));
    REQUIRE(s.endStroke());
    int right = 0, left = 0, wrong = 0;
    for (Index f = 0; f < obj.mesh.faceCount(); ++f) {
      const Vec3 c = obj.mesh.faceCentroid(f);
      if (std::abs(c.z) >= 0.15f) continue;
      const std::int32_t set = obj.mesh.faceSets[f];
      right += c.x > 0.0f && set == 2;
      left += c.x < 0.0f && set == 3;
      wrong += (c.x > 0.0f && set == 3) || (c.x < 0.0f && set == 2);
    }
    CHECK(right > 2);
    CHECK(left == right);
    CHECK(wrong == 0);
  }
  {
    // Overlapping copies with different sets: the first copy's set wins the shared faces.
    StrokeOptions o = withSymmetry(spec(true, false, false), 1.0f);
    o.extendFaceSet = true;
    Sculptor s;
    s.beginStroke(obj, brush, o, "Face Set");
    const Vec3 seam = test::surfacePoint(obj, {0.0f, -0.2f, 1.0f});
    REQUIRE(s.dab(test::surfacePoint(obj, {0.1f, -0.2f, 1.0f}), 0.3f, 1.0f));
    CHECK(s.lastDab().clusters == 2);
    REQUIRE(s.endStroke());
    int first = 0, other = 0;
    for (Index f = 0; f < obj.mesh.faceCount(); ++f) {
      if (glm::length(obj.mesh.faceCentroid(f) - seam) > 0.06f) continue;
      first += obj.mesh.faceSets[f] == 2;
      other += obj.mesh.faceSets[f] == 3;
    }
    CHECK(first > 0);
    CHECK(other == 0);
  }
}

TEST_CASE("symmetry: a symmetric stroke on a layer commits into it") {
  Scene scene;
  SceneObject& obj = scene.add("S", makeQuadSphere(32));
  LayerWorkspace ws;
  std::string error;
  REQUIRE(addLayer(obj, ws, &error));
  const std::vector<Vec3> base = obj.mesh.layers.base;
  DrawBrush draw;
  Sculptor s;
  StrokeOptions o = withSymmetry(spec(true, false, false, Axis::Y, 3), 0.7f);
  o.layerTarget = obj.mesh.layers.active;
  s.beginStroke(obj, draw, o, "Draw");
  REQUIRE(strokePath(s, obj, {0.05f, 1.0f, 0.0f}, {0.5f, 0.6f, 0.2f}, 0.25f, 20));
  REQUIRE(s.endStroke());
  const ValidationResult r = validateLayers(obj.mesh, true);
  INFO(r.message);
  CHECK(r.ok);
  CHECK(test::hashBits(obj.mesh.layers.base) == test::hashBits(base));
  // Still refused on a hidden layer.
  obj.mesh.layers.list[0].visible = false;
  CHECK_FALSE(layerTargetRefusal(obj, &draw, o.layerTarget).empty());
  Sculptor refused;
  refused.beginStroke(obj, draw, o, "Draw");
  CHECK_FALSE(refused.active());
}

TEST_CASE("symmetry: a symmetric stroke on a subdivision level syncs") {
  Scene scene;
  SceneObject& obj = scene.add("S", makeQuadSphere(6));
  obj.bvh.build(obj.mesh, Bvh::Params{256});
  obj.topologyVersion = nextTopologyVersion();
  SyncWorkspace ws;
  std::string error;
  REQUIRE(subdivideObject(obj, ws, &error, 256));
  REQUIRE(subdivideObject(obj, ws, &error, 256));
  REQUIRE(setActiveLevel(obj, 1, ws));
  DrawBrush draw;
  Sculptor s;
  s.beginStroke(obj, draw, withSymmetry(spec(true, false, false), 1.0f), "Draw");
  REQUIRE(strokePath(s, obj, {0.5f, 0.8f, 0.0f}, {0.6f, 0.6f, 0.3f}, 0.4f, 10));
  REQUIRE(s.endStroke());
  REQUIRE(setActiveLevel(obj, 2, ws));
  // The top level picked up the mirrored bumps on both sides.
  const float right = test::radiusAlong(obj, {0.55f, 0.7f, 0.15f});
  const float left = test::radiusAlong(obj, {-0.55f, 0.7f, 0.15f});
  CHECK(right > 1.0f);
  CHECK(left == doctest::Approx(right).epsilon(0.01));
}

TEST_CASE("symmetry: overlapping copies give the same result on one thread and many") {
  auto run = [](bool serial) {
    setSerialForTesting(serial);
    Scene scene;
    SceneObject& obj = scene.add("S", makeQuadSphere(48));
    Brushes b;
    for (const Brush* brush : b.moving()) {
      Sculptor s;
      s.beginStroke(obj, *brush, withSymmetry(spec(true, false, true, Axis::Y, 4), 0.7f), brush->name());
      strokePath(s, obj, {0.1f, 1.0f, 0.05f}, {0.3f, 0.9f, 0.2f}, 0.3f, 15, 0.8f);
      s.endStroke();
    }
    setSerialForTesting(false);
    return std::pair{test::hashBits(obj.mesh.positions), test::hashBits(obj.mesh.normals)};
  };
  CHECK(run(true) == run(false));
}

TEST_CASE("symmetry: copies far apart act exactly like separate dabs") {
  // With the copies beyond each other's reach, a symmetric stroke equals dabbing every copy by
  // hand in the same stroke, bit for bit.
  Brushes b;
  for (const Brush* brush : b.moving()) {
    INFO(brush->name());
    Scene scene;
    SceneObject& a = scene.add("A", makeQuadSphere(40));
    SceneObject& m = scene.add("B", makeQuadSphere(40));
    roughen(a);
    roughen(m);
    Sculptor sa, sm;
    sa.beginStroke(a, *brush, withSymmetry(spec(true, false, false), 0.7f), "A");
    sm.beginStroke(m, *brush, withSymmetry({}, 0.7f), "B");
    for (int i = 0; i < 20; ++i) {
      const float t = static_cast<float>(i) / 19.0f;
      const Vec3 c = test::surfacePoint(a, glm::mix(Vec3{0.85f, 0.5f, -0.2f}, Vec3{0.75f, -0.4f, 0.4f}, t));
      REQUIRE(sa.dab(c, 0.22f, 0.8f));
      const Vec3 c2 = test::surfacePoint(m, glm::mix(Vec3{0.85f, 0.5f, -0.2f}, Vec3{0.75f, -0.4f, 0.4f}, t));
      REQUIRE(sm.dab(c2, 0.22f, 0.8f));
      REQUIRE(sm.dab({-c2.x, c2.y, c2.z}, 0.22f, 0.8f));
    }
    sa.endStroke();
    sm.endStroke();
    CHECK(test::hashBits(a.mesh.positions) == test::hashBits(m.mesh.positions));
    CHECK(test::hashBits(a.mesh.normals) == test::hashBits(m.mesh.normals));
  }
}

TEST_CASE("symmetry: dynamic topology is off above its copy limit") {
  Scene scene;
  SceneObject& obj = scene.add("S", makeQuadSphere(24));
  const Index faces = obj.mesh.faceCount();
  DrawBrush draw;
  Sculptor s;
  StrokeOptions o = withSymmetry(spec(false, false, false, Axis::Y, 32), 0.6f);
  o.dyntopo = true;
  o.dyntopoOptions.timeBudgetMs = 0.0;
  s.beginStroke(obj, draw, o, "Draw");
  CHECK(s.topologyOffForCopies());
  CHECK(s.dyntopo() == nullptr);
  REQUIRE(strokePath(s, obj, {0.5f, 0.8f, 0.0f}, {0.6f, 0.6f, 0.2f}, 0.2f, 5, 1.0f, 0.01f));
  REQUIRE(s.endStroke());
  CHECK(s.lastStrokeTopology().tooManyCopies);
  CHECK(obj.mesh.faceCount() == faces);
  // At the limit it runs.
  o.radialCount = kMaxDyntopoCopies;
  s.beginStroke(obj, draw, o, "Draw");
  CHECK_FALSE(s.topologyOffForCopies());
  REQUIRE(strokePath(s, obj, {0.5f, 0.8f, 0.0f}, {0.6f, 0.6f, 0.2f}, 0.2f, 5, 1.0f, 0.01f));
  REQUIRE(s.endStroke());
  CHECK(obj.mesh.faceCount() > faces);
  test::requireValid(obj.mesh);
}
