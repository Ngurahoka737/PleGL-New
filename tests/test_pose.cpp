#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>
#include <tuple>

#include "TestUtil.h"
#include "core/Parallel.h"
#include "mesh/Primitives.h"
#include "multires/MultiresOps.h"
#include "multires/Propagate.h"
#include "scene/Scene.h"
#include "sculpt/LayerOps.h"
#include "sculpt/PoseOps.h"
#include "sculpt/Sculptor.h"
#include "sculpt/Undo.h"

using namespace plegl;

namespace {

constexpr float kPi = std::numbers::pi_v<float>;

Vec3 hitFrom(const SceneObject& obj, Vec3 from, Vec3 dir) {
  RayHit hit;
  const bool ok = obj.bvh.raycast(obj.mesh, Ray{from, glm::normalize(dir)}, hit);
  REQUIRE(ok);
  return hit.position;
}

// Both meshes in one, the second moved by `offset`.
Mesh combine(const Mesh& a, const Mesh& b, Vec3 offset) {
  std::vector<Vec3> p = a.positions;
  for (const Vec3& q : b.positions) p.push_back(q + offset);
  std::vector<Index> idx, sizes;
  for (const Mesh* m : {&a, &b}) {
    const Index base = m == &a ? 0 : a.vertexCount();
    for (Index f = 0; f < m->faceCount(); ++f) {
      sizes.push_back(m->faceSize(f));
      m->forEachFaceVertex(f, [&](Index v) { idx.push_back(v + base); });
    }
  }
  Mesh out = buildMesh(std::move(p), idx, sizes);
  out.computeNormals();
  return out;
}

Mesh shifted(Mesh m, Vec3 offset) {
  for (Vec3& p : m.positions) p += offset;
  return m;
}

// x turned by the rotation vector r about `pivot`, with glm's quaternions.
Vec3 turned(const Vec3& x, const Vec3& pivot, const Vec3& r) {
  const float angle = glm::length(r);
  if (angle == 0.0f) return x;
  return pivot + glm::angleAxis(angle, r / angle) * (x - pivot);
}

// A target that turns `click` about `pivot` by `angle` about `axis`.
Vec3 targetFor(const Vec3& click, const Vec3& pivot, const Vec3& axis, float angle) {
  return turned(click, pivot, glm::normalize(axis) * angle);
}

StrokeOptions withSymmetry(SymmetrySpec s = {}) {
  StrokeOptions o;
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

void requireMirrored(const Mesh& m, const std::vector<Index>& image, const Mat3& L, float tolerance = 2e-5f) {
  for (Index v = 0; v < m.vertexCount(); ++v) {
    const Index u = image[v];
    if (u == kInvalid) continue;
    const float d = glm::length(m.positions[u] - transform(L, m.positions[v]));
    if (d > tolerance) {
      INFO("vertex " << v << " and its image " << u << " differ by " << d);
      REQUIRE(d <= tolerance);
    }
  }
}

// The weight of every vertex in the build (0 when it does not move).
std::vector<float> weights(const Mesh& m, const PoseBuild& build) {
  std::vector<float> w(static_cast<std::size_t>(m.vertexCount()), 0.0f);
  for (const PoseRig& rig : build.rigs)
    for (std::size_t i = 0; i < rig.verts.size(); ++i) w[rig.verts[i]] = rig.weight[i];
  return w;
}

const Mat3 kMirrorX{Vec3{-1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}};

// A flat mesh of the grid cells (x, y), x < w and y < h, for which inside(x, y) holds, each a
// quad `cell` wide.
template <class Fn>
Mesh gridShape(int w, int h, float cell, Fn&& inside) {
  std::vector<Index> id(static_cast<std::size_t>((w + 1) * (h + 1)), kInvalid);
  std::vector<Vec3> p;
  std::vector<Index> idx, sizes;
  const auto vert = [&](int x, int y) {
    Index& i = id[static_cast<std::size_t>(y * (w + 1) + x)];
    if (i == kInvalid) {
      i = static_cast<Index>(p.size());
      p.push_back({static_cast<float>(x) * cell, static_cast<float>(y) * cell, 0.0f});
    }
    return i;
  };
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      if (!inside(x, y)) continue;
      idx.insert(idx.end(), {vert(x, y), vert(x + 1, y), vert(x + 1, y + 1), vert(x, y + 1)});
      sizes.push_back(4);
    }
  Mesh m = buildMesh(std::move(p), idx, sizes);
  m.computeNormals();
  return m;
}

// The tube of test::makeTube laid along X and centred on the origin, so X mirrors it onto itself.
Mesh tubeAlongX(int segments, int rings, float radius, float length) {
  Mesh m = test::makeTube(segments, rings, radius, length);
  for (Vec3& p : m.positions) p = {p.y - 0.5f * length, -p.x, p.z};
  m.computeNormals();
  return m;
}

// The largest change in displacement along any edge, over the largest displacement.
float tearRatio(const Mesh& before, const Mesh& after) {
  float most = 0.0f, step = 0.0f;
  for (Index v = 0; v < before.vertexCount(); ++v)
    most = std::max(most, glm::length(after.positions[v] - before.positions[v]));
  for (Index f = 0; f < before.faceCount(); ++f) {
    std::vector<Index> corners;
    before.forEachFaceVertex(f, [&](Index v) { corners.push_back(v); });
    for (std::size_t i = 0; i < corners.size(); ++i) {
      const Index a = corners[i], b = corners[(i + 1) % corners.size()];
      const Vec3 da = after.positions[a] - before.positions[a];
      const Vec3 db = after.positions[b] - before.positions[b];
      step = std::max(step, glm::length(da - db));
    }
  }
  return most > 0.0f ? step / most : 0.0f;
}

}  // namespace

TEST_CASE("pose: topology mode bends a tube end about the loop one radius away") {
  Scene scene;
  SceneObject& obj = scene.add("T", test::makeTube(16, 40, 0.1f, 2.0f));
  const Mesh before = obj.mesh;
  const float L = 0.5f;
  const Vec3 click = hitFrom(obj, {1.0f, 1.8f, 0.0f}, {-1, 0, 0});
  Sculptor s;
  REQUIRE(s.beginPose(obj, withSymmetry(), {}, click, L, "Pose"));
  const PoseBuild& build = *s.poseBuild();
  REQUIRE(build.rigs.size() == 1);
  const PoseRig& rig = build.rigs[0];
  CHECK_FALSE(rig.loose);
  // The loop runs around the tube where the distance along the surface reaches L: straight down on
  // the pressed side, partly around it on the far side.
  REQUIRE(rig.rootLoop.size() >= 16);
  for (const Vec3& q : rig.rootLoop) {
    CHECK(q.y >= 1.8f - L - 0.03f);
    CHECK(q.y <= 1.8f - (L - kPi * 0.1f) + 0.03f);  // At worst, around then down.
    CHECK(std::abs(glm::length(Vec2{q.x, q.z}) - 0.1f) < 0.005f);
  }
  CHECK(std::abs(rig.pivot.x) < 0.02f * L);
  CHECK(std::abs(rig.pivot.z) < 0.02f * L);
  // A quarter turn about Z: the far end turns rigidly with it.
  const Vec3 target = targetFor(click, rig.pivot, {0, 0, 1}, 0.5f * kPi);
  const Vec3 r = poseRotation(click, rig.pivot, target, {0, 0, 1});
  CHECK(glm::length(r) == doctest::Approx(0.5f * kPi).epsilon(1e-4));
  const Vec3 pivot = rig.pivot;
  const std::vector<float> w = weights(obj.mesh, build);
  s.pose(target, {0, 0, 1});
  CHECK(s.poseAngle() == doctest::Approx(90.0f).epsilon(1e-3));
  Index tip = kInvalid;
  for (Index v = 0; v < before.vertexCount(); ++v)
    if (glm::length(before.positions[v] - Vec3{0, 2, 0}) < 1e-6f) tip = v;
  REQUIRE(tip != kInvalid);
  CHECK(w[tip] == 1.0f);
  CHECK(glm::length(obj.mesh.positions[tip] - turned(before.positions[tip], pivot, r)) < 1e-5f);
  // Faces that turn whole keep their edge lengths.
  int rigid = 0;
  for (Index f = 0; f < before.faceCount(); ++f) {
    bool whole = true;
    before.forEachFaceVertex(f, [&](Index v) { whole &= w[v] == 1.0f; });
    if (!whole) continue;
    ++rigid;
    const Index h0 = before.faceHe[f];
    Index h = h0;
    do {
      const Index a = before.heVert[h], b = before.heTarget(h);
      const float l0 = glm::length(before.positions[a] - before.positions[b]);
      const float l1 = glm::length(obj.mesh.positions[a] - obj.mesh.positions[b]);
      REQUIRE(std::abs(l1 - l0) <= 1e-5f * l0 + 1e-7f);
      h = before.heNext[h];
    } while (h != h0);
  }
  CHECK(rigid > 50);
  // The root stays put.
  for (Index v = 0; v < before.vertexCount(); ++v)
    if (before.positions[v].y < 1.0f) REQUIRE(test::hashBits({obj.mesh.positions[v]}) == test::hashBits({before.positions[v]}));
  REQUIRE(s.endStroke());
  test::requireValid(obj.mesh);
  test::requireBoundsContainPositions(obj);
}

TEST_CASE("pose: the bend keeps the tube's cross-section") {
  Scene scene;
  SceneObject& obj = scene.add("T", test::makeTube(16, 40, 0.1f, 2.0f));
  const Mesh before = obj.mesh;
  const Vec3 click = hitFrom(obj, {1.0f, 1.8f, 0.0f}, {-1, 0, 0});
  Sculptor s;
  PoseOptions o;
  o.softness = 0.6f;
  REQUIRE(s.beginPose(obj, withSymmetry(), o, click, 0.5f, "Pose"));
  const Vec3 pivot = s.poseBuild()->rigs[0].pivot;
  s.pose(targetFor(click, pivot, {0, 0, 1}, 0.5f * kPi), {0, 0, 1});
  // Every ring of the tube, wherever it is in the band, stays as wide.
  std::map<long, std::vector<Index>> rings;
  for (Index v = 0; v < before.vertexCount(); ++v) {
    const Vec3 p = before.positions[v];
    if (std::abs(glm::length(Vec2{p.x, p.z}) - 0.1f) < 1e-4f) rings[std::lround(p.y * 20.0f)].push_back(v);
  }
  float narrowest = 1.0f;
  for (const auto& [ring, verts] : rings) {
    Vec3 c{0.0f};
    for (Index v : verts) c += obj.mesh.positions[v];
    c /= static_cast<float>(verts.size());
    float r = 0.0f;
    for (Index v : verts) r += glm::length(obj.mesh.positions[v] - c);
    narrowest = std::min(narrowest, r / static_cast<float>(verts.size()));
  }
  CHECK(narrowest >= 0.9f * 0.1f);
  s.endStroke();
}

TEST_CASE("pose: parts beyond the other loop turn rigidly and the longer side stays") {
  Scene scene;
  SceneObject& obj = scene.add("T", test::makeTube(16, 40, 0.1f, 2.0f));
  const Mesh before = obj.mesh;
  // A press at y = 1.2 with a reach past half the circumference makes a band with two loops; the
  // top part (0.6 long) turns, the bottom part (1.0 long) is the root.
  const Vec3 click = hitFrom(obj, {1.0f, 1.2f, 0.0f}, {-1, 0, 0});
  Sculptor s;
  REQUIRE(s.beginPose(obj, withSymmetry(), {}, click, 0.4f, "Pose"));
  const PoseRig& rig = s.poseBuild()->rigs[0];
  CHECK(rig.pivot.y < 1.0f);
  const Vec3 pivot = rig.pivot;
  const Vec3 target = targetFor(click, pivot, {0, 0, 1}, 0.3f);
  const Vec3 r = poseRotation(click, pivot, target, {0, 0, 1});
  s.pose(target, {0, 0, 1});
  for (Index v = 0; v < before.vertexCount(); ++v) {
    const Vec3 p = before.positions[v];
    if (p.y > 1.75f) {
      REQUIRE(glm::length(obj.mesh.positions[v] - turned(p, pivot, r)) < 1e-5f);
    } else if (p.y < 0.6f) {
      REQUIRE(test::hashBits({obj.mesh.positions[v]}) == test::hashBits({p}));
    }
  }
  REQUIRE(s.endStroke());
}

TEST_CASE("pose: a loose part turns about its centroid and leaves the others alone") {
  Scene scene;
  SceneObject& obj = scene.add("S", combine(makeQuadSphere(12), makeQuadSphere(12), {3.0f, 0.0f, 0.0f}));
  const Mesh before = obj.mesh;
  const Vec3 click = hitFrom(obj, {0.0f, 3.0f, 0.0f}, {0, -1, 0});
  Sculptor s;
  REQUIRE(s.beginPose(obj, withSymmetry(), {}, click, 4.0f, "Pose"));
  const PoseRig& rig = s.poseBuild()->rigs[0];
  CHECK(rig.loose);
  CHECK(glm::length(rig.pivot) < 1e-4f);
  s.pose(targetFor(click, rig.pivot, {0, 0, 1}, 1.0f), {0, 0, 1});
  glm::dvec3 c0{0.0}, c1{0.0};
  int n = 0;
  for (Index v = 0; v < before.vertexCount(); ++v) {
    if (before.positions[v].x > 1.5f) {
      REQUIRE(test::hashBits({obj.mesh.positions[v]}) == test::hashBits({before.positions[v]}));
      continue;
    }
    c0 += glm::dvec3(before.positions[v]);
    c1 += glm::dvec3(obj.mesh.positions[v]);
    ++n;
    REQUIRE(std::abs(glm::length(obj.mesh.positions[v]) - glm::length(before.positions[v])) < 1e-5f);
  }
  CHECK(glm::length(c1 - c0) / n < 1e-5);
  CHECK(glm::length(obj.mesh.positions[0] - before.positions[0]) > 0.1f);
  REQUIRE(s.endStroke());
}

TEST_CASE("pose: the root is the loop of the largest part, deterministically, within the visit cap") {
  Scene scene;
  Mesh mesh = makeQuadSphere(32);
  mesh.ensureMask();
  // Three masked caps: the one around +X reaches deepest.
  for (Index v = 0; v < mesh.vertexCount(); ++v) {
    const Vec3 p = mesh.positions[v];
    mesh.mask[v] = p.y > 0.9f || p.y < -0.75f || p.x > 0.7f ? 1.0f : 0.0f;
  }
  SceneObject& obj = scene.add("S", std::move(mesh));
  const auto fans = findNonManifoldFans(obj.mesh);
  const SymmetryGroup group{SymmetrySpec{}};
  PoseOptions o;
  o.mode = PoseMode::Mask;
  const Vec3 click = test::surfacePoint(obj, {-1.0f, 0.2f, 0.3f});
  const PoseBuild a = buildPose(obj.mesh, obj.bvh, fans, group, {}, click, 0.3f, o);
  REQUIRE(a.failure == PoseFailure::None);
  REQUIRE(a.rigs.size() == 1);
  CHECK(glm::length(a.rigs[0].pivot - Vec3{0.7f, 0.0f, 0.0f}) < 0.03f);
  setSerialForTesting(true);
  const PoseBuild b = buildPose(obj.mesh, obj.bvh, fans, group, {}, click, 0.3f, o);
  setSerialForTesting(false);
  CHECK(test::hashBits({a.rigs[0].pivot}) == test::hashBits({b.rigs[0].pivot}));
  CHECK(a.rigs[0].verts == b.rigs[0].verts);
  CHECK(a.rigs[0].weight == b.rigs[0].weight);
  PoseLimits capped;
  capped.raceVisitCap = 50;
  const PoseBuild c = buildPose(obj.mesh, obj.bvh, fans, group, {}, click, 0.3f, o, capped);
  CHECK(c.failure == PoseFailure::None);
  CHECK(c.visited <= 50);
}

TEST_CASE("pose: mask mode turns the unmasked part about its border and never moves the mask") {
  Scene scene;
  Mesh mesh = makeQuadSphere(24);
  mesh.ensureMask();
  for (Index v = 0; v < mesh.vertexCount(); ++v) mesh.mask[v] = mesh.positions[v].y < 0.0f ? 1.0f : 0.0f;
  SceneObject& obj = scene.add("S", std::move(mesh));
  const Mesh before = obj.mesh;
  PoseOptions o;
  o.mode = PoseMode::Mask;
  const Vec3 click = test::surfacePoint(obj, {0, 1, 0});
  Sculptor s;
  REQUIRE(s.beginPose(obj, withSymmetry(), o, click, 0.3f, "Pose"));
  const PoseRig& rig = s.poseBuild()->rigs[0];
  CHECK(std::abs(rig.pivot.x) < 1e-3f);
  CHECK(std::abs(rig.pivot.z) < 1e-3f);
  CHECK(std::abs(rig.pivot.y) < 0.05f);
  s.pose(targetFor(click, rig.pivot, {1, 0, 0}, 0.25f * kPi), {0, 0, 1});
  int moved = 0;
  for (Index v = 0; v < before.vertexCount(); ++v) {
    const bool same = test::hashBits({obj.mesh.positions[v]}) == test::hashBits({before.positions[v]});
    if (before.mask[v] == 1.0f) REQUIRE(same);
    moved += !same;
  }
  CHECK(moved > 100);
  REQUIRE(s.endStroke());
}

TEST_CASE("pose: mirrored tube ends turn by the mirrored rotation") {
  Scene scene;
  const Mesh tube = test::makeTube(16, 40, 0.1f, 2.0f);
  SceneObject& obj = scene.add("T", combine(shifted(tube, {1.0f, 0.0f, 0.0f}), tube, {-1.0f, 0.0f, 0.0f}));
  const std::vector<Index> image = imageMap(obj.mesh, kMirrorX);
  const Vec3 click = hitFrom(obj, {2.0f, 1.8f, 0.0f}, {-1, 0, 0});
  Sculptor s;
  REQUIRE(s.beginPose(obj, withSymmetry({true, false, false}), {}, click, 0.5f, "Pose"));
  const PoseBuild& build = *s.poseBuild();
  REQUIRE(build.rigs.size() == 2);
  CHECK_FALSE(build.rigs[0].cluster.merged());
  CHECK(glm::length(build.rigs[1].pivot - transform(kMirrorX, build.rigs[0].pivot)) < 1e-5f);
  // A turn with a part along every axis, so the mirrored rotation differs from the original.
  s.pose(targetFor(click, build.rigs[0].pivot, {0.3f, 0.5f, 1.0f}, 0.8f), {0, 0, 1});
  requireMirrored(obj.mesh, image, kMirrorX);
  REQUIRE(s.endStroke());
  requireMirrored(obj.mesh, image, kMirrorX);
}

TEST_CASE("pose: a part across the mirror plane turns as one about the plane") {
  Scene scene;
  SceneObject& obj = scene.add("S", makeQuadSphere(24));
  const Mesh before = obj.mesh;
  const std::vector<Index> image = imageMap(obj.mesh, kMirrorX);
  const Vec3 click = test::surfacePoint(obj, {0, 1, 0});
  Sculptor s;
  REQUIRE(s.beginPose(obj, withSymmetry({true, false, false}), {}, click, 0.8f, "Pose"));
  const PoseBuild& build = *s.poseBuild();
  REQUIRE(build.rigs.size() == 1);
  CHECK(build.rigs[0].cluster.merged());
  CHECK(build.rigs[0].pivot.x == 0.0f);
  // Only the turn about X (a nod) is symmetric: the rest is dropped.
  const Vec3 r = poseRotation(click, build.rigs[0].pivot, targetFor(click, build.rigs[0].pivot, {1, 0, 1}, 1.0f), {0, 0, 1});
  const Vec3 rg = transform(build.rigs[0].cluster.axial, r);
  CHECK(rg.x == doctest::Approx(r.x));
  CHECK(rg.y == 0.0f);
  CHECK(rg.z == 0.0f);
  s.pose(targetFor(click, build.rigs[0].pivot, {1, 0, 1}, 1.0f), {0, 0, 1});
  int onPlane = 0;
  for (Index v = 0; v < before.vertexCount(); ++v) {
    if (before.positions[v].x != 0.0f) continue;
    ++onPlane;
    REQUIRE(obj.mesh.positions[v].x == 0.0f);
    REQUIRE_FALSE(std::signbit(obj.mesh.positions[v].x));
  }
  CHECK(onPlane > 20);
  requireMirrored(obj.mesh, image, kMirrorX);
  REQUIRE(s.endStroke());
}

TEST_CASE("pose: radial copies merged around the axis only twist") {
  Scene scene;
  SceneObject& obj = scene.add("S", makeQuadSphere(24));
  const Mesh before = obj.mesh;
  const Vec3 click = test::surfacePoint(obj, {0.02f, 1.0f, 0.0f});
  Sculptor s;
  REQUIRE(s.beginPose(obj, withSymmetry({false, false, false, Axis::Y, 4}), {}, click, 0.8f, "Pose"));
  const PoseBuild& build = *s.poseBuild();
  REQUIRE(build.rigs.size() == 1);
  CHECK(build.rigs[0].cluster.merged());
  s.pose(targetFor(click, build.rigs[0].pivot, {0, 1, 0}, 0.5f), {0, 0, 1});
  float most = 0.0f;
  for (Index v = 0; v < before.vertexCount(); ++v) {
    const Vec3 p0 = before.positions[v], p1 = obj.mesh.positions[v];
    REQUIRE(std::abs(p1.y - p0.y) < 1e-5f);
    REQUIRE(std::abs(glm::length(Vec2{p1.x, p1.z}) - glm::length(Vec2{p0.x, p0.z})) < 1e-5f);
    most = std::max(most, glm::length(p1 - p0));
  }
  CHECK(most > 1e-4f);
  REQUIRE(s.endStroke());
}

TEST_CASE("pose: normals follow during the drag and are exact after it, and undo restores everything") {
  Scene scene;
  SceneObject& obj = scene.add("T", test::makeTube(16, 40, 0.1f, 2.0f));
  const Mesh before = obj.mesh;
  const Vec3 click = hitFrom(obj, {1.0f, 1.8f, 0.0f}, {-1, 0, 0});
  Sculptor s;
  REQUIRE(s.beginPose(obj, withSymmetry(), {}, click, 0.5f, "Pose"));
  const Vec3 pivot = s.poseBuild()->rigs[0].pivot;
  for (float angle : {0.3f, 0.9f, 1.4f}) {
    s.pose(targetFor(click, pivot, {0.2f, 0.0f, 1.0f}, angle), {0, 0, 1});
    test::requireNormalsMatchFullRecompute(obj.mesh, 1e-5f);
  }
  auto undo = s.endStroke();
  REQUIRE(undo);
  test::requireNormalsBitExact(obj.mesh);
  test::requireBoundsContainPositions(obj);
  UndoStack stack;
  stack.push(std::move(*undo));
  REQUIRE_FALSE(stack.undo(scene).empty());
  CHECK(test::hashBits(obj.mesh.positions) == test::hashBits(before.positions));
  CHECK(test::hashBits(obj.mesh.normals) == test::hashBits(before.normals));
  test::requireBoundsContainPositions(obj);
}

TEST_CASE("pose: a stroke on a sculpt layer commits into it and undoes") {
  Scene scene;
  SceneObject& obj = scene.add("T", test::makeTube(16, 40, 0.1f, 2.0f));
  LayerWorkspace ws;
  std::string error;
  REQUIRE(addLayer(obj, ws, &error));
  const std::vector<Vec3> base = obj.mesh.layers.base;
  const std::vector<Vec3> offsets = obj.mesh.layers.list[0].offset;
  const std::vector<Vec3> positions = obj.mesh.positions;
  StrokeOptions o = withSymmetry();
  o.layerTarget = obj.mesh.layers.active;
  const Vec3 click = hitFrom(obj, {1.0f, 1.8f, 0.0f}, {-1, 0, 0});
  Sculptor s;
  REQUIRE(s.beginPose(obj, o, {}, click, 0.5f, "Pose"));
  s.pose(targetFor(click, s.poseBuild()->rigs[0].pivot, {0, 0, 1}, 0.7f), {0, 0, 1});
  auto undo = s.endStroke();
  REQUIRE(undo);
  const ValidationResult valid = validateLayers(obj.mesh, true);
  INFO(valid.message);
  CHECK(valid.ok);
  CHECK(test::hashBits(obj.mesh.layers.base) == test::hashBits(base));
  CHECK(test::hashBits(obj.mesh.layers.list[0].offset) != test::hashBits(offsets));
  UndoStack stack;
  stack.push(std::move(*undo));
  REQUIRE_FALSE(stack.undo(scene).empty());
  CHECK(test::hashBits(obj.mesh.layers.list[0].offset) == test::hashBits(offsets));
  CHECK(test::hashBits(obj.mesh.positions) == test::hashBits(positions));
}

TEST_CASE("pose: on a low subdivision level the detail above rides along rigidly") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = scene.add("T", test::makeTube(16, 40, 0.1f, 2.0f));
  for (int i = 0; i < 2; ++i) {
    std::string error;
    INFO(error);
    REQUIRE(subdivideObject(obj, ws, &error));
  }
  // Detail on the top level: every vertex near the tip pushed out a little.
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
    const Vec3 p = obj.mesh.positions[v];
    if (p.y > 1.5f) obj.mesh.positions[v] += Vec3{p.x, 0.0f, p.z} * 0.2f * std::sin(p.y * 40.0f);
  }
  obj.mesh.computeNormals();
  obj.bvh.refit(obj.mesh);
  REQUIRE(setActiveLevel(obj, 0, ws));
  const std::vector<Vec3> top = obj.multires->levels[2].mesh.positions;
  const std::vector<Vec3> low = obj.mesh.positions;
  const Vec3 click = hitFrom(obj, {1.0f, 1.8f, 0.0f}, {-1, 0, 0});
  Sculptor s;
  REQUIRE(s.beginPose(obj, withSymmetry(), {}, click, 0.5f, "Pose"));
  const Vec3 pivot = s.poseBuild()->rigs[0].pivot;
  const Vec3 r = Vec3{0.0f, 0.0f, 1.0f} * 0.8f;
  s.pose(targetFor(click, pivot, r, glm::length(r)), {0, 0, 1});
  auto entry = s.endStroke();
  REQUIRE(entry);
  // The entry undoes and redoes on its level.
  UndoStack stack;
  stack.push(std::move(*entry));
  REQUIRE_FALSE(stack.undo(scene).empty());
  CHECK(test::hashBits(obj.mesh.positions) == test::hashBits(low));
  REQUIRE_FALSE(stack.redo(scene).empty());
  // Up on the top level, the tip (all of whose support turned rigidly) turned with its detail.
  REQUIRE(setActiveLevel(obj, 2, ws));
  float worst = 0.0f;
  int checked = 0;
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
    if (top[v].y < 1.6f) continue;
    worst = std::max(worst, glm::length(obj.mesh.positions[v] - turned(top[v], pivot, r)));
    ++checked;
  }
  CHECK(checked > 100);
  CHECK(worst < 1e-4f);
  // The root end did not move.
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v)
    if (top[v].y < 0.8f) REQUIRE(sameBits(obj.mesh.positions[v], top[v]));
}

TEST_CASE("pose: every update starts from the press, so going back gives the same bits") {
  const auto run = [](std::initializer_list<float> angles) {
    Scene scene;
    SceneObject& obj = scene.add("T", test::makeTube(16, 40, 0.1f, 2.0f));
    const Vec3 click = hitFrom(obj, {1.0f, 1.8f, 0.0f}, {-1, 0, 0});
    Sculptor s;
    REQUIRE(s.beginPose(obj, withSymmetry(), {}, click, 0.5f, "Pose"));
    const Vec3 pivot = s.poseBuild()->rigs[0].pivot;
    for (float a : angles) s.pose(targetFor(click, pivot, {0, 0, 1}, a), {0, 0, 1});
    const auto during = std::pair{test::hashBits(obj.mesh.positions), test::hashBits(obj.mesh.normals)};
    s.endStroke();
    return std::tuple{during.first, during.second, test::hashBits(obj.mesh.positions), test::hashBits(obj.mesh.normals)};
  };
  CHECK(run({0.6f}) == run({0.6f, 1.3f, 0.6f}));
}

TEST_CASE("pose: hidden faces, locked borders and auto-masking hold parts back") {
  const auto setup = [](Scene& scene, bool hide) -> SceneObject& {
    Mesh mesh = makeQuadSphere(24);
    mesh.ensureFaceSets();
    for (Index f = 0; f < mesh.faceCount(); ++f) mesh.faceSets[f] = mesh.faceCentroid(f).y > 0.5f ? 2 : (hide ? -3 : 3);
    return scene.add("S", std::move(mesh));
  };
  // Vertices with every face in set 3 must not move.
  const auto check = [](const SceneObject& obj, const Mesh& before, bool border) {
    int still = 0, moved = 0;
    for (Index v = 0; v < before.vertexCount(); ++v) {
      bool any2 = false, any3 = false;
      before.forEachOutgoing(v, [&](Index h) {
        const std::int32_t id = faceSetId(before.faceSets[before.heFace[h]]);
        any2 |= id == 2;
        any3 |= id == 3;
      });
      const bool same = test::hashBits({obj.mesh.positions[v]}) == test::hashBits({before.positions[v]});
      if (!any2 || (border && any3)) {
        REQUIRE(same);
        ++still;
      } else {
        moved += !same;
      }
    }
    CHECK(still > 100);
    CHECK(moved > 50);
  };
  for (int c = 0; c < 3; ++c) {
    INFO("case " << c);
    Scene scene;
    SceneObject& obj = setup(scene, c == 1);
    const Mesh before = obj.mesh;
    StrokeOptions o = withSymmetry();
    o.faceSetAutoMask = c == 0;
    o.lockFaceSetBoundaries = c == 2;
    const Vec3 click = test::surfacePoint(obj, {0, 1, 0});
    Sculptor s;
    REQUIRE(s.beginPose(obj, o, {}, click, 2.0f, "Pose"));
    s.pose(targetFor(click, s.poseBuild()->rigs[0].pivot, {1, 0, 0}, 0.6f), {0, 0, 1});
    REQUIRE(s.endStroke());
    check(obj, before, c == 2);
  }
}

TEST_CASE("pose: failures and a reused build") {
  {
    Scene scene;
    Mesh mesh = makeQuadSphere(16);
    mesh.ensureMask();
    std::fill(mesh.mask.begin(), mesh.mask.end(), 1.0f);
    SceneObject& obj = scene.add("S", std::move(mesh));
    Sculptor s;
    CHECK_FALSE(s.beginPose(obj, withSymmetry(), {}, test::surfacePoint(obj, {0, 1, 0}), 0.5f, "Pose"));
    CHECK(s.lastPoseFailure() == PoseFailure::Masked);
    CHECK_FALSE(s.active());
  }
  {
    Scene scene;
    SceneObject& obj = scene.add("P", makePlane(20, 2.0f));
    Sculptor s;
    CHECK_FALSE(s.beginPose(obj, withSymmetry(), {}, Vec3{0.0f}, 0.3f, "Pose"));
    CHECK(s.lastPoseFailure() == PoseFailure::NoJoint);
  }
  {
    Scene scene;
    SceneObject& obj = scene.add("S", makeQuadSphere(64));
    FanCache fans;
    PoseLimits limits;
    limits.timeCapMs = 1e-6;
    const PoseBuild build = buildPoseFor(obj, withSymmetry(), {}, test::surfacePoint(obj, {0, 1, 0}), 4.0f, limits, fans);
    CHECK(build.failure == PoseFailure::Aborted);
  }
  {
    // A build made before the press (the hover preview) gives the same stroke.
    const auto run = [](bool reuse) {
      Scene scene;
      SceneObject& obj = scene.add("T", test::makeTube(16, 40, 0.1f, 2.0f));
      const Vec3 click = hitFrom(obj, {1.0f, 1.8f, 0.0f}, {-1, 0, 0});
      StrokeOptions o = withSymmetry({true, false, false});
      FanCache fans;
      PoseBuild preview = buildPoseFor(obj, o, {}, click, 0.5f, {}, fans);
      Sculptor s;
      REQUIRE(s.beginPose(obj, o, {}, click, 0.5f, "Pose", reuse ? &preview : nullptr));
      // The mirror copy on the tube's far side merges with the press: only the nod about X stays.
      s.pose(targetFor(click, s.poseBuild()->rigs[0].pivot, {1.0f, 0.0f, 0.3f}, 0.9f), {0, 0, 1});
      REQUIRE(s.endStroke());
      return std::pair{test::hashBits(obj.mesh.positions), test::hashBits(obj.mesh.normals)};
    };
    CHECK(run(true) == run(false));
  }
}

TEST_CASE("pose: the root is the largest part even when a smaller one reaches deeper") {
  // A wide block (441 vertices, about 30 rings deep from the neck) and a long thin strip (213
  // vertices, 70 rings) on either side of the press: the block stays and the strip turns.
  Scene scene;
  const float cell = 0.02f;
  SceneObject& obj = scene.add("G", gridShape(100, 20, cell, [](int x, int y) {
    return (x < 20) || (x < 30 && y >= 8 && y < 12) || (y >= 9 && y < 11);
  }));
  const auto fans = findNonManifoldFans(obj.mesh);
  const SymmetryGroup group{SymmetrySpec{}};
  const Vec3 click{25.0f * cell, 10.0f * cell, 0.0f};
  const PoseBuild build = buildPose(obj.mesh, obj.bvh, fans, group, {}, click, 6.0f * cell, {});
  REQUIRE(build.failure == PoseFailure::None);
  REQUIRE(build.rigs.size() == 1);
  CHECK(build.rigs[0].pivot.x < 21.0f * cell);  // The joint at the block, not the strip.
  const std::vector<float> w = weights(obj.mesh, build);
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
    const float x = obj.mesh.positions[v].x / cell;
    if (x < 15.0f) REQUIRE(w[v] == 0.0f);
    if (x > 35.0f) REQUIRE(w[v] == 1.0f);
  }
}

TEST_CASE("pose: a part the visit cap cuts off stays rather than turning half of it") {
  // Both loops lead to parts larger than the cap: both stay, and only the region bends.
  Scene scene;
  SceneObject& obj = scene.add("T", test::makeTube(16, 40, 0.1f, 2.0f));
  const auto fans = findNonManifoldFans(obj.mesh);
  const SymmetryGroup group{SymmetrySpec{}};
  const Vec3 click = hitFrom(obj, {1.0f, 1.0f, 0.0f}, {-1, 0, 0});
  PoseLimits capped;
  capped.raceVisitCap = 150;  // About half of either part.
  const float radius = 0.45f;
  const PoseBuild build = buildPose(obj.mesh, obj.bvh, fans, group, {}, click, radius, {}, capped);
  REQUIRE(build.failure == PoseFailure::None);
  REQUIRE(build.visited <= 150);
  const std::vector<float> w = weights(obj.mesh, build);
  int moving = 0;
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
    if (std::abs(obj.mesh.positions[v].y - 1.0f) > radius + 0.5f * 0.25f * radius + 0.01f) REQUIRE(w[v] == 0.0f);
    moving += w[v] > 0.0f;
  }
  CHECK(moving > 100);
}

TEST_CASE("pose: a locked face set border on the far loop is a joint, not a tear") {
  Scene scene;
  Mesh mesh = test::makeTube(16, 40, 0.1f, 2.0f);
  mesh.ensureFaceSets();
  for (Index f = 0; f < mesh.faceCount(); ++f) mesh.faceSets[f] = mesh.faceCentroid(f).y > 1.5f ? 3 : 2;
  SceneObject& obj = scene.add("T", std::move(mesh));
  const Mesh before = obj.mesh;
  StrokeOptions o = withSymmetry();
  o.lockFaceSetBoundaries = true;
  PoseOptions pose;
  pose.softness = 1.0f;
  const Vec3 click = hitFrom(obj, {1.0f, 1.15f, 0.0f}, {-1, 0, 0});
  Sculptor s;
  REQUIRE(s.beginPose(obj, o, pose, click, 0.45f, "Pose"));
  const Vec3 pivot = s.poseBuild()->rigs[0].pivot;
  CHECK(pivot.y < 1.0f);  // The bottom loop is the root.
  s.pose(targetFor(click, pivot, {0, 0, 1}, 0.5f), {0, 0, 1});
  REQUIRE(s.endStroke());
  float most = 0.0f;
  for (Index v = 0; v < before.vertexCount(); ++v) {
    const Vec3 p = before.positions[v];
    const float moved = glm::length(obj.mesh.positions[v] - p);
    most = std::max(most, moved);
    if (p.y > 1.49f) REQUIRE(moved == 0.0f);  // The border and the tip beyond it stay.
  }
  CHECK(most > 0.05f);
  CHECK(tearRatio(before, obj.mesh) < 0.35f);
}

TEST_CASE("pose: mirrored rigs close to the plane keep their bands to themselves") {
  // Each rig's root band reaches past the plane; it stops at the other rig's region, the gap
  // goes to the nearer joint, and the plane stays the plane.
  Scene scene;
  SceneObject& obj = scene.add("T", tubeAlongX(16, 40, 0.1f, 2.0f));
  const Mesh before = obj.mesh;
  const std::vector<Index> image = imageMap(before, kMirrorX);
  PoseOptions pose;
  pose.softness = 1.0f;
  const Vec3 click = hitFrom(obj, {0.6f, 1.0f, 0.0f}, {0, -1, 0});
  Sculptor s;
  REQUIRE(s.beginPose(obj, withSymmetry({true, false, false}), pose, click, 0.5f, "Pose"));
  const PoseBuild& build = *s.poseBuild();
  REQUIRE(build.rigs.size() == 2);
  const std::vector<float> w = weights(obj.mesh, build);
  for (Index v = 0; v < before.vertexCount(); ++v) {
    if (image[v] == kInvalid || before.positions[v].x == 0.0f) continue;
    INFO("vertex at x " << before.positions[v].x);
    REQUIRE(std::abs(w[v] - w[image[v]]) < 1e-4f);
  }
  const Vec3 pivot = build.rigs[0].pivot;
  s.pose(targetFor(click, pivot, {0, 0, 1}, 0.4f), {0, 0, 1});
  REQUIRE(s.endStroke());
  requireMirrored(obj.mesh, image, kMirrorX, 1e-4f);
  for (Index v = 0; v < before.vertexCount(); ++v)
    if (before.positions[v].x == 0.0f) REQUIRE(obj.mesh.positions[v].x == 0.0f);
}

TEST_CASE("pose: a loose part much smaller than the brush still turns") {
  Mesh small = makeQuadSphere(12);
  for (Vec3& p : small.positions) p *= 0.04f;
  small.computeNormals();
  Scene scene;
  SceneObject& obj = scene.add("S", combine(small, makeQuadSphere(12), {3.0f, 0.0f, 0.0f}));
  const Mesh before = obj.mesh;
  const Vec3 click = hitFrom(obj, {0.0f, 1.0f, 0.0f}, {0, -1, 0});
  Sculptor s;
  REQUIRE(s.beginPose(obj, withSymmetry(), {}, click, 1.0f, "Pose"));
  CHECK(s.poseBuild()->rigs[0].loose);
  s.pose(targetFor(click, s.poseBuild()->rigs[0].pivot, {0, 0, 1}, 1.0f), {0, 0, 1});
  REQUIRE(s.endStroke());
  CHECK(glm::length(obj.mesh.positions[0] - before.positions[0]) > 0.01f);
}
