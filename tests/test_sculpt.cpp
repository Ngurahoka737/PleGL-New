#include <cmath>
#include <random>

#include "TestUtil.h"
#include "mesh/Primitives.h"
#include "scene/Scene.h"
#include "sculpt/Sculptor.h"
#include "sculpt/StrokeSampler.h"

using namespace plegl;

namespace {

// Surface point on the unit sphere in the given direction (nearest hit from outside).
Vec3 surfacePoint(const SceneObject& obj, Vec3 dir) {
  dir = glm::normalize(dir);
  RayHit hit;
  const bool ok = obj.bvh.raycast(obj.mesh, Ray{dir * 3.0f, -dir}, hit);
  REQUIRE(ok);
  return hit.position;
}

void requireNormalsMatchFullRecompute(const Mesh& m) {
  Mesh fresh = m;
  fresh.computeNormals();
  for (Index v = 0; v < m.vertexCount(); ++v) {
    const float d = glm::length(fresh.normals[v] - m.normals[v]);
    if (d > 1e-4f) {
      INFO("vertex " << v << " normal is stale");
      REQUIRE(d <= 1e-4f);
    }
  }
}

void requireBoundsContainPositions(const SceneObject& obj) {
  for (const BvhLeaf& leaf : obj.bvh.leaves()) {
    for (Index f = leaf.faceBegin; f < leaf.faceEnd; ++f) {
      obj.mesh.forEachFaceVertex(f, [&](Index v) {
        const Vec3& p = obj.mesh.positions[v];
        const bool inside = glm::all(glm::greaterThanEqual(p, leaf.bounds.min)) &&
                            glm::all(glm::lessThanEqual(p, leaf.bounds.max));
        REQUIRE(inside);
      });
    }
  }
}

}  // namespace

TEST_CASE("falloff curves are 1 at the centre and 0 outside") {
  for (Falloff f : {Falloff::Smooth, Falloff::Sharp, Falloff::Linear, Falloff::Constant}) {
    CHECK(falloffWeight(f, 0.0f) == doctest::Approx(1.0f));
    CHECK(falloffWeight(f, 1.0f) == 0.0f);
    CHECK(falloffWeight(f, 1.5f) == 0.0f);
    CHECK(falloffWeight(f, 0.5f) <= 1.0f);
  }
  CHECK(falloffWeight(Falloff::Smooth, 0.5f) == doctest::Approx(0.5f));
  CHECK(falloffWeight(Falloff::Linear, 0.25f) == doctest::Approx(0.75f));
  CHECK(falloffWeight(Falloff::Sharp, 0.5f) == doctest::Approx(0.25f));
}

TEST_CASE("stroke sampler spaces dabs evenly whatever the event rate") {
  StrokeSampler s;
  std::vector<StrokeSample> out;
  s.begin({0, 0, 0.0f}, 10.0f, out);
  // 100 px in uneven steps.
  for (float x : {3.0f, 4.0f, 25.0f, 26.0f, 61.0f, 100.0f}) s.moveTo({x, 0, x / 100.0f}, out);
  REQUIRE(out.size() == 11);  // 0, 10, ..., 100
  for (std::size_t i = 0; i < out.size(); ++i) {
    CHECK(out[i].x == doctest::Approx(10.0f * static_cast<float>(i)));
    CHECK(out[i].pressure == doctest::Approx(out[i].x / 100.0f));
  }
}

TEST_CASE("draw brush raises the surface inside the radius only") {
  Scene scene;
  SceneObject& obj = scene.add("Sphere", makeQuadSphere(48));
  const Mesh before = obj.mesh;
  DrawBrush draw;
  Sculptor sculptor;
  sculptor.beginStroke(obj, draw, {}, "Draw");
  const Vec3 c = surfacePoint(obj, {0.2f, 1.0f, 0.1f});
  const float radius = 0.3f;
  REQUIRE(sculptor.dab(c, radius, 1.0f));
  auto undo = sculptor.endStroke();
  REQUIRE(undo);

  int moved = 0;
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
    const float d = glm::length(before.positions[v] - c);
    const float rise = glm::length(obj.mesh.positions[v]) - glm::length(before.positions[v]);
    if (d >= radius) {
      CHECK(obj.mesh.positions[v] == before.positions[v]);
    } else if (d < radius * 0.8f) {
      CHECK(rise > 0.0f);
      ++moved;
    }
  }
  CHECK(moved > 10);
  test::requireValid(obj.mesh);
  requireNormalsMatchFullRecompute(obj.mesh);
  requireBoundsContainPositions(obj);
  CHECK_FALSE(obj.dirtyLeaves.empty());
}

TEST_CASE("inverted draw pushes the surface in") {
  Scene scene;
  SceneObject& obj = scene.add("Sphere", makeQuadSphere(32));
  DrawBrush draw;
  Sculptor sculptor;
  StrokeOptions opts;
  opts.invert = true;
  sculptor.beginStroke(obj, draw, opts, "Draw");
  const Vec3 c = surfacePoint(obj, {0, 0, 1});
  REQUIRE(sculptor.dab(c, 0.3f, 1.0f));
  sculptor.endStroke();
  RayHit hit;
  REQUIRE(obj.bvh.raycast(obj.mesh, Ray{{0, 0, 3}, {0, 0, -1}}, hit));
  CHECK(hit.position.z < 0.995f);
}

TEST_CASE("smooth brush flattens noise without touching the outside") {
  Scene scene;
  SceneObject& obj = scene.add("Sphere", makeQuadSphere(40));
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> u(-0.01f, 0.01f);
  for (Vec3& p : obj.mesh.positions) p *= 1.0f + u(rng);
  obj.mesh.computeNormals();
  obj.bvh.refit(obj.mesh);
  const Mesh before = obj.mesh;

  // Spread of the vertex radii around their mean: noise, not the slight shrink smoothing causes.
  auto roughness = [&](const Mesh& m, const Vec3& c, float r) {
    std::vector<double> lens;
    for (Index v = 0; v < m.vertexCount(); ++v)
      if (glm::length(before.positions[v] - c) < r * 0.5f) lens.push_back(glm::length(m.positions[v]));
    double mean = 0, var = 0;
    for (double l : lens) mean += l;
    mean /= static_cast<double>(lens.size());
    for (double l : lens) var += (l - mean) * (l - mean);
    return std::sqrt(var / static_cast<double>(lens.size()));
  };

  SmoothBrush smooth;
  Sculptor sculptor;
  sculptor.beginStroke(obj, smooth, {.strength = 1.0f}, "Smooth");
  const Vec3 c = surfacePoint(obj, {1, 0.3f, 0});
  for (int i = 0; i < 10; ++i) REQUIRE(sculptor.dab(c, 0.4f, 1.0f));
  sculptor.endStroke();

  CHECK(roughness(obj.mesh, c, 0.4f) < roughness(before, c, 0.4f) * 0.5);
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v)
    if (glm::length(before.positions[v] - c) >= 0.4f) CHECK(obj.mesh.positions[v] == before.positions[v]);
  requireNormalsMatchFullRecompute(obj.mesh);
  requireBoundsContainPositions(obj);
}

TEST_CASE("smooth keeps an open border on the border") {
  Scene scene;
  SceneObject& obj = scene.add("Plane", makePlane(20));
  SmoothBrush smooth;
  Sculptor sculptor;
  sculptor.beginStroke(obj, smooth, {.strength = 1.0f}, "Smooth");
  for (int i = 0; i < 20; ++i) sculptor.dab({1.0f, 0.0f, 0.0f}, 0.5f, 1.0f);  // Centred on the +X edge.
  sculptor.endStroke();
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
    if (!obj.mesh.isBoundaryVertex(v)) continue;
    const Vec3& p = obj.mesh.positions[v];
    const bool onEdge = std::abs(std::abs(p.x) - 1.0f) < 1e-4f || std::abs(std::abs(p.z) - 1.0f) < 1e-4f;
    CHECK(onEdge);
  }
}

TEST_CASE("undo and redo restore a stroke exactly") {
  Scene scene;
  SceneObject& obj = scene.add("Sphere", makeQuadSphere(40));
  const Mesh original = obj.mesh;
  DrawBrush draw;
  Sculptor sculptor;
  UndoStack stack;

  sculptor.beginStroke(obj, draw, {.strength = 0.8f}, "Draw");
  for (int i = 0; i < 12; ++i) sculptor.dab(surfacePoint(obj, {0.1f * i, 1.0f, 0.2f}), 0.25f, 1.0f);
  stack.push(*sculptor.endStroke());
  const Mesh sculpted = obj.mesh;
  REQUIRE(sculpted.positions != original.positions);

  CHECK(stack.undo(scene) == "Draw");
  CHECK(obj.mesh.positions == original.positions);
  CHECK(obj.mesh.normals == original.normals);
  requireBoundsContainPositions(obj);

  CHECK(stack.redo(scene) == "Draw");
  CHECK(obj.mesh.positions == sculpted.positions);
  CHECK(obj.mesh.normals == sculpted.normals);
  requireBoundsContainPositions(obj);

  CHECK(stack.undo(scene) == "Draw");
  CHECK(stack.undo(scene).empty());  // Nothing left.
}

TEST_CASE("undo skips entries whose object was rebuilt") {
  Scene scene;
  SceneObject& obj = scene.add("Sphere", makeQuadSphere(16));
  DrawBrush draw;
  Sculptor sculptor;
  UndoStack stack;
  sculptor.beginStroke(obj, draw, {}, "Draw");
  sculptor.dab(surfacePoint(obj, {0, 1, 0}), 0.3f, 1.0f);
  stack.push(*sculptor.endStroke());
  obj.rebuildSpatial();  // New vertex order: the entry no longer applies.
  CHECK(stack.undo(scene).empty());
}

TEST_CASE("x symmetry mirrors the stroke") {
  Scene scene;
  SceneObject& obj = scene.add("Sphere", makeQuadSphere(40));
  DrawBrush draw;
  Sculptor sculptor;
  sculptor.beginStroke(obj, draw, {.strength = 1.0f, .symmetryX = true}, "Draw");
  REQUIRE(sculptor.dab(surfacePoint(obj, {0.7f, 0.5f, 0.5f}), 0.3f, 1.0f));
  sculptor.endStroke();
  RayHit right, left;
  const Vec3 d = glm::normalize(Vec3{0.7f, 0.5f, 0.5f});
  const Vec3 dm{-d.x, d.y, d.z};
  REQUIRE(obj.bvh.raycast(obj.mesh, Ray{d * 3.0f, -d}, right));
  REQUIRE(obj.bvh.raycast(obj.mesh, Ray{dm * 3.0f, -dm}, left));
  CHECK(glm::length(right.position) > 1.003f);
  CHECK(glm::length(left.position) == doctest::Approx(glm::length(right.position)).epsilon(0.002));
}

TEST_CASE("undo stack drops the oldest entries over budget") {
  UndoStack stack(1000);
  for (int i = 0; i < 5; ++i) {
    SculptUndo e;
    e.label = std::to_string(i);
    e.before.push_back({0, std::vector<Vec3>(20), {}});  // 240 bytes
    stack.push(std::move(e));
  }
  CHECK(stack.bytes() <= 1000);
  CHECK(stack.size() == 4);
}
