#include <algorithm>
#include <limits>
#include <random>

#include "TestUtil.h"
#include "mesh/Primitives.h"
#include "spatial/Bvh.h"

using namespace plegl;

namespace {

// Reference: nearest hit by testing every triangle.
float bruteForce(const Mesh& m, const Ray& ray) {
  float best = std::numeric_limits<float>::infinity();
  for (Index f = 0; f < m.faceCount(); ++f) {
    std::vector<Index> vs;
    m.forEachFaceVertex(f, [&](Index v) { vs.push_back(v); });
    for (std::size_t i = 1; i + 1 < vs.size(); ++i) {
      const Vec3 a = m.positions[vs[0]], b = m.positions[vs[i]], c = m.positions[vs[i + 1]];
      const Vec3 e1 = b - a, e2 = c - a, p = glm::cross(ray.dir, e2);
      const float det = glm::dot(e1, p);
      if (std::abs(det) < 1e-20f) continue;
      const Vec3 s = ray.origin - a;
      const float u = glm::dot(s, p) / det;
      const Vec3 q = glm::cross(s, e1);
      const float v = glm::dot(ray.dir, q) / det;
      const float t = glm::dot(e2, q) / det;
      if (u >= 0 && v >= 0 && u + v <= 1 && t >= 0) best = std::min(best, t);
    }
  }
  return best;
}

}  // namespace

TEST_CASE("BVH build reorders mesh into contiguous leaves") {
  Mesh m = makeQuadSphere(40);  // 9602 vertices, 9600 faces
  Bvh bvh;
  bvh.build(m, {.maxLeafFaces = 256});
  test::requireValid(m);
  test::requireOutward(m);

  const auto leaves = bvh.leaves();
  REQUIRE(leaves.size() > 8);
  Index expectFace = 0, expectVert = 0;
  for (std::size_t i = 0; i < leaves.size(); ++i) {
    const BvhLeaf& leaf = leaves[i];
    CHECK(leaf.faceBegin == expectFace);
    CHECK(leaf.vertBegin == expectVert);
    CHECK(leaf.faceEnd - leaf.faceBegin <= 256);
    expectFace = leaf.faceEnd;
    expectVert = leaf.vertEnd;
    for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) CHECK(bvh.leafOfVertex(v) == static_cast<Index>(i));
    for (Index f = leaf.faceBegin; f < leaf.faceEnd; ++f) {
      m.forEachFaceVertex(f, [&](Index v) {
        const Vec3& p = m.positions[v];
        CHECK(p.x >= leaf.bounds.min.x);
        CHECK(p.x <= leaf.bounds.max.x);
      });
    }
  }
  CHECK(expectFace == m.faceCount());
  CHECK(expectVert == m.vertexCount());
}

TEST_CASE("BVH raycast matches brute force") {
  Mesh m = makeIcosphere(4);
  Bvh bvh;
  bvh.build(m, {.maxLeafFaces = 64});
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  int hits = 0;
  for (int i = 0; i < 300; ++i) {
    Ray ray;
    ray.origin = Vec3(u(rng), u(rng), u(rng)) * 3.0f;
    ray.dir = Vec3(u(rng), u(rng), u(rng)) * 0.5f - ray.origin * 0.2f;
    if (glm::length(ray.dir) < 1e-3f) continue;
    const float expected = bruteForce(m, ray);
    RayHit hit;
    const bool got = bvh.raycast(m, ray, hit);
    CHECK(got == std::isfinite(expected));
    if (got) {
      ++hits;
      CHECK(hit.t == doctest::Approx(expected).epsilon(1e-5));
      CHECK(glm::length(hit.smoothNormal) == doctest::Approx(1.0f));
      const BvhLeaf& leaf = bvh.leaves()[hit.leaf];
      CHECK(hit.face >= leaf.faceBegin);
      CHECK(hit.face < leaf.faceEnd);
    }
  }
  CHECK(hits > 50);
}

TEST_CASE("BVH sphere query finds every leaf touching the sphere") {
  Mesh m = makeQuadSphere(30);
  Bvh bvh;
  bvh.build(m, {.maxLeafFaces = 128});
  const Vec3 center{0.0f, 0.0f, 1.0f};
  const float radius = 0.3f;
  std::vector<Index> found;
  bvh.querySphere(center, radius, found);
  CHECK_FALSE(found.empty());
  // Every vertex inside the sphere must belong to a face in a returned leaf.
  for (Index v = 0; v < m.vertexCount(); ++v) {
    if (glm::length(m.positions[v] - center) > radius) continue;
    const Index leaf = bvh.leafOfVertex(v);
    CHECK(std::find(found.begin(), found.end(), leaf) != found.end());
  }
}

TEST_CASE("BVH refit tracks moved vertices") {
  Mesh m = makeQuadSphere(20);
  Bvh bvh;
  bvh.build(m, {.maxLeafFaces = 100});
  const BvhLeaf leaf = bvh.leaves()[0];
  for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) m.positions[v] *= 2.0f;
  const Index leaves[] = {0};
  bvh.refitLeaves(m, leaves);
  for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
    const Vec3& p = m.positions[v];
    CHECK(p.x <= bvh.bounds().max.x);
    CHECK(p.x >= bvh.bounds().min.x);
    CHECK(p.y <= bvh.leaves()[0].bounds.max.y);
  }
  // A ray straight at a moved vertex now hits it at the new distance.
  const Vec3 target = m.positions[leaf.vertBegin];
  Ray ray{target * 3.0f, -target * 3.0f};
  RayHit hit;
  REQUIRE(bvh.raycast(m, ray, hit));
  CHECK(glm::length(hit.position) > 1.5f);
}
