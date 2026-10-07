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

TEST_CASE("BVH raycast handles axis-aligned rays through split planes") {
  Mesh m = makeQuadSphere(32);
  Bvh bvh;
  bvh.build(m, {.maxLeafFaces = 64});
  const Vec3 dirs[] = {{0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {0, -1, 0}, {-1, 0, 0}};
  for (const Vec3& d : dirs) {
    RayHit hit;
    REQUIRE(bvh.raycast(m, Ray{d * 3.0f, -d}, hit));
    CHECK(hit.t == doctest::Approx(2.0f).epsilon(1e-3));
  }
}

TEST_CASE("rays through shared edges hit the near side") {
  // A quad sphere has edges exactly on the z = 0 plane. Rays inside that plane used to slip
  // between the two triangles of an edge and report the far side of the sphere.
  Mesh m = makeQuadSphere(48);
  Bvh bvh;
  bvh.build(m);
  for (int i = 0; i < 64; ++i) {
    const float a = static_cast<float>(i) * 0.0981f;
    const Vec3 dir{std::cos(a), std::sin(a), 0.0f};
    RayHit hit;
    REQUIRE(bvh.raycast(m, Ray{dir * 3.0f, -dir}, hit));
    CHECK(glm::dot(hit.position, dir) > 0.99f);
  }
}

TEST_CASE("BVH leaves own contiguous half-edge ranges laid out face by face") {
  for (int shape = 0; shape < 3; ++shape) {
    Mesh m = shape == 0 ? makeQuadSphere(40) : shape == 1 ? makeIcosphere(4) : makePlane(30);
    Bvh bvh;
    bvh.build(m, {256});
    REQUIRE(bvh.leaves().size() >= 4);
    Index h = 0;
    for (const BvhLeaf& leaf : bvh.leaves()) {
      CHECK(leaf.heBegin == h);
      CHECK(leaf.heBegin == m.faceHe[leaf.faceBegin]);
      h = leaf.heEnd;
      for (Index e = leaf.heBegin; e < leaf.heEnd; ++e) CHECK(bvh.leafOfHalfEdge(e) == bvh.leafOfFace(m.heFace[e]));
    }
    CHECK(h == m.halfEdgeCount());
    const ValidationResult r = validateLayout(m, bvh);
    INFO(r.message);
    CHECK(r.ok);
  }
}

TEST_CASE("rebuilding the tree keeps leaf indices and skips empty leaves") {
  Mesh m = makeQuadSphere(32);
  Bvh bvh;
  bvh.build(m, {128});
  // Insert two empty leaves (as dynamic topology leaves behind) and rebuild the tree.
  std::vector<BvhLeaf> leaves(bvh.leaves().begin(), bvh.leaves().end());
  const std::size_t original = leaves.size();
  BvhLeaf empty;
  empty.faceBegin = empty.faceEnd = leaves[3].faceEnd;
  empty.vertBegin = empty.vertEnd = leaves[3].vertEnd;
  empty.heBegin = empty.heEnd = leaves[3].heEnd;
  leaves.insert(leaves.begin() + 4, empty);
  BvhLeaf last = leaves.back();
  last.faceBegin = last.faceEnd;
  last.vertBegin = last.vertEnd = m.vertexCount();
  last.heBegin = last.heEnd;
  leaves.push_back(last);
  bvh.setLeaves(leaves);
  REQUIRE(bvh.leaves().size() == original + 2);
  const ValidationResult r = validateLayout(m, bvh);
  INFO(r.message);
  CHECK(r.ok);
  CHECK(bvh.leafOfFace(leaves[5].faceBegin) == 5);
  CHECK(bvh.leafOfVertex(leaves[5].vertBegin) == 5);

  std::mt19937 rng(3);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  for (int i = 0; i < 200; ++i) {
    Ray ray{Vec3{u(rng), u(rng), u(rng)} * 3.0f, {u(rng), u(rng), u(rng)}};
    RayHit hit;
    const bool any = bvh.raycast(m, ray, hit);
    const float ref = bruteForce(m, ray);
    CHECK(any == std::isfinite(ref));
    if (any) CHECK(hit.t == doctest::Approx(ref).epsilon(1e-4));
  }
}

TEST_CASE("BVH tail leaves are found by every query in dynamic mode") {
  Mesh m = makeQuadSphere(16);
  Bvh bvh;
  bvh.build(m, {64});
  bvh.beginDynamic(m);
  // Append a lone triangle far outside the root box, as a tail face.
  const Index v0 = m.vertexCount();
  for (const Vec3& p : {Vec3{5, 0, 0}, Vec3{5, 1, 0}, Vec3{5, 0, 1}}) {
    m.positions.push_back(p);
    m.normals.push_back({1, 0, 0});
    m.vertHe.push_back(m.halfEdgeCount() + static_cast<Index>(m.positions.size()) - 1 - v0);
  }
  const Index h0 = m.halfEdgeCount(), f = m.faceCount();
  for (Index k = 0; k < 3; ++k) {
    m.heNext.push_back(h0 + (k + 1) % 3);
    m.heTwin.push_back(kInvalid);
    m.heVert.push_back(v0 + k);
    m.heFace.push_back(f);
  }
  m.faceHe.push_back(h0);
  bvh.growTail(m);
  const Index tail = bvh.firstTailLeaf();
  REQUIRE(static_cast<Index>(bvh.leaves().size()) == tail + 1);
  CHECK(bvh.isTail(tail));
  CHECK(bvh.leafOfFace(f) == tail);
  CHECK(bvh.leafOfVertex(v0 + 1) == tail);
  CHECK(bvh.leafOfHalfEdge(h0 + 2) == tail);
  const Index tailLeaf[] = {tail};
  bvh.refitLeaves(m, tailLeaf);  // Has no node; must not touch the tree.

  std::vector<Index> found;
  bvh.querySphere({5, 0.3f, 0.3f}, 0.2f, found);
  CHECK(found == std::vector<Index>{tail});
  RayHit hit;
  REQUIRE(bvh.raycast(m, Ray{{6, 0.2f, 0.2f}, {-1, 0, 0}}, hit));
  CHECK(hit.face == f);
  Bvh::ClosestHit near;
  REQUIRE(bvh.closestPoint(m, {5.5f, 0.2f, 0.2f}, 1.0f, near));
  CHECK(near.face == f);

  // A removed face in a flagged leaf is skipped.
  m.faceHe[f] = kInvalid;
  CHECK_FALSE(bvh.raycast(m, Ray{{6, 0.2f, 0.2f}, {-1, 0, 0}}, hit, 2.0f));
  bvh.endDynamic();
}
