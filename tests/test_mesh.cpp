#include <algorithm>
#include <numeric>
#include <random>

#include "TestUtil.h"
#include "mesh/Mesh.h"

using namespace plegl;

namespace {

// Unit cube made of 6 quads, outward winding.
Mesh cubeMesh() {
  std::vector<Vec3> p = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                         {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};
  std::vector<Index> idx = {0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 2, 3, 7, 6, 1, 2, 6, 5, 0, 4, 7, 3};
  std::vector<Index> sizes(6, 4);
  return buildMesh(p, idx, sizes);
}

}  // namespace

TEST_CASE("buildMesh links a closed quad cube") {
  BuildReport rep;
  std::vector<Vec3> p = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                         {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};
  std::vector<Index> idx = {0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 2, 3, 7, 6, 1, 2, 6, 5, 0, 4, 7, 3};
  std::vector<Index> sizes(6, 4);
  Mesh m = buildMesh(p, idx, sizes, &rep);
  CHECK(rep.ok());
  CHECK(rep.nonManifoldVertices == 0);
  test::requireValid(m);
  test::requireOutward(m);
  CHECK(m.edgeCount() == 12);
  CHECK(test::eulerCharacteristic(m) == 2);
  for (Index v = 0; v < m.vertexCount(); ++v) {
    CHECK(m.valence(v) == 3);
    CHECK_FALSE(m.isBoundaryVertex(v));
    CHECK(glm::dot(m.normals[v], m.positions[v]) > 0.0f);
  }
}

TEST_CASE("open fans: a single quad has boundary vertices of valence 2") {
  std::vector<Vec3> p = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
  std::vector<Index> idx = {0, 1, 2, 3};
  std::vector<Index> sizes = {4};
  Mesh m = buildMesh(p, idx, sizes);
  test::requireValid(m);
  CHECK(m.edgeCount() == 4);
  for (Index v = 0; v < 4; ++v) {
    CHECK(m.isBoundaryVertex(v));
    CHECK(m.valence(v) == 2);
    CHECK(m.normals[v].z == doctest::Approx(1.0f));
  }
}

TEST_CASE("buildMesh reports degenerate and non-manifold input") {
  std::vector<Vec3> p = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  // Face 1 repeats a vertex (degenerate); face 2 duplicates face 0's winding (non-manifold edges).
  std::vector<Index> idx = {0, 1, 2, 0, 1, 1, 0, 1, 3};
  std::vector<Index> sizes = {3, 3, 3};
  BuildReport rep;
  Mesh m = buildMesh(p, idx, sizes, &rep);
  CHECK(rep.degenerateFaces == 1);
  CHECK(rep.nonManifoldEdges == 1);
  CHECK(m.faceCount() == 2);
}

TEST_CASE("reorder keeps topology valid and renumbers vertices by first use") {
  Mesh m = cubeMesh();
  std::vector<Index> order(m.faceCount());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(7);
  std::shuffle(order.begin(), order.end(), rng);

  std::vector<Vec3> oldCentroids;
  for (Index f : order) oldCentroids.push_back(m.faceCentroid(f));

  const std::vector<Index> first = m.reorder(order);
  test::requireValid(m);
  test::requireOutward(m);
  REQUIRE(first.size() == static_cast<std::size_t>(m.faceCount() + 1));
  CHECK(first.front() == 0);
  CHECK(first.back() == 8);
  for (Index f = 0; f < m.faceCount(); ++f) {
    const Vec3 d = m.faceCentroid(f) - oldCentroids[f];
    CHECK(glm::dot(d, d) < 1e-12f);
    // Every vertex of face f was first used at or before f.
    m.forEachFaceVertex(f, [&](Index v) { CHECK(v < first[f + 1]); });
  }
}

TEST_CASE("weldVertices merges coincident points") {
  std::vector<Vec3> p = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 0, 1e-7f}, {0, 1, 0}, {1, 1, 0}};
  std::vector<Index> idx = {0, 1, 2, 3, 5, 4};
  weldVertices(p, idx, 1e-5f);
  CHECK(p.size() == 4);
  CHECK(idx[3] == idx[1]);
  CHECK(idx[5] == idx[2]);
}
