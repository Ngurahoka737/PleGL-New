#include <algorithm>
#include <cmath>
#include <random>

#include "TestUtil.h"
#include "core/Geometry.h"
#include "mesh/MeshEdit.h"
#include "mesh/Primitives.h"
#include "spatial/Bvh.h"

using namespace plegl;

namespace {

// Compacts and checks the invariants every edit must keep on a closed surface.
void requireClosedValid(MeshEditor& ed, Index euler = 2) {
  ed.compact();
  const Mesh& m = ed.mesh();
  test::requireValid(m);
  CHECK(test::eulerCharacteristic(m) == euler);
  for (Index v = 0; v < m.vertexCount(); ++v) {
    CHECK_FALSE(m.isBoundaryVertex(v));
    CHECK(m.valence(v) >= 3);
  }
}

// First interior half-edge of a face with `size` corners.
Index halfEdgeOfFaceSize(const Mesh& m, Index size) {
  for (Index h = 0; h < m.halfEdgeCount(); ++h)
    if (m.faceSize(m.heFace[h]) == size) return h;
  return kInvalid;
}

}  // namespace

TEST_CASE("rotateEdge flips triangle edges and rotates quad edges") {
  SUBCASE("triangles") {
    Mesh m = makeIcosphere(2);
    const Index faces = m.faceCount();
    MeshEditor ed(m);
    int done = 0;
    for (Index h = 0; h < m.halfEdgeCount(); h += 7) done += ed.rotateEdge(h);
    CHECK(done > 0);
    requireClosedValid(ed);
    CHECK(m.faceCount() == faces);
    for (Index f = 0; f < m.faceCount(); ++f) CHECK(m.faceSize(f) == 3);
  }
  SUBCASE("quads") {
    Mesh m = makeQuadSphere(6);
    MeshEditor ed(m);
    int done = 0;
    for (Index h = 0; h < m.halfEdgeCount(); h += 11) done += ed.rotateEdge(h);
    CHECK(done > 0);
    requireClosedValid(ed);
    for (Index f = 0; f < m.faceCount(); ++f) CHECK(m.faceSize(f) == 4);
  }
}

TEST_CASE("rotateEdge refuses when an endpoint would drop below valence 3") {
  Mesh m = makeCube(1);  // Every vertex has valence 3.
  MeshEditor ed(m);
  for (Index h = 0; h < m.halfEdgeCount(); ++h) CHECK_FALSE(ed.rotateEdge(h));
  requireClosedValid(ed);
}

TEST_CASE("splitEdge and splitFace keep the mesh closed") {
  Mesh m = makeQuadSphere(4);
  MeshEditor ed(m);
  const Index h = 5;
  const Vec3 mid = (m.positions[m.heVert[h]] + m.positions[m.heTarget(h)]) * 0.5f;
  const Index v = ed.splitEdge(h);
  CHECK(m.positions[v].x == doctest::Approx(mid.x));
  CHECK(ed.valence(v) == 2);  // Valid half-edge mesh, but the vertex needs the face splits below.
  // Each neighbouring face is now a pentagon; split both through the new vertex.
  for (Index e : {m.vertHe[v], m.heTwin[m.vertHe[v]]}) {
    const Index start = m.heVert[e] == v ? e : m.heNext[e];
    REQUIRE(m.heVert[start] == v);
    const Index opposite = m.heNext[m.heNext[m.heNext[start]]];
    CHECK(ed.splitFace(start, opposite) != kInvalid);
  }
  ed.compact();
  test::requireValid(m);
  CHECK(test::eulerCharacteristic(m) == 2);
  CHECK(m.valence(ed.mesh().vertexCount() - 1) == 4);
}

TEST_CASE("splitFace refuses neighbours and existing edges") {
  Mesh m = makeQuadSphere(3);
  MeshEditor ed(m);
  const Index h = halfEdgeOfFaceSize(m, 4);
  CHECK(ed.splitFace(h, m.heNext[h]) == kInvalid);
  CHECK(ed.splitFace(h, h) == kInvalid);
  const Index g = ed.splitFace(h, m.heNext[m.heNext[h]]);
  CHECK(g != kInvalid);
  CHECK(m.faceSize(g) == 3);
  CHECK(m.faceSize(m.heFace[h]) == 3);
  requireClosedValid(ed);
}

TEST_CASE("collapseEdge on triangles removes two faces and keeps the surface closed") {
  Mesh m = makeIcosphere(3);
  const Index v0 = m.vertexCount(), f0 = m.faceCount();
  MeshEditor ed(m);
  int done = 0;
  for (Index h = 0; h < m.halfEdgeCount() && done < 50; h += 13) {
    if (!ed.halfEdgeAlive(h)) continue;
    const Vec3 mid = (m.positions[m.heVert[h]] + m.positions[m.heTarget(h)]) * 0.5f;
    done += ed.collapseEdge(h, mid);
  }
  CHECK(done == 50);
  requireClosedValid(ed);
  CHECK(m.vertexCount() == v0 - 50);
  CHECK(m.faceCount() == f0 - 100);
}

TEST_CASE("collapseEdge refuses the tetrahedron and other link violations") {
  const std::vector<Vec3> p{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  const std::vector<Index> idx{0, 2, 1, 0, 1, 3, 1, 2, 3, 0, 3, 2};
  const std::vector<Index> sizes{3, 3, 3, 3};
  Mesh m = buildMesh(p, idx, sizes);
  MeshEditor ed(m);
  for (Index h = 0; h < m.halfEdgeCount(); ++h) CHECK_FALSE(ed.collapseEdge(h, Vec3{0.0f}));
  requireClosedValid(ed);
}

TEST_CASE("collapseDiagonal removes quads and leaves an all-quad mesh") {
  Mesh m = makeQuadSphere(8);
  const Index f0 = m.faceCount();
  MeshEditor ed(m);
  int done = 0;
  for (Index h = 0; h < m.halfEdgeCount(); h += 37) {
    if (!ed.halfEdgeAlive(h)) continue;
    done += ed.collapseDiagonal(h);
  }
  CHECK(done > 10);
  requireClosedValid(ed);
  CHECK(m.faceCount() == f0 - done);
  for (Index f = 0; f < m.faceCount(); ++f) CHECK(m.faceSize(f) == 4);
}

TEST_CASE("collapseDiagonal refuses at valence-3 corners") {
  Mesh m = makeCube(1);
  MeshEditor ed(m);
  for (Index h = 0; h < m.halfEdgeCount(); ++h) CHECK_FALSE(ed.collapseDiagonal(h));
  requireClosedValid(ed);
}

TEST_CASE("Bvh::closestPoint matches brute force") {
  Mesh m = makeUvSphere(32, 16, 1.0f);
  Bvh bvh;
  bvh.build(m);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-1.6f, 1.6f);
  for (int i = 0; i < 300; ++i) {
    const Vec3 p{u(rng), u(rng), u(rng)};
    float best = 1e30f;
    for (Index f = 0; f < m.faceCount(); ++f) {
      // Fan triangulation, as the BVH uses.
      const Index h0 = m.faceHe[f];
      const Vec3 a = m.positions[m.heVert[h0]];
      for (Index h = m.heNext[h0]; m.heNext[h] != h0; h = m.heNext[h]) {
        const Vec3 q = closestPointOnTriangle(p, a, m.positions[m.heVert[h]], m.positions[m.heTarget(h)]);
        best = std::min(best, glm::dot(q - p, q - p));
      }
    }
    Bvh::ClosestHit hit;
    REQUIRE(bvh.closestPoint(m, p, 10.0f, hit));
    CHECK(hit.distSq == doctest::Approx(best).epsilon(1e-4));
    Bvh::ClosestHit limited;
    if (std::sqrt(best) > 0.2f) CHECK_FALSE(bvh.closestPoint(m, p, 0.2f, limited));
  }
}
