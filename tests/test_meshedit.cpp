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

namespace {

// Records every element an edit reports, so tests can check that nothing changes unreported.
struct RecordingObserver : EditObserver {
  std::vector<Index> verts, faces, halfEdges;
  void beforeWrite(ElementKind kind, Index i) override {
    (kind == ElementKind::Vertex ? verts : kind == ElementKind::Face ? faces : halfEdges).push_back(i);
  }
  bool has(const std::vector<Index>& list, Index i) const { return std::find(list.begin(), list.end(), i) != list.end(); }
};

// Every element that existed before the edit and differs after it must have been reported.
void requireReported(const Mesh& before, const Mesh& after, const RecordingObserver& obs) {
  for (Index v = 0; v < before.vertexCount(); ++v) {
    const bool changed = before.positions[v] != after.positions[v] || before.vertHe[v] != after.vertHe[v] ||
                         (!before.mask.empty() && before.mask[v] != after.mask[v]);
    if (changed && !obs.has(obs.verts, v)) {
      INFO("vertex " << v << " changed unreported");
      REQUIRE(false);
    }
  }
  for (Index f = 0; f < before.faceCount(); ++f) {
    const bool changed = before.faceHe[f] != after.faceHe[f] ||
                         (!before.faceSets.empty() && before.faceSets[f] != after.faceSets[f]);
    if (changed && !obs.has(obs.faces, f)) {
      INFO("face " << f << " changed unreported");
      REQUIRE(false);
    }
  }
  for (Index h = 0; h < before.halfEdgeCount(); ++h) {
    const bool changed = before.heNext[h] != after.heNext[h] || before.heTwin[h] != after.heTwin[h] ||
                         before.heVert[h] != after.heVert[h] || before.heFace[h] != after.heFace[h];
    if (changed && !obs.has(obs.halfEdges, h)) {
      INFO("half-edge " << h << " changed unreported");
      REQUIRE(false);
    }
  }
}

}  // namespace

TEST_CASE("an edit observer hears about every element an edit changes") {
  for (int shape = 0; shape < 3; ++shape) {
    Mesh m = shape == 0 ? makeIcosphere(2) : shape == 1 ? makeQuadSphere(6) : makeUvSphere(12, 8);
    m.mask.assign(m.positions.size(), 0.25f);
    m.faceSets.resize(m.faceHe.size());
    for (Index f = 0; f < m.faceCount(); ++f) m.faceSets[f] = f % 7 == 0 ? -(f % 3 + 1) : f % 3 + 1;
    MeshEditor ed(m);
    RecordingObserver obs;
    ed.setObserver(&obs);
    std::mt19937 rng(1234 + shape);
    int applied[5] = {};
    for (int step = 0; step < 400; ++step) {
      const Index nh = m.halfEdgeCount();
      const Index h = std::uniform_int_distribution<Index>(0, nh - 1)(rng);
      if (!ed.halfEdgeAlive(h)) continue;
      const int op = static_cast<int>(rng() % 5);
      const Mesh before = m;
      obs = {};
      bool ok = false;
      switch (op) {
        case 0: ok = ed.rotateEdge(h); break;
        case 1: ok = ed.splitEdge(h, 0.3f) != kInvalid; break;
        case 2: {
          const Index hb = m.heNext[m.heNext[h]];
          ok = m.heNext[hb] != h && ed.splitFace(h, hb) != kInvalid;
          break;
        }
        case 3: ok = ed.collapseEdge(h, m.positions[m.heVert[h]]); break;
        case 4: ok = ed.collapseDiagonal(h); break;
      }
      if (!ok) {
        // A refused edit changes nothing at all.
        CHECK(m.positions == before.positions);
        CHECK(m.heNext == before.heNext);
        continue;
      }
      ++applied[op];
      requireReported(before, m, obs);
      const ValidationResult r = validateLive(m);
      INFO(r.message);
      REQUIRE(r.ok);
    }
    for (int op = 0; op < 5; ++op) {
      INFO("shape " << shape << " op " << op);
      if (shape != 0 || op != 4) CHECK(applied[op] > 0);  // collapseDiagonal needs quads.
    }
  }
}

TEST_CASE("an edit observer changes nothing about the edit") {
  Mesh a = makeQuadSphere(8), b = a;
  MeshEditor ea(a), eb(b);
  RecordingObserver obs;
  eb.setObserver(&obs);
  std::mt19937 rng(7);
  for (int step = 0; step < 200; ++step) {
    const Index h = std::uniform_int_distribution<Index>(0, a.halfEdgeCount() - 1)(rng);
    if (!ea.halfEdgeAlive(h)) continue;
    const bool ra = (step % 2) ? ea.collapseDiagonal(h) : ea.rotateEdge(h);
    const bool rb = (step % 2) ? eb.collapseDiagonal(h) : eb.rotateEdge(h);
    REQUIRE(ra == rb);
  }
  CHECK(a.positions == b.positions);
  CHECK(a.heNext == b.heNext);
  CHECK(a.heTwin == b.heTwin);
  CHECK(a.vertHe == b.vertHe);
}

TEST_CASE("mesh headroom avoids reallocation") {
  Mesh m = makeIcosphere(1);
  m.reserveHeadroom(m.vertexCount() + 10, m.faceCount() + 40, m.halfEdgeCount() + 100);
  CHECK(m.hasHeadroom(10, 40, 100));
  CHECK_FALSE(m.hasHeadroom(11, 40, 100));
  const Vec3* data = m.positions.data();
  MeshEditor ed(m);
  for (int i = 0; i < 10; ++i) ed.splitEdge(static_cast<Index>(i * 3));
  CHECK(m.positions.data() == data);
}
