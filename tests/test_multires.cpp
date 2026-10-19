#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

#include "TestUtil.h"
#include "mesh/Primitives.h"
#include "multires/Subdivide.h"
#include "scene/Scene.h"

using namespace plegl;

namespace {

bool sameBits(const Vec3& a, const Vec3& b) { return std::memcmp(&a, &b, sizeof(Vec3)) == 0; }

Mesh makePentagonPrism() {
  std::vector<Vec3> p;
  for (int ring = 0; ring < 2; ++ring)
    for (int i = 0; i < 5; ++i) {
      const float a = 6.2831853f * static_cast<float>(i) / 5.0f;
      p.push_back({std::cos(a), ring == 0 ? -0.5f : 0.5f, std::sin(a)});
    }
  std::vector<Index> idx, sizes;
  // Bottom and top, then the sides, all wound outward.
  for (int i = 0; i < 5; ++i) idx.push_back(i);
  sizes.push_back(5);
  for (int i = 4; i >= 0; --i) idx.push_back(5 + i);
  sizes.push_back(5);
  for (int i = 0; i < 5; ++i) {
    const int j = (i + 1) % 5;
    idx.insert(idx.end(), {i, 5 + i, 5 + j, j});
    sizes.push_back(4);
  }
  return buildMesh(std::move(p), idx, sizes);
}

// A mesh in canonical order and the result of one level of subdivision, provisional numbering.
SubdivisionResult subdivideFlat(const Mesh& m) {
  SubdivideOptions o;
  o.buildBvh = false;
  auto r = subdivide(m, identityCanonicalMap(m), classifyVertices(m), o);
  REQUIRE(r);
  return std::move(*r);
}

Index quadCount(const Mesh& m) {
  Index n = 0;
  for (Index f = 0; f < m.faceCount(); ++f) n += m.faceSize(f) == 4 ? 1 : 0;
  return n;
}

// Positions of a level in canonical vertex order.
std::vector<Vec3> canonicalPositions(const Mesh& m, const CanonicalMap& c) {
  std::vector<Vec3> out(m.positions.size());
  for (Index v = 0; v < m.vertexCount(); ++v) out[c.vert[v]] = m.positions[v];
  return out;
}

}  // namespace

TEST_CASE("catmull-clark counts, validity and all quads") {
  const std::vector<std::pair<const char*, Mesh>> meshes = {
      {"cube 1", makeCube(1)},           {"cube 4", makeCube(4)},       {"icosphere 1", makeIcosphere(1)},
      {"uv sphere", makeUvSphere(8, 6)}, {"plane 4", makePlane(4)},     {"pentagon prism", makePentagonPrism()},
  };
  for (const auto& [name, base] : meshes) {
    INFO(name);
    Mesh m = base;
    Bvh bvh;
    bvh.build(m, Bvh::Params{64});
    test::requireValid(m);
    auto r = subdivide(m, identityCanonicalMap(m), classifyVertices(m), SubdivideOptions{kMaxMultiresFaces, 64});
    REQUIRE(r);
    const Mesh& c = r->mesh;
    CHECK(c.vertexCount() == m.vertexCount() + m.edgeCount() + m.faceCount());
    CHECK(c.faceCount() == m.halfEdgeCount());
    CHECK(c.halfEdgeCount() == 4 * m.halfEdgeCount());
    CHECK(test::eulerCharacteristic(c) == test::eulerCharacteristic(m));
    CHECK(quadCount(c) == c.faceCount());
    test::requireValid(c);
    const ValidationResult layout = validateLayout(c, r->bvh);
    INFO(layout.message);
    CHECK(layout.ok);
    CHECK(c.normals.size() == c.positions.size());
    // A second level from the first.
    auto r2 = subdivide(c, r->canon, r->rule, SubdivideOptions{kMaxMultiresFaces, 64});
    REQUIRE(r2);
    test::requireValid(r2->mesh);
    CHECK(r2->mesh.faceCount() == 4 * c.faceCount());
    CHECK(r2->rule == classifyVertices(r2->mesh));
  }
}

TEST_CASE("closed-form catmull-clark topology equals buildMesh") {
  for (Mesh m : {makeCube(3), makeIcosphere(1), makeUvSphere(7, 5), makePlane(3), makePentagonPrism()}) {
    const SubdivisionResult r = subdivideFlat(m);
    std::vector<Index> idx, sizes;
    for (Index f = 0; f < r.mesh.faceCount(); ++f) {
      r.mesh.forEachFaceVertex(f, [&](Index v) { idx.push_back(v); });
      sizes.push_back(4);
    }
    BuildReport report;
    const Mesh built = buildMesh(r.mesh.positions, idx, sizes, &report);
    CHECK(report.ok());
    REQUIRE(built.heTwin.size() == r.mesh.heTwin.size());
    CHECK(built.heTwin == r.mesh.heTwin);
    CHECK(built.heNext == r.mesh.heNext);
    CHECK(built.heVert == r.mesh.heVert);
  }
}

TEST_CASE("catmull-clark hand values") {
  SUBCASE("cube") {
    const Mesh m = makeCube(1, 2.0f);  // Corners at +-1.
    const SubdivisionResult r = subdivideFlat(m);
    const Mesh& c = r.mesh;
    for (Index v = 0; v < m.vertexCount(); ++v) {
      const Vec3 expected = m.positions[v] * (5.0f / 9.0f);
      CHECK(glm::length(c.positions[v] - expected) < 1e-6f);
    }
    // Edge and face points.
    int edgePoints = 0, facePoints = 0;
    for (Index v = m.vertexCount(); v < c.vertexCount(); ++v) {
      const Vec3 a = glm::abs(c.positions[v]);
      const float big = std::max({a.x, a.y, a.z});
      const int zeros = (a.x < 1e-6f) + (a.y < 1e-6f) + (a.z < 1e-6f);
      if (zeros == 1) {
        ++edgePoints;
        CHECK(std::abs(big - 0.75f) < 1e-6f);
        CHECK(std::abs(a.x + a.y + a.z - 1.5f) < 1e-6f);
      } else if (zeros == 2) {
        ++facePoints;
        CHECK(std::abs(big - 1.0f) < 1e-6f);
      }
    }
    CHECK(edgePoints == 12);
    CHECK(facePoints == 6);
  }
  SUBCASE("regular interior vertex: 9/16, 3/32, 1/64") {
    Mesh m = makePlane(4);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> h(-0.2f, 0.2f);
    for (Vec3& p : m.positions) p.y = h(rng);
    const SubdivisionResult r = subdivideFlat(m);
    int checked = 0;
    for (Index v = 0; v < m.vertexCount(); ++v) {
      if (m.isBoundaryVertex(v)) continue;
      Vec3 edges{0.0f}, diagonals{0.0f};
      m.forEachOutgoing(v, [&](Index e) {
        edges += m.positions[m.heTarget(e)];
        diagonals += m.positions[m.heTarget(m.heNext[e])];  // Opposite corner of the quad.
      });
      const Vec3 expected = m.positions[v] * (9.0f / 16.0f) + edges * (3.0f / 32.0f) + diagonals * (1.0f / 64.0f);
      CHECK(glm::length(r.mesh.positions[v] - expected) < 1e-6f);
      ++checked;
    }
    CHECK(checked == 9);
  }
}

TEST_CASE("open borders follow the cubic B-spline and corners stay put") {
  const Mesh m = makePlane(3);
  const std::vector<VertexRule> rules = classifyVertices(m);
  const SubdivisionResult r = subdivideFlat(m);
  int corners = 0, border = 0;
  for (Index v = 0; v < m.vertexCount(); ++v) {
    const Vec3& p = m.positions[v];
    const bool onX = std::abs(std::abs(p.x) - 1.0f) < 1e-6f, onZ = std::abs(std::abs(p.z) - 1.0f) < 1e-6f;
    if (onX && onZ) {
      CHECK(rules[v] == VertexRule::Pinned);
      CHECK(sameBits(r.mesh.positions[v], p));
      ++corners;
    } else if (onX || onZ) {
      CHECK(rules[v] == VertexRule::Boundary);
      Index a, b;
      borderNeighbours(m, v, a, b);
      CHECK(a != b);
      const Vec3 expected = p * 0.75f + (m.positions[a] + m.positions[b]) * 0.125f;
      CHECK(sameBits(r.mesh.positions[v], expected));
      ++border;
    } else {
      CHECK(rules[v] == VertexRule::Smooth);
    }
  }
  CHECK(corners == 4);
  CHECK(border == 8);
  for (const Vec3& p : r.mesh.positions) CHECK(p.y == 0.0f);
  // Border edge points are plain midpoints.
  for (Index h = 0; h < m.halfEdgeCount(); ++h) {
    if (m.heTwin[h] != kInvalid) continue;
    const Vec3 mid = (m.positions[m.heVert[h]] + m.positions[m.heTarget(h)]) * 0.5f;
    CHECK(sameBits(r.mesh.positions[r.links.edgeChild[h]], mid));
  }
  // On an evenly spaced border every rule that keeps straight lines gives the same point, so
  // check the weights on a bent, unevenly spaced one.
  Mesh bent = m;
  for (Index v = 0; v < bent.vertexCount(); ++v) bent.positions[v] += Vec3{0.05f * std::sin(1.7f * float(v)), 0.1f * std::cos(2.3f * float(v)), 0.0f};
  const SubdivisionResult rb = subdivideFlat(bent);
  int checked = 0;
  for (Index v = 0; v < bent.vertexCount(); ++v) {
    if (rules[v] != VertexRule::Boundary) continue;
    Index a, b;
    borderNeighbours(bent, v, a, b);
    const Vec3 expected = bent.positions[v] * 0.75f + (bent.positions[a] + bent.positions[b]) * 0.125f;
    CHECK(glm::length(rb.mesh.positions[v] - expected) < 1e-6f);
    CHECK(glm::length(rb.mesh.positions[v] - bent.positions[v]) > 1e-4f);
    ++checked;
  }
  CHECK(checked == 8);
}

TEST_CASE("non-manifold and isolated vertices are pinned") {
  // Two quads that touch at vertex 0 only (a bowtie), plus an isolated vertex.
  std::vector<Vec3> p = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {-1, 0, 0}, {-1, -1, 0}, {0, -1, 0}, {5, 5, 5}};
  const std::vector<Index> idx = {0, 1, 2, 3, 0, 4, 5, 6};
  const std::vector<Index> sizes = {4, 4};
  BuildReport report;
  const Mesh m = buildMesh(p, idx, sizes, &report);
  CHECK(report.nonManifoldVertices == 1);
  const std::vector<VertexRule> rules = classifyVertices(m);
  CHECK(rules[0] == VertexRule::Pinned);
  CHECK(rules[7] == VertexRule::Pinned);
  const SubdivisionResult r = subdivideFlat(m);
  CHECK(sameBits(r.mesh.positions[0], m.positions[0]));
  CHECK(sameBits(r.mesh.positions[7], m.positions[7]));
  CHECK(r.mesh.vertHe[7] == kInvalid);
  CHECK(r.mesh.faceCount() == 8);
  CHECK(r.rule[0] == VertexRule::Pinned);
}

TEST_CASE("subdivision gives the same bits for any live order, with the same canonical numbering") {
  Mesh base = makeQuadSphere(6);
  base.mask.resize(base.positions.size());
  for (Index v = 0; v < base.vertexCount(); ++v) base.mask[v] = base.positions[v].x > 0.2f ? 1.0f : 0.25f;
  base.faceSets.resize(base.faceHe.size());
  for (Index f = 0; f < base.faceCount(); ++f) base.faceSets[f] = base.faceCentroid(f).y > 0.0f ? 2 : -3;
  const CanonicalMap baseCanon = identityCanonicalMap(base);

  // Two live orders of the same canonical base: BVH builds with different leaf sizes.
  auto liveCopy = [&](int leafFaces) {
    Mesh m = base;
    Bvh bvh;
    ReorderMap map;
    bvh.build(m, Bvh::Params{leafFaces}, &map);
    CanonicalMap c;
    c.vert = map.vertOld;  // New -> old, and old is canonical.
    c.face = map.faceOld;
    c.faceStart = baseCanon.faceStart;
    return std::pair{std::move(m), std::move(c)};
  };
  auto [a, ca] = liveCopy(16);
  auto [b, cb] = liveCopy(1024);
  REQUIRE(a.heVert != b.heVert);
  const auto rules = [](const Mesh& m) { return classifyVertices(m); };
  auto ra = subdivide(a, ca, rules(a), SubdivideOptions{kMaxMultiresFaces, 64});
  auto rb = subdivide(b, cb, rules(b), SubdivideOptions{kMaxMultiresFaces, 1024});
  REQUIRE(ra);
  REQUIRE(rb);
  REQUIRE(ra->mesh.heVert != rb->mesh.heVert);
  std::vector<std::uint32_t> sa, pa, sb, pb;
  canonicalPolygons(ra->mesh, ra->canon, sa, pa);
  canonicalPolygons(rb->mesh, rb->canon, sb, pb);
  CHECK(sa == sb);
  CHECK(pa == pb);
  const auto posA = canonicalPositions(ra->mesh, ra->canon), posB = canonicalPositions(rb->mesh, rb->canon);
  CHECK(std::memcmp(posA.data(), posB.data(), posA.size() * sizeof(Vec3)) == 0);
  // Normals too: fans start at the same canonical half-edge.
  std::vector<Vec3> na(posA.size()), nb(posB.size());
  for (Index v = 0; v < ra->mesh.vertexCount(); ++v) na[ra->canon.vert[v]] = ra->mesh.normals[v];
  for (Index v = 0; v < rb->mesh.vertexCount(); ++v) nb[rb->canon.vert[v]] = rb->mesh.normals[v];
  CHECK(std::memcmp(na.data(), nb.data(), na.size() * sizeof(Vec3)) == 0);
  // And the next level.
  auto ra2 = subdivide(ra->mesh, ra->canon, ra->rule, SubdivideOptions{kMaxMultiresFaces, 256});
  auto rb2 = subdivide(rb->mesh, rb->canon, rb->rule);
  REQUIRE(ra2);
  REQUIRE(rb2);
  const auto posA2 = canonicalPositions(ra2->mesh, ra2->canon), posB2 = canonicalPositions(rb2->mesh, rb2->canon);
  CHECK(std::memcmp(posA2.data(), posB2.data(), posA2.size() * sizeof(Vec3)) == 0);
  canonicalPolygons(ra2->mesh, ra2->canon, sa, pa);
  canonicalPolygons(rb2->mesh, rb2->canon, sb, pb);
  CHECK(pa == pb);
  // Repeating a subdivision gives the same bits.
  auto again = subdivide(a, ca, rules(a), SubdivideOptions{kMaxMultiresFaces, 64});
  REQUIRE(again);
  CHECK(std::memcmp(again->mesh.positions.data(), ra->mesh.positions.data(), posA.size() * sizeof(Vec3)) == 0);
  CHECK(again->mesh.heTwin == ra->mesh.heTwin);
}

TEST_CASE("subdivision links point both ways") {
  Mesh m = makeIcosphere(1);
  Bvh bvh;
  bvh.build(m, Bvh::Params{32});
  auto r = subdivide(m, identityCanonicalMap(m), classifyVertices(m), SubdivideOptions{kMaxMultiresFaces, 32});
  REQUIRE(r);
  const SubdivisionLinks& L = r->links;
  const Mesh& c = r->mesh;
  REQUIRE(L.parent.size() == c.positions.size());
  REQUIRE(L.parentHalfEdge.size() == c.faceHe.size());
  for (Index v = 0; v < m.vertexCount(); ++v) {
    CHECK(L.parent[L.vertexChild[v]] == makeParent(kParentVertex, v));
  }
  for (Index f = 0; f < m.faceCount(); ++f) CHECK(L.parent[L.faceChild[f]] == makeParent(kParentFace, f));
  for (Index h = 0; h < m.halfEdgeCount(); ++h) {
    const std::uint32_t p = L.parent[L.edgeChild[h]];
    CHECK(parentKind(p) == kParentEdge);
    CHECK((parentIndex(p) == h || parentIndex(p) == m.heTwin[h]));
    CHECK(L.parentHalfEdge[L.childFace[h]] == h);
    // The child quad starts at the vertex child and runs edge child, face child, previous edge child.
    const Index q = L.childFace[h];
    Index he = c.faceHe[q];
    CHECK(c.heVert[he] == L.vertexChild[m.heVert[h]]);
    he = c.heNext[he];
    CHECK(c.heVert[he] == L.edgeChild[h]);
    he = c.heNext[he];
    CHECK(c.heVert[he] == L.faceChild[m.heFace[h]]);
    he = c.heNext[he];
    CHECK(c.heVert[he] == L.edgeChild[m.hePrev(h)]);
  }
}

TEST_CASE("new levels inherit face sets and hidden faces and interpolate the mask") {
  Mesh m = makeCube(2);
  m.mask.assign(m.positions.size(), 0.0f);
  for (Index v = 0; v < m.vertexCount(); ++v)
    if (m.positions[v].x > 0.5f) m.mask[v] = 1.0f;
  m.faceSets.resize(m.faceHe.size());
  for (Index f = 0; f < m.faceCount(); ++f) {
    const Vec3 c = m.faceCentroid(f);
    m.faceSets[f] = c.y > 0.5f ? -4 : (c.x < 0.0f ? 7 : 1);
  }
  const SubdivisionResult r = subdivideFlat(m);
  const Mesh& c = r.mesh;
  for (Index h = 0; h < m.halfEdgeCount(); ++h) CHECK(c.faceSets[r.links.childFace[h]] == m.faceSets[m.heFace[h]]);
  for (Index v = 0; v < m.vertexCount(); ++v) CHECK(c.mask[r.links.vertexChild[v]] == m.mask[v]);
  for (Index h = 0; h < m.halfEdgeCount(); ++h) {
    const float a = m.mask[m.heVert[h]], b = m.mask[m.heTarget(h)];
    const float e = c.mask[r.links.edgeChild[h]];
    if (a == b) CHECK(e == a);  // 0 and 1 stay exact.
    else CHECK(e == 0.5f);
  }
  for (Index f = 0; f < m.faceCount(); ++f) {
    bool all = true, none = true;
    m.forEachFaceVertex(f, [&](Index v) {
      all &= m.mask[v] == 1.0f;
      none &= m.mask[v] == 0.0f;
    });
    const float value = c.mask[r.links.faceChild[f]];
    if (all) CHECK(value == 1.0f);
    if (none) CHECK(value == 0.0f);
  }
}

TEST_CASE("subdivide refuses past the face cap and on empty meshes") {
  const Mesh m = makeCube(2);
  std::string error;
  SubdivideOptions o;
  o.maxFaces = m.halfEdgeCount() - 1;
  CHECK_FALSE(subdivide(m, identityCanonicalMap(m), classifyVertices(m), o, &error));
  CHECK(!error.empty());
  o.maxFaces = m.halfEdgeCount();
  CHECK(subdivide(m, identityCanonicalMap(m), classifyVertices(m), o));
  const Mesh empty;
  CHECK_FALSE(subdivide(empty, identityCanonicalMap(empty), {}, {}, &error));
}

TEST_CASE("vertexNormal equals computeNormals bit for bit, also on degenerate fans") {
  Mesh m = makeQuadSphere(5);
  // Collapse one face onto a point so the fans around it hold zero-area faces.
  const Index f = 3;
  const Vec3 c = m.faceCentroid(f);
  m.forEachFaceVertex(f, [&](Index v) { m.positions[v] = c; });
  m.positions[0] = m.positions[1];
  m.computeNormals();
  for (Index v = 0; v < m.vertexCount(); ++v) CHECK(sameBits(m.vertexNormal(v), m.normals[v]));
  std::vector<Vec3> before = m.normals;
  m.computeNormals(0, m.vertexCount());
  CHECK(std::memcmp(before.data(), m.normals.data(), before.size() * sizeof(Vec3)) == 0);
}
