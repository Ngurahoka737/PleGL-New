#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#include "TestUtil.h"
#include "io/Project.h"
#include "mesh/Primitives.h"
#include "multires/MultiresIo.h"
#include "multires/MultiresOps.h"
#include "multires/Subdivide.h"
#include "scene/Scene.h"
#include "sculpt/Sculptor.h"

using namespace plegl;

namespace {

template <class T>
bool sameArray(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

const Mesh& levelMesh(const SceneObject& o, int k) {
  return k == o.multires->active ? o.mesh : o.multires->levels[static_cast<std::size_t>(k)].mesh;
}

template <class T>
std::vector<T> canonical(const std::vector<T>& live, const std::vector<Index>& map, T fill) {
  std::vector<T> out(map.size(), fill);
  for (std::size_t i = 0; i < live.size(); ++i) out[static_cast<std::size_t>(map[i])] = live[i];
  return out;
}

// Every level of an object in canonical order, so objects with different BVH orders compare.
void requireSameCanonical(const SceneObject& a, const SceneObject& b) {
  REQUIRE(a.multires);
  REQUIRE(b.multires);
  REQUIRE(a.multires->levelCount() == b.multires->levelCount());
  CHECK(a.multires->active == b.multires->active);
  for (int k = 0; k < a.multires->levelCount(); ++k) {
    INFO("level " << k);
    const Mesh &ma = levelMesh(a, k), &mb = levelMesh(b, k);
    const CanonicalMap& ca = a.multires->levels[static_cast<std::size_t>(k)].canon;
    const CanonicalMap& cb = b.multires->levels[static_cast<std::size_t>(k)].canon;
    REQUIRE(ma.vertexCount() == mb.vertexCount());
    CHECK(sameArray(canonical(ma.positions, ca.vert, Vec3{0}), canonical(mb.positions, cb.vert, Vec3{0})));
    CHECK(sameArray(canonical(ma.normals, ca.vert, Vec3{0}), canonical(mb.normals, cb.vert, Vec3{0})));
    const std::vector<float> za(ma.positions.size(), 0.0f), zb(mb.positions.size(), 0.0f);
    CHECK(sameArray(canonical(ma.mask.empty() ? za : ma.mask, ca.vert, 0.0f),
                    canonical(mb.mask.empty() ? zb : mb.mask, cb.vert, 0.0f)));
    const std::vector<std::int32_t> da(ma.faceHe.size(), kDefaultFaceSet), db(mb.faceHe.size(), kDefaultFaceSet);
    CHECK(sameArray(canonical(ma.faceSets.empty() ? da : ma.faceSets, ca.face, 0),
                    canonical(mb.faceSets.empty() ? db : mb.faceSets, cb.face, 0)));
    std::vector<std::uint32_t> sa, pa, sb, pb;
    canonicalPolygons(ma, ca, sa, pa);
    canonicalPolygons(mb, cb, sb, pb);
    CHECK(sa == sb);
    CHECK(pa == pb);
  }
  // The reference too, through the pending edits.
  const MultiresFileData fa = captureLevels(a), fb = captureLevels(b);
  CHECK(fa.pendingPosIndex == fb.pendingPosIndex);
  CHECK(sameArray(fa.pendingPos, fb.pendingPos));
  CHECK(fa.pendingMaskIndex == fb.pendingMaskIndex);
  CHECK(fa.pendingSetIndex == fb.pendingSetIndex);
  CHECK(fa.baseTwins == fb.baseTwins);
  CHECK(fa.baseVertHe == fb.baseVertHe);
}

Vec3 surfacePoint(const SceneObject& obj, Vec3 dir) {
  dir = glm::normalize(dir);
  RayHit hit;
  REQUIRE(obj.bvh.raycast(obj.mesh, Ray{dir * 3.0f, -dir}, hit, std::numeric_limits<float>::infinity(), true));
  return hit.position;
}

void stroke(SceneObject& obj, Vec3 dir, float radius) {
  static DrawBrush draw;
  Sculptor sculptor;
  sculptor.beginStroke(obj, draw, {}, "Draw");
  const Vec3 c = surfacePoint(obj, dir);
  for (int i = 0; i < 3; ++i) sculptor.dab(c, radius, 0.7f);
  REQUIRE(sculptor.endStroke());
}

// Saves and opens a scene, returning the opened project's scene.
std::unique_ptr<Scene> roundTrip(const Scene& scene, std::string* error = nullptr) {
  const std::vector<std::uint8_t> bytes = serializeProject(scene, "");
  std::optional<Project> p = parseProject(bytes.data(), bytes.size(), error);
  if (!p || !buildProject(*p, error)) return nullptr;
  auto out = std::make_unique<Scene>();
  for (ProjectObject& po : p->objects) addProjectObject(*out, po);
  return out;
}

std::size_t findChunk(const std::vector<std::uint8_t>& bytes, const char* tag) {
  std::size_t at = 12;  // Magic and version.
  while (at + 12 <= bytes.size()) {
    if (std::memcmp(bytes.data() + at, tag, 4) == 0) return at;
    std::uint64_t size = 0;
    std::memcpy(&size, bytes.data() + at + 4, 8);
    at += 12 + std::size_t(size);
  }
  return std::string::npos;
}

void fixChecksum(std::vector<std::uint8_t>& bytes) {
  const std::uint32_t crc = crc32(bytes.data(), bytes.size() - 16);
  std::memcpy(bytes.data() + bytes.size() - 4, &crc, 4);
}

// A sculpted object with three levels, active on level 1, with pending edits on every channel.
SceneObject& makeSculpted(Scene& scene, SyncWorkspace& ws) {
  SceneObject& obj = scene.add("Head", makeQuadSphere(5));
  for (int i = 0; i < 3; ++i) REQUIRE(subdivideObject(obj, ws));
  stroke(obj, {0, 1, 0}, 0.3f);
  obj.mesh.ensureMask();
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) obj.mesh.mask[v] = obj.mesh.positions[v].x > 0.6f ? 1.0f : 0.0f;
  REQUIRE(setActiveLevel(obj, 1, ws));
  stroke(obj, {-1, 0, 0}, 0.4f);
  obj.mesh.ensureFaceSets();
  for (Index f = 0; f < obj.mesh.faceCount(); ++f)
    if (obj.mesh.faceCentroid(f).z > 0.5f) obj.mesh.faceSets[f] = -3;
  obj.mesh.mask[2] = 0.5f;
  return obj;
}

}  // namespace

TEST_CASE("levels round-trip bit-exactly with pending edits") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = makeSculpted(scene, ws);
  REQUIRE(!diffActive(*obj.multires, obj.mesh).empty());
  std::string error;
  auto loaded = roundTrip(scene, &error);
  INFO(error);
  REQUIRE(loaded);
  SceneObject& copy = *loaded->objects()[0];
  {
    const ValidationResult r = validateMultires(copy);
    INFO(r.message);
    REQUIRE(r.ok);
  }
  requireSameCanonical(obj, copy);
  // A level step after loading gives the same bits as one before saving.
  SyncWorkspace ws2;
  REQUIRE(setActiveLevel(obj, 3, ws));
  REQUIRE(setActiveLevel(copy, 3, ws2));
  requireSameCanonical(obj, copy);
  REQUIRE(setActiveLevel(obj, 0, ws));
  REQUIRE(setActiveLevel(copy, 0, ws2));
  requireSameCanonical(obj, copy);
  // Versions are fresh.
  for (int k = 0; k < 4; ++k)
    CHECK(copy.multires->levels[static_cast<std::size_t>(k)].version != obj.multires->levels[static_cast<std::size_t>(k)].version);
}

TEST_CASE("levels round-trip from the base and from the top") {
  for (int active : {0, 2}) {
    Scene scene;
    SyncWorkspace ws;
    SceneObject& obj = scene.add("Head", makeQuadSphere(4));
    REQUIRE(subdivideObject(obj, ws));
    REQUIRE(subdivideObject(obj, ws));
    stroke(obj, {0, 0, 1}, 0.4f);
    if (active != 2) REQUIRE(setActiveLevel(obj, active, ws));
    stroke(obj, {0, 1, 0}, 0.4f);
    auto loaded = roundTrip(scene);
    REQUIRE(loaded);
    requireSameCanonical(obj, *loaded->objects()[0]);
  }
}

TEST_CASE("a reader that skips MRES opens the active level") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = makeSculpted(scene, ws);
  std::vector<std::uint8_t> bytes = serializeProject(scene, "");
  const std::size_t at = findChunk(bytes, "MRES");
  REQUIRE(at != std::string::npos);
  std::uint64_t size = 0;
  std::memcpy(&size, bytes.data() + at + 4, 8);
  bytes.erase(bytes.begin() + static_cast<std::ptrdiff_t>(at), bytes.begin() + static_cast<std::ptrdiff_t>(at + 12 + size));
  fixChecksum(bytes);
  std::optional<Project> p = parseProject(bytes.data(), bytes.size());
  REQUIRE(p);
  REQUIRE(buildProject(*p));
  REQUIRE(!p->objects[0].multires);
  const Mesh& plain = p->objects[0].mesh;
  const CanonicalLevel c = canonicalActiveLevel(obj);
  // The plain object is the active level, in canonical order before its BVH build reordered it:
  // compare as sets of positions with their masks.
  REQUIRE(plain.vertexCount() == obj.mesh.vertexCount());
  REQUIRE(plain.faceCount() == obj.mesh.faceCount());
  std::vector<std::pair<std::array<float, 4>, int>> a, b;
  for (Index v = 0; v < plain.vertexCount(); ++v)
    a.push_back({{plain.positions[v].x, plain.positions[v].y, plain.positions[v].z, plain.mask[v]}, 0});
  for (std::size_t v = 0; v < c.positions.size(); ++v)
    b.push_back({{c.positions[v].x, c.positions[v].y, c.positions[v].z, c.mask[v]}, 0});
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  CHECK(a == b);
  CHECK(plain.anyHidden());
}

TEST_CASE("a non-manifold base reloads with identical twins") {
  // A strip of quads plus two quads that meet the strip at one vertex only.
  std::vector<Vec3> p;
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 4; ++i) p.push_back({float(i), float(j), 0.0f});
  p.push_back({4.0f, 3.0f, 0.5f});
  p.push_back({5.0f, 3.0f, 0.5f});
  p.push_back({5.0f, 4.0f, 0.5f});
  std::vector<Index> idx, sizes;
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 3; ++i) {
      const Index v = j * 4 + i;
      idx.insert(idx.end(), {v, v + 1, v + 5, v + 4});
      sizes.push_back(4);
    }
  idx.insert(idx.end(), {11, 12, 13, 14});  // Shares vertex 11 with the grid.
  sizes.push_back(4);
  p.push_back({4.0f, 4.0f, 0.5f});
  BuildReport report;
  Mesh m = buildMesh(p, idx, sizes, &report);
  REQUIRE(report.nonManifoldVertices == 1);
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = scene.add("Bowtie", std::move(m));
  REQUIRE(subdivideObject(obj, ws));
  REQUIRE(subdivideObject(obj, ws));
  REQUIRE(!obj.multires->levels[0].nonManifoldFaces.empty());
  REQUIRE(setActiveLevel(obj, 0, ws));
  for (Vec3& q : obj.mesh.positions) q.z += q.x * 0.25f;
  obj.mesh.computeNormals();
  obj.bvh.refit(obj.mesh);
  REQUIRE(setActiveLevel(obj, 2, ws));
  auto loaded = roundTrip(scene);
  REQUIRE(loaded);
  // validateMultires() refuses non-manifold fans (validateLive does), so compare only.
  requireSameCanonical(obj, *loaded->objects()[0]);
}

TEST_CASE("a base face that repeats a corner is refused at any active level") {
  // A hexagon a b x b a y with no twins passes every twin and fan check, and its level 1 quads
  // all have distinct corners, so only the corner check catches it. Saving again from level 0
  // would fail, because buildMesh drops the hexagon.
  Mesh base;
  base.positions = {{0, 0, 0}, {1, 0, 0}, {2, 1, 0}, {0, 2, 0}};
  base.heVert = {0, 1, 2, 1, 0, 3};
  base.heNext = {1, 2, 3, 4, 5, 0};
  base.heFace.assign(6, 0);
  base.heTwin.assign(6, kInvalid);
  base.faceHe = {0};
  base.vertHe = {0, 1, 2, 5};
  base.computeNormals();
  MultiresFileData d;
  d.active = 1;
  d.baseVertices = 4;
  d.baseSizes = {6};
  d.baseCorners = {0, 1, 2, 1, 0, 3};
  d.baseTwins.assign(6, kInvalid);
  d.baseVertHe = {0, 1, 2, 5};
  std::vector<Mesh> meshes{base};
  std::vector<CanonicalMap> canons{identityCanonicalMap(base)};
  std::vector<std::vector<VertexRule>> rules{classifyVertices(base)};
  for (int k = 1; k < 3; ++k) {
    SubdivideOptions o;
    o.buildBvh = false;
    auto r = subdivide(meshes.back(), canons.back(), rules.back(), o);
    REQUIRE(r);
    meshes.push_back(std::move(r->mesh));
    canons.push_back(std::move(r->canon));
    rules.push_back(std::move(r->rule));
  }
  Mesh objs;
  for (int k = 0; k < 3; ++k) {
    const Mesh& m = meshes[static_cast<std::size_t>(k)];
    const CanonicalMap& c = canons[static_cast<std::size_t>(k)];
    MultiresFileLevel l;
    l.vertices = static_cast<std::uint32_t>(m.vertexCount());
    l.faces = static_cast<std::uint32_t>(m.faceCount());
    std::vector<Vec3> cp(m.positions.size());
    for (Index v = 0; v < m.vertexCount(); ++v) cp[static_cast<std::size_t>(c.vert[v])] = m.positions[v];
    if (k == d.active) {
      std::vector<std::uint32_t> sizes, corners;
      canonicalPolygons(m, c, sizes, corners);
      const std::vector<Index> idx(corners.begin(), corners.end()), sz(sizes.begin(), sizes.end());
      objs = buildMesh(cp, idx, sz);
      REQUIRE(objs.faceCount() == m.faceCount());
    } else {
      l.channels = kLevelPositions;
      l.positions = std::move(cp);
    }
    d.levels.push_back(std::move(l));
  }
  Bvh bvh;
  std::shared_ptr<Multires> stack;
  std::string error;
  CHECK_FALSE(restoreLevels(d, objs, bvh, stack, &error));
  CHECK(error.find("base face corners") != std::string::npos);
}

TEST_CASE("a mesh with non-finite positions is not subdivided") {
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = scene.add("Head", makeQuadSphere(4));
  obj.mesh.positions[3].z = std::numeric_limits<float>::quiet_NaN();
  std::string error;
  CHECK_FALSE(prepareSubdivide(obj, &error));
  CHECK(error.find("invalid coordinates") != std::string::npos);
  CHECK(!obj.multires);
}

TEST_CASE("damaged level data is refused") {
  Scene scene;
  SyncWorkspace ws;
  makeSculpted(scene, ws);
  const std::vector<std::uint8_t> good = serializeProject(scene, "");
  const std::size_t at = findChunk(good, "MRES");
  REQUIRE(at != std::string::npos);
  const std::size_t body = at + 12;  // u32 count, then u32 index, u8 encoding, count, active, reserved.

  SUBCASE("chunk layout") {
    struct Case {
      const char* name;
      std::size_t offset;
      std::uint8_t value;
    };
    for (const Case& c : {Case{"object index", body + 4, 7}, Case{"encoding", body + 8, 1},
                          Case{"level count", body + 9, 1}, Case{"level count", body + 9, 9},
                          Case{"active level", body + 10, 4}, Case{"reserved", body + 11, 1}}) {
      INFO(c.name);
      std::vector<std::uint8_t> bytes = good;
      bytes[c.offset] = c.value;
      fixChecksum(bytes);
      std::string error;
      std::optional<Project> p = parseProject(bytes.data(), bytes.size(), &error);
      CHECK((!p || !buildProject(*p, &error)));
      CHECK(error.find("subdivision level data") != std::string::npos);
    }
  }
  SUBCASE("two entries for one object") {
    // Repeat the only entry: count 2, then the same bytes twice.
    std::uint64_t size = 0;
    std::memcpy(&size, good.data() + at + 4, 8);
    const std::vector<std::uint8_t> entry(good.begin() + static_cast<std::ptrdiff_t>(body + 4),
                                          good.begin() + static_cast<std::ptrdiff_t>(body + size));
    std::vector<std::uint8_t> bytes(good.begin(), good.begin() + static_cast<std::ptrdiff_t>(at + 4));
    const std::uint64_t newSize = 4 + 2 * entry.size();
    const std::uint32_t two = 2;
    bytes.insert(bytes.end(), reinterpret_cast<const std::uint8_t*>(&newSize),
                 reinterpret_cast<const std::uint8_t*>(&newSize) + 8);
    bytes.insert(bytes.end(), reinterpret_cast<const std::uint8_t*>(&two), reinterpret_cast<const std::uint8_t*>(&two) + 4);
    bytes.insert(bytes.end(), entry.begin(), entry.end());
    bytes.insert(bytes.end(), entry.begin(), entry.end());
    bytes.insert(bytes.end(), good.begin() + static_cast<std::ptrdiff_t>(body + size), good.end());
    fixChecksum(bytes);
    std::string error;
    CHECK(!parseProject(bytes.data(), bytes.size(), &error));
    CHECK(error.find("subdivision level data") != std::string::npos);
  }
  SUBCASE("truncated chunk") {
    std::vector<std::uint8_t> bytes = good;
    // Claim a huge base face count: the reader must refuse before allocating.
    const std::uint32_t huge = 0x7FFFFFF0u;
    std::memcpy(bytes.data() + body + 16, &huge, 4);
    fixChecksum(bytes);
    std::string error;
    CHECK(!parseProject(bytes.data(), bytes.size(), &error));
    CHECK(error.find("subdivision level data") != std::string::npos);
  }
  SUBCASE("contents") {
    std::optional<Project> p = parseProject(good.data(), good.size());
    REQUIRE(p);
    const MultiresFileData original = *p->objects[0].levelData;
    const Mesh objs = p->objects[0].mesh;
    auto refused = [&](const char* name, auto&& damage) {
      INFO(name);
      MultiresFileData d = original;
      Mesh m = objs;
      damage(d, m);
      Bvh bvh;
      std::shared_ptr<Multires> stack;
      std::string error;
      CHECK_FALSE(restoreLevels(d, m, bvh, stack, &error));
      CHECK(error.find("subdivision level data") != std::string::npos);
    };
    {
      MultiresFileData d = original;
      Mesh m = objs;
      Bvh bvh;
      std::shared_ptr<Multires> stack;
      REQUIRE(restoreLevels(d, m, bvh, stack));
    }
    refused("face size", [](MultiresFileData& d, Mesh&) { d.baseSizes[0] = 2; });
    refused("corner range", [](MultiresFileData& d, Mesh&) { d.baseCorners[3] = d.baseVertices; });
    refused("twin not involutive", [](MultiresFileData& d, Mesh&) { std::swap(d.baseTwins[0], d.baseTwins[1]); });
    refused("twin of itself", [](MultiresFileData& d, Mesh&) { d.baseTwins[d.baseTwins[5]] = 5, d.baseTwins[5] = 5; });
    refused("one-sided twin", [](MultiresFileData& d, Mesh&) {
      // Keeps every endpoint and the edge count right; only the missing way back is wrong.
      std::size_t h = 0;
      while (d.baseTwins[h] <= std::int32_t(h)) ++h;
      d.baseTwins[h] = kInvalid;
    });
    refused("fan start", [](MultiresFileData& d, Mesh&) { d.baseVertHe[0] = d.baseVertHe[1]; });
    refused("unused fan start", [](MultiresFileData& d, Mesh&) { d.baseVertHe[4] = kInvalid; });
    refused("level count", [](MultiresFileData& d, Mesh&) { d.levels[2].vertices += 1; });
    refused("missing positions", [](MultiresFileData& d, Mesh&) { d.levels[0].channels = 0; });
    refused("active channels", [](MultiresFileData& d, Mesh&) { d.levels[1].channels = kLevelPositions; });
    refused("mask size", [](MultiresFileData& d, Mesh&) {
      d.levels[3].channels |= kLevelMask;
      d.levels[3].mask.assign(5, 0.0f);
    });
    refused("pending order", [](MultiresFileData& d, Mesh&) {
      REQUIRE(d.pendingPosIndex.size() > 1);
      std::swap(d.pendingPosIndex[0], d.pendingPosIndex[1]);
    });
    refused("pending range", [](MultiresFileData& d, Mesh&) { d.pendingSetIndex.back() = 1u << 30; });
    refused("pending at the end", [](MultiresFileData& d, Mesh&) {
      d.pendingPosIndex.back() = d.levels[static_cast<std::size_t>(d.active)].vertices;
    });
    refused("face set pending at the end", [](MultiresFileData& d, Mesh&) {
      d.pendingSetIndex.back() = d.levels[static_cast<std::size_t>(d.active)].faces;
    });
    refused("active polygons", [](MultiresFileData&, Mesh& m) {
      // Same counts, other corners: rotate one face's corner order.
      std::vector<Index> idx, sizes;
      for (Index f = 0; f < m.faceCount(); ++f) {
        std::vector<Index> c;
        m.forEachFaceVertex(f, [&](Index v) { c.push_back(v); });
        if (f == 3) std::rotate(c.begin(), c.begin() + 1, c.end());
        idx.insert(idx.end(), c.begin(), c.end());
        sizes.push_back(static_cast<Index>(c.size()));
      }
      Mesh rebuilt = buildMesh(m.positions, idx, sizes);
      rebuilt.mask = m.mask;
      rebuilt.faceSets = m.faceSets;
      m = std::move(rebuilt);
    });
    refused("active size", [](MultiresFileData&, Mesh& m) { m.positions.push_back(Vec3{0}), m.vertHe.push_back(kInvalid); });
    refused("active NaN", [](MultiresFileData&, Mesh& m) { m.positions[7].y = std::numeric_limits<float>::quiet_NaN(); });
    refused("active infinity", [](MultiresFileData&, Mesh& m) { m.positions[0].x = std::numeric_limits<float>::infinity(); });
    refused("repeated base corner", [](MultiresFileData& d, Mesh&) { d.baseCorners[1] = d.baseCorners[0]; });
  }
}
