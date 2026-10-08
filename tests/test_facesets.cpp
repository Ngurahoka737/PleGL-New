#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <set>

#include "TestUtil.h"
#include "io/Project.h"
#include "mesh/MeshEdit.h"
#include "mesh/Primitives.h"
#include "remesh/QuadRemesh.h"
#include "scene/Scene.h"
#include "sculpt/FaceSetOps.h"
#include "sculpt/MaskOps.h"
#include "sculpt/Sculptor.h"
#include "spatial/LeafLayout.h"

using namespace plegl;

namespace {

// A face set value that depends only on where the face is, so it can be checked after the faces
// were renumbered: octants get sets 1..8, and the octant with x, y and z all negative is hidden.
std::int32_t setOf(const Vec3& c) {
  const std::int32_t id = 1 + (c.x > 0.0f ? 1 : 0) + (c.y > 0.0f ? 2 : 0) + (c.z > 0.0f ? 4 : 0);
  return id == 1 ? -id : id;
}

void setFaceSets(Mesh& m, std::int32_t (*f)(const Vec3&)) {
  m.faceSets.resize(m.faceHe.size());
  for (Index g = 0; g < m.faceCount(); ++g) m.faceSets[g] = f(m.faceCentroid(g));
}

void requireSetsMatch(const Mesh& m, std::int32_t (*f)(const Vec3&)) {
  REQUIRE(m.faceSets.size() == m.faceHe.size());
  for (Index g = 0; g < m.faceCount(); ++g) {
    if (m.faceSets[g] != f(m.faceCentroid(g))) {
      INFO("face " << g);
      REQUIRE(m.faceSets[g] == f(m.faceCentroid(g)));
    }
  }
}

Vec3 surfacePoint(const SceneObject& obj, Vec3 dir) {
  dir = glm::normalize(dir);
  RayHit hit;
  const bool ok = obj.bvh.raycast(obj.mesh, Ray{dir * 3.0f, -dir}, hit, std::numeric_limits<float>::infinity(), true);
  REQUIRE(ok);
  return hit.position;
}

std::optional<SculptUndo> stroke(SceneObject& obj, const Brush& brush, Vec3 dir, float radius, StrokeOptions opts = {},
                                 float strength = 1.0f, int dabs = 1) {
  Sculptor sculptor;
  sculptor.beginStroke(obj, brush, opts, brush.name());
  const Vec3 c = surfacePoint(obj, dir);
  for (int i = 0; i < dabs; ++i) sculptor.dab(c, radius, strength);
  auto undo = sculptor.endStroke();
  if (!undo) return std::nullopt;
  return std::get<SculptUndo>(std::move(*undo));
}

// Byte offset of the first chunk with this tag, or npos.
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

// Two separate spheres in one mesh.
Mesh twoSpheres() {
  std::vector<Vec3> pos;
  std::vector<Index> idx, sizes;
  for (float x : {-1.5f, 1.5f}) {
    const Mesh s = makeIcosphere(2);
    const Index base = static_cast<Index>(pos.size());
    for (const Vec3& p : s.positions) pos.push_back(p + Vec3{x, 0.0f, 0.0f});
    for (Index f = 0; f < s.faceCount(); ++f) {
      Index n = 0;
      s.forEachFaceVertex(f, [&](Index v) {
        idx.push_back(base + v);
        ++n;
      });
      sizes.push_back(n);
    }
  }
  return buildMesh(std::move(pos), idx, sizes);
}

std::set<std::int32_t> idsIn(const Mesh& m) {
  std::set<std::int32_t> ids;
  for (Index f = 0; f < m.faceCount(); ++f) ids.insert(faceSetId(m.faceSetValue(f)));
  return ids;
}

}  // namespace

// ----- Mesh plumbing -------------------------------------------------------------------------

TEST_CASE("face sets follow their faces through BVH reordering and duplication") {
  Scene scene;
  Mesh m = makeQuadSphere(12);
  setFaceSets(m, setOf);
  SceneObject& obj = scene.add("A", std::move(m));  // Builds the BVH, which reorders the mesh.
  requireSetsMatch(obj.mesh, setOf);
  test::requireValid(obj.mesh);
  CHECK(obj.mesh.anyFaceSet());
  CHECK(obj.mesh.anyHidden());
  CHECK(obj.mesh.hasFaceSetData());
  CHECK(obj.mesh.maxFaceSetId() == 8);

  SceneObject* copy = scene.duplicate(obj.id);
  REQUIRE(copy);
  requireSetsMatch(copy->mesh, setOf);
}

TEST_CASE("a mesh without face sets reads as one visible default set") {
  const Mesh m = makeIcosphere(2);
  CHECK(m.faceSets.empty());
  CHECK(m.faceSetValue(0) == kDefaultFaceSet);
  CHECK_FALSE(m.faceHidden(0));
  CHECK_FALSE(m.anyFaceSet());
  CHECK_FALSE(m.anyHidden());
  CHECK_FALSE(m.hasFaceSetData());
  CHECK(m.maxFaceSetId() == kDefaultFaceSet);
  CHECK(m.vertexVisible(0));
}

TEST_CASE("a vertex is visible while any of its faces is") {
  Mesh m = makeIcosphere(1);
  m.ensureFaceSets();
  const Index v = 0;
  std::vector<Index> faces;
  m.forEachOutgoing(v, [&](Index h) { faces.push_back(m.heFace[h]); });
  REQUIRE(faces.size() >= 3);
  for (std::size_t i = 0; i + 1 < faces.size(); ++i) m.faceSets[faces[i]] = -kDefaultFaceSet;
  CHECK(m.vertexVisible(v));
  m.faceSets[faces.back()] = -kDefaultFaceSet;
  CHECK_FALSE(m.vertexVisible(v));
}

TEST_CASE("validate rejects face sets of the wrong size or with a zero") {
  Mesh m = makeIcosphere(1);
  m.faceSets.assign(m.faceHe.size() - 1, 1);
  CHECK_FALSE(validate(m).ok);
  m.faceSets.assign(m.faceHe.size(), 1);
  CHECK(validate(m).ok);
  m.faceSets[3] = 0;
  CHECK_FALSE(validate(m).ok);
}

TEST_CASE("splitting a face keeps both halves in its set, and compact keeps every set") {
  Mesh m = makeQuadSphere(6);
  setFaceSets(m, setOf);
  MeshEditor ed(m);
  const Index f = 5;
  const std::int32_t before = m.faceSets[f];
  const Index ha = m.faceHe[f];
  const Index g = ed.splitFace(ha, m.heNext[m.heNext[ha]]);
  REQUIRE(g != kInvalid);
  CHECK(m.faceSets.size() == m.faceHe.size());
  CHECK(m.faceSets[f] == before);
  CHECK(m.faceSets[g] == before);
  // Collapse an edge somewhere else and compact: dead faces drop out, every live one keeps its set.
  std::vector<std::int32_t> byCentroid;
  REQUIRE(ed.collapseEdge(m.faceHe[40], m.positions[m.heVert[m.faceHe[40]]]));
  std::vector<std::pair<Vec3, std::int32_t>> live;
  for (Index k = 0; k < m.faceCount(); ++k)
    if (m.faceHe[k] != kInvalid) live.emplace_back(m.faceCentroid(k), m.faceSets[k]);
  ed.compact();
  test::requireValid(m);
  REQUIRE(static_cast<std::size_t>(m.faceCount()) == live.size());
  for (Index k = 0; k < m.faceCount(); ++k) {
    CHECK(glm::length(m.faceCentroid(k) - live[k].first) < 1e-6f);
    CHECK(m.faceSets[k] == live[k].second);
  }
}

TEST_CASE("headroom covers face sets") {
  Mesh m = makeIcosphere(2);
  m.ensureFaceSets();
  m.faceSets.shrink_to_fit();
  m.reserveHeadroom(m.vertexCount() + 10, m.faceCount() + 10, m.halfEdgeCount() + 30);
  CHECK(m.hasHeadroom(10, 10, 30));
  CHECK(m.faceSets.capacity() >= m.faceSets.size() + 10);
  m.faceSets.shrink_to_fit();
  CHECK_FALSE(m.hasHeadroom(0, 1, 0));
}

// ----- Files and remeshing -----------------------------------------------------------------------

TEST_CASE("projects keep face sets and hidden faces, and plain projects have no face set chunk") {
  Scene scene;
  Mesh a = makeQuadSphere(8);
  setFaceSets(a, setOf);
  scene.add("A", std::move(a));
  Mesh b = makeIcosphere(2);
  b.ensureFaceSets();  // All default: nothing worth saving.
  scene.add("B", std::move(b));
  const auto bytes = serializeProject(scene, "");
  std::string error;
  auto project = parseProject(bytes.data(), bytes.size(), &error);
  INFO(error);
  REQUIRE(project);
  REQUIRE(project->objects.size() == 2);
  requireSetsMatch(project->objects[0].mesh, setOf);
  CHECK(project->objects[1].mesh.faceSets.empty());
  // Loading builds the BVH, which reorders faces; the sets must follow.
  Scene loaded;
  SceneObject& obj = loaded.add("A", std::move(project->objects[0].mesh));
  requireSetsMatch(obj.mesh, setOf);

  Scene plain;
  plain.add("P", makeIcosphere(2));
  const auto plainBytes = serializeProject(plain, "");
  CHECK(findChunk(plainBytes, "FSET") == std::string::npos);
}

TEST_CASE("a project without its face set chunk still opens") {
  Scene scene;
  Mesh a = makeQuadSphere(8);
  setFaceSets(a, setOf);
  scene.add("A", std::move(a));
  auto bytes = serializeProject(scene, "");
  const std::size_t at = findChunk(bytes, "FSET");
  REQUIRE(at != std::string::npos);
  bytes[at] = 'X';  // An unknown chunk, as an older build would see it.
  fixChecksum(bytes);
  auto project = parseProject(bytes.data(), bytes.size());
  REQUIRE(project);
  CHECK(project->objects[0].mesh.faceSets.empty());
}

TEST_CASE("damaged face set chunks are refused") {
  Scene scene;
  Mesh a = makeQuadSphere(8);
  setFaceSets(a, setOf);
  scene.add("A", std::move(a));
  const auto good = serializeProject(scene, "");
  const std::size_t at = findChunk(good, "FSET");
  REQUIRE(at != std::string::npos);
  const std::size_t payload = at + 12;  // u32 count, then u32 index, u32 face count, u8 encoding.
  std::string error;
  auto refused = [&](std::vector<std::uint8_t> b) {
    fixChecksum(b);
    return !parseProject(b.data(), b.size(), &error);
  };
  SUBCASE("wrong face count") {
    auto b = good;
    std::uint32_t n = 0;
    std::memcpy(&n, b.data() + payload + 8, 4);
    ++n;
    std::memcpy(b.data() + payload + 8, &n, 4);
    CHECK(refused(b));
    CHECK(error.find("face set") != std::string::npos);
  }
  SUBCASE("object index out of range") {
    auto b = good;
    const std::uint32_t index = 7;
    std::memcpy(b.data() + payload + 4, &index, 4);
    CHECK(refused(b));
  }
  SUBCASE("unknown encoding") {
    auto b = good;
    b[payload + 12] = 9;
    CHECK(refused(b));
  }
  SUBCASE("a zero or the smallest integer as a value") {
    for (std::int32_t bad : {0, std::numeric_limits<std::int32_t>::min()}) {
      auto b = good;
      std::memcpy(b.data() + payload + 13 + 4 * 3, &bad, 4);
      CHECK(refused(b));
    }
  }
}

TEST_CASE("remeshing carries face sets and hidden faces over to the new surface") {
  Mesh in = makeQuadSphere(32);
  setFaceSets(in, [](const Vec3& c) -> std::int32_t { return c.y > 0.0f ? 5 : -3; });
  for (int rounds : {3, 0}) {
    INFO("rounds " << rounds);
    auto out = quadRemesh(in, {.targetEdge = 0.06f, .rounds = rounds});
    REQUIRE(out);
    REQUIRE(out->faceSets.size() == out->faceHe.size());
    test::requireValid(*out);
    int checked = 0;
    for (Index f = 0; f < out->faceCount(); ++f) {
      const float y = out->faceCentroid(f).y;
      if (y > 0.1f) {
        CHECK(out->faceSets[f] == 5);
        ++checked;
      } else if (y < -0.1f) {
        CHECK(out->faceSets[f] == -3);
        ++checked;
      }
    }
    CHECK(checked > 100);
  }
  // The far side of a thin slab keeps its own set.
  Mesh slab = makeCube(16, 2.0f);
  for (Vec3& p : slab.positions) p.y *= 0.04f;  // 0.08 thick, thinner than a few edges.
  slab.computeNormals();
  setFaceSets(slab, [](const Vec3& c) -> std::int32_t { return c.y > 0.0f ? 2 : 3; });
  auto thin = quadRemesh(slab, {.targetEdge = 0.05f, .rounds = 0});
  REQUIRE(thin);
  int wrong = 0, big = 0;
  for (Index f = 0; f < thin->faceCount(); ++f) {
    const Vec3 n = thin->faceAreaNormal(f);
    const float len = glm::length(n);
    if (len <= 0.0f || std::abs(n.y / len) < 0.9f) continue;  // Only the two large sides.
    ++big;
    wrong += thin->faceSets[f] != (n.y > 0.0f ? 2 : 3);
  }
  CHECK(big > 100);
  CHECK(wrong == 0);
  // No face set data in, none out.
  Mesh plainIn = makeQuadSphere(16);
  plainIn.ensureFaceSets();
  auto plain = quadRemesh(plainIn, {.targetEdge = 0.1f});
  REQUIRE(plain);
  CHECK(plain->faceSets.empty());
}

TEST_CASE("undo memory counts face sets") {
  MeshState state;
  state.mesh = makeQuadSphere(8);
  const std::size_t plain = state.bytes();
  state.mesh.ensureFaceSets();
  CHECK(state.bytes() == plain + state.mesh.faceSets.size() * sizeof(std::int32_t));
  SculptUndo e;
  e.before.push_back({0, {}, {}, {}, std::vector<std::int32_t>(10)});
  CHECK(e.bytes() == 10 * sizeof(std::int32_t));
  Mesh m = makeIcosphere(1);
  m.ensureFaceSets();
  Bvh bvh;
  bvh.build(m);
  const TopoLeafState leaf = captureLeaf(m, bvh, 0);
  CHECK(leaf.faceSets.size() == leaf.faceHe.size());
}

// ----- Face set brush ----------------------------------------------------------------------------

TEST_CASE("the face set brush paints a new set under the dab, and undo takes it back") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(24));
  const Mesh before = obj.mesh;
  FaceSetBrush brush;
  auto first = stroke(obj, brush, {0, 1, 0}, 0.3f);
  REQUIRE(first);
  REQUIRE(obj.mesh.faceSets.size() == obj.mesh.faceHe.size());
  CHECK(obj.mesh.positions == before.positions);  // Painting never moves the surface.
  const Vec3 top = surfacePoint(obj, {0, 1, 0});
  int painted = 0;
  for (Index f = 0; f < obj.mesh.faceCount(); ++f) {
    const float d = glm::length(obj.mesh.faceCentroid(f) - top);
    if (d < 0.25f) CHECK(obj.mesh.faceSets[f] == 2);
    if (d > 0.32f) CHECK(obj.mesh.faceSets[f] == kDefaultFaceSet);
    painted += obj.mesh.faceSets[f] == 2;
  }
  CHECK(painted > 10);
  CHECK_FALSE(obj.faceSetDirtyLeaves.empty());
  CHECK(first->before.size() == first->after.size());
  for (const LeafState& s : first->before) {
    CHECK(s.positions.empty());
    CHECK(s.mask.empty());
    CHECK_FALSE(s.faceSets.empty());
  }

  // The next stroke starts another new set, even where it overlaps the first.
  auto second = stroke(obj, brush, {0.3f, 1, 0}, 0.3f);
  REQUIRE(second);
  CHECK(idsIn(obj.mesh) == std::set<std::int32_t>{1, 2, 3});

  UndoStack stack;
  stack.push(std::move(*first));
  stack.push(std::move(*second));
  const auto after = obj.mesh.faceSets;
  CHECK(stack.undo(scene) == "Face Set");
  CHECK(idsIn(obj.mesh) == std::set<std::int32_t>{1, 2});
  CHECK_FALSE(stack.undo(scene).empty());
  CHECK(idsIn(obj.mesh) == std::set<std::int32_t>{1});
  CHECK_FALSE(stack.redo(scene).empty());
  CHECK_FALSE(stack.redo(scene).empty());
  CHECK(obj.mesh.faceSets == after);
}

TEST_CASE("the face set brush can continue the set under the stroke start") {
  Scene scene;
  Mesh m = makeQuadSphere(24);
  setFaceSets(m, [](const Vec3& c) -> std::int32_t { return c.x > 0.0f ? 4 : 7; });
  SceneObject& obj = scene.add("A", std::move(m));
  FaceSetBrush brush;
  StrokeOptions opts;
  opts.extendFaceSet = true;
  // Start on the x > 0 side and reach well across: the set there (4) grows.
  auto undo = stroke(obj, brush, {0.1f, 1, 0}, 0.5f, opts);
  REQUIRE(undo);
  CHECK(idsIn(obj.mesh) == std::set<std::int32_t>{4, 7});
  const Vec3 c = surfacePoint(obj, {0.1f, 1, 0});
  for (Index f = 0; f < obj.mesh.faceCount(); ++f)
    if (glm::length(obj.mesh.faceCentroid(f) - c) < 0.4f) CHECK(obj.mesh.faceSets[f] == 4);
}

TEST_CASE("a light touch paints a thinner line, and masked or hidden faces keep their set") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(32));
  FaceSetBrush brush;
  auto count = [&](std::int32_t id) {
    return std::count(obj.mesh.faceSets.begin(), obj.mesh.faceSets.end(), id);
  };
  stroke(obj, brush, {0, 1, 0}, 0.4f, {}, 1.0f);
  const auto full = count(2);
  stroke(obj, brush, {0, -1, 0}, 0.4f, {}, 0.3f);
  const auto light = count(3);
  CHECK(light > 0);
  CHECK(light < full / 2);

  // Mask one side and hide the other; a stroke across both paints neither.
  Mesh& m = obj.mesh;
  m.ensureMask();
  for (Index v = 0; v < m.vertexCount(); ++v) m.mask[v] = m.positions[v].z > 0.0f ? 1.0f : 0.0f;
  for (Index f = 0; f < m.faceCount(); ++f)
    if (m.faceCentroid(f).z < 0.0f) m.faceSets[f] = -faceSetId(m.faceSets[f]);
  const auto sets = m.faceSets;
  CHECK_FALSE(stroke(obj, brush, {1, 0, 0.05f}, 0.6f));
  CHECK(m.faceSets == sets);
}

TEST_CASE("face set strokes mirror across X with one new set for both sides") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(24));
  FaceSetBrush brush;
  StrokeOptions opts;
  opts.symmetryX = true;
  REQUIRE(stroke(obj, brush, {1, 0.2f, 0}, 0.3f, opts));
  const Vec3 c = surfacePoint(obj, {1, 0.2f, 0});
  const Vec3 mirrored{-c.x, c.y, c.z};
  int left = 0, right = 0;
  for (Index f = 0; f < obj.mesh.faceCount(); ++f) {
    const Vec3 k = obj.mesh.faceCentroid(f);
    if (obj.mesh.faceSets[f] != 2) continue;
    right += glm::length(k - c) < 0.31f;
    left += glm::length(k - mirrored) < 0.31f;
  }
  CHECK(right > 5);
  CHECK(left == right);
  CHECK(idsIn(obj.mesh) == std::set<std::int32_t>{1, 2});
}

// ----- Auto-masking and hiding ---------------------------------------------------------------------

TEST_CASE("face set auto-masking moves only the set under the stroke start") {
  for (bool symmetry : {false, true}) {
    INFO("symmetry " << symmetry);
    Scene scene;
    Mesh m = makeQuadSphere(24);
    // Two bands: z > 0.1 is set 2, the rest set 3. The mirror side (x < 0) has the same bands.
    setFaceSets(m, [](const Vec3& c) -> std::int32_t { return c.z > 0.1f ? 2 : 3; });
    SceneObject& obj = scene.add("A", std::move(m));
    const Mesh before = obj.mesh;
    DrawBrush draw;
    StrokeOptions opts;
    opts.faceSetAutoMask = true;
    opts.symmetryX = symmetry;
    REQUIRE(stroke(obj, draw, {0.7f, 0, 0.7f}, 0.8f, opts, 1.0f, 4));  // Starts in set 2.
    int moved = 0;
    for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
      bool inSet = false;
      before.forEachOutgoing(v, [&](Index h) { inSet |= before.faceSets[before.heFace[h]] == 2; });
      const bool changed = obj.mesh.positions[v] != before.positions[v];
      if (!inSet) CHECK_FALSE(changed);
      moved += changed && before.positions[v].x < 0.0f;
    }
    CHECK((moved > 0) == symmetry);
  }
}

TEST_CASE("locked face set boundaries stay put") {
  Scene scene;
  Mesh m = makeQuadSphere(24);
  setFaceSets(m, [](const Vec3& c) -> std::int32_t { return c.y > 0.0f ? 2 : 3; });
  SceneObject& obj = scene.add("A", std::move(m));
  const Mesh before = obj.mesh;
  for (bool grab : {false, true}) {
    INFO("grab " << grab);
    StrokeOptions opts;
    opts.lockFaceSetBoundaries = true;
    if (grab) {
      Sculptor s;
      REQUIRE(s.beginGrab(obj, opts, surfacePoint(obj, {1, 0, 0}), 0.6f, "Grab"));
      s.grab({0.0f, 0.3f, 0.0f});
      s.endStroke();
    } else {
      SmoothBrush smooth;
      REQUIRE(stroke(obj, smooth, {1, 0, 0}, 0.6f, opts, 1.0f, 3));
    }
    int interior = 0;
    for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
      std::set<std::int32_t> ids;
      before.forEachOutgoing(v, [&](Index h) { ids.insert(before.faceSets[before.heFace[h]]); });
      if (ids.size() > 1) {
        CHECK(obj.mesh.positions[v] == before.positions[v]);
      } else {
        interior += obj.mesh.positions[v] != before.positions[v];
      }
    }
    CHECK(interior > 0);
  }
}

TEST_CASE("hidden faces cannot be picked or sculpted") {
  Scene scene;
  Mesh m = makeQuadSphere(24);
  setFaceSets(m, [](const Vec3& c) -> std::int32_t { return c.y > 0.0f ? -2 : 3; });  // Top hidden.
  SceneObject& obj = scene.add("A", std::move(m));
  // A ray from above passes the hidden top and hits the bottom half from inside.
  const auto pick = scene.pick(Ray{{0.0f, 3.0f, 0.0f}, {0.0f, -1.0f, 0.0f}});
  REQUIRE(pick);
  CHECK(pick->worldPosition.y < -0.9f);
  // Drawing over the equator moves only vertices of visible faces.
  const Mesh before = obj.mesh;
  DrawBrush draw;
  InflateBrush inflate;
  for (const Brush* b : {static_cast<const Brush*>(&draw), static_cast<const Brush*>(&inflate)}) {
    REQUIRE(stroke(obj, *b, {1, -0.05f, 0}, 0.6f, {}, 1.0f, 3));
  }
  int moved = 0;
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
    const bool changed = obj.mesh.positions[v] != before.positions[v];
    if (!before.vertexVisible(v)) CHECK_FALSE(changed);
    moved += changed;
  }
  CHECK(moved > 0);
  // Mask operations leave hidden vertices alone too.
  REQUIRE(applyMaskOp(obj, MaskOp::Fill));
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) CHECK(obj.mesh.mask[v] == (obj.mesh.vertexVisible(v) ? 1.0f : 0.0f));
}

TEST_CASE("dynamic topology leaves hidden faces and face set boundaries in place") {
  Scene scene;
  Mesh m = makeQuadSphere(16);
  setFaceSets(m, [](const Vec3& c) -> std::int32_t {
    if (c.y > 0.55f) return -4;  // Hidden cap, partly under the brush.
    return c.x > 0.0f ? 2 : 3;
  });
  SceneObject& obj = scene.add("A", std::move(m));
  const Mesh before = obj.mesh;
  std::vector<Vec3> hiddenFaces, boundary;
  for (Index f = 0; f < before.faceCount(); ++f)
    if (before.faceSets[f] < 0) hiddenFaces.push_back(before.faceCentroid(f));
  for (Index v = 0; v < before.vertexCount(); ++v) {
    std::set<std::int32_t> ids;
    before.forEachOutgoing(v, [&](Index h) { ids.insert(faceSetId(before.faceSets[before.heFace[h]])); });
    if (ids.size() > 1) boundary.push_back(before.positions[v]);
  }
  Sculptor sculptor;
  StrokeOptions opts;
  opts.dyntopo = true;
  opts.dyntopoOptions.timeBudgetMs = 0.0;
  UndoStack stack;
  SmoothBrush smooth;  // Barely moves anything, so positions show what the topology did.
  int splits = 0, collapses = 0;
  for (float detail : {0.03f, 0.4f}) {  // Refine, then coarsen.
    sculptor.beginStroke(obj, smooth, opts, "Dyntopo");
    for (int i = 0; i < 6; ++i)
      sculptor.dab(surfacePoint(obj, {0.1f * i - 0.3f, 0.3f, 1.0f}), 0.6f, 0.0f, {.detail = detail});
    auto undo = sculptor.endStroke();
    REQUIRE(undo);
    splits += sculptor.lastStrokeTopology().splits;
    collapses += sculptor.lastStrokeTopology().collapses;
    stack.push(std::move(*undo));
  }
  CHECK(splits > 100);
  CHECK(collapses > 100);
  test::requireValid(obj.mesh);
  // Every hidden face is still there, unchanged.
  std::vector<Vec3> hiddenNow;
  for (Index f = 0; f < obj.mesh.faceCount(); ++f)
    if (obj.mesh.faceSets[f] < 0) hiddenNow.push_back(obj.mesh.faceCentroid(f));
  auto byXyz = [](const Vec3& a, const Vec3& b) { return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z); };
  std::sort(hiddenFaces.begin(), hiddenFaces.end(), byXyz);
  std::sort(hiddenNow.begin(), hiddenNow.end(), byXyz);
  CHECK(hiddenNow == hiddenFaces);
  // The boundary between sets 2 and 3 stays on the x = 0 plane it started on: refining adds
  // points on the line, coarsening removes only points the line runs through.
  int onLine = 0;
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
    std::set<std::int32_t> ids;
    obj.mesh.forEachOutgoing(v, [&](Index h) { ids.insert(obj.mesh.faceSets[obj.mesh.heFace[h]]); });
    if (ids != std::set<std::int32_t>{2, 3}) continue;
    CHECK(std::abs(obj.mesh.positions[v].x) < 1e-6f);
    ++onLine;
  }
  CHECK(onLine > 20);
  for (Index f = 0; f < obj.mesh.faceCount(); ++f) {
    const Vec3 c = obj.mesh.faceCentroid(f);
    if (obj.mesh.faceSets[f] > 0) CHECK(obj.mesh.faceSets[f] == (c.x > 0.0f ? 2 : 3));
  }
  CHECK_FALSE(stack.undo(scene).empty());
  CHECK_FALSE(stack.undo(scene).empty());
  CHECK(obj.mesh.faceSets == before.faceSets);
  CHECK(obj.mesh.positions == before.positions);
}

// ----- Face set operations -----------------------------------------------------------------------

TEST_CASE("face set from mask, and masking a face set") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(16));
  CHECK_FALSE(applyFaceSetOp(obj, FaceSetOp::FromMask));  // Nothing masked.
  Mesh& m = obj.mesh;
  m.ensureMask();
  for (Index v = 0; v < m.vertexCount(); ++v) m.mask[v] = m.positions[v].y > 0.2f ? 1.0f : 0.0f;
  auto undo = applyFaceSetOp(obj, FaceSetOp::FromMask);
  REQUIRE(undo);
  for (Index f = 0; f < m.faceCount(); ++f) {
    bool all = true;
    m.forEachFaceVertex(f, [&](Index v) { all &= m.positions[v].y > 0.2f; });
    CHECK(m.faceSets[f] == (all ? 2 : 1));
  }
  CHECK_FALSE(obj.faceSetDirtyLeaves.empty());
  CHECK(obj.topoDirtyLeaves.empty());  // Nothing was hidden or shown.

  // Mask the new set from scratch: exactly the vertices its faces use.
  REQUIRE(applyMaskOp(obj, MaskOp::Clear));
  auto masked = maskFaceSet(obj, 2);
  REQUIRE(masked);
  for (Index v = 0; v < m.vertexCount(); ++v) {
    bool inSet = false;
    m.forEachOutgoing(v, [&](Index h) { inSet |= m.faceSets[m.heFace[h]] == 2; });
    CHECK(m.mask[v] == (inSet ? 1.0f : 0.0f));
  }
  CHECK_FALSE(maskFaceSet(obj, 2));  // Already masked.
  CHECK_FALSE(maskFaceSet(obj, 9));  // No such set.
}

TEST_CASE("loose parts get their own sets, and clear puts everything back") {
  Scene scene;
  SceneObject& obj = scene.add("A", twoSpheres());
  REQUIRE(applyFaceSetOp(obj, FaceSetOp::FromLooseParts));
  const Mesh& m = obj.mesh;
  CHECK(idsIn(m) == std::set<std::int32_t>{2, 3});
  std::int32_t leftSet = 0, rightSet = 0;
  for (Index f = 0; f < m.faceCount(); ++f) (m.faceCentroid(f).x < 0.0f ? leftSet : rightSet) = m.faceSets[f];
  for (Index f = 0; f < m.faceCount(); ++f) CHECK(m.faceSets[f] == (m.faceCentroid(f).x < 0.0f ? leftSet : rightSet));
  CHECK(leftSet != rightSet);
  CHECK_FALSE(applyFaceSetOp(obj, FaceSetOp::FromLooseParts));  // Same result: nothing changed.
  REQUIRE(applyFaceSetOp(obj, FaceSetOp::Clear));
  CHECK(idsIn(m) == std::set<std::int32_t>{1});
  CHECK_FALSE(m.hasFaceSetData());
}

TEST_CASE("hide, isolate, reveal and invert visibility") {
  Scene scene;
  Mesh mesh = makeQuadSphere(16);
  setFaceSets(mesh, [](const Vec3& c) -> std::int32_t { return c.y > 0.0f ? 2 : 3; });
  SceneObject& obj = scene.add("A", std::move(mesh));
  const Mesh& m = obj.mesh;
  auto hiddenIs = [&](auto&& want) {
    for (Index f = 0; f < m.faceCount(); ++f)
      if (m.faceHidden(f) != want(faceSetId(m.faceSets[f]))) return false;
    return true;
  };
  UndoStack stack;
  auto run = [&](FaceSetOp op, std::int32_t id = 0) {
    obj.clearDirty();
    auto e = applyFaceSetOp(obj, op, id);
    if (e) stack.push(std::move(*e));
    return e.has_value();
  };
  REQUIRE(run(FaceSetOp::Hide, 2));
  CHECK(hiddenIs([](std::int32_t id) { return id == 2; }));
  CHECK_FALSE(obj.topoDirtyLeaves.empty());  // Leaves that draw fewer triangles now.
  CHECK_FALSE(run(FaceSetOp::Hide, 2));      // Already hidden.
  REQUIRE(run(FaceSetOp::RevealAll));
  CHECK(hiddenIs([](std::int32_t) { return false; }));
  REQUIRE(run(FaceSetOp::Isolate, 3));
  CHECK(hiddenIs([](std::int32_t id) { return id != 3; }));
  REQUIRE(run(FaceSetOp::Isolate, 3));  // Isolating again shows everything.
  CHECK(hiddenIs([](std::int32_t) { return false; }));
  REQUIRE(run(FaceSetOp::Isolate, 3));
  REQUIRE(run(FaceSetOp::InvertVisibility));
  CHECK(hiddenIs([](std::int32_t id) { return id == 3; }));
  CHECK_FALSE(run(FaceSetOp::Hide, 0));
  // Undo walks it all back, marking leaves whose visibility changes.
  REQUIRE(stack.size() == 6);
  for (int i = 0; i < 6; ++i) {
    obj.clearDirty();
    CHECK_FALSE(stack.undo(scene).empty());
    CHECK_FALSE(obj.topoDirtyLeaves.empty());
  }
  CHECK(hiddenIs([](std::int32_t) { return false; }));
  CHECK_FALSE(m.anyHidden());
}

TEST_CASE("a face set undo entry applies even if the face sets were dropped since") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(12));
  FaceSetBrush brush;
  auto undo = stroke(obj, brush, {0, 1, 0}, 0.3f);
  REQUIRE(undo);
  UndoStack stack;
  stack.push(std::move(*undo));
  const auto painted = obj.mesh.faceSets;
  CHECK_FALSE(stack.undo(scene).empty());
  obj.mesh.faceSets.clear();
  CHECK_FALSE(stack.redo(scene).empty());
  CHECK(obj.mesh.faceSets == painted);
}
