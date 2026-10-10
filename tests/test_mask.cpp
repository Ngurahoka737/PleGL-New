#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "TestUtil.h"
#include "io/Project.h"
#include "mesh/MeshEdit.h"
#include "mesh/Primitives.h"
#include "remesh/QuadRemesh.h"
#include "scene/Scene.h"
#include "sculpt/MaskOps.h"
#include "sculpt/Sculptor.h"

using namespace plegl;

namespace {

// A mask value that depends only on the vertex position, so it can be checked after reordering.
float maskOf(const Vec3& p) { return std::clamp(p.y * 0.5f + 0.5f, 0.0f, 1.0f); }

void setMask(Mesh& m, float (*f)(const Vec3&)) {
  m.mask.resize(m.positions.size());
  for (Index v = 0; v < m.vertexCount(); ++v) m.mask[v] = f(m.positions[v]);
}

Vec3 surfacePoint(const SceneObject& obj, Vec3 dir) {
  dir = glm::normalize(dir);
  RayHit hit;
  const bool ok = obj.bvh.raycast(obj.mesh, Ray{dir * 3.0f, -dir}, hit);
  REQUIRE(ok);
  return hit.position;
}

Index nearestVertex(const Mesh& m, const Vec3& p) {
  Index best = 0;
  float bestD = std::numeric_limits<float>::infinity();
  for (Index v = 0; v < m.vertexCount(); ++v) {
    const float d = glm::length(m.positions[v] - p);
    if (d < bestD) bestD = d, best = v;
  }
  return best;
}

std::optional<SculptUndo> maskStroke(SceneObject& obj, const Brush& brush, Vec3 dir, float radius, bool invert = false,
                                     bool symmetry = false, int dabs = 1) {
  Sculptor sculptor;
  StrokeOptions opts;
  opts.invert = invert;
  opts.symmetryX = symmetry;
  sculptor.beginStroke(obj, brush, opts, brush.name());
  const Vec3 c = surfacePoint(obj, dir);
  for (int i = 0; i < dabs; ++i) REQUIRE(sculptor.dab(c, radius, 1.0f));
  auto undo = sculptor.endStroke();
  if (!undo) return std::nullopt;
  return std::get<SculptUndo>(std::move(*undo));
}

void requireMaskInRange(const Mesh& m) {
  for (float v : m.mask) REQUIRE((v >= 0.0f && v <= 1.0f));
}

std::size_t intermediateCount(const Mesh& m) {
  std::size_t n = 0;
  for (float v : m.mask) n += (v > 0.01f && v < 0.99f) ? 1 : 0;
  return n;
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

}  // namespace

// ----- Mesh plumbing -------------------------------------------------------------------------

TEST_CASE("the mask follows its vertices through BVH reordering, duplication and editing") {
  Scene scene;
  Mesh m = makeQuadSphere(12);
  setMask(m, maskOf);
  SceneObject& obj = scene.add("A", std::move(m));  // Builds the BVH, which reorders the mesh.
  REQUIRE(obj.mesh.mask.size() == obj.mesh.positions.size());
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) CHECK(obj.mesh.mask[v] == maskOf(obj.mesh.positions[v]));
  test::requireValid(obj.mesh);

  SceneObject* copy = scene.duplicate(obj.id);
  REQUIRE(copy);
  CHECK(copy->mesh.mask == obj.mesh.mask);

  Mesh e = makeQuadSphere(4);
  setMask(e, maskOf);
  MeshEditor ed(e);
  // Use edges whose ends have clearly different masks, so copying either end would fail.
  Index hs = 0;
  while (hs < e.halfEdgeCount() && e.mask[e.heTarget(hs)] - e.mask[e.heVert[hs]] < 0.05f) ++hs;
  REQUIRE(hs < e.halfEdgeCount());
  const float ma = e.mask[e.heVert[hs]], mb = e.mask[e.heTarget(hs)];
  const Index v = ed.splitEdge(hs, 0.25f);
  CHECK(e.mask[v] == doctest::Approx(ma + (mb - ma) * 0.25f));
  // A collapse keeps the stronger of the two masks on the surviving (start) vertex, which here
  // starts with the weaker one.
  bool collapsed = false;
  for (Index h = 0; h < e.halfEdgeCount() && !collapsed; ++h) {
    const Index start = e.heVert[h], end = e.heTarget(h);
    if (start == v || end == v || e.mask[end] - e.mask[start] < 0.05f) continue;
    const float stronger = e.mask[end];
    if (ed.collapseEdge(h, e.positions[start])) {
      collapsed = true;
      CHECK(e.mask[start] == stronger);
    }
  }
  REQUIRE(collapsed);
  ed.compact();
  REQUIRE(e.mask.size() == e.positions.size());
}

TEST_CASE("compact keeps the mask on the surviving vertices") {
  Mesh m = makeQuadSphere(6);
  setMask(m, maskOf);
  MeshEditor ed(m);
  bool collapsed = false;
  for (Index f = 0; f < m.faceCount() && !collapsed; ++f) collapsed = ed.collapseDiagonal(m.faceHe[f]);
  REQUIRE(collapsed);
  ed.compact();
  test::requireValid(m);
  REQUIRE(m.mask.size() == m.positions.size());
  // Every value still belongs to an original vertex, except the one merged vertex (max of two).
  int mismatches = 0;
  for (Index v = 0; v < m.vertexCount(); ++v) mismatches += m.mask[v] != maskOf(m.positions[v]) ? 1 : 0;
  CHECK(mismatches <= 1);
}

TEST_CASE("validate rejects a mask of the wrong size or out of range") {
  Mesh m = makeQuadSphere(3);
  m.mask.assign(m.positions.size() - 1, 0.0f);
  CHECK_FALSE(validate(m).ok);
  m.mask.assign(m.positions.size(), 0.0f);
  CHECK(validate(m).ok);
  m.mask[3] = std::numeric_limits<float>::quiet_NaN();
  CHECK_FALSE(validate(m).ok);
  m.mask[3] = 1.5f;
  CHECK_FALSE(validate(m).ok);
}

// ----- Brushes respect the mask ---------------------------------------------------------------

TEST_CASE("fully masked vertices never move, for every brush") {
  DrawBrush draw;
  ClayBrush clay;
  SmoothBrush smooth;
  InflateBrush inflate;
  FlattenBrush flatten;
  CreaseBrush crease;
  for (const Brush* brush : std::initializer_list<const Brush*>{&draw, &clay, &smooth, &inflate, &flatten, &crease}) {
    INFO(brush->name());
    Scene scene;
    SceneObject& obj = scene.add("A", makeQuadSphere(48));
    setMask(obj.mesh, [](const Vec3& p) { return p.x > 0.0f ? 1.0f : 0.0f; });
    const Mesh before = obj.mesh;
    Sculptor sculptor;
    sculptor.beginStroke(obj, *brush, {}, brush->name());
    const Vec3 c = surfacePoint(obj, {0, 1, 0});
    for (int i = 0; i < 6; ++i) REQUIRE(sculptor.dab(c, 0.4f, 1.0f));
    REQUIRE(sculptor.endStroke());
    int movedFree = 0;
    for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
      if (before.mask[v] == 1.0f) {
        REQUIRE(obj.mesh.positions[v] == before.positions[v]);  // Bit exact.
      } else if (obj.mesh.positions[v] != before.positions[v]) {
        ++movedFree;
      }
    }
    CHECK(movedFree > 20);
    CHECK(obj.mesh.mask == before.mask);
  }
}

TEST_CASE("a stroke over a fully masked area records nothing") {
  DrawBrush draw;
  ClayBrush clay;
  SmoothBrush smooth;
  InflateBrush inflate;
  FlattenBrush flatten;
  CreaseBrush crease;
  for (const Brush* brush : std::initializer_list<const Brush*>{&draw, &clay, &smooth, &inflate, &flatten, &crease}) {
    INFO(brush->name());
    Scene scene;
    SceneObject& obj = scene.add("A", makeQuadSphere(32));
    obj.mesh.mask.assign(obj.mesh.positions.size(), 1.0f);
    const Mesh before = obj.mesh;
    Sculptor sculptor;
    sculptor.beginStroke(obj, *brush, {}, brush->name());
    const Vec3 c = surfacePoint(obj, {0, 1, 0});
    for (int i = 0; i < 4; ++i) sculptor.dab(c, 0.3f, 1.0f);
    CHECK_FALSE(sculptor.endStroke());  // No undo entry, so the redo history survives.
    CHECK(obj.mesh.positions == before.positions);
    CHECK(obj.mesh.normals == before.normals);
  }
}

TEST_CASE("a half mask halves the draw displacement") {
  Scene scene;
  SceneObject& free = scene.add("Free", makeQuadSphere(48));
  SceneObject& half = scene.add("Half", makeQuadSphere(48));
  half.mesh.mask.assign(half.mesh.positions.size(), 0.5f);
  const Mesh before = free.mesh;
  DrawBrush draw;
  for (SceneObject* obj : {&free, &half}) {
    Sculptor sculptor;
    sculptor.beginStroke(*obj, draw, {}, "Draw");
    REQUIRE(sculptor.dab(surfacePoint(*obj, {0, 1, 0}), 0.3f, 1.0f));
    REQUIRE(sculptor.endStroke());
  }
  int checked = 0;
  for (Index v = 0; v < before.vertexCount(); ++v) {
    const float dFree = glm::length(free.mesh.positions[v] - before.positions[v]);
    const float dHalf = glm::length(half.mesh.positions[v] - before.positions[v]);
    if (dFree < 1e-4f) continue;
    CHECK(dHalf == doctest::Approx(0.5f * dFree).epsilon(1e-3));
    ++checked;
  }
  CHECK(checked > 20);
}

TEST_CASE("grab leaves masked vertices in place") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(48));
  setMask(obj.mesh, [](const Vec3& p) { return p.x > 0.0f ? 1.0f : 0.0f; });
  const Mesh before = obj.mesh;
  Sculptor sculptor;
  StrokeOptions opts;
  opts.strength = 1.0f;
  REQUIRE(sculptor.beginGrab(obj, opts, surfacePoint(obj, {0, 1, 0}), 0.5f, "Grab"));
  sculptor.grab({0.0f, 0.3f, 0.0f});
  auto entry = sculptor.endStroke();
  REQUIRE(entry);
  int moved = 0;
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
    if (before.mask[v] == 1.0f) REQUIRE(obj.mesh.positions[v] == before.positions[v]);
    else moved += obj.mesh.positions[v] != before.positions[v] ? 1 : 0;
  }
  CHECK(moved > 20);
}

// ----- Mask brush ------------------------------------------------------------------------------

TEST_CASE("the mask brush paints inside the dab only and never moves the surface") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(48));
  REQUIRE(obj.mesh.mask.empty());
  const Mesh before = obj.mesh;
  MaskBrush mask;
  const Vec3 c = surfacePoint(obj, {0, 1, 0});
  auto entry = maskStroke(obj, mask, {0, 1, 0}, 0.3f);
  REQUIRE(entry);
  REQUIRE(obj.mesh.mask.size() == obj.mesh.positions.size());
  requireMaskInRange(obj.mesh);
  CHECK(obj.mesh.positions == before.positions);
  CHECK(obj.mesh.normals == before.normals);
  int painted = 0;
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v) {
    const float d = glm::length(obj.mesh.positions[v] - c);
    if (d >= 0.3f) REQUIRE(obj.mesh.mask[v] == 0.0f);
    painted += obj.mesh.mask[v] > 0.0f ? 1 : 0;
  }
  CHECK(painted > 20);
  CHECK(obj.mesh.mask[nearestVertex(obj.mesh, c)] > 0.95f);
  // Only mask ranges were marked for upload, and undo recorded the mask channel only.
  CHECK(obj.dirtyLeaves.empty());
  CHECK_FALSE(obj.maskDirtyLeaves.empty());
  for (const LeafState& s : entry->before) {
    CHECK(s.positions.empty());
    CHECK(s.normals.empty());
    CHECK_FALSE(s.mask.empty());
  }
  CHECK(entry->before.size() < obj.bvh.leaves().size());
}

TEST_CASE("inverted mask strokes erase, and repainting a full mask records nothing") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(48));
  MaskBrush mask;
  REQUIRE(maskStroke(obj, mask, {0, 1, 0}, 0.3f, false, false, 8));
  const Vec3 c = surfacePoint(obj, {0, 1, 0});
  const Index centre = nearestVertex(obj.mesh, c);
  CHECK(obj.mesh.mask[centre] == 1.0f);
  // Painting the same spot again changes nothing, so there is nothing to undo.
  const std::vector<float> full = obj.mesh.mask;
  auto again = maskStroke(obj, mask, {0, 1, 0}, 0.05f);
  if (again) CHECK(obj.mesh.mask != full);
  else CHECK(obj.mesh.mask == full);
  REQUIRE(maskStroke(obj, mask, {0, 1, 0}, 0.3f, true, false, 8));
  CHECK(obj.mesh.mask[centre] == 0.0f);
  requireMaskInRange(obj.mesh);
}

TEST_CASE("mask strokes mirror across X with symmetry") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(48));
  MaskBrush mask;
  REQUIRE(maskStroke(obj, mask, {0.6f, 0.8f, 0.0f}, 0.25f, false, true, 4));
  const Vec3 c = surfacePoint(obj, {0.6f, 0.8f, 0.0f});
  CHECK(obj.mesh.mask[nearestVertex(obj.mesh, c)] > 0.95f);
  CHECK(obj.mesh.mask[nearestVertex(obj.mesh, {-c.x, c.y, c.z})] > 0.95f);
  CHECK(obj.mesh.mask[nearestVertex(obj.mesh, {c.x, -c.y, c.z})] == 0.0f);
}

TEST_CASE("undo and redo restore a mask stroke exactly") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(48));
  MaskBrush mask;
  REQUIRE(maskStroke(obj, mask, {1, 0, 0}, 0.4f, false, false, 3));
  const std::vector<float> first = obj.mesh.mask;
  const Mesh geometry = obj.mesh;
  auto entry = maskStroke(obj, mask, {0, 1, 0}, 0.4f, false, false, 3);
  REQUIRE(entry);
  const std::vector<float> second = obj.mesh.mask;
  UndoStack stack;
  stack.push(std::move(*entry));
  obj.clearDirty();
  CHECK(stack.undo(scene) == "Mask");
  CHECK(obj.mesh.mask == first);
  CHECK(obj.mesh.positions == geometry.positions);
  CHECK(obj.dirtyLeaves.empty());  // No geometry upload for a mask-only entry.
  CHECK(stack.redo(scene) == "Mask");
  CHECK(obj.mesh.mask == second);
}

TEST_CASE("a mask undo entry applies even if the mask was dropped since") {
  // For example: remesh (mask empty), paint, undo, undo the remesh, redo the remesh (restores a
  // mesh without a mask), redo the paint.
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(32));
  MaskBrush mask;
  auto entry = maskStroke(obj, mask, {0, 1, 0}, 0.4f);
  REQUIRE(entry);
  const std::vector<float> painted = obj.mesh.mask;
  UndoStack stack;
  stack.push(std::move(*entry));
  REQUIRE(stack.undo(scene) == "Mask");
  obj.mesh.mask.clear();
  REQUIRE(stack.redo(scene) == "Mask");
  CHECK(obj.mesh.mask == painted);
  test::requireValid(obj.mesh);
}

TEST_CASE("smooth mask softens a hard mask edge") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(48));
  setMask(obj.mesh, [](const Vec3& p) { return p.y > 0.0f ? 1.0f : 0.0f; });
  const std::vector<float> step = obj.mesh.mask;
  MaskSmoothBrush smooth;
  REQUIRE(maskStroke(obj, smooth, {1, 0, 0}, 0.3f, false, false, 4));
  requireMaskInRange(obj.mesh);
  CHECK(intermediateCount(obj.mesh) > 10);
  const Vec3 c = surfacePoint(obj, {1, 0, 0});
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v)
    if (glm::length(obj.mesh.positions[v] - c) >= 0.3f) REQUIRE(obj.mesh.mask[v] == step[v]);
}

// ----- Whole-mesh mask operations -------------------------------------------------------------

TEST_CASE("clear and fill do nothing when there is nothing to change") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(16));
  CHECK_FALSE(applyMaskOp(obj, MaskOp::Clear));
  CHECK(obj.mesh.mask.empty());  // No allocation either.
  REQUIRE(applyMaskOp(obj, MaskOp::Fill));
  CHECK_FALSE(applyMaskOp(obj, MaskOp::Fill));
  CHECK_FALSE(applyMaskOp(obj, MaskOp::Blur));  // A constant mask stays constant.
}

TEST_CASE("invert masks everything and undo restores it exactly") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(48));
  auto entry = applyMaskOp(obj, MaskOp::Invert);
  REQUIRE(entry);
  CHECK(entry->label == "Invert Mask");
  CHECK(entry->before.size() == obj.bvh.leaves().size());
  CHECK((obj.maskDirtyAll || obj.maskDirtyLeaves.size() == obj.bvh.leaves().size()));
  for (float v : obj.mesh.mask) REQUIRE(v == 1.0f);
  UndoStack stack;
  stack.push(std::move(*entry));
  stack.undo(scene);
  for (float v : obj.mesh.mask) REQUIRE(v == 0.0f);
  stack.redo(scene);
  for (float v : obj.mesh.mask) REQUIRE(v == 1.0f);

  // Inverting a 0/1 mask swaps the regions.
  setMask(obj.mesh, [](const Vec3& p) { return p.y > 0.0f ? 1.0f : 0.0f; });
  REQUIRE(applyMaskOp(obj, MaskOp::Invert));
  for (Index v = 0; v < obj.mesh.vertexCount(); ++v)
    REQUIRE(obj.mesh.mask[v] == (obj.mesh.positions[v].y > 0.0f ? 0.0f : 1.0f));
}

TEST_CASE("a mask operation over many leaves asks for one whole upload") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(128));
  REQUIRE(obj.bvh.leaves().size() > kMaskDirtyAllLeaves);
  REQUIRE(applyMaskOp(obj, MaskOp::Fill));
  CHECK(obj.maskDirtyAll);
  CHECK(obj.maskDirtyLeaves.empty());
}

TEST_CASE("clear records only the leaves that had a mask") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(48));
  MaskBrush mask;
  REQUIRE(maskStroke(obj, mask, {0, 1, 0}, 0.2f));
  const std::vector<float> painted = obj.mesh.mask;
  std::size_t maskedLeaves = 0;
  for (const BvhLeaf& l : obj.bvh.leaves()) {
    bool any = false;
    for (Index v = l.vertBegin; v < l.vertEnd; ++v) any |= painted[v] > 0.0f;
    maskedLeaves += any ? 1 : 0;
  }
  obj.clearDirty();
  auto entry = applyMaskOp(obj, MaskOp::Clear);
  REQUIRE(entry);
  CHECK(entry->before.size() == maskedLeaves);
  CHECK(obj.maskDirtyLeaves.size() == maskedLeaves);
  CHECK_FALSE(obj.mesh.anyMasked());
  UndoStack stack;
  stack.push(std::move(*entry));
  stack.undo(scene);
  CHECK(obj.mesh.mask == painted);
}

TEST_CASE("blur softens and sharpen hardens mask edges") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(48));
  setMask(obj.mesh, [](const Vec3& p) { return p.y > 0.0f ? 1.0f : 0.0f; });
  REQUIRE(applyMaskOp(obj, MaskOp::Blur, 3));
  requireMaskInRange(obj.mesh);
  const std::size_t soft = intermediateCount(obj.mesh);
  CHECK(soft > 50);
  // Far from the edge the mask is untouched.
  CHECK(obj.mesh.mask[nearestVertex(obj.mesh, {0, 1, 0})] == 1.0f);
  CHECK(obj.mesh.mask[nearestVertex(obj.mesh, {0, -1, 0})] == 0.0f);
  REQUIRE(applyMaskOp(obj, MaskOp::Sharpen, 3));
  requireMaskInRange(obj.mesh);
  CHECK(intermediateCount(obj.mesh) < soft);
}

TEST_CASE("mask filters reach the open border of a mesh") {
  // Everything but the border is masked. The border must pick up the mask from its inner
  // neighbours instead of averaging only along the border, which would keep it at zero.
  auto insideOnly = [](const Vec3& p) { return std::max(std::abs(p.x), std::abs(p.z)) > 0.99f ? 0.0f : 1.0f; };
  const Vec3 edgeMiddle{1.0f, 0.0f, 0.0f};
  {
    Scene scene;
    SceneObject& obj = scene.add("Plane", makePlane(16, 2.0f));
    setMask(obj.mesh, insideOnly);
    REQUIRE(applyMaskOp(obj, MaskOp::Blur, 1));
    // Three neighbours (two on the border, one inside), halfway towards their mean: 1/6.
    CHECK(obj.mesh.mask[nearestVertex(obj.mesh, edgeMiddle)] == doctest::Approx(1.0f / 6.0f));
  }
  {
    Scene scene;
    SceneObject& obj = scene.add("Plane", makePlane(16, 2.0f));
    setMask(obj.mesh, insideOnly);
    MaskSmoothBrush smoothMask;
    Sculptor sculptor;
    sculptor.beginStroke(obj, smoothMask, {}, smoothMask.name());
    REQUIRE(sculptor.dab(edgeMiddle, 0.3f, 1.0f));
    REQUIRE(sculptor.endStroke());
    CHECK(obj.mesh.mask[nearestVertex(obj.mesh, edgeMiddle)] > 0.2f);
  }
}

// ----- Project files ---------------------------------------------------------------------------

TEST_CASE("projects keep masks, and unmasked projects have no mask chunk") {
  Scene scene;
  SceneObject& head = scene.add("Head", makeQuadSphere(16));
  setMask(head.mesh, maskOf);
  scene.add("Eye", makeIcosphere(2));
  SceneObject& zero = scene.add("Zero", makeQuadSphere(4));
  zero.mesh.ensureMask();  // Allocated but all zero: nothing to store.

  const auto bytes = serializeProject(scene, "");
  REQUIRE(findChunk(bytes, "MASK") != std::string::npos);
  std::string error;
  auto project = parseProject(bytes.data(), bytes.size(), &error);
  INFO(error);
  REQUIRE(project);
  CHECK(project->objects[0].mesh.mask == head.mesh.mask);  // Bit exact, same vertex order.
  CHECK(project->objects[1].mesh.mask.empty());
  CHECK(project->objects[2].mesh.mask.empty());

  Scene plain;
  plain.add("Head", makeQuadSphere(16));
  const auto plainBytes = serializeProject(plain, "");
  CHECK(findChunk(plainBytes, "MASK") == std::string::npos);
}

TEST_CASE("a project without its mask chunk still opens, as in builds without masking") {
  Scene scene;
  SceneObject& head = scene.add("Head", makeQuadSphere(16));
  setMask(head.mesh, maskOf);
  auto bytes = serializeProject(scene, "k v\n");
  const std::size_t at = findChunk(bytes, "MASK");
  REQUIRE(at != std::string::npos);
  std::uint64_t size = 0;
  std::memcpy(&size, bytes.data() + at + 4, 8);
  bytes.erase(bytes.begin() + at, bytes.begin() + at + 12 + std::size_t(size));
  fixChecksum(bytes);
  auto project = parseProject(bytes.data(), bytes.size());
  REQUIRE(project);
  CHECK(project->objects[0].mesh.mask.empty());
  CHECK(project->settings == "k v\n");
}

TEST_CASE("damaged mask chunks are refused and odd values are sanitized") {
  Scene scene;
  SceneObject& head = scene.add("Head", makeQuadSphere(8));
  setMask(head.mesh, maskOf);
  const auto good = serializeProject(scene, "");
  const std::size_t at = findChunk(good, "MASK");
  REQUIRE(at != std::string::npos);
  const std::size_t payload = at + 12;  // u32 count, then u32 index, u32 vertex count, u8 encoding.
  std::string error;

  SUBCASE("wrong vertex count") {
    auto b = good;
    std::uint32_t n = 0;
    std::memcpy(&n, b.data() + payload + 8, 4);
    ++n;
    std::memcpy(b.data() + payload + 8, &n, 4);
    fixChecksum(b);
    CHECK_FALSE(parseProject(b.data(), b.size(), &error));
    CHECK(error.find("mask") != std::string::npos);
  }
  SUBCASE("object index out of range") {
    auto b = good;
    const std::uint32_t index = 7;
    std::memcpy(b.data() + payload + 4, &index, 4);
    fixChecksum(b);
    CHECK_FALSE(parseProject(b.data(), b.size(), &error));
  }
  SUBCASE("unknown encoding") {
    auto b = good;
    b[payload + 12] = 9;
    fixChecksum(b);
    CHECK_FALSE(parseProject(b.data(), b.size(), &error));
  }
  SUBCASE("NaN, negative and too large values are clamped") {
    auto b = good;
    const std::size_t values = payload + 13;
    const float odd[3] = {std::numeric_limits<float>::quiet_NaN(), -1.0f, 2.0f};
    std::memcpy(b.data() + values, odd, sizeof(odd));
    fixChecksum(b);
    auto project = parseProject(b.data(), b.size(), &error);
    INFO(error);
    REQUIRE(project);
    const auto& m = project->objects[0].mesh.mask;
    CHECK(m[0] == 0.0f);
    CHECK(m[1] == 0.0f);
    CHECK(m[2] == 1.0f);
    test::requireValid(project->objects[0].mesh);
  }
}

// ----- Remesh ------------------------------------------------------------------------------------

TEST_CASE("remeshing carries the mask over to the new surface") {
  Mesh in = makeQuadSphere(32);
  setMask(in, [](const Vec3& p) { return p.y > 0.0f ? 1.0f : 0.0f; });
  for (int rounds : {3, 0}) {
    INFO("rounds " << rounds);
    auto out = quadRemesh(in, {.targetEdge = 0.06f, .rounds = rounds});
    REQUIRE(out);
    REQUIRE(out->mask.size() == out->positions.size());
    test::requireValid(*out);
    int checked = 0;
    for (Index v = 0; v < out->vertexCount(); ++v) {
      const float y = out->positions[v].y;
      if (y > 0.15f) {
        CHECK(out->mask[v] > 0.99f);
        ++checked;
      } else if (y < -0.15f) {
        CHECK(out->mask[v] < 0.01f);
        ++checked;
      }
    }
    CHECK(checked > 100);
  }
  // An unmasked input gives an unmasked result.
  auto plain = quadRemesh(makeQuadSphere(16), {.targetEdge = 0.1f});
  REQUIRE(plain);
  CHECK(plain->mask.empty());
}

TEST_CASE("undo memory counts masks") {
  MeshState state;
  state.mesh = makeQuadSphere(8);
  const std::size_t plain = state.bytes();
  state.mesh.ensureMask();
  CHECK(state.bytes() == plain + state.mesh.mask.size() * sizeof(float));
  SculptUndo e;
  e.before.push_back({0, {}, {}, std::vector<float>(10), {}, {}});
  CHECK(e.bytes() == 10 * sizeof(float));
}
