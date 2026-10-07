#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <tuple>

#include "TestUtil.h"
#include "io/Obj.h"
#include "io/Project.h"
#include "mesh/Primitives.h"
#include "remesh/QuadRemesh.h"
#include "scene/Scene.h"
#include "sculpt/Sculptor.h"
#include "sculpt/Undo.h"
#include "spatial/LeafLayout.h"

using namespace plegl;

namespace {

constexpr float kPi = 3.14159265358979f;

StrokeOptions dyntopoStroke(DyntopoRefine refine = DyntopoRefine::SplitCollapse) {
  StrokeOptions o;
  o.dyntopo = true;
  o.dyntopoOptions.refine = refine;
  o.dyntopoOptions.timeBudgetMs = 0.0;  // Deterministic: only the per-pass counts limit the work.
  return o;
}

void requireDynamic(const SceneObject& obj) {
  const ValidationResult r = validateDynamic(obj.mesh, obj.bvh);
  INFO(r.message);
  REQUIRE(r.ok);
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

// A compact mesh again, with a canonical BVH layout and fresh normals.
void requireCompact(const SceneObject& obj) {
  test::requireValid(obj.mesh);
  const ValidationResult r = validateLayout(obj.mesh, obj.bvh);
  INFO(r.message);
  REQUIRE(r.ok);
  CHECK_FALSE(obj.bvh.dynamic());
  requireNormalsMatchFullRecompute(obj.mesh);
}

// Everything undo must bring back, bit for bit.
struct State {
  Mesh mesh;
  std::vector<BvhLeaf> leaves;
  std::uint64_t version = 0;
};

State capture(const SceneObject& obj) {
  return {obj.mesh, {obj.bvh.leaves().begin(), obj.bvh.leaves().end()}, obj.topologyVersion};
}

bool sameLeaves(std::span<const BvhLeaf> a, std::span<const BvhLeaf> b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const BvhLeaf &x = a[i], &y = b[i];
    if (x.faceBegin != y.faceBegin || x.faceEnd != y.faceEnd || x.vertBegin != y.vertBegin ||
        x.vertEnd != y.vertEnd || x.heBegin != y.heBegin || x.heEnd != y.heEnd || x.flags != y.flags ||
        x.bounds.min != y.bounds.min || x.bounds.max != y.bounds.max)
      return false;
  }
  return true;
}

void requireState(const SceneObject& obj, const State& s) {
  const Mesh &a = obj.mesh, &b = s.mesh;
  CHECK(obj.topologyVersion == s.version);
  CHECK(a.positions == b.positions);
  CHECK(a.normals == b.normals);
  CHECK(a.mask == b.mask);
  CHECK(a.vertHe == b.vertHe);
  CHECK(a.faceHe == b.faceHe);
  CHECK(a.heNext == b.heNext);
  CHECK(a.heTwin == b.heTwin);
  CHECK(a.heVert == b.heVert);
  CHECK(a.heFace == b.heFace);
  CHECK(sameLeaves(obj.bvh.leaves(), s.leaves));
}

// Nearest surface point seen from outside along `dir` (meshes here surround the origin).
Vec3 surfacePoint(const SceneObject& obj, Vec3 dir) {
  dir = glm::normalize(dir);
  RayHit hit;
  const bool ok = obj.bvh.raycast(obj.mesh, Ray{dir * 4.0f, -dir}, hit);
  REQUIRE(ok);
  return hit.position;
}

// Surface points along the great arc from one direction to another.
std::vector<Vec3> arc(const SceneObject& obj, Vec3 from, Vec3 to, int n) {
  from = glm::normalize(from);
  to = glm::normalize(to);
  std::vector<Vec3> out;
  for (int i = 0; i < n; ++i) {
    const float t = n > 1 ? static_cast<float>(i) / static_cast<float>(n - 1) : 0.0f;
    out.push_back(surfacePoint(obj, glm::normalize(from * (1.0f - t) + to * t)));
  }
  return out;
}

// The face the app would pass as the hint: the one under the cursor.
Index faceNear(const SceneObject& obj, const Vec3& p, float radius) {
  Bvh::ClosestHit hit;
  return obj.bvh.closestPoint(obj.mesh, p, radius, hit) ? hit.face : kInvalid;
}

// Runs one dyntopo stroke through the dabs and checks the in-stroke layout after each one.
std::optional<StrokeUndo> stroke(Sculptor& sculptor, SceneObject& obj, const Brush& brush, const StrokeOptions& options,
                                 std::span<const Vec3> dabs, float radius, float strength, float detail,
                                 bool checkEachDab = true) {
  sculptor.beginStroke(obj, brush, options, brush.name());
  for (const Vec3& c : dabs) {
    sculptor.dab(c, radius, strength, DabTopology{detail, faceNear(obj, c, radius)});
    if (checkEachDab && obj.bvh.dynamic()) requireDynamic(obj);
  }
  auto undo = sculptor.endStroke();
  CHECK_FALSE(sculptor.lastStrokeTopology().lostUndo);
  CHECK_FALSE(sculptor.lastStrokeTopology().faulted);
  requireCompact(obj);
  return undo;
}

float segmentDist(const Vec3& p, const Vec3& a, const Vec3& b) {
  const Vec3 ab = b - a;
  const float len2 = glm::dot(ab, ab);
  const float t = len2 > 0.0f ? std::clamp(glm::dot(p - a, ab) / len2, 0.0f, 1.0f) : 0.0f;
  return glm::length(a + ab * t - p);
}

// Calls fn(a, b) with the end positions of every edge once.
template <class Fn>
void forEachEdge(const Mesh& m, Fn&& fn) {
  for (Index h = 0; h < m.halfEdgeCount(); ++h) {
    const Index t = m.heTwin[h];
    if (t != kInvalid && t < h) continue;
    fn(m.positions[m.heVert[h]], m.positions[m.heTarget(h)], t == kInvalid);
  }
}

float maxEdge(const Mesh& m) {
  float out = 0.0f;
  forEachEdge(m, [&](const Vec3& a, const Vec3& b, bool) { out = std::max(out, glm::length(a - b)); });
  return out;
}

int valence(const Mesh& m, Index v) {
  int n = 0;
  m.forEachOutgoing(v, [&](Index) { ++n; });
  return n;
}

int faceSize(const Mesh& m, Index f) {
  int n = 0;
  m.forEachFaceVertex(f, [&](Index) { ++n; });
  return n;
}

// Smallest corner angle of any triangle, in degrees. Repeated splits next to a long edge make
// slivers whose angles shrink toward zero.
float minTriangleAngle(const Mesh& m) {
  float out = 180.0f;
  for (Index f = 0; f < m.faceCount(); ++f) {
    std::vector<Vec3> c;
    m.forEachFaceVertex(f, [&](Index v) { c.push_back(m.positions[v]); });
    if (c.size() != 3) continue;
    for (int i = 0; i < 3; ++i) {
      const Vec3 u = glm::normalize(c[(i + 1) % 3] - c[i]), w = glm::normalize(c[(i + 2) % 3] - c[i]);
      out = std::min(out, std::acos(std::clamp(glm::dot(u, w), -1.0f, 1.0f)) * 180.0f / kPi);
    }
  }
  return out;
}

int duplicatePositions(const Mesh& m) {
  std::set<std::tuple<float, float, float>> seen;
  int dup = 0;
  for (const Vec3& p : m.positions) dup += !seen.insert({p.x, p.y, p.z}).second;
  return dup;
}

// A face by its corner positions in order, starting at the smallest: survives renumbering.
using FaceKey = std::vector<float>;

FaceKey faceKey(const Mesh& m, Index f) {
  std::vector<Vec3> c;
  m.forEachFaceVertex(f, [&](Index v) { c.push_back(m.positions[v]); });
  const auto less = [](const Vec3& a, const Vec3& b) { return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z); };
  const std::size_t first = static_cast<std::size_t>(std::min_element(c.begin(), c.end(), less) - c.begin());
  FaceKey k;
  for (std::size_t i = 0; i < c.size(); ++i) {
    const Vec3& p = c[(first + i) % c.size()];
    k.insert(k.end(), {p.x, p.y, p.z});
  }
  return k;
}

std::set<FaceKey> faceKeys(const Mesh& m) {
  std::set<FaceKey> out;
  for (Index f = 0; f < m.faceCount(); ++f) out.insert(faceKey(m, f));
  return out;
}

// Faces of `before` whose corners all satisfy `keep` and that are missing from `after`.
template <class Pred>
int lostFaces(const Mesh& before, const Mesh& after, Pred&& keep) {
  const std::set<FaceKey> now = faceKeys(after);
  int lost = 0;
  for (Index f = 0; f < before.faceCount(); ++f) {
    bool all = true;
    before.forEachFaceVertex(f, [&](Index v) { all &= keep(v); });
    if (all && !now.count(faceKey(before, f))) ++lost;
  }
  return lost;
}

// Builds a mesh from polygons, flipping any face that points toward the origin.
Mesh buildOutward(const std::vector<Vec3>& p, std::vector<std::vector<Index>> faces) {
  std::vector<Index> idx, sizes;
  for (auto& f : faces) {
    Vec3 n{0.0f}, c{0.0f};
    for (std::size_t i = 0; i < f.size(); ++i) {
      n += glm::cross(p[f[i]], p[f[(i + 1) % f.size()]]);
      c += p[f[i]];
    }
    if (glm::dot(n, c) < 0.0f) std::reverse(f.begin(), f.end());
    idx.insert(idx.end(), f.begin(), f.end());
    sizes.push_back(static_cast<Index>(f.size()));
  }
  return buildMesh(p, idx, sizes);
}

Mesh makeTetrahedron() {
  const std::vector<Vec3> p{{0.7f, 0.7f, 0.7f}, {0.7f, -0.7f, -0.7f}, {-0.7f, 0.7f, -0.7f}, {-0.7f, -0.7f, 0.7f}};
  return buildOutward(p, {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}});
}

// A closed prism with n-gon caps and quad sides: faces with more than 4 corners must stay as
// they are.
Mesh makeNgonPrism(int n, int rings) {
  std::vector<Vec3> p;
  std::vector<std::vector<Index>> faces;
  for (int r = 0; r <= rings; ++r) {
    for (int i = 0; i < n; ++i) {
      const float a = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(n);
      p.push_back({std::cos(a), -1.0f + 2.0f * static_cast<float>(r) / static_cast<float>(rings), std::sin(a)});
    }
  }
  for (int r = 0; r < rings; ++r)
    for (int i = 0; i < n; ++i)
      faces.push_back({r * n + i, r * n + (i + 1) % n, (r + 1) * n + (i + 1) % n, (r + 1) * n + i});
  std::vector<Index> bottom, top;
  for (int i = 0; i < n; ++i) {
    bottom.push_back(i);
    top.push_back(rings * n + i);
  }
  faces.push_back(bottom);
  faces.push_back(top);
  return buildOutward(p, std::move(faces));
}

// Two UV spheres touching at one shared pole: a vertex whose faces form two fans.
Mesh makePinchedSpheres(BuildReport& report) {
  std::vector<Vec3> p;
  std::vector<Index> idx, sizes;
  for (float dy : {0.5f, -0.5f}) {
    const Mesh s = makeUvSphere(12, 8, 0.5f);
    const Index base = static_cast<Index>(p.size());
    for (const Vec3& q : s.positions) p.push_back(q + Vec3{0.0f, dy, 0.0f});
    for (Index f = 0; f < s.faceCount(); ++f) {
      Index n = 0;
      s.forEachFaceVertex(f, [&](Index v) {
        idx.push_back(base + v);
        ++n;
      });
      sizes.push_back(n);
    }
  }
  weldVertices(p, idx, 1e-6f);
  return buildMesh(std::move(p), idx, sizes, &report);
}

}  // namespace

TEST_CASE("dynamic topology keeps every kind of mesh valid after every dab and undoes exactly") {
  struct Case {
    const char* name;
    Mesh mesh;
  };
  std::vector<Case> cases;
  cases.push_back({"quad sphere", makeQuadSphere(12)});
  cases.push_back({"icosphere", makeIcosphere(3)});
  cases.push_back({"uv sphere", makeUvSphere(32, 16)});
  cases.push_back({"cube", makeCube(3)});
  cases.push_back({"plane", makePlane(8)});
  cases.push_back({"n-gon prism", makeNgonPrism(8, 6)});
  cases.push_back({"tetrahedron", makeTetrahedron()});
  DrawBrush draw;
  SmoothBrush smooth;
  for (Case& c : cases) {
    INFO(c.name);
    test::requireValid(c.mesh);
    Scene scene;
    SceneObject& obj = scene.add(c.name, std::move(c.mesh));
    const State start = capture(obj);
    const bool flat = std::string(c.name) == "plane";
    // Across the top, so every mesh gets both fine and coarse dabs.
    std::vector<Vec3> path;
    if (flat) {
      for (int i = 0; i < 10; ++i) path.push_back({-0.9f + 0.2f * static_cast<float>(i), 0.0f, 0.3f});
    } else {
      path = arc(obj, {-0.6f, 1.0f, 0.2f}, {0.6f, 1.0f, -0.1f}, 10);
    }
    UndoStack stack;
    Sculptor sculptor;
    auto fine = stroke(sculptor, obj, draw, dyntopoStroke(), path, 0.45f, 0.3f, 0.06f);
    REQUIRE(fine);
    CHECK(std::holds_alternative<DyntopoUndo>(*fine));
    CHECK(sculptor.lastStrokeTopology().splits > 0);
    stack.push(std::move(*fine));
    const State afterFine = capture(obj);
    auto coarse = stroke(sculptor, obj, smooth, dyntopoStroke(), path, 0.45f, 0.5f, 0.4f);
    REQUIRE(coarse);
    stack.push(std::move(*coarse));
    const State afterCoarse = capture(obj);

    for (int round = 0; round < 2; ++round) {
      CHECK_FALSE(stack.undo(scene).empty());
      requireState(obj, afterFine);
      CHECK_FALSE(stack.undo(scene).empty());
      requireState(obj, start);
      CHECK_FALSE(stack.redo(scene).empty());
      requireState(obj, afterFine);
      CHECK_FALSE(stack.redo(scene).empty());
      requireState(obj, afterCoarse);
    }
    requireCompact(obj);
  }
}

TEST_CASE("faces with more than four corners are never touched") {
  Scene scene;
  SceneObject& obj = scene.add("Prism", makeNgonPrism(8, 4));
  const Mesh before = obj.mesh;
  DrawBrush draw;
  Sculptor sculptor;
  // On the cap, then down the side next to it.
  const std::vector<Vec3> dabs{{0.0f, 1.0f, 0.0f}, {0.5f, 1.0f, 0.0f}, {0.95f, 0.8f, 0.0f}, {0.95f, 0.5f, 0.1f}};
  REQUIRE(stroke(sculptor, obj, draw, dyntopoStroke(), dabs, 0.4f, 0.0f, 0.08f));
  CHECK(obj.mesh.faceCount() > before.faceCount());
  int ngons = 0;
  for (Index f = 0; f < obj.mesh.faceCount(); ++f) ngons += faceSize(obj.mesh, f) > 4;
  CHECK(ngons == 2);
  CHECK(duplicatePositions(obj.mesh) == 0);
  CHECK(minTriangleAngle(obj.mesh) > 10.0f);
  // Both caps keep their exact corners.
  std::set<FaceKey> after = faceKeys(obj.mesh);
  for (Index f = 0; f < before.faceCount(); ++f)
    if (faceSize(before, f) > 4) CHECK(after.count(faceKey(before, f)) == 1);
}

TEST_CASE("refinement brings every edge the brush reaches under the detail size and nothing else") {
  Scene scene;
  SceneObject& obj = scene.add("Ico", makeIcosphere(3));
  const Mesh before = obj.mesh;
  const float edge = maxEdge(before);
  DrawBrush draw;
  Sculptor sculptor;
  const Vec3 c = surfacePoint(obj, {0.2f, 1.0f, 0.1f});
  const float r = 0.3f, detail = 0.05f;
  StrokeOptions options = dyntopoStroke(DyntopoRefine::SplitOnly);
  options.dyntopoOptions.maxSplits = 1 << 20;
  const std::vector<Vec3> dabs{c};
  REQUIRE(stroke(sculptor, obj, draw, options, dabs, r, 0.0f, detail));
  const Mesh& m = obj.mesh;
  CHECK(m.faceCount() > before.faceCount() * 3 / 2);
  CHECK(test::eulerCharacteristic(m) == 2);

  int long_ = 0, inside = 0;
  forEachEdge(m, [&](const Vec3& a, const Vec3& b, bool) {
    if (segmentDist(c, a, b) > r) return;
    ++inside;
    long_ += glm::length(a - b) > detail * 1.0001f;
  });
  CHECK(inside > 100);
  CHECK(long_ == 0);
  CHECK(minTriangleAngle(m) > 20.0f);
  // Faces beyond the reach of any split edge are untouched, and no vertex moved.
  const float reach = r + 2.0f * edge;
  CHECK(lostFaces(before, m, [&](Index v) { return glm::length(before.positions[v] - c) > reach; }) == 0);
  std::set<std::tuple<float, float, float>> now;
  for (const Vec3& p : m.positions) now.insert({p.x, p.y, p.z});
  int moved = 0;
  for (const Vec3& p : before.positions) moved += now.count({p.x, p.y, p.z}) == 0;
  CHECK(moved == 0);
}

TEST_CASE("a face larger than the brush is refined toward the cursor") {
  Scene scene;
  SceneObject& obj = scene.add("Cube", makeCube(1));  // Six 2 x 2 quads.
  DrawBrush draw;
  Sculptor sculptor;
  const Vec3 c{0.1f, 1.0f, 0.05f};

  // Without a hint no edge is within the brush, so nothing changes.
  sculptor.beginStroke(obj, draw, dyntopoStroke(), "Draw");
  sculptor.dab(c, 0.2f, 0.0f, DabTopology{0.05f, kInvalid});
  CHECK_FALSE(sculptor.endStroke());
  CHECK(obj.mesh.faceCount() == 6);

  sculptor.beginStroke(obj, draw, dyntopoStroke(), "Draw");
  for (int i = 0; i < 12; ++i) {
    sculptor.dab(c, 0.2f, 0.0f, DabTopology{0.05f, faceNear(obj, c, 0.2f)});
    requireDynamic(obj);
  }
  REQUIRE(sculptor.endStroke());
  requireCompact(obj);
  const Index f = faceNear(obj, c, 0.2f);
  REQUIRE(f != kInvalid);
  float longest = 0.0f;
  const Index h0 = obj.mesh.faceHe[f];
  Index h = h0;
  do {
    longest = std::max(longest, glm::length(obj.mesh.positions[obj.mesh.heTarget(h)] - obj.mesh.positions[obj.mesh.heVert[h]]));
    h = obj.mesh.heNext[h];
  } while (h != h0);
  CHECK(longest < 0.5f);
}

TEST_CASE("coarsening merges short edges without folds, long edges or high valence") {
  Scene scene;
  SceneObject& obj = scene.add("Ico", makeIcosphere(5));
  const Index facesBefore = obj.mesh.faceCount();
  DrawBrush draw;
  Sculptor sculptor;
  const Vec3 c = surfacePoint(obj, {0.0f, 1.0f, 0.0f});
  const float r = 0.4f, detail = 0.3f;
  const std::vector<Vec3> dabs(8, c);
  REQUIRE(stroke(sculptor, obj, draw, dyntopoStroke(DyntopoRefine::CollapseOnly), dabs, r, 0.0f, detail));
  const Mesh& m = obj.mesh;
  // About 800 faces lie inside the brush; most of them go.
  CHECK(sculptor.lastStrokeTopology().collapses > 200);
  CHECK(m.faceCount() < facesBefore - 400);
  CHECK(test::eulerCharacteristic(m) == 2);
  test::requireOutward(m);
  int high = 0;
  for (Index v = 0; v < m.vertexCount(); ++v)
    if (glm::length(m.positions[v] - c) < r) high += valence(m, v) > 12;
  CHECK(high == 0);
  int long_ = 0;
  forEachEdge(m, [&](const Vec3& a, const Vec3& b, bool) {
    if (glm::length((a + b) * 0.5f - c) < r) long_ += glm::length(a - b) > detail;
  });
  CHECK(long_ == 0);
}

TEST_CASE("masked vertices and every face around them stay exactly as they were") {
  Scene scene;
  Mesh mesh = makeQuadSphere(24);
  mesh.mask.resize(mesh.positions.size());
  for (Index v = 0; v < mesh.vertexCount(); ++v) mesh.mask[v] = std::clamp((mesh.positions[v].x + 0.1f) / 0.2f, 0.0f, 1.0f);
  SceneObject& obj = scene.add("A", std::move(mesh));
  const Mesh before = obj.mesh;
  DrawBrush draw;
  Sculptor sculptor;
  const std::vector<Vec3> dabs = arc(obj, {-0.2f, 1.0f, -0.3f}, {0.2f, 1.0f, 0.3f}, 6);
  REQUIRE(stroke(sculptor, obj, draw, dyntopoStroke(), dabs, 0.4f, 0.0f, 0.03f));
  CHECK(obj.mesh.faceCount() > before.faceCount());
  REQUIRE(stroke(sculptor, obj, draw, dyntopoStroke(), dabs, 0.4f, 0.0f, 0.5f));
  const Mesh& m = obj.mesh;
  CHECK(sculptor.lastStrokeTopology().collapses > 0);

  // Any face with a locked corner survives as it was.
  CHECK(lostFaces(before, m, [&](Index) { return true; }) > 0);  // The free side did change.
  int lost = 0;
  const std::set<FaceKey> now = faceKeys(m);
  for (Index f = 0; f < before.faceCount(); ++f) {
    bool locked = false;
    before.forEachFaceVertex(f, [&](Index v) { locked |= before.mask[v] >= 0.5f; });
    if (locked && !now.count(faceKey(before, f))) ++lost;
  }
  CHECK(lost == 0);
  // The locked vertices are exactly the ones there were, with the same mask.
  std::map<std::tuple<float, float, float>, float> lockedNow;
  for (Index v = 0; v < m.vertexCount(); ++v)
    if (m.mask[v] >= 0.5f) lockedNow[{m.positions[v].x, m.positions[v].y, m.positions[v].z}] = m.mask[v];
  int lockedBefore = 0, kept = 0;
  for (Index v = 0; v < before.vertexCount(); ++v) {
    if (before.mask[v] < 0.5f) continue;
    ++lockedBefore;
    const auto it = lockedNow.find({before.positions[v].x, before.positions[v].y, before.positions[v].z});
    kept += it != lockedNow.end() && it->second == before.mask[v];
  }
  CHECK(kept == lockedBefore);
  CHECK(static_cast<int>(lockedNow.size()) == lockedBefore);
}

TEST_CASE("a vertex with two separate fans locks its faces and the rest still refines") {
  BuildReport report;
  Mesh mesh = makePinchedSpheres(report);
  REQUIRE(report.nonManifoldVertices == 1);
  REQUIRE(report.nonManifoldEdges == 0);
  Scene scene;
  SceneObject& obj = scene.add("Pinch", std::move(mesh));
  const State start = capture(obj);
  const Mesh before = obj.mesh;
  Index pinch = kInvalid;
  for (Index v = 0; v < before.vertexCount(); ++v)
    if (glm::length(before.positions[v]) < 1e-6f) pinch = v;
  REQUIRE(pinch != kInvalid);

  DrawBrush draw;
  Sculptor sculptor;
  // Over the pinch, so its faces are in reach, and over the top sphere.
  const std::vector<Vec3> dabs{{0.0f, 0.0f, 0.0f}, {0.1f, 0.15f, 0.0f}, {0.3f, 0.5f, 0.2f}};
  sculptor.beginStroke(obj, draw, dyntopoStroke(), "Draw");
  REQUIRE(sculptor.dyntopo());
  CHECK(sculptor.dyntopo()->lockedVertices() == 1);
  for (const Vec3& c : dabs) sculptor.dab(c, 0.3f, 0.0f, DabTopology{0.04f, faceNear(obj, c, 0.3f)});
  auto undo = sculptor.endStroke();
  REQUIRE(undo);
  CHECK_FALSE(sculptor.lastStrokeTopology().lostUndo);
  // validateLayout() refuses the non-manifold vertex itself; the leaves still cover everything.
  CHECK_FALSE(obj.bvh.dynamic());
  REQUIRE_FALSE(obj.bvh.leaves().empty());
  CHECK(obj.bvh.leaves().back().faceEnd == obj.mesh.faceCount());
  CHECK(obj.bvh.leaves().back().heEnd == obj.mesh.halfEdgeCount());
  CHECK(obj.mesh.faceCount() > before.faceCount());

  // The faces at the pinch are unchanged, and it is still the only non-manifold vertex.
  CHECK(lostFaces(before, obj.mesh, [&](Index) { return true; }) > 0);
  const std::set<FaceKey> now = faceKeys(obj.mesh);
  int lost = 0;
  for (Index f = 0; f < before.faceCount(); ++f) {
    bool atPinch = false;
    before.forEachFaceVertex(f, [&](Index v) { atPinch |= v == pinch; });
    if (atPinch && !now.count(faceKey(before, f))) ++lost;
  }
  CHECK(lost == 0);
  std::vector<Index> idx, sizes;
  for (Index f = 0; f < obj.mesh.faceCount(); ++f) {
    Index n = 0;
    obj.mesh.forEachFaceVertex(f, [&](Index v) {
      idx.push_back(v);
      ++n;
    });
    sizes.push_back(n);
  }
  BuildReport rebuilt;
  buildMesh(obj.mesh.positions, idx, sizes, &rebuilt);
  CHECK(rebuilt.nonManifoldVertices == 1);
  CHECK(rebuilt.nonManifoldEdges == 0);

  UndoStack stack;
  stack.push(std::move(*undo));
  CHECK_FALSE(stack.undo(scene).empty());
  requireState(obj, start);
}

TEST_CASE("with x symmetry both sides are refined alike") {
  Scene scene;
  SceneObject& obj = scene.add("Ico", makeIcosphere(3));
  const Mesh before = obj.mesh;
  DrawBrush draw;
  Sculptor sculptor;
  StrokeOptions options = dyntopoStroke(DyntopoRefine::SplitOnly);
  options.symmetryX = true;
  const float r = 0.3f, detail = 0.04f;
  const std::vector<Vec3> dabs = arc(obj, {0.5f, 0.8f, 0.3f}, {0.6f, 0.7f, -0.1f}, 4);
  REQUIRE(stroke(sculptor, obj, draw, options, dabs, r, 0.0f, detail));
  const Mesh& m = obj.mesh;
  for (const float side : {1.0f, -1.0f}) {
    INFO("side " << side);
    int long_ = 0, inside = 0;
    forEachEdge(m, [&](const Vec3& a, const Vec3& b, bool) {
      bool near = false;
      for (const Vec3& d : dabs) near |= segmentDist({side * d.x, d.y, d.z}, a, b) <= r;
      if (!near) return;
      ++inside;
      long_ += glm::length(a - b) > detail * 1.0001f;
    });
    CHECK(inside > 200);
    CHECK(long_ == 0);
  }
  auto facesNear = [&](float side) {
    int n = 0;
    for (Index f = 0; f < m.faceCount(); ++f) {
      const Vec3 fc = m.faceCentroid(f);
      for (const Vec3& d : dabs)
        if (glm::length(fc - Vec3{side * d.x, d.y, d.z}) < r) {
          ++n;
          break;
        }
    }
    return n;
  };
  const int right = facesNear(1.0f), left = facesNear(-1.0f);
  CHECK(right > 0);
  CHECK(static_cast<float>(left) > 0.85f * static_cast<float>(right));
  CHECK(static_cast<float>(left) < 1.15f * static_cast<float>(right));
}

TEST_CASE("every moving brush works with dynamic topology and undoes exactly") {
  DrawBrush draw;
  ClayBrush clay;
  SmoothBrush smooth;
  InflateBrush inflate;
  FlattenBrush flatten;
  CreaseBrush crease;
  for (const Brush* brush : std::initializer_list<const Brush*>{&draw, &clay, &smooth, &inflate, &flatten, &crease}) {
    INFO(brush->name());
    Scene scene;
    SceneObject& obj = scene.add("A", makeQuadSphere(16));
    const State start = capture(obj);
    Sculptor sculptor;
    const std::vector<Vec3> dabs = arc(obj, {-0.5f, 1.0f, 0.0f}, {0.5f, 1.0f, 0.3f}, 8);
    StrokeOptions options = dyntopoStroke();
    options.symmetryX = true;
    auto undo = stroke(sculptor, obj, *brush, options, dabs, 0.35f, 0.6f, 0.04f);
    REQUIRE(undo);
    REQUIRE(std::holds_alternative<DyntopoUndo>(*undo));
    const State after = capture(obj);
    UndoStack stack;
    stack.push(std::move(*undo));
    CHECK(stack.undo(scene) == brush->name());
    requireState(obj, start);
    CHECK(stack.redo(scene) == brush->name());
    requireState(obj, after);
  }
}

TEST_CASE("mask and grab strokes ignore dynamic topology") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(16));
  const Index faces = obj.mesh.faceCount();
  Sculptor sculptor;
  MaskBrush mask;
  sculptor.beginStroke(obj, mask, dyntopoStroke(), "Mask");
  CHECK(sculptor.dyntopo() == nullptr);
  const Vec3 c = surfacePoint(obj, {0, 1, 0});
  sculptor.dab(c, 0.4f, 1.0f, DabTopology{0.01f, faceNear(obj, c, 0.4f)});
  REQUIRE(sculptor.endStroke());
  CHECK(obj.mesh.faceCount() == faces);

  REQUIRE(sculptor.beginGrab(obj, dyntopoStroke(), c, 0.4f, "Grab"));
  CHECK(sculptor.dyntopo() == nullptr);
  sculptor.grab({0.0f, 0.2f, 0.0f});
  auto grab = sculptor.endStroke();
  REQUIRE(grab);
  CHECK(std::holds_alternative<SculptUndo>(*grab));
  CHECK(obj.mesh.faceCount() == faces);
  requireCompact(obj);
}

TEST_CASE("a dyntopo stroke that changes no topology records a plain sculpt entry") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeIcosphere(3));
  const Index faces = obj.mesh.faceCount();
  const float edge = maxEdge(obj.mesh);
  DrawBrush draw;
  Sculptor sculptor;
  const std::vector<Vec3> dabs = arc(obj, {0, 1, 0}, {0.3f, 1, 0}, 3);
  // Every edge is shorter than the detail size and longer than the collapse limit.
  auto undo = stroke(sculptor, obj, draw, dyntopoStroke(), dabs, 0.3f, 0.5f, edge * 1.1f);
  REQUIRE(undo);
  CHECK(std::holds_alternative<SculptUndo>(*undo));
  CHECK(obj.mesh.faceCount() == faces);
  CHECK_FALSE(sculptor.lastStrokeTopology().dyntopo);
}

TEST_CASE("open borders keep their shape: split at midpoints, never merged") {
  Scene scene;
  SceneObject& obj = scene.add("Plane", makePlane(8));
  struct Segment {
    Vec3 a, b;
  };
  auto border = [](const Mesh& m) {
    std::vector<Segment> out;
    forEachEdge(m, [&](const Vec3& a, const Vec3& b, bool open) {
      if (open) out.push_back({a, b});
    });
    return out;
  };
  auto length = [](const std::vector<Segment>& s) {
    double n = 0.0;
    for (const Segment& e : s) n += glm::length(e.b - e.a);
    return n;
  };
  const std::vector<Segment> before = border(obj.mesh);
  const Index facesBefore = obj.mesh.faceCount();
  DrawBrush draw;
  Sculptor sculptor;
  std::vector<Vec3> dabs;
  for (int i = 0; i < 8; ++i) dabs.push_back({0.95f, 0.0f, -0.8f + 0.2f * static_cast<float>(i)});
  REQUIRE(stroke(sculptor, obj, draw, dyntopoStroke(), dabs, 0.3f, 0.0f, 0.05f));
  CHECK(obj.mesh.faceCount() > facesBefore);
  CHECK(minTriangleAngle(obj.mesh) > 20.0f);
  const std::vector<Segment> refined = border(obj.mesh);
  CHECK(refined.size() > before.size());
  CHECK(length(refined) == doctest::Approx(length(before)).epsilon(1e-5));
  // Every border edge now lies on one of the original border edges.
  int off = 0;
  for (const Segment& e : refined) {
    bool on = false;
    for (const Segment& o : before) on |= segmentDist(e.a, o.a, o.b) < 1e-6f && segmentDist(e.b, o.a, o.b) < 1e-6f;
    off += !on;
  }
  CHECK(off == 0);
  // Coarsening merges inside, never along the border.
  REQUIRE(stroke(sculptor, obj, draw, dyntopoStroke(), dabs, 0.3f, 0.0f, 0.6f));
  CHECK(sculptor.lastStrokeTopology().collapses > 0);
  CHECK(border(obj.mesh).size() == refined.size());
}

TEST_CASE("per-pass limits, the time budget and the detail clamp bound the work") {
  CHECK(DyntopoSession::effectiveDetail(0.05f, 0.2f, 2.0f) == 0.05f);
  CHECK(DyntopoSession::effectiveDetail(1e-9f, 0.2f, 2.0f) == doctest::Approx(0.01f));
  CHECK(DyntopoSession::effectiveDetail(0.0f, 0.0f, 1e6f) == doctest::Approx(1.0f));

  DrawBrush draw;
  auto oneDab = [&](const StrokeOptions& options, float detail, float r, int subdivisions) {
    Scene scene;
    SceneObject& obj = scene.add("Ico", makeIcosphere(subdivisions));
    Sculptor sculptor;
    sculptor.beginStroke(obj, draw, options, "Draw");
    const Vec3 c = surfacePoint(obj, {0, 1, 0});
    sculptor.dab(c, r, 0.0f, DabTopology{detail, faceNear(obj, c, r)});
    const DabTiming t = sculptor.lastDab();
    sculptor.endStroke();
    requireCompact(obj);
    return std::tuple{t.splits, t.collapses, obj.mesh.faceCount()};
  };

  StrokeOptions limited = dyntopoStroke();
  limited.dyntopoOptions.maxSplits = 7;
  CHECK(std::get<0>(oneDab(limited, 0.02f, 1.0f, 2)) == 7);
  limited = dyntopoStroke();
  limited.dyntopoOptions.maxCollapses = 3;
  CHECK(std::get<1>(oneDab(limited, 0.5f, 0.6f, 4)) == 3);

  StrokeOptions hurried = dyntopoStroke();
  hurried.dyntopoOptions.timeBudgetMs = 1e-9;
  const int quick = std::get<0>(oneDab(hurried, 0.02f, 1.0f, 2));
  const int full = std::get<0>(oneDab(dyntopoStroke(), 0.02f, 1.0f, 2));
  CHECK(quick > 0);
  CHECK(quick < 32);
  CHECK(full > 100);

  // A detail finer than radius / 20 behaves exactly like radius / 20.
  CHECK(std::get<2>(oneDab(dyntopoStroke(), 1e-9f, 0.4f, 3)) == std::get<2>(oneDab(dyntopoStroke(), 0.02f, 0.4f, 3)));
}

TEST_CASE("every element a stroke changes sits in a leaf recorded before it changed") {
  Scene scene;
  Mesh mesh = makeQuadSphere(20);
  mesh.ensureMask();
  SceneObject& obj = scene.add("A", std::move(mesh));
  const Mesh start = obj.mesh;
  const Bvh startBvh = obj.bvh;
  ClayBrush clay;
  Sculptor sculptor;
  StrokeOptions options = dyntopoStroke();
  options.symmetryX = true;
  sculptor.beginStroke(obj, clay, options, "Clay");
  const std::vector<Vec3> fine = arc(obj, {-0.5f, 1.0f, 0.2f}, {0.6f, 0.8f, -0.2f}, 6);
  for (int i = 0; i < 12; ++i) {
    const Vec3& c = fine[static_cast<std::size_t>(i) % fine.size()];
    // Alternate fine and coarse dabs so the stroke both splits and collapses.
    sculptor.dab(c, 0.35f, 0.5f, DabTopology{i % 2 == 0 ? 0.03f : 0.3f, faceNear(obj, c, 0.35f)});
  }
  const DyntopoSession* session = sculptor.dyntopo();
  REQUIRE(session);
  CHECK(session->totalSplits() > 0);
  CHECK(session->totalCollapses() > 0);
  const ClaimRecorder& rec = session->recorder();
  const Mesh& m = obj.mesh;
  int missed = 0;
  for (Index v = 0; v < start.vertexCount(); ++v) {
    const bool changed = start.vertHe[v] != m.vertHe[v] || start.mask[v] != m.mask[v];
    if (changed && !rec.claimed(startBvh.leafOfVertex(v))) ++missed;
  }
  for (Index f = 0; f < start.faceCount(); ++f)
    if (start.faceHe[f] != m.faceHe[f] && !rec.claimed(startBvh.leafOfFace(f))) ++missed;
  for (Index h = 0; h < start.halfEdgeCount(); ++h) {
    const bool changed = start.heNext[h] != m.heNext[h] || start.heTwin[h] != m.heTwin[h] ||
                         start.heVert[h] != m.heVert[h] || start.heFace[h] != m.heFace[h];
    if (changed && !rec.claimed(startBvh.leafOfHalfEdge(h))) ++missed;
  }
  CHECK(missed == 0);
  REQUIRE(sculptor.endStroke());
  CHECK_FALSE(sculptor.lastStrokeTopology().lostUndo);
  requireCompact(obj);
}

TEST_CASE("a dyntopo stroke's undo entry stays a small fraction of the mesh") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(96));
  const Mesh& m = obj.mesh;
  const std::size_t meshBytes = (m.positions.size() + m.normals.size()) * sizeof(Vec3) +
                                (m.vertHe.size() + m.faceHe.size() + m.heNext.size() + m.heTwin.size() +
                                 m.heVert.size() + m.heFace.size()) * sizeof(Index);
  DrawBrush draw;
  Sculptor sculptor;
  const std::vector<Vec3> dabs = arc(obj, {0.0f, 1.0f, 0.0f}, {0.3f, 1.0f, 0.0f}, 5);
  auto undo = stroke(sculptor, obj, draw, dyntopoStroke(), dabs, 0.15f, 0.5f, 0.01f, false);
  REQUIRE(undo);
  REQUIRE(std::holds_alternative<DyntopoUndo>(*undo));
  const std::size_t entry = std::get<DyntopoUndo>(*undo).bytes();
  INFO("entry " << entry << " bytes, mesh " << meshBytes << " bytes");
  CHECK(entry * 10 < meshBytes);
  UndoStack stack;
  stack.push(std::move(*undo));
  stack.undo(scene);  // Captures the after side.
  CHECK(stack.bytes() * 4 < meshBytes);
  stack.redo(scene);  // And drops it again.
  CHECK(stack.bytes() == entry);
}

TEST_CASE("a dyntopo result saves, loads, exports, duplicates and remeshes like any mesh") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(16));
  DrawBrush draw;
  Sculptor sculptor;
  const std::vector<Vec3> dabs = arc(obj, {-0.5f, 1.0f, 0.0f}, {0.5f, 1.0f, 0.3f}, 8);
  REQUIRE(stroke(sculptor, obj, draw, dyntopoStroke(), dabs, 0.35f, 0.6f, 0.03f));

  const std::vector<std::uint8_t> bytes = serializeProject(scene, "");
  auto project = parseProject(bytes.data(), bytes.size());
  REQUIRE(project);
  REQUIRE(project->objects.size() == 1);
  const Mesh& loaded = project->objects[0].mesh;
  CHECK(loaded.positions == obj.mesh.positions);
  CHECK(faceKeys(loaded) == faceKeys(obj.mesh));
  test::requireValid(loaded);

  const ObjImportResult obj2 = parseObj(writeObj(obj.mesh));
  REQUIRE(obj2.ok);
  CHECK(obj2.report.ok());
  CHECK(obj2.mesh.faceCount() == obj.mesh.faceCount());
  test::requireValid(obj2.mesh);

  SceneObject* copy = scene.duplicate(obj.id);
  REQUIRE(copy);
  requireCompact(*copy);

  QuadRemeshParams params;
  params.targetEdge = 0.08f;
  auto remeshed = quadRemesh(obj.mesh, params);
  REQUIRE(remeshed);
  CHECK(remeshed->faceCount() > 100);
  test::requireValid(*remeshed);
}

TEST_CASE("an open stroke is finished before the next one starts") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeIcosphere(3));
  DrawBrush draw;
  Sculptor sculptor;
  const Vec3 c = surfacePoint(obj, {0, 1, 0});
  sculptor.beginStroke(obj, draw, dyntopoStroke(), "Draw");
  sculptor.dab(c, 0.3f, 0.5f, DabTopology{0.03f, faceNear(obj, c, 0.3f)});
  REQUIRE(obj.bvh.dynamic());
  sculptor.beginStroke(obj, draw, {}, "Draw");  // No endStroke() in between.
  CHECK_FALSE(obj.bvh.dynamic());
  test::requireValid(obj.mesh);
  sculptor.dab(c, 0.3f, 0.5f);
  REQUIRE(sculptor.endStroke());
  requireCompact(obj);
}

TEST_CASE("random strokes, masks and undo keep the mesh valid and history exact") {
  DrawBrush draw;
  ClayBrush clay;
  SmoothBrush smooth;
  InflateBrush inflate;
  FlattenBrush flatten;
  CreaseBrush crease;
  MaskBrush mask;
  const std::vector<const Brush*> brushes{&draw, &clay, &smooth, &inflate, &flatten, &crease};
  const DyntopoRefine refines[] = {DyntopoRefine::SplitCollapse, DyntopoRefine::SplitOnly, DyntopoRefine::CollapseOnly};
  for (unsigned seed : {1u, 2u, 3u}) {
    INFO("seed " << seed);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    Scene scene;
    Mesh mesh = makeIcosphere(3);
    mesh.ensureMask();  // So undoing the first mask stroke has a mask to return to.
    SceneObject& obj = scene.add("A", std::move(mesh));
    auto workspace = std::make_shared<LayoutWorkspace>();
    UndoStack stack;
    stack.setLayoutWorkspace(workspace);
    Sculptor sculptor;
    sculptor.setLayoutWorkspace(workspace);
    std::vector<State> history{capture(obj)};
    std::size_t cursor = 0;
    for (int step = 0; step < 40; ++step) {
      INFO("step " << step);
      const float roll = uni(rng);
      if (roll < 0.2f && cursor > 0) {
        CHECK_FALSE(stack.undo(scene).empty());
        requireState(obj, history[--cursor]);
        continue;
      }
      if (roll < 0.3f && cursor + 1 < history.size()) {
        CHECK_FALSE(stack.redo(scene).empty());
        requireState(obj, history[++cursor]);
        continue;
      }
      const bool maskStroke = uni(rng) < 0.15f;
      const Brush& brush = maskStroke ? static_cast<const Brush&>(mask) : *brushes[rng() % brushes.size()];
      StrokeOptions options = uni(rng) < 0.15f ? StrokeOptions{} : dyntopoStroke(refines[rng() % 3]);
      options.symmetryX = uni(rng) < 0.5f;
      options.invert = uni(rng) < 0.3f;
      const float r = 0.1f + 0.4f * uni(rng);
      const float detail = 0.03f + 0.37f * uni(rng);
      const float strength = uni(rng);
      std::vector<Vec3> dabs;
      Vec3 dir{uni(rng) - 0.5f, uni(rng) - 0.5f, uni(rng) - 0.5f};
      if (glm::length(dir) < 1e-3f) dir = {0, 1, 0};
      const int n = 3 + static_cast<int>(rng() % 6);
      for (int i = 0; i < n; ++i) {
        dabs.push_back(surfacePoint(obj, dir));
        dir += Vec3{uni(rng) - 0.5f, uni(rng) - 0.5f, uni(rng) - 0.5f} * 0.3f;
        if (glm::length(dir) < 1e-3f) dir = {0, 1, 0};
      }
      auto undo = stroke(sculptor, obj, brush, options, dabs, r, strength, detail);
      if (!undo) {
        requireState(obj, history[cursor]);  // Nothing changed, so nothing was recorded.
        continue;
      }
      stack.push(std::move(*undo));
      history.resize(cursor + 1);
      history.push_back(capture(obj));
      ++cursor;
    }
    // Walk the whole history back and forth.
    while (cursor > 0) {
      CHECK_FALSE(stack.undo(scene).empty());
      requireState(obj, history[--cursor]);
    }
    while (cursor + 1 < history.size()) {
      CHECK_FALSE(stack.redo(scene).empty());
      requireState(obj, history[++cursor]);
    }
    requireCompact(obj);
  }
}
