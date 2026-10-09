#include <algorithm>
#include <memory>
#include <random>

#include "TestUtil.h"
#include "mesh/MeshEdit.h"
#include "mesh/Primitives.h"
#include "remesh/VoxelRemesh.h"
#include "scene/Scene.h"
#include "sculpt/MaskOps.h"
#include "sculpt/Sculptor.h"
#include "sculpt/Undo.h"
#include "spatial/Bvh.h"
#include "spatial/LeafLayout.h"

using namespace plegl;

namespace {

// A dynamic topology stroke driven by hand: random refinements and collapses near a point, made
// through a MeshEditor that reports to a ClaimRecorder, as the dyntopo brush does.
struct HandStroke {
  Mesh& m;
  Bvh& bvh;
  Mesh startMesh;
  std::shared_ptr<const Bvh> startBvh;
  ClaimRecorder rec;
  std::unique_ptr<MeshEditor> ed;

  HandStroke(Mesh& mesh, Bvh& b) : m(mesh), bvh(b), startMesh(mesh), startBvh(std::make_shared<const Bvh>(b)) {
    bvh.beginDynamic(m);
    rec.begin(m, bvh);
    ed = std::make_unique<MeshEditor>(m);
    ed->setObserver(&rec);
  }

  // Splits the edge of h at its midpoint and cuts both faces from the new vertex.
  bool refine(Index h) {
    const Index t = m.heTwin[h];
    if (m.heFace[h] == kInvalid || t == kInvalid) return false;
    ed->splitEdge(h, 0.5f);
    for (Index s : {m.heNext[h], m.heNext[t]}) {  // Both start at the new vertex.
      const Index across = m.heNext[m.heNext[s]];
      if (m.heNext[across] != s) ed->splitFace(s, across);
    }
    bvh.growTail(m);
    return true;
  }

  // Collapses the edge of h at its midpoint; the end vertex dies.
  bool collapse(Index h) {
    if (m.heFace[h] == kInvalid || m.heTwin[h] == kInvalid) return false;
    const Index a = m.heVert[h], b = m.heTarget(h);
    if (!ed->collapseEdge(h, (m.positions[a] + m.positions[b]) * 0.5f)) return false;
    m.positions[b] = kDeadPosition;  // b was reported (and its leaf claimed) by the collapse.
    return true;
  }

  // Random edits on live faces whose centroid lies within r of c. Returns the edits made.
  int edit(const Vec3& c, float r, int attempts, double collapseShare, std::mt19937& rng) {
    int done = 0;
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    for (int batch = 0; batch < attempts; batch += 20) {
      std::vector<Index> faces;
      for (Index f = 0; f < m.faceCount(); ++f)
        if (m.faceHe[f] != kInvalid && glm::length(m.faceCentroid(f) - c) < r) faces.push_back(f);
      if (faces.empty()) break;
      for (int i = 0; i < 20; ++i) {
        const Index f = faces[rng() % faces.size()];
        if (m.faceHe[f] == kInvalid) continue;
        Index h = m.faceHe[f];
        for (unsigned k = rng() % 4; k > 0; --k) h = m.heNext[h];
        done += coin(rng) < collapseShare ? collapse(h) : refine(h);
      }
    }
    return done;
  }

  // Every element that existed when the stroke began and has changed lies in a claimed leaf.
  void requireClaimsCover() const {
    const Mesh& a = startMesh;
    int missed = 0;
    for (Index v = 0; v < a.vertexCount(); ++v) {
      const bool changed = a.positions[v] != m.positions[v] || a.vertHe[v] != m.vertHe[v] ||
                           (!a.mask.empty() && a.mask[v] != m.mask[v]);
      if (changed && !rec.claimed(startBvh->leafOfVertex(v))) ++missed;
    }
    for (Index f = 0; f < a.faceCount(); ++f)
      if (a.faceHe[f] != m.faceHe[f] && !rec.claimed(startBvh->leafOfFace(f))) ++missed;
    for (Index h = 0; h < a.halfEdgeCount(); ++h) {
      const bool changed = a.heNext[h] != m.heNext[h] || a.heTwin[h] != m.heTwin[h] ||
                           a.heVert[h] != m.heVert[h] || a.heFace[h] != m.heFace[h];
      if (changed && !rec.claimed(startBvh->leafOfHalfEdge(h))) ++missed;
    }
    CHECK(missed == 0);
  }

  std::vector<Index> claimedLeaves() const { return {rec.claimedLeaves().begin(), rec.claimedLeaves().end()}; }
};

void requireLayout(const Mesh& m, const Bvh& bvh) {
  const ValidationResult r = validateLayout(m, bvh);
  INFO(r.message);
  REQUIRE(r.ok);
}

void requireDynamic(const Mesh& m, const Bvh& bvh) {
  const ValidationResult r = validateDynamic(m, bvh);
  INFO(r.message);
  REQUIRE(r.ok);
}

void requireSameMesh(const Mesh& a, const Mesh& b) {
  CHECK(a.positions == b.positions);
  CHECK(a.normals == b.normals);
  CHECK(a.mask == b.mask);
  CHECK(a.faceSets == b.faceSets);
  CHECK(a.vertHe == b.vertHe);
  CHECK(a.faceHe == b.faceHe);
  CHECK(a.heNext == b.heNext);
  CHECK(a.heTwin == b.heTwin);
  CHECK(a.heVert == b.heVert);
  CHECK(a.heFace == b.heFace);
}

bool sameBox(const Aabb& a, const Aabb& b) { return a.min == b.min && a.max == b.max; }

void requireSameBvh(const Bvh& a, const Bvh& b) {
  REQUIRE(a.leaves().size() == b.leaves().size());
  REQUIRE(a.nodes().size() == b.nodes().size());
  int diff = 0;
  for (std::size_t i = 0; i < a.leaves().size(); ++i) {
    const BvhLeaf &x = a.leaves()[i], &y = b.leaves()[i];
    diff += x.faceBegin != y.faceBegin || x.faceEnd != y.faceEnd || x.vertBegin != y.vertBegin ||
            x.vertEnd != y.vertEnd || x.heBegin != y.heBegin || x.heEnd != y.heEnd || x.flags != y.flags ||
            !sameBox(x.bounds, y.bounds);
  }
  for (std::size_t i = 0; i < a.nodes().size(); ++i) {
    const BvhNode &x = a.nodes()[i], &y = b.nodes()[i];
    diff += x.left != y.left || x.right != y.right || x.leaf != y.leaf || !sameBox(x.bounds, y.bounds);
  }
  CHECK(diff == 0);
}

// leafOf* agree with the leaf ranges for every element.
void requireLeafLookups(const Mesh& m, const Bvh& bvh) {
  int wrong = 0;
  const auto leaves = bvh.leaves();
  for (Index l = 0; l < static_cast<Index>(leaves.size()); ++l) {
    for (Index f = leaves[l].faceBegin; f < leaves[l].faceEnd; ++f) wrong += bvh.leafOfFace(f) != l;
    for (Index h = leaves[l].heBegin; h < leaves[l].heEnd; ++h) wrong += bvh.leafOfHalfEdge(h) != l;
    for (Index v = leaves[l].vertBegin; v < leaves[l].vertEnd; ++v) wrong += bvh.leafOfVertex(v) != l;
  }
  const Index iso = leaves.empty() ? 0 : leaves.back().vertEnd;
  for (Index v = iso; v < m.vertexCount(); ++v) wrong += bvh.leafOfVertex(v) != kInvalid;
  CHECK(wrong == 0);
}

}  // namespace

TEST_CASE("a stroke with no edits leaves the layout alone") {
  Mesh m = makeIcosphere(3);
  Bvh bvh;
  bvh.build(m, {.maxLeafFaces = 128});
  const Mesh start = m;
  const Bvh startBvh = bvh;
  HandStroke s(m, bvh);
  LayoutWorkspace ws;
  const ConsolidateResult r = consolidate(m, bvh, s.claimedLeaves(), ws);
  CHECK(r.delta.regionLeaves.empty());
  CHECK_FALSE(bvh.dynamic());
  requireSameMesh(m, start);
  requireSameBvh(bvh, startBvh);
}

TEST_CASE("consolidating a stroke rebuilds only the region and keeps a canonical layout") {
  for (int quads = 0; quads < 2; ++quads) {
    CAPTURE(quads);
    Mesh m = quads ? makeQuadSphere(24) : makeIcosphere(4);
    Bvh bvh;
    bvh.build(m, {.maxLeafFaces = 256});
    const Mesh start = m;
    const Bvh startBvh = bvh;
    const Index euler = test::eulerCharacteristic(m);

    HandStroke s(m, bvh);
    std::mt19937 rng(7 + quads);
    CHECK(s.edit(glm::normalize(Vec3{0.8f, 0.4f, 0.3f}), 0.4f, 400, 0.4, rng) > 100);
    bvh.refit(m);
    requireDynamic(m, bvh);
    s.requireClaimsCover();

    const std::vector<Index> claimed = s.claimedLeaves();
    LayoutWorkspace ws;
    const ConsolidateResult r = consolidate(m, bvh, claimed, ws);
    CHECK_FALSE(r.missedClaim);
    CHECK_FALSE(bvh.dynamic());
    test::requireValid(m);
    requireLayout(m, bvh);
    requireLeafLookups(m, bvh);
    CHECK(test::eulerCharacteristic(m) == euler);
    for (const BvhLeaf& leaf : bvh.leaves()) CHECK(leaf.faceEnd - leaf.faceBegin <= 256);
    for (Index l : claimed) CHECK(std::binary_search(r.delta.regionLeaves.begin(), r.delta.regionLeaves.end(), l));
    CHECK(r.beforeFaceCount == start.faceCount());
    CHECK(r.beforeVertexCount == start.vertexCount());

    // Untouched leaves keep their index, size, bounds and vertices; they only shift.
    int untouched = 0;
    for (Index l = 0; l < static_cast<Index>(startBvh.leaves().size()); ++l) {
      if (std::binary_search(r.delta.regionLeaves.begin(), r.delta.regionLeaves.end(), l)) continue;
      ++untouched;
      const BvhLeaf& was = startBvh.leaves()[l];
      const BvhLeaf& now = bvh.leaves()[l];
      REQUIRE(now.faceEnd - now.faceBegin == was.faceEnd - was.faceBegin);
      REQUIRE(now.vertEnd - now.vertBegin == was.vertEnd - was.vertBegin);
      CHECK(sameBox(now.bounds, was.bounds));
      CHECK(std::equal(start.positions.begin() + was.vertBegin, start.positions.begin() + was.vertEnd,
                       m.positions.begin() + now.vertBegin));
      for (Index k = 0; k < now.heEnd - now.heBegin; ++k) {
        // References that stay inside the leaf shift with it.
        const Index next = start.heNext[was.heBegin + k];
        CHECK(m.heNext[now.heBegin + k] == next - was.heBegin + now.heBegin);
      }
    }
    CHECK(untouched > 0);
    CHECK(r.delta.vertRefs.size() + r.delta.heRefs.size() > 0);
  }
}

TEST_CASE("relayout switches between a stroke's two layouts bit for bit") {
  Mesh m = makeQuadSphere(24);
  Bvh bvh;
  bvh.build(m, {.maxLeafFaces = 200});
  m.ensureMask();
  for (Index v = 0; v < m.vertexCount(); ++v) m.mask[v] = m.positions[v].x > 0.3f ? 0.25f : 0.0f;
  const Mesh before = m;
  const Bvh beforeBvh = bvh;

  HandStroke s(m, bvh);
  std::mt19937 rng(11);
  CHECK(s.edit(glm::normalize(Vec3{0.2f, 0.9f, 0.1f}), 0.45f, 400, 0.35, rng) > 100);

  // A brush also moved a leaf the topology edits did not claim.
  Index moved = kInvalid;
  for (Index l = 0; l < bvh.firstTailLeaf() && moved == kInvalid; ++l)
    if (!s.rec.claimed(l) && !bvh.leaves()[l].empty()) moved = l;
  REQUIRE(moved != kInvalid);
  LeafState movedBefore;
  movedBefore.leaf = moved;
  const BvhLeaf movedLeaf = bvh.leaves()[moved];
  movedBefore.positions.assign(m.positions.begin() + movedLeaf.vertBegin, m.positions.begin() + movedLeaf.vertEnd);
  movedBefore.normals.assign(m.normals.begin() + movedLeaf.vertBegin, m.normals.begin() + movedLeaf.vertEnd);
  for (Index v = movedLeaf.vertBegin; v < movedLeaf.vertEnd; ++v) {
    m.positions[v] *= 1.01f;
    m.normals[v] = glm::normalize(m.normals[v] + Vec3{0.0f, 0.01f, 0.0f});
  }
  bvh.refit(m);
  requireDynamic(m, bvh);

  LayoutWorkspace ws;
  const ConsolidateResult r = consolidate(m, bvh, s.claimedLeaves(), ws);
  REQUIRE_FALSE(r.missedClaim);
  REQUIRE_FALSE(std::binary_search(r.delta.regionLeaves.begin(), r.delta.regionLeaves.end(), moved));
  requireLayout(m, bvh);
  const Mesh after = m;
  const Bvh afterBvh = bvh;

  const LayoutSide beforeLayout = beforeSide(s.startBvh, s.rec.takeClaims(), {movedBefore}, r, 1);
  const Index movedLeaves[] = {moved};
  const LayoutSide afterLayout = captureSide(m, bvh, r.delta, movedLeaves, 2);
  CHECK(beforeLayout.bytes() > 0);
  CHECK(beforeLayout.bytes() < before.positions.size() * 60);  // Far less than the whole mesh.

  for (int round = 0; round < 3; ++round) {
    CAPTURE(round);
    REQUIRE(relayout(m, bvh, beforeLayout, r.delta, false, ws));
    requireSameMesh(m, before);
    requireSameBvh(bvh, beforeBvh);
    requireLayout(m, bvh);
    REQUIRE(relayout(m, bvh, afterLayout, r.delta, true, ws));
    requireSameMesh(m, after);
    requireSameBvh(bvh, afterBvh);
  }

  // Applying a layout to a mesh that is not on the other side is refused.
  const Mesh current = m;
  Mesh other = makeIcosphere(2);
  Bvh otherBvh;
  otherBvh.build(other);
  CHECK_FALSE(relayout(other, otherBvh, beforeLayout, r.delta, false, ws));
  requireSameMesh(m, current);
}

TEST_CASE("a change the recorder never heard about is caught and every leaf rebuilt") {
  Mesh m = makeIcosphere(4);
  Bvh bvh;
  bvh.build(m, {.maxLeafFaces = 256});
  const Index euler = test::eulerCharacteristic(m);
  HandStroke s(m, bvh);
  std::mt19937 rng(3);
  const Vec3 c = glm::normalize(Vec3{1.0f, 0.2f, 0.1f});
  s.edit(c, 0.3f, 100, 0.5, rng);

  // An edit on the far side, made without the observer.
  s.ed->setObserver(nullptr);
  bool collapsed = false;
  for (Index h = 0; h < m.halfEdgeCount() && !collapsed; ++h) {
    if (m.heFace[h] == kInvalid || m.heVert[h] >= bvh.tailStartVertex()) continue;
    if (glm::dot(m.positions[m.heVert[h]], c) > -0.8f) continue;
    if (s.rec.claimed(bvh.leafOfHalfEdge(h))) continue;
    const Index b = m.heTarget(h);
    if (s.ed->collapseEdge(h, m.positions[m.heVert[h]])) {
      m.positions[b] = kDeadPosition;
      collapsed = true;
    }
  }
  REQUIRE(collapsed);
  bvh.refit(m);

  LayoutWorkspace ws;
  const ConsolidateResult r = consolidate(m, bvh, s.claimedLeaves(), ws);
  CHECK(r.missedClaim);
  test::requireValid(m);
  requireLayout(m, bvh);
  requireLeafLookups(m, bvh);
  CHECK(test::eulerCharacteristic(m) == euler);
}

TEST_CASE("a vertex no rebuilt face uses stays owned, bounded and findable") {
  Mesh m = makeIcosphere(3);
  Bvh bvh;
  bvh.build(m, {.maxLeafFaces = 128});
  // Hand leaf 0 the first vertex of leaf 1: leaf 0 then owns a vertex none of its faces use.
  std::vector<BvhLeaf> leaves(bvh.leaves().begin(), bvh.leaves().end());
  REQUIRE(leaves.size() > 3);
  const Index orphan = leaves[1].vertBegin;
  const Vec3 orphanPos = m.positions[orphan];
  leaves[0].vertEnd += 1;
  leaves[1].vertBegin += 1;
  leaves[0].flags |= kLeafOwnsOrphans;
  leaves[0].bounds.expand(orphanPos);
  bvh.setLeaves(leaves);
  requireLayout(m, bvh);
  const Mesh before = m;
  const Bvh beforeBvh = bvh;

  HandStroke s(m, bvh);
  s.rec.claimLeaf(0);  // Leaf 1, whose faces use the vertex, stays untouched.
  bvh.refit(m);
  requireDynamic(m, bvh);
  LayoutWorkspace ws;
  const ConsolidateResult r = consolidate(m, bvh, s.claimedLeaves(), ws);
  REQUIRE_FALSE(r.missedClaim);
  requireLayout(m, bvh);
  CHECK_FALSE(r.delta.vertRefs.empty());

  Index now = kInvalid;
  for (Index v = 0; v < m.vertexCount(); ++v)
    if (m.positions[v] == orphanPos) now = v;
  REQUIRE(now != kInvalid);
  const Index owner = bvh.leafOfVertex(now);
  REQUIRE(owner != kInvalid);
  CHECK((bvh.leaves()[owner].flags & kLeafOwnsOrphans) != 0);
  std::vector<Index> found;
  bvh.querySphere(orphanPos, 1e-4f, found);
  CHECK(std::find(found.begin(), found.end(), owner) != found.end());

  const Mesh after = m;
  const LayoutSide b = beforeSide(s.startBvh, s.rec.takeClaims(), {}, r, 1);
  const LayoutSide a = captureSide(m, bvh, r.delta, {}, 2);
  REQUIRE(relayout(m, bvh, b, r.delta, false, ws));
  requireSameMesh(m, before);
  requireSameBvh(bvh, beforeBvh);
  REQUIRE(relayout(m, bvh, a, r.delta, true, ws));
  requireSameMesh(m, after);
}

TEST_CASE("leaves emptied by coarsening are reused by later strokes") {
  Mesh m = makeIcosphere(4);
  Bvh bvh;
  bvh.build(m, {.maxLeafFaces = 256});
  LayoutWorkspace ws;
  std::mt19937 rng(5);
  const Vec3 c = glm::normalize(Vec3{0.3f, -0.8f, 0.5f});

  // Coarsen hard in one place.
  {
    HandStroke s(m, bvh);
    CHECK(s.edit(c, 0.6f, 2000, 1.0, rng) > 300);
    bvh.refit(m);
    requireDynamic(m, bvh);
    const ConsolidateResult r = consolidate(m, bvh, s.claimedLeaves(), ws);
    REQUIRE_FALSE(r.missedClaim);
  }
  requireLayout(m, bvh);
  requireLeafLookups(m, bvh);
  std::vector<Index> emptied;
  for (Index l = 0; l < static_cast<Index>(bvh.leaves().size()); ++l)
    if (bvh.leaves()[l].empty()) emptied.push_back(l);
  REQUIRE_FALSE(emptied.empty());

  // Refine hard elsewhere: the new leaves take the empty indices before growing the leaf list.
  const std::size_t leafCount = bvh.leaves().size();
  {
    HandStroke s(m, bvh);
    CHECK(s.edit(-c, 0.6f, 1200, 0.0, rng) > 500);
    bvh.refit(m);
    requireDynamic(m, bvh);
    for (Index l : emptied) CHECK_FALSE(s.rec.claimed(l));
    const ConsolidateResult r = consolidate(m, bvh, s.claimedLeaves(), ws);
    REQUIRE_FALSE(r.missedClaim);
    const std::size_t reused = static_cast<std::size_t>(std::count_if(emptied.begin(), emptied.end(), [&](Index l) {
      return std::binary_search(r.delta.regionLeaves.begin(), r.delta.regionLeaves.end(), l);
    }));
    CHECK(reused > 0);
    if (reused < emptied.size()) CHECK(bvh.leaves().size() == leafCount);
  }
  test::requireValid(m);
  requireLayout(m, bvh);
  requireLeafLookups(m, bvh);
  CHECK(test::eulerCharacteristic(m) == 2);
}

namespace {

// A hand-driven dynamic topology stroke on a scene object, ending the way Sculptor::endStroke
// does: consolidate, bump the version and build the undo entry.
DyntopoUndo handDyntopoStroke(SceneObject& obj, const Vec3& c, float r, double collapseShare, std::mt19937& rng,
                              LayoutWorkspace& ws) {
  const std::uint64_t version = obj.topologyVersion;
  HandStroke s(obj.mesh, obj.bvh);
  CHECK(s.edit(c, r, 300, collapseShare, rng) > 50);
  // Bring normals up to date, keeping the old normals of untouched leaves for undo, as the
  // sculptor does.
  Mesh& m = obj.mesh;
  for (Index v = 0; v < m.vertexCount(); ++v)
    if (m.vertHe[v] != kInvalid) m.computeNormals(v, v + 1);
  std::vector<LeafState> moved;
  for (Index l = 0; l < obj.bvh.firstTailLeaf(); ++l) {
    const BvhLeaf& leaf = obj.bvh.leaves()[l];
    if (s.rec.claimed(l) ||
        std::equal(m.normals.begin() + leaf.vertBegin, m.normals.begin() + leaf.vertEnd,
                   s.startMesh.normals.begin() + leaf.vertBegin))
      continue;
    LeafState state;
    state.leaf = l;
    state.positions.assign(m.positions.begin() + leaf.vertBegin, m.positions.begin() + leaf.vertEnd);
    state.normals.assign(s.startMesh.normals.begin() + leaf.vertBegin, s.startMesh.normals.begin() + leaf.vertEnd);
    moved.push_back(std::move(state));
  }
  obj.bvh.refit(m);
  requireDynamic(m, obj.bvh);
  const ConsolidateResult result = consolidate(m, obj.bvh, s.claimedLeaves(), ws);
  REQUIRE_FALSE(result.missedClaim);
  obj.topologyVersion = nextTopologyVersion();
  obj.clearDirty();
  DyntopoUndo undo;
  undo.label = "Dyntopo";
  undo.objectId = obj.id;
  undo.afterVersion = obj.topologyVersion;
  undo.delta = result.delta;
  undo.before = beforeSide(s.startBvh, s.rec.takeClaims(), std::move(moved), result, version);
  return undo;
}

struct ObjectState {
  Mesh mesh;
  Bvh bvh;
  std::uint64_t version = 0;
};

ObjectState stateOf(const SceneObject& obj) { return {obj.mesh, obj.bvh, obj.topologyVersion}; }

// Same mesh, BVH and version; an empty mask counts as all zeros, empty face sets as all default.
void requireState(const SceneObject& obj, const ObjectState& want) {
  CHECK(obj.topologyVersion == want.version);
  Mesh a = obj.mesh, b = want.mesh;
  if (a.mask.empty() != b.mask.empty()) {
    a.ensureMask();
    b.ensureMask();
  }
  if (a.faceSets.empty() != b.faceSets.empty()) {
    a.ensureFaceSets();
    b.ensureFaceSets();
  }
  requireSameMesh(a, b);
  requireSameBvh(obj.bvh, want.bvh);
}

}  // namespace

TEST_CASE("dynamic topology undo captures the after side on the first undo only") {
  Scene scene;
  SceneObject& obj = scene.add("sphere", makeQuadSphere(96));  // 55K faces, default leaves.
  LayoutWorkspace ws;
  std::mt19937 rng(21);
  const ObjectState start = stateOf(obj);
  DyntopoUndo entry = handDyntopoStroke(obj, glm::normalize(Vec3{0.3f, 0.9f, 0.2f}), 0.15f, 0.3, rng, ws);
  const ObjectState done = stateOf(obj);
  CHECK(entry.after == nullptr);
  const std::size_t stored = entry.bytes();
  const std::size_t whole = MeshState{start.mesh, start.bvh, start.version, nullptr}.bytes();
  CHECK(stored < whole / 5);

  UndoStack stack;
  stack.push(std::move(entry));
  CHECK(stack.bytes() == stored);
  for (int round = 0; round < 2; ++round) {
    CHECK(stack.undo(scene) == "Dyntopo");
    requireState(obj, start);
    CHECK(stack.bytes() > stored);  // The after side is held while the stroke can be redone.
    CHECK(stack.redo(scene) == "Dyntopo");
    requireState(obj, done);
    CHECK(stack.bytes() == stored);
  }
  CHECK(stack.undo(scene) == "Dyntopo");
  CHECK(stack.undo(scene).empty());
}

TEST_CASE("dynamic topology undo interleaves with sculpt, mask and remesh entries") {
  Scene scene;
  SceneObject& obj = scene.add("sphere", makeQuadSphere(64));
  LayoutWorkspace ws;
  std::mt19937 rng(22);
  UndoStack stack;
  const ObjectState initial = stateOf(obj);
  auto surface = [&](Vec3 dir) {
    dir = glm::normalize(dir);
    RayHit hit;
    REQUIRE(obj.bvh.raycast(obj.mesh, Ray{dir * 3.0f, -dir}, hit));
    return hit.position;
  };
  auto stroke = [&](const Brush& brush, Vec3 dir, const char* label) {
    Sculptor sculptor;
    sculptor.beginStroke(obj, brush, {}, label);
    const Vec3 c = surface(dir);
    for (int i = 0; i < 4; ++i) REQUIRE(sculptor.dab(c, 0.3f, 1.0f));
    auto undo = sculptor.endStroke();
    REQUIRE(undo);
    stack.push(std::move(*undo));
  };

  DrawBrush draw;
  MaskBrush mask;
  stroke(draw, {0.2f, 1.0f, 0.1f}, "S1");
  stroke(mask, {1.0f, 0.3f, 0.0f}, "M1");
  stack.push(handDyntopoStroke(obj, glm::normalize(Vec3{0.2f, 1.0f, 0.1f}), 0.35f, 0.4, rng, ws));
  stroke(draw, {0.2f, 1.0f, 0.1f}, "S2");
  stack.push(handDyntopoStroke(obj, glm::normalize(Vec3{0.9f, 0.4f, 0.2f}), 0.35f, 0.2, rng, ws));
  {
    auto remeshed = voxelRemesh(obj.mesh, {.voxelSize = 0.05f});
    REQUIRE(remeshed);
    TopologyUndo entry;
    entry.label = "Remesh";
    entry.objectId = obj.id;
    entry.before = std::make_shared<MeshState>(MeshState{obj.mesh, obj.bvh, obj.topologyVersion, nullptr});
    obj.mesh = std::move(*remeshed);
    obj.rebuildSpatial();
    entry.after = std::make_shared<MeshState>(MeshState{obj.mesh, obj.bvh, obj.topologyVersion, nullptr});
    stack.push(std::move(entry));
  }
  const ObjectState final = stateOf(obj);

  const char* labels[] = {"Remesh", "Dyntopo", "S2", "Dyntopo", "M1", "S1"};
  for (const char* label : labels) CHECK(stack.undo(scene) == label);
  requireState(obj, initial);
  for (int i = 5; i >= 0; --i) CHECK(stack.redo(scene) == labels[i]);
  requireState(obj, final);
  test::requireValid(obj.mesh);
}
