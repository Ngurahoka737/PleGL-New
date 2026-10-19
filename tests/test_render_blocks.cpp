#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <random>
#include <set>

#include "TestUtil.h"
#include "mesh/Primitives.h"
#include "render/LeafIndexBlocks.h"
#include "scene/Scene.h"
#include "sculpt/Sculptor.h"

using namespace plegl;

namespace {

// What Renderer::upload() drew before indices were split per leaf: one walk over everything.
std::vector<std::uint32_t> globalTriangles(const Mesh& m) {
  std::vector<std::uint32_t> out;
  for (Index f = 0; f < m.faceCount(); ++f) {
    const Index h0 = m.faceHe[f];
    for (Index h = m.heNext[h0]; m.heNext[h] != h0; h = m.heNext[h])
      out.insert(out.end(), {static_cast<std::uint32_t>(m.heVert[h0]), static_cast<std::uint32_t>(m.heVert[h]),
                             static_cast<std::uint32_t>(m.heTarget(h))});
  }
  return out;
}

std::vector<std::uint32_t> globalEdges(const Mesh& m) {
  std::vector<std::uint32_t> out;
  for (Index h = 0; h < m.halfEdgeCount(); ++h)
    if (m.heTwin[h] == kInvalid || h < m.heTwin[h])
      out.insert(out.end(), {static_cast<std::uint32_t>(m.heVert[h]), static_cast<std::uint32_t>(m.heTarget(h))});
  return out;
}

using Tri = std::array<std::uint32_t, 3>;

// Triangles as rotation-normalised corner triples, so the fan start does not matter.
std::multiset<Tri> triangleSet(const std::vector<std::uint32_t>& idx) {
  std::multiset<Tri> out;
  for (std::size_t i = 0; i + 2 < idx.size(); i += 3) {
    Tri t{idx[i], idx[i + 1], idx[i + 2]};
    std::rotate(t.begin(), std::min_element(t.begin(), t.end()), t.end());
    out.insert(t);
  }
  return out;
}

// Every live face of a mesh that may hold removed elements, fan-triangulated.
std::multiset<Tri> liveTriangles(const Mesh& m) {
  std::vector<std::uint32_t> idx;
  for (Index f = 0; f < m.faceCount(); ++f) {
    const Index h0 = m.faceHe[f];
    if (h0 == kInvalid) continue;
    for (Index h = m.heNext[h0]; m.heNext[h] != h0; h = m.heNext[h])
      idx.insert(idx.end(), {static_cast<std::uint32_t>(m.heVert[h0]), static_cast<std::uint32_t>(m.heVert[h]),
                             static_cast<std::uint32_t>(m.heTarget(h))});
  }
  return triangleSet(idx);
}

// Every live edge once, as a sorted vertex pair.
std::multiset<std::pair<std::uint32_t, std::uint32_t>> edgeSet(const std::vector<std::uint32_t>& idx) {
  std::multiset<std::pair<std::uint32_t, std::uint32_t>> out;
  for (std::size_t i = 0; i + 1 < idx.size(); i += 2) out.insert(std::minmax(idx[i], idx[i + 1]));
  return out;
}

std::multiset<std::pair<std::uint32_t, std::uint32_t>> liveEdges(const Mesh& m) {
  std::multiset<std::pair<std::uint32_t, std::uint32_t>> out;
  for (Index h = 0; h < m.halfEdgeCount(); ++h) {
    if (m.heFace[h] == kInvalid) continue;
    const Index t = m.heTwin[h];
    if (t == kInvalid || h < t) out.insert(std::minmax(static_cast<std::uint32_t>(m.heVert[h]),
                                                       static_cast<std::uint32_t>(m.heTarget(h))));
  }
  return out;
}

std::vector<std::vector<std::uint32_t>> perLeaf(LeafIndexKind kind, const SceneObject& obj) {
  std::vector<std::vector<std::uint32_t>> out(obj.bvh.leaves().size());
  for (std::size_t l = 0; l < out.size(); ++l) appendLeafIndices(kind, obj.mesh, obj.bvh.leaves()[l], out[l]);
  return out;
}

Vec3 surfacePoint(const SceneObject& obj, Vec3 dir) {
  dir = glm::normalize(dir);
  RayHit hit;
  const bool ok = obj.bvh.raycast(obj.mesh, Ray{dir * 4.0f, -dir}, hit);
  REQUIRE(ok);
  return hit.position;
}

Index faceNear(const SceneObject& obj, const Vec3& p, float radius) {
  Bvh::ClosestHit hit;
  return obj.bvh.closestPoint(obj.mesh, p, radius, hit) ? hit.face : kInvalid;
}

// The renderer's index buffer on the CPU: packed on upload, then rewritten leaf by leaf.
struct CpuIndexBuffer {
  LeafIndexBlocks blocks;
  std::vector<std::uint32_t> data;

  void upload(LeafIndexKind kind, const SceneObject& obj) { data = buildLeafIndices(kind, obj.mesh, obj.bvh, blocks); }

  void sync(LeafIndexKind kind, const SceneObject& obj, std::span<const Index> leaves) {
    std::vector<std::uint32_t> scratch;
    for (Index l : leaves) {
      scratch.clear();
      appendLeafIndices(kind, obj.mesh, obj.bvh.leaves()[l], scratch);
      blocks.place(l, static_cast<std::uint32_t>(scratch.size()));
      if (data.size() < blocks.size()) data.resize(blocks.size(), 0xFFFFFFFFu);  // Growth keeps the old data.
      std::copy(scratch.begin(), scratch.end(), data.begin() + blocks.block(l).first);
    }
  }

  // What a multi-draw over the runs reads.
  std::vector<std::uint32_t> drawn() const {
    std::vector<std::uint32_t> first, count, out;
    blocks.runs(first, count);
    for (std::size_t i = 0; i < first.size(); ++i)
      out.insert(out.end(), data.begin() + first[i], data.begin() + first[i] + count[i]);
    return out;
  }
};

}  // namespace

TEST_CASE("per-leaf indices at rest are exactly the old whole-mesh lists, drawn as one run") {
  for (Mesh mesh : {makeQuadSphere(24), makeIcosphere(4), makePlane(16), makeUvSphere(24, 12)}) {
    Scene scene;
    SceneObject& obj = scene.add("A", std::move(mesh));
    LeafIndexBlocks tri, edge;
    CHECK(buildLeafIndices(LeafIndexKind::Triangles, obj.mesh, obj.bvh, tri) == globalTriangles(obj.mesh));
    CHECK(buildLeafIndices(LeafIndexKind::Edges, obj.mesh, obj.bvh, edge) == globalEdges(obj.mesh));
    std::vector<std::uint32_t> first, count;
    tri.runs(first, count);
    REQUIRE(first.size() == 1);
    CHECK(first[0] == 0);
    CHECK(count[0] == tri.size());
    CHECK(tri.consistent());
    CHECK(edge.consistent());
  }
}

TEST_CASE("leaf blocks stay put while they fit, move with slack when they do not, and never overlap") {
  LeafIndexBlocks b;
  b.pack(std::vector<std::uint32_t>{30, 0, 60, 90});
  CHECK(b.size() == 180);
  // Shrinking or refilling in place keeps the block.
  CHECK_FALSE(b.place(2, 45));
  CHECK(b.block(2).first == 30);
  CHECK_FALSE(b.place(2, 60));
  // Outgrowing it moves the block to the end, with room to spare.
  CHECK(b.place(0, 31));
  CHECK(b.block(0).first == 180);
  CHECK(b.block(0).capacity > 31);
  // The freed range is reused by the next block that fits in it (5 indices plus slack < 30).
  CHECK(b.place(5, 5));
  CHECK(b.block(5).first == 0);
  CHECK(b.leafCount() == 6);
  CHECK(b.consistent());

  std::mt19937 rng(7);
  for (int i = 0; i < 5000; ++i) {
    const auto leaf = static_cast<Index>(rng() % 64);
    const std::uint32_t before = leaf < static_cast<Index>(b.leafCount()) ? b.block(leaf).first : 0;
    const std::uint32_t cap = leaf < static_cast<Index>(b.leafCount()) ? b.block(leaf).capacity : 0;
    const auto count = static_cast<std::uint32_t>(rng() % 400);
    const bool moved = b.place(leaf, count);
    if (count <= cap) CHECK(b.block(leaf).first == before);
    if (count <= cap) CHECK_FALSE(moved);
    if (i % 97 == 0) {
      INFO("step " << i);
      REQUIRE(b.consistent());
    }
  }
  CHECK(b.consistent());
  // Runs cover exactly the used part of every block.
  std::vector<std::uint32_t> first, count;
  b.runs(first, count);
  std::uint64_t covered = 0;
  for (std::uint32_t c : count) covered += c;
  CHECK(covered == b.used());
  b.truncate(10);
  CHECK(b.leafCount() == 10);
  CHECK(b.consistent());
}

TEST_CASE("mid-stroke leaf indices cover every live triangle and edge exactly once") {
  Scene scene;
  SceneObject& obj = scene.add("A", makeQuadSphere(20));
  DrawBrush draw;
  Sculptor sculptor;
  StrokeOptions options;
  options.dyntopo = true;
  options.dyntopoOptions.timeBudgetMs = 0.0;
  options.symmetryX = true;
  sculptor.beginStroke(obj, draw, options, "Draw");
  for (int i = 0; i < 10; ++i) {
    const Vec3 c = surfacePoint(obj, {-0.5f + 0.1f * static_cast<float>(i), 1.0f, 0.2f});
    // Alternate refining and coarsening.
    sculptor.dab(c, 0.35f, 0.5f, DabTopology{i % 3 == 2 ? 0.3f : 0.03f, faceNear(obj, c, 0.35f)});
    std::vector<std::uint32_t> tris, edges;
    for (const BvhLeaf& leaf : obj.bvh.leaves()) {
      appendLeafTriangles(obj.mesh, leaf, tris);
      appendLeafEdges(obj.mesh, leaf, edges);
    }
    INFO("dab " << i);
    CHECK(triangleSet(tris) == liveTriangles(obj.mesh));
    CHECK(edgeSet(edges) == liveEdges(obj.mesh));
  }
  REQUIRE(sculptor.lastDab().splits + sculptor.dabCount() > 0);
  sculptor.endStroke();
}

TEST_CASE("leaves a dab does not mark keep their indices, and every changed vertex is marked") {
  Scene scene;
  Mesh mesh = makeQuadSphere(20);
  mesh.ensureMask();
  for (Index v = 0; v < mesh.vertexCount(); ++v) mesh.mask[v] = std::clamp(mesh.positions[v].z, 0.0f, 0.45f);
  SceneObject& obj = scene.add("A", std::move(mesh));
  DrawBrush draw;
  Sculptor sculptor;
  StrokeOptions options;
  options.dyntopo = true;
  options.dyntopoOptions.timeBudgetMs = 0.0;
  options.symmetryX = true;
  sculptor.beginStroke(obj, draw, options, "Draw");
  CpuIndexBuffer gpuTris, gpuEdges;
  gpuTris.upload(LeafIndexKind::Triangles, obj);
  gpuEdges.upload(LeafIndexKind::Edges, obj);
  int topoDabs = 0;
  for (int i = 0; i < 12; ++i) {
    INFO("dab " << i);
    const auto trisBefore = perLeaf(LeafIndexKind::Triangles, obj);
    const auto edgesBefore = perLeaf(LeafIndexKind::Edges, obj);
    const Mesh before = obj.mesh;
    obj.clearDirty();
    const Vec3 c = surfacePoint(obj, {-0.4f + 0.08f * static_cast<float>(i), 1.0f, 0.1f * static_cast<float>(i % 3)});
    sculptor.dab(c, 0.3f, 0.5f, DabTopology{i % 4 == 3 ? 0.3f : 0.03f, faceNear(obj, c, 0.3f)});
    topoDabs += sculptor.lastDab().splits + sculptor.lastDab().collapses > 0;

    std::set<Index> topo(obj.topoDirtyLeaves.begin(), obj.topoDirtyLeaves.end());
    std::set<Index> verts(obj.dirtyLeaves.begin(), obj.dirtyLeaves.end());
    const auto trisAfter = perLeaf(LeafIndexKind::Triangles, obj);
    const auto edgesAfter = perLeaf(LeafIndexKind::Edges, obj);
    int stale = 0;
    for (std::size_t l = 0; l < trisAfter.size(); ++l) {
      if (topo.count(static_cast<Index>(l))) continue;
      if (l >= trisBefore.size()) {
        ++stale;  // A new leaf must be marked.
        continue;
      }
      stale += trisAfter[l] != trisBefore[l];
      stale += edgesAfter[l] != edgesBefore[l];
    }
    CHECK(stale == 0);
    // Moved, re-shaded or new vertices are in a vertex-dirty leaf; new mask values in a
    // topology-dirty one.
    int unmarked = 0;
    const Mesh& m = obj.mesh;
    for (Index v = 0; v < m.vertexCount(); ++v) {
      if (m.vertHe[v] == kInvalid) continue;  // Removed: nothing draws it.
      const bool added = v >= before.vertexCount();
      const Index owner = obj.bvh.leafOfVertex(v);
      if ((added || m.positions[v] != before.positions[v] || m.normals[v] != before.normals[v]) && !verts.count(owner))
        ++unmarked;
      if ((added || m.mask[v] != before.mask[v]) && !topo.count(owner)) ++unmarked;
    }
    CHECK(unmarked == 0);

    // The renderer's buffers, updated for the marked leaves only, draw the current mesh.
    gpuTris.sync(LeafIndexKind::Triangles, obj, obj.topoDirtyLeaves);
    gpuEdges.sync(LeafIndexKind::Edges, obj, obj.topoDirtyLeaves);
    CHECK(gpuTris.blocks.consistent());
    CHECK(triangleSet(gpuTris.drawn()) == liveTriangles(m));
    CHECK(edgeSet(gpuEdges.drawn()) == liveEdges(m));
  }
  CHECK(topoDabs > 6);
  sculptor.endStroke();
}

TEST_CASE("refining and then merging under a moving brush keeps every drawn edge current") {
  // A merge rewrites where the edges coming into the removed vertex end. After earlier splits
  // those edges are often drawn by a leaf that holds neither their face nor the merged vertex.
  int topoDabs = 0;
  for (int seed = 1; seed <= 6; ++seed) {
    INFO("seed " << seed);
    Scene scene;
    SceneObject& obj = scene.add("A", makeQuadSphere(16));
    DrawBrush draw;
    Sculptor sculptor;
    StrokeOptions options;
    options.dyntopo = true;
    options.dyntopoOptions.timeBudgetMs = 0.0;
    sculptor.beginStroke(obj, draw, options, "Draw");
    CpuIndexBuffer gpuTris, gpuEdges;
    gpuTris.upload(LeafIndexKind::Triangles, obj);
    gpuEdges.upload(LeafIndexKind::Edges, obj);
    float t = 1.3f * static_cast<float>(seed);
    for (int i = 0; i < 160; ++i) {
      INFO("dab " << i);
      // A zigzag over the same area, with the detail drifting finer and coarser as in Relative mode.
      t += 0.05f;
      const Vec3 c = surfacePoint(obj, {-0.6f + 0.25f * std::sin(t * 1.7f), 0.8f, 0.2f + 0.25f * std::sin(t * 0.9f)});
      const float detail = 0.04f + 0.025f * std::sin(t * 0.6f);
      const auto trisBefore = perLeaf(LeafIndexKind::Triangles, obj);
      const auto edgesBefore = perLeaf(LeafIndexKind::Edges, obj);
      obj.clearDirty();
      sculptor.dab(c, 0.2f, 0.5f, DabTopology{detail, faceNear(obj, c, 0.2f)});
      topoDabs += sculptor.lastDab().splits + sculptor.lastDab().collapses > 0;
      const std::set<Index> topo(obj.topoDirtyLeaves.begin(), obj.topoDirtyLeaves.end());
      const auto trisAfter = perLeaf(LeafIndexKind::Triangles, obj);
      const auto edgesAfter = perLeaf(LeafIndexKind::Edges, obj);
      int stale = 0;
      for (std::size_t l = 0; l < trisBefore.size(); ++l) {
        if (topo.count(static_cast<Index>(l))) continue;
        stale += trisAfter[l] != trisBefore[l];
        stale += edgesAfter[l] != edgesBefore[l];
      }
      REQUIRE(stale == 0);
      gpuTris.sync(LeafIndexKind::Triangles, obj, obj.topoDirtyLeaves);
      gpuEdges.sync(LeafIndexKind::Edges, obj, obj.topoDirtyLeaves);
      if (i % 20 == 19) {  // The whole-mesh comparison is slow; the per-leaf check above is not.
        REQUIRE(edgeSet(gpuEdges.drawn()) == liveEdges(obj.mesh));
        REQUIRE(triangleSet(gpuTris.drawn()) == liveTriangles(obj.mesh));
      }
    }
    sculptor.endStroke();
  }
  CHECK(topoDabs > 300);
}
