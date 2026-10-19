#include <algorithm>
#include <cmath>

#include "TestUtil.h"
#include "mesh/Primitives.h"
#include "remesh/QuadRemesh.h"
#include "remesh/VoxelGrid.h"
#include "remesh/VoxelRemesh.h"
#include "scene/Scene.h"
#include "sculpt/Sculptor.h"
#include "sculpt/Undo.h"
#include "spatial/Bvh.h"

using namespace plegl;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Polygon soup of several meshes, optionally moved and with reversed faces.
struct Soup {
  std::vector<Vec3> positions;
  std::vector<Index> indices, sizes;

  void add(const Mesh& m, Vec3 offset = Vec3{0.0f}, bool reverse = false, Index skipFace = kInvalid) {
    const Index base = static_cast<Index>(positions.size());
    for (const Vec3& p : m.positions) positions.push_back(p + offset);
    std::vector<Index> fv;
    for (Index f = 0; f < m.faceCount(); ++f) {
      if (f == skipFace) continue;
      fv.clear();
      m.forEachFaceVertex(f, [&](Index v) { fv.push_back(base + v); });
      if (reverse) std::reverse(fv.begin(), fv.end());
      indices.insert(indices.end(), fv.begin(), fv.end());
      sizes.push_back(static_cast<Index>(fv.size()));
    }
  }
  Mesh build() const { return buildMesh(positions, indices, sizes); }
};

// Remeshes and checks what every result must satisfy: valid, closed, manifold, all quads,
// facing outward.
Mesh remeshClosed(const Mesh& in, float voxel) {
  VoxelRemeshStats st;
  std::string error;
  auto out = voxelRemesh(in, {.voxelSize = voxel}, &st, &error);
  INFO(error);
  REQUIRE(out);
  test::requireValid(*out);
  CHECK(st.report.nonManifoldEdges == 0);
  CHECK(st.report.nonManifoldVertices == 0);
  CHECK(st.report.degenerateFaces == 0);
  for (Index h = 0; h < out->halfEdgeCount(); ++h) REQUIRE(out->heTwin[h] != kInvalid);  // Closed.
  for (Index f = 0; f < out->faceCount(); ++f) REQUIRE(out->faceSize(f) == 4);
  CHECK(meshVolume(*out) > 0.0);
  return std::move(*out);
}

}  // namespace

TEST_CASE("voxel grid signs and distances match a sphere") {
  VoxelGrid grid;
  REQUIRE(grid.build(makeQuadSphere(64), {.voxelSize = 0.05f}));
  int checked = 0;
  for (int k = 0; k < grid.nz(); ++k)
    for (int j = 0; j < grid.ny(); ++j)
      for (int i = 0; i < grid.nx(); ++i) {
        const float r = glm::length(grid.nodePosition(i, j, k));
        if (std::abs(r - 1.0f) < 0.01f) continue;  // Too close to the faceted surface to judge.
        REQUIRE(grid.inside(i, j, k) == (r < 1.0f));
        if (std::abs(r - 1.0f) < 0.08f) {
          // Distance in voxels; the sphere is faceted, so allow a little slack.
          CHECK(std::abs(grid.distance(i, j, k) * 0.05f - (r - 1.0f)) < 0.004f);
          ++checked;
        }
      }
  CHECK(checked > 1000);
}

TEST_CASE("voxel remesh of a sphere is closed, all quads and keeps the volume") {
  const Mesh in = makeQuadSphere(48);
  const Mesh out = remeshClosed(in, 0.04f);
  const double v0 = meshVolume(in), v1 = meshVolume(out);
  CHECK(std::abs(v1 - v0) / v0 < 0.02);
  test::requireOutward(out);
  test::requireValid(out);
  CHECK(test::eulerCharacteristic(out) == 2);  // Still a sphere.
  for (const Vec3& p : out.positions) CHECK(std::abs(glm::length(p) - 1.0f) < 0.02f);
  // Plain Surface Nets on a curved surface gives roughly half valence-4 vertices; the valence
  // optimisation phase raises this. Guard the baseline so it does not regress.
  Index v4 = 0;
  for (Index v = 0; v < out.vertexCount(); ++v) {
    int valence = 0;
    out.forEachOutgoing(v, [&](Index) { ++valence; });
    v4 += valence == 4;
  }
  CHECK(static_cast<double>(v4) / out.vertexCount() > 0.45);
}

TEST_CASE("voxel remesh evens out density regardless of the input topology") {
  // A UV sphere has tiny faces at the poles and large ones at the equator; the remesh does not.
  const Mesh out = remeshClosed(makeUvSphere(64, 32), 0.05f);
  double minLen = 1e9, maxLen = 0;
  for (Index h = 0; h < out.halfEdgeCount(); ++h) {
    const double l = glm::length(out.positions[out.heTarget(h)] - out.positions[out.heVert[h]]);
    minLen = std::min(minLen, l);
    maxLen = std::max(maxLen, l);
  }
  CHECK(maxLen < 0.05 * 1.8);
  CHECK(minLen > 0.0);
}

TEST_CASE("voxel remesh merges overlapping parts into one surface") {
  Soup soup;
  soup.add(makeQuadSphere(48), {-0.5f, 0.0f, 0.0f});
  soup.add(makeQuadSphere(48), {0.5f, 0.0f, 0.0f});
  const Mesh out = remeshClosed(soup.build(), 0.04f);
  CHECK(test::eulerCharacteristic(out) == 2);  // One closed surface, no inner walls.
  // Union of two unit spheres one unit apart: two balls minus the lens they share.
  const double ball = 4.0 / 3.0 * kPi;
  const double lens = kPi * (4.0 + 1.0) * 1.0 / 12.0;
  const double expected = 2.0 * ball - lens;
  CHECK(std::abs(meshVolume(out) - expected) / expected < 0.03);
}

TEST_CASE("voxel remesh ignores face orientation and closes small holes") {
  Soup soup;
  soup.add(makeQuadSphere(32), Vec3{0.0f}, /*reverse=*/true, /*skipFace=*/100);
  const Mesh out = remeshClosed(soup.build(), 0.05f);
  test::requireOutward(out);
  CHECK(test::eulerCharacteristic(out) == 2);
}

TEST_CASE("voxel remesh keeps a cube's shape") {
  const Mesh in = makeCube(8);
  const Mesh out = remeshClosed(in, 0.04f);
  const double v0 = meshVolume(in), v1 = meshVolume(out);
  CHECK(std::abs(v1 - v0) / v0 < 0.02);
}

TEST_CASE("voxel remesh refuses grids over the limit") {
  std::string error;
  CHECK_FALSE(voxelRemesh(makeQuadSphere(8), {.voxelSize = 0.001f, .maxResolution = 256}, nullptr, &error));
  CHECK(error.find("too small") != std::string::npos);
}

TEST_CASE("undoing a remesh restores the mesh and earlier sculpt undo still applies") {
  Scene scene;
  SceneObject& obj = scene.add("Sphere", makeQuadSphere(24));
  UndoStack stack;

  // A sculpt stroke, then a remesh on top of it.
  DrawBrush draw;
  Sculptor sculptor;
  sculptor.beginStroke(obj, draw, {}, "Draw");
  RayHit hit;
  REQUIRE(obj.bvh.raycast(obj.mesh, Ray{{0, 3, 0}, {0, -1, 0}}, hit));
  REQUIRE(sculptor.dab(hit.position, 0.4f, 1.0f));
  stack.push(*sculptor.endStroke());
  const Mesh sculpted = obj.mesh;
  const std::uint64_t sculptedVersion = obj.topologyVersion;

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
  CHECK(obj.topologyVersion != sculptedVersion);
  const Index remeshedVerts = obj.mesh.vertexCount();

  CHECK(stack.undo(scene) == "Remesh");
  CHECK(obj.topologyVersion == sculptedVersion);
  CHECK(obj.mesh.positions == sculpted.positions);
  CHECK(stack.undo(scene) == "Draw");  // Applies again because the topology is back.
  CHECK(stack.redo(scene) == "Draw");
  CHECK(stack.redo(scene) == "Remesh");
  CHECK(obj.mesh.vertexCount() == remeshedVerts);
  RayHit after;
  CHECK(obj.bvh.raycast(obj.mesh, Ray{{0, 3, 0}, {0, -1, 0}}, after));
}

namespace {

void requireQuadRemeshOk(const Mesh& out) {
  test::requireValid(out);
  for (Index h = 0; h < out.halfEdgeCount(); ++h) REQUIRE(out.heTwin[h] != kInvalid);
  for (Index f = 0; f < out.faceCount(); ++f) REQUIRE(out.faceSize(f) == 4);
  for (Index v = 0; v < out.vertexCount(); ++v) REQUIRE(out.valence(v) >= 3);
}

void printQuality(const std::string& name, const MeshQuality& voxel, const QuadRemeshStats& st) {
  MESSAGE(name << ": voxel v4 " << voxel.valence4Ratio << " cv " << voxel.edgeLengthCv << " err " << voxel.meanError
               << " -> quad v4 " << st.optimized.valence4Ratio << " cv " << st.optimized.edgeLengthCv << " err "
               << st.optimized.meanError << "/" << st.optimized.maxError << " edge " << st.optimized.meanEdge
               << " collapsed " << st.collapsed << " rotated " << st.rotated << " ms " << st.optimizeMs);
}

}  // namespace

TEST_CASE("quad remesh reaches 98% valence 4 and fits the surface better than the voxel remesh") {
  struct Case {
    const char* name;
    Mesh mesh;
    float h;
    bool sharp;  // Has sharp edges, which the coarse layout rounds a little more.
  };
  Soup two;
  two.add(makeQuadSphere(48), {-0.5f, 0.0f, 0.0f});
  two.add(makeQuadSphere(48), {0.5f, 0.0f, 0.0f});
  std::vector<Case> cases;
  cases.push_back({"sphere", makeQuadSphere(48), 0.04f, false});
  cases.push_back({"uv sphere", makeUvSphere(64, 32), 0.05f, false});
  cases.push_back({"cube", makeCube(16), 0.06f, true});
  cases.push_back({"two spheres", two.build(), 0.04f, false});
  for (const Case& c : cases) {
    INFO(c.name);
    Mesh ref = c.mesh;
    Bvh refBvh;
    refBvh.build(ref);
    const Mesh voxel = *voxelRemesh(c.mesh, {.voxelSize = c.h});
    const MeshQuality vq = measureQuality(voxel, &ref, &refBvh);

    QuadRemeshStats st;
    std::string error;
    auto out = quadRemesh(c.mesh, {.targetEdge = c.h, .measureError = true}, &st, &error);
    INFO(error);
    REQUIRE(out);
    printQuality(c.name, vq, st);
    requireQuadRemeshOk(*out);
    CHECK(test::eulerCharacteristic(*out) == 2);
    test::requireOutward(*out);
    // Against the voxel result: the input of "two spheres" counts the overlap twice.
    const double v0 = meshVolume(voxel), v1 = meshVolume(*out);
    CHECK(std::abs(v1 - v0) / v0 < 0.01);
    CHECK(st.optimized.valence4Ratio >= 0.98);
    CHECK(st.optimized.edgeLengthCv < vq.edgeLengthCv);
    CHECK(st.optimized.meanEdge == doctest::Approx(c.h).epsilon(0.15));
    if (c.sharp) {
      CHECK(st.optimized.meanError < 2.0 * vq.meanError);
    } else {
      CHECK(st.optimized.meanError < vq.meanError);
    }
  }
}

TEST_CASE("quad remesh without subdivision still improves valence") {
  const Mesh in = makeQuadSphere(48);
  QuadRemeshStats st;
  auto out = quadRemesh(in, {.targetEdge = 0.04f, .subdivisions = 0}, &st);
  REQUIRE(out);
  requireQuadRemeshOk(*out);
  CHECK(st.raw.valence4Ratio < 0.6);
  CHECK(st.optimized.valence4Ratio > 0.95);
  QuadRemeshStats plain;
  auto voxelOnly = quadRemesh(in, {.targetEdge = 0.04f, .rounds = 0}, &plain);
  REQUIRE(voxelOnly);
  CHECK(plain.optimized.valence4Ratio < 0.6);  // rounds = 0 is the plain voxel remesh.
}
