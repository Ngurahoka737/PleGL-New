#include "remesh/VoxelRemesh.h"

#include "core/Timer.h"
#include "remesh/SurfaceNets.h"
#include "remesh/VoxelGrid.h"

namespace plegl {

std::optional<Mesh> voxelRemesh(const Mesh& input, const VoxelRemeshParams& params, VoxelRemeshStats* stats,
                                std::string* error) {
  VoxelRemeshStats local;
  VoxelRemeshStats& st = stats ? *stats : local;
  Timer t;
  VoxelGrid grid;
  if (!grid.build(input, {params.voxelSize, params.maxResolution}, error)) return std::nullopt;
  st.resolution[0] = grid.nx();
  st.resolution[1] = grid.ny();
  st.resolution[2] = grid.nz();
  st.gridBytes = grid.memoryBytes();
  st.gridMs = t.ms();

  t.reset();
  SurfaceNetsMesh sn = extractSurfaceNets(grid);
  st.extractMs = t.ms();
  if (sn.quads.empty()) {
    if (error) *error = "No closed volume found. The mesh may be open, or thinner than the voxel size.";
    return std::nullopt;
  }

  t.reset();
  std::vector<Index> sizes(sn.quads.size() / 4, 4);
  Mesh out = buildMesh(std::move(sn.positions), sn.quads, sizes, &st.report);
  st.buildMs = t.ms();
  return out;
}

double meshVolume(const Mesh& mesh) {
  double v = 0.0;
  std::vector<Index> fv;
  for (Index f = 0; f < mesh.faceCount(); ++f) {
    fv.clear();
    mesh.forEachFaceVertex(f, [&](Index x) { fv.push_back(x); });
    const glm::dvec3 a(mesh.positions[fv[0]]);
    for (std::size_t i = 1; i + 1 < fv.size(); ++i) {
      const glm::dvec3 b(mesh.positions[fv[i]]), c(mesh.positions[fv[i + 1]]);
      v += glm::dot(a, glm::cross(b, c));
    }
  }
  return v / 6.0;
}

}  // namespace plegl
