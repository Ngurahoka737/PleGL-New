// Headless engine benchmark. Prints one row per mesh size so results can be compared across
// commits and machines. Usage: plegl_bench [--quick]
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "core/Parallel.h"
#include "core/Timer.h"
#include "io/Obj.h"
#include "mesh/Primitives.h"
#include "spatial/Bvh.h"

using namespace plegl;

int main(int argc, char** argv) {
  const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
  // Quad sphere resolutions giving roughly 50K, 100K, 500K and 1M vertices (6 * r^2 + 2).
  std::vector<int> resolutions = {91, 129, 289, 408};
  if (quick) resolutions = {91, 129};
  const int rays = 20000;

  std::printf("PleGL engine benchmark, %zu worker threads\n\n", workerCount());
  std::printf("%10s %10s %10s %10s %10s %11s %11s %10s %10s\n", "vertices", "build ms", "normals", "bvh ms",
              "leaves", "ray us avg", "query us", "refit ms", "obj w ms");

  for (int res : resolutions) {
    Timer t;
    Mesh mesh = makeQuadSphere(res);
    const double buildMs = t.ms();

    t.reset();
    mesh.computeNormals();
    const double normalsMs = t.ms();

    t.reset();
    Bvh bvh;
    bvh.build(mesh);
    const double bvhMs = t.ms();

    // Random rays from outside the unit sphere aimed near its centre: the cursor case.
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    int hits = 0;
    t.reset();
    for (int i = 0; i < rays; ++i) {
      Vec3 o{u(rng), u(rng), u(rng)};
      o = glm::normalize(o + Vec3{1e-3f}) * 4.0f;
      Ray ray{o, Vec3{u(rng), u(rng), u(rng)} * 0.3f - o};
      RayHit hit;
      hits += bvh.raycast(mesh, ray, hit);
    }
    const double rayUs = t.us() / rays;

    std::vector<Index> found;
    t.reset();
    for (int i = 0; i < 1000; ++i) {
      found.clear();
      bvh.querySphere(glm::normalize(Vec3{u(rng), u(rng), u(rng)}), 0.1f, found);
    }
    const double queryUs = t.us() / 1000;

    t.reset();
    bvh.refit(mesh);
    const double refitMs = t.ms();

    t.reset();
    const std::string obj = writeObj(mesh);
    const double objMs = t.ms();

    std::printf("%10d %10.1f %10.1f %10.1f %10zu %11.2f %11.2f %10.2f %10.1f\n", mesh.vertexCount(), buildMs,
                normalsMs, bvhMs, bvh.leaves().size(), rayUs, queryUs, refitMs, objMs);
    if (hits == 0) std::printf("  warning: no ray hit the mesh\n");
  }
  return 0;
}
