// Headless engine benchmark. Prints one row per mesh size so results can be compared across
// commits and machines. Usage: plegl_bench [--quick]
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

#include "core/Parallel.h"
#include "core/Timer.h"
#include "io/Obj.h"
#include "mesh/Primitives.h"
#include "scene/Scene.h"
#include "sculpt/Sculptor.h"
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

  // Brush latency: one Draw stroke of 200 dabs along an arc, then the same with Smooth. A dab
  // includes the BVH query, brush, normals, refit, undo snapshots and dirty marking; GPU upload
  // and drawing are not included.
  std::printf("\nStroke benchmark (ms per dab, 200 dabs)\n\n");
  std::printf("%10s %8s %8s %10s %10s %10s %10s\n", "vertices", "radius", "brush", "verts/dab", "avg ms", "p95 ms",
              "max ms");
  std::vector<int> strokeRes = quick ? std::vector<int>{129} : std::vector<int>{289, 408};
  for (int res : strokeRes) {
    Scene scene;
    SceneObject& obj = scene.add("Sphere", makeQuadSphere(res));
    DrawBrush draw;
    SmoothBrush smooth;
    for (const Brush* brush : {static_cast<const Brush*>(&draw), static_cast<const Brush*>(&smooth)}) {
      for (float radius : {0.05f, 0.15f, 0.4f}) {
        Sculptor sculptor;
        sculptor.beginStroke(obj, *brush, {.strength = 0.5f}, brush->name());
        std::vector<double> times;
        double verts = 0;
        for (int i = 0; i < 200; ++i) {
          const float a = -0.8f + 1.6f * static_cast<float>(i) / 199.0f;
          const Vec3 dir = glm::normalize(Vec3{std::sin(a), 0.3f, std::cos(a)});
          RayHit hit;
          if (!obj.bvh.raycast(obj.mesh, Ray{dir * 3.0f, -dir}, hit)) continue;
          sculptor.dab(hit.position, radius, 0.5f);
          times.push_back(sculptor.lastDab().totalMs);
          verts += sculptor.lastDab().vertices;
        }
        sculptor.endStroke();
        obj.dirtyLeaves.clear();
        std::sort(times.begin(), times.end());
        double avg = 0;
        for (double t : times) avg += t;
        avg /= static_cast<double>(times.size());
        std::printf("%10d %8.2f %8s %10.0f %10.3f %10.3f %10.3f\n", obj.mesh.vertexCount(), radius, brush->name(),
                    verts / static_cast<double>(times.size()), avg, times.at(times.size() * 95 / 100), times.back());
      }
    }
  }
  return 0;
}
