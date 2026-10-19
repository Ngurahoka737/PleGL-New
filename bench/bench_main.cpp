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
#include "remesh/QuadRemesh.h"
#include "remesh/VoxelRemesh.h"
#include "sculpt/MaskOps.h"
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

  // Brush latency: one stroke of 200 dabs along an arc per brush and radius. A dab includes the
  // BVH query, brush, normals, refit, undo snapshots and dirty marking; GPU upload and drawing
  // are not included. Grab is measured as 200 cursor moves after one capture.
  std::printf("\nStroke benchmark (ms per dab, 200 dabs)\n\n");
  std::printf("%10s %8s %8s %10s %10s %10s %10s\n", "vertices", "radius", "brush", "verts/dab", "avg ms", "p95 ms",
              "max ms");
  auto report = [](const SceneObject& obj, float radius, const char* name, std::vector<double>& times, double verts) {
    std::sort(times.begin(), times.end());
    double avg = 0;
    for (double t : times) avg += t;
    avg /= static_cast<double>(times.size());
    std::printf("%10d %8.2f %8s %10.0f %10.3f %10.3f %10.3f\n", obj.mesh.vertexCount(), radius, name,
                verts / static_cast<double>(times.size()), avg, times.at(times.size() * 95 / 100), times.back());
  };
  std::vector<int> strokeRes = quick ? std::vector<int>{129} : std::vector<int>{289, 408};
  for (int res : strokeRes) {
    DrawBrush draw;
    ClayBrush clay;
    SmoothBrush smooth;
    InflateBrush inflate;
    FlattenBrush flatten;
    CreaseBrush crease;
    MaskBrush mask;
    const Brush* brushes[] = {&draw, &clay, &smooth, &inflate, &flatten, &crease, &mask};
    for (const Brush* brush : brushes) {
      for (float radius : {0.05f, 0.15f, 0.4f}) {
        Scene scene;  // Fresh sphere per run so earlier strokes do not change the next.
        SceneObject& obj = scene.add("Sphere", makeQuadSphere(res));
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
        report(obj, radius, brush->name(), times, verts);
      }
    }
    for (float radius : {0.05f, 0.15f, 0.4f}) {
      Scene scene;
      SceneObject& obj = scene.add("Sphere", makeQuadSphere(res));
      Sculptor sculptor;
      if (!sculptor.beginGrab(obj, {.strength = 1.0f}, {0.0f, 0.0f, 1.0f}, radius, "Grab")) continue;
      std::vector<double> times;
      double verts = 0;
      for (int i = 0; i < 200; ++i) {
        const float t = static_cast<float>(i) / 199.0f;
        sculptor.grab(Vec3{0.3f * t, 0.1f * t, 0.5f * t});
        times.push_back(sculptor.lastDab().totalMs);
        verts += sculptor.lastDab().vertices;
      }
      sculptor.endStroke();
      report(obj, radius, "Grab", times, verts);
    }
  }
  // Whole-mesh mask operations, run on the main thread in the app. Blur and Sharpen use two
  // iterations (the app default). The undo entry is built inside the timed call.
  std::printf("\nMask operation benchmark (ms)\n\n");
  std::printf("%10s %10s %10s %10s %10s %10s\n", "vertices", "invert", "blur x2", "sharpen x2", "clear", "undo MB");
  for (int res : strokeRes) {
    Scene scene;
    SceneObject& obj = scene.add("Sphere", makeQuadSphere(res));
    double ms[4];
    std::size_t undoBytes = 0;
    const MaskOp ops[4] = {MaskOp::Invert, MaskOp::Blur, MaskOp::Sharpen, MaskOp::Clear};
    // Start from a half mask so blur and sharpen have an edge to work on.
    obj.mesh.mask.resize(obj.mesh.positions.size());
    for (Index v = 0; v < obj.mesh.vertexCount(); ++v) obj.mesh.mask[v] = obj.mesh.positions[v].y > 0.0f ? 1.0f : 0.0f;
    for (int i = 0; i < 4; ++i) {
      Timer t;
      auto entry = applyMaskOp(obj, ops[i], 2);
      ms[i] = t.ms();
      if (entry) undoBytes = std::max(undoBytes, entry->bytes());
    }
    std::printf("%10d %10.1f %10.1f %10.1f %10.1f %10.1f\n", obj.mesh.vertexCount(), ms[0], ms[1], ms[2], ms[3],
                static_cast<double>(undoBytes) / (1024.0 * 1024.0));
  }

  // Voxel remesh: whole pipeline (sign, narrow-band distance, Surface Nets, half-edge build) at
  // two voxel sizes. The PRD targets 100K < 1 s, 500K < 3 s, 1M < 5 s.
  std::printf("\nVoxel remesh benchmark\n\n");
  std::printf("%10s %8s %12s %10s %10s %10s %10s %10s %8s\n", "in verts", "voxel", "grid", "out verts", "grid ms",
              "extract ms", "build ms", "total ms", "MB");
  std::vector<int> remeshRes = quick ? std::vector<int>{129} : std::vector<int>{129, 289, 408};
  for (int res : remeshRes) {
    const Mesh in = makeQuadSphere(res);
    for (float voxel : {0.01f, 0.005f}) {
      VoxelRemeshStats st;
      std::string error;
      Timer t;
      auto out = voxelRemesh(in, {.voxelSize = voxel}, &st, &error);
      const double total = t.ms();
      if (!out) {
        std::printf("  remesh failed: %s\n", error.c_str());
        continue;
      }
      char grid[32];
      std::snprintf(grid, sizeof(grid), "%dx%dx%d", st.resolution[0], st.resolution[1], st.resolution[2]);
      std::printf("%10d %8.3f %12s %10d %10.0f %10.0f %10.0f %10.0f %8.0f\n", in.vertexCount(), voxel, grid,
                  out->vertexCount(), st.gridMs, st.extractMs, st.buildMs, total,
                  static_cast<double>(st.gridBytes) / (1024.0 * 1024.0));
    }
  }
  // Quad remesh: voxel remesh plus valence optimisation and relaxation onto the input. Same
  // PRD targets as the voxel remesh.
  std::printf("\nQuad remesh benchmark (masked: half the input masked, so the mask is transferred)\n\n");
  std::printf("%10s %8s %7s %10s %10s %10s %10s %8s %8s\n", "in verts", "edge", "masked", "out verts", "voxel ms",
              "optim ms", "total ms", "val4 %", "edge cv");
  for (int res : remeshRes) {
    Mesh in = makeQuadSphere(res);
    for (float edge : {0.01f, 0.005f, -0.01f}) {
      const bool masked = edge < 0.0f;  // Last row: the first edge again, with a mask.
      edge = std::abs(edge);
      if (masked) {
        in.mask.resize(in.positions.size());
        for (Index v = 0; v < in.vertexCount(); ++v) in.mask[v] = in.positions[v].y > 0.0f ? 1.0f : 0.0f;
      }
      QuadRemeshStats st;
      std::string error;
      Timer t;
      auto out = quadRemesh(in, {.targetEdge = edge}, &st, &error);
      const double total = t.ms();
      if (!out) {
        std::printf("  remesh failed: %s\n", error.c_str());
        continue;
      }
      std::printf("%10d %8.3f %7s %10d %10.0f %10.0f %10.0f %8.1f %8.3f\n", in.vertexCount(), edge,
                  masked ? "yes" : "no", out->vertexCount(), st.voxel.gridMs + st.voxel.extractMs + st.voxel.buildMs,
                  st.optimizeMs, total, 100.0 * st.optimized.valence4Ratio, st.optimized.edgeLengthCv);
    }
  }
  return 0;
}
