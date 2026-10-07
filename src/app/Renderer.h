#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <glad/gl.h>

#include "Camera.h"
#include "scene/Scene.h"

namespace plegl {

struct ViewSettings {
  int matcap = 0;
  bool wireframe = false;
  float wireframeOpacity = 0.35f;
  bool grid = true;
  Vec3 backgroundTop{0.26f, 0.27f, 0.30f};
  Vec3 backgroundBottom{0.11f, 0.11f, 0.13f};
  bool showMask = true;
  float maskOpacity = 0.7f;  // How dark fully masked areas are drawn.
};

struct ViewportRect {
  int x = 0, y = 0, width = 1, height = 1;  // Framebuffer pixels, origin bottom-left.
};

struct CursorMarker {
  bool visible = false;
  Vec3 position{0.0f};
  Vec3 normal{0.0f, 1.0f, 0.0f};
  float radius = 0.05f;
};

struct RenderStats {
  std::size_t uploadedBytes = 0;   // This frame, partial updates included.
  int partialUploads = 0;          // Leaf ranges uploaded this frame.
  int fullUploads = 0;             // Meshes rebuilt this frame.
};

class Renderer {
 public:
  static constexpr const char* kMatcapNames[] = {"Clay", "Red Wax", "Studio Gray", "Jade"};
  static constexpr int kMatcapCount = 4;

  bool init(std::string* error);
  void shutdown();

  // Uploads changed meshes: full rebuild on topology change, per-leaf ranges otherwise.
  // Consumes each object's dirty geometry and mask ranges.
  void sync(Scene& scene);

  void render(const Scene& scene, const Camera& camera, const ViewportRect& rect, const ViewSettings& settings,
              std::uint32_t selectedId, const CursorMarker& cursor);

  const RenderStats& stats() const { return stats_; }

 private:
  struct GpuMesh {
    GLuint vao = 0, positions = 0, normals = 0, mask = 0, triangles = 0, edges = 0;
    bool maskOnGpu = false;  // The mask buffer may hold non-zero values.
    GLsizei triangleIndexCount = 0;
    GLsizei edgeIndexCount = 0;
    std::uint64_t topologyVersion = 0;
  };

  void upload(GpuMesh& gpu, const SceneObject& object);
  void syncMask(GpuMesh& gpu, SceneObject& object);
  void destroy(GpuMesh& gpu);

  GLuint meshProgram_ = 0, lineProgram_ = 0, backgroundProgram_ = 0;
  GLuint matcaps_[kMatcapCount] = {};
  GLuint emptyVao_ = 0;
  GLuint gridVao_ = 0, gridVbo_ = 0;
  GLsizei gridVertexCount_ = 0;
  GLuint cursorVao_ = 0, cursorVbo_ = 0;
  std::unordered_map<std::uint32_t, GpuMesh> meshes_;
  RenderStats stats_;
};

}  // namespace plegl
