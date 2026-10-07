#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <glad/gl.h>

#include "Camera.h"
#include "render/LeafIndexBlocks.h"
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
  int indexBlockUploads = 0;       // Leaf index blocks rewritten this frame (dynamic topology).
  int bufferGrowths = 0;           // GPU buffers that ran out of room and were enlarged this frame.
  int drawRuns = 0;                // Index ranges drawn for triangles (1 when the mesh is at rest).
};

class Renderer {
 public:
  static constexpr const char* kMatcapNames[] = {"Clay", "Red Wax", "Studio Gray", "Jade"};
  static constexpr int kMatcapCount = 4;

  bool init(std::string* error);
  void shutdown();

  // Uploads changed meshes: full rebuild on topology change, per-leaf ranges otherwise. During a
  // dynamic topology stroke it rewrites the index blocks of the leaves whose faces changed.
  // Consumes each object's dirty geometry, mask and topology ranges. Edge (wireframe) indices
  // are only kept up to date while `wireframe` is on and rebuilt when it is turned on.
  void sync(Scene& scene, bool wireframe);

  void render(const Scene& scene, const Camera& camera, const ViewportRect& rect, const ViewSettings& settings,
              std::uint32_t selectedId, const CursorMarker& cursor);

  const RenderStats& stats() const { return stats_; }

 private:
  // An index buffer split into per-leaf blocks, drawn with one (multi-)draw call.
  struct IndexBuffer {
    GLuint buffer = 0;
    std::uint32_t capacity = 0;  // In indices.
    LeafIndexBlocks blocks;
    std::vector<GLsizei> counts;         // Runs to draw, rebuilt when blocks change.
    std::vector<const void*> offsets;
    bool runsStale = true;
  };
  struct GpuMesh {
    GLuint vao = 0, positions = 0, normals = 0, mask = 0;
    std::uint32_t vertexCapacity = 0;  // Vertices the buffers have room for; strokes append more.
    IndexBuffer triangles, edges;
    bool edgesValid = false;  // Edge blocks are kept up to date only while the wireframe is shown.
    bool maskOnGpu = false;   // The mask buffer may hold non-zero values.
    std::uint64_t topologyVersion = 0;
  };

  void upload(GpuMesh& gpu, const SceneObject& object, bool wireframe);
  void uploadIndices(IndexBuffer& ib, LeafIndexKind kind, const SceneObject& object);
  void syncLeaves(GpuMesh& gpu, IndexBuffer& ib, LeafIndexKind kind, const SceneObject& object,
                  std::span<const Index> leaves);
  void ensureVertexCapacity(GpuMesh& gpu, std::uint32_t vertices);
  void growIndexBuffer(GpuMesh& gpu, IndexBuffer& ib, std::uint32_t indices);
  void syncMask(GpuMesh& gpu, SceneObject& object);
  void draw(const IndexBuffer& ib, GLenum mode);
  void destroy(GpuMesh& gpu);

  GLuint meshProgram_ = 0, lineProgram_ = 0, backgroundProgram_ = 0;
  GLuint matcaps_[kMatcapCount] = {};
  GLuint emptyVao_ = 0;
  GLuint gridVao_ = 0, gridVbo_ = 0;
  GLsizei gridVertexCount_ = 0;
  GLuint cursorVao_ = 0, cursorVbo_ = 0;
  std::unordered_map<std::uint32_t, GpuMesh> meshes_;
  std::vector<std::uint32_t> scratch_;
  RenderStats stats_;
};

}  // namespace plegl
