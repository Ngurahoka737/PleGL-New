#include "Renderer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_set>

#include <glm/gtc/type_ptr.hpp>

namespace plegl {
namespace {

// ----- Shaders --------------------------------------------------------------------------------

const char* kMeshVs = R"(#version 450 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 3) in float aMask;  // Location 2 is the line shader's vertex colour.
layout(location = 4) in uint aFirstIndex;  // Per draw command: where its run of triangles starts.
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform mat3 uNormalMat;  // view * model, inverse-transposed
out vec3 vNormalView;
out float vMask;
flat out uint vFirstSlot;
void main() {
  vNormalView = uNormalMat * aNormal;
  vMask = aMask;
  vFirstSlot = aFirstIndex / 3u;
  gl_Position = uProj * uView * uModel * vec4(aPosition, 1.0);
}
)";

const char* kMeshFs = R"(#version 450 core
in vec3 vNormalView;
in float vMask;
flat in uint vFirstSlot;
layout(std430, binding = 0) readonly buffer FaceSets { int faceSet[]; };  // Per triangle slot.
uniform sampler2D uMatcap;
uniform float uSelected;
uniform float uMaskOpacity;
uniform float uFaceSetOpacity;  // 0 when face sets are off or the mesh has none.
out vec4 fragColor;
// A distinct pastel per set: consecutive ids step around the hue circle by the golden ratio.
vec3 faceSetColor(int id) {
  float hue = fract(float(id) * 0.618034 + 0.12);
  float sat = 0.45 + 0.3 * fract(float(id) * 0.381966);
  vec3 rgb = clamp(abs(mod(hue * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
  return mix(vec3(1.0), rgb, sat);
}
void main() {
  vec3 n = normalize(vNormalView);
  if (!gl_FrontFacing) n = -n;  // Open meshes: shade the inside too.
  vec3 color = texture(uMatcap, n.xy * 0.49 + 0.5).rgb;
  if (uFaceSetOpacity > 0.0) {
    int id = abs(faceSet[vFirstSlot + uint(gl_PrimitiveID)]);
    if (id > 1) color *= mix(vec3(1.0), faceSetColor(id), uFaceSetOpacity);  // Set 1 is the default.
  }
  color *= 1.0 - uMaskOpacity * clamp(vMask, 0.0, 1.0);  // Masked areas read as darker.
  // Selected objects get a soft rim so they read without an outline pass.
  float rim = pow(1.0 - clamp(n.z, 0.0, 1.0), 3.0);
  color += uSelected * rim * vec3(1.0, 0.55, 0.15) * 0.6;
  fragColor = vec4(color, 1.0);
}
)";

const char* kLineVs = R"(#version 450 core
layout(location = 0) in vec3 aPosition;
layout(location = 2) in vec4 aColor;
uniform mat4 uMvp;
uniform vec4 uColor;
uniform float uUseVertexColor;
out vec4 vColor;
void main() {
  vColor = mix(uColor, aColor, uUseVertexColor);
  gl_Position = uMvp * vec4(aPosition, 1.0);
}
)";

const char* kLineFs = R"(#version 450 core
in vec4 vColor;
out vec4 fragColor;
void main() { fragColor = vColor; }
)";

const char* kBackgroundVs = R"(#version 450 core
out float vT;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);  // Fullscreen triangle.
  vT = p.y * 0.5;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";

const char* kBackgroundFs = R"(#version 450 core
in float vT;
uniform vec3 uTop;
uniform vec3 uBottom;
out vec4 fragColor;
void main() { fragColor = vec4(mix(uBottom, uTop, clamp(vT, 0.0, 1.0)), 1.0); }
)";

GLuint compile(GLenum type, const char* src, std::string* error) {
  const GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[2048];
    glGetShaderInfoLog(s, sizeof(log), nullptr, log);
    if (error) *error = std::string("Shader compile failed: ") + log;
    glDeleteShader(s);
    return 0;
  }
  return s;
}

GLuint link(const char* vs, const char* fs, std::string* error) {
  const GLuint v = compile(GL_VERTEX_SHADER, vs, error);
  const GLuint f = v ? compile(GL_FRAGMENT_SHADER, fs, error) : 0;
  if (!v || !f) return 0;
  const GLuint p = glCreateProgram();
  glAttachShader(p, v);
  glAttachShader(p, f);
  glLinkProgram(p);
  glDeleteShader(v);
  glDeleteShader(f);
  GLint ok = 0;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[2048];
    glGetProgramInfoLog(p, sizeof(log), nullptr, log);
    if (error) *error = std::string("Program link failed: ") + log;
    glDeleteProgram(p);
    return 0;
  }
  return p;
}

GLint loc(GLuint program, const char* name) { return glGetUniformLocation(program, name); }

// ----- Procedural matcaps ---------------------------------------------------------------------

struct MatcapStyle {
  Vec3 base;
  Vec3 rim;
  float specular;
  float shininess;
};

GLuint makeMatcap(const MatcapStyle& style) {
  constexpr int kSize = 256;
  std::vector<std::uint8_t> pixels(kSize * kSize * 4);
  const Vec3 key = glm::normalize(Vec3{-0.45f, 0.65f, 0.62f});
  const Vec3 fill = glm::normalize(Vec3{0.7f, -0.2f, 0.5f});
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < kSize; ++x) {
      const float u = (static_cast<float>(x) + 0.5f) / kSize * 2.0f - 1.0f;
      const float v = (static_cast<float>(y) + 0.5f) / kSize * 2.0f - 1.0f;
      const float r2 = std::min(u * u + v * v, 1.0f);
      const Vec3 n{u, v, std::sqrt(1.0f - r2)};
      const float wrap = std::pow(std::max((glm::dot(n, key) + 0.2f) / 1.2f, 0.0f), 1.3f);
      const float fillL = std::max(glm::dot(n, fill), 0.0f) * 0.25f;
      const Vec3 h = glm::normalize(key + Vec3{0.0f, 0.0f, 1.0f});
      const float spec = std::pow(std::max(glm::dot(n, h), 0.0f), style.shininess) * style.specular;
      const float rim = std::pow(1.0f - n.z, 2.5f) * 0.35f;
      // Cavity-like darkening toward grazing angles keeps forms readable.
      const float occl = 0.45f + 0.55f * n.z;
      Vec3 c = style.base * (0.10f + 0.95f * wrap + fillL) * occl + Vec3{spec} + style.rim * rim;
      c = glm::clamp(c, Vec3{0.0f}, Vec3{1.0f});
      std::uint8_t* px = &pixels[(y * kSize + x) * 4];
      px[0] = static_cast<std::uint8_t>(std::pow(c.x, 1.0f / 1.1f) * 255.0f);
      px[1] = static_cast<std::uint8_t>(std::pow(c.y, 1.0f / 1.1f) * 255.0f);
      px[2] = static_cast<std::uint8_t>(std::pow(c.z, 1.0f / 1.1f) * 255.0f);
      px[3] = 255;
    }
  }
  GLuint tex = 0;
  glCreateTextures(GL_TEXTURE_2D, 1, &tex);
  glTextureStorage2D(tex, 1, GL_RGBA8, kSize, kSize);
  glTextureSubImage2D(tex, 0, 0, 0, kSize, kSize, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
  glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  return tex;
}

struct LineVertex {
  Vec3 position;
  Vec4 color;
};

GLuint makeLineVao(GLuint vbo) {
  GLuint vao = 0;
  glCreateVertexArrays(1, &vao);
  glVertexArrayVertexBuffer(vao, 0, vbo, 0, sizeof(LineVertex));
  glEnableVertexArrayAttrib(vao, 0);
  glVertexArrayAttribFormat(vao, 0, 3, GL_FLOAT, GL_FALSE, offsetof(LineVertex, position));
  glVertexArrayAttribBinding(vao, 0, 0);
  glEnableVertexArrayAttrib(vao, 2);
  glVertexArrayAttribFormat(vao, 2, 4, GL_FLOAT, GL_FALSE, offsetof(LineVertex, color));
  glVertexArrayAttribBinding(vao, 2, 0);
  return vao;
}

constexpr int kCursorSegments = 48;

}  // namespace

bool Renderer::init(std::string* error) {
  meshProgram_ = link(kMeshVs, kMeshFs, error);
  lineProgram_ = link(kLineVs, kLineFs, error);
  backgroundProgram_ = link(kBackgroundVs, kBackgroundFs, error);
  if (!meshProgram_ || !lineProgram_ || !backgroundProgram_) return false;

  const MatcapStyle styles[kMatcapCount] = {
      {{0.80f, 0.64f, 0.52f}, {0.9f, 0.8f, 0.7f}, 0.10f, 18.0f},  // Clay
      {{0.72f, 0.20f, 0.16f}, {1.0f, 0.6f, 0.5f}, 0.35f, 40.0f},  // Red wax
      {{0.62f, 0.63f, 0.66f}, {0.8f, 0.85f, 1.0f}, 0.20f, 30.0f}, // Studio gray
      {{0.30f, 0.62f, 0.48f}, {0.7f, 1.0f, 0.9f}, 0.45f, 60.0f},  // Jade
  };
  for (int i = 0; i < kMatcapCount; ++i) matcaps_[i] = makeMatcap(styles[i]);

  glCreateVertexArrays(1, &emptyVao_);

  // Ground grid on XZ: 1 unit major lines, 0.25 unit minor lines, colored axes.
  std::vector<LineVertex> grid;
  const int half = 40;  // In quarter units: covers [-10, 10].
  for (int i = -half; i <= half; ++i) {
    const float c = static_cast<float>(i) * 0.25f;
    const float extent = static_cast<float>(half) * 0.25f;
    Vec4 color = (i % 4 == 0) ? Vec4{0.5f, 0.5f, 0.55f, 0.35f} : Vec4{0.5f, 0.5f, 0.55f, 0.12f};
    Vec4 colorX = color, colorZ = color;
    if (i == 0) {
      colorX = {0.85f, 0.25f, 0.25f, 0.8f};  // Line along X at z = 0.
      colorZ = {0.25f, 0.45f, 0.9f, 0.8f};   // Line along Z at x = 0.
    }
    grid.push_back({{-extent, 0.0f, c}, colorX});
    grid.push_back({{extent, 0.0f, c}, colorX});
    grid.push_back({{c, 0.0f, -extent}, colorZ});
    grid.push_back({{c, 0.0f, extent}, colorZ});
  }
  gridVertexCount_ = static_cast<GLsizei>(grid.size());
  glCreateBuffers(1, &gridVbo_);
  glNamedBufferStorage(gridVbo_, static_cast<GLsizeiptr>(grid.size() * sizeof(LineVertex)), grid.data(), 0);
  gridVao_ = makeLineVao(gridVbo_);

  glCreateBuffers(1, &cursorVbo_);
  glNamedBufferStorage(cursorVbo_, (kCursorSegments + 2) * sizeof(LineVertex), nullptr, GL_DYNAMIC_STORAGE_BIT);
  cursorVao_ = makeLineVao(cursorVbo_);
  return true;
}

void Renderer::shutdown() {
  for (auto& [id, gpu] : meshes_) destroy(gpu);
  meshes_.clear();
  glDeleteTextures(kMatcapCount, matcaps_);
  glDeleteBuffers(1, &gridVbo_);
  glDeleteBuffers(1, &cursorVbo_);
  glDeleteVertexArrays(1, &gridVao_);
  glDeleteVertexArrays(1, &cursorVao_);
  glDeleteVertexArrays(1, &emptyVao_);
  glDeleteProgram(meshProgram_);
  glDeleteProgram(lineProgram_);
  glDeleteProgram(backgroundProgram_);
}

void Renderer::destroy(GpuMesh& gpu) {
  const GLuint buffers[] = {gpu.positions,     gpu.normals,  gpu.mask,    gpu.triangles.buffer,
                            gpu.edges.buffer, gpu.commands, gpu.faceSets};
  glDeleteBuffers(static_cast<GLsizei>(std::size(buffers)), buffers);
  glDeleteVertexArrays(1, &gpu.vao);
  gpu = {};
}

namespace {

// A new buffer of `newBytes` holding the first `keepBytes` of the old one, copied on the GPU.
GLuint regrow(GLuint old, std::size_t keepBytes, std::size_t newBytes) {
  GLuint fresh = 0;
  glCreateBuffers(1, &fresh);
  glNamedBufferStorage(fresh, static_cast<GLsizeiptr>(newBytes), nullptr, GL_DYNAMIC_STORAGE_BIT);
  if (old != 0 && keepBytes > 0) glCopyNamedBufferSubData(old, fresh, 0, 0, static_cast<GLsizeiptr>(keepBytes));
  glDeleteBuffers(1, &old);
  return fresh;
}

// Room for the vertices a dynamic topology stroke appends, so most strokes never regrow.
std::uint32_t vertexRoom(std::size_t vertices) {
  return static_cast<std::uint32_t>(vertices + std::max<std::size_t>(65536, vertices / 4));
}

// Triangle slots (index / 3) an index buffer of `indices` can hold.
std::uint32_t slotsFor(std::uint32_t indices) { return (indices + 2) / 3; }

}  // namespace

void Renderer::upload(GpuMesh& gpu, const SceneObject& object, bool wireframe) {
  destroy(gpu);
  const Mesh& m = object.mesh;
  const std::size_t nv = m.positions.size();
  gpu.vertexCapacity = vertexRoom(nv);
  const std::size_t vbytes = nv * sizeof(Vec3);
  const std::size_t vcap = static_cast<std::size_t>(gpu.vertexCapacity) * sizeof(Vec3);
  glCreateBuffers(1, &gpu.positions);
  glNamedBufferStorage(gpu.positions, static_cast<GLsizeiptr>(vcap), nullptr, GL_DYNAMIC_STORAGE_BIT);
  glCreateBuffers(1, &gpu.normals);
  glNamedBufferStorage(gpu.normals, static_cast<GLsizeiptr>(vcap), nullptr, GL_DYNAMIC_STORAGE_BIT);
  if (vbytes > 0) {
    glNamedBufferSubData(gpu.positions, 0, static_cast<GLsizeiptr>(vbytes), m.positions.data());
    glNamedBufferSubData(gpu.normals, 0, static_cast<GLsizeiptr>(vbytes), m.normals.data());
  }
  // One float per vertex; an unmasked mesh gets a zero-filled buffer so the shader can always read it.
  const bool hasMask = !m.mask.empty();
  glCreateBuffers(1, &gpu.mask);
  glNamedBufferStorage(gpu.mask, static_cast<GLsizeiptr>(gpu.vertexCapacity * sizeof(float)), nullptr,
                       GL_DYNAMIC_STORAGE_BIT);
  glClearNamedBufferData(gpu.mask, GL_R32F, GL_RED, GL_FLOAT, nullptr);
  if (hasMask) glNamedBufferSubData(gpu.mask, 0, static_cast<GLsizeiptr>(nv * sizeof(float)), m.mask.data());
  gpu.maskOnGpu = hasMask;

  glCreateVertexArrays(1, &gpu.vao);
  glVertexArrayVertexBuffer(gpu.vao, 0, gpu.positions, 0, sizeof(Vec3));
  glVertexArrayVertexBuffer(gpu.vao, 1, gpu.normals, 0, sizeof(Vec3));
  glEnableVertexArrayAttrib(gpu.vao, 0);
  glVertexArrayAttribFormat(gpu.vao, 0, 3, GL_FLOAT, GL_FALSE, 0);
  glVertexArrayAttribBinding(gpu.vao, 0, 0);
  glEnableVertexArrayAttrib(gpu.vao, 1);
  glVertexArrayAttribFormat(gpu.vao, 1, 3, GL_FLOAT, GL_FALSE, 0);
  glVertexArrayAttribBinding(gpu.vao, 1, 1);
  glVertexArrayVertexBuffer(gpu.vao, 2, gpu.mask, 0, sizeof(float));
  glEnableVertexArrayAttrib(gpu.vao, 3);
  glVertexArrayAttribFormat(gpu.vao, 3, 1, GL_FLOAT, GL_FALSE, 0);
  glVertexArrayAttribBinding(gpu.vao, 3, 2);

  // Indices per BVH leaf, back to back, so a dynamic topology stroke can rewrite single leaves.
  uploadIndices(gpu.triangles, LeafIndexKind::Triangles, object);
  glVertexArrayElementBuffer(gpu.vao, gpu.triangles.buffer);
  gpu.edgesValid = wireframe;
  if (wireframe) uploadIndices(gpu.edges, LeafIndexKind::Edges, object);

  // Draw commands, also read per command as the first index of its run (see GpuMesh). The
  // attribute is enabled for every draw with this VAO, so the buffer always holds a command.
  gpu.commandCapacity = 16;
  glCreateBuffers(1, &gpu.commands);
  glNamedBufferStorage(gpu.commands, gpu.commandCapacity * sizeof(DrawCommand), nullptr, GL_DYNAMIC_STORAGE_BIT);
  glClearNamedBufferData(gpu.commands, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
  glVertexArrayVertexBuffer(gpu.vao, 3, gpu.commands, offsetof(DrawCommand, firstIndex), sizeof(DrawCommand));
  glVertexArrayBindingDivisor(gpu.vao, 3, 1);
  glEnableVertexArrayAttrib(gpu.vao, 4);
  glVertexArrayAttribIFormat(gpu.vao, 4, 1, GL_UNSIGNED_INT, 0);
  glVertexArrayAttribBinding(gpu.vao, 4, 3);
  gpu.triangles.runsStale = true;

  // Face sets: the real buffer only for meshes that have some; others get a one-slot stand-in
  // that the shader never reads.
  if (m.hasFaceSetData()) {
    uploadFaceSets(gpu, object);
  } else {
    gpu.faceSetSlots = 1;
    glCreateBuffers(1, &gpu.faceSets);
    glNamedBufferStorage(gpu.faceSets, 4, nullptr, GL_DYNAMIC_STORAGE_BIT);
    gpu.faceSetsOnGpu = false;
  }

  gpu.topologyVersion = object.topologyVersion;
  stats_.uploadedBytes += vbytes * 2 + m.mask.size() * sizeof(float);
  ++stats_.fullUploads;
}

void Renderer::uploadIndices(IndexBuffer& ib, LeafIndexKind kind, const SceneObject& object) {
  glDeleteBuffers(1, &ib.buffer);
  const std::vector<std::uint32_t> indices = buildLeafIndices(kind, object.mesh, object.bvh, ib.blocks);
  // Some slack for the blocks a stroke moves or appends; running out grows the buffer on the GPU.
  // Whole triangles, so the face set buffer covers every slot.
  ib.capacity = static_cast<std::uint32_t>((indices.size() + indices.size() / 8 + 4096 + 2) / 3 * 3);
  glCreateBuffers(1, &ib.buffer);
  glNamedBufferStorage(ib.buffer, static_cast<GLsizeiptr>(ib.capacity) * 4, nullptr, GL_DYNAMIC_STORAGE_BIT);
  if (!indices.empty())
    glNamedBufferSubData(ib.buffer, 0, static_cast<GLsizeiptr>(indices.size() * 4), indices.data());
  ib.runsStale = true;
  stats_.uploadedBytes += indices.size() * 4;
}

void Renderer::ensureVertexCapacity(GpuMesh& gpu, std::uint32_t vertices) {
  if (vertices <= gpu.vertexCapacity) return;
  const std::uint32_t cap = std::max(vertices, gpu.vertexCapacity + gpu.vertexCapacity / 2);
  const std::size_t oldCap = gpu.vertexCapacity;
  gpu.positions = regrow(gpu.positions, oldCap * sizeof(Vec3), cap * sizeof(Vec3));
  gpu.normals = regrow(gpu.normals, oldCap * sizeof(Vec3), cap * sizeof(Vec3));
  gpu.mask = regrow(gpu.mask, oldCap * sizeof(float), cap * sizeof(float));
  // The copy keeps the old values; the new tail of the mask must read as unmasked.
  glClearNamedBufferSubData(gpu.mask, GL_R32F, static_cast<GLintptr>(oldCap * sizeof(float)),
                            static_cast<GLsizeiptr>((cap - oldCap) * sizeof(float)), GL_RED, GL_FLOAT, nullptr);
  glVertexArrayVertexBuffer(gpu.vao, 0, gpu.positions, 0, sizeof(Vec3));
  glVertexArrayVertexBuffer(gpu.vao, 1, gpu.normals, 0, sizeof(Vec3));
  glVertexArrayVertexBuffer(gpu.vao, 2, gpu.mask, 0, sizeof(float));
  gpu.vertexCapacity = cap;
  ++stats_.bufferGrowths;
}

void Renderer::growIndexBuffer(GpuMesh& gpu, IndexBuffer& ib, std::uint32_t indices) {
  if (indices <= ib.capacity) return;
  const std::uint32_t cap = (std::max(indices, ib.capacity + ib.capacity / 2) + 2) / 3 * 3;
  ib.buffer = regrow(ib.buffer, static_cast<std::size_t>(ib.capacity) * 4, static_cast<std::size_t>(cap) * 4);
  ib.capacity = cap;
  if (&ib == &gpu.triangles) {
    glVertexArrayElementBuffer(gpu.vao, ib.buffer);
    if (gpu.faceSetsOnGpu) {
      const std::uint32_t slots = slotsFor(cap);
      gpu.faceSets = regrow(gpu.faceSets, static_cast<std::size_t>(gpu.faceSetSlots) * 4, static_cast<std::size_t>(slots) * 4);
      gpu.faceSetSlots = slots;
    }
  }
  ++stats_.bufferGrowths;
}

void Renderer::syncLeaves(GpuMesh& gpu, IndexBuffer& ib, LeafIndexKind kind, const SceneObject& object,
                          std::span<const Index> leaves) {
  const auto all = object.bvh.leaves();
  for (Index l : leaves) {
    if (l < 0 || static_cast<std::size_t>(l) >= all.size()) continue;
    scratch_.clear();
    appendLeafIndices(kind, object.mesh, all[l], scratch_);
    ib.blocks.place(l, static_cast<std::uint32_t>(scratch_.size()));
    growIndexBuffer(gpu, ib, ib.blocks.size());
    if (!scratch_.empty()) {
      const auto bytes = static_cast<GLsizeiptr>(scratch_.size() * 4);
      glNamedBufferSubData(ib.buffer, static_cast<GLintptr>(ib.blocks.block(l).first) * 4, bytes, scratch_.data());
      stats_.uploadedBytes += static_cast<std::size_t>(bytes);
    }
    if (kind == LeafIndexKind::Triangles && gpu.faceSetsOnGpu) {
      faceSetScratch_.clear();
      appendLeafTriangleFaceSets(object.mesh, all[l], faceSetScratch_);
      if (!faceSetScratch_.empty()) {
        const auto bytes = static_cast<GLsizeiptr>(faceSetScratch_.size() * 4);
        glNamedBufferSubData(gpu.faceSets, static_cast<GLintptr>(ib.blocks.block(l).first / 3) * 4, bytes,
                             faceSetScratch_.data());
        stats_.uploadedBytes += static_cast<std::size_t>(bytes);
      }
    }
    ++stats_.indexBlockUploads;
  }
  ib.runsStale = true;
}

void Renderer::uploadFaceSets(GpuMesh& gpu, const SceneObject& object) {
  glDeleteBuffers(1, &gpu.faceSets);
  gpu.faceSetSlots = std::max<std::uint32_t>(slotsFor(gpu.triangles.capacity), 1);
  glCreateBuffers(1, &gpu.faceSets);
  glNamedBufferStorage(gpu.faceSets, static_cast<GLsizeiptr>(gpu.faceSetSlots) * 4, nullptr, GL_DYNAMIC_STORAGE_BIT);
  const std::vector<std::int32_t> slots = buildTriangleFaceSets(object.mesh, object.bvh, gpu.triangles.blocks);
  if (!slots.empty())
    glNamedBufferSubData(gpu.faceSets, 0, static_cast<GLsizeiptr>(slots.size() * 4), slots.data());
  gpu.faceSetsOnGpu = true;
  stats_.uploadedBytes += slots.size() * 4;
  ++stats_.faceSetUploads;
}

void Renderer::syncFaceSets(GpuMesh& gpu, SceneObject& obj) {
  const Mesh& m = obj.mesh;
  if ((obj.faceSetDirtyAll || !obj.faceSetDirtyLeaves.empty()) && !m.faceSets.empty()) {
    if (!gpu.faceSetsOnGpu || obj.faceSetDirtyAll) {
      uploadFaceSets(gpu, obj);
    } else {
      std::sort(obj.faceSetDirtyLeaves.begin(), obj.faceSetDirtyLeaves.end());
      obj.faceSetDirtyLeaves.erase(std::unique(obj.faceSetDirtyLeaves.begin(), obj.faceSetDirtyLeaves.end()),
                                   obj.faceSetDirtyLeaves.end());
      const auto leaves = obj.bvh.leaves();
      for (Index l : obj.faceSetDirtyLeaves) {
        if (l < 0 || static_cast<std::size_t>(l) >= leaves.size() ||
            static_cast<std::size_t>(l) >= gpu.triangles.blocks.leafCount())
          continue;
        faceSetScratch_.clear();
        appendLeafTriangleFaceSets(m, leaves[l], faceSetScratch_);
        const auto& block = gpu.triangles.blocks.block(l);
        // A paint stroke never changes which triangles a leaf draws; hiding does, and rewrites the
        // leaf's slots along with its indices.
        if (faceSetScratch_.empty() || faceSetScratch_.size() * 3 != block.count) continue;
        const auto bytes = static_cast<GLsizeiptr>(faceSetScratch_.size() * 4);
        glNamedBufferSubData(gpu.faceSets, static_cast<GLintptr>(block.first / 3) * 4, bytes, faceSetScratch_.data());
        stats_.uploadedBytes += static_cast<std::size_t>(bytes);
        ++stats_.faceSetUploads;
      }
    }
  }
  obj.faceSetDirtyLeaves.clear();
  obj.faceSetDirtyAll = false;
}

void Renderer::uploadCommands(GpuMesh& gpu) {
  const IndexBuffer& ib = gpu.triangles;
  commandScratch_.clear();
  for (std::size_t i = 0; i < ib.counts.size(); ++i) {
    const auto first = static_cast<GLuint>(reinterpret_cast<std::uintptr_t>(ib.offsets[i]) / 4);
    commandScratch_.push_back({static_cast<GLuint>(ib.counts[i]), 1, first, 0, static_cast<GLuint>(i)});
  }
  if (commandScratch_.size() > gpu.commandCapacity) {
    gpu.commandCapacity = static_cast<std::uint32_t>(std::max(commandScratch_.size(), std::size_t{gpu.commandCapacity} * 2));
    glDeleteBuffers(1, &gpu.commands);
    glCreateBuffers(1, &gpu.commands);
    glNamedBufferStorage(gpu.commands, gpu.commandCapacity * sizeof(DrawCommand), nullptr, GL_DYNAMIC_STORAGE_BIT);
    glClearNamedBufferData(gpu.commands, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
    glVertexArrayVertexBuffer(gpu.vao, 3, gpu.commands, offsetof(DrawCommand, firstIndex), sizeof(DrawCommand));
  }
  if (!commandScratch_.empty())
    glNamedBufferSubData(gpu.commands, 0, static_cast<GLsizeiptr>(commandScratch_.size() * sizeof(DrawCommand)),
                         commandScratch_.data());
}

void Renderer::syncMask(GpuMesh& gpu, SceneObject& obj) {
  const Mesh& m = obj.mesh;
  if (m.mask.empty()) {
    if (gpu.maskOnGpu) glClearNamedBufferData(gpu.mask, GL_R32F, GL_RED, GL_FLOAT, nullptr);
    gpu.maskOnGpu = false;
  } else if (obj.maskDirtyAll) {
    const std::size_t bytes = m.mask.size() * sizeof(float);
    glNamedBufferSubData(gpu.mask, 0, static_cast<GLsizeiptr>(bytes), m.mask.data());
    stats_.uploadedBytes += bytes;
    ++stats_.partialUploads;
    gpu.maskOnGpu = true;
  } else if (!obj.maskDirtyLeaves.empty()) {
    std::sort(obj.maskDirtyLeaves.begin(), obj.maskDirtyLeaves.end());
    obj.maskDirtyLeaves.erase(std::unique(obj.maskDirtyLeaves.begin(), obj.maskDirtyLeaves.end()),
                              obj.maskDirtyLeaves.end());
    const auto leaves = obj.bvh.leaves();
    for (Index l : obj.maskDirtyLeaves) {
      const BvhLeaf& leaf = leaves[l];
      const auto bytes = static_cast<GLsizeiptr>((leaf.vertEnd - leaf.vertBegin) * sizeof(float));
      if (bytes == 0) continue;
      glNamedBufferSubData(gpu.mask, static_cast<GLintptr>(leaf.vertBegin * sizeof(float)), bytes,
                           &m.mask[leaf.vertBegin]);
      stats_.uploadedBytes += static_cast<std::size_t>(bytes);
      ++stats_.partialUploads;
    }
    gpu.maskOnGpu = true;
  }
  obj.maskDirtyLeaves.clear();
  obj.maskDirtyAll = false;
}

void Renderer::sync(Scene& scene, bool wireframe) {
  stats_ = {};
  std::unordered_set<std::uint32_t> alive;
  for (const auto& objPtr : scene.objects()) {
    SceneObject& obj = *objPtr;
    alive.insert(obj.id);
    GpuMesh& gpu = meshes_[obj.id];
    if (gpu.vao == 0 || gpu.topologyVersion != obj.topologyVersion) {
      upload(gpu, obj, wireframe);
      obj.clearDirty();
      continue;
    }
    const Mesh& m = obj.mesh;
    ensureVertexCapacity(gpu, static_cast<std::uint32_t>(m.positions.size()));
    const std::size_t leafCount = obj.bvh.leaves().size();
    if (gpu.triangles.blocks.leafCount() > leafCount) {  // Defensive: leaves only grow mid-stroke.
      gpu.triangles.blocks.truncate(leafCount);
      gpu.edges.blocks.truncate(leafCount);
      gpu.triangles.runsStale = gpu.edges.runsStale = true;
    }
    if (!obj.topoDirtyLeaves.empty()) {
      std::sort(obj.topoDirtyLeaves.begin(), obj.topoDirtyLeaves.end());
      obj.topoDirtyLeaves.erase(std::unique(obj.topoDirtyLeaves.begin(), obj.topoDirtyLeaves.end()),
                                obj.topoDirtyLeaves.end());
      if (obj.topoDirtyLeaves.size() > 16 && obj.topoDirtyLeaves.size() * 2 > leafCount) {
        // Most leaves changed (revealing everything, say): one packed rebuild beats moving each
        // block to the end of a growing buffer.
        uploadIndices(gpu.triangles, LeafIndexKind::Triangles, obj);
        glVertexArrayElementBuffer(gpu.vao, gpu.triangles.buffer);
        if (gpu.faceSetsOnGpu) uploadFaceSets(gpu, obj);
        gpu.edgesValid = false;  // Rebuilt below when the wireframe is shown.
        ++stats_.fullUploads;
      } else {
        syncLeaves(gpu, gpu.triangles, LeafIndexKind::Triangles, obj, obj.topoDirtyLeaves);
        if (gpu.edgesValid && wireframe) {
          syncLeaves(gpu, gpu.edges, LeafIndexKind::Edges, obj, obj.topoDirtyLeaves);
        } else {
          gpu.edgesValid = false;  // Rebuilt in one go when the wireframe is turned on.
        }
      }
      // New vertices carry interpolated mask values that no mask stroke announced.
      if (!m.mask.empty()) {
        const auto leaves = obj.bvh.leaves();
        for (Index l : obj.topoDirtyLeaves) {
          if (l < 0 || static_cast<std::size_t>(l) >= leaves.size()) continue;
          const BvhLeaf& leaf = leaves[l];
          const auto bytes = static_cast<GLsizeiptr>((leaf.vertEnd - leaf.vertBegin) * sizeof(float));
          if (bytes == 0) continue;
          glNamedBufferSubData(gpu.mask, static_cast<GLintptr>(leaf.vertBegin * sizeof(float)), bytes,
                               &m.mask[leaf.vertBegin]);
          stats_.uploadedBytes += static_cast<std::size_t>(bytes);
        }
        gpu.maskOnGpu = true;
      }
      obj.topoDirtyLeaves.clear();
    }
    if (wireframe && !gpu.edgesValid) {
      uploadIndices(gpu.edges, LeafIndexKind::Edges, obj);
      gpu.edgesValid = true;
    }
    syncMask(gpu, obj);
    syncFaceSets(gpu, obj);
    if (obj.dirtyLeaves.empty()) continue;
    std::sort(obj.dirtyLeaves.begin(), obj.dirtyLeaves.end());
    obj.dirtyLeaves.erase(std::unique(obj.dirtyLeaves.begin(), obj.dirtyLeaves.end()), obj.dirtyLeaves.end());
    const auto leaves = obj.bvh.leaves();
    for (Index l : obj.dirtyLeaves) {
      if (l < 0 || static_cast<std::size_t>(l) >= leaves.size()) continue;
      const BvhLeaf& leaf = leaves[l];
      const auto offset = static_cast<GLintptr>(leaf.vertBegin * sizeof(Vec3));
      const auto bytes = static_cast<GLsizeiptr>((leaf.vertEnd - leaf.vertBegin) * sizeof(Vec3));
      if (bytes == 0) continue;
      glNamedBufferSubData(gpu.positions, offset, bytes, &m.positions[leaf.vertBegin]);
      glNamedBufferSubData(gpu.normals, offset, bytes, &m.normals[leaf.vertBegin]);
      stats_.uploadedBytes += static_cast<std::size_t>(bytes) * 2;
      ++stats_.partialUploads;
    }
    obj.dirtyLeaves.clear();
  }
  for (auto it = meshes_.begin(); it != meshes_.end();) {
    if (!alive.count(it->first)) {
      destroy(it->second);
      it = meshes_.erase(it);
    } else {
      ++it;
    }
  }
  // Draw lists for the blocks that changed.
  for (auto& [id, gpu] : meshes_) {
    for (IndexBuffer* ib : {&gpu.triangles, &gpu.edges}) {
      if (!ib->runsStale) continue;
      std::vector<std::uint32_t> first, count;
      ib->blocks.runs(first, count);
      ib->counts.assign(count.begin(), count.end());
      ib->offsets.resize(first.size());
      for (std::size_t i = 0; i < first.size(); ++i)
        ib->offsets[i] = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(first[i]) * 4);
      ib->runsStale = false;
      if (ib == &gpu.triangles) uploadCommands(gpu);
    }
    stats_.drawRuns += static_cast<int>(gpu.triangles.counts.size());
  }
}

void Renderer::draw(const IndexBuffer& ib, GLenum mode) {
  if (ib.counts.size() == 1) {
    glDrawElements(mode, ib.counts[0], GL_UNSIGNED_INT, ib.offsets[0]);
  } else if (!ib.counts.empty()) {
    glMultiDrawElements(mode, ib.counts.data(), GL_UNSIGNED_INT, ib.offsets.data(),
                        static_cast<GLsizei>(ib.counts.size()));
  }
}

void Renderer::drawTriangles(const GpuMesh& gpu) {
  if (gpu.triangles.counts.empty()) return;
  // GL_DRAW_INDIRECT_BUFFER is context state, not VAO state: bind it for this call only.
  glBindBuffer(GL_DRAW_INDIRECT_BUFFER, gpu.commands);
  glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, nullptr, static_cast<GLsizei>(gpu.triangles.counts.size()),
                              sizeof(DrawCommand));
  glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
}

void Renderer::render(const Scene& scene, const Camera& camera, const ViewportRect& rect,
                      const ViewSettings& settings, std::uint32_t selectedId, const CursorMarker& cursor) {
  glViewport(rect.x, rect.y, rect.width, rect.height);
  glEnable(GL_SCISSOR_TEST);
  glScissor(rect.x, rect.y, rect.width, rect.height);
  glClearDepth(1.0);
  glClear(GL_DEPTH_BUFFER_BIT);

  // Background gradient.
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glUseProgram(backgroundProgram_);
  glUniform3fv(loc(backgroundProgram_, "uTop"), 1, glm::value_ptr(settings.backgroundTop));
  glUniform3fv(loc(backgroundProgram_, "uBottom"), 1, glm::value_ptr(settings.backgroundBottom));
  glBindVertexArray(emptyVao_);
  glDrawArrays(GL_TRIANGLES, 0, 3);

  const Mat4 view = camera.view();
  const Mat4 proj = camera.projection();
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  glDisable(GL_CULL_FACE);

  // Solid pass, pushed back slightly so wireframe lines win the depth test.
  glEnable(GL_POLYGON_OFFSET_FILL);
  glPolygonOffset(1.0f, 1.0f);
  glUseProgram(meshProgram_);
  glUniformMatrix4fv(loc(meshProgram_, "uView"), 1, GL_FALSE, glm::value_ptr(view));
  glUniformMatrix4fv(loc(meshProgram_, "uProj"), 1, GL_FALSE, glm::value_ptr(proj));
  glUniform1i(loc(meshProgram_, "uMatcap"), 0);
  glUniform1f(loc(meshProgram_, "uMaskOpacity"), settings.showMask ? std::clamp(settings.maskOpacity, 0.0f, 1.0f) : 0.0f);
  const float faceSetOpacity = settings.showFaceSets ? std::clamp(settings.faceSetOpacity, 0.0f, 1.0f) : 0.0f;
  const GLint faceSetOpacityLoc = loc(meshProgram_, "uFaceSetOpacity");
  glBindTextureUnit(0, matcaps_[std::clamp(settings.matcap, 0, kMatcapCount - 1)]);
  for (const auto& obj : scene.objects()) {
    if (!obj->visible) continue;
    auto it = meshes_.find(obj->id);
    if (it == meshes_.end()) continue;
    const Mat4 model = obj->transform.matrix();
    const Mat3 normalMat = glm::transpose(glm::inverse(Mat3(view * model)));
    glUniformMatrix4fv(loc(meshProgram_, "uModel"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix3fv(loc(meshProgram_, "uNormalMat"), 1, GL_FALSE, glm::value_ptr(normalMat));
    glUniform1f(loc(meshProgram_, "uSelected"), obj->id == selectedId ? 1.0f : 0.0f);
    glUniform1f(faceSetOpacityLoc, it->second.faceSetsOnGpu ? faceSetOpacity : 0.0f);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, it->second.faceSets);
    glBindVertexArray(it->second.vao);
    drawTriangles(it->second);
  }
  glDisable(GL_POLYGON_OFFSET_FILL);

  // Lines: grid, wireframe, cursor.
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDepthMask(GL_FALSE);
  glUseProgram(lineProgram_);
  const GLint mvpLoc = loc(lineProgram_, "uMvp");
  const GLint colorLoc = loc(lineProgram_, "uColor");
  const GLint useVcLoc = loc(lineProgram_, "uUseVertexColor");

  if (settings.grid) {
    const Mat4 mvp = proj * view;
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform1f(useVcLoc, 1.0f);
    glBindVertexArray(gridVao_);
    glDrawArrays(GL_LINES, 0, gridVertexCount_);
  }

  glUniform1f(useVcLoc, 0.0f);
  for (const auto& obj : scene.objects()) {
    if (!settings.wireframe || !obj->visible) continue;
    const bool selected = obj->id == selectedId;
    auto it = meshes_.find(obj->id);
    if (it == meshes_.end() || !it->second.edgesValid) continue;
    const Mat4 mvp = proj * view * obj->transform.matrix();
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));
    const Vec4 color = selected ? Vec4{1.0f, 0.65f, 0.3f, settings.wireframeOpacity}
                                : Vec4{0.05f, 0.05f, 0.06f, settings.wireframeOpacity};
    glUniform4fv(colorLoc, 1, glm::value_ptr(color));
    glBindVertexArray(it->second.vao);
    glVertexArrayElementBuffer(it->second.vao, it->second.edges.buffer);
    draw(it->second.edges, GL_LINES);
    glVertexArrayElementBuffer(it->second.vao, it->second.triangles.buffer);
  }

  if (cursor.visible) {
    LineVertex pts[kCursorSegments + 2];
    const Vec3 n = glm::normalize(cursor.normal);
    const Vec3 t = glm::normalize(std::abs(n.y) < 0.9f ? glm::cross(n, Vec3{0, 1, 0}) : glm::cross(n, Vec3{1, 0, 0}));
    const Vec3 b = glm::cross(n, t);
    const Vec4 c{1.0f, 1.0f, 1.0f, 0.9f};
    for (int i = 0; i < kCursorSegments; ++i) {
      const float a = 6.2831853f * static_cast<float>(i) / kCursorSegments;
      pts[i] = {cursor.position + (t * std::cos(a) + b * std::sin(a)) * cursor.radius, c};
    }
    pts[kCursorSegments] = {cursor.position, c};
    pts[kCursorSegments + 1] = {cursor.position + n * cursor.radius, c};
    glNamedBufferSubData(cursorVbo_, 0, sizeof(pts), pts);
    const Mat4 mvp = proj * view;
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform1f(useVcLoc, 1.0f);
    glDisable(GL_DEPTH_TEST);  // The cursor must stay visible over the surface it sits on.
    glBindVertexArray(cursorVao_);
    glDrawArrays(GL_LINE_LOOP, 0, kCursorSegments);
    glDrawArrays(GL_LINES, kCursorSegments, 2);
    glEnable(GL_DEPTH_TEST);
  }

  glDepthMask(GL_TRUE);
  glDisable(GL_BLEND);
  glDisable(GL_SCISSOR_TEST);
  glBindVertexArray(0);
}

}  // namespace plegl
