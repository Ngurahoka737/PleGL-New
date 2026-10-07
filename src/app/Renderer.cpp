#include "Renderer.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

#include <glm/gtc/type_ptr.hpp>

namespace plegl {
namespace {

// ----- Shaders --------------------------------------------------------------------------------

const char* kMeshVs = R"(#version 450 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 3) in float aMask;  // Location 2 is the line shader's vertex colour.
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform mat3 uNormalMat;  // view * model, inverse-transposed
out vec3 vNormalView;
out float vMask;
void main() {
  vNormalView = uNormalMat * aNormal;
  vMask = aMask;
  gl_Position = uProj * uView * uModel * vec4(aPosition, 1.0);
}
)";

const char* kMeshFs = R"(#version 450 core
in vec3 vNormalView;
in float vMask;
uniform sampler2D uMatcap;
uniform float uSelected;
uniform float uMaskOpacity;
out vec4 fragColor;
void main() {
  vec3 n = normalize(vNormalView);
  if (!gl_FrontFacing) n = -n;  // Open meshes: shade the inside too.
  vec3 color = texture(uMatcap, n.xy * 0.49 + 0.5).rgb;
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
  const GLuint buffers[] = {gpu.positions, gpu.normals, gpu.mask, gpu.triangles, gpu.edges};
  glDeleteBuffers(5, buffers);
  glDeleteVertexArrays(1, &gpu.vao);
  gpu = {};
}

void Renderer::upload(GpuMesh& gpu, const SceneObject& object) {
  destroy(gpu);
  const Mesh& m = object.mesh;

  // Fan-triangulate faces in order, so each leaf's triangles stay contiguous.
  std::vector<std::uint32_t> tris;
  tris.reserve(static_cast<std::size_t>(m.halfEdgeCount()) * 3);
  for (Index f = 0; f < m.faceCount(); ++f) {
    const Index h0 = m.faceHe[f];
    const auto a = static_cast<std::uint32_t>(m.heVert[h0]);
    for (Index h = m.heNext[h0]; m.heNext[h] != h0; h = m.heNext[h]) {
      tris.push_back(a);
      tris.push_back(static_cast<std::uint32_t>(m.heVert[h]));
      tris.push_back(static_cast<std::uint32_t>(m.heTarget(h)));
    }
  }
  // Real polygon edges, so quads draw as quads in the wireframe.
  std::vector<std::uint32_t> edges;
  edges.reserve(static_cast<std::size_t>(m.halfEdgeCount()));
  for (Index h = 0; h < m.halfEdgeCount(); ++h) {
    if (m.heTwin[h] == kInvalid || h < m.heTwin[h]) {
      edges.push_back(static_cast<std::uint32_t>(m.heVert[h]));
      edges.push_back(static_cast<std::uint32_t>(m.heTarget(h)));
    }
  }

  const auto vbytes = static_cast<GLsizeiptr>(m.positions.size() * sizeof(Vec3));
  glCreateBuffers(1, &gpu.positions);
  glNamedBufferStorage(gpu.positions, vbytes, m.positions.data(), GL_DYNAMIC_STORAGE_BIT);
  glCreateBuffers(1, &gpu.normals);
  glNamedBufferStorage(gpu.normals, vbytes, m.normals.data(), GL_DYNAMIC_STORAGE_BIT);
  // One float per vertex; an unmasked mesh gets a zero-filled buffer so the shader can always read it.
  const bool hasMask = !m.mask.empty();
  glCreateBuffers(1, &gpu.mask);
  glNamedBufferStorage(gpu.mask, static_cast<GLsizeiptr>(std::max<std::size_t>(m.positions.size(), 1) * sizeof(float)),
                       hasMask ? m.mask.data() : nullptr, GL_DYNAMIC_STORAGE_BIT);
  if (!hasMask) glClearNamedBufferData(gpu.mask, GL_R32F, GL_RED, GL_FLOAT, nullptr);
  gpu.maskOnGpu = hasMask;
  glCreateBuffers(1, &gpu.triangles);
  glNamedBufferStorage(gpu.triangles, static_cast<GLsizeiptr>(tris.size() * 4), tris.data(), 0);
  glCreateBuffers(1, &gpu.edges);
  glNamedBufferStorage(gpu.edges, static_cast<GLsizeiptr>(edges.size() * 4), edges.data(), 0);

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
  glVertexArrayElementBuffer(gpu.vao, gpu.triangles);

  gpu.triangleIndexCount = static_cast<GLsizei>(tris.size());
  gpu.edgeIndexCount = static_cast<GLsizei>(edges.size());
  gpu.topologyVersion = object.topologyVersion;
  stats_.uploadedBytes += static_cast<std::size_t>(vbytes) * 2 + m.mask.size() * sizeof(float) + tris.size() * 4 +
                         edges.size() * 4;
  ++stats_.fullUploads;
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

void Renderer::sync(Scene& scene) {
  stats_ = {};
  std::unordered_set<std::uint32_t> alive;
  for (const auto& objPtr : scene.objects()) {
    SceneObject& obj = *objPtr;
    alive.insert(obj.id);
    GpuMesh& gpu = meshes_[obj.id];
    if (gpu.topologyVersion != obj.topologyVersion) {
      upload(gpu, obj);
      obj.clearDirty();
      continue;
    }
    syncMask(gpu, obj);
    if (obj.dirtyLeaves.empty()) continue;
    std::sort(obj.dirtyLeaves.begin(), obj.dirtyLeaves.end());
    obj.dirtyLeaves.erase(std::unique(obj.dirtyLeaves.begin(), obj.dirtyLeaves.end()), obj.dirtyLeaves.end());
    const auto leaves = obj.bvh.leaves();
    for (Index l : obj.dirtyLeaves) {
      const BvhLeaf& leaf = leaves[l];
      const auto offset = static_cast<GLintptr>(leaf.vertBegin * sizeof(Vec3));
      const auto bytes = static_cast<GLsizeiptr>((leaf.vertEnd - leaf.vertBegin) * sizeof(Vec3));
      if (bytes == 0) continue;
      glNamedBufferSubData(gpu.positions, offset, bytes, &obj.mesh.positions[leaf.vertBegin]);
      glNamedBufferSubData(gpu.normals, offset, bytes, &obj.mesh.normals[leaf.vertBegin]);
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
    glBindVertexArray(it->second.vao);
    glDrawElements(GL_TRIANGLES, it->second.triangleIndexCount, GL_UNSIGNED_INT, nullptr);
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
    if (it == meshes_.end()) continue;
    const Mat4 mvp = proj * view * obj->transform.matrix();
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));
    const Vec4 color = selected ? Vec4{1.0f, 0.65f, 0.3f, settings.wireframeOpacity}
                                : Vec4{0.05f, 0.05f, 0.06f, settings.wireframeOpacity};
    glUniform4fv(colorLoc, 1, glm::value_ptr(color));
    glBindVertexArray(it->second.vao);
    glVertexArrayElementBuffer(it->second.vao, it->second.edges);
    glDrawElements(GL_LINES, it->second.edgeIndexCount, GL_UNSIGNED_INT, nullptr);
    glVertexArrayElementBuffer(it->second.vao, it->second.triangles);
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
