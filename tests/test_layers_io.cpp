#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "TestUtil.h"
#include "io/Obj.h"
#include "io/Project.h"
#include "mesh/Primitives.h"
#include "multires/MultiresIo.h"
#include "multires/MultiresOps.h"
#include "scene/Scene.h"
#include "sculpt/LayerOps.h"
#include "sculpt/Sculptor.h"

using namespace plegl;

namespace {

bool bitwise(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(Vec3)) == 0);
}

bool sameFloat(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

Mesh& levelMesh(SceneObject& o, int k) {
  return k == o.multires->active ? o.mesh : o.multires->levels[static_cast<std::size_t>(k)].mesh;
}

// For every vertex of b, the vertex of a at the same position (opening a project rebuilds the BVH,
// which may number the vertices differently). As a CanonicalMap-style map it puts b's arrays in
// a's order: toCanonical(bArray, map).
std::vector<Index> matchVertices(const Mesh& a, const Mesh& b) {
  auto key = [](const Vec3& p) {
    std::array<std::uint32_t, 3> k;
    std::memcpy(k.data(), &p, sizeof(Vec3));
    return k;
  };
  std::map<std::array<std::uint32_t, 3>, Index> at;
  for (Index v = 0; v < a.vertexCount(); ++v) at.emplace(key(a.positions[v]), v);
  REQUIRE(at.size() == a.positions.size());
  REQUIRE(b.positions.size() == a.positions.size());
  std::vector<Index> map(b.positions.size());
  for (Index v = 0; v < b.vertexCount(); ++v) {
    const auto it = at.find(key(b.positions[v]));
    REQUIRE(it != at.end());
    map[static_cast<std::size_t>(v)] = it->second;
  }
  return map;
}

// The same positions, whatever the vertex order.
bool sameShape(const Mesh& a, const Mesh& b) {
  const std::vector<Index> map = matchVertices(a, b);
  return bitwise(toCanonical(b.positions, map), a.positions);
}

Vec3 surfacePoint(const SceneObject& obj, Vec3 dir) {
  dir = glm::normalize(dir);
  RayHit hit;
  REQUIRE(obj.bvh.raycast(obj.mesh, Ray{dir * 3.0f, -dir}, hit, std::numeric_limits<float>::infinity(), true));
  return hit.position;
}

void stroke(SceneObject& obj, Vec3 dir, float radius = 0.35f) {
  static DrawBrush draw;
  Sculptor sculptor;
  StrokeOptions o;
  o.layerTarget = obj.mesh.layers.active;
  sculptor.beginStroke(obj, draw, o, "Draw");
  REQUIRE(sculptor.active());
  const Vec3 c = surfacePoint(obj, dir);
  for (int i = 0; i < 3; ++i) sculptor.dab(c, radius, 0.7f);
  REQUIRE(sculptor.endStroke());
}

void sculptLayer(SceneObject& obj, LayerWorkspace& ws, Vec3 dir, float strength = 1.0f) {
  std::string error;
  REQUIRE(addLayer(obj, ws, &error));
  stroke(obj, dir);
  if (strength != 1.0f) REQUIRE(setLayerStrength(obj, obj.mesh.layers.active, strength, ws, &error));
}

void requireLayers(const Mesh& m) {
  const ValidationResult r = validateLayers(m, true);
  INFO(r.message);
  REQUIRE(r.ok);
}

// Everything a file keeps of a stack (not the epoch). `map` puts b's arrays in a's vertex order.
void requireSameStack(const LayerStack& a, const LayerStack& b, const std::vector<Index>* aCanon = nullptr,
                      const std::vector<Index>* bCanon = nullptr) {
  auto order = [](const std::vector<Vec3>& v, const std::vector<Index>* canon) {
    return canon ? toCanonical(v, *canon) : v;
  };
  CHECK(bitwise(order(a.base, aCanon), order(b.base, bCanon)));
  CHECK(a.active == b.active);
  CHECK(a.nextId == b.nextId);
  REQUIRE(a.list.size() == b.list.size());
  for (std::size_t k = 0; k < a.list.size(); ++k) {
    INFO("layer " << k);
    CHECK(a.list[k].id == b.list[k].id);
    CHECK(a.list[k].name == b.list[k].name);
    CHECK(sameFloat(a.list[k].strength, b.list[k].strength));
    CHECK(a.list[k].visible == b.list[k].visible);
    std::vector<Vec3> oa = order(a.list[k].offset, aCanon), ob = order(b.list[k].offset, bCanon);
    for (Vec3& o : oa)
      if (isZero(o)) o = Vec3{0.0f};  // -0 is stored as +0; both add nothing.
    CHECK(bitwise(oa, ob));
  }
}

struct Opened {
  Project project;
  std::unique_ptr<Scene> scene;
};

Opened open(const std::vector<std::uint8_t>& bytes) {
  std::string error;
  std::optional<Project> p = parseProject(bytes.data(), bytes.size(), &error);
  INFO(error);
  REQUIRE(p);
  REQUIRE(buildProject(*p, &error));
  Opened out{std::move(*p), std::make_unique<Scene>()};
  for (ProjectObject& po : out.project.objects) addProjectObject(*out.scene, po);
  return out;
}

std::size_t findChunk(const std::vector<std::uint8_t>& bytes, const char* tag) {
  std::size_t at = 12;  // Magic and version.
  while (at + 12 <= bytes.size()) {
    if (std::memcmp(bytes.data() + at, tag, 4) == 0) return at;
    std::uint64_t size = 0;
    std::memcpy(&size, bytes.data() + at + 4, 8);
    at += 12 + std::size_t(size);
  }
  return std::string::npos;
}

std::uint64_t chunkSize(const std::vector<std::uint8_t>& bytes, std::size_t at) {
  std::uint64_t size = 0;
  std::memcpy(&size, bytes.data() + at + 4, 8);
  return size;
}

void fixChecksum(std::vector<std::uint8_t>& bytes) {
  const std::uint32_t crc = crc32(bytes.data(), bytes.size() - 16);
  std::memcpy(bytes.data() + bytes.size() - 4, &crc, 4);
}

// The file with its LAYR chunk replaced by one with `payload` (or removed when empty).
std::vector<std::uint8_t> withLayr(std::vector<std::uint8_t> bytes, const std::vector<std::uint8_t>& payload) {
  const std::size_t at = findChunk(bytes, "LAYR");
  if (at != std::string::npos)
    bytes.erase(bytes.begin() + static_cast<std::ptrdiff_t>(at),
                bytes.begin() + static_cast<std::ptrdiff_t>(at + 12 + chunkSize(bytes, at)));
  if (!payload.empty()) {
    std::vector<std::uint8_t> chunk = {'L', 'A', 'Y', 'R'};
    const std::uint64_t size = payload.size();
    chunk.insert(chunk.end(), reinterpret_cast<const std::uint8_t*>(&size), reinterpret_cast<const std::uint8_t*>(&size) + 8);
    chunk.insert(chunk.end(), payload.begin(), payload.end());
    bytes.insert(bytes.end() - 16, chunk.begin(), chunk.end());
  }
  fixChecksum(bytes);
  return bytes;
}

// A LAYR payload written field by field, so tests can damage any of them.
struct RawBlock {
  std::uint8_t code = 1;
  std::vector<std::uint32_t> index;
  std::vector<Vec3> values;  // Dense: one per vertex.
  std::uint32_t count() const { return static_cast<std::uint32_t>(index.size()); }
};
struct RawLayer {
  std::uint32_t id = 1;
  float strength = 1.0f;
  std::uint8_t flags = 1;
  std::string name = "Layer";
  RawBlock offsets;
};
struct RawStack {
  std::uint32_t object = 0;
  std::uint8_t level = 255;
  int layerCount = -1;  // -1: the number of layers.
  std::uint8_t activeSlot = 0;
  std::uint32_t vertexCount = 0, nextId = 2;
  RawBlock base{2, {}, {}};
  std::vector<RawLayer> layers;
};
struct RawLayr {
  std::uint8_t encoding = 0;
  std::vector<RawStack> stacks;
};

template <class T>
void put(std::vector<std::uint8_t>& out, const T& v) {
  out.insert(out.end(), reinterpret_cast<const std::uint8_t*>(&v), reinterpret_cast<const std::uint8_t*>(&v) + sizeof(T));
}

void putBlock(std::vector<std::uint8_t>& out, const RawBlock& b) {
  put(out, b.code);
  if (b.code != 0) put(out, b.count());
  for (std::uint32_t i : b.index) put(out, i);
  for (const Vec3& v : b.values) put(out, v);
}

std::vector<std::uint8_t> encode(const RawLayr& l) {
  std::vector<std::uint8_t> out;
  put(out, l.encoding);
  put(out, std::array<std::uint8_t, 3>{});
  put(out, static_cast<std::uint32_t>(l.stacks.size()));
  for (const RawStack& s : l.stacks) {
    put(out, s.object);
    put(out, s.level);
    put(out, static_cast<std::uint8_t>(s.layerCount < 0 ? s.layers.size() : std::size_t(s.layerCount)));
    put(out, s.activeSlot);
    put(out, std::uint8_t{0});
    put(out, s.vertexCount);
    put(out, s.nextId);
    putBlock(out, s.base);
    for (const RawLayer& layer : s.layers) {
      put(out, layer.id);
      put(out, layer.strength);
      put(out, layer.flags);
      put(out, static_cast<std::uint8_t>(layer.name.size()));
      out.insert(out.end(), layer.name.begin(), layer.name.end());
      putBlock(out, layer.offsets);
    }
  }
  return out;
}

constexpr const char* kDamaged = "Sculpt layers could not be read and were dropped; the sculpted shape is kept.";

}  // namespace

TEST_CASE("layer files: a plain object's layers round-trip") {
  Scene scene;
  LayerWorkspace ws;
  SceneObject& obj = scene.add("Head", makeIcosphere(4));
  sculptLayer(obj, ws, {0, 1, 0});
  sculptLayer(obj, ws, {1, 0.2f, 0}, -0.8f);
  std::string error;
  REQUIRE(renameLayer(obj, obj.mesh.layers.list[1].id, "D\xC3\xA9tail \xE2\x9C\x93", &error));
  // A dense layer: every vertex moves.
  REQUIRE(addLayer(obj, ws, &error));
  {
    SculptLayer& l = obj.mesh.layers.list.back();
    for (std::size_t v = 0; v < l.offset.size(); ++v) l.offset[v] = Vec3{0.001f * float(v % 7), 0.002f, -0.001f};
    l.offset[3] = Vec3{-0.0f, 0.0f, -0.0f};  // Reads back as +0.
    obj.mesh.positions = [&] {
      std::vector<Vec3> p(obj.mesh.positions.size());
      for (std::size_t v = 0; v < p.size(); ++v) p[v] = composeVertex(obj.mesh.layers, static_cast<Index>(v));
      return p;
    }();
    obj.mesh.computeNormals();
  }
  REQUIRE(setLayerVisible(obj, obj.mesh.layers.list[0].id, false, ws, &error));
  REQUIRE(selectLayer(obj, obj.mesh.layers.list[1].id));
  requireLayers(obj.mesh);

  const std::vector<std::uint8_t> bytes = serializeProject(scene, "");
  Opened o = open(bytes);
  CHECK(o.project.warnings.empty());
  const SceneObject& copy = *o.scene->objects()[0];
  requireLayers(copy.mesh);
  const std::vector<Index> map = matchVertices(obj.mesh, copy.mesh);
  requireSameStack(obj.mesh.layers, copy.mesh.layers, nullptr, &map);
  CHECK(copy.mesh.layers.epoch != 0);
  CHECK(copy.mesh.layers.epoch != obj.mesh.layers.epoch);
  CHECK(copy.mesh.layers.list[1].name == "D\xC3\xA9tail \xE2\x9C\x93");

  // The chunk size follows from the codes: the base over the composite, the two stroked layers
  // sparse, the full one dense.
  const LayerStack& s = obj.mesh.layers;
  const std::size_t V = s.base.size();
  std::size_t baseCount = 0;
  for (std::size_t v = 0; v < V; ++v) baseCount += !sameBits(s.base[v], obj.mesh.positions[v]);
  std::size_t expected = 8 + 16 + 1 + 4 + 16 * baseCount;
  for (const SculptLayer& l : s.list) {
    std::size_t count = 0;
    for (const Vec3& v : l.offset) count += !isZero(v);
    const bool sparse = 4 + 16 * count < 12 * V;
    expected += 10 + l.name.size() + 1 + (sparse ? 4 + 16 * count : 12 * V);
    CHECK(sparse == (&l != &s.list.back()));
  }
  const std::size_t at = findChunk(bytes, "LAYR");
  REQUIRE(at != std::string::npos);
  CHECK(chunkSize(bytes, at) == expected);
}

TEST_CASE("layer files: stacks on several levels round-trip with pending edits") {
  Scene scene;
  SyncWorkspace sws;
  LayerWorkspace ws;
  SceneObject& obj = scene.add("Head", makeQuadSphere(5));
  sculptLayer(obj, ws, {0, 1, 0});  // Stays on level 0.
  for (int i = 0; i < 3; ++i) REQUIRE(subdivideObject(obj, sws));
  REQUIRE(setActiveLevel(obj, 2, sws));
  sculptLayer(obj, ws, {1, 0.3f, 0}, 0.5f);
  sculptLayer(obj, ws, {0, 0.2f, 1}, -1.5f);
  REQUIRE(setActiveLevel(obj, 1, sws));
  sculptLayer(obj, ws, {0.3f, 1, 0.3f});
  REQUIRE(!diffActive(*obj.multires, obj.mesh).empty());

  Opened o = open(serializeProject(scene, ""));
  CHECK(o.project.warnings.empty());
  SceneObject& copy = *o.scene->objects()[0];
  {
    const ValidationResult r = validateMultires(copy);
    INFO(r.message);
    REQUIRE(r.ok);
  }
  auto compare = [&] {
    REQUIRE(copy.multires->active == obj.multires->active);
    for (int k = 0; k < 4; ++k) {
      INFO("level " << k);
      const std::vector<Index>& ca = obj.multires->levels[static_cast<std::size_t>(k)].canon.vert;
      const std::vector<Index>& cb = copy.multires->levels[static_cast<std::size_t>(k)].canon.vert;
      CHECK(bitwise(toCanonical(levelMesh(obj, k).positions, ca), toCanonical(levelMesh(copy, k).positions, cb)));
      CHECK(levelMesh(copy, k).layers.list.size() == levelMesh(obj, k).layers.list.size());
      if (!levelMesh(obj, k).layers.empty()) requireSameStack(levelMesh(obj, k).layers, levelMesh(copy, k).layers, &ca, &cb);
    }
    CHECK(bitwise(toCanonical(obj.multires->reference.positions,
                              obj.multires->levels[static_cast<std::size_t>(obj.multires->active)].canon.vert),
                  toCanonical(copy.multires->reference.positions,
                              copy.multires->levels[static_cast<std::size_t>(copy.multires->active)].canon.vert)));
  };
  compare();
  // Level steps after opening give the same bits as before saving.
  SyncWorkspace sws2;
  for (int level : {3, 0}) {
    REQUIRE(setActiveLevel(obj, level, sws));
    REQUIRE(setActiveLevel(copy, level, sws2));
    compare();
  }
}

TEST_CASE("layer files: builds without layers and newer layer data open the sculpted shape") {
  Scene scene;
  LayerWorkspace ws;
  SceneObject& obj = scene.add("Head", makeIcosphere(3));
  sculptLayer(obj, ws, {0, 1, 0});
  sculptLayer(obj, ws, {0, 0.5f, 1}, 0.4f);
  const std::vector<std::uint8_t> bytes = serializeProject(scene, "");
  const std::size_t at = findChunk(bytes, "LAYR");
  REQUIRE(at != std::string::npos);
  SUBCASE("no LAYR") {
    Opened o = open(withLayr(bytes, {}));
    CHECK(o.project.warnings.empty());
    const SceneObject& copy = *o.scene->objects()[0];
    CHECK(copy.mesh.layers.empty());
    CHECK(sameShape(copy.mesh, obj.mesh));
    requireLayers(copy.mesh);
  }
  SUBCASE("newer encoding") {
    std::vector<std::uint8_t> b = bytes;
    b[at + 12] = 1;
    fixChecksum(b);
    Opened o = open(b);
    REQUIRE(o.project.warnings.size() == 1);
    CHECK(o.project.warnings[0] == "Sculpt layers were saved by a newer version and were dropped.");
    CHECK(o.scene->objects()[0]->mesh.layers.empty());
    CHECK(sameShape(o.scene->objects()[0]->mesh, obj.mesh));
  }
  SUBCASE("a second LAYR is ignored") {
    const std::vector<std::uint8_t> chunk(bytes.begin() + static_cast<std::ptrdiff_t>(at),
                                          bytes.begin() + static_cast<std::ptrdiff_t>(at + 12 + chunkSize(bytes, at)));
    std::vector<std::uint8_t> b = bytes;
    std::vector<std::uint8_t> junk = chunk;
    junk[12] = 7;  // A damaged copy after the good one.
    b.insert(b.end() - 16, junk.begin(), junk.end());
    fixChecksum(b);
    Opened o = open(b);
    CHECK(o.project.warnings.empty());
    CHECK(o.scene->objects()[0]->mesh.layers.list.size() == 2);
  }
}

TEST_CASE("layer files: damaged layer data opens without layers and says so") {
  Scene scene;
  SceneObject& a = scene.add("A", makeIcosphere(2));
  scene.add("B", makeIcosphere(1));
  const std::vector<std::uint8_t> bytes = serializeProject(scene, "");
  const auto V = static_cast<std::uint32_t>(a.mesh.vertexCount());
  // A valid hand-written chunk: one stack on A with two sparse layers.
  RawLayr good;
  RawStack s;
  s.vertexCount = V;
  s.nextId = 5;
  s.activeSlot = 2;
  s.layers.push_back({1, 1.0f, 1, "One", {1, {3, 9}, {Vec3{0.01f, 0, 0}, Vec3{0, 0.02f, 0}}}});
  s.layers.push_back({4, -0.5f, 0, "Two", {1, {9}, {Vec3{0, 0, 0.03f}}}});
  // The base under them: the stored positions minus the visible layer.
  s.base.index = {3, 9};
  s.base.values = {a.mesh.positions[3] - Vec3{0.01f, 0, 0}, a.mesh.positions[9] - Vec3{0, 0.02f, 0}};
  good.stacks.push_back(s);
  {
    Opened o = open(withLayr(bytes, encode(good)));
    // Rounding in the hand-made base is recomposed, so no warning.
    CHECK(o.project.warnings.empty());
    const Mesh& m = o.scene->objects()[0]->mesh;
    REQUIRE(m.layers.list.size() == 2);
    CHECK(m.layers.active == 4);
    CHECK(!m.layers.list[1].visible);
    requireLayers(m);
  }
  auto damaged = [&](const char* name, auto&& damage) {
    INFO(name);
    RawLayr l = good;
    damage(l);
    Opened o = open(withLayr(bytes, encode(l)));
    REQUIRE(o.project.warnings.size() == 1);
    CHECK(o.project.warnings[0] == kDamaged);
    for (const auto& obj : o.scene->objects()) {
      CHECK(obj->mesh.layers.empty());
      requireLayers(obj->mesh);
    }
    CHECK(sameShape(o.scene->objects()[0]->mesh, a.mesh));
  };
  damaged("stacks out of order", [&](RawLayr& l) {
    RawStack b = l.stacks[0];
    b.object = 1;
    b.vertexCount = static_cast<std::uint32_t>(scene.objects()[1]->mesh.vertexCount());
    b.base.index.clear();
    b.base.values.clear();
    l.stacks.insert(l.stacks.begin(), b);
  });
  damaged("the same stack twice", [](RawLayr& l) { l.stacks.push_back(l.stacks[0]); });
  damaged("count above the vertex count", [&](RawLayr& l) {
    RawBlock& b = l.stacks[0].layers[0].offsets;
    b.index.resize(V + 1);
    b.values.resize(V + 1);
    for (std::uint32_t i = 0; i <= V; ++i) b.index[i] = i;
  });
  damaged("indices not ascending", [](RawLayr& l) { std::swap(l.stacks[0].layers[0].offsets.index[0], l.stacks[0].layers[0].offsets.index[1]); });
  damaged("index out of range", [&](RawLayr& l) { l.stacks[0].layers[1].offsets.index[0] = V; });
  damaged("NaN offset", [](RawLayr& l) { l.stacks[0].layers[0].offsets.values[1].y = std::numeric_limits<float>::quiet_NaN(); });
  damaged("infinite base", [](RawLayr& l) { l.stacks[0].base.values[0].x = std::numeric_limits<float>::infinity(); });
  damaged("duplicate id", [](RawLayr& l) { l.stacks[0].layers[1].id = 1; });
  damaged("id at next id", [](RawLayr& l) { l.stacks[0].layers[1].id = 5; });
  damaged("id zero", [](RawLayr& l) { l.stacks[0].layers[0].id = 0; });
  damaged("next id too small", [](RawLayr& l) { l.stacks[0].nextId = 1; });
  damaged("strength out of range", [](RawLayr& l) { l.stacks[0].layers[0].strength = 10.5f; });
  damaged("strength NaN", [](RawLayr& l) { l.stacks[0].layers[0].strength = std::numeric_limits<float>::quiet_NaN(); });
  damaged("unknown flags", [](RawLayr& l) { l.stacks[0].layers[0].flags = 3; });
  damaged("name too long", [](RawLayr& l) { l.stacks[0].layers[0].name = std::string(64, 'x'); });
  damaged("level on an object without levels", [](RawLayr& l) { l.stacks[0].level = 0; });
  damaged("object out of range", [](RawLayr& l) { l.stacks[0].object = 2; });
  damaged("wrong vertex count", [](RawLayr& l) { l.stacks[0].vertexCount -= 1; });
  damaged("no layers", [](RawLayr& l) { l.stacks[0].layerCount = 0; });
  damaged("active slot past the layers", [](RawLayr& l) { l.stacks[0].activeSlot = 3; });
  damaged("more layers claimed than stored", [](RawLayr& l) { l.stacks[0].layerCount = 3; });
  damaged("code 2 on offsets", [](RawLayr& l) { l.stacks[0].layers[0].offsets.code = 2; });
  damaged("unknown code", [](RawLayr& l) { l.stacks[0].base.code = 3; });
  damaged("no stacks", [](RawLayr& l) { l.stacks.clear(); });

  SUBCASE("a name that is not UTF-8 is replaced") {
    RawLayr l = good;
    l.stacks[0].layers[0].name = "Bad \xC3";
    Opened o = open(withLayr(bytes, encode(l)));
    CHECK(o.project.warnings.empty());
    REQUIRE(o.scene->objects()[0]->mesh.layers.list.size() == 2);
    CHECK(o.scene->objects()[0]->mesh.layers.list[0].name == "Layer 1");
  }
  SUBCASE("a truncated chunk") {
    std::vector<std::uint8_t> payload = encode(good);
    payload.resize(payload.size() - 5);
    Opened o = open(withLayr(bytes, payload));
    REQUIRE(o.project.warnings.size() == 1);
    CHECK(o.project.warnings[0] == kDamaged);
    CHECK(o.scene->objects()[0]->mesh.layers.empty());
  }
}

TEST_CASE("layer files: layers too large for an object are refused before decoding") {
  // 255 layers on a level of 700K+ vertices pass 2 GiB.
  Scene scene;
  SyncWorkspace ws;
  SceneObject& obj = scene.add("Big", makeQuadSphere(6));
  for (int i = 0; i < 6; ++i) REQUIRE(subdivideObject(obj, ws));
  const auto V = static_cast<std::uint32_t>(obj.mesh.vertexCount());
  REQUIRE((255.0 + 1.0) * V * 12.0 > double(kMaxFileObjectLayerBytes));
  const std::vector<std::uint8_t> bytes = serializeProject(scene, "");
  RawLayr l;
  RawStack s;
  s.level = 6;
  s.vertexCount = V;
  s.nextId = 300;
  s.layerCount = 255;  // Nothing else follows: the guard must refuse before reading on.
  l.stacks.push_back(s);
  std::string error;
  const std::vector<std::uint8_t> b = withLayr(bytes, encode(l));
  std::optional<Project> p = parseProject(b.data(), b.size(), &error);
  REQUIRE(p);
  REQUIRE(p->warnings.size() == 1);
  CHECK(p->warnings[0] == kDamaged);
  CHECK(p->objects[0].levelLayers.empty());
}

TEST_CASE("layer files: composites off by rounding are recomposed, others dropped") {
  SUBCASE("object without levels") {
    for (float off : {0.0f, 1e-2f}) {
      INFO("off by " << off);
      Scene scene;
      LayerWorkspace ws;
      SceneObject& obj = scene.add("Head", makeIcosphere(3));
      sculptLayer(obj, ws, {0, 1, 0});
      Index v = 0;
      while (isZero(obj.mesh.layers.list[0].offset[v])) ++v;
      const Vec3 composite = obj.mesh.positions[v];
      // Stored a little off from what the layers add up to.
      obj.mesh.positions[v].x = off == 0.0f ? std::nextafter(composite.x, 10.0f) : composite.x + off;
      const Vec3 stored = obj.mesh.positions[v];
      Opened o = open(serializeProject(scene, ""));
      const SceneObject& copy = *o.scene->objects()[0];
      requireLayers(copy.mesh);
      // The copy's vertex order may differ: find v by its stored or composite position.
      bool found = false;
      for (Index w = 0; w < copy.mesh.vertexCount(); ++w) {
        if (off == 0.0f && sameBits(copy.mesh.positions[w], composite)) found = true;
        if (off != 0.0f && sameBits(copy.mesh.positions[w], stored)) found = true;
      }
      CHECK(found);
      if (off == 0.0f) {
        CHECK(o.project.warnings.empty());
        CHECK(copy.mesh.layers.list.size() == 1);
      } else {
        REQUIRE(o.project.warnings.size() == 1);
        CHECK(o.project.warnings[0] == "Sculpt layers of 'Head' did not match the saved shape and were dropped.");
        CHECK(copy.mesh.layers.empty());
      }
    }
  }
  SUBCASE("active level: the reference moves along") {
    Scene scene;
    SyncWorkspace sws;
    LayerWorkspace ws;
    SceneObject& obj = scene.add("Head", makeQuadSphere(4));
    REQUIRE(subdivideObject(obj, sws));
    sculptLayer(obj, ws, {0, 1, 0});
    REQUIRE(setActiveLevel(obj, 0, sws));
    REQUIRE(setActiveLevel(obj, 1, sws));
    REQUIRE(diffActive(*obj.multires, obj.mesh).empty());
    Index v = 0;
    while (isZero(obj.mesh.layers.list[0].offset[v])) ++v;
    // Stored and referenced 1 ulp off, so the file has no pending edit.
    obj.mesh.positions[v].x = std::nextafter(obj.mesh.positions[v].x, 10.0f);
    obj.multires->reference.positions[v] = obj.mesh.positions[v];
    Opened o = open(serializeProject(scene, ""));
    CHECK(o.project.warnings.empty());
    SceneObject& copy = *o.scene->objects()[0];
    const ValidationResult r = validateMultires(copy);
    INFO(r.message);
    CHECK(r.ok);
    CHECK(copy.mesh.layers.list.size() == 1);
    CHECK(diffActive(*copy.multires, copy.mesh).empty());
  }
}

TEST_CASE("layer files: the fast CRC equals the byte-wise one") {
  auto reference = [](const std::uint8_t* p, std::size_t n) {
    std::uint32_t crc = ~0u;
    for (std::size_t i = 0; i < n; ++i) {
      crc ^= p[i];
      for (int k = 0; k < 8; ++k) crc = (crc & 1u) ? 0xEDB88320u ^ (crc >> 1) : crc >> 1;
    }
    return ~crc;
  };
  std::mt19937 rng(7);
  std::vector<std::uint8_t> buf((1 << 20) + 16);
  for (std::uint8_t& b : buf) b = static_cast<std::uint8_t>(rng());
  for (std::size_t offset : {0, 1, 3}) {
    for (std::size_t n = 0; n <= 17; ++n) CHECK(crc32(buf.data() + offset, n) == reference(buf.data() + offset, n));
    CHECK(crc32(buf.data() + offset, 1 << 20) == reference(buf.data() + offset, 1 << 20));
  }
  CHECK(crc32(reinterpret_cast<const std::uint8_t*>("123456789"), 9) == 0xCBF43926u);
  // Continuing a CRC gives the same as one pass.
  CHECK(crc32(buf.data() + 13, 1000, crc32(buf.data(), 13)) == crc32(buf.data(), 1013));
}

TEST_CASE("layer files: a draft keeps what it captured while the scene changes") {
  Scene scene;
  SyncWorkspace sws;
  LayerWorkspace ws;
  SceneObject& plain = scene.add("Plain", makeIcosphere(3));
  sculptLayer(plain, ws, {0, 1, 0});
  SceneObject& leveled = scene.add("Leveled", makeQuadSphere(4));
  REQUIRE(subdivideObject(leveled, sws));
  sculptLayer(leveled, ws, {1, 0, 0}, 0.5f);
  const std::vector<std::uint8_t> expected = serializeProject(scene, "k v\n");
  ProjectDraft draft = draftProject(scene, "k v\n");
  // Change everything the draft copied.
  std::string error;
  REQUIRE(setLayerStrength(plain, plain.mesh.layers.list[0].id, 0.25f, ws, &error));
  REQUIRE(addLayer(leveled, ws, &error));
  stroke(leveled, {0, 0, 1});
  CHECK(finishProject(std::move(draft)) == expected);
}

TEST_CASE("layer files: OBJ export writes the shape you see") {
  Scene scene;
  LayerWorkspace ws;
  SceneObject& obj = scene.add("Head", makeIcosphere(3));
  sculptLayer(obj, ws, {0, 1, 0}, 0.6f);
  sculptLayer(obj, ws, {1, 0, 0});
  std::string error;
  REQUIRE(setLayerVisible(obj, obj.mesh.layers.list[1].id, false, ws, &error));
  const std::string text = writeObj(obj.mesh);
  CHECK(text == writeObj(copyWithoutLayers(obj.mesh)));
  Mesh plain = obj.mesh;
  plain.positions = obj.mesh.layers.base;
  CHECK(text != writeObj(copyWithoutLayers(plain)));
}
