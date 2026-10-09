#include "io/Project.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <optional>
#include <fstream>
#include <system_error>

namespace plegl {

static_assert(std::endian::native == std::endian::little, "The .psculpt writer assumes a little-endian CPU.");

namespace {

constexpr char kMagic[8] = {'P', 'S', 'C', 'U', 'L', 'P', 'T', '\x1A'};

constexpr std::uint32_t tag(const char (&s)[5]) {
  return std::uint32_t(std::uint8_t(s[0])) | std::uint32_t(std::uint8_t(s[1])) << 8 |
         std::uint32_t(std::uint8_t(s[2])) << 16 | std::uint32_t(std::uint8_t(s[3])) << 24;
}
constexpr std::uint32_t kTagSettings = tag("SETT");
constexpr std::uint32_t kTagObjects = tag("OBJS");
constexpr std::uint32_t kTagMask = tag("MASK");
constexpr std::uint8_t kMaskEncodingF32 = 0;
constexpr std::uint32_t kTagFaceSets = tag("FSET");
constexpr std::uint8_t kFaceSetEncodingI32 = 0;
constexpr std::uint32_t kTagLevels = tag("MRES");
constexpr std::uint8_t kLevelsEncoding = 0;
constexpr std::uint32_t kTagEnd = tag("END ");

class Writer {
 public:
  std::vector<std::uint8_t> bytes;

  template <class T>
  void put(const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    bytes.insert(bytes.end(), p, p + sizeof(T));
  }
  template <class T>
  void putArray(const T* data, std::size_t count) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(data);
    bytes.insert(bytes.end(), p, p + count * sizeof(T));
  }
  void putString(const std::string& s) {
    put(static_cast<std::uint32_t>(s.size()));
    putArray(s.data(), s.size());
  }
  // Starts a chunk; returns the offset of its size field for endChunk().
  std::size_t beginChunk(std::uint32_t t) {
    put(t);
    const std::size_t at = bytes.size();
    put(std::uint64_t{0});
    return at;
  }
  void endChunk(std::size_t sizeAt) {
    const std::uint64_t size = bytes.size() - sizeAt - sizeof(std::uint64_t);
    std::memcpy(bytes.data() + sizeAt, &size, sizeof(size));
  }
};

class Reader {
 public:
  Reader(const std::uint8_t* data, std::size_t size) : p_(data), end_(data + size) {}

  bool ok() const { return ok_; }
  std::size_t remaining() const { return std::size_t(end_ - p_); }
  const std::uint8_t* position() const { return p_; }

  template <class T>
  T get() {
    T v{};
    if (!need(sizeof(T))) return v;
    std::memcpy(&v, p_, sizeof(T));
    p_ += sizeof(T);
    return v;
  }
  template <class T>
  bool getArray(std::vector<T>& out, std::size_t count) {
    if (count > remaining() / sizeof(T)) return fail();
    out.resize(count);
    std::memcpy(out.data(), p_, count * sizeof(T));
    p_ += count * sizeof(T);
    return true;
  }
  bool getString(std::string& out) {
    const auto n = get<std::uint32_t>();
    if (!ok_ || n > remaining()) return fail();
    out.assign(reinterpret_cast<const char*>(p_), n);
    p_ += n;
    return true;
  }
  void skip(std::size_t n) {
    if (need(n)) p_ += n;
  }
  bool fail() {
    ok_ = false;
    return false;
  }

 private:
  bool need(std::size_t n) { return n <= remaining() ? true : fail(); }

  const std::uint8_t* p_;
  const std::uint8_t* end_;
  bool ok_ = true;
};

bool setError(std::string* error, const std::string& message) {
  if (error) *error = message;
  return false;
}

void writeObject(Writer& w, const SceneObject& obj, const CanonicalLevel* canonical) {
  w.putString(obj.name);
  const Transform& t = obj.transform;
  w.putArray(&t.position.x, 3);
  const float q[4] = {t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z};
  w.putArray(q, 4);
  w.putArray(&t.scale.x, 3);
  w.put(static_cast<std::uint8_t>(obj.visible ? 1 : 0));

  const Mesh& m = obj.mesh;
  w.put(static_cast<std::uint32_t>(m.vertexCount()));
  if (canonical) {
    w.putArray(reinterpret_cast<const float*>(canonical->positions.data()), canonical->positions.size() * 3);
    w.put(static_cast<std::uint32_t>(canonical->sizes.size()));
    w.put(static_cast<std::uint32_t>(canonical->corners.size()));
    w.putArray(canonical->sizes.data(), canonical->sizes.size());
    w.putArray(canonical->corners.data(), canonical->corners.size());
    return;
  }
  w.putArray(reinterpret_cast<const float*>(m.positions.data()), m.positions.size() * 3);
  std::vector<std::uint32_t> sizes(m.faceCount());
  std::vector<std::uint32_t> indices;
  indices.reserve(m.halfEdgeCount());
  for (Index f = 0; f < m.faceCount(); ++f) {
    const std::size_t before = indices.size();
    m.forEachFaceVertex(f, [&](Index v) { indices.push_back(static_cast<std::uint32_t>(v)); });
    sizes[f] = static_cast<std::uint32_t>(indices.size() - before);
  }
  w.put(static_cast<std::uint32_t>(sizes.size()));
  w.put(static_cast<std::uint32_t>(indices.size()));
  w.putArray(sizes.data(), sizes.size());
  w.putArray(indices.data(), indices.size());
}

bool readObject(Reader& r, ProjectObject& out, std::string* error) {
  if (!r.getString(out.name)) return setError(error, "Object name is cut off.");
  float pos[3], q[4], scale[3];
  for (float& v : pos) v = r.get<float>();
  for (float& v : q) v = r.get<float>();
  for (float& v : scale) v = r.get<float>();
  out.transform.position = Vec3{pos[0], pos[1], pos[2]};
  out.transform.rotation = Quat{q[0], q[1], q[2], q[3]};
  out.transform.scale = Vec3{scale[0], scale[1], scale[2]};
  out.visible = r.get<std::uint8_t>() != 0;

  const auto vertexCount = r.get<std::uint32_t>();
  std::vector<float> coords;
  if (!r.ok() || vertexCount > std::uint32_t(INT32_MAX) || !r.getArray(coords, std::size_t(vertexCount) * 3))
    return setError(error, "Vertex data of \"" + out.name + "\" is cut off.");
  const auto faceCount = r.get<std::uint32_t>();
  const auto indexCount = r.get<std::uint32_t>();
  std::vector<std::uint32_t> sizes, indices;
  if (!r.ok() || !r.getArray(sizes, faceCount) || !r.getArray(indices, indexCount))
    return setError(error, "Face data of \"" + out.name + "\" is cut off.");

  std::uint64_t total = 0;
  for (std::uint32_t s : sizes) {
    if (s < 3) return setError(error, "\"" + out.name + "\" has a face with fewer than 3 corners.");
    total += s;
  }
  if (total != indexCount) return setError(error, "Face sizes of \"" + out.name + "\" do not match its indices.");
  for (std::uint32_t i : indices)
    if (i >= vertexCount) return setError(error, "\"" + out.name + "\" has a face index out of range.");

  std::vector<Vec3> positions(vertexCount);
  std::memcpy(positions.data(), coords.data(), coords.size() * sizeof(float));
  std::vector<Index> idx(indices.begin(), indices.end());
  std::vector<Index> sz(sizes.begin(), sizes.end());
  out.mesh = buildMesh(std::move(positions), idx, sz);
  return true;
}

using Canonicals = std::vector<std::optional<CanonicalLevel>>;

void writeMasks(Writer& w, const Scene& scene, const Canonicals& canonical) {
  std::vector<std::uint32_t> masked;
  for (std::size_t i = 0; i < scene.objects().size(); ++i)
    if (scene.objects()[i]->mesh.anyMasked()) masked.push_back(static_cast<std::uint32_t>(i));
  if (masked.empty()) return;
  const std::size_t at = w.beginChunk(kTagMask);
  w.put(static_cast<std::uint32_t>(masked.size()));
  for (std::uint32_t i : masked) {
    const Mesh& m = scene.objects()[i]->mesh;
    const std::vector<float>& values = canonical[i] ? canonical[i]->mask : m.mask;
    w.put(i);
    w.put(static_cast<std::uint32_t>(values.size()));
    w.put(kMaskEncodingF32);
    w.putArray(values.data(), values.size());
  }
  w.endChunk(at);
}

// Attaches the masks of a MASK chunk to the parsed objects.
bool readMasks(Reader& r, std::size_t size, std::vector<ProjectObject>& objects, std::string* error) {
  const auto count = r.get<std::uint32_t>();
  if (!r.ok() || count > size) return setError(error, "The mask data is damaged.");
  for (std::uint32_t k = 0; k < count; ++k) {
    const auto index = r.get<std::uint32_t>();
    const auto vertexCount = r.get<std::uint32_t>();
    const auto encoding = r.get<std::uint8_t>();
    if (!r.ok() || index >= objects.size() || encoding != kMaskEncodingF32 ||
        vertexCount != std::uint32_t(objects[index].mesh.vertexCount()))
      return setError(error, "The mask data is damaged.");
    std::vector<float> values;
    if (!r.getArray(values, vertexCount)) return setError(error, "The mask data is cut off.");
    for (float& v : values) v = v > 0.0f ? std::min(v, 1.0f) : 0.0f;  // Also turns NaN into 0.
    objects[index].mesh.mask = std::move(values);
  }
  return true;
}

void writeFaceSets(Writer& w, const Scene& scene, const Canonicals& canonical) {
  std::vector<std::uint32_t> withSets;
  for (std::size_t i = 0; i < scene.objects().size(); ++i)
    if (scene.objects()[i]->mesh.hasFaceSetData()) withSets.push_back(static_cast<std::uint32_t>(i));
  if (withSets.empty()) return;
  const std::size_t at = w.beginChunk(kTagFaceSets);
  w.put(static_cast<std::uint32_t>(withSets.size()));
  for (std::uint32_t i : withSets) {
    const Mesh& m = scene.objects()[i]->mesh;
    const std::vector<std::int32_t>& values = canonical[i] ? canonical[i]->faceSets : m.faceSets;
    w.put(i);
    w.put(static_cast<std::uint32_t>(values.size()));
    w.put(kFaceSetEncodingI32);
    w.putArray(values.data(), values.size());
  }
  w.endChunk(at);
}

// Attaches the face sets of an FSET chunk to the parsed objects. The face count is compared with
// the built mesh, which is what the values are indexed by.
bool readFaceSets(Reader& r, std::size_t size, std::vector<ProjectObject>& objects, std::string* error) {
  const auto count = r.get<std::uint32_t>();
  if (!r.ok() || count > size) return setError(error, "The face set data is damaged.");
  for (std::uint32_t k = 0; k < count; ++k) {
    const auto index = r.get<std::uint32_t>();
    const auto faceCount = r.get<std::uint32_t>();
    const auto encoding = r.get<std::uint8_t>();
    if (!r.ok() || index >= objects.size() || encoding != kFaceSetEncodingI32 ||
        faceCount != std::uint32_t(objects[index].mesh.faceCount()))
      return setError(error, "The face set data is damaged.");
    std::vector<std::int32_t> values;
    if (!r.getArray(values, faceCount)) return setError(error, "The face set data is cut off.");
    for (std::int32_t v : values)
      if (!validFaceSetValue(v)) return setError(error, "The face set data is damaged.");
    objects[index].mesh.faceSets = std::move(values);
  }
  return true;
}

template <class T>
void putList(Writer& w, const std::vector<std::uint32_t>& index, const std::vector<T>& values) {
  w.put(static_cast<std::uint32_t>(index.size()));
  w.putArray(index.data(), index.size());
  w.putArray(values.data(), values.size());
}

void writeLevels(Writer& w, const Scene& scene) {
  std::vector<std::uint32_t> withLevels;
  for (std::size_t i = 0; i < scene.objects().size(); ++i)
    if (scene.objects()[i]->multires) withLevels.push_back(static_cast<std::uint32_t>(i));
  if (withLevels.empty()) return;
  const std::size_t at = w.beginChunk(kTagLevels);
  w.put(static_cast<std::uint32_t>(withLevels.size()));
  for (std::uint32_t i : withLevels) {
    const MultiresFileData d = captureLevels(*scene.objects()[i]);
    w.put(i);
    w.put(kLevelsEncoding);
    w.put(static_cast<std::uint8_t>(d.levels.size()));
    w.put(static_cast<std::uint8_t>(d.active));
    w.put(std::uint8_t{0});
    w.put(d.baseVertices);
    w.put(static_cast<std::uint32_t>(d.baseSizes.size()));
    w.put(static_cast<std::uint32_t>(d.baseCorners.size()));
    w.putArray(d.baseSizes.data(), d.baseSizes.size());
    w.putArray(d.baseCorners.data(), d.baseCorners.size());
    w.putArray(d.baseTwins.data(), d.baseTwins.size());
    w.putArray(d.baseVertHe.data(), d.baseVertHe.size());
    for (const MultiresFileLevel& l : d.levels) {
      w.put(l.vertices);
      w.put(l.faces);
      w.put(l.channels);
      if (l.channels & kLevelPositions) w.putArray(l.positions.data(), l.positions.size());
      if (l.channels & kLevelMask) w.putArray(l.mask.data(), l.mask.size());
      if (l.channels & kLevelFaceSets) w.putArray(l.faceSets.data(), l.faceSets.size());
    }
    putList(w, d.pendingPosIndex, d.pendingPos);
    putList(w, d.pendingMaskIndex, d.pendingMask);
    putList(w, d.pendingSetIndex, d.pendingSets);
  }
  w.endChunk(at);
}

bool finite(const Vec3& p) { return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z); }

template <class T>
bool getList(Reader& r, std::vector<std::uint32_t>& index, std::vector<T>& values) {
  const auto n = r.get<std::uint32_t>();
  return r.ok() && r.getArray(index, n) && r.getArray(values, n);
}

// Attaches the level data of an MRES chunk to the parsed objects. Only the layout and the values
// are checked here; restoreLevels() checks that the levels fit their object.
bool readLevels(Reader& r, std::size_t size, std::vector<ProjectObject>& objects, std::string* error) {
  const std::string damaged = "The subdivision level data is damaged.";
  const auto count = r.get<std::uint32_t>();
  if (!r.ok() || count > size) return setError(error, damaged);
  std::int64_t last = -1;
  for (std::uint32_t k = 0; k < count; ++k) {
    const auto index = r.get<std::uint32_t>();
    const auto encoding = r.get<std::uint8_t>();
    const auto levelCount = r.get<std::uint8_t>();
    const auto active = r.get<std::uint8_t>();
    const auto reserved = r.get<std::uint8_t>();
    if (!r.ok() || index >= objects.size() || std::int64_t(index) <= last || encoding != kLevelsEncoding ||
        levelCount < 2 || levelCount > kMaxMultiresLevels || active >= levelCount || reserved != 0)
      return setError(error, damaged);
    last = index;
    auto d = std::make_shared<MultiresFileData>();
    d->active = active;
    d->baseVertices = r.get<std::uint32_t>();
    const auto faces = r.get<std::uint32_t>();
    const auto halfEdges = r.get<std::uint32_t>();
    if (!r.ok() || !r.getArray(d->baseSizes, faces) || !r.getArray(d->baseCorners, halfEdges) ||
        !r.getArray(d->baseTwins, halfEdges) || !r.getArray(d->baseVertHe, d->baseVertices))
      return setError(error, damaged);
    d->levels.resize(levelCount);
    for (MultiresFileLevel& l : d->levels) {
      l.vertices = r.get<std::uint32_t>();
      l.faces = r.get<std::uint32_t>();
      l.channels = r.get<std::uint8_t>();
      if (!r.ok() || (l.channels & ~(kLevelPositions | kLevelMask | kLevelFaceSets))) return setError(error, damaged);
      if ((l.channels & kLevelPositions) && !r.getArray(l.positions, l.vertices)) return setError(error, damaged);
      if ((l.channels & kLevelMask) && !r.getArray(l.mask, l.vertices)) return setError(error, damaged);
      if ((l.channels & kLevelFaceSets) && !r.getArray(l.faceSets, l.faces)) return setError(error, damaged);
      for (const Vec3& p : l.positions)
        if (!finite(p)) return setError(error, damaged);
      for (float& m : l.mask) m = m > 0.0f ? std::min(m, 1.0f) : 0.0f;  // Also turns NaN into 0.
      for (std::int32_t s : l.faceSets)
        if (!validFaceSetValue(s)) return setError(error, damaged);
    }
    if (!getList(r, d->pendingPosIndex, d->pendingPos) || !getList(r, d->pendingMaskIndex, d->pendingMask) ||
        !getList(r, d->pendingSetIndex, d->pendingSets))
      return setError(error, damaged);
    for (const Vec3& p : d->pendingPos)
      if (!finite(p)) return setError(error, damaged);
    for (float& m : d->pendingMask) m = m > 0.0f ? std::min(m, 1.0f) : 0.0f;
    for (std::int32_t s : d->pendingSets)
      if (!validFaceSetValue(s)) return setError(error, damaged);
    objects[index].levelData = std::move(d);
  }
  return true;
}

}  // namespace

std::uint32_t crc32(const std::uint8_t* data, std::size_t size, std::uint32_t crc) {
  static const std::array<std::uint32_t, 256> table = [] {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t i = 0; i < 256; ++i) {
      std::uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      t[i] = c;
    }
    return t;
  }();
  crc = ~crc;
  for (std::size_t i = 0; i < size; ++i) crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
  return ~crc;
}

std::vector<std::uint8_t> serializeProject(const Scene& scene, const std::string& settings) {
  Writer w;
  std::size_t total = 64 + 32 + settings.size();  // 32: MASK and FSET chunk headers and counts.
  for (const auto& obj : scene.objects())
    total += 64 + obj->name.size() + obj->mesh.positions.size() * 12 + std::size_t(obj->mesh.faceCount()) * 4 +
             std::size_t(obj->mesh.halfEdgeCount()) * 4 + (obj->mesh.mask.empty() ? 0 : 9 + obj->mesh.mask.size() * 4) +
             (obj->mesh.faceSets.empty() ? 0 : 9 + obj->mesh.faceSets.size() * 4) +
             (obj->multires ? obj->multires->bytes() / 3 : 0);
  w.bytes.reserve(total);

  w.putArray(kMagic, sizeof(kMagic));
  w.put(kProjectVersion);
  std::size_t at = w.beginChunk(kTagSettings);
  w.putArray(settings.data(), settings.size());
  w.endChunk(at);
  // Objects with levels store their active level in canonical order.
  Canonicals canonical(scene.objects().size());
  for (std::size_t i = 0; i < scene.objects().size(); ++i)
    if (scene.objects()[i]->multires) canonical[i] = canonicalActiveLevel(*scene.objects()[i]);
  at = w.beginChunk(kTagObjects);
  w.put(static_cast<std::uint32_t>(scene.objects().size()));
  for (std::size_t i = 0; i < scene.objects().size(); ++i)
    writeObject(w, *scene.objects()[i], canonical[i] ? &*canonical[i] : nullptr);
  w.endChunk(at);
  writeMasks(w, scene, canonical);
  writeFaceSets(w, scene, canonical);
  writeLevels(w, scene);
  const std::uint32_t crc = crc32(w.bytes.data(), w.bytes.size());
  at = w.beginChunk(kTagEnd);
  w.put(crc);
  w.endChunk(at);
  return std::move(w.bytes);
}

std::optional<Project> parseProject(const std::uint8_t* data, std::size_t size, std::string* error) {
  if (size < sizeof(kMagic) + 4 || std::memcmp(data, kMagic, sizeof(kMagic)) != 0) {
    setError(error, "Not a PleGL Sculpt project.");
    return std::nullopt;
  }
  Reader r(data + sizeof(kMagic), size - sizeof(kMagic));
  const auto version = r.get<std::uint32_t>();
  if (version > kProjectVersion) {
    setError(error,
             "The project was saved by a newer version of PleGL Sculpt (format " + std::to_string(version) + ").");
    return std::nullopt;
  }
  Project project;
  bool sawObjects = false;
  const std::uint8_t* maskData = nullptr;  // Applied after the loop, once the objects exist.
  std::size_t maskSize = 0;
  const std::uint8_t* faceSetData = nullptr;
  std::size_t faceSetSize = 0;
  const std::uint8_t* levelData = nullptr;
  std::size_t levelSize = 0;
  for (;;) {
    const std::uint8_t* chunkStart = r.position();
    const auto t = r.get<std::uint32_t>();
    const auto chunkSize = r.get<std::uint64_t>();
    if (!r.ok() || chunkSize > r.remaining()) {
      setError(error, "The project file is cut off or damaged.");
      return std::nullopt;
    }
    if (t == kTagEnd) {
      const auto stored = r.get<std::uint32_t>();
      if (!r.ok() || stored != crc32(data, std::size_t(chunkStart - data))) {
        setError(error, "The project file is damaged (checksum mismatch).");
        return std::nullopt;
      }
      break;
    }
    Reader chunk(r.position(), std::size_t(chunkSize));
    r.skip(std::size_t(chunkSize));
    if (t == kTagSettings) {
      project.settings.assign(reinterpret_cast<const char*>(chunk.position()), std::size_t(chunkSize));
    } else if (t == kTagObjects) {
      const auto count = chunk.get<std::uint32_t>();
      if (!chunk.ok() || count > chunkSize) {
        setError(error, "The object list is damaged.");
        return std::nullopt;
      }
      project.objects.resize(count);
      for (ProjectObject& obj : project.objects)
        if (!readObject(chunk, obj, error)) return std::nullopt;
      sawObjects = true;
    } else if (t == kTagMask) {
      maskData = chunk.position();
      maskSize = std::size_t(chunkSize);
    } else if (t == kTagFaceSets) {
      faceSetData = chunk.position();
      faceSetSize = std::size_t(chunkSize);
    } else if (t == kTagLevels) {
      levelData = chunk.position();
      levelSize = std::size_t(chunkSize);
    }
    // Unknown chunks are skipped.
  }
  if (!sawObjects) {
    setError(error, "The project file has no object list.");
    return std::nullopt;
  }
  if (maskData) {
    Reader masks(maskData, maskSize);
    if (!readMasks(masks, maskSize, project.objects, error)) return std::nullopt;
  }
  if (faceSetData) {
    Reader faceSets(faceSetData, faceSetSize);
    if (!readFaceSets(faceSets, faceSetSize, project.objects, error)) return std::nullopt;
  }
  if (levelData) {
    Reader levels(levelData, levelSize);
    if (!readLevels(levels, levelSize, project.objects, error)) return std::nullopt;
  }
  return project;
}

bool buildProject(Project& project, std::string* error) {
  for (ProjectObject& obj : project.objects) {
    if (obj.levelData) {
      if (!restoreLevels(*obj.levelData, obj.mesh, obj.bvh, obj.multires, error)) return false;
      obj.levelData.reset();
    } else {
      obj.bvh.build(obj.mesh);
    }
  }
  return true;
}

SceneObject& addProjectObject(Scene& scene, ProjectObject& po) {
  SceneObject& obj = scene.add(po.name, std::move(po.mesh), std::move(po.bvh));
  obj.name = po.name;  // Keep names exactly, even duplicates.
  obj.transform = po.transform;
  obj.visible = po.visible;
  if (po.multires) {
    obj.multires = std::move(po.multires);
    obj.topologyVersion = obj.multires->levels[static_cast<std::size_t>(obj.multires->active)].version;
  }
  return obj;
}

bool writeFileAtomic(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes, std::string* error) {
  std::filesystem::path tmp = path;
  tmp += ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) return setError(error, "Cannot write " + tmp.string());
    out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    out.flush();
    if (!out) {
      out.close();
      std::error_code ec;
      std::filesystem::remove(tmp, ec);
      return setError(error, "Writing " + tmp.string() + " failed (disk full?)");
    }
  }
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);  // Replaces an existing file, on Windows too.
  if (ec) {
    std::filesystem::remove(tmp, ec);
    return setError(error, "Cannot replace " + path.string() + ": " + ec.message());
  }
  return true;
}

std::optional<Project> loadProject(const std::filesystem::path& path, std::string* error) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    setError(error, "Cannot open " + path.string());
    return std::nullopt;
  }
  const std::streamsize size = in.tellg();
  in.seekg(0);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size > 0 ? size : 0));
  if (size > 0 && !in.read(reinterpret_cast<char*>(bytes.data()), size)) {
    setError(error, "Cannot read " + path.string());
    return std::nullopt;
  }
  return parseProject(bytes.data(), bytes.size(), error);
}

}  // namespace plegl
