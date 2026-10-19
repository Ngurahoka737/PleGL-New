#include "io/Project.h"

#include <array>
#include <bit>
#include <cstring>
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

void writeObject(Writer& w, const SceneObject& obj) {
  w.putString(obj.name);
  const Transform& t = obj.transform;
  w.putArray(&t.position.x, 3);
  const float q[4] = {t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z};
  w.putArray(q, 4);
  w.putArray(&t.scale.x, 3);
  w.put(static_cast<std::uint8_t>(obj.visible ? 1 : 0));

  const Mesh& m = obj.mesh;
  w.put(static_cast<std::uint32_t>(m.vertexCount()));
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
  std::size_t total = 64 + settings.size();
  for (const auto& obj : scene.objects())
    total += 64 + obj->name.size() + obj->mesh.positions.size() * 12 + std::size_t(obj->mesh.faceCount()) * 4 +
             std::size_t(obj->mesh.halfEdgeCount()) * 4;
  w.bytes.reserve(total);

  w.putArray(kMagic, sizeof(kMagic));
  w.put(kProjectVersion);
  std::size_t at = w.beginChunk(kTagSettings);
  w.putArray(settings.data(), settings.size());
  w.endChunk(at);
  at = w.beginChunk(kTagObjects);
  w.put(static_cast<std::uint32_t>(scene.objects().size()));
  for (const auto& obj : scene.objects()) writeObject(w, *obj);
  w.endChunk(at);
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
    }
    // Unknown chunks are skipped.
  }
  if (!sawObjects) {
    setError(error, "The project file has no object list.");
    return std::nullopt;
  }
  return project;
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
