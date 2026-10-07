#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "scene/Scene.h"

namespace plegl {

// One object of a project file. Topology is stored as polygons and rebuilt on load.
struct ProjectObject {
  std::string name;
  Transform transform;
  bool visible = true;
  Mesh mesh;
};

struct Project {
  std::vector<ProjectObject> objects;
  // Application state (brush, camera, symmetry, viewport...) as "key value" lines. The engine
  // stores it verbatim; the app decides what goes in it, so new settings need no format change.
  std::string settings;
};

// The .psculpt format: a small binary container.
//
//   "PSCULPT\x1A"  u32 version
//   chunks:        u32 tag, u64 payload size, payload
//   "END " chunk:  u32 CRC-32 of every byte before it
//
// Chunks: "SETT" (settings text), "OBJS" (objects: name, transform, visibility, positions,
// face sizes, face indices) and the optional "MASK" (u32 count, then per masked object: u32 object
// index, u32 vertex count, u8 encoding (0 = f32), the values). MASK is written only when some
// object has a non-zero mask. Readers skip unknown chunks, so later versions can add data that
// older builds ignore (a build without masking opens a masked file and drops the mask).
// Little-endian.
inline constexpr std::uint32_t kProjectVersion = 1;

// Serializes in memory, so the caller can write the bytes on another thread while the scene keeps
// changing (autosave does this).
std::vector<std::uint8_t> serializeProject(const Scene& scene, const std::string& settings);

// Parses and validates; every count and index is checked, and a damaged or truncated file is
// refused instead of producing a broken mesh.
std::optional<Project> parseProject(const std::uint8_t* data, std::size_t size, std::string* error = nullptr);

// Writes to "<path>.tmp" and renames it over `path`, so a crash while saving never leaves a
// half-written project behind.
bool writeFileAtomic(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes,
                     std::string* error = nullptr);

std::optional<Project> loadProject(const std::filesystem::path& path, std::string* error = nullptr);

std::uint32_t crc32(const std::uint8_t* data, std::size_t size, std::uint32_t crc = 0);

}  // namespace plegl
