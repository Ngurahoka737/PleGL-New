#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "multires/MultiresIo.h"
#include "scene/Scene.h"

namespace plegl {

// One object of a project file. Topology is stored as polygons and rebuilt on load.
struct ProjectObject {
  std::string name;
  Transform transform;
  bool visible = true;
  Mesh mesh;
  // Subdivision levels as read from the file, rebuilt by buildProject().
  std::shared_ptr<MultiresFileData> levelData;
  // Filled by buildProject(): the mesh's BVH and, for an object with levels, its stack (the mesh
  // is then the active level and the BVH the level's own).
  Bvh bvh;
  std::shared_ptr<Multires> multires;
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
// object has a non-zero mask. The optional "FSET" has the same layout per object, with a u32 face
// count and i32 face set values (encoding 0) in face order; it is written only for objects with a
// face set other than the default or a hidden face. Readers skip unknown chunks, so later versions
// can add data that older builds ignore (a build without masking opens a masked file and drops the
// mask).
//
// The optional "MRES" holds subdivision levels (see multires/MultiresIo.h): u32 count, then per
// object with levels: u32 object index (ascending), u8 encoding (0), u8 level count (2..8), u8
// active level, u8 reserved (0); the base level as u32 vertex, face and half-edge counts, u32 face
// sizes, u32 corners, i32 twins and i32 vertex fan starts (canonical half-edges, -1 for none);
// per level u32 vertex and face counts, u8 channels (1 positions, 2 mask, 4 face sets; 0 on the
// active level) and the channels' values; then the active level's pending edits as three lists
// (u32 count, u32 canonical indices, values) for positions, mask and face sets. For such objects
// OBJS, MASK and FSET hold the active level in canonical order, so a build without levels opens
// the level that was being sculpted.
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

// Builds every object's BVH and rebuilds subdivision levels (the slow part of opening a project;
// worker-safe). Fails, refusing the whole project, when level data does not fit its object.
bool buildProject(Project& project, std::string* error = nullptr);

// Moves a built object (see buildProject) into the scene, keeping its name exactly.
SceneObject& addProjectObject(Scene& scene, ProjectObject& object);

std::uint32_t crc32(const std::uint8_t* data, std::size_t size, std::uint32_t crc = 0);

}  // namespace plegl
