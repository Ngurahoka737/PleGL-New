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
  // Sculpt layers as read from the file, attached by buildProject(): the stack of an object
  // without levels (file vertex order), or the stacks of an object's levels by level (canonical
  // order). Both have no epoch yet.
  LayerStack layers;
  std::vector<std::pair<int, LayerStack>> levelLayers;
};

struct Project {
  std::vector<ProjectObject> objects;
  // Application state (brush, camera, symmetry, viewport...) as "key value" lines. The engine
  // stores it verbatim; the app decides what goes in it, so new settings need no format change.
  std::string settings;
  // Data that could not be read and was left out while the project still opened (for the status
  // line).
  std::vector<std::string> warnings;
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
//
// The optional "LAYR" holds sculpt layers (written after MRES): u8 encoding (0), u8[3] reserved
// (0), u32 stack count, then per stack, ascending by object index and level (255 for an object
// without levels, which sorts last): u32 object index, u8 level, u8 layer count (1..255), u8
// active slot (0 = base, k = k-th layer from the bottom), u8 reserved (0), u32 vertex count, u32
// next id, the base as an array block, and per layer bottom to top: u32 id, f32 strength, u8
// flags (bit 0 = visible), u8 name length (0..63), the name (UTF-8), the offsets as an array
// block. An array block is u8 code, then f32[3 x vertex count] (code 0, dense), or u32 count,
// u32 indices (ascending) and f32[3 x count] (code 1: zero elsewhere; code 2, base only: the
// composite elsewhere). Vertices are in file order, or a level's canonical order. The composite
// is what OBJS (or MRES, for a parked level) stores, so a build without layers opens the shape
// that was being sculpted; a damaged LAYR drops the layers with a warning and keeps that shape.
// Little-endian.
inline constexpr std::uint32_t kProjectVersion = 1;

// A sparse or dense array captured for LAYR.
struct CapturedArray {
  std::vector<std::uint32_t> index;  // Ascending, unless `unsorted` (canonical order, not sorted yet).
  std::vector<Vec3> values;          // Parallel to `index`, or every vertex's value when `dense`.
  bool unsorted = false;
  bool dense = false;
};
struct CapturedLayer {
  std::uint32_t id = 0;
  std::string name;
  float strength = 1.0f;
  bool visible = true;
  CapturedArray offsets;  // Non-zero offsets.
};
struct CapturedStack {
  std::uint32_t objectIndex = 0;
  std::uint8_t level = 255;
  std::uint8_t activeSlot = 0;
  std::uint32_t vertexCount = 0, nextId = 1;
  CapturedArray base;  // Where the base differs from the composite.
  std::vector<CapturedLayer> layers;
};
// A save in two halves: draftProject() copies what the scene holds (main thread, quick) and
// finishProject() sorts, encodes and checksums it (any thread), so the scene can keep changing
// while the bytes are finished and written (autosave does this).
struct ProjectDraft {
  std::vector<std::uint8_t> bytes;  // Every chunk before LAYR.
  std::vector<CapturedStack> layers;
};
ProjectDraft draftProject(const Scene& scene, const std::string& settings);
std::vector<std::uint8_t> finishProject(ProjectDraft&& draft);

inline std::vector<std::uint8_t> serializeProject(const Scene& scene, const std::string& settings) {
  return finishProject(draftProject(scene, settings));
}

// Parses and validates; every count and index is checked, and a damaged or truncated file is
// refused instead of producing a broken mesh.
std::optional<Project> parseProject(const std::uint8_t* data, std::size_t size, std::string* error = nullptr);

// Writes to "<path>.tmp" and renames it over `path`, so a crash while saving never leaves a
// half-written project behind.
bool writeFileAtomic(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes,
                     std::string* error = nullptr);

std::optional<Project> loadProject(const std::filesystem::path& path, std::string* error = nullptr);

// Builds every object's BVH, rebuilds subdivision levels and attaches sculpt layers (the slow part
// of opening a project; worker-safe). Fails, refusing the whole project, when level data does not
// fit its object. Layers whose composite is off from the stored shape by rounding are recomposed;
// layers further off are dropped with a warning in project.warnings.
bool buildProject(Project& project, std::string* error = nullptr);

// Moves a built object (see buildProject) into the scene, keeping its name exactly.
SceneObject& addProjectObject(Scene& scene, ProjectObject& object);

std::uint32_t crc32(const std::uint8_t* data, std::size_t size, std::uint32_t crc = 0);

}  // namespace plegl
