#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "scene/Scene.h"

namespace plegl {

// Subdivision levels as a project file stores them (the MRES chunk, see io/Project.h): the base
// topology and every level's values in canonical order, so they load back bit for bit whatever
// order the BVH builds pick. The active level's values are not here: they are the object's
// regular vertices, mask and face sets (OBJS, MASK, FSET), also in canonical order, so a build
// without multires opens the level the artist was looking at.
struct MultiresFileLevel {
  std::uint32_t vertices = 0, faces = 0;
  std::uint8_t channels = 0;  // kLevelPositions | kLevelMask | kLevelFaceSets; 0 on the active level.
  std::vector<Vec3> positions;
  std::vector<float> mask;
  std::vector<std::int32_t> faceSets;
};
inline constexpr std::uint8_t kLevelPositions = 1, kLevelMask = 2, kLevelFaceSets = 4;

struct MultiresFileData {
  int active = 0;
  // Level 0 in canonical order: face sizes, corners (canonical vertices from each start corner),
  // the twin of every canonical half-edge (or -1) and the fan start of every vertex (or -1).
  std::uint32_t baseVertices = 0;
  std::vector<std::uint32_t> baseSizes, baseCorners;
  std::vector<std::int32_t> baseTwins, baseVertHe;
  std::vector<MultiresFileLevel> levels;
  // Pending edits of the active level: reference values where the live level differs, by
  // canonical index, ascending.
  std::vector<std::uint32_t> pendingPosIndex, pendingMaskIndex, pendingSetIndex;
  std::vector<Vec3> pendingPos;
  std::vector<float> pendingMask;
  std::vector<std::int32_t> pendingSets;
};

// The active level of an object with levels in canonical order, as OBJS, MASK and FSET store it.
struct CanonicalLevel {
  std::vector<Vec3> positions;
  std::vector<std::uint32_t> sizes, corners;
  std::vector<float> mask;               // Empty when the level has none.
  std::vector<std::int32_t> faceSets;    // Empty when the level has none.
};
CanonicalLevel canonicalActiveLevel(const SceneObject& object);

// Everything else about the object's levels (main thread, no sync: pending edits are stored).
MultiresFileData captureLevels(const SceneObject& object);

// Rebuilds a stack from file data. `mesh` holds the active level as read from OBJS, MASK and FSET
// (canonical order); on success it becomes the live active level, `bvh` its BVH and `stack` the
// levels, each with a fresh topology version. Every level is regenerated from the base by
// subdivision and must match the stored counts and the OBJS polygons exactly; anything else
// refuses the data. Worker-safe.
bool restoreLevels(const MultiresFileData& data, Mesh& mesh, Bvh& bvh, std::shared_ptr<Multires>& stack,
                   std::string* error = nullptr);

}  // namespace plegl
