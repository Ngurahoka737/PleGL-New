#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "mesh/Mesh.h"
#include "multires/Multires.h"
#include "spatial/Bvh.h"

namespace plegl {

struct Transform {
  Vec3 position{0.0f};
  Quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
  Vec3 scale{1.0f};

  Mat4 matrix() const;
  static Transform fromMatrix(const Mat4& m);
};

// Process-wide counter, so a topology version is never reused for a different mesh (undo entries
// are matched by object id and version).
std::uint64_t nextTopologyVersion();

// Above this many changed leaves, a mask or face set edit asks for one whole upload instead of
// per-leaf uploads.
inline constexpr std::size_t kMaskDirtyAllLeaves = 64;

struct SceneObject {
  std::uint32_t id = 0;
  std::string name;
  Mesh mesh;
  Bvh bvh;
  Transform transform;
  bool visible = true;

  // Bumped whenever topology or vertex order changes; renderers rebuild their buffers then.
  std::uint64_t topologyVersion = 1;
  // Leaves whose vertex positions or normals changed since the renderer last uploaded them.
  std::vector<Index> dirtyLeaves;
  // Leaves whose mask values changed, or the whole mask when maskDirtyAll is set.
  std::vector<Index> maskDirtyLeaves;
  bool maskDirtyAll = false;
  // Leaves whose face set values changed (colours only), or all of them when faceSetDirtyAll is
  // set.
  std::vector<Index> faceSetDirtyLeaves;
  bool faceSetDirtyAll = false;
  // Leaves whose faces changed during a dynamic topology stroke (topologyVersion only changes
  // when the stroke ends), or whose faces were hidden or revealed, so their GPU index data must
  // be rebuilt.
  std::vector<Index> topoDirtyLeaves;
  // topologyVersion at which the mesh was last found free of non-manifold vertices.
  std::uint64_t manifoldCheckedVersion = 0;
  // Subdivision levels, or null for a plain object. While set, `mesh` and `bvh` hold the active
  // level and the stack holds the others (see multires/Multires.h). Never shared between objects:
  // the pointer is shared only so background work can keep a stack alive.
  std::shared_ptr<Multires> multires;

  // A face set id no face uses on any level, or 0 when ids have run out.
  std::int32_t newFaceSetId() const;

  // Rebuilds the BVH (which reorders the mesh) and bumps topologyVersion. Not for objects with
  // subdivision levels, whose layouts are frozen.
  void rebuildSpatial();
  void markLeafDirty(Index leaf) { dirtyLeaves.push_back(leaf); }
  void markMaskDirty(Index leaf) { maskDirtyLeaves.push_back(leaf); }
  void markTopologyDirty(Index leaf) { topoDirtyLeaves.push_back(leaf); }
  void markMaskDirtyAll() {
    maskDirtyAll = true;
    maskDirtyLeaves.clear();
  }
  void markFaceSetDirty(Index leaf) { faceSetDirtyLeaves.push_back(leaf); }
  // Faces of `leaf` were hidden or shown: the leaf draws other triangles and edges now, and so may
  // the leaves next to it (an edge is drawn by whichever of its two faces shows).
  void markVisibilityDirty(Index leaf);
  void markFaceSetDirtyAll() {
    faceSetDirtyAll = true;
    faceSetDirtyLeaves.clear();
  }
  // Asks for every leaf's positions, mask, face sets and triangles again. For a mesh swapped back
  // in under a version the renderer may already hold with other values (a subdivision level that
  // left and came back within one frame).
  void markAllDirty();
  // Drops pending partial uploads; for use after a topology change, which re-uploads everything.
  void clearDirty() {
    dirtyLeaves.clear();
    maskDirtyLeaves.clear();
    maskDirtyAll = false;
    faceSetDirtyLeaves.clear();
    faceSetDirtyAll = false;
    topoDirtyLeaves.clear();
  }
};

struct ScenePick {
  std::uint32_t objectId = 0;
  RayHit localHit;          // In the object's local space.
  Vec3 worldPosition{0.0f};
  Vec3 worldNormal{0.0f};   // Smooth normal in world space.
  float t = 0.0f;           // Along the world ray, in units of its direction.
};

class Scene {
 public:
  // Builds the BVH (reordering the mesh) and adds the object.
  SceneObject& add(std::string name, Mesh mesh);
  // Adds a mesh whose BVH was already built, for example on a background thread.
  SceneObject& add(std::string name, Mesh mesh, Bvh bvh);
  SceneObject* duplicate(std::uint32_t id);
  bool remove(std::uint32_t id);
  void clear() { objects_.clear(); }

  SceneObject* find(std::uint32_t id);
  const SceneObject* find(std::uint32_t id) const;

  // Nearest visible object hit by a world-space ray. Hidden faces are not hit.
  std::optional<ScenePick> pick(const Ray& worldRay) const;

  Aabb worldBounds() const;
  std::string uniqueName(const std::string& base) const;

  const std::vector<std::unique_ptr<SceneObject>>& objects() const { return objects_; }

 private:
  std::vector<std::unique_ptr<SceneObject>> objects_;
  std::uint32_t nextId_ = 1;
};

}  // namespace plegl
