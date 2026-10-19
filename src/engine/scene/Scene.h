#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "mesh/Mesh.h"
#include "spatial/Bvh.h"

namespace plegl {

struct Transform {
  Vec3 position{0.0f};
  Quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
  Vec3 scale{1.0f};

  Mat4 matrix() const;
  static Transform fromMatrix(const Mat4& m);
};

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

  // Rebuilds the BVH (which reorders the mesh) and bumps topologyVersion.
  void rebuildSpatial();
  void markLeafDirty(Index leaf) { dirtyLeaves.push_back(leaf); }
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

  // Nearest visible object hit by a world-space ray.
  std::optional<ScenePick> pick(const Ray& worldRay) const;

  Aabb worldBounds() const;
  std::string uniqueName(const std::string& base) const;

  const std::vector<std::unique_ptr<SceneObject>>& objects() const { return objects_; }

 private:
  std::vector<std::unique_ptr<SceneObject>> objects_;
  std::uint32_t nextId_ = 1;
};

}  // namespace plegl
