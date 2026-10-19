#include "scene/Scene.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>

namespace plegl {

Mat4 Transform::matrix() const {
  Mat4 m = glm::translate(Mat4{1.0f}, position);
  m *= glm::mat4_cast(rotation);
  m = glm::scale(m, scale);
  return m;
}

Transform Transform::fromMatrix(const Mat4& m) {
  Transform t;
  Vec3 skew;
  Vec4 perspective;
  glm::decompose(m, t.scale, t.rotation, t.position, skew, perspective);
  return t;
}

std::uint64_t nextTopologyVersion() {
  static std::atomic<std::uint64_t> counter{1};
  return ++counter;
}

void SceneObject::rebuildSpatial() {
  assert(!multires && "levels keep their BVH layout for life");
  bvh.build(mesh);
  clearDirty();
  topologyVersion = nextTopologyVersion();
}

std::int32_t SceneObject::newFaceSetId() const {
  std::int32_t largest = mesh.maxFaceSetId();
  if (multires) largest = std::max(largest, multires->faceSetIdBound);
  return largest < kMaxFaceSetId ? largest + 1 : 0;
}

void SceneObject::markVisibilityDirty(Index leaf) {
  topoDirtyLeaves.push_back(leaf);
  const BvhLeaf& l = bvh.leaves()[leaf];
  Index last = leaf;
  for (Index h = l.heBegin; h < l.heEnd; ++h) {
    const Index t = mesh.heTwin[h];
    if (t == kInvalid || (t >= l.heBegin && t < l.heEnd)) continue;
    const Index other = bvh.leafOfHalfEdge(t);
    if (other == last || other == kInvalid) continue;
    topoDirtyLeaves.push_back(other);
    last = other;
  }
}

void SceneObject::markAllDirty() {
  clearDirty();
  const auto count = static_cast<Index>(bvh.leaves().size());
  dirtyLeaves.reserve(static_cast<std::size_t>(count));
  topoDirtyLeaves.reserve(static_cast<std::size_t>(count));
  for (Index l = 0; l < count; ++l) {
    dirtyLeaves.push_back(l);
    topoDirtyLeaves.push_back(l);
  }
  maskDirtyAll = true;
  faceSetDirtyAll = true;
}

SceneObject& Scene::add(std::string name, Mesh mesh) {
  auto obj = std::make_unique<SceneObject>();
  obj->id = nextId_++;
  obj->name = uniqueName(name);
  obj->mesh = std::move(mesh);
  obj->rebuildSpatial();
  objects_.push_back(std::move(obj));
  return *objects_.back();
}

SceneObject& Scene::add(std::string name, Mesh mesh, Bvh bvh) {
  auto obj = std::make_unique<SceneObject>();
  obj->id = nextId_++;
  obj->name = uniqueName(name);
  obj->mesh = std::move(mesh);
  obj->bvh = std::move(bvh);
  obj->topologyVersion = nextTopologyVersion();
  objects_.push_back(std::move(obj));
  return *objects_.back();
}

SceneObject* Scene::duplicate(std::uint32_t id) {
  const SceneObject* src = find(id);
  if (!src) return nullptr;
  auto obj = std::make_unique<SceneObject>();
  obj->id = nextId_++;
  obj->name = uniqueName(src->name);
  obj->mesh = src->mesh;
  obj->bvh = src->bvh;  // Same vertex order, so the copy is valid as is.
  obj->topologyVersion = nextTopologyVersion();
  if (src->multires) {
    // Every level gets a new version; pending edits come along as they are.
    obj->multires = std::shared_ptr<Multires>(src->multires->clone(true));
    obj->topologyVersion = obj->multires->levels[static_cast<std::size_t>(obj->multires->active)].version;
  }
  obj->transform = src->transform;
  obj->visible = src->visible;
  objects_.push_back(std::move(obj));
  return objects_.back().get();
}

bool Scene::remove(std::uint32_t id) {
  auto it = std::find_if(objects_.begin(), objects_.end(), [&](const auto& o) { return o->id == id; });
  if (it == objects_.end()) return false;
  objects_.erase(it);
  return true;
}

SceneObject* Scene::find(std::uint32_t id) {
  for (auto& o : objects_)
    if (o->id == id) return o.get();
  return nullptr;
}

const SceneObject* Scene::find(std::uint32_t id) const {
  for (const auto& o : objects_)
    if (o->id == id) return o.get();
  return nullptr;
}

std::optional<ScenePick> Scene::pick(const Ray& worldRay) const {
  std::optional<ScenePick> best;
  for (const auto& o : objects_) {
    if (!o->visible) continue;
    const Mat4 model = o->transform.matrix();
    const Mat4 inv = glm::inverse(model);
    // An affine map keeps the ray parameter, so local t equals world t.
    Ray local;
    local.origin = Vec3(inv * Vec4(worldRay.origin, 1.0f));
    local.dir = Vec3(inv * Vec4(worldRay.dir, 0.0f));
    RayHit hit;
    const float tMax = best ? best->t : std::numeric_limits<float>::infinity();
    if (!o->bvh.raycast(o->mesh, local, hit, tMax, true)) continue;
    ScenePick p;
    p.objectId = o->id;
    p.localHit = hit;
    p.t = hit.t;
    p.worldPosition = worldRay.origin + worldRay.dir * hit.t;
    const Mat3 normalMat = glm::transpose(glm::inverse(Mat3(model)));
    p.worldNormal = glm::normalize(normalMat * hit.smoothNormal);
    best = p;
  }
  return best;
}

Aabb Scene::worldBounds() const {
  Aabb b;
  for (const auto& o : objects_) {
    if (!o->visible || o->bvh.empty()) continue;
    const Aabb lb = o->bvh.bounds();
    const Mat4 m = o->transform.matrix();
    for (int i = 0; i < 8; ++i) {
      const Vec3 c{(i & 1) ? lb.max.x : lb.min.x, (i & 2) ? lb.max.y : lb.min.y, (i & 4) ? lb.max.z : lb.min.z};
      b.expand(Vec3(m * Vec4(c, 1.0f)));
    }
  }
  return b;
}

std::string Scene::uniqueName(const std::string& base) const {
  auto taken = [&](const std::string& n) {
    return std::any_of(objects_.begin(), objects_.end(), [&](const auto& o) { return o->name == n; });
  };
  if (!taken(base)) return base;
  // "Sphere.002" duplicates as "Sphere.003", not "Sphere.002.001".
  std::string stem = base;
  if (stem.size() > 4 && stem[stem.size() - 4] == '.' &&
      std::all_of(stem.end() - 3, stem.end(), [](char c) { return c >= '0' && c <= '9'; }))
    stem.resize(stem.size() - 4);
  for (int i = 1;; ++i) {
    char suffix[16];
    std::snprintf(suffix, sizeof(suffix), ".%03d", i);
    if (!taken(stem + suffix)) return stem + suffix;
  }
}

}  // namespace plegl
