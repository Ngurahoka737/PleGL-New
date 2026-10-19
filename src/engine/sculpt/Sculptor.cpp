#include "sculpt/Sculptor.h"

#include <algorithm>
#include <cmath>

#include "core/Parallel.h"
#include "core/Timer.h"

namespace plegl {

void Sculptor::beginStroke(SceneObject& object, const Brush& brush, const StrokeOptions& options, std::string label) {
  object_ = &object;
  brush_ = &brush;
  options_ = options;
  undo_ = SculptUndo{};
  undo_.label = std::move(label);
  undo_.objectId = object.id;
  undo_.topologyVersion = object.topologyVersion;
  snapshotIndex_.clear();
  if (vertexStamp_.size() != object.mesh.positions.size()) {
    vertexStamp_.assign(object.mesh.positions.size(), 0);
    stamp_ = 0;
  }
  dabCount_ = 0;
}

void Sculptor::snapshot(Index leaf) {
  if (snapshotIndex_.count(leaf)) return;
  const BvhLeaf& l = object_->bvh.leaves()[leaf];
  const Mesh& m = object_->mesh;
  LeafState s;
  s.leaf = leaf;
  s.positions.assign(m.positions.begin() + l.vertBegin, m.positions.begin() + l.vertEnd);
  s.normals.assign(m.normals.begin() + l.vertBegin, m.normals.begin() + l.vertEnd);
  snapshotIndex_.emplace(leaf, undo_.before.size());
  undo_.before.push_back(std::move(s));
}

Vec3 Sculptor::areaNormal(const Vec3& center, float radius, std::span<const Index> leaves) const {
  const Mesh& m = object_->mesh;
  const float r2 = radius * radius;
  Vec3 sum{0.0f};
  for (Index li : leaves) {
    const BvhLeaf& leaf = object_->bvh.leaves()[li];
    for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
      const Vec3 d = m.positions[v] - center;
      const float dist2 = glm::dot(d, d);
      if (dist2 < r2) sum += m.normals[v] * falloffWeight(options_.falloff, std::sqrt(dist2) / radius);
    }
  }
  const float len = glm::length(sum);
  return len > 1e-12f ? sum / len : Vec3{0.0f};
}

bool Sculptor::dab(const Vec3& center, float radius, float strength) {
  if (!object_ || radius <= 0.0f) return false;
  Timer total;
  lastDab_ = {};
  Dab d;
  d.center = center;
  d.radius = radius;
  d.strength = std::clamp(strength, 0.0f, 1.0f);
  d.falloff = options_.falloff;
  d.invert = options_.invert;
  bool any = applyOne(d);
  if (options_.symmetryX) {
    Dab mirrored = d;
    mirrored.center.x = -mirrored.center.x;
    // Skip the mirror when the dab sits on the plane, or the seam would get double strength.
    if (std::abs(center.x) > radius * 0.05f) any |= applyOne(mirrored);
  }
  lastDab_.totalMs = total.ms();
  if (any) ++dabCount_;
  return any;
}

bool Sculptor::applyOne(const Dab& dabIn) {
  Mesh& m = object_->mesh;
  Bvh& bvh = object_->bvh;
  Dab dab = dabIn;

  leaves_.clear();
  bvh.querySphere(dab.center, dab.radius, leaves_);
  if (leaves_.empty()) return false;
  if (brush_->needsAreaNormal()) {
    dab.areaNormal = areaNormal(dab.center, dab.radius, leaves_);
    if (dab.areaNormal == Vec3{0.0f}) return false;  // No vertex inside the dab.
  }
  for (Index l : leaves_) snapshot(l);

  Timer t;
  BrushContext ctx{m, bvh, leaves_, dab};
  brush_->apply(ctx);
  lastDab_.brushMs += t.ms();

  // Brushes only move vertices inside the dab sphere. Every face touching such a vertex lies in a
  // queried leaf (its bounds contain the vertex), so the vertices of those faces are exactly the
  // ones whose normals can have changed. Positions are read after the brush ran, so the sphere
  // test uses a slightly larger radius to cover vertices that moved outward across it.
  t.reset();
  if (++stamp_ == 0) {
    std::fill(vertexStamp_.begin(), vertexStamp_.end(), 0);
    stamp_ = 1;
  }
  normalVerts_.clear();
  const float reach = dab.radius * 1.25f;
  const float reach2 = reach * reach;
  for (Index li : leaves_) {
    const BvhLeaf& leaf = bvh.leaves()[li];
    for (Index f = leaf.faceBegin; f < leaf.faceEnd; ++f) {
      bool touched = false;
      m.forEachFaceVertex(f, [&](Index v) {
        const Vec3 d = m.positions[v] - dab.center;
        touched |= glm::dot(d, d) < reach2;
      });
      if (!touched) continue;
      m.forEachFaceVertex(f, [&](Index v) {
        if (vertexStamp_[v] != stamp_) {
          vertexStamp_[v] = stamp_;
          normalVerts_.push_back(v);
        }
      });
    }
  }
  // Vertices owned by leaves outside the query change normals too; snapshot and upload those.
  dirtyLeaves_.assign(leaves_.begin(), leaves_.end());
  {
    // Owners of the normal vertices, found once per contiguous run.
    Index lastOwner = kInvalid;
    std::sort(normalVerts_.begin(), normalVerts_.end());  // Also improves locality below.
    for (Index v : normalVerts_) {
      if (lastOwner != kInvalid) {
        const BvhLeaf& l = bvh.leaves()[lastOwner];
        if (v >= l.vertBegin && v < l.vertEnd) continue;
      }
      lastOwner = bvh.leafOfVertex(v);
      if (lastOwner == kInvalid) continue;
      snapshot(lastOwner);
      dirtyLeaves_.push_back(lastOwner);
    }
  }
  parallelFor(0, normalVerts_.size(), 2048, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const Index v = normalVerts_[i];
      Vec3 n{0.0f};
      m.forEachOutgoing(v, [&](Index h) { n += m.faceAreaNormal(m.heFace[h]); });
      const float len = glm::length(n);
      if (len > 1e-20f) m.normals[v] = n / len;
    }
  });
  lastDab_.normalsMs += t.ms();
  lastDab_.vertices += static_cast<int>(normalVerts_.size());
  lastDab_.leaves += static_cast<int>(leaves_.size());

  bvh.refitLeaves(m, leaves_);
  std::sort(dirtyLeaves_.begin(), dirtyLeaves_.end());
  dirtyLeaves_.erase(std::unique(dirtyLeaves_.begin(), dirtyLeaves_.end()), dirtyLeaves_.end());
  for (Index l : dirtyLeaves_) object_->markLeafDirty(l);
  return true;
}

std::optional<SculptUndo> Sculptor::endStroke() {
  if (!object_) return std::nullopt;
  SceneObject* obj = object_;
  object_ = nullptr;
  if (undo_.before.empty()) return std::nullopt;
  const Mesh& m = obj->mesh;
  undo_.after.reserve(undo_.before.size());
  for (const LeafState& b : undo_.before) {
    const BvhLeaf& l = obj->bvh.leaves()[b.leaf];
    LeafState a;
    a.leaf = b.leaf;
    a.positions.assign(m.positions.begin() + l.vertBegin, m.positions.begin() + l.vertEnd);
    a.normals.assign(m.normals.begin() + l.vertBegin, m.normals.begin() + l.vertEnd);
    undo_.after.push_back(std::move(a));
  }
  return std::move(undo_);
}

}  // namespace plegl
