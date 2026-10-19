#include "sculpt/Sculptor.h"

#include <algorithm>
#include <cmath>

#include "core/Parallel.h"
#include "core/Timer.h"

namespace plegl {

void Sculptor::beginStroke(SceneObject& object, const Brush& brush, const StrokeOptions& options, std::string label) {
  start(object, options, std::move(label));
  brush_ = &brush;
  maskStroke_ = brush.editsMask();
  // Allocate before the first snapshot, so the before-state holds real (zero) values.
  if (maskStroke_) object.mesh.ensureMask();
}

void Sculptor::start(SceneObject& object, const StrokeOptions& options, std::string label) {
  object_ = &object;
  brush_ = nullptr;
  maskStroke_ = false;
  grabVerts_.clear();
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
  if (maskStroke_) {
    s.mask.assign(m.mask.begin() + l.vertBegin, m.mask.begin() + l.vertEnd);
  } else {
    s.positions.assign(m.positions.begin() + l.vertBegin, m.positions.begin() + l.vertEnd);
    s.normals.assign(m.normals.begin() + l.vertBegin, m.normals.begin() + l.vertEnd);
  }
  snapshotIndex_.emplace(leaf, undo_.before.size());
  undo_.before.push_back(std::move(s));
}

bool Sculptor::computeArea(Dab& dab, std::span<const Index> leaves) const {
  const Mesh& m = object_->mesh;
  const float r2 = dab.radius * dab.radius;
  Vec3 normalSum{0.0f}, centerSum{0.0f};
  float weightSum = 0.0f;
  for (Index li : leaves) {
    const BvhLeaf& leaf = object_->bvh.leaves()[li];
    for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
      const Vec3 d = m.positions[v] - dab.center;
      const float dist2 = glm::dot(d, d);
      if (dist2 >= r2) continue;
      // Constant falloff would give rim vertices full say; a smooth weight keeps the plane stable.
      const float w = falloffWeight(Falloff::Smooth, std::sqrt(dist2) / dab.radius) + 1e-3f;
      normalSum += m.normals[v] * w;
      centerSum += m.positions[v] * w;
      weightSum += w;
    }
  }
  const float len = glm::length(normalSum);
  if (weightSum <= 0.0f || len <= 1e-12f) return false;
  dab.areaNormal = normalSum / len;
  dab.areaCenter = centerSum / weightSum;
  return true;
}

bool Sculptor::dab(const Vec3& center, float radius, float strength) {
  if (!object_ || !brush_ || radius <= 0.0f) return false;
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
  if (brush_->needsArea() && !computeArea(dab, leaves_)) return false;  // No vertex inside the dab.
  for (Index l : leaves_) snapshot(l);

  Timer t;
  BrushContext ctx{m, bvh, leaves_, dab};
  brush_->apply(ctx);
  lastDab_.brushMs += t.ms();

  if (maskStroke_) {  // Positions did not move: no normals, bounds or geometry upload.
    for (Index l : leaves_) object_->markMaskDirty(l);
    lastDab_.leaves += static_cast<int>(leaves_.size());
    return true;
  }

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
  recomputeNormals(normalVerts_);
  lastDab_.normalsMs += t.ms();
  lastDab_.vertices += static_cast<int>(normalVerts_.size());
  lastDab_.leaves += static_cast<int>(leaves_.size());

  bvh.refitLeaves(m, leaves_);
  std::sort(dirtyLeaves_.begin(), dirtyLeaves_.end());
  dirtyLeaves_.erase(std::unique(dirtyLeaves_.begin(), dirtyLeaves_.end()), dirtyLeaves_.end());
  for (Index l : dirtyLeaves_) object_->markLeafDirty(l);
  return true;
}

void Sculptor::recomputeNormals(std::span<const Index> verts) {
  Mesh& m = object_->mesh;
  parallelFor(0, verts.size(), 2048, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const Index v = verts[i];
      Vec3 n{0.0f};
      m.forEachOutgoing(v, [&](Index h) { n += m.faceAreaNormal(m.heFace[h]); });
      const float len = glm::length(n);
      if (len > 1e-20f) m.normals[v] = n / len;
    }
  });
}

bool Sculptor::beginGrab(SceneObject& object, const StrokeOptions& options, const Vec3& center, float radius,
                         std::string label) {
  if (radius <= 0.0f) return false;
  start(object, options, std::move(label));
  Mesh& m = object.mesh;
  const Bvh& bvh = object.bvh;
  const float strength = std::clamp(options.strength, 0.0f, 1.0f);

  // Collect weights per vertex; a vertex can sit in both spheres when they overlap at the seam.
  std::unordered_map<Index, std::size_t> slot;
  auto capture = [&](const Vec3& c, bool mirror) {
    leaves_.clear();
    bvh.querySphere(c, radius, leaves_);
    const float r2 = radius * radius;
    for (Index li : leaves_) {
      const BvhLeaf& leaf = bvh.leaves()[li];
      for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
        const Vec3 d = m.positions[v] - c;
        const float dist2 = glm::dot(d, d);
        if (dist2 >= r2) continue;
        float w = strength * falloffWeight(options.falloff, std::sqrt(dist2) / radius);
        if (!m.mask.empty()) w *= 1.0f - m.mask[v];  // Fully masked vertices stay put.
        if (w <= 0.0f) continue;
        auto [it, inserted] = slot.try_emplace(v, grabVerts_.size());
        if (inserted) grabVerts_.push_back({v, m.positions[v], 0.0f, 0.0f});
        (mirror ? grabVerts_[it->second].mirrorWeight : grabVerts_[it->second].weight) += w;
      }
    }
  };
  capture(center, false);
  if (options.symmetryX && std::abs(center.x) > radius * 0.05f) capture({-center.x, center.y, center.z}, true);
  if (grabVerts_.empty()) {
    object_ = nullptr;
    return false;
  }

  // Faces around captured vertices are the only ones that change shape: their vertices need new
  // normals and their leaves new bounds. Topology is fixed during a stroke, so compute this once.
  if (++stamp_ == 0) {
    std::fill(vertexStamp_.begin(), vertexStamp_.end(), 0);
    stamp_ = 1;
  }
  std::vector<Index> faces;
  for (const GrabVertex& g : grabVerts_) {
    m.forEachOutgoing(g.v, [&](Index h) {
      const Index f = m.heFace[h];
      if (f != kInvalid) faces.push_back(f);
    });
  }
  std::sort(faces.begin(), faces.end());
  faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
  grabNormalVerts_.clear();
  grabRefitLeaves_.clear();
  for (Index f : faces) {
    grabRefitLeaves_.push_back(bvh.leafOfFace(f));
    m.forEachFaceVertex(f, [&](Index v) {
      if (vertexStamp_[v] != stamp_) {
        vertexStamp_[v] = stamp_;
        grabNormalVerts_.push_back(v);
      }
    });
  }
  std::sort(grabRefitLeaves_.begin(), grabRefitLeaves_.end());
  grabRefitLeaves_.erase(std::unique(grabRefitLeaves_.begin(), grabRefitLeaves_.end()), grabRefitLeaves_.end());
  std::sort(grabNormalVerts_.begin(), grabNormalVerts_.end());
  grabDirtyLeaves_.clear();
  for (Index v : grabNormalVerts_) {
    const Index owner = bvh.leafOfVertex(v);
    if (owner != kInvalid && (grabDirtyLeaves_.empty() || grabDirtyLeaves_.back() != owner))
      grabDirtyLeaves_.push_back(owner);
  }
  for (Index l : grabDirtyLeaves_) snapshot(l);
  return true;
}

void Sculptor::grab(const Vec3& offset) {
  if (!object_ || grabVerts_.empty()) return;
  Timer total;
  lastDab_ = {};
  Mesh& m = object_->mesh;
  const Vec3 mirrored{-offset.x, offset.y, offset.z};
  Timer t;
  parallelFor(0, grabVerts_.size(), 4096, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const GrabVertex& g = grabVerts_[i];
      m.positions[g.v] = g.start + offset * g.weight + mirrored * g.mirrorWeight;
    }
  });
  lastDab_.brushMs = t.ms();
  t.reset();
  recomputeNormals(grabNormalVerts_);
  lastDab_.normalsMs = t.ms();
  object_->bvh.refitLeaves(m, grabRefitLeaves_);
  for (Index l : grabDirtyLeaves_) object_->markLeafDirty(l);
  lastDab_.vertices = static_cast<int>(grabNormalVerts_.size());
  lastDab_.leaves = static_cast<int>(grabRefitLeaves_.size());
  lastDab_.totalMs = total.ms();
  ++dabCount_;
}

std::optional<SculptUndo> Sculptor::endStroke() {
  if (!object_) return std::nullopt;
  SceneObject* obj = object_;
  object_ = nullptr;
  if (undo_.before.empty()) return std::nullopt;
  const Mesh& m = obj->mesh;
  if (maskStroke_) {
    // Drop leaves the mask brush passed over without changing (for example already fully masked).
    std::vector<LeafState> before;
    for (LeafState& b : undo_.before) {
      const BvhLeaf& l = obj->bvh.leaves()[b.leaf];
      if (std::equal(b.mask.begin(), b.mask.end(), m.mask.begin() + l.vertBegin)) continue;
      LeafState a;
      a.leaf = b.leaf;
      a.mask.assign(m.mask.begin() + l.vertBegin, m.mask.begin() + l.vertEnd);
      undo_.after.push_back(std::move(a));
      before.push_back(std::move(b));
    }
    if (before.empty()) return std::nullopt;
    undo_.before = std::move(before);
    return std::move(undo_);
  }
  // Drop leaves the stroke did not change, for example because they are fully masked. A leaf
  // whose positions stayed but whose normals changed (a neighbour moved) is kept.
  std::vector<LeafState> before;
  before.reserve(undo_.before.size());
  undo_.after.reserve(undo_.before.size());
  for (LeafState& b : undo_.before) {
    const BvhLeaf& l = obj->bvh.leaves()[b.leaf];
    if (std::equal(b.positions.begin(), b.positions.end(), m.positions.begin() + l.vertBegin) &&
        std::equal(b.normals.begin(), b.normals.end(), m.normals.begin() + l.vertBegin))
      continue;
    LeafState a;
    a.leaf = b.leaf;
    a.positions.assign(m.positions.begin() + l.vertBegin, m.positions.begin() + l.vertEnd);
    a.normals.assign(m.normals.begin() + l.vertBegin, m.normals.begin() + l.vertEnd);
    undo_.after.push_back(std::move(a));
    before.push_back(std::move(b));
  }
  if (before.empty()) return std::nullopt;
  undo_.before = std::move(before);
  return std::move(undo_);
}

}  // namespace plegl
