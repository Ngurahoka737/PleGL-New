#include "sculpt/Sculptor.h"

#include <algorithm>
#include <cmath>
#include <iterator>

#include "core/Parallel.h"
#include "core/Timer.h"

namespace plegl {

namespace {
// An auto-mask set no face has (ids stop at kMaxFaceSetId): the stroke finds nothing to change
// on that side.
constexpr std::int32_t kNoFaceSet = INT32_MAX;
}  // namespace

void Sculptor::beginStroke(SceneObject& object, const Brush& brush, const StrokeOptions& options, std::string label) {
  start(object, options, std::move(label));
  brush_ = &brush;
  maskStroke_ = brush.editsMask();
  faceSetStroke_ = brush.editsFaceSets();
  // Allocate before the first snapshot, so the before-state holds real (default) values.
  if (maskStroke_) object.mesh.ensureMask();
  if (faceSetStroke_) object.mesh.ensureFaceSets();
  // Subdivision levels keep their topology for life, so dynamic topology never runs on them.
  if (options.dyntopo && !object.multires && !maskStroke_ && !faceSetStroke_ && object.mesh.faceCount() > 0) {
    dyntopo_ = std::make_unique<DyntopoSession>(object, options.dyntopoOptions);
    mergedClaims_ = 0;
  }
}

void Sculptor::start(SceneObject& object, const StrokeOptions& options, std::string label) {
  if (object_) endStroke();  // A stroke left open must not leave its mesh mid-edit.
  object_ = &object;
  brush_ = nullptr;
  maskStroke_ = false;
  faceSetStroke_ = false;
  hidden_ = object.mesh.anyHidden();
  faceSetsResolved_ = false;
  onlySet_[0] = onlySet_[1] = 0;
  paintSet_[0] = paintSet_[1] = 0;
  grabVerts_.clear();
  options_ = options;
  undo_ = SculptUndo{};
  undo_.label = std::move(label);
  undo_.objectId = object.id;
  undo_.topologyVersion = object.topologyVersion;
  snapshotIndex_.clear();
  strokeRefit_.clear();
  if (vertexStamp_.size() != object.mesh.positions.size()) {
    vertexStamp_.assign(object.mesh.positions.size(), 0);
    stamp_ = 0;
  }
  dabCount_ = 0;
}

void Sculptor::snapshot(Index leaf) {
  if (snapshotIndex_.count(leaf)) return;
  // Leaves the topology pass changed are recorded whole, and tail leaves hold only new elements.
  if (dyntopo_ && (dyntopo_->recorder().claimed(leaf) || object_->bvh.isTail(leaf))) return;
  const BvhLeaf& l = object_->bvh.leaves()[leaf];
  const Mesh& m = object_->mesh;
  LeafState s;
  s.leaf = leaf;
  if (maskStroke_) {
    s.mask.assign(m.mask.begin() + l.vertBegin, m.mask.begin() + l.vertEnd);
  } else if (faceSetStroke_) {
    s.faceSets.assign(m.faceSets.begin() + l.faceBegin, m.faceSets.begin() + l.faceEnd);
  } else {
    s.positions.assign(m.positions.begin() + l.vertBegin, m.positions.begin() + l.vertEnd);
    s.normals.assign(m.normals.begin() + l.vertBegin, m.normals.begin() + l.vertEnd);
  }
  snapshotIndex_.emplace(leaf, undo_.before.size());
  undo_.before.push_back(std::move(s));
}

bool Sculptor::computeArea(Dab& dab, std::span<const Index> leaves, const FaceSetFilter& filter) const {
  const Mesh& m = object_->mesh;
  const float r2 = dab.radius * dab.radius;
  Vec3 normalSum{0.0f}, centerSum{0.0f};
  float weightSum = 0.0f;
  for (Index li : leaves) {
    const BvhLeaf& leaf = object_->bvh.leaves()[li];
    for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
      const Vec3 d = m.positions[v] - dab.center;
      const float dist2 = glm::dot(d, d);
      if (dist2 >= r2 || !filter.allows(m, v)) continue;
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

bool Sculptor::dab(const Vec3& center, float radius, float strength, const DabTopology& topology) {
  if (!object_ || !brush_ || radius <= 0.0f) return false;
  Timer total;
  lastDab_ = {};
  Dab d;
  d.center = center;
  d.radius = radius;
  d.strength = std::clamp(strength, 0.0f, 1.0f);
  d.falloff = options_.falloff;
  d.invert = options_.invert;
  if (!faceSetsResolved_) resolveFaceSets(center, radius);
  bool any = applyOne(d, topology, 0);
  if (options_.symmetryX) {
    Dab mirrored = d;
    mirrored.center.x = -mirrored.center.x;
    // Skip the mirror when the dab sits on the plane, or the seam would get double strength.
    if (std::abs(center.x) > radius * 0.05f) {
      DabTopology mirroredTopology = topology;
      if (dyntopo_ && topology.detail > 0.0f) {
        Bvh::ClosestHit hit;
        mirroredTopology.hintFace =
            object_->bvh.closestPoint(object_->mesh, mirrored.center, radius, hit) ? hit.face : kInvalid;
      }
      any |= applyOne(mirrored, mirroredTopology, 1);
    }
  }
  lastDab_.totalMs = total.ms();
  if (any) ++dabCount_;
  return any;
}

void Sculptor::resolveFaceSets(const Vec3& center, float radius) {
  faceSetsResolved_ = true;
  const Mesh& m = object_->mesh;
  const bool wantSet = options_.faceSetAutoMask || (faceSetStroke_ && options_.extendFaceSet);
  std::int32_t under[2] = {0, 0};  // Set of the visible face under each side's first dab, 0 if none.
  if (wantSet && !m.faceSets.empty()) {
    for (int side = 0; side < 2; ++side) {
      const Vec3 c = side == 0 ? center : Vec3{-center.x, center.y, center.z};
      Bvh::ClosestHit hit;
      if (object_->bvh.closestPoint(m, c, radius, hit, true)) under[side] = faceSetId(m.faceSets[hit.face]);
    }
  } else if (wantSet) {
    under[0] = under[1] = kDefaultFaceSet;  // Every face is in the default set.
  }
  // 0 (paint nothing) once ids run out.
  const std::int32_t fresh = faceSetStroke_ && !options_.extendFaceSet ? object_->newFaceSetId() : 0;
  for (int side = 0; side < 2; ++side) {
    if (options_.faceSetAutoMask) onlySet_[side] = under[side] != 0 ? under[side] : kNoFaceSet;
    if (faceSetStroke_) paintSet_[side] = options_.extendFaceSet ? under[side] : fresh;  // One new set for both sides.
  }
}

FaceSetFilter Sculptor::filter(int side) const {
  FaceSetFilter f;
  const Mesh& m = object_->mesh;
  if (m.faceSets.empty() || (!hidden_ && onlySet_[side] == 0 && !options_.lockFaceSetBoundaries)) return f;
  f.faceSets = m.faceSets.data();
  f.skipHidden = hidden_;
  f.onlySet = onlySet_[side];
  f.lockBoundaries = options_.lockFaceSetBoundaries;
  return f;
}

void Sculptor::mergeClaims() {
  // A leaf claimed after it already had a position snapshot: the snapshot holds the values from
  // before the stroke, the claim only those from before the topology edit.
  ClaimRecorder& rec = dyntopo_->recorder();
  const auto claimed = rec.claimedLeaves();
  for (; mergedClaims_ < claimed.size(); ++mergedClaims_) {
    const Index leaf = claimed[mergedClaims_];
    const auto it = snapshotIndex_.find(leaf);
    if (it == snapshotIndex_.end()) continue;
    LeafState& state = undo_.before[it->second];
    TopoLeafState* claim = rec.claim(leaf);
    if (!state.positions.empty()) claim->positions = std::move(state.positions);
    if (!state.normals.empty()) claim->normals = std::move(state.normals);
    state = LeafState{};  // leaf = kInvalid: merged. The index entry stays so it is not taken again.
  }
}

void Sculptor::applyTopology(const Dab& dab, const DabTopology& topology) {
  Timer t;
  Mesh& m = object_->mesh;
  Bvh& bvh = object_->bvh;
  const DyntopoPass& pass = dyntopo_->pass(dab.center, dab.radius, topology);
  if (pass.changed()) {
    mergeClaims();
    if (vertexStamp_.size() < m.positions.size()) vertexStamp_.resize(m.positions.size(), 0);
    // Vertices around the edits get new normals; snapshot their owners first.
    dirtyLeaves_.clear();
    Index lastOwner = kInvalid;
    for (Index v : pass.changedVerts) {
      if (lastOwner != kInvalid) {
        const BvhLeaf& l = bvh.leaves()[lastOwner];
        if (v >= l.vertBegin && v < l.vertEnd) continue;
      }
      lastOwner = bvh.leafOfVertex(v);
      if (lastOwner == kInvalid) continue;
      snapshot(lastOwner);
      dirtyLeaves_.push_back(lastOwner);
    }
    recomputeNormals(pass.changedVerts);
    bvh.refitLeaves(m, pass.refitLeaves);
    for (Index l : pass.topoDirtyLeaves) object_->markTopologyDirty(l);
    std::sort(dirtyLeaves_.begin(), dirtyLeaves_.end());
    dirtyLeaves_.erase(std::unique(dirtyLeaves_.begin(), dirtyLeaves_.end()), dirtyLeaves_.end());
    for (Index l : dirtyLeaves_) object_->markLeafDirty(l);
  }
  lastDab_.splits += pass.splits;
  lastDab_.collapses += pass.collapses;
  lastDab_.topologyMs += t.ms();
}

bool Sculptor::applyOne(const Dab& dabIn, const DabTopology& topology, int side) {
  Mesh& m = object_->mesh;
  Bvh& bvh = object_->bvh;
  Dab dab = dabIn;
  if (faceSetStroke_ && paintSet_[side] == 0) return false;  // No set to continue on this side.
  // The topology pass honours the face set limits too; with no set under this side's first dab
  // there is nothing it may change.
  if (dyntopo_ && topology.detail > 0.0f && onlySet_[side] != kNoFaceSet) {
    DabTopology limited = topology;
    limited.onlySet = onlySet_[side];
    limited.lockFaceSetBorders = options_.lockFaceSetBoundaries;
    applyTopology(dab, limited);
  }

  leaves_.clear();
  bvh.querySphere(dab.center, dab.radius, leaves_);
  if (leaves_.empty()) return false;
  const FaceSetFilter faceSetFilter = filter(side);  // After the topology pass: it may grow the array.
  if (brush_->needsArea() && !computeArea(dab, leaves_, faceSetFilter)) return false;  // No vertex inside the dab.
  for (Index l : leaves_) snapshot(l);

  Timer t;
  BrushContext ctx{m, bvh, leaves_, dab, faceSetFilter, paintSet_[side]};
  brush_->apply(ctx);
  lastDab_.brushMs += t.ms();

  if (maskStroke_ || faceSetStroke_) {  // Positions did not move: no normals, bounds or geometry upload.
    for (Index l : leaves_) {
      if (maskStroke_) {
        object_->markMaskDirty(l);
      } else {
        object_->markFaceSetDirty(l);
      }
    }
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
  const bool mayHaveDead = bvh.dynamic();
  for (Index li : leaves_) {
    const BvhLeaf& leaf = bvh.leaves()[li];
    for (Index f = leaf.faceBegin; f < leaf.faceEnd; ++f) {
      if (mayHaveDead && m.faceHe[f] == kInvalid) continue;
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
  strokeRefit_.insert(strokeRefit_.end(), leaves_.begin(), leaves_.end());
  if (strokeRefit_.size() > 2 * bvh.leaves().size() + 4096) {  // Long strokes revisit leaves.
    std::sort(strokeRefit_.begin(), strokeRefit_.end());
    strokeRefit_.erase(std::unique(strokeRefit_.begin(), strokeRefit_.end()), strokeRefit_.end());
  }
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
      m.normals[v] = m.vertexNormal(v);
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
  resolveFaceSets(center, radius);

  // Collect weights per vertex; a vertex can sit in both spheres when they overlap at the seam.
  std::unordered_map<Index, std::size_t> slot;
  auto capture = [&](const Vec3& c, bool mirror) {
    leaves_.clear();
    bvh.querySphere(c, radius, leaves_);
    const float r2 = radius * radius;
    const FaceSetFilter faceSetFilter = filter(mirror ? 1 : 0);
    for (Index li : leaves_) {
      const BvhLeaf& leaf = bvh.leaves()[li];
      for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
        const Vec3 d = m.positions[v] - c;
        const float dist2 = glm::dot(d, d);
        if (dist2 >= r2) continue;
        float w = strength * falloffWeight(options.falloff, std::sqrt(dist2) / radius);
        if (!m.mask.empty()) w *= 1.0f - m.mask[v];  // Fully masked vertices stay put.
        if (w <= 0.0f || !faceSetFilter.allows(m, v)) continue;
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
  strokeRefit_ = grabRefitLeaves_;
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

std::optional<StrokeUndo> Sculptor::endDyntopoStroke(SceneObject& obj, DyntopoSession& session) {
  Timer t;
  Mesh& m = obj.mesh;
  lastStroke_.dyntopo = true;
  lastStroke_.splits = session.totalSplits();
  lastStroke_.collapses = session.totalCollapses();
  lastStroke_.facesBefore = obj.bvh.tailStartFace();
  lastStroke_.faulted = session.faulted();
  lastStroke_.outOfRoom = session.outOfRoom();
  // Position snapshots of untouched leaves that really changed; old leaves keep their ranges
  // until the consolidation, so they are compared in place.
  std::vector<LeafState> moved;
  for (LeafState& b : undo_.before) {
    if (b.leaf == kInvalid) continue;  // Merged into a claim.
    const BvhLeaf& l = obj.bvh.leaves()[b.leaf];
    if (std::equal(b.positions.begin(), b.positions.end(), m.positions.begin() + l.vertBegin) &&
        std::equal(b.normals.begin(), b.normals.end(), m.normals.begin() + l.vertBegin))
      continue;
    moved.push_back(std::move(b));
  }
  ClaimRecorder& rec = session.recorder();
  const std::vector<Index> claimed(rec.claimedLeaves().begin(), rec.claimedLeaves().end());
  std::vector<TopoLeafState> claims = rec.takeClaims();
  if (!workspace_) workspace_ = std::make_shared<LayoutWorkspace>();
  const ConsolidateResult result = consolidate(m, obj.bvh, claimed, *workspace_);
  const std::uint64_t before = obj.topologyVersion;
  obj.topologyVersion = nextTopologyVersion();
  // Dynamic topology never creates non-manifold vertices.
  if (obj.manifoldCheckedVersion == before) obj.manifoldCheckedVersion = obj.topologyVersion;
  obj.clearDirty();
  lastStroke_.facesAfter = m.faceCount();
  lastStroke_.consolidateMs = t.ms();
  if (result.missedClaim) {
    lastStroke_.lostUndo = true;
    return std::nullopt;
  }
  DyntopoUndo undo;
  undo.label = undo_.label;
  undo.objectId = obj.id;
  undo.afterVersion = obj.topologyVersion;
  undo.delta = result.delta;
  undo.before = beforeSide(session.beforeBvh(), std::move(claims), std::move(moved), result, before);
  return StrokeUndo{std::move(undo)};
}

std::optional<StrokeUndo> Sculptor::endStroke() {
  if (!object_) return std::nullopt;
  SceneObject* obj = object_;
  object_ = nullptr;
  lastStroke_ = {};
  if (std::unique_ptr<DyntopoSession> session = std::move(dyntopo_)) {
    lastStroke_.lockedVertices = session->lockedVertices();
    if (session->changedTopology()) return endDyntopoStroke(*obj, *session);
    obj->bvh.endDynamic();
  }
  if (undo_.before.empty()) return std::nullopt;
  const Mesh& m = obj->mesh;
  if (faceSetStroke_) {
    // Drop leaves the brush passed over without repainting a face.
    std::vector<LeafState> before;
    for (LeafState& b : undo_.before) {
      const BvhLeaf& l = obj->bvh.leaves()[b.leaf];
      if (std::equal(b.faceSets.begin(), b.faceSets.end(), m.faceSets.begin() + l.faceBegin)) continue;
      LeafState a;
      a.leaf = b.leaf;
      a.faceSets.assign(m.faceSets.begin() + l.faceBegin, m.faceSets.begin() + l.faceEnd);
      undo_.after.push_back(std::move(a));
      before.push_back(std::move(b));
    }
    if (before.empty()) return std::nullopt;
    undo_.before = std::move(before);
    return StrokeUndo{std::move(undo_)};
  }
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
    return StrokeUndo{std::move(undo_)};
  }
  // Drop leaves the stroke did not change, for example because they are fully masked. A leaf
  // whose positions stayed but whose normals changed (a neighbour moved) is kept.
  std::vector<LeafState> before;
  before.reserve(undo_.before.size());
  undo_.after.reserve(undo_.before.size());
  kept_.clear();
  for (LeafState& b : undo_.before) {
    const BvhLeaf& l = obj->bvh.leaves()[b.leaf];
    if (std::equal(b.positions.begin(), b.positions.end(), m.positions.begin() + l.vertBegin) &&
        std::equal(b.normals.begin(), b.normals.end(), m.normals.begin() + l.vertBegin))
      continue;
    kept_.push_back(b.leaf);
    LeafState a;
    a.leaf = b.leaf;
    a.positions.assign(m.positions.begin() + l.vertBegin, m.positions.begin() + l.vertEnd);
    a.normals.assign(m.normals.begin() + l.vertBegin, m.normals.begin() + l.vertEnd);
    undo_.after.push_back(std::move(a));
    before.push_back(std::move(b));
  }
  if (before.empty()) return std::nullopt;
  undo_.before = std::move(before);
  // Leaves refit without a snapshot that survived: their bounds must follow the vertices of the
  // others back and forth, or a raycast may miss them after undo.
  std::sort(strokeRefit_.begin(), strokeRefit_.end());
  strokeRefit_.erase(std::unique(strokeRefit_.begin(), strokeRefit_.end()), strokeRefit_.end());
  std::sort(kept_.begin(), kept_.end());
  std::set_difference(strokeRefit_.begin(), strokeRefit_.end(), kept_.begin(), kept_.end(),
                      std::back_inserter(undo_.refit));
  return StrokeUndo{std::move(undo_)};
}

}  // namespace plegl
