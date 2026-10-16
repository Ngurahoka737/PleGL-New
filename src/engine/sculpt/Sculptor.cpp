#include "sculpt/Sculptor.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iterator>
#include <span>
#include <unordered_map>

#include "core/Parallel.h"
#include "core/Timer.h"
#include "sculpt/LayerOps.h"

namespace plegl {

namespace {
// An auto-mask set no face has (ids stop at kMaxFaceSetId): the stroke finds nothing to change
// on that side.
constexpr std::int32_t kNoFaceSet = INT32_MAX;
}  // namespace

void Sculptor::beginStroke(SceneObject& object, const Brush& requested, const StrokeOptions& options,
                           std::string label) {
  const Brush& brush = strokeBrush(requested, object, options);
  if (!layerTargetRefusal(object, &brush, options.layerTarget).empty()) {
    if (object_) endStroke();  // As start() would: no stroke stays open.
    return;
  }
  start(object, options, std::move(label));
  brush_ = &brush;
  maskStroke_ = brush.editsMask();
  faceSetStroke_ = brush.editsFaceSets();
  // Allocate before the first snapshot, so the before-state holds real (default) values.
  if (maskStroke_) object.mesh.ensureMask();
  if (faceSetStroke_) object.mesh.ensureFaceSets();
  beginLayers();
  // Subdivision levels keep their topology for life, so dynamic topology never runs on them; nor
  // does it on meshes with sculpt layers.
  // Nor above kMaxDyntopoCopies symmetric copies: every copy would run its own pass.
  if (options.dyntopo && !object.multires && object.mesh.layers.empty() && !maskStroke_ && !faceSetStroke_ &&
      object.mesh.faceCount() > 0) {
    if (group_.size() > kMaxDyntopoCopies) {
      tooManyCopies_ = true;
    } else {
      DyntopoOptions topo = options.dyntopoOptions;
      if (group_.size() > 1) {
        topo.pins = {group_.mirrorPlanes(), group_.rotationAxes(), pinEps_};
        topo.copies = group_.size();
      }
      dyntopo_ = std::make_unique<DyntopoSession>(object, topo);
      mergedClaims_ = 0;
    }
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
  onlySet_.fill(0);
  paintSet_.fill(0);
  grabVerts_.clear();
  grabEntries_.clear();
  grabPins_.clear();
  snakeStroke_ = false;
  snakeClusterCount_ = 0;
  poseStroke_ = false;
  pose_ = PoseBuild{};
  options_ = options;
  tooManyCopies_ = false;
  if (const SymmetrySpec spec = options.symmetry(); !(spec == group_.spec())) group_ = SymmetryGroup(spec);
  {
    const float diagonal = glm::length(object.bvh.bounds().extent());
    pinEps_ = std::isfinite(diagonal) ? kPinTolerance * diagonal : 0.0f;
  }
  layerArray_ = nullptr;
  layerTarget_ = {};
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

void Sculptor::beginLayers() {
  LayerStack& stack = object_->mesh.layers;
  if (stack.empty() || maskStroke_ || faceSetStroke_) return;
  // The refusal checked the target exists.
  layerArray_ = stack.array(options_.layerTarget);
  layerTarget_ = {&stack, layerArray_, targetStrength(stack, options_.layerTarget)};
  undo_.layerTarget = options_.layerTarget;
  undo_.layerKey = layerStateKey(stack);
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
    if (layerArray_) s.layer.assign(layerArray_->begin() + l.vertBegin, layerArray_->begin() + l.vertEnd);
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
  const bool any = applyCopies(prepareCopies(d, topology), {});
  lastDab_.totalMs = total.ms();
  if (any) ++dabCount_;
  return any;
}

int Sculptor::prepareCopies(const Dab& d, const DabTopology& topology) {
  const int n = clusterCopies(group_, d.center, kCopyMergeFraction * d.radius, clusters_);
  if (copies_.size() < static_cast<std::size_t>(n)) copies_.resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    CopyDab& c = copies_[static_cast<std::size_t>(i)];
    c.cluster = clusters_[static_cast<std::size_t>(i)];
    c.dab = d;
    c.dab.center = c.cluster.center;
    c.topology = topology;
    // The first copy has the face under the cursor; the others look for theirs.
    if (i > 0 && dyntopo_ && topology.detail > 0.0f) {
      Bvh::ClosestHit hit;
      c.topology.hintFace =
          object_->bvh.closestPoint(object_->mesh, c.dab.center, d.radius, hit) ? hit.face : kInvalid;
    }
    c.live = false;
  }
  lastDab_.copies = group_.size();
  lastDab_.clusters = n;
  return n;
}

OpMask Sculptor::pinOps(const Vec3& center, float radius) const {
  // Ops whose fixed plane or axis comes within the reach of normal updates around this centre.
  OpMask ops = 0;
  const float reach = 1.25f * radius;
  for (int k = 1; k < group_.size(); ++k)
    if (glm::length(center - transform(group_.fixProjector(k), center)) < reach) ops |= OpMask{1} << k;
  return ops;
}

bool Sculptor::pinCandidate(const Vec3& p, OpMask ops) const {
  for (int k = 1; k < group_.size(); ++k)
    if ((ops >> k & 1) && glm::length(p - transform(group_.fixProjector(k), p)) <= pinEps_) return true;
  return false;
}

void Sculptor::collectPins(const CopyCluster& cluster, const Vec3& center, float radius,
                           std::span<const Index> leaves) {
  (void)cluster;
  const OpMask ops = pinOps(center, radius);
  if (!ops) return;
  const Mesh& m = object_->mesh;
  const float r2 = radius * radius;
  for (Index li : leaves) {
    const BvhLeaf& leaf = object_->bvh.leaves()[li];
    for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
      const Vec3& p = m.positions[v];
      const Vec3 d = p - center;
      if (glm::dot(d, d) >= r2 || !pinCandidate(p, ops)) continue;
      pins_.push_back({v, p, seamPin(group_, p, pinEps_)});
    }
  }
}

const ShareView* Sculptor::buildZone(int count, ShareView& view) {
  // Copies that some other copy comes close to: a vertex within 1.2 r of two centres can be read
  // by one copy after another moved it (moves are at most 0.2 r). Brushes that read neighbours
  // look one edge further.
  const bool readers = brush_->readsNeighbours();
  const float r = copies_[0].dab.radius;
  const float pairDist = (readers ? 2.5f : 2.4f) * r;
  bool overlapping[kMaxSymmetryCopies] = {};
  bool any = false;
  for (int i = 0; i < count; ++i) {
    const CopyDab& a = copies_[static_cast<std::size_t>(i)];
    if (!a.live) continue;
    for (int j = i + 1; j < count; ++j) {
      const CopyDab& b = copies_[static_cast<std::size_t>(j)];
      if (!b.live || glm::length(a.dab.center - b.dab.center) >= pairDist) continue;
      overlapping[i] = overlapping[j] = true;
      any = true;
    }
  }
  if (!any) return nullptr;

  Timer t;
  const Mesh& m = object_->mesh;
  ShareZone& z = zone_;
  if (z.stamp.size() < m.positions.size()) {
    z.stamp.resize(m.positions.size(), 0);
    z.slot.resize(m.positions.size(), 0);
  }
  if (++z.current == 0) {
    std::fill(z.stamp.begin(), z.stamp.end(), 0);
    z.current = 1;
  }
  z.denom.clear();
  z.count.clear();
  z.verts.clear();
  const float reach = r * (1.0f + kMaxDabMove);
  const float reach2 = reach * reach;
  const float r2 = r * r;
  const float invR = 1.0f / r;
  // Copies in order, leaves in query order and a serial sum: the shares do not depend on the
  // number of threads.
  for (int i = 0; i < count; ++i) {
    const CopyDab& c = copies_[static_cast<std::size_t>(i)];
    if (!c.live || !overlapping[i]) continue;
    if (z.gathered.size() < c.leaves.size()) z.gathered.resize(c.leaves.size());
    const Vec3 center = c.dab.center;
    const Falloff falloff = c.dab.falloff;
    parallelFor(0, c.leaves.size(), 1, [&](std::size_t b, std::size_t e) {
      for (std::size_t k = b; k < e; ++k) {
        auto& out = z.gathered[k];
        out.clear();
        const BvhLeaf& leaf = object_->bvh.leaves()[c.leaves[k]];
        for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
          const Vec3 delta = m.positions[v] - center;
          const float dist2 = glm::dot(delta, delta);
          if (dist2 >= reach2) continue;
          // The same falloff the brush computes for this vertex.
          out.emplace_back(v, dist2 >= r2 ? 0.0f : falloffWeight(falloff, std::sqrt(dist2) * invR));
        }
      }
    });
    for (std::size_t k = 0; k < c.leaves.size(); ++k) {
      for (const auto& [v, f] : z.gathered[k]) {
        const float fp = kShareExponent == 1.0f ? f : std::pow(f, kShareExponent);
        if (z.stamp[v] != z.current) {
          z.stamp[v] = z.current;
          z.slot[v] = static_cast<std::uint32_t>(z.verts.size());
          z.verts.push_back(v);
          z.denom.push_back(fp);
          z.count.push_back(1);
        } else {
          const std::uint32_t s = z.slot[v];
          z.denom[s] += fp;
          if (z.count[s] < 255) ++z.count[s];
        }
      }
    }
  }
  z.pending.assign(z.verts.size(), Vec3{0.0f});
  if (maskStroke_) z.pendingScalar.assign(z.verts.size(), 0.0f);
  view.stamp = z.stamp.data();
  view.current = z.current;
  view.slot = z.slot.data();
  view.denom = z.denom.data();
  view.count = z.count.data();
  view.pending = z.pending.data();
  view.pendingScalar = maskStroke_ ? z.pendingScalar.data() : nullptr;
  view.deferAll = readers;
  lastDab_.shareMs += t.ms();
  return &view;
}

void Sculptor::resolveFaceSets(const Vec3& center, float radius) {
  faceSetsResolved_ = true;
  const Mesh& m = object_->mesh;
  const bool wantSet = options_.faceSetAutoMask || (faceSetStroke_ && options_.extendFaceSet);
  // Set of the visible face under each op's first dab, 0 if none.
  std::array<std::int32_t, kMaxSymmetryCopies> under{};
  const int ops = group_.size();
  if (wantSet && !m.faceSets.empty()) {
    for (int k = 0; k < ops; ++k) {
      const Vec3 c = k == 0 ? center : transform(group_.op(k).L, center);
      Bvh::ClosestHit hit;
      if (object_->bvh.closestPoint(m, c, radius, hit, true))
        under[static_cast<std::size_t>(k)] = faceSetId(m.faceSets[hit.face]);
    }
  } else if (wantSet) {
    under.fill(kDefaultFaceSet);  // Every face is in the default set.
  }
  // 0 (paint nothing) once ids run out.
  const std::int32_t fresh = faceSetStroke_ && !options_.extendFaceSet ? object_->newFaceSetId() : 0;
  for (int k = 0; k < ops; ++k) {
    const std::size_t i = static_cast<std::size_t>(k);
    if (options_.faceSetAutoMask) onlySet_[i] = under[i] != 0 ? under[i] : kNoFaceSet;
    if (faceSetStroke_) paintSet_[i] = options_.extendFaceSet ? under[i] : fresh;  // One new set for every op.
  }
}

FaceSetFilter Sculptor::filter(int op) const {
  FaceSetFilter f;
  const Mesh& m = object_->mesh;
  const std::int32_t only = onlySet_[static_cast<std::size_t>(op)];
  if (m.faceSets.empty() || (!hidden_ && only == 0 && !options_.lockFaceSetBoundaries)) return f;
  f.faceSets = m.faceSets.data();
  f.skipHidden = hidden_;
  f.onlySet = only;
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

double Sculptor::topologyBudget(int passes) const {
  // Symmetric dabs spend at most twice a single pass, split evenly; -1 leaves the options' budget.
  const double budget = options_.dyntopoOptions.timeBudgetMs;
  return budget > 0.0 && passes > 0 ? std::max(0.25, std::min(2.0 * budget, budget * passes) / passes) : -1.0;
}

bool Sculptor::applyCopies(int count, const ApplyFlags& flags) {
  Mesh& m = object_->mesh;
  Bvh& bvh = object_->bvh;
  const auto copy = [&](int i) -> CopyDab& { return copies_[static_cast<std::size_t>(i)]; };
  // The topology pass honours the face set limits too; with no set under an op's first dab there
  // is nothing it may change, and a face set stroke with nothing to continue skips the op.
  const auto skipped = [&](const CopyDab& c) { return faceSetStroke_ && paintSet_[c.cluster.op] == 0; };

  // Phase 0: topology, copy by copy, each pass seeing the edits of the passes before it. The time
  // budget is split so symmetric strokes spend at most twice a single pass.
  if (dyntopo_) {
    int passes = 0;
    for (int i = 0; i < count; ++i)
      passes += !skipped(copy(i)) && copy(i).topology.detail > 0.0f && onlySet_[copy(i).cluster.op] != kNoFaceSet;
    const double perPass = topologyBudget(passes);
    for (int i = 0; i < count; ++i) {
      CopyDab& c = copy(i);
      const std::int32_t only = onlySet_[c.cluster.op];
      if (skipped(c) || c.topology.detail <= 0.0f || only == kNoFaceSet) continue;
      DabTopology limited = c.topology;
      limited.onlySet = only;
      limited.lockFaceSetBorders = options_.lockFaceSetBoundaries;
      limited.timeBudgetMs = perPass;
      applyTopology(c.dab, limited);
    }
  }

  // Phase A: every copy queries, measures its area and snapshots before any copy writes.
  const bool positions = !maskStroke_ && !faceSetStroke_;
  const bool pinning = positions && group_.size() > 1 && !brush_->writesLayer();
  pins_.clear();
  int live = 0;
  for (int i = 0; i < count; ++i) {
    CopyDab& c = copy(i);
    c.live = false;
    if (skipped(c)) continue;
    c.leaves.clear();
    bvh.querySphere(c.dab.center, c.dab.radius, c.leaves);
    if (c.leaves.empty()) continue;
    c.filter = filter(c.cluster.op);  // After the topology pass: it may grow the array.
    c.paintSet = paintSet_[c.cluster.op];
    if (brush_->needsArea()) {
      if (!computeArea(c.dab, c.leaves, c.filter)) continue;  // No vertex inside the dab.
      if (c.cluster.merged()) {
        // Merged copies agree only on part of the area: keep the normal and centre on the plane or
        // axis they share.
        const Vec3 n = c.cluster.projectLocal(c.dab.areaNormal);
        if (glm::length(n) >= 0.1f) c.dab.areaNormal = glm::normalize(n);
        c.dab.areaCenter = c.cluster.projectLocal(c.dab.areaCenter);
      }
    }
    for (Index l : c.leaves) snapshot(l);
    if (pinning) collectPins(c.cluster, c.dab.center, c.dab.radius, c.leaves);
    c.live = true;
    ++live;
  }
  if (live == 0) return false;
  if (pins_.size() > 1) {
    std::sort(pins_.begin(), pins_.end(), [](const PinnedVertex& a, const PinnedVertex& b) { return a.v < b.v; });
    pins_.erase(std::unique(pins_.begin(), pins_.end(),
                            [](const PinnedVertex& a, const PinnedVertex& b) { return a.v == b.v; }),
                pins_.end());
  }

  // Phase Z: shares where copies overlap.
  ShareView view;
  const ShareView* shares = live >= 2 && !faceSetStroke_ ? buildZone(count, view) : nullptr;
  std::uint32_t* claims = nullptr;
  if (faceSetStroke_ && live >= 2) {
    if (faceClaim_.size() < static_cast<std::size_t>(m.faceCount())) faceClaim_.resize(static_cast<std::size_t>(m.faceCount()), 0);
    if (++claimStamp_ == 0) {
      std::fill(faceClaim_.begin(), faceClaim_.end(), 0);
      claimStamp_ = 1;
    }
    claims = faceClaim_.data();
  }

  // Phase B: the brush, copy by copy.
  Timer t;
  LayerTarget* layer = brush_->writesLayer() && layerArray_ ? &layerTarget_ : nullptr;
  for (int i = 0; i < count; ++i) {
    CopyDab& c = copy(i);
    if (!c.live) continue;
    BrushContext ctx{m, bvh, c.leaves, c.dab, c.filter, c.paintSet, layer, shares, claims, claimStamp_};
    brush_->apply(ctx);
  }

  // Phase B2: moves that waited for every copy.
  if (shares) {
    const ShareZone& z = zone_;
    parallelFor(0, z.verts.size(), 4096, [&](std::size_t b, std::size_t e) {
      for (std::size_t s = b; s < e; ++s) {
        if (!view.deferred(static_cast<std::int64_t>(s))) continue;
        const Index v = z.verts[s];
        if (maskStroke_) {
          if (z.pendingScalar[s] != 0.0f) m.mask[v] = std::clamp(m.mask[v] + z.pendingScalar[s], 0.0f, 1.0f);
          continue;
        }
        const Vec3& move = z.pending[s];
        if (isZero(move)) continue;
        if (layer) {
          Vec3& o = (*layer->offset)[v];
          o.x = o.x + move.x;
          o.y = o.y + move.y;
          o.z = o.z + move.z;
          m.positions[v] = composeVertex(*layer->stack, v);
        } else {
          m.positions[v] += move;
        }
      }
    });
  }

  // Phase P: vertices that started on a symmetry plane or axis and moved go back onto it; a
  // vertex the dab left alone (masked, filtered, outside the falloff) stays where it is.
  for (const PinnedVertex& pin : pins_)
    if (!sameBits(m.positions[pin.v], pin.start)) m.positions[pin.v] = snapPinned(pin.pin, pin.start, m.positions[pin.v]);
  lastDab_.pinned = static_cast<int>(pins_.size());
  lastDab_.brushMs += t.ms();

  if (maskStroke_ || faceSetStroke_) {  // Positions did not move: no normals, bounds or geometry upload.
    for (int i = 0; i < count; ++i) {
      const CopyDab& c = copy(i);
      if (!c.live) continue;
      for (Index l : c.leaves) {
        if (maskStroke_) {
          object_->markMaskDirty(l);
        } else {
          object_->markFaceSetDirty(l);
        }
      }
      lastDab_.leaves += static_cast<int>(c.leaves.size());
    }
    return true;
  }

  // Phase C, once for every copy: brushes only move vertices inside their dab sphere. Every face
  // touching such a vertex lies in a queried leaf (its bounds contain the vertex), so the vertices
  // of those faces are exactly the ones whose normals can have changed. Positions are read after
  // the brush ran, so the sphere test uses a slightly larger radius to cover vertices that moved
  // outward across it.
  t.reset();
  if (++stamp_ == 0) {
    std::fill(vertexStamp_.begin(), vertexStamp_.end(), 0);
    stamp_ = 1;
  }
  normalVerts_.clear();
  leaves_.clear();
  const bool mayHaveDead = bvh.dynamic();
  for (int i = 0; i < count; ++i) {
    const CopyDab& c = copy(i);
    if (!c.live) continue;
    const float reach = c.dab.radius * 1.25f;
    const float reach2 = reach * reach;
    const Vec3 center = c.dab.center;
    leaves_.insert(leaves_.end(), c.leaves.begin(), c.leaves.end());
    for (Index li : c.leaves) {
      const BvhLeaf& leaf = bvh.leaves()[li];
      for (Index f = leaf.faceBegin; f < leaf.faceEnd; ++f) {
        if (mayHaveDead && m.faceHe[f] == kInvalid) continue;
        bool touched = false;
        m.forEachFaceVertex(f, [&](Index v) {
          const Vec3 d = m.positions[v] - center;
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
  }
  if (live > 1) {
    std::sort(leaves_.begin(), leaves_.end());
    leaves_.erase(std::unique(leaves_.begin(), leaves_.end()), leaves_.end());
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
  if (flags.deferNormals) {
    pendingNormals_.insert(pendingNormals_.end(), normalVerts_.begin(), normalVerts_.end());
  } else {
    recomputeNormals(normalVerts_);
  }
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
  if (!normals_) normals_ = std::make_shared<NormalWorkspace>();
  // Dynamic topology leaves dead faces until the stroke ends, so only whole-mesh passes are off.
  recomputeVertexNormals(object_->mesh, verts, *normals_, !object_->bvh.dynamic());
}

bool Sculptor::beginGrab(SceneObject& object, const StrokeOptions& options, const Vec3& center, float radius,
                         std::string label) {
  if (radius <= 0.0f || !layerTargetRefusal(object, nullptr, options.layerTarget).empty()) return false;
  start(object, options, std::move(label));
  beginLayers();
  Mesh& m = object.mesh;
  const Bvh& bvh = object.bvh;
  const float strength = std::clamp(options.strength, 0.0f, 1.0f);
  resolveFaceSets(center, radius);

  // Every cluster of symmetric copies captures the vertices inside its sphere. A vertex inside
  // several spheres splits between them by falloff, like a dab, so overlapping copies do not
  // stack; on a plane the copies merge into one cluster whose pull stays in the plane.
  const int clusters = clusterCopies(group_, center, kCopyMergeFraction * radius, grabClusters_);
  grabClusterCount_ = clusters;
  struct Hit {
    Index v;
    std::uint32_t cluster;
    float weight;  // Strength, falloff and mask.
    float share;   // Raw falloff, then the share.
  };
  std::vector<Hit> hits;
  std::unordered_map<Index, float> shareSum;  // Vertex -> sum of f^p over clusters containing it.
  const float r2 = radius * radius;
  OpMask pinOpsAll = 0;
  for (int k = 0; k < clusters; ++k) {
    const CopyCluster& cl = grabClusters_[static_cast<std::size_t>(k)];
    const Vec3 c = cl.center;
    if (group_.size() > 1) pinOpsAll |= pinOps(c, radius);
    leaves_.clear();
    bvh.querySphere(c, radius, leaves_);
    const FaceSetFilter faceSetFilter = filter(cl.op);
    for (Index li : leaves_) {
      const BvhLeaf& leaf = bvh.leaves()[li];
      for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
        const Vec3 d = m.positions[v] - c;
        const float dist2 = glm::dot(d, d);
        if (dist2 >= r2) continue;
        const float f = falloffWeight(options.falloff, std::sqrt(dist2) / radius);
        if (clusters > 1) shareSum[v] += kShareExponent == 1.0f ? f : std::pow(f, kShareExponent);
        float w = strength * f;
        if (!m.mask.empty()) w *= 1.0f - m.mask[v];  // Fully masked vertices stay put.
        if (w <= 0.0f || !faceSetFilter.allows(m, v)) continue;
        hits.push_back({v, static_cast<std::uint32_t>(k), w, f});
      }
    }
  }
  // Entries grouped by vertex, vertices in the order they were first captured.
  std::unordered_map<Index, std::size_t> slot;
  std::vector<std::vector<GrabEntry>> entries;
  for (const Hit& hit : hits) {
    float weight = hit.weight;
    if (clusters > 1) {
      const float f = hit.share;
      const float denom = shareSum[hit.v];
      weight *= denom > 0.0f ? (kShareExponent == 1.0f ? f : std::pow(f, kShareExponent)) / denom : 0.0f;
    }
    auto [it, inserted] = slot.try_emplace(hit.v, grabVerts_.size());
    if (inserted) {
      grabVerts_.push_back({hit.v, m.positions[hit.v], 0, 0});
      entries.emplace_back();
    }
    entries[it->second].push_back({hit.cluster, weight});
  }
  for (std::size_t i = 0; i < grabVerts_.size(); ++i) {
    grabVerts_[i].firstEntry = static_cast<std::uint32_t>(grabEntries_.size());
    grabVerts_[i].entryCount = static_cast<std::uint32_t>(entries[i].size());
    grabEntries_.insert(grabEntries_.end(), entries[i].begin(), entries[i].end());
    if (pinOpsAll && pinCandidate(grabVerts_[i].start, pinOpsAll))
      grabPins_.emplace_back(static_cast<std::uint32_t>(i), seamPin(group_, grabVerts_[i].start, pinEps_));
  }
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
  // Every fan of a pinched vertex: faces in its other fans change shape too.
  const std::span<const std::pair<Index, Index>> fans = nonManifoldFans(object, fans_);
  std::vector<Index> faces;
  for (const GrabVertex& g : grabVerts_) forEachFaceAt(m, fans, g.v, [&](Index f) { faces.push_back(f); });
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
  // The offset as each cluster sees it: mirrored, turned, and on a plane projected into it.
  std::array<Vec3, kMaxSymmetryCopies> moves;
  for (int k = 0; k < grabClusterCount_; ++k)
    moves[static_cast<std::size_t>(k)] = transform(grabClusters_[static_cast<std::size_t>(k)].polar, offset);
  Timer t;
  parallelFor(0, grabVerts_.size(), 4096, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const GrabVertex& g = grabVerts_[i];
      Vec3 p = g.start;
      for (std::uint32_t k = 0; k < g.entryCount; ++k) {
        const GrabEntry& entry = grabEntries_[g.firstEntry + k];
        p += moves[entry.cluster] * entry.weight;
      }
      m.positions[g.v] = p;
    }
  });
  for (const auto& [i, pin] : grabPins_) {
    const GrabVertex& g = grabVerts_[i];
    if (!sameBits(m.positions[g.v], g.start)) m.positions[g.v] = snapPinned(pin, g.start, m.positions[g.v]);
  }
  lastDab_.brushMs = t.ms();
  lastDab_.copies = group_.size();
  lastDab_.clusters = grabClusterCount_;
  lastDab_.pinned = static_cast<int>(grabPins_.size());
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

float snakeGain(float strength, double dtMs) {
  const double s = std::clamp(static_cast<double>(strength), 0.0, 1.0);
  if (s >= 1.0) return 1.0f;
  const double dt = std::clamp(dtMs, 1.0, 50.0);
  return static_cast<float>(1.0 - std::pow(1.0 - s, dt / (1000.0 / 60.0)));
}

Index Sculptor::findHook(const CopyCluster& cluster, float radius) {
  const Mesh& m = object_->mesh;
  const FaceSetFilter allowed = filter(cluster.op);
  // Coordinates the merged copies agree are zero: a hook with those zero stays on the plane or
  // axis, where the pins keep it.
  std::uint8_t onSeam = 0;
  if (cluster.merged())
    for (int a = 0; a < 3; ++a)
      if (cluster.localProjector[a] == Vec3{0.0f}) onSeam |= static_cast<std::uint8_t>(1 << a);
  const float seamDist2 = kCopyMergeFraction * radius * kCopyMergeFraction * radius;
  Index best = kInvalid, seam = kInvalid;
  float bestDist2 = radius * radius, bestSeam2 = seamDist2;
  leaves_.clear();
  object_->bvh.querySphere(cluster.center, radius, leaves_);
  for (Index li : leaves_) {
    const BvhLeaf& leaf = object_->bvh.leaves()[li];
    for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
      const Vec3 d = m.positions[v] - cluster.center;
      const float dist2 = glm::dot(d, d);
      // Removed and isolated vertices have no face; hidden ones none that is visible.
      if ((dist2 >= bestDist2 && dist2 >= bestSeam2) || m.vertHe[v] == kInvalid || !allowed.allows(m, v)) continue;
      if (dist2 < bestDist2 || (dist2 == bestDist2 && v < best)) {
        bestDist2 = dist2;
        best = v;
      }
      if (onSeam && dist2 < bestSeam2 &&
          (pinnedAxes(group_.mirrorPlanes(), group_.rotationAxes(), m.positions[v], pinEps_) & onSeam) == onSeam) {
        bestSeam2 = dist2;
        seam = v;
      }
    }
  }
  return seam != kInvalid ? seam : best;
}

bool Sculptor::beginSnake(SceneObject& object, const Brush& snake, const StrokeOptions& options, const Vec3& center,
                          float radius, float pinch, std::string label) {
  if (!(radius > 0.0f)) return false;
  beginStroke(object, snake, options, std::move(label));
  if (!active()) return false;
  resolveFaceSets(center, radius);
  snakePinch_ = std::clamp(pinch, 0.0f, 1.0f);
  // The clusters stay as they are at the press for the whole stroke, so the copies keep their
  // roles however far the hook travels.
  snakeClusterCount_ = clusterCopies(group_, center, kCopyMergeFraction * radius, snakeClusters_);
  for (int k = 0; k < snakeClusterCount_; ++k)
    snakeHooks_[static_cast<std::size_t>(k)] = findHook(snakeClusters_[static_cast<std::size_t>(k)], radius);
  const Index hook = snakeHooks_[0];
  const Mesh& m = object.mesh;
  if (hook == kInvalid || (!m.mask.empty() && m.mask[hook] >= 0.999f)) {
    endStroke();
    return false;
  }
  snakeStroke_ = true;
  return true;
}

void Sculptor::reglueHooks(float radius) {
  // A topology pass keeps only its own hook; one that another copy's pass merged away moves to
  // the nearest live vertex that may move.
  const Mesh& m = object_->mesh;
  for (int k = 0; k < snakeClusterCount_; ++k) {
    Index& hook = snakeHooks_[static_cast<std::size_t>(k)];
    if (hook == kInvalid || m.vertHe[hook] != kInvalid) continue;
    CopyCluster around = snakeClusters_[static_cast<std::size_t>(k)];
    around.center = m.positions[hook];  // Removed vertices keep their last position.
    hook = findHook(around, radius);
  }
}

Mat3 Sculptor::snakeSteering(float radius) const {
  // How the primary hook moves for a primary pull of one: every copy reaching it adds its mirrored
  // pull, weighted by its falloff there and its share, all scaled by the mask.
  const Mesh& m = object_->mesh;
  const Index h0 = snakeHooks_[0];
  const Vec3 p0 = m.positions[h0];
  std::array<float, kMaxSymmetryCopies> f{};
  float sum = 0.0f;
  for (int k = 0; k < snakeClusterCount_; ++k) {
    const Index h = snakeHooks_[static_cast<std::size_t>(k)];
    if (h == kInvalid || (k > 0 && !filter(snakeClusters_[static_cast<std::size_t>(k)].op).allows(m, h0))) continue;
    const float fk = k == 0 ? 1.0f : falloffWeight(options_.falloff, glm::length(p0 - m.positions[h]) / radius);
    f[static_cast<std::size_t>(k)] = fk;
    sum += kShareExponent == 1.0f ? fk : std::pow(fk, kShareExponent);
  }
  Mat3 E{0.0f};
  if (sum <= 0.0f) return E;
  for (int k = 0; k < snakeClusterCount_; ++k) {
    const float fk = f[static_cast<std::size_t>(k)];
    if (fk <= 0.0f) continue;
    const float share = (kShareExponent == 1.0f ? fk : std::pow(fk, kShareExponent)) / sum;
    E += snakeClusters_[static_cast<std::size_t>(k)].polar * (share * fk);
  }
  const float free = m.mask.empty() ? 1.0f : 1.0f - m.mask[h0];
  return E * free;
}

SnakeStep Sculptor::snakeTo(const Vec3& target, float radius, float gain, const DabTopology& topology,
                            double budgetMs) {
  SnakeStep step;
  if (!object_ || !snakeStroke_ || !(radius > 0.0f)) return step;
  Timer total;
  lastDab_ = {};
  Mesh& m = object_->mesh;
  const auto hookAt = [&](int k) { return snakeHooks_[static_cast<std::size_t>(k)]; };

  // 1. Topology around every hook, each pass keeping its hook vertex.
  if (dyntopo_ && topology.detail > 0.0f) {
    int passes = 0;
    for (int k = 0; k < snakeClusterCount_; ++k)
      passes += hookAt(k) != kInvalid && onlySet_[snakeClusters_[static_cast<std::size_t>(k)].op] != kNoFaceSet;
    const double perPass = topologyBudget(passes);
    for (int k = 0; k < snakeClusterCount_; ++k) {
      const Index hook = hookAt(k);
      const std::int32_t only = onlySet_[snakeClusters_[static_cast<std::size_t>(k)].op];
      if (hook == kInvalid || only == kNoFaceSet || m.vertHe[hook] == kInvalid) continue;
      DabTopology t = topology;
      t.keepVertex = hook;
      t.hintFace = m.heFace[m.vertHe[hook]];
      t.onlySet = only;
      t.lockFaceSetBorders = options_.lockFaceSetBoundaries;
      t.timeBudgetMs = perPass;
      Dab d;
      d.center = m.positions[hook];
      d.radius = radius;
      applyTopology(d, t);
      reglueHooks(radius);
    }
    if (hookAt(0) == kInvalid) {  // Nothing left to pull (a damaged area).
      step.ms = total.ms();
      return step;
    }
  }

  // 2. The commanded pull: gain of the way to the target, steered so the hook really gets there
  // where copies or the mask hold it back, but never more than three times the plain pull.
  const Vec3 p0 = m.positions[hookAt(0)];
  const Vec3 tick = (target - p0) * std::clamp(gain, 0.0f, 1.0f);
  Vec3 command = tick;
  const Mat3 E = snakeSteering(radius);
  float offIdentity = 0.0f;
  for (int c = 0; c < 3; ++c)
    for (int r = 0; r < 3; ++r) offIdentity = std::max(offIdentity, std::abs(E[c][r] - (c == r ? 1.0f : 0.0f)));
  if (offIdentity >= 1e-6f) {
    constexpr float kLambda = 0.01f;
    const Mat3 Et = glm::transpose(E);
    command = glm::inverse(Et * E + Mat3{kLambda}) * (Et * tick);
    const float cap = 3.0f * glm::length(tick);
    if (const float len = glm::length(command); len > cap) command *= cap / len;
  }

  // 3. Sub-dabs glued to the hooks. One that would pull farther than a step is cut to a step and
  // the next tick covers the rest.
  const float length = glm::length(command);
  const float stepLen = kSnakeStep * radius;
  if (length > 1e-7f * radius) {
    const int n = std::clamp(static_cast<int>(std::ceil(length / stepLen)), 1, kMaxSnakeSubDabs);
    Vec3 part = command / static_cast<float>(n);
    if (glm::length(part) > stepLen) part *= stepLen / glm::length(part);
    pendingNormals_.clear();
    Dab base;
    base.radius = radius;
    base.strength = 1.0f;
    base.falloff = options_.falloff;
    base.pinch = snakePinch_;
    for (int i = 0; i < n; ++i) {
      int count = 0;
      for (int k = 0; k < snakeClusterCount_; ++k) {
        const Index hook = hookAt(k);
        if (hook == kInvalid) continue;
        if (copies_.size() <= static_cast<std::size_t>(count)) copies_.resize(static_cast<std::size_t>(count) + 1);
        CopyDab& c = copies_[static_cast<std::size_t>(count++)];
        c.cluster = snakeClusters_[static_cast<std::size_t>(k)];
        c.dab = base;
        c.dab.center = m.positions[hook];
        c.dab.pull = transform(c.cluster.polar, part);
        c.topology = {};
        c.live = false;
      }
      lastDab_.copies = group_.size();
      lastDab_.clusters = count;
      if (!applyCopies(count, {.deferNormals = true})) break;
      ++step.subDabs;
      if (budgetMs >= 0.0 && total.ms() >= budgetMs) break;
    }
    // 4. Normals once for every sub-dab of the tick; their owners were snapshotted already.
    if (!pendingNormals_.empty()) {
      Timer t;
      std::sort(pendingNormals_.begin(), pendingNormals_.end());
      pendingNormals_.erase(std::unique(pendingNormals_.begin(), pendingNormals_.end()), pendingNormals_.end());
      recomputeNormals(pendingNormals_);
      lastDab_.normalsMs += t.ms();
      lastDab_.vertices = static_cast<int>(pendingNormals_.size());
      pendingNormals_.clear();
    }
  }
  if (step.subDabs > 0) ++dabCount_;
  step.remaining = glm::length(target - m.positions[hookAt(0)]);
  step.ms = total.ms();
  lastDab_.totalMs = step.ms;
  return step;
}

std::array<FaceSetFilter, kMaxSymmetryCopies> strokeFilters(const SceneObject& object, const StrokeOptions& options,
                                                            const SymmetryGroup& group, const Vec3& center,
                                                            float radius) {
  // As Sculptor::resolveFaceSets and Sculptor::filter would set them.
  std::array<FaceSetFilter, kMaxSymmetryCopies> out{};
  const Mesh& m = object.mesh;
  const bool hidden = m.anyHidden();
  if (m.faceSets.empty() || (!hidden && !options.faceSetAutoMask && !options.lockFaceSetBoundaries)) return out;
  for (int k = 0; k < group.size(); ++k) {
    std::int32_t only = 0;
    if (options.faceSetAutoMask) {
      const Vec3 c = k == 0 ? center : transform(group.op(k).L, center);
      Bvh::ClosestHit hit;
      only = object.bvh.closestPoint(m, c, radius, hit, true) ? faceSetId(m.faceSets[hit.face]) : kNoFaceSet;
    }
    FaceSetFilter& f = out[static_cast<std::size_t>(k)];
    f.faceSets = m.faceSets.data();
    f.skipHidden = hidden;
    f.onlySet = only;
    f.lockBoundaries = options.lockFaceSetBoundaries;
  }
  return out;
}

PoseBuild buildPoseFor(const SceneObject& object, const StrokeOptions& options, const PoseOptions& pose,
                       const Vec3& click, float radius, const PoseLimits& limits, FanCache& fans) {
  const SymmetryGroup group(options.symmetry());
  const auto filters = strokeFilters(object, options, group, click, radius);
  return buildPose(object.mesh, object.bvh, nonManifoldFans(object, fans), group,
                   std::span(filters.data(), static_cast<std::size_t>(group.size())), click, radius, pose, limits);
}

bool Sculptor::beginPose(SceneObject& object, const StrokeOptions& options, const PoseOptions& pose,
                         const Vec3& click, float radius, std::string label, PoseBuild* reuse) {
  lastPoseFailure_ = PoseFailure::None;
  if (!(radius > 0.0f) || !layerTargetRefusal(object, nullptr, options.layerTarget).empty()) return false;
  start(object, options, std::move(label));
  beginLayers();
  pose_ = reuse ? std::move(*reuse) : buildPoseFor(object, options, pose, click, radius, {}, fans_);
  if (pose_.failure != PoseFailure::None || pose_.rigs.empty()) {
    lastPoseFailure_ = pose_.failure != PoseFailure::None ? pose_.failure : PoseFailure::NoSurface;
    pose_ = PoseBuild{};
    object_ = nullptr;
    return false;
  }
  for (Index l : pose_.snapshotLeaves) snapshot(l);
  poseStroke_ = true;
  poseClick_ = click;
  poseAngle_ = 0.0f;
  poseDeferNormals_ = pose_.bentNormals.size() > kLivePoseNormals;
  return true;
}

void Sculptor::pose(const Vec3& target, const Vec3& viewNormal) {
  if (!object_ || !poseStroke_) return;
  Timer total;
  lastDab_ = {};
  Mesh& m = object_->mesh;
  const Vec3 r = poseRotation(poseClick_, pose_.rigs[0].pivot, target, viewNormal);
  Timer t;
  for (const PoseRig& rig : pose_.rigs) applyPoseRig(m, rig, transform(rig.cluster.axial, r));
  poseAngle_ = glm::degrees(glm::length(transform(pose_.rigs[0].cluster.axial, r)));
  lastDab_.brushMs = t.ms();
  t.reset();
  if (!poseDeferNormals_) recomputeNormals(pose_.bentNormals);
  lastDab_.normalsMs = t.ms();
  // Bounds wait for the end of the stroke; the GPU takes the new positions now.
  if (pose_.snapshotLeaves.size() > 64) {
    object_->markPositionsDirtyAll(true);
  } else {
    for (Index l : pose_.snapshotLeaves) object_->markLeafDirty(l);
  }
  lastDab_.copies = group_.size();
  lastDab_.clusters = static_cast<int>(pose_.rigs.size());
  lastDab_.vertices = static_cast<int>(pose_.bentNormals.size());
  lastDab_.leaves = static_cast<int>(pose_.snapshotLeaves.size());
  lastDab_.totalMs = total.ms();
  ++dabCount_;
}

void Sculptor::finishPose() {
  // Exact normals for everything that moved and the ring around it, and bounds for every face
  // that changed shape.
  poseStroke_ = false;
  Mesh& m = object_->mesh;
  // Every list is ascending, so merging keeps the union ascending.
  std::vector<Index> verts = pose_.bentNormals, merged;
  for (const PoseRig& rig : pose_.rigs) {
    merged.clear();
    merged.reserve(verts.size() + rig.verts.size());
    std::set_union(verts.begin(), verts.end(), rig.verts.begin(), rig.verts.end(), std::back_inserter(merged));
    verts.swap(merged);
  }
  recomputeNormals(verts);
  object_->bvh.refitLeaves(m, pose_.refitLeaves);
  strokeRefit_.insert(strokeRefit_.end(), pose_.refitLeaves.begin(), pose_.refitLeaves.end());
  for (Index l : pose_.snapshotLeaves) object_->markLeafDirty(l);
  pose_ = PoseBuild{};
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

void Sculptor::commitLayerStroke() {
  Mesh& m = object_->mesh;
  Bvh& bvh = object_->bvh;
  const LayerStack& stack = m.layers;
  std::vector<Vec3>& target = *layerArray_;
  const float s = layerTarget_.strength;
  const bool base = undo_.layerTarget == 0;
  if (++stamp_ == 0) {
    std::fill(vertexStamp_.begin(), vertexStamp_.end(), 0);
    stamp_ = 1;
  }
  const std::uint32_t stamp = stamp_;
  const std::span<const BvhLeaf> leafList = bvh.leaves();
  const std::span<const std::pair<Index, Index>> fans = nonManifoldFans(*object_, fans_);
  const std::size_t snapshots = undo_.before.size();
  layerChanged_.resize(snapshots);
  layerRing_.resize(snapshots);
  layerLeaves_.resize(snapshots);
  // Snapshots cover disjoint vertex ranges, so they run in parallel.
  parallelFor(0, snapshots, 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t k = b; k < e; ++k) {
      // 1. Each vertex the stroke moved adds its change, divided by the target's strength, to the
      // target; then it takes the composite. Usually that is the position the brush left, else it
      // differs by rounding. On the base where no layer moved the vertex (the base equals the old
      // position), the base takes the new position as is, so the brush's result stays exact.
      std::vector<Index>& changed = layerChanged_[k];
      changed.clear();
      const LeafState& before = undo_.before[k];
      const Index begin = leafList[before.leaf].vertBegin;
      for (std::size_t i = 0; i < before.positions.size(); ++i) {
        const Index v = begin + static_cast<Index>(i);
        const Vec3 p0 = before.positions[i];
        const Vec3 p1 = m.positions[v];
        if (sameBits(p0, p1)) continue;
        Vec3& t = target[v];
        if (base && sameBits(t, p0)) {
          t = p1;
        } else {
          const float dx = p1.x - p0.x;
          const float dy = p1.y - p0.y;
          const float dz = p1.z - p0.z;
          if (dx != 0.0f) t.x = t.x + dx / s;
          if (dy != 0.0f) t.y = t.y + dy / s;
          if (dz != 0.0f) t.z = t.z + dz / s;
        }
        const Vec3 p = composeVertex(stack, v);
        if (!sameBits(p, p1)) {
          m.positions[v] = p;
          changed.push_back(v);
        }
      }
      // 2. Vertices sharing a face with a recomposed vertex get new normals, and the leaves of
      // those faces new bounds. Each vertex is claimed once, by whichever snapshot stamps it
      // first. Leaf lookups start from the last leaf found, which neighbours almost always share.
      std::vector<Index>& ring = layerRing_[k];
      std::vector<Index>& faceLeaves = layerLeaves_[k];
      ring.clear();
      faceLeaves.clear();
      Index faceLeaf = kInvalid;
      for (Index v : changed) {
        forEachFaceAt(m, fans, v, [&](Index f) {
          if (faceLeaf == kInvalid || f < leafList[faceLeaf].faceBegin || f >= leafList[faceLeaf].faceEnd) {
            faceLeaf = bvh.leafOfFace(f);
            if (faceLeaf != kInvalid) faceLeaves.push_back(faceLeaf);
          }
          m.forEachFaceVertex(f, [&](Index u) {
            std::atomic_ref<std::uint32_t> mark(vertexStamp_[u]);
            if (mark.load(std::memory_order_relaxed) != stamp &&
                mark.exchange(stamp, std::memory_order_relaxed) != stamp)
              ring.push_back(u);
          });
        });
      }
    }
  });

  // Their leaves are snapshotted before any normal is written: on a coarse mesh a face can reach
  // past the normals the dabs recomputed, and such a leaf still holds its values from before the
  // stroke.
  normalVerts_.clear();
  leaves_.clear();  // Leaves whose faces moved: their bounds change.
  for (std::size_t k = 0; k < snapshots; ++k) {
    normalVerts_.insert(normalVerts_.end(), layerRing_[k].begin(), layerRing_[k].end());
    leaves_.insert(leaves_.end(), layerLeaves_[k].begin(), layerLeaves_[k].end());
  }
  if (normalVerts_.empty()) return;
  dirtyLeaves_.clear();
  Index owner = kInvalid;
  for (Index v : normalVerts_) {
    if (owner != kInvalid && v >= leafList[owner].vertBegin && v < leafList[owner].vertEnd) continue;
    owner = bvh.leafOfVertex(v);
    if (owner != kInvalid) dirtyLeaves_.push_back(owner);
  }
  std::sort(dirtyLeaves_.begin(), dirtyLeaves_.end());
  dirtyLeaves_.erase(std::unique(dirtyLeaves_.begin(), dirtyLeaves_.end()), dirtyLeaves_.end());
  for (Index l : dirtyLeaves_) snapshot(l);

  // 3. Normals, bounds and uploads.
  recomputeNormals(normalVerts_);
  std::sort(leaves_.begin(), leaves_.end());
  leaves_.erase(std::unique(leaves_.begin(), leaves_.end()), leaves_.end());
  bvh.refitLeaves(m, leaves_);
  strokeRefit_.insert(strokeRefit_.end(), leaves_.begin(), leaves_.end());
  for (Index l : dirtyLeaves_) object_->markLeafDirty(l);
}

std::optional<StrokeUndo> Sculptor::endStroke() {
  if (!object_) return std::nullopt;
  if (poseStroke_) finishPose();
  // Commit the stroke into its layer target first: the after-snapshots must hold the committed
  // values. Dynamic topology never runs on layered meshes.
  if (layerArray_ && !dyntopo_ && !(brush_ && brush_->writesLayer()) && !undo_.before.empty()) commitLayerStroke();
  const std::vector<Vec3>* layer = layerArray_;
  SceneObject* obj = object_;
  object_ = nullptr;
  lastStroke_ = {};
  lastStroke_.tooManyCopies = tooManyCopies_;
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
  // whose positions stayed but whose normals changed (a neighbour moved) is kept, and so is one
  // where only the layer target changed (a layer brush whose change rounded away).
  std::vector<LeafState> before;
  before.reserve(undo_.before.size());
  undo_.after.reserve(undo_.before.size());
  kept_.clear();
  for (LeafState& b : undo_.before) {
    const BvhLeaf& l = obj->bvh.leaves()[b.leaf];
    if (std::equal(b.positions.begin(), b.positions.end(), m.positions.begin() + l.vertBegin) &&
        std::equal(b.normals.begin(), b.normals.end(), m.normals.begin() + l.vertBegin) &&
        (!layer || std::equal(b.layer.begin(), b.layer.end(), layer->begin() + l.vertBegin, sameBits)))
      continue;
    kept_.push_back(b.leaf);
    LeafState a;
    a.leaf = b.leaf;
    a.positions.assign(m.positions.begin() + l.vertBegin, m.positions.begin() + l.vertEnd);
    a.normals.assign(m.normals.begin() + l.vertBegin, m.normals.begin() + l.vertEnd);
    if (!b.layer.empty()) a.layer.assign(layer->begin() + l.vertBegin, layer->begin() + l.vertEnd);
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
