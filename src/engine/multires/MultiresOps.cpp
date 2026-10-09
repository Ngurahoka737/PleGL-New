#include "multires/MultiresOps.h"

#include <algorithm>
#include <iterator>

namespace plegl {

namespace {

int countOf(const SceneObject& o) { return o.multires ? o.multires->levelCount() : 0; }
int activeOf(const SceneObject& o) { return o.multires ? o.multires->active : 0; }

void takeReference(SceneObject& obj) {
  LevelReference& ref = obj.multires->reference;
  ref.positions = obj.mesh.positions;
  ref.mask = obj.mesh.mask;
  ref.faceSets = obj.mesh.faceSets;
}

// Moves the active level from the object into the stack.
void park(SceneObject& obj) {
  MultiresLevel& l = obj.multires->levels[static_cast<std::size_t>(obj.multires->active)];
  l.mesh = std::exchange(obj.mesh, Mesh{});
  l.bvh = std::exchange(obj.bvh, Bvh{});
}

// Moves level k from the stack into the object and makes it active. Its BVH is only refit, never
// rebuilt, so its leaf ranges and topology version are the ones its undo entries were made on.
void unpark(SceneObject& obj, int k) {
  Multires& s = *obj.multires;
  MultiresLevel& l = s.levels[static_cast<std::size_t>(k)];
  obj.mesh = std::exchange(l.mesh, Mesh{});
  obj.bvh = std::exchange(l.bvh, Bvh{});
  if (l.boundsStale) {
    obj.bvh.refit(obj.mesh);
    l.boundsStale = false;
  }
  obj.topologyVersion = l.version;
  s.active = k;
  takeReference(obj);
  obj.clearDirty();
  ++s.serial;
}

void growBound(SceneObject& obj) {
  obj.multires->faceSetIdBound = std::max(obj.multires->faceSetIdBound, obj.mesh.maxFaceSetId());
}

SyncDelta syncPending(const SceneObject& obj, SyncWorkspace& ws) {
  const Multires& s = *obj.multires;
  return computeSync(s, obj.mesh, diffActive(s, obj.mesh), ws);
}

// A parked level dressed as a scene object, so object operations can run on it.
class ParkedObject {
 public:
  ParkedObject(const SceneObject& owner, MultiresLevel& level) : level_(level) {
    obj.id = owner.id;
    obj.topologyVersion = level.version;
    obj.mesh = std::exchange(level.mesh, Mesh{});
    obj.bvh = std::exchange(level.bvh, Bvh{});
  }
  ~ParkedObject() {
    level_.mesh = std::exchange(obj.mesh, Mesh{});
    level_.bvh = std::exchange(obj.bvh, Bvh{});
  }
  ParkedObject(const ParkedObject&) = delete;
  ParkedObject& operator=(const ParkedObject&) = delete;
  SceneObject obj;

 private:
  MultiresLevel& level_;
};

MultiresUndo makeEntry(const SceneObject& obj, MultiresOp op, std::string label) {
  MultiresUndo u;
  u.label = std::move(label);
  u.objectId = obj.id;
  u.op = op;
  u.levelBefore = u.levelAfter = activeOf(obj);
  u.countBefore = u.countAfter = countOf(obj);
  u.versionBefore = u.versionAfter = obj.topologyVersion;
  return u;
}

// Delete Higher and Delete Lower, shared by the commands and their redo.
void dropHigher(SceneObject& obj, MultiresUndo& u) {
  Multires& s = *obj.multires;
  const auto keep = static_cast<std::size_t>(s.active + 1);
  u.held.assign(std::make_move_iterator(s.levels.begin() + static_cast<std::ptrdiff_t>(keep)),
                std::make_move_iterator(s.levels.end()));
  s.levels.resize(keep);
  ++s.serial;
  if (s.levelCount() == 1) u.heldStack = std::exchange(obj.multires, nullptr);
}

void dropLower(SceneObject& obj, MultiresUndo& u) {
  Multires& s = *obj.multires;
  const auto drop = static_cast<std::ptrdiff_t>(s.active);
  u.held.assign(std::make_move_iterator(s.levels.begin()), std::make_move_iterator(s.levels.begin() + drop));
  s.levels.erase(s.levels.begin(), s.levels.begin() + drop);
  u.heldLinks = std::exchange(s.levels[0].links, SubdivisionLinks{});
  s.active = 0;
  ++s.serial;
  if (s.levelCount() == 1) u.heldStack = std::exchange(obj.multires, nullptr);
}

template <class Op>
std::optional<MultiresUndo> allLevels(SceneObject& obj, std::string label, SyncWorkspace& ws, Op&& op) {
  if (!obj.multires) return std::nullopt;
  Multires& s = *obj.multires;
  MultiresUndo u = makeEntry(obj, MultiresOp::AllLevels, std::move(label));
  u.sync = syncPending(obj, ws);
  applyDelta(s, u.sync, true, ws);
  growBound(obj);
  for (int k = 0; k < s.levelCount(); ++k) {
    std::optional<SculptUndo> r;
    if (k == s.active) {
      r = op(obj);
    } else {
      ParkedObject parked(obj, s.levels[static_cast<std::size_t>(k)]);
      r = op(parked.obj);
    }
    if (r) u.perLevel.emplace_back(k, std::move(*r));
  }
  if (u.perLevel.empty()) {
    applyDelta(s, u.sync, false, ws);  // Nothing changed: leave the edits pending as they were.
    return std::nullopt;
  }
  takeReference(obj);
  ++s.serial;
  return u;
}

void applyLevelStates(SceneObject& obj, MultiresUndo& u, bool redo) {
  Multires& s = *obj.multires;
  auto applyOne = [&](std::pair<int, SculptUndo>& p) {
    const std::vector<LeafState>& states = redo ? p.second.after : p.second.before;
    if (p.first == s.active) {
      applySculptStates(obj, p.second, states);
    } else {
      ParkedObject parked(obj, s.levels[static_cast<std::size_t>(p.first)]);
      applySculptStates(parked.obj, p.second, states);
    }
  };
  if (redo) {
    for (auto& p : u.perLevel) applyOne(p);
  } else {
    for (auto it = u.perLevel.rbegin(); it != u.perLevel.rend(); ++it) applyOne(*it);
  }
}

}  // namespace

std::optional<SubdivideJob> prepareSubdivide(const SceneObject& o, std::string* error) {
  auto fail = [&](const char* message) -> std::optional<SubdivideJob> {
    if (error) *error = message;
    return std::nullopt;
  };
  if (o.mesh.faceCount() == 0) return fail("The object has no faces.");
  if (o.multires) {
    const Multires& s = *o.multires;
    if (s.active != s.top()) return fail("Subdivide works from the highest level.");
    if (s.levelCount() >= kMaxMultiresLevels) return fail("The object already has the most levels (8).");
  }
  if (o.mesh.halfEdgeCount() > kMaxMultiresFaces) return fail("The next level would have more than 8.4 million faces.");
  SubdivideJob job;
  job.objectId = o.id;
  job.version = o.topologyVersion;
  job.plain = !o.multires;
  job.serial = o.multires ? o.multires->serial : 0;
  job.maxLeafFaces = o.bvh.maxLeafFaces();
  job.mesh = o.mesh;
  if (!job.plain) {
    const MultiresLevel& top = o.multires->levels[static_cast<std::size_t>(o.multires->active)];
    job.canon = top.canon;
    job.rule = top.rule;
  }
  return job;
}

void runSubdivideJob(SubdivideJob& job) {
  if (job.plain) {
    job.canon = identityCanonicalMap(job.mesh);
    job.rule = classifyVertices(job.mesh);
    job.baseNonManifold = findNonManifoldFans(job.mesh);
  }
  SubdivideOptions options;
  options.maxLeafFaces = job.maxLeafFaces;
  job.result = subdivide(job.mesh, job.canon, job.rule, options, &job.error);
  if (job.result) job.nonManifold = findNonManifoldFans(job.result->mesh);
  job.mesh = Mesh{};
}

std::optional<MultiresUndo> finishSubdivide(SceneObject& obj, SubdivideJob& job, SyncWorkspace& ws) {
  if (!job.result || obj.id != job.objectId || obj.topologyVersion != job.version) return std::nullopt;
  if (job.plain != !obj.multires) return std::nullopt;
  if (!job.plain && (obj.multires->serial != job.serial || obj.multires->active != obj.multires->top()))
    return std::nullopt;
  MultiresUndo u = makeEntry(obj, MultiresOp::Subdivide, "Subdivide");
  if (job.plain) {
    // The existing mesh becomes the base as it is: same layout, same version, so every undo entry
    // recorded on it still applies.
    auto stack = std::make_shared<Multires>();
    MultiresLevel base;
    base.version = obj.topologyVersion;
    base.layoutHash = layoutHash(obj.bvh);
    base.rule = std::move(job.rule);
    base.nonManifoldFaces = std::move(job.baseNonManifold);
    base.canon = std::move(job.canon);
    stack->levels.push_back(std::move(base));
    obj.multires = std::move(stack);
    takeReference(obj);
  } else {
    u.sync = syncPending(obj, ws);
    applyDelta(*obj.multires, u.sync, true, ws);
  }
  growBound(obj);
  Multires& s = *obj.multires;
  SubdivisionResult& r = *job.result;
  MultiresLevel level;
  level.mesh = std::move(r.mesh);
  level.bvh = std::move(r.bvh);
  level.version = nextTopologyVersion();
  level.layoutHash = layoutHash(level.bvh);
  level.rule = std::move(r.rule);
  level.nonManifoldFaces = std::move(job.nonManifold);
  level.canon = std::move(r.canon);
  level.links = std::move(r.links);
  s.levels.push_back(std::move(level));
  job.result.reset();
  park(obj);
  unpark(obj, s.top());
  u.levelAfter = s.active;
  u.countAfter = s.levelCount();
  u.versionAfter = obj.topologyVersion;
  return u;
}

std::optional<MultiresUndo> subdivideObject(SceneObject& obj, SyncWorkspace& ws, std::string* error, int maxLeafFaces) {
  std::optional<SubdivideJob> job = prepareSubdivide(obj, error);
  if (!job) return std::nullopt;
  job->maxLeafFaces = maxLeafFaces;
  runSubdivideJob(*job);
  if (!job->result) {
    if (error) *error = job->error;
    return std::nullopt;
  }
  return finishSubdivide(obj, *job, ws);
}

std::optional<MultiresUndo> setActiveLevel(SceneObject& obj, int target, SyncWorkspace& ws) {
  if (!obj.multires || target < 0 || target >= obj.multires->levelCount() || target == obj.multires->active)
    return std::nullopt;
  return setActiveLevel(obj, target, syncPending(obj, ws), ws);
}

std::optional<MultiresUndo> setActiveLevel(SceneObject& obj, int target, SyncDelta sync, SyncWorkspace& ws) {
  if (!obj.multires || target < 0 || target >= obj.multires->levelCount() || target == obj.multires->active)
    return std::nullopt;
  Multires& s = *obj.multires;
  MultiresUndo u = makeEntry(obj, MultiresOp::Switch, "Level " + std::to_string(target));
  applyDelta(s, sync, true, ws);
  growBound(obj);
  park(obj);
  unpark(obj, target);
  u.levelAfter = target;
  u.versionAfter = obj.topologyVersion;
  u.sync = std::move(sync);
  return u;
}

std::optional<MultiresUndo> deleteHigherLevels(SceneObject& obj) {
  if (!obj.multires || obj.multires->active == obj.multires->top()) return std::nullopt;
  MultiresUndo u = makeEntry(obj, MultiresOp::DeleteHigher, "Delete Higher Levels");
  dropHigher(obj, u);
  u.countAfter = countOf(obj);
  return u;
}

std::optional<MultiresUndo> deleteLowerLevels(SceneObject& obj) {
  if (!obj.multires || obj.multires->active == 0) return std::nullopt;
  MultiresUndo u = makeEntry(obj, MultiresOp::DeleteLower, "Delete Lower Levels");
  dropLower(obj, u);
  u.levelAfter = activeOf(obj);
  u.countAfter = countOf(obj);
  return u;
}

std::optional<MultiresUndo> applyMaskOpAllLevels(SceneObject& obj, MaskOp op, SyncWorkspace& ws) {
  return allLevels(obj, maskOpName(op), ws, [&](SceneObject& o) { return applyMaskOp(o, op); });
}

std::optional<MultiresUndo> applyFaceSetOpAllLevels(SceneObject& obj, FaceSetOp op, SyncWorkspace& ws) {
  return allLevels(obj, faceSetOpName(op), ws, [&](SceneObject& o) { return applyFaceSetOp(o, op); });
}

bool applyMultiresUndo(Scene& scene, MultiresUndo& u, bool redo, SyncWorkspace& ws) {
  SceneObject* obj = scene.find(u.objectId);
  if (!obj) return false;
  // The object must be exactly where the entry left it (undo) or found it (redo).
  const int count = redo ? u.countBefore : u.countAfter;
  const int level = redo ? u.levelBefore : u.levelAfter;
  const std::uint64_t version = redo ? u.versionBefore : u.versionAfter;
  if (countOf(*obj) != count || activeOf(*obj) != level || obj->topologyVersion != version) return false;

  switch (u.op) {
    case MultiresOp::Switch:
      if (!redo) {
        park(*obj);
        unpark(*obj, u.levelBefore);
        applyDelta(*obj->multires, u.sync, false, ws);  // The reference is back: edits are pending again.
      } else {
        applyDelta(*obj->multires, u.sync, true, ws);
        park(*obj);
        unpark(*obj, u.levelAfter);
      }
      return true;

    case MultiresOp::Subdivide:
      if (!redo) {
        Multires& s = *obj->multires;
        park(*obj);
        u.held.clear();
        u.held.push_back(std::move(s.levels.back()));
        s.levels.pop_back();
        unpark(*obj, u.levelBefore);
        applyDelta(s, u.sync, false, ws);
        if (u.countBefore == 0) u.heldStack = std::exchange(obj->multires, nullptr);
      } else {
        if (u.held.size() != 1) return false;
        if (u.countBefore == 0) {
          if (!u.heldStack) return false;
          obj->multires = std::exchange(u.heldStack, nullptr);
          takeReference(*obj);
        }
        Multires& s = *obj->multires;
        applyDelta(s, u.sync, true, ws);
        s.levels.push_back(std::move(u.held[0]));
        u.held.clear();
        park(*obj);
        unpark(*obj, s.top());
      }
      return true;

    case MultiresOp::DeleteHigher:
      if (!redo) {
        if (u.heldStack) obj->multires = std::exchange(u.heldStack, nullptr);
        Multires& s = *obj->multires;
        for (MultiresLevel& l : u.held) s.levels.push_back(std::move(l));
        u.held.clear();
        ++s.serial;
      } else {
        dropHigher(*obj, u);
      }
      return true;

    case MultiresOp::DeleteLower:
      if (!redo) {
        if (u.heldStack) obj->multires = std::exchange(u.heldStack, nullptr);
        Multires& s = *obj->multires;
        s.levels.insert(s.levels.begin(), std::make_move_iterator(u.held.begin()), std::make_move_iterator(u.held.end()));
        u.held.clear();
        s.levels[static_cast<std::size_t>(u.levelBefore)].links = std::exchange(u.heldLinks, SubdivisionLinks{});
        s.active = u.levelBefore;
        ++s.serial;
      } else {
        dropLower(*obj, u);
      }
      return true;

    case MultiresOp::AllLevels: {
      Multires& s = *obj->multires;
      if (!redo) {
        applyLevelStates(*obj, u, false);
        takeReference(*obj);
        applyDelta(s, u.sync, false, ws);
      } else {
        applyDelta(s, u.sync, true, ws);
        applyLevelStates(*obj, u, true);
        takeReference(*obj);
      }
      ++s.serial;
      return true;
    }
  }
  return false;
}

ValidationResult validateMultires(const SceneObject& obj) {
  auto fail = [](std::string msg) { return ValidationResult{false, std::move(msg)}; };
  if (!obj.multires) return fail("no levels");
  const Multires& s = *obj.multires;
  const int count = s.levelCount();
  if (count < 2 || count > kMaxMultiresLevels) return fail("level count " + std::to_string(count));
  if (s.active < 0 || s.active >= count) return fail("active level out of range");
  if (obj.topologyVersion != s.levels[static_cast<std::size_t>(s.active)].version)
    return fail("object version is not the active level's");
  auto meshOf = [&](int k) -> const Mesh& { return k == s.active ? obj.mesh : s.levels[static_cast<std::size_t>(k)].mesh; };
  auto bvhOf = [&](int k) -> const Bvh& { return k == s.active ? obj.bvh : s.levels[static_cast<std::size_t>(k)].bvh; };
  for (int k = 0; k < count; ++k) {
    const std::string at = " on level " + std::to_string(k);
    const MultiresLevel& l = s.levels[static_cast<std::size_t>(k)];
    const Mesh& m = meshOf(k);
    if (k == s.active && l.mesh.vertexCount() != 0) return fail("active level also parked");
    for (int j = 0; j < k; ++j)
      if (s.levels[static_cast<std::size_t>(j)].version == l.version) return fail("shared version" + at);
    // Parked levels refit only when they become active, so check ranges on refit bounds.
    Bvh refit = bvhOf(k);
    refit.refit(m);
    const ValidationResult layout = validateLayout(m, refit);
    if (!layout.ok) return fail(layout.message + at);
    if (layoutHash(bvhOf(k)) != l.layoutHash) return fail("leaf layout changed" + at);
    if (l.rule != classifyVertices(m)) return fail("vertex rules" + at);
    if (l.nonManifoldFaces != findNonManifoldFans(m)) return fail("non-manifold fans" + at);
    if (l.canon.vert.size() != m.positions.size() || l.canon.face.size() != m.faceHe.size())
      return fail("canonical map size" + at);
    std::vector<std::uint8_t> seen(m.positions.size(), 0);
    for (Index c : l.canon.vert) {
      if (c < 0 || c >= m.vertexCount() || seen[c]) return fail("canonical vertex map" + at);
      seen[c] = 1;
    }
    seen.assign(m.faceHe.size(), 0);
    for (Index c : l.canon.face) {
      if (c < 0 || c >= m.faceCount() || seen[c]) return fail("canonical face map" + at);
      seen[c] = 1;
    }
    if (k == 0) continue;
    const Mesh& c = meshOf(k - 1);
    const SubdivisionLinks& L = l.links;
    if (m.vertexCount() != c.vertexCount() + c.edgeCount() + c.faceCount() || m.faceCount() != c.halfEdgeCount())
      return fail("level counts" + at);
    if (L.vertexChild.size() != c.positions.size() || L.edgeChild.size() != c.heNext.size() ||
        L.faceChild.size() != c.faceHe.size() || L.childFace.size() != c.heNext.size() ||
        L.parent.size() != m.positions.size() || L.parentHalfEdge.size() != m.faceHe.size())
      return fail("link sizes" + at);
    for (Index v = 0; v < c.vertexCount(); ++v)
      if (L.parent[L.vertexChild[v]] != makeParent(kParentVertex, v)) return fail("vertex links" + at);
    for (Index f = 0; f < c.faceCount(); ++f)
      if (L.parent[L.faceChild[f]] != makeParent(kParentFace, f)) return fail("face links" + at);
    for (Index h = 0; h < c.halfEdgeCount(); ++h) {
      const std::uint32_t p = L.parent[L.edgeChild[h]];
      if (parentKind(p) != kParentEdge || (parentIndex(p) != h && parentIndex(p) != c.heTwin[h]))
        return fail("edge links" + at);
      if (L.parentHalfEdge[L.childFace[h]] != h) return fail("child face links" + at);
      if (m.heVert[m.faceHe[L.childFace[h]]] != L.vertexChild[c.heVert[h]]) return fail("child face start" + at);
    }
  }
  const Mesh& live = obj.mesh;
  if (s.reference.positions.size() != live.positions.size() ||
      (!s.reference.mask.empty() && s.reference.mask.size() != live.positions.size()) ||
      (!s.reference.faceSets.empty() && s.reference.faceSets.size() != live.faceHe.size()))
    return fail("reference size");
  for (int k = 0; k < count; ++k)
    if (k != s.active && s.levels[static_cast<std::size_t>(k)].mesh.maxFaceSetId() > s.faceSetIdBound)
      return fail("face set id above the bound on level " + std::to_string(k));
  return {};
}

}  // namespace plegl
