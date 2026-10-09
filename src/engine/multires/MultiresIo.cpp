#include "multires/MultiresIo.h"

#include <algorithm>
#include <numeric>

#include "multires/Propagate.h"
#include "multires/Subdivide.h"

namespace plegl {

namespace {

const Mesh& levelMesh(const SceneObject& o, int k) {
  return k == o.multires->active ? o.mesh : o.multires->levels[static_cast<std::size_t>(k)].mesh;
}

template <class T>
std::vector<T> toCanonical(const std::vector<T>& live, const std::vector<Index>& canon) {
  std::vector<T> out(live.size());
  for (std::size_t i = 0; i < live.size(); ++i) out[static_cast<std::size_t>(canon[i])] = live[i];
  return out;
}

template <class T>
std::vector<T> toLive(const std::vector<T>& canonical, const std::vector<Index>& canon) {
  std::vector<T> out(canonical.size());
  for (std::size_t i = 0; i < canon.size(); ++i) out[i] = canonical[static_cast<std::size_t>(canon[i])];
  return out;
}

// The polygons of a mesh in its own face order.
void polygons(const Mesh& m, std::vector<std::uint32_t>& sizes, std::vector<std::uint32_t>& corners) {
  sizes.resize(m.faceHe.size());
  corners.clear();
  corners.reserve(m.heNext.size());
  for (Index f = 0; f < m.faceCount(); ++f) {
    const std::size_t before = corners.size();
    m.forEachFaceVertex(f, [&](Index v) { corners.push_back(static_cast<std::uint32_t>(v)); });
    sizes[f] = static_cast<std::uint32_t>(corners.size() - before);
  }
}

}  // namespace

CanonicalLevel canonicalActiveLevel(const SceneObject& obj) {
  const Multires& s = *obj.multires;
  const CanonicalMap& canon = s.levels[static_cast<std::size_t>(s.active)].canon;
  const Mesh& m = obj.mesh;
  CanonicalLevel out;
  out.positions = toCanonical(m.positions, canon.vert);
  canonicalPolygons(m, canon, out.sizes, out.corners);
  if (!m.mask.empty()) out.mask = toCanonical(m.mask, canon.vert);
  if (!m.faceSets.empty()) out.faceSets = toCanonical(m.faceSets, canon.face);
  return out;
}

MultiresFileData captureLevels(const SceneObject& obj) {
  const Multires& s = *obj.multires;
  MultiresFileData data;
  data.active = s.active;
  const Mesh& base = levelMesh(obj, 0);
  const CanonicalMap& canon0 = s.levels[0].canon;
  canonicalPolygons(base, canon0, data.baseSizes, data.baseCorners);
  const std::vector<Index> canonHe = canonicalHalfEdges(base, canon0);
  data.baseVertices = static_cast<std::uint32_t>(base.vertexCount());
  data.baseTwins.assign(base.heNext.size(), kInvalid);
  for (Index h = 0; h < base.halfEdgeCount(); ++h)
    if (base.heTwin[h] != kInvalid) data.baseTwins[static_cast<std::size_t>(canonHe[h])] = canonHe[base.heTwin[h]];
  data.baseVertHe.assign(base.positions.size(), kInvalid);
  for (Index v = 0; v < base.vertexCount(); ++v)
    if (base.vertHe[v] != kInvalid) data.baseVertHe[static_cast<std::size_t>(canon0.vert[v])] = canonHe[base.vertHe[v]];

  for (int k = 0; k < s.levelCount(); ++k) {
    const Mesh& m = levelMesh(obj, k);
    const CanonicalMap& canon = s.levels[static_cast<std::size_t>(k)].canon;
    MultiresFileLevel fl;
    fl.vertices = static_cast<std::uint32_t>(m.vertexCount());
    fl.faces = static_cast<std::uint32_t>(m.faceCount());
    if (k != s.active) {
      fl.channels = kLevelPositions;
      fl.positions = toCanonical(m.positions, canon.vert);
      if (m.anyMasked()) {
        fl.channels |= kLevelMask;
        fl.mask = toCanonical(m.mask, canon.vert);
      }
      if (m.hasFaceSetData()) {
        fl.channels |= kLevelFaceSets;
        fl.faceSets = toCanonical(m.faceSets, canon.face);
      }
    }
    data.levels.push_back(std::move(fl));
  }

  // Pending edits: where the live level differs from the reference, the reference values.
  const CanonicalMap& canon = s.levels[static_cast<std::size_t>(s.active)].canon;
  const SyncDelta diff = diffActive(s, obj.mesh);
  if (!diff.levels.empty()) {
    const LevelDelta& d = diff.levels[0];
    auto sortPairs = [](auto& index, auto& values, const std::vector<Index>& map, const auto& live,
                        const auto& before) {
      std::vector<std::pair<std::uint32_t, std::size_t>> order;
      for (std::size_t i = 0; i < live.size(); ++i)
        order.emplace_back(static_cast<std::uint32_t>(map[static_cast<std::size_t>(live[i])]), i);
      std::sort(order.begin(), order.end());
      for (const auto& [c, i] : order) {
        index.push_back(c);
        values.push_back(before[i]);
      }
    };
    sortPairs(data.pendingPosIndex, data.pendingPos, canon.vert, d.posIndex, d.posBefore);
    sortPairs(data.pendingMaskIndex, data.pendingMask, canon.vert, d.maskIndex, d.maskBefore);
    sortPairs(data.pendingSetIndex, data.pendingSets, canon.face, d.setIndex, d.setBefore);
  }
  return data;
}

bool restoreLevels(const MultiresFileData& data, Mesh& mesh, Bvh& bvh, std::shared_ptr<Multires>& out,
                   std::string* error) {
  auto fail = [&](const char* why) {
    if (error) *error = std::string("The subdivision level data is damaged (") + why + ").";
    return false;
  };
  const int count = static_cast<int>(data.levels.size());
  const int active = data.active;
  if (count < 2 || count > kMaxMultiresLevels || active < 0 || active >= count) return fail("level count");

  // Base topology.
  const std::uint64_t V0 = data.baseVertices;
  const std::uint64_t F0 = data.baseSizes.size();
  const std::uint64_t H0 = data.baseCorners.size();
  if (V0 > std::uint64_t(INT32_MAX) || H0 > std::uint64_t(INT32_MAX) / 4 || F0 == 0) return fail("base size");
  std::uint64_t total = 0;
  for (std::uint32_t n : data.baseSizes) {
    if (n < 3) return fail("base face size");
    total += n;
  }
  if (total != H0 || data.baseTwins.size() != H0 || data.baseVertHe.size() != V0) return fail("base arrays");
  for (std::uint32_t c : data.baseCorners)
    if (c >= V0) return fail("base corner");
  std::vector<Index> heNext(H0), heFace(H0), faceStart(F0 + 1);
  {
    Index h = 0;
    for (std::size_t f = 0; f < F0; ++f) {
      faceStart[f] = h;
      const Index n = static_cast<Index>(data.baseSizes[f]);
      for (Index i = 0; i < n; ++i) {
        heNext[static_cast<std::size_t>(h + i)] = h + (i + 1) % n;
        heFace[static_cast<std::size_t>(h + i)] = static_cast<Index>(f);
      }
      h += n;
    }
    faceStart[F0] = h;
  }
  std::uint64_t E0 = 0;
  std::vector<std::uint8_t> used(V0, 0);
  for (std::size_t h = 0; h < H0; ++h) {
    used[data.baseCorners[h]] = 1;
    const std::int32_t t = data.baseTwins[h];
    if (t == kInvalid) {
      ++E0;
      continue;
    }
    if (t < 0 || std::uint64_t(t) >= H0 || std::size_t(t) == h || data.baseTwins[std::size_t(t)] != Index(h))
      return fail("base twins");
    if (data.baseCorners[std::size_t(t)] != data.baseCorners[std::size_t(heNext[h])] ||
        data.baseCorners[std::size_t(heNext[std::size_t(t)])] != data.baseCorners[h])
      return fail("base twin endpoints");
    if (Index(h) < t) ++E0;
  }
  for (std::size_t v = 0; v < V0; ++v) {
    const std::int32_t h = data.baseVertHe[v];
    if (h == kInvalid) {
      if (used[v]) return fail("base fan start");
      continue;
    }
    if (h < 0 || std::uint64_t(h) >= H0 || data.baseCorners[std::size_t(h)] != v) return fail("base fan start");
  }

  // Counts every level must have.
  {
    std::uint64_t V = V0, E = E0, F = F0, H = H0;
    for (int k = 0; k < count; ++k) {
      const MultiresFileLevel& l = data.levels[static_cast<std::size_t>(k)];
      if (l.vertices != V || l.faces != F) return fail("level counts");
      if (k > 0 && F > std::uint64_t(kMaxMultiresFaces)) return fail("level size");
      const std::uint64_t nV = V + E + F, nE = 2 * E + H, nF = H, nH = 4 * H;
      V = nV, E = nE, F = nF, H = nH;
    }
  }
  for (int k = 0; k < count; ++k) {
    const MultiresFileLevel& l = data.levels[static_cast<std::size_t>(k)];
    if (k == active) {
      if (l.channels != 0) return fail("active level channels");
      continue;
    }
    if (!(l.channels & kLevelPositions) || (l.channels & ~(kLevelPositions | kLevelMask | kLevelFaceSets)))
      return fail("level channels");
    if (l.positions.size() != l.vertices) return fail("level positions");
    if ((l.channels & kLevelMask) ? l.mask.size() != l.vertices : !l.mask.empty()) return fail("level mask");
    if ((l.channels & kLevelFaceSets) ? l.faceSets.size() != l.faces : !l.faceSets.empty()) return fail("level face sets");
  }
  const MultiresFileLevel& activeLevel = data.levels[static_cast<std::size_t>(active)];
  if (std::uint64_t(mesh.vertexCount()) != activeLevel.vertices || std::uint64_t(mesh.faceCount()) != activeLevel.faces)
    return fail("active level size");

  // Values of level k in canonical order: the object's own for the active level.
  auto positionsOf = [&](int k) -> const std::vector<Vec3>& {
    return k == active ? mesh.positions : data.levels[static_cast<std::size_t>(k)].positions;
  };
  auto maskOf = [&](int k) -> const std::vector<float>& {
    return k == active ? mesh.mask : data.levels[static_cast<std::size_t>(k)].mask;
  };
  auto setsOf = [&](int k) -> const std::vector<std::int32_t>& {
    return k == active ? mesh.faceSets : data.levels[static_cast<std::size_t>(k)].faceSets;
  };
  std::vector<std::uint32_t> objSizes, objCorners;
  polygons(mesh, objSizes, objCorners);

  // Level 0, straight from the stored arrays (buildMesh could pair a non-manifold base otherwise).
  auto stack = std::make_shared<Multires>();
  stack->levels.resize(static_cast<std::size_t>(count));
  {
    if (active == 0 && (objSizes != data.baseSizes || objCorners != data.baseCorners)) return fail("base polygons");
    Mesh base;
    base.positions = positionsOf(0);
    base.mask = maskOf(0);
    base.faceSets = setsOf(0);
    base.heNext = std::move(heNext);
    base.heFace = std::move(heFace);
    base.heVert.assign(data.baseCorners.begin(), data.baseCorners.end());
    base.heTwin.assign(data.baseTwins.begin(), data.baseTwins.end());
    base.vertHe.assign(data.baseVertHe.begin(), data.baseVertHe.end());
    base.faceHe.assign(faceStart.begin(), faceStart.end() - 1);
    base.computeNormals();
    MultiresLevel& l0 = stack->levels[0];
    ReorderMap map;
    l0.bvh.build(base, Bvh::Params{}, &map);
    l0.canon.vert = std::move(map.vertOld);
    l0.canon.face = std::move(map.faceOld);
    l0.canon.faceStart = std::move(faceStart);
    l0.rule = classifyVertices(base);
    l0.nonManifoldFaces = findNonManifoldFans(base);
    l0.mesh = std::move(base);
  }
  for (int k = 1; k < count; ++k) {
    MultiresLevel& parent = stack->levels[static_cast<std::size_t>(k - 1)];
    std::string why;
    auto r = subdivide(parent.mesh, parent.canon, parent.rule, SubdivideOptions{}, &why);
    if (!r) return fail("subdivision");
    if (k == active) {
      std::vector<std::uint32_t> sizes, corners;
      canonicalPolygons(r->mesh, r->canon, sizes, corners);
      if (sizes != objSizes || corners != objCorners) return fail("active level polygons");
    }
    Mesh& m = r->mesh;
    m.positions = toLive(positionsOf(k), r->canon.vert);
    m.mask = maskOf(k).empty() ? std::vector<float>{} : toLive(maskOf(k), r->canon.vert);
    m.faceSets = setsOf(k).empty() ? std::vector<std::int32_t>{} : toLive(setsOf(k), r->canon.face);
    m.computeNormals();
    r->bvh.refit(m);
    MultiresLevel& l = stack->levels[static_cast<std::size_t>(k)];
    l.mesh = std::move(m);
    l.bvh = std::move(r->bvh);
    l.rule = std::move(r->rule);
    l.nonManifoldFaces = findNonManifoldFans(l.mesh);
    l.canon = std::move(r->canon);
    l.links = std::move(r->links);
  }
  // Level 0 values were taken in canonical order before the BVH build reordered them.
  for (MultiresLevel& l : stack->levels) {
    l.version = nextTopologyVersion();
    l.layoutHash = layoutHash(l.bvh);
    stack->faceSetIdBound = std::max(stack->faceSetIdBound, l.mesh.maxFaceSetId());
  }

  // The active level goes to the object; its reference gets the pending edits back.
  MultiresLevel& a = stack->levels[static_cast<std::size_t>(active)];
  Mesh live = std::exchange(a.mesh, Mesh{});
  Bvh liveBvh = std::exchange(a.bvh, Bvh{});
  stack->active = active;
  LevelReference& ref = stack->reference;
  ref.positions = live.positions;
  ref.mask = live.mask;
  ref.faceSets = live.faceSets;
  const std::vector<Index> vertLive = invertMap(a.canon.vert), faceLive = invertMap(a.canon.face);
  auto ascending = [](const std::vector<std::uint32_t>& idx, std::size_t limit) {
    for (std::size_t i = 0; i < idx.size(); ++i)
      if (idx[i] >= limit || (i > 0 && idx[i] <= idx[i - 1])) return false;
    return true;
  };
  if (!ascending(data.pendingPosIndex, live.positions.size()) || data.pendingPos.size() != data.pendingPosIndex.size() ||
      !ascending(data.pendingMaskIndex, live.positions.size()) ||
      data.pendingMask.size() != data.pendingMaskIndex.size() ||
      !ascending(data.pendingSetIndex, live.faceHe.size()) || data.pendingSets.size() != data.pendingSetIndex.size())
    return fail("pending edits");
  for (std::size_t i = 0; i < data.pendingPosIndex.size(); ++i)
    ref.positions[static_cast<std::size_t>(vertLive[data.pendingPosIndex[i]])] = data.pendingPos[i];
  if (!data.pendingMaskIndex.empty() && ref.mask.empty()) ref.mask.assign(ref.positions.size(), 0.0f);
  for (std::size_t i = 0; i < data.pendingMaskIndex.size(); ++i)
    ref.mask[static_cast<std::size_t>(vertLive[data.pendingMaskIndex[i]])] = data.pendingMask[i];
  if (!data.pendingSetIndex.empty() && ref.faceSets.empty()) ref.faceSets.assign(live.faceHe.size(), kDefaultFaceSet);
  for (std::size_t i = 0; i < data.pendingSetIndex.size(); ++i)
    ref.faceSets[static_cast<std::size_t>(faceLive[data.pendingSetIndex[i]])] = data.pendingSets[i];
  for (std::int32_t v : ref.faceSets) stack->faceSetIdBound = std::max(stack->faceSetIdBound, faceSetId(v));

  mesh = std::move(live);
  bvh = std::move(liveBvh);
  out = std::move(stack);
  return true;
}

}  // namespace plegl
