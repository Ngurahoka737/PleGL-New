#include "spatial/Bvh.h"

#include "core/Geometry.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "core/Parallel.h"

namespace plegl {
namespace {

struct BuildItem {
  Index begin, end;  // Range in the face order array.
  Index parent;      // Parent node, kInvalid for the root.
  bool isLeft;
};

// Slab test; returns entry distance or +inf when missed.
float rayAabb(const Vec3& origin, const Vec3& invDir, const Aabb& b, float tMax) {
  const Vec3 t0 = (b.min - origin) * invDir;
  const Vec3 t1 = (b.max - origin) * invDir;
  const Vec3 tn = glm::min(t0, t1);
  const Vec3 tf = glm::max(t0, t1);
  const float enter = std::max(std::max(tn.x, tn.y), std::max(tn.z, 0.0f));
  const float exit = std::min(std::min(tf.x, tf.y), std::min(tf.z, tMax));
  return enter <= exit ? enter : std::numeric_limits<float>::infinity();
}

// Möller-Trumbore, double sided. Writes t, u, v on hit. Barycentrics get a small tolerance so a
// ray through a shared edge or vertex cannot slip between neighbouring triangles (it would then
// hit the far side of the mesh).
constexpr float kBaryEps = 1e-5f;

bool rayTriangle(const Ray& ray, const Vec3& a, const Vec3& b, const Vec3& c, float& t, float& u, float& v) {
  const Vec3 e1 = b - a;
  const Vec3 e2 = c - a;
  const Vec3 p = glm::cross(ray.dir, e2);
  const float det = glm::dot(e1, p);
  if (std::abs(det) < 1e-20f) return false;
  const float inv = 1.0f / det;
  const Vec3 s = ray.origin - a;
  u = glm::dot(s, p) * inv;
  if (u < -kBaryEps || u > 1.0f + kBaryEps) return false;
  const Vec3 q = glm::cross(s, e1);
  v = glm::dot(ray.dir, q) * inv;
  if (v < -kBaryEps || u + v > 1.0f + kBaryEps) return false;
  t = glm::dot(e2, q) * inv;
  return t >= 0.0f;
}

}  // namespace

void Bvh::build(Mesh& mesh, const Params& params, ReorderMap* map) {
  nodes_.clear();
  leaves_.clear();
  leafNode_.clear();
  parent_.clear();
  dynamic_ = false;
  const Index maxLeaf = std::max(params.maxLeafFaces, 1);
  maxLeafFaces_ = maxLeaf;
  const Index nf = mesh.faceCount();
  if (nf == 0) {
    if (map) {
      map->faceOld.clear();
      map->vertOld.resize(mesh.positions.size());
      std::iota(map->vertOld.begin(), map->vertOld.end(), 0);
    }
    return;
  }

  std::vector<Vec3> centroids(nf);
  parallelFor(0, static_cast<std::size_t>(nf), 8192, [&](std::size_t b, std::size_t e) {
    for (std::size_t f = b; f < e; ++f) centroids[f] = mesh.faceCentroid(static_cast<Index>(f));
  });

  std::vector<Index> order(nf);
  std::iota(order.begin(), order.end(), 0);

  // Depth-first build. Nodes are allocated when popped and the left child is pushed last, so
  // nodes end up in preorder and leaves in left-to-right order.
  std::vector<BuildItem> stack{{0, nf, kInvalid, true}};
  while (!stack.empty()) {
    const BuildItem item = stack.back();
    stack.pop_back();
    const Index node = static_cast<Index>(nodes_.size());
    nodes_.push_back({});
    parent_.push_back(item.parent);
    if (item.parent != kInvalid) (item.isLeft ? nodes_[item.parent].left : nodes_[item.parent].right) = node;

    const Index count = item.end - item.begin;
    if (count <= maxLeaf) {
      nodes_[node].leaf = static_cast<Index>(leaves_.size());
      leafNode_.push_back(node);
      BvhLeaf leaf;
      leaf.faceBegin = item.begin;
      leaf.faceEnd = item.end;
      leaves_.push_back(leaf);
      continue;
    }
    Aabb cb;
    for (Index i = item.begin; i < item.end; ++i) cb.expand(centroids[order[i]]);
    const Vec3 ext = cb.extent();
    const int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);
    const Index mid = item.begin + count / 2;
    std::nth_element(order.begin() + item.begin, order.begin() + mid, order.begin() + item.end,
                     [&](Index a, Index b) { return centroids[a][axis] < centroids[b][axis]; });
    stack.push_back({mid, item.end, node, false});
    stack.push_back({item.begin, mid, node, true});
  }

  // Leaves were emitted left to right over `order`, so their face ranges are already
  // contiguous in it. Reorder the mesh to match and record each leaf's vertex range.
  const std::vector<Index> firstVertex = mesh.reorder(order, map);
  // reorder() lays half-edges out face by face, so faceHe[f] is the first half-edge of face f.
  const Index nh = mesh.halfEdgeCount();
  for (BvhLeaf& leaf : leaves_) {
    leaf.vertBegin = firstVertex[leaf.faceBegin];
    leaf.vertEnd = firstVertex[leaf.faceEnd];
    leaf.heBegin = mesh.faceHe[leaf.faceBegin];
    leaf.heEnd = leaf.faceEnd < nf ? mesh.faceHe[leaf.faceEnd] : nh;
  }
  refit(mesh);
}

Aabb Bvh::leafBounds(const Mesh& mesh, const BvhLeaf& leaf) const {
  Aabb b;
  // Removed faces are skipped even in leaves dynamic topology never flagged: an edit it did not
  // hear about (which consolidate() then repairs) must not send the walk off the arrays.
  for (Index f = leaf.faceBegin; f < leaf.faceEnd; ++f) {
    if (mesh.faceHe[f] == kInvalid) continue;
    mesh.forEachFaceVertex(f, [&](Index v) { b.expand(mesh.positions[v]); });
  }
  if (leaf.flags == 0) return b;
  // Leaves touched by dynamic topology also include owned vertices, which may no longer be used
  // by any face of the leaf.
  for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v)
    if (mesh.vertHe[v] != kInvalid) b.expand(mesh.positions[v]);
  return b;
}

void Bvh::refit(const Mesh& mesh) {
  parallelFor(0, leaves_.size(), 4, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) leaves_[i].bounds = leafBounds(mesh, leaves_[i]);
  });
  // Children follow parents in preorder, so a reverse sweep sees children first.
  for (Index n = static_cast<Index>(nodes_.size()) - 1; n >= 0; --n) {
    BvhNode& node = nodes_[n];
    if (node.isLeaf()) {
      node.bounds = leaves_[node.leaf].bounds;
    } else {
      node.bounds = nodes_[node.left].bounds;
      node.bounds.expand(nodes_[node.right].bounds);
    }
  }
}

void Bvh::refitLeaves(const Mesh& mesh, std::span<const Index> leaves) {
  parallelFor(0, leaves.size(), 4, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      BvhLeaf& leaf = leaves_[leaves[i]];
      leaf.bounds = leafBounds(mesh, leaf);
      const Index node = leafNode_[leaves[i]];
      if (node != kInvalid) nodes_[node].bounds = leaf.bounds;
    }
  });
  // Recompute every ancestor (bounds can shrink, so no early exit). Depth is about
  // log2(leaf count), so this stays cheap. Tail and empty leaves have no node.
  for (Index leaf : leaves) {
    if (leafNode_[leaf] == kInvalid) continue;
    for (Index n = parent_[leafNode_[leaf]]; n != kInvalid; n = parent_[n]) {
      BvhNode& node = nodes_[n];
      node.bounds = nodes_[node.left].bounds;
      node.bounds.expand(nodes_[node.right].bounds);
    }
  }
}

Index Bvh::leafOfVertex(Index v) const {
  auto it = std::upper_bound(leaves_.begin(), leaves_.end(), v,
                             [](Index value, const BvhLeaf& leaf) { return value < leaf.vertEnd; });
  if (it == leaves_.end() || v < it->vertBegin) return kInvalid;
  return static_cast<Index>(it - leaves_.begin());
}

bool Bvh::closestPoint(const Mesh& mesh, const Vec3& p, float maxDist, ClosestHit& out, bool visibleOnly) const {
  float best = maxDist * maxDist;
  bool found = false;
  const std::int32_t* sets = visibleOnly && !mesh.faceSets.empty() ? mesh.faceSets.data() : nullptr;
  auto visitLeaf = [&](const BvhLeaf& leaf) {
    const bool mayHaveDead = (leaf.flags & kLeafMayHaveDead) != 0;
    for (Index f = leaf.faceBegin; f < leaf.faceEnd; ++f) {
      const Index h0 = mesh.faceHe[f];
      if ((mayHaveDead && h0 == kInvalid) || (sets && sets[f] < 0)) continue;
      const Vec3& a = mesh.positions[mesh.heVert[h0]];
      Index h = mesh.heNext[h0];
      for (Index hn = mesh.heNext[h]; hn != h0; h = hn, hn = mesh.heNext[hn]) {
        const Vec3& b = mesh.positions[mesh.heVert[h]];
        const Vec3& c = mesh.positions[mesh.heVert[hn]];
        const Vec3 q = closestPointOnTriangle(p, a, b, c);
        const float d = glm::dot(q - p, q - p);
        if (d <= best) {
          best = d;
          found = true;
          out.position = q;
          out.face = f;
          out.distSq = d;
          out.corners[0] = mesh.heVert[h0];
          out.corners[1] = mesh.heVert[h];
          out.corners[2] = mesh.heVert[hn];
          const Vec3 n = glm::cross(b - a, c - a);
          const float len = glm::length(n);
          out.faceNormal = len > 0.0f ? n / len : Vec3{0.0f};
        }
      }
    }
  };
  forEachTailLeaf([&](Index l) {
    if (distanceSq(leaves_[l].bounds, p) <= best) visitLeaf(leaves_[l]);
  });
  if (nodes_.empty()) return found;
  Index stack[128];
  int sp = 0;
  stack[sp++] = 0;
  while (sp > 0) {
    const BvhNode& node = nodes_[stack[--sp]];
    if (distanceSq(node.bounds, p) > best) continue;
    if (node.isLeaf()) {
      visitLeaf(leaves_[node.leaf]);
      continue;
    }
    // Visit the nearer child first so `best` shrinks early.
    const float dl = distanceSq(nodes_[node.left].bounds, p);
    const float dr = distanceSq(nodes_[node.right].bounds, p);
    if (sp + 2 > 128) continue;
    if (dl <= dr) {
      stack[sp++] = node.right;
      stack[sp++] = node.left;
    } else {
      stack[sp++] = node.left;
      stack[sp++] = node.right;
    }
  }
  return found;
}

std::size_t Bvh::memoryBytes() const {
  return nodes_.size() * sizeof(BvhNode) + leaves_.size() * sizeof(BvhLeaf) +
         (leafNode_.size() + parent_.size()) * sizeof(Index);
}

Index Bvh::leafOfHalfEdge(Index h) const {
  auto it = std::upper_bound(leaves_.begin(), leaves_.end(), h,
                             [](Index value, const BvhLeaf& leaf) { return value < leaf.heEnd; });
  if (it == leaves_.end() || h < it->heBegin) return kInvalid;
  return static_cast<Index>(it - leaves_.begin());
}

Index Bvh::leafOfFace(Index f) const {
  auto it = std::upper_bound(leaves_.begin(), leaves_.end(), f,
                             [](Index value, const BvhLeaf& leaf) { return value < leaf.faceEnd; });
  if (it == leaves_.end() || f < it->faceBegin) return kInvalid;
  return static_cast<Index>(it - leaves_.begin());
}

bool Bvh::raycast(const Mesh& mesh, const Ray& ray, RayHit& hit, float tMax, bool visibleOnly) const {
  if (nodes_.empty() && !dynamic_) return false;
  // Avoid 0 * inf = NaN in the slab test for axis-aligned rays starting on a box plane.
  Vec3 safeDir = ray.dir;
  for (int a = 0; a < 3; ++a)
    if (std::abs(safeDir[a]) < 1e-20f) safeDir[a] = std::copysign(1e-20f, safeDir[a]);
  const Vec3 invDir = 1.0f / safeDir;
  float best = tMax;
  Index bestFace = kInvalid, bestLeaf = kInvalid;
  Index bestA = 0, bestB = 0, bestC = 0;
  float bestU = 0.0f, bestV = 0.0f;

  struct Entry {
    Index node;
    float tEnter;
  };
  const std::int32_t* sets = visibleOnly && !mesh.faceSets.empty() ? mesh.faceSets.data() : nullptr;
  auto visitLeaf = [&](Index leafIndex) {
    const BvhLeaf& leaf = leaves_[leafIndex];
    const bool mayHaveDead = (leaf.flags & kLeafMayHaveDead) != 0;
    for (Index f = leaf.faceBegin; f < leaf.faceEnd; ++f) {
      const Index h0 = mesh.faceHe[f];
      if ((mayHaveDead && h0 == kInvalid) || (sets && sets[f] < 0)) continue;
      const Index a = mesh.heVert[h0];
      Index h = mesh.heNext[h0];
      Index b = mesh.heVert[h];
      for (h = mesh.heNext[h]; h != h0; h = mesh.heNext[h]) {
        const Index c = mesh.heVert[h];
        float t, u, v;
        if (rayTriangle(ray, mesh.positions[a], mesh.positions[b], mesh.positions[c], t, u, v) && t < best) {
          best = t;
          bestFace = f;
          bestLeaf = leafIndex;
          bestA = a, bestB = b, bestC = c;
          bestU = u, bestV = v;
        }
        b = c;
      }
    }
  };
  // Tail leaves first: they are few and can lie outside the root box.
  forEachTailLeaf([&](Index l) {
    if (std::isfinite(rayAabb(ray.origin, invDir, leaves_[l].bounds, best))) visitLeaf(l);
  });

  Entry stack[128];
  int sp = 0;
  const float rootT = nodes_.empty() ? std::numeric_limits<float>::infinity()
                                     : rayAabb(ray.origin, invDir, nodes_[0].bounds, best);
  if (std::isfinite(rootT)) stack[sp++] = {0, rootT};

  while (sp > 0) {
    const Entry e = stack[--sp];
    if (e.tEnter > best) continue;
    const BvhNode& node = nodes_[e.node];
    if (node.isLeaf()) {
      visitLeaf(node.leaf);
      continue;
    }
    const float tl = rayAabb(ray.origin, invDir, nodes_[node.left].bounds, best);
    const float tr = rayAabb(ray.origin, invDir, nodes_[node.right].bounds, best);
    // Push the farther child first so the nearer one is processed next.
    if (tl <= tr) {
      if (std::isfinite(tr) && sp < 128) stack[sp++] = {node.right, tr};
      if (std::isfinite(tl) && sp < 128) stack[sp++] = {node.left, tl};
    } else {
      if (std::isfinite(tl) && sp < 128) stack[sp++] = {node.left, tl};
      if (std::isfinite(tr) && sp < 128) stack[sp++] = {node.right, tr};
    }
  }

  if (bestFace == kInvalid) return false;
  const Vec3& pa = mesh.positions[bestA];
  const Vec3& pb = mesh.positions[bestB];
  const Vec3& pc = mesh.positions[bestC];
  hit.t = best;
  hit.face = bestFace;
  hit.leaf = bestLeaf;
  hit.position = ray.origin + ray.dir * best;
  hit.faceNormal = glm::normalize(glm::cross(pb - pa, pc - pa));
  if (mesh.normals.size() == mesh.positions.size()) {
    const Vec3 n = (1.0f - bestU - bestV) * mesh.normals[bestA] + bestU * mesh.normals[bestB] + bestV * mesh.normals[bestC];
    const float len = glm::length(n);
    hit.smoothNormal = len > 1e-12f ? n / len : hit.faceNormal;
  } else {
    hit.smoothNormal = hit.faceNormal;
  }
  return true;
}

void Bvh::querySphere(const Vec3& center, float radius, std::vector<Index>& outLeaves) const {
  const float r2 = radius * radius;
  auto overlaps = [&](const Aabb& b) {
    const Vec3 d = glm::max(glm::max(b.min - center, center - b.max), Vec3{0.0f});
    return glm::dot(d, d) <= r2;
  };
  if (nodes_.empty()) {
    forEachTailLeaf([&](Index l) {
      if (overlaps(leaves_[l].bounds)) outLeaves.push_back(l);
    });
    return;
  }
  Index stack[128];
  int sp = 0;
  stack[sp++] = 0;
  while (sp > 0) {
    const BvhNode& node = nodes_[stack[--sp]];
    if (!overlaps(node.bounds)) continue;
    if (node.isLeaf()) {
      outLeaves.push_back(node.leaf);
    } else if (sp + 2 <= 128) {
      stack[sp++] = node.right;
      stack[sp++] = node.left;
    }
  }
  forEachTailLeaf([&](Index l) {
    if (overlaps(leaves_[l].bounds)) outLeaves.push_back(l);
  });
}

// --- Dynamic topology ----------------------------------------------------------------------

void Bvh::beginDynamic(const Mesh& mesh) {
  dynamic_ = true;
  firstTailLeaf_ = static_cast<Index>(leaves_.size());
  tailStartFace_ = mesh.faceCount();
  tailStartVert_ = mesh.vertexCount();
  tailStartHe_ = mesh.halfEdgeCount();
}

void Bvh::growTail(const Mesh& mesh) {
  if (!dynamic_) return;
  const Index nf = mesh.faceCount(), nv = mesh.vertexCount(), nh = mesh.halfEdgeCount();
  const bool haveOpen = static_cast<Index>(leaves_.size()) > firstTailLeaf_;
  if (haveOpen && leaves_.back().faceEnd - leaves_.back().faceBegin < kTailLeafFaces) {
    BvhLeaf& open = leaves_.back();
    open.faceEnd = nf;
    open.vertEnd = nv;
    open.heEnd = nh;
    return;
  }
  const Index fb = haveOpen ? leaves_.back().faceEnd : tailStartFace_;
  const Index vb = haveOpen ? leaves_.back().vertEnd : tailStartVert_;
  const Index hb = haveOpen ? leaves_.back().heEnd : tailStartHe_;
  if (fb == nf && vb == nv && hb == nh) return;  // Nothing appended since.
  BvhLeaf leaf;
  leaf.faceBegin = fb;
  leaf.faceEnd = nf;
  leaf.vertBegin = vb;
  leaf.vertEnd = nv;
  leaf.heBegin = hb;
  leaf.heEnd = nh;
  leaf.flags = kLeafMayHaveDead;
  leaves_.push_back(leaf);
  leafNode_.push_back(kInvalid);
}

void Bvh::setLeaves(std::vector<BvhLeaf> leaves) {
  leaves_ = std::move(leaves);
  dynamic_ = false;
  rebuildTree();
}

void Bvh::rebuildTree() {
  nodes_.clear();
  parent_.clear();
  leafNode_.assign(leaves_.size(), kInvalid);
  std::vector<Index> order;
  order.reserve(leaves_.size());
  for (Index l = 0; l < static_cast<Index>(leaves_.size()); ++l)
    if (!leaves_[l].empty()) order.push_back(l);
  if (order.empty()) return;
  std::vector<Vec3> centers(leaves_.size());
  for (Index l : order) centers[l] = leaves_[l].bounds.valid() ? leaves_[l].bounds.center() : Vec3{0.0f};

  std::vector<BuildItem> stack{{0, static_cast<Index>(order.size()), kInvalid, true}};
  while (!stack.empty()) {
    const BuildItem item = stack.back();
    stack.pop_back();
    const Index node = static_cast<Index>(nodes_.size());
    nodes_.push_back({});
    parent_.push_back(item.parent);
    if (item.parent != kInvalid) (item.isLeft ? nodes_[item.parent].left : nodes_[item.parent].right) = node;
    const Index count = item.end - item.begin;
    if (count == 1) {
      nodes_[node].leaf = order[item.begin];
      leafNode_[order[item.begin]] = node;
      continue;
    }
    Aabb cb;
    for (Index i = item.begin; i < item.end; ++i) cb.expand(centers[order[i]]);
    const Vec3 ext = cb.extent();
    const int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);
    const Index mid = item.begin + count / 2;
    std::nth_element(order.begin() + item.begin, order.begin() + mid, order.begin() + item.end,
                     [&](Index a, Index b) { return centers[a][axis] < centers[b][axis]; });
    stack.push_back({mid, item.end, node, false});
    stack.push_back({item.begin, mid, node, true});
  }
  for (Index n = static_cast<Index>(nodes_.size()) - 1; n >= 0; --n) {
    BvhNode& node = nodes_[n];
    if (node.isLeaf()) {
      node.bounds = leaves_[node.leaf].bounds;
    } else {
      node.bounds = nodes_[node.left].bounds;
      node.bounds.expand(nodes_[node.right].bounds);
    }
  }
}

std::vector<Index> partitionFaces(std::span<const Vec3> centroids, std::vector<Index>& order, Index maxLeafFaces) {
  maxLeafFaces = std::max<Index>(maxLeafFaces, 1);
  std::vector<Index> bounds{0};
  if (order.empty()) return bounds;
  // Depth first, left half first, so groups come out in order.
  std::vector<std::pair<Index, Index>> stack{{0, static_cast<Index>(order.size())}};
  while (!stack.empty()) {
    const auto [begin, end] = stack.back();
    stack.pop_back();
    const Index count = end - begin;
    if (count <= maxLeafFaces) {
      bounds.push_back(end);
      continue;
    }
    Aabb cb;
    for (Index i = begin; i < end; ++i) cb.expand(centroids[order[i]]);
    const Vec3 ext = cb.extent();
    const int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);
    const Index mid = begin + count / 2;
    std::nth_element(order.begin() + begin, order.begin() + mid, order.begin() + end,
                     [&](Index a, Index b) { return centroids[a][axis] < centroids[b][axis]; });
    stack.push_back({mid, end});
    stack.push_back({begin, mid});
  }
  return bounds;
}

ValidationResult validateLayout(const Mesh& m, const Bvh& bvh) {
  auto fail = [](std::string msg) { return ValidationResult{false, std::move(msg)}; };
  if (bvh.dynamic()) return fail("BVH is in dynamic mode");
  const ValidationResult mv = validate(m);
  if (!mv.ok) return mv;
  const auto leaves = bvh.leaves();
  Index f = 0, h = 0, v = 0;
  for (Index l = 0; l < static_cast<Index>(leaves.size()); ++l) {
    const BvhLeaf& leaf = leaves[l];
    const std::string at = " at leaf " + std::to_string(l);
    if (leaf.faceBegin != f || leaf.heBegin != h || leaf.vertBegin != v) return fail("ranges not contiguous" + at);
    if (leaf.faceEnd < leaf.faceBegin || leaf.heEnd < leaf.heBegin || leaf.vertEnd < leaf.vertBegin)
      return fail("negative range" + at);
    if ((leaf.flags & kLeafMayHaveDead) != 0) return fail("dynamic flag left set" + at);
    for (Index face = leaf.faceBegin; face < leaf.faceEnd; ++face) {
      if (m.faceHe[face] != h) return fail("half-edges not laid out face by face" + at);
      const Index size = m.faceSize(face);
      for (Index k = 0; k < size; ++k) {
        if (m.heFace[h + k] != face) return fail("face loop not consecutive" + at);
        if (m.heNext[h + k] != (k + 1 < size ? h + k + 1 : h)) return fail("face loop not consecutive" + at);
      }
      h += size;
    }
    if (h != leaf.heEnd) return fail("half-edge range mismatch" + at);
    f = leaf.faceEnd;
    v = leaf.vertEnd;
    // Owned vertices are used by the leaf's own faces, unless the leaf owns orphans.
    if ((leaf.flags & kLeafOwnsOrphans) == 0) {
      std::vector<char> used(static_cast<std::size_t>(leaf.vertEnd - leaf.vertBegin), 0);
      for (Index e = leaf.heBegin; e < leaf.heEnd; ++e) {
        const Index u = m.heVert[e];
        if (u >= leaf.vertBegin && u < leaf.vertEnd) used[static_cast<std::size_t>(u - leaf.vertBegin)] = 1;
      }
      if (std::find(used.begin(), used.end(), 0) != used.end()) return fail("owned vertex unused by the leaf" + at);
    }
    for (Index e = leaf.heBegin; e < leaf.heEnd; ++e) {
      const Vec3& p = m.positions[m.heVert[e]];
      if (glm::any(glm::lessThan(p, leaf.bounds.min)) || glm::any(glm::greaterThan(p, leaf.bounds.max)))
        return fail("leaf bounds miss a vertex" + at);
    }
    for (Index u = leaf.vertBegin; u < leaf.vertEnd; ++u) {
      if (m.vertHe[u] == kInvalid) return fail("owned vertex without faces" + at);
      const Vec3& p = m.positions[u];
      if (glm::any(glm::lessThan(p, leaf.bounds.min)) || glm::any(glm::greaterThan(p, leaf.bounds.max)))
        return fail("leaf bounds miss an owned vertex" + at);
    }
  }
  if (f != m.faceCount() || h != m.halfEdgeCount()) return fail("leaves do not cover every face");
  for (Index u = v; u < m.vertexCount(); ++u)
    if (m.vertHe[u] != kInvalid) return fail("vertex with faces after the leaves: " + std::to_string(u));

  // Tree: every non-empty leaf reached exactly once, children after parents, bounds nested.
  const auto nodes = bvh.nodes();
  std::vector<int> reached(leaves.size(), 0);
  for (Index n = 0; n < static_cast<Index>(nodes.size()); ++n) {
    const BvhNode& node = nodes[n];
    auto contains = [](const Aabb& outer, const Aabb& inner) {
      return !inner.valid() || (glm::all(glm::lessThanEqual(outer.min, inner.min)) &&
                                glm::all(glm::greaterThanEqual(outer.max, inner.max)));
    };
    if (node.isLeaf()) {
      if (node.leaf < 0 || node.leaf >= static_cast<Index>(leaves.size())) return fail("node leaf out of range");
      ++reached[node.leaf];
      if (!contains(node.bounds, leaves[node.leaf].bounds)) return fail("leaf node bounds too small");
    } else {
      if (node.left <= n || node.right <= n) return fail("child before parent at node " + std::to_string(n));
      if (!contains(node.bounds, nodes[node.left].bounds) || !contains(node.bounds, nodes[node.right].bounds))
        return fail("node bounds too small at " + std::to_string(n));
    }
  }
  for (Index l = 0; l < static_cast<Index>(leaves.size()); ++l) {
    if (reached[l] != (leaves[l].empty() ? 0 : 1)) return fail("leaf reached " + std::to_string(reached[l]) + " times: " + std::to_string(l));
  }
  return {};
}

}  // namespace plegl
