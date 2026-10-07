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

void Bvh::build(Mesh& mesh, const Params& params) {
  nodes_.clear();
  leaves_.clear();
  leafNode_.clear();
  parent_.clear();
  const Index nf = mesh.faceCount();
  if (nf == 0) return;
  const Index maxLeaf = std::max(params.maxLeafFaces, 1);

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
  const std::vector<Index> firstVertex = mesh.reorder(order);
  for (BvhLeaf& leaf : leaves_) {
    leaf.vertBegin = firstVertex[leaf.faceBegin];
    leaf.vertEnd = firstVertex[leaf.faceEnd];
  }
  refit(mesh);
}

Aabb Bvh::leafBounds(const Mesh& mesh, const BvhLeaf& leaf) const {
  Aabb b;
  for (Index f = leaf.faceBegin; f < leaf.faceEnd; ++f) {
    mesh.forEachFaceVertex(f, [&](Index v) { b.expand(mesh.positions[v]); });
  }
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
      nodes_[leafNode_[leaves[i]]].bounds = leaf.bounds;
    }
  });
  // Recompute every ancestor (bounds can shrink, so no early exit). Depth is about
  // log2(leaf count), so this stays cheap.
  for (Index leaf : leaves) {
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

bool Bvh::closestPoint(const Mesh& mesh, const Vec3& p, float maxDist, ClosestHit& out) const {
  if (nodes_.empty()) return false;
  float best = maxDist * maxDist;
  bool found = false;
  Index stack[128];
  int sp = 0;
  stack[sp++] = 0;
  while (sp > 0) {
    const BvhNode& node = nodes_[stack[--sp]];
    if (distanceSq(node.bounds, p) > best) continue;
    if (node.isLeaf()) {
      const BvhLeaf& leaf = leaves_[node.leaf];
      for (Index f = leaf.faceBegin; f < leaf.faceEnd; ++f) {
        const Index h0 = mesh.faceHe[f];
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

Index Bvh::leafOfFace(Index f) const {
  auto it = std::upper_bound(leaves_.begin(), leaves_.end(), f,
                             [](Index value, const BvhLeaf& leaf) { return value < leaf.faceEnd; });
  if (it == leaves_.end() || f < it->faceBegin) return kInvalid;
  return static_cast<Index>(it - leaves_.begin());
}

bool Bvh::raycast(const Mesh& mesh, const Ray& ray, RayHit& hit, float tMax) const {
  if (nodes_.empty()) return false;
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
  Entry stack[128];
  int sp = 0;
  const float rootT = rayAabb(ray.origin, invDir, nodes_[0].bounds, best);
  if (!std::isfinite(rootT)) return false;
  stack[sp++] = {0, rootT};

  while (sp > 0) {
    const Entry e = stack[--sp];
    if (e.tEnter > best) continue;
    const BvhNode& node = nodes_[e.node];
    if (node.isLeaf()) {
      const BvhLeaf& leaf = leaves_[node.leaf];
      for (Index f = leaf.faceBegin; f < leaf.faceEnd; ++f) {
        const Index h0 = mesh.faceHe[f];
        const Index a = mesh.heVert[h0];
        Index h = mesh.heNext[h0];
        Index b = mesh.heVert[h];
        for (h = mesh.heNext[h]; h != h0; h = mesh.heNext[h]) {
          const Index c = mesh.heVert[h];
          float t, u, v;
          if (rayTriangle(ray, mesh.positions[a], mesh.positions[b], mesh.positions[c], t, u, v) && t < best) {
            best = t;
            bestFace = f;
            bestLeaf = node.leaf;
            bestA = a, bestB = b, bestC = c;
            bestU = u, bestV = v;
          }
          b = c;
        }
      }
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
  if (nodes_.empty()) return;
  const float r2 = radius * radius;
  auto overlaps = [&](const Aabb& b) {
    const Vec3 d = glm::max(glm::max(b.min - center, center - b.max), Vec3{0.0f});
    return glm::dot(d, d) <= r2;
  };
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
}

}  // namespace plegl
