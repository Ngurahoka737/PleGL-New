#include "multires/Subdivide.h"

#include <algorithm>
#include <numeric>

#include "core/Parallel.h"

namespace plegl {

std::vector<VertexRule> classifyVertices(const Mesh& m) {
  const Index nv = m.vertexCount();
  std::vector<Index> outCount(static_cast<std::size_t>(nv), 0);
  for (Index h = 0; h < m.halfEdgeCount(); ++h) ++outCount[m.heVert[h]];
  std::vector<VertexRule> rule(static_cast<std::size_t>(nv), VertexRule::Smooth);
  parallelFor(0, static_cast<std::size_t>(nv), 4096, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const Index v = static_cast<Index>(i);
      if (m.vertHe[v] == kInvalid) {
        rule[i] = VertexRule::Pinned;
        continue;
      }
      Index faces = 0;
      bool open = false;
      m.forEachOutgoing(v, [&](Index h) {
        ++faces;
        open |= m.heTwin[h] == kInvalid;
      });
      if (faces != outCount[i]) {
        rule[i] = VertexRule::Pinned;
      } else if (open) {
        rule[i] = faces >= 2 ? VertexRule::Boundary : VertexRule::Pinned;
      } else {
        rule[i] = faces >= 3 ? VertexRule::Smooth : VertexRule::Pinned;
      }
    }
  });
  return rule;
}

std::optional<SubdivisionResult> subdivide(const Mesh& P, const CanonicalMap& parentCanon,
                                           std::span<const VertexRule> parentRule, const SubdivideOptions& options,
                                           std::string* error) {
  auto fail = [&](const char* message) -> std::optional<SubdivisionResult> {
    if (error) *error = message;
    return std::nullopt;
  };
  const Index nV = P.vertexCount();
  const Index nF = P.faceCount();
  const Index nH = P.halfEdgeCount();
  if (nF == 0) return fail("The mesh is empty.");
  if (nH > options.maxFaces) return fail("The next level would have too many faces.");

  // Edges: the half-edge with the smaller live index owns its edge (open edges own themselves).
  std::vector<Index> edgeOf(static_cast<std::size_t>(nH));
  std::vector<Index> edgeOwner;
  edgeOwner.reserve(static_cast<std::size_t>(nH));
  for (Index h = 0; h < nH; ++h) {
    const Index t = P.heTwin[h];
    if (t == kInvalid || h < t) {
      edgeOf[h] = static_cast<Index>(edgeOwner.size());
      edgeOwner.push_back(h);
    }
  }
  const Index nE = static_cast<Index>(edgeOwner.size());
  for (Index h = 0; h < nH; ++h) {
    const Index t = P.heTwin[h];
    if (t != kInvalid && t < h) edgeOf[h] = edgeOf[t];
  }
  std::vector<Index> prevOf(static_cast<std::size_t>(nH));
  for (Index h = 0; h < nH; ++h) prevOf[P.heNext[h]] = h;

  const std::int64_t childVerts = std::int64_t{nV} + nE + nF;
  if (childVerts > INT32_MAX / 4) return fail("The next level would have too many vertices.");
  const Index cV = static_cast<Index>(childVerts);
  const Index cF = nH;
  const Index cH = 4 * nH;
  const Index edgeBase = nV, faceBase = nV + nE;

  SubdivisionResult out;
  Mesh& C = out.mesh;
  C.positions.resize(static_cast<std::size_t>(cV));
  C.vertHe.resize(static_cast<std::size_t>(cV));
  C.faceHe.resize(static_cast<std::size_t>(cF));
  C.heNext.resize(static_cast<std::size_t>(cH));
  C.heTwin.resize(static_cast<std::size_t>(cH));
  C.heVert.resize(static_cast<std::size_t>(cH));
  C.heFace.resize(static_cast<std::size_t>(cH));

  // Topology. Child quad q(h) = h owns half-edges 4h..4h+3, starting at the vertex child.
  parallelFor(0, static_cast<std::size_t>(nH), 8192, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const Index h = static_cast<Index>(i);
      const Index p = prevOf[h];
      const Index base = 4 * h;
      C.heVert[base + 0] = P.heVert[h];
      C.heVert[base + 1] = edgeBase + edgeOf[h];
      C.heVert[base + 2] = faceBase + P.heFace[h];
      C.heVert[base + 3] = edgeBase + edgeOf[p];
      for (Index k = 0; k < 4; ++k) {
        C.heNext[base + k] = base + (k + 1) % 4;
        C.heFace[base + k] = h;
      }
      const Index t = P.heTwin[h];
      const Index tp = P.heTwin[p];
      C.heTwin[base + 0] = t == kInvalid ? kInvalid : 4 * P.heNext[t] + 3;
      C.heTwin[base + 1] = 4 * P.heNext[h] + 2;
      C.heTwin[base + 2] = 4 * p + 1;
      C.heTwin[base + 3] = tp == kInvalid ? kInvalid : 4 * tp;
      C.faceHe[h] = base;
    }
  });
  // Canonical numbering of the provisional child elements.
  const CanonicalMap identity = parentCanon.vert.empty() ? identityCanonicalMap(P) : CanonicalMap{};
  const CanonicalMap& pc = parentCanon.vert.empty() ? identity : parentCanon;
  const std::vector<Index> canonHe = canonicalHalfEdges(P, pc);
  // Rank of every canonical owner (the half-edge with the smaller canonical id) in canonical order.
  std::vector<Index> rank(static_cast<std::size_t>(nH), 0);
  for (Index h = 0; h < nH; ++h) {
    const Index t = P.heTwin[h];
    if (t == kInvalid || canonHe[h] < canonHe[t]) rank[canonHe[h]] = 1;
  }
  {
    Index running = 0;
    for (Index& r : rank) {
      const Index flag = r;
      r = running;
      running += flag;
    }
  }
  auto canonicalEdge = [&](Index h) {
    const Index t = P.heTwin[h];
    const Index owner = (t == kInvalid || canonHe[h] < canonHe[t]) ? h : t;
    return rank[canonHe[owner]];
  };
  // Every fan starts at a half-edge picked by canonical rules (the parent's vertHe, the canonical
  // owner of an edge, the start corner of a face), so sums around a vertex run in the same order
  // for any live order of the same level.
  for (Index v = 0; v < nV; ++v) C.vertHe[v] = P.vertHe[v] == kInvalid ? kInvalid : 4 * P.vertHe[v];
  for (Index e = 0; e < nE; ++e) {
    const Index h = edgeOwner[e], t = P.heTwin[h];
    C.vertHe[edgeBase + e] = 4 * (t == kInvalid || canonHe[h] < canonHe[t] ? h : t) + 1;
  }
  for (Index f = 0; f < nF; ++f) C.vertHe[faceBase + f] = 4 * P.faceHe[f] + 2;

  // Positions: face points first, then edge and vertex points read them.
  std::vector<Vec3> facePoint(static_cast<std::size_t>(nF));
  const auto X = [&](Index v) -> const Vec3& { return P.positions[v]; };
  const auto FP = [&](Index f) -> const Vec3& { return facePoint[f]; };
  parallelFor(0, static_cast<std::size_t>(nF), 8192, [&](std::size_t b, std::size_t e) {
    for (std::size_t f = b; f < e; ++f) {
      facePoint[f] = ccFacePoint(P, static_cast<Index>(f), X);
      C.positions[faceBase + f] = facePoint[f];
    }
  });
  parallelFor(0, static_cast<std::size_t>(nE), 8192, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) C.positions[edgeBase + i] = ccEdgePoint(P, edgeOwner[i], X, FP);
  });
  parallelFor(0, static_cast<std::size_t>(nV), 4096, [&](std::size_t b, std::size_t e) {
    for (std::size_t v = b; v < e; ++v)
      C.positions[v] = ccVertexPoint(P, static_cast<Index>(v), parentRule[v], X, FP);
  });

  // Mask, face sets and rules.
  if (!P.mask.empty()) {
    C.mask.resize(static_cast<std::size_t>(cV));
    std::copy(P.mask.begin(), P.mask.end(), C.mask.begin());
    for (Index e = 0; e < nE; ++e) {
      const Index h = edgeOwner[e];
      C.mask[edgeBase + e] = (P.mask[P.heVert[h]] + P.mask[P.heTarget(h)]) * 0.5f;
    }
    for (Index f = 0; f < nF; ++f) {
      float sum = 0.0f;
      Index n = 0;
      P.forEachFaceVertex(f, [&](Index v) {
        sum += P.mask[v];
        ++n;
      });
      C.mask[faceBase + f] = sum / static_cast<float>(n);
    }
  }
  if (!P.faceSets.empty()) {
    C.faceSets.resize(static_cast<std::size_t>(cF));
    for (Index h = 0; h < nH; ++h) C.faceSets[h] = P.faceSets[P.heFace[h]];
  }
  std::vector<VertexRule> rule(static_cast<std::size_t>(cV), VertexRule::Smooth);
  std::copy(parentRule.begin(), parentRule.end(), rule.begin());
  for (Index e = 0; e < nE; ++e)
    rule[edgeBase + e] = P.heTwin[edgeOwner[e]] == kInvalid ? VertexRule::Boundary : VertexRule::Smooth;
  C.computeNormals();

  std::vector<Index> provVert(static_cast<std::size_t>(cV)), provFace(static_cast<std::size_t>(cF));
  for (Index v = 0; v < nV; ++v) provVert[v] = pc.vert[v];
  for (Index e = 0; e < nE; ++e) provVert[edgeBase + e] = edgeBase + canonicalEdge(edgeOwner[e]);
  for (Index f = 0; f < nF; ++f) provVert[faceBase + f] = faceBase + pc.face[f];
  for (Index h = 0; h < nH; ++h) provFace[h] = canonHe[h];

  if (!options.buildBvh) {
    out.canon.vert = std::move(provVert);
    out.canon.face = std::move(provFace);
    SubdivisionLinks& L = out.links;
    L.vertexChild.resize(static_cast<std::size_t>(nV));
    std::iota(L.vertexChild.begin(), L.vertexChild.end(), 0);
    L.edgeChild.resize(static_cast<std::size_t>(nH));
    L.childFace.resize(static_cast<std::size_t>(nH));
    for (Index h = 0; h < nH; ++h) {
      L.edgeChild[h] = edgeBase + edgeOf[h];
      L.childFace[h] = h;
    }
    L.faceChild.resize(static_cast<std::size_t>(nF));
    for (Index f = 0; f < nF; ++f) L.faceChild[f] = faceBase + f;
    L.parent.resize(static_cast<std::size_t>(cV));
    for (Index v = 0; v < nV; ++v) L.parent[v] = makeParent(kParentVertex, v);
    for (Index e = 0; e < nE; ++e) L.parent[edgeBase + e] = makeParent(kParentEdge, edgeOwner[e]);
    for (Index f = 0; f < nF; ++f) L.parent[faceBase + f] = makeParent(kParentFace, f);
    L.parentHalfEdge.resize(static_cast<std::size_t>(cF));
    std::iota(L.parentHalfEdge.begin(), L.parentHalfEdge.end(), 0);
    out.rule = std::move(rule);
    return out;
  }

  ReorderMap map;
  Bvh::Params params;
  params.maxLeafFaces = options.maxLeafFaces;
  out.bvh.build(C, params, &map);

  std::vector<Index> vertNew(static_cast<std::size_t>(cV)), faceNew(static_cast<std::size_t>(cF));
  for (Index v = 0; v < cV; ++v) vertNew[map.vertOld[v]] = v;
  for (Index f = 0; f < cF; ++f) faceNew[map.faceOld[f]] = f;

  out.canon.vert.resize(static_cast<std::size_t>(cV));
  out.rule.resize(static_cast<std::size_t>(cV));
  SubdivisionLinks& L = out.links;
  L.parent.resize(static_cast<std::size_t>(cV));
  for (Index v = 0; v < cV; ++v) {
    const Index old = map.vertOld[v];
    out.canon.vert[v] = provVert[old];
    out.rule[v] = rule[old];
    if (old < edgeBase) {
      L.parent[v] = makeParent(kParentVertex, old);
    } else if (old < faceBase) {
      L.parent[v] = makeParent(kParentEdge, edgeOwner[old - edgeBase]);
    } else {
      L.parent[v] = makeParent(kParentFace, old - faceBase);
    }
  }
  out.canon.face.resize(static_cast<std::size_t>(cF));
  L.parentHalfEdge.resize(static_cast<std::size_t>(cF));
  for (Index f = 0; f < cF; ++f) {
    out.canon.face[f] = provFace[map.faceOld[f]];
    L.parentHalfEdge[f] = map.faceOld[f];
  }
  L.vertexChild.resize(static_cast<std::size_t>(nV));
  for (Index v = 0; v < nV; ++v) L.vertexChild[v] = vertNew[v];
  L.edgeChild.resize(static_cast<std::size_t>(nH));
  L.childFace.resize(static_cast<std::size_t>(nH));
  for (Index h = 0; h < nH; ++h) {
    L.edgeChild[h] = vertNew[edgeBase + edgeOf[h]];
    L.childFace[h] = faceNew[h];
  }
  L.faceChild.resize(static_cast<std::size_t>(nF));
  for (Index f = 0; f < nF; ++f) L.faceChild[f] = vertNew[faceBase + f];
  return out;
}

}  // namespace plegl
