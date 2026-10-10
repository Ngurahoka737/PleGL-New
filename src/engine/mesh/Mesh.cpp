#include "mesh/Mesh.h"

#include <algorithm>
#include <cmath>

#include "core/Parallel.h"

namespace plegl {

Index Mesh::hePrev(Index h) const {
  Index p = h;
  while (heNext[p] != h) p = heNext[p];
  return p;
}

Index Mesh::faceSize(Index f) const {
  Index n = 0;
  const Index start = faceHe[f];
  Index h = start;
  do {
    ++n;
    h = heNext[h];
  } while (h != start);
  return n;
}

bool Mesh::isBoundaryVertex(Index v) const {
  bool boundary = false;
  forEachOutgoing(v, [&](Index h) {
    if (heTwin[h] == kInvalid || heTwin[hePrev(h)] == kInvalid) boundary = true;
  });
  return boundary;
}

int Mesh::valence(Index v) const {
  int outgoing = 0;
  bool boundary = false;
  forEachOutgoing(v, [&](Index h) {
    ++outgoing;
    if (heTwin[h] == kInvalid) boundary = true;
  });
  // An open fan has one more edge than faces: the incoming boundary edge.
  return boundary ? outgoing + 1 : outgoing;
}

Vec3 Mesh::faceAreaNormal(Index f) const {
  Vec3 n{0.0f};
  const Index start = faceHe[f];
  Index h = start;
  do {
    const Vec3& a = positions[heVert[h]];
    const Vec3& b = positions[heVert[heNext[h]]];
    n.x += (a.y - b.y) * (a.z + b.z);
    n.y += (a.z - b.z) * (a.x + b.x);
    n.z += (a.x - b.x) * (a.y + b.y);
    h = heNext[h];
  } while (h != start);
  return n;
}

Vec3 Mesh::faceCentroid(Index f) const {
  Vec3 c{0.0f};
  Index n = 0;
  forEachFaceVertex(f, [&](Index v) {
    c += positions[v];
    ++n;
  });
  return c / static_cast<float>(n);
}

Aabb Mesh::bounds() const {
  Aabb b;
  for (const Vec3& p : positions) b.expand(p);
  return b;
}

Vec3 Mesh::vertexNormal(Index v) const {
  Vec3 n{0.0f};
  forEachOutgoing(v, [&](Index h) { n += faceAreaNormal(heFace[h]); });
  return safeNormalize(n);
}

void Mesh::computeNormals() {
  normals.resize(positions.size());
  // Same sums in the same order as vertexNormal(), with each face normal computed once.
  std::vector<Vec3> faceN(faceHe.size());
  parallelFor(0, faceHe.size(), 4096, [&](std::size_t b, std::size_t e) {
    for (std::size_t f = b; f < e; ++f) faceN[f] = faceAreaNormal(static_cast<Index>(f));
  });
  parallelFor(0, positions.size(), 4096, [&](std::size_t b, std::size_t e) {
    for (std::size_t v = b; v < e; ++v) {
      Vec3 n{0.0f};
      forEachOutgoing(static_cast<Index>(v), [&](Index h) { n += faceN[heFace[h]]; });
      normals[v] = safeNormalize(n);
    }
  });
}

void Mesh::computeNormals(Index vertBegin, Index vertEnd) {
  normals.resize(positions.size());
  for (Index v = vertBegin; v < vertEnd; ++v) normals[v] = vertexNormal(v);
}

Index Mesh::edgeCount() const {
  Index n = 0;
  for (Index h = 0; h < halfEdgeCount(); ++h) {
    if (heTwin[h] == kInvalid || h < heTwin[h]) ++n;
  }
  return n;
}

std::vector<Index> Mesh::reorder(std::span<const Index> faceOrder, ReorderMap* map) {
  const Index nf = faceCount();
  const Index nv = vertexCount();
  const Index nh = halfEdgeCount();

  std::vector<Index> vertMap(nv, kInvalid);  // old -> new
  std::vector<Index> heMap(nh, kInvalid);    // old -> new
  std::vector<Index> heOld(nh);              // new -> old
  std::vector<Index> vertOld;                // new -> old
  vertOld.reserve(nv);
  std::vector<Index> firstVertex(nf + 1);

  Index nextHe = 0;
  for (Index nfIdx = 0; nfIdx < nf; ++nfIdx) {
    firstVertex[nfIdx] = static_cast<Index>(vertOld.size());
    const Index oldF = faceOrder[nfIdx];
    const Index start = faceHe[oldF];
    Index h = start;
    do {
      heMap[h] = nextHe;
      heOld[nextHe] = h;
      ++nextHe;
      const Index v = heVert[h];
      if (vertMap[v] == kInvalid) {
        vertMap[v] = static_cast<Index>(vertOld.size());
        vertOld.push_back(v);
      }
      h = heNext[h];
    } while (h != start);
  }
  firstVertex[nf] = static_cast<Index>(vertOld.size());
  // Isolated vertices keep their relative order at the end.
  for (Index v = 0; v < nv; ++v) {
    if (vertMap[v] == kInvalid) {
      vertMap[v] = static_cast<Index>(vertOld.size());
      vertOld.push_back(v);
    }
  }

  std::vector<Index> newNext(nh), newTwin(nh), newVert(nh), newFace(nh);
  for (Index h = 0; h < nh; ++h) {
    const Index o = heOld[h];
    newNext[h] = heMap[heNext[o]];
    newTwin[h] = heTwin[o] == kInvalid ? kInvalid : heMap[heTwin[o]];
    newVert[h] = vertMap[heVert[o]];
  }
  std::vector<Index> newFaceHe(nf);
  std::vector<std::int32_t> newFaceSets(faceSets.size() == faceHe.size() ? nf : 0);
  for (Index f = 0; f < nf; ++f) {
    newFaceHe[f] = heMap[faceHe[faceOrder[f]]];
    if (!newFaceSets.empty()) newFaceSets[f] = faceSets[faceOrder[f]];
    Index h = newFaceHe[f];
    do {
      newFace[h] = f;
      h = newNext[h];
    } while (h != newFaceHe[f]);
  }

  std::vector<Vec3> newPos(nv), newNrm(normals.size() == positions.size() ? nv : 0);
  std::vector<float> newMask(mask.size() == positions.size() ? nv : 0);
  std::vector<Index> newVertHe(nv);
  for (Index v = 0; v < nv; ++v) {
    const Index o = vertOld[v];
    newPos[v] = positions[o];
    if (!newNrm.empty()) newNrm[v] = normals[o];
    if (!newMask.empty()) newMask[v] = mask[o];
    newVertHe[v] = vertHe[o] == kInvalid ? kInvalid : heMap[vertHe[o]];
  }
  auto permute = [&](std::vector<Vec3>& a) {
    if (a.size() != static_cast<std::size_t>(nv)) return;
    std::vector<Vec3> out(a.size());
    for (Index v = 0; v < nv; ++v) out[v] = a[vertOld[v]];
    a = std::move(out);
  };
  if (!layers.empty()) {
    permute(layers.base);
    for (SculptLayer& l : layers.list) permute(l.offset);
  }

  positions = std::move(newPos);
  normals = std::move(newNrm);
  mask = std::move(newMask);
  faceSets = std::move(newFaceSets);
  vertHe = std::move(newVertHe);
  faceHe = std::move(newFaceHe);
  heNext = std::move(newNext);
  heTwin = std::move(newTwin);
  heVert = std::move(newVert);
  heFace = std::move(newFace);
  if (map) {
    map->faceOld.assign(faceOrder.begin(), faceOrder.end());
    map->vertOld = std::move(vertOld);
  }
  return firstVertex;
}

void Mesh::clear() { *this = Mesh{}; }

Mesh copyWithoutLayers(const Mesh& m) {
  Mesh c;
  c.positions = m.positions;
  c.normals = m.normals;
  c.mask = m.mask;
  c.faceSets = m.faceSets;
  c.vertHe = m.vertHe;
  c.faceHe = m.faceHe;
  c.heNext = m.heNext;
  c.heTwin = m.heTwin;
  c.heVert = m.heVert;
  c.heFace = m.heFace;
  return c;
}

void Mesh::reserveHeadroom(Index vertices, Index faces, Index halfEdges) {
  const auto nv = static_cast<std::size_t>(vertices), nf = static_cast<std::size_t>(faces),
             nh = static_cast<std::size_t>(halfEdges);
  positions.reserve(nv);
  if (!normals.empty()) normals.reserve(nv);
  if (!mask.empty()) mask.reserve(nv);
  vertHe.reserve(nv);
  faceHe.reserve(nf);
  if (!faceSets.empty()) faceSets.reserve(nf);
  heNext.reserve(nh);
  heTwin.reserve(nh);
  heVert.reserve(nh);
  heFace.reserve(nh);
}

bool Mesh::hasHeadroom(Index vertices, Index faces, Index halfEdges) const {
  auto fits = [](const auto& v, Index n) { return v.size() + static_cast<std::size_t>(n) <= v.capacity(); };
  return fits(positions, vertices) && (normals.empty() || fits(normals, vertices)) &&
         (mask.empty() || fits(mask, vertices)) && fits(vertHe, vertices) && fits(faceHe, faces) &&
         (faceSets.empty() || fits(faceSets, faces)) &&
         fits(heNext, halfEdges) && fits(heTwin, halfEdges) && fits(heVert, halfEdges) && fits(heFace, halfEdges);
}

bool Mesh::anyMasked() const {
  for (float v : mask) {
    if (v > 0.0f) return true;
  }
  return false;
}

bool Mesh::anyFaceSet() const {
  for (std::size_t f = 0; f < faceSets.size(); ++f) {
    if (faceSetId(faceSets[f]) != kDefaultFaceSet && faceHe[f] != kInvalid) return true;
  }
  return false;
}

bool Mesh::anyHidden() const {
  for (std::size_t f = 0; f < faceSets.size(); ++f) {
    if (faceSets[f] < 0 && faceHe[f] != kInvalid) return true;
  }
  return false;
}

bool Mesh::hasFaceSetData() const {
  for (std::size_t f = 0; f < faceSets.size(); ++f) {
    if (faceSets[f] != kDefaultFaceSet && faceHe[f] != kInvalid) return true;
  }
  return false;
}

std::int32_t Mesh::maxFaceSetId() const {
  std::int32_t id = kDefaultFaceSet;
  for (std::size_t f = 0; f < faceSets.size(); ++f) {
    if (faceHe[f] != kInvalid) id = std::max(id, faceSetId(faceSets[f]));
  }
  return id;
}

std::int32_t Mesh::newFaceSetId() const {
  const std::int32_t id = maxFaceSetId();
  return id < kMaxFaceSetId ? id + 1 : 0;
}

bool Mesh::vertexVisible(Index v) const {
  if (vertHe[v] == kInvalid) return false;
  if (faceSets.empty()) return true;
  bool visible = false;
  forEachOutgoing(v, [&](Index h) {
    const Index f = heFace[h];
    visible |= f != kInvalid && faceSets[f] > 0;
  });
  return visible;
}

// ---------------------------------------------------------------------------------------------

Mesh buildMesh(std::vector<Vec3> positions, std::span<const Index> faceIndices,
               std::span<const Index> faceSizes, BuildReport* report) {
  BuildReport rep;
  Mesh m;
  m.positions = std::move(positions);
  const Index nv = m.vertexCount();

  // Clean faces: drop consecutive duplicates; skip faces that still repeat a vertex.
  std::vector<Index> cleanIdx;
  std::vector<Index> cleanSizes;
  cleanIdx.reserve(faceIndices.size());
  cleanSizes.reserve(faceSizes.size());
  std::vector<Index> poly;
  std::size_t cursor = 0;
  for (Index size : faceSizes) {
    poly.clear();
    for (Index i = 0; i < size; ++i) {
      const Index v = faceIndices[cursor + i];
      if (v < 0 || v >= nv) {
        poly.clear();
        break;
      }
      if (poly.empty() || poly.back() != v) poly.push_back(v);
    }
    cursor += static_cast<std::size_t>(size);
    while (poly.size() > 1 && poly.front() == poly.back()) poly.pop_back();
    bool repeated = poly.size() < 3;
    if (!repeated) {
      std::vector<Index> sorted = poly;
      std::sort(sorted.begin(), sorted.end());
      repeated = std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end();
    }
    if (repeated) {
      ++rep.degenerateFaces;
      continue;
    }
    cleanIdx.insert(cleanIdx.end(), poly.begin(), poly.end());
    cleanSizes.push_back(static_cast<Index>(poly.size()));
  }

  const Index nf = static_cast<Index>(cleanSizes.size());
  const Index nh = static_cast<Index>(cleanIdx.size());
  m.faceHe.resize(nf);
  m.heNext.resize(nh);
  m.heTwin.assign(nh, kInvalid);
  m.heVert.resize(nh);
  m.heFace.resize(nh);

  Index h = 0;
  for (Index f = 0; f < nf; ++f) {
    const Index size = cleanSizes[f];
    m.faceHe[f] = h;
    for (Index i = 0; i < size; ++i) {
      m.heVert[h + i] = cleanIdx[h + i];
      m.heFace[h + i] = f;
      m.heNext[h + i] = h + (i + 1) % size;
    }
    h += size;
  }

  // Outgoing half-edges per vertex in CSR form.
  std::vector<Index> outStart(nv + 1, 0);
  for (Index e = 0; e < nh; ++e) ++outStart[m.heVert[e] + 1];
  for (Index v = 0; v < nv; ++v) outStart[v + 1] += outStart[v];
  std::vector<Index> outList(nh);
  {
    std::vector<Index> fill(outStart.begin(), outStart.end() - 1);
    for (Index e = 0; e < nh; ++e) outList[fill[m.heVert[e]]++] = e;
  }

  for (Index e = 0; e < nh; ++e) {
    const Index a = m.heVert[e];
    const Index b = m.heTarget(e);
    // The same directed edge used twice means inconsistent winding or a non-manifold edge.
    for (Index k = outStart[a]; k < outStart[a + 1]; ++k) {
      const Index g = outList[k];
      if (g < e && m.heTarget(g) == b) {
        ++rep.nonManifoldEdges;
        break;
      }
    }
    if (m.heTwin[e] != kInvalid) continue;
    for (Index k = outStart[b]; k < outStart[b + 1]; ++k) {
      const Index g = outList[k];
      if (m.heTwin[g] == kInvalid && m.heTarget(g) == a) {
        m.heTwin[e] = g;
        m.heTwin[g] = e;
        break;
      }
    }
  }

  m.vertHe.assign(nv, kInvalid);
  for (Index v = 0; v < nv; ++v) {
    if (outStart[v] == outStart[v + 1]) {
      ++rep.isolatedVertices;
      continue;
    }
    // Prefer an outgoing boundary half-edge so fans of open vertices start at the boundary.
    Index pick = outList[outStart[v]];
    for (Index k = outStart[v]; k < outStart[v + 1]; ++k) {
      if (m.heTwin[m.hePrev(outList[k])] == kInvalid) {
        pick = outList[k];
        break;
      }
    }
    m.vertHe[v] = pick;
    Index visited = 0;
    m.forEachOutgoing(v, [&](Index) { ++visited; });
    if (visited != outStart[v + 1] - outStart[v]) ++rep.nonManifoldVertices;
  }

  m.computeNormals();
  if (report) *report = rep;
  return m;
}

void weldVertices(std::vector<Vec3>& positions, std::vector<Index>& faceIndices, float epsilon) {
  // Grid cells are much larger than epsilon, so most points only need their own cell; a
  // neighbouring cell is checked only when the point lies within epsilon of that side.
  const float cell = epsilon * 64.0f;
  const float inv = 1.0f / cell;
  const float eps2 = epsilon * epsilon;

  struct Slot {
    glm::ivec3 key;
    Index head = kInvalid;  // First output vertex in this cell; chained through `chain`.
  };
  std::size_t capacity = 16;
  while (capacity < positions.size() * 2) capacity <<= 1;
  std::vector<Slot> table(capacity);
  auto hash = [&](const glm::ivec3& c) {
    std::uint64_t h = static_cast<std::uint32_t>(c.x) * 0x9E3779B97F4A7C15ull;
    h ^= static_cast<std::uint32_t>(c.y) * 0xC2B2AE3D27D4EB4Full + (h << 6) + (h >> 2);
    h ^= static_cast<std::uint32_t>(c.z) * 0x165667B19E3779F9ull + (h << 6) + (h >> 2);
    return static_cast<std::size_t>(h) & (capacity - 1);
  };
  auto slotFor = [&](const glm::ivec3& c) -> Slot& {
    std::size_t i = hash(c);
    while (table[i].head != kInvalid && table[i].key != c) i = (i + 1) & (capacity - 1);
    return table[i];
  };

  std::vector<Vec3> out;
  out.reserve(positions.size());
  std::vector<Index> chain;
  chain.reserve(positions.size());
  std::vector<Index> remap(positions.size());

  for (std::size_t i = 0; i < positions.size(); ++i) {
    const Vec3& p = positions[i];
    const Vec3 g = p * inv;
    const glm::ivec3 c(static_cast<int>(std::floor(g.x)), static_cast<int>(std::floor(g.y)),
                       static_cast<int>(std::floor(g.z)));
    // Which neighbours can hold a point within epsilon: -1, 0 or +1 per axis.
    glm::ivec3 lo(0), hi(0);
    for (int a = 0; a < 3; ++a) {
      const float f = (g[a] - static_cast<float>(c[a])) * cell;
      if (f < epsilon) lo[a] = -1;
      if (cell - f < epsilon) hi[a] = 1;
    }
    Index found = kInvalid;
    for (int dz = lo.z; dz <= hi.z && found == kInvalid; ++dz)
      for (int dy = lo.y; dy <= hi.y && found == kInvalid; ++dy)
        for (int dx = lo.x; dx <= hi.x && found == kInvalid; ++dx) {
          const Slot& s = slotFor(c + glm::ivec3(dx, dy, dz));
          for (Index v = s.head; v != kInvalid; v = chain[v]) {
            const Vec3 d = out[v] - p;
            if (glm::dot(d, d) <= eps2) {
              found = v;
              break;
            }
          }
        }
    if (found == kInvalid) {
      found = static_cast<Index>(out.size());
      out.push_back(p);
      Slot& s = slotFor(c);
      s.key = c;
      chain.push_back(s.head);
      s.head = found;
    }
    remap[i] = found;
  }
  for (Index& idx : faceIndices) idx = remap[idx];
  positions = std::move(out);
}

namespace {
ValidationResult validateImpl(const Mesh& m, bool live) {
  auto fail = [](std::string msg) { return ValidationResult{false, std::move(msg)}; };
  const Index nv = m.vertexCount();
  const Index nf = m.faceCount();
  const Index nh = m.halfEdgeCount();

  if (static_cast<Index>(m.vertHe.size()) != nv) return fail("vertHe size mismatch");
  if (!m.normals.empty() && static_cast<Index>(m.normals.size()) != nv) return fail("normals size mismatch");
  if (!m.mask.empty() && static_cast<Index>(m.mask.size()) != nv) return fail("mask size mismatch");
  for (Index v = 0; v < static_cast<Index>(m.mask.size()); ++v) {
    if (!(m.mask[v] >= 0.0f && m.mask[v] <= 1.0f)) return fail("mask out of range at vertex " + std::to_string(v));
  }
  if (!m.layers.empty() || !m.layers.base.empty()) {
    if (m.layers.base.size() != static_cast<std::size_t>(nv)) return fail("layer base size mismatch");
    for (const SculptLayer& l : m.layers.list) {
      if (l.offset.size() != static_cast<std::size_t>(nv)) return fail("layer offset size mismatch");
    }
  }
  if (!m.faceSets.empty() && static_cast<Index>(m.faceSets.size()) != nf) return fail("face set size mismatch");
  for (Index f = 0; f < static_cast<Index>(m.faceSets.size()); ++f) {
    if (!validFaceSetValue(m.faceSets[f]) && (!live || m.faceHe[f] != kInvalid))
      return fail("face set value " + std::to_string(m.faceSets[f]) + " at face " + std::to_string(f));
  }
  if (static_cast<Index>(m.heTwin.size()) != nh || static_cast<Index>(m.heVert.size()) != nh ||
      static_cast<Index>(m.heFace.size()) != nh)
    return fail("half-edge array size mismatch");

  for (Index v = 0; v < nv; ++v) {
    const Vec3& p = m.positions[v];
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
      return fail("non-finite position at vertex " + std::to_string(v));
  }

  Index liveHalfEdges = 0;
  for (Index h = 0; h < nh; ++h) {
    if (live && m.heFace[h] == kInvalid) continue;
    ++liveHalfEdges;
    if (m.heNext[h] < 0 || m.heNext[h] >= nh) return fail("heNext out of range at " + std::to_string(h));
    if (m.heVert[h] < 0 || m.heVert[h] >= nv) return fail("heVert out of range at " + std::to_string(h));
    if (m.heFace[h] < 0 || m.heFace[h] >= nf) return fail("heFace out of range at " + std::to_string(h));
    if (live && m.faceHe[m.heFace[h]] == kInvalid) return fail("half-edge of a removed face at " + std::to_string(h));
    if (m.heFace[m.heNext[h]] != m.heFace[h]) return fail("heNext leaves its face at " + std::to_string(h));
    const Index t = m.heTwin[h];
    if (t != kInvalid) {
      if (t < 0 || t >= nh) return fail("heTwin out of range at " + std::to_string(h));
      if (t == h) return fail("half-edge is its own twin at " + std::to_string(h));
      if (m.heTwin[t] != h) return fail("twin not symmetric at " + std::to_string(h));
      if (m.heVert[t] != m.heTarget(h) || m.heTarget(t) != m.heVert[h])
        return fail("twin endpoints mismatch at " + std::to_string(h));
    }
  }

  Index walked = 0;
  for (Index f = 0; f < nf; ++f) {
    if (live && m.faceHe[f] == kInvalid) continue;
    const Index start = m.faceHe[f];
    if (start < 0 || start >= nh || m.heFace[start] != f) return fail("faceHe invalid at " + std::to_string(f));
    Index h = start;
    Index n = 0;
    do {
      if (++n > nh) return fail("face loop does not close at " + std::to_string(f));
      h = m.heNext[h];
    } while (h != start);
    if (n < 3) return fail("face with fewer than 3 sides at " + std::to_string(f));
    walked += n;
  }
  if (walked != liveHalfEdges) return fail("some half-edges belong to no face loop");

  std::vector<Index> outCount(nv, 0);
  for (Index h = 0; h < nh; ++h)
    if (!live || m.heFace[h] != kInvalid) ++outCount[m.heVert[h]];
  for (Index v = 0; v < nv; ++v) {
    const Index start = m.vertHe[v];
    if (start == kInvalid) {
      if (outCount[v] != 0) return fail("vertex with faces has no vertHe at " + std::to_string(v));
      continue;
    }
    if (start < 0 || start >= nh || m.heVert[start] != v || m.heFace[start] == kInvalid)
      return fail("vertHe invalid at " + std::to_string(v));
    Index visited = 0;
    m.forEachOutgoing(v, [&](Index) { ++visited; });
    if (visited != outCount[v]) return fail("non-manifold vertex fan at " + std::to_string(v));
  }
  return {};
}
}  // namespace

ValidationResult validate(const Mesh& m) { return validateImpl(m, false); }
ValidationResult validateLive(const Mesh& m) { return validateImpl(m, true); }

ValidationResult validateLayers(const Mesh& m, bool checkComposite) {
  auto fail = [](std::string msg) { return ValidationResult{false, "layers: " + std::move(msg)}; };
  const LayerStack& s = m.layers;
  if (s.empty()) {
    if (!s.base.empty() || s.epoch != 0 || s.nextId != 1 || s.active != 0) return fail("empty stack carries state");
    return {};
  }
  const std::size_t nv = m.positions.size();
  if (s.base.size() != nv) return fail("base size mismatch");
  if (s.epoch == 0) return fail("no epoch");
  if (s.list.size() > static_cast<std::size_t>(kMaxFileLayers)) return fail("too many layers");
  auto finite = [](const Vec3& v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); };
  for (std::size_t i = 0; i < nv; ++i) {
    if (!finite(s.base[i])) return fail("non-finite base at vertex " + std::to_string(i));
  }
  for (std::size_t k = 0; k < s.list.size(); ++k) {
    const SculptLayer& l = s.list[k];
    if (l.id == 0 || l.id >= s.nextId) return fail("layer id out of range");
    for (std::size_t j = 0; j < k; ++j) {
      if (s.list[j].id == l.id) return fail("duplicate layer id");
    }
    if (!std::isfinite(l.strength) || std::abs(l.strength) > kMaxLayerStrength) return fail("strength out of range");
    if (l.name.empty() || l.name.size() > kMaxLayerNameBytes) return fail("bad name length");
    if (l.offset.size() != nv) return fail("offset size mismatch");
    for (std::size_t i = 0; i < nv; ++i) {
      if (!finite(l.offset[i])) return fail("non-finite offset at vertex " + std::to_string(i));
    }
  }
  if (s.active != 0 && !s.find(s.active)) return fail("active layer missing");
  if (!checkComposite) return {};
  for (std::size_t i = 0; i < nv; ++i) {
    if (!sameBits(m.positions[i], composeVertex(s, static_cast<Index>(i))))
      return fail("position differs from the composite at vertex " + std::to_string(i));
  }
  if (m.normals.size() == nv) {
    Mesh copy;
    copy.positions = m.positions;
    copy.vertHe = m.vertHe;
    copy.faceHe = m.faceHe;
    copy.heNext = m.heNext;
    copy.heTwin = m.heTwin;
    copy.heVert = m.heVert;
    copy.heFace = m.heFace;
    copy.computeNormals();
    for (std::size_t i = 0; i < nv; ++i) {
      if (!sameBits(m.normals[i], copy.normals[i])) return fail("stale normal at vertex " + std::to_string(i));
    }
  }
  return {};
}

}  // namespace plegl
