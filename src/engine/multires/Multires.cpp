#include "multires/Multires.h"

#include "scene/Scene.h"

namespace plegl {

namespace {
template <class T>
std::size_t vecBytes(const std::vector<T>& v) {
  return v.size() * sizeof(T);
}
std::size_t meshBytes(const Mesh& m) {
  return vecBytes(m.positions) + vecBytes(m.normals) + vecBytes(m.mask) + vecBytes(m.faceSets) + vecBytes(m.vertHe) +
         vecBytes(m.faceHe) + vecBytes(m.heNext) + vecBytes(m.heTwin) + vecBytes(m.heVert) + vecBytes(m.heFace);
}
}  // namespace

std::size_t SubdivisionLinks::bytes() const {
  return vecBytes(vertexChild) + vecBytes(edgeChild) + vecBytes(faceChild) + vecBytes(childFace) + vecBytes(parent) +
         vecBytes(parentHalfEdge);
}

std::size_t MultiresLevel::bytes() const {
  return meshBytes(mesh) + bvh.memoryBytes() + vecBytes(rule) + vecBytes(nonManifoldFaces) + vecBytes(canon.vert) +
         vecBytes(canon.face) + vecBytes(canon.faceStart) + links.bytes();
}

std::size_t LevelReference::bytes() const { return vecBytes(positions) + vecBytes(mask) + vecBytes(faceSets); }

std::size_t LevelDelta::bytes() const {
  return vecBytes(posIndex) + vecBytes(posBefore) + vecBytes(posAfter) + vecBytes(maskIndex) + vecBytes(maskBefore) +
         vecBytes(maskAfter) + vecBytes(setIndex) + vecBytes(setBefore) + vecBytes(setAfter);
}

bool SyncDelta::empty() const {
  for (const LevelDelta& d : levels)
    if (!d.empty() || d.maskCreated || d.setsCreated) return false;
  return true;
}

std::size_t SyncDelta::bytes() const {
  std::size_t n = 0;
  for (const LevelDelta& d : levels) n += d.bytes();
  return n;
}

std::unique_ptr<Multires> Multires::clone(bool freshVersions) const {
  auto copy = std::make_unique<Multires>(*this);
  if (freshVersions)
    for (MultiresLevel& l : copy->levels) l.version = nextTopologyVersion();
  return copy;
}

std::size_t Multires::bytes() const {
  std::size_t n = reference.bytes();
  for (const MultiresLevel& l : levels) n += l.bytes();
  return n;
}

std::uint64_t layoutHash(const Bvh& bvh) {
  std::uint64_t h = 1469598103934665603ull;
  auto mix = [&](Index v) {
    auto u = static_cast<std::uint32_t>(v);
    for (int i = 0; i < 4; ++i) {
      h ^= (u >> (8 * i)) & 0xFFu;
      h *= 1099511628211ull;
    }
  };
  for (const BvhLeaf& l : bvh.leaves()) {
    mix(l.faceBegin);
    mix(l.faceEnd);
    mix(l.vertBegin);
    mix(l.vertEnd);
    mix(l.heBegin);
    mix(l.heEnd);
  }
  return h;
}

CanonicalMap identityCanonicalMap(const Mesh& m) {
  CanonicalMap c;
  c.vert.resize(m.positions.size());
  for (Index v = 0; v < m.vertexCount(); ++v) c.vert[v] = v;
  c.face.resize(m.faceHe.size());
  c.faceStart.resize(m.faceHe.size() + 1);
  Index start = 0;
  for (Index f = 0; f < m.faceCount(); ++f) {
    c.face[f] = f;
    c.faceStart[f] = start;
    start += m.faceSize(f);
  }
  c.faceStart[m.faceHe.size()] = start;
  return c;
}

std::vector<Index> canonicalHalfEdges(const Mesh& m, const CanonicalMap& canon) {
  std::vector<Index> out(m.heNext.size(), kInvalid);
  for (Index f = 0; f < m.faceCount(); ++f) {
    Index id = canon.firstHalfEdge(canon.face[f]);
    const Index start = m.faceHe[f];
    Index h = start;
    do {
      out[h] = id++;
      h = m.heNext[h];
    } while (h != start);
  }
  return out;
}

void canonicalPolygons(const Mesh& m, const CanonicalMap& canon, std::vector<std::uint32_t>& sizes,
                       std::vector<std::uint32_t>& corners) {
  const std::vector<Index> liveFace = invertMap(canon.face);
  sizes.resize(liveFace.size());
  corners.clear();
  corners.reserve(m.heNext.size());
  for (std::size_t c = 0; c < liveFace.size(); ++c) {
    const std::size_t before = corners.size();
    m.forEachFaceVertex(liveFace[c], [&](Index v) { corners.push_back(static_cast<std::uint32_t>(canon.vert[v])); });
    sizes[c] = static_cast<std::uint32_t>(corners.size() - before);
  }
}

std::vector<Index> invertMap(const std::vector<Index>& map) {
  std::vector<Index> inv(map.size());
  for (std::size_t i = 0; i < map.size(); ++i) inv[map[i]] = static_cast<Index>(i);
  return inv;
}

}  // namespace plegl
