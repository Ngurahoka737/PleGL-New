#include "mesh/MeshEdit.h"

#include <algorithm>

namespace plegl {

MeshEditor::MeshEditor(Mesh& mesh) : m_(mesh), deadVertex_(mesh.positions.size(), 0) {}

void MeshEditor::collectOutgoing(Index v, std::vector<Index>& out) const {
  out.clear();
  m_.forEachOutgoing(v, [&](Index h) { out.push_back(h); });
}

bool MeshEditor::closedFan(Index v) const {
  if (deadVertex_[v] || m_.vertHe[v] == kInvalid) return false;
  bool closed = true;
  m_.forEachOutgoing(v, [&](Index h) { closed &= m_.heTwin[h] != kInvalid; });
  return closed;
}

int MeshEditor::valence(Index v) const {
  int n = 0;
  m_.forEachOutgoing(v, [&](Index) { ++n; });
  return n;
}

bool MeshEditor::connected(Index u, Index w) const {
  bool found = false;
  m_.forEachOutgoing(u, [&](Index h) { found |= m_.heTarget(h) == w; });
  return found;
}

void MeshEditor::killHalfEdge(Index h) {
  m_.heFace[h] = kInvalid;
  m_.heNext[h] = kInvalid;
  m_.heTwin[h] = kInvalid;
}

bool MeshEditor::rotateEdge(Index h) {
  const Index t = m_.heTwin[h];
  if (t == kInvalid) return false;
  const Index F = m_.heFace[h], G = m_.heFace[t];
  if (F == G) return false;
  const Index a = m_.heVert[h], b = m_.heVert[t];
  const Index hn = m_.heNext[h], hnn = m_.heNext[hn], hp = m_.hePrev(h);
  const Index tn = m_.heNext[t], tnn = m_.heNext[tn], tp = m_.hePrev(t);
  const Index c = m_.heVert[hnn], d = m_.heVert[tnn];
  if (c == d || c == a || d == b) return false;
  if (!closedFan(a) || !closedFan(b) || !closedFan(c) || !closedFan(d)) return false;
  if (valence(a) <= 3 || valence(b) <= 3) return false;
  if (connected(c, d)) return false;

  if (obs_) {
    for (Index e : {hp, tn, h, tp, hn, t}) touchH(e);
    touchV(a);
    touchV(b);
    touchF(F);
    touchF(G);
  }
  // F = [a, b, c, ...] becomes [d, c, ..., a]; G = [b, a, d, ...] becomes [c, d, ..., b].
  m_.heNext[hp] = tn;
  m_.heNext[tn] = h;
  m_.heNext[h] = hnn;
  m_.heNext[tp] = hn;
  m_.heNext[hn] = t;
  m_.heNext[t] = tnn;
  m_.heFace[tn] = F;
  m_.heFace[hn] = G;
  m_.heVert[h] = d;
  m_.heVert[t] = c;
  m_.vertHe[a] = tn;
  m_.vertHe[b] = hn;
  m_.faceHe[F] = h;
  m_.faceHe[G] = t;
  return true;
}

Index MeshEditor::splitEdge(Index h, float t) {
  const Index tw = m_.heTwin[h];
  const Index a = m_.heVert[h], b = m_.heTarget(h);
  touchH(h);
  touchH(tw);
  const Index v = static_cast<Index>(m_.positions.size());
  m_.positions.push_back(glm::mix(m_.positions[a], m_.positions[b], t));
  if (!m_.normals.empty()) {
    const Vec3 n = m_.normals[a] + m_.normals[b];
    const float len = glm::length(n);
    m_.normals.push_back(len > 0.0f ? n / len : m_.normals[a]);
  }
  if (!m_.mask.empty()) m_.mask.push_back(glm::mix(m_.mask[a], m_.mask[b], t));
  deadVertex_.push_back(0);

  auto newHalfEdge = [&](Index vert, Index face, Index next) {
    const Index e = static_cast<Index>(m_.heNext.size());
    m_.heNext.push_back(next);
    m_.heTwin.push_back(kInvalid);
    m_.heVert.push_back(vert);
    m_.heFace.push_back(face);
    return e;
  };
  // h: a -> b becomes a -> v, followed by h2: v -> b.
  const Index h2 = newHalfEdge(v, m_.heFace[h], m_.heNext[h]);
  m_.heNext[h] = h2;
  m_.vertHe.push_back(h2);
  if (tw != kInvalid) {
    // tw: b -> a becomes b -> v, followed by tw2: v -> a.
    const Index tw2 = newHalfEdge(v, m_.heFace[tw], m_.heNext[tw]);
    m_.heNext[tw] = tw2;
    m_.heTwin[h] = tw2;
    m_.heTwin[tw2] = h;
    m_.heTwin[h2] = tw;
    m_.heTwin[tw] = h2;
  }
  return v;
}

Index MeshEditor::splitFace(Index ha, Index hb) {
  const Index f = m_.heFace[ha];
  if (ha == hb || m_.heFace[hb] != f || m_.heNext[ha] == hb || m_.heNext[hb] == ha) return kInvalid;
  const Index u = m_.heVert[ha], w = m_.heVert[hb];
  if (connected(u, w)) return kInvalid;
  const Index pa = m_.hePrev(ha), pb = m_.hePrev(hb);
  if (obs_) {
    touchH(pa);
    touchH(pb);
    for (Index e = hb; e != ha; e = m_.heNext[e]) touchH(e);  // These move to the new face.
    touchF(f);
  }
  const Index g = static_cast<Index>(m_.faceHe.size());
  m_.faceHe.push_back(hb);
  if (!m_.faceSets.empty()) m_.faceSets.push_back(m_.faceSets[f]);  // Both halves stay in f's set.
  const Index d1 = static_cast<Index>(m_.heNext.size());  // u -> w, closes the new face g.
  const Index d2 = d1 + 1;                                // w -> u, closes f.
  m_.heNext.insert(m_.heNext.end(), {hb, ha});
  m_.heTwin.insert(m_.heTwin.end(), {d2, d1});
  m_.heVert.insert(m_.heVert.end(), {u, w});
  m_.heFace.insert(m_.heFace.end(), {g, f});
  m_.heNext[pb] = d2;
  m_.heNext[pa] = d1;
  for (Index e = hb; e != d1; e = m_.heNext[e]) m_.heFace[e] = g;
  m_.faceHe[f] = ha;
  return g;
}

bool MeshEditor::collapseEdge(Index h, const Vec3& position) {
  const Index t = m_.heTwin[h];
  if (t == kInvalid) return false;
  const Index a = m_.heVert[h], b = m_.heVert[t];
  if (!closedFan(a) || !closedFan(b)) return false;

  // Apexes of the triangles on the edge; these are the only neighbours a and b may share.
  Index apex[2] = {kInvalid, kInvalid};
  int apexCount = 0;
  for (Index s : {h, t}) {
    if (m_.faceSize(m_.heFace[s]) == 3) {
      apex[apexCount++] = m_.heTarget(m_.heNext[s]);
    }
  }
  if (apexCount == 2 && apex[0] == apex[1]) return false;
  for (int i = 0; i < apexCount; ++i)
    if (valence(apex[i]) <= 3) return false;
  int shared = 0;
  bool linkOk = true;
  m_.forEachOutgoing(a, [&](Index e) {
    const Index n = m_.heTarget(e);
    if (n == b || !connected(n, b)) return;
    ++shared;
    linkOk &= (n == apex[0] || n == apex[1]);
  });
  if (!linkOk || shared != apexCount) return false;
  if (valence(a) + valence(b) - 2 - apexCount < 3) return false;

  collectOutgoing(a, scratchA_);
  collectOutgoing(b, scratchB_);
  if (obs_) {
    for (Index s : {h, t}) {
      const Index sn = m_.heNext[s], sp = m_.hePrev(s);
      touchF(m_.heFace[s]);
      touchH(s);
      touchH(sp);
      if (m_.heNext[sn] == sp) {  // Triangle: all three die and the outer twins are relinked.
        touchH(sn);
        touchH(m_.heTwin[sn]);
        touchH(m_.heTwin[sp]);
        touchV(m_.heVert[sp]);
      }
    }
    for (Index e : scratchB_) touchH(e);
    touchV(a);
    touchV(b);
  }
  for (Index s : {h, t}) {
    const Index F = m_.heFace[s];
    const Index sn = m_.heNext[s], sp = m_.hePrev(s);
    if (m_.heNext[sn] == sp) {  // Triangle: it disappears and its two other edges merge.
      const Index tw1 = m_.heTwin[sn], tw2 = m_.heTwin[sp];
      m_.heTwin[tw1] = tw2;
      m_.heTwin[tw2] = tw1;
      const Index c = m_.heVert[sp];
      if (m_.vertHe[c] == sp) m_.vertHe[c] = tw1;
      killHalfEdge(s);
      killHalfEdge(sn);
      killHalfEdge(sp);
      m_.faceHe[F] = kInvalid;
    } else {  // Larger face: just drop the corner.
      m_.heNext[sp] = sn;
      if (m_.faceHe[F] == s) m_.faceHe[F] = sn;
      killHalfEdge(s);
    }
  }
  for (Index e : scratchB_)
    if (halfEdgeAlive(e)) m_.heVert[e] = a;
  m_.positions[a] = position;
  // Keep the stronger mask so a collapse never shrinks a protected region.
  if (!m_.mask.empty()) m_.mask[a] = std::max(m_.mask[a], m_.mask[b]);
  deadVertex_[b] = 1;
  m_.vertHe[b] = kInvalid;
  m_.vertHe[a] = kInvalid;
  for (const auto* list : {&scratchA_, &scratchB_})
    for (Index e : *list)
      if (m_.vertHe[a] == kInvalid && halfEdgeAlive(e)) m_.vertHe[a] = e;
  return true;
}

bool MeshEditor::collapseDiagonal(Index h) {
  const Index F = m_.heFace[h];
  const Index e0 = h, e1 = m_.heNext[e0], e2 = m_.heNext[e1], e3 = m_.heNext[e2];
  if (m_.heNext[e3] != e0) return false;  // Not a quad.
  const Index v0 = m_.heVert[e0], v1 = m_.heVert[e1], v2 = m_.heVert[e2], v3 = m_.heVert[e3];
  const Index t0 = m_.heTwin[e0], t1 = m_.heTwin[e1], t2 = m_.heTwin[e2], t3 = m_.heTwin[e3];
  if (t0 == kInvalid || t1 == kInvalid || t2 == kInvalid || t3 == kInvalid) return false;
  if (m_.heFace[t0] == m_.heFace[t1] || m_.heFace[t2] == m_.heFace[t3]) return false;
  for (Index v : {v0, v1, v2, v3})
    if (!closedFan(v)) return false;
  if (valence(v1) <= 3 || valence(v3) <= 3) return false;
  bool linkOk = true;
  m_.forEachOutgoing(v0, [&](Index e) {
    const Index n = m_.heTarget(e);
    if (n != v1 && n != v3 && connected(n, v2)) linkOk = false;
  });
  if (!linkOk || connected(v0, v2)) return false;

  collectOutgoing(v2, scratchB_);
  if (obs_) {
    for (Index e : {e0, e1, e2, e3, t0, t1, t2, t3}) touchH(e);
    for (Index e : scratchB_) touchH(e);
    for (Index v : {v0, v1, v2, v3}) touchV(v);
    touchF(F);
  }
  // The quad's opposite edges pair up: v1-v0 with v1-v2, and v3-v0 with v3-v2.
  m_.heTwin[t0] = t1;
  m_.heTwin[t1] = t0;
  m_.heTwin[t2] = t3;
  m_.heTwin[t3] = t2;
  for (Index e : {e0, e1, e2, e3}) killHalfEdge(e);
  m_.faceHe[F] = kInvalid;
  for (Index e : scratchB_)
    if (halfEdgeAlive(e)) m_.heVert[e] = v0;
  m_.positions[v0] = (m_.positions[v0] + m_.positions[v2]) * 0.5f;
  if (!m_.mask.empty()) m_.mask[v0] = std::max(m_.mask[v0], m_.mask[v2]);
  deadVertex_[v2] = 1;
  m_.vertHe[v2] = kInvalid;
  m_.vertHe[v0] = t3;
  m_.vertHe[v1] = t0;
  m_.vertHe[v3] = t2;
  return true;
}

void MeshEditor::compact() {
  const Index nv = m_.vertexCount(), nf = m_.faceCount(), nh = m_.halfEdgeCount();
  std::vector<Index> vmap(nv, kInvalid), fmap(nf, kInvalid), hmap(nh, kInvalid);
  Index cv = 0, cf = 0, ch = 0;
  for (Index v = 0; v < nv; ++v)
    if (!deadVertex_[v]) vmap[v] = cv++;
  for (Index f = 0; f < nf; ++f)
    if (m_.faceHe[f] != kInvalid) fmap[f] = cf++;
  for (Index h = 0; h < nh; ++h)
    if (m_.heFace[h] != kInvalid) hmap[h] = ch++;

  auto remap = [](Index i, const std::vector<Index>& map) { return i == kInvalid ? kInvalid : map[i]; };
  Mesh out;
  out.positions.resize(cv);
  out.vertHe.resize(cv);
  if (!m_.mask.empty()) out.mask.resize(cv);
  for (Index v = 0; v < nv; ++v) {
    if (vmap[v] == kInvalid) continue;
    out.positions[vmap[v]] = m_.positions[v];
    if (!out.mask.empty()) out.mask[vmap[v]] = m_.mask[v];
    out.vertHe[vmap[v]] = remap(m_.vertHe[v], hmap);
  }
  out.faceHe.resize(cf);
  if (!m_.faceSets.empty()) out.faceSets.resize(cf);
  for (Index f = 0; f < nf; ++f) {
    if (fmap[f] == kInvalid) continue;
    out.faceHe[fmap[f]] = hmap[m_.faceHe[f]];
    if (!out.faceSets.empty()) out.faceSets[fmap[f]] = m_.faceSets[f];
  }
  out.heNext.resize(ch);
  out.heTwin.resize(ch);
  out.heVert.resize(ch);
  out.heFace.resize(ch);
  for (Index h = 0; h < nh; ++h) {
    const Index n = hmap[h];
    if (n == kInvalid) continue;
    out.heNext[n] = hmap[m_.heNext[h]];
    out.heTwin[n] = remap(m_.heTwin[h], hmap);
    out.heVert[n] = vmap[m_.heVert[h]];
    out.heFace[n] = fmap[m_.heFace[h]];
  }
  out.computeNormals();
  m_ = std::move(out);
  deadVertex_.assign(m_.positions.size(), 0);
}

}  // namespace plegl
