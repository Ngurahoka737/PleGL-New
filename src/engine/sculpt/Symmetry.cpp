#include "sculpt/Symmetry.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace plegl {

namespace {

// Mirror combinations in op order: I, Sx, Sy, Sz, SxSy, SxSz, SySz, SxSySz (bit a flips axis a).
constexpr int kFlipOrder[8] = {0, 1, 2, 4, 3, 5, 6, 7};

float maxAbsDiff(const Mat3& a, const Mat3& b) {
  float d = 0.0f;
  for (int c = 0; c < 3; ++c)
    for (int r = 0; r < 3; ++r) d = std::max(d, std::abs(a[c][r] - b[c][r]));
  return d;
}

// Entries within 1e-6 of 0 or +-1 become exact, so projectors of mirrors and quarter turns are
// exact and their zero rows can be recognised.
Mat3 clean(Mat3 m) {
  for (int c = 0; c < 3; ++c) {
    for (int r = 0; r < 3; ++r) {
      float& e = m[c][r];
      if (std::abs(e) < 1e-6f) e = 0.0f;
      else if (std::abs(e - 1.0f) < 1e-6f) e = 1.0f;
      else if (std::abs(e + 1.0f) < 1e-6f) e = -1.0f;
    }
  }
  return m;
}

// Right-handed turn by 2 pi j / n around a coordinate axis. Multiples of a quarter turn are
// written exactly; other angles are computed in double and rounded.
Mat3 radialTurn(Axis axis, int j, int n) {
  double c = 1.0, s = 0.0;
  if ((4 * j) % n == 0) {
    static constexpr double kCos[4] = {1.0, 0.0, -1.0, 0.0};
    static constexpr double kSin[4] = {0.0, 1.0, 0.0, -1.0};
    const int quarter = (4 * j / n) % 4;
    c = kCos[quarter];
    s = kSin[quarter];
  } else {
    const double t = 2.0 * std::numbers::pi * static_cast<double>(j) / static_cast<double>(n);
    c = std::cos(t);
    s = std::sin(t);
  }
  const float cf = static_cast<float>(c), sf = static_cast<float>(s);
  Mat3 m{1.0f};  // m[column][row].
  switch (axis) {
    case Axis::X:  // (x, c y - s z, s y + c z)
      m[1] = Vec3{0.0f, cf, sf};
      m[2] = Vec3{0.0f, -sf, cf};
      break;
    case Axis::Y:  // (c x + s z, y, -s x + c z)
      m[0] = Vec3{cf, 0.0f, -sf};
      m[2] = Vec3{sf, 0.0f, cf};
      break;
    case Axis::Z:  // (c x - s y, s x + c y, z)
      m[0] = Vec3{cf, sf, 0.0f};
      m[1] = Vec3{-sf, cf, 0.0f};
      break;
  }
  return m;
}

// Every distinct L = R_j * D in op order, with its determinant.
void generate(const SymmetrySpec& spec, std::vector<SymmetryOp>& out) {
  out.clear();
  const int n = std::max(1, spec.radialCount);
  const bool on[3] = {spec.x, spec.y, spec.z};
  for (int flips : kFlipOrder) {
    bool allowed = true;
    for (int a = 0; a < 3; ++a) allowed &= !(flips & (1 << a)) || on[a];
    if (!allowed) continue;
    for (int j = 0; j < n; ++j) {
      Mat3 L = radialTurn(spec.radialAxis, j, n);
      float det = 1.0f;
      for (int a = 0; a < 3; ++a) {
        if (!(flips & (1 << a))) continue;
        L[a] = -L[a];  // R * D: column a of R times s_a, exact.
        det = -det;
      }
      bool duplicate = false;
      for (const SymmetryOp& o : out) duplicate |= maxAbsDiff(o.L, L) <= 1e-6f;
      if (!duplicate) out.push_back({L, det});
    }
  }
}

}  // namespace

Vec3 transform(const Mat3& L, const Vec3& p) {
  Vec3 out{0.0f};
  for (int r = 0; r < 3; ++r) {
    float s = 0.0f;
    bool first = true;
    for (int c = 0; c < 3; ++c) {
      const float a = L[c][r];
      if (a == 0.0f) continue;
      const float t = a == 1.0f ? p[c] : a == -1.0f ? -p[c] : a * p[c];
      s = first ? t : s + t;
      first = false;
    }
    out[r] = s;
  }
  return out;
}

int copyCount(const SymmetrySpec& spec) {
  std::vector<SymmetryOp> ops;
  generate(spec, ops);
  return static_cast<int>(ops.size());
}

int maxRadialCount(const SymmetrySpec& spec) {
  SymmetrySpec s = spec;
  for (int n = kMaxRadialCount; n > 1; --n) {
    s.radialCount = n;
    if (copyCount(s) <= kMaxSymmetryCopies) return n;
  }
  return 1;
}

SymmetrySpec sanitized(SymmetrySpec spec) {
  if (static_cast<int>(spec.radialAxis) > 2) spec.radialAxis = Axis::Y;
  spec.radialCount = std::clamp(spec.radialCount, 1, kMaxRadialCount);
  // The count is not monotonic (an even count can merge with two mirrors), so step down to the
  // nearest count that fits.
  while (spec.radialCount > 1 && copyCount(spec) > kMaxSymmetryCopies) --spec.radialCount;
  return spec;
}

SymmetryGroup::SymmetryGroup() {
  ops_[0] = {};
  mul_[0][0] = 0;
  inv_[0] = 0;
  fix_[0] = Mat3{1.0f};
}

SymmetryGroup::SymmetryGroup(const SymmetrySpec& spec) : spec_(spec) {
  std::vector<SymmetryOp> ops;
  generate(spec, ops);
  size_ = std::min(static_cast<int>(ops.size()), kMaxSymmetryCopies);
  for (int k = 0; k < size_; ++k) ops_[static_cast<std::size_t>(k)] = ops[static_cast<std::size_t>(k)];
  for (int a = 0; a < size_; ++a) {
    for (int b = 0; b < size_; ++b) {
      const Mat3 m = ops_[static_cast<std::size_t>(a)].L * ops_[static_cast<std::size_t>(b)].L;
      int found = 0;
      float best = 1e30f;
      for (int k = 0; k < size_; ++k) {
        const float d = maxAbsDiff(m, ops_[static_cast<std::size_t>(k)].L);
        if (d < best) {
          best = d;
          found = k;
        }
      }
      mul_[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] = static_cast<std::uint8_t>(found);
    }
  }
  for (int a = 0; a < size_; ++a) {
    for (int b = 0; b < size_; ++b) {
      if (mul(a, b) == 0) {
        inv_[static_cast<std::size_t>(a)] = static_cast<std::uint8_t>(b);
        break;
      }
    }
    // The cyclic subgroup of a: the mean of its powers projects onto what a fixes.
    Mat3 sum{1.0f};
    int count = 1;
    for (int x = a; x != 0 && count < kMaxSymmetryCopies; x = mul(x, a)) {
      sum += ops_[static_cast<std::size_t>(x)].L;
      ++count;
    }
    fix_[static_cast<std::size_t>(a)] = clean(sum * (1.0f / static_cast<float>(count)));
  }
  for (int k = 1; k < size_; ++k) {
    const SymmetryOp& o = ops_[static_cast<std::size_t>(k)];
    for (int a = 0; a < 3; ++a) {
      Mat3 mirror{1.0f};
      mirror[a][a] = -1.0f;
      if (maxAbsDiff(o.L, mirror) == 0.0f) planes_ |= static_cast<std::uint8_t>(1 << a);
      Mat3 line{0.0f};
      line[a][a] = 1.0f;
      if (o.det > 0.0f && maxAbsDiff(fix_[static_cast<std::size_t>(k)], line) == 0.0f)
        axes_ |= static_cast<std::uint8_t>(1 << a);
    }
  }
}

OpMask SymmetryGroup::closure(OpMask generators) const {
  generators &= all();
  OpMask members = OpMask{1} | generators;
  for (;;) {
    OpMask next = members;
    for (int a = 0; a < size_; ++a) {
      if (!(members >> a & 1)) continue;
      for (int g = 0; g < size_; ++g)
        if (generators >> g & 1) next |= OpMask{1} << mul(a, g);
    }
    if (next == members) return members;
    members = next;
  }
}

Mat3 SymmetryGroup::projector(OpMask h) const {
  Mat3 sum{0.0f};
  int count = 0;
  for (int k = 0; k < size_; ++k) {
    if (!(h >> k & 1)) continue;
    sum += ops_[static_cast<std::size_t>(k)].L;
    ++count;
  }
  if (count <= 1) return Mat3{1.0f};
  return clean(sum * (1.0f / static_cast<float>(count)));
}

Mat3 SymmetryGroup::axialProjector(OpMask h) const {
  Mat3 sum{0.0f};
  int count = 0;
  for (int k = 0; k < size_; ++k) {
    if (!(h >> k & 1)) continue;
    sum += ops_[static_cast<std::size_t>(k)].L * ops_[static_cast<std::size_t>(k)].det;
    ++count;
  }
  if (count <= 1) return Mat3{1.0f};
  return clean(sum * (1.0f / static_cast<float>(count)));
}

OpMask SymmetryGroup::stabilizer(const Vec3& p, float eps) const {
  OpMask generators = 0;
  const float limit = 2.0f * eps;
  for (int k = 1; k < size_; ++k)
    if (glm::length(transform(ops_[static_cast<std::size_t>(k)].L, p) - p) <= limit) generators |= OpMask{1} << k;
  return closure(generators);
}

Vec3 CopyCluster::projectLocal(const Vec3& v) const { return merged() ? transform(localProjector, v) : v; }

int clusterCopies(const SymmetryGroup& group, const Vec3& c, float mergeDist, CopyClusters& out) {
  // An op merges with the primary copy when its copy lies within the merge distance, or the centre
  // lies within it of what the op fixes: a mirror merges within that distance of its plane (as
  // the X mirror always did), small turns already when neighbouring copies come that close.
  OpMask near = 0;
  for (int k = 1; k < group.size(); ++k) {
    const float apart = glm::length(transform(group.op(k).L, c) - c);
    const float offFixed = glm::length(c - transform(group.fixProjector(k), c));
    if (std::min(apart, offFixed) < mergeDist) near |= OpMask{1} << k;
  }
  return clustersOf(group, group.closure(near), c, out);
}

int clustersOf(const SymmetryGroup& group, OpMask subgroup, const Vec3& c, CopyClusters& out) {
  subgroup = group.closure(subgroup);
  const bool merged = subgroup != 1;
  const Mat3 P = merged ? group.projector(subgroup) : Mat3{1.0f};
  const Mat3 Q = merged ? group.axialProjector(subgroup) : Mat3{1.0f};
  const Vec3 fixed = merged ? transform(P, c) : c;
  OpMask assigned = 0;
  int n = 0;
  for (int k = 0; k < group.size(); ++k) {
    if (assigned >> k & 1) continue;
    OpMask coset = 0;
    for (int h = 0; h < group.size(); ++h)
      if (subgroup >> h & 1) coset |= OpMask{1} << group.mul(k, h);
    assigned |= coset;
    const SymmetryOp& o = group.op(k);
    CopyCluster& cl = out[static_cast<std::size_t>(n++)];
    cl.op = k;
    cl.members = coset;
    cl.subgroup = subgroup;
    cl.center = k == 0 ? fixed : transform(o.L, fixed);
    cl.polar = merged ? o.L * P : o.L;
    cl.axial = (merged ? o.L * Q : o.L) * o.det;
    cl.localProjector = merged ? clean(o.L * P * glm::transpose(o.L)) : Mat3{1.0f};
  }
  return n;
}

SeamPin seamPin(const SymmetryGroup& group, const Vec3& p, float eps) {
  SeamPin pin;
  pin.P = group.projector(group.stabilizer(p, eps));
  for (int a = 0; a < 3; ++a)
    if (pin.P[a][0] == 0.0f && pin.P[a][1] == 0.0f && pin.P[a][2] == 0.0f) pin.zeroAxes |= static_cast<std::uint8_t>(1 << a);
  return pin;
}

Vec3 snapPinned(const SeamPin& pin, const Vec3& start, const Vec3& p) {
  Vec3 q = transform(pin.P, start + transform(pin.P, p - start));
  for (int a = 0; a < 3; ++a)
    if (pin.zeroAxes >> a & 1) q[a] = 0.0f;
  return q;
}

std::uint8_t pinnedAxes(std::uint8_t planes, std::uint8_t axes, const Vec3& p, float eps) {
  std::uint8_t bits = 0;
  for (int a = 0; a < 3; ++a) {
    if ((planes >> a & 1) && std::abs(p[a]) <= eps) bits |= static_cast<std::uint8_t>(1 << a);
    const int b = (a + 1) % 3, c = (a + 2) % 3;
    if ((axes >> a & 1) && std::abs(p[b]) <= eps && std::abs(p[c]) <= eps)
      bits |= static_cast<std::uint8_t>((1 << b) | (1 << c));
  }
  return bits;
}

}  // namespace plegl
