#include "render/LeafIndexBlocks.h"

#include <algorithm>
#include <cstring>

#include "core/Parallel.h"

namespace plegl {

void appendLeafTriangles(const Mesh& m, const BvhLeaf& leaf, std::vector<std::uint32_t>& out) {
  for (Index f = leaf.faceBegin; f < leaf.faceEnd; ++f) {
    const Index h0 = m.faceHe[f];
    if (h0 == kInvalid) continue;
    const auto a = static_cast<std::uint32_t>(m.heVert[h0]);
    for (Index h = m.heNext[h0]; m.heNext[h] != h0; h = m.heNext[h]) {
      out.push_back(a);
      out.push_back(static_cast<std::uint32_t>(m.heVert[h]));
      out.push_back(static_cast<std::uint32_t>(m.heTarget(h)));
    }
  }
}

void appendLeafEdges(const Mesh& m, const BvhLeaf& leaf, std::vector<std::uint32_t>& out) {
  for (Index h = leaf.heBegin; h < leaf.heEnd; ++h) {
    if (m.heFace[h] == kInvalid) continue;
    const Index t = m.heTwin[h];
    if (t != kInvalid && t < h) continue;
    out.push_back(static_cast<std::uint32_t>(m.heVert[h]));
    out.push_back(static_cast<std::uint32_t>(m.heTarget(h)));
  }
}

void appendLeafIndices(LeafIndexKind kind, const Mesh& m, const BvhLeaf& leaf, std::vector<std::uint32_t>& out) {
  if (kind == LeafIndexKind::Triangles) {
    appendLeafTriangles(m, leaf, out);
  } else {
    appendLeafEdges(m, leaf, out);
  }
}

std::vector<std::uint32_t> buildLeafIndices(LeafIndexKind kind, const Mesh& m, const Bvh& bvh, LeafIndexBlocks& blocks) {
  const auto leaves = bvh.leaves();
  std::vector<std::vector<std::uint32_t>> perLeaf(leaves.size());
  parallelFor(0, leaves.size(), 8, [&](std::size_t b, std::size_t e) {
    for (std::size_t l = b; l < e; ++l) appendLeafIndices(kind, m, leaves[l], perLeaf[l]);
  });
  std::vector<std::uint32_t> counts(leaves.size());
  for (std::size_t l = 0; l < leaves.size(); ++l) counts[l] = static_cast<std::uint32_t>(perLeaf[l].size());
  blocks.pack(counts);
  std::vector<std::uint32_t> out(blocks.size());
  parallelFor(0, leaves.size(), 8, [&](std::size_t b, std::size_t e) {
    for (std::size_t l = b; l < e; ++l)
      if (!perLeaf[l].empty())
        std::memcpy(out.data() + blocks.block(static_cast<Index>(l)).first, perLeaf[l].data(),
                    perLeaf[l].size() * sizeof(std::uint32_t));
  });
  return out;
}

void LeafIndexBlocks::pack(std::span<const std::uint32_t> counts) {
  blocks_.resize(counts.size());
  free_.clear();
  std::uint32_t at = 0;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    blocks_[i] = {at, counts[i], counts[i]};
    at += counts[i];
  }
  end_ = at;
}

bool LeafIndexBlocks::place(Index leaf, std::uint32_t count) {
  const auto l = static_cast<std::size_t>(leaf);
  bool moved = false;
  if (l >= blocks_.size()) {
    blocks_.resize(l + 1);
    moved = true;
  }
  Block& b = blocks_[l];
  if (count <= b.capacity) {
    b.count = count;
    return moved;
  }
  release(b.first, b.capacity);
  // Half again as much room, so a leaf that keeps growing moves only now and then.
  const std::uint32_t room = count + count / 2 + 16;
  b = {allocate(room), room, count};
  return true;
}

void LeafIndexBlocks::truncate(std::size_t leafCount) {
  while (blocks_.size() > leafCount) {
    release(blocks_.back().first, blocks_.back().capacity);
    blocks_.pop_back();
  }
}

std::uint64_t LeafIndexBlocks::used() const {
  std::uint64_t n = 0;
  for (const Block& b : blocks_) n += b.count;
  return n;
}

void LeafIndexBlocks::release(std::uint32_t first, std::uint32_t size) {
  if (size == 0) return;
  // Insert in order and merge with touching neighbours.
  auto it = std::lower_bound(free_.begin(), free_.end(), std::pair{first, 0u});
  it = free_.insert(it, {first, size});
  if (it + 1 != free_.end() && it->first + it->second == (it + 1)->first) {
    it->second += (it + 1)->second;
    free_.erase(it + 1);
  }
  if (it != free_.begin() && (it - 1)->first + (it - 1)->second == it->first) {
    (it - 1)->second += it->second;
    it = free_.erase(it) - 1;
  }
  // Room at the very end goes back to the end.
  if (it->first + it->second == end_) {
    end_ = it->first;
    free_.erase(it);
  }
}

std::uint32_t LeafIndexBlocks::allocate(std::uint32_t size) {
  for (auto it = free_.begin(); it != free_.end(); ++it) {
    if (it->second < size) continue;
    const std::uint32_t first = it->first;
    it->first += size;
    it->second -= size;
    if (it->second == 0) free_.erase(it);
    return first;
  }
  const std::uint32_t first = end_;
  end_ += size;
  return first;
}

void LeafIndexBlocks::runs(std::vector<std::uint32_t>& first, std::vector<std::uint32_t>& count) const {
  first.clear();
  count.clear();
  std::vector<std::pair<std::uint32_t, std::uint32_t>> used;
  used.reserve(blocks_.size());
  for (const Block& b : blocks_)
    if (b.count > 0) used.emplace_back(b.first, b.count);
  std::sort(used.begin(), used.end());
  for (const auto& [f, c] : used) {
    if (!first.empty() && first.back() + count.back() == f) {
      count.back() += c;
    } else {
      first.push_back(f);
      count.push_back(c);
    }
  }
}

bool LeafIndexBlocks::consistent() const {
  std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;
  for (const Block& b : blocks_) {
    if (b.count > b.capacity) return false;
    if (b.capacity > 0) ranges.emplace_back(b.first, b.capacity);
  }
  ranges.insert(ranges.end(), free_.begin(), free_.end());
  std::sort(ranges.begin(), ranges.end());
  std::uint64_t at = 0;
  for (const auto& [f, s] : ranges) {
    if (f < at) return false;
    at = static_cast<std::uint64_t>(f) + s;
  }
  return at <= end_;
}

}  // namespace plegl
