#include "mesh/LayerStack.h"

#include <algorithm>
#include <numeric>

#include "core/Parallel.h"

namespace plegl {

int LayerStack::indexOf(std::uint32_t id) const {
  for (std::size_t k = 0; k < list.size(); ++k) {
    if (list[k].id == id) return static_cast<int>(k);
  }
  return -1;
}

SculptLayer* LayerStack::find(std::uint32_t id) {
  const int k = indexOf(id);
  return k < 0 ? nullptr : &list[static_cast<std::size_t>(k)];
}

const SculptLayer* LayerStack::find(std::uint32_t id) const {
  const int k = indexOf(id);
  return k < 0 ? nullptr : &list[static_cast<std::size_t>(k)];
}

std::vector<Vec3>* LayerStack::array(std::uint32_t id) {
  if (empty()) return nullptr;
  if (id == 0) return &base;
  SculptLayer* l = find(id);
  return l ? &l->offset : nullptr;
}

const std::vector<Vec3>* LayerStack::array(std::uint32_t id) const {
  return const_cast<LayerStack*>(this)->array(id);
}

std::size_t LayerStack::bytes() const {
  std::size_t b = base.size() * sizeof(Vec3);
  for (const SculptLayer& l : list) b += l.offset.size() * sizeof(Vec3) + l.name.size() + sizeof(SculptLayer);
  return b;
}

void composeAll(const LayerStack& s, std::vector<Vec3>& out) {
  out.resize(s.base.size());
  parallelFor(0, s.base.size(), 16384, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) out[i] = composeVertex(s, static_cast<Index>(i));
  });
}

void layerSupport(const std::vector<Vec3>& offset, std::vector<Index>& out) {
  out.clear();
  // Two passes over fixed chunks: count, then fill, so the list comes out ascending.
  constexpr std::size_t kChunk = 65536;
  const std::size_t chunks = (offset.size() + kChunk - 1) / kChunk;
  std::vector<std::size_t> count(chunks + 1, 0);
  parallelFor(0, chunks, 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t c = b; c < e; ++c) {
      const std::size_t end = std::min(offset.size(), (c + 1) * kChunk);
      std::size_t n = 0;
      for (std::size_t i = c * kChunk; i < end; ++i) n += isZero(offset[i]) ? 0 : 1;
      count[c + 1] = n;
    }
  });
  for (std::size_t c = 0; c < chunks; ++c) count[c + 1] += count[c];
  out.resize(count[chunks]);
  parallelFor(0, chunks, 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t c = b; c < e; ++c) {
      const std::size_t end = std::min(offset.size(), (c + 1) * kChunk);
      std::size_t at = count[c];
      for (std::size_t i = c * kChunk; i < end; ++i) {
        if (!isZero(offset[i])) out[at++] = static_cast<Index>(i);
      }
    }
  });
}

std::size_t layerSupportSize(const std::vector<Vec3>& offset) {
  constexpr std::size_t kChunk = 65536;
  const std::size_t chunks = (offset.size() + kChunk - 1) / kChunk;
  std::vector<std::size_t> count(chunks, 0);
  parallelFor(0, chunks, 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t c = b; c < e; ++c) {
      const std::size_t end = std::min(offset.size(), (c + 1) * kChunk);
      std::size_t n = 0;
      for (std::size_t i = c * kChunk; i < end; ++i) n += isZero(offset[i]) ? 0 : 1;
      count[c] = n;
    }
  });
  return std::accumulate(count.begin(), count.end(), std::size_t{0});
}

std::uint64_t layerStateKey(const LayerStack& s) {
  if (s.empty()) return 0;
  std::uint64_t h = 1469598103934665603ull;
  auto mix = [&](const void* data, std::size_t n) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
      h ^= p[i];
      h *= 1099511628211ull;
    }
  };
  mix(&s.epoch, sizeof s.epoch);
  for (const SculptLayer& l : s.list) {
    const unsigned char visible = l.visible ? 1 : 0;
    mix(&l.id, sizeof l.id);
    mix(&visible, 1);
    mix(&l.strength, sizeof l.strength);
  }
  return h == 0 ? 1 : h;
}

float targetStrength(const LayerStack& s, std::uint32_t id) {
  if (id == 0) return 1.0f;
  const SculptLayer* l = s.find(id);
  return l ? l->strength : 0.0f;
}

std::string clampLayerName(std::string_view name, std::uint32_t id) {
  auto space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
  while (!name.empty() && space(name.front())) name.remove_prefix(1);
  while (!name.empty() && space(name.back())) name.remove_suffix(1);
  std::string out(name.substr(0, std::min(name.size(), kMaxLayerNameBytes)));
  if (out.size() < name.size()) {
    // Cut back to the start of the code point that did not fit (continuation bytes are 10xxxxxx).
    std::size_t end = out.size();
    while (end > 0 && (static_cast<unsigned char>(name[end]) & 0xC0) == 0x80) --end;
    out.resize(end);
    while (!out.empty() && space(out.back())) out.pop_back();
  }
  // Control characters would break the panel; replace them.
  for (char& c : out) {
    if (static_cast<unsigned char>(c) < 0x20) c = ' ';
  }
  if (out.empty()) out = "Layer " + std::to_string(id);
  return out;
}

}  // namespace plegl
