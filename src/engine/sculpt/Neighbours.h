#pragma once

#include "mesh/Mesh.h"

namespace plegl {

// Calls fn(mean) with the average of value(u) over the neighbours u of v, if v has any.
// Open-border vertices average only along the border, so open meshes do not shrink inward from
// their edges. Shared by the smoothing brushes and the mask filters.
template <typename T, typename Value, typename Fn>
void neighbourMean(const Mesh& m, Index v, Value&& value, Fn&& fn) {
  if (m.vertHe[v] == kInvalid) return;
  T sum{0.0f}, borderSum{0.0f};
  int count = 0, borderCount = 0;
  m.forEachOutgoing(v, [&](Index h) {
    const T q = value(m.heTarget(h));
    sum += q;
    ++count;
    if (m.heTwin[h] == kInvalid) {
      borderSum += q;
      ++borderCount;
    }
    const Index prev = m.hePrev(h);
    if (m.heTwin[prev] == kInvalid) {  // Incoming border edge: its start is a neighbour too.
      borderSum += value(m.heVert[prev]);
      ++borderCount;
    }
  });
  fn(borderCount > 0 ? borderSum / static_cast<float>(borderCount) : sum / static_cast<float>(count));
}

// Like neighbourMean, but border vertices average over all their neighbours. Used for the mask,
// where the border rule would cut the open edge off from the rest of the mask.
template <typename T, typename Value, typename Fn>
void neighbourMeanAll(const Mesh& m, Index v, Value&& value, Fn&& fn) {
  if (m.vertHe[v] == kInvalid) return;
  T sum{0.0f};
  int count = 0;
  m.forEachOutgoing(v, [&](Index h) {
    sum += value(m.heTarget(h));
    ++count;
    const Index prev = m.hePrev(h);
    if (m.heTwin[prev] == kInvalid) {  // Incoming border edge: no outgoing edge reaches its start.
      sum += value(m.heVert[prev]);
      ++count;
    }
  });
  fn(sum / static_cast<float>(count));
}

}  // namespace plegl
