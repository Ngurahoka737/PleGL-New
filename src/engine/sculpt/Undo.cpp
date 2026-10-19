#include "sculpt/Undo.h"

#include <algorithm>

namespace plegl {

std::size_t SculptUndo::bytes() const {
  std::size_t n = 0;
  for (const auto* list : {&before, &after})
    for (const LeafState& s : *list) n += (s.positions.size() + s.normals.size()) * sizeof(Vec3);
  return n;
}

void UndoStack::push(SculptUndo entry) {
  while (entries_.size() > cursor_) {
    bytes_ -= entries_.back().bytes();
    entries_.pop_back();
  }
  bytes_ += entry.bytes();
  entries_.push_back(std::move(entry));
  cursor_ = entries_.size();
  while (bytes_ > maxBytes_ && entries_.size() > 1) {
    bytes_ -= entries_.front().bytes();
    entries_.erase(entries_.begin());
    --cursor_;
  }
}

void UndoStack::clear() {
  entries_.clear();
  cursor_ = 0;
  bytes_ = 0;
}

bool UndoStack::apply(Scene& scene, const SculptUndo& entry, const std::vector<LeafState>& states) {
  SceneObject* obj = scene.find(entry.objectId);
  if (!obj || obj->topologyVersion != entry.topologyVersion) return false;
  Mesh& m = obj->mesh;
  std::vector<Index> leaves;
  leaves.reserve(states.size());
  for (const LeafState& s : states) {
    const BvhLeaf& leaf = obj->bvh.leaves()[s.leaf];
    std::copy(s.positions.begin(), s.positions.end(), m.positions.begin() + leaf.vertBegin);
    std::copy(s.normals.begin(), s.normals.end(), m.normals.begin() + leaf.vertBegin);
    obj->markLeafDirty(s.leaf);
    leaves.push_back(s.leaf);
  }
  // Every leaf whose faces moved has its own snapshot, so refitting these is enough.
  obj->bvh.refitLeaves(m, leaves);
  return true;
}

std::string UndoStack::undo(Scene& scene) {
  // Skip entries whose object is gone or was rebuilt; they can never apply again.
  while (cursor_ > 0) {
    const SculptUndo& e = entries_[--cursor_];
    if (apply(scene, e, e.before)) return e.label;
  }
  return {};
}

std::string UndoStack::redo(Scene& scene) {
  while (cursor_ < entries_.size()) {
    const SculptUndo& e = entries_[cursor_++];
    if (apply(scene, e, e.after)) return e.label;
  }
  return {};
}

}  // namespace plegl
