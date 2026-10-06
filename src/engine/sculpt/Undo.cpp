#include "sculpt/Undo.h"

#include <algorithm>

namespace plegl {

std::size_t SculptUndo::bytes() const {
  std::size_t n = 0;
  for (const auto* list : {&before, &after})
    for (const LeafState& s : *list) n += (s.positions.size() + s.normals.size()) * sizeof(Vec3);
  return n;
}

std::size_t MeshState::bytes() const {
  const Mesh& m = mesh;
  return (m.positions.size() + m.normals.size()) * sizeof(Vec3) +
         (m.heNext.size() + m.heTwin.size() + m.heVert.size() + m.heFace.size() + m.vertHe.size() +
          m.faceHe.size()) * sizeof(Index) +
         bvh.memoryBytes();
}

std::size_t TopologyUndo::bytes() const {
  return (before ? before->bytes() : 0) + (after ? after->bytes() : 0);
}

std::size_t UndoStack::bytesOf(const Entry& e) {
  return std::visit([](const auto& x) { return x.bytes(); }, e);
}

void UndoStack::push(SculptUndo entry) { pushEntry(std::move(entry)); }
void UndoStack::push(TopologyUndo entry) { pushEntry(std::move(entry)); }

void UndoStack::pushEntry(Entry entry) {
  while (entries_.size() > cursor_) {
    bytes_ -= bytesOf(entries_.back());
    entries_.pop_back();
  }
  bytes_ += bytesOf(entry);
  entries_.push_back(std::move(entry));
  cursor_ = entries_.size();
  while (bytes_ > maxBytes_ && entries_.size() > 1) {
    bytes_ -= bytesOf(entries_.front());
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

bool UndoStack::apply(Scene& scene, const TopologyUndo& entry, const MeshState& from, const MeshState& to) {
  SceneObject* obj = scene.find(entry.objectId);
  if (!obj || obj->topologyVersion != from.topologyVersion) return false;
  obj->mesh = to.mesh;
  obj->bvh = to.bvh;
  obj->topologyVersion = to.topologyVersion;
  obj->dirtyLeaves.clear();
  return true;
}

std::string UndoStack::undo(Scene& scene) {
  // Skip entries whose object is gone or was rebuilt; they can never apply again.
  while (cursor_ > 0) {
    const Entry& entry = entries_[--cursor_];
    if (const auto* s = std::get_if<SculptUndo>(&entry)) {
      if (apply(scene, *s, s->before)) return s->label;
    } else if (const auto* t = std::get_if<TopologyUndo>(&entry)) {
      if (apply(scene, *t, *t->after, *t->before)) return t->label;
    }
  }
  return {};
}

std::string UndoStack::redo(Scene& scene) {
  while (cursor_ < entries_.size()) {
    const Entry& entry = entries_[cursor_++];
    if (const auto* s = std::get_if<SculptUndo>(&entry)) {
      if (apply(scene, *s, s->after)) return s->label;
    } else if (const auto* t = std::get_if<TopologyUndo>(&entry)) {
      if (apply(scene, *t, *t->before, *t->after)) return t->label;
    }
  }
  return {};
}

}  // namespace plegl
