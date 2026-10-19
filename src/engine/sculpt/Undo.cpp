#include "sculpt/Undo.h"

#include <algorithm>

namespace plegl {

std::size_t SculptUndo::bytes() const {
  std::size_t n = 0;
  for (const auto* list : {&before, &after})
    for (const LeafState& s : *list)
      n += (s.positions.size() + s.normals.size()) * sizeof(Vec3) + s.mask.size() * sizeof(float) +
           s.faceSets.size() * sizeof(std::int32_t);
  return n + refit.size() * sizeof(Index);
}

std::size_t MeshState::bytes() const {
  const Mesh& m = mesh;
  return (m.positions.size() + m.normals.size()) * sizeof(Vec3) + m.mask.size() * sizeof(float) +
         m.faceSets.size() * sizeof(std::int32_t) +
         (m.heNext.size() + m.heTwin.size() + m.heVert.size() + m.heFace.size() + m.vertHe.size() +
          m.faceHe.size()) * sizeof(Index) +
         bvh.memoryBytes();
}

std::size_t TopologyUndo::bytes() const {
  return (before ? before->bytes() : 0) + (after ? after->bytes() : 0);
}

std::size_t DyntopoUndo::bytes() const {
  return delta.bytes() + before.bytes() + (after ? after->bytes() : 0);
}

std::size_t UndoStack::bytesOf(const Entry& e) {
  return std::visit([](const auto& x) { return x.bytes(); }, e);
}

void UndoStack::push(SculptUndo entry) { pushEntry(std::move(entry)); }
void UndoStack::push(TopologyUndo entry) { pushEntry(std::move(entry)); }
void UndoStack::push(DyntopoUndo entry) { pushEntry(std::move(entry)); }
void UndoStack::push(StrokeUndo entry) {
  std::visit([this](auto&& e) { pushEntry(std::move(e)); }, std::move(entry));
}

void UndoStack::pushEntry(Entry entry) {
  while (entries_.size() > cursor_) {
    bytes_ -= bytesOf(entries_.back());
    entries_.pop_back();
  }
  bytes_ += bytesOf(entry);
  entries_.push_back(std::move(entry));
  cursor_ = entries_.size();
  trim();
}

void UndoStack::trim() {
  // Never drop the newest entry, nor anything that can still be redone.
  while (bytes_ > maxBytes_ && cursor_ > 0 && entries_.size() > 1) {
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
  leaves.reserve(states.size() + entry.refit.size());
  std::size_t maskLeaves = 0, faceSetLeaves = 0;
  for (const LeafState& s : states) {
    maskLeaves += s.mask.empty() ? 0 : 1;
    faceSetLeaves += s.faceSets.empty() ? 0 : 1;
  }
  // A mask operation can touch every leaf; one full mask upload is cheaper than many small ones.
  const bool maskAll = maskLeaves > kMaskDirtyAllLeaves;
  const bool faceSetAll = faceSetLeaves > kMaskDirtyAllLeaves;
  if (maskLeaves > 0) m.ensureMask();  // The mask may have been dropped by a redone remesh.
  if (faceSetLeaves > 0) m.ensureFaceSets();
  for (const LeafState& s : states) {
    const BvhLeaf& leaf = obj->bvh.leaves()[s.leaf];
    std::copy(s.positions.begin(), s.positions.end(), m.positions.begin() + leaf.vertBegin);
    std::copy(s.normals.begin(), s.normals.end(), m.normals.begin() + leaf.vertBegin);
    std::copy(s.mask.begin(), s.mask.end(), m.mask.begin() + leaf.vertBegin);
    if (!s.positions.empty()) {
      obj->markLeafDirty(s.leaf);
      leaves.push_back(s.leaf);
    }
    if (!s.mask.empty() && !maskAll) obj->markMaskDirty(s.leaf);
    if (!s.faceSets.empty()) {
      const auto dst = m.faceSets.begin() + leaf.faceBegin;
      // Hiding or revealing changes which triangles the leaf draws.
      bool visibility = false;
      for (std::size_t i = 0; i < s.faceSets.size() && !visibility; ++i)
        visibility = (dst[static_cast<std::ptrdiff_t>(i)] < 0) != (s.faceSets[i] < 0);
      std::copy(s.faceSets.begin(), s.faceSets.end(), dst);
      if (visibility) obj->markVisibilityDirty(s.leaf);
      if (!faceSetAll) obj->markFaceSetDirty(s.leaf);
    }
  }
  if (maskAll) obj->markMaskDirtyAll();
  if (faceSetAll) obj->markFaceSetDirtyAll();
  if (!leaves.empty()) {
    for (Index l : entry.refit) leaves.push_back(l);
    obj->bvh.refitLeaves(m, leaves);
  }
  return true;
}

bool UndoStack::apply(Scene& scene, const TopologyUndo& entry, const MeshState& from, const MeshState& to) {
  SceneObject* obj = scene.find(entry.objectId);
  if (!obj || obj->topologyVersion != from.topologyVersion) return false;
  obj->mesh = to.mesh;
  obj->bvh = to.bvh;
  obj->topologyVersion = to.topologyVersion;
  obj->clearDirty();
  return true;
}

bool UndoStack::apply(Scene& scene, DyntopoUndo& entry, bool redo) {
  SceneObject* obj = scene.find(entry.objectId);
  if (!obj) return false;
  if (!workspace_) workspace_ = std::make_shared<LayoutWorkspace>();
  if (!redo) {
    if (obj->topologyVersion != entry.afterVersion) return false;
    if (!entry.after) {
      // Undo is last in, first out, so the current mesh is exactly the state after the stroke.
      std::vector<Index> moved;
      for (const LeafState& s : entry.before.posOnly) moved.push_back(s.leaf);
      entry.after = std::make_unique<LayoutSide>(captureSide(obj->mesh, obj->bvh, entry.delta, moved, entry.afterVersion));
      bytes_ += entry.after->bytes();
    }
    if (!relayout(obj->mesh, obj->bvh, entry.before, entry.delta, false, *workspace_)) return false;
    obj->topologyVersion = entry.before.topologyVersion;
  } else {
    if (obj->topologyVersion != entry.before.topologyVersion || !entry.after) return false;
    if (!relayout(obj->mesh, obj->bvh, *entry.after, entry.delta, true, *workspace_)) return false;
    obj->topologyVersion = entry.afterVersion;
    // The current mesh is the after side again; capture it anew if the stroke is undone again.
    bytes_ -= entry.after->bytes();
    entry.after.reset();
  }
  obj->clearDirty();
  return true;
}

std::string UndoStack::undo(Scene& scene) {
  // Skip entries whose object is gone or was rebuilt; they can never apply again.
  while (cursor_ > 0) {
    Entry& entry = entries_[--cursor_];
    if (const auto* s = std::get_if<SculptUndo>(&entry)) {
      if (apply(scene, *s, s->before)) return s->label;
    } else if (const auto* t = std::get_if<TopologyUndo>(&entry)) {
      if (apply(scene, *t, *t->after, *t->before)) return t->label;
    } else if (auto* d = std::get_if<DyntopoUndo>(&entry)) {
      if (apply(scene, *d, false)) {
        std::string label = d->label;
        trim();  // The captured after side may push the stack over budget.
        return label;
      }
    }
  }
  return {};
}

std::string UndoStack::redo(Scene& scene) {
  while (cursor_ < entries_.size()) {
    Entry& entry = entries_[cursor_++];
    if (const auto* s = std::get_if<SculptUndo>(&entry)) {
      if (apply(scene, *s, s->after)) return s->label;
    } else if (const auto* t = std::get_if<TopologyUndo>(&entry)) {
      if (apply(scene, *t, *t->before, *t->after)) return t->label;
    } else if (auto* d = std::get_if<DyntopoUndo>(&entry)) {
      if (apply(scene, *d, true)) return d->label;
    }
  }
  return {};
}

}  // namespace plegl
