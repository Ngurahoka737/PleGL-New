#include "sculpt/Undo.h"

#include <algorithm>
#include <cstring>

#include "multires/MultiresOps.h"
#include "sculpt/LayerOps.h"

namespace plegl {

std::size_t SculptUndo::bytes() const {
  std::size_t n = 0;
  for (const auto* list : {&before, &after})
    for (const LeafState& s : *list)
      n += (s.positions.size() + s.normals.size() + s.layer.size()) * sizeof(Vec3) + s.mask.size() * sizeof(float) +
           s.faceSets.size() * sizeof(std::int32_t);
  return n + refit.size() * sizeof(Index);
}

std::size_t MeshState::bytes() const {
  const Mesh& m = mesh;
  return (m.positions.size() + m.normals.size()) * sizeof(Vec3) + m.mask.size() * sizeof(float) +
         m.faceSets.size() * sizeof(std::int32_t) +
         (m.heNext.size() + m.heTwin.size() + m.heVert.size() + m.heFace.size() + m.vertHe.size() +
          m.faceHe.size()) * sizeof(Index) +
         m.layers.bytes() + bvh.memoryBytes() + (multires ? multires->bytes() : 0);
}

std::size_t TopologyUndo::bytes() const {
  return (before ? before->bytes() : 0) + (after ? after->bytes() : 0);
}

std::size_t DyntopoUndo::bytes() const {
  return delta.bytes() + before.bytes() + (after ? after->bytes() : 0);
}

std::size_t MultiresUndo::bytes() const {
  std::size_t n = sizeof(MultiresUndo) + sync.bytes() + heldLinks.bytes() + (heldStack ? heldStack->bytes() : 0);
  for (const auto& [level, entry] : perLevel) n += entry.bytes();
  for (const MultiresLevel& l : held) n += l.bytes();
  return n;
}

bool LayerMeta::operator==(const LayerMeta& o) const {
  return id == o.id && name == o.name && std::memcmp(&strength, &o.strength, sizeof strength) == 0 &&
         visible == o.visible;
}

bool LayerSide::sameState(const LayerSide& o) const {
  return hasStack == o.hasStack && epoch == o.epoch && nextId == o.nextId && layers == o.layers;
}

std::size_t LayerUndo::bytes() const {
  std::size_t n = sizeof(LayerUndo) + label.size() + recomposeIds.size() * sizeof(std::uint32_t) +
                  refIndex.size() * sizeof(Index) + (refBefore.size() + refAfter.size()) * sizeof(Vec3);
  for (const LayerSide* side : {&before, &after})
    for (const LayerMeta& l : side->layers) n += sizeof(LayerMeta) + l.name.size();
  for (const HeldArray& h : held) n += sizeof(HeldArray) + h.index.size() * sizeof(Index) + h.values.size() * sizeof(Vec3);
  return n;
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

void UndoStack::push(MultiresUndo entry) {
  if (entry.op == MultiresOp::Switch && entry.sync.empty() && cursor_ == entries_.size() && !entries_.empty()) {
    auto* top = std::get_if<MultiresUndo>(&entries_.back());
    if (top && top->op == MultiresOp::Switch && top->objectId == entry.objectId &&
        top->levelAfter == entry.levelBefore && top->versionAfter == entry.versionBefore &&
        top->countAfter == entry.countBefore) {
      top->levelAfter = entry.levelAfter;
      top->versionAfter = entry.versionAfter;
      top->label = entry.label;
      if (top->levelBefore == top->levelAfter && top->sync.empty()) {
        bytes_ -= bytesOf(entries_.back());
        entries_.pop_back();
        cursor_ = entries_.size();
      }
      return;
    }
  }
  pushEntry(std::move(entry));
}

void UndoStack::push(LayerUndo entry) {
  auto eye = [](LayerOp op) {
    return op == LayerOp::Visibility || op == LayerOp::Solo || op == LayerOp::ShowAll || op == LayerOp::HideAll;
  };
  if (eye(entry.op) && entry.held.empty() && cursor_ == entries_.size() && !entries_.empty()) {
    auto* top = std::get_if<LayerUndo>(&entries_.back());
    if (top && eye(top->op) && top->held.empty() && top->objectId == entry.objectId && top->level == entry.level &&
        top->topologyVersion == entry.topologyVersion && top->after.sameState(entry.before)) {
      bytes_ -= bytesOf(entries_.back());
      top->after = std::move(entry.after);
      top->label = "Layer Visibility";
      // The positions to recompose are those of every layer whose eye changed in the run.
      for (std::uint32_t id : entry.recomposeIds)
        if (std::find(top->recomposeIds.begin(), top->recomposeIds.end(), id) == top->recomposeIds.end())
          top->recomposeIds.push_back(id);
      top->recompose = !top->recomposeIds.empty();
      if (top->before.sameState(top->after)) {
        entries_.pop_back();
        cursor_ = entries_.size();
      } else {
        bytes_ += bytesOf(entries_.back());
      }
      return;
    }
  }
  pushEntry(std::move(entry));
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

void applySculptStates(SceneObject& object, const SculptUndo& entry, const std::vector<LeafState>& states) {
  SceneObject* obj = &object;
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
  std::vector<Vec3>* target = m.layers.array(entry.layerTarget);
  for (const LeafState& s : states) {
    const BvhLeaf& leaf = obj->bvh.leaves()[s.leaf];
    if (target) std::copy(s.layer.begin(), s.layer.end(), target->begin() + leaf.vertBegin);
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
}

bool sculptUndoApplies(const SceneObject& object, const SculptUndo& entry, const std::vector<LeafState>& states) {
  if (object.topologyVersion != entry.topologyVersion) return false;
  bool positions = false, layer = false;
  for (const LeafState& s : states) {
    positions |= !s.positions.empty();
    layer |= !s.layer.empty();
  }
  // Positions are composites of the layer state they were recorded under; mask and face set
  // values do not depend on it.
  if ((positions || layer) && entry.layerKey != layerStateKey(object.mesh.layers)) return false;
  return !layer || object.mesh.layers.array(entry.layerTarget) != nullptr;
}

bool UndoStack::apply(Scene& scene, const SculptUndo& entry, const std::vector<LeafState>& states) {
  SceneObject* obj = scene.find(entry.objectId);
  if (!obj || !sculptUndoApplies(*obj, entry, states)) return false;
  applySculptStates(*obj, entry, states);
  return true;
}

bool UndoStack::apply(Scene& scene, const TopologyUndo& entry, const MeshState& from, const MeshState& to) {
  SceneObject* obj = scene.find(entry.objectId);
  if (!obj || obj->topologyVersion != from.topologyVersion || (obj->multires != nullptr) != (from.multires != nullptr))
    return false;
  obj->mesh = to.mesh;
  obj->bvh = to.bvh;
  obj->topologyVersion = to.topologyVersion;
  // A copy with the same versions: the state stays intact for the next undo or redo.
  obj->multires = to.multires ? std::shared_ptr<Multires>(to.multires->clone(false)) : nullptr;
  obj->clearDirty();
  return true;
}

bool UndoStack::apply(Scene& scene, DyntopoUndo& entry, bool redo) {
  SceneObject* obj = scene.find(entry.objectId);
  // Layouts never carry sculpt layers (dynamic topology refuses them), so the entry cannot apply.
  if (!obj || obj->multires || !obj->mesh.layers.empty()) return false;
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

bool UndoStack::apply(Scene& scene, LayerUndo& entry, bool redo) {
  if (!layerWorkspace_) layerWorkspace_ = std::make_shared<LayerWorkspace>();
  // Arrays move between the object and the entry, so its size changes.
  const std::size_t before = entry.bytes();
  if (!applyLayerUndo(scene, entry, redo, *layerWorkspace_)) return false;
  bytes_ = bytes_ - before + entry.bytes();
  return true;
}

bool UndoStack::apply(Scene& scene, MultiresUndo& entry, bool redo) {
  if (!syncWorkspace_) syncWorkspace_ = std::make_shared<SyncWorkspace>();
  // Held levels move in and out of the entry, so its size changes.
  const std::size_t before = entry.bytes();
  if (!applyMultiresUndo(scene, entry, redo, *syncWorkspace_)) return false;
  bytes_ = bytes_ - before + entry.bytes();
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
    } else if (auto* m = std::get_if<MultiresUndo>(&entry)) {
      // No trim: levels an undo or redo moves into the entry were in the object a moment ago, so
      // memory did not grow, and trimming would throw away older history to pay for redo data.
      if (apply(scene, *m, false)) return m->label;
    } else if (auto* l = std::get_if<LayerUndo>(&entry)) {
      if (apply(scene, *l, false)) return l->label;
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
    } else if (auto* m = std::get_if<MultiresUndo>(&entry)) {
      if (apply(scene, *m, true)) return m->label;
    } else if (auto* l = std::get_if<LayerUndo>(&entry)) {
      if (apply(scene, *l, true)) return l->label;
    }
  }
  return {};
}

}  // namespace plegl
