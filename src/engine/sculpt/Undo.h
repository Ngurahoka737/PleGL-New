#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "scene/Scene.h"
#include "spatial/LeafLayout.h"

namespace plegl {

// One sculpt stroke or mask edit: only the leaves it touched, before and after. A stroke over a
// 1M vertex mesh that touches 20 leaves stores about 20 * 1024 vertices, not the whole mesh.
struct SculptUndo {
  std::string label;
  std::uint32_t objectId = 0;
  std::uint64_t topologyVersion = 0;  // The entry only applies to this exact vertex order.
  std::vector<LeafState> before;
  std::vector<LeafState> after;
  std::size_t bytes() const;
};

// A whole mesh with its BVH, as one topology state of an object.
struct MeshState {
  Mesh mesh;
  Bvh bvh;
  std::uint64_t topologyVersion = 0;
  std::size_t bytes() const;
};

// A topology change such as a remesh: the object's complete mesh before and after. Restoring a
// state also restores its topologyVersion, so sculpt entries recorded on that topology apply
// again after undoing the remesh.
struct TopologyUndo {
  std::string label;
  std::uint32_t objectId = 0;
  std::shared_ptr<const MeshState> before;
  std::shared_ptr<const MeshState> after;
  std::size_t bytes() const;
};

// A dynamic topology stroke. Only the leaves it rebuilt are stored, as whole-leaf slices, plus the
// position-only changes it made to other leaves, so the entry costs about as much as the area the
// stroke covered. The after side is captured the first time the stroke is undone (until then it
// is simply the current mesh) and dropped again after a redo.
struct DyntopoUndo {
  std::string label;
  std::uint32_t objectId = 0;
  std::uint64_t afterVersion = 0;  // topologyVersion right after the stroke.
  LayoutDelta delta;
  LayoutSide before;
  std::unique_ptr<LayoutSide> after;
  std::size_t bytes() const;
};

// What a sculpt stroke leaves for undo.
using StrokeUndo = std::variant<SculptUndo, DyntopoUndo>;

class UndoStack {
 public:
  explicit UndoStack(std::size_t maxBytes = std::size_t{1} << 30) : maxBytes_(maxBytes) {}

  // Adds an entry and drops anything that could be redone. Old entries are dropped once the
  // stack exceeds its memory budget.
  void push(SculptUndo entry);
  void push(TopologyUndo entry);
  void push(DyntopoUndo entry);
  void push(StrokeUndo entry);

  // Scratch memory for dynamic topology undo, shared with the sculptor so only one exists.
  // Without one the stack makes its own on first use.
  void setLayoutWorkspace(std::shared_ptr<LayoutWorkspace> workspace) { workspace_ = std::move(workspace); }

  // Return the label of what was undone or redone, or empty when nothing applied.
  std::string undo(Scene& scene);
  std::string redo(Scene& scene);

  bool canUndo() const { return cursor_ > 0; }
  bool canRedo() const { return cursor_ < entries_.size(); }
  std::size_t size() const { return entries_.size(); }
  std::size_t bytes() const { return bytes_; }
  void clear();

 private:
  using Entry = std::variant<SculptUndo, TopologyUndo, DyntopoUndo>;
  void pushEntry(Entry entry);
  static std::size_t bytesOf(const Entry& e);
  static bool apply(Scene& scene, const SculptUndo& entry, const std::vector<LeafState>& states);
  static bool apply(Scene& scene, const TopologyUndo& entry, const MeshState& from, const MeshState& to);
  bool apply(Scene& scene, DyntopoUndo& entry, bool redo);
  // Drops the oldest undoable entries while over budget.
  void trim();

  std::vector<Entry> entries_;
  std::size_t cursor_ = 0;  // Entries before the cursor are undoable.
  std::size_t bytes_ = 0;
  std::size_t maxBytes_;
  std::shared_ptr<LayoutWorkspace> workspace_;
};

}  // namespace plegl
