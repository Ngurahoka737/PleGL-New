#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "scene/Scene.h"

namespace plegl {

// Vertex data owned by one BVH leaf at one moment.
struct LeafState {
  Index leaf = kInvalid;
  std::vector<Vec3> positions;
  std::vector<Vec3> normals;
};

// One sculpt stroke: only the leaves it touched, before and after. A stroke over a 1M vertex
// mesh that touches 20 leaves stores about 20 * 1024 vertices, not the whole mesh.
struct SculptUndo {
  std::string label;
  std::uint32_t objectId = 0;
  std::uint64_t topologyVersion = 0;  // The entry only applies to this exact vertex order.
  std::vector<LeafState> before;
  std::vector<LeafState> after;
  std::size_t bytes() const;
};

class UndoStack {
 public:
  explicit UndoStack(std::size_t maxBytes = std::size_t{1} << 30) : maxBytes_(maxBytes) {}

  // Adds an entry and drops anything that could be redone. Old entries are dropped once the
  // stack exceeds its memory budget.
  void push(SculptUndo entry);

  // Return the label of what was undone or redone, or empty when nothing applied.
  std::string undo(Scene& scene);
  std::string redo(Scene& scene);

  bool canUndo() const { return cursor_ > 0; }
  bool canRedo() const { return cursor_ < entries_.size(); }
  std::size_t size() const { return entries_.size(); }
  std::size_t bytes() const { return bytes_; }
  void clear();

 private:
  static bool apply(Scene& scene, const SculptUndo& entry, const std::vector<LeafState>& states);

  std::vector<SculptUndo> entries_;
  std::size_t cursor_ = 0;  // Entries before the cursor are undoable.
  std::size_t bytes_ = 0;
  std::size_t maxBytes_;
};

}  // namespace plegl
