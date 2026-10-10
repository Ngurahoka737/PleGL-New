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

// One sculpt stroke, mask edit or face set edit: only the leaves it touched, before and after. A
// stroke over a 1M vertex mesh that touches 20 leaves stores about 20 * 1024 vertices, not the
// whole mesh.
struct SculptUndo {
  std::string label;
  std::uint32_t objectId = 0;
  std::uint64_t topologyVersion = 0;  // The entry only applies to this exact vertex order.
  // Sculpt layers: the array the stroke committed into (a layer id, or 0 for the base) and the
  // layer state key it was recorded under (0 without layers). Entries that hold positions or layer
  // values apply only under the same key, because their positions are composites of that state.
  std::uint32_t layerTarget = 0;
  std::uint64_t layerKey = 0;
  std::vector<LeafState> before;
  std::vector<LeafState> after;
  // Leaves whose bounds the stroke changed although none of their own vertices did (their faces
  // use vertices of other leaves). They are refit along with the leaves above.
  std::vector<Index> refit;
  std::size_t bytes() const;
};

// A whole mesh with its BVH, as one topology state of an object, with the object's subdivision
// levels if it had any (the mesh is then the active level).
struct MeshState {
  Mesh mesh;
  Bvh bvh;
  std::uint64_t topologyVersion = 0;
  std::shared_ptr<const Multires> multires;
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

// Copies the channels `states` recorded into `object` leaf by leaf, marks the changed GPU ranges
// and refits the moved leaves (plus entry.refit). The caller checks that the entry belongs to
// this object and topology version.
void applySculptStates(SceneObject& object, const SculptUndo& entry, const std::vector<LeafState>& states);
// True if `states` of `entry` may be applied to `object`: same topology version and, when they hold
// positions or layer values, the same layer state key and an existing target array.
bool sculptUndoApplies(const SceneObject& object, const SculptUndo& entry, const std::vector<LeafState>& states);

// What a sculpt stroke leaves for undo.
using StrokeUndo = std::variant<SculptUndo, DyntopoUndo>;

// Multiresolution commands (see multires/MultiresOps.h). Every change of the active level is an
// undo step of its own, so the history stays linear: a stroke is always undone on the level, leaf
// layout and topology version it was recorded on.
enum class MultiresOp : std::uint8_t {
  Switch,        // Changed the active level.
  Subdivide,     // Added a level on top and made it active.
  DeleteHigher,  // Dropped the levels above the active one.
  DeleteLower,   // Dropped the levels below the active one, which became the base.
  AllLevels,     // Ran a mask or face set operation on every level.
};

struct MultiresUndo {
  std::string label;
  std::uint32_t objectId = 0;
  MultiresOp op = MultiresOp::Switch;
  int levelBefore = 0, levelAfter = 0;  // Active level on each side.
  int countBefore = 0, countAfter = 0;  // Level count on each side; 0 for an object without levels.
  std::uint64_t versionBefore = 0, versionAfter = 0;  // The object's topologyVersion on each side.
  SyncDelta sync;                                     // Pending edits the command folded in.
  std::vector<std::pair<int, SculptUndo>> perLevel;   // AllLevels: what changed on each level.
  std::vector<MultiresLevel> held;                    // Levels that exist on one side only.
  SubdivisionLinks heldLinks;                         // DeleteLower: the new base's old links.
  std::shared_ptr<Multires> heldStack;                // The stack while the object has none.
  std::size_t bytes() const;
};

// Sculpt layer operations (see sculpt/LayerOps.h).
enum class LayerOp : std::uint8_t {
  Add, Duplicate, Delete, Rename, Visibility, Strength, Solo, ShowAll, HideAll, Invert, MergeDown, Apply,
  ApplyAll,
};

// A layer's settings, without its offsets. == compares the strength's bits.
struct LayerMeta {
  std::uint32_t id = 0;
  std::string name;
  float strength = 1.0f;
  bool visible = true;
  bool operator==(const LayerMeta& o) const;
};

// Everything about a layer stack except its arrays, on one side of a layer operation.
struct LayerSide {
  bool hasStack = false;
  std::uint64_t epoch = 0;
  std::uint32_t nextId = 1;
  std::uint32_t active = 0;      // Restored, but never compared: selecting is not an edit.
  std::vector<LayerMeta> layers;  // Bottom to top.
  // Equal apart from `active`.
  bool sameState(const LayerSide& o) const;
};

// An array the live stack does not have right now: a whole array (a layer that exists on the other
// side only, or the other side's contents of an array both sides have), or a patch of values at
// some vertices. Applying the entry swaps these with the live arrays, so the entry always holds
// the side that is not live.
struct HeldArray {
  std::uint32_t id = 0;       // A layer id, or 0 for the base.
  std::vector<Index> index;   // Empty: a whole array. Otherwise a patch at these vertices.
  std::vector<Vec3> values;
};

// One layer operation on one object (or the active level of one, see `level`).
struct LayerUndo {
  std::string label;
  std::uint32_t objectId = 0;
  std::uint64_t topologyVersion = 0;
  int level = -1;  // Active subdivision level when recorded; -1 for an object without levels.
  LayerOp op = LayerOp::Add;
  LayerSide before, after;
  std::vector<HeldArray> held;
  // Positions to recompose after a swap: where these layers (in live or held arrays) or the
  // patches are not zero. Empty when the composite does not change.
  std::vector<std::uint32_t> recomposeIds;
  bool recompose = false;
  // Subdivision levels: reference positions the operation moved along with a rounding-only change
  // of the composite (see rebaseReference), ascending.
  std::vector<Index> refIndex;
  std::vector<Vec3> refBefore, refAfter;
  std::size_t bytes() const;
};

struct SyncWorkspace;
struct LayerWorkspace;

class UndoStack {
 public:
  explicit UndoStack(std::size_t maxBytes = std::size_t{1} << 30) : maxBytes_(maxBytes) {}

  // Adds an entry and drops anything that could be redone. Old entries are dropped once the
  // stack exceeds its memory budget.
  void push(SculptUndo entry);
  void push(TopologyUndo entry);
  void push(DyntopoUndo entry);
  void push(StrokeUndo entry);
  // Adds a multires entry. Level steps that follow each other with nothing in between merge into
  // one entry (a step that ends where the first began disappears), so browsing levels does not
  // flood the history.
  void push(MultiresUndo entry);
  // Adds a layer entry. Eye toggles (visibility, solo, show or hide all) that follow each other
  // on the same object merge into one entry, which disappears when the run ends where it began.
  void push(LayerUndo entry);

  // Scratch memory for dynamic topology undo, shared with the sculptor so only one exists.
  // Without one the stack makes its own on first use.
  void setLayoutWorkspace(std::shared_ptr<LayoutWorkspace> workspace) { workspace_ = std::move(workspace); }
  // Scratch memory for multires syncs, shared with the app's level commands.
  void setSyncWorkspace(std::shared_ptr<SyncWorkspace> workspace) { syncWorkspace_ = std::move(workspace); }
  // Scratch memory for layer operations, shared with the app.
  void setLayerWorkspace(std::shared_ptr<LayerWorkspace> workspace) { layerWorkspace_ = std::move(workspace); }

  // Return the label of what was undone or redone, or empty when nothing applied.
  std::string undo(Scene& scene);
  std::string redo(Scene& scene);

  bool canUndo() const { return cursor_ > 0; }
  bool canRedo() const { return cursor_ < entries_.size(); }
  std::size_t size() const { return entries_.size(); }
  std::size_t bytes() const { return bytes_; }
  void clear();

 private:
  using Entry = std::variant<SculptUndo, TopologyUndo, DyntopoUndo, MultiresUndo, LayerUndo>;
  void pushEntry(Entry entry);
  static std::size_t bytesOf(const Entry& e);
  static bool apply(Scene& scene, const SculptUndo& entry, const std::vector<LeafState>& states);
  static bool apply(Scene& scene, const TopologyUndo& entry, const MeshState& from, const MeshState& to);
  bool apply(Scene& scene, DyntopoUndo& entry, bool redo);
  bool apply(Scene& scene, MultiresUndo& entry, bool redo);
  bool apply(Scene& scene, LayerUndo& entry, bool redo);
  // Drops the oldest undoable entries while over budget.
  void trim();

  std::vector<Entry> entries_;
  std::size_t cursor_ = 0;  // Entries before the cursor are undoable.
  std::size_t bytes_ = 0;
  std::size_t maxBytes_;
  std::shared_ptr<LayoutWorkspace> workspace_;
  std::shared_ptr<SyncWorkspace> syncWorkspace_;
  std::shared_ptr<LayerWorkspace> layerWorkspace_;
};

}  // namespace plegl
