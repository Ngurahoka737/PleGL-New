#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "multires/Propagate.h"
#include "scene/Scene.h"
#include "sculpt/Brush.h"
#include "sculpt/Sculptor.h"
#include "sculpt/Undo.h"

namespace plegl {

// Sculpt layer rules and operations on scene objects. The data model is in mesh/LayerStack.h.
//
// Every operation changes the stack of the object's live mesh (on an object with subdivision
// levels, the active level's), recomposes the positions it affects and returns an undo entry.
// A refused operation returns nothing and sets *error; one that changes nothing returns nothing
// and leaves *error empty. A change of the composite on a subdivision level is a pending edit
// like a stroke and reaches the other levels on the next sync; rounding-only changes (Merge Down,
// Apply) move the level's reference along, so they do not.

// ---- Strokes -----------------------------------------------------------------------------------

// Why a stroke with `brush` (nullptr for Grab) on `object` may not run, or an empty string if it
// may. Checks dynamic topology (which layered meshes cannot use) and the stroke's layer target:
// it must exist, be visible and have a strength of at least kMinStrokeStrength, and Erase Layer
// needs a layer rather than the base. Mask and face set brushes are never refused. The text is
// meant for the status bar.
std::string layerStrokeRefusal(const SceneObject& object, const Brush* brush, const StrokeOptions& options);

// The same without the dynamic topology check. The sculptor uses this one: it turns dynamic
// topology off by itself on layered meshes.
std::string layerTargetRefusal(const SceneObject& object, const Brush* brush, std::uint32_t target);

// The brush a stroke really runs: LayerSmoothBrush in place of SmoothBrush when `options` asks for
// "This layer only" and the target is a layer of `object`, otherwise `brush` itself.
const Brush& strokeBrush(const Brush& brush, const SceneObject& object, const StrokeOptions& options);

// ---- Recomposing -------------------------------------------------------------------------------

// Scratch memory for layer operations, kept between them.
struct LayerWorkspace {
  std::vector<Index> support, changed, ring, leaves, faces;
  std::vector<Vec3> next;  // New positions, parallel to the vertices being recomposed.
  std::vector<Vec3> oldP;  // Old positions, parallel to `changed`.
  std::vector<Vec3> faceN;
  SyncStamps faceMark, vertMark;
  std::size_t bytes() const;
  void release() { *this = LayerWorkspace{}; }
};

enum class NormalsMode : std::uint8_t { Now, Defer };

// Sets positions[v] to the composite for every v in `verts` (ascending, no repeats, not one of
// the workspace's own lists). The vertices whose bits changed end up in ws.changed (old positions
// in ws.oldP). With NormalsMode::Now their normals and those of every vertex sharing a face with
// them are recomputed; the leaves of their faces are refit when `refit` is set; the GPU ranges
// are marked either way. Returns the number of changed vertices.
std::size_t recomposeVertices(SceneObject& object, std::span<const Index> verts, LayerWorkspace& ws, NormalsMode mode,
                              bool refit);

// The second half of recomposeVertices for positions already written: normals (when `normals`)
// and bounds (when `refit`) around `changed`, then GPU marks. `changed` may be a superset.
void finishPositions(SceneObject& object, std::span<const Index> changed, LayerWorkspace& ws, bool normals, bool refit);

// Normals of `verts` (ascending) from their faces, bit for bit what Mesh::computeNormals gives.
void leafNormals(Mesh& mesh, std::span<const Index> verts, LayerWorkspace& ws);

// Subdivision levels: for every changed vertex whose reference position still equals its old
// position, moves the reference to the new live position, so a rounding-only change of the
// composite is not a pending edit. Vertices that already had pending edits keep them. Records
// what it moved in `record` when given. `changed` is ascending, `oldP` parallel to it.
void rebaseReference(Multires& stack, std::span<const Index> changed, std::span<const Vec3> oldP, const Mesh& live,
                     LayerUndo* record);

// Layer memory of the object over all its levels.
std::size_t objectLayerBytes(const SceneObject& object);

// ---- Operations --------------------------------------------------------------------------------

// Adds an empty layer on top and makes it the target; the first layer creates the stack. The
// shape does not change. Refused at kMaxLayers layers or when the object's layers would pass
// `byteLimit`.
std::optional<LayerUndo> addLayer(SceneObject& object, LayerWorkspace& ws, std::string* error,
                                  std::size_t byteLimit = kMaxObjectLayerBytes);
// A hidden copy of layer `id`, just above it. The target stays on the original.
std::optional<LayerUndo> duplicateLayer(SceneObject& object, std::uint32_t id, LayerWorkspace& ws, std::string* error,
                                        std::size_t byteLimit = kMaxObjectLayerBytes);
// Removes the layer and what it adds to the shape. Deleting the last layer drops the stack.
std::optional<LayerUndo> deleteLayer(SceneObject& object, std::uint32_t id, LayerWorkspace& ws, std::string* error);
std::optional<LayerUndo> renameLayer(SceneObject& object, std::uint32_t id, std::string_view name, std::string* error);
std::optional<LayerUndo> setLayerVisible(SceneObject& object, std::uint32_t id, bool visible, LayerWorkspace& ws,
                                         std::string* error);
// Sets a typed strength (clamped to +-kMaxLayerStrength). Dragging uses StrengthDrag.
std::optional<LayerUndo> setLayerStrength(SceneObject& object, std::uint32_t id, float strength, LayerWorkspace& ws,
                                          std::string* error);
// Shows only layer `id`; when it already is the only visible one, shows every layer.
std::optional<LayerUndo> soloLayer(SceneObject& object, std::uint32_t id, LayerWorkspace& ws, std::string* error);
std::optional<LayerUndo> setAllLayersVisible(SceneObject& object, bool visible, LayerWorkspace& ws,
                                             std::string* error);
// Negates the strength, which turns the layer inside out without touching its offsets.
std::optional<LayerUndo> invertLayer(SceneObject& object, std::uint32_t id, LayerWorkspace& ws, std::string* error);
// Folds layer `id` into the layer below it, which keeps its place and takes strength 1.
std::optional<LayerUndo> mergeLayerDown(SceneObject& object, std::uint32_t id, LayerWorkspace& ws, std::string* error);
// Folds layer `id` into the base. Applying the last layer drops the stack.
std::optional<LayerUndo> applyLayer(SceneObject& object, std::uint32_t id, LayerWorkspace& ws, std::string* error);
// Makes the current shape the mesh and drops every layer, hidden ones too.
std::optional<LayerUndo> applyAllLayers(SceneObject& object, LayerWorkspace& ws, std::string* error);
// Masks the mesh by how far layer `id` moves each vertex: 1 where it moves most, 0 where not at all.
std::optional<SculptUndo> maskFromLayer(SceneObject& object, std::uint32_t id, std::string* error);
// Makes `id` (a layer, or 0 for the base) the target of strokes. Not an edit: no undo entry.
bool selectLayer(SceneObject& object, std::uint32_t id);

// A strength drag on the layer panel: positions follow the slider every frame, and the whole
// drag becomes one undo entry. The object must stay alive until end() or cancel().
class StrengthDrag {
 public:
  // Above `liveNormalsLimit` affected vertices, normals wait for end() (tests lower it).
  bool begin(SceneObject& object, std::uint32_t id, LayerWorkspace& ws, std::string* error,
             std::size_t liveNormalsLimit = kLiveNormalsLimit);
  // At most once per frame. After an update that took long the next one only stores its value,
  // which the following update (or end) applies, so the panel stays responsive.
  void update(float strength);
  // Final normals and bounds; nothing when the strength ended where it began.
  std::optional<LayerUndo> end();
  // Puts the starting strength back.
  void cancel();
  bool active() const { return object_ != nullptr; }
  std::uint32_t objectId() const { return entry_.objectId; }
  std::uint32_t layerId() const { return id_; }

 private:
  void apply(float strength);
  void finish();

  SceneObject* object_ = nullptr;
  LayerWorkspace* ws_ = nullptr;
  std::uint32_t id_ = 0;
  std::size_t liveNormalsLimit_ = kLiveNormalsLimit;
  float start_ = 1.0f;
  float pending_ = 0.0f;
  bool hasPending_ = false;
  bool skipNext_ = false;
  bool deferred_ = false;  // Some update left normals for the end.
  bool moved_ = false;     // Some update recomposed positions.
  LayerUndo entry_;
  std::vector<Index> support_;
};

// Undoes (or redoes) a layer operation: checks the object still has the side the entry left,
// swaps the arrays and settings with the entry, recomposes and restores reference rows. Returns
// false, changing nothing, when the entry does not apply.
bool applyLayerUndo(Scene& scene, LayerUndo& entry, bool redo, LayerWorkspace& ws);

// The settings of a stack, for undo entries.
LayerSide layerSide(const LayerStack& stack);

}  // namespace plegl
