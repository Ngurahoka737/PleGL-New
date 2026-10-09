#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "multires/Propagate.h"
#include "multires/Subdivide.h"
#include "sculpt/FaceSetOps.h"
#include "sculpt/MaskOps.h"
#include "sculpt/Undo.h"

namespace plegl {

// Multiresolution commands on a scene object. Each returns its undo entry, or nothing (leaving the
// object as it was) when there is nothing to do. Commands that need every level current first
// sync the pending edits of the active level (see multires/Propagate.h).

// A subdivision prepared on the main thread and run on a worker: a copy of the top level with
// what subdividing it needs. The worker never touches the scene.
struct SubdivideJob {
  std::uint32_t objectId = 0;
  std::uint64_t version = 0;  // The object's topologyVersion when the job was made.
  std::uint64_t serial = 0;   // Its stack's serial (0 for a plain object).
  bool plain = false;         // The object had no levels: the job also prepares the base.
  int maxLeafFaces = 1024;
  Mesh mesh;                  // Copy of the top level.
  CanonicalMap canon;         // Its canonical map; filled by runSubdivideJob for a plain object.
  std::vector<VertexRule> rule;
  std::vector<std::pair<Index, Index>> baseNonManifold;  // Plain object: the base's fans.
  std::vector<std::pair<Index, Index>> nonManifold;      // The new level's fans.
  std::optional<SubdivisionResult> result;
  std::string error;
};

// Refuses (with a message) when the object is not at its top level or has 8 levels.
std::optional<SubdivideJob> prepareSubdivide(const SceneObject& object, std::string* error = nullptr);
// Worker-safe.
void runSubdivideJob(SubdivideJob& job);
// Main thread: adds the job's level on top and makes it active. Returns nothing if the job failed
// or the object changed since it was prepared.
std::optional<MultiresUndo> finishSubdivide(SceneObject& object, SubdivideJob& job, SyncWorkspace& ws);
// All three in a row, for tests and small meshes.
std::optional<MultiresUndo> subdivideObject(SceneObject& object, SyncWorkspace& ws, std::string* error = nullptr,
                                            int maxLeafFaces = 1024);

// Makes `target` the active level after syncing the pending edits.
std::optional<MultiresUndo> setActiveLevel(SceneObject& object, int target, SyncWorkspace& ws);
// The same with a sync computed elsewhere (on a worker) from the current pending edits.
std::optional<MultiresUndo> setActiveLevel(SceneObject& object, int target, SyncDelta sync, SyncWorkspace& ws);

// Drops every level above the active one (no sync: edits stay pending). From the base the object
// becomes a plain mesh.
std::optional<MultiresUndo> deleteHigherLevels(SceneObject& object);
// Drops every level below the active one, which becomes the base. From the top the object becomes
// a plain mesh.
std::optional<MultiresUndo> deleteLowerLevels(SceneObject& object);

// Runs a mask operation on every level, so mask that exists only on one level is cleared too.
// For Clear, Fill and Invert (Blur and Sharpen act on the active level and propagate).
std::optional<MultiresUndo> applyMaskOpAllLevels(SceneObject& object, MaskOp op, SyncWorkspace& ws);
// Runs a face set operation on every level. For Clear, RevealAll and InvertVisibility.
std::optional<MultiresUndo> applyFaceSetOpAllLevels(SceneObject& object, FaceSetOp op, SyncWorkspace& ws);

// Undoes or redoes a multires entry. Returns false (changing nothing) when the object is not in
// the state the entry expects.
bool applyMultiresUndo(Scene& scene, MultiresUndo& entry, bool redo, SyncWorkspace& ws);

// Checks every invariant of an object's levels: counts, links, canonical maps, rules, frozen
// layouts and versions, reference sizes. Expensive; for tests and debug builds.
ValidationResult validateMultires(const SceneObject& object);

}  // namespace plegl
