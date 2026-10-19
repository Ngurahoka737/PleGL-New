#pragma once

#include <cstdint>
#include <optional>

#include "sculpt/Undo.h"

namespace plegl {

enum class FaceSetOp {
  FromMask,          // Visible faces whose corners are all masked (>= 0.5) form a new set.
  FromLooseParts,    // Every connected part gets its own set (hidden faces stay hidden).
  Clear,             // Every face back in the default set (hidden faces stay hidden).
  Hide,              // Hides the faces of the given set.
  Isolate,           // Hides every face outside the given set; when that is already so, shows all.
  RevealAll,         // Shows every hidden face.
  InvertVisibility,  // Hides what is shown and shows what is hidden.
};

const char* faceSetOpName(FaceSetOp op);

// Applies `op` to the face sets of `object` (`faceSet` names the set for Hide and Isolate) and
// marks the changed GPU ranges: colours for face set changes, index data for leaves whose
// visibility changed. Returns an undo entry holding only the leaves whose face sets changed, or
// nothing if nothing changed. Leaves the topology and the vertices alone.
std::optional<SculptUndo> applyFaceSetOp(SceneObject& object, FaceSetOp op, std::int32_t faceSet = 0);

// Builds the undo entry for the leaves whose face sets differ from `old` (the whole array before
// the change) and marks them for upload: colours always, index data where a face was hidden or
// shown. Nothing when no value changed.
std::optional<SculptUndo> recordFaceSets(SceneObject& object, const std::vector<std::int32_t>& old, const char* label);

// Masks (mask 1) every vertex that a visible face of `faceSet` uses, leaving other values alone.
// Returns a mask undo entry, or nothing if no value changed.
std::optional<SculptUndo> maskFaceSet(SceneObject& object, std::int32_t faceSet);

}  // namespace plegl
