#pragma once

#include <optional>

#include "sculpt/Undo.h"

namespace plegl {

enum class MaskOp {
  Invert,   // m -> 1 - m (masks everything when nothing is masked yet).
  Clear,    // m -> 0
  Fill,     // m -> 1
  Blur,     // Softens mask edges: each value moves halfway to its neighbours' mean.
  Sharpen,  // Hardens mask edges: each value moves away from its neighbours' mean.
};

const char* maskOpName(MaskOp op);

// Applies `op` to the whole mask of `object` (`iterations` times for Blur and Sharpen) and marks
// the changed GPU ranges. Vertices used only by hidden faces keep their values. Returns an undo entry holding only the leaves whose mask changed, or
// nothing if no value changed (for example Clear on an unmasked object).
std::optional<SculptUndo> applyMaskOp(SceneObject& object, MaskOp op, int iterations = 1);

// Finishes a whole-mesh mask edit: puts back `old` values on vertices used only by hidden faces,
// marks the changed GPU ranges and returns an undo entry with the leaves whose mask changed, or
// nothing if none did. `old` is the mask before the edit (same size as the current one).
std::optional<SculptUndo> recordMaskEdit(SceneObject& object, const std::vector<float>& old, std::string label);

}  // namespace plegl
