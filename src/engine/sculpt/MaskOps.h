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
// the changed GPU ranges. Returns an undo entry holding only the leaves whose mask changed, or
// nothing if no value changed (for example Clear on an unmasked object).
std::optional<SculptUndo> applyMaskOp(SceneObject& object, MaskOp op, int iterations = 1);

}  // namespace plegl
