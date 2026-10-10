#pragma once

#include <cstdint>
#include <string>

#include "scene/Scene.h"
#include "sculpt/Brush.h"
#include "sculpt/Sculptor.h"

namespace plegl {

// Sculpt layer rules and operations on scene objects. The data model is in mesh/LayerStack.h.

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

}  // namespace plegl
