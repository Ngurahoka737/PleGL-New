#include "sculpt/LayerOps.h"

#include <cmath>

namespace plegl {

namespace {
bool movesVertices(const Brush* brush) { return !brush || (!brush->editsMask() && !brush->editsFaceSets()); }
}  // namespace

std::string layerTargetRefusal(const SceneObject& object, const Brush* brush, std::uint32_t target) {
  if (!movesVertices(brush)) return {};
  const LayerStack& stack = object.mesh.layers;
  const bool erase = brush && dynamic_cast<const EraseLayerBrush*>(brush);
  if (stack.empty() || target == 0) {
    return erase ? "Erase Layer works on a layer. Pick a layer under Sculpt layers." : std::string{};
  }
  const SculptLayer* layer = stack.find(target);
  if (!layer) return "Pick a layer under Sculpt layers.";
  if (!layer->visible) return "Layer '" + layer->name + "' is hidden. Show it (L) to sculpt on it.";
  if (!(std::abs(layer->strength) >= kMinStrokeStrength)) {
    const long percent = std::lround(layer->strength * 100.0f);
    return "Layer '" + layer->name + "' is at " + std::to_string(percent) +
           " %. Raise its strength above 5 % to sculpt on it.";
  }
  return {};
}

std::string layerStrokeRefusal(const SceneObject& object, const Brush* brush, const StrokeOptions& options) {
  // Grab never runs dynamic topology, and subdivision levels never do either.
  if (brush && movesVertices(brush) && options.dyntopo && !object.multires && !object.mesh.layers.empty())
    return "Dynamic topology is off while this object has sculpt layers. Use Layers > Apply All Layers to sculpt "
           "with it.";
  return layerTargetRefusal(object, brush ? &strokeBrush(*brush, object, options) : nullptr, options.layerTarget);
}

const Brush& strokeBrush(const Brush& brush, const SceneObject& object, const StrokeOptions& options) {
  static const LayerSmoothBrush layerSmooth;
  if (options.smoothLayerOnly && options.layerTarget != 0 && object.mesh.layers.find(options.layerTarget) &&
      dynamic_cast<const SmoothBrush*>(&brush))
    return layerSmooth;
  return brush;
}

}  // namespace plegl
