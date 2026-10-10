// Sculpt layer commands of the app: the engine operations (sculpt/LayerOps.h) on the selected
// object, with undo, status messages and the strength slider's drag.
#include <cmath>
#include <cstdio>

#include "App.h"
#include "core/Timer.h"

namespace plegl {

SceneObject* App::layerObject() { return editableObject(selectedId, "sculpt layers"); }

const SculptLayer* App::activeLayer(SceneObject*& obj) {
  obj = layerObject();
  if (!obj) return nullptr;
  const LayerStack& s = obj->mesh.layers;
  const SculptLayer* layer = s.find(s.active);
  if (!layer)
    statusMessage = s.empty() ? obj->name + " has no sculpt layers. Ctrl+L adds one."
                              : std::string("The base is selected. Pick a layer under Sculpt layers.");
  return layer;
}

void App::pushEdit(std::optional<LayerUndo> entry, const char* name, const std::string& error, double ms) {
  stats.maskOpMs = ms;
  if (!entry) {
    statusMessage = error.empty() ? std::string(name) + ": nothing to change" : error;
    return;
  }
  ++editCounter_;
  const std::string label = entry->label;
  undoStack.push(std::move(*entry));
  hover_.reset();  // The surface under the cursor may have moved.
  char buf[128];
  std::snprintf(buf, sizeof(buf), "%s (%.1f ms)", label.c_str(), ms);
  statusMessage = buf;
}

void App::addLayer() {
  SceneObject* obj = layerObject();
  if (!obj) return;
  Timer t;
  std::string error;
  auto entry = plegl::addLayer(*obj, *layerWorkspace_, &error);
  pushEdit(std::move(entry), "Add Layer", error, t.ms());
}

void App::duplicateLayer() {
  SceneObject* obj = nullptr;
  const SculptLayer* layer = activeLayer(obj);
  if (!layer) return;
  Timer t;
  std::string error;
  auto entry = plegl::duplicateLayer(*obj, layer->id, *layerWorkspace_, &error);
  pushEdit(std::move(entry), "Duplicate Layer", error, t.ms());
}

void App::deleteLayer() {
  SceneObject* obj = nullptr;
  const SculptLayer* layer = activeLayer(obj);
  if (!layer) return;
  Timer t;
  std::string error;
  auto entry = plegl::deleteLayer(*obj, layer->id, *layerWorkspace_, &error);
  pushEdit(std::move(entry), "Delete Layer", error, t.ms());
}

void App::renameLayer(std::uint32_t id, const std::string& name) {
  SceneObject* obj = layerObject();
  if (!obj) return;
  std::string error;
  auto entry = plegl::renameLayer(*obj, id, name, &error);
  pushEdit(std::move(entry), "Rename Layer", error, 0.0);
}

void App::setLayerVisible(std::uint32_t id, bool visible) {
  SceneObject* obj = layerObject();
  if (!obj) return;
  Timer t;
  std::string error;
  auto entry = plegl::setLayerVisible(*obj, id, visible, *layerWorkspace_, &error);
  pushEdit(std::move(entry), "Layer Visibility", error, t.ms());
}

void App::toggleActiveLayerVisible() {
  SceneObject* obj = nullptr;
  const SculptLayer* layer = activeLayer(obj);
  if (!layer) return;
  setLayerVisible(layer->id, !layer->visible);
}

void App::soloLayer(std::uint32_t id) {
  SceneObject* obj = layerObject();
  if (!obj) return;
  Timer t;
  std::string error;
  auto entry = plegl::soloLayer(*obj, id, *layerWorkspace_, &error);
  pushEdit(std::move(entry), "Solo Layer", error, t.ms());
}

void App::setAllLayersVisible(bool visible) {
  SceneObject* obj = layerObject();
  if (!obj) return;
  Timer t;
  std::string error;
  auto entry = plegl::setAllLayersVisible(*obj, visible, *layerWorkspace_, &error);
  pushEdit(std::move(entry), visible ? "Show All Layers" : "Hide All Layers", error, t.ms());
}

void App::invertLayer() {
  SceneObject* obj = nullptr;
  const SculptLayer* layer = activeLayer(obj);
  if (!layer) return;
  Timer t;
  std::string error;
  auto entry = plegl::invertLayer(*obj, layer->id, *layerWorkspace_, &error);
  pushEdit(std::move(entry), "Invert Layer", error, t.ms());
}

void App::mergeLayerDown() {
  SceneObject* obj = nullptr;
  const SculptLayer* layer = activeLayer(obj);
  if (!layer) return;
  Timer t;
  std::string error;
  auto entry = plegl::mergeLayerDown(*obj, layer->id, *layerWorkspace_, &error);
  pushEdit(std::move(entry), "Merge Down", error, t.ms());
}

void App::applyLayer() {
  SceneObject* obj = nullptr;
  const SculptLayer* layer = activeLayer(obj);
  if (!layer) return;
  Timer t;
  std::string error;
  auto entry = plegl::applyLayer(*obj, layer->id, *layerWorkspace_, &error);
  pushEdit(std::move(entry), "Apply Layer", error, t.ms());
}

void App::applyAllLayers() {
  SceneObject* obj = layerObject();
  if (!obj) return;
  Timer t;
  std::string error;
  auto entry = plegl::applyAllLayers(*obj, *layerWorkspace_, &error);
  pushEdit(std::move(entry), "Apply All Layers", error, t.ms());
}

void App::maskFromLayer() {
  SceneObject* obj = nullptr;
  const SculptLayer* layer = activeLayer(obj);
  if (!layer) return;
  Timer t;
  std::string error;
  auto entry = plegl::maskFromLayer(*obj, layer->id, &error);
  if (!entry && !error.empty()) {
    statusMessage = error;
    return;
  }
  pushEdit(std::move(entry), "Mask from Layer", t.ms());
}

void App::selectLayer(std::uint32_t id) {
  SceneObject* obj = scene.find(selectedId);
  if (!obj || !plegl::selectLayer(*obj, id)) return;
  const SculptLayer* layer = obj->mesh.layers.find(id);
  statusMessage = "Sculpting on " + (layer ? layer->name : std::string("the base"));
  // Erase Layer has nothing to erase on the base.
  if (!layer && sculpt.brush == BrushKind::EraseLayer) sculpt.brush = BrushKind::Draw;
}

void App::setLayerStrength(std::uint32_t id, float strength) {
  SceneObject* obj = layerObject();
  if (!obj) return;
  Timer t;
  std::string error;
  auto entry = plegl::setLayerStrength(*obj, id, strength, *layerWorkspace_, &error);
  pushEdit(std::move(entry), "Layer Strength", error, t.ms());
}

void App::beginLayerStrengthDrag(std::uint32_t id) {
  SceneObject* obj = layerObject();  // Also ends a drag still running.
  if (!obj) return;
  std::string error;
  if (!strengthDrag_.begin(*obj, id, *layerWorkspace_, &error)) statusMessage = error;
}

void App::updateLayerStrengthDrag(float strength) {
  if (strengthDrag_.active()) strengthDrag_.update(strength);
}

void App::endLayerStrengthDrag() { finishLayerStrengthDrag(); }

void App::finishLayerStrengthDrag() {
  if (!strengthDrag_.active()) return;
  Timer t;
  std::optional<LayerUndo> entry = strengthDrag_.end();
  if (!entry) return;  // Ended where it began.
  pushEdit(std::move(entry), "Layer Strength", {}, t.ms());
}

std::string App::layerStrokeHint() const {
  const SceneObject* obj = scene.find(selectedId);
  if (!obj || obj->mesh.layers.empty()) return {};
  StrokeOptions o;
  o.layerTarget = obj->mesh.layers.active;
  o.smoothLayerOnly = sculpt.smoothLayerOnly;
  o.dyntopo = sculpt.dyntopo;
  return layerStrokeRefusal(*obj, brushFor(sculpt.brush), o);
}

}  // namespace plegl
