#pragma once

#include <SDL3/SDL.h>

#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "Camera.h"
#include "Renderer.h"
#include "scene/Scene.h"

namespace plegl {

enum class Mode { Object, Sculpt };
enum class GizmoOp { Translate, Rotate, Scale };

struct PenState {
  bool inProximity = false;
  bool down = false;
  bool eraser = false;
  float pressure = 0.0f;
  float tiltX = 0.0f;  // Degrees, -90..90.
  float tiltY = 0.0f;
  std::uint64_t lastEventNs = 0;
  std::uint32_t deviceId = 0;
};

struct PrimitiveSettings {
  int uvSegments = 64;
  int uvRings = 32;
  int icoSubdivisions = 5;
  int cubeResolution = 16;
  int quadSphereResolution = 64;
  int planeResolution = 64;
};

struct FrameStats {
  double frameMs = 0.0;
  double fps = 0.0;
  double raycastUs = 0.0;
  double partialTestMs = 0.0;
};

class App {
 public:
  bool init(std::string* error);
  void run();
  void shutdown();

  // ---- Used by the UI ----
  Scene scene;
  Camera camera;
  ViewSettings view;
  PrimitiveSettings primitives;
  Mode mode = Mode::Object;
  GizmoOp gizmo = GizmoOp::Translate;
  std::uint32_t selectedId = 0;
  PenState pen;
  FrameStats stats;
  bool vsync = true;
  bool showDemo = false;
  std::string statusMessage;
  std::string glInfo;

  void addPrimitive(const std::string& name, Mesh mesh);
  void newScene();
  void requestImport();
  void requestExport();
  void deleteSelected();
  void duplicateSelected();
  void frameScene();
  void setVsync(bool on);
  void bumpUnderCursor();  // Debug path for per-leaf GPU updates.
  void importFile(const std::filesystem::path& path);
  void quit() { running_ = false; }
  const RenderStats& renderStats() const { return renderer_.stats(); }
  const std::optional<ScenePick>& hover() const { return hover_; }

 private:
  void handleEvent(const SDL_Event& e);
  void handleShortcut(const SDL_KeyboardEvent& key);
  void updateHover();
  void updateViewportRect();
  void drawUi();
  void drawGizmo();
  void processAsyncResults();
  void exportFile(const std::filesystem::path& path);

  SDL_Window* window_ = nullptr;
  SDL_GLContext gl_ = nullptr;
  Renderer renderer_;
  bool rendererReady_ = false;
  bool running_ = true;

  // Viewport area between the side panels, in window points (top-left origin).
  float vpX_ = 0, vpY_ = 0, vpW_ = 1, vpH_ = 1;
  float pixelScale_ = 1.0f;

  // Navigation.
  enum class Drag { None, Orbit, Pan, Zoom } drag_ = Drag::None;
  Vec3 orbitPivot_{0.0f};
  float mouseX_ = 0, mouseY_ = 0;
  bool mouseInViewport_ = false;
  std::optional<ScenePick> hover_;
  bool hoverDirty_ = true;

  // Background work (file dialogs, imports) hands results to the main thread through here.
  std::mutex asyncMutex_;
  std::deque<std::function<void()>> asyncResults_;
  std::vector<std::jthread> workers_;
  int pendingImports_ = 0;

  std::uint64_t lastFrameNs_ = 0;
};

}  // namespace plegl
