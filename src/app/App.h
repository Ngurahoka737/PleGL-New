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
#include "sculpt/Sculptor.h"
#include "sculpt/StrokeSampler.h"
#include "sculpt/Undo.h"

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

enum class BrushKind { Draw, Clay, Smooth, Grab, Inflate, Flatten, Crease };
inline constexpr int kBrushCount = 7;
inline constexpr const char* kBrushNames[kBrushCount] = {"Draw", "Clay", "Smooth", "Grab", "Inflate", "Flatten", "Crease"};
inline constexpr const char* kBrushKeys[kBrushCount] = {"D", "C", "S", "G", "I", "T", "Shift+C"};
enum class PressureMap { Strength, Radius, Both, None };

struct SculptSettings {
  BrushKind brush = BrushKind::Draw;
  float radiusPx = 60.0f;      // Screen-space radius, like most sculpting tools.
  float strength[kBrushCount] = {0.5f, 0.5f, 0.5f, 1.0f, 0.5f, 0.5f, 0.5f};  // Per brush.
  Falloff falloff = Falloff::Smooth;
  bool invert = false;         // Brushes subtract instead of add (Ctrl flips it per stroke).
  PressureMap pressure = PressureMap::Strength;
  bool symmetryX = true;       // PRD default for character sculpting.
  float spacing = 0.1f;        // Dab spacing as a fraction of the radius.
};

struct RemeshSettings {
  float voxelSize = 0.01f;  // In object units; about the edge length of the result.
  bool optimizeQuads = true;  // Valence optimisation and relaxation after the voxel remesh.
};

struct FrameStats {
  double frameMs = 0.0;
  double fps = 0.0;
  double raycastUs = 0.0;
  double partialTestMs = 0.0;
  double dabMs = 0.0;           // Last dab, CPU.
  double inputToDabMs = 0.0;    // OS input timestamp to dab applied.
  double inputToFrameMs = 0.0;  // OS input timestamp to the frame showing it being swapped.
  int dabsLastFrame = 0;
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
  SculptSettings sculpt;
  RemeshSettings remesh;
  std::string lastRemeshInfo;
  UndoStack undoStack;
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
  // Voxel remesh of the selected object on a worker thread (Ctrl+R).
  void requestRemesh();
  bool remeshing() const { return remeshObjectId_ != 0; }
  void requestImport();
  void requestExport();
  void deleteSelected();
  void duplicateSelected();
  void frameScene();
  void setVsync(bool on);
  void bumpUnderCursor();  // Debug path for per-leaf GPU updates.
  void undo();
  void redo();
  bool strokeActive() const { return sculptor_.active(); }
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
  void beginStroke(float x, float y, std::uint64_t timestampNs);
  void continueStroke(float x, float y, std::uint64_t timestampNs);
  void endStroke();
  void applySamples(std::uint64_t timestampNs);
  float currentPressure() const;
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
  std::uint32_t remeshObjectId_ = 0;  // Object being remeshed, 0 when idle.
  int pendingImports_ = 0;

  std::uint64_t lastFrameNs_ = 0;

  // Sculpting.
  Sculptor sculptor_;
  const Brush* brushFor(BrushKind kind) const;  // nullptr for Grab, which has its own stroke path.
  DrawBrush drawBrush_;
  ClayBrush clayBrush_;
  SmoothBrush smoothBrush_;
  InflateBrush inflateBrush_;
  FlattenBrush flattenBrush_;
  CreaseBrush creaseBrush_;
  StrokeSampler sampler_;
  std::vector<StrokeSample> samples_;
  BrushKind strokeBrush_ = BrushKind::Draw;  // Brush of the running stroke (Shift turns it into Smooth).
  // Grab drags the captured region in the plane through the grab point facing the camera.
  Vec3 grabStartWorld_{0.0f};
  Vec3 grabPlaneNormal_{0.0f, 0.0f, 1.0f};
  bool adjustingRadius_ = false;     // F held.
  std::uint64_t oldestInputThisFrameNs_ = 0;
  int dabsThisFrame_ = 0;
};

}  // namespace plegl
