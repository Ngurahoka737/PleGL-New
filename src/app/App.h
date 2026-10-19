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
#include "sculpt/FaceSetOps.h"
#include "sculpt/MaskOps.h"
#include "sculpt/Sculptor.h"
#include "sculpt/StrokeSampler.h"
#include "io/Project.h"
#include "multires/MultiresOps.h"
#include "sculpt/LayerOps.h"
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

// Projects store the brush as its index here, so new brushes go at the end. Names are also
// settings keys, so they have no spaces; labels are what the panel shows.
enum class BrushKind { Draw, Clay, Smooth, Grab, Inflate, Flatten, Crease, Mask, FaceSet, EraseLayer };
inline constexpr int kBrushCount = 10;
inline constexpr const char* kBrushNames[kBrushCount] = {"Draw",    "Clay",   "Smooth", "Grab",    "Inflate",
                                                         "Flatten", "Crease", "Mask",   "FaceSet", "EraseLayer"};
inline constexpr const char* kBrushLabels[kBrushCount] = {"Draw",    "Clay",   "Smooth", "Grab",     "Inflate",
                                                          "Flatten", "Crease", "Mask",   "Face Set", "Erase Layer"};
inline constexpr const char* kBrushKeys[kBrushCount] = {"D", "C", "S", "G", "I", "T", "Shift+C", "M", "P", "E"};
enum class PressureMap { Strength, Radius, Both, None };
// How the dynamic topology detail size is given: in screen pixels at the cursor (so zooming in
// adds detail), or as a fixed edge length in object units.
enum class DetailMode { Relative, Constant };

struct SculptSettings {
  BrushKind brush = BrushKind::Draw;
  float radiusPx = 60.0f;      // Screen-space radius, like most sculpting tools.
  float strength[kBrushCount] = {0.5f, 0.5f, 0.5f, 1.0f, 0.5f, 0.5f, 0.5f, 1.0f, 1.0f, 0.5f};  // Per brush.
  Falloff falloff = Falloff::Smooth;
  bool invert = false;         // Brushes subtract instead of add (Ctrl flips it per stroke).
  PressureMap pressure = PressureMap::Strength;
  bool symmetryX = true;       // PRD default for character sculpting.
  float spacing = 0.1f;        // Dab spacing as a fraction of the radius.
  int maskFilterSteps = 2;     // Iterations per Blur Mask / Sharpen Mask.
  // Face set auto-masking for every brush: only the set under the stroke's start changes, and
  // vertices where sets meet stay put.
  bool faceSetAutoMask = false;
  bool lockFaceSetBorders = false;
  // Smooth (and Shift+drag) on a sculpt layer smooths only that layer's own detail.
  bool smoothLayerOnly = false;
  // Dynamic topology (Ctrl+D): brushes add and remove triangles under the cursor so detail can go
  // anywhere. Not used by Grab and Mask.
  bool dyntopo = false;
  DyntopoRefine dyntopoRefine = DyntopoRefine::SplitCollapse;
  DetailMode detailMode = DetailMode::Relative;
  float detailPx = 8.0f;      // Relative: target edge length in screen pixels (hold R to change).
  float detailSize = 0.02f;   // Constant: target edge length in object units.
};

struct RemeshSettings {
  float voxelSize = 0.01f;  // Target edge length of the result, in object units.
  bool optimizeQuads = true;  // Valence optimisation and relaxation after the voxel remesh.
};

struct ProjectSettings {
  float autosaveMinutes = 5.0f;  // 0 turns autosave off.
};

// What to do once the user has answered "save changes?".
enum class PendingAction { None, Quit, NewScene, OpenProject };

struct FrameStats {
  double frameMs = 0.0;
  double fps = 0.0;
  double raycastUs = 0.0;
  double partialTestMs = 0.0;
  double dabMs = 0.0;           // Last dab, CPU.
  double inputToDabMs = 0.0;    // OS input timestamp to dab applied.
  double inputToFrameMs = 0.0;  // OS input timestamp to the frame showing it being swapped.
  int dabsLastFrame = 0;
  double maskOpMs = 0.0;        // Last whole-mesh mask or face set operation (invert, blur, hide...).
  double topologyMs = 0.0;      // Dynamic topology part of the last dab.
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
  ProjectSettings projectSettings;
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
  // Voxel remesh of the selected object on a worker thread (Ctrl+R). Drops subdivision levels.
  void requestRemesh();
  // Background work on one object (remesh, subdivide). Only one runs at a time; the object can not
  // be edited meanwhile, and undo waits.
  enum class Job { None, Remesh, Subdivide };
  Job job() const { return job_; }
  bool busy() const { return job_ != Job::None; }
  bool remeshing() const { return job_ == Job::Remesh; }
  // Subdivision levels (multiresolution) of the selected object. Subdivide adds a Catmull-Clark
  // level on top (on a worker thread; from a lower level it first steps to the top).
  void requestSubdivide();
  // Makes `level` the active one (clamped); edits since the last switch spread to every level.
  void setLevel(int level);
  void stepLevel(int delta);
  void deleteHigherLevels();
  void deleteLowerLevels();
  void requestImport();
  void requestExport();
  // Project files (.psculpt). New, Open and Quit ask about unsaved changes first.
  void requestNewScene();
  void requestOpenProject();
  void requestSaveProject(bool saveAs);
  void requestQuit();
  void openProject(const std::filesystem::path& path);
  // Opens a project or imports a mesh, by extension.
  void openFile(const std::filesystem::path& path);
  bool hasUnsavedChanges() const { return sceneFingerprint() != savedFingerprint_; }
  const std::filesystem::path& projectPath() const { return projectPath_; }
  // Settings stored in project files: brush, camera, symmetry, viewport, remesh, selection.
  std::string settingsText() const;
  void applySettings(const std::string& text);
  void deleteSelected();
  void duplicateSelected();
  void frameScene();
  void setVsync(bool on);
  void bumpUnderCursor();  // Debug path for per-leaf GPU updates.
  void undo();
  void redo();
  // Whole-mesh mask operations on the selected object, undoable.
  void applyMask(MaskOp op);
  bool canEditMask() const;
  // Whole-mesh face set operations (from mask, loose parts, clear, reveal all, invert
  // visibility) on the selected object, undoable. Hide and Isolate act on the set under the
  // cursor; see faceSetOpUnderCursor().
  void applyFaceSets(FaceSetOp op);
  // Hide or isolate the face set under the cursor, on the object under it.
  void faceSetOpUnderCursor(FaceSetOp op);
  // Masks the face set under the cursor (adds to the mask).
  void maskFaceSetUnderCursor();
  bool strokeActive() const { return sculptor_.active(); }
  // Sculpt layers of the selected object (on an object with levels, of the active level). Each is
  // undoable; refusals go to the status line.
  void addLayer();
  void duplicateLayer();
  void deleteLayer();
  void renameLayer(std::uint32_t id, const std::string& name);
  void setLayerVisible(std::uint32_t id, bool visible);
  void toggleActiveLayerVisible();  // L
  void soloLayer(std::uint32_t id);
  void setAllLayersVisible(bool visible);
  void invertLayer();
  void mergeLayerDown();
  void applyLayer();
  void applyAllLayers();
  void maskFromLayer();
  void selectLayer(std::uint32_t id);  // 0: the base.
  // Typed strengths (Ctrl+click on the slider); drags go through the three calls below.
  void setLayerStrength(std::uint32_t id, float strength);
  // A strength slider drag: positions follow every frame, the whole drag is one undo step.
  void beginLayerStrengthDrag(std::uint32_t id);
  void updateLayerStrengthDrag(float strength);
  void endLayerStrengthDrag();
  bool layerStrengthDragActive() const { return strengthDrag_.active(); }
  // Why a stroke on the selected object would be refused right now (empty when it would run).
  std::string layerStrokeHint() const;
  // Whether layer edits on the selected object can run now (no stroke, no job on it).
  bool canEditLayers() const { return canEditMask(); }
  void importFile(const std::filesystem::path& path);
  void quit() { running_ = false; }
  const RenderStats& renderStats() const { return renderer_.stats(); }
  const StrokeTopologyStats& lastStrokeTopology() const { return sculptor_.lastStrokeTopology(); }
  // Detail size the next dab would use at a surface point, in object units (dynamic topology).
  float detailSizeAt(const SceneObject& obj, const Vec3& worldPoint) const;
  const std::optional<ScenePick>& hover() const { return hover_; }

 private:
  void handleEvent(const SDL_Event& e);
  void handleShortcut(const SDL_KeyboardEvent& key);
  void updateHover();
  // The surface under the mouse right now. Keys that act on the face under the cursor use this
  // rather than hover_, which is from the last frame (or from before an orbit), while an undo or
  // the end of a dynamic topology stroke since may have renumbered the faces.
  std::optional<ScenePick> pickUnderMouse() const;
  void updateViewportRect();
  void drawUi();
  void drawLayersMenu();
  void drawLayerPanel(SceneObject& obj);
  void drawGizmo();
  void processAsyncResults();
  void beginStroke(float x, float y, std::uint64_t timestampNs);
  void continueStroke(float x, float y, std::uint64_t timestampNs);
  void endStroke();
  void applySamples(std::uint64_t timestampNs);
  float currentPressure() const;
  void exportFile(const std::filesystem::path& path);
  void drawProjectDialogs();
  void runPendingAction();
  // Serializes on the main thread and writes on a worker. `then` runs on the main thread after a
  // successful write.
  void saveProjectTo(const std::filesystem::path& path, std::function<void()> then = {});
  void applyProject(Project project, const std::filesystem::path& path, bool recovered);
  void loadProjectAsync(const std::filesystem::path& path, bool recovered);
  void autosaveTick();
  void updateWindowTitle();
  std::uint64_t sceneFingerprint() const;
  void initRecovery();
  void finishRecovery();
  // The object a whole-mesh edit of `what` (mask, face sets) may change now, or nullptr with a
  // status message saying why not.
  SceneObject* editableObject(std::uint32_t id, const char* what);
  void pushEdit(std::optional<SculptUndo> entry, const char* name, double ms);
  void pushEdit(std::optional<MultiresUndo> entry, const char* name, double ms);
  // A layer operation's result: the entry, or the refusal in `error` (empty: nothing changed).
  void pushEdit(std::optional<LayerUndo> entry, const char* name, const std::string& error, double ms);
  // The selected object, if a layer edit may change it now (see editableObject). Ends a running
  // strength drag first.
  SceneObject* layerObject();
  // The active layer of the selected object, or nullptr with a status message.
  const SculptLayer* activeLayer(SceneObject*& obj);
  // Ends a running strength drag and records it (before any other edit, undo, save...).
  void finishLayerStrengthDrag();
  // "Wait for the remesh to finish before <action>." when `id` is the busy object (0: any).
  bool waitForJob(std::uint32_t id, const std::string& action);
  // Gives back the sync and layer scratch, which grow with the largest meshes ever edited.
  void releaseSyncScratch();

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
  Job job_ = Job::None;
  std::uint32_t jobObjectId_ = 0;  // Object the job works on.
  int pendingImports_ = 0;

  std::uint64_t lastFrameNs_ = 0;

  // Project files, autosave and crash recovery.
  std::filesystem::path projectPath_;     // Empty until saved or opened.
  std::uint64_t editCounter_ = 0;         // Bumped by strokes and undo/redo (mesh edits).
  std::uint64_t savedFingerprint_ = 0;    // sceneFingerprint() when last saved or opened.
  std::uint64_t autosavedFingerprint_ = 0;
  std::uint64_t lastAutosaveNs_ = 0;
  bool autosaveRunning_ = false;
  bool savingProject_ = false;
  std::filesystem::path dataDir_;         // Per-user folder for autosave and the session lock.
  bool offerRecovery_ = false;            // Previous session ended without a clean exit.
  PendingAction pendingAction_ = PendingAction::None;
  bool askSaveChanges_ = false;
  std::string windowTitle_;

  // Sculpting.
  Sculptor sculptor_;
  // Scratch for spreading edits between subdivision levels, shared with the undo stack.
  std::shared_ptr<SyncWorkspace> syncWorkspace_;
  // Scratch for layer operations, also shared with the undo stack.
  std::shared_ptr<LayerWorkspace> layerWorkspace_;
  StrengthDrag strengthDrag_;
  const Brush* brushFor(BrushKind kind) const;  // nullptr for Grab, which has its own stroke path.
  DrawBrush drawBrush_;
  ClayBrush clayBrush_;
  SmoothBrush smoothBrush_;
  InflateBrush inflateBrush_;
  FlattenBrush flattenBrush_;
  CreaseBrush creaseBrush_;
  MaskBrush maskBrush_;
  MaskSmoothBrush maskSmoothBrush_;  // Shift with the Mask brush smooths the mask.
  FaceSetBrush faceSetBrush_;
  EraseLayerBrush eraseLayerBrush_;
  StrokeSampler sampler_;
  std::vector<StrokeSample> samples_;
  // Brush of the running stroke. Shift turns it into Smooth, except for Mask, which stays Mask and
  // smooths the mask instead.
  BrushKind strokeBrush_ = BrushKind::Draw;
  // Grab drags the captured region in the plane through the grab point facing the camera.
  Vec3 grabStartWorld_{0.0f};
  Vec3 grabPlaneNormal_{0.0f, 0.0f, 1.0f};
  bool adjustingRadius_ = false;     // F held.
  bool adjustingDetail_ = false;     // R held in Sculpt mode.
  std::uint64_t oldestInputThisFrameNs_ = 0;
  int dabsThisFrame_ = 0;
};

}  // namespace plegl
