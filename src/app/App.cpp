#include "App.h"

#include "remesh/QuadRemesh.h"

#include <glad/gl.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl3.h>
#include <ImGuizmo.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <glm/gtc/type_ptr.hpp>

#include "core/Timer.h"
#include "io/Obj.h"
#include "mesh/Primitives.h"

namespace plegl {
namespace {

#ifndef NDEBUG
void GLAPIENTRY glDebugCallback(GLenum, GLenum type, GLuint, GLenum severity, GLsizei, const GLchar* message,
                                const void*) {
  if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) return;
  SDL_LogWarn(SDL_LOG_CATEGORY_RENDER, "GL %s: %s", type == GL_DEBUG_TYPE_ERROR ? "error" : "message", message);
}
#endif

SDL_GLContext createContext(SDL_Window* window, int minor, bool msaa) {
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, minor);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, msaa ? 1 : 0);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, msaa ? 4 : 0);
  return SDL_GL_CreateContext(window);
}

}  // namespace

bool App::init(std::string* error) {
  SDL_SetAppMetadata("PleGL Sculpt", "0.1.0", "app.plegl.sculpt");
  if (!SDL_Init(SDL_INIT_VIDEO)) {
    *error = std::string("SDL_Init failed: ") + SDL_GetError();
    return false;
  }

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
#ifndef NDEBUG
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
#endif

  window_ = SDL_CreateWindow("PleGL Sculpt", 1600, 900,
                             SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
  if (!window_) {
    *error = std::string("Could not create window: ") + SDL_GetError();
    return false;
  }
  // OpenGL 4.6 first, then 4.5 (which already has direct state access), then without MSAA.
  for (const auto& [minor, msaa] : {std::pair{6, true}, std::pair{5, true}, std::pair{6, false}, std::pair{5, false}}) {
    gl_ = createContext(window_, minor, msaa);
    if (gl_) break;
  }
  if (!gl_) {
    *error = std::string("PleGL Sculpt needs OpenGL 4.5 or newer. ") + SDL_GetError();
    return false;
  }
  SDL_GL_MakeCurrent(window_, gl_);
  const int version = gladLoadGL(reinterpret_cast<GLADloadfunc>(SDL_GL_GetProcAddress));
  if (version == 0 || GLAD_VERSION_MAJOR(version) * 10 + GLAD_VERSION_MINOR(version) < 45) {
    *error = "PleGL Sculpt needs OpenGL 4.5 or newer.";
    return false;
  }
#ifndef NDEBUG
  if (GLAD_GL_KHR_debug) {
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(glDebugCallback, nullptr);
  }
#endif
  glInfo = std::string(reinterpret_cast<const char*>(glGetString(GL_RENDERER))) + "\nOpenGL " +
           reinterpret_cast<const char*>(glGetString(GL_VERSION));
  glEnable(GL_MULTISAMPLE);
  setVsync(true);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.IniFilename = nullptr;  // Fixed layout; nothing to persist yet.
  ImGui::StyleColorsDark();
  ImGuiStyle& style = ImGui::GetStyle();
  style.WindowRounding = 0.0f;
  style.FrameRounding = 4.0f;
  style.GrabRounding = 4.0f;
  style.WindowBorderSize = 0.0f;
  const float dpi = SDL_GetWindowDisplayScale(window_);
  if (dpi > 0.0f) {
    style.ScaleAllSizes(dpi);
    style.FontScaleDpi = dpi;
  }
  ImGui_ImplSDL3_InitForOpenGL(window_, gl_);
  ImGui_ImplOpenGL3_Init("#version 450");

  if (!renderer_.init(error)) return false;
  // Dynamic topology strokes and their undo share one scratch mesh, so neither reallocates it.
  const auto workspace = std::make_shared<LayoutWorkspace>();
  sculptor_.setLayoutWorkspace(workspace);
  undoStack.setLayoutWorkspace(workspace);
  syncWorkspace_ = std::make_shared<SyncWorkspace>();
  undoStack.setSyncWorkspace(syncWorkspace_);
  rendererReady_ = true;

  initRecovery();
  addPrimitive("Sphere", makeQuadSphere(primitives.quadSphereResolution));
  frameScene();
  statusMessage = "Ready. Alt + left drag to orbit.";
  lastFrameNs_ = SDL_GetTicksNS();
  lastAutosaveNs_ = lastFrameNs_;
  savedFingerprint_ = autosavedFingerprint_ = sceneFingerprint();
  updateWindowTitle();
  return true;
}

void App::shutdown() {
  workers_.clear();  // Joins background imports and saves.
  finishRecovery();  // A clean exit: nothing to recover next time.
  if (rendererReady_) renderer_.shutdown();
  if (ImGui::GetCurrentContext()) {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
  }
  if (gl_) SDL_GL_DestroyContext(gl_);
  if (window_) SDL_DestroyWindow(window_);
  SDL_Quit();
}

void App::setVsync(bool on) {
  vsync = on;
  if (on && !SDL_GL_SetSwapInterval(-1)) SDL_GL_SetSwapInterval(1);  // Prefer adaptive vsync.
  if (!on) SDL_GL_SetSwapInterval(0);
}

void App::run() {
  while (running_) {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
      ImGui_ImplSDL3_ProcessEvent(&e);
      handleEvent(e);
    }
    processAsyncResults();
    autosaveTick();
    updateWindowTitle();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    ImGuizmo::BeginFrame();

    drawUi();  // Also lays out the viewport rectangle.
    drawGizmo();
    updateHover();
    renderer_.sync(scene, view.wireframe);
    ImGui::Render();

    int fbw = 0, fbh = 0;
    SDL_GetWindowSizeInPixels(window_, &fbw, &fbh);
    glViewport(0, 0, fbw, fbh);
    glClearColor(0.09f, 0.09f, 0.10f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ViewportRect rect;
    rect.x = static_cast<int>(vpX_ * pixelScale_);
    rect.width = std::max(1, static_cast<int>(vpW_ * pixelScale_));
    rect.height = std::max(1, static_cast<int>(vpH_ * pixelScale_));
    rect.y = fbh - static_cast<int>((vpY_ + vpH_) * pixelScale_);

    CursorMarker cursor;
    if (hover_ && (drag_ == Drag::None || drag_ == Drag::Pan)) {
      cursor.visible = true;
      cursor.position = hover_->worldPosition;
      cursor.normal = hover_->worldNormal;
      const float px = mode == Mode::Sculpt ? sculpt.radiusPx : 14.0f;
      cursor.radius = camera.worldPerPixel(cursor.position) * px;
    }
    renderer_.render(scene, camera, rect, view, selectedId, cursor);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(window_);
    if (oldestInputThisFrameNs_ != 0) {
      const double ms = static_cast<double>(SDL_GetTicksNS() - oldestInputThisFrameNs_) / 1e6;
      stats.inputToFrameMs = stats.inputToFrameMs == 0.0 ? ms : stats.inputToFrameMs * 0.8 + ms * 0.2;
      oldestInputThisFrameNs_ = 0;
    }
    stats.dabsLastFrame = dabsThisFrame_;
    dabsThisFrame_ = 0;

    const std::uint64_t now = SDL_GetTicksNS();
    const double ms = static_cast<double>(now - lastFrameNs_) / 1e6;
    lastFrameNs_ = now;
    stats.frameMs = stats.frameMs == 0.0 ? ms : stats.frameMs * 0.9 + ms * 0.1;
    stats.fps = stats.frameMs > 0.0 ? 1000.0 / stats.frameMs : 0.0;
  }
}

void App::updateViewportRect() {
  int w = 0, h = 0, fbw = 0;
  SDL_GetWindowSize(window_, &w, &h);
  SDL_GetWindowSizeInPixels(window_, &fbw, nullptr);
  pixelScale_ = w > 0 ? static_cast<float>(fbw) / static_cast<float>(w) : 1.0f;
  camera.setViewport(static_cast<int>(vpW_), static_cast<int>(vpH_));
}

void App::handleEvent(const SDL_Event& e) {
  const ImGuiIO& io = ImGui::GetIO();
  switch (e.type) {
    case SDL_EVENT_QUIT:
      requestQuit();
      break;
    case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
      if (e.window.windowID == SDL_GetWindowID(window_)) requestQuit();
      break;
    case SDL_EVENT_DROP_FILE:
      if (e.drop.data) openFile(std::filesystem::path(reinterpret_cast<const char8_t*>(e.drop.data)));
      break;

    case SDL_EVENT_MOUSE_MOTION: {
      mouseX_ = e.motion.x;
      mouseY_ = e.motion.y;
      mouseInViewport_ = mouseX_ >= vpX_ && mouseX_ < vpX_ + vpW_ && mouseY_ >= vpY_ && mouseY_ < vpY_ + vpH_;
      const float dx = e.motion.xrel, dy = e.motion.yrel;
      if (adjustingRadius_) {
        sculpt.radiusPx = std::clamp(sculpt.radiusPx * std::exp(dx * 0.01f), 2.0f, 1000.0f);
        break;
      }
      if (adjustingDetail_) {
        if (sculpt.detailMode == DetailMode::Relative) {
          sculpt.detailPx = std::clamp(sculpt.detailPx * std::exp(dx * 0.01f), 2.0f, 64.0f);
          statusMessage = "Detail " + std::to_string(static_cast<int>(std::lround(sculpt.detailPx))) + " px";
        } else {
          sculpt.detailSize = std::clamp(sculpt.detailSize * std::exp(dx * 0.01f), 1e-5f, 10.0f);
          char text[48];
          std::snprintf(text, sizeof(text), "Detail %.4g units", sculpt.detailSize);
          statusMessage = text;
        }
        break;
      }
      if (sculptor_.active()) {
        continueStroke(mouseX_, mouseY_, e.motion.timestamp);
        break;
      }
      switch (drag_) {
        case Drag::Orbit: camera.orbit(dx, dy, orbitPivot_); break;
        case Drag::Pan: camera.pan(dx, dy); break;
        case Drag::Zoom: camera.dolly(dy); break;
        case Drag::None: break;
      }
      break;
    }

    case SDL_EVENT_MOUSE_BUTTON_DOWN: {
      if (drag_ != Drag::None || !mouseInViewport_ || io.WantCaptureMouse) break;
      const bool alt = (SDL_GetModState() & SDL_KMOD_ALT) != 0;
      if (alt) {
        if (e.button.button == SDL_BUTTON_LEFT) {
          drag_ = Drag::Orbit;
          orbitPivot_ = hover_ ? hover_->worldPosition : camera.target;  // Orbit around what is under the cursor.
        } else if (e.button.button == SDL_BUTTON_MIDDLE) {
          drag_ = Drag::Pan;
        } else if (e.button.button == SDL_BUTTON_RIGHT) {
          drag_ = Drag::Zoom;
        }
      } else if (e.button.button == SDL_BUTTON_LEFT && mode == Mode::Sculpt) {
        beginStroke(e.button.x, e.button.y, e.button.timestamp);
      } else if (e.button.button == SDL_BUTTON_LEFT && mode == Mode::Object && !ImGuizmo::IsUsing() &&
                 !(scene.find(selectedId) && ImGuizmo::IsOver())) {
        selectedId = hover_ ? hover_->objectId : 0;
      }
      break;
    }
    case SDL_EVENT_MOUSE_BUTTON_UP:
      if (e.button.button == SDL_BUTTON_LEFT && sculptor_.active()) endStroke();
      drag_ = Drag::None;
      break;
    case SDL_EVENT_MOUSE_WHEEL:
      if (mouseInViewport_ && !io.WantCaptureMouse) camera.zoomSteps(e.wheel.y);
      break;

    case SDL_EVENT_KEY_DOWN:
      // Typing into a text field, dragging a widget or an open dialog blocks shortcuts. Not
      // WantCaptureKeyboard: it is also true whenever a panel has keyboard focus, which happens
      // after any click on a button or slider, and would leave every shortcut dead until the
      // viewport was clicked. The dialog flags are set before ImGui opens the popup.
      if (!io.WantTextInput && !ImGui::IsAnyItemActive() && !askSaveChanges_ && !offerRecovery_ && !e.key.repeat)
        handleShortcut(e.key);
      break;
    case SDL_EVENT_KEY_UP:
      if (e.key.key == SDLK_F) adjustingRadius_ = false;
      if (e.key.key == SDLK_R) adjustingDetail_ = false;
      break;

    // Pen tablets. SDL also turns pen input into mouse events, so navigation works with a pen;
    // here we only track the stylus state, which Phase 1 feeds into the brush.
    case SDL_EVENT_PEN_PROXIMITY_IN:
      pen.inProximity = true;
      pen.deviceId = e.pproximity.which;
      break;
    case SDL_EVENT_PEN_PROXIMITY_OUT:
      pen = PenState{};
      break;
    case SDL_EVENT_PEN_DOWN:
      pen.down = true;
      pen.eraser = e.ptouch.eraser;
      pen.lastEventNs = e.ptouch.timestamp;
      break;
    case SDL_EVENT_PEN_UP:
      pen.down = false;
      pen.lastEventNs = e.ptouch.timestamp;
      break;
    case SDL_EVENT_PEN_AXIS:
      pen.inProximity = true;
      pen.lastEventNs = e.paxis.timestamp;
      if (e.paxis.axis == SDL_PEN_AXIS_PRESSURE) pen.pressure = e.paxis.value;
      if (e.paxis.axis == SDL_PEN_AXIS_XTILT) pen.tiltX = e.paxis.value;
      if (e.paxis.axis == SDL_PEN_AXIS_YTILT) pen.tiltY = e.paxis.value;
      break;
    default:
      break;
  }
}

void App::handleShortcut(const SDL_KeyboardEvent& key) {
  const bool ctrl = (key.mod & SDL_KMOD_CTRL) != 0;
  const bool shift = (key.mod & SDL_KMOD_SHIFT) != 0;
  if (ctrl) {
    if (key.key == SDLK_Z && !shift) undo();
    if ((key.key == SDLK_Z && shift) || key.key == SDLK_Y) redo();
    if (key.key == SDLK_S) requestSaveProject(shift);
    if (key.key == SDLK_O) requestOpenProject();
    if (key.key == SDLK_I) {
      // Ctrl+I inverts the mask while sculpting (as in other sculpting tools); Ctrl+Shift+I imports.
      if (mode == Mode::Sculpt && !shift) applyMask(MaskOp::Invert);
      else requestImport();
    }
    if (key.key == SDLK_E) requestExport();
    if (key.key == SDLK_N) requestNewScene();
    if (key.key == SDLK_R) requestRemesh();
    if (key.key == SDLK_PAGEUP) {
      requestSubdivide();
      ImGui::SetWindowFocus(nullptr);  // Panels would also scroll on Page Up.
    }
    if (key.key == SDLK_D && mode == Mode::Sculpt) {
      sculpt.dyntopo = !sculpt.dyntopo;
      statusMessage = sculpt.dyntopo ? "Dynamic topology on" : "Dynamic topology off";
      const SceneObject* obj = scene.find(selectedId);
      if (sculpt.dyntopo && obj && obj->multires) statusMessage += " (not on objects with subdivision levels)";
    }
    return;
  }
  switch (key.key) {
    case SDLK_TAB:
      if (!sculptor_.active()) mode = mode == Mode::Object ? Mode::Sculpt : Mode::Object;
      // Drop panel focus so ImGui does not also use this Tab to turn the last slider into a text field.
      ImGui::SetWindowFocus(nullptr);
      break;
    case SDLK_HOME:
    case SDLK_KP_PERIOD:
      frameScene();
      break;
    case SDLK_PAGEUP:
    case SDLK_PAGEDOWN: {
      const bool up = key.key == SDLK_PAGEUP;
      if (shift) setLevel(up ? kMaxMultiresLevels : 0);
      else stepLevel(up ? 1 : -1);
      ImGui::SetWindowFocus(nullptr);  // Panels would also scroll on Page Up and Page Down.
      break;
    }
    default:
      break;
  }
  if ((key.mod & SDL_KMOD_ALT) != 0) {
    // Mask and visibility operations work in both modes, like their menus. Alt is the navigation
    // modifier, so no other Alt+key does anything.
    switch (key.key) {
      case SDLK_M: applyMask(shift ? MaskOp::Fill : MaskOp::Clear); break;
      case SDLK_B: applyMask(shift ? MaskOp::Sharpen : MaskOp::Blur); break;
      case SDLK_H: applyFaceSets(FaceSetOp::RevealAll); break;
      default: break;
    }
    return;
  }
  if (mode == Mode::Sculpt) {
    switch (key.key) {
      case SDLK_D: sculpt.brush = BrushKind::Draw; break;
      case SDLK_M:
        if (shift) maskFaceSetUnderCursor();
        else sculpt.brush = BrushKind::Mask;
        break;
      case SDLK_P: sculpt.brush = BrushKind::FaceSet; break;
      case SDLK_H: faceSetOpUnderCursor(shift ? FaceSetOp::Isolate : FaceSetOp::Hide); break;
      case SDLK_C: sculpt.brush = shift ? BrushKind::Crease : BrushKind::Clay; break;
      case SDLK_S: sculpt.brush = BrushKind::Smooth; break;
      case SDLK_G: sculpt.brush = BrushKind::Grab; break;
      case SDLK_I: sculpt.brush = BrushKind::Inflate; break;
      case SDLK_T: sculpt.brush = BrushKind::Flatten; break;
      case SDLK_F: adjustingRadius_ = true; break;  // Move the mouse sideways while holding F.
      case SDLK_R: adjustingDetail_ = true; break;  // The same for the dynamic topology detail.
      case SDLK_LEFTBRACKET: sculpt.radiusPx = std::max(2.0f, sculpt.radiusPx / 1.15f); break;
      case SDLK_RIGHTBRACKET: sculpt.radiusPx = std::min(1000.0f, sculpt.radiusPx * 1.15f); break;
      case SDLK_X: sculpt.symmetryX = !sculpt.symmetryX; break;
      default: break;
    }
    return;
  }
  switch (key.key) {
    case SDLK_G: gizmo = GizmoOp::Translate; break;
    case SDLK_R: gizmo = GizmoOp::Rotate; break;
    case SDLK_S: gizmo = GizmoOp::Scale; break;
    case SDLK_X:
    case SDLK_DELETE: deleteSelected(); break;
    case SDLK_D:
      if (shift) duplicateSelected();
      break;
    case SDLK_B:
      if (!shift) bumpUnderCursor();
      break;
    case SDLK_ESCAPE: selectedId = 0; break;
    default: break;
  }
}

void App::updateHover() {
  if (!mouseInViewport_ || drag_ == Drag::Orbit || ImGui::GetIO().WantCaptureMouse) {
    if (drag_ == Drag::None) hover_.reset();
    return;
  }
  const Ray ray = camera.rayThroughPixel(mouseX_ - vpX_, mouseY_ - vpY_);
  Timer t;
  hover_ = scene.pick(ray);
  const double us = t.us();
  stats.raycastUs = stats.raycastUs == 0.0 ? us : stats.raycastUs * 0.9 + us * 0.1;
}

std::optional<ScenePick> App::pickUnderMouse() const {
  if (!mouseInViewport_ || ImGui::GetIO().WantCaptureMouse) return std::nullopt;
  return scene.pick(camera.rayThroughPixel(mouseX_ - vpX_, mouseY_ - vpY_));
}

void App::drawGizmo() {
  if (mode != Mode::Object) return;
  SceneObject* obj = scene.find(selectedId);
  if (!obj || !obj->visible) return;
  // Navigation always wins: with Alt held (or mid-drag) the gizmo must not grab the mouse.
  const bool navigating = drag_ != Drag::None || (SDL_GetModState() & SDL_KMOD_ALT) != 0;
  ImGuizmo::Enable(!navigating);
  ImGuizmo::SetOrthographic(false);
  ImGuizmo::SetDrawlist(ImGui::GetBackgroundDrawList());
  ImGuizmo::SetRect(vpX_, vpY_, vpW_, vpH_);
  const Mat4 viewM = camera.view();
  const Mat4 projM = camera.projection();
  Mat4 model = obj->transform.matrix();
  const ImGuizmo::OPERATION op = gizmo == GizmoOp::Translate ? ImGuizmo::TRANSLATE
                                 : gizmo == GizmoOp::Rotate  ? ImGuizmo::ROTATE
                                                             : ImGuizmo::SCALE;
  const ImGuizmo::MODE space = gizmo == GizmoOp::Translate ? ImGuizmo::WORLD : ImGuizmo::LOCAL;
  if (ImGuizmo::Manipulate(glm::value_ptr(viewM), glm::value_ptr(projM), op, space, glm::value_ptr(model))) {
    obj->transform = Transform::fromMatrix(model);
  }
}

void App::processAsyncResults() {
  std::deque<std::function<void()>> pending;
  {
    std::lock_guard lock(asyncMutex_);
    pending.swap(asyncResults_);
  }
  for (auto& fn : pending) fn();
}

void App::addPrimitive(const std::string& name, Mesh mesh) {
  SceneObject& obj = scene.add(name, std::move(mesh));
  selectedId = obj.id;
  statusMessage = "Added " + obj.name + " (" + std::to_string(obj.mesh.vertexCount()) + " vertices)";
}

void App::newScene() {
  if (sculptor_.active()) endStroke();  // Never act on a mesh in the middle of a stroke.
  scene.clear();
  undoStack.clear();
  selectedId = 0;
  hover_.reset();
  projectPath_.clear();
  savedFingerprint_ = autosavedFingerprint_ = sceneFingerprint();
  statusMessage = "New scene";
}

void App::deleteSelected() {
  if (sculptor_.active()) endStroke();  // Never act on a mesh in the middle of a stroke.
  if (selectedId && scene.remove(selectedId)) {
    statusMessage = "Deleted object";
    selectedId = 0;
    hover_.reset();
  }
}

void App::duplicateSelected() {
  if (sculptor_.active()) endStroke();  // Never act on a mesh in the middle of a stroke.
  if (SceneObject* copy = scene.duplicate(selectedId)) {
    selectedId = copy->id;
    statusMessage = "Duplicated as " + copy->name;
  }
}

void App::frameScene() {
  const SceneObject* obj = scene.find(selectedId);
  Aabb b;
  if (obj && !obj->bvh.empty()) {
    const Aabb lb = obj->bvh.bounds();
    const Mat4 m = obj->transform.matrix();
    for (int i = 0; i < 8; ++i) {
      b.expand(Vec3(m * Vec4((i & 1) ? lb.max.x : lb.min.x, (i & 2) ? lb.max.y : lb.min.y,
                             (i & 4) ? lb.max.z : lb.min.z, 1.0f)));
    }
  } else {
    b = scene.worldBounds();
  }
  camera.frame(b);
}

void App::bumpUnderCursor() {
  if (!hover_) {
    statusMessage = "Hover over a mesh, then press B";
    return;
  }
  SceneObject* obj = editableObject(hover_->objectId, "mesh");
  if (!obj) return;
  Timer t;
  Mesh& m = obj->mesh;
  const Vec3 center = hover_->localHit.position;
  const Vec3 dir = hover_->localHit.smoothNormal;
  const float radius = glm::length(obj->bvh.bounds().extent()) * 0.06f;
  std::vector<Index> leaves;
  obj->bvh.querySphere(center, radius, leaves);
  for (Index l : leaves) {
    const BvhLeaf& leaf = obj->bvh.leaves()[l];
    for (Index v = leaf.vertBegin; v < leaf.vertEnd; ++v) {
      const float d = glm::length(m.positions[v] - center) / radius;
      if (d >= 1.0f) continue;
      const float falloff = 1.0f - d * d * (3.0f - 2.0f * d);  // Smoothstep from 1 at centre to 0.
      m.positions[v] += dir * (radius * 0.15f * falloff);
    }
  }
  for (Index l : leaves) {
    const BvhLeaf& leaf = obj->bvh.leaves()[l];
    m.computeNormals(leaf.vertBegin, leaf.vertEnd);
    obj->markLeafDirty(l);
  }
  obj->bvh.refitLeaves(m, leaves);
  stats.partialTestMs = t.ms();
  statusMessage = "Bumped " + std::to_string(leaves.size()) + " leaves";
}

// ---- Sculpting -----------------------------------------------------------------------------

float App::currentPressure() const {
  // SDL turns pen input into mouse events too; while the pen touches, use its pressure.
  return pen.down ? std::clamp(pen.pressure, 0.0f, 1.0f) : 1.0f;
}

void App::beginStroke(float x, float y, std::uint64_t timestampNs) {
  // A press without a matching release (focus lost mid-stroke) must not drop the open stroke's
  // undo entry: end it properly first.
  if (sculptor_.active()) endStroke();
  if (!hover_ || ImGui::GetIO().WantCaptureMouse) return;
  SceneObject* obj = scene.find(hover_->objectId);
  if (!obj) return;
  selectedId = obj->id;
  if (waitForJob(obj->id, "sculpting this object")) return;
  const SDL_Keymod mods = SDL_GetModState();
  const bool shift = (mods & SDL_KMOD_SHIFT) != 0, ctrl = (mods & SDL_KMOD_CTRL) != 0;
  const bool masking = sculpt.brush == BrushKind::Mask;
  strokeBrush_ = shift && !masking ? BrushKind::Smooth : sculpt.brush;
  StrokeOptions opts;
  opts.falloff = sculpt.falloff;
  opts.symmetryX = sculpt.symmetryX;
  opts.strength = sculpt.strength[static_cast<int>(strokeBrush_)];
  // The Mask brush ignores Add/Subtract: it paints, and erases with Ctrl or the pen's eraser end.
  opts.invert = masking ? ctrl || (pen.down && pen.eraser) : sculpt.invert != ctrl;
  opts.dyntopo = sculpt.dyntopo;  // The sculptor ignores it for Grab, Mask and Face Set.
  opts.dyntopoOptions.refine = sculpt.dyntopoRefine;
  opts.faceSetAutoMask = sculpt.faceSetAutoMask;
  opts.lockFaceSetBoundaries = sculpt.lockFaceSetBorders;
  // The Face Set brush starts a new set per stroke; Ctrl grows the set under the cursor instead.
  opts.extendFaceSet = ctrl;
  if (strokeBrush_ == BrushKind::Grab) {
    // Grab captures once at the press; pressure only scales the radius here.
    const float p = currentPressure();
    const bool mapRadius = sculpt.pressure == PressureMap::Radius || sculpt.pressure == PressureMap::Both;
    const float px = sculpt.radiusPx * (mapRadius ? std::max(p, 0.1f) : 1.0f);
    const Mat4 inv = glm::inverse(obj->transform.matrix());
    const float scale = (obj->transform.scale.x + obj->transform.scale.y + obj->transform.scale.z) / 3.0f;
    grabStartWorld_ = hover_->worldPosition;
    grabPlaneNormal_ = camera.forward();
    const float radius = camera.worldPerPixel(grabStartWorld_) * px / std::max(scale, 1e-6f);
    if (sculptor_.beginGrab(*obj, opts, Vec3(inv * Vec4(grabStartWorld_, 1.0f)), radius, "Grab")) {
      stats.dabMs = 0.0;
      if (timestampNs != 0 && (oldestInputThisFrameNs_ == 0 || timestampNs < oldestInputThisFrameNs_))
        oldestInputThisFrameNs_ = timestampNs;
    }
    return;
  }
  const Brush& brush = masking && shift ? maskSmoothBrush_ : *brushFor(strokeBrush_);
  sculptor_.beginStroke(*obj, brush, opts, brush.name());
  samples_.clear();
  sampler_.begin({x - vpX_, y - vpY_, currentPressure()}, std::max(1.0f, sculpt.radiusPx * sculpt.spacing), samples_);
  applySamples(timestampNs);
}

void App::continueStroke(float x, float y, std::uint64_t timestampNs) {
  if (strokeBrush_ == BrushKind::Grab) {
    SceneObject* obj = sculptor_.object();
    const Ray ray = camera.rayThroughPixel(x - vpX_, y - vpY_);
    const float denom = glm::dot(ray.dir, grabPlaneNormal_);
    if (!obj || std::abs(denom) < 1e-8f) return;
    const float t = glm::dot(grabStartWorld_ - ray.origin, grabPlaneNormal_) / denom;
    const Vec3 offsetWorld = ray.origin + ray.dir * t - grabStartWorld_;
    const Mat4 inv = glm::inverse(obj->transform.matrix());
    sculptor_.grab(Vec3(inv * Vec4(offsetWorld, 0.0f)));
    ++dabsThisFrame_;
    stats.dabMs = sculptor_.lastDab().totalMs;
    if (timestampNs != 0) {
      const double ms = static_cast<double>(SDL_GetTicksNS() - timestampNs) / 1e6;
      stats.inputToDabMs = stats.inputToDabMs == 0.0 ? ms : stats.inputToDabMs * 0.8 + ms * 0.2;
      if (oldestInputThisFrameNs_ == 0 || timestampNs < oldestInputThisFrameNs_) oldestInputThisFrameNs_ = timestampNs;
    }
    return;
  }
  samples_.clear();
  sampler_.setSpacing(std::max(1.0f, sculpt.radiusPx * sculpt.spacing));
  sampler_.moveTo({x - vpX_, y - vpY_, currentPressure()}, samples_);
  applySamples(timestampNs);
}

void App::applySamples(std::uint64_t timestampNs) {
  SceneObject* obj = sculptor_.object();
  if (!obj || samples_.empty()) return;
  const Mat4 model = obj->transform.matrix();
  const Mat4 inv = glm::inverse(model);
  const float scale = (obj->transform.scale.x + obj->transform.scale.y + obj->transform.scale.z) / 3.0f;
  const float baseStrength = sculpt.strength[static_cast<int>(strokeBrush_)];
  for (const StrokeSample& smp : samples_) {
    const Ray world = camera.rayThroughPixel(smp.x, smp.y);
    Ray local{Vec3(inv * Vec4(world.origin, 1.0f)), Vec3(inv * Vec4(world.dir, 0.0f))};
    RayHit hit;
    if (!obj->bvh.raycast(obj->mesh, local, hit, std::numeric_limits<float>::infinity(), true)) continue;  // Off the mesh: no dab.
    const Vec3 worldHit = world.origin + world.dir * hit.t;
    const float p = smp.pressure;
    const bool mapRadius = sculpt.pressure == PressureMap::Radius || sculpt.pressure == PressureMap::Both;
    const bool mapStrength = sculpt.pressure == PressureMap::Strength || sculpt.pressure == PressureMap::Both;
    const float px = sculpt.radiusPx * (mapRadius ? std::max(p, 0.1f) : 1.0f);
    const float radius = camera.worldPerPixel(worldHit) * px / std::max(scale, 1e-6f);
    const float strength = baseStrength * (mapStrength ? p : 1.0f);
    DabTopology topology;
    if (sculpt.dyntopo) {
      topology.detail = detailSizeAt(*obj, worldHit);
      topology.hintFace = hit.face;
    }
    if (sculptor_.dab(hit.position, radius, strength, topology)) {
      ++dabsThisFrame_;
      stats.dabMs = sculptor_.lastDab().totalMs;
      stats.topologyMs = sculptor_.lastDab().topologyMs;
    }
  }
  if (timestampNs != 0) {
    const double ms = static_cast<double>(SDL_GetTicksNS() - timestampNs) / 1e6;
    stats.inputToDabMs = stats.inputToDabMs == 0.0 ? ms : stats.inputToDabMs * 0.8 + ms * 0.2;
    if (oldestInputThisFrameNs_ == 0 || timestampNs < oldestInputThisFrameNs_) oldestInputThisFrameNs_ = timestampNs;
  }
}

void App::endStroke() {
  const SceneObject* obj = sculptor_.object();
  const bool levels = obj && obj->multires;
  auto entry = sculptor_.endStroke();
  const StrokeTopologyStats& topo = sculptor_.lastStrokeTopology();
  if (entry) {
    ++editCounter_;
    const int dabs = sculptor_.dabCount();
    const std::string label = std::visit([](const auto& e) { return e.label; }, *entry);
    statusMessage = label + " stroke, " + std::to_string(dabs) + (strokeBrush_ == BrushKind::Grab ? " moves" : " dabs");
    if (topo.dyntopo) {
      const long long diff = static_cast<long long>(topo.facesAfter) - topo.facesBefore;
      statusMessage += ", " + std::string(diff >= 0 ? "+" : "") + std::to_string(diff) + " faces";
    }
    undoStack.push(std::move(*entry));
  } else if (topo.lostUndo) {
    // The mesh is fine, but the stroke could not be recorded; older history no longer applies.
    ++editCounter_;
    statusMessage = "Stroke could not be recorded for undo; earlier history was cut";
  }
  if (sculpt.dyntopo && levels && entry && strokeBrush_ != BrushKind::Grab && strokeBrush_ != BrushKind::Mask &&
      strokeBrush_ != BrushKind::FaceSet)
    statusMessage += " (dynamic topology is off on objects with subdivision levels)";
  if (topo.outOfRoom) statusMessage += " (dynamic topology stopped: stroke too large)";
  if (topo.faulted) statusMessage += " (dynamic topology stopped on a damaged area)";
}

float App::detailSizeAt(const SceneObject& obj, const Vec3& worldPoint) const {
  if (sculpt.detailMode == DetailMode::Constant) return sculpt.detailSize;
  const float scale = (obj.transform.scale.x + obj.transform.scale.y + obj.transform.scale.z) / 3.0f;
  return camera.worldPerPixel(worldPoint) * sculpt.detailPx / std::max(scale, 1e-6f);
}

const Brush* App::brushFor(BrushKind kind) const {
  switch (kind) {
    case BrushKind::Draw: return &drawBrush_;
    case BrushKind::Clay: return &clayBrush_;
    case BrushKind::Smooth: return &smoothBrush_;
    case BrushKind::Grab: return nullptr;
    case BrushKind::Inflate: return &inflateBrush_;
    case BrushKind::Flatten: return &flattenBrush_;
    case BrushKind::Crease: return &creaseBrush_;
    case BrushKind::Mask: return &maskBrush_;
    case BrushKind::FaceSet: return &faceSetBrush_;
  }
  return nullptr;
}

void App::undo() {
  if (sculptor_.active()) return;
  // A remesh or subdivision result is built from the mesh as it was, so it would bring this change
  // back.
  if (waitForJob(0, "undoing")) return;
  const std::string label = undoStack.undo(scene);
  if (!label.empty()) ++editCounter_;
  statusMessage = label.empty() ? "Nothing to undo" : "Undo " + label;
}

void App::redo() {
  if (sculptor_.active()) return;
  if (waitForJob(0, "redoing")) return;
  const std::string label = undoStack.redo(scene);
  if (!label.empty()) ++editCounter_;
  statusMessage = label.empty() ? "Nothing to redo" : "Redo " + label;
}

bool App::canEditMask() const {
  const SceneObject* obj = scene.find(selectedId);
  return obj && !sculptor_.active() && !(busy() && obj->id == jobObjectId_);
}

bool App::waitForJob(std::uint32_t id, const std::string& action) {
  if (!busy() || (id != 0 && id != jobObjectId_)) return false;
  statusMessage = std::string("Wait for the ") + (job_ == Job::Remesh ? "remesh" : "subdivision") +
                  " to finish before " + action + ".";
  return true;
}

SceneObject* App::editableObject(std::uint32_t id, const char* what) {
  SceneObject* obj = scene.find(id);
  if (!obj) {
    statusMessage = "Select an object first.";
    return nullptr;
  }
  if (sculptor_.active()) return nullptr;
  // The job's result is built from the mesh as it was, so this edit would be lost.
  if (waitForJob(obj->id, std::string("editing this object's ") + what)) return nullptr;
  return obj;
}

void App::pushEdit(std::optional<SculptUndo> entry, const char* name, double ms) {
  stats.maskOpMs = ms;
  if (!entry) {
    statusMessage = std::string(name) + ": nothing to change";
    return;
  }
  ++editCounter_;
  undoStack.push(std::move(*entry));
  char buf[96];
  std::snprintf(buf, sizeof(buf), "%s (%.1f ms)", name, ms);
  statusMessage = buf;
}

void App::pushEdit(std::optional<MultiresUndo> entry, const char* name, double ms) {
  stats.maskOpMs = ms;
  if (!entry) {
    statusMessage = std::string(name) + ": nothing to change";
    return;
  }
  ++editCounter_;
  undoStack.push(std::move(*entry));
  hover_.reset();
  char buf[96];
  std::snprintf(buf, sizeof(buf), "%s (%.1f ms)", name, ms);
  statusMessage = buf;
}

void App::applyMask(MaskOp op) {
  SceneObject* obj = editableObject(selectedId, "mask");
  if (!obj) return;
  const bool filter = op == MaskOp::Blur || op == MaskOp::Sharpen;
  Timer t;
  if (obj->multires && !filter) {
    // Clear, Fill and Invert act on every level at once; mask made on one level only would
    // otherwise survive on the others.
    auto entry = applyMaskOpAllLevels(*obj, op, *syncWorkspace_);
    pushEdit(std::move(entry), maskOpName(op), t.ms());
    return;
  }
  auto entry = applyMaskOp(*obj, op, filter ? std::clamp(sculpt.maskFilterSteps, 1, 10) : 1);
  pushEdit(std::move(entry), maskOpName(op), t.ms());
}

void App::applyFaceSets(FaceSetOp op) {
  SceneObject* obj = editableObject(selectedId, "face sets");
  if (!obj) return;
  Timer t;
  if (obj->multires && (op == FaceSetOp::Clear || op == FaceSetOp::RevealAll || op == FaceSetOp::InvertVisibility)) {
    auto entry = applyFaceSetOpAllLevels(*obj, op, *syncWorkspace_);
    pushEdit(std::move(entry), faceSetOpName(op), t.ms());
    return;
  }
  auto entry = applyFaceSetOp(*obj, op);
  pushEdit(std::move(entry), faceSetOpName(op), t.ms());
}

void App::faceSetOpUnderCursor(FaceSetOp op) {
  const std::optional<ScenePick> pick = pickUnderMouse();
  if (!pick) {
    statusMessage = "Hover over a face set, then press H (Shift+H shows only that set).";
    return;
  }
  SceneObject* obj = editableObject(pick->objectId, "face sets");
  if (!obj) return;
  selectedId = obj->id;
  const std::int32_t set = faceSetId(obj->mesh.faceSetValue(pick->localHit.face));
  Timer t;
  auto entry = applyFaceSetOp(*obj, op, set);
  pushEdit(std::move(entry), faceSetOpName(op), t.ms());
}

void App::maskFaceSetUnderCursor() {
  const std::optional<ScenePick> pick = pickUnderMouse();
  if (!pick) {
    statusMessage = "Hover over a face set, then press Shift+M to mask it.";
    return;
  }
  SceneObject* obj = editableObject(pick->objectId, "mask");
  if (!obj) return;
  selectedId = obj->id;
  const std::int32_t set = faceSetId(obj->mesh.faceSetValue(pick->localHit.face));
  Timer t;
  auto entry = maskFaceSet(*obj, set);
  pushEdit(std::move(entry), "Mask Face Set", t.ms());
}

void App::requestRemesh() {
  SceneObject* obj = scene.find(selectedId);
  if (!obj) {
    statusMessage = "Select an object to remesh.";
    return;
  }
  if (sculptor_.active() || waitForJob(0, "remeshing")) return;
  job_ = Job::Remesh;
  jobObjectId_ = obj->id;
  statusMessage = "Remeshing " + obj->name + "...";
  // The worker gets its own copy; the scene is only touched on the main thread.
  auto input = std::make_shared<Mesh>(obj->mesh);
  const std::uint32_t id = obj->id;
  const std::uint64_t version = obj->topologyVersion;
  const QuadRemeshParams params{.targetEdge = remesh.voxelSize, .rounds = remesh.optimizeQuads ? 3 : 0};
  workers_.emplace_back([this, input, id, version, params] {
    Timer t;
    auto remeshStats = std::make_shared<QuadRemeshStats>();
    auto error = std::make_shared<std::string>();
    auto result = std::make_shared<std::optional<Mesh>>(quadRemesh(*input, params, remeshStats.get(), error.get()));
    auto bvh = std::make_shared<Bvh>();
    if (*result) bvh->build(**result);
    const double ms = t.ms();
    std::lock_guard lock(asyncMutex_);
    asyncResults_.push_back([this, id, version, remeshStats, error, result, bvh, ms] {
      job_ = Job::None;
      jobObjectId_ = 0;
      SceneObject* target = scene.find(id);
      if (!*result) {
        statusMessage = "Remesh failed: " + *error;
        return;
      }
      if (!target || target->topologyVersion != version) {
        statusMessage = "Remesh discarded: the object changed while it ran.";
        return;
      }
      TopologyUndo entry;
      entry.label = "Remesh";
      entry.objectId = id;
      // The new topology has no subdivision levels; undo brings them back.
      const bool hadLevels = target->multires != nullptr;
      entry.before = std::make_shared<MeshState>(MeshState{std::move(target->mesh), std::move(target->bvh),
                                                           target->topologyVersion, std::move(target->multires)});
      target->multires.reset();
      target->mesh = std::move(**result);
      target->bvh = std::move(*bvh);
      target->topologyVersion = nextTopologyVersion();
      target->clearDirty();
      entry.after =
          std::make_shared<MeshState>(MeshState{target->mesh, target->bvh, target->topologyVersion, nullptr});
      undoStack.push(std::move(entry));
      const int* res = remeshStats->voxel.resolution;
      char buf[256];
      std::snprintf(buf, sizeof(buf), "%d vertices, %d quads, %.0f%% valence 4, grid %dx%dx%d, %.0f ms",
                    target->mesh.vertexCount(), target->mesh.faceCount(), 100.0 * remeshStats->optimized.valence4Ratio,
                    res[0], res[1], res[2], ms);
      lastRemeshInfo = buf;
      statusMessage = "Remeshed " + target->name + ": " + lastRemeshInfo;
      if (hadLevels) statusMessage += " (subdivision levels removed; undo restores them)";
    });
  });
}

// ---- Subdivision levels --------------------------------------------------------------------

void App::requestSubdivide() {
  SceneObject* obj = editableObject(selectedId, "levels");
  if (!obj || waitForJob(0, "subdividing")) return;
  if (obj->multires && obj->multires->active != obj->multires->top()) {
    // New levels go on top, so step there first (its own undo step).
    setLevel(obj->multires->top());
    if (!obj->multires || obj->multires->active != obj->multires->top()) return;
  }
  std::string error;
  std::optional<SubdivideJob> prepared = prepareSubdivide(*obj, &error);
  if (!prepared) {
    statusMessage = "Cannot subdivide " + obj->name + ": " + error;
    return;
  }
  // The worker gets a copy of the top level; the scene is only touched on the main thread.
  auto job = std::make_shared<SubdivideJob>(std::move(*prepared));
  job_ = Job::Subdivide;
  jobObjectId_ = obj->id;
  statusMessage = "Subdividing " + obj->name + "...";
  workers_.emplace_back([this, job] {
    Timer t;
    runSubdivideJob(*job);
    const double ms = t.ms();
    std::lock_guard lock(asyncMutex_);
    asyncResults_.push_back([this, job, ms] {
      job_ = Job::None;
      jobObjectId_ = 0;
      if (!job->result) {
        statusMessage = "Subdivide failed: " + job->error;
        return;
      }
      SceneObject* target = scene.find(job->objectId);
      Timer finish;
      std::optional<MultiresUndo> entry;
      if (target) entry = finishSubdivide(*target, *job, *syncWorkspace_);
      if (!entry) {
        statusMessage = "Subdivide discarded: the object changed while it ran.";
        return;
      }
      undoStack.push(std::move(*entry));
      ++editCounter_;
      hover_.reset();
      char buf[160];
      std::snprintf(buf, sizeof(buf), "Subdivided %s: level %d of %d, %d faces (%.0f ms)", target->name.c_str(),
                    target->multires->active, target->multires->top(), target->mesh.faceCount(), ms + finish.ms());
      statusMessage = buf;
    });
  });
}

void App::setLevel(int level) {
  SceneObject* obj = editableObject(selectedId, "levels");
  if (!obj) return;
  if (!obj->multires) {
    statusMessage = obj->name + " has no subdivision levels. Ctrl+Page Up subdivides.";
    return;
  }
  const Multires& s = *obj->multires;
  level = std::clamp(level, 0, s.top());
  if (level == s.active) {
    statusMessage = level == s.top() ? "Already at the highest level. Ctrl+Page Up adds one."
                                     : "Already at the lowest level.";
    return;
  }
  Timer t;
  auto entry = setActiveLevel(*obj, level, *syncWorkspace_);
  const double ms = t.ms();
  if (!entry) {
    statusMessage = "Could not switch levels.";
    return;
  }
  undoStack.push(std::move(*entry));
  ++editCounter_;
  hover_.reset();  // The surface under the cursor moved.
  char buf[128];
  std::snprintf(buf, sizeof(buf), "Level %d of %d, %d faces (%.0f ms)", obj->multires->active, obj->multires->top(),
                obj->mesh.faceCount(), ms);
  statusMessage = buf;
}

void App::stepLevel(int delta) {
  const SceneObject* obj = scene.find(selectedId);
  setLevel(obj && obj->multires ? obj->multires->active + delta : 0);
}

void App::deleteHigherLevels() {
  SceneObject* obj = editableObject(selectedId, "levels");
  if (!obj || !obj->multires) return;
  Timer t;
  pushEdit(plegl::deleteHigherLevels(*obj), "Delete Higher Levels", t.ms());
}

void App::deleteLowerLevels() {
  SceneObject* obj = editableObject(selectedId, "levels");
  if (!obj || !obj->multires) return;
  Timer t;
  pushEdit(plegl::deleteLowerLevels(*obj), "Delete Lower Levels", t.ms());
}

// ---- Files ---------------------------------------------------------------------------------

namespace {
const SDL_DialogFileFilter kObjFilter[] = {{"Wavefront OBJ", "obj"}};

struct DialogContext {
  App* app;
  std::mutex* mutex;
  std::deque<std::function<void()>>* queue;
};
}  // namespace

void App::requestImport() {
  auto* ctx = new DialogContext{this, &asyncMutex_, &asyncResults_};
  SDL_ShowOpenFileDialog(
      [](void* user, const char* const* files, int) {
        auto* c = static_cast<DialogContext*>(user);
        std::function<void()> fn;
        if (!files) {
          std::string err = SDL_GetError();
          fn = [app = c->app, err] { app->statusMessage = "File dialog failed: " + err; };
        } else if (files[0]) {
          std::filesystem::path path(reinterpret_cast<const char8_t*>(files[0]));
          fn = [app = c->app, path] { app->importFile(path); };
        }
        if (fn) {
          std::lock_guard lock(*c->mutex);
          c->queue->push_back(std::move(fn));
        }
        delete c;
      },
      ctx, window_, kObjFilter, 1, nullptr, false);
}

void App::requestExport() {
  if (!scene.find(selectedId)) {
    statusMessage = "Select an object to export";
    return;
  }
  auto* ctx = new DialogContext{this, &asyncMutex_, &asyncResults_};
  SDL_ShowSaveFileDialog(
      [](void* user, const char* const* files, int) {
        auto* c = static_cast<DialogContext*>(user);
        std::function<void()> fn;
        if (!files) {
          std::string err = SDL_GetError();
          fn = [app = c->app, err] { app->statusMessage = "File dialog failed: " + err; };
        } else if (files[0]) {
          std::filesystem::path path(reinterpret_cast<const char8_t*>(files[0]));
          fn = [app = c->app, path] { app->exportFile(path); };
        }
        if (fn) {
          std::lock_guard lock(*c->mutex);
          c->queue->push_back(std::move(fn));
        }
        delete c;
      },
      ctx, window_, kObjFilter, 1, nullptr);
}

void App::importFile(const std::filesystem::path& path) {
  ++pendingImports_;
  statusMessage = "Importing " + path.filename().string() + "...";
  // Parse and build the BVH off the UI thread; the scene is only touched on the main thread.
  workers_.emplace_back([this, path] {
    Timer t;
    auto result = std::make_shared<ObjImportResult>(importObj(path));
    auto bvh = std::make_shared<Bvh>();
    if (result->ok) bvh->build(result->mesh);
    const double ms = t.ms();
    std::lock_guard lock(asyncMutex_);
    asyncResults_.push_back([this, path, result, bvh, ms] {
      --pendingImports_;
      if (!result->ok) {
        statusMessage = "Import failed: " + result->error;
        return;
      }
      SceneObject& obj = scene.add(path.stem().string(), std::move(result->mesh), std::move(*bvh));
      selectedId = obj.id;
      char buf[256];
      std::snprintf(buf, sizeof(buf), "Imported %s: %d vertices, %d faces in %.0f ms", obj.name.c_str(),
                    obj.mesh.vertexCount(), obj.mesh.faceCount(), ms);
      statusMessage = buf;
      const BuildReport& r = result->report;
      if (!r.ok() || r.nonManifoldVertices)
        statusMessage += " (warning: " + std::to_string(r.nonManifoldEdges) + " non-manifold edges, " +
                         std::to_string(r.nonManifoldVertices) + " non-manifold vertices, " +
                         std::to_string(r.degenerateFaces) + " degenerate faces skipped)";
      frameScene();
    });
  });
}

void App::exportFile(const std::filesystem::path& requested) {
  if (sculptor_.active()) endStroke();  // A dynamic topology stroke leaves holes until it ends.
  const SceneObject* obj = scene.find(selectedId);
  if (!obj) return;
  std::filesystem::path path = requested;
  if (path.extension().empty()) path += ".obj";
  // Bake the object transform so the file matches what is on screen.
  Mesh baked = obj->mesh;
  const Mat4 m = obj->transform.matrix();
  const Mat3 nm = glm::transpose(glm::inverse(Mat3(m)));
  for (Vec3& p : baked.positions) p = Vec3(m * Vec4(p, 1.0f));
  for (Vec3& n : baked.normals) n = glm::normalize(nm * n);
  std::string err;
  if (exportObj(baked, path, &err)) {
    statusMessage = "Exported " + path.filename().string();
    if (obj->multires)
      statusMessage += " (level " + std::to_string(obj->multires->active) + " of " +
                       std::to_string(obj->multires->top()) + ")";
  } else {
    statusMessage = "Export failed: " + err;
  }
}

}  // namespace plegl
