// Project files (.psculpt), autosave and crash recovery.

#include <imgui.h>

#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <sstream>
#include <system_error>

#include "App.h"
#include "core/Timer.h"

namespace plegl {

namespace {

const SDL_DialogFileFilter kProjectFilter[] = {{"PleGL Sculpt project", "psculpt"}};
const SDL_DialogFileFilter kOpenFilter[] = {{"PleGL Sculpt project or OBJ mesh", "psculpt;obj"}};

constexpr const char* kAutosaveName = "autosave.psculpt";
constexpr const char* kLockName = "session.lock";

// ---- Settings text: one "key value" pair per line ----

class SettingsWriter {
 public:
  std::string text;

  void put(const char* key, const std::string& value) {
    text += key;
    text += ' ';
    text += value;
    text += '\n';
  }
  void put(const char* key, float v) {
    char buf[32];
    const auto r = std::to_chars(buf, buf + sizeof(buf), v);  // Shortest exact form, locale free.
    put(key, std::string(buf, r.ptr));
  }
  void put(const char* key, int v) { put(key, std::to_string(v)); }
  void put(const char* key, bool v) { put(key, std::string(v ? "1" : "0")); }
  void put(const char* key, const Vec3& v) {
    std::string s;
    for (int i = 0; i < 3; ++i) {
      char buf[32];
      const auto r = std::to_chars(buf, buf + sizeof(buf), v[i]);
      if (i) s += ' ';
      s.append(buf, r.ptr);
    }
    put(key, s);
  }
};

class SettingsReader {
 public:
  explicit SettingsReader(const std::string& text) {
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
      const auto space = line.find(' ');
      if (space == std::string::npos || space == 0) continue;
      values_[line.substr(0, space)] = line.substr(space + 1);
    }
  }

  // Each getter leaves `out` unchanged when the key is missing or malformed, so files from older
  // versions keep the current defaults for settings they do not have.
  void get(const char* key, std::string& out) const {
    if (auto it = values_.find(key); it != values_.end()) out = it->second;
  }
  void get(const char* key, float& out) const {
    if (auto it = values_.find(key); it != values_.end()) parseFloats(it->second, &out, 1);
  }
  void get(const char* key, Vec3& out) const {
    if (auto it = values_.find(key); it != values_.end()) parseFloats(it->second, &out.x, 3);
  }
  void get(const char* key, bool& out) const {
    if (auto it = values_.find(key); it != values_.end()) out = it->second == "1";
  }
  void get(const char* key, int& out) const {
    if (auto it = values_.find(key); it != values_.end()) {
      int v = 0;
      const char* s = it->second.data();
      if (std::from_chars(s, s + it->second.size(), v).ec == std::errc()) out = v;
    }
  }
  template <class E>
  void getEnum(const char* key, E& out, int count) const {
    int v = static_cast<int>(out);
    get(key, v);
    if (v >= 0 && v < count) out = static_cast<E>(v);
  }

 private:
  static void parseFloats(const std::string& s, float* out, int n) {
    float tmp[3] = {};
    const char* p = s.data();
    const char* end = p + s.size();
    for (int i = 0; i < n; ++i) {
      while (p < end && *p == ' ') ++p;
      const auto r = std::from_chars(p, end, tmp[i]);
      if (r.ec != std::errc() || !std::isfinite(tmp[i])) return;
      p = r.ptr;
    }
    std::memcpy(out, tmp, sizeof(float) * n);
  }

  std::map<std::string, std::string> values_;
};

std::uint64_t fnv(std::uint64_t h, const void* data, std::size_t n) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

struct DialogContext {
  App* app;
  std::mutex* mutex;
  std::deque<std::function<void()>>* queue;
  std::function<void(App&, std::filesystem::path)> onPath;
  std::function<void(App&)> onCancel;
};

void SDLCALL dialogDone(void* user, const char* const* files, int) {
  auto* c = static_cast<DialogContext*>(user);
  std::function<void()> fn;
  if (!files) {
    std::string err = SDL_GetError();
    fn = [c, err] {
      c->app->statusMessage = "File dialog failed: " + err;
      if (c->onCancel) c->onCancel(*c->app);
    };
  } else if (files[0]) {
    std::filesystem::path path(reinterpret_cast<const char8_t*>(files[0]));
    fn = [c, path] { c->onPath(*c->app, path); };
  } else {
    fn = [c] {
      if (c->onCancel) c->onCancel(*c->app);
    };
  }
  // The context is freed on the main thread, after the callback that uses it.
  std::lock_guard lock(*c->mutex);
  c->queue->push_back([c, fn = std::move(fn)] {
    fn();
    delete c;
  });
}

std::string pathToUtf8(const std::filesystem::path& p) {
  const std::u8string s = p.u8string();
  return std::string(s.begin(), s.end());
}

std::filesystem::path pathFromUtf8(const std::string& s) {
  return std::filesystem::path(std::u8string(s.begin(), s.end()));
}

}  // namespace

// ---- Settings ----------------------------------------------------------------------------------

std::string App::settingsText() const {
  SettingsWriter w;
  w.put("mode", static_cast<int>(mode));
  int selectedIndex = -1;
  for (std::size_t i = 0; i < scene.objects().size(); ++i)
    if (scene.objects()[i]->id == selectedId) selectedIndex = static_cast<int>(i);
  w.put("selected", selectedIndex);

  w.put("camera.target", camera.target);
  w.put("camera.distance", camera.distance);
  w.put("camera.yaw", camera.yaw);
  w.put("camera.pitch", camera.pitch);
  w.put("camera.fov", camera.fovY);

  w.put("brush", static_cast<int>(sculpt.brush));
  w.put("brush.radius_px", sculpt.radiusPx);
  for (int i = 0; i < kBrushCount; ++i)
    w.put(("brush.strength." + std::string(kBrushNames[i])).c_str(), sculpt.strength[i]);
  w.put("brush.falloff", static_cast<int>(sculpt.falloff));
  w.put("brush.invert", sculpt.invert);
  w.put("brush.pressure", static_cast<int>(sculpt.pressure));
  w.put("brush.spacing", sculpt.spacing);
  w.put("symmetry.x", sculpt.symmetryX);
  w.put("mask.filter_steps", sculpt.maskFilterSteps);
  w.put("face_sets.automask", sculpt.faceSetAutoMask);
  w.put("face_sets.lock_borders", sculpt.lockFaceSetBorders);
  w.put("dyntopo.enabled", sculpt.dyntopo);
  w.put("dyntopo.refine", static_cast<int>(sculpt.dyntopoRefine));
  w.put("dyntopo.detail_mode", static_cast<int>(sculpt.detailMode));
  w.put("dyntopo.detail_px", sculpt.detailPx);
  w.put("dyntopo.detail_size", sculpt.detailSize);

  w.put("view.matcap", view.matcap);
  w.put("view.wireframe", view.wireframe);
  w.put("view.wireframe_opacity", view.wireframeOpacity);
  w.put("view.grid", view.grid);
  w.put("view.background_top", view.backgroundTop);
  w.put("view.background_bottom", view.backgroundBottom);
  w.put("view.mask", view.showMask);
  w.put("view.mask_opacity", view.maskOpacity);
  w.put("view.face_sets", view.showFaceSets);
  w.put("view.face_set_opacity", view.faceSetOpacity);

  w.put("remesh.edge", remesh.voxelSize);
  w.put("remesh.optimize", remesh.optimizeQuads);
  if (!projectPath_.empty()) w.put("project.path", pathToUtf8(projectPath_));
  return w.text;
}

void App::applySettings(const std::string& text) {
  const SettingsReader r(text);
  r.getEnum("mode", mode, 2);
  int selectedIndex = 0;  // Files without a selection select the first object.
  r.get("selected", selectedIndex);
  selectedId = selectedIndex >= 0 && selectedIndex < static_cast<int>(scene.objects().size())
                   ? scene.objects()[selectedIndex]->id
                   : 0;

  r.get("camera.target", camera.target);
  r.get("camera.distance", camera.distance);
  r.get("camera.yaw", camera.yaw);
  r.get("camera.pitch", camera.pitch);
  r.get("camera.fov", camera.fovY);
  camera.distance = std::max(camera.distance, 1e-4f);
  camera.fovY = std::clamp(camera.fovY, 0.1f, 2.5f);

  r.getEnum("brush", sculpt.brush, kBrushCount);
  r.get("brush.radius_px", sculpt.radiusPx);
  for (int i = 0; i < kBrushCount; ++i) {
    r.get(("brush.strength." + std::string(kBrushNames[i])).c_str(), sculpt.strength[i]);
    sculpt.strength[i] = std::clamp(sculpt.strength[i], 0.0f, 1.0f);
  }
  r.getEnum("brush.falloff", sculpt.falloff, 4);
  r.get("brush.invert", sculpt.invert);
  r.getEnum("brush.pressure", sculpt.pressure, 4);
  r.get("brush.spacing", sculpt.spacing);
  r.get("symmetry.x", sculpt.symmetryX);
  r.get("mask.filter_steps", sculpt.maskFilterSteps);
  r.get("face_sets.automask", sculpt.faceSetAutoMask);
  r.get("face_sets.lock_borders", sculpt.lockFaceSetBorders);
  sculpt.maskFilterSteps = std::clamp(sculpt.maskFilterSteps, 1, 10);
  sculpt.radiusPx = std::clamp(sculpt.radiusPx, 2.0f, 2000.0f);
  sculpt.spacing = std::clamp(sculpt.spacing, 0.01f, 1.0f);
  r.get("dyntopo.enabled", sculpt.dyntopo);
  r.getEnum("dyntopo.refine", sculpt.dyntopoRefine, 3);
  r.getEnum("dyntopo.detail_mode", sculpt.detailMode, 2);
  r.get("dyntopo.detail_px", sculpt.detailPx);
  r.get("dyntopo.detail_size", sculpt.detailSize);
  sculpt.detailPx = std::isfinite(sculpt.detailPx) ? std::clamp(sculpt.detailPx, 2.0f, 64.0f) : 8.0f;
  sculpt.detailSize = std::isfinite(sculpt.detailSize) ? std::clamp(sculpt.detailSize, 1e-5f, 10.0f) : 0.02f;

  r.get("view.matcap", view.matcap);
  r.get("view.wireframe", view.wireframe);
  r.get("view.wireframe_opacity", view.wireframeOpacity);
  r.get("view.grid", view.grid);
  r.get("view.background_top", view.backgroundTop);
  r.get("view.background_bottom", view.backgroundBottom);
  r.get("view.mask", view.showMask);
  r.get("view.mask_opacity", view.maskOpacity);
  r.get("view.face_sets", view.showFaceSets);
  r.get("view.face_set_opacity", view.faceSetOpacity);
  view.matcap = std::max(view.matcap, 0);
  view.maskOpacity = std::clamp(view.maskOpacity, 0.1f, 1.0f);
  view.faceSetOpacity = std::isfinite(view.faceSetOpacity) ? std::clamp(view.faceSetOpacity, 0.1f, 1.0f) : 0.6f;

  r.get("remesh.edge", remesh.voxelSize);
  r.get("remesh.optimize", remesh.optimizeQuads);
  remesh.voxelSize = std::clamp(remesh.voxelSize, 0.0005f, 1.0f);
}

// ---- Change tracking ---------------------------------------------------------------------------

std::uint64_t App::sceneFingerprint() const {
  // Everything a project file stores about the scene. Mesh edits are covered by editCounter_
  // (strokes, undo, redo) and topologyVersion (remesh, import), so no vertex data is hashed.
  std::uint64_t h = 1469598103934665603ull;
  h = fnv(h, &editCounter_, sizeof(editCounter_));
  for (const auto& o : scene.objects()) {
    h = fnv(h, &o->id, sizeof(o->id));
    h = fnv(h, &o->topologyVersion, sizeof(o->topologyVersion));
    h = fnv(h, o->name.data(), o->name.size());
    h = fnv(h, &o->visible, sizeof(o->visible));
    h = fnv(h, &o->transform.position, sizeof(Vec3));
    h = fnv(h, &o->transform.rotation, sizeof(Quat));
    h = fnv(h, &o->transform.scale, sizeof(Vec3));
  }
  return h;
}

void App::updateWindowTitle() {
  std::string title = "PleGL Sculpt - ";
  title += projectPath_.empty() ? std::string("Untitled") : pathToUtf8(projectPath_.filename());
  if (hasUnsavedChanges()) title += " *";
  if (title != windowTitle_) {
    windowTitle_ = title;
    SDL_SetWindowTitle(window_, title.c_str());
  }
}

// ---- Save and open -----------------------------------------------------------------------------

void App::saveProjectTo(const std::filesystem::path& requested, std::function<void()> then) {
  if (savingProject_) {
    statusMessage = "Already saving...";
    return;
  }
  if (sculptor_.active()) endStroke();
  std::filesystem::path path = requested;
  if (path.extension() != ".psculpt") path += ".psculpt";
  // Remember the path before serializing, so the file's own settings point at itself.
  const std::filesystem::path previousPath = projectPath_;
  projectPath_ = path;
  auto bytes = std::make_shared<std::vector<std::uint8_t>>(serializeProject(scene, settingsText()));
  projectPath_ = previousPath;
  const std::uint64_t fingerprint = sceneFingerprint();
  savingProject_ = true;
  statusMessage = "Saving " + pathToUtf8(path.filename()) + "...";
  workers_.emplace_back([this, path, bytes, fingerprint, then = std::move(then)] {
    Timer t;
    auto error = std::make_shared<std::string>();
    const bool ok = writeFileAtomic(path, *bytes, error.get());
    const double ms = t.ms();
    std::lock_guard lock(asyncMutex_);
    asyncResults_.push_back([this, path, bytes, fingerprint, then, ok, error, ms] {
      savingProject_ = false;
      if (!ok) {
        statusMessage = "Save failed: " + *error;
        pendingAction_ = PendingAction::None;
        return;
      }
      projectPath_ = path;
      savedFingerprint_ = fingerprint;
      // The project on disk is now newer than any autosave.
      std::error_code ec;
      std::filesystem::remove(dataDir_ / kAutosaveName, ec);
      autosavedFingerprint_ = fingerprint;
      char buf[128];
      std::snprintf(buf, sizeof(buf), " (%.1f MB, %.0f ms)", double(bytes->size()) / (1024.0 * 1024.0), ms);
      statusMessage = "Saved " + pathToUtf8(path.filename()) + buf;
      if (then) then();
    });
  });
}

void App::requestSaveProject(bool saveAs) {
  if (!saveAs && !projectPath_.empty()) {
    saveProjectTo(projectPath_);
    return;
  }
  auto* ctx = new DialogContext{this, &asyncMutex_, &asyncResults_,
                                [](App& app, std::filesystem::path path) { app.saveProjectTo(path); }, nullptr};
  SDL_ShowSaveFileDialog(dialogDone, ctx, window_, kProjectFilter, 1, nullptr);
}

void App::loadProjectAsync(const std::filesystem::path& path, bool recovered) {
  statusMessage = (recovered ? "Recovering " : "Opening ") + pathToUtf8(path.filename()) + "...";
  ++pendingImports_;
  workers_.emplace_back([this, path, recovered] {
    Timer t;
    auto error = std::make_shared<std::string>();
    auto project = std::make_shared<std::optional<Project>>(loadProject(path, error.get()));
    auto bvhs = std::make_shared<std::vector<Bvh>>();
    if (*project) {
      bvhs->resize((*project)->objects.size());
      for (std::size_t i = 0; i < bvhs->size(); ++i) (*bvhs)[i].build((*project)->objects[i].mesh);
    }
    const double ms = t.ms();
    std::lock_guard lock(asyncMutex_);
    asyncResults_.push_back([this, path, recovered, error, project, bvhs, ms] {
      --pendingImports_;
      if (!*project) {
        statusMessage = "Could not open " + pathToUtf8(path.filename()) + ": " + *error;
        return;
      }
      applyProject(std::move(**project), std::move(*bvhs), path, recovered);
      char buf[64];
      std::snprintf(buf, sizeof(buf), " in %.0f ms", ms);
      statusMessage += buf;
    });
  });
}

void App::applyProject(Project project, std::vector<Bvh> bvhs, const std::filesystem::path& path, bool recovered) {
  if (sculptor_.active()) endStroke();
  scene.clear();
  undoStack.clear();
  hover_.reset();
  std::size_t vertices = 0;
  for (std::size_t i = 0; i < project.objects.size(); ++i) {
    ProjectObject& po = project.objects[i];
    vertices += po.mesh.positions.size();
    SceneObject& obj = scene.add(po.name, std::move(po.mesh), std::move(bvhs[i]));
    obj.name = po.name;  // Keep names exactly, even duplicates.
    obj.transform = po.transform;
    obj.visible = po.visible;
  }
  applySettings(project.settings);
  if (recovered) {
    // Back to the file the user was working on, but still unsaved.
    std::string original;
    SettingsReader(project.settings).get("project.path", original);
    projectPath_ = original.empty() ? std::filesystem::path() : pathFromUtf8(original);
    savedFingerprint_ = 0;
    autosavedFingerprint_ = sceneFingerprint();
    statusMessage = "Recovered previous session (" + std::to_string(scene.objects().size()) + " objects)";
  } else {
    projectPath_ = path;
    savedFingerprint_ = autosavedFingerprint_ = sceneFingerprint();
    statusMessage = "Opened " + pathToUtf8(path.filename()) + " (" + std::to_string(scene.objects().size()) +
                    " objects, " + std::to_string(vertices) + " vertices)";
  }
}

void App::openProject(const std::filesystem::path& path) { loadProjectAsync(path, false); }

void App::openFile(const std::filesystem::path& path) {
  std::string ext = pathToUtf8(path.extension());
  for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (ext == ".psculpt")
    openProject(path);
  else
    importFile(path);
}

// ---- New / Open / Quit with unsaved changes ------------------------------------------------------

void App::runPendingAction() {
  const PendingAction action = pendingAction_;
  pendingAction_ = PendingAction::None;
  switch (action) {
    case PendingAction::Quit:
      running_ = false;
      break;
    case PendingAction::NewScene:
      newScene();
      break;
    case PendingAction::OpenProject: {
      auto* ctx = new DialogContext{this, &asyncMutex_, &asyncResults_,
                                    [](App& app, std::filesystem::path path) { app.openFile(path); }, nullptr};
      SDL_ShowOpenFileDialog(dialogDone, ctx, window_, kOpenFilter, 1, nullptr, false);
      break;
    }
    case PendingAction::None:
      break;
  }
}

static void askOrRun(App& app, bool unsaved, PendingAction action, PendingAction& pending, bool& ask,
                     void (App::*run)()) {
  pending = action;
  if (unsaved)
    ask = true;
  else
    (app.*run)();
}

void App::requestNewScene() {
  askOrRun(*this, hasUnsavedChanges(), PendingAction::NewScene, pendingAction_, askSaveChanges_,
           &App::runPendingAction);
}

void App::requestOpenProject() {
  askOrRun(*this, hasUnsavedChanges(), PendingAction::OpenProject, pendingAction_, askSaveChanges_,
           &App::runPendingAction);
}

void App::requestQuit() {
  askOrRun(*this, hasUnsavedChanges(), PendingAction::Quit, pendingAction_, askSaveChanges_, &App::runPendingAction);
}

// ---- Autosave and recovery ---------------------------------------------------------------------

void App::initRecovery() {
  char* pref = SDL_GetPrefPath("PleGL", "PleGL Sculpt");
  if (!pref) return;  // No writable per-user folder: autosave stays off.
  dataDir_ = std::filesystem::path(reinterpret_cast<const char8_t*>(pref));
  SDL_free(pref);
  std::error_code ec;
  // A lock left behind means the last session did not exit cleanly.
  offerRecovery_ =
      std::filesystem::exists(dataDir_ / kLockName, ec) && std::filesystem::exists(dataDir_ / kAutosaveName, ec);
  if (!offerRecovery_) std::filesystem::remove(dataDir_ / kAutosaveName, ec);
  writeFileAtomic(dataDir_ / kLockName, {});
}

void App::finishRecovery() {
  if (dataDir_.empty()) return;
  std::error_code ec;
  std::filesystem::remove(dataDir_ / kAutosaveName, ec);
  std::filesystem::remove(dataDir_ / kLockName, ec);
}

void App::autosaveTick() {
  const std::uint64_t now = SDL_GetTicksNS();
  if (dataDir_.empty() || projectSettings.autosaveMinutes <= 0.0f || autosaveRunning_ || offerRecovery_) return;
  if (now - lastAutosaveNs_ < static_cast<std::uint64_t>(projectSettings.autosaveMinutes * 60.0e9)) return;
  if (sculptor_.active()) return;  // Never in the middle of a stroke; retried next frame.
  lastAutosaveNs_ = now;
  const std::uint64_t fingerprint = sceneFingerprint();
  if (fingerprint == savedFingerprint_ || fingerprint == autosavedFingerprint_) return;
  autosaveRunning_ = true;
  auto bytes = std::make_shared<std::vector<std::uint8_t>>(serializeProject(scene, settingsText()));
  const std::filesystem::path path = dataDir_ / kAutosaveName;
  workers_.emplace_back([this, path, bytes, fingerprint] {
    auto error = std::make_shared<std::string>();
    const bool ok = writeFileAtomic(path, *bytes, error.get());
    std::lock_guard lock(asyncMutex_);
    asyncResults_.push_back([this, ok, error, fingerprint] {
      autosaveRunning_ = false;
      if (ok) {
        autosavedFingerprint_ = fingerprint;
        statusMessage = "Autosaved";
      } else {
        statusMessage = "Autosave failed: " + *error;
      }
    });
  });
}

void App::drawProjectDialogs() {
  if (offerRecovery_ && !ImGui::IsPopupOpen("Recover Previous Session?")) ImGui::OpenPopup("Recover Previous Session?");
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (ImGui::BeginPopupModal("Recover Previous Session?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextUnformatted("PleGL Sculpt did not close normally last time.");
    std::error_code ec;
    const auto when = std::filesystem::last_write_time(dataDir_ / kAutosaveName, ec);
    if (!ec) {
      const auto age =
          std::chrono::duration_cast<std::chrono::minutes>(std::filesystem::file_time_type::clock::now() - when);
      const int minutes = static_cast<int>(age.count());
      if (minutes < 1)
        ImGui::TextUnformatted("An autosave from less than a minute ago is available.");
      else
        ImGui::Text("An autosave from %d minute%s ago is available.", minutes, minutes == 1 ? "" : "s");
    }
    ImGui::Spacing();
    if (ImGui::Button("Recover", ImVec2(120, 0))) {
      offerRecovery_ = false;
      loadProjectAsync(dataDir_ / kAutosaveName, true);
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Discard", ImVec2(120, 0))) {
      offerRecovery_ = false;
      std::filesystem::remove(dataDir_ / kAutosaveName, ec);
      statusMessage = "Discarded the previous session";
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  if (askSaveChanges_ && !ImGui::IsPopupOpen("Save changes?")) ImGui::OpenPopup("Save changes?");
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (ImGui::BeginPopupModal("Save changes?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    const std::string name = projectPath_.empty() ? "Untitled" : pathToUtf8(projectPath_.filename());
    ImGui::Text("Save changes to %s before continuing?", name.c_str());
    ImGui::Spacing();
    const bool busy = savingProject_;
    ImGui::BeginDisabled(busy);
    if (ImGui::Button(busy ? "Saving..." : "Save", ImVec2(110, 0))) {
      askSaveChanges_ = false;
      ImGui::CloseCurrentPopup();
      auto then = [this] { runPendingAction(); };
      if (!projectPath_.empty()) {
        saveProjectTo(projectPath_, then);
      } else {
        auto* ctx = new DialogContext{this, &asyncMutex_, &asyncResults_,
                                      [then](App& app, std::filesystem::path path) { app.saveProjectTo(path, then); },
                                      [](App& app) { app.pendingAction_ = PendingAction::None; }};
        SDL_ShowSaveFileDialog(dialogDone, ctx, window_, kProjectFilter, 1, nullptr);
      }
    }
    ImGui::SameLine();
    if (ImGui::Button("Don't Save", ImVec2(110, 0))) {
      askSaveChanges_ = false;
      ImGui::CloseCurrentPopup();
      runPendingAction();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(110, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
      askSaveChanges_ = false;
      pendingAction_ = PendingAction::None;
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndDisabled();
    ImGui::EndPopup();
  }
}

}  // namespace plegl
