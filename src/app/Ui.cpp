// Dear ImGui panels. Deliberately minimal: the PRD puts engine and brush feel before UI.
#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>

#include <glm/gtc/quaternion.hpp>

#include "App.h"
#include "core/Parallel.h"
#include "mesh/Primitives.h"

namespace plegl {
namespace {

struct MeshInfo {
  std::uint64_t version = 0;
  Index triangles = 0, quads = 0, ngons = 0, edges = 0;
  Index valence4 = 0, interior = 0;
};

const MeshInfo& meshInfo(const SceneObject& obj) {
  static std::unordered_map<std::uint32_t, MeshInfo> cache;
  MeshInfo& info = cache[obj.id];
  if (info.version == obj.topologyVersion) return info;
  info = {};
  info.version = obj.topologyVersion;
  const Mesh& m = obj.mesh;
  for (Index f = 0; f < m.faceCount(); ++f) {
    const Index n = m.faceSize(f);
    (n == 3 ? info.triangles : n == 4 ? info.quads : info.ngons)++;
  }
  info.edges = m.edgeCount();
  for (Index v = 0; v < m.vertexCount(); ++v) {
    if (m.vertHe[v] == kInvalid || m.isBoundaryVertex(v)) continue;
    ++info.interior;
    info.valence4 += m.valence(v) == 4;
  }
  return info;
}

void sectionHeader(const char* label) {
  ImGui::Spacing();
  ImGui::TextDisabled("%s", label);
  ImGui::Separator();
}

bool primitiveRow(const char* button, int* value, int lo, int hi, const char* fmt) {
  ImGui::PushID(button);
  const bool pressed = ImGui::Button(button, ImVec2(ImGui::GetContentRegionAvail().x * 0.48f, 0));
  ImGui::SameLine();
  ImGui::SetNextItemWidth(-1);
  ImGui::SliderInt("##res", value, lo, hi, fmt);
  ImGui::PopID();
  return pressed;
}

}  // namespace

void App::drawUi() {
  const ImGuiViewport* vp = ImGui::GetMainViewport();
  const float scale = ImGui::GetStyle().FontScaleDpi > 0 ? ImGui::GetStyle().FontScaleDpi : 1.0f;
  const float leftW = 210.0f * scale;
  const float rightW = 270.0f * scale;

  // ---- Menu bar ----
  float menuH = 0.0f;
  if (ImGui::BeginMainMenuBar()) {
    menuH = ImGui::GetWindowSize().y;
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("New Scene", "Ctrl+N")) newScene();
      if (ImGui::MenuItem("Import OBJ...", "Ctrl+O")) requestImport();
      if (ImGui::MenuItem("Export Selected OBJ...", "Ctrl+E", false, scene.find(selectedId) != nullptr)) requestExport();
      ImGui::Separator();
      if (ImGui::MenuItem("Quit")) quit();
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Add")) {
      if (ImGui::MenuItem("Quad Sphere")) addPrimitive("Sphere", makeQuadSphere(primitives.quadSphereResolution));
      if (ImGui::MenuItem("UV Sphere")) addPrimitive("UV Sphere", makeUvSphere(primitives.uvSegments, primitives.uvRings));
      if (ImGui::MenuItem("Icosphere")) addPrimitive("Icosphere", makeIcosphere(primitives.icoSubdivisions));
      if (ImGui::MenuItem("Cube")) addPrimitive("Cube", makeCube(primitives.cubeResolution));
      if (ImGui::MenuItem("Plane")) addPrimitive("Plane", makePlane(primitives.planeResolution));
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
      ImGui::MenuItem("Wireframe", nullptr, &view.wireframe);
      ImGui::MenuItem("Grid", nullptr, &view.grid);
      if (ImGui::MenuItem("Frame Selected", "Home")) frameScene();
      ImGui::Separator();
      ImGui::MenuItem("ImGui Demo", nullptr, &showDemo);
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Help")) {
      ImGui::TextUnformatted("Alt + Left drag     Orbit (around the surface under the cursor)");
      ImGui::TextUnformatted("Alt + Middle drag   Pan");
      ImGui::TextUnformatted("Alt + Right drag    Zoom (or mouse wheel)");
      ImGui::TextUnformatted("G / R / S           Move / Rotate / Scale gizmo");
      ImGui::TextUnformatted("Shift + D           Duplicate");
      ImGui::TextUnformatted("X / Delete          Delete");
      ImGui::TextUnformatted("Home                Frame selected");
      ImGui::TextUnformatted("Tab                 Object / Sculpt mode");
      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }

  const float statusH = ImGui::GetFrameHeight() + 4.0f * scale;
  const float bodyY = vp->Pos.y + menuH;
  const float bodyH = std::max(1.0f, vp->Size.y - menuH - statusH);
  const ImGuiWindowFlags panelFlags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                      ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar |
                                      ImGuiWindowFlags_NoSavedSettings;

  // ---- Left panel: tools ----
  ImGui::SetNextWindowPos(ImVec2(vp->Pos.x, bodyY));
  ImGui::SetNextWindowSize(ImVec2(leftW, bodyH));
  if (ImGui::Begin("Tools", nullptr, panelFlags)) {
    int m = static_cast<int>(mode);
    ImGui::RadioButton("Object", &m, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Sculpt", &m, 1);
    if (!strokeActive()) mode = static_cast<Mode>(m);

    if (mode == Mode::Sculpt) {
      sectionHeader("Brush");
      // Two columns of brush buttons; the active one is highlighted.
      const float bw = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
      for (int i = 0; i < kBrushCount; ++i) {
        if (i % 2 == 1) ImGui::SameLine();
        const bool active = static_cast<int>(sculpt.brush) == i;
        if (active) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        if (ImGui::Button(kBrushNames[i], ImVec2(bw, 0))) sculpt.brush = static_cast<BrushKind>(i);
        if (active) ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Shortcut: %s", kBrushKeys[i]);
      }

      ImGui::SetNextItemWidth(-1);
      ImGui::SliderFloat("##radius", &sculpt.radiusPx, 2.0f, 1000.0f, "radius %.0f px", ImGuiSliderFlags_Logarithmic);
      float& strength = sculpt.strength[static_cast<int>(sculpt.brush)];
      ImGui::SetNextItemWidth(-1);
      ImGui::SliderFloat("##strength", &strength, 0.0f, 1.0f, "strength %.2f");
      if (sculpt.brush != BrushKind::Smooth && sculpt.brush != BrushKind::Grab) {
        int inv = sculpt.invert ? 1 : 0;
        ImGui::RadioButton("Add", &inv, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Subtract", &inv, 1);
        sculpt.invert = inv != 0;
      }
      static const char* kFalloffs[] = {"Smooth falloff", "Sharp falloff", "Linear falloff", "Constant falloff"};
      int f = static_cast<int>(sculpt.falloff);
      ImGui::SetNextItemWidth(-1);
      if (ImGui::Combo("##falloff", &f, kFalloffs, 4)) sculpt.falloff = static_cast<Falloff>(f);
      if (sculpt.brush != BrushKind::Grab) {
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##spacing", &sculpt.spacing, 0.02f, 1.0f, "spacing %.2f");
      }

      sectionHeader("Pen pressure");
      static const char* kPressure[] = {"Strength", "Radius", "Both", "Off"};
      int pm = static_cast<int>(sculpt.pressure);
      ImGui::SetNextItemWidth(-1);
      if (ImGui::Combo("##pressure", &pm, kPressure, 4)) sculpt.pressure = static_cast<PressureMap>(pm);

      sectionHeader("Symmetry");
      ImGui::Checkbox("Mirror X (X)", &sculpt.symmetryX);

      sectionHeader("Remesh");
      ImGui::SetNextItemWidth(-1);
      ImGui::DragFloat("##voxel", &remesh.voxelSize, remesh.voxelSize * 0.01f, 0.0005f, 1.0f, "voxel size %.4f",
                       ImGuiSliderFlags_Logarithmic);
      remesh.voxelSize = std::clamp(remesh.voxelSize, 0.0005f, 1.0f);
      if (const SceneObject* sel = scene.find(selectedId)) {
        const Aabb b = sel->bvh.bounds();
        if (b.valid()) {
          const float longest = std::max({b.extent().x, b.extent().y, b.extent().z});
          const int across = static_cast<int>(std::ceil(longest / remesh.voxelSize));
          ImGui::TextDisabled("%d voxels across", across);
          if (across + 7 > 512) {
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.4f, 1.0f), "Voxel size too small (limit 505).");
          } else if (across < 48) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.4f, 1.0f));
            ImGui::TextWrapped("Large voxel size may remove surface details.");
            ImGui::PopStyleColor();
          }
        }
      }
      ImGui::BeginDisabled(remeshing() || strokeActive() || !scene.find(selectedId));
      if (ImGui::Button(remeshing() ? "Remeshing..." : "Voxel Remesh (Ctrl+R)", ImVec2(-1, 0))) requestRemesh();
      ImGui::EndDisabled();
      if (!lastRemeshInfo.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("Last: %s", lastRemeshInfo.c_str());
        ImGui::PopStyleColor();
      }

      sectionHeader("History");
      ImGui::BeginDisabled(!undoStack.canUndo() || strokeActive());
      if (ImGui::Button("Undo")) undo();
      ImGui::EndDisabled();
      ImGui::SameLine();
      ImGui::BeginDisabled(!undoStack.canRedo() || strokeActive());
      if (ImGui::Button("Redo")) redo();
      ImGui::EndDisabled();
      ImGui::SameLine();
      ImGui::TextDisabled("%.0f MB", static_cast<double>(undoStack.bytes()) / (1024.0 * 1024.0));

      ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
      ImGui::TextWrapped("Shift+drag: Smooth. Ctrl+drag: invert. [ ]: radius, hold F and move: radius.");
      ImGui::PopStyleColor();
    } else {
      sectionHeader("Transform");
      int g = static_cast<int>(gizmo);
      ImGui::RadioButton("Move", &g, 0);
      ImGui::SameLine();
      ImGui::RadioButton("Rotate", &g, 1);
      ImGui::SameLine();
      ImGui::RadioButton("Scale", &g, 2);
      gizmo = static_cast<GizmoOp>(g);
    }

    sectionHeader("Add");
    if (primitiveRow("Quad Sphere", &primitives.quadSphereResolution, 4, 256, "res %d"))
      addPrimitive("Sphere", makeQuadSphere(primitives.quadSphereResolution));
    if (primitiveRow("UV Sphere", &primitives.uvSegments, 8, 512, "seg %d")) {
      primitives.uvRings = std::max(primitives.uvSegments / 2, 2);
      addPrimitive("UV Sphere", makeUvSphere(primitives.uvSegments, primitives.uvRings));
    }
    if (primitiveRow("Icosphere", &primitives.icoSubdivisions, 0, 8, "sub %d"))
      addPrimitive("Icosphere", makeIcosphere(primitives.icoSubdivisions));
    if (primitiveRow("Cube", &primitives.cubeResolution, 1, 256, "res %d"))
      addPrimitive("Cube", makeCube(primitives.cubeResolution));
    if (primitiveRow("Plane", &primitives.planeResolution, 1, 512, "res %d"))
      addPrimitive("Plane", makePlane(primitives.planeResolution));
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("Quad Sphere is the best start for sculpting.");
    ImGui::PopStyleColor();

    sectionHeader("View");
    ImGui::SetNextItemWidth(-1);
    ImGui::Combo("##matcap", &view.matcap, Renderer::kMatcapNames, Renderer::kMatcapCount);
    ImGui::Checkbox("Wireframe", &view.wireframe);
    if (view.wireframe) {
      ImGui::SetNextItemWidth(-1);
      ImGui::SliderFloat("##wireopacity", &view.wireframeOpacity, 0.05f, 1.0f, "opacity %.2f");
    }
    ImGui::Checkbox("Grid", &view.grid);
    ImGui::ColorEdit3("Top", &view.backgroundTop.x, ImGuiColorEditFlags_NoInputs);
    ImGui::SameLine();
    ImGui::ColorEdit3("Bottom", &view.backgroundBottom.x, ImGuiColorEditFlags_NoInputs);
    bool vs = vsync;
    if (ImGui::Checkbox("VSync", &vs)) setVsync(vs);
  }
  ImGui::End();

  // ---- Right panel: scene, object, input, performance ----
  ImGui::SetNextWindowPos(ImVec2(vp->Pos.x + vp->Size.x - rightW, bodyY));
  ImGui::SetNextWindowSize(ImVec2(rightW, bodyH));
  if (ImGui::Begin("Inspector", nullptr, panelFlags)) {
    sectionHeader("Scene");
    std::uint32_t toDelete = 0;
    if (ImGui::BeginChild("objects", ImVec2(0, 140.0f * scale), ImGuiChildFlags_Borders)) {
      for (const auto& o : scene.objects()) {
        ImGui::PushID(static_cast<int>(o->id));
        ImGui::Checkbox("##vis", &o->visible);
        ImGui::SameLine();
        if (ImGui::Selectable(o->name.c_str(), o->id == selectedId)) selectedId = o->id;
        if (ImGui::BeginPopupContextItem()) {
          if (ImGui::MenuItem("Duplicate")) {
            selectedId = o->id;
            duplicateSelected();
          }
          if (ImGui::MenuItem("Delete")) toDelete = o->id;
          ImGui::EndPopup();
        }
        ImGui::PopID();
      }
      if (scene.objects().empty()) ImGui::TextDisabled("Empty. Use Add to create a mesh.");
    }
    ImGui::EndChild();
    if (toDelete) {
      selectedId = toDelete;
      deleteSelected();
    }

    if (SceneObject* obj = scene.find(selectedId)) {
      sectionHeader("Object");
      ImGui::SetNextItemWidth(-1);
      ImGui::InputText("##name", &obj->name);
      ImGui::DragFloat3("Position", &obj->transform.position.x, 0.01f);
      // Edit rotation as Euler degrees; keep a cache so values do not jump while dragging.
      static std::uint32_t eulerId = 0;
      static Quat eulerQuat{1, 0, 0, 0};
      static Vec3 eulerDeg{0.0f};
      if (eulerId != obj->id || eulerQuat != obj->transform.rotation) {
        eulerDeg = glm::degrees(glm::eulerAngles(obj->transform.rotation));
        eulerId = obj->id;
      }
      if (ImGui::DragFloat3("Rotation", &eulerDeg.x, 0.5f)) obj->transform.rotation = Quat(glm::radians(eulerDeg));
      eulerQuat = obj->transform.rotation;
      ImGui::DragFloat3("Scale", &obj->transform.scale.x, 0.01f, 0.001f, 1000.0f);
      if (ImGui::Button("Duplicate")) duplicateSelected();
      ImGui::SameLine();
      if (ImGui::Button("Delete")) deleteSelected();
      ImGui::SameLine();
      if (ImGui::Button("Frame")) frameScene();

      if (SceneObject* still = scene.find(selectedId)) {
        const MeshInfo& info = meshInfo(*still);
        const Mesh& mesh = still->mesh;
        ImGui::Text("Vertices   %d", mesh.vertexCount());
        ImGui::Text("Faces      %d", mesh.faceCount());
        ImGui::Text("Edges      %d", info.edges);
        const float total = static_cast<float>(std::max<Index>(mesh.faceCount(), 1));
        ImGui::Text("Quads %.1f%%  Tris %.1f%%  N-gons %.1f%%", 100.0f * info.quads / total,
                    100.0f * info.triangles / total, 100.0f * info.ngons / total);
        if (info.interior > 0)
          ImGui::Text("Valence 4  %.2f%%", 100.0f * info.valence4 / info.interior);
        ImGui::Text("BVH leaves %zu", still->bvh.leaves().size());
      }
    }

    sectionHeader("Pen tablet");
    if (pen.deviceId == 0 && !pen.inProximity) {
      ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
      ImGui::TextWrapped("No pen seen yet. Bring the stylus near the tablet.");
      ImGui::PopStyleColor();
    } else {
      ImGui::Text("%s%s", pen.inProximity ? "In range" : "Out of range", pen.eraser ? " (eraser)" : "");
      char overlay[32];
      std::snprintf(overlay, sizeof(overlay), "pressure %.3f", pen.pressure);
      ImGui::ProgressBar(pen.pressure, ImVec2(-1, 0), overlay);
      ImGui::Text("Tilt  x %.0f  y %.0f", pen.tiltX, pen.tiltY);
    }

    sectionHeader("Performance");
    ImGui::Text("%.0f FPS  (%.2f ms)", stats.fps, stats.frameMs);
    ImGui::Text("Hover raycast  %.1f us", stats.raycastUs);
    ImGui::Text("Last dab  %.2f ms", stats.dabMs);
    ImGui::Text("Input to dab  %.2f ms", stats.inputToDabMs);
    ImGui::Text("Input to frame  %.1f ms", stats.inputToFrameMs);
    const RenderStats& rs = renderStats();
    ImGui::Text("Uploads  %d full, %d partial", rs.fullUploads, rs.partialUploads);
    ImGui::Text("Worker threads  %zu", workerCount());
    if (ImGui::Button("Bump under cursor (B)")) bumpUnderCursor();
    if (stats.partialTestMs > 0) {
      ImGui::SameLine();
      ImGui::Text("%.2f ms", stats.partialTestMs);
    }
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("%s", glInfo.c_str());
    ImGui::PopStyleColor();
  }
  ImGui::End();

  // ---- Status bar ----
  ImGui::SetNextWindowPos(ImVec2(vp->Pos.x, vp->Pos.y + vp->Size.y - statusH));
  ImGui::SetNextWindowSize(ImVec2(vp->Size.x, statusH));
  if (ImGui::Begin("Status", nullptr, panelFlags | ImGuiWindowFlags_NoScrollbar)) {
    ImGui::TextUnformatted(statusMessage.c_str());
    const char* hint = mode == Mode::Object ? "Object mode  |  Alt+drag navigate  |  G R S transform  |  Tab sculpt"
                                            : "Sculpt mode  |  Alt+drag navigate  |  Tab object";
    const float w = ImGui::CalcTextSize(hint).x;
    ImGui::SameLine(std::max(ImGui::GetCursorPosX() + 20.0f, ImGui::GetWindowWidth() - w - 12.0f * scale));
    ImGui::TextDisabled("%s", hint);
  }
  ImGui::End();

  if (showDemo) ImGui::ShowDemoWindow(&showDemo);

  // The viewport is whatever the panels leave.
  vpX_ = leftW;
  vpY_ = menuH;
  vpW_ = std::max(1.0f, vp->Size.x - leftW - rightW);
  vpH_ = bodyH;
  updateViewportRect();
}

}  // namespace plegl
