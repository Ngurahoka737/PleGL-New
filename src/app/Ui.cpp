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
  std::uint64_t seen = 0;  // Newest version noticed; counted one frame later.
  Index triangles = 0, quads = 0, ngons = 0, edges = 0;
  Index valence4 = 0, interior = 0;
};

// Face and valence statistics, recounted in parallel when the topology changes. Never during a
// stroke (a dynamic topology mesh holds removed elements then) and never in the frame that
// noticed the change, which already pays for the stroke's compaction and GPU upload.
const MeshInfo& meshInfo(const SceneObject& obj, bool strokeActive) {
  static std::unordered_map<std::uint32_t, MeshInfo> cache;
  MeshInfo& info = cache[obj.id];
  if (info.version == obj.topologyVersion || strokeActive) return info;
  if (info.seen != obj.topologyVersion) {
    info.seen = obj.topologyVersion;
    if (info.version != 0) return info;  // Show the old numbers for one more frame.
  }
  const Mesh& m = obj.mesh;
  struct Counts {
    Index triangles = 0, quads = 0, ngons = 0, edges = 0, valence4 = 0, interior = 0;
  };
  constexpr std::size_t kChunk = 16384;
  const std::size_t nf = static_cast<std::size_t>(m.faceCount()), nv = static_cast<std::size_t>(m.vertexCount());
  const std::size_t chunks = (std::max(nf, nv) + kChunk - 1) / kChunk;
  std::vector<Counts> part(chunks);
  parallelFor(0, chunks, 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t c = b; c < e; ++c) {
      Counts& k = part[c];
      for (std::size_t f = c * kChunk; f < std::min(nf, (c + 1) * kChunk); ++f) {
        const Index n = m.faceSize(static_cast<Index>(f));
        (n == 3 ? k.triangles : n == 4 ? k.quads : k.ngons)++;
        // Each edge once: from its lower half-edge, or its only one on a border.
        const Index h0 = m.faceHe[f];
        Index h = h0;
        do {
          const Index t = m.heTwin[h];
          k.edges += t == kInvalid || h < t;
          h = m.heNext[h];
        } while (h != h0);
      }
      for (std::size_t v = c * kChunk; v < std::min(nv, (c + 1) * kChunk); ++v) {
        const auto vi = static_cast<Index>(v);
        if (m.vertHe[vi] == kInvalid || m.isBoundaryVertex(vi)) continue;
        ++k.interior;
        k.valence4 += m.valence(vi) == 4;
      }
    }
  });
  info = {};
  info.version = info.seen = obj.topologyVersion;
  for (const Counts& k : part) {
    info.triangles += k.triangles;
    info.quads += k.quads;
    info.ngons += k.ngons;
    info.edges += k.edges;
    info.valence4 += k.valence4;
    info.interior += k.interior;
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
      if (ImGui::MenuItem("New Scene", "Ctrl+N")) requestNewScene();
      if (ImGui::MenuItem("Open Project...", "Ctrl+O")) requestOpenProject();
      if (ImGui::MenuItem("Save Project", "Ctrl+S")) requestSaveProject(false);
      if (ImGui::MenuItem("Save Project As...", "Ctrl+Shift+S")) requestSaveProject(true);
      ImGui::Separator();
      if (ImGui::MenuItem("Import OBJ...", "Ctrl+Shift+I")) requestImport();
      if (ImGui::MenuItem("Export Selected OBJ...", "Ctrl+E", false, scene.find(selectedId) != nullptr)) requestExport();
      ImGui::Separator();
      ImGui::SetNextItemWidth(120.0f * scale);
      ImGui::SliderFloat("Autosave (min)", &projectSettings.autosaveMinutes, 0.0f, 30.0f, projectSettings.autosaveMinutes > 0 ? "%.0f" : "off");
      ImGui::Separator();
      if (ImGui::MenuItem("Quit")) requestQuit();
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
    if (ImGui::BeginMenu("Mask")) {
      // Ctrl+I imports in Object mode, so it inverts the mask only in Sculpt mode.
      const bool can = canEditMask();
      if (ImGui::MenuItem("Invert Mask", mode == Mode::Sculpt ? "Ctrl+I" : nullptr, false, can))
        applyMask(MaskOp::Invert);
      if (ImGui::MenuItem("Clear Mask", "Alt+M", false, can)) applyMask(MaskOp::Clear);
      if (ImGui::MenuItem("Mask All", "Alt+Shift+M", false, can)) applyMask(MaskOp::Fill);
      if (ImGui::MenuItem("Blur Mask", "Alt+B", false, can)) applyMask(MaskOp::Blur);
      if (ImGui::MenuItem("Sharpen Mask", "Alt+Shift+B", false, can)) applyMask(MaskOp::Sharpen);
      ImGui::Separator();
      ImGui::MenuItem("Show Mask", nullptr, &view.showMask);
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Face Sets")) {
      const bool can = canEditMask();
      if (ImGui::MenuItem("Face Set from Mask", nullptr, false, can)) applyFaceSets(FaceSetOp::FromMask);
      if (ImGui::MenuItem("Face Sets from Loose Parts", nullptr, false, can)) applyFaceSets(FaceSetOp::FromLooseParts);
      if (ImGui::MenuItem("Clear Face Sets", nullptr, false, can)) applyFaceSets(FaceSetOp::Clear);
      ImGui::Separator();
      if (ImGui::MenuItem("Reveal All", "Alt+H", false, can)) applyFaceSets(FaceSetOp::RevealAll);
      if (ImGui::MenuItem("Invert Visibility", nullptr, false, can)) applyFaceSets(FaceSetOp::InvertVisibility);
      // These act on the set under the cursor, which the menu covers, so they are keys only.
      ImGui::MenuItem("Hide Set Under Cursor", "H", false, false);
      ImGui::MenuItem("Show Only Set Under Cursor", "Shift+H", false, false);
      ImGui::MenuItem("Mask Set Under Cursor", "Shift+M", false, false);
      ImGui::Separator();
      ImGui::MenuItem("Show Face Sets", nullptr, &view.showFaceSets);
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Multires")) {
      const SceneObject* sel = scene.find(selectedId);
      const Multires* levels = sel ? sel->multires.get() : nullptr;
      const bool can = canEditMask();
      const std::string refusal = sel ? subdivideRefusal(*sel) : std::string();
      if (ImGui::MenuItem(job() == Job::Subdivide ? "Subdividing..." : "Subdivide", "Ctrl+PgUp", false,
                          can && !busy() && refusal.empty()))
        requestSubdivide();
      if (!refusal.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("%s", refusal.c_str());
      ImGui::Separator();
      const bool canSwitch = can && levels;
      if (ImGui::MenuItem("Level Up", "PgUp", false, canSwitch && levels->active < levels->top())) stepLevel(1);
      if (ImGui::MenuItem("Level Down", "PgDn", false, canSwitch && levels->active > 0)) stepLevel(-1);
      if (ImGui::MenuItem("Highest Level", "Shift+PgUp", false, canSwitch && levels->active < levels->top()))
        setLevel(kMaxMultiresLevels);
      if (ImGui::MenuItem("Lowest Level", "Shift+PgDn", false, canSwitch && levels->active > 0)) setLevel(0);
      ImGui::Separator();
      if (ImGui::MenuItem("Delete Higher Levels", nullptr, false, canSwitch && levels->active < levels->top()))
        deleteHigherLevels();
      if (ImGui::MenuItem("Delete Lower Levels", nullptr, false, canSwitch && levels->active > 0)) deleteLowerLevels();
      ImGui::EndMenu();
    }
    drawLayersMenu();
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
      ImGui::Separator();
      ImGui::TextUnformatted("Alt + M             Clear mask (Alt+Shift+M: mask all)");
      ImGui::TextUnformatted("Alt + B             Blur mask (Alt+Shift+B: sharpen)");
      ImGui::TextUnformatted("Alt + H             Reveal all hidden faces");
      ImGui::TextUnformatted("Ctrl + Page Up      Subdivide (add a level on top)");
      ImGui::TextUnformatted("Page Up / Down      Higher / lower subdivision level");
      ImGui::TextUnformatted("Shift + Page Up/Dn  Highest / lowest level");
      ImGui::TextUnformatted("Ctrl + L            New sculpt layer");
      ImGui::TextUnformatted("Alt + click an eye  Show only that layer (again: show all)");
      ImGui::Separator();
      ImGui::TextUnformatted("Sculpt mode:");
      ImGui::TextUnformatted("M                   Mask brush (Ctrl+drag erases, Shift+drag smooths)");
      ImGui::TextUnformatted("Ctrl + I            Invert mask");
      ImGui::TextUnformatted("P                   Face Set brush (Ctrl+drag grows the set under it)");
      ImGui::TextUnformatted("H / Shift + H       Hide / show only the face set under the cursor");
      ImGui::TextUnformatted("Shift + M           Mask the face set under the cursor");
      ImGui::TextUnformatted("Ctrl + D            Dynamic topology on / off");
      ImGui::TextUnformatted("L                   Show / hide the active sculpt layer");
      ImGui::TextUnformatted("E                   Erase Layer brush");
      ImGui::TextUnformatted("F / R + move        Brush radius / detail size");
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
      const SceneObject* target = scene.find(selectedId);
      const bool onLayer = target && target->mesh.layers.active != 0;
      for (int i = 0; i < kBrushCount; ++i) {
        if (i % 2 == 1) ImGui::SameLine();
        const bool active = static_cast<int>(sculpt.brush) == i;
        // Erase Layer takes detail out of a layer, so it needs one to be the target.
        const bool erase = static_cast<BrushKind>(i) == BrushKind::EraseLayer;
        ImGui::BeginDisabled(erase && !onLayer && !active);
        if (active) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        if (ImGui::Button(kBrushLabels[i], ImVec2(bw, 0))) sculpt.brush = static_cast<BrushKind>(i);
        if (active) ImGui::PopStyleColor();
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
          if (erase)
            ImGui::SetTooltip("Shortcut: E\nRemoves the active sculpt layer's detail where you paint.%s",
                              onLayer ? "" : "\nPick a layer under Sculpt layers first.");
          else
            ImGui::SetTooltip("Shortcut: %s", kBrushKeys[i]);
        }
      }

      ImGui::SetNextItemWidth(-1);
      ImGui::SliderFloat("##radius", &sculpt.radiusPx, 2.0f, 1000.0f, "radius %.0f px", ImGuiSliderFlags_Logarithmic);
      float& strength = sculpt.strength[static_cast<int>(sculpt.brush)];
      ImGui::SetNextItemWidth(-1);
      ImGui::SliderFloat("##strength", &strength, 0.0f, 1.0f, "strength %.2f");
      if (sculpt.brush == BrushKind::Mask) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("Masked areas are protected from every brush. Ctrl+drag: erase. Shift+drag: smooth mask.");
        ImGui::PopStyleColor();
      } else if (sculpt.brush == BrushKind::FaceSet) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("Each stroke paints a new face set. Ctrl+drag: grow the set under the cursor. "
                           "Lower strength paints a smaller core.");
        ImGui::PopStyleColor();
      } else if (sculpt.brush == BrushKind::EraseLayer) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("Takes the active layer's detail back out. The base and other layers stay.");
        ImGui::PopStyleColor();
      } else if (sculpt.brush != BrushKind::Smooth && sculpt.brush != BrushKind::Grab) {
        int inv = sculpt.invert ? 1 : 0;
        ImGui::RadioButton("Add", &inv, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Subtract", &inv, 1);
        sculpt.invert = inv != 0;
      }
      if (sculpt.brush == BrushKind::Smooth && onLayer) {
        ImGui::Checkbox("This layer only", &sculpt.smoothLayerOnly);
        if (ImGui::IsItemHovered())
          ImGui::SetTooltip("Smooths only the active layer's own detail, also with Shift+drag.\n"
                            "Off: smooths the shape you see and keeps the result on this layer.");
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

      sectionHeader("Dynamic topology");
      ImGui::Checkbox("Enabled (Ctrl+D)", &sculpt.dyntopo);
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Brushes add triangles where detail is needed and merge them where it is not.\n"
                          "Apply or delete all sculpt layers to use it on an object with layers.");
      ImGui::BeginDisabled(!sculpt.dyntopo);
      static const char* kDetailModes[] = {"Relative (pixels)", "Constant (units)"};
      int dm = static_cast<int>(sculpt.detailMode);
      ImGui::SetNextItemWidth(-1);
      if (ImGui::Combo("##detailMode", &dm, kDetailModes, 2)) sculpt.detailMode = static_cast<DetailMode>(dm);
      ImGui::SetNextItemWidth(-1);
      if (sculpt.detailMode == DetailMode::Relative) {
        ImGui::SliderFloat("##detailPx", &sculpt.detailPx, 2.0f, 64.0f, "detail %.0f px", ImGuiSliderFlags_AlwaysClamp);
      } else {
        ImGui::DragFloat("##detailSize", &sculpt.detailSize, sculpt.detailSize * 0.01f, 1e-5f, 10.0f,
                         "detail %.4g units", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Target edge length. Hold R and move the mouse to change it.");
      static const char* kRefine[] = {"Subdivide and collapse", "Subdivide only", "Collapse only"};
      int rm = static_cast<int>(sculpt.dyntopoRefine);
      ImGui::SetNextItemWidth(-1);
      if (ImGui::Combo("##refine", &rm, kRefine, 3)) sculpt.dyntopoRefine = static_cast<DyntopoRefine>(rm);
      ImGui::EndDisabled();
      if (const SceneObject* sel = scene.find(selectedId); sculpt.dyntopo && sel && sel->multires) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("Off on this object: it has subdivision levels.");
        ImGui::PopStyleColor();
      } else if (sculpt.dyntopo && sel && !sel->mesh.layers.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.4f, 1.0f));
        ImGui::TextWrapped("This object has sculpt layers, so strokes are refused. Turn this off or apply all layers.");
        ImGui::PopStyleColor();
      } else if (sculpt.dyntopo && (sculpt.brush == BrushKind::Grab || sculpt.brush == BrushKind::Mask ||
                                    sculpt.brush == BrushKind::FaceSet)) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("%s does not change topology.", kBrushLabels[static_cast<int>(sculpt.brush)]);
        ImGui::PopStyleColor();
      }

      sectionHeader("Mask");
      {
        const float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        ImGui::BeginDisabled(!canEditMask());
        if (ImGui::Button("Invert", ImVec2(half, 0))) applyMask(MaskOp::Invert);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Ctrl+I");
        ImGui::SameLine();
        if (ImGui::Button("Clear", ImVec2(half, 0))) applyMask(MaskOp::Clear);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Alt+M");
        if (ImGui::Button("Blur", ImVec2(half, 0))) applyMask(MaskOp::Blur);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Alt+B");
        ImGui::SameLine();
        if (ImGui::Button("Sharpen", ImVec2(half, 0))) applyMask(MaskOp::Sharpen);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Alt+Shift+B");
        ImGui::EndDisabled();
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderInt("##maskSteps", &sculpt.maskFilterSteps, 1, 10, "blur/sharpen steps %d",
                         ImGuiSliderFlags_AlwaysClamp);
        ImGui::Checkbox("Show", &view.showMask);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##maskOpacity", &view.maskOpacity, 0.1f, 1.0f, "opacity %.2f", ImGuiSliderFlags_AlwaysClamp);
      }

      sectionHeader("Face sets");
      {
        ImGui::Checkbox("Limit to one set", &sculpt.faceSetAutoMask);
        if (ImGui::IsItemHovered())
          ImGui::SetTooltip("Every brush changes only the face set where the stroke starts.");
        ImGui::Checkbox("Keep set borders", &sculpt.lockFaceSetBorders);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Vertices where face sets meet do not move.");
        const float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        ImGui::BeginDisabled(!canEditMask());
        if (ImGui::Button("From Mask", ImVec2(half, 0))) applyFaceSets(FaceSetOp::FromMask);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
          ImGui::SetTooltip("Masked faces become a new face set.");
        ImGui::SameLine();
        if (ImGui::Button("Loose Parts", ImVec2(half, 0))) applyFaceSets(FaceSetOp::FromLooseParts);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
          ImGui::SetTooltip("Every separate piece gets its own face set.");
        if (ImGui::Button("Reveal All", ImVec2(half, 0))) applyFaceSets(FaceSetOp::RevealAll);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
          ImGui::SetTooltip("Alt+H. H hides the set under the cursor, Shift+H shows only it.");
        ImGui::SameLine();
        if (ImGui::Button("Clear##faceSets", ImVec2(half, 0))) applyFaceSets(FaceSetOp::Clear);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
          ImGui::SetTooltip("Every face back in one set. Hidden faces stay hidden.");
        ImGui::EndDisabled();
        ImGui::Checkbox("Show##faceSets", &view.showFaceSets);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##faceSetOpacity", &view.faceSetOpacity, 0.1f, 1.0f, "opacity %.2f",
                           ImGuiSliderFlags_AlwaysClamp);
      }

      sectionHeader("Remesh");
      ImGui::SetNextItemWidth(-1);
      ImGui::DragFloat("##voxel", &remesh.voxelSize, remesh.voxelSize * 0.01f, 0.0005f, 1.0f, "edge length %.4f",
                       ImGuiSliderFlags_Logarithmic);
      remesh.voxelSize = std::clamp(remesh.voxelSize, 0.0005f, 1.0f);
      if (const SceneObject* sel = scene.find(selectedId)) {
        const Aabb b = sel->bvh.bounds();
        if (b.valid()) {
          const float longest = std::max({b.extent().x, b.extent().y, b.extent().z});
          // Optimized quads build their layout at twice the edge length, then subdivide.
          const float voxel = remesh.voxelSize * (remesh.optimizeQuads ? 2.0f : 1.0f);
          const int across = static_cast<int>(std::ceil(longest / voxel));
          ImGui::TextDisabled("%d voxels across", across);
          if (across + 7 > 512) {
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.4f, 1.0f), "Edge length too small (grid limit 505).");
          } else if (across < 48) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.4f, 1.0f));
            ImGui::TextWrapped("Large edge length may remove surface details.");
            ImGui::PopStyleColor();
          }
        }
      }
      ImGui::Checkbox("Optimize quads", &remesh.optimizeQuads);
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Raise valence-4 vertices, even out edge lengths and fit the original surface.");
      ImGui::BeginDisabled(busy() || strokeActive() || !scene.find(selectedId));
      if (ImGui::Button(remeshing() ? "Remeshing..." : "Remesh (Ctrl+R)", ImVec2(-1, 0))) requestRemesh();
      ImGui::EndDisabled();
      if (const SceneObject* sel = scene.find(selectedId); sel && sel->multires) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("Remeshing removes the subdivision levels (undo brings them back).");
        ImGui::PopStyleColor();
      } else if (sel && !sel->mesh.layers.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("Remeshing bakes the sculpt layers into the mesh (undo brings them back).");
        ImGui::PopStyleColor();
      }
      if (!lastRemeshInfo.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("Last: %s", lastRemeshInfo.c_str());
        ImGui::PopStyleColor();
      }

      sectionHeader("History");
      ImGui::BeginDisabled(!undoStack.canUndo() || strokeActive() || busy());
      if (ImGui::Button("Undo")) undo();
      ImGui::EndDisabled();
      ImGui::SameLine();
      ImGui::BeginDisabled(!undoStack.canRedo() || strokeActive() || busy());
      if (ImGui::Button("Redo")) redo();
      ImGui::EndDisabled();
      ImGui::SameLine();
      ImGui::TextDisabled("%.0f MB", static_cast<double>(undoStack.bytes()) / (1024.0 * 1024.0));

      ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
      ImGui::TextWrapped("Shift+drag: Smooth. Ctrl+drag: invert. [ ]: radius, hold F and move: radius. M: mask. "
                         "H: hide face set.");
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
        std::string label = o->name;
        if (o->multires)
          label += "  L" + std::to_string(o->multires->active) + "/" + std::to_string(o->multires->top());
        label += "##object";  // The id stays the same when the level changes.
        if (ImGui::Selectable(label.c_str(), o->id == selectedId)) selectedId = o->id;
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
        const MeshInfo& info = meshInfo(*still, strokeActive());
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

        sectionHeader("Subdivision levels");
        const bool can = canEditMask();
        if (const Multires* levels = still->multires.get()) {
          ImGui::BeginDisabled(!can);
          for (int k = levels->top(); k >= 0; --k) {
            const Mesh& lm = k == levels->active ? still->mesh : levels->levels[static_cast<std::size_t>(k)].mesh;
            char row[64];
            std::snprintf(row, sizeof(row), "Level %d%s##level%d", k, k == 0 ? " (base)" : "", k);
            ImGui::PushID(k);
            if (ImGui::Selectable(row, k == levels->active, ImGuiSelectableFlags_AllowOverlap) && k != levels->active)
              setLevel(k);
            ImGui::SameLine(ImGui::GetContentRegionAvail().x * 0.55f);
            if (lm.layers.empty())
              ImGui::TextDisabled("%d faces", lm.faceCount());
            else
              ImGui::TextDisabled("%d faces, %zu layer%s", lm.faceCount(), lm.layers.list.size(),
                                  lm.layers.list.size() == 1 ? "" : "s");
            ImGui::PopID();
          }
          ImGui::EndDisabled();
        } else {
          ImGui::TextDisabled("None. Subdivide to add a level.");
        }
        const std::string refusal = subdivideRefusal(*still);
        ImGui::BeginDisabled(!can || busy() || !refusal.empty());
        if (ImGui::Button(job() == Job::Subdivide ? "Subdividing..." : "Subdivide (Ctrl+Page Up)", ImVec2(-1, 0)))
          requestSubdivide();
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
          if (!refusal.empty())
            ImGui::SetTooltip("%s", refusal.c_str());
          else
            ImGui::SetTooltip("Adds a level on top: every face splits into four, smoothly (Catmull-Clark).\n"
                              "Shape on low levels, detail on high ones; edits carry across levels.\n"
                              "Page Up / Page Down switch levels.");
        }
        if (const Multires* levels = still->multires.get()) {
          const float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
          ImGui::BeginDisabled(!can || levels->active == levels->top());
          if (ImGui::Button("Delete Higher", ImVec2(half, 0))) deleteHigherLevels();
          ImGui::EndDisabled();
          if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("Removes the levels above the current one.");
          ImGui::SameLine();
          ImGui::BeginDisabled(!can || levels->active == 0);
          if (ImGui::Button("Delete Lower", ImVec2(half, 0))) deleteLowerLevels();
          ImGui::EndDisabled();
          if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("Makes the current level the base and removes the ones below it.");
          ImGui::TextDisabled("Other levels  %.0f MB", static_cast<double>(levels->bytes()) / (1024.0 * 1024.0));
        }
        drawLayerPanel(*still);
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
    if (sculpt.dyntopo || rs.indexBlockUploads > 0) {
      ImGui::Text("Topology  %.2f ms last dab", stats.topologyMs);
      ImGui::Text("Index blocks  %d  (draw runs %d)", rs.indexBlockUploads, rs.drawRuns);
      const StrokeTopologyStats& ts = lastStrokeTopology();
      if (ts.dyntopo) {
        ImGui::Text("Last stroke  +%d / -%d edges", ts.splits, ts.collapses);
        ImGui::Text("  faces %d -> %d, end %.1f ms", ts.facesBefore, ts.facesAfter, ts.consolidateMs);
      }
    }
    if (stats.maskOpMs > 0.0) ImGui::Text("Last mask/set op  %.1f ms", stats.maskOpMs);
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
    std::string hint;
    if (const SceneObject* sel = scene.find(selectedId); sel && sel->multires)
      hint = "Level " + std::to_string(sel->multires->active) + "/" + std::to_string(sel->multires->top()) + ", " +
             std::to_string(sel->mesh.faceCount()) + " faces  |  PgUp/PgDn level  |  ";
    hint += mode == Mode::Object ? "Object mode  |  Alt+drag navigate  |  G R S transform  |  Tab sculpt"
                                 : "Sculpt mode  |  Alt+drag navigate  |  M mask  |  H hide  |  Ctrl+D dyntopo  |  Tab object";
    // The stroke target on objects with sculpt layers, in amber when strokes would be refused.
    std::string layerText;
    bool refused = false;
    if (const SceneObject* sel = scene.find(selectedId); mode == Mode::Sculpt && sel && !sel->mesh.layers.empty()) {
      const SculptLayer* layer = sel->mesh.layers.find(sel->mesh.layers.active);
      char buf[128];
      if (layer)
        std::snprintf(buf, sizeof(buf), "Layer: %s (%.0f %%)", layer->name.c_str(), layer->strength * 100.0f);
      else
        std::snprintf(buf, sizeof(buf), "Layer: Base");
      layerText = buf;
      refused = !layerStrokeHint().empty();
      if (refused && layer && !layer->visible) layerText += " - hidden, strokes refused";
      else if (refused) layerText += " - strokes refused";
      layerText += "  |  ";
    }
    const float w = ImGui::CalcTextSize(hint.c_str()).x + ImGui::CalcTextSize(layerText.c_str()).x;
    ImGui::SameLine(std::max(ImGui::GetCursorPosX() + 20.0f, ImGui::GetWindowWidth() - w - 12.0f * scale));
    if (!layerText.empty()) {
      if (refused)
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "%s", layerText.c_str());
      else
        ImGui::TextDisabled("%s", layerText.c_str());
      if (refused && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", layerStrokeHint().c_str());
      ImGui::SameLine(0.0f, 0.0f);
    }
    ImGui::TextDisabled("%s", hint.c_str());
  }
  ImGui::End();

  drawProjectDialogs();
  if (showDemo) ImGui::ShowDemoWindow(&showDemo);

  // The viewport is whatever the panels leave.
  vpX_ = leftW;
  vpY_ = menuH;
  vpW_ = std::max(1.0f, vp->Size.x - leftW - rightW);
  vpH_ = bodyH;
  updateViewportRect();
}

void App::drawLayersMenu() {
  if (!ImGui::BeginMenu("Layers")) return;
  const SceneObject* sel = scene.find(selectedId);
  const LayerStack* stack = sel ? &sel->mesh.layers : nullptr;
  const SculptLayer* layer = stack ? stack->find(stack->active) : nullptr;
  const bool can = canEditLayers();
  const bool any = can && stack && !stack->empty();
  const bool one = can && layer;
  if (ImGui::MenuItem("New Layer", "Ctrl+L", false, can && sel)) addLayer();
  if (ImGui::MenuItem("Duplicate Layer", nullptr, false, one)) duplicateLayer();
  if (ImGui::MenuItem("Delete Layer", nullptr, false, one)) deleteLayer();
  ImGui::Separator();
  if (ImGui::MenuItem("Merge Down", nullptr, false, one && stack->indexOf(layer->id) > 0)) mergeLayerDown();
  if (ImGui::MenuItem("Apply Layer", nullptr, false, one)) applyLayer();
  if (ImGui::MenuItem("Apply All Layers", nullptr, false, any)) applyAllLayers();
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    ImGui::SetTooltip("Makes the shape you see the mesh and removes every layer, hidden ones too.");
  ImGui::Separator();
  if (ImGui::MenuItem("Invert Layer", nullptr, false, one)) invertLayer();
  if (ImGui::MenuItem("Mask from Layer", nullptr, false, one)) maskFromLayer();
  ImGui::Separator();
  if (ImGui::MenuItem(layer && !layer->visible ? "Show Layer" : "Hide Layer", "L", false, one))
    toggleActiveLayerVisible();
  if (ImGui::MenuItem("Solo Layer", "Alt+click eye", false, one)) soloLayer(layer->id);
  if (ImGui::MenuItem("Show All Layers", nullptr, false, any)) setAllLayersVisible(true);
  if (ImGui::MenuItem("Hide All Layers", nullptr, false, any)) setAllLayersVisible(false);
  ImGui::EndMenu();
}

void App::drawLayerPanel(SceneObject& obj) {
  const float scale = ImGui::GetStyle().FontScaleDpi > 0 ? ImGui::GetStyle().FontScaleDpi : 1.0f;
  if (obj.multires) {
    const std::string title = "Sculpt layers - level " + std::to_string(obj.multires->active);
    sectionHeader(title.c_str());
  } else {
    sectionHeader("Sculpt layers");
  }
  const LayerStack& stack = obj.mesh.layers;
  const SculptLayer* active = stack.find(stack.active);
  const bool can = canEditLayers();
  ImGui::BeginDisabled(!can);

  // Add, Duplicate and Delete, with the reason when Add or Duplicate cannot run.
  std::string full;
  if (stack.list.size() >= static_cast<std::size_t>(kMaxLayers))
    full = "An object (or subdivision level) can have at most " + std::to_string(kMaxLayers) + " sculpt layers.";
  else if (objectLayerBytes(obj) + (stack.empty() ? 2 : 1) * obj.mesh.positions.size() * sizeof(Vec3) > kMaxObjectLayerBytes)
    full = "The sculpt layers of one object may use at most 1 GB.";
  const float third = (ImGui::GetContentRegionAvail().x - 2.0f * ImGui::GetStyle().ItemSpacing.x) / 3.0f;
  ImGui::BeginDisabled(!full.empty());
  if (ImGui::Button("Add", ImVec2(third, 0))) addLayer();
  ImGui::EndDisabled();
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    ImGui::SetTooltip("%s", full.empty() ? "A new empty layer on top (Ctrl+L). Strokes go into it." : full.c_str());
  ImGui::SameLine();
  ImGui::BeginDisabled(!active || !full.empty());
  if (ImGui::Button("Duplicate", ImVec2(third, 0))) duplicateLayer();
  ImGui::EndDisabled();
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    ImGui::SetTooltip("%s", !full.empty() ? full.c_str() : "A hidden copy of the active layer.");
  ImGui::SameLine();
  ImGui::BeginDisabled(!active);
  if (ImGui::Button("Delete##layer", ImVec2(third, 0))) deleteLayer();
  ImGui::EndDisabled();

  if (stack.empty()) {
    ImGui::EndDisabled();
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("No sculpt layers. Strokes change the mesh directly.");
    if (obj.multires) ImGui::TextWrapped("Each subdivision level has its own layers.");
    ImGui::PopStyleColor();
    return;
  }

  // One row per layer, top layer first: eye, name, strength.
  static std::uint32_t renaming = 0, renamingObject = 0;
  static std::string renameText;
  static bool focusRename = false;
  std::uint32_t contextLayer = 0;
  for (int k = static_cast<int>(stack.list.size()) - 1; k >= 0; --k) {
    const SculptLayer& layer = stack.list[static_cast<std::size_t>(k)];
    const std::uint32_t id = layer.id;
    ImGui::PushID(static_cast<int>(id));
    bool visible = layer.visible;
    if (ImGui::Checkbox("##eye", &visible)) {
      if (ImGui::GetIO().KeyAlt)
        soloLayer(id);
      else
        setLayerVisible(id, visible);
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show or hide (L). Alt+click: show only this layer.");
    ImGui::SameLine();
    const float nameW = ImGui::GetContentRegionAvail().x * 0.48f;
    if (renaming == id && renamingObject == obj.id) {
      ImGui::SetNextItemWidth(nameW);
      if (focusRename) {
        ImGui::SetKeyboardFocusHere();
        focusRename = false;
      }
      const bool enter = ImGui::InputText("##rename", &renameText,
                                          ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
      if (enter) renameLayer(id, renameText);
      if (enter || ImGui::IsItemDeactivated()) renaming = 0;  // Esc or clicking away cancels.
    } else {
      if (!layer.visible) ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
      const std::string label = layer.name + "##name";
      if (ImGui::Selectable(label.c_str(), id == stack.active, ImGuiSelectableFlags_AllowDoubleClick,
                            ImVec2(nameW, 0))) {
        selectLayer(id);
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
          renaming = id;
          renamingObject = obj.id;
          renameText = layer.name;
          focusRename = true;
        }
      }
      if (!layer.visible) ImGui::PopStyleColor();
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Click: sculpt on this layer. Double-click: rename. Right-click: more.");
      if (ImGui::BeginPopupContextItem("##layerMenu")) {
        contextLayer = id;
        ImGui::EndPopup();
      }
    }
    ImGui::SameLine();
    // Strength in percent: dragging moves the shape live and becomes one undo step; Ctrl+click
    // types an exact value (up to 1000 %).
    float percent = layer.strength * 100.0f;
    ImGui::SetNextItemWidth(-1);
    const bool changed = ImGui::SliderFloat("##strength", &percent, -100.0f, 100.0f, "%.0f %%");
    if (ImGui::IsItemActivated()) beginLayerStrengthDrag(id);
    if (changed) updateLayerStrengthDrag(percent / 100.0f);
    if (ImGui::IsItemDeactivated()) endLayerStrengthDrag();
    ImGui::PopID();
  }
  // Right-click menu of a layer row; it acts on that layer, so it becomes the target first.
  if (contextLayer != 0) ImGui::OpenPopup("layerActions");
  static std::uint32_t menuLayer = 0;
  if (contextLayer != 0) menuLayer = contextLayer;
  if (ImGui::BeginPopup("layerActions")) {
    const int index = stack.indexOf(menuLayer);
    const SculptLayer* layer = index >= 0 ? &stack.list[static_cast<std::size_t>(index)] : nullptr;
    if (!layer) {
      ImGui::CloseCurrentPopup();
    } else {
      ImGui::TextDisabled("%s", layer->name.c_str());
      ImGui::Separator();
      auto onLayer = [&](void (App::*op)()) {
        selectLayer(menuLayer);
        (this->*op)();
      };
      if (ImGui::MenuItem("Duplicate", nullptr, false, full.empty())) onLayer(&App::duplicateLayer);
      if (ImGui::MenuItem("Merge Down", nullptr, false, index > 0)) onLayer(&App::mergeLayerDown);
      if (ImGui::MenuItem("Apply")) onLayer(&App::applyLayer);
      if (ImGui::MenuItem("Invert")) onLayer(&App::invertLayer);
      if (ImGui::MenuItem("Mask from Layer")) onLayer(&App::maskFromLayer);
      if (ImGui::MenuItem("Rename")) {
        renaming = menuLayer;
        renamingObject = obj.id;
        renameText = layer->name;
        focusRename = true;
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Delete")) onLayer(&App::deleteLayer);
    }
    ImGui::EndPopup();
  }
  // The base: strokes on it change the mesh under every layer.
  ImGui::Dummy(ImVec2(ImGui::GetFrameHeight(), 0));
  ImGui::SameLine();
  if (ImGui::Selectable("Base##layers", stack.active == 0)) selectLayer(0);
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Sculpt the mesh under all layers.");
  ImGui::EndDisabled();

  ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
  ImGui::Text("Sculpting on: %s", active ? active->name.c_str() : "Base");
  if (obj.multires) ImGui::TextWrapped("Each subdivision level has its own layers.");
  ImGui::Text("Layers use %.1f MB", static_cast<double>(objectLayerBytes(obj)) / (1024.0 * 1024.0));
  ImGui::PopStyleColor();
  (void)scale;
}

}  // namespace plegl
