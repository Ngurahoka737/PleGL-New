#pragma once

#include <filesystem>
#include <string>

#include "mesh/Mesh.h"

namespace plegl {

struct ObjImportResult {
  bool ok = false;
  std::string error;   // Set when ok is false.
  Mesh mesh;
  BuildReport report;  // Topology problems found while building; the mesh is still usable.
};

// Reads positions and faces (any polygon size, v, v/vt, v//vn and v/vt/vn forms, negative
// indices). Texture coordinates and file normals are ignored; normals are recomputed.
// All groups and objects in the file are merged into one mesh.
ObjImportResult importObj(const std::filesystem::path& path);
ObjImportResult parseObj(std::string_view text);

// Writes vertex positions, vertex normals and polygon faces.
bool exportObj(const Mesh& mesh, const std::filesystem::path& path, std::string* error = nullptr);
std::string writeObj(const Mesh& mesh);

}  // namespace plegl
