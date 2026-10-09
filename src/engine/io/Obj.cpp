#include "io/Obj.h"

#include <charconv>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <memory>
#include <system_error>

namespace plegl {
namespace {

bool isSpace(char c) { return c == ' ' || c == '\t' || c == '\r'; }

const char* skipSpace(const char* p, const char* end) {
  while (p < end && isSpace(*p)) ++p;
  return p;
}

const char* parseFloat(const char* p, const char* end, float& out) {
  p = skipSpace(p, end);
  if (p < end && *p == '+') ++p;  // from_chars rejects a leading '+'.
  auto [next, ec] = std::from_chars(p, end, out);
  // "nan" and "inf" parse, but a mesh with them cannot be sculpted, subdivided or saved with levels.
  return ec == std::errc() && std::isfinite(out) ? next : nullptr;
}

const char* parseInt(const char* p, const char* end, long& out) {
  if (p < end && *p == '+') ++p;
  auto [next, ec] = std::from_chars(p, end, out);
  return ec == std::errc() ? next : nullptr;
}

}  // namespace

ObjImportResult parseObj(std::string_view text) {
  ObjImportResult result;
  std::vector<Vec3> positions;
  std::vector<Index> indices;
  std::vector<Index> sizes;
  positions.reserve(text.size() / 64);
  indices.reserve(text.size() / 16);

  const char* p = text.data();
  const char* end = p + text.size();
  std::size_t line = 0;
  while (p < end) {
    ++line;
    const char* eol = static_cast<const char*>(std::memchr(p, '\n', static_cast<std::size_t>(end - p)));
    if (!eol) eol = end;
    const char* s = skipSpace(p, eol);

    if (eol - s >= 2 && s[0] == 'v' && isSpace(s[1])) {
      Vec3 v;
      const char* q = parseFloat(s + 2, eol, v.x);
      if (q) q = parseFloat(q, eol, v.y);
      if (q) q = parseFloat(q, eol, v.z);
      if (!q) {
        result.error = "Invalid vertex on line " + std::to_string(line);
        return result;
      }
      positions.push_back(v);
    } else if (eol - s >= 2 && s[0] == 'f' && isSpace(s[1])) {
      const char* q = s + 2;
      Index count = 0;
      for (;;) {
        q = skipSpace(q, eol);
        if (q >= eol) break;
        long idx = 0;
        const char* after = parseInt(q, eol, idx);
        if (!after || idx == 0) {
          result.error = "Invalid face index on line " + std::to_string(line);
          return result;
        }
        // Skip /vt/vn parts.
        while (after < eol && !isSpace(*after)) ++after;
        const long resolved = idx > 0 ? idx - 1 : static_cast<long>(positions.size()) + idx;
        if (resolved < 0 || resolved >= static_cast<long>(positions.size())) {
          result.error = "Face index out of range on line " + std::to_string(line);
          return result;
        }
        indices.push_back(static_cast<Index>(resolved));
        ++count;
        q = after;
      }
      if (count > 0) sizes.push_back(count);
    }
    p = eol + (eol < end ? 1 : 0);
  }

  if (sizes.empty()) {
    result.error = "File contains no faces";
    return result;
  }
  result.mesh = buildMesh(std::move(positions), indices, sizes, &result.report);
  result.ok = true;
  return result;
}

ObjImportResult importObj(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    ObjImportResult r;
    r.error = "Cannot open " + path.string();
    return r;
  }
  const auto size = static_cast<std::size_t>(in.tellg());
  std::string text(size, '\0');
  in.seekg(0);
  in.read(text.data(), static_cast<std::streamsize>(size));
  return parseObj(text);
}

std::string writeObj(const Mesh& mesh) {
  std::string out;
  out.reserve(static_cast<std::size_t>(mesh.vertexCount()) * 70 + static_cast<std::size_t>(mesh.faceCount()) * 40);
  out += "# PleGL Sculpt\n";
  char buf[32];
  auto num = [&](float f) {
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), f);
    out.append(buf, ptr);
  };
  auto integer = [&](long i) {
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), i);
    out.append(buf, ptr);
  };
  for (const Vec3& p : mesh.positions) {
    out += "v ";
    num(p.x);
    out += ' ';
    num(p.y);
    out += ' ';
    num(p.z);
    out += '\n';
  }
  const bool hasNormals = mesh.normals.size() == mesh.positions.size();
  if (hasNormals) {
    for (const Vec3& n : mesh.normals) {
      out += "vn ";
      num(n.x);
      out += ' ';
      num(n.y);
      out += ' ';
      num(n.z);
      out += '\n';
    }
  }
  for (Index f = 0; f < mesh.faceCount(); ++f) {
    out += 'f';
    mesh.forEachFaceVertex(f, [&](Index v) {
      out += ' ';
      integer(v + 1);
      if (hasNormals) {
        out += "//";
        integer(v + 1);
      }
    });
    out += '\n';
  }
  return out;
}

bool exportObj(const Mesh& mesh, const std::filesystem::path& path, std::string* error) {
  const std::string text = writeObj(mesh);
  std::ofstream outFile(path, std::ios::binary | std::ios::trunc);
  if (!outFile) {
    if (error) *error = "Cannot write " + path.string();
    return false;
  }
  outFile.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!outFile) {
    if (error) *error = "Write failed for " + path.string();
    return false;
  }
  return true;
}

}  // namespace plegl
