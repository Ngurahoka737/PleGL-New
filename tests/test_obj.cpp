#include <filesystem>

#include "TestUtil.h"
#include "io/Obj.h"
#include "mesh/Primitives.h"

using namespace plegl;

TEST_CASE("parseObj reads all face index forms and negative indices") {
  const char* text =
      "# comment\n"
      "o thing\n"
      "v 0 0 0\n"
      "v 1 0 0\r\n"
      "v 1 1 0\n"
      "v 0 1 +0\n"
      "vt 0 0\n"
      "vn 0 0 1\n"
      "f 1/1/1 2/1/1 3/1/1\n"
      "f -4//1 -2//1 -1//1\n";
  ObjImportResult r = parseObj(text);
  REQUIRE(r.ok);
  CHECK(r.mesh.vertexCount() == 4);
  CHECK(r.mesh.faceCount() == 2);
  test::requireValid(r.mesh);
  CHECK(r.mesh.edgeCount() == 5);
}

TEST_CASE("parseObj rejects bad input with a line number") {
  CHECK_FALSE(parseObj("v 0 0\nf 1 2 3\n").ok);
  ObjImportResult r = parseObj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n");
  CHECK_FALSE(r.ok);
  CHECK(r.error.find("line 4") != std::string::npos);
  CHECK_FALSE(parseObj("v 0 0 0\n").ok);
  // Non-finite coordinates parse as numbers but could never be sculpted or saved with levels.
  for (const char* bad : {"nan", "inf", "-inf", "infinity"}) {
    INFO(bad);
    const std::string text = std::string("v 0 0 0\nv 1 0 0\nv ") + bad + " 1 0\nf 1 2 3\n";
    ObjImportResult n = parseObj(text);
    CHECK_FALSE(n.ok);
    CHECK(n.error.find("line 3") != std::string::npos);
  }
}

TEST_CASE("OBJ round trip preserves positions, faces and topology") {
  const Mesh src = makeQuadSphere(6);
  const auto path = std::filesystem::temp_directory_path() / "plegl_roundtrip.obj";
  REQUIRE(exportObj(src, path));
  ObjImportResult r = importObj(path);
  std::filesystem::remove(path);
  REQUIRE(r.ok);
  CHECK(r.report.ok());
  const Mesh& m = r.mesh;
  REQUIRE(m.vertexCount() == src.vertexCount());
  REQUIRE(m.faceCount() == src.faceCount());
  for (Index v = 0; v < m.vertexCount(); ++v) CHECK(m.positions[v] == src.positions[v]);  // Exact: shortest round-trip floats.
  for (Index f = 0; f < m.faceCount(); ++f) {
    std::vector<Index> a, b;
    m.forEachFaceVertex(f, [&](Index v) { a.push_back(v); });
    src.forEachFaceVertex(f, [&](Index v) { b.push_back(v); });
    CHECK(a == b);
  }
  test::requireValid(m);
}
