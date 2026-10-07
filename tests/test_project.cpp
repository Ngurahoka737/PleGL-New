#include <cstring>
#include <filesystem>
#include <fstream>

#include "TestUtil.h"
#include "io/Project.h"
#include "mesh/Primitives.h"

using namespace plegl;

namespace {

// Faces of a mesh as vertex lists, for comparing topology.
std::vector<std::vector<Index>> faceLists(const Mesh& m) {
  std::vector<std::vector<Index>> out(m.faceCount());
  for (Index f = 0; f < m.faceCount(); ++f) m.forEachFaceVertex(f, [&](Index v) { out[f].push_back(v); });
  return out;
}

Scene makeScene() {
  Scene scene;
  SceneObject& a = scene.add("Head", makeQuadSphere(16));
  a.transform.position = {1.0f, 2.0f, 3.0f};
  a.transform.rotation = glm::angleAxis(0.5f, glm::normalize(Vec3{1.0f, 1.0f, 0.0f}));
  a.transform.scale = {1.0f, 2.0f, 0.5f};
  SceneObject& b = scene.add("Eye (left)", makeIcosphere(2));
  b.visible = false;
  // An n-gon and a triangle survive too.
  const std::vector<Vec3> p{{0, 0, 0}, {1, 0, 0}, {1.5f, 1, 0}, {0.5f, 1.5f, 0}, {-0.5f, 1, 0}, {0, 0, 1}};
  const std::vector<Index> idx{0, 1, 2, 3, 4, 0, 5, 1};
  const std::vector<Index> sizes{5, 3};
  scene.add("Mixed", buildMesh(p, idx, sizes));
  return scene;
}

}  // namespace

TEST_CASE("project round trip keeps meshes, transforms, visibility and settings") {
  const Scene scene = makeScene();
  const std::string settings = "brush Clay\ncamera.distance 4.5\nname with spaces ok\n";
  const std::vector<std::uint8_t> bytes = serializeProject(scene, settings);
  std::string error;
  auto project = parseProject(bytes.data(), bytes.size(), &error);
  INFO(error);
  REQUIRE(project);
  CHECK(project->settings == settings);
  REQUIRE(project->objects.size() == scene.objects().size());
  for (std::size_t i = 0; i < project->objects.size(); ++i) {
    const SceneObject& src = *scene.objects()[i];
    const ProjectObject& dst = project->objects[i];
    CHECK(dst.name == src.name);
    CHECK(dst.visible == src.visible);
    CHECK(dst.transform.position == src.transform.position);
    CHECK(dst.transform.rotation == src.transform.rotation);
    CHECK(dst.transform.scale == src.transform.scale);
    CHECK(dst.mesh.positions == src.mesh.positions);  // Bit exact.
    CHECK(faceLists(dst.mesh) == faceLists(src.mesh));
    test::requireValid(dst.mesh);
    REQUIRE(dst.mesh.normals.size() == dst.mesh.positions.size());
  }
}

TEST_CASE("empty scene round trips") {
  const Scene scene;
  const auto bytes = serializeProject(scene, "");
  auto project = parseProject(bytes.data(), bytes.size());
  REQUIRE(project);
  CHECK(project->objects.empty());
}

TEST_CASE("damaged project files are refused") {
  const auto bytes = serializeProject(makeScene(), "x 1\n");
  std::string error;

  SUBCASE("wrong magic") {
    auto b = bytes;
    b[0] = 'X';
    CHECK_FALSE(parseProject(b.data(), b.size(), &error));
    CHECK(error.find("Not a PleGL") != std::string::npos);
  }
  SUBCASE("every truncation") {
    // Cutting the file anywhere must fail cleanly, never crash or return a partial project.
    for (std::size_t n = 0; n < bytes.size(); n += 1 + n / 50) CHECK_FALSE(parseProject(bytes.data(), n));
  }
  SUBCASE("flipped bit") {
    for (std::size_t at : {std::size_t{40}, bytes.size() / 2, bytes.size() - 20}) {
      auto b = bytes;
      b[at] ^= 0x10;
      CHECK_FALSE(parseProject(b.data(), b.size(), &error));
    }
  }
  SUBCASE("newer format") {
    auto b = bytes;
    b[8] = 99;
    CHECK_FALSE(parseProject(b.data(), b.size(), &error));
    CHECK(error.find("newer") != std::string::npos);
  }
}

TEST_CASE("unknown chunks are skipped") {
  auto bytes = serializeProject(makeScene(), "a 1\n");
  // Insert a chunk "FUTR" with 5 payload bytes after the header, then fix the checksum.
  const std::uint8_t extra[] = {'F', 'U', 'T', 'R', 5, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5};
  bytes.insert(bytes.begin() + 12, std::begin(extra), std::end(extra));
  const std::size_t endChunk = bytes.size() - 16;  // "END " tag, u64 size, u32 crc.
  const std::uint32_t crc = crc32(bytes.data(), endChunk);
  std::memcpy(bytes.data() + bytes.size() - 4, &crc, 4);
  auto project = parseProject(bytes.data(), bytes.size());
  REQUIRE(project);
  CHECK(project->objects.size() == 3);
  CHECK(project->settings == "a 1\n");
}

TEST_CASE("saving writes atomically and loads back") {
  const auto dir = std::filesystem::temp_directory_path() / "plegl_test_project";
  std::filesystem::create_directories(dir);
  const auto path = dir / "head.psculpt";
  std::string error;
  REQUIRE(writeFileAtomic(path, serializeProject(makeScene(), "k v\n"), &error));
  CHECK_FALSE(std::filesystem::exists(dir / "head.psculpt.tmp"));
  // Overwriting an existing file works too.
  REQUIRE(writeFileAtomic(path, serializeProject(makeScene(), "k w\n"), &error));
  auto project = loadProject(path, &error);
  INFO(error);
  REQUIRE(project);
  CHECK(project->settings == "k w\n");
  CHECK(project->objects.size() == 3);
  CHECK_FALSE(loadProject(dir / "missing.psculpt", &error));
  std::filesystem::remove_all(dir);
}

TEST_CASE("crc32 matches the standard check value") {
  const char* s = "123456789";
  CHECK(crc32(reinterpret_cast<const std::uint8_t*>(s), 9) == 0xCBF43926u);
}
