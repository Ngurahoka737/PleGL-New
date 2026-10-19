#include "TestUtil.h"
#include "mesh/Primitives.h"

using namespace plegl;

TEST_CASE("closed primitives are valid, outward and genus 0") {
  struct Case {
    const char* name;
    Mesh mesh;
  };
  Case cases[] = {
      {"uv sphere", makeUvSphere(24, 12)},
      {"icosphere", makeIcosphere(3)},
      {"cube", makeCube(5)},
      {"quad sphere", makeQuadSphere(8)},
  };
  for (auto& c : cases) {
    INFO(c.name);
    test::requireValid(c.mesh);
    test::requireOutward(c.mesh);
    CHECK(test::eulerCharacteristic(c.mesh) == 2);
    for (Index v = 0; v < c.mesh.vertexCount(); ++v) CHECK_FALSE(c.mesh.isBoundaryVertex(v));
  }
}

TEST_CASE("primitive counts match their formulas") {
  CHECK(makeUvSphere(24, 12).vertexCount() == 24 * 11 + 2);
  CHECK(makeIcosphere(2).faceCount() == 20 * 16);
  CHECK(makeQuadSphere(10).vertexCount() == 6 * 100 + 2);
  CHECK(makeCube(3).faceCount() == 6 * 9);
  CHECK(makePlane(4).vertexCount() == 25);
}

TEST_CASE("quad sphere is all quads with valence 4 except 8 corners") {
  const Mesh m = makeQuadSphere(12);
  for (Index f = 0; f < m.faceCount(); ++f) CHECK(m.faceSize(f) == 4);
  int valence3 = 0, valence4 = 0;
  for (Index v = 0; v < m.vertexCount(); ++v) {
    const int val = m.valence(v);
    valence3 += val == 3;
    valence4 += val == 4;
  }
  CHECK(valence3 == 8);
  CHECK(valence4 == m.vertexCount() - 8);
  for (const Vec3& p : m.positions) CHECK(glm::length(p) == doctest::Approx(1.0f));
}

TEST_CASE("plane faces +Y and has an open border") {
  const Mesh m = makePlane(3);
  test::requireValid(m);
  for (Index f = 0; f < m.faceCount(); ++f) CHECK(m.faceAreaNormal(f).y > 0.0f);
  int boundary = 0;
  for (Index v = 0; v < m.vertexCount(); ++v) boundary += m.isBoundaryVertex(v);
  CHECK(boundary == 12);
}
