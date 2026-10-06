#include <glm/gtc/matrix_transform.hpp>

#include "TestUtil.h"
#include "mesh/Primitives.h"
#include "scene/Scene.h"

using namespace plegl;

TEST_CASE("scene picks the nearest visible object through its transform") {
  Scene scene;
  SceneObject& a = scene.add("Sphere", makeQuadSphere(10));
  SceneObject& b = scene.add("Sphere", makeQuadSphere(10));
  CHECK(b.name == "Sphere.001");
  a.transform.position = {0, 0, -5};
  b.transform.position = {0, 0, -10};
  b.transform.scale = Vec3{3.0f};

  const Ray ray{{0, 0, 0}, {0, 0, -1}};
  auto pick = scene.pick(ray);
  REQUIRE(pick);
  CHECK(pick->objectId == a.id);
  CHECK(pick->worldPosition.z == doctest::Approx(-4.0f).epsilon(0.01));
  CHECK(pick->worldNormal.z == doctest::Approx(1.0f).epsilon(0.01));

  a.visible = false;
  pick = scene.pick(ray);
  REQUIRE(pick);
  CHECK(pick->objectId == b.id);
  CHECK(pick->worldPosition.z == doctest::Approx(-7.0f).epsilon(0.01));
}

TEST_CASE("duplicate and remove keep names unique") {
  Scene scene;
  SceneObject& a = scene.add("Head", makeIcosphere(1));
  SceneObject* copy = scene.duplicate(a.id);
  REQUIRE(copy);
  CHECK(copy->name == "Head.001");
  SceneObject* copy2 = scene.duplicate(copy->id);
  CHECK(copy2->name == "Head.002");
  const std::uint32_t removedId = a.id;  // `a` dangles after remove.
  CHECK(scene.remove(removedId));
  CHECK(scene.objects().size() == 2);
  CHECK(scene.find(removedId) == nullptr);
}

TEST_CASE("transform matrix round trips through decompose") {
  Transform t;
  t.position = {1, 2, 3};
  t.rotation = glm::angleAxis(0.7f, glm::normalize(Vec3{1, 1, 0}));
  t.scale = {2, 2, 2};
  const Transform back = Transform::fromMatrix(t.matrix());
  CHECK(glm::length(back.position - t.position) < 1e-4f);
  CHECK(glm::length(back.scale - t.scale) < 1e-4f);
  CHECK(std::abs(glm::dot(back.rotation, t.rotation)) == doctest::Approx(1.0f));
}
