#pragma once

#include <doctest/doctest.h>

#include "mesh/Mesh.h"

namespace plegl::test {

inline void requireValid(const Mesh& m) {
  const ValidationResult r = validate(m);
  INFO(r.message);
  REQUIRE(r.ok);
}

// For a closed, outward-wound mesh centred at the origin, every face normal points away from it.
inline void requireOutward(const Mesh& m) {
  for (Index f = 0; f < m.faceCount(); ++f) {
    const float d = glm::dot(m.faceAreaNormal(f), m.faceCentroid(f));
    if (d <= 0.0f) {
      INFO("face " << f << " points inward");
      REQUIRE(d > 0.0f);
    }
  }
}

inline Index eulerCharacteristic(const Mesh& m) { return m.vertexCount() - m.edgeCount() + m.faceCount(); }

}  // namespace plegl::test
