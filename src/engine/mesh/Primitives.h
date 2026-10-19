#pragma once

#include "mesh/Mesh.h"

namespace plegl {

// All primitives are centred at the origin with +Y up and outward-facing, counter-clockwise faces.

// Quads between rings, triangle fans at the poles. segments >= 3, rings >= 2.
Mesh makeUvSphere(int segments, int rings, float radius = 1.0f);

// Subdivided icosahedron, all triangles. 20 * 4^subdivisions faces.
Mesh makeIcosphere(int subdivisions, float radius = 1.0f);

// Cube with `resolution` x `resolution` quads per side, edge length `size`.
Mesh makeCube(int resolution, float size = 2.0f);

// Cube projected onto a sphere with an equal-angle mapping. All quads, valence 4 everywhere except
// the 8 cube corners (valence 3). The preferred starting shape for sculpting.
// Vertex count is 6 * resolution^2 + 2.
Mesh makeQuadSphere(int resolution, float radius = 1.0f);

// Flat grid in the XZ plane facing +Y with `resolution` x `resolution` quads.
Mesh makePlane(int resolution, float size = 2.0f);

}  // namespace plegl
