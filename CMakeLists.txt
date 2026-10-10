cmake_minimum_required(VERSION 3.24)
project(PleGLSculpt VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(PLEGL_BUILD_APP "Build the desktop application (needs SDL3, Dear ImGui)" ON)
option(PLEGL_BUILD_TESTS "Build engine unit tests" ON)
option(PLEGL_BUILD_BENCH "Build engine benchmarks" ON)

include(cmake/Dependencies.cmake)

# Common warnings for our own targets only.
add_library(plegl_warnings INTERFACE)
if(MSVC)
  target_compile_options(plegl_warnings INTERFACE /W4 /permissive- /utf-8)
  target_compile_definitions(plegl_warnings INTERFACE NOMINMAX _USE_MATH_DEFINES)
else()
  target_compile_options(plegl_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow)
endif()

# --- Engine: headless, no window or GPU dependencies -------------------------
add_library(plegl_engine STATIC
  src/engine/core/Parallel.cpp
  src/engine/mesh/Mesh.cpp
  src/engine/mesh/LayerStack.cpp
  src/engine/mesh/MeshEdit.cpp
  src/engine/mesh/Primitives.cpp
  src/engine/io/Obj.cpp
  src/engine/io/Project.cpp
  src/engine/spatial/Bvh.cpp
  src/engine/spatial/LeafLayout.cpp
  src/engine/render/LeafIndexBlocks.cpp
  src/engine/scene/Scene.cpp
  src/engine/multires/Multires.cpp
  src/engine/multires/Subdivide.cpp
  src/engine/multires/Propagate.cpp
  src/engine/multires/MultiresOps.cpp
  src/engine/multires/MultiresIo.cpp
  src/engine/sculpt/Brush.cpp
  src/engine/sculpt/Dyntopo.cpp
  src/engine/sculpt/LayerOps.cpp
  src/engine/sculpt/MaskOps.cpp
  src/engine/sculpt/FaceSetOps.cpp
  src/engine/sculpt/Sculptor.cpp
  src/engine/sculpt/StrokeSampler.cpp
  src/engine/sculpt/Undo.cpp
  src/engine/remesh/VoxelGrid.cpp
  src/engine/remesh/SurfaceNets.cpp
  src/engine/remesh/VoxelRemesh.cpp
  src/engine/remesh/QuadRemesh.cpp
)
target_include_directories(plegl_engine PUBLIC src/engine)
target_link_libraries(plegl_engine PUBLIC glm::glm PRIVATE plegl_warnings)
target_compile_definitions(plegl_engine PUBLIC GLM_ENABLE_EXPERIMENTAL GLM_FORCE_XYZW_ONLY)
# Bit-exact results (sculpt layer composites, multires kernels, normals) must not depend on whether
# the compiler fuses a*b+c into one rounding. MSVC's default /fp:precise does not contract.
if(NOT MSVC)
  target_compile_options(plegl_engine PUBLIC -ffp-contract=off)
endif()
find_package(Threads REQUIRED)
target_link_libraries(plegl_engine PUBLIC Threads::Threads)

# --- Application -------------------------------------------------------------
if(PLEGL_BUILD_APP)
  add_library(glad STATIC third_party/glad/src/gl.c)
  target_include_directories(glad PUBLIC third_party/glad/include)

  add_executable(plegl_sculpt WIN32
    src/app/main.cpp
    src/app/App.cpp
    src/app/AppProject.cpp
    src/app/Camera.cpp
    src/app/Renderer.cpp
    src/app/Ui.cpp
  )
  target_link_libraries(plegl_sculpt PRIVATE plegl_engine glad imgui imguizmo SDL3::SDL3-static plegl_warnings)
  set_target_properties(plegl_sculpt PROPERTIES OUTPUT_NAME "PleGLSculpt")
endif()

# --- Tests and benchmarks ----------------------------------------------------
if(PLEGL_BUILD_TESTS)
  enable_testing()
  add_executable(plegl_tests
    tests/test_main.cpp
    tests/test_mesh.cpp
    tests/test_meshedit.cpp
    tests/test_primitives.cpp
    tests/test_obj.cpp
    tests/test_mask.cpp
    tests/test_facesets.cpp
    tests/test_multires.cpp
    tests/test_multires_levels.cpp
    tests/test_multires_io.cpp
    tests/test_project.cpp
    tests/test_bvh.cpp
    tests/test_leaflayout.cpp
    tests/test_dyntopo.cpp
    tests/test_render_blocks.cpp
    tests/test_scene.cpp
    tests/test_sculpt.cpp
    tests/test_remesh.cpp
    tests/test_layers.cpp
    tests/test_layers_undo.cpp
    tests/test_layers_multires.cpp
  )
  target_link_libraries(plegl_tests PRIVATE plegl_engine doctest::doctest plegl_warnings)
  add_test(NAME plegl_tests COMMAND plegl_tests)
endif()

if(PLEGL_BUILD_BENCH)
  add_executable(plegl_bench bench/bench_main.cpp)
  target_link_libraries(plegl_bench PRIVATE plegl_engine plegl_warnings)
endif()
