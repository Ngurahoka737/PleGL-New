#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "mesh/Mesh.h"
#include "sculpt/Brush.h"
#include "sculpt/Symmetry.h"
#include "spatial/Bvh.h"

namespace plegl {

// How the Pose brush finds the part to turn.
enum class PoseMode : std::uint8_t {
  Topology = 0,  // Everything within one brush radius of the press, measured along the surface.
  Mask = 1,      // The unmasked part connected to the press; it turns about its border with the mask.
};

struct PoseOptions {
  PoseMode mode = PoseMode::Topology;
  // Width of the bend at the joint as a fraction of the brush radius, 0..1 (0: a hard step).
  float softness = 0.25f;
  bool operator==(const PoseOptions&) const = default;
};

struct PoseLimits {
  double timeCapMs = 0.0;        // A build that takes longer stops with Aborted; 0 for none.
  std::size_t raceVisitCap = 0;  // Most vertices the root search visits; 0: max(1M, 2 x region).
};

enum class PoseFailure : std::uint8_t {
  None,
  NoSurface,  // Nothing under the press.
  Masked,     // The press is on a masked (Topology mode: fully masked) vertex.
  NoJoint,    // The joint loop lies around the press, so there is no part to bend (a flat area).
  Aborted,    // PoseLimits::timeCapMs ran out.
};

// One group of symmetric copies that turns together: one coset of the copies whose parts touch.
struct PoseRig {
  CopyCluster cluster;     // Its frame: `axial` carries the primary rotation to this rig.
  Vec3 pivot{0.0f};        // The joint: length-weighted centre of the loop it turns about.
  bool loose = false;      // The part has no joint loop (a whole loose piece): it turns about its centroid.
  std::vector<Index> verts;          // Vertices that move, ascending.
  std::vector<float> weight;         // Per vertex, share of the rotation in (0, 1].
  std::vector<Vec3> start, startNormal;
  std::vector<std::uint32_t> rigidSlots;  // Indices into verts whose faces all turn rigidly.
  std::vector<Vec3> rootLoop;        // The joint loop as segment end point pairs, at most kPoseLoopPoints.
};

inline constexpr std::size_t kPoseLoopPoints = 512;

// The parts a Pose stroke turns, built once at the press.
struct PoseBuild {
  std::vector<PoseRig> rigs;  // rigs[0] holds the press itself.
  PoseFailure failure = PoseFailure::None;
  OpMask missingOps = 0;      // Symmetric copies of the press that found no surface.
  std::vector<Index> bentNormals;     // Vertices whose normals are recomputed during the drag, ascending.
  std::vector<Index> snapshotLeaves;  // Leaves owning every vertex whose position or normal changes.
  std::vector<Index> refitLeaves;     // Leaves of every face around a moving vertex.
  double buildMs = 0.0;
  std::size_t visited = 0;    // Vertices the root search visited.
};

// Finds the parts to turn for a press at `click` (local space): per symmetric copy the region
// the mode gives, grown from the corners of the face nearest that copy of the press; regions of
// copies that touch form one rig. The region's border loops are the candidate joints; a search
// over the rest of the mesh, one ring per loop at a time, finds the loop the largest part hangs
// on (the root, which stays put) and the parts beyond the other loops (which turn rigidly with
// the region). Weights fall from 1 to 0 across a band of softness x radius centred on the root
// loop, and are scaled by (1 - mask). `filterPerOp` holds each op's face set limits (empty: none):
// vertices they refuse never join a region and never move. `fans` is findNonManifoldFans(mesh).
// Serial and deterministic.
PoseBuild buildPose(const Mesh& mesh, const Bvh& bvh, std::span<const std::pair<Index, Index>> fans,
                    const SymmetryGroup& group, std::span<const FaceSetFilter> filterPerOp, const Vec3& click,
                    float radius, const PoseOptions& options, const PoseLimits& limits = {});

// The rotation vector (axis times angle, radians) that turns `from` about `pivot` toward the
// direction of `target`. Zero when either lies on the pivot or the angle is below 1e-6; a half
// turn goes about the axis perpendicular to `from` and `viewNormal`.
Vec3 poseRotation(const Vec3& from, const Vec3& pivot, const Vec3& target, const Vec3& viewNormal);

// Moves a rig's vertices to their start positions turned by `rotation` (this rig's frame) about
// its pivot, each by its weight's share of the angle, and turns the normals of its rigid slots.
void applyPoseRig(Mesh& mesh, const PoseRig& rig, const Vec3& rotation);

}  // namespace plegl
