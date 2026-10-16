#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <optional>
#include <unordered_map>
#include <vector>

#include "multires/Propagate.h"
#include "sculpt/Brush.h"
#include "sculpt/Dyntopo.h"
#include "sculpt/Normals.h"
#include "sculpt/PoseOps.h"
#include "sculpt/Undo.h"

namespace plegl {

struct StrokeOptions {
  float strength = 0.5f;
  Falloff falloff = Falloff::Smooth;
  bool invert = false;
  // Symmetry: mirror every dab across the object's local X = 0 / Y = 0 / Z = 0 planes, and repeat
  // it radialCount times around the radial axis (1 = off). See Symmetry.h.
  bool symmetryX = false;
  bool symmetryY = false;
  bool symmetryZ = false;
  Axis radialAxis = Axis::Y;
  int radialCount = 1;
  // Dynamic topology: dabs that pass a detail size split and collapse edges under the brush.
  // Ignored by grab and mask strokes.
  bool dyntopo = false;
  DyntopoOptions dyntopoOptions{};
  // Face set auto-masking: only vertices of the face set under the first dab move (each side of
  // the symmetry plane takes the set under its own first dab). Face set brushes only repaint
  // faces of that set.
  bool faceSetAutoMask = false;
  // Vertices where face sets meet stay put.
  bool lockFaceSetBoundaries = false;
  // Face set brushes: continue the set under the first dab instead of starting a new one.
  bool extendFaceSet = false;
  // Sculpt layers: the layer (by id) strokes commit into, or 0 for the base. Ignored on meshes
  // without layers.
  std::uint32_t layerTarget = 0;
  // Smooth on a layer smooths only that layer's offsets (see LayerSmoothBrush).
  bool smoothLayerOnly = false;

  SymmetrySpec symmetry() const { return sanitized({symmetryX, symmetryY, symmetryZ, radialAxis, radialCount}); }
};

struct DabTiming {
  double totalMs = 0.0;    // Whole dab: query, brush, normals, refit, bookkeeping.
  double brushMs = 0.0;    // Brush kernel only.
  double normalsMs = 0.0;
  int leaves = 0;
  int vertices = 0;        // Vertices whose normals were recomputed.
  double topologyMs = 0.0; // Dynamic topology pass, normals and refit after it.
  int splits = 0;
  int collapses = 0;
  int copies = 1;          // Symmetric copies of the dab (the group's size).
  int clusters = 1;        // Copies left after merging those that nearly coincide.
  double shareMs = 0.0;    // Building the shares of overlapping copies.
  int pinned = 0;          // Vertices held on a symmetry plane or axis.
};

// Snake Hook follow gain for one tick: the share of the way to the cursor the hook covers in
// dtMs, from the brush strength s (1 sticks to the cursor). Ticks of any length compose the same
// as one long tick, so the feel does not depend on the frame rate. dtMs is clamped to [1, 50].
float snakeGain(float strength, double dtMs);

// Most sub-dabs one Snake Hook tick runs; the rest of the way waits for the next tick.
inline constexpr int kMaxSnakeSubDabs = 32;

// What one Snake Hook tick did.
struct SnakeStep {
  int subDabs = 0;
  float remaining = 0.0f;  // Distance from the primary hook to the target after the tick.
  double ms = 0.0;
};

// Pose: above this many recomputed normals, the drag leaves them to the end of the stroke.
inline constexpr std::size_t kLivePoseNormals = 600'000;

// The face set limits a stroke with these options applies per symmetry op for a first dab at
// `center` (local space), as Sculptor uses them.
std::array<FaceSetFilter, kMaxSymmetryCopies> strokeFilters(const SceneObject& object, const StrokeOptions& options,
                                                            const SymmetryGroup& group, const Vec3& center,
                                                            float radius);
// The Pose build a stroke with these options makes for a press at `click`: Sculptor::beginPose
// takes it as `reuse` when nothing changed in between (the hover preview).
PoseBuild buildPoseFor(const SceneObject& object, const StrokeOptions& options, const PoseOptions& pose,
                       const Vec3& click, float radius, const PoseLimits& limits, FanCache& fans);

// What the last ended stroke did to topology.
struct StrokeTopologyStats {
  bool dyntopo = false;    // The stroke ran with dynamic topology and changed the mesh.
  int splits = 0;
  int collapses = 0;
  Index facesBefore = 0;
  Index facesAfter = 0;
  double consolidateMs = 0.0;
  bool lostUndo = false;   // Undo could not be recorded (a missed claim); history was cut.
  bool faulted = false;    // Topology stopped part way through the stroke.
  bool outOfRoom = false;  // Splits stopped because the stroke's headroom ran out.
  Index lockedVertices = 0;
  bool tooManyCopies = false;  // Dynamic topology was off: more than kMaxDyntopoCopies copies.
};

// Runs strokes on one scene object.
//
// Symmetry (StrokeOptions::symmetry()): every dab, and a Grab's capture, runs once per cluster of
// symmetric copies (see clusterCopies). Copies that nearly coincide merge into one dab on the
// plane or axis between them. Copies that overlap share each vertex by their falloff (ShareView),
// and every copy reads the mesh as it was before the dab. Vertices that start on a symmetry plane
// or axis (within kPinTolerance of the bounds diagonal) are snapped back onto it after each dab.
// Normals, bounds and GPU marks are updated once per dab for the union of all copies.
// For every dab it queries the BVH, snapshots the touched
// leaves for undo, runs the brush, recomputes normals around the moved vertices, refits the BVH
// and marks GPU ranges dirty. Brushes only move positions. Mask and face set brushes take a
// shorter path: the sculptor allocates the mask (or face sets), snapshots only those values and
// skips normals and refitting. Hidden faces (see Mesh::faceSets) are never changed: their
// vertices do not move unless a visible face also uses them.
//
// With dynamic topology each dab first refines or coarsens the mesh under the brush (see
// DyntopoSession), then runs the brush as usual. Leaves the topology pass changed are recorded
// whole; other leaves keep the usual position snapshots. endStroke() compacts the mesh again.
//
// On a mesh with sculpt layers, brushes still move the composite positions dab by dab, exactly as
// without layers. endStroke() then adds each moved vertex's change, divided by the target's
// strength, to the target (StrokeOptions::layerTarget: a layer's offsets or the base) and
// recomposes, so positions stay equal to the composite. Layer brushes (Brush::writesLayer()) edit
// the target's offsets directly instead. Undo records the target's values with the positions.
// Dynamic topology is off on such meshes.
class Sculptor {
 public:
  // Starts a stroke. A stroke the layer rules refuse (see layerTargetRefusal) does not start, and
  // active() stays false; callers show layerStrokeRefusal() first.
  void beginStroke(SceneObject& object, const Brush& brush, const StrokeOptions& options, std::string label);
  bool active() const { return object_ != nullptr; }
  SceneObject* object() const { return object_; }

  // Applies one dab at a surface point in the object's local space. `radius` and `strength`
  // already include pen pressure. `topology` only matters for dynamic topology strokes. Returns
  // false if nothing was under the dab.
  bool dab(const Vec3& center, float radius, float strength, const DabTopology& topology = {});

  // Grab stroke: captures the vertices inside the sphere (and its X mirror when symmetry is on)
  // and then moves them rigidly with the cursor, weighted by falloff, until the stroke ends.
  // Returns false (and starts nothing) if no vertex is inside the sphere or the layer rules refuse
  // the stroke.
  bool beginGrab(SceneObject& object, const StrokeOptions& options, const Vec3& center, float radius,
                 std::string label);
  // Moves the captured vertices to their start position plus `offset` (local space).
  void grab(const Vec3& offset);

  // Snake Hook stroke: picks a hook vertex near `center` (local space) for every cluster of
  // symmetric copies, the nearest that may move (on the plane or axis for merged copies when one
  // lies within the merge distance). Returns false, starting nothing, if none lies within
  // `radius` of the press, the hook is fully masked, or the layer rules refuse the stroke.
  bool beginSnake(SceneObject& object, const Brush& snake, const StrokeOptions& options, const Vec3& center,
                  float radius, float pinch, std::string label);
  // One tick: with dynamic topology refines around every hook (keeping the hook), then pulls the
  // primary hook `gain` of the way to `target` (local space) in sub-dabs of at most kSnakeStep *
  // radius, each centred on the hook vertex, with every copy pulling its mirrored share. Steers
  // the pull so a hook that copies or the mask hold back still reaches the target. Stops after
  // kMaxSnakeSubDabs sub-dabs or once budgetMs (negative: none) has passed after a sub-dab; the
  // next tick picks up the rest from the hook's real position.
  SnakeStep snakeTo(const Vec3& target, float radius, float gain, const DabTopology& topology = {},
                    double budgetMs = -1.0);
  // The running Snake Hook stroke's hook vertex for cluster k (cluster 0 is under the cursor),
  // or kInvalid.
  Index snakeHook(int k = 0) const {
    return snakeStroke_ && k >= 0 && k < snakeClusterCount_ ? snakeHooks_[static_cast<std::size_t>(k)] : kInvalid;
  }

  // Pose stroke: finds the parts to turn for a press at `click` (local space, see buildPose), or
  // takes `reuse`, a buildPoseFor result for the same object, options and press. Returns false,
  // starting nothing, if the build fails (see lastPoseFailure) or the layer rules refuse it.
  bool beginPose(SceneObject& object, const StrokeOptions& options, const PoseOptions& pose, const Vec3& click,
                 float radius, std::string label, PoseBuild* reuse = nullptr);
  // Turns the parts from their start positions so the press points from the joint toward
  // `target` (local space); symmetric parts turn by the mirrored rotation, merged ones by the part
  // of it they share. `viewNormal` picks the axis of a half turn. Normals follow (rigidly turned
  // where a face turns whole); bounds are refit when the stroke ends.
  void pose(const Vec3& target, const Vec3& viewNormal = {0.0f, 0.0f, 1.0f});
  // The running Pose stroke's parts, or nullptr.
  const PoseBuild* poseBuild() const { return poseStroke_ ? &pose_ : nullptr; }
  PoseFailure lastPoseFailure() const { return lastPoseFailure_; }
  // Angle of the last pose() on the part under the press, in degrees.
  float poseAngle() const { return poseAngle_; }

  // Ends the stroke. Returns the undo entry, or nothing if the stroke changed nothing. Mask and
  // face set strokes keep only the leaves whose values actually changed. A dynamic topology
  // stroke is compacted here and returns a DyntopoUndo.
  std::optional<StrokeUndo> endStroke();

  // Scratch memory for compacting dynamic topology strokes, shared with the undo stack. Without
  // one the sculptor makes its own on first use.
  void setLayoutWorkspace(std::shared_ptr<LayoutWorkspace> workspace) { workspace_ = std::move(workspace); }
  // Scratch for normal recomputation, may be shared with the layer operations. Without one the
  // sculptor makes its own on first use.
  void setNormalWorkspace(std::shared_ptr<NormalWorkspace> workspace) { normals_ = std::move(workspace); }
  // The running stroke's topology session, or nullptr.
  const DyntopoSession* dyntopo() const { return dyntopo_.get(); }
  // The face set the running (or last) face set stroke paints on the primary side.
  std::int32_t paintFaceSet() const { return paintSet_[0]; }
  // The running stroke's symmetry group.
  const SymmetryGroup& symmetry() const { return group_; }
  // True while the running stroke asked for dynamic topology but has too many symmetric copies.
  bool topologyOffForCopies() const { return tooManyCopies_; }

  const DabTiming& lastDab() const { return lastDab_; }
  int dabCount() const { return dabCount_; }
  const StrokeTopologyStats& lastStrokeTopology() const { return lastStroke_; }

 private:
  void start(SceneObject& object, const StrokeOptions& options, std::string label);
  void beginLayers();
  void commitLayerStroke();
  struct ApplyFlags {
    bool deferNormals = false;  // Leave normals for a later recomputeNormals of pendingNormals_.
  };
  int prepareCopies(const Dab& dab, const DabTopology& topology);
  bool applyCopies(int count, const ApplyFlags& flags);
  const ShareView* buildZone(int count, ShareView& view);
  void collectPins(const CopyCluster& cluster, const Vec3& center, float radius, std::span<const Index> leaves);
  OpMask pinOps(const Vec3& center, float radius) const;
  bool pinCandidate(const Vec3& p, OpMask ops) const;
  void resolveFaceSets(const Vec3& center, float radius);
  FaceSetFilter filter(int op) const;
  void applyTopology(const Dab& dab, const DabTopology& topology);
  double topologyBudget(int passes) const;
  Index findHook(const CopyCluster& cluster, float radius);
  void reglueHooks(float radius, std::span<const Vec3> before);
  void finishPose();
  Mat3 snakeSteering(float radius) const;
  void mergeClaims();
  std::optional<StrokeUndo> endDyntopoStroke(SceneObject& obj, DyntopoSession& session);
  void recomputeNormals(std::span<const Index> verts);
  void snapshot(Index leaf);
  bool computeArea(Dab& dab, std::span<const Index> leaves, const FaceSetFilter& filter) const;

  SceneObject* object_ = nullptr;
  const Brush* brush_ = nullptr;
  bool maskStroke_ = false;
  bool faceSetStroke_ = false;
  bool hidden_ = false;           // The mesh had hidden faces when the stroke began.
  bool faceSetsResolved_ = false;
  // Per symmetry op: the auto-mask set (0 none) and the set painted (0 nothing).
  std::array<std::int32_t, kMaxSymmetryCopies> onlySet_{};
  std::array<std::int32_t, kMaxSymmetryCopies> paintSet_{};
  SymmetryGroup group_;
  float pinEps_ = 0.0f;
  bool tooManyCopies_ = false;
  // One symmetric copy of the current dab.
  struct CopyDab {
    Dab dab;
    CopyCluster cluster;
    DabTopology topology;
    FaceSetFilter filter;
    std::int32_t paintSet = 0;
    std::vector<Index> leaves;
    bool live = false;
  };
  CopyClusters clusters_;
  std::vector<CopyDab> copies_;
  // Vertices within reach of two or more copies: per-vertex slots (kept for the stroke) and
  // per-dab sums, see ShareView.
  struct ShareZone {
    std::vector<std::uint32_t> stamp, slot;
    std::uint32_t current = 0;
    std::vector<float> denom;
    std::vector<std::uint8_t> count;
    std::vector<Vec3> pending;
    std::vector<float> pendingScalar;
    std::vector<Index> verts;
    std::vector<std::vector<std::pair<Index, float>>> gathered;  // Per leaf: vertex, raw falloff.
  };
  ShareZone zone_;
  struct PinnedVertex {
    Index v;
    Vec3 start;
    SeamPin pin;
  };
  std::vector<PinnedVertex> pins_;
  std::vector<std::uint32_t> faceClaim_;
  std::uint32_t claimStamp_ = 0;
  std::vector<Index> pendingNormals_;
  // Snake Hook state: clusters fixed at the press and each one's hook vertex (kInvalid: none).
  bool snakeStroke_ = false;
  float snakePinch_ = 0.0f;
  CopyClusters snakeClusters_;
  int snakeClusterCount_ = 0;
  std::array<Index, kMaxSymmetryCopies> snakeHooks_{};
  // Pose state.
  bool poseStroke_ = false;
  PoseBuild pose_;
  Vec3 poseClick_{0.0f};
  float poseAngle_ = 0.0f;
  bool poseDeferNormals_ = false;
  PoseFailure lastPoseFailure_ = PoseFailure::None;
  StrokeOptions options_;
  // Sculpt layers: the target array of a position stroke on a layered mesh (else nullptr), and
  // the same as layer brushes see it.
  std::vector<Vec3>* layerArray_ = nullptr;
  LayerTarget layerTarget_;
  std::vector<std::vector<Index>> layerChanged_;  // Per snapshot: vertices the commit recomposed.
  std::vector<std::vector<Index>> layerRing_, layerLeaves_;  // Per snapshot: their face rings and leaves.
  FanCache fans_;  // Non-manifold fans of the object's mesh, for the commit's face rings.
  SculptUndo undo_;
  std::unordered_map<Index, std::size_t> snapshotIndex_;  // leaf -> index in undo_.before
  std::vector<std::uint32_t> vertexStamp_;
  std::uint32_t stamp_ = 0;
  std::vector<Index> leaves_, normalVerts_, dirtyLeaves_;
  // Grab state: captured vertices, their start positions and their weight per symmetric cluster,
  // plus the fixed set of vertices whose normals and leaves a grab touches.
  struct GrabVertex {
    Index v;
    Vec3 start;
    std::uint32_t firstEntry;
    std::uint32_t entryCount;
  };
  struct GrabEntry {
    std::uint32_t cluster;
    float weight;
  };
  std::vector<GrabVertex> grabVerts_;
  std::vector<GrabEntry> grabEntries_;
  std::vector<std::pair<std::uint32_t, SeamPin>> grabPins_;  // Index in grabVerts_, pin.
  CopyClusters grabClusters_;
  int grabClusterCount_ = 0;
  std::vector<Index> grabNormalVerts_, grabRefitLeaves_, grabDirtyLeaves_;
  std::vector<Index> strokeRefit_;  // Every leaf refit during the stroke, for undo.
  std::vector<Index> kept_;
  DabTiming lastDab_;
  int dabCount_ = 0;
  std::unique_ptr<DyntopoSession> dyntopo_;
  std::size_t mergedClaims_ = 0;  // Claims already merged with position snapshots.
  std::shared_ptr<LayoutWorkspace> workspace_;
  std::shared_ptr<NormalWorkspace> normals_;
  StrokeTopologyStats lastStroke_;
};

}  // namespace plegl
