import argparse
import csv
import sys
import os
import time
import logging
import traceback
import faulthandler
import signal
from datetime import datetime, timezone
import numpy as np
import torch
import matplotlib.pyplot as plt

os.environ["PYTHONUNBUFFERED"] = "1"

faulthandler.enable(all_threads=True)
 

 
#   PERFORMANCE TUNING 
DEBUG_LOOP = False  # Set to True for debugging, False for final silent runs
HEARTBEAT_EVERY = 200
RENDER_EVERY = 1
UI_UPDATE_EVERY = 1
POSE_LOG_EVERY = 100

PROFILE_LOOP = True
PROFILE_EVERY = 300

# Determine log level based on the master switch
log_level = logging.DEBUG if DEBUG_LOOP else logging.WARNING

LOG_PATH = os.path.abspath("isaaclab_debug.log")
logging.basicConfig(
    level=log_level,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("dbg")

def _make_loop_profiler():
    return {
        "art_update": 0.0,
        "adapter_step": 0.0,
        "goal_update": 0.0,
        "ui_update": 0.0,
        "pose_log": 0.0,
        "sim_step": 0.0,
        "total_loop": 0.0,
        "count": 0,
    }


def _print_loop_profiler_stats(profiler, frame, sim_time):

    if not DEBUG_LOOP:
            return

    n = max(profiler["count"], 1)

    msg = (
        f"[PROFILE] frame={frame} sim_time={sim_time:.2f}s | "
        f"avg_loop={profiler['total_loop']/n:.5f}s | "
        f"art_update={profiler['art_update']/n:.5f}s | "
        f"adapter_step={profiler['adapter_step']/n:.5f}s | "
        f"goal_update={profiler['goal_update']/n:.5f}s | "
        f"ui_update={profiler['ui_update']/n:.5f}s | "
        f"pose_log={profiler['pose_log']/n:.5f}s | "
        f"sim_step={profiler['sim_step']/n:.5f}s"
    )
    log.debug(msg)
    print(msg, flush=True)


def _reset_loop_profiler(profiler):
    for k in profiler:
        profiler[k] = 0.0
    profiler["count"] = 0

COLLISION_LOG_PATH = os.path.join("logs", "collision_summary.csv")
COLLISION_LOG_FIELDS = [
    "timestamp", "algorithm", "num_drones", "known_obstacles",
    "unknown_obstacles", "unknown_obstacles_detected_pct", "lidar_enabled",
    "collisions", "collisions_per_robot", "success",
]


def _unknown_obs_detected_pct(controller_list, unknown_obs, tol=0.15) -> float:
    """% of ground-truth unknown obstacles detected by at least one robot.

    Detection isn't noisy in this sim (a detected obstacle's reported
    position/radius is the ground-truth one), so a small position tolerance
    is enough to match a robot's detected_unknown_obs_memory entries back to
    the config's unknown_obstacles list.
    """
    if len(unknown_obs) == 0:
        return 0.0

    detected_xy = []
    for ctrl in controller_list:
        mem = getattr(ctrl.robot, "detected_unknown_obs_memory", None)
        if mem is not None and len(mem) > 0:
            detected_xy.append(np.asarray(mem)[:, :2])
    if not detected_xy:
        return 0.0
    detected_xy = np.concatenate(detected_xy, axis=0)

    truth_xy = np.asarray(unknown_obs)[:, :2]
    hits = 0
    for gx, gy in truth_xy:
        dists = np.linalg.norm(detected_xy - np.array([gx, gy]), axis=1)
        if np.any(dists <= tol):
            hits += 1
    return 100.0 * hits / len(truth_xy)


def _log_collision_summary(**fields) -> None:
    """Append one minimal per-run summary row. Independent of ExperimentLogger
    (which is disabled by default) so this always runs, headless or not."""
    try:
        os.makedirs(os.path.dirname(COLLISION_LOG_PATH), exist_ok=True)
        file_exists = os.path.isfile(COLLISION_LOG_PATH)
        with open(COLLISION_LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLLISION_LOG_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(fields)
    except Exception:
        log.exception("Failed writing collision summary log")


def checkpoint(name: str) -> None:
    log.debug(f"CHECKPOINT: {name}")


def _dump_stacks(signum, frame):
    print(f"\n\n=== RECEIVED SIGNAL {signum} -> dumping stacks ===", flush=True)
    faulthandler.dump_traceback(all_threads=True)


try:
    signal.signal(signal.SIGUSR1, _dump_stacks)
except Exception:
    pass


from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IsaacLab YAML Experiment")
parser.add_argument("--config", "-c", required=True, help="Path to YAML config")
parser.add_argument("--explore-ui", action="store_true", help="Show Matplotlib UI")
parser.add_argument("--use-rtx-lidar", action="store_true", help="Enable Lidar")
parser.add_argument(
    "--low-memory",
    action="store_true",
    help="Reduce renderer memory use for machines with limited RAM",
)
parser.add_argument(
    "--record-dir",
    default=None,
    help=(
        "Capture a numbered PNG per rendered frame into this directory "
        "(created if missing), for stitching into a video afterward with "
        "ffmpeg. Only available with --viz kit (viewport capture isn't "
        "loaded in the headless experience); silently disabled otherwise."
    ),
)
parser.add_argument(
    "--record-every",
    type=int,
    default=4,
    help="Only capture every Nth rendered frame with --record-dir (default 4).",
)
parser.add_argument(
    "--exit-after-collision-secs",
    type=float,
    default=None,
    help=(
        "Once the first collision happens, keep running/recording for this "
        "many more sim-seconds, then stop early instead of running to "
        "max_ticks/exploration_complete. Useful for a short demo clip: "
        "capture the fall-to-the-ground, then quit."
    ),
)
parser.add_argument(
    "--no-trails",
    action="store_true",
    help=(
        "Disable the persistent trajectory-trail lines and frontier markers "
        "(both purely cosmetic overlays). For a clean asset-showcase "
        "recording; leave enabled for experiment demos."
    ),
)
parser.add_argument(
    "--hide-fov",
    action="store_true",
    help=(
        "Hide the per-robot FOV wedge mesh entirely (it's otherwise always "
        "attached). For a clean asset-showcase recording; leave enabled for "
        "experiment demos where seeing sensor coverage matters."
    ),
)
parser.add_argument(
    "--no-warehouse",
    action="store_true",
    help=(
        "Skip loading the warehouse USD reference -- just the default "
        "ground plane grid + lights, no shelving/walls. For asset-showcase "
        "recordings; experiment demos rely on the warehouse for scale/"
        "camera-framing context."
    ),
)
parser.add_argument(
    "--force-track-robot",
    type=int,
    default=None,
    help=(
        "Keep the camera locked in track mode on this robot index for the "
        "entire run, instead of the usual global/per-robot cycling. For "
        "asset-showcase recordings, not experiment demos."
    ),
)
parser.add_argument(
    "--track-offset",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help=(
        "Camera eye offset (world-frame, relative to the tracked robot) used "
        "with --force-track-robot. Default (-3,-3,6) is the same distant "
        "offset the normal camera director uses; pass something closer "
        "(e.g. -1.2 -1.2 0.8) for a close-up showcase angle."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.low_memory and args_cli.use_rtx_lidar:
    parser.error("--low-memory cannot be combined with --use-rtx-lidar")

checkpoint("Args parsed")

checkpoint("Creating AppLauncher")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
checkpoint("AppLauncher created (simulation_app exists)")
#  INSERT SETTINGS HERE
import carb
settings = carb.settings.get_settings()
# 0. Isaac Lab's default experience file enables FSD's
# rtx.hydra.readTransformsFromFabricInRenderDelegate, which Kit itself warns
# is known to break transform streaming for dynamic objects when combined
# with streamed geometry (our large external warehouse.usd) -- exactly our
# setup, since the robots are teleported every physics frame. Disabling it
# here may be too late if Kit already cached the FSD path at extension load;
# if the viewport is still black, pass it at Kit startup instead via
# --kit_args "--/rtx/hydra/readTransformsFromFabricInRenderDelegate=false".
settings.set("/rtx/hydra/readTransformsFromFabricInRenderDelegate", False)
# 1. Disable Motion Blur completely
settings.set("/rtx/post/motionblur/enabled", False)
# 2. Disable TAA/DLSS (Temporal Anti-Aliasing) to stop smearing
#settings.set("/rtx/post/aa/op", 0) 
# 3. Increase rendering sharpness
settings.set("/rtx/post/sharpen/enabled", True)
settings.set("/rtx/post/sharpen/strength", 0.5)
# 4. Use a reduced render scale when explicitly requested. Full resolution is
# retained by default for existing runs.
if args_cli.low_memory:
    settings.set("/rtx/dynamicResampling/enabled", True)
    settings.set("/rtx/resampling/factor", 0.65)
    settings.set("/app/renderer/resolution/width", 1280)
    settings.set("/app/renderer/resolution/height", 720)
else:
    settings.set("/rtx/dynamicResampling/enabled", False)
    settings.set("/rtx/resampling/factor", 1.0)
 
# settings.set("/app/renderer/resolution/width", 1920)
# settings.set("/app/renderer/resolution/height", 1080)
# ----------------------------
render_every = 2 if args_cli.low_memory else RENDER_EVERY
import omni.usd
from pxr import UsdGeom, Gf

from isaaclab.sim import SimulationContext, SimulationCfg, GroundPlaneCfg, DomeLightCfg
from isaaclab.assets import Articulation
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg


sys.path.append(os.getcwd())
import scripts.utils as utils
import scripts.sensor as sensor_utils
import scripts.scene as scene_utils
import scripts.adapter as adapter_utils
from exploration import ExplorationManager
from safe_control.utils import env
from scripts.sensor import RobotSensorSuite, RobotSensorSuiteCfg
from tools.record_video import start_frames, stop_frames


# def print_children(path):
#     stage = omni.usd.get_context().get_stage()
#     prim = stage.GetPrimAtPath(path)
#     print(f"[children] {path} valid={prim.IsValid()}")
#     if not prim.IsValid():
#         return
#     for child in prim.GetChildren():
#         print(" ", child.GetPath())


 
# def print_transform(path):
#     stage = omni.usd.get_context().get_stage()
#     prim = stage.GetPrimAtPath(path)

#     xf = UsdGeom.Xformable(prim)
#     ops = xf.GetOrderedXformOps()

#     print("Transforms for", path)

#     for op in ops:
#         print(op.GetOpName(), op.Get())



        
def _safe_update(n: int = 1):
    for _ in range(n):
        simulation_app.update()


# def _run_checked(__label: str, fn, *args, **kwargs):
#     checkpoint(f"START: {__label}")
#     try:
#         out = fn(*args, **kwargs)
#         checkpoint(f"OK: {__label}")
#         return out
#     except Exception:
#         log.exception(f"FAILED: {__label}")
#         raise

def _run_checked(__label: str, fn, *args, **kwargs):
    return fn(*args, **kwargs)



def main():
    checkpoint("main() entered")

    cfg = None
    exploration = None
    sim = None
    logger = None
    adapter = None
    success = False
    sim_time = 0.0
    frame = 0
    start_wall_time = time.time()
    loop_profiler = _make_loop_profiler()
 

    # Load the structured config object instead of raw dicts
    cfg = _run_checked("load_config", utils.load_config, args_cli.config)


    # Extract sub-configs for easier access
    exp_cfg = cfg
    env_cfg = cfg.environment
    ctrl_cfg = cfg.controller

    SHOW_EXPLORE_UI = args_cli.explore_ui  
    checkpoint(f"SHOW_EXPLORE_UI={SHOW_EXPLORE_UI} use_rtx_lidar={args_cli.use_rtx_lidar}")

    sim_cfg = _run_checked(
        "SimulationCfg",
        SimulationCfg,
        device=args_cli.device,
        physics=PhysxCfg(enable_external_forces_every_iteration=True),
        # Without this, SimulationContext reuses whatever stage it finds in
        # Kit's stage cache / thread-local context (e.g. restored from the
        # persistent isaac_omniverse/isaac_cache_kit Docker volumes across
        # runs), which is what produced the "A prim already exists at prim
        # path: '/World/Drone_0'" warnings -- leftover prims from a prior
        # run colliding with this run's spawn. Forcing a fresh in-memory
        # stage every launch avoids that; Isaac Lab still attaches it to
        # Kit's UsdContext for the viewport regardless of this flag.
        create_stage_in_memory=True,
    )
    sim = _run_checked("SimulationContext", SimulationContext, sim_cfg)

    checkpoint("Getting USD stage")
    stage = _run_checked("get_stage", omni.usd.get_context().get_stage)

    warehouse_url = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/warehouse.usd"

    # initialize camera manager
    checkpoint("Initializing Camera Manager")
    camera_manager = utils.CameraManager(sim, env_cfg)

    # Optional frame recorder (--record-dir): dumps a numbered PNG per
    # captured frame for stitching into a video afterward with ffmpeg. Uses
    # the same viewport-capture API as the one-off debug screenshots used
    # during development this session -- only available with --viz kit (the
    # headless experience doesn't load the viewport extension set), so this
    # degrades to a no-op instead of crashing headless runs.
    record_get_viewport = None
    record_capture_fn = None
    record_saved_count = 0
    if args_cli.record_dir:
        try:
            from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
            os.makedirs(args_cli.record_dir, exist_ok=True)
            record_get_viewport = get_active_viewport
            record_capture_fn = capture_viewport_to_file
            checkpoint(f"Recording enabled: {args_cli.record_dir} (every {args_cli.record_every} rendered frames)")
        except ModuleNotFoundError:
            log.warning("--record-dir requested but viewport capture isn't available (need --viz kit); ignoring.")

    # Warehouse wall/floor scaling.
    #
    # The asset's actual interior footprint is 24m x 36m -- measured directly
    # from the wall (SM_WallA_*) and floor (SM_floor*/GroundPlane) prim
    # bounding boxes after resolving their referenced sub-assets, NOT the
    # "20m x 20m square including apron" this used to assume, which produced
    # a warehouse scaled far larger than the environment grid and distorted
    # non-uniformly (different X/Y factors against a square reference that
    # was itself wrong).
    #
    # Scaling uniformly (same factor on X and Y) is required to avoid
    # stretching/squishing the walls, pillars, and racking out of proportion.
    # Since the warehouse's native aspect ratio (24:36, i.e. 2:3) generally
    # won't match the requested grid's aspect ratio, a uniform scale can only
    # guarantee the grid is fully enclosed on both axes, not that the walls
    # hug the grid tightly on every side -- whichever axis has slack will
    # show extra warehouse floor beyond the red boundary lines. Trying both
    # the native orientation and a 90deg-rotated one and keeping whichever
    # leaves less total slack picks the better-aligned fit automatically,
    # for any width/height combination (not just this particular grid).
    NATIVE_WAREHOUSE_WIDTH = 24.0
    NATIVE_WAREHOUSE_HEIGHT = 36.0

    def _warehouse_fit(native_w, native_h):
        scale = max(env_cfg.width / native_w, env_cfg.height / native_h)
        slack = (native_w * scale - env_cfg.width) + (native_h * scale - env_cfg.height)
        return scale, slack

    scale_native, slack_native = _warehouse_fit(NATIVE_WAREHOUSE_WIDTH, NATIVE_WAREHOUSE_HEIGHT)
    scale_rotated, slack_rotated = _warehouse_fit(NATIVE_WAREHOUSE_HEIGHT, NATIVE_WAREHOUSE_WIDTH)

    if slack_rotated < slack_native:
        warehouse_scale = scale_rotated
        warehouse_rotate_z = 90.0
        used_native_w, used_native_h = NATIVE_WAREHOUSE_HEIGHT, NATIVE_WAREHOUSE_WIDTH
    else:
        warehouse_scale = scale_native
        warehouse_rotate_z = 0.0
        used_native_w, used_native_h = NATIVE_WAREHOUSE_WIDTH, NATIVE_WAREHOUSE_HEIGHT

    # Manual cosmetic size bump on top of the automatic fit above.
    WAREHOUSE_MANUAL_SCALE_BOOST = 1.08
    warehouse_scale *= WAREHOUSE_MANUAL_SCALE_BOOST

    # The top-right corner gets anchored below at a fixed (env_width +
    # WAREHOUSE_MARGIN_X, env_height + WAREHOUSE_MARGIN_Y) offset regardless
    # of scale -- but the opposite (left/bottom) edges only get whatever
    # slack happens to be left over from warehouse_scale, which isn't
    # guaranteed to be enough: at scale 1.08 the left edge landed almost
    # exactly on the wall, letting the red boundary line clip through it.
    # Solve for the scale that guarantees the left/bottom edges get AT LEAST
    # as much clearance as MIN_LEFT_MARGIN/MIN_BOTTOM_MARGIN, then take
    # whichever of (cosmetic boost, this requirement) is larger -- this can
    # only grow the warehouse from here, never shrink the already-good
    # right/top margins or the bottom clearance that was previously tuned.
    WAREHOUSE_MARGIN_X = 5.0  # meters, world +X (screen-right), fixed anchor
    WAREHOUSE_MARGIN_Y = 1.5  # meters, world +Y (screen-up), fixed anchor
    MIN_LEFT_MARGIN = 7.0     # meters, world -X: generous, past the anchored side
    MIN_BOTTOM_MARGIN = 2.0   # meters, world -Y: generous, past the anchored side

    required_half_x = (env_cfg.width + WAREHOUSE_MARGIN_X + MIN_LEFT_MARGIN) / 2.0
    required_half_y = (env_cfg.height + WAREHOUSE_MARGIN_Y + MIN_BOTTOM_MARGIN) / 2.0
    required_scale = max(
        required_half_x / (used_native_w / 2.0),
        required_half_y / (used_native_h / 2.0),
    )
    warehouse_scale = max(warehouse_scale, required_scale)

    scene_cfg = scene_utils.WarehouseSceneCfg(
        env_translate=(0.0, 0.0, 0.0),
        env_scale=(warehouse_scale, warehouse_scale, 1.0),
        env_rotate_deg=(0.0, 0.0, warehouse_rotate_z),
    )

    scene = _run_checked("WarehouseScene construct", scene_utils.WarehouseScene, scene_cfg, stage=stage)
    _run_checked("scene.setup", scene.setup, warehouse_url, skip_env=args_cli.no_warehouse)
    _safe_update(15)

    # Anchor the warehouse's top-right corner exactly to the top-right corner
    # of the red env-handler boundary (env_cfg.width, env_cfg.height), rather
    # than centering it -- any leftover slack (see _warehouse_fit above) then
    # shows up as extra floor past the bottom/left edges instead of being
    # split evenly on all sides. The warehouse's own local origin sits at its
    # bounding-box center (verified directly from the wall/floor geometry:
    # x in [-12,12], y in [-18,18], native/unrotated), so after rotating
    # warehouse_rotate_z degrees and scaling by warehouse_scale, its
    # half-extents swap when rotated 90deg:
    if warehouse_rotate_z == 90.0:
        half_x = (NATIVE_WAREHOUSE_HEIGHT / 2.0) * warehouse_scale
        half_y = (NATIVE_WAREHOUSE_WIDTH / 2.0) * warehouse_scale
    else:
        half_x = (NATIVE_WAREHOUSE_WIDTH / 2.0) * warehouse_scale
        half_y = (NATIVE_WAREHOUSE_HEIGHT / 2.0) * warehouse_scale

    # The goal isn't to touch the red border exactly -- it's for the red
    # boundary (what robots actually explore within) to sit fully inside the
    # warehouse's walls, with visible wall clearance on every side. So the
    # anchor point is offset past the corner by a margin rather than landing
    # exactly on it. Empirically tuned by capturing the actual rendered
    # birdview and comparing pixel positions: the mesh's own bounding-box
    # geometry lands the corner exactly on the border (verified directly via
    # ComputeWorldBound), but the *rendered* wall silhouette sits noticeably
    # inward of that in the perspective birdview -- the walls are ~9m tall,
    # and at this camera altitude a tall wall's visible top edge shifts
    # toward image-center relative to its true ground-plane (z=0) position,
    # while the red boundary lines are flat at z~0. X needs much more margin
    # than Y here because the warehouse is rotated 90deg, putting its long
    # (native 36m) axis on X, farther from the camera's optical axis.
    # (WAREHOUSE_MARGIN_X/Y are computed above, alongside MIN_LEFT_MARGIN/
    # MIN_BOTTOM_MARGIN, since the scale solve needs them too.)
    #
    # Pure rigid shift on top of the anchor above -- moves the whole
    # warehouse box (env_handler/red boundary is untouched), trading a bit
    # of the left/bottom surplus margin for more right/top clearance.
    #
    # WAREHOUSE_SHIFT_Y used to be 1.5, which (combined with
    # MIN_BOTTOM_MARGIN's solved 2.0m guarantee above) skewed the split to
    # top=3.0m/bottom=0.5m instead of the intended "both sides comfortable"
    # -- a robot exploring anywhere near the bottom edge of the play area
    # could end up flush against the warehouse's actual (9m-tall) wall,
    # only 0.5m past the boundary. Shifting down by much less instead
    # rebalances top/bottom to ~1.75m each (both sides drawing off the same
    # ~3.5m of combined Y slack this map size has, split evenly rather than
    # handing nearly all of it to the top).
    WAREHOUSE_SHIFT_X = 2.0   # meters, world +X (screen-right)
    WAREHOUSE_SHIFT_Y = 0.25  # meters, world +Y (screen-up)

    target_translate = (
        env_cfg.width - half_x + WAREHOUSE_MARGIN_X + WAREHOUSE_SHIFT_X,
        env_cfg.height - half_y + WAREHOUSE_MARGIN_Y + WAREHOUSE_SHIFT_Y,
        0.0,
    )

    env_prim = stage.GetPrimAtPath("/World/Env/Main")
    xform = UsdGeom.Xformable(env_prim)

    # Apply translation to the container prim
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*target_translate))

    log.info(
        f"Warehouse scaled uniformly ({warehouse_scale:.3f}x, native {NATIVE_WAREHOUSE_WIDTH}x{NATIVE_WAREHOUSE_HEIGHT}m, "
        f"rotate_z={warehouse_rotate_z}deg) to enclose {env_cfg.width}x{env_cfg.height} grid."
    )
    _safe_update(10)

    try:
        log.info(f"Sun valid: {stage.GetPrimAtPath('/World/Lights/Sun').IsValid()}")
        log.info(f"Sky valid: {stage.GetPrimAtPath('/World/Lights/Sky').IsValid()}")
    except Exception:
        log.exception("Failed querying light prim validity")

    known_obs = _run_checked("build_known_obs", utils.build_known_obs, env_cfg)
    unknown_obs = np.asarray(env_cfg.unknown_obstacles, dtype=float)
##################################################################
##################################################################
##################################################################

    robot_specs = _run_checked("parse_robot_specs", utils.parse_robot_specs, cfg)
    X0s = [np.array(inst.x0) for inst in cfg.robots.instances]
    controller_type = {"pos": ctrl_cfg.pos, "att": ctrl_cfg.att}

    robots_sim = []
    sensors = []

    robot_radius = cfg.robots.common.radius
    checkpoint(f"Global robot_radius set to: {robot_radius}")
    # 1. Define the robot configuration outside of the loop
    CUSTOM_DRONE_CFG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/seamlis/isaaclab_assets/drone_articulation.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=2,
            ),
        ),
        prim_path="{PXR_PREFIX}",
        actuators={
            "motors": ImplicitActuatorCfg(
                joint_names_expr=["joint_prop_.*"],
                stiffness=0.0,
                damping=0.02,
                effort_limit_sim=0.02,
                velocity_limit_sim=500.0,
            )
        },
    )
 
    CHOSEN_ORIENTATION = [1.0, 0.0, 0.0, 0.0]
    PROP_SPEED = 100.0

    robot_specs = _run_checked("parse_robot_specs", utils.parse_robot_specs, cfg)
    X0s = [np.array(inst.x0) for inst in cfg.robots.instances]
    controller_type = {"pos": ctrl_cfg.pos, "att": ctrl_cfg.att}

    robots_sim = []
    sensors = []
    robot_radius = cfg.robots.common.radius
    checkpoint(f"Global robot_radius set to: {robot_radius}")

    checkpoint("Spawning robots begin")
    for i, spec in enumerate(robot_specs):
        prim_path = f"/World/Drone_{i}"
        spawn_pos = [spec["x0"][0], spec["x0"][1], cfg.robots.common.flight_z]
        checkpoint(f"Robot {i} start prim={prim_path} spawn_pos={spawn_pos}")

        try:
            # Articulation.__init__ (via AssetBase.__init__) already calls
            # cfg.spawn.func(...) itself whenever cfg.spawn is set -- spawning
            # here too meant every one of these USD-heavy, many-mesh drones
            # (dozens of individual collision meshes, each needing a convex-hull
            # fallback) had its spawn/collision-authoring work done twice, and
            # the second call always hit the "prim already exists" branch and
            # silently no-op'd. Setting init_state instead lets Articulation do
            # the one real spawn, at the right position/orientation.
            bot_cfg = CUSTOM_DRONE_CFG.replace(
                prim_path=prim_path,
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=tuple(spawn_pos),
                    rot=tuple(CHOSEN_ORIENTATION),
                ),
            )
            checkpoint(f"{prim_path}: constructing Articulation")
            art = Articulation(bot_cfg)
            checkpoint(f"{prim_path}: Articulation constructed")
     

            #  SENSOR SETUP 
            # Put corrective rotation on SensorFrame, not directly on FOV mesh.
            sensor_cfg = RobotSensorSuiteCfg(
                fov_deg=spec["fov_angle"],
                rng=spec["cam_range"],
                use_lidar=args_cli.use_rtx_lidar,
                color=sensor_utils.hsv_to_rgb_deg(i * 360.0 / len(robot_specs)),
                sensor_frame_translation=(0.0, 0.0, 0.0),
                sensor_frame_rotate_xyz_deg=(0.0, 0.0, 0.0),  # try (0,90,0) if needed
                hide_fov=args_cli.hide_fov,
            )

            checkpoint(f"{prim_path}: RobotSensorSuite attach begin")
            suite_obj = RobotSensorSuite(robot_root=prim_path, cfg=sensor_cfg)
            suite = suite_obj.attach()
            checkpoint(f"{prim_path}: RobotSensorSuite attach done")

            # Debug where the suite actually attached
            print(f"[DEBUG] {prim_path}")
            print(f"  parent_path       = {suite.parent_path}")
            print(f"  sensor_frame_path = {suite.sensor_frame_path}")
            print(f"  fov_path          = {suite.fov_path}")

            if suite.fov_path is not None:
                _run_checked(
                    f"{prim_path}: bind_preview_material",
                    sensor_utils.bind_preview_material,
                    suite.fov_path,
                    color=sensor_cfg.color,
                    opacity=1.0,
                    emissive_strength=5.0,
                )

            try:
                if suite.fov_path is not None:
                    log.info(f"{prim_path}: FOV prim valid: {stage.GetPrimAtPath(suite.fov_path).IsValid()}")
                if suite.sensor_frame_path is not None:
                    log.info(
                        f"{prim_path}: SensorFrame prim valid: "
                        f"{stage.GetPrimAtPath(suite.sensor_frame_path).IsValid()}"
                    )
            except Exception:
                log.exception(f"{prim_path}: failed checking sensor prim validity")

            sensors.append(suite)



            robots_sim.append(
                {
                    "name": f"Drone_{i}",
                    "path": prim_path,
                    "art": art,
                    "directions": None,
                    "lidar": getattr(suite, "lidar", None),
                    "sensors": suite,
                    "fov_color": sensor_cfg.color,
                }
            )

            checkpoint(f"Robot {i} complete prim={prim_path}")

        except Exception:
            log.exception(f"FAILED while setting up robot {i} prim={prim_path}")
            raise

    checkpoint("Spawning robots done")
  ##################################################################
  ##################################################################
  ####################################################################################################################################
  ##################################################################
  ##################################################################

    # print_transform("/World/Drone_0")
    # print_transform("/World/Drone_0/SensorFrame")
    # print_transform("/World/CF_0/body/SensorFrame")
    # print_transform("/World/CF_1/body/SensorFrame")
    # print_transform("/World/CF_2/body/SensorFrame")


    known_obs_array = utils.build_known_obs(cfg.environment)
    

    checkpoint("Initializing Environment Handler")
    env_handler = env.Env(
        width=cfg.environment.width,
        height=cfg.environment.height,
        known_obs=known_obs_array,
        resolution=cfg.environment.resolution,
    )

    # Red exploration-boundary lines -- useful for debugging env_handler
    # alignment against the warehouse mesh, not meant for normal runs.
    # Uncomment to bring them back.
    # checkpoint("Spawning Boundary Visualizers")
    # scene_utils.spawn_boundary_lines(
    #     stage=stage,
    #     width=cfg.environment.width,
    #     height=cfg.environment.height,
    #     color=(1, 0, 0),
    # )

    checkpoint("Constructing ExplorationManager")
    exploration = _run_checked(
        "ExplorationManager",
        ExplorationManager,
        X0s,
        robot_specs,
        controller_type,
        exploration_algorithm=exp_cfg.exploration_algorithm,
        dt=exp_cfg.dt,
        show_animation=SHOW_EXPLORE_UI,
        env_handler=env_handler,
        unknown_obs=unknown_obs,
    )

    if SHOW_EXPLORE_UI:
        if hasattr(exploration, "init_visualization"):
            exploration.init_visualization()
        if hasattr(exploration, "update_visualization"):
            exploration.update_visualization()

    _run_checked(
        "scene.sync_all_obstacles",
        scene.sync_all_obstacles,
        env_handler,
        unknown_obs_list=exploration.unknown_obs,
    )
    #_run_checked("scene.spawn_grid_markers", scene.spawn_grid_markers, env_handler)

    checkpoint("Constructing adapter")
    adapter = _run_checked(
        "ExploreAdapter",
        adapter_utils.ExploreAdapter,
        exploration,
        robots_sim,
        fixed_z=cfg.robots.common.flight_z,
        draw_trails=not args_cli.no_trails,
        draw_frontiers=not args_cli.no_trails,
    )

    checkpoint("Initializing goals/frontiers")
    exploration.frontiers = _run_checked("get_frontiers", exploration.get_frontiers)
    _run_checked("update_all_goals", exploration.update_all_goals)
    adapter.draw_frontiers(exploration.frontiers)

    # logger = _run_checked(
    #     "ExperimentLogger",
    #     utils.ExperimentLogger,
    #     experiment_name=cfg.experiment_name,
    #     controller_pos=cfg.controller.pos,
    #     controller_att=cfg.controller.att,
    # )

    checkpoint("sim.reset() begin")
    _run_checked("sim.reset", sim.reset)
    checkpoint("sim.reset() done")

    dt = sim.get_physics_dt()

    checkpoint("Initializing propeller directions post-reset")
    for r in robots_sim:
        art = r["art"]
        art.update(dt)

        joint_names = art.data.joint_names
        checkpoint(f"{r['name']}: detected joints = {joint_names}")

        directions_list = []
        for n in joint_names:
            if "joint_prop_" in n:
                if "front_left" in n or "back_right" in n:
                    directions_list.append(1.0)
                else:
                    directions_list.append(-1.0)
            else:
                directions_list.append(0.0)

        r["directions"] = torch.tensor(
            directions_list,
            device=sim.device,
            dtype=torch.float32,
        ).unsqueeze(0)

    checkpoint("post-reset updates begin")
    _safe_update(20)
    checkpoint("post-reset updates done")

    checkpoint("Frame recording disabled")
    time_since_last_plan = 0.0

    checkpoint("Entering main sim loop")
    print(
        f"DEBUG: Manager has {len(exploration.env_handler.obs_circle)} circles and "
        f"{len(exploration.env_handler.obs_superellipsoid)} walls."
    )

    # IMPORTANT: allocate these ONCE, outside the loop
    pose_history = {i: [] for i in range(len(robots_sim))}
    time_history = []
    frame_history = []
    collision_exit_deadline_frame = None  # set once the first collision happens
    try:
        while simulation_app.is_running():
            loop_start = time.perf_counter()

            dt = sim.get_physics_dt()
            sim_time += dt
            time_since_last_plan += dt

            if frame % HEARTBEAT_EVERY == 0:
                log.info(f"HEARTBEAT frame={frame} sim_time={sim_time:.3f} dt={dt:.6f}")

            # --------------------------------------------------
            # 1. Update articulation buffers
            # --------------------------------------------------
            t0 = time.perf_counter()
            try:
                for r in robots_sim:
                    r["art"].update(dt)
            except Exception:
                log.exception(f"FAILED during Articulation.update at frame={frame}")
                raise
            loop_profiler["art_update"] += time.perf_counter() - t0

            # --------------------------------------------------
            # 1b. Re-assert pose every physics frame, so contact/reaction
            # forces from the environment and the spinning propellers never
            # get a window to nudge the body before the next correction.
            # Sensor USD transforms are visual-only, so they're skipped when
            # this frame won't be rendered (see step 6 below for do_render).
            # --------------------------------------------------
            do_render = (frame % RENDER_EVERY == 0)
            t0 = time.perf_counter()
            try:
                adapter.pin_step(dt, render_this_frame=do_render)
            except Exception:
                log.exception(f"FAILED during adapter.pin_step at frame={frame}")
                raise
            loop_profiler["adapter_step"] += time.perf_counter() - t0

            # --------------------------------------------------
            # 2. Planner update at planner rate
            # --------------------------------------------------
            if time_since_last_plan >= exp_cfg.dt:
                t0 = time.perf_counter()
                try:
                    reached = adapter.plan_step(exp_cfg.dt)
                except Exception:
                    log.exception(f"FAILED during adapter.plan_step at frame={frame}")
                    raise
                loop_profiler["adapter_step"] += time.perf_counter() - t0

                if adapter.last_collision_events:
                    camera_manager.focus_on_collision(adapter.last_collision_events[-1]["pos"])
                    if args_cli.exit_after_collision_secs is not None and collision_exit_deadline_frame is None:
                        collision_exit_deadline_frame = frame + int(args_cli.exit_after_collision_secs / dt)
                        checkpoint(
                            f"loop frame={frame}: first collision -- exiting at frame="
                            f"{collision_exit_deadline_frame} ({args_cli.exit_after_collision_secs}s later)"
                        )

                t0 = time.perf_counter()
                try:
                    if any(reached):
                        if DEBUG_LOOP:
                            checkpoint(f"loop frame={frame}: updating frontiers")
                        exploration.frontiers = exploration.get_frontiers()
                        exploration.update_goals_for_completed(reached)
                        adapter.draw_frontiers(exploration.frontiers)
                except Exception:
                    log.exception(f"FAILED during goal/frontier update at frame={frame}")
                    raise
                loop_profiler["goal_update"] += time.perf_counter() - t0

                if SHOW_EXPLORE_UI and frame % UI_UPDATE_EVERY == 0:
                    t0 = time.perf_counter()
                    try:
                        exploration.update_visualization()
                        if hasattr(exploration, "fig") and exploration.fig:
                            exploration.fig.canvas.draw_idle()
                            exploration.fig.canvas.flush_events()
                    except Exception:
                        log.exception(f"FAILED during exploration UI update at frame={frame}")
                    loop_profiler["ui_update"] += time.perf_counter() - t0

                time_since_last_plan -= exp_cfg.dt

            # --------------------------------------------------
            # 3. Spin propellers every frame
            # --------------------------------------------------
            try:
                for r in robots_sim:
                    r["art"].set_joint_velocity_target_index(
                        target=PROP_SPEED * r["directions"]
                    )
                    r["art"].write_data_to_sim()
            except Exception:
                log.exception(f"FAILED during propeller command at frame={frame}")
                raise

            # --------------------------------------------------
            # 4. Log poses occasionally
            # --------------------------------------------------
            t0 = time.perf_counter()
            try:
                if logger is not None and frame % POSE_LOG_EVERY == 0:
                    time_history.append(sim_time)
                    frame_history.append(frame)

                    for i, r in enumerate(robots_sim):
                        pose = r["art"].data.root_state_w[0, :7].clone()
                        pose_history[i].append(pose)
            except Exception:
                log.exception(f"FAILED during logging at frame={frame}")
                raise
            loop_profiler["pose_log"] += time.perf_counter() - t0

            # --------------------------------------------------
            # 5. Camera director
            # --------------------------------------------------
            if args_cli.force_track_robot is not None:
                if frame == 0:
                    camera_manager.set_view(
                        mode="track",
                        track_idx=args_cli.force_track_robot,
                        offset=tuple(args_cli.track_offset) if args_cli.track_offset else (-3.0, -3.0, 6.0),
                    )
            else:
                DRONE_VIEW_DURATION = 100
                num_robots = len(robots_sim)
                # One slot for the global/top view plus one slot per robot, then
                # repeat -- previously the global view only ever appeared once, at
                # phase 0, and every phase after that cycled through robots forever.
                cycle_len = num_robots + 1
                phase_in_cycle = (frame // DRONE_VIEW_DURATION) % cycle_len

                if frame % DRONE_VIEW_DURATION == 0:
                    if phase_in_cycle == 0:
                        camera_manager.set_view(mode="global")
                    else:
                        robot_idx = phase_in_cycle - 1
                        # Robot 0 is shot from the opposite side (x/y mirrored,
                        # height unchanged) so its segment isn't framed identically
                        # to every other robot's.
                        track_offset = (3.0, 3.0, 6.0) if robot_idx == 0 else (-3.0, -3.0, 6.0)
                        camera_manager.set_view(
                            mode="track",
                            track_idx=robot_idx,
                            offset=track_offset,
                        )

            camera_manager.update(robots_sim)

            # --------------------------------------------------
            # 6. Physics/render step
            # --------------------------------------------------
            t0 = time.perf_counter()
            checkpoint(f"loop frame={frame}: sim.step(render=True) begin")

            try:
                sim.step(render=do_render)
            except Exception:
                log.exception(f"FAILED during sim.step at frame={frame}")
                raise
            loop_profiler["sim_step"] += time.perf_counter() - t0

            if record_capture_fn is not None and do_render and frame % args_cli.record_every == 0:
                try:
                    out_path = os.path.join(args_cli.record_dir, f"frame_{record_saved_count:06d}.png")
                    record_capture_fn(record_get_viewport(), out_path)
                    record_saved_count += 1
                except Exception:
                    log.exception(f"FAILED capturing recorder frame at frame={frame}")

            # --------------------------------------------------
            # 7. Profiling
            # --------------------------------------------------
            loop_profiler["total_loop"] += time.perf_counter() - loop_start
            loop_profiler["count"] += 1

            if PROFILE_LOOP and frame > 0 and frame % PROFILE_EVERY == 0:
                _print_loop_profiler_stats(loop_profiler, frame, sim_time)
                _reset_loop_profiler(loop_profiler)

            # --------------------------------------------------
            # 8. Stopping conditions
            # --------------------------------------------------
            try:
                if exploration.exploration_complete():
                    success = True
                    checkpoint("Stopping condition reached: exploration complete")
                    break

                if exp_cfg.max_ticks and frame >= exp_cfg.max_ticks:
                    success = False
                    checkpoint("Stopping condition reached: max_ticks reached")
                    break

                if collision_exit_deadline_frame is not None and frame >= collision_exit_deadline_frame:
                    success = False
                    checkpoint("Stopping condition reached: exit-after-collision window elapsed")
                    break
            except Exception:
                log.exception(f"FAILED during stopping condition check at frame={frame}")
                raise

            frame += 1

    
    finally:
        if record_saved_count > 0:
            print(f"[SUMMARY] recorded {record_saved_count} frames to {args_cli.record_dir}")

        if adapter is not None:
            detected_pct = _unknown_obs_detected_pct(exploration.controller_list, unknown_obs)
            print(
                f"[SUMMARY] total_collisions={adapter.total_collisions} "
                f"per_robot={adapter.collision_count} "
                f"unknown_obs_detected={detected_pct:.1f}%"
            )
            _log_collision_summary(
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                algorithm=exp_cfg.exploration_algorithm,
                num_drones=len(robots_sim),
                known_obstacles=len(known_obs_array),
                unknown_obstacles=len(unknown_obs),
                unknown_obstacles_detected_pct=round(detected_pct, 1),
                lidar_enabled=bool(args_cli.use_rtx_lidar),
                collisions=adapter.total_collisions,
                collisions_per_robot=adapter.collision_count,
                success=success,
            )

        if logger is not None and frame_history:
            for i, r in enumerate(robots_sim):
                if len(pose_history[i]) > 0:
                    batched_poses = torch.stack(pose_history[i]).cpu().numpy()

                    for step_idx in range(len(batched_poses)):
                        logger.log_pose(
                            frame=frame_history[step_idx],
                            sim_time=time_history[step_idx],
                            robot_idx=i,
                            robot_name=r["name"],
                            pose=batched_poses[step_idx],
                        )

        try:
            if SHOW_EXPLORE_UI:
                plt.close("all")
        except Exception:
            pass

        try:
            simulation_app.close()
        except Exception:
            log.exception("FAILED during simulation_app.close()")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("TOP-LEVEL FAILURE (main crashed)")
        raise
    finally:
        log.info(f"Log written to: {LOG_PATH}")