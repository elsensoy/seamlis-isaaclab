# In scripts/utils.py
import yaml
import os
import datetime
import csv
import numpy as np
from typing import Optional, Literal
from pydantic import BaseModel

# class ExperimentConfig(BaseModel):
#     name: str
#     dt: float = 0.05
#     exploration_algorithm: Literal["CoScan", "Frontier"] = "CoScan"
#     show_animation: bool = False
#     save_animation: bool = False
#     max_ticks: Optional[int] = None
#     seed: Optional[int] = None
#     video_dir: str = "videos"
#     video_filename: Optional[str] = None

import yaml
import os
import csv
import numpy as np
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

# 1. CONFIGURATION MODELS (TK's Hardcoded Defaults) 

class DeadlockConfig(BaseModel):
    window_s: float = 3.0
    position_eps: float = 0.28
    speed_eps: float = 0.06
    goal_margin: float = 0.9
    cooldown_s: float = 3.5
    max_recoveries: int = 12

class GatekeeperConfig(BaseModel):
    nominal: str = "visibility_area"
    backup: str = "velocity_tracking_yaw"
    nominal_horizon: float = 0.4
    backup_horizon: float = 1.8
    event_offset: float = 0.0
    horizon_discount: float = 0.05
    validation_slack: float = 0.30
    braking_margin: float = 0.90

class RobotCommonConfig(BaseModel):
    model: str = "DoubleIntegrator2D"
    v_max: float = 1.35
    a_max: float = 1.9
    radius: float = 0.15 # default
    flight_z: float = 1.0  # cruise altitude (m); drones lock to this until a collision sends them to the ground for good
    fov_angle: float = 70.0
    cam_range: float = 4.5
    deadlock: DeadlockConfig = DeadlockConfig()
    mpc_horizon: int = 10
    mpc_cbf_alpha1: float = 0.55
    mpc_cbf_alpha2: float = 0.55
    sensor: str = "rgbd"
    exploration: bool = True
    reached_threshold: float = 1.8
    min_goal_distance: float = 2.6
    nominal_k_v: float = 2.2
    nominal_k_a: float = 2.2
    unknown_obs_detection: str = "fov"
    visibility_violation_mode: str = "point_mass"
    visibility_violation_tolerance: float = 0.02

class RobotInstance(BaseModel):
    robot_id: int
    x0: List[float]  # [x, y, yaw]

class EnvironmentConfig(BaseModel):
    use_default_layout: bool = True
    layout_type: Literal["indoor", "open"] = "indoor"
    width: float = 24.0
    height: float = 18.0
    resolution: float = 0.16
    circles: List[List[float]] = []
    superellipsoids: List[List[float]] = []
    unknown_obstacles: List[List[float]] = []


class ControllerConfig(BaseModel):
    pos: str = "mpc_cbf"
    att: str = "gatekeeper"
    gatekeeper: GatekeeperConfig = GatekeeperConfig()

class FullConfig(BaseModel):
    experiment_name: str = "Default_Exp"
    dt: float = 0.1
    tf: float = 300.0
    exploration_algorithm: Literal["CoScan", "Frontier"] = "Frontier"
     
    show_animation: bool = False
    max_ticks: Optional[int] = 5000  # give it a default so it doesn't crash
    seed: Optional[int] = None
    
    environment: EnvironmentConfig = EnvironmentConfig()
    controller: ControllerConfig = ControllerConfig()
  
    
    # NEW: A wrapper class to handle the common and instances nesting in YAML
    class RobotSection(BaseModel):
        common: RobotCommonConfig = RobotCommonConfig()
        instances: List[RobotInstance] = []

    robots: RobotSection #  matches the robots key  

#  2. UTILITY FUNCTIONS 

def load_config(path: str) -> FullConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # The Extended Behavioral Test Bank (configs/test_0..test_11*.yaml) nests
    # experiment-level fields under an `experiment:` key, using `name` instead
    # of `experiment_name`. FullConfig expects these at the top level, so
    # pydantic silently ignores the unrecognized `experiment` key wholesale
    # (dropping dt/exploration_algorithm/max_ticks/etc. to defaults) unless
    # it's hoisted first -- this was confirmed actually happening (dt fell
    # back to 0.1 instead of the intended 0.05, exploration_algorithm to
    # "Frontier" regardless of what was configured) before this fix.
    experiment_block = data.pop("experiment", None)
    if isinstance(experiment_block, dict):
        if "name" in experiment_block and "experiment_name" not in data:
            experiment_block["experiment_name"] = experiment_block.pop("name")
        for key, value in experiment_block.items():
            data.setdefault(key, value)

    return FullConfig(**data)

 
def configure_planner_from_yaml(manager, env_cfg_obj):
    # 1. Access the manager's internal environment handler
    env_handler = manager.env_handler
    
    # 2. Update Basic Properties
    env_handler.width = float(env_cfg_obj.width)
    env_handler.height = float(env_cfg_obj.height)
    env_handler.resolution = float(env_cfg_obj.resolution)
    
     
    # This actually creates the new numpy array of the correct size
    if hasattr(env_handler, 'initialize_grid'):
        env_handler.initialize_grid()
    elif hasattr(env_handler, '_setup_grid'):
        env_handler._setup_grid()
    # ------------------------------------------------
    
#     # 3. re-build known obstacles using our existing helper
#     # We pass the env_cfg_obj directly to build_known_obs
    yaml_known_obs = build_known_obs(env_cfg_obj)
    
    if yaml_known_obs is not None:
        # reset current obstacles in the manager
        env_handler.obs_circle = []
        env_handler.obs_superellipsoid = []
        
        # populate with new data based on the flag at the end of the 7D vector
        for obs in yaml_known_obs:
            # obs format: [params..., flag]
            # flag=0.0 -> Circle, flag=1.0 -> Superellipsoid
            if obs[-1] == 0.0:
                env_handler.obs_circle.append(obs)
            else:
                env_handler.obs_superellipsoid.append(obs)
 
    manager.set_env_obstacles(env_handler)
    manager.set_env_workspace(env_handler)
   
    if hasattr(manager, 'ax') and manager.ax is not None:
        manager.ax.set_xlim(0, env_handler.width)
        manager.ax.set_ylim(0, env_handler.height)
        # If the manager uses an extent for imshow, update it:
        if hasattr(manager, 'img'):
            manager.img.set_extent([0, env_handler.width, 0, env_handler.height])
    # --------------------------------------
    
    for ctrl in manager.controller_list:
        ctrl.env = env_handler
        
    print(f"[Utils] Planner configured and Grid Reset: {env_handler.width}x{env_handler.height}m")
    return env_handler


def parse_robot_specs(cfg: FullConfig) -> List[Dict[str, Any]]:
    """Flattens Pydantic config into the dictionary format the Manager expects."""
    specs = []
    
 
    common_dict = cfg.robots.common.model_dump() 
    ctrl_dict = cfg.controller.model_dump()

    for inst in cfg.robots.instances:
        # Start with common defaults, override with instance-specific x0
        spec = common_dict.copy()
        spec.update(inst.model_dump())
        
        # Flatten Nested Deadlock
        if "deadlock" in spec:
            dl = spec.pop("deadlock")
            for k, v in dl.items():
                spec[f"deadlock_{k}"] = v
            
        # Add Gatekeeper logic if active
        if cfg.controller.att == "gatekeeper":
            gk = ctrl_dict["gatekeeper"]
            for k, v in gk.items():
                key = f"gatekeeper_{k}"
                if k == "braking_margin": key = "gatekeeper_braking_distance_margin"
                spec[key] = v
        
        specs.append(spec)
    return specs

def build_known_obs(env_cfg: EnvironmentConfig):
    """Formats obstacles into the 7D array for the environment handler."""
    rows = []
    for c in env_cfg.circles:
        if len(c) == 3: rows.append([*c, 0.0, 0.0, 0.0, 0.0])
    
    for s in env_cfg.superellipsoids:
        if len(s) == 6: rows.append([*s, 1.0])

    # env.Env's known_obs default is [], and it iterates the value directly
    # (`for obs in known_obs`) with no None-guard -- an empty map (no circles,
    # no superellipsoids, e.g. configs/test_0_empty_map.yaml) must return an
    # empty array here, not None, or env.Env(known_obs=None) crashes outright.
    return np.asarray(rows, dtype=float) if rows else np.empty((0, 7))

def get_scene_scale(env_cfg: EnvironmentConfig, asset_native_size: float = 20.0):
    """
    Calculates the scale vector for a USD asset to match YAML dimensions.
    asset_native_size: The default width/length of USD warehouse.
    """
    scale_x = env_cfg.width / asset_native_size
    scale_y = env_cfg.height / asset_native_size
    # Return (x, y, z). We keep Z at 1.0 so walls don't get taller/shorter.
    return (scale_x, scale_y, 1.0)

# 3. LOGGING & SIMULATION SYNC 
 
class ExperimentLogger:
    def __init__(self, experiment_name, controller_pos, controller_att):
        self.exp_name = experiment_name
        self.ctrl_pos = controller_pos
        self.ctrl_att = controller_att
        self.comparison_file = "logs/comparison_results.csv"
        self.pose_file = "logs/pose_log.csv"
        # Track unique grid indices visited while they were unknown (-1)
        self.blind_cells_visited = set()
        self.total_cells_entered = set()

    def record_blind_move(self, grid_coords, is_unknown):
        """Call this every frame for every robot."""
        self.total_cells_entered.add(grid_coords)
        if is_unknown:
            self.blind_cells_visited.add(grid_coords)

    def log_summary(self, total_frames, sim_time, real_duration, collisions, first_collision_expl):
        fieldnames = [
            "experiment", "pos_ctrl", "att_ctrl", "total_frames", 
            "sim_time", "collisions", "first_collision_expl", 
            "blind_cells_count", "blind_ratio", "success"
        ]
        
        success = 1 if collisions == 0 else 0
        
        # Calculate the ratio of "blind" movement vs total movement
        blind_count = len(self.blind_cells_visited)
        total_moved = len(self.total_cells_entered)
        blind_ratio = (blind_count / total_moved) if total_moved > 0 else 0
        
        file_exists = os.path.isfile(self.comparison_file)
        
        with open(self.comparison_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                "experiment": self.exp_name,
                "pos_ctrl": self.ctrl_pos,
                "att_ctrl": self.ctrl_att,
                "total_frames": total_frames,
                "sim_time": round(sim_time, 2),
                "collisions": collisions,
                "first_collision_expl": round(first_collision_expl, 2),
                "blind_cells_count": blind_count,
                "blind_ratio": round(blind_ratio, 4),
                "success": success
            })

 
    def log_pose(self, frame, sim_time, robot_id, robot_name, pose7):
        """
        pose7: [x,y,z,qx,qy,qz,qw] in world frame
        """
        fieldnames = ["experiment", "pos_ctrl", "att_ctrl",
                      "frame", "sim_time",
                      "robot_id", "robot_name",
                      "x", "y", "z"]
        file_exists = os.path.isfile(self.pose_file)

        with open(self.pose_file, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
            w.writerow({
                "experiment": self.exp_name,
                "pos_ctrl": self.ctrl_pos,
                "att_ctrl": self.ctrl_att,
                "frame": int(frame),
                "sim_time": float(sim_time),
                "robot_id": int(robot_id),
                "robot_name": robot_name,
                "x": float(pose7[0]),
                "y": float(pose7[1]),
                "z": float(pose7[2]),
            })

class CameraManager:
    def __init__(self, sim, env_cfg):
        self.sim = sim

        # Store the global view from YAML dimensions
        self.center_x = env_cfg.width / 2.0
        self.center_y = env_cfg.height / 2.0
        # Straight-down birdview centered on the environment, matching the
        # top-down framing of the 2D Matplotlib exploration plots (which show
        # the full [0, width] x [0, height] rectangle from directly above)
        # rather than the previous oblique ~45deg pulled-back angle. Altitude
        # scaled with a margin so the full footprint stays in frame.
        env_span = max(env_cfg.width, env_cfg.height)
        # A perfectly vertical eye->target vector is degenerate for a look-at
        # camera (the up-vector cross product goes to zero), which rendered as
        # a black screen. Nudge the eye a hair off-axis to keep the look-at
        # rotation well-defined -- specifically along -Y (not X or an
        # arbitrary axis): tilting the eye slightly toward -Y makes +Y
        # reliably "far"/screen-up and +X screen-right (the standard
        # north-up map convention, matching the 2D exploration plots),
        # regardless of whatever up-vector fallback the look-at math uses
        # for a near-degenerate view. An X (or diagonal) nudge leaves that
        # mapping to an internal fallback convention instead, which is what
        # produced a visibly skewed/rotated framing here.
        self.global_eye = (
            self.center_x,
            self.center_y - env_span * 0.01,
            env_span * 1.5,
        )
        self.global_target = (self.center_x, self.center_y, 0.0)

        # State variables
        self.mode = "global"  # Options: "global", "track"
        self.track_idx = 0    # Which robot to follow

        # Relative camera position for tracking: [x_offset, y_offset, z_offset]
        # Closer and more tilted than a near-top-down look, so motion/orientation
        # detail on the tracked drone is actually visible.
        self.offset = np.array([-3.0, -3.0, 6.0])

        # Collision-focus override: takes priority over whatever `mode` is set
        # to (global/track) for a fixed number of frames, then falls back on
        # its own -- callers just call focus_on_collision(), no need to
        # restore the previous mode afterward.
        self.collision_focus_offset = np.array([-2.5, -2.5, 3.5])
        self.collision_focus_hold_frames = 150
        self._collision_point = None
        self._collision_frames_left = 0

    def set_view(self, mode: str, track_idx: int = 0, offset: tuple = (-3.0, -3.0, 6.0)):
        """Update the active camera perspective."""
        self.mode = mode
        self.track_idx = track_idx
        self.offset = np.array(offset)

    def focus_on_collision(self, point_xyz, hold_frames: int = None):
        """Cut to a birdview centered on a collision site for a few seconds."""
        self._collision_point = tuple(point_xyz)
        self._collision_frames_left = hold_frames if hold_frames is not None else self.collision_focus_hold_frames

    def update(self, robots_sim):
        """Call this every simulation frame to update the viewport."""
        try:
            if self._collision_frames_left > 0 and self._collision_point is not None:
                cx, cy, cz = self._collision_point
                eye = (
                    cx + self.collision_focus_offset[0],
                    cy + self.collision_focus_offset[1],
                    cz + self.collision_focus_offset[2],
                )
                self.sim.set_camera_view(eye=eye, target=(cx, cy, cz))
                self._collision_frames_left -= 1
                return

            if self.mode == "global":
                self.sim.set_camera_view(eye=self.global_eye, target=self.global_target)

            elif self.mode == "track":
                if 0 <= self.track_idx < len(robots_sim):
                    # Keep it as a tensor for speed, only convert to list at the last second
                    root_state = robots_sim[self.track_idx]["art"].data.root_state_w[0, :3]
                    pos_np = root_state.tolist() # Faster than .detach().cpu().numpy() for single vectors

                    eye = [pos_np[0] + self.offset[0],
                        pos_np[1] + self.offset[1],
                        pos_np[2] + self.offset[2]]

                    self.sim.set_camera_view(eye=tuple(eye), target=tuple(pos_np))
                else:
                    # Fallback if an invalid robot index is passed
                    self.sim.set_camera_view(eye=self.global_eye, target=self.global_target)
        except Exception as e:
            # We don't want a camera glitch to crash a long-running training/eval loop
            pass

 