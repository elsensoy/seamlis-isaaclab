import math
import torch
from tools.math_utils import map_xy

XY_SCALE = 1.0

# Crash-landing feel. Tune here if the fall should be quicker/slower.
CRASH_FALL_TIME = 0.35  # seconds to fall from cruise altitude down to the ground
GROUND_Z = 0.08         # resting height once crashed (small clearance above the floor mesh)

# The CBF/planning safety radius (robots.common.radius, e.g. 0.25m in
# blind_corner_isaac.yaml) is a padded margin, not the drone's actual visual
# footprint -- so the logical collision fires while the rendered body is
# still short of the obstacle surface. This is the assumed visual radius used
# to nudge the frozen crash position flush against whatever it hit, purely
# cosmetic (matches the project's more common robots.common.radius default).
CRASH_VISUAL_RADIUS = 0.15


def _ease_in(frac: float) -> float:
    # Accelerate into the fall, like gravity, instead of a constant-speed dip.
    return frac ** 2


class ExploreAdapter:
    """
    Kinematically puppeteers each drone's root pose to exactly match the
    exploration manager's planner state, with a fall-and-stay animation on z
    whenever that robot's planner step reports a collision: the robot falls
    to the ground and remains there, frozen in place, for the rest of the run.

    Two entry points, called at different rates:
      - plan_step(): advances the planner itself. Call once per planner tick.
      - pin_step(): re-writes the last planner target's pose and zeroes root
        velocity. Call every physics substep, so the sim never gets a window
        (between planner ticks) where contact/reaction forces from the
        environment or the spinning propellers can nudge the body before the
        next correction -- that residual drift is what reads as "floating".
    """

    TRAIL_ALPHA = 1.0  # trace transparency: 1.0 opaque, 0.0 invisible

    # Robot 0's FOV/trail color comes out pure red from the hue-by-index scheme
    # in run_isaac.py (hue 0 deg) -- overridden here to a warmer peach-pink for
    # the trail specifically, without touching the FOV mesh color or the hue
    # assignment itself.
    ROBOT_0_TRAIL_COLOR = (1.0, 0.55, 0.45)

    FRONTIER_COLOR = (0.15, 0.45, 1.0, 0.9)  # blue markers for current frontier set
    FRONTIER_PICKED_COLOR = (1.0, 0.55, 0.0, 1.0)  # orange: this robot's currently picked goal
    FRONTIER_POINT_SIZE = 9.0
    FRONTIER_PICKED_POINT_SIZE = 14.0
    FRONTIER_Z = 0.05  # just above the floor, so it reads as a map marker not a flight target

    def __init__(self, exploration, robots, fixed_z=1.0, draw_trails=True, draw_frontiers=True):
        self.exploration = exploration
        self.robots = robots
        self.fixed_z = fixed_z
        self.num_robots = len(robots)

        # Persistent trajectory traces in the Kit viewport (cosmetic, for
        # screenshots/video), one polyline per robot colored to match its FOV
        # mesh color. Drawn incrementally: one new line segment per plan_step()
        # position update, rather than clearing and redrawing the whole trail
        # every frame -- isaacsim.util.debug_draw's buffer persists on its own
        # until explicitly cleared.
        self.draw_trails = draw_trails
        # Current frontier set (the unexplored-boundary candidates the planner
        # is choosing goals from), redrawn as blue points whenever the caller
        # recomputes exploration.frontiers -- replaces the previous set rather
        # than accumulating, since it reflects "now", not history.
        self.frontiers_enabled = draw_frontiers
        self._trail_draw = None
        self._trail_color = []
        self._trail_last_point = []
        self._num_frontier_points = 0
        if self.draw_trails or self.frontiers_enabled:
            try:
                from isaacsim.util.debug_draw import _debug_draw
                self._trail_draw = _debug_draw.acquire_debug_draw_interface()
            except ModuleNotFoundError:
                # Only available with the full Kit/viewport extension set
                # (--viz kit); the headless experience doesn't load it.
                # Both features are purely cosmetic, so degrade gracefully
                # instead of crashing headless runs.
                self.draw_trails = False
                self.frontiers_enabled = False
            if self.draw_trails:
                for i, ctrl in enumerate(exploration.controller_list):
                    x, y = map_xy(ctrl.robot.get_position())
                    if i == 0:
                        r, g, b = self.ROBOT_0_TRAIL_COLOR
                    else:
                        r, g, b = robots[i].get("fov_color", (1.0, 1.0, 1.0))
                    self._trail_color.append((r, g, b, self.TRAIL_ALPHA))
                    self._trail_last_point.append((x, y, fixed_z))

        self.device = None
        self.dtype = None
        # Preallocated per-robot write buffers (populated lazily once the sim
        # device is known -- see pin_step()). Reused in place every frame so
        # pin_step() never triggers a CUDA allocation on the hot path.
        self._pose_gpu = None    # list[Tensor(1,7)] on device, one per robot
        self._pose_cpu = None    # list[Tensor(1,7)] pinned host staging buffer
        self._zero_vel_gpu = None  # single shared Tensor(1,6), always zero

        # Per-robot crash state machine: "flying" -> "falling" -> "crashed" (terminal).
        # Once crashed, the robot is frozen in place (x, y, yaw, and z) for good.
        self._crash_phase = ["flying"] * self.num_robots
        self._crash_timer = [0.0] * self.num_robots

        # Collision bookkeeping. One collision = one flying->falling transition,
        # i.e. counted once per crash, not once per tick the robot stays down.
        self.collision_count = [0] * self.num_robots
        self.total_collisions = 0
        # Collision events detected during the most recent plan_step() call,
        # for callers (e.g. the camera director) to react to. Reset every call.
        self.last_collision_events = []

        # Latest planner target per robot (x, y, yaw), refreshed in plan_step()
        # and held constant by pin_step() between planner ticks. Seeded from
        # the exploration manager's initial state so pin_step() has something
        # valid to write even before the first plan_step() call.
        self._target_xy_yaw = []
        for ctrl in exploration.controller_list:
            x, y = map_xy(ctrl.robot.get_position())
            self._target_xy_yaw.append((x, y, float(ctrl.robot.get_orientation())))

    def clear_trails(self):
        """Erase all drawn trajectory traces (they keep accumulating otherwise)."""
        if self.draw_trails:
            self._trail_draw.clear_lines()

    def draw_frontiers(self, frontiers):
        """Replace the drawn frontier markers with the current frontier set,
        plus each robot's currently picked goal drawn larger and in orange
        so it's obvious which frontier got picked.

        `frontiers` is exploration.frontiers: either a shapely LineString (as
        used by ExplorationManager) or a plain iterable of (x, y) pairs. The
        picked goals are read from exploration.robot_goals (one entry per
        robot, None until assigned). Call this whenever the caller recomputes
        exploration.frontiers or reassigns robot_goals -- e.g. after
        update_all_goals()/update_goals_for_completed(), not before, so the
        just-picked goal is reflected in this draw. Uses the points buffer,
        which is independent of the trails feature's lines buffer, so this
        never disturbs the persistent trajectory traces.
        """
        if not self.frontiers_enabled:
            return
        if self._num_frontier_points:
            self._trail_draw.clear_points()
            self._num_frontier_points = 0

        coords = list(getattr(frontiers, "coords", frontiers))
        points = [(*map_xy((x, y)), self.FRONTIER_Z) for x, y in coords]
        colors = [self.FRONTIER_COLOR] * len(points)
        sizes = [self.FRONTIER_POINT_SIZE] * len(points)

        for goal in getattr(self.exploration, "robot_goals", []):
            if goal is None:
                continue
            gx, gy = float(goal[0]), float(goal[1])
            points.append((*map_xy((gx, gy)), self.FRONTIER_Z))
            colors.append(self.FRONTIER_PICKED_COLOR)
            sizes.append(self.FRONTIER_PICKED_POINT_SIZE)

        if not points:
            return
        self._trail_draw.draw_points(points, colors, sizes)
        self._num_frontier_points = len(points)

    def _crash_touch_point(self, i: int, x: float, y: float) -> tuple:
        """Nudge a just-crashed robot's frozen (x, y) flush against the obstacle
        it hit, purely cosmetic -- doesn't affect collision counts, the
        planner, or self.exploration state at all. The safety-margin radius
        that triggered the collision is bigger than the drone's rendered
        footprint, so without this the frozen position visually falls short
        of the obstacle surface. Only applies when the obstacle that was hit
        is known (exploration.last_collision_info identifies it as this
        robot's unknown-obstacle hit); otherwise returns (x, y) unchanged.
        """
        info = getattr(self.exploration, "last_collision_info", None)
        if not isinstance(info, dict) or info.get("robot_idx") != i or info.get("type") != "unknown":
            return x, y
        obs = info.get("obs")
        if not obs or len(obs) < 3:
            return x, y
        obs_x, obs_y, obs_r = float(obs[0]), float(obs[1]), float(obs[2])
        dx, dy = x - obs_x, y - obs_y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return x, y
        touch_dist = obs_r + CRASH_VISUAL_RADIUS
        scale = touch_dist / dist
        return obs_x + dx * scale, obs_y + dy * scale

    def _register_collision(self, i: int, x: float, y: float, z: float):
        self.collision_count[i] += 1
        self.total_collisions += 1
        event = {"robot": i, "pos": (x, y, z), "count": self.collision_count[i]}
        self.last_collision_events.append(event)
        print(
            f"[COLLISION] robot={i} at ({x:.2f}, {y:.2f}) | "
            f"robot_count={self.collision_count[i]} total={self.total_collisions}"
        )

    def plan_step(self, dt_plan):
        """Advance the planner and refresh cached targets. Call at planner rate."""
        self.last_collision_events = []
        reached = self.exploration.move_robots()

        last_step_status = getattr(self.exploration, "last_step_status", None)
        for i, ctrl in enumerate(self.exploration.controller_list):
            # A crashed robot is frozen for the rest of the run -- don't pull a
            # fresh x/y/yaw from the controller even if it keeps ticking, or
            # the crashed drone would visibly snap ("jump") back onto whatever
            # position/orientation the controller's internal state drifts to.
            if self._crash_phase[i] != "flying":
                continue

            target_x, target_y = map_xy(ctrl.robot.get_position())
            target_yaw = float(ctrl.robot.get_orientation())

            status = None
            if last_step_status is not None and i < len(last_step_status):
                status = last_step_status[i]
            if status == -2:
                target_x, target_y = self._crash_touch_point(i, target_x, target_y)
                self._crash_phase[i] = "falling"
                self._crash_timer[i] = 0.0
                self._register_collision(i, target_x, target_y, self.fixed_z)

            self._target_xy_yaw[i] = (target_x, target_y, target_yaw)

            if self.draw_trails:
                new_point = (target_x, target_y, self.fixed_z)
                last_point = self._trail_last_point[i]
                if new_point != last_point:
                    self._trail_draw.draw_lines(
                        [last_point], [new_point], [self._trail_color[i]], [3.0]
                    )
                    self._trail_last_point[i] = new_point

        return reached

    def _z_for_robot(self, i: int, dt_frame: float) -> float:
        phase = self._crash_phase[i]
        if phase == "falling":
            self._crash_timer[i] += dt_frame
            frac = min(self._crash_timer[i] / CRASH_FALL_TIME, 1.0)
            z = self.fixed_z - (self.fixed_z - GROUND_Z) * _ease_in(frac)
            if frac >= 1.0:
                self._crash_phase[i] = "crashed"
                self._crash_timer[i] = 0.0
            return z

        if phase == "crashed":
            # Terminal: stays on the ground for the rest of the run.
            return GROUND_Z

        return self.fixed_z

    def pin_step(self, dt_frame, render_this_frame=True):
        """Re-assert each robot's pose and zero its root velocity. Call every physics frame.

        render_this_frame: whether this physics frame will actually be rendered.
        The sensor-frame USD transform is only visible to cameras/renders, so it's
        skipped on frames that won't be rendered -- the physics pose/velocity
        writes below still run every frame regardless, since those are what keep
        the body from drifting between corrections.
        """
        if self.device is None:
            self.device = torch.device(str(self.robots[0]["art"].data.root_state_w.device))
            self.dtype = torch.float32
            self._pose_gpu = [
                torch.zeros((1, 7), dtype=self.dtype, device=self.device)
                for _ in range(self.num_robots)
            ]
            self._pose_cpu = [
                torch.zeros((1, 7), dtype=self.dtype, pin_memory=(self.device.type == "cuda"))
                for _ in range(self.num_robots)
            ]
            self._zero_vel_gpu = torch.zeros((1, 6), dtype=self.dtype, device=self.device)

        with torch.inference_mode():
            for i in range(self.num_robots):
                art = self.robots[i]["art"]
                target_x, target_y, target_yaw = self._target_xy_yaw[i]
                target_z = self._z_for_robot(i, dt_frame)

                # Build body orientation quaternion in xyzw for drone_root, the
                # prim carrying PhysicsArticulationRootAPI -- write_root_pose_to_sim
                # sets ITS world pose directly, so drone_root's own baked USD
                # rotation is irrelevant (overridden every frame regardless).
                # drone_body (RigidBodyAPI, the visible mesh and propeller-joint
                # parent) is a *child* of drone_root with its own fixed local
                # +90deg X offset (confirmed via direct USD inspection). The raw
                # mesh geometry under drone_body is itself authored upside-down
                # (CAD import convention mismatch) -- Rx(-90deg) here, which
                # cancels the local +90deg exactly, reproduces that raw upside-down
                # orientation, not "upright". Empirically confirmed via the actual
                # rendered result: Rx(+180deg) left it tilted 90deg on its side,
                # Rx(-90deg) left it upside down (a 180deg flip apart, as expected
                # since both are rotations about the same X axis). The correction
                # that actually renders upright is Rx(+90deg) -- which also just
                # reproduces drone_root's own originally-authored USD rotation,
                # further rotated by the desired yaw.
                half = 0.5 * target_yaw
                c, s = math.cos(half), math.sin(half)
                k = 0.7071067811865476  # sqrt(0.5): Rx(+90deg) half-angle factor

                # Fill the preallocated host staging buffer (cheap CPU writes,
                # no allocation) then copy in place into the preallocated GPU
                # buffer -- avoids a fresh CUDA allocation every robot/frame.
                cpu_buf = self._pose_cpu[i]
                cpu_buf[0, 0] = target_x
                cpu_buf[0, 1] = target_y
                cpu_buf[0, 2] = target_z
                cpu_buf[0, 3] = k * c  # qx
                cpu_buf[0, 4] = k * s  # qy
                cpu_buf[0, 5] = k * s  # qz
                cpu_buf[0, 6] = k * c  # qw

                pose_buf = self._pose_gpu[i]
                pose_buf.copy_(cpu_buf, non_blocking=True)

                art.write_root_pose_to_sim_index(root_pose=pose_buf)

                # Zero root velocity every frame: without this, contact/depenetration
                # forces from the environment and reaction torque from the spinning
                # propeller joints accumulate velocity between corrective writes and
                # the body drifts/tumbles -- since we're kinematically puppeteering
                # pose here, physics should never be left to integrate it.
                # The buffer is always all-zero, so it's allocated once and reused
                # (read-only) for every robot on every frame.
                art.write_root_velocity_to_sim_index(root_velocity=self._zero_vel_gpu)

                # update sensor frame to match the pose (visual-only; skip on
                # frames that won't be rendered)
                if render_this_frame:
                    suite = self.robots[i].get("sensors", None)
                    if suite is not None:
                        suite.set_sensor_frame_pose_xyyaw(
                            x=target_x,
                            y=target_y,
                            z=target_z,
                            yaw_rad=target_yaw,
                        )
