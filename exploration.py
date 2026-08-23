import sys
import os
# Add the safe_control directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'safe_control'))

import numpy as np
import heapq
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon, Point, LineString

import matplotlib.pyplot as plt

from tracking_controller import LocalTrackingController
from algorithms.co_scan import CoScanPlanner
from algorithms.frontier_vanilla import FrontierPlanner
from safe_control.utils import plotting
from safe_control.utils import env
from safe_control.utils.geometry import custom_merge

"""
Created on June 22nd, 2024
@author: Taekyung Kim

@description: 

@functions to be implemented: 
    An Exploration class
        1. correctly assign frontier as goal
        2. assign global goal at every iterations? or just assign it after it reaches the goal?

@required-scripts: tracking.py, utils/plotting.py, utils/env.py, algorithms/co_scan.py
"""


def _normalize_obs_array(obs):
    obs = np.array([] if obs is None else obs, dtype=float)
    if obs.size == 0:
        return np.empty((0, 7))
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if obs.shape[1] == 3:
        obs = np.hstack((obs, np.zeros((obs.shape[0], 4))))
    elif obs.shape[1] < 7:
        obs = np.hstack((obs, np.zeros((obs.shape[0], 7 - obs.shape[1]))))
    elif obs.shape[1] > 7:
        obs = obs[:, :7]
    return obs


class ExplorationManager:
    def __init__(self, X0s, robot_specs, controller_type, exploration_algorithm='CoScan',
                  dt=0.1,
                  show_animation=False, save_animation=False,
                  env_handler=None, env_width=20.0, env_height=20.0,
                  known_obs=None, unknown_obs=None, env_resolution=0.1,
                  use_astar_waypoints=False, coverage_target=0.98):
        self.robot_specs = robot_specs
        self.num_robot = len(robot_specs)
        self.dt = dt
        self.failed = False
        self.last_step_status = [0] * self.num_robot
        self.last_collision_info = None
        self.last_termination_reason = None
        self.last_step_count = 0
        self.last_sim_time = 0.0
        self.last_success = False
        self.latest_obstacle_map = None
        self.use_astar_waypoints = bool(use_astar_waypoints)
        self.coverage_target = float(np.clip(coverage_target, 0.0, 1.0))
        self._force_reassign_agents = set()
        self._termination_marker_drawn = False

        self.known_obs = _normalize_obs_array(known_obs)
        self.unknown_obs = _normalize_obs_array(unknown_obs)

        if env_handler is None:
            self.env_handler = env.Env(
                width=env_width,
                height=env_height,
                known_obs=self.known_obs,
                resolution=env_resolution,
            )
        else:
            self.env_handler = env_handler
            if self.known_obs.size == 0:
                self.known_obs = self._collect_known_obs_from_env(self.env_handler)

        self.show_animation = show_animation
        self.save_animation = save_animation

        if self.show_animation:
            self.plot_handler = plotting.Plotting(
                width=self.env_handler.width,
                height=self.env_handler.height,
                known_obs=self.known_obs,
            )
            # Keep animation output untitled for cleaner hero figures/videos.
            self.ax, self.fig = self.plot_handler.plot_grid("")
        else:
            # No Matplotlib figure/axes/artists for headless runs.
            self.plot_handler = None
            self.fig = None
            self.ax = None

        self.controller_list = []
        for i in range(self.num_robot):
            tracking_controller = LocalTrackingController(X0s[i], self.robot_specs[i], controller_type,
                                         dt=dt,
                                         show_animation=show_animation,
                                         save_animation=save_animation,
                                         ax=self.ax, fig=self.fig,
                                         env=self.env_handler)
            self.controller_list.append(tracking_controller)
        self._apply_environment_to_controllers()
        self.merged_global_map = Polygon()
        self.frontiers = LineString()

        if self.show_animation:
            self.frontiers_scatter = self.ax.scatter([],[],s=10,facecolors='orange',edgecolors='orange')
            self.global_goals_scatter = self.ax.scatter([],[],s=10,facecolors='blue',edgecolors='blue')
        else:
            self.frontiers_scatter = None
            self.global_goals_scatter = None

        # Set up the environment
        self.set_env_obstacles(self.env_handler)
        self.set_env_workspace(self.env_handler)
        self.static_obstacle_map = self._compute_static_obstacle_map()
        free_space_geom = self.env_workspace.difference(self.env_obstacles)
        self.free_space_area = max(float(free_space_geom.area), 1e-6)
        self.last_coverage_ratio = 0.0
        self.robot_goals = [None] * self.num_robot
        self._setup_deadlock_monitor()

        if exploration_algorithm == 'CoScan':
            self.exploration_algorithm = CoScanPlanner()
        elif exploration_algorithm == 'Frontier':
            self.exploration_algorithm = FrontierPlanner(fov_angle=self.controller_list[0].robot.fov_angle)
        else:
            raise ValueError(f"Exploration algorithm {exploration_algorithm} is not implemented")

    def _setup_deadlock_monitor(self):
        ref_spec = self.robot_specs[0] if len(self.robot_specs) > 0 else {}
        self.deadlock_enabled = bool(ref_spec.get('enable_deadlock_recovery', True))
        if not self.deadlock_enabled:
            self.deadlock_window_steps = 0
            self.deadlock_position_eps = 0.0
            self.deadlock_speed_eps = 0.0
            self.deadlock_goal_margin = 0.0
            self.deadlock_cooldown_steps = 0
            self._deadlock_hist = [[] for _ in range(self.num_robot)]
            self._deadlock_cooldown = np.zeros(self.num_robot, dtype=np.int32)
            self._deadlock_recovery_count = np.zeros(self.num_robot, dtype=np.int32)
            self.deadlock_max_recoveries = 0
            self.deadlock_exclusion_radius_cells = 0
            self.deadlock_exclusion_ttl_steps = 0
            self.deadlock_max_exclusions = 0
            self._deadlock_exclusions = [[] for _ in range(self.num_robot)]
            return

        deadlock_window_s = float(ref_spec.get('deadlock_window_s', 3.0))
        self.deadlock_window_steps = max(int(np.ceil(deadlock_window_s / max(self.dt, 1e-3))), 12)
        self.deadlock_position_eps = float(ref_spec.get('deadlock_position_eps', 0.28))
        self.deadlock_speed_eps = float(ref_spec.get('deadlock_speed_eps', 0.06))
        self.deadlock_goal_margin = float(ref_spec.get('deadlock_goal_margin', 0.9))
        deadlock_cooldown_s = float(ref_spec.get('deadlock_cooldown_s', 2.5))
        self.deadlock_cooldown_steps = max(int(np.ceil(deadlock_cooldown_s / max(self.dt, 1e-3))), 8)
        self._deadlock_hist = [[] for _ in range(self.num_robot)]
        self._deadlock_cooldown = np.zeros(self.num_robot, dtype=np.int32)
        self._deadlock_recovery_count = np.zeros(self.num_robot, dtype=np.int32)
        self.deadlock_max_recoveries = int(ref_spec.get('deadlock_max_recoveries', 6))
        exclusion_radius = float(
            ref_spec.get(
                'deadlock_exclusion_radius',
                max(1.8, self.deadlock_goal_margin + 0.8),
            )
        )
        self.deadlock_exclusion_radius_cells = max(
            int(np.ceil(exclusion_radius / max(self.env_handler.resolution, 1e-3))),
            4,
        )
        exclusion_ttl_s = float(
            ref_spec.get(
                'deadlock_exclusion_ttl_s',
                max(9.0, deadlock_window_s + deadlock_cooldown_s + 2.0),
            )
        )
        self.deadlock_exclusion_ttl_steps = max(
            int(np.ceil(exclusion_ttl_s / max(self.dt, 1e-3))),
            self.deadlock_cooldown_steps + 1,
        )
        self.deadlock_max_exclusions = int(ref_spec.get('deadlock_max_exclusions', 5))
        self._deadlock_exclusions = [[] for _ in range(self.num_robot)]

    def _reset_deadlock_history(self, idx):
        if idx < 0 or idx >= self.num_robot:
            return
        self._deadlock_hist[idx] = []

    def _tick_deadlock_exclusions(self, idx):
        if idx < 0 or idx >= self.num_robot:
            return
        if not hasattr(self, '_deadlock_exclusions'):
            return
        entries = self._deadlock_exclusions[idx]
        if len(entries) == 0:
            return
        updated = []
        for entry in entries:
            ttl = int(entry.get('ttl', 0)) - 1
            if ttl <= 0:
                continue
            updated.append({'xy': np.array(entry['xy'], dtype=np.int32), 'ttl': ttl})
        self._deadlock_exclusions[idx] = updated

    def _register_deadlock_exclusion(self, idx, goal):
        if idx < 0 or idx >= self.num_robot:
            return
        if goal is None:
            return
        if not hasattr(self, '_deadlock_exclusions'):
            return

        goal_arr = np.asarray(goal, dtype=float).reshape(-1)
        if goal_arr.size < 2:
            return
        goal_grid = self.env_handler.f_to_grid(goal_arr[:2])
        if np.asarray(goal_grid).size < 2:
            return
        goal_xy = np.array([int(goal_grid[0]), int(goal_grid[1])], dtype=np.int32)

        entries = self._deadlock_exclusions[idx]
        merge_radius = max(2.0, 0.5 * float(self.deadlock_exclusion_radius_cells))
        for entry in entries:
            entry_xy = np.array(entry['xy'], dtype=float).reshape(2)
            if np.linalg.norm(entry_xy - goal_xy.astype(float)) <= merge_radius:
                entry['xy'] = goal_xy
                entry['ttl'] = int(self.deadlock_exclusion_ttl_steps)
                return

        entries.append({'xy': goal_xy, 'ttl': int(self.deadlock_exclusion_ttl_steps)})
        max_entries = max(int(self.deadlock_max_exclusions), 1)
        if len(entries) > max_entries:
            self._deadlock_exclusions[idx] = entries[-max_entries:]

    def _mark_agent_deadlock_if_needed(self, idx, step_status, has_reached_goal):
        if not getattr(self, 'deadlock_enabled', True):
            return False
        if idx < 0 or idx >= self.num_robot:
            return False

        if self._deadlock_cooldown[idx] > 0:
            self._deadlock_cooldown[idx] -= 1

        controller = self.controller_list[idx]
        pos = np.asarray(controller.robot.get_position(), dtype=float).reshape(-1)
        hist = self._deadlock_hist[idx]
        hist.append(pos[:2].copy())
        if len(hist) > self.deadlock_window_steps:
            del hist[0]

        if has_reached_goal or step_status in [-1, -2]:
            self._reset_deadlock_history(idx)
            return False
        if self._deadlock_cooldown[idx] > 0:
            return False

        goal = self.robot_goals[idx]
        if goal is None:
            return False
        if len(hist) < self.deadlock_window_steps:
            return False

        goal_xy = np.asarray(goal, dtype=float).reshape(-1)[:2]
        goal_dist = float(np.linalg.norm(pos[:2] - goal_xy))
        reached_threshold = float(self.robot_specs[idx].get('reached_threshold', 0.3))
        min_goal_dist = max(reached_threshold + self.deadlock_goal_margin, 1.6 * reached_threshold)
        if goal_dist < min_goal_dist:
            return False

        hist_arr = np.asarray(hist, dtype=float)
        displacement = float(np.linalg.norm(hist_arr[-1] - hist_arr[0]))
        step_disp = np.linalg.norm(np.diff(hist_arr, axis=0), axis=1)
        mean_speed = float(np.mean(step_disp) / max(self.dt, 1e-3)) if step_disp.size > 0 else 0.0

        if displacement <= self.deadlock_position_eps and mean_speed <= self.deadlock_speed_eps:
            self._register_deadlock_exclusion(idx, goal)
            self._deadlock_cooldown[idx] = self.deadlock_cooldown_steps
            self._deadlock_recovery_count[idx] += 1
            self._reset_deadlock_history(idx)
            print(
                f"Deadlock recovery trigger: robot={idx}, goal_dist={goal_dist:.3f}, "
                f"disp={displacement:.3f}, speed={mean_speed:.3f}. Reassigning goal."
            )
            return True
        return False

    def _draw_termination_marker(self, target_robot_ids=None):
        if (not self.show_animation) or self._termination_marker_drawn:
            return
        self._termination_marker_drawn = True

        if target_robot_ids is None:
            target_robot_ids = list(range(self.num_robot))
        if not isinstance(target_robot_ids, (list, tuple, np.ndarray)):
            target_robot_ids = [int(target_robot_ids)]

        valid_targets = []
        for rid in target_robot_ids:
            rid_int = int(rid)
            if 0 <= rid_int < self.num_robot and rid_int not in valid_targets:
                valid_targets.append(rid_int)
        if len(valid_targets) == 0:
            valid_targets = [0]

        for controller in self.controller_list:
            controller.robot.render_plot()

        for rid in valid_targets:
            pos = np.asarray(self.controller_list[rid].robot.get_position(), dtype=float).reshape(-1)
            self.ax.text(
                float(pos[0]) + 0.45,
                float(pos[1]) + 0.45,
                '!',
                color='red',
                weight='bold',
                fontsize=24,
                zorder=30,
            )

        try:
            self.controller_list[0].draw_plot(pause=0.35, force_save=True)
        except Exception:
            pass

    def _finalize_animation(self):
        if self.save_animation and len(self.controller_list) > 0:
            self.controller_list[0].export_video()

    def close(self):
        '''
        Release this manager's Matplotlib figure, if it created one.
        No-op for headless (show_animation=False) managers, which never
        create a figure in the first place. Callers that construct many
        ExplorationManager instances in a loop (e.g. benchmark sweeps with
        show_animation=True) should call this when done with each one to
        avoid accumulating open figures.
        '''
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def _finalize_failure(self, target_robot_ids=None):
        self._draw_termination_marker(target_robot_ids=target_robot_ids)
        self._finalize_animation()

    @staticmethod
    def _collect_known_obs_from_env(env_handler):
        known_obs_parts = []
        circle_obs = _normalize_obs_array(getattr(env_handler, 'obs_circle', []))
        if circle_obs.size > 0:
            known_obs_parts.append(circle_obs)
        super_obs = _normalize_obs_array(getattr(env_handler, 'obs_superellipsoid', []))
        if super_obs.size > 0:
            known_obs_parts.append(super_obs)
        if len(known_obs_parts) == 0:
            return np.empty((0, 7))
        return np.vstack(known_obs_parts)

    @staticmethod
    def _superellipsoid_to_polygon(obs_info, num_points=100):
        ox, oy, a, b, e, theta = np.asarray(obs_info[:6], dtype=float)
        a = max(abs(a), 1e-3)
        b = max(abs(b), 1e-3)
        e = max(abs(e), 2.0)

        phi = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
        x = a * np.sign(np.cos(phi)) * np.abs(np.cos(phi)) ** (2.0 / e)
        y = b * np.sign(np.sin(phi)) * np.abs(np.sin(phi)) ** (2.0 / e)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        points = (rot @ np.vstack((x, y))).T + np.array([ox, oy])
        return Polygon(points)

    def _apply_environment_to_controllers(self):
        known_obs = np.copy(self.known_obs)
        unknown_obs = np.copy(self.unknown_obs)
        for controller in self.controller_list:
            controller.obs = known_obs
            controller.set_unknown_obs(unknown_obs)

    def set_env_workspace(self, env_handler):
        # Add workspace boundary
        self.env_workspace = Polygon([(0, 0), (0, env_handler.height), 
                             (env_handler.width, env_handler.height), 
                             (env_handler.width, 0)])
        return self.env_workspace
        
    def set_env_obstacles(self, env_handler):
        # Get obstacle information
        obs_rectangle = env_handler.obs_rectangle
        obs_circle = env_handler.obs_circle
        obs_superellipsoid = env_handler.obs_superellipsoid

        # Create obstacle geometries
        obstacle_geometries = []
        
        for rect in obs_rectangle:
            x, y, w, h = rect
            obstacle_geometries.append(Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)]))
        for circle in obs_circle:
            x, y, r = circle[:3]
            obstacle_geometries.append(Point(x, y).buffer(r))
        for obs in obs_superellipsoid:
            obstacle_geometries.append(self._superellipsoid_to_polygon(obs))

        # Combine all obstacles
        if len(obstacle_geometries) == 0:
            self.env_obstacles = Polygon()
        else:
            self.env_obstacles = unary_union(obstacle_geometries)
        return self.env_obstacles

    def set_waypoints(self, robot_ids, waypoints):
        # Convert robot_ids and waypoints to lists if a single instance is provided
        if not isinstance(robot_ids, list):
            robot_ids = [robot_ids]
            waypoints = [waypoints]
        
        # Ensure the number of robot_ids matches the number of waypoints
        if len(robot_ids) != len(waypoints):
            raise ValueError("The number of robot_ids must match the number of waypoints")

        # Set waypoints for each specified robot
        for robot_id, waypoint in zip(robot_ids, waypoints):
            if 0 <= robot_id < self.num_robot:
                self.controller_list[robot_id].set_waypoints(waypoint)
            else:
                raise ValueError(f"Invalid robot_id: {robot_id}. Must be between 0 and {self.num_robot - 1}")
            
    def merge_sensing_footprints(self):
        '''
        Merge all sensing footprints from all robots into a global, centralized map
        '''
        all_footprints = []

        # Collect sensing footprints from all robots
        for controller in self.controller_list:
            robot_footprints = controller.robot.sensing_footprints
            all_footprints.append(robot_footprints)

        # Merge all footprints using shapely's unary_union
        return custom_merge(all_footprints).simplify(0.3)

    def get_frontiers(self):
        '''
        Merge global map and extract frontiers
        '''
        self.merged_global_map = self.merge_sensing_footprints()
        if isinstance(self.merged_global_map, Polygon):
            boundaries = [self.merged_global_map.exterior]
            interiors = [self.merged_global_map.interiors]
        elif isinstance(self.merged_global_map, MultiPolygon):
            boundaries = [poly.exterior for poly in self.merged_global_map.geoms]
            interiors = [poly.interiors for poly in self.merged_global_map.geoms]
        else:
            raise ValueError(f"Unexpected type for merged_global_map: {type(self.merged_global_map)}")
        
        frontiers = []
        for boundary in boundaries:
            coords = np.array(boundary.coords)
            frontiers.extend(self.interpolate_and_filter_frontier(coords))
        for interior in interiors:
            for hole in interior:
                coords = np.array(hole.coords)
                frontiers.extend(self.interpolate_and_filter_frontier(coords))

        if not frontiers or len(frontiers) == 1: # a single frontier point is negligible
            return LineString()
        return LineString(frontiers)  # Return frontiers as a LineString

    def interpolate_and_filter_frontier(self, coords, step=0.3):
        '''
        @description: Vectorized interpolation between points and checking if interpolated points are valid frontiers
        @note: shapely object's exterior only contains the boundary points, so we need to interpolate between them
        '''
        # Calculate distances between consecutive points
        diff = np.diff(coords, axis=0)
        distances = np.sqrt((diff**2).sum(axis=1))
        
        # Calculate number of steps for each segment
        steps = np.maximum(np.ceil(distances / step).astype(int), 1)
        
        # Generate interpolation parameters
        cum_steps = np.cumsum(steps)
        total_steps = cum_steps[-1]
        
        indices = np.arange(total_steps)
        segment_indices = np.searchsorted(cum_steps, indices, side='right')
        
        # Correct calculation of segment_steps
        segment_starts = np.concatenate(([0], cum_steps[:-1]))
        segment_steps = indices - segment_starts[segment_indices]
        
        # Ensure alphas are calculated correctly
        steps_broadcast = steps[segment_indices]
        alphas = segment_steps / steps_broadcast

        # Interpolate points
        start_points = coords[segment_indices]
        end_points = coords[segment_indices + 1]
        
        interpolated_points = start_points + alphas[:, np.newaxis] * (end_points - start_points)
        
        # Check validity of interpolated points
        valid_mask = np.array([self.env_workspace.contains(Point(p)) and not self.env_obstacles.contains(Point(p))
                            for p in interpolated_points])
        
        return interpolated_points[valid_mask]

    def explore(self, max_steps=None):
        '''
        Main exploration loop with goal-based updates
        '''
        self.failed = False
        self.last_collision_info = None
        self.last_termination_reason = None
        self.last_step_count = 0
        self.last_sim_time = 0.0
        self.last_success = False
        self._termination_marker_drawn = False
        self._force_reassign_agents = set()
        self._setup_deadlock_monitor()
        self.frontiers = self.get_frontiers()  # initially get frontiers
        self.last_coverage_ratio = self.get_coverage_ratio()
        if self.last_coverage_ratio >= self.coverage_target:
            self._finalize_animation()
            self.last_termination_reason = 'coverage_target_reached'
            self.last_success = True
            print(f"Exploration complete! Coverage={self.last_coverage_ratio:.3f}")
            return True
        if self.exploration_complete():
            self.last_termination_reason = 'frontiers_exhausted_initial'
            print(f"Exploration frontiers exhausted, but coverage={self.last_coverage_ratio:.3f} < target={self.coverage_target:.3f}.")
            self._finalize_failure()
            return False

        self.update_all_goals()  # Initial goal assignment for all robots
        if self.global_goals is None:
            self.last_coverage_ratio = self.get_coverage_ratio()
            if self.last_coverage_ratio >= self.coverage_target:
                self._finalize_animation()
                self.last_termination_reason = 'coverage_target_reached_no_goals'
                self.last_success = True
                print(f"Exploration complete! Coverage={self.last_coverage_ratio:.3f}")
                return True
            self.last_termination_reason = 'no_initial_goals'
            print(f"Exploration has no new goals, but coverage={self.last_coverage_ratio:.3f} < target={self.coverage_target:.3f}.")
            self._finalize_failure()
            return False

        if self.show_animation:
            self.update_visualization()

        step_count = 0
        while True:
            if max_steps is not None and step_count >= max_steps:
                self.last_step_count = int(step_count)
                self.last_sim_time = float(step_count * self.dt)
                self.last_termination_reason = 'max_steps'
                print(
                    f"Exploration stopped after reaching max_steps={max_steps}. "
                    f"Coverage={self.last_coverage_ratio:.3f}."
                )
                self._finalize_failure()
                return False

            robots_reached_goals = self.move_robots()
            step_count += 1
            self.last_step_count = int(step_count)
            self.last_sim_time = float(step_count * self.dt)

            if self.failed:
                self.last_termination_reason = 'collision_or_infeasible'
                print(
                    "Exploration failed: collision or infeasible optimization. "
                    f"Coverage={self.last_coverage_ratio:.3f}."
                )
                marker_targets = None
                if isinstance(self.last_collision_info, dict):
                    if "robot_idx" in self.last_collision_info:
                        marker_targets = [int(self.last_collision_info["robot_idx"])]
                    elif self.last_collision_info.get("type") == "inter_agent":
                        pair = self.last_collision_info.get("pair", [])
                        marker_targets = [int(p) for p in pair]
                self._finalize_failure(target_robot_ids=marker_targets)
                return False

            refresh_map = any(robots_reached_goals) or (step_count % 15 == 0)
            if refresh_map:
                self.frontiers = self.get_frontiers()
                self.last_coverage_ratio = self.get_coverage_ratio()
                if self.last_coverage_ratio >= self.coverage_target:
                    self._finalize_animation()
                    self.last_termination_reason = 'coverage_target_reached'
                    self.last_success = True
                    print(f"Exploration complete! Coverage={self.last_coverage_ratio:.3f}")
                    return True
                if self.exploration_complete():
                    self.last_termination_reason = 'frontiers_exhausted'
                    print(
                        f"Exploration frontiers exhausted, but coverage={self.last_coverage_ratio:.3f} "
                        f"< target={self.coverage_target:.3f}."
                    )
                    self._finalize_failure()
                    return False

            if any(robots_reached_goals):
                self.update_goals_for_completed(robots_reached_goals)

            if self.show_animation:
                self.update_visualization()

    def update_all_goals(self):
        '''
        Initially assign goals to all robots (later, goal assignment is asychronous)
        '''
        self.global_goals = self.update_global_goals()
        if self.global_goals is None:
            self.robot_goals = [None] * self.num_robot
            return
        for i, controller in enumerate(self.controller_list):
            self.robot_goals[i] = self.global_goals[i]
            if self.use_astar_waypoints:
                waypoints = self._build_waypoints_for_robot(i, self.global_goals[i])
            else:
                waypoints = np.array([self.global_goals[i][:2]], dtype=float)
            controller.set_waypoints(waypoints)
        
        if self.show_animation:
            self.global_goals_scatter.set_offsets(self.global_goals)

    def _extract_planar_velocity(self, controller):
        robot = controller.robot
        model = str(controller.robot_spec.get('model', ''))
        x = np.asarray(robot.X, dtype=float).reshape(-1)

        if model == 'DoubleIntegrator2D' and x.shape[0] >= 4:
            return np.array([x[2], x[3]], dtype=float)
        if model == 'SingleIntegrator2D':
            u = np.asarray(robot.U, dtype=float).reshape(-1)
            if u.shape[0] >= 2:
                return np.array([u[0], u[1]], dtype=float)
            return np.zeros(2, dtype=float)
        if model in ['Unicycle2D', 'DynamicUnicycle2D', 'KinematicBicycle2D', 'KinematicBicycle2D_C3BF', 'KinematicBicycle2D_DPCBF']:
            yaw = float(robot.get_orientation())
            if model == 'Unicycle2D':
                u = np.asarray(robot.U, dtype=float).reshape(-1)
                speed = float(u[0]) if u.shape[0] > 0 else 0.0
            else:
                speed = float(x[3]) if x.shape[0] > 3 else 0.0
            return np.array([speed * np.cos(yaw), speed * np.sin(yaw)], dtype=float)
        return np.zeros(2, dtype=float)

    def _build_agent_state_snapshot(self):
        snapshot = []
        for controller in self.controller_list:
            pos = np.asarray(controller.robot.get_position(), dtype=float).reshape(-1)
            vel = self._extract_planar_velocity(controller)
            radius = float(controller.robot.robot_radius)
            snapshot.append([pos[0], pos[1], vel[0], vel[1], radius])
        if len(snapshot) == 0:
            return np.empty((0, 5), dtype=float)
        return np.array(snapshot, dtype=float)

    def _check_inter_agent_collision(self):
        if self.num_robot < 2:
            return None
        for i in range(self.num_robot):
            pi = np.asarray(self.controller_list[i].robot.get_position(), dtype=float).reshape(-1)
            ri = float(self.controller_list[i].robot.robot_radius)
            for j in range(i + 1, self.num_robot):
                pj = np.asarray(self.controller_list[j].robot.get_position(), dtype=float).reshape(-1)
                rj = float(self.controller_list[j].robot.robot_radius)
                dist = float(np.linalg.norm(pi - pj))
                if dist < (ri + rj):
                    return i, j, dist, (ri + rj)
        return None

    def move_robots(self):
        '''
        Move all robots and return a list indicating which robots have reached their goals
        '''
        robots_reached_goals = [False] * self.num_robot
        self.last_step_status = [0] * self.num_robot
        agent_snapshot = self._build_agent_state_snapshot()
        for i, controller in enumerate(self.controller_list):
            self._tick_deadlock_exclusions(i)
            if hasattr(controller, 'set_other_agents'):
                if agent_snapshot.shape[0] > 0:
                    other_agents = np.delete(agent_snapshot, i, axis=0)
                else:
                    other_agents = np.empty((0, 5), dtype=float)
                controller.set_other_agents(other_agents)
            pre_detected_unknown = np.array(
                getattr(controller.robot, 'detected_unknown_obs_memory', np.empty((0, 7))),
                dtype=float,
                copy=True,
            )
            step_status = controller.control_step()
            self.last_step_status[i] = step_status
            if step_status == -2:
                self.failed = True
                self._record_collision_info(i, controller, pre_detected_unknown)
            has_reached_goal = controller.has_reached_goal()
            if has_reached_goal:
                robots_reached_goals[i] = True
                self._reset_deadlock_history(i)
                if hasattr(self, '_deadlock_exclusions'):
                    self._deadlock_exclusions[i] = []
            elif self._mark_agent_deadlock_if_needed(i, step_status=step_status, has_reached_goal=has_reached_goal):
                if self._deadlock_recovery_count[i] > self.deadlock_max_recoveries:
                    # Do not terminate as collision/infeasible for deadlock;
                    # keep trying recovery and let max_steps decide timeout.
                    print(
                        f"Deadlock saturation: robot={i}, recoveries={self._deadlock_recovery_count[i]}. "
                        "Continuing with fresh reassignment."
                    )
                    self._deadlock_recovery_count[i] = 0
                    self._deadlock_cooldown[i] = self.deadlock_cooldown_steps
                robots_reached_goals[i] = True
                self._force_reassign_agents.add(i)

        inter_agent_collision = self._check_inter_agent_collision()
        if inter_agent_collision is not None:
            i, j, dist, threshold = inter_agent_collision
            self.failed = True
            # Both robots need status -2, not just self.failed/last_collision_info:
            # last_step_status is what ExploreAdapter.plan_step() actually watches
            # to trigger the fall-and-stay crash animation and collision count --
            # without this, a robot-robot hit was detected and logged here but
            # never visibly registered (no fall, no crash count) in the Isaac
            # Sim recording pipeline.
            self.last_step_status[i] = -2
            self.last_step_status[j] = -2
            self.last_collision_info = {
                'type': 'inter_agent',
                'pair': [int(i), int(j)],
                'distance': float(dist),
                'threshold': float(threshold),
            }
            print(
                f"Inter-agent collision detected between robot {i} and {j}: "
                f"distance={dist:.3f}, threshold={threshold:.3f}"
            )
        return robots_reached_goals

    def _record_collision_info(self, robot_idx, controller, pre_detected_unknown):
        robot_pos = np.asarray(controller.robot.get_position(), dtype=float).reshape(-1)
        robot_radius = float(controller.robot.robot_radius)

        unknown_idx = None
        unknown_obs = None
        for idx, obs in enumerate(self.unknown_obs):
            if np.linalg.norm(robot_pos - obs[:2]) < (robot_radius + obs[2]):
                unknown_idx = idx
                unknown_obs = obs
                break

        if unknown_obs is None:
            self.last_collision_info = {
                'robot_idx': int(robot_idx),
                'type': 'known_or_infeasible',
            }
            return

        post_detected_unknown = np.array(
            getattr(controller.robot, 'detected_unknown_obs_memory', np.empty((0, 7))),
            dtype=float,
            copy=True,
        )
        detected_pre = self._obs_in_memory(pre_detected_unknown, unknown_obs)
        detected_post = self._obs_in_memory(post_detected_unknown, unknown_obs)
        if detected_pre:
            stage = 'already_detected_before_collision'
        elif detected_post:
            stage = 'detected_too_late_same_step'
        else:
            stage = 'not_detected_before_collision'

        self.last_collision_info = {
            'robot_idx': int(robot_idx),
            'type': 'unknown',
            'obs_idx': int(unknown_idx),
            'stage': stage,
            'obs': unknown_obs[:3].tolist(),
            'robot_pos': robot_pos[:2].tolist(),
        }
        print(
            f"Unknown collision analysis: robot={robot_idx}, obs_idx={unknown_idx}, "
            f"stage={stage}, obs={unknown_obs[:3]}, pos={robot_pos[:2]}"
        )

    @staticmethod
    def _obs_in_memory(obs_memory, target_obs, pos_tol=1e-3, radius_tol=1e-2):
        if obs_memory is None:
            return False
        mem = np.array(obs_memory, dtype=float)
        if mem.size == 0:
            return False
        if mem.ndim == 1:
            mem = mem.reshape(1, -1)
        if mem.shape[1] < 3:
            return False
        center_diff = np.linalg.norm(mem[:, :2] - target_obs[:2], axis=1)
        radius_diff = np.abs(mem[:, 2] - target_obs[2])
        return bool(np.any((center_diff <= pos_tol) & (radius_diff <= radius_tol)))

    def update_goals_for_completed(self, robots_reached_goals):
        '''
        Asynchronously update goals only for robots that have reached their goals
        '''
        new_global_goals = self.update_global_goals()
        self.global_goals = new_global_goals
        if new_global_goals is not None:
            for i, reached in enumerate(robots_reached_goals):
                if reached: #only update goals with goal-reached robots
                    was_forced_reassign = i in self._force_reassign_agents
                    goal_i = np.array(new_global_goals[i], dtype=float)
                    if was_forced_reassign:
                        recovery_goal = self._select_recovery_goal(i, avoid_goal=goal_i)
                        if recovery_goal is not None:
                            goal_i = recovery_goal
                    self._force_reassign_agents.discard(i)
                    self.robot_goals[i] = goal_i
                    if self.use_astar_waypoints:
                        waypoints = self._build_waypoints_for_robot(i, goal_i)
                    else:
                        waypoints = np.array([goal_i[:2]], dtype=float)
                    self.controller_list[i].set_waypoints(waypoints)
                    self._reset_deadlock_history(i)
                    if not was_forced_reassign:
                        self._deadlock_recovery_count[i] = 0
            
            if self.show_animation:
                self.global_goals_scatter.set_offsets(self.robot_goals)

    def _select_recovery_goal(self, robot_idx, avoid_goal=None):
        obstacle_map = self.latest_obstacle_map if self.latest_obstacle_map is not None else self.get_obstacle_map()
        frontier_map = self.get_frontier_map()
        frontier_idx = np.argwhere(frontier_map == 1)  # [y, x]
        if frontier_idx.size == 0:
            return None

        frontier_xy = np.column_stack((frontier_idx[:, 1], frontier_idx[:, 0])).astype(np.int32)
        agent_positions = self.get_robot_positions()
        start_xy = agent_positions[robot_idx][:2].astype(int)
        distances = np.linalg.norm(frontier_xy - start_xy.reshape(1, 2), axis=1)

        avoid_grid = None
        if avoid_goal is not None:
            avoid_grid = self.env_handler.f_to_grid(np.asarray(avoid_goal, dtype=float).reshape(1, -1)[:, :2])[0][:2]
        min_separation_cells = max(int(np.ceil(2.2 / max(self.env_handler.resolution, 1e-3))), 7)

        deadlock_exclusions = []
        if hasattr(self, '_deadlock_exclusions') and 0 <= robot_idx < len(self._deadlock_exclusions):
            deadlock_exclusions = self._deadlock_exclusions[robot_idx]
        exclusion_xy = (
            np.array([np.array(entry['xy'], dtype=float).reshape(2) for entry in deadlock_exclusions], dtype=float)
            if len(deadlock_exclusions) > 0
            else np.empty((0, 2), dtype=float)
        )
        exclusion_radius = float(getattr(self, 'deadlock_exclusion_radius_cells', 0))

        if avoid_grid is not None:
            avoid_dist = np.linalg.norm(
                frontier_xy.astype(float) - np.array(avoid_grid, dtype=float).reshape(1, 2), axis=1
            )
            if exclusion_xy.size > 0:
                exclusion_dist = np.min(
                    np.linalg.norm(
                        frontier_xy[:, None, :].astype(float) - exclusion_xy[None, :, :],
                        axis=2,
                    ),
                    axis=1,
                )
                # Prefer alternatives far from both planner goal and deadlocked goals.
                score = 0.50 * avoid_dist + 0.25 * distances + 0.25 * exclusion_dist
            else:
                # Prefer far-away alternatives to escape local deadlock basins.
                score = 0.75 * avoid_dist + 0.25 * distances
            order = np.argsort(-score)
        elif exclusion_xy.size > 0:
            exclusion_dist = np.min(
                np.linalg.norm(
                    frontier_xy[:, None, :].astype(float) - exclusion_xy[None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            score = 0.65 * exclusion_dist + 0.35 * distances
            order = np.argsort(-score)
        else:
            order = np.argsort(-distances)

        for idx in order[:350]:
            x = int(frontier_xy[idx, 0])
            y = int(frontier_xy[idx, 1])
            if obstacle_map[y, x] == 1:
                continue
            if exclusion_xy.size > 0 and exclusion_radius > 0.0:
                d_exclusion = np.min(
                    np.linalg.norm(
                        exclusion_xy - np.array([x, y], dtype=float).reshape(1, 2),
                        axis=1,
                    )
                )
                if d_exclusion < exclusion_radius:
                    continue
            if avoid_grid is not None:
                if np.linalg.norm(np.array([x, y], dtype=float) - np.array(avoid_grid, dtype=float)) < min_separation_cells:
                    continue
            if self.use_astar_waypoints:
                path = self._astar_grid(obstacle_map, (int(start_xy[0]), int(start_xy[1])), (x, y))
                if path is None or len(path) < 2:
                    continue
            return self.env_handler.grid_to_f(np.array([[x, y]], dtype=np.int32))[0]
        return None

    def update_global_goals(self):
        np_obstacle_map = self.get_obstacle_map()
        self.latest_obstacle_map = np_obstacle_map
        np_frontier_map = self.get_frontier_map()
        #cv2.imshow('frontier', (np_frontier_map * 255).astype(np.uint8))
        #cv2.imshow('obstacle', (np_obstacle_map * 255).astype(np.uint8))
        #cv2.waitKey(10)

        agent_positions = self.get_robot_positions()
        agent_orientations = self.get_robot_orientations()
        global_goals = self.exploration_algorithm.get_long_term_goals(np_obstacle_map, np_frontier_map, agent_positions, agent_orientations)
        # if any of the goals is None, return None for all goals (exploration is complete)
        if any(goal is None for goal in global_goals):
            return None
        global_goals = np.array(global_goals, dtype=np.int32)
        global_goals = self._adjust_goals_for_progress(
            global_goals,
            agent_positions,
            np_frontier_map,
            np_obstacle_map,
        )
        return self.env_handler.grid_to_f(global_goals)

    def _adjust_goals_for_progress(self, global_goals, agent_positions, frontier_map, obstacle_map):
        adjusted_goals = np.array(global_goals, dtype=np.int32, copy=True)
        resolution = float(self.env_handler.resolution)
        for i in range(self.num_robot):
            cur = agent_positions[i][:2].astype(int)
            goal = adjusted_goals[i]
            reached_threshold = float(self.robot_specs[i].get('reached_threshold', 0.3))
            requested_min_goal_distance = float(
                self.robot_specs[i].get(
                    'min_goal_distance',
                    max(1.4 * reached_threshold, 0.9),
                )
            )
            # Prevent near-goal deadlock (especially in CoScan), but avoid
            # aggressive far-goal forcing that causes cross-map assignment swaps.
            required_progress_distance = min(
                requested_min_goal_distance,
                max(reached_threshold + 0.25, 1.1 * reached_threshold),
            )
            reassignment_cells = max(int(np.ceil(required_progress_distance / resolution)), 2)

            if np.linalg.norm(goal - cur) >= reassignment_cells:
                continue

            replacement = self._select_distant_reachable_frontier(
                start_xy=(int(cur[0]), int(cur[1])),
                frontier_map=frontier_map,
                obstacle_map=obstacle_map,
                min_dist_cells=reassignment_cells,
                preferred_xy=(int(goal[0]), int(goal[1])),
            )
            if replacement is not None:
                adjusted_goals[i] = replacement
        return adjusted_goals

    def _select_distant_reachable_frontier(self, start_xy, frontier_map, obstacle_map, min_dist_cells, preferred_xy=None):
        frontier_idx = np.argwhere(frontier_map == 1)  # [y, x]
        if frontier_idx.size == 0:
            return None

        frontier_xy = np.column_stack((frontier_idx[:, 1], frontier_idx[:, 0])).astype(np.int32)
        distances = np.linalg.norm(frontier_xy - np.array(start_xy, dtype=float), axis=1)
        valid_mask = distances >= float(min_dist_cells)
        if np.any(valid_mask):
            candidates = frontier_xy[valid_mask]
            cand_dist = distances[valid_mask]
            min_path_len = int(max(min_dist_cells, 2))
            if preferred_xy is not None:
                preferred_xy = np.array(preferred_xy, dtype=float).reshape(1, 2)
                preferred_dist = np.linalg.norm(candidates.astype(float) - preferred_xy, axis=1)
                # Favor candidates close to the original planner assignment, then shorter travel.
                score = preferred_dist + 0.15 * cand_dist
                order = np.argsort(score)
            else:
                order = np.argsort(cand_dist)
        else:
            # Startup and narrow-corridor fallback: pick the farthest reachable
            # frontier even if none satisfy the preferred distance.
            candidates = frontier_xy
            cand_dist = distances
            min_path_len = 2
            order = np.argsort(-cand_dist)
        max_checks = min(200, order.shape[0])

        for idx in order[:max_checks]:
            x, y = int(candidates[idx, 0]), int(candidates[idx, 1])
            if obstacle_map[y, x] == 1:
                continue
            if self.use_astar_waypoints:
                path = self._astar_grid(obstacle_map, start_xy, (x, y))
                if path is None or len(path) < min_path_len:
                    continue
            return np.array([x, y], dtype=np.int32)
        return None

    def _build_waypoints_for_robot(self, robot_idx, goal_f, waypoint_stride_cells=None):
        goal_f = np.asarray(goal_f, dtype=float).reshape(-1)
        if goal_f.size < 2:
            return np.array([goal_f[:2]], dtype=float)

        obstacle_map = self.latest_obstacle_map
        if obstacle_map is None:
            return np.array([goal_f[:2]], dtype=float)

        start_f = np.asarray(self.controller_list[robot_idx].robot.get_position(), dtype=float).reshape(-1)
        start_grid = self.env_handler.f_to_grid(start_f[:2])
        goal_grid = self.env_handler.f_to_grid(goal_f[:2])

        height, width = obstacle_map.shape
        sx = int(np.clip(start_grid[0], 0, width - 1))
        sy = int(np.clip(start_grid[1], 0, height - 1))
        gx = int(np.clip(goal_grid[0], 0, width - 1))
        gy = int(np.clip(goal_grid[1], 0, height - 1))

        if obstacle_map[gy, gx] == 1 or obstacle_map[sy, sx] == 1:
            return np.array([goal_f[:2]], dtype=float)

        path = self._astar_grid(obstacle_map, (sx, sy), (gx, gy))
        if path is None or len(path) < 2:
            return np.array([goal_f[:2]], dtype=float)

        path_grid = np.array(path, dtype=int)
        path_f = self.env_handler.grid_to_f(path_grid)[:, :2]
        if path_f.shape[0] < 2:
            return np.array([goal_f[:2]], dtype=float)

        reached_threshold = float(self.robot_specs[robot_idx].get('reached_threshold', 0.3))
        if waypoint_stride_cells is None:
            waypoint_spacing = max(1.35 * reached_threshold, 1.2)
        else:
            waypoint_spacing = max(float(waypoint_stride_cells) * self.env_handler.resolution, 1e-3)

        segment_lengths = np.linalg.norm(np.diff(path_f, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        sample_distances = np.arange(waypoint_spacing, cumulative[-1], waypoint_spacing)

        sample_indices = []
        for dist in sample_distances:
            idx = int(np.searchsorted(cumulative, dist, side='left'))
            idx = min(max(idx, 1), len(path_f) - 1)
            if len(sample_indices) == 0 or sample_indices[-1] != idx:
                sample_indices.append(idx)

        if len(sample_indices) == 0 or sample_indices[-1] != (len(path_f) - 1):
            sample_indices.append(len(path_f) - 1)

        sampled_f = path_f[np.array(sample_indices, dtype=int)]
        if sampled_f.size == 0:
            return np.array([goal_f[:2]], dtype=float)
        return sampled_f

    @staticmethod
    def _astar_grid(obstacle_map, start, goal):
        if start == goal:
            return [start]

        height, width = obstacle_map.shape
        moves = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, np.sqrt(2.0)), (1, -1, np.sqrt(2.0)),
            (-1, 1, np.sqrt(2.0)), (-1, -1, np.sqrt(2.0)),
        ]

        def heuristic(node):
            return np.hypot(node[0] - goal[0], node[1] - goal[1])

        open_heap = []
        heapq.heappush(open_heap, (heuristic(start), 0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        visited = set()

        while open_heap:
            _, current_g, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            cx, cy = current
            for dx, dy, move_cost in moves:
                nx = cx + dx
                ny = cy + dy
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                if obstacle_map[ny, nx] == 1:
                    continue
                neighbor = (nx, ny)
                tentative_g = current_g + move_cost
                if tentative_g >= g_score.get(neighbor, np.inf):
                    continue
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                heapq.heappush(open_heap, (tentative_g + heuristic(neighbor), tentative_g, neighbor))

        return None

    def get_coverage_ratio(self):
        if self.merged_global_map is None or self.merged_global_map.is_empty:
            return 0.0

        observed = self.merged_global_map.intersection(self.env_workspace)
        if not self.env_obstacles.is_empty:
            observed = observed.difference(self.env_obstacles)

        coverage_ratio = float(observed.area) / self.free_space_area
        return float(np.clip(coverage_ratio, 0.0, 1.0))

    def _compute_static_obstacle_map(self):
        obstacle_map = np.zeros(self.env_handler.get_map_shape(), dtype=np.int8)
        if self.env_obstacles.is_empty:
            return obstacle_map

        height, width = self.env_handler.get_map_shape()
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        grid_points = np.column_stack((x.ravel(), y.ravel()))
        cont_points = self.env_handler.grid_to_f(grid_points)

        contains_mask = np.fromiter(
            (self.env_obstacles.contains(Point(p)) for p in cont_points),
            dtype=np.bool_,
            count=cont_points.shape[0],
        )
        return contains_mask.reshape(height, width).astype(np.int8)

    def get_obstacle_map(self):
        return self.static_obstacle_map.copy()
    
    def get_frontier_map(self):
        frontier_map = np.zeros(self.env_handler.get_map_shape(), dtype=np.int8)
        if len(self.frontiers.coords) == 0:
            return frontier_map

        frontier_points = np.array(self.frontiers.coords)
        grid_points = self.env_handler.f_to_grid(frontier_points)
        
        # Ensure all points are within the map bounds
        height, width = self.env_handler.get_map_shape()
        mask = (grid_points[:, 0] >= 0) & (grid_points[:, 0] < width) & \
               (grid_points[:, 1] >= 0) & (grid_points[:, 1] < height)
        grid_points = grid_points[mask]
        
        # Set frontier points to 1 in the frontier map
        frontier_map[grid_points[:, 1], grid_points[:, 0]] = 1
        return frontier_map

    def get_robot_positions(self):
        positions = np.array([controller.robot.get_position() for controller in self.controller_list])
        return self.env_handler.f_to_grid(positions)
    
    def get_robot_orientations(self):
        orientations = np.array([controller.robot.get_orientation() for controller in self.controller_list])
        return orientations

    def update_visualization(self):
        '''
        Update visualization elements
        '''
        if len(self.frontiers.coords) > 0:
            self.frontiers_scatter.set_offsets(np.array(self.frontiers.coords))
        else:
            self.frontiers_scatter.set_offsets(np.empty((0, 2)))
        self.controller_list[0].draw_plot()

    def exploration_complete(self):
        return len(self.frontiers.coords) == 0
        

def main():
    dt = 0.1

    # temporal
    waypoints = [
        [2, 2, math.pi/2],
        [2, 12, 0],
        [10, 12, 0],
        [15, 2, math.pi/2]
    ]
    waypoints = np.array(waypoints, dtype=np.float64)
    x_init = waypoints[0]
    x_init2 = waypoints[-1]

    # define as much robot specs as you want
    robot_spec_1 = {
        'model': 'DoubleIntegrator2D',
        'sensor': 'rgbd',
        'cam_range': 5.0,
        'reached_threshold': 2.0,
        'exploration': True
    }
    robot_spec_2 = {
        'model': 'DoubleIntegrator2D',
        'sensor': 'rgbd',
        'cam_range': 7.0,
        'reached_threshold': 2.0,
        'exploration': True
    }

    robot_specs = [robot_spec_1, robot_spec_2]
    for i, robot_spec in enumerate(robot_specs):
        robot_spec['robot_id'] = i

    controller_type = {
        'pos': 'mpc_cbf',
        'att': 'gatekeeper'
    }

    exploration = ExplorationManager([x_init, x_init2], robot_specs, controller_type,
                                    exploration_algorithm='CoScan',
                                     dt=dt,
                                    show_animation=True,
                                    save_animation=False)
    exploration.explore()
    # to check the set_waypoints function
        # exploration.set_waypoints(0, waypoints)
        # exploration.set_waypoints(1, waypoints[::-1])
        # for i in range(1000):
        #     exploration.control_step()
    

if __name__ == "__main__":
    from utils import plotting
    from utils import env
    import math
    
    main()

    
