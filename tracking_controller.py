import glob
import os
import re
import shutil
import subprocess

import numpy as np
from shapely.geometry import Point

from safe_control.tracking import LocalTrackingController as _BaseLocalTrackingController


class LocalTrackingController(_BaseLocalTrackingController):
    _DYNAMIC_OBS_FLAG = 2.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.other_agents = np.empty((0, 5), dtype=float)
        self._show_visibility_violations = bool(
            self.robot_spec.get('show_visibility_violations', True)
        )
        if not self._show_visibility_violations and getattr(self.robot, 'unsafe_points_handle', None) is not None:
            self.robot.unsafe_points_handle.set_visible(False)

        # Use local MPC-CBF extension that supports dynamic-obstacle CBF terms.
        if self.pos_controller_type == 'mpc_cbf':
            from position_controller import SeamlisMPCCBF

            self.pos_controller = SeamlisMPCCBF(
                self.robot,
                self.robot_spec,
                show_mpc_traj=self.show_mpc_traj,
                num_obs=self.num_constraints,
            )
            if self.att_controller_type == 'gatekeeper' and self.att_controller is not None:
                if hasattr(self.att_controller, 'setup_pos_controller'):
                    self.att_controller.setup_pos_controller(self.pos_controller)

    def set_other_agents(self, other_agents):
        other_agents = np.array(other_agents, dtype=float)
        if other_agents.size == 0:
            self.other_agents = np.empty((0, 5), dtype=float)
            return
        if other_agents.ndim == 1:
            other_agents = other_agents.reshape(1, -1)
        if other_agents.shape[1] > 5:
            other_agents = other_agents[:, :5]
        elif other_agents.shape[1] < 5:
            pad = np.zeros((other_agents.shape[0], 5 - other_agents.shape[1]), dtype=float)
            other_agents = np.hstack((other_agents, pad))
        self.other_agents = other_agents

    def _build_inter_agent_dynamic_obs(self):
        if self.other_agents.size == 0:
            return np.empty((0, 7), dtype=float)

        robot_pos = np.asarray(self.robot.get_position(), dtype=float).reshape(-1)
        agent_margin = float(self.robot_spec.get('inter_agent_margin', 0.08))
        max_range = float(self.robot_spec.get('inter_agent_obs_range', 8.0))
        obs_flag = self._DYNAMIC_OBS_FLAG if self.pos_controller_type == 'mpc_cbf' else 0.0

        obs_rows = []
        for agent in self.other_agents:
            ox, oy, vx, vy, other_radius = agent[:5]
            dx = ox - robot_pos[0]
            dy = oy - robot_pos[1]
            if np.hypot(dx, dy) > max_range:
                continue

            obs_rows.append([
                float(ox),
                float(oy),
                float(max(other_radius + agent_margin, 1e-3)),
                float(vx),
                float(vy),
                0.0,
                obs_flag,
            ])

        if len(obs_rows) == 0:
            return np.empty((0, 7), dtype=float)
        return np.array(obs_rows, dtype=float)

    def update_goal(self):
        if self.robot_spec['model'] in ['Quad3D']:
            n_pos = 3
        else:
            n_pos = 2

        # Disable rotate/stop state-machine branches in seamlis.
        # Yaw should always be produced by the active attitude controller.
        if self.state_machine in ['rotate', 'stop']:
            self.state_machine = 'track'
            self.u_att = None

        if self.current_goal_index >= len(self.waypoints):
            return None

        if self.goal_reached(self.robot.X, np.array(self.waypoints[self.current_goal_index]).reshape(-1, 1)):
            self.current_goal_index += 1
            if self.current_goal_index >= len(self.waypoints):
                self.state_machine = 'idle'
                return None

        goal = np.array(self.waypoints[self.current_goal_index][0:n_pos])
        return goal

    def set_waypoints(self, waypoints):
        if type(waypoints) == list:
            waypoints = np.array(waypoints, dtype=float)
        waypoints = np.array(waypoints, dtype=float)
        if waypoints.ndim == 1 and waypoints.size > 0:
            waypoints = waypoints.reshape(1, -1)

        filtered_waypoints = self.filter_waypoints(waypoints)
        # Guard against exploration deadlock: filtering can remove all
        # waypoints when spacing is below reached_threshold.
        if filtered_waypoints.size == 0 and waypoints.size > 0:
            if waypoints.ndim == 2 and waypoints.shape[1] >= 2:
                robot_pos = self.robot.get_position()
                distances = np.linalg.norm(waypoints[:, :2] - robot_pos.reshape(1, -1), axis=1)
                keep_idx = int(np.argmax(distances))
                filtered_waypoints = waypoints[keep_idx:keep_idx + 1]
            else:
                filtered_waypoints = waypoints[-1:].copy()

        self.waypoints = filtered_waypoints
        self.current_goal_index = 0

        self.goal = self.update_goal()
        if self.goal is not None:
            self.state_machine = 'track'
        else:
            self.state_machine = 'idle'
            self.u_att = None

        if self.show_animation:
            self.waypoints_scatter.set_offsets(self.waypoints[:, :2])

    def get_nearest_unpassed_obs(self, detected_obs, angle_unpassed=np.pi * 2, obs_num=5):
        def angle_normalize(x):
            return (((x + np.pi) % (2 * np.pi)) - np.pi)

        if self.robot_spec['model'] in ['SingleIntegrator2D', 'DoubleIntegrator2D', 'Quad2D', 'Quad3D']:
            angle_unpassed = np.pi * 2
        elif self.robot_spec['model'] in ['Unicycle2D', 'DynamicUnicycle2D', 'VTOL2D']:
            angle_unpassed = np.pi * 1.2
        elif 'KinematicBicycle2D' in self.robot_spec['model']:
            angle_unpassed = np.pi * 2.0

        detected_arr = np.array([], dtype=float)
        if detected_obs is not None and len(detected_obs) != 0:
            detected_arr = np.array(detected_obs, dtype=float)
            if detected_arr.ndim == 1:
                detected_arr = detected_arr.reshape(1, -1)
            # Slightly inflate detected-unknown radius inside the CBF layer to
            # counter discretization / solver tolerance effects.
            unknown_margin = float(
                self.robot_spec.get(
                    'unknown_obs_cbf_margin',
                    max(0.08, 0.5 * float(self.robot.robot_radius)),
                )
            )
            detected_arr = detected_arr.copy()
            detected_arr[:, 2] = detected_arr[:, 2] + unknown_margin

        dynamic_agents = self._build_inter_agent_dynamic_obs()

        obs_blocks = []
        mask_blocks = []

        if len(self.obs) != 0:
            known_obs = np.array(self.obs, dtype=float)
            if known_obs.ndim == 1:
                known_obs = known_obs.reshape(1, -1)
            obs_blocks.append(known_obs)
            mask_blocks.append(np.zeros(known_obs.shape[0], dtype=bool))

        if detected_arr.size != 0:
            obs_blocks.append(detected_arr)
            mask_blocks.append(np.ones(detected_arr.shape[0], dtype=bool))

        if dynamic_agents.size != 0:
            obs_blocks.append(dynamic_agents)
            mask_blocks.append(np.zeros(dynamic_agents.shape[0], dtype=bool))

        if len(obs_blocks) == 0:
            return None

        all_obs = np.vstack(obs_blocks)
        unknown_mask = np.concatenate(mask_blocks)

        if all_obs.ndim == 1:
            all_obs = all_obs.reshape(1, -1)

        # Inflate obstacle size used by CBF constraints to absorb solver and
        # discretization errors near obstacle boundaries.
        obs_margin = float(self.robot_spec.get('known_obs_cbf_margin', 0.08))
        if obs_margin > 0.0:
            all_obs = all_obs.copy()
            shape_flag = all_obs[:, 6] if all_obs.shape[1] > 6 else np.zeros(all_obs.shape[0])
            is_super = np.isclose(shape_flag, 1.0)
            # Circle-like obstacles: radius field a/r at column 2.
            all_obs[~is_super, 2] = all_obs[~is_super, 2] + obs_margin
            # Superellipse obstacles: inflate both semi-axes a and b.
            if np.any(is_super):
                all_obs[is_super, 2] = all_obs[is_super, 2] + obs_margin
                if all_obs.shape[1] > 3:
                    all_obs[is_super, 3] = all_obs[is_super, 3] + obs_margin

        robot_pos = self.robot.get_position()
        robot_yaw = self.robot.get_orientation()
        to_obs_vectors = all_obs[:, :2] - robot_pos.reshape(1, -1)
        distances = np.linalg.norm(to_obs_vectors, axis=1)
        radii = all_obs[:, 2]
        clearances = distances - (radii + float(self.robot.robot_radius))

        angle_to_obs = np.arctan2(to_obs_vectors[:, 1], to_obs_vectors[:, 0])
        angle_diff = np.abs(np.array([angle_normalize(a - robot_yaw) for a in angle_to_obs]))
        unpassed_mask = angle_diff <= (angle_unpassed / 2.0)

        # Always keep very close obstacles regardless of heading to avoid
        # collisions with already-detected side/rear obstacles.
        v_max = float(self.robot_spec.get('v_max', 1.0))
        critical_clearance = float(self.robot_spec.get('critical_obs_clearance', max(0.2, 0.55 * v_max * self.dt)))
        critical_mask = clearances <= critical_clearance

        candidate_mask = np.logical_or(unpassed_mask, critical_mask)
        if np.any(candidate_mask):
            candidate_indices = np.where(candidate_mask)[0]
        else:
            candidate_indices = np.arange(all_obs.shape[0], dtype=int)

        ordered_candidates = candidate_indices[np.argsort(clearances[candidate_indices])]
        nearest_indices = ordered_candidates[:obs_num].astype(int, copy=True)

        unknown_candidates = candidate_indices[unknown_mask[candidate_indices]]
        ordered_unknown = unknown_candidates[np.argsort(clearances[unknown_candidates])]
        unknown_priority_count = int(
            self.robot_spec.get(
                'unknown_obs_priority_count',
                min(4, max(2, int(max(obs_num, 1) // 6))),
            )
        )
        unknown_quota = min(obs_num, max(0, unknown_priority_count))
        if unknown_quota > 0 and ordered_unknown.size > 0 and nearest_indices.size > 0:
            selected_set = set(int(i) for i in nearest_indices.tolist())
            unknown_selected = int(np.sum(unknown_mask[nearest_indices]))
            missing_unknown = max(0, unknown_quota - unknown_selected)

            if missing_unknown > 0:
                unknown_pool = [int(i) for i in ordered_unknown.tolist() if int(i) not in selected_set]
                for unknown_idx in unknown_pool:
                    if missing_unknown <= 0:
                        break
                    non_unknown_positions = [
                        pos for pos, idx in enumerate(nearest_indices.tolist())
                        if not unknown_mask[int(idx)]
                    ]
                    if len(non_unknown_positions) == 0:
                        break
                    replace_pos = max(
                        non_unknown_positions,
                        key=lambda p: clearances[int(nearest_indices[p])],
                    )
                    nearest_indices[replace_pos] = int(unknown_idx)
                    selected_set.add(int(unknown_idx))
                    missing_unknown -= 1

            nearest_indices = nearest_indices[np.argsort(clearances[nearest_indices])]
        return all_obs[nearest_indices]

    def control_step(self):
        # Keep a snapshot of the known footprint before stepping so the
        # point-mass visibility metric can detect boundary crossing this step.
        pre_step_footprint = self.robot.sensing_footprints
        ret = super().control_step()
        self._apply_visibility_plot_toggle()

        if ret == -2:
            return ret
        if 'sensor' not in self.robot_spec or self.robot_spec['sensor'] != 'rgbd':
            return ret

        violation_mode = str(
            self.robot_spec.get('visibility_violation_mode', 'safety_area')
        ).lower()
        if violation_mode != 'point_mass':
            return ret

        if pre_step_footprint is None or pre_step_footprint.is_empty:
            return ret

        # Apply a small tolerance around the known boundary to avoid counting
        # near-boundary numerical jitter as a visibility violation.
        vis_tol = float(self.robot_spec.get('visibility_violation_tolerance', 0.02))
        vis_tol = max(vis_tol, 0.0)
        known_region = pre_step_footprint.buffer(vis_tol + 1e-9)
        p = Point(float(self.robot.X[0, 0]), float(self.robot.X[1, 0]))
        if not known_region.covers(p):
            self.robot.unsafe_point_count += 1
            self.robot.unsafe_points.append((self.robot.X[0, 0], self.robot.X[1, 0]))
            # Only convert normal return codes to violation; terminal statuses
            # (-1: done, -2: collision/infeasible) are preserved.
            if ret == 0:
                self._apply_visibility_plot_toggle()
                return 1
        self._apply_visibility_plot_toggle()
        return ret

    def _apply_visibility_plot_toggle(self):
        if self._show_visibility_violations:
            return
        # Keep internal violation counting intact, but hide visual markers.
        if getattr(self.robot, 'unsafe_points_handle', None) is not None:
            self.robot.unsafe_points_handle.set_visible(False)
            try:
                self.robot.unsafe_points_handle.set_offsets(np.empty((0, 2)))
            except Exception:
                pass

    def _duplicate_last_frame_for_hold(self):
        if not (self.show_animation and self.save_animation):
            return
        hold_seconds = float(self.robot_spec.get('end_frame_hold_seconds', 1.0))
        if hold_seconds <= 0.0:
            return
        input_fps = float(self.robot_spec.get('animation_input_fps', 30.0))
        hold_frames = int(np.round(max(0.0, hold_seconds) * max(input_fps, 1.0)))
        if hold_frames <= 0:
            return

        frame_glob = os.path.join(self.current_directory_path, "output/animations/t_step_*.png")
        frame_files = sorted(glob.glob(frame_glob))
        if len(frame_files) == 0:
            return

        last_file = frame_files[-1]
        match = re.search(r"t_step_(\d+)\.png$", os.path.basename(last_file))
        if match is None:
            return
        last_idx = int(match.group(1))
        for k in range(1, hold_frames + 1):
            dst = os.path.join(
                self.current_directory_path,
                "output/animations",
                f"t_step_{last_idx + k:04d}.png",
            )
            shutil.copyfile(last_file, dst)

    def export_video(self):
        if self.show_animation and self.save_animation:
            self._duplicate_last_frame_for_hold()
            subprocess.call([
                'ffmpeg',
                '-framerate', str(int(self.robot_spec.get('animation_input_fps', 30))),
                '-i', self.current_directory_path + "/output/animations/t_step_%04d.png",
                '-vf', 'scale=1920:982,fps=60',
                '-pix_fmt', 'yuv420p',
                self.current_directory_path + "/output/animations/tracking.mp4",
            ])

            for file_name in glob.glob(self.current_directory_path + "/output/animations/*.png"):
                os.remove(file_name)
