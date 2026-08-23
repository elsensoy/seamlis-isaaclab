import argparse
import math
import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def build_indoor_exploration_env():
    env_width = 24.0
    env_height = 18.0
    e_wall = 2.0

    def _split_with_doors(start, end, door_intervals, max_seg_len=1.8):
        intervals = [(float(start), float(end))]
        for ds, de in sorted(door_intervals):
            next_intervals = []
            for s0, e0 in intervals:
                if de <= s0 or ds >= e0:
                    next_intervals.append((s0, e0))
                    continue
                if ds > s0:
                    next_intervals.append((s0, min(ds, e0)))
                if de < e0:
                    next_intervals.append((max(de, s0), e0))
            intervals = next_intervals

        split_intervals = []
        for s0, e0 in intervals:
            length = max(e0 - s0, 0.0)
            if length < 0.25:
                continue
            num_seg = max(int(np.ceil(length / max_seg_len)), 1)
            seg_len = length / num_seg
            for i in range(num_seg):
                ss = s0 + i * seg_len
                ee = s0 + (i + 1) * seg_len
                if ee - ss > 0.2:
                    split_intervals.append((ss, ee))
        return split_intervals

    def _vertical_wall(x, y_min, y_max, door_intervals, half_thickness=0.35):
        segs = _split_with_doors(y_min, y_max, door_intervals)
        return [[x, 0.5 * (s + e), half_thickness, 0.5 * (e - s), e_wall, 0.0, 1.0] for s, e in segs]

    def _horizontal_wall(y, x_min, x_max, door_intervals, half_thickness=0.35):
        segs = _split_with_doors(x_min, x_max, door_intervals)
        return [[0.5 * (s + e), y, 0.5 * (e - s), half_thickness, e_wall, 0.0, 1.0] for s, e in segs]

    interior_walls = []
    # Boundary-connected indoor partitions with wide doors.
    interior_walls += _vertical_wall(16.0, 0.0, 18.0, door_intervals=[(1.2, 13.4), (14.8, 17.2)])
    interior_walls += _horizontal_wall(13.0, 0.0, 7.0, door_intervals=[(1.2, 5.8)])
    interior_walls = np.array(interior_walls, dtype=np.float64)

    known_circles = np.array(
        [
            [1.9609824671993221, 3.571635852132964, 0.34050950984221157],
            [1.676274334947608, 14.685978826913212, 0.34888886514506134],
            [10.154313911104943, 6.197057802522608, 0.31567569237250975],
            [11.172968489948397, 13.709415691971323, 0.44],
            [12.840940316109451, 5.668353014220707, 0.28],
            [17.84911288084617, 6.673288202225253, 0.35923213699101575],
            [20.74134371070081, 14.657228646665182, 0.4132990115393822],
            [19.49278534976608, 9.762322881834, 0.3082173123587094],
        ],
        dtype=np.float64,
    )
    known_circles = np.hstack((known_circles, np.zeros((known_circles.shape[0], 4))))

    known_obs = np.vstack((known_circles, interior_walls))

    unknown_obs = np.array(
        [
            [11.362271107787219, 8.337044872202748, 0.3534059153097643],
            [7.618, 8.824, 0.2809135884004253],
            [14.614, 10.842, 0.2723120820423194],
            [3.6076386528860906, 7.946496393342197, 0.3511538410747291],
            [11.046, 11.226, 0.33635577409748685],
            [6.3811796772763865, 5.40084144215512, 0.342438685717245],
            [8.612, 12.648, 0.3288366773537649],
            [11.362, 6.518, 0.31487221430580004],
            [4.097645485203005, 5.961445020410135, 0.3013563252909263],
            [5.006429344726173, 5.811399027777369, 0.32015699814621223], #causing deadlock
            [10.460569204900429, 16.168491431352955, 0.26213281684933154],
            [10.436, 12.462, 0.29772772592575547],
            [11.975285417129745, 17.507220605630696, 0.23964754358894777],
            [11.087933619108204, 1.3033333032753358, 0.3506232510240296],
            [14.228, 6.214, 0.32986688289530625],
            [5.974553486302494, 9.50883337247491, 0.33155439236552264],
            [20.307339013256755, 10.710580277796133, 0.3048422148980015],
            [7.182, 6.628, 0.27776001969831526],
            [20.297004786237103, 8.235727438424906, 0.3112703435811964],
            [3.106378022744236, 1.8346281020628696, 0.33431962687934746],
            [4.548, 13.562, 0.30322071292590325],
            [3.925470482939058, 16.01888543328257, 0.26344685434937043],
        ],
        dtype=np.float64,
    )

    return env_width, env_height, known_obs, unknown_obs


def build_open_exploration_env():
    env_width = 24.0
    env_height = 18.0
    e_wall = 2.0

    # Open map: only a few short wall pieces, but many known/unknown obstacles.
    short_walls = np.array(
        [
            [21.0, 13.0, 1.6, 0.35, e_wall, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    known_circles = np.array(
        [
            [3.2, 3.0, 0.45],
            [3.2, 9.0, 0.45],
            [3.2, 15.0, 0.45],
            [7.2, 4.0, 0.48],
            [7.2, 10.0, 0.48],
            [7.2, 15.0, 0.48],
            [12.0, 6.2, 0.48],
            [12.0, 12.0, 0.48],
            [16.8, 3.2, 0.48],
            [16.8, 9.2, 0.48],
            [16.8, 15.0, 0.48],
            [21.8, 6.0, 0.46],
            [21.8, 11.8, 0.46],
        ],
        dtype=np.float64,
    )
    known_circles = np.hstack((known_circles, np.zeros((known_circles.shape[0], 4))))

    known_obs = np.vstack((known_circles, short_walls))

    unknown_obs = np.array(
        [
            [2.4, 6.0, 0.22],
            [2.6, 12.0, 0.22],
            [4.8, 8.0, 0.22],
            [5.6, 13.4, 0.22],
            [8.6, 6.2, 0.22],
            [8.8, 12.8, 0.22],
            [10.6, 3.6, 0.22],
            [10.8, 9.6, 0.22],
            [10.8, 15.0, 0.22],
            [13.6, 4.4, 0.22],
            [13.6, 10.0, 0.22],
            [13.6, 14.8, 0.22],
            [17.4, 5.8, 0.22],
            [17.6, 11.2, 0.22],
            [18.4, 14.6, 0.22],
            [21.0, 8.6, 0.22],
            [22.2, 10.0, 0.22],
            [22.2, 14.2, 0.22],
        ],
        dtype=np.float64,
    )

    return env_width, env_height, known_obs, unknown_obs


def build_stress_unknown_obs(layout):
    if layout == 'indoor':
        return np.array(
            [
                [5.2, 5.0, 0.24],
                [6.2, 8.8, 0.24],
                [7.2, 13.8, 0.24],
                [9.0, 7.2, 0.24],
                [11.0, 5.4, 0.24],
                [11.6, 9.8, 0.24],
                [13.8, 9.2, 0.24],
                [15.0, 11.8, 0.24],
                [17.0, 5.6, 0.24],
                [18.8, 8.8, 0.24],
                [19.8, 13.8, 0.24],
                [21.2, 9.6, 0.24],
            ],
            dtype=np.float64,
        )

    return np.array(
        [
            [4.6, 5.6, 0.24],
            [5.4, 11.0, 0.24],
            [6.8, 8.0, 0.24],
            [8.8, 4.8, 0.24],
            [9.2, 13.8, 0.24],
            [11.0, 8.0, 0.24],
            [12.6, 4.6, 0.24],
            [13.2, 13.4, 0.24],
            [15.4, 7.0, 0.24],
            [17.2, 4.8, 0.24],
            [17.8, 13.2, 0.24],
            [20.4, 9.8, 0.24],
        ],
        dtype=np.float64,
    )


def build_initial_states(num_agent):
    candidates = np.array(
        [
            [2.0, 2.0, 0.0],
            [2.0, 16.0, -math.pi / 2.0],
            [22.0, 4.0, math.pi],
        ],
        dtype=np.float64,
    )
    if num_agent < 1 or num_agent > candidates.shape[0]:
        raise ValueError("num_agent must be in [1, 3] for this test scenario.")
    return [candidates[i] for i in range(num_agent)]


def get_robot_specs(num_agent, use_astar):
    robot_specs = []
    for robot_id in range(num_agent):
        if use_astar:
            robot_spec = {
                'model': 'DoubleIntegrator2D',
                'v_max': 1.35,
                'a_max': 1.9,
                'radius': 0.15,
                'sensor': 'rgbd',
                'fov_angle': 70.0,
                'cam_range': 4.5,
                'num_constraints': 20,
                'reached_threshold': 1.8,
                'min_goal_distance': 2.6,
                'nominal_k_v': 2.2,
                'nominal_k_a': 2.2,
                'unknown_obs_detection': 'fov',
                'exploration': True,
                'robot_id': robot_id,
                'visibility_violation_mode': 'point_mass',
                'visibility_violation_tolerance': 0.02,
                'deadlock_window_s': 3.0,
                'deadlock_position_eps': 0.28,
                'deadlock_speed_eps': 0.06,
                'deadlock_goal_margin': 0.9,
                'deadlock_cooldown_s': 3.5,
                'deadlock_max_recoveries': 12,
                'mpc_horizon': 10,
                'mpc_cbf_alpha1': 0.55,
                'mpc_cbf_alpha2': 0.55,
            }
        else:
            robot_spec = {
                'model': 'DoubleIntegrator2D',
                'v_max': 1.45,
                'a_max': 2.0,
                'radius': 0.18,
                'sensor': 'rgbd',
                'fov_angle': 70.0,
                'cam_range': 4.5,
                'num_constraints': 16,
                'reached_threshold': 1.6,
                'min_goal_distance': 2.2,
                'nominal_k_v': 2.4,
                'nominal_k_a': 2.3,
                'unknown_obs_detection': 'fov',
                'exploration': True,
                'robot_id': robot_id,
                'visibility_violation_mode': 'point_mass',
                'visibility_violation_tolerance': 0.02,
                'deadlock_window_s': 3.0,
                'deadlock_position_eps': 0.28,
                'deadlock_speed_eps': 0.06,
                'deadlock_goal_margin': 0.9,
                'deadlock_cooldown_s': 3.5,
                'deadlock_max_recoveries': 12,
                'mpc_horizon': 8,
                'mpc_cbf_alpha1': 0.45,
                'mpc_cbf_alpha2': 0.45,
            }
        robot_specs.append(robot_spec)
    return robot_specs


def parse_args():
    parser = argparse.ArgumentParser(description='Run exploration test scenario.')
    parser.add_argument('--num_agent', type=int, default=2, help='Number of robots (supported: 1, 2, 3).')
    parser.add_argument(
        '--algo',
        type=str,
        default='frontier',
        choices=['coscan', 'frontier'],
        help='Exploration algorithm: coscan or frontier.',
    )
    parser.add_argument(
        '--layout',
        type=str,
        default='indoor',
        choices=['indoor', 'open'],
        help='Environment layout: indoor (wall-heavy) or open (obstacle-heavy).',
    )
    astar_group = parser.add_mutually_exclusive_group()
    astar_group.add_argument('--use_astar', dest='use_astar', action='store_true', help='Enable A* corridor waypoints.')
    astar_group.add_argument('--no-astar', dest='use_astar', action='store_false', help='Disable A* waypoints.')
    parser.set_defaults(use_astar=None)
    parser.add_argument(
        '--attitude',
        type=str,
        default='velocity_tracking_yaw',
        choices=['velocity_tracking_yaw', 'visibility_area', 'simple', 'visibility_raycast', 'gatekeeper', 'visibility'],
        help='Attitude controller name.',
    )
    parser.add_argument(
        '--gatekeeper_nominal',
        type=str,
        default='visibility_area',
        choices=['visibility_area', 'simple', 'velocity_tracking_yaw'],
        help='Nominal attitude controller used inside gatekeeper.',
    )
    parser.add_argument(
        '--gatekeeper_backup',
        type=str,
        default='velocity_tracking_yaw',
        choices=['velocity_tracking_yaw', 'simple'],
        help='Backup attitude controller used inside gatekeeper.',
    )
    parser.add_argument(
        '--gatekeeper_nominal_horizon',
        type=float,
        default=0.4,
        help='Gatekeeper nominal horizon [s].',
    )
    parser.add_argument(
        '--gatekeeper_backup_horizon',
        type=float,
        default=1.8,
        help='Gatekeeper backup horizon [s].',
    )
    parser.add_argument(
        '--gatekeeper_event_offset',
        type=float,
        default=0.0,
        help='Gatekeeper event offset [s].',
    )
    parser.add_argument(
        '--gatekeeper_horizon_discount',
        type=float,
        default=0.05,
        help='Gatekeeper nominal-horizon discount step [s].',
    )
    parser.add_argument(
        '--gatekeeper_validation_slack',
        type=float,
        default=0.30,
        help='Extra slack [m] for braking-distance monitor.',
    )
    parser.add_argument(
        '--gatekeeper_braking_margin',
        type=float,
        default=0.90,
        help='Extra conservative braking margin [m].',
    )
    parser.add_argument(
        '--pos_controller',
        type=str,
        default='mpc_cbf',
        choices=['cbf_qp', 'mpc_cbf'],
        help='Position controller name.',
    )
    parser.add_argument('--coverage_target', type=float, default=0.98, help='Coverage ratio target for success.')
    parser.add_argument(
        '--unknown_profile',
        type=str,
        default='default',
        choices=['default', 'stress'],
        help='Unknown-obstacle profile: default or stress (denser, harder).',
    )
    parser.add_argument('--map_resolution', type=float, default=0.16, help='Exploration map resolution [m/cell].')
    parser.add_argument('--fov_angle', type=float, default=None, help='Override robot FoV angle in degrees.')
    parser.add_argument('--cam_range', type=float, default=None, help='Override robot camera range in meters.')
    parser.add_argument('--w_max', type=float, default=None, help='Override robot max yaw rate [rad/s].')
    parser.add_argument(
        '--hide_visibility_violations',
        action='store_true',
        help='Hide visibility-violation red markers in the animation (counting is unchanged).',
    )
    parser.add_argument('--save_anim', action='store_true', help='Save animation as mp4 (rendering required).')
    parser.add_argument('--no_render', action='store_true', help='Disable live rendering (headless run).')
    parser.add_argument('--dt', type=float, default=0.1, help='Simulation step size.')
    parser.add_argument('--tf', type=float, default=300.0, help='Simulation horizon in seconds.')
    unknown_group = parser.add_mutually_exclusive_group()
    unknown_group.add_argument('--unknown', dest='unknown', action='store_true', help='Enable unknown obstacles.')
    unknown_group.add_argument('--no-unknown', dest='unknown', action='store_false', help='Disable unknown obstacles.')
    parser.set_defaults(unknown=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_render:
        import matplotlib

        matplotlib.use('Agg')

    from exploration import ExplorationManager
    from safe_control.utils import env

    layout = args.layout
    use_astar = (layout == 'indoor') if args.use_astar is None else bool(args.use_astar)

    if layout == 'indoor':
        env_width, env_height, known_obs, unknown_obs = build_indoor_exploration_env()
    else:
        env_width, env_height, known_obs, unknown_obs = build_open_exploration_env()

    if not args.unknown:
        unknown_obs = np.empty((0, 3), dtype=np.float64)
    elif args.unknown_profile == 'stress':
        unknown_obs = np.vstack((unknown_obs, build_stress_unknown_obs(layout)))

    show_animation = not args.no_render
    save_animation = args.save_anim and show_animation
    if args.no_render and args.save_anim:
        print('`--save_anim` requires rendering. Ignoring save request because `--no_render` is set.')

    x0s = build_initial_states(args.num_agent)
    robot_specs = get_robot_specs(args.num_agent, use_astar=use_astar)
    for robot_spec in robot_specs:
        if args.fov_angle is not None:
            robot_spec['fov_angle'] = float(args.fov_angle)
        if args.cam_range is not None:
            robot_spec['cam_range'] = float(args.cam_range)
        if args.w_max is not None:
            robot_spec['w_max'] = float(args.w_max)
        if args.hide_visibility_violations:
            robot_spec['show_visibility_violations'] = False
        # Keep unknown-obstacle memory persistent for each agent.
        robot_spec['unknown_obs_persistent_fov'] = True
        if args.attitude == 'gatekeeper':
            robot_spec['w_max'] = float(robot_spec.get('w_max', 1.2))
            robot_spec['visibility_violation_mode'] = 'point_mass'
            robot_spec['gatekeeper_nominal'] = args.gatekeeper_nominal
            robot_spec['gatekeeper_backup'] = args.gatekeeper_backup
            robot_spec['gatekeeper_nominal_horizon'] = float(args.gatekeeper_nominal_horizon)
            robot_spec['gatekeeper_backup_horizon'] = float(args.gatekeeper_backup_horizon)
            robot_spec['gatekeeper_event_offset'] = float(args.gatekeeper_event_offset)
            robot_spec['gatekeeper_horizon_discount'] = float(args.gatekeeper_horizon_discount)
            robot_spec['gatekeeper_validation_slack'] = float(args.gatekeeper_validation_slack)
            robot_spec['gatekeeper_braking_distance_margin'] = float(args.gatekeeper_braking_margin)
    env_handler = env.Env(
        width=env_width,
        height=env_height,
        known_obs=known_obs,
        resolution=args.map_resolution,
    )

    controller_type = {
        'pos': args.pos_controller,
        'att': args.attitude,
    }

    exploration_algorithm = 'CoScan' if args.algo == 'coscan' else 'Frontier'
    manager = ExplorationManager(
        x0s,
        robot_specs,
        controller_type,
        exploration_algorithm=exploration_algorithm,
        dt=args.dt,
        show_animation=show_animation,
        save_animation=save_animation,
        env_handler=env_handler,
        known_obs=known_obs,
        unknown_obs=unknown_obs,
        use_astar_waypoints=use_astar,
        coverage_target=args.coverage_target,
    )

    max_steps = int(args.tf / args.dt)
    success = manager.explore(max_steps=max_steps)
    violation_counts = [controller.robot.unsafe_point_count for controller in manager.controller_list]
    print(f'Visibility violations per robot: {violation_counts} (total={sum(violation_counts)})')
    if args.attitude == 'gatekeeper':
        gk_stats = []
        for i, controller in enumerate(manager.controller_list):
            att_ctrl = getattr(controller, 'att_controller', None)
            if att_ctrl is not None and hasattr(att_ctrl, 'get_stats'):
                stats = att_ctrl.get_stats()
                gk_stats.append(
                    f"r{i}: replans={stats['replans']}, accepted={stats['accepted']}, rejected={stats['rejected']}, "
                    f"nominal_commits={stats['nominal_commits']}, "
                    f"nominal_max={stats['nominal_seconds_max']:.2f}s, nominal_avg={stats['nominal_seconds_avg_per_commit']:.2f}s"
                )
        if gk_stats:
            print('Gatekeeper nominal usage -> ' + ' | '.join(gk_stats))
    if success:
        print('Success!')
    else:
        print('Failed!')


if __name__ == '__main__':
    main()
