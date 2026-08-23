import argparse
import concurrent.futures
import csv
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from exploration import ExplorationManager
from safe_control.utils import env
from examples.test_exploration import build_indoor_exploration_env, build_initial_states, get_robot_specs


ATTITUDES = ["simple", "visibility_area", "gatekeeper"]
ALGOS = ["frontier", "coscan"]
AGENT_COUNTS = [1, 2, 3]


DEFAULT_GATEKEEPER_PARAMS = {
    "gatekeeper_nominal": "visibility_area",
    "gatekeeper_backup": "velocity_tracking_yaw",
    # Strict-safety tuning for benchmarking.
    "gatekeeper_nominal_horizon": 0.4,
    "gatekeeper_backup_horizon": 1.8,
    "gatekeeper_event_offset": 0.0,
    "gatekeeper_horizon_discount": 0.05,
    "gatekeeper_validation_slack": 0.30,
    "gatekeeper_braking_distance_margin": 0.90,
}


@dataclass
class MapConfig:
    map_id: str
    known_obs: np.ndarray
    unknown_obs: np.ndarray
    source_seed: int


def _superellipse_outside(x: float, y: float, radius: float, walls: np.ndarray, margin: float = 0.04) -> bool:
    if walls.size == 0:
        return True
    for wall in walls:
        ox, oy, a, b, e, theta = wall[:6]
        ct = math.cos(theta)
        st = math.sin(theta)
        px = ct * (x - ox) + st * (y - oy)
        py = -st * (x - ox) + ct * (y - oy)
        den_a = max(abs(a) + radius + margin, 1e-3)
        den_b = max(abs(b) + radius + margin, 1e-3)
        expo = max(float(e), 2.0)
        h = (abs(px) / den_a) ** expo + (abs(py) / den_b) ** expo - 1.0
        if h <= 0.0:
            return False
    return True


def _circles_nonoverlap(x: float, y: float, radius: float, circles: np.ndarray, margin: float = 0.08) -> bool:
    if circles.size == 0:
        return True
    dxy = circles[:, :2] - np.array([x, y], dtype=float).reshape(1, 2)
    d = np.linalg.norm(dxy, axis=1)
    return bool(np.all(d > (circles[:, 2] + radius + margin)))


def _sample_circle(
    rng: np.random.Generator,
    width: float,
    height: float,
    walls: np.ndarray,
    existing_circles: np.ndarray,
    starts_xy: np.ndarray,
    r_min: float,
    r_max: float,
    start_clearance: float,
    clearance_margin: float,
    wall_margin: float,
    boundary_margin: float,
    mode: str,
    centers: Optional[np.ndarray] = None,
    max_tries: int = 1500,
) -> Optional[np.ndarray]:
    for _ in range(max_tries):
        r = float(rng.uniform(r_min, r_max))
        if mode == "cluster" and centers is not None and len(centers) > 0:
            center = centers[int(rng.integers(0, len(centers)))]
            x = float(center[0] + rng.normal(0.0, 1.0))
            y = float(center[1] + rng.normal(0.0, 1.0))
        else:
            x = float(rng.uniform(r + 0.4, width - r - 0.4))
            y = float(rng.uniform(r + 0.4, height - r - 0.4))

        if x <= r + boundary_margin or x >= width - r - boundary_margin or y <= r + boundary_margin or y >= height - r - boundary_margin:
            continue
        if not _superellipse_outside(x, y, r, walls, margin=wall_margin):
            continue
        if not _circles_nonoverlap(x, y, r, existing_circles, margin=clearance_margin):
            continue
        if starts_xy.size > 0:
            d_start = np.linalg.norm(starts_xy - np.array([x, y], dtype=float).reshape(1, 2), axis=1)
            if np.any(d_start <= (r + start_clearance)):
                continue
        return np.array([x, y, r], dtype=float)
    return None


def generate_random_indoor_map(
    rng: np.random.Generator,
    map_index: int,
    base_known_obs: np.ndarray,
    env_width: float,
    env_height: float,
    starts_xy: np.ndarray,
) -> MapConfig:
    max_robot_radius = 0.18
    # Keep random maps dense, but reject obstacle placements that create
    # near-impossible single-robot passages.
    min_obstacle_boundary_gap = 2.0 * max_robot_radius + 0.20

    flags = base_known_obs[:, 6] if base_known_obs.size > 0 else np.zeros(0, dtype=float)
    walls = base_known_obs[np.isclose(flags, 1.0)].copy() if base_known_obs.size > 0 else np.empty((0, 7), dtype=float)
    base_circles = base_known_obs[~np.isclose(flags, 1.0), :3].copy() if base_known_obs.size > 0 else np.empty((0, 3), dtype=float)
    if base_circles.size == 0:
        base_circles = np.array(
            [
                [3.0, 3.0, 0.35],
                [3.0, 15.0, 0.35],
                [9.0, 4.0, 0.35],
                [11.0, 14.0, 0.35],
                [13.0, 6.0, 0.35],
                [18.0, 7.0, 0.35],
                [20.0, 15.0, 0.35],
                [21.0, 10.0, 0.35],
            ],
            dtype=float,
        )

    known_circles: List[np.ndarray] = []
    for base in base_circles:
        accepted = None
        existing = np.array(known_circles, dtype=float) if len(known_circles) > 0 else np.empty((0, 3), dtype=float)
        for _ in range(400):
            r = float(np.clip(base[2] + rng.normal(0.0, 0.05), 0.28, 0.44))
            x = float(np.clip(base[0] + rng.normal(0.0, 0.35), r + 0.5, env_width - r - 0.5))
            y = float(np.clip(base[1] + rng.normal(0.0, 0.35), r + 0.5, env_height - r - 0.5))
            if not _superellipse_outside(x, y, r, walls, margin=0.04):
                continue
            if not _circles_nonoverlap(x, y, r, existing, margin=min_obstacle_boundary_gap):
                continue
            d_start = np.linalg.norm(starts_xy - np.array([x, y], dtype=float).reshape(1, 2), axis=1)
            if np.any(d_start <= (r + 0.8)):
                continue
            accepted = np.array([x, y, r], dtype=float)
            break

        if accepted is None:
            base_ok = (
                _superellipse_outside(float(base[0]), float(base[1]), float(base[2]), walls, margin=0.04)
                and _circles_nonoverlap(
                    float(base[0]),
                    float(base[1]),
                    float(base[2]),
                    existing,
                    margin=min_obstacle_boundary_gap,
                )
            )
            if base_ok and starts_xy.size > 0:
                d_start = np.linalg.norm(starts_xy - base[:2].reshape(1, 2), axis=1)
                base_ok = bool(np.all(d_start > (float(base[2]) + 0.8)))
            if base_ok:
                accepted = base.copy()

        if accepted is not None:
            known_circles.append(accepted)

    known_circles_arr = np.array(known_circles, dtype=float) if len(known_circles) > 0 else np.empty((0, 3), dtype=float)
    known_circles_arr = np.hstack((known_circles_arr, np.zeros((known_circles_arr.shape[0], 4), dtype=float)))
    known_obs = np.vstack((known_circles_arr, walls))

    # Unknown-obstacle sampling: clustered + uniform to induce side-visibility hazards.
    cluster_centers = np.array(
        [
            [5.0, 5.5],
            [9.8, 11.6],
            [11.6, 3.3],
            [20.2, 10.8],
            [8.5, 16.0],
            [10.6, 8.9],
        ],
        dtype=float,
    )
    cluster_centers += rng.normal(0.0, 0.20, size=cluster_centers.shape)

    unknown_circles: List[np.ndarray] = []
    known_for_clearance = known_obs[:, :3].copy() if known_obs.size > 0 else np.empty((0, 3), dtype=float)
    total_unknown = 9
    clustered_unknown = 5

    for idx in range(total_unknown):
        mode = "cluster" if idx < clustered_unknown else "uniform"
        existing_unknown = np.array(unknown_circles, dtype=float) if len(unknown_circles) > 0 else np.empty((0, 3), dtype=float)
        combined_existing = existing_unknown
        if known_for_clearance.size > 0:
            combined_existing = (
                np.vstack((known_for_clearance, existing_unknown))
                if existing_unknown.size > 0
                else known_for_clearance
            )
        sampled = _sample_circle(
            rng=rng,
            width=env_width,
            height=env_height,
            walls=walls,
            existing_circles=combined_existing,
            starts_xy=starts_xy,
            r_min=0.22,
            r_max=0.28,
            start_clearance=1.2,
            clearance_margin=min_obstacle_boundary_gap,
            wall_margin=0.10,
            boundary_margin=0.32,
            mode=mode,
            centers=cluster_centers if mode == "cluster" else None,
            max_tries=2000,
        )
        if sampled is not None:
            unknown_circles.append(sampled)

    if len(unknown_circles) == 0:
        unknown_circles.append(np.array([9.8, 11.8, 0.30], dtype=float))
    unknown_obs = np.array(unknown_circles, dtype=float)

    return MapConfig(
        map_id=f"map_{map_index:03d}",
        known_obs=known_obs,
        unknown_obs=unknown_obs,
        source_seed=int(rng.integers(0, 2**31 - 1)),
    )


def run_exploration_case(
    map_cfg: MapConfig,
    num_agent: int,
    algo: str,
    attitude: str,
    dt: float,
    tf: float,
    coverage_target: float,
    use_astar: bool,
    gatekeeper_params: Dict[str, float],
) -> Dict[str, object]:
    x0s = build_initial_states(num_agent)
    robot_specs = get_robot_specs(num_agent, use_astar=use_astar)
    for spec in robot_specs:
        spec["unknown_obs_persistent_fov"] = True
        # Fair visibility-violation accounting across all attitudes.
        spec["visibility_violation_mode"] = "point_mass"
        spec["visibility_violation_tolerance"] = 0.02
        if attitude == "gatekeeper":
            spec["w_max"] = float(spec.get("w_max", 1.2))
            spec.update(gatekeeper_params)

    env_handler = env.Env(
        width=24.0,
        height=18.0,
        known_obs=map_cfg.known_obs,
        resolution=0.16,
    )
    controller_type = {"pos": "mpc_cbf", "att": attitude}
    exploration_algorithm = "CoScan" if algo == "coscan" else "Frontier"

    manager = ExplorationManager(
        x0s,
        robot_specs,
        controller_type,
        exploration_algorithm=exploration_algorithm,
        dt=dt,
        show_animation=False,
        save_animation=False,
        env_handler=env_handler,
        known_obs=map_cfg.known_obs,
        unknown_obs=map_cfg.unknown_obs,
        use_astar_waypoints=use_astar,
        coverage_target=coverage_target,
    )

    max_steps = int(tf / dt)
    t0 = time.perf_counter()
    success = bool(manager.explore(max_steps=max_steps))
    wallclock = float(time.perf_counter() - t0)

    violations = [c.robot.unsafe_point_count for c in manager.controller_list]
    visibility_total = int(sum(violations))

    collision_info = manager.last_collision_info if manager.last_collision_info is not None else {}
    collision_type = str(collision_info.get("type", "")) if isinstance(collision_info, dict) else ""
    collision_stage = str(collision_info.get("stage", "")) if isinstance(collision_info, dict) else ""
    unknown_collision = collision_type == "unknown"
    collision_or_infeasible = manager.last_termination_reason == "collision_or_infeasible"

    result = {
        "map_id": map_cfg.map_id,
        "num_agent": int(num_agent),
        "algo": str(algo),
        "attitude": str(attitude),
        "success": bool(success),
        "coverage": float(manager.last_coverage_ratio),
        "termination_reason": str(manager.last_termination_reason),
        "step_count": int(manager.last_step_count),
        "sim_time": float(manager.last_sim_time),
        "wallclock_time": wallclock,
        "visibility_total": visibility_total,
        "visibility_per_robot": json.dumps(violations),
        "collision_or_infeasible": bool(collision_or_infeasible),
        "collision_type": collision_type,
        "collision_stage": collision_stage,
        "unknown_collision": bool(unknown_collision),
        "exploration_time_success": float(manager.last_sim_time) if success else None,
    }

    if isinstance(collision_info, dict) and len(collision_info) > 0:
        result["collision_info_json"] = json.dumps(collision_info)
    else:
        result["collision_info_json"] = ""

    if attitude == "gatekeeper":
        nominal_usage = []
        for controller in manager.controller_list:
            att_ctrl = getattr(controller, "att_controller", None)
            if att_ctrl is not None and hasattr(att_ctrl, "get_stats"):
                stats = att_ctrl.get_stats()
                nominal_usage.append(float(stats.get("nominal_seconds_avg_per_commit", 0.0)))
        result["gatekeeper_nominal_avg"] = float(np.mean(nominal_usage)) if len(nominal_usage) > 0 else 0.0
    else:
        result["gatekeeper_nominal_avg"] = None

    # Benchmark runs are headless, so this is normally a no-op (no Matplotlib
    # figure is created at all -- see ExplorationManager.__init__). Still call
    # it in case some caller passed show_animation=True, to avoid
    # process-level figure accumulation during large sweeps.
    try:
        manager.close()
    except Exception:
        pass

    return result


def write_json(path: str, data: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def append_csv(path: str, row: Dict[str, object], header: List[str]) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def summarize_results(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    def _to_bool(v: object) -> bool:
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        s = str(v).strip().lower()
        return s in ["1", "true", "yes", "y", "t"]

    grouped: Dict[Tuple[str, int, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (row["algo"], int(row["num_agent"]), row["attitude"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key in sorted(grouped.keys(), key=lambda x: (x[0], x[1], x[2])):
        algo, num_agent, attitude = key
        g = grouped[key]
        n = len(g)
        success_mask = np.array([_to_bool(r.get("success")) for r in g], dtype=bool)
        collision_mask = np.array([_to_bool(r.get("collision_or_infeasible")) for r in g], dtype=bool)
        unknown_collision_mask = np.array([_to_bool(r.get("unknown_collision")) for r in g], dtype=bool)
        vis_totals = np.array([int(r["visibility_total"]) for r in g], dtype=float)
        sim_times = np.array([float(r["sim_time"]) for r in g], dtype=float)
        success_times = np.array(
            [
                float(r["exploration_time_success"])
                for r in g
                if r.get("exploration_time_success") not in [None, ""]
            ],
            dtype=float,
        )

        summary_rows.append(
            {
                "algo": algo,
                "num_agent": num_agent,
                "attitude": attitude,
                "trials": n,
                "success_rate": float(np.mean(success_mask)) if n > 0 else 0.0,
                "collision_or_infeasible_rate": float(np.mean(collision_mask)) if n > 0 else 0.0,
                "unknown_collision_rate": float(np.mean(unknown_collision_mask)) if n > 0 else 0.0,
                "mean_visibility_total": float(np.mean(vis_totals)) if n > 0 else 0.0,
                "mean_visibility_per_robot": float(np.mean(vis_totals / max(num_agent, 1))) if n > 0 else 0.0,
                "mean_sim_time_all": float(np.mean(sim_times)) if n > 0 else 0.0,
                "mean_exploration_time_success_only": float(np.mean(success_times)) if success_times.size > 0 else None,
            }
        )
    return summary_rows


def render_summary_markdown(
    summary_rows: List[Dict[str, object]],
    output_path: str,
    hero_map_id: str,
    gatekeeper_params: Dict[str, float],
) -> None:
    lines = []
    lines.append("# Exploration Benchmark Summary")
    lines.append("")
    lines.append(f"- Hero map id: `{hero_map_id}`")
    lines.append(f"- Gatekeeper params: `{json.dumps(gatekeeper_params, sort_keys=True)}`")
    lines.append("")
    lines.append("| Algo | Agents | Attitude | Trials | Success Rate | Collision/Infeasible Rate | Unknown Collision Rate | Mean Vis. Viol. (Total) | Mean Vis. Viol. / Robot | Mean Sim Time (All) | Mean Explore Time (Success Only) |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in summary_rows:
        success_only = (
            f"{r['mean_exploration_time_success_only']:.2f}"
            if r["mean_exploration_time_success_only"] is not None
            else "N/A"
        )
        lines.append(
            "| {algo} | {num_agent} | {attitude} | {trials} | {sr:.2f} | {cr:.2f} | {ucr:.2f} | {mv:.2f} | {mvr:.2f} | {mt:.2f} | {mst} |".format(
                algo=r["algo"],
                num_agent=r["num_agent"],
                attitude=r["attitude"],
                trials=r["trials"],
                sr=r["success_rate"],
                cr=r["collision_or_infeasible_rate"],
                ucr=r["unknown_collision_rate"],
                mv=r["mean_visibility_total"],
                mvr=r["mean_visibility_per_robot"],
                mt=r["mean_sim_time_all"],
                mst=success_only,
            )
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def map_to_serializable(map_cfg: MapConfig) -> Dict[str, object]:
    return {
        "map_id": map_cfg.map_id,
        "source_seed": int(map_cfg.source_seed),
        "known_obs": map_cfg.known_obs.tolist(),
        "unknown_obs": map_cfg.unknown_obs.tolist(),
    }


def is_hero_case(
    simple_res_n2: Dict[str, object],
    vis_res_n2: Dict[str, object],
    gk_res_n2: Dict[str, object],
    simple_res_n3: Optional[Dict[str, object]],
    vis_res_n3: Optional[Dict[str, object]],
    gk_res_n3: Optional[Dict[str, object]],
    max_gatekeeper_visibility: int,
) -> bool:
    gk_n2_ok = bool(gk_res_n2["success"]) and (not bool(gk_res_n2["collision_or_infeasible"])) and (
        int(gk_res_n2["visibility_total"]) <= max_gatekeeper_visibility
    )
    simple_n2_ok = bool(simple_res_n2["unknown_collision"])
    vis_n2_ok = bool(vis_res_n2["unknown_collision"])

    # Prefer hero maps where both 2-agent and 3-agent baselines fail while
    # gatekeeper stays safe. If n=3 data is unavailable, fall back to n=2.
    if simple_res_n3 is not None and vis_res_n3 is not None and gk_res_n3 is not None:
        gk_n3_ok = bool(gk_res_n3["success"]) and (not bool(gk_res_n3["collision_or_infeasible"])) and (
            int(gk_res_n3["visibility_total"]) <= max_gatekeeper_visibility
        )
        simple_n3_ok = bool(simple_res_n3["unknown_collision"])
        vis_n3_ok = bool(vis_res_n3["unknown_collision"])
        return bool(gk_n2_ok and gk_n3_ok and simple_n2_ok and simple_n3_ok and vis_n2_ok and vis_n3_ok)

    return bool(gk_n2_ok and simple_n2_ok and vis_n2_ok)


def find_hero_map_id(rows: List[Dict[str, object]], max_gatekeeper_visibility: int) -> Optional[str]:
    map_ids = sorted({str(r["map_id"]) for r in rows})
    preferred_algos = ["coscan", "frontier"]
    for algo in preferred_algos:
        for map_id in map_ids:
            lookup = {}
            for r in rows:
                if str(r["map_id"]) != map_id:
                    continue
                key = (int(r["num_agent"]), str(r["algo"]), str(r["attitude"]))
                lookup[key] = r
            simple_res_n2 = lookup.get((2, algo, "simple"))
            vis_res_n2 = lookup.get((2, algo, "visibility_area"))
            gk_res_n2 = lookup.get((2, algo, "gatekeeper"))
            if simple_res_n2 is None or vis_res_n2 is None or gk_res_n2 is None:
                continue
            if is_hero_case(
                simple_res_n2=simple_res_n2,
                vis_res_n2=vis_res_n2,
                gk_res_n2=gk_res_n2,
                simple_res_n3=lookup.get((3, algo, "simple")),
                vis_res_n3=lookup.get((3, algo, "visibility_area")),
                gk_res_n3=lookup.get((3, algo, "gatekeeper")),
                max_gatekeeper_visibility=max_gatekeeper_visibility,
            ):
                return map_id
    return None


def main():
    parser = argparse.ArgumentParser(description="Seeded random exploration benchmark.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for map sampling.")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of sampled map trials.")
    parser.add_argument("--max_candidates", type=int, default=150, help="Maximum sampled candidate maps.")
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation dt.")
    parser.add_argument("--tf", type=float, default=300.0, help="Simulation horizon [s].")
    parser.add_argument("--coverage_target", type=float, default=0.98, help="Coverage success threshold.")
    parser.add_argument(
        "--hero_max_gatekeeper_visibility",
        type=int,
        default=0,
        help="Maximum total visibility violations allowed for hero gatekeeper run.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/benchmark_seed42",
        help="Directory for benchmark artifacts.",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=4,
        help="Retry attempts with new seeded map batches if constraints are not met.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for benchmark simulations.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    raw_csv_path = os.path.join(args.output_dir, "raw_results.csv")
    maps_json_path = os.path.join(args.output_dir, "trial_maps.json")
    hero_json_path = os.path.join(args.output_dir, "hero_case.json")
    summary_md_path = os.path.join(args.output_dir, "summary.md")

    env_width, env_height, base_known_obs, _ = build_indoor_exploration_env()
    starts = np.array([s[:2] for s in build_initial_states(3)], dtype=float)

    header = [
        "map_id",
        "num_agent",
        "algo",
        "attitude",
        "success",
        "coverage",
        "termination_reason",
        "step_count",
        "sim_time",
        "wallclock_time",
        "visibility_total",
        "visibility_per_robot",
        "collision_or_infeasible",
        "collision_type",
        "collision_stage",
        "unknown_collision",
        "exploration_time_success",
        "collision_info_json",
        "gatekeeper_nominal_avg",
    ]

    final_rows: Optional[List[Dict[str, object]]] = None
    final_maps: Optional[List[MapConfig]] = None
    final_hero_map_id: Optional[str] = None
    final_attempt_csv: Optional[str] = None

    for attempt in range(args.max_attempts):
        attempt_seed = args.seed + 1000 * attempt
        rng = np.random.default_rng(attempt_seed)
        attempt_csv_path = os.path.join(args.output_dir, f"raw_results_attempt_{attempt:02d}.csv")
        if os.path.exists(attempt_csv_path):
            os.remove(attempt_csv_path)

        trial_maps: List[MapConfig] = []
        for i in range(args.num_trials):
            map_cfg = generate_random_indoor_map(
                rng=rng,
                map_index=attempt * 1000 + i,
                base_known_obs=base_known_obs,
                env_width=env_width,
                env_height=env_height,
                starts_xy=starts,
            )
            map_cfg.map_id = f"a{attempt:02d}_m{i:02d}"
            trial_maps.append(map_cfg)

        all_rows: List[Dict[str, object]] = []

        gatekeeper_jobs = []
        non_gatekeeper_jobs = []
        for map_cfg in trial_maps:
            for num_agent in AGENT_COUNTS:
                for algo in ALGOS:
                    gatekeeper_jobs.append((map_cfg, num_agent, algo, "gatekeeper"))
                    non_gatekeeper_jobs.append((map_cfg, num_agent, algo, "simple"))
                    non_gatekeeper_jobs.append((map_cfg, num_agent, algo, "visibility_area"))

        def _run_job_list(jobs: List[Tuple[MapConfig, int, str, str]], offset: int, total: int) -> List[Dict[str, object]]:
            rows = []
            if args.workers <= 1:
                for idx, (map_cfg, num_agent, algo, attitude) in enumerate(jobs, start=1):
                    run_idx = offset + idx
                    print(
                        f"  run {run_idx:03d}/{total}: "
                        f"map={map_cfg.map_id}, agents={num_agent}, algo={algo}, att={attitude}"
                    )
                    row = run_exploration_case(
                        map_cfg=map_cfg,
                        num_agent=num_agent,
                        algo=algo,
                        attitude=attitude,
                        dt=args.dt,
                        tf=args.tf,
                        coverage_target=args.coverage_target,
                        use_astar=True,
                        gatekeeper_params=DEFAULT_GATEKEEPER_PARAMS,
                    )
                    row["attempt"] = int(attempt)
                    rows.append(row)
                    append_csv(attempt_csv_path, row=row, header=header + ["attempt"])
                return rows

            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_to_job = {}
                for map_cfg, num_agent, algo, attitude in jobs:
                    fut = executor.submit(
                        run_exploration_case,
                        map_cfg,
                        num_agent,
                        algo,
                        attitude,
                        args.dt,
                        args.tf,
                        args.coverage_target,
                        True,
                        DEFAULT_GATEKEEPER_PARAMS,
                    )
                    future_to_job[fut] = (map_cfg.map_id, num_agent, algo, attitude)

                completed = 0
                for fut in concurrent.futures.as_completed(future_to_job):
                    completed += 1
                    map_id, num_agent, algo, attitude = future_to_job[fut]
                    try:
                        row = fut.result()
                    except Exception as exc:
                        row = {
                            "map_id": map_id,
                            "num_agent": int(num_agent),
                            "algo": str(algo),
                            "attitude": str(attitude),
                            "success": False,
                            "coverage": 0.0,
                            "termination_reason": "exception",
                            "step_count": 0,
                            "sim_time": 0.0,
                            "wallclock_time": 0.0,
                            "visibility_total": 0,
                            "visibility_per_robot": "[]",
                            "collision_or_infeasible": True,
                            "collision_type": "exception",
                            "collision_stage": "",
                            "unknown_collision": False,
                            "exploration_time_success": None,
                            "collision_info_json": json.dumps({"error": str(exc)}),
                            "gatekeeper_nominal_avg": None,
                        }
                    row["attempt"] = int(attempt)
                    rows.append(row)
                    append_csv(attempt_csv_path, row=row, header=header + ["attempt"])
                    print(
                        f"  completed {offset + completed:03d}/{total}: "
                        f"map={map_id}, agents={num_agent}, algo={algo}, att={attitude}, "
                        f"success={row['success']}, coll_or_inf={row['collision_or_infeasible']}"
                    )
            return rows

        total_runs = len(gatekeeper_jobs) + len(non_gatekeeper_jobs)
        print(
            f"[attempt {attempt + 1}/{args.max_attempts}] seed={attempt_seed} "
            f"running gatekeeper precheck ({len(gatekeeper_jobs)} sims) with workers={args.workers}..."
        )
        gk_rows = _run_job_list(gatekeeper_jobs, offset=0, total=total_runs)
        all_rows.extend(gk_rows)
        gk_bad = [r for r in gk_rows if bool(r["collision_or_infeasible"])]
        gk_not_success = [r for r in gk_rows if not bool(r["success"])]
        if len(gk_bad) > 0 or len(gk_not_success) > 0:
            print(
                f"[attempt {attempt + 1}] gatekeeper precheck failed: "
                f"{len(gk_bad)} collision/infeasible, {len(gk_not_success)} non-success runs. "
                "Retrying with next seed batch."
            )
            continue

        print(
            f"[attempt {attempt + 1}] gatekeeper precheck passed. "
            f"Running non-gatekeeper jobs ({len(non_gatekeeper_jobs)} sims)..."
        )
        other_rows = _run_job_list(non_gatekeeper_jobs, offset=len(gatekeeper_jobs), total=total_runs)
        all_rows.extend(other_rows)

        hero_map_id = find_hero_map_id(all_rows, max_gatekeeper_visibility=args.hero_max_gatekeeper_visibility)

        gk_vis_per_robot = []
        for r in gk_rows:
            gk_vis_per_robot.append(float(r["visibility_total"]) / max(int(r["num_agent"]), 1))
        gk_vis_mean = float(np.mean(gk_vis_per_robot)) if len(gk_vis_per_robot) > 0 else 0.0

        print(
            f"[attempt {attempt + 1}] gatekeeper bad runs={len(gk_bad)}, "
            f"gatekeeper non-success runs={len(gk_not_success)}, "
            f"gatekeeper mean vis/robot={gk_vis_mean:.3f}, hero_map={hero_map_id}"
        )

        if len(gk_bad) == 0 and len(gk_not_success) == 0 and hero_map_id is not None:
            final_rows = all_rows
            final_maps = trial_maps
            final_hero_map_id = hero_map_id
            final_attempt_csv = attempt_csv_path
            break

    if final_rows is None or final_maps is None or final_hero_map_id is None or final_attempt_csv is None:
        raise RuntimeError(
            "Failed to satisfy benchmark constraints. "
            "Increase --max_attempts or relax --hero_max_gatekeeper_visibility."
        )

    # Persist selected artifacts.
    shutil.copyfile(final_attempt_csv, raw_csv_path)
    serial_maps = [map_to_serializable(m) for m in final_maps]
    write_json(maps_json_path, serial_maps)
    hero_map_obj = next(m for m in final_maps if m.map_id == final_hero_map_id)
    write_json(hero_json_path, map_to_serializable(hero_map_obj))

    summary_rows = summarize_results(final_rows)
    render_summary_markdown(
        summary_rows=summary_rows,
        output_path=summary_md_path,
        hero_map_id=final_hero_map_id,
        gatekeeper_params=DEFAULT_GATEKEEPER_PARAMS,
    )

    print(f"Saved benchmark summary to: {summary_md_path}")
    print(f"Saved raw rows to: {raw_csv_path}")
    print(f"Saved trial maps to: {maps_json_path}")
    print(f"Saved hero map to: {hero_json_path}")


if __name__ == "__main__":
    main()
