#!/usr/bin/env python3
"""Run matched, headless exploration benchmarks from YAML scenario files.

The suite deliberately reports physical collisions separately from controller
infeasibility and timeouts. This keeps the collision metric comparable across
attitude policies and prevents failed optimization from being labeled a crash.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exploration import ExplorationManager
from safe_control.utils import env
from scripts.utils import build_known_obs, load_config, parse_robot_specs


DEFAULT_SUITE = REPO_ROOT / "configs" / "benchmarks" / "headless_safety" / "suite.yaml"
PHYSICAL_COLLISION_TYPES = {"unknown", "inter_agent"}


def classify_outcome(
    success: bool,
    termination_reason: Optional[str],
    collision_info: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Classify mutually useful benchmark outcomes from manager diagnostics."""
    info = collision_info if isinstance(collision_info, dict) else {}
    collision_type = str(info.get("type", ""))
    collision = collision_type in PHYSICAL_COLLISION_TYPES
    infeasible = termination_reason == "collision_or_infeasible" and not collision
    timeout = termination_reason == "max_steps"
    return {
        "success": bool(success),
        "collision": collision,
        "infeasible": infeasible,
        "timeout": timeout,
        "collision_type": collision_type,
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _selected(values: Iterable[str], filters: Optional[list[str]]) -> list[str]:
    values = list(values)
    if not filters:
        return values
    wanted = {item.lower() for item in filters}
    selected = [item for item in values if item.lower() in wanted]
    missing = wanted - {item.lower() for item in selected}
    if missing:
        raise ValueError(f"Unknown selection(s): {sorted(missing)}; available={values}")
    return selected


def _scenario_paths(suite_path: Path, suite: dict[str, Any], filters: Optional[list[str]]) -> list[Path]:
    paths = [(suite_path.parent / item).resolve() for item in suite.get("scenarios", [])]
    if not paths:
        raise ValueError("The suite contains no scenarios")
    if not filters:
        return paths
    wanted = {item.lower() for item in filters}
    selected = [path for path in paths if path.stem.lower() in wanted or path.name.lower() in wanted]
    missing = wanted - {path.stem.lower() for path in selected} - {path.name.lower() for path in selected}
    if missing:
        raise ValueError(f"Unknown scenario(s): {sorted(missing)}")
    return selected


def _max_steps(cfg: Any, timeout_s: float, explicit_max_steps: Optional[int]) -> int:
    limits = [max(1, int(np.floor(float(timeout_s) / float(cfg.dt))))]
    if cfg.tf is not None:
        limits.append(max(1, int(np.floor(float(cfg.tf) / float(cfg.dt)))))
    if cfg.max_ticks is not None:
        limits.append(max(1, int(cfg.max_ticks)))
    if explicit_max_steps is not None:
        limits.append(max(1, int(explicit_max_steps)))
    return min(limits)


def run_case(
    config_path: Path,
    algorithm: str,
    attitude: str,
    seed: int,
    coverage_target: float,
    timeout_s: float,
    use_astar: bool,
    explicit_max_steps: Optional[int] = None,
) -> dict[str, Any]:
    np.random.seed(seed)
    cfg = load_config(str(config_path)).model_copy(deep=True)
    cfg.exploration_algorithm = algorithm
    cfg.controller.att = attitude
    cfg.show_animation = False

    known_obs = build_known_obs(cfg.environment)
    if known_obs is None:
        known_obs = np.empty((0, 7), dtype=float)
    unknown_obs = np.asarray(cfg.environment.unknown_obstacles, dtype=float)
    if unknown_obs.size == 0:
        unknown_obs = np.empty((0, 3), dtype=float)

    robot_specs = parse_robot_specs(cfg)
    for spec in robot_specs:
        spec["unknown_obs_persistent_fov"] = True
        spec["visibility_violation_mode"] = "point_mass"

    x0s = [np.asarray(instance.x0, dtype=float) for instance in cfg.robots.instances]
    env_handler = env.Env(
        width=float(cfg.environment.width),
        height=float(cfg.environment.height),
        known_obs=known_obs,
        resolution=float(cfg.environment.resolution),
    )
    manager = ExplorationManager(
        x0s,
        robot_specs,
        {"pos": cfg.controller.pos, "att": attitude},
        exploration_algorithm=algorithm,
        dt=float(cfg.dt),
        show_animation=False,
        save_animation=False,
        env_handler=env_handler,
        known_obs=known_obs,
        unknown_obs=unknown_obs,
        use_astar_waypoints=use_astar,
        coverage_target=coverage_target,
    )

    max_steps = _max_steps(cfg, timeout_s=timeout_s, explicit_max_steps=explicit_max_steps)
    started = time.perf_counter()
    try:
        success = bool(manager.explore(max_steps=max_steps))
        wallclock_s = time.perf_counter() - started
        collision_info = manager.last_collision_info if isinstance(manager.last_collision_info, dict) else {}
        outcome = classify_outcome(success, manager.last_termination_reason, collision_info)
        violations = [int(controller.robot.unsafe_point_count) for controller in manager.controller_list]
        result = {
            "scenario": config_path.stem,
            "config": str(config_path),
            "seed": int(seed),
            "robots": len(robot_specs),
            "algorithm": algorithm,
            "attitude": attitude,
            **outcome,
            "termination_reason": str(manager.last_termination_reason),
            "coverage": float(manager.last_coverage_ratio),
            "steps": int(manager.last_step_count),
            "sim_time_s": float(manager.last_sim_time),
            "wallclock_s": float(wallclock_s),
            "violations_total": int(sum(violations)),
            "violations_per_robot": json.dumps(violations),
            "collision_info": json.dumps(collision_info, sort_keys=True),
            "error": "",
        }
        return result
    finally:
        try:
            manager.close()
        except Exception:
            pass


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["scenario"], row["algorithm"], row["attitude"], int(row["robots"]))
        groups[key].append(row)

    output: list[dict[str, Any]] = []
    for (scenario, algorithm, attitude, robots), group in sorted(groups.items()):
        successful_times = [float(row["sim_time_s"]) for row in group if row["success"]]
        output.append(
            {
                "scenario": scenario,
                "robots": robots,
                "algorithm": algorithm,
                "attitude": attitude,
                "trials": len(group),
                "success_rate": float(np.mean([row["success"] for row in group])),
                "collision_rate": float(np.mean([row["collision"] for row in group])),
                "infeasible_rate": float(np.mean([row["infeasible"] for row in group])),
                "timeout_rate": float(np.mean([row["timeout"] for row in group])),
                "mean_violations_per_robot": float(
                    np.mean([row["violations_total"] / max(robots, 1) for row in group])
                ),
                "mean_time_success_s": float(np.mean(successful_times)) if successful_times else None,
                "mean_coverage": float(np.mean([row["coverage"] for row in group])),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary_rows: list[dict[str, Any]], suite_name: str) -> None:
    lines = [
        f"# {suite_name}",
        "",
        "Collision means geometric contact with an unknown obstacle or another robot. "
        "Infeasibility and timeout are reported separately.",
        "",
        "| Scenario | R | Algorithm | Attitude | N | Success | Collision | Infeasible | Timeout | Violations / robot | Time, successes [s] | Coverage |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        success_time = "N/A" if row["mean_time_success_s"] is None else f"{row['mean_time_success_s']:.2f}"
        lines.append(
            f"| {row['scenario']} | {row['robots']} | {row['algorithm']} | {row['attitude']} | "
            f"{row['trials']} | {row['success_rate']:.2f} | {row['collision_rate']:.2f} | "
            f"{row['infeasible_rate']:.2f} | {row['timeout_rate']:.2f} | "
            f"{row['mean_violations_per_robot']:.2f} | {success_time} | {row['mean_coverage']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched YAML exploration benchmarks headlessly.")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "headless_safety_benchmark")
    parser.add_argument("--scenario", action="append", help="Scenario stem/name to include; repeat as needed.")
    parser.add_argument("--algorithm", action="append", help="Algorithm to include; repeat as needed.")
    parser.add_argument("--attitude", action="append", help="Attitude policy to include; repeat as needed.")
    parser.add_argument("--repeats", type=int, help="Override repeats from the suite.")
    parser.add_argument("--max-steps", type=int, help="Optional short cap for smoke testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_path = args.suite.resolve()
    suite = _read_yaml(suite_path)
    scenarios = _scenario_paths(suite_path, suite, args.scenario)
    algorithms = _selected(suite.get("algorithms", []), args.algorithm)
    attitudes = _selected(suite.get("attitudes", []), args.attitude)
    repeats = int(args.repeats if args.repeats is not None else suite.get("repeats", 1))
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    timeouts = {int(key): float(value) for key, value in suite.get("timeouts_s", {}).items()}
    master_seed = int(suite.get("master_seed", 0))
    coverage_target = float(suite.get("coverage_target", 0.98))
    use_astar = bool(suite.get("use_astar", True))

    rows: list[dict[str, Any]] = []
    total = len(scenarios) * len(algorithms) * len(attitudes) * repeats
    run_index = 0
    for scenario_index, config_path in enumerate(scenarios):
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        robot_count = len(load_config(str(config_path)).robots.instances)
        if robot_count not in timeouts:
            raise ValueError(f"No timeout configured for R={robot_count}")
        for repeat in range(repeats):
            seed = master_seed + scenario_index * 10_000 + repeat
            for algorithm in algorithms:
                for attitude in attitudes:
                    run_index += 1
                    print(
                        f"[{run_index}/{total}] scenario={config_path.stem} R={robot_count} "
                        f"algorithm={algorithm} attitude={attitude} seed={seed}",
                        flush=True,
                    )
                    try:
                        row = run_case(
                            config_path=config_path,
                            algorithm=algorithm,
                            attitude=attitude,
                            seed=seed,
                            coverage_target=coverage_target,
                            timeout_s=timeouts[robot_count],
                            use_astar=use_astar,
                            explicit_max_steps=args.max_steps,
                        )
                    except Exception as exc:
                        row = {
                            "scenario": config_path.stem,
                            "config": str(config_path),
                            "seed": int(seed),
                            "robots": robot_count,
                            "algorithm": algorithm,
                            "attitude": attitude,
                            "success": False,
                            "collision": False,
                            "infeasible": False,
                            "timeout": False,
                            "collision_type": "",
                            "termination_reason": "exception",
                            "coverage": 0.0,
                            "steps": 0,
                            "sim_time_s": 0.0,
                            "wallclock_s": 0.0,
                            "violations_total": 0,
                            "violations_per_robot": "[]",
                            "collision_info": "{}",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        print(f"  ERROR: {row['error']}", flush=True)
                    rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize(rows)
    _write_csv(args.output_dir / "raw_results.csv", rows)
    _write_csv(args.output_dir / "summary.csv", summary_rows)
    _write_markdown(
        args.output_dir / "summary.md",
        summary_rows,
        suite_name=str(suite.get("suite_name", "benchmark")),
    )
    print(f"Wrote benchmark outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
