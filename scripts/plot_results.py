import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import yaml
import matplotlib.pyplot as plt

def _load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _plot_circle(ax, x, y, r, dashed=False):
    circ = plt.Circle((x, y), r, fill=False, linestyle="--" if dashed else "-")
    ax.add_patch(circ)

def _plot_superellipse_outline(ax, cx, cy, a, b, e, theta, n=200):
    """
    Draws an outline of the 2D superellipse:
        (|x|/a)^e + (|y|/b)^e = 1
    rotated by theta and translated to (cx, cy).
    """
    t = np.linspace(0, 2*np.pi, n)
    # Parametric form for superellipse (Lamé curve)
    # Use sign-preserving power to avoid complex values
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    x = a * np.sign(cos_t) * (np.abs(cos_t) ** (2.0 / e))
    y = b * np.sign(sin_t) * (np.abs(sin_t) ** (2.0 / e))

    # Rotate by theta
    ct, st = np.cos(theta), np.sin(theta)
    xr = ct * x - st * y
    yr = st * x + ct * y

    ax.plot(cx + xr, cy + yr)

def overlay_obstacles_from_yaml(ax, yaml_path: str):
    cfg = _load_yaml(yaml_path)
    env = cfg.get("environment", {})

    # Known circles
    for c in env.get("circles", []) or []:
        if len(c) >= 3:
            _plot_circle(ax, c[0], c[1], c[2], dashed=False)

    # Unknown obstacles (circles)
    for u in env.get("unknown_obstacles", []) or []:
        if len(u) >= 3:
            _plot_circle(ax, u[0], u[1], u[2], dashed=True)

    # Superellipsoids (superellipses)
    for s in env.get("superellipsoids", []) or []:
        if len(s) >= 6:
            cx, cy, a, b, e, theta = s[:6]
            _plot_superellipse_outline(ax, cx, cy, a, b, e, theta)

# def _split_case_and_ctrl(exp_name: str):
#     # expects names like "EXP_01_blind_corner_gatekeeper" or "..._visibilityarea"
#     m = re.match(r"^(.*)_(gatekeeper|visibilityarea)$", exp_name)
#     if m:
#         return m.group(1), m.group(2)
#     return exp_name, "unknown"

def generate_comparison_plots(csv_path="logs/comparison_results.csv"):
    df = pd.read_csv(csv_path)
    
    # Create 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    sns.set_theme(style="whitegrid")

    # 1. Total Collisions
    sns.barplot(data=df, x="experiment", y="collisions", hue="att_ctrl", ax=ax1)
    ax1.set_title("Total Safety Violations", fontweight='bold')

    # 2. Exploration at Failure (The "EUC/AUC" Metric)
    # This shows how much they explored before the first crash
    sns.barplot(data=df, x="experiment", y="first_collision_expl", hue="att_ctrl", ax=ax2)
    ax2.set_ylim(0, 105)
    ax2.set_title("Exploration % at First Failure", fontweight='bold')
    ax2.set_ylabel("Explored Area %")

    # 3. Efficiency (Sim Time)
    sns.barplot(data=df, x="experiment", y="sim_time", hue="att_ctrl", ax=ax3)
    ax3.set_title("Mission Completion Time", fontweight='bold')

    plt.suptitle("Controller Comparison: Gatekeeper vs. Visibility Area", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("detailed_comparison.png")
    plt.show()

def plot_trajectories_from_pose_log(
    pose_csv="logs/pose_log.csv",
    experiments=None,
    controllers=None,
    downsample=1,
    save_path="trajectories.png",
    obstacle_yaml_by_experiment=None,  # NEW: { "blind_corner_test": "tests/test_1_blind_corner_gatekeeper.yaml", ... }
):
    df = pd.read_csv(pose_csv)

    if experiments is not None:
        df = df[df["experiment"].isin(experiments)]
    if controllers is not None:
        df = df[df["att_ctrl"].isin(controllers)]
    if downsample > 1:
        df = df[df["frame"] % downsample == 0]

    if df.empty:
        print("No pose data found for the requested filter.")
        return

    exp_list = sorted(df["experiment"].unique())
    ctrl_list = sorted(df["att_ctrl"].unique())

    fig, axes = plt.subplots(
        len(exp_list), len(ctrl_list),
        figsize=(6 * len(ctrl_list), 5 * len(exp_list)),
        squeeze=False
    )

    for r, exp in enumerate(exp_list):
        for c, ctrl in enumerate(ctrl_list):
            ax = axes[r][c]
            sub = df[(df["experiment"] == exp) & (df["att_ctrl"] == ctrl)]

            if sub.empty:
                ax.set_title(f"{exp} | {ctrl} (no data)")
                ax.axis("off")
                continue

            # Obstacle overlay (optional)
            if obstacle_yaml_by_experiment and exp in obstacle_yaml_by_experiment:
                overlay_obstacles_from_yaml(ax, obstacle_yaml_by_experiment[exp])

            for robot_id, g in sub.groupby("robot_id"):
                g = g.sort_values("frame")
                ax.plot(g["x"].to_numpy(), g["y"].to_numpy(), label=f"robot {robot_id}")
                ax.scatter([g["x"].iloc[0]], [g["y"].iloc[0]], marker="o")
                ax.scatter([g["x"].iloc[-1]], [g["y"].iloc[-1]], marker="x")

            ax.set_title(f"{exp} | {ctrl}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()



if __name__ == "__main__":

#     plot_trajectories_grouped(
#     cases=["EXP_01_blind_corner", "EXP_02_narrow_gateway"],
#     downsample=2,
#     save_path="traj_core.png"
# )
    plot_trajectories_from_pose_log(
        experiments=["blind_corner_test"],
        controllers=["gatekeeper", "visibility_area"],
        downsample=2,
        save_path="traj_blind_corner.png"
    )

    generate_comparison_plots()

# plot_trajectories_from_pose_log(
#     experiments=["blind_corner_test"],
#     controllers=["gatekeeper", "visibility_area"],
#     downsample=2,
#     save_path="traj_blind_corner.png",
#     obstacle_yaml_by_experiment={
#         "blind_corner_test": "tests/test_1_blind_corner_gatekeeper.yaml"
#     }
# )