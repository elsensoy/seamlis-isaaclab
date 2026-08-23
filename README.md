 


<a id="top"></a>

# SEAMLIS

`seamlis` is a **safe exploration and mapping framework** utilizing [`safe_control`](https://github.com/tkkim-robot/safe_control) for 2D planning and **Isaac Lab / Isaac Sim** for photorealistic, GPU-accelerated 3D simulation.

It features a modular architecture that bridges high-level CBF-based planners with low-level physics simulation, all running inside a containerized environment with full GUI support.

## Explore the Documentation

Use these links to jump directly to the part of the project you want to explore:

| Section | What you will find |
| --- | --- |
| [Isaac Lab implementation](#core-logic) | How the planar exploration stack, safety controllers, adapter, and 3D simulation work together. |
| [System overview](#system-overview) | How the planner, Isaac simulation, and adapter work together. |
| [Assets](#assets) | Drone and environment asset information. |
| [Installation and running](INSTALLATION_AND_RUNNING.md) | Host setup, Docker, Isaac Lab, simulation commands, and troubleshooting. |
| [Architecture](#architecture) | Planner, simulation, and adapter code responsibilities. |
| [Experiment configuration](#configuration) | YAML configuration structure and controller selection. |
| [Outputs and logging](#outputs) | Visualizations, data logs, collision summaries, and detection logs. |
| [Experiments](#experiments) | Test scenarios, their purpose, and direct links to every YAML configuration. |
| [Testing](#testing) | Headless safety benchmarks and drone smoke tests. |
| [Headless benchmark guide](configs/benchmarks/headless_safety/README.md) | Scenarios, filters, metrics, repeats, and result files. |
| [Coordinate validation](#coordinate-validation) | Planner-to-Isaac alignment checks and troubleshooting. |
| [Warehouse asset](#warehouse-asset) | Baked warehouse asset details. |

<a id="core-logic"></a>

## Isaac Lab Implementation Overview

This repository provides an **Isaac Lab implementation and experiment harness** for
SEAMLiS. The paper contains the formal safety definitions, controller construction, and
guarantees; this README focuses on how those components are configured, executed, and
evaluated in simulation.

The implementation connects an existing planar exploration and control stack to a 3D Isaac
Sim scene. The planner remains responsible for mapping, frontier selection, goal assignment,
and safe planar control. Isaac Lab supplies the drone and environment assets, simulation
loop, sensor and FoV visualization, cameras, and GUI or headless execution.

### From the planner to Isaac Lab

At a high level, data moves through the repository as follows:

`YAML scenario -> ExplorationManager -> safety controllers -> ExploreAdapter -> Isaac stage -> logs and metrics`

1. A YAML file defines the environment, robot limits and sensors, exploration strategy,
   controller choices, and run duration.
2. `ExplorationManager` advances the 2D exploration state, updates sensing and frontiers,
   assigns goals, and invokes the configured position and attitude controllers.
3. `ExploreAdapter` synchronizes the planar robot position and yaw with the corresponding
   Crazyflie asset and FoV visualization in the Isaac stage.
4. The runner advances the simulation, renders the scene when requested, and records the
   outputs used by the experiment and headless benchmark tools.

The current adapter synchronizes the planner state to the 3D drone representation; it does
not model motor-level quadrotor flight dynamics. This keeps visual and headless runs tied to
the same exploration and safety-controller implementation.

### Configurable controller interfaces

The safety components are exposed as controller choices instead of being hard-coded into a
single experiment. Their theory is described in the paper; in this repository they enter
the simulation through these YAML interfaces:

| Command channel | Configuration | Implementation role |
| --- | --- | --- |
| Position / acceleration | `controller.pos: "mpc_cbf"` or `"cbf_qp"` | Filters the planar motion command using the selected CBF controller. |
| Yaw / sensor heading | `controller.att: "gatekeeper"` | Runs the nominal and backup attitude policies used by the visibility monitor. |
| Goal allocation | `exploration_algorithm` | Selects the configured frontier-assignment strategy. |

Controller horizons, gains, sensing limits, and robot motion limits are supplied by the
experiment configuration, which makes the same scene reusable across controller variants
and parameter sweeps.

### Implementation map

| Responsibility | Main entry point |
| --- | --- |
| 2D exploration, frontiers, goals, and control loop | `exploration.py` (`ExplorationManager`) |
| Position and attitude controllers | `safe_control/` |
| Planner-to-stage state synchronization | `scripts/adapter.py` (`ExploreAdapter`) |
| Isaac scene, drone assets, and simulation loop | `scripts/run_isaac.py` and `scripts/scene.py` |
| Scenario and benchmark definitions | `configs/` |

Continue with the [system overview](#system-overview) for the layer boundaries, the
[architecture](#architecture) for file-level responsibilities, or the
[experiments](#experiments) for runnable scenarios and benchmark configurations.

<a id="system-overview"></a>

## High-Level System: Two Layers + Adapter

This project is structured as **two coupled layers**:

1. **Planner Layer (2D, logic)**
   - Implemented by `ExplorationManager`
   - Operates over a 2D workspace (`env_handler`) containing:
     - workspace size: `width`, `height`
     - map resolution: `resolution`
     - known obstacles: circles and superellipsoids
     - unknown obstacles: used for evaluation / safety checks

2. **Simulation Layer (3D, physics + rendering)**
   - Isaac Lab / Isaac Sim stage (`USD stage`)
   - Contains:
     - environment asset (warehouse USD)
     - robots (Crazyflie articulations)
     - visual FoV meshes
     - optional RTX LiDAR sensors
     - obstacle visualization as USD primitives

3. **The Adapter Layer (bridge)**
   - `scripts/adapter.py` (ExploreAdapter)
   - Synchronizes planner state <-> simulation state:
     - pushes planner outputs to simulated robot pose
     - optionally rotates FoV for visualization
     - reads robot states from sim for planner updates

---

<a id="assets"></a>

## Assets

We use the dev quad custom design from DASC lab. For patching details CAD->USD, after redesigning existing hierarchy on blender, we patch the usd for spawning. For more details, see tools/patch_drone_usd.py. The environment asset used for the 3D backdrop is documented separately below, in [Warehouse asset](#warehouse-asset).

https://github.com/user-attachments/assets/ff7f51d1-c830-4621-8874-ddbb953d71d8

<a id="warehouse-asset"></a>

### Warehouse asset

We use NVIDIA's stock Isaac Sim demo warehouse (`Isaac/Environments/Simple_Warehouse/warehouse.usd`, streamed from the Omniverse content-library S3 bucket by URL, not vendored into this repo) as the visual backdrop for the 3D scene.

* **Purely cosmetic, not a collision surface.** The warehouse mesh has no physics collision enabled (`enable_warehouse_collision()` in `scripts/scene.py` exists but is left disabled) -- robots are kinematically driven by the planner/adapter, and the actual safety constraints (obstacle radii, wall margins) come from the 2D `environment:` grid in the YAML config, not from the warehouse geometry. The warehouse is fit around that grid, never the reverse.
* **Auto-fit scale and orientation.** `scripts/run_isaac.py`'s `_warehouse_fit()` measures the asset's real interior footprint (24m x 36m, taken directly from the wall/floor prim bounding boxes) and tries both the native orientation and a 90-degree-rotated one, keeping whichever leaves less unused floor space around the configured `environment.width` x `environment.height` grid. A small manual scale boost (1.08x) plus solved per-side margins (`WAREHOUSE_MARGIN_X/Y`, `MIN_LEFT_MARGIN`, `MIN_BOTTOM_MARGIN`) then guarantee the grid sits fully inside the walls with clearance on every side.
* **Cleanup pass.** On load, `WarehouseScene._cleanup_env()` hides pillars, posts, ceiling panels, roof trusses, and light fixtures (matched by name pattern and by a bounding-box height heuristic) so they don't block the top-down camera or clutter the shelving/racking that's deliberately kept as background obstacle context.
* **Skippable.** Passing `--no-warehouse` to `scripts/run_isaac.py` skips loading it entirely and falls back to the default flat grid ground plane -- used for the drone asset-showcase recording above, where the warehouse would only be a distraction from the asset itself.

<a id="installation"></a>

## Installation and Running

Setup requirements, Docker and Isaac Lab installation, container startup,
simulation commands, headless benchmark commands, and troubleshooting are in
[Installation and Running](INSTALLATION_AND_RUNNING.md).

<a id="architecture"></a>

## Architecture

The system is split into two distinct layers bridged by an Adapter.

### 1. Planner Layer (`safe_control`)

* **Space:** 2D Configuration Space ().
* **Logic:** Uses `ExplorationManager` to calculate frontiers and safe control inputs.
* **Controllers:**
* **Position:** MPC-CBF, CBF-QP.
* **Attitude:** Gatekeeper, Visibility Area.

### 2. Simulation Layer (Isaac Lab)

* **Space:** 3D Euclidean Space ().
* **Physics:** High-fidelity physics stepping, contact dynamics, and sensor rendering (RTX Lidar).
* **Assets:** Procedurally generates obstacles in 3D to match the 2D planner configuration.

### 3. The Adapter (`scripts/adapter.py`)

Synchronizes the two layers:

* **Time Buffering:** Accumulates high-freq physics steps (60Hz) to trigger low-freq planning steps (20Hz).
* **State Authority:** "Force Writes" planned poses to the simulation while respecting collision physics.

---

<a id="configuration"></a>

## Configuration (`configs/*.yaml`)

Experiments are defined via YAML.

```yaml
experiment:
  name: "gatekeeper_demo"
  dt: 0.05
  exploration_algorithm: "coscan"

environment:
  width: 10
  height: 10
  circles: [[5, 5, 0.5]] # [x, y, radius]

robots:
  common:
    model: "DoubleIntegrator2D"
    sensor: "sector"
  instances:
    - { robot_id: 0, x0: [1.0, 1.0, 0.0] }

controller:
  pos: "mpc_cbf"
  att: "gatekeeper"

```

---

<a id="outputs"></a>

## Outputs & Logging

### 1. Visualization


* **Isaac Viewport:** Shows real-time 3D drone behavior. The camera cycles automatically
  every 100 frames through a top-down birdview and a tracking shot of each robot in turn
  (`global -> robot 0 -> robot 1 -> ... -> global -> ...`), so a long `--viz kit` run
  periodically returns to the birdview instead of only showing it once at startup.
* **Trajectory traces:** Each robot's followed path is drawn as a persistent, semi-transparent
  colored line in the Isaac viewport, matching that robot's FoV mesh color -- useful for
  screenshots/video without needing the Matplotlib UI. Purely cosmetic; only renders when the
  full Kit viewport extensions are loaded (`--viz kit`), so it's automatically skipped (not an
  error) on headless runs. Controlled via `ExploreAdapter(..., draw_trails=True)` in
  `scripts/run_isaac.py`; see `ExploreAdapter.clear_trails()` to reset it mid-session.
* **Exploration UI:** Shows the internal map belief, frontiers, and sensing wedges (`--explore-ui`).

### 2. Data Logs

* **`logs/comparison_results.csv`**: Master summary of all runs (collisions, time, success rate).
  Written by `ExperimentLogger`, which is **disabled by default** (see `logger = None` in
  `scripts/run_isaac.py`) -- enable it there if you need frame-level pose logging too.
* **`logs/<experiment_name>/pose_log.csv`**: Frame-by-frame trajectory data, also gated behind
  `ExperimentLogger`.

### 3. Collision & Detection Summary (`logs/collision_summary.csv`)

The main log for **comparing algorithm robustness across runs**. One row is appended per run,
independent of `ExperimentLogger` above, so it's written for every run -- headless or `--viz
kit` -- with no extra setup. Safe to run many experiments back to back and load the whole file
into a spreadsheet or pandas afterward.

| Column | Meaning |
| --- | --- |
| `timestamp` | UTC time the run finished (ISO 8601). |
| `algorithm` | `experiment.exploration_algorithm` from the config (e.g. `Frontier`, `CoScan`). |
| `num_drones` | Number of robot instances in the run. |
| `known_obstacles` | Count of known (map-given) obstacles. |
| `unknown_obstacles` | Count of hidden obstacles the robots must discover. |
| `unknown_obstacles_detected_pct` | % of `unknown_obstacles` seen by *at least one* robot by run end -- aggregated across all robots' `detected_unknown_obs_memory` and matched against the config's ground-truth obstacle positions. |
| `lidar_enabled` | Whether `--use-rtx-lidar` was passed. |
| `collisions` | Total collision count across all robots (`ExploreAdapter.total_collisions`). |
| `collisions_per_robot` | Per-robot breakdown, e.g. `[0, 1, 0]`. |
| `success` | Whether the run reached `exploration_complete()` (`True`) or was cut off by `max_ticks` (`False`). |

Example row:

```csv
timestamp,algorithm,num_drones,known_obstacles,unknown_obstacles,unknown_obstacles_detected_pct,lidar_enabled,collisions,collisions_per_robot,success
2026-08-21T18:02:35+00:00,Frontier,3,8,6,66.7,False,1,"[0, 1, 0]",True
```

`logs/` is git-ignored, so this file accumulates locally across your experiment sweep and
won't get committed by accident.

---

<a id="experiments"></a>

## Experiments

The YAML files under `configs/` define repeatable environments, robot starting
states, sensing limits, exploration algorithms, and controller selections. Use
the tables below to choose a scenario before running it. The **Media** column is
reserved for videos or GIFs of representative runs.

For controlled comparisons, keep the environment, starting states, timing, and
robot limits identical and change only the controller or algorithm being tested.
The paired filenames below identify the intended variants; verify their YAML
fields before treating older files as a strict A/B comparison.

### Recorded videos

The tables above are the quick-reference source of truth for what each run
actually did; this section is just for watching them. Entries not yet
uploaded fall back to a plain link to the file in the repo.

#### Headless benchmark scenarios

**Decentralized hidden obstacle**

https://github.com/user-attachments/assets/382b98c3-9224-49d7-bd3a-1f2243372973

#### Core controller experiments

**EXP_01 Sparse forest -- Gatekeeper**

https://github.com/user-attachments/assets/4d475aaf-f3be-4449-9e48-c280188a8de6

**EXP_01 Sparse forest -- Visibility area**

https://github.com/user-attachments/assets/b1c2ea35-c483-46d7-9f44-7784dde046d1

**EXP_02 Narrow gateway -- Gatekeeper**

https://github.com/user-attachments/assets/3bec8ebd-9545-4577-98ec-9fc47badb6f2

**EXP_02 Narrow gateway -- Visibility area**

https://github.com/user-attachments/assets/802324e5-6484-4562-8a35-518044c8ea28

#### Multi-robot stress test (STRESS_01)

**Gatekeeper**

https://github.com/user-attachments/assets/2e38ff99-cc38-443d-b7b3-32e8f2d1c710

**Visibility area**

https://github.com/user-attachments/assets/b0485d58-8fe1-48b9-b580-fce7b974eb8b

#### Validation experiments

**VAL_01 Start in contact**

https://github.com/user-attachments/assets/13e2e7bb-ff4a-4fb4-9b6c-547be2ae8965

#### Small-FoV comparison

**Gatekeeper**

https://github.com/user-attachments/assets/f248b545-c11a-414d-bb76-2d80de6731c2

**Visibility area**

https://github.com/user-attachments/assets/838e1d51-6f59-4625-895e-8f2b7b9f86bc

#### Known-obstacle and sensor ablations

**Visibility-area RGB-D**

Pending upload -- [file in repo](media/experiments/exp04_known_obstacles_5robots_visibility_area.mp4)

**Gatekeeper RGB-D**

https://github.com/user-attachments/assets/96f13270-0b98-4c6d-b46e-45a7a02932c9

**Gatekeeper LiDAR**

https://github.com/user-attachments/assets/05056e07-8174-42cd-a595-ee3aaaf40ea7

### Example configurations

Use these as starting points when authoring a new experiment:

- [Indoor exploration example](configs/example_1.yaml)
- [Three-robot indoor stress example](configs/example_test_indoor.yaml)


<a id="testing"></a>

### Headless benchmark scenarios

These configurations are used by `examples/benchmark_config_suite.py`. The
runner applies both `Frontier` and `CoScan` and overrides the attitude policy
with `simple`, `visibility_area`, `velocity_tracking_yaw`, or `gatekeeper`.

| Scenario | Robots | Obstacles (known+unknown) | Feature under test | YAML | Result | Media |
| --- | ---: | ---: | --- | --- | --- | --- |
| Blind corner | 1 | 6+3=9 | Occlusion and late discovery of a hidden obstacle immediately beyond a turn. | [Configuration](configs/benchmarks/headless_safety/blind_corner.yaml) | Not re-verified this session | [Video](media/experiments/exp06_headless_safety_blind_corner.mp4) |
| Decentralized hidden obstacle | 2 | 8+4=12 | Robots with different local obstacle knowledge and no shared obstacle map. | [Configuration](configs/benchmarks/headless_safety/decentralized_hidden_obstacle.yaml) | 0 collisions, both robots succeeded (re-recorded this session) | [Video](media/experiments/exp05_decentralized.mp4) |

The complete matrix is defined in [suite.yaml](configs/benchmarks/headless_safety/suite.yaml),
and its metrics and filters are documented in the
[headless benchmark guide](configs/benchmarks/headless_safety/README.md).

### Core controller experiments

These scenarios compare Gatekeeper and visibility-area attitude control across
increasingly difficult exploration conditions.

| ID | Experiment | Robots | Obstacles (known+unknown) | Feature under test | Gatekeeper YAML | Visibility-area YAML | Result | Media |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| EXP_01 | Sparse forest | 2 | 0+15=15 | Long-horizon exploration and controller feasibility in dense hidden-obstacle constraints. | [Gatekeeper](configs/test_4_sparse_forest.yaml) | [Visibility area](configs/test_4_sparse_forest_visibility_area.yaml) | Both variants: 1 collision (robot 0, falls and stays down); robot 1 continues exploring | [Gatekeeper](media/experiments/exp01_sparse_forest_gatekeeper.mp4), [Visibility area](media/experiments/exp01_sparse_forest_visibility_area.mp4) |
| EXP_02 | Narrow gateway | 2 | 5+4=9 | Multi-robot interaction, reciprocal avoidance, and possible deadlock in a constrained passage. | [Gatekeeper](configs/test_2_narrow_gateway.yaml) | [Visibility area](configs/test_2_narrow_gateway_visibility_area.yaml) | Both variants: 0 collisions, 100% of unknown obstacles detected, full 5000-tick run completed | [Gatekeeper](media/experiments/exp02_narrow_gateway_gatekeeper.mp4), [Visibility area](media/experiments/exp02_narrow_gateway_visibility_area.mp4) |

> EXP_01's videos were recorded at a shortened `max_ticks` (800 instead of the file's 5000) to keep
> recording time reasonable -- a representative clip, not a full run to exploration-complete.
> EXP_02's videos run the full 5000-tick file, sped up in playback so the whole run is still a
> reasonably short clip to watch.
>
> EXP_02's wall is a line of circles (radius 0.7, spaced 1.4 apart so they touch), alternating known
> (mapped in advance) and unknown (must be detected) obstacles, with a single 0.7m gap at the
> vertical center -- just wide enough for a 0.25-radius robot (0.5m diameter) to squeeze through.
> Watching this run surfaced a real bug in `scripts/run_isaac.py`'s warehouse-fit code:
> `WAREHOUSE_SHIFT_Y` (a rigid post-anchor shift meant to trade left/bottom surplus margin for more
> right/top clearance) skewed the top/bottom split to 3.0m/0.5m instead of the intended "both sides
> comfortable" -- a robot exploring near the bottom edge could end up flush against the warehouse's
> actual 9m-tall wall, only 0.5m past the boundary (confirmed visually: the drone's mesh sitting
> right at the wall-floor seam). Fixed by reducing the shift so top/bottom land at ~1.75m each off
> the same ~3.5m of combined Y slack this map size has; verified with a quick partial re-recording
> before committing to the full run.
>
> EXP_01 (sparse forest, config file still named `test_4_sparse_forest*.yaml` -- only the
> showcase ID here was renumbered)'s videos were re-recorded twice. First, after the crash-and-stay
> fix below: robot 0 now hits an unknown obstacle once and stays down, instead of the old
> bump-and-recover animation letting it climb back up and re-collide with it repeatedly (12 times
> in a row, in the original version of this video). That re-recording still crashed right next to
> warehouse shelving though, because both robots spawned at (2,2)/(2,10) -- the same map corner
> already flagged as camera-unfriendly in `configs/blind_corner_isaac.yaml`. So both spawns were
> changed to start from the center of the map (9.5, 9.0) and (10.5, 9.0), 1m apart, facing opposite
> directions -- clear of any corner, and a cleaner "two robots explore outward from the middle"
> setup than the original corner-to-corner one.

### Multi-robot stress test

A denser scale-up of the core controller experiments above: 5 robots and 20
obstacles (12 known -- 10 circles and 2 wall segments -- plus 8 unknown) in a
26x20 map, comparing Gatekeeper and visibility-area under the same layout.

| ID | Experiment | Robots | Obstacles (known+unknown) | Feature under test | Gatekeeper YAML | Visibility-area YAML | Result | Media |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| STRESS_01 | 5-robot / 20-obstacle | 5 | 12+8=20 | Controller scaling: five robots exploring concurrently through a dense known+unknown obstacle field. | [Gatekeeper](configs/stress_5robots_gatekeeper.yaml) | [Visibility area](configs/stress_5robots_visibility_area.yaml) | Both variants: 0 collisions across all 5 robots (Gatekeeper detected 50% of unknown obstacles, visibility-area 87.5%) | [Gatekeeper](media/experiments/stress_5robots_gatekeeper.mp4), [Visibility area](media/experiments/stress_5robots_visibility_area.mp4) |

> Recorded as a scaling/safety demonstration rather than a collision-vs-success comparison, since
> neither controller failed under this particular layout.

### Validation experiments

Validation files test the collision instrumentation and the numerical boundary
between a safe near miss and a collision.

| ID | Experiment | Robots | Obstacles (known+unknown) | Expected use | YAML | Result | Media |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| VAL_01 | Start in contact | 5 | 5+5=10 | Confirms that an initial overlap is detected immediately rather than after movement. | [Configuration](configs/test_5_start_in_contact.yaml) | 1 collision (robot 0, at tick 0, by design) | [Video](media/experiments/val01_start_in_contact.mp4) |

> VAL_01's recording variant is [`configs/val01_start_in_contact_isaac.yaml`](configs/val01_start_in_contact_isaac.yaml)
> (a 5-robot layout; robots 1-4 and the extra known/unknown obstacles are shared scenery, added
> since this check fires at tick zero and isn't sensitive to what else is in the scene). Robot 0 is
> spawned 0.75m from the obstacle center (radius 0.6 + robot radius 0.25 = 0.85m collision
> threshold, so 0.75m is comfortably inside it) rather than at the exact same point as the obstacle
> -- landing the drone mesh exactly on the obstacle's center looked like a rendering glitch (the
> drone teleported inside solid geometry) rather than a collision. Facing directly at the obstacle
> so it reads as "just ran into it nose-first" instead of an arbitrary orientation.
>
> VAL_02A/VAL_02B (near-miss pass/fail) aren't recorded at all: retrofitting 5 robots into those
> broke the precise 0.34-vs-0.36m margin they're built to demonstrate -- two otherwise-identical
> headless runs of the same 5-robot layout produced completely different early trajectories for
> robot 0, since CoScan's goal/frontier assignment isn't deterministic enough across robot counts
> to keep guaranteeing which obstacle it grazes and by how much.

### Small-FoV comparison

This focused pair compares how Gatekeeper and visibility-area yaw behave around
unknown obstacles when sensing coverage is deliberately narrow.

| Experiment | Robots | Obstacles (known+unknown) | YAML | Result | Media |
| --- | ---: | ---: | --- | --- | --- |
| Gatekeeper with unknown obstacles and small FoV | 5 | 5+6=11 | [Configuration](configs/test_10_unknown_smallfov_gatekeeper.yaml) | 1 collision (robot 0, into the unknown obstacle near (6,6)); 66.7% of unknown obstacles detected | [Video](media/experiments/exp03_smallfov_5robots_gatekeeper.mp4) |
| Visibility-area control with unknown obstacles and small FoV | 5 | 5+6=11 | [Configuration](configs/test_11_visibilityarea_unknown_smallfov.yaml.yaml) | 0 collisions across all 5 robots; 66.7% of unknown obstacles detected | [Video](media/experiments/exp03_smallfov_5robots_visibility_area.mp4) |

> Videos captured from the 5-robot `smallfov_5robots_gatekeeper_isaac.yaml` /
> `smallfov_5robots_visibility_area_isaac.yaml` variants (same narrow 40deg FoV and the original 2
> unknown obstacles, plus 5 known and 4 more unknown obstacles shared with the other _isaac variants
> recorded alongside these). **Gatekeeper is the one that collided here**, while visibility-area
> completed with 0 collisions across all 5 robots -- worth noting since Gatekeeper's dual
> nominal+backup solve is generally the more conservative controller elsewhere in this README (e.g.
> the Core controller experiments note above). Not reliably reproducible given the goal-assignment
> non-determinism noted under VAL_01 -- a re-run could come out differently.

### Known-obstacle and sensor ablations

These files isolate controller and sensor choices in an environment with known
obstacles as its primary feature (the recorded 5-robot variants below also add
a few unknown obstacles between them, at the user's request, so they're no
longer purely known-obstacle-only).

| Experiment | Robots | Obstacles (known+unknown) | Difference being isolated | YAML | Result | Media |
| --- | ---: | ---: | --- | --- | --- | --- |
| Visibility-area RGB-D baseline | 5 | 9+4=13 | Visibility-area attitude controller with heterogeneous FoV/range settings. | [Configuration](configs/test_7_gatekeeper_known.yaml) | 0 collisions across all 5 robots; 100% of unknown obstacles detected | [Video](media/experiments/exp04_known_obstacles_5robots_visibility_area.mp4) |
| Gatekeeper RGB-D | 5 | 9+4=13 | Gatekeeper under the same known-obstacle environment. | [Configuration](configs/test_8_gatekeeperknown.yaml) | 0 collisions across all 5 robots; 100% of unknown obstacles detected | [Video](media/experiments/exp04_known_obstacles_5robots_gatekeeper.mp4) |
| Gatekeeper LiDAR | 5 | 9+4=13 | Sensor-type ablation using the LiDAR configuration. | [Configuration](configs/test_9_gatekeeperknown.yaml) | 0 collisions across all 5 robots; 75% of unknown obstacles detected | [Video](media/experiments/exp04_known_obstacles_5robots_gatekeeper_lidar.mp4) |

> Videos captured from the `known_obstacles_5robots_*_isaac.yaml` variants: 5 robots (up from 2),
> and known obstacles scaled from 4 to 9 (the original small circle pair + 2 angled walls,
> repositioned, plus 5 shared with the other _isaac variants recorded alongside these), plus 4
> unknown obstacles interspersed between them.
>
> Gatekeeper LiDAR's robots visibly spin in place more than the RGB-D runs. The attitude
> controllers (`visibility_area`/`gatekeeper`) drive a `stop` -> `rotate` -> `track` state machine
> in `safe_control/tracking.py`: whenever a robot needs to face a new frontier or a freshly
> detected obstacle, it stops and rotates to face that direction before moving again. That part is
> normal for every controller/sensor. What's dense here is the obstacle count (9 known + 4 unknown
> in a tight shared field for 5 robots), which makes that reorientation trigger far more often --
> and LiDAR gets no benefit from it being omnidirectional, since it hits the exact same
> `fov_angle`-driven cone logic as RGB-D. In short: LiDAR was attached, but it didn't change the
> underlying behavior.


## Testing Suite

We categorize experiments into three tiers.(see test.md under scripts/):

| Tier | Prefix | Description | Example |
| --- | --- | --- | --- |
| **Core** | `EXP_` | Primary evaluation cases (Blind Corner, Narrow Gateway). | `test_1_blind_corner.yaml` |
| **Validation** | `VAL_` | Sanity checks for physics and collision logic. | `test_5_start_in_contact.yaml` |
| **Comparison** | `CMP_` | Direct A/B testing of Gatekeeper vs Baselines. | `test_10_gatekeeper_unknown.yaml` |

### Drone Spawn Smoke Test

Use the standalone test to verify that `drone_articulation.usd` loads as an
Isaac Lab articulation, exposes its joints and bodies, and can step in the
simulation while holding a fixed altitude. Run these commands inside the
`seamlis-dev` container.

Quick headless check:

```bash
docker compose exec seamlis-dev \
  /workspace/isaaclab/_isaac_sim/python.sh \
  /workspace/seamlis/tests/test_drone_articulation_spawn.py \
  --headless --steps 2 --fixed-z 0.5 --prop-speed 500
```

To inspect the spawned drone in the Isaac Lab viewport, omit `--headless` and
use `--viz kit`:

```bash
docker compose exec seamlis-dev \
  /workspace/isaaclab/_isaac_sim/python.sh \
  /workspace/seamlis/tests/test_drone_articulation_spawn.py \
  --viz kit --fixed-z 0.5 --prop-speed 500
```

The test uses `tools/math_utils.py:set_pose_constant_z` to reassert the drone's
altitude before each physics step and spins the propellers with alternating
directions. Use `--no-props` to disable propeller motion or change the speed
with `--prop-speed`.

The default orientation applies the same 180-degree X-axis body correction as
`run_isaac.py`, which keeps the drone upright. Adjust the orientation with
`--roll-deg`, `--pitch-deg`, and `--yaw-deg`:

```bash
docker compose exec seamlis-dev \
  /workspace/isaaclab/_isaac_sim/python.sh \
  /workspace/seamlis/tests/test_drone_articulation_spawn.py \
  --viz kit --fixed-z 0.5 --prop-speed 500 \
  --roll-deg 180 --pitch-deg 0 --yaw-deg 90
```

Change `--fixed-z` to test another hover height, or pass
`--usd /workspace/seamlis/isaaclab_assets/<asset>.usd` to test another USD
file.

To stop the container when finished:

```bash
docker compose down

```

### Lidar Attachment with CLI

What stays exactly the same:
- Which obstacles get "detected" (still the geometric cam_range check in detect_unknown_obs())
- When/whether a collision happens (is_collide_unknown(), same geometric check)
- The robot's avoidance behavior, paths, everything the planner does

What actually changes:
- A real RTX Lidar sensor prim gets attached to each drone (scripts/sensor.py), producing a live point cloud -- but nothing in the pipeline reads it
- Higher GPU/render cost (heavier -- that's why it's mutually exclusive with --low-memory in the CLI)
- The lidar_enabled column in collision_summary.csv flips to True (pure metadata, doesn't affect any other column's value)
- If debug_print() gets called, you'd see a point count printed

<a id="coordinate-validation"></a>

## Coordinate Validation & Grid Alignment

The planner and Isaac Sim currently use the same horizontal coordinate frame:

* Planner coordinates are `[x, y]` in `[0, width] x [0, height]`.
* Isaac world coordinates use the same `[x, y]` values; altitude is added as `z`.
* Grid arrays use `(row, column) = (y, x)` internally, but this does not swap planner coordinates.
* Robot poses, boundary lines, obstacle markers, and grid markers use the same origin and scale.

`run_isaac.py` validates the environment immediately after constructing `env_handler`.
The validation checks positive dimensions, map shape, coordinate ranges, finite and
in-bounds obstacle centers, and the direct mapping settings:
`xy_scale == 1.0` and `env_origin_world_xy == (0.0, 0.0)`.

### How to verify

Run a simulation from inside the container:

```bash
OMNI_KIT_ALLOW_ROOT=1 /workspace/isaaclab/_isaac_sim/python.sh \
  /workspace/seamlis/scripts/run_isaac.py \
  -c /workspace/seamlis/configs/example_test_indoor.yaml \
  --viz kit
```

The startup log should contain a line similar to:

```text
[Scene] Coordinates validated: planner [0, 24.0] x [0, 18.0] -> world [0.0, 24.0] x [0.0, 18.0]
```

Then check the red boundary lines and survey poles in the Isaac viewport:

> The red boundary lines are **disabled by default** now that the warehouse/env_handler
> alignment is verified (commented out in `run_isaac.py`, search `spawn_boundary_lines`) --
> uncomment that call to bring them back for this kind of alignment debugging.

* The poles define the same rectangle as the planner workspace.
* `Origin_BL` is at world `(0, 0)`.
* Known and unknown obstacle markers match the planner UI locations.
* No poles or obstacles float outside the warehouse floor or intersect walls unexpectedly.

For a syntax-only check that does not start Isaac Sim:

```bash
python3 -m py_compile /workspace/seamlis/scripts/scene.py \
  /workspace/seamlis/scripts/run_isaac.py
```



### for a comparison run, please run run_comparison.sh in the docker container. 
