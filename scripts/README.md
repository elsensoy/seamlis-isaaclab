# Scripts Architecture Overview (Isaac Lab + ExplorationManager Bridge)


## `run_isaac.py`: What It Does
`run_isaac.py` is the “orchestrator” script.

It should:
- parse CLI args + read YAML config
- spawn simulation context
- call modular setup classes for scene + sensors
- instantiate `ExplorationManager`
- instantiate `ExploreAdapter`
- run simulation loop (planner ticks, physics pin, camera director, optional frame recording)

Those responsibilities are delegated to `scripts/scene.py`, `scripts/sensor.py`, `scripts/utils.py`, and `scripts/adapter.py`.

### CLI flags

Beyond `--config`, `--use-rtx-lidar`, and the standard `AppLauncher` args (`--viz kit|headless`, `--enable_cameras`, ...), a few flags exist purely to support clean **asset-showcase** recordings (no experiment, just the drone asset) as opposed to normal experiment demos:

| Flag | Effect |
| --- | --- |
| `--no-trails` | Disable the persistent trajectory-trail lines and frontier markers (cosmetic overlays from `adapter.py`). |
| `--hide-fov` | Skip creating the per-robot FOV wedge mesh entirely (`RobotSensorSuiteCfg.hide_fov`). |
| `--no-warehouse` | Skip loading the warehouse USD reference; falls back to the default ground-plane grid + lights (`WarehouseScene.setup(..., skip_env=True)`). |
| `--force-track-robot N` | Lock the camera in `track` mode on robot `N` for the whole run instead of the usual global/per-robot cycling. |
| `--track-offset X Y Z` | Camera eye offset (world-frame, relative to the tracked robot) used with `--force-track-robot`; pass something close (e.g. `-1 -1 0.6`) for a close-up angle. |

Recording-related flags (used for both experiment demos and showcases):

| Flag | Effect |
| --- | --- |
| `--record-dir DIR` | Capture a numbered PNG per rendered frame into `DIR` (stitch into an mp4 afterward with `ffmpeg`). Only works with `--viz kit` (needs the viewport extension); silently disabled otherwise. |
| `--record-every N` | Only capture every Nth rendered frame (default 4). |
| `--exit-after-collision-secs S` | Once the first collision happens, keep running/recording for `S` more sim-seconds, then stop early — useful for a short "fall and stay down" demo clip instead of running to `max_ticks`. |

---

## `scripts/utils.py`: Config + Planner Override + Camera + Logging

### Responsibilities
`utils.py` provides:
1. **YAML loading**
2. **Robot spec parsing** (common + instances → flat list)
3. **Known obstacle conversion** from YAML into planner format
4. **Planner override hook**: apply YAML environment into `ExplorationManager`
5. **`CameraManager`**: viewport control (global birdview / per-robot tracking / collision focus)
6. **Experiment logging** (CSV summary + pose trace)

### Key functions

#### `load_yaml(path)`
Reads a YAML into a Python dictionary.

#### `parse_robot_specs(robots_cfg)`
Flattens:
```yaml
robots:
  common: {...}
  instances: [...]
````

into:

* `robot_specs`: list of per-robot dict configs
* `X0s`: list of initial states

#### `build_known_obs(env_cfg)`

Builds known obstacles in the planner’s expected format:

* circles: `[x, y, r, 0, 0, 0, flag=0]`
* superellipsoids: `[cx, cy, a, b, e, theta, flag=1]`

#### `configure_planner_from_yaml(exploration, env_cfg)`

This is the **main planner synchronization method**.

It updates `exploration.env_handler` with YAML values:

* width
* height
* resolution
* known obstacles (circles/superellipsoids)

Then it forces internal refresh:

* `manager.set_env_obstacles(env_handler)`
* `manager.set_env_workspace(env_handler)`
* updates each controller’s `env` reference

**Important:** this changes planner internals and must be called before computing frontiers/goals.

### `CameraManager`

Owns the Kit viewport camera. Two modes:

* `mode="global"`: a fixed top-down birdview centered on the configured `environment.width`/`height` rectangle (matches the framing of the 2D Matplotlib exploration plots).
* `mode="track"`: eye = `robot_pos + offset` (world-frame, not robot-relative), target = `robot_pos`, recomputed every frame via `update(robots_sim)`. `offset` and the tracked robot index are set with `set_view(mode, track_idx, offset)`.

`focus_on_collision(point_xyz)` overrides whichever mode is active for a fixed number of frames (`collision_focus_hold_frames`) to cut to a birdview centered on a collision site, then falls back on its own — callers don't need to restore the previous mode afterward.

`run_isaac.py`'s main loop drives this as a "camera director": by default it cycles between the global view and each robot's track view every `DRONE_VIEW_DURATION` (100) frames, with robot 0 shot from a mirrored offset so its segment isn't framed identically to the others. `--force-track-robot`/`--track-offset` bypass the cycling entirely and lock onto one robot for the whole run (used for asset-showcase recordings).

---

## `scripts/scene.py`: WarehouseScene (Environment + Lighting + Visual Debug)

### Purpose

`WarehouseScene` is a modular wrapper around:

* loading the warehouse USD (or skipping it for a bare ground plane)
* applying transforms, including an auto-fit scale/orientation solve
* spawning ground + lights
* cleanup/hiding clutter prims (pillars/ceilings)
* spawning debug grid markers
* spawning obstacle visualization USD shapes

The **planner stays 2D**, but we build a synchronized 3D “debuggable” world.

### Key objects

#### `WarehouseSceneCfg`

Defines scene constants:

* coordinate mapping (planner env → USD world)
* environment prim path and transforms
* lighting setup
* debug root prim paths (materials, markers, obstacles)
* cleanup toggles

#### `WarehouseScene.setup(env_url, skip_env=False)`

Called once from `run_isaac.py`.

It:

1. Loads the warehouse USD reference under `/World/Env/Main` — or, if `skip_env=True` (`--no-warehouse`), leaves that container prim empty and skips the cleanup pass, so downstream transform math still has a valid prim to work with, it just has no warehouse geometry in it.
2. Applies translate/scale to align with planner coordinates
3. Spawns ground + lights (if enabled) — with no warehouse loaded, this is the default `GroundPlaneCfg()` grid seen in asset-showcase recordings.
4. Runs cleanup filters (optional, skipped when `skip_env=True`)

Returns the env prim.

The warehouse's on-screen size/position isn't fixed — `run_isaac.py` solves for it before calling `setup()`:

* `_warehouse_fit()` measures the asset's real interior footprint (24m x 36m, from the wall/floor prim bounding boxes) and tries both the native orientation and a 90°-rotated one, picking whichever leaves less unused floor space around the configured `environment.width` x `environment.height` grid.
* A manual cosmetic scale boost (`WAREHOUSE_MANUAL_SCALE_BOOST`) and solved per-side margins (`WAREHOUSE_MARGIN_X/Y`, `MIN_LEFT_MARGIN`, `MIN_BOTTOM_MARGIN`, `WAREHOUSE_SHIFT_X/Y`) then guarantee the grid sits fully inside the walls with clearance on every side — the previous version of this math is what let robots visually clip through a wall near their spawn corner; see the README's core-controller experiments section for the incident.

#### `WarehouseScene.spawn_grid_markers(env_handler)`

Spawns 5 red poles:

* (0,0), (w,0), (0,h), (w,h), (w/2,h/2)

This is a **critical alignment debugging tool**:

* if markers don’t match the warehouse floor corners, update `env_origin_world_xy` and/or transforms.

#### `WarehouseScene.sync_all_obstacles(env_handler, unknown_obs_list)`

Visualizes obstacles in the USD stage:

* known obstacles from `env_handler.obs_circle` → blue cylinders
* unknown obstacles from YAML → orange cylinders

This function is **visual only**. It does not affect planner logic.
Planner logic is controlled by `configure_planner_from_yaml`.

---

## `scripts/sensor.py`: RobotSensorSuite (FoV Mesh + Optional RTX LiDAR)

### Why a sensor suite class?

Sensors were previously implemented as standalone functions. That made it hard to:

* debug which prim paths exist
* reliably bind materials
* attach LiDAR without crashing runs
* evolve toward “add LiDAR later” workflows

`RobotSensorSuite` centralizes sensor-related logic and exposes a stable API.

### What is mandatory vs optional?

Normally mandatory, but skippable for showcase recordings:

* FoV wedge mesh under a dedicated `SensorFrame` (used by exploration visualization and debugging) — set `RobotSensorSuiteCfg.hide_fov=True` (`--hide-fov`) to skip creating it entirely, e.g. for a clean asset-showcase shot.

Optional:

* RTX LiDAR (enabled with `--use-rtx-lidar`)
* Replicator pointcloud annotator (best effort)

### Key objects

#### `RobotSensorSuiteCfg`

Defines:

* FoV geometry: `fov_deg`, `rng`, `z_thickness`
* FoV visuals: `color`, `opacity`, `emissive_strength`
* `hide_fov`: skip FoV mesh creation entirely
* SensorFrame pose: `sensor_frame_translation`, `sensor_frame_rotate_xyz_deg`
* LiDAR toggles + config: `use_lidar`, `scan_rate_hz`, config file, pose offsets

#### `RobotSensorSuite(robot_root, cfg).attach()`

Creates a dedicated `SensorFrame` prim under `/World` (not under the robot's own prim hierarchy), moved every frame by the adapter via `set_sensor_frame_pose_xyyaw(x, y, z, yaw_rad)` to track the robot's pose — that call only updates cached transform ops, it never recreates mesh topology.

Then, unless `hide_fov=True`:

1. Creates FoV mesh prim at `{SensorFrame}/FOV`
2. Binds a `UsdPreviewSurface` material (strong binding)

Independently of `hide_fov`:

3. If `use_lidar=True`, attempts to attach `LidarRtx` under `{parent}/Lidar`
4. Optionally attaches Replicator annotator (`RtxLidarPointCloud`)

LiDAR is **best-effort**:

* failures are logged
* sim continues
* `suite.lidar` becomes `None`

#### Useful methods

* `suite.lidar_point_count()`: returns point count if annotator exists
* `suite.update_fov(...)`: rebuilds mesh geometry
* `suite.set_fov_color(...)`: recolors wedge during runtime
* `suite.set_sensor_frame_pose_xyyaw(x, y, z, yaw_rad)`: moves the SensorFrame to match the robot's current planner pose (called from `adapter.py`'s `pin_step()`, only on frames that will actually be rendered)

---

## `scripts/adapter.py`: ExploreAdapter (Planner ↔ Sim State + Crash Animation)

### Why this file exists

The planner (`ExplorationManager`) is 2D and knows nothing about USD/physics. `ExploreAdapter` is the only thing that writes robot pose into the Isaac stage, and the only thing that reads sim state back out. Two entry points, called at different rates:

* **`plan_step(dt_plan)`** — advances the planner (`exploration.move_robots()`) and refreshes each robot's cached target `(x, y, yaw)`. Call once per planner tick.
* **`pin_step(dt_frame, render_this_frame)`** — re-writes the *last* cached target's pose and zeroes root velocity via `write_root_pose_to_sim_index`/`write_root_velocity_to_sim_index`. Call every physics substep, so contact/depenetration forces and propeller reaction torque never get a window to drift the body between planner ticks (that residual drift is what reads as "floating"). Preallocates its GPU/CPU pose buffers lazily on first call and reuses them every frame — no per-frame CUDA allocation.

### Crash state machine

Per-robot state: `"flying" -> "falling" -> "crashed"` (terminal). When `exploration.last_step_status[i] == -2` (a collision, either an unknown-obstacle hit or an inter-agent collision), the robot's `(x, y)` are frozen — nudged flush against the obstacle surface by `_crash_touch_point()` first, since the CBF safety-margin radius that triggered the collision is larger than the drone's rendered footprint (`CRASH_VISUAL_RADIUS`) and would otherwise leave a visible gap — and `z` eases down to `GROUND_Z` over `CRASH_FALL_TIME` seconds. Once `"crashed"`, the robot is skipped entirely by `plan_step()` (no further pose/trail updates), so it can never "jump" back onto wherever the controller's internal state has since drifted to.

Collisions are counted once per `flying -> falling` transition (`collision_count`, `total_collisions`), not once per tick the robot stays down. `last_collision_events` holds this planner-tick's new collisions for callers (the camera director's `focus_on_collision()`) to react to, and is reset every `plan_step()` call.

### Cosmetic overlays (optional, `draw_trails`/`draw_frontiers`)

* **Trails**: one polyline per robot, colored to match its FOV mesh hue (robot 0 overridden to a peach-pink, `ROBOT_0_TRAIL_COLOR`, since the hue-by-index scheme gives it pure red otherwise). Drawn incrementally, one new segment per `plan_step()` position update — not cleared and redrawn every frame.
* **Frontiers**: `draw_frontiers(frontiers)` replaces the drawn frontier-point markers with the current frontier set (blue) plus each robot's currently-picked goal (orange, larger). Call after `update_all_goals()`/`update_goals_for_completed()`, not before.

Both degrade gracefully to disabled if `isaacsim.util.debug_draw` isn't available (headless runs), and both can be disabled outright via `--no-trails` (clean asset-showcase recordings).

---

## How `run_isaac.py` Connects Everything

### Recommended init order (important)

1. **Parse CLI args, load YAML**
2. **Initialize SimulationContext**
3. **Initialize `CameraManager`** (needs `env_cfg` for the global-view framing)
4. **Create `WarehouseScene` and call `scene.setup(env_url, skip_env=args_cli.no_warehouse)`**
5. **Spawn robots**
6. **Attach `RobotSensorSuite` per robot** (`hide_fov=args_cli.hide_fov`)
7. **Instantiate `ExplorationManager`**
8. **Override `env_handler` from YAML**

   ```python
   env_handler = utils.configure_planner_from_yaml(exploration, env_cfg_data)
   ```
9. **Set unknown obstacles**
10. **Sync visuals**

    ```python
    scene.sync_all_obstacles(env_handler, unknown_obs)
    scene.spawn_grid_markers(env_handler)
    ```
11. **Create `ExploreAdapter`** (`draw_trails`/`draw_frontiers=not args_cli.no_trails`)
12. **Initialize goals**

```python
exploration.frontiers = exploration.get_frontiers()
exploration.update_all_goals()
```

Then the main loop runs: `adapter.plan_step()` at the planner rate, `adapter.pin_step()` every physics substep, the camera director updates the viewport, and (if `--record-dir`) a PNG is captured every Nth rendered frame.

### Why this ordering?

* The planner’s env (`env_handler`) must be finalized *before* computing frontiers/goals.
* Scene obstacle visuals must be spawned using the finalized `env_handler`.
* The adapter relies on consistent robot lists and planner state.

---

## Safety Notes and Debug Strategy

### 1) Planner vs Visuals

* `configure_planner_from_yaml()` affects planner logic.
* `scene.sync_all_obstacles()` affects visualization only.
* The warehouse USD itself is **always** visual-only — it has no physics collision enabled (`enable_warehouse_collision()` exists in `scene.py` but isn't called), so a robot never physically collides with the warehouse mesh. Its only job is to visually contain the planner's `environment:` grid; if it's scaled/positioned too small for that grid, robots can appear to fly through a wall even though nothing "collided" from the planner's point of view. See `_warehouse_fit()` in `run_isaac.py`.

### 2) Coordinate alignment

If robots hit “invisible walls,” the mapping between:

* planner env coords
* USD world coords

is wrong.

Use `scene.spawn_grid_markers()` to verify.

### 3) RTX LiDAR is optional and fragile

The sensor suite is designed so that a LiDAR failure never kills the simulation.

In Isaac Sim 4.5, LiDAR and replicator often require:

* `sim.reset()`
* one `sim.step(render=True)`
  before the render product path becomes valid.

---

## Summary: Who Owns What?

| Component      | Owns                                                          | Does NOT own          |
| -------------- | -------------------------------------------------------------- | --------------------- |
| `run_isaac.py` | orchestration + main loop + camera director + recording         | low-level USD logic   |
| `utils.py`     | YAML parsing, planner override, `CameraManager`, logging        | rendering / sensors   |
| `scene.py`     | env asset (or its absence), lights, warehouse auto-fit, obstacle/marker visualization | planner logic         |
| `sensor.py`    | FoV mesh (optional), optional LiDAR attach, pointcloud hooks    | planner behavior      |
| `adapter.py`   | planner↔sim pose sync, crash state machine, trails/frontiers    | environment creation  |

---

## Typical Debuggings

### "FoV is invisible / not colorful"

* check that `{SensorFrame}/FOV` exists (and that `--hide-fov`/`hide_fov` wasn't left on by mistake)
* ensure strong material binding is being applied
* increase `emissive_strength` and decrease opacity (0.25–0.40 range is typical)

### "LiDAR always returns no points"

* confirm `{parent}/Lidar` prim exists
* print `lidar.get_render_product_path()`
* ensure we stepped at least once with `render=True`

### "Robot collides with stuff planner thinks is free"

* check `spawn_grid_markers()` alignment
* validate that known obstacles are being loaded into planner (YAML circles/superellipsoids)
* validate environment transform (translate/scale)

### "Robot visually clips through the warehouse wall near its spawn point"

* the warehouse is visual-only (no collision) and is fit around the planner grid by `_warehouse_fit()` in `run_isaac.py` — if a robot spawns close to one edge of the grid, check that edge's solved margin (`WAREHOUSE_MARGIN_X/Y`, `MIN_LEFT_MARGIN`, `MIN_BOTTOM_MARGIN`, `WAREHOUSE_SHIFT_X/Y`) is actually still positive after any manual shift; a shift on one axis silently erodes the opposite side's margin guarantee.

### "Crashed robot jumps back into the air / snaps to a new position"

* a robot's `_crash_phase` must be checked in `plan_step()` before pulling a fresh pose from the controller — if it's not `"flying"`, skip the update entirely instead of re-reading the controller's (still-advancing) internal state.

---
