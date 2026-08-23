# scripts/scene.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, Gf, Sdf, UsdPhysics, UsdShade, UsdLux
 
    
    
import isaaclab.sim as sim_utils


Vec3f = Tuple[float, float, float]
 
def enable_warehouse_collision(stage, root_path="/World/Env/Main"):
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print(f"[Collision] Invalid root path: {root_path}")
        return

    count = 0
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)

            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api.CreateApproximationAttr().Set("convexHull")
            count += 1

    print(f"[Collision] Enabled collision on {count} meshes under {root_path}")

 
@dataclass
class WarehouseSceneCfg:
    # Coordinates mapping from planner-grid (env) -> USD world
    xy_scale: float = 1.0
    env_origin_world_xy: Tuple[float, float] = (0.0, 0.0)

    # Where to mount the environment USD reference
    env_path: str = "/World/Env/Main"

    # Transform applied to the env container prim
    env_translate: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    env_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    env_rotate_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Lighting
    spawn_ground: bool = True
    spawn_lights: bool = True
    lights_root: str = "/World/Lights"

    sun_path: str = "/World/Lights/Sun"
    sun_intensity: float = 3000.0
    sun_angle: float = 1.0
    sun_color: Vec3f = (1.0, 0.98, 0.9)
    sun_rotate_deg: Vec3f = (-45.0, 0.0, -30.0)

    sky_path: str = "/World/Lights/Sky"
    sky_intensity: float = 800.0
    sky_color: Vec3f = (0.9, 0.95, 1.0)
    # Simple IsaacLab demo light (matches forward_demo.py)
    use_demo_light: bool = True
    demo_light_path: str = "/World/lightDistant"
    demo_light_intensity: float = 3000.0
    demo_light_color: Vec3f = (0.75, 0.75, 0.75)
    demo_light_translation: Vec3f = (1.0, 0.0, 10.0)

    # If True, also spawn the UsdLux sun/sky lights 
    use_usd_lights: bool = False

    # Cleanup toggles
    cleanup_by_name_patterns: bool = True
    cleanup_by_bbox_heuristics: bool = True

    # Debug paths
    materials_root: str = "/World/Materials"
    debug_markers_root: str = "/World/DebugMarkers"
    obstacles_root: str = "/World/EnvObstacles"




class WarehouseScene:
    """
    One-stop scene setup:
      - env USD reference + transform
      - ground + lights
      - optional cleanup (hide pillars/ceilings etc.)
      - optional debug markers
      - optional obstacle visualization
    """

    def __init__(self, cfg: WarehouseSceneCfg, stage=None):
        self.cfg = cfg
        self.stage = stage or omni.usd.get_context().get_stage()



    def setup(self, env_url: str, skip_env: bool = False) -> Usd.Prim:
        """
        Main entrypoint we call from main().
        Returns the env container prim (cfg.env_path).

        skip_env=True loads no warehouse geometry at all (default ground
        plane grid + lights only) -- for asset-showcase recordings.
        """
        env_prim = self._load_env(env_url, skip_reference=skip_env)
        #enable_warehouse_collision(self.stage, self.cfg.env_path)
        if self.cfg.spawn_lights or self.cfg.spawn_ground:
            self._spawn_default_ground_and_lights()

        if not skip_env and (self.cfg.cleanup_by_name_patterns or self.cfg.cleanup_by_bbox_heuristics):
            self._cleanup_env(env_prim)

        return env_prim

    def planner_bounds_world(self, env_handler) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return the world-space bounds represented by the planner environment."""
        width = float(env_handler.width)
        height = float(env_handler.height)
        corners = np.asarray(
            [self._env_to_world_xy((0.0, 0.0)), self._env_to_world_xy((width, height))],
            dtype=float,
        )
        return (
            (float(min(corners[:, 0])), float(min(corners[:, 1]))),
            (float(max(corners[:, 0])), float(max(corners[:, 1]))),
        )

    def validate_env_handler(self, env_handler, tolerance: float = 1e-6) -> None:
        """Validate that planner coordinates have a consistent world mapping."""
        width = float(env_handler.width)
        height = float(env_handler.height)
        resolution = float(env_handler.resolution)
        errors = []

        if not np.isfinite([width, height, resolution]).all() or width <= 0 or height <= 0 or resolution <= 0:
            errors.append("width, height, and resolution must be finite and positive")
        if not np.isclose(self.cfg.xy_scale, 1.0, atol=tolerance):
            errors.append("xy_scale must be 1.0 because robot poses use planner coordinates directly")
        if not np.allclose(self.cfg.env_origin_world_xy, (0.0, 0.0), atol=tolerance):
            errors.append("env_origin_world_xy must be (0.0, 0.0) because robot poses use planner coordinates directly")

        if errors:
            raise ValueError("[Scene] Environment coordinate validation failed: " + "; ".join(errors))

        expected_shape = (int(height / resolution), int(width / resolution))
        get_map_shape = getattr(env_handler, "get_map_shape", None)
        if callable(get_map_shape) and tuple(get_map_shape()) != expected_shape:
            errors.append(
                f"map shape {tuple(get_map_shape())} does not match handler dimensions {expected_shape}"
            )

        x_range = getattr(env_handler, "x_range", (0.0, width))
        y_range = getattr(env_handler, "y_range", (0.0, height))
        if not np.allclose(x_range, (0.0, width), atol=tolerance):
            errors.append(f"x_range {x_range} is not [0, {width}]")
        if not np.allclose(y_range, (0.0, height), atol=tolerance):
            errors.append(f"y_range {y_range} is not [0, {height}]")

        for name in ("obs_circle", "obs_superellipsoid"):
            for index, obstacle in enumerate(getattr(env_handler, name, [])):
                values = np.asarray(obstacle, dtype=float)
                if values.ndim != 1 or values.size < 2 or not np.isfinite(values[:2]).all():
                    errors.append(f"{name}[{index}] does not contain finite [x, y] coordinates")
                elif not (0.0 <= values[0] <= width and 0.0 <= values[1] <= height):
                    errors.append(f"{name}[{index}] center {values[:2].tolist()} is outside planner bounds")

        if errors:
            raise ValueError("[Scene] Environment coordinate validation failed: " + "; ".join(errors))

        world_min, world_max = self.planner_bounds_world(env_handler)
        print(
            f"[Scene] Coordinates validated: planner [0, {width}] x [0, {height}] "
            f"-> world [{world_min[0]}, {world_max[0]}] x [{world_min[1]}, {world_max[1]}]"
        )
        
 
      #  return offset
    def auto_align_warehouse(self, env_path):
            from pxr import UsdGeom, Usd, Gf
            
            floor_prim = None
            root_prim = self.stage.GetPrimAtPath(env_path)
            # 1. Find the floor
            for prim in Usd.PrimRange(root_prim):
                if "floor" in prim.GetName().lower():
                    floor_prim = prim
                    break
            
            if not floor_prim:
                return (0.0, 0.0, 0.0), (1.0, 1.0)

            # 2. Get the WORLD bounding box (takes current scaling into account)
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            bbox = bbox_cache.ComputeWorldBound(floor_prim)
            aligned_box = bbox.ComputeAlignedBox()
            
            f_min = aligned_box.GetMin()
            f_max = aligned_box.GetMax()
            
            size = (float(f_max[0] - f_min[0]), float(f_max[1] - f_min[1]))
            offset = (-float(f_min[0]), -float(f_min[1]), 0.0)
            
            return offset, size
# width, height = (20, 30)
    def spawn_grid_markers(self, env_handler) -> None:
        """Spawns tall red poles at corners and center of planner grid."""
        w = float(env_handler.width)
        h = float(env_handler.height)

        points = [
            (0.0, 0.0, "Origin_BL"),
            (w,   0.0, "Corner_BR"),
            (0.0, h,   "Corner_TL"),
            (w,   h,   "Corner_TR"),
            (w/2, h/2, "Center"),
        ]

        if not self.stage.GetPrimAtPath(self.cfg.debug_markers_root).IsValid():
            UsdGeom.Xform.Define(self.stage, self.cfg.debug_markers_root)

        mat = self._get_or_create_preview_material(
            "RedMarker", Gf.Vec3f(1.0, 0.0, 0.0), roughness=0.4
        )

        for x, y, name in points:
            path = f"{self.cfg.debug_markers_root}/{name}"
            cyl = UsdGeom.Cylinder.Define(self.stage, path)
            cyl.CreateHeightAttr(5.0)
            cyl.CreateRadiusAttr(0.05)

            xw, yw = self._env_to_world_xy((x, y))
            UsdGeom.XformCommonAPI(cyl).SetTranslate((float(xw), float(yw), 2.5))

            UsdShade.MaterialBindingAPI(cyl).Bind(mat)

    def sync_all_obstacles(self, env_handler, unknown_obs_list: Sequence[Sequence[float]]) -> None:
        """
        Spawns cylinders for:
          - known obs (env_handler.obs_circle) in blue
          - unknown_obs_list in orange
        """
        mat_known = self._get_or_create_preview_material(
            "ObstacleKnown", Gf.Vec3f(0.0, 0.4, 0.9), roughness=0.5
        )
        mat_unknown = self._get_or_create_preview_material(
            "ObstacleUnknown", Gf.Vec3f(1.0, 0.5, 0.0), roughness=0.5
        )

        # Known (blue)
        for j, obs in enumerate(getattr(env_handler, "obs_circle", [])):
            x_env, y_env, r_env = obs[:3]
            xw, yw = self._env_to_world_xy((x_env, y_env))
            self._spawn_cylinder(
                f"{self.cfg.obstacles_root}/Known_{j}",
                xw, yw, float(r_env) * self.cfg.xy_scale,
                z=1.0,
                material=mat_known,
                collision=True,
            )

        # Unknown (orange)
        for k, obs in enumerate(unknown_obs_list):
            if len(obs) < 3:
                continue
            x_env, y_env, r_env = obs[:3]
            xw, yw = self._env_to_world_xy((x_env, y_env))
            self._spawn_cylinder(
                f"{self.cfg.obstacles_root}/Unknown_{k}",
                xw, yw, float(r_env) * self.cfg.xy_scale,
                z=1.0,
                material=mat_unknown,
                collision=True,
            )

        #  Internals 
    def _load_env(self, env_url: str, skip_reference: bool = False) -> Usd.Prim:
        env_path = self.cfg.env_path

        # Define-or-get container prim
        xform = UsdGeom.Xform.Define(self.stage, env_path)
        prim = xform.GetPrim()

        # skip_reference leaves this container prim empty (no warehouse
        # geometry referenced in) -- used for asset-showcase recordings that
        # want the default ground-plane grid instead of the warehouse, while
        # keeping this prim around so the transform math downstream (which
        # targets env_path unconditionally) still has something valid to
        # operate on.
        if not skip_reference:
            # Get existing references and only add if not already present
            refs = prim.GetReferences()

            # inspect authored references on the prim
            existing = []
            try:
                # USD stores references in a listOp; this reads authored items
                list_op = prim.GetMetadata("references")
                if list_op is not None:
                    existing = [r.assetPath for r in list_op.GetAddedOrExplicitItems()]
            except Exception:
                existing = []

            if env_url not in existing:
                refs.AddReference(env_url)

        # Apply transform. Op order matters: xformOpOrder lists ops
        # outermost-first, so [translate, rotate, scale] means a point is
        # scaled first (in the asset's own local axes), then rotated about
        # its local origin, then translated to its final world position.
        xformable = UsdGeom.Xformable(xform)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*self.cfg.env_translate))
        if any(self.cfg.env_rotate_deg):
            xformable.AddRotateXYZOp().Set(Gf.Vec3f(*self.cfg.env_rotate_deg))
        xformable.AddScaleOp().Set(Gf.Vec3f(*self.cfg.env_scale))
        
        env_prim = self.stage.GetPrimAtPath(env_path)
        if not env_prim.IsValid():
            raise RuntimeError(f"[Scene] Env prim not found at {env_path}")

        print(
            f"[Scene] Loaded env at {env_path} (reference guarded) "
            f"T={self.cfg.env_translate}, R={self.cfg.env_rotate_deg}, S={self.cfg.env_scale}"
        )
        return env_prim

    def _spawn_default_ground_and_lights(self) -> None:
        # Ground (exactly like the demo)
        if self.cfg.spawn_ground:
            cfg_ground = sim_utils.GroundPlaneCfg()
            cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

        # Demo-style distant light (exactly like the demo)
        if self.cfg.spawn_lights and self.cfg.use_demo_light:
            cfg_light_distant = sim_utils.DistantLightCfg(
                intensity=float(self.cfg.demo_light_intensity),
                color=tuple(self.cfg.demo_light_color),
            )
            cfg_light_distant.func(
                self.cfg.demo_light_path,
                cfg_light_distant,
                translation=tuple(self.cfg.demo_light_translation),
            )

        # Optional: keep ld UsdLux sun/sky available, but OFF by default
        if self.cfg.spawn_lights and self.cfg.use_usd_lights:
            if not self.stage.GetPrimAtPath(self.cfg.lights_root).IsValid():
                UsdGeom.Xform.Define(self.stage, self.cfg.lights_root)

            if not self.stage.GetPrimAtPath(self.cfg.sun_path).IsValid():
                sun = UsdLux.DistantLight.Define(self.stage, self.cfg.sun_path)
                sun.CreateIntensityAttr(self.cfg.sun_intensity)
                sun.CreateAngleAttr(self.cfg.sun_angle)
                sun.CreateColorAttr(Gf.Vec3f(*self.cfg.sun_color))
                UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(*self.cfg.sun_rotate_deg))

            if not self.stage.GetPrimAtPath(self.cfg.sky_path).IsValid():
                sky = UsdLux.DomeLight.Define(self.stage, self.cfg.sky_path)
                sky.CreateIntensityAttr(self.cfg.sky_intensity)
                sky.CreateColorAttr(Gf.Vec3f(*self.cfg.sky_color))

        print("[Scene] Ground/lights ready.")


    def _cleanup_env(self, env_prim: Usd.Prim) -> None:
        hidden_paths = set()

        if self.cfg.cleanup_by_name_patterns:
            hide_patterns = [
                "sm_pillara", "sm_pillarparta", "sm_pillara_9m",
                "pillar", "column", "post", "support",
                "beam", "sm_beama", "sm_beamparta", "sm_beampartb", "sm_beamsupporta",
                "lamp", "ceilinglight", "fixture",
                "roof", "truss", "ceiling",
            ]

            for prim in Usd.PrimRange(self.stage.GetPseudoRoot()):
                name_lower = prim.GetName().lower()
                if any(pat in name_lower for pat in hide_patterns):
                    img = UsdGeom.Imageable(prim)
                    if img:
                        img.MakeInvisible()
                        hidden_paths.add(prim.GetPath())

        if self.cfg.cleanup_by_bbox_heuristics:
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            )

            env_bbox = bbox_cache.ComputeWorldBound(env_prim)
            env_range = env_bbox.ComputeAlignedBox()
            if env_range.IsEmpty():
                return

            global_min = env_range.GetMin()
            global_max = env_range.GetMax()
            ceiling_z_threshold = global_max[2] - 0.3 * (global_max[2] - global_min[2])

            for prim in Usd.PrimRange(env_prim):
                if prim.GetPath() in hidden_paths:
                    continue
                if not prim.IsA(UsdGeom.Mesh):
                    continue

                pbbox = bbox_cache.ComputeWorldBound(prim)
                prange = pbbox.ComputeAlignedBox()
                if prange.IsEmpty():
                    continue

                psize = prange.GetSize()
                center = (prange.GetMin() + prange.GetMax()) * 0.5

                # Ceiling panels
                is_flat = psize[2] < min(psize[0], psize[1]) * 0.3
                is_high = center[2] > ceiling_z_threshold
                if is_flat and is_high:
                    UsdGeom.Imageable(prim).MakeInvisible()
                    hidden_paths.add(prim.GetPath())
                    continue

                # Pillars
                is_tall = psize[2] > 2.0
                is_thin = (psize[0] < 1.0 and psize[1] < 1.0)
                if is_tall and is_thin:
                    UsdGeom.Imageable(prim).MakeInvisible()
                    hidden_paths.add(prim.GetPath())

    def _env_to_world_xy(self, xy: Tuple[float, float]) -> Tuple[float, float]:
        origin = np.asarray(self.cfg.env_origin_world_xy, dtype=np.float32)
        xy = np.asarray(xy, dtype=np.float32)
        out = origin + self.cfg.xy_scale * xy[:2]
        return float(out[0]), float(out[1])

    def _get_or_create_preview_material(self, name: str, color: Gf.Vec3f, roughness: float = 0.5) -> UsdShade.Material:
        mat_path = f"{self.cfg.materials_root}/{name}"
        if self.stage.GetPrimAtPath(mat_path).IsValid():
            return UsdShade.Material(self.stage.GetPrimAtPath(mat_path))

        mat = UsdShade.Material.Define(self.stage, mat_path)
        pbr = UsdShade.Shader.Define(self.stage, f"{mat_path}/PBRShader")
        pbr.CreateIdAttr("UsdPreviewSurface")
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
        return mat

    def _spawn_cylinder(
        self,
        path: str,
        x: float,
        y: float,
        r: float,
        z: float = 1.0,
        material: Optional[UsdShade.Material] = None,
        collision: bool = True,
    ) -> None:
        parent_path = Sdf.Path(path).GetParentPath().pathString
        if not self.stage.GetPrimAtPath(parent_path).IsValid():
            UsdGeom.Xform.Define(self.stage, parent_path)

        if not self.stage.GetPrimAtPath(path).IsValid():
            UsdGeom.Cylinder.Define(self.stage, path)

        prim = self.stage.GetPrimAtPath(path)
        xf = UsdGeom.XformCommonAPI(prim)
        xf.SetTranslate(Gf.Vec3d(float(x), float(y), float(z / 2.0)))
        xf.SetScale(Gf.Vec3f(float(r), float(r), float(z)))

        if collision:
            UsdPhysics.CollisionAPI.Apply(prim)

        if material is not None:
            UsdShade.MaterialBindingAPI(prim).Bind(material)

 
# def spawn_boundary_lines(stage, width, height, color=(1, 0, 0)):
  
    
#     # Create a parent Xform for the boundaries
#     boundary_path = "/World/Boundary"
#     if not stage.GetPrimAtPath(boundary_path):
#         UsdGeom.Xform.Define(stage, boundary_path)

#     # Simplified point logic
#     points = [
#         [(0,0,0), (width,0,0)],
#         [(width,0,0), (width,height,0)],
#         [(width,height,0), (0,height,0)],
#         [(0,height,0), (0,0,0)]
#     ]
    
#     for i, p in enumerate(points):
#         path = f"{boundary_path}/Line_{i}"
#         # Use UsdGeom.Cube directly 
#         cube = UsdGeom.Cube.Define(stage, path)
        
#         mid = (np.array(p[0]) + np.array(p[1])) / 2
#         dist = np.linalg.norm(np.array(p[1]) - np.array(p[0]))
        
#         # Scale and Translate using USD Attributes
#         scale = (dist/2 if i%2==0 else 0.05, 0.05 if i%2==0 else dist/2, 0.01)
#         cube.AddTranslateOp().Set(Gf.Vec3f(mid[0], mid[1], 0.01))
#         cube.AddScaleOp().Set(Gf.Vec3f(scale[0], scale[1], scale[2]))
        
#         # Set Color
#         cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

def spawn_boundary_lines(stage, width, height, color=(1, 0, 0)):
    boundary_path = "/World/Boundary"
    if not stage.GetPrimAtPath(boundary_path):
        UsdGeom.Xform.Define(stage, boundary_path)

    points = [
        [(0,0,0), (width,0,0)],
        [(width,0,0), (width,height,0)],
        [(width,height,0), (0,height,0)],
        [(0,height,0), (0,0,0)]
    ]
    
    for i, p in enumerate(points):
        path = f"{boundary_path}/Line_{i}"
        cube = UsdGeom.Cube.Define(stage, path)
        
        mid = (np.array(p[0]) + np.array(p[1])) / 2
        dist = np.linalg.norm(np.array(p[1]) - np.array(p[0]))
        
        scale = (dist/2 if i%2==0 else 0.05, 0.05 if i%2==0 else dist/2, 0.1)
        cube.AddTranslateOp().Set(Gf.Vec3f(mid[0], mid[1], 0.05))
        cube.AddScaleOp().Set(Gf.Vec3f(scale[0], scale[1], scale[2]))
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

        prim = stage.GetPrimAtPath(path)
        UsdPhysics.CollisionAPI.Apply(prim)