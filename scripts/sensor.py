# creates /World/CF_i/SensorFrame

# attaches FOV under SensorFrame

# attaches Lidar under SensorFrame


# scripts/sensor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import omni.usd
import torch
from pxr import UsdGeom, UsdShade, Sdf, Gf

# Optional LiDAR imports guarded inside attach_lidar()
HAVE_REP = True
try:
    import omni.replicator.core as rep
except Exception:
    HAVE_REP = False
    rep = None

import math 
# ------------------------------------------------------------
# Geometry helpers (FoV wedge mesh)
# ------------------------------------------------------------

def _make_or_get_mesh(path: str) -> UsdGeom.Mesh:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(path))
        mesh.CreateDoubleSidedAttr(True)
    else:
        mesh = UsdGeom.Mesh(prim)
        if not mesh.GetDoubleSidedAttr().HasAuthoredValueOpinion():
            mesh.CreateDoubleSidedAttr(True)
    return mesh


def _set_mesh_xy_wedge_local(mesh: UsdGeom.Mesh, verts_xy_local, z=0.02) -> None:
    verts = [Gf.Vec3f(float(x), float(y), 0.0) for (x, y) in verts_xy_local] + \
            [Gf.Vec3f(float(x), float(y), float(z)) for (x, y) in verts_xy_local]
    n = len(verts_xy_local)

    side = []
    for i in range(n - 1):
        A, B = i, i + 1
        C, D = i + n, i + 1 + n
        side += [[A, B, C], [B, D, C]]

    lower = [[0, i, i + 1] for i in range(1, n - 1)]
    upper = [[n, n + i + 1, n + i] for i in range(1, n - 1)]
    tris = lower + upper + side

    mesh.CreateFaceVertexCountsAttr([3] * len(tris))
    mesh.CreateFaceVertexIndicesAttr([j for tri in tris for j in tri])
    mesh.CreatePointsAttr(verts)


def _fov_wedge_points_local(fov_deg: float, rng: float, samples: int = 40):
    half = 0.5 * float(fov_deg)
    thetas = np.linspace(-half, half, samples)
    pts = [(0.0, 0.0)] + [
        (rng * np.cos(np.radians(t)), rng * np.sin(np.radians(t)))
        for t in thetas
    ]
    return pts

def _update_robot_fov_mesh(parent_prim_path: str, fov_deg: float, rng: float, z_thickness: float) -> str:
    fov_path = f"{parent_prim_path}/FOV"
    mesh = _make_or_get_mesh(fov_path)
    verts_xy_local = _fov_wedge_points_local(fov_deg, rng, samples=40)
    _set_mesh_xy_wedge_local(mesh, verts_xy_local, z=z_thickness)
    return fov_path


def _make_sensor_frame(
    parent_path: str,
    frame_name: str,
    translation=(0.0, 0.0, 0.0),
    rotate_xyz_deg=(0.0, 0.0, 0.0),
) -> str:
    """
    Create a dedicated Xform prim to serve as the stable sensor mount frame.

    Example result:
        /World/CF_0/SensorFrame
    """
    stage = omni.usd.get_context().get_stage()
    frame_path = f"{parent_path}/{frame_name}"

    prim = stage.GetPrimAtPath(frame_path)
    if not prim.IsValid():
        xform = UsdGeom.Xform.Define(stage, Sdf.Path(frame_path))
        prim = xform.GetPrim()

    xf = UsdGeom.Xformable(prim)

    # Clear old ops so reruns do not stack transforms
    xf.ClearXformOpOrder()

    t_op = xf.AddTranslateOp()
    t_op.Set(Gf.Vec3d(*translation))

    r_op = xf.AddRotateXYZOp()
    r_op.Set(Gf.Vec3f(*rotate_xyz_deg))

    return frame_path


# ------------------------------------------------------------
# Material binding
# ------------------------------------------------------------

def bind_preview_material(
    prim_path: str,
    color=(0.0, 0.8, 0.8),
    opacity: float = 0.40,
    emissive_strength: float = 0.35,
    roughness: float = 0.5,
    metallic: float = 0.0,
    stronger: bool = True,
):
    """
    Robust UsdPreviewSurface material binding for ANY prim path.
    Call this AFTER the prim exists (e.g. after suite.attach()).
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[mat] prim invalid: {prim_path}")
        return None

    img = UsdGeom.Imageable(prim)
    if img:
        img.MakeVisible()

    mat_path = f"{prim_path}_Material"
    shader_path = f"{mat_path}/Shader"

    mat = UsdShade.Material.Get(stage, mat_path)
    if not mat:
        mat = UsdShade.Material.Define(stage, mat_path)

    sh = UsdShade.Shader.Get(stage, shader_path)
    if not sh:
        sh = UsdShade.Shader.Define(stage, shader_path)
        sh.CreateIdAttr("UsdPreviewSurface")
    else:
        sh.CreateIdAttr("UsdPreviewSurface")

    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)

    em = Gf.Vec3f(
        color[0] * emissive_strength,
        color[1] * emissive_strength,
        color[2] * emissive_strength,
    )
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(em)

    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))

    mat_surface = mat.CreateSurfaceOutput()
    sh_surface = sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat_surface.ConnectToSource(sh_surface)

    bind_api = UsdShade.MaterialBindingAPI(prim)
    if stronger:
        try:
            bind_api.Bind(mat, UsdShade.Tokens.strongerThanDescendantsBinding)
        except Exception:
            bind_api.Bind(mat)
    else:
        bind_api.Bind(mat)

    mesh = UsdGeom.Mesh(prim)
    if mesh:
        mesh.CreateDoubleSidedAttr(True)

    return mat


def set_fov_color(mesh_prim_path: str, color=(0.0, 0.8, 0.8), opacity=0.35, emissive_strength=0.6):
    return bind_preview_material(
        mesh_prim_path,
        color=color,
        opacity=opacity,
        emissive_strength=emissive_strength,
        roughness=0.35,
        metallic=0.0,
        stronger=True,
    )


# ------------------------------------------------------------
# Small color helper
# ------------------------------------------------------------

def _hsv_to_rgb(h, s, v):
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if   i == 0: r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else:        r, g, b = v, p, q
    return (r, g, b)


def hsv_to_rgb_deg(h, s=0.85, v=0.9):
    return _hsv_to_rgb(h / 360.0, s, v)


# ------------------------------------------------------------
# Sensor suite
# ------------------------------------------------------------
 
@dataclass
class RobotSensorSuiteCfg:
    fov_deg: float = 45.0 # i need to add a pydantic integration to set this to the value in yaml fov_angle
    rng: float = 3.0
    z_thickness: float = 0.01

    # visual
    color: Tuple[float, float, float] = (0.0, 0.8, 0.8)
    opacity: float = 0.001
    emissive_strength: float = 0.4
    hide_fov: bool = False  # skip creating the FOV wedge mesh entirely (asset-showcase recordings)

    # dedicated sensor frame
    sensor_frame_name: str = "SensorFrame"
    sensor_frame_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    sensor_frame_rotate_xyz_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # lidar
    use_lidar: bool = False
    lidar_translation: Tuple[float, float, float] = (0.0, 0.0, 0.03)
    lidar_orientation_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    lidar_config: str = "Example_Rotary"
    scan_rate_hz: int = 20


class RobotSensorSuite:
    """
    Mandatory: FoV wedge mesh under a dedicated SensorFrame.
    Optional: RTX LiDAR + replicator pointcloud annotator.
    """

    def __init__(self, robot_root: str, cfg: RobotSensorSuiteCfg):
        self.robot_root = robot_root
        self.cfg = cfg
        self.stage = omni.usd.get_context().get_stage()

        self.parent_path: Optional[str] = None
        self.sensor_frame_path: Optional[str] = None
        self.fov_path: Optional[str] = None

        self.lidar = None
        self.pc_annot = None

    def attach(self) -> "RobotSensorSuite":
        self.parent_path = "/World"

        self.sensor_frame_path = _make_sensor_frame(
            parent_path=self.parent_path,
            frame_name=f"{self.robot_root.split('/')[-1]}_{self.cfg.sensor_frame_name}",
            translation=(0.0, 0.0, 0.0),
            rotate_xyz_deg=(0.0, 0.0, 0.0),
        )

        if not self.cfg.hide_fov:
            self.fov_path = _update_robot_fov_mesh(
                self.sensor_frame_path,
                self.cfg.fov_deg,
                self.cfg.rng,
                self.cfg.z_thickness,
            )

            bind_preview_material(
                self.fov_path,
                color=self.cfg.color,
                opacity=self.cfg.opacity,
                emissive_strength=self.cfg.emissive_strength,
                stronger=True,
            )

        if self.cfg.use_lidar:
            self._try_attach_lidar()

        return self
    
    
    def set_frame_world_pose(self, pose: torch.Tensor) -> None:
        """
        pose = [x, y, z, qw, qx, qy, qz]
        """
        if self.sensor_frame_path is None:
            return

        prim = self.stage.GetPrimAtPath(self.sensor_frame_path)
        if not prim.IsValid():
            return

        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()

        t = xf.AddTranslateOp()
        t.Set(Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2])))

        # keep visual correction as local rotateXYZ 
        r = xf.AddRotateXYZOp()
        r.Set(Gf.Vec3f(*self.cfg.sensor_frame_rotate_xyz_deg))

 
    def set_sensor_frame_pose_xyyaw(self, x: float, y: float, z: float, yaw_rad: float) -> None:
            """
            Move the separate SensorFrame to the drone position and rotate it only by yaw.
            Optimized to only update values without modifying USD topology.
            """
            if self.sensor_frame_path is None:
                return

            # Lazy initialization of the transform operations
            if getattr(self, "_translate_op", None) is None:
                prim = self.stage.GetPrimAtPath(self.sensor_frame_path)
                if not prim.IsValid():
                    return
                
                xf = UsdGeom.Xformable(prim)
                xf.ClearXformOpOrder()
                
                # Create the operators ONCE and cache them
                self._translate_op = xf.AddTranslateOp()
                self._rotate_op = xf.AddRotateXYZOp()

            # Update the values directly (extremely fast, no topology changes)
            self._translate_op.Set(Gf.Vec3d(x, y, z))
            self._rotate_op.Set(Gf.Vec3f(0.0, 0.0, math.degrees(yaw_rad)))

        
    def update_fov(self, fov_deg: Optional[float] = None, rng: Optional[float] = None) -> None:
        if self.sensor_frame_path is None:
            raise RuntimeError("SensorSuite.attach() must be called before update_fov().")

        if fov_deg is not None:
            self.cfg.fov_deg = float(fov_deg)
        if rng is not None:
            self.cfg.rng = float(rng)

        self.fov_path = _update_robot_fov_mesh(
            self.sensor_frame_path,
            self.cfg.fov_deg,
            self.cfg.rng,
            self.cfg.z_thickness,
        )

        bind_preview_material(
            self.fov_path,
            color=self.cfg.color,
            opacity=self.cfg.opacity,
            emissive_strength=self.cfg.emissive_strength,
            stronger=True,
        )

    def attach_lidar_late(self) -> None:
        """Call after sim.reset() and one sim.step(render=True) if you want late attach."""
        if not self.cfg.use_lidar:
            return
        if self.sensor_frame_path is None:
            return
        if self.lidar is not None:
            return
        self._try_attach_lidar()

    def lidar_point_count(self) -> Optional[int]:
        if self.pc_annot is None:
            return None
        try:
            data = self.pc_annot.get_data()
        except Exception:
            return None

        if isinstance(data, dict):
            for v in data.values():
                if hasattr(v, "shape") and len(v.shape) >= 2:
                    return int(v.shape[0])
            return None

        if hasattr(data, "shape") and len(data.shape) >= 2:
            return int(data.shape[0])
        return None

    def debug_print(self) -> None:
        print(f"[Sensors] {self.robot_root}")
        print(f"  parent:        {self.parent_path}")
        print(f"  sensor_frame:  {self.sensor_frame_path}")
        print(f"  fov:           {self.fov_path} (valid={self.stage.GetPrimAtPath(self.fov_path).IsValid() if self.fov_path else False})")
        print(f"  lidar:         {'ON' if self.lidar is not None else 'OFF'}")
        if self.pc_annot is not None:
            print(f"  pc_pts: {self.lidar_point_count()}")

    # internals 

    def _resolve_parent_path(self, robot_root: str) -> str:
        """
        Keep this in case still want to inspect /body, no longer
        attaching FoV/LiDAR to it directly.
        """
        cand = f"{robot_root}/body"
        if self.stage.GetPrimAtPath(cand).IsValid():
            return cand
        return robot_root

    def _try_attach_lidar(self) -> None:
        try:
            from isaacsim.sensors.rtx import LidarRtx
        except Exception as e:
            print(f"[Sensors] RTX LiDAR import failed (continuing without lidar): {e}")
            self.lidar = None
            self.pc_annot = None
            return

        try:
            self.lidar = LidarRtx(
                prim_path=f"{self.sensor_frame_path}/Lidar",
                parent_prim_path=self.sensor_frame_path,
                translation=np.array(self.cfg.lidar_translation, dtype=np.float32),
                orientation=np.array(self.cfg.lidar_orientation_wxyz, dtype=np.float32),
                config_file_name=self.cfg.lidar_config,
                **{"omni:sensor:Core:scanRateBaseHz": int(self.cfg.scan_rate_hz)},
            )

            if HAVE_REP and rep is not None:
                try:
                    self.pc_annot = rep.AnnotatorRegistry.get_annotator("RtxLidarPointCloud")
                    rp = self.lidar.get_render_product_path()
                    if rp:
                        self.pc_annot.attach([rp])
                except Exception as e:
                    print(f"[Sensors] Replicator annotator attach failed (lidar ok): {e}")
                    self.pc_annot = None

        except Exception as e:
            print(f"[Sensors] Failed to attach lidar (continuing without): {e}")
            self.lidar = None
            self.pc_annot = None


 