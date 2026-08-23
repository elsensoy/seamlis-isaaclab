import numpy as np
import torch
import omni
import omni.usd
from pxr import UsdGeom, Gf
import omni.kit.app
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
import omni.replicator.core as rep
from pxr import Gf, Sdf, UsdGeom, Usd, UsdShade, UsdPhysics

XY_SCALE = 1


def set_pose_constant_z(
  robot,
  step_size: float = 0.02,
  fixed_z: float | None = None,
  yaw_rad: float = 0.0,
  set_yaw: bool = True,
):
  """
  Move the robot forward by 'step_size' along the given yaw heading,
  keeping altitude constant. Optionally rotate the robot to match yaw.




  Args:
      robot: Isaac Lab Articulation handle
      step_size: forward distance per call (m)
      fixed_z: lock altitude (use current z if None)
      yaw_rad: heading in radians (0 = +X)
      set_yaw: if True, sets the root quaternion to match yaw
  Returns:
      fixed_z for reuse across frames
  """
  # get pose [x, y, z, qx, qy, qz, qw] -- Isaac Lab's Warp-backed root pose is XYZW
  pose = robot.data.root_state_w[0, :7].clone()
  pos = pose[:3]
  # initialize altitude if first call
  if fixed_z is None:
      fixed_z = float(pos[2].item())
  # forward step in XY plane
  c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
  pos[0] += step_size * c
  pos[1] += step_size * s
  pos[2] = torch.tensor(fixed_z, device=pos.device, dtype=pos.dtype)
  pose[:3] = pos

  #  set yaw orientation (x, y, z, w) = (0, 0, sin(y/2), cos(y/2))
  if set_yaw:
      cy, sy = np.cos(0.5 * yaw_rad), np.sin(0.5 * yaw_rad)
      quat = torch.tensor([0.0, 0.0, sy, cy],
                          device=pose.device, dtype=pose.dtype)
      pose[3:7] = quat
  # write new state back into sim
  robot.write_root_pose_to_sim_index(root_pose=pose.unsqueeze(0))
  robot.write_root_velocity_to_sim_index(
      root_velocity=torch.zeros((1, 6), device=pose.device, dtype=pose.dtype)
  )


  return fixed_z
def yaw_from_quat_wxyz(qw, qx, qy, qz):
  siny_cosp = 2.0*(qw*qz + qx*qy)
  cosy_cosp = 1.0 - 2.0*(qy*qy + qz*qz)
  return np.arctan2(siny_cosp, cosy_cosp)


def map_xy(xy_ctrl):
  return XY_SCALE * xy_ctrl[0], XY_SCALE * xy_ctrl[1]

def world_xy_to_exploration_xy(xy_world):
    # xy_world: (N,2) in Isaac meters
    return xy_world / XY_SCALE


def map_xy(xy_ctrl):
    return XY_SCALE * xy_ctrl[0], XY_SCALE * xy_ctrl[1]

def world_xy_to_exploration_xy(xy_world):
    return xy_world / XY_SCALE




def get_world_transform_matrix(prim_path: str) -> Gf.Matrix4d:
    """
    Return the 4x4 world transform matrix for a given prim path.
    """
    import omni
    import omni.usd
    from pxr import UsdGeom, Gf
    import omni.replicator.core as rep
    from pxr import Gf, Sdf, UsdGeom, Usd, UsdShade, UsdPhysics
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not valid at {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    # Use default timecode 0; if using time-varying transforms, adapt this
    return xformable.ComputeLocalToWorldTransform(UsdGeom.GetStageUpAxis(stage))
  # or return xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())