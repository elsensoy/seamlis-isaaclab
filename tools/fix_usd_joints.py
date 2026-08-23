# Save as fix_usd_joints.py
#!/usr/bin/env python3
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Flight test with orientation fix.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Usd, UsdPhysics, UsdGeom
import os

stage = Usd.Stage.Open("/workspace/seamlis/isaaclab_assets/drone_articulation.usd")

for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.RevoluteJoint):
        joint = UsdPhysics.RevoluteJoint(prim)
        
        # 1. REMOVE LIMITS (Crucial for propellers)
        joint.CreateLowerLimitAttr().Set(-1e10)
        joint.CreateUpperLimitAttr().Set(1e10)
        
        # 2. ENABLE VELOCITY DRIVE
        # Damping/effort kept low and bounded to match the runtime
        # ImplicitActuatorCfg override in run_isaac.py (joint_prop_.* motors) --
        # damping=1e5 with an unlimited effort limit produced enormous reaction
        # torque against the kinematically-pinned root every physics step.
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr().Set("velocity")
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(0.02)
        drive.CreateMaxForceAttr().Set(0.02)  # bound effort; USD default is unlimited
        
        print(f"Fixed Joint: {prim.GetPath()}")

stage.GetRootLayer().Save()