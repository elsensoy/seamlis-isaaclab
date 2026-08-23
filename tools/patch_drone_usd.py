#!/usr/bin/env python3
import os
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Flatten and Patch Drone USD.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--input", "-i", required=True)
parser.add_argument("--output", "-o", required=True)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

def _world_transform(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    return UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)

def patch_drone(input_usd, output_usd):
    stage = Usd.Stage.Open(input_usd)
    
    # 1. Identify key paths
    # Based on logs, the paths are /root/drone_root/
    drone_root_path = Sdf.Path("/root/drone_root")
    drone_body_path = drone_root_path.AppendChild("drone_body")
    
    drone_root = stage.GetPrimAtPath(drone_root_path)
    body = stage.GetPrimAtPath(drone_body_path)

    if not body.IsValid():
        raise RuntimeError(f"Could not find body at {drone_body_path}")

    # 2. Articulation Root goes on the top-most common parent
    UsdPhysics.ArticulationRootAPI.Apply(drone_root)
    
    # 3. Make body a rigid body
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(1.5)
    # Apply collisions to all meshes inside the body
    for p in Usd.PrimRange(body):
        if p.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(p)

    # 4. Move Propellers to flatten hierarchy
    # Current: body -> motor -> prop | Target: drone_root -> prop
    prop_map = {
        "front_left":  "motor_front_left/prop_front_left",
        "front_right": "motor_front_right/prop_front_right",
        "back_left":   "motor_back_left/prop_back_left",
        "back_right":  "motor_back_right/prop_back_right",
    }

    joints_scope = UsdGeom.Scope.Define(stage, drone_root_path.AppendChild("joints"))

    for key, subpath in prop_map.items():
        old_path = drone_body_path.AppendPath(subpath)
        motor_path = drone_body_path.AppendChild(f"motor_{key}")
        new_path = drone_root_path.AppendChild(f"prop_{key}")
        
        # Capture World Transforms before moving anything
        M_prop_w = _world_transform(stage, old_path)
        M_root_w = _world_transform(stage, drone_root_path)
        M_motor_w = _world_transform(stage, motor_path)
        
        # THE FIX: we need the prim correctly 
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(old_path, new_path)
        stage.GetRootLayer().Apply(edit)
        
        prop_prim = stage.GetPrimAtPath(new_path)
        
        # Set Local Transform relative to drone_root (this avoids cluttering at the global axis)
        # Formula: M_local = M_world_prop * inv(M_world_parent)
        M_local = M_prop_w * M_root_w.GetInverse()
        xf = UsdGeom.Xformable(prop_prim)
        xf.ClearXformOpOrder()
        xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(M_local)

        # Physics for the Propeller
        UsdPhysics.RigidBodyAPI.Apply(prop_prim)
        UsdPhysics.MassAPI.Apply(prop_prim).CreateMassAttr(0.05)
        for p in Usd.PrimRange(prop_prim):
            if p.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(p)

        # 5. Create the Joint
        joint_path = joints_scope.GetPath().AppendChild(f"joint_prop_{key}")
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([drone_body_path])
        joint.CreateBody1Rel().SetTargets([new_path])
        joint.CreateAxisAttr("Y")

        # Set anchors (Motor origin in Body and Prop local frames)
        p_w = M_motor_w.ExtractTranslation()
        p_body = _world_transform(stage, drone_body_path).GetInverse().Transform(p_w)
        p_prop = M_prop_w.GetInverse().Transform(p_w)

        joint.CreateLocalPos0Attr(Gf.Vec3f(p_body))
        joint.CreateLocalPos1Attr(Gf.Vec3f(p_prop))
        print(f"[INFO] Created Joint and Flattened: {key}")

    # Save
    stage.GetRootLayer().Export(output_usd)
    print(f" SUCCESS: Patched file saved to {output_usd} ")

if __name__ == "__main__":
    patch_drone(os.path.abspath(args_cli.input), os.path.abspath(args_cli.output))
    app.close()