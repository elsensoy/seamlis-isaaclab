#!/usr/bin/env python3
"""Bake the referenced warehouse into a local, transformed USD asset."""

from __future__ import annotations

import argparse
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True, help="Warehouse USD path or URL")
parser.add_argument("--output", required=True, help="Output USD path")
parser.add_argument("--planner-width", type=float, required=True, help="Planner width in meters")
parser.add_argument("--planner-height", type=float, required=True, help="Planner height in meters")
parser.add_argument("--native-size", type=float, default=20.0)
parser.add_argument("--padding", type=float, default=1.25)
args = parser.parse_args()

from pxr import Gf, Sdf, Usd, UsdGeom


def bake_warehouse(input_usd: str, output_usd: str, width: float, height: float,
                   native_size: float, padding: float) -> None:
    if width <= 0 or height <= 0 or native_size <= 0 or padding <= 0:
        raise ValueError("width, height, native-size, and padding must be positive")

    stage = Usd.Stage.CreateInMemory()
    root_path = Sdf.Path("/World/Env/Main")
    root = UsdGeom.Xform.Define(stage, root_path)
    root.GetPrim().GetReferences().AddReference(input_usd)
    if not any(prim.IsDefined() for prim in stage.Traverse() if prim != root.GetPrim()):
        raise RuntimeError(f"Could not resolve warehouse geometry from {input_usd}")

    xform = UsdGeom.Xformable(root)
    xform.AddTranslateOp().Set(Gf.Vec3d(width / 2.0, height / 2.0, 0.0))
    xform.AddScaleOp().Set(Gf.Vec3f(
        (width / native_size) * padding,
        (height / native_size) * padding,
        1.0,
    ))

    os.makedirs(os.path.dirname(os.path.abspath(output_usd)), exist_ok=True)
    flattened = stage.Flatten()
    flattened.Export(output_usd)
    print(f"[Bake] Saved local warehouse asset to {output_usd}")


bake_warehouse(
    args.input,
    os.path.abspath(args.output),
    args.planner_width,
    args.planner_height,
    args.native_size,
    args.padding,
)
