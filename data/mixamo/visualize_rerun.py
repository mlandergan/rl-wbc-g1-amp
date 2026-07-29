"""Replay a GMR-retargeted G1 motion (.pkl) in Rerun, using the real G1 MuJoCo
model for forward kinematics and rendering the actual robot mesh geometry.

Usage:
    mjpython visualize_rerun.py --pkl Strut_Walking_g1.pkl \
        --mjcf ../../third_party/gmr/assets/unitree_g1/g1_mocap_29dof.xml
"""

import argparse
import pickle
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import rerun as rr


def parse_mesh_files(mjcf_path: str) -> dict[str, str]:
    """Map MJCF <mesh name=...> to its resolved file path, honoring <compiler meshdir=...>."""
    root = ET.parse(mjcf_path).getroot()
    mjcf_dir = Path(mjcf_path).resolve().parent
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    mesh_files = {}
    for mesh in root.iter("mesh"):
        name = mesh.get("name")
        file = mesh.get("file")
        if name and file:
            mesh_files[name] = str(mjcf_dir / meshdir / file)
    return mesh_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--fps", type=float, default=None, help="Override playback fps")
    parser.add_argument("--realtime", action="store_true", default=True,
                         help="Pace logging to real motion duration (default: on)")
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        motion = pickle.load(f)

    root_pos = motion["root_pos"]  # (N, 3)
    root_rot = motion["root_rot"]  # (N, 4), xyzw
    dof_pos = motion["dof_pos"]  # (N, 29)
    fps = args.fps or motion["fps"]
    n_frames = root_pos.shape[0]

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}" for i in range(model.nbody)]
    mesh_files = parse_mesh_files(args.mjcf)

    # body_id -> mesh file path, for bodies that have exactly one mesh geom at local origin
    body_mesh_file = {}
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        body_id = model.geom_bodyid[g]
        mesh_id = model.geom_dataid[g]
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        if mesh_name in mesh_files:
            body_mesh_file[body_id] = mesh_files[mesh_name]

    rr.init("g1_strut_walk", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # log each body's mesh once, at the body's own local origin (no scale/offset needed per XML)
    for body_id, mesh_path in body_mesh_file.items():
        name = body_names[body_id]
        rr.log(f"g1/{name}/mesh", rr.Asset3D(path=mesh_path), static=True)

    parents = [model.body_parentid[i] for i in range(model.nbody)]
    edges = [(i, parents[i]) for i in range(1, model.nbody) if parents[i] >= 0]

    frame_dt = 1.0 / fps
    t_start = time.time()
    for t in range(n_frames):
        rr.set_time_seconds("motion_time", t / fps)

        qpos = np.zeros(model.nq)
        qpos[0:3] = root_pos[t]
        x, y, z, w = root_rot[t]
        qpos[3:7] = [w, x, y, z]  # MuJoCo freejoint quat is wxyz; saved root_rot is xyzw
        qpos[7:] = dof_pos[t]
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        for body_id in body_mesh_file:
            name = body_names[body_id]
            rr.log(
                f"g1/{name}",
                rr.Transform3D(translation=data.xpos[body_id], quaternion=data.xquat[body_id][[1, 2, 3, 0]]),
            )

        xpos = data.xpos.copy()
        segments = [[xpos[c], xpos[p]] for c, p in edges]
        rr.log("g1/skeleton", rr.LineStrips3D(segments, colors=(58, 90, 122), radii=0.004))

        if args.realtime:
            target = t_start + (t + 1) * frame_dt
            sleep_s = target - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)

    print(f"Logged {n_frames} frames ({n_frames/fps:.1f}s) at {fps} fps to Rerun viewer.")
    print(f"Rendered mesh for {len(body_mesh_file)}/{model.nbody} bodies.")


if __name__ == "__main__":
    main()
