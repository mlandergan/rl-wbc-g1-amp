"""Replay a converted AMP .npz motion (the exact data the training env consumes -- after
convert_gmr_to_npz.py's +X heading alignment and ground alignment) in Rerun, using the real
G1 MuJoCo model for forward kinematics and rendering the actual robot mesh geometry.

Differences from visualize_rerun.py (which replays the raw GMR .pkl): reads the npz's own
pelvis pose + dof positions, draws the z=0 ground grid the clip was aligned to, logs
left/right foot trails so lateral foot placement (e.g. crossed feet) is visible at a glance,
and prints a foot-separation analysis to the terminal.

Usage:
    python visualize_npz_rerun.py --npz ../../isaaclab_project/g1_amp/motions/G1_strut_walk.npz \
        --mjcf ../../third_party/gmr/assets/unitree_g1/g1_mocap_29dof.xml
"""

import argparse
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
    parser.add_argument("--npz", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--fps", type=float, default=None, help="Override playback fps")
    parser.add_argument("--realtime", action="store_true", default=True,
                         help="Pace logging to real motion duration (default: on)")
    args = parser.parse_args()

    d = np.load(args.npz)
    body_names_npz = d["body_names"].tolist()
    pel = body_names_npz.index("pelvis")
    la = body_names_npz.index("left_ankle_roll_link")
    ra = body_names_npz.index("right_ankle_roll_link")
    root_pos = d["body_positions"][:, pel]  # (N, 3), aligned + grounded
    root_rot_wxyz = d["body_rotations"][:, pel]  # (N, 4), wxyz
    dof_pos = d["dof_positions"]  # (N, 29), same joint order as the MJCF (see converter)
    fps = args.fps or float(d["fps"])
    n_frames = dof_pos.shape[0]

    # terminal analysis first: lateral foot placement in the heading frame (clip is +X-aligned,
    # so world y IS the lateral axis). Crossing = left foot ending up right of the right foot.
    ly = d["body_positions"][:, la, 1]
    ry = d["body_positions"][:, ra, 1]
    sep = ly - ry  # positive = feet on their own sides
    print(f"Lateral foot separation (left_y - right_y), {n_frames} frames:")
    print(f"  mean {sep.mean():+.3f} m | min {sep.min():+.3f} | max {sep.max():+.3f}")
    print(f"  frames with feet CROSSED (separation < 0): {(sep < 0).sum()} / {n_frames}"
          f"  ({100.0 * (sep < 0).mean():.1f}%)")
    lz = d["body_positions"][:, la, 2]
    rz = d["body_positions"][:, ra, 2]
    print(f"Foot heights: left z [{lz.min():.3f}, {lz.max():.3f}], "
          f"right z [{rz.min():.3f}, {rz.max():.3f}] (sole ~= z - 0.035)")

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}" for i in range(model.nbody)]
    mesh_files = parse_mesh_files(args.mjcf)

    body_mesh_file = {}
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        body_id = model.geom_bodyid[g]
        mesh_id = model.geom_dataid[g]
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        if mesh_name in mesh_files:
            body_mesh_file[body_id] = mesh_files[mesh_name]

    rr.init("g1_strut_walk_npz", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # the z=0 ground grid the clip was aligned to (the whole point of the ground-align fix)
    grid = []
    for i in range(-2, 11):
        grid.append([[i, -2.0, 0.0], [i, 3.0, 0.0]])
    for j in range(-2, 4):
        grid.append([[-2.0, j, 0.0], [10.0, j, 0.0]])
    rr.log("world/ground_grid", rr.LineStrips3D(grid, colors=(90, 90, 90), radii=0.002), static=True)

    for body_id, mesh_path in body_mesh_file.items():
        name = body_names[body_id]
        rr.log(f"g1/{name}/mesh", rr.Asset3D(path=mesh_path), static=True)

    parents = [model.body_parentid[i] for i in range(model.nbody)]
    edges = [(i, parents[i]) for i in range(1, model.nbody) if parents[i] >= 0]

    frame_dt = 1.0 / fps
    t_start = time.time()
    left_trail: list[np.ndarray] = []
    right_trail: list[np.ndarray] = []
    for t in range(n_frames):
        rr.set_time_seconds("motion_time", t / fps)

        qpos = np.zeros(model.nq)
        qpos[0:3] = root_pos[t]
        qpos[3:7] = root_rot_wxyz[t]  # npz stores wxyz (from MuJoCo xquat), freejoint wants wxyz
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

        # foot trails: left = orange, right = cyan; crossing shows up as the trails swapping sides
        left_trail.append(d["body_positions"][t, la].copy())
        right_trail.append(d["body_positions"][t, ra].copy())
        rr.log("g1/foot_trail/left", rr.Points3D(np.array(left_trail), colors=(230, 140, 30), radii=0.008))
        rr.log("g1/foot_trail/right", rr.Points3D(np.array(right_trail), colors=(40, 180, 220), radii=0.008))

        if args.realtime:
            target = t_start + (t + 1) * frame_dt
            sleep_s = target - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)

    print(f"Logged {n_frames} frames ({n_frames/fps:.1f}s) at {fps} fps to Rerun viewer.")
    print(f"Rendered mesh for {len(body_mesh_file)}/{model.nbody} bodies.")


if __name__ == "__main__":
    main()
