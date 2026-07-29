"""Render a GMR-retargeted G1 motion (.pkl) to an animated GIF using MuJoCo's
own offscreen renderer (correct meshes/materials/lighting out of the box).

Usage:
    python render_gif.py --pkl Strut_Walking_loop_g1.pkl \
        --mjcf ../../third_party/gmr/assets/unitree_g1/g1_mocap_29dof.xml \
        --out strut_walk.gif
"""

import argparse
import pickle

import mujoco
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--render_fps", type=float, default=20.0)
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        motion = pickle.load(f)

    root_pos = motion["root_pos"]
    root_rot = motion["root_rot"]  # xyzw
    dof_pos = motion["dof_pos"]
    src_fps = motion["fps"]
    n_frames = root_pos.shape[0]

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    cam = mujoco.MjvCamera()
    cam.distance = 3.0
    cam.elevation = -15
    cam.azimuth = 110

    step = max(1, round(src_fps / args.render_fps))
    frame_indices = list(range(0, n_frames, step))

    images = []
    for t in frame_indices:
        qpos = np.zeros(model.nq)
        qpos[0:3] = root_pos[t]
        x, y, z, w = root_rot[t]
        qpos[3:7] = [w, x, y, z]
        qpos[7:] = dof_pos[t]
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        cam.lookat[:] = root_pos[t] + np.array([0.0, 0.0, 0.6])
        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()
        images.append(Image.fromarray(pixels))

    duration_ms = int(1000 / args.render_fps)
    images[0].save(
        args.out,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Wrote {args.out}: {len(images)} frames at {args.render_fps} fps")


if __name__ == "__main__":
    main()
