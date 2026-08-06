"""Sim-to-sim: run the trained AMP policy (checkpoint from Isaac Sim / PhysX training) inside
Unitree's own official G1 MuJoCo model, driven by a standard external PD loop -- the same shape
every real G1 RL deployment uses (unitree_rl_gym, GR00T, etc.): the policy outputs a target
joint-position offset, a PD controller outside the model turns that into torque, physics steps.

No Isaac Lab dependency. Everything the policy needs (network weights, observation
normalization stats) is read directly out of the skrl checkpoint; the observation vector is
reconstructed here to match g1_amp_env.py's compute_obs() exactly, from this MuJoCo model's own
state, gathered by joint/body NAME (not raw array index) so this doesn't silently break if the
model's internal ordering ever changes.

PD gains are NOT borrowed from any other project's G1 config -- they're the actual gains this
policy was trained against in Isaac Lab (G1_29DOF_CFG's stock leg/foot values, this project's
softened arm/waist override). Using different gains here would test the policy against control
dynamics it never saw, which isn't a sim-to-sim comparison.

Usage:
    python rollout_policy.py --checkpoint /path/to/best_agent.pt --out sim2sim_strut.mp4
"""

import argparse

import imageio
import mujoco
import numpy as np
import torch
import torch.nn as nn

# Same 29-joint order as convert_gmr_to_npz.py's DOF_NAMES / the AMP env's action_dof_indexes.
DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Same order as g1_amp_env.py's key_body_names. This MJCF (unlike GMR's) has no separate hand
# body -- the hand mesh is a geom on wrist_yaw_link, the last body in the arm chain -- so
# *_hand_palm_link becomes *_wrist_yaw_link, the closest equivalent end-effector body present.
KEY_BODY_NAMES = [
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_elbow_link", "right_elbow_link",
    "right_hip_yaw_link", "left_hip_yaw_link",
    "right_wrist_yaw_link", "left_wrist_yaw_link",
    "right_ankle_roll_link", "left_ankle_roll_link",
]

# default_joint_pos, matching Isaac Lab's G1_29DOF_CFG init_state.joint_pos (only these three
# joint groups get a nonzero standing default; everything else is 0).
DEFAULT_JOINT_POS = {n: 0.0 for n in DOF_NAMES}
for n in DOF_NAMES:
    if n.endswith("hip_pitch_joint"):
        DEFAULT_JOINT_POS[n] = -0.1
    elif n.endswith("knee_joint"):
        DEFAULT_JOINT_POS[n] = 0.3
    elif n.endswith("ankle_pitch_joint"):
        DEFAULT_JOINT_POS[n] = -0.2

# PD gains this policy actually trained against (g1_amp_env_cfg.py's robot actuators): stock
# Isaac Lab G1_29DOF_CFG legs/feet, this project's softened arms/waist override.
GAINS = {}
for side in ("left", "right"):
    GAINS[f"{side}_hip_yaw_joint"] = (100.0, 2.5)
    GAINS[f"{side}_hip_roll_joint"] = (100.0, 2.5)
    GAINS[f"{side}_hip_pitch_joint"] = (100.0, 2.5)
    GAINS[f"{side}_knee_joint"] = (200.0, 5.0)
    GAINS[f"{side}_ankle_pitch_joint"] = (20.0, 0.2)
    GAINS[f"{side}_ankle_roll_joint"] = (20.0, 0.1)
    for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw"):
        GAINS[f"{side}_{j}_joint"] = (40.0, 10.0)
for j in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
    GAINS[j] = (200.0, 5.0)

# torque limits, this MJCF's own actuator ctrlrange (real G1 hardware effort limits).
TORQUE_LIMIT = {}
for side in ("left", "right"):
    TORQUE_LIMIT[f"{side}_hip_yaw_joint"] = 88.0
    TORQUE_LIMIT[f"{side}_hip_roll_joint"] = 88.0
    TORQUE_LIMIT[f"{side}_hip_pitch_joint"] = 88.0
    TORQUE_LIMIT[f"{side}_knee_joint"] = 139.0
    TORQUE_LIMIT[f"{side}_ankle_pitch_joint"] = 50.0
    TORQUE_LIMIT[f"{side}_ankle_roll_joint"] = 50.0
    for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"):
        TORQUE_LIMIT[f"{side}_{j}_joint"] = 25.0
    for j in ("wrist_pitch", "wrist_yaw"):
        TORQUE_LIMIT[f"{side}_{j}_joint"] = 5.0
TORQUE_LIMIT["waist_yaw_joint"] = 88.0
TORQUE_LIMIT["waist_roll_joint"] = 50.0
TORQUE_LIMIT["waist_pitch_joint"] = 50.0

ACTION_SCALE = 0.5
CONTROL_HZ = 30.0
EPISODE_LENGTH_S = 10.0
TERMINATION_HEIGHT = 0.5


def quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class PolicyNet(nn.Module):
    """Matches skrl_g1_amp_linden_cfg.yaml's policy network exactly: [1024, 512] relu -> 29
    actions. Deterministic rollout, so log_std (used only for sampling during training) is
    loaded but never used."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net_container = nn.Sequential(
            nn.Linear(obs_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, act_dim),
        )

    def forward(self, x):
        return self.net_container(x)


def load_policy(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    obs_dim = ckpt["policy"]["net_container.0.weight"].shape[1]
    act_dim = ckpt["policy"]["net_container.4.weight"].shape[0]
    policy = PolicyNet(obs_dim, act_dim)
    state_dict = {k: v for k, v in ckpt["policy"].items() if k.startswith("net_container")}
    policy.load_state_dict(state_dict)
    policy.eval()
    mean = ckpt["state_preprocessor"]["running_mean"].numpy()
    var = ckpt["state_preprocessor"]["running_variance"].numpy()
    return policy, mean, var


class G1MujocoRollout:
    def __init__(self, mjcf_path: str, command_xyz: tuple[float, float, float],
                 motion_npz: str | None = None, start_frame: int = 0):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.command = np.array(command_xyz, dtype=np.float32)

        self.qpos_adr = {n: self.model.joint(n).qposadr[0] for n in DOF_NAMES}
        self.dof_adr = {n: self.model.joint(n).dofadr[0] for n in DOF_NAMES}
        self.act_adr = {self.model.actuator(i).name: i for i in range(self.model.nu)}
        self.pelvis_id = self.model.body("pelvis").id
        self.key_body_ids = [self.model.body(n).id for n in KEY_BODY_NAMES]

        self.default_qpos = np.array([DEFAULT_JOINT_POS[n] for n in DOF_NAMES], dtype=np.float32)
        self.gains = np.array([GAINS[n] for n in DOF_NAMES], dtype=np.float32)  # (29, 2): kp, kd
        self.torque_limit = np.array([TORQUE_LIMIT[n] for n in DOF_NAMES], dtype=np.float32)

        self.substeps = max(1, round(1.0 / CONTROL_HZ / self.model.opt.timestep))
        self.reset(motion_npz, start_frame)

    def reset(self, motion_npz: str | None = None, start_frame: int = 0):
        # cfg.reset_strategy = "random" for this policy: every episode it ever saw, in training
        # AND in eval/video rollouts, started mid-stride -- sampled from the reference clip with
        # that frame's own joint/root velocities, never a static zero-velocity default pose. A
        # static idle reset is out-of-distribution for it and was the actual cause of the
        # instant collapse seen before this fix (action magnitude grew every step, the policy
        # fighting a starting state it never learned to recover from).
        mujoco.mj_resetData(self.model, self.data)
        if motion_npz is not None:
            d = np.load(motion_npz)
            bn = d["body_names"].tolist()
            pel = bn.index("pelvis")
            quat = d["body_rotations"][start_frame, pel]  # wxyz
            self.data.qpos[0:3] = d["body_positions"][start_frame, pel]
            self.data.qpos[3:7] = quat
            self.data.qvel[0:3] = d["body_linear_velocities"][start_frame, pel]  # world frame
            # npz angular velocity is world-frame (see convert_gmr_to_npz.py); MuJoCo's free
            # joint qvel[3:6] wants body-frame, so rotate world -> body with R^T.
            self.data.qvel[3:6] = quat_to_rotmat(quat).T @ d["body_angular_velocities"][start_frame, pel]
            dof_names_npz = d["dof_names"].tolist()
            for n in DOF_NAMES:
                i = dof_names_npz.index(n)
                self.data.qpos[self.qpos_adr[n]] = d["dof_positions"][start_frame, i]
                self.data.qvel[self.dof_adr[n]] = d["dof_velocities"][start_frame, i]
        else:
            self.data.qpos[0:3] = [0.0, 0.0, 0.793]
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            for n in DOF_NAMES:
                self.data.qpos[self.qpos_adr[n]] = DEFAULT_JOINT_POS[n]
        mujoco.mj_forward(self.model, self.data)

    def get_dof(self, arr_qpos_or_qvel: np.ndarray, adr_map: dict) -> np.ndarray:
        return np.array([arr_qpos_or_qvel[adr_map[n]] for n in DOF_NAMES], dtype=np.float32)

    def get_observation(self) -> np.ndarray:
        d = self.data
        dof_pos = self.get_dof(d.qpos, self.qpos_adr)
        dof_vel = self.get_dof(d.qvel, self.dof_adr)

        root_quat = d.qpos[3:7].copy()  # wxyz
        rotmat = quat_to_rotmat(root_quat)
        tangent = rotmat[:, 0]
        normal = rotmat[:, 2]

        root_lin_vel_w = d.qvel[0:3].copy()  # free joint linear qvel IS world-frame in MuJoCo
        root_ang_vel_body = d.qvel[3:6].copy()  # free joint angular qvel is body-frame
        root_ang_vel_w = rotmat @ root_ang_vel_body

        pelvis_pos = d.xpos[self.pelvis_id].copy()
        root_height = np.array([pelvis_pos[2]], dtype=np.float32)
        key_body_rel = np.array(
            [d.xpos[bid] - pelvis_pos for bid in self.key_body_ids], dtype=np.float32
        ).flatten()

        amp_obs = np.concatenate([
            dof_pos, dof_vel, root_height, tangent, normal,
            root_lin_vel_w, root_ang_vel_w, key_body_rel,
        ]).astype(np.float32)
        return np.concatenate([amp_obs, self.command])

    def apply_action(self, action: np.ndarray):
        # the target (setpoint) is held fixed for the whole control step, matching training's
        # decimation -- but PD torque is a feedback law and must be recomputed every physics
        # substep from the current qpos/qvel, not frozen at the state when the target was set.
        # A torque held constant across 34ms (17 substeps at dt=0.002) never backs off as the
        # joint approaches the target, which just pumps energy in -- this was the actual bug
        # behind the robot launching upward instead of walking.
        target = self.default_qpos + ACTION_SCALE * action
        kp, kd = self.gains[:, 0], self.gains[:, 1]
        act_idx = [self.act_adr[n.replace("_joint", "")] for n in DOF_NAMES]
        for _ in range(self.substeps):
            dof_pos = self.get_dof(self.data.qpos, self.qpos_adr)
            dof_vel = self.get_dof(self.data.qvel, self.dof_adr)
            torque = kp * (target - dof_pos) - kd * dof_vel
            torque = np.clip(torque, -self.torque_limit, self.torque_limit)
            self.data.ctrl[act_idx] = torque
            mujoco.mj_step(self.model, self.data)

    def pelvis_height(self) -> float:
        return float(self.data.xpos[self.pelvis_id, 2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mjcf", default="unitree_robots/g1/scene_29dof.xml")
    parser.add_argument("--out", default="sim2sim_rollout.mp4")
    parser.add_argument("--command", type=float, nargs=3, default=[0.8, 0.0, 0.0],
                         help="lin_vel_x lin_vel_y ang_vel_z, base frame")
    parser.add_argument("--motion_npz", default=None,
                         help="reference motion npz to sample the reset state from (matches "
                              "cfg.reset_strategy='random' -- this policy never trained from a "
                              "static default pose). Omit for a static default-pose reset.")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--render_fps", type=float, default=30.0)
    parser.add_argument("--distance", type=float, default=3.2)
    parser.add_argument("--elevation", type=float, default=-10.0)
    parser.add_argument("--azimuth", type=float, default=90.0)
    args = parser.parse_args()

    policy, obs_mean, obs_var = load_policy(args.checkpoint)
    sim = G1MujocoRollout(args.mjcf, tuple(args.command), args.motion_npz, args.start_frame)

    renderer = mujoco.Renderer(sim.model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = args.distance, args.elevation, args.azimuth

    n_steps = int(EPISODE_LENGTH_S * CONTROL_HZ)
    images = []
    terminated_at = None
    for t in range(n_steps):
        obs = sim.get_observation()
        obs_norm = (obs - obs_mean) / np.sqrt(obs_var + 1e-8)
        obs_norm = np.clip(obs_norm, -5.0, 5.0)
        with torch.no_grad():
            action = policy(torch.from_numpy(obs_norm).float().unsqueeze(0)).squeeze(0).numpy()
        sim.apply_action(action)

        pelvis_pos = sim.data.xpos[sim.pelvis_id]
        cam.lookat[:] = pelvis_pos + np.array([0.0, 0.0, -0.2])
        renderer.update_scene(sim.data, camera=cam)
        images.append(renderer.render())

        h = sim.pelvis_height()
        if h < TERMINATION_HEIGHT and terminated_at is None:
            terminated_at = t
            print(f"[step {t}] pelvis height {h:.3f} < {TERMINATION_HEIGHT} -- fell")

    writer = imageio.get_writer(args.out, fps=args.render_fps, quality=8)
    for im in images:
        writer.append_data(im)
    writer.close()

    status = f"fell at step {terminated_at}/{n_steps}" if terminated_at is not None else \
        f"stayed up the full {n_steps} steps"
    print(f"Wrote {args.out}: {len(images)} frames. Result: {status}")


if __name__ == "__main__":
    main()
