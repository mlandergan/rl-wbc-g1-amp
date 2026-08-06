"""Project 2: G1 AMP environment — adapted from linden713/humanoid_amp's G1AmpEnv
(BSD-3-Clause), which is pure imitation (all task reward scales zero, no command).

Adds on top of that reference:
  - a resampled base-frame velocity command (lin_vel_x, lin_vel_y, ang_vel_z), matching
    Project 1's task, sampled from the narrow range in g1_amp_env_cfg.py (chosen to match
    Strut_Walking_loop_g1's own ~0.83 m/s pace rather than Project 1's full command range)
  - the command appended to the *policy* observation only — the AMP observation buffer
    stays exactly the reference's style-only features, so the discriminator never sees
    the command and can't use it to shortcut style scoring
  - a track_lin_vel_xy_exp / track_ang_vel_z_exp task reward (mirrors Project 1's
    manager-based reward terms, reimplemented directly since this is a direct-workflow
    env with no reward manager)

NOTE: the task-reward terms use *base-frame* velocity (root_lin_vel_b/root_ang_vel_b) —
what the command means. The AMP style features below still use *world-frame* velocity
(body_lin_vel_w/body_ang_vel_w), unchanged from the reference implementation. These are
deliberately different quantities for different roles; see the note in
convert_gmr_to_npz.py about aligning the reference clip's world-frame heading to +X so
the two don't fight each other.
"""

from __future__ import annotations

import re

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, sample_uniform

from .g1_amp_env_cfg import G1AmpEnvCfg
from .motions import MotionLoader


class G1AmpEnv(DirectRLEnv):
    cfg: G1AmpEnvCfg

    def __init__(self, cfg: G1AmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # load motion
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        # G1_29DOF_CFG has 43 joints (29 "real" body joints + 14 finger joints); our motion
        # data only covers the 29 real ones (retargeted with no finger animation). action_
        # dof_indexes is where those 29 land in the robot's full 43-joint ordering -- fingers
        # are excluded from the action space entirely and held at their default pose (see
        # _apply_action/_reset_strategy_random), since nothing here cares about finger motion.
        motion_joint_names = [n for n in self.robot.data.joint_names if n in self._motion_loader.dof_names]
        assert len(motion_joint_names) == self.cfg.action_space, (
            f"Expected {self.cfg.action_space} robot joints covered by the motion file, "
            f"found {len(motion_joint_names)}: {motion_joint_names}"
        )
        self.action_dof_indexes = [self.robot.data.joint_names.index(n) for n in motion_joint_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(motion_joint_names)

        # joint-group index subsets (positions *within* the 29-long action_dof_indexes slice, not
        # into the robot's full 43-joint space) for the regularization terms below that only apply
        # to specific joint groups, matching Project 1's asset_cfg(joint_names=...) restrictions.
        def _match(patterns: list[str]) -> list[int]:
            return [i for i, n in enumerate(motion_joint_names) if any(re.fullmatch(p, n) for p in patterns)]

        self.hip_knee_dof_indexes = _match([r".*_hip_.*", r".*_knee_joint"])
        self.ankle_dof_indexes = _match([r".*_ankle_pitch_joint", r".*_ankle_roll_joint"])
        self.hip_deviation_dof_indexes = _match([r".*_hip_yaw_joint", r".*_hip_roll_joint"])
        self.arm_deviation_dof_indexes = _match([
            r".*_shoulder_pitch_joint", r".*_shoulder_roll_joint", r".*_shoulder_yaw_joint", r".*_elbow_joint",
        ])
        self.torso_deviation_dof_indexes = _match([r"waist_yaw_joint", r"waist_roll_joint", r"waist_pitch_joint"])

        # NOTE: no per-joint action_offset/action_scale tensors here (that was the original
        # AMP-reference-implementation convention: target = joint_limit_midpoint + full_joint_range
        # * action). Project 1's actual action space is target = default_joint_pos + 0.5 * action
        # (a small, bounded nudge from the robot's own standing pose, not a swing across the whole
        # range of motion centered on the joint-limit midpoint) -- see cfg.action_scale and
        # _apply_action. The full-range version was traced as the likely root cause of episodes
        # reliably dying within ~6-8 steps across every reward/PPO-hyperparameter variant tried:
        # even a single std of Gaussian action noise could swing a PD target across a joint's
        # entire range, untethered from the actual current pose, regardless of policy quality.

        # DOF and key body indexes
        key_body_names = [
            "left_shoulder_pitch_link", "right_shoulder_pitch_link",
            "left_elbow_link", "right_elbow_link",
            "right_hip_yaw_link", "left_hip_yaw_link",
            "right_hand_palm_link", "left_hand_palm_link",
            "right_ankle_roll_link", "left_ankle_roll_link",
        ]

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # reconfigure AMP observation space according to the number of observations and create the buffer
        # (style-only — the velocity command is never part of this, see module docstring)
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

        # velocity command buffer: (lin_vel_x, lin_vel_y, ang_vel_z) in the robot's base frame
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

        # action buffers for action_rate_l2 (penalizes actions changing too fast between steps,
        # matching Project 1's mdp.action_rate_l2 term -- there's no action manager here to hold
        # this for us since this is a direct-workflow env, so it's tracked by hand)
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

        # resample commands whose window has elapsed (mid-episode, in general; with the
        # default config command_resample_time_s == episode_length_s so this only actually
        # fires on reset, but stays correct if the two are ever set independently)
        resample_steps = int(self.cfg.command_resample_time_s / (self.cfg.sim.dt * self.cfg.decimation))
        due = (self.episode_length_buf % max(resample_steps, 1) == 0) & (self.episode_length_buf > 0)
        env_ids = due.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._resample_commands(env_ids)

    def _resample_commands(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        r = self.cfg
        self.commands[env_ids, 0] = sample_uniform(*r.command_lin_vel_x_range, (n,), self.device)
        self.commands[env_ids, 1] = sample_uniform(*r.command_lin_vel_y_range, (n,), self.device)
        self.commands[env_ids, 2] = sample_uniform(*r.command_ang_vel_z_range, (n,), self.device)

    def _apply_action(self):
        # fingers (not in action_dof_indexes) hold their default pose; only the 29
        # motion-covered joints are actually driven by the policy's actions, as a small nudge
        # away from the robot's own default/standing pose -- matches Project 1's actual action
        # space (JointPositionAction, scale=0.5, use_default_offset=true), not the original AMP
        # reference implementation's "swing across the whole joint range" convention (see the
        # note in __init__ for why that was likely the real root cause of instant falls).
        target = self.robot.data.default_joint_pos.clone()
        target[:, self.action_dof_indexes] += self.cfg.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        # style-only AMP observation, unchanged from the reference implementation except
        # restricted to the 29 motion-covered joints (not the full 43 incl. fingers) so
        # this stays dimensionally consistent with the reference motion's own 29-dim data
        amp_obs = compute_obs(
            self.robot.data.joint_pos[:, self.action_dof_indexes],
            self.robot.data.joint_vel[:, self.action_dof_indexes],
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = amp_obs.clone()
        # _get_rewards() runs before _get_observations() in DirectRLEnv.step() and sets
        # self.extras["log"] for the reward breakdown -- update here, don't reassign
        # self.extras wholesale, or that breakdown gets silently wiped out before it ever
        # reaches the trainer's info dict (confirmed: this is exactly what was happening).
        self.extras["amp_obs"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)

        # policy observation = style obs + velocity command, so the policy can condition its
        # actions on what it's being asked to do (the discriminator above never sees this)
        policy_obs = torch.cat((amp_obs, self.commands), dim=-1)
        return {"policy": policy_obs}

    def _get_rewards(self) -> torch.Tensor:
        # pre-slice per-joint-group tensors here (plain python/torch indexing) rather than inside
        # the jit-scripted compute_rewards, so that function only ever does simple reductions and
        # never needs to know about index lists (TorchScript's typing for that is a headache).
        joint_pos_29 = self.robot.data.joint_pos[:, self.action_dof_indexes]
        default_joint_pos_29 = self.robot.data.default_joint_pos[:, self.action_dof_indexes]
        soft_limits_29 = self.robot.data.soft_joint_pos_limits[:, self.action_dof_indexes]
        joint_acc_29 = self.robot.data.joint_acc[:, self.action_dof_indexes]
        applied_torque_29 = self.robot.data.applied_torque[:, self.action_dof_indexes]

        total_reward, reward_log = compute_rewards(
            self.cfg.rew_lin_vel_xy,
            self.cfg.rew_ang_vel_z,
            self.cfg.rew_track_sigma,
            self.cfg.rew_termination,
            self.cfg.rew_action_rate_l2,
            self.cfg.rew_joint_pos_limits,
            self.cfg.rew_joint_acc_l2,
            self.cfg.rew_dof_torques_l2,
            self.cfg.rew_flat_orientation_l2,
            self.cfg.rew_lin_vel_z_l2,
            self.cfg.rew_ang_vel_xy_l2,
            self.cfg.rew_joint_deviation_hip,
            self.cfg.rew_joint_deviation_arms,
            self.cfg.rew_joint_deviation_torso,
            self.commands,
            self.robot.data.root_lin_vel_b,
            self.robot.data.root_ang_vel_b,
            self.reset_terminated,
            self.actions,
            self.prev_actions,
            self.robot.data.projected_gravity_b,
            joint_acc_29[:, self.hip_knee_dof_indexes],
            applied_torque_29[:, self.hip_knee_dof_indexes],
            joint_pos_29[:, self.ankle_dof_indexes],
            soft_limits_29[:, self.ankle_dof_indexes],
            joint_pos_29[:, self.hip_deviation_dof_indexes] - default_joint_pos_29[:, self.hip_deviation_dof_indexes],
            joint_pos_29[:, self.arm_deviation_dof_indexes] - default_joint_pos_29[:, self.arm_deviation_dof_indexes],
            joint_pos_29[:, self.torso_deviation_dof_indexes] - default_joint_pos_29[:, self.torso_deviation_dof_indexes],
        )
        self.extras["log"] = reward_log
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._resample_commands(env_ids)

    # reset strategies

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        motion_torso_index = self._motion_loader.get_body_index(["pelvis"])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, motion_torso_index] + self.scene.env_origins[env_ids]
        # small clearance so the stance foot doesn't spawn interpenetrating the ground. The
        # motion data itself is ground-aligned by the converter (lowest sole at z=0) -- an
        # earlier version of the npz floated ~0.08-0.09 m AND this offset was 0.05, so every
        # reset started with a ~0.13-0.17 m free-fall drop onto the ground.
        root_state[:, 2] += 0.02
        root_state[:, 3:7] = body_rotations[:, motion_torso_index]
        root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
        root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]
        # fingers (not in action_dof_indexes) reset to the robot's own default pose; only
        # the 29 motion-covered joints get values from the sampled reference motion
        dof_pos = self.robot.data.default_joint_pos[env_ids].clone()
        dof_vel = self.robot.data.default_joint_vel[env_ids].clone()
        dof_pos[:, self.action_dof_indexes] = dof_positions[:, self.motion_dof_indexes]
        dof_vel[:, self.action_dof_indexes] = dof_velocities[:, self.motion_dof_indexes]

        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    # env methods

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )
        return amp_observation.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_rewards(
    rew_lin_vel_xy: float,
    rew_ang_vel_z: float,
    rew_track_sigma: float,
    rew_scale_termination: float,
    rew_scale_action_rate_l2: float,
    rew_scale_joint_pos_limits: float,
    rew_scale_joint_acc_l2: float,
    rew_scale_dof_torques_l2: float,
    rew_scale_flat_orientation_l2: float,
    rew_scale_lin_vel_z_l2: float,
    rew_scale_ang_vel_xy_l2: float,
    rew_scale_joint_deviation_hip: float,
    rew_scale_joint_deviation_arms: float,
    rew_scale_joint_deviation_torso: float,
    commands: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    reset_terminated: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    hip_knee_joint_acc: torch.Tensor,
    hip_knee_applied_torque: torch.Tensor,
    ankle_joint_pos: torch.Tensor,
    ankle_soft_joint_pos_limits: torch.Tensor,
    hip_deviation: torch.Tensor,
    arm_deviation: torch.Tensor,
    torso_deviation: torch.Tensor,
):
    # task reward: velocity tracking, base frame — mirrors Project 1's
    # track_lin_vel_xy_exp / track_ang_vel_z_exp manager-based reward terms
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - root_lin_vel_b[:, :2]), dim=1)
    rew_task_lin_vel = rew_lin_vel_xy * torch.exp(-lin_vel_error / rew_track_sigma)

    ang_vel_error = torch.square(commands[:, 2] - root_ang_vel_b[:, 2])
    rew_task_ang_vel = rew_ang_vel_z * torch.exp(-ang_vel_error / rew_track_sigma)

    # regularization — matches Project 1's stock g1_flat run (see g1_amp_env_cfg.py)
    rew_termination = rew_scale_termination * reset_terminated.float()
    rew_action_rate_l2 = rew_scale_action_rate_l2 * torch.sum(torch.square(actions - prev_actions), dim=1)

    out_of_limits = -(ankle_joint_pos - ankle_soft_joint_pos_limits[:, :, 0]).clip(max=0.0)
    out_of_limits += (ankle_joint_pos - ankle_soft_joint_pos_limits[:, :, 1]).clip(min=0.0)
    rew_joint_pos_limits = rew_scale_joint_pos_limits * torch.sum(out_of_limits, dim=1)

    rew_joint_acc_l2 = rew_scale_joint_acc_l2 * torch.sum(torch.square(hip_knee_joint_acc), dim=1)
    rew_dof_torques_l2 = rew_scale_dof_torques_l2 * torch.sum(torch.square(hip_knee_applied_torque), dim=1)
    rew_flat_orientation_l2 = rew_scale_flat_orientation_l2 * torch.sum(
        torch.square(projected_gravity_b[:, :2]), dim=1
    )
    rew_lin_vel_z_l2 = rew_scale_lin_vel_z_l2 * torch.square(root_lin_vel_b[:, 2])
    rew_ang_vel_xy_l2 = rew_scale_ang_vel_xy_l2 * torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1)
    rew_joint_deviation_hip = rew_scale_joint_deviation_hip * torch.sum(torch.abs(hip_deviation), dim=1)
    rew_joint_deviation_arms = rew_scale_joint_deviation_arms * torch.sum(torch.abs(arm_deviation), dim=1)
    rew_joint_deviation_torso = rew_scale_joint_deviation_torso * torch.sum(torch.abs(torso_deviation), dim=1)

    total_reward = (
        rew_task_lin_vel + rew_task_ang_vel
        + rew_termination + rew_action_rate_l2 + rew_joint_pos_limits
        + rew_joint_acc_l2 + rew_dof_torques_l2 + rew_flat_orientation_l2
        + rew_lin_vel_z_l2 + rew_ang_vel_xy_l2
        + rew_joint_deviation_hip + rew_joint_deviation_arms + rew_joint_deviation_torso
    )

    log = {
        "rew_task_lin_vel": rew_task_lin_vel.mean(),
        "rew_task_ang_vel": rew_task_ang_vel.mean(),
        "rew_termination": rew_termination.mean(),
        "rew_action_rate_l2": rew_action_rate_l2.mean(),
        "rew_joint_pos_limits": rew_joint_pos_limits.mean(),
        "rew_joint_acc_l2": rew_joint_acc_l2.mean(),
        "rew_dof_torques_l2": rew_dof_torques_l2.mean(),
        "rew_flat_orientation_l2": rew_flat_orientation_l2.mean(),
        "rew_lin_vel_z_l2": rew_lin_vel_z_l2.mean(),
        "rew_ang_vel_xy_l2": rew_ang_vel_xy_l2.mean(),
        "rew_joint_deviation_hip": rew_joint_deviation_hip.mean(),
        "rew_joint_deviation_arms": rew_joint_deviation_arms.mean(),
        "rew_joint_deviation_torso": rew_joint_deviation_torso.mean(),
    }
    return total_reward, log
