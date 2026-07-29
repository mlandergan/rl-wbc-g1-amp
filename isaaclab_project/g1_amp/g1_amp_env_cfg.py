"""Project 2: G1 AMP environment config — velocity-tracking task reward layered on
an AMP style reward (see g1_amp_env.py for the reward/observation logic).

Adapted from linden713/humanoid_amp (BSD-3-Clause), which wires Isaac Lab's stock AMP
machinery to a direct-workflow G1 env but only for pure imitation (all task reward
scales are zero there). This adds a velocity-command task reward on top, and uses
Isaac Lab's own G1_MINIMAL_CFG (isaaclab_assets) instead of that reference's bundled
custom robot config, so the robot model matches Project 1 (rl-wbc-g1-baseline) exactly.
"""

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg

from isaaclab_assets import G1_MINIMAL_CFG

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class G1AmpEnvCfg(DirectRLEnvCfg):
    """G1 AMP environment config: velocity-tracking task reward + AMP style reward."""

    # task reward (velocity tracking, mirrors Project 1's track_lin_vel_xy_exp/track_ang_vel_z_exp
    # scales/shape, reimplemented in g1_amp_env.py since this is a direct-workflow env, not a
    # manager-based one, so there's no reward-manager term to reuse directly)
    rew_lin_vel_xy = 1.0
    rew_ang_vel_z = 0.5
    # this is std**2, not std itself — Isaac Lab's own track_*_exp mdp functions take `std` and
    # divide error by std**2 internally; 0.25 here == Project 1's std=math.sqrt(0.25), same weights
    rew_track_sigma = 0.25

    # regularization (kept small/zero to start; the AMP style reward is doing most of the
    # "look natural" work Project 1's joint_deviation_* terms did by hand)
    rew_termination = -0.0
    rew_action_l2 = -0.0
    rew_joint_pos_limits = -0.0
    rew_joint_acc_l2 = -0.0
    rew_joint_vel_l2 = -0.0

    # velocity command range — narrowed to roughly match Strut_Walking_loop_g1's own pace
    # (~0.83 m/s forward, near-straight-line) rather than reusing Project 1's full command range
    # (0.0-1.0 m/s x, +/-0.5 m/s y, +/-1.0 rad/s yaw), per the option-1 plan: keep the task reward's
    # commanded velocity consistent with what the reference clip's own root motion actually shows,
    # so the task reward and the AMP style reward aren't pulling in different directions.
    command_lin_vel_x_range = (0.5, 1.0)
    command_lin_vel_y_range = (-0.1, 0.1)
    command_ang_vel_z_range = (-0.2, 0.2)
    command_resample_time_s = 10.0

    # env
    episode_length_s = 10.0
    decimation = 2

    # spaces — 71 task-obs dims (29 dof_pos + 29 dof_vel + 1 root height + 6 tangent/normal
    # + 3 root lin vel + 3 root ang vel) + 3*10 key-body relative positions + 3 velocity command
    observation_space = 71 + 3 * 10 + 3
    action_space = 29
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 71 + 3 * 10  # AMP obs stays style-only, no command — see g1_amp_env.py

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING
    reference_body = "pelvis"
    reset_strategy = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot — Isaac Lab's own G1_MINIMAL_CFG, same asset Project 1 uses (via the stock
    # G1FlatEnvCfg), not the reference implementation's separately-bundled robot config.
    robot: ArticulationCfg = G1_MINIMAL_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class G1AmpStrutWalkEnvCfg(G1AmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "G1_strut_walk.npz")


@configclass
class G1AmpStrutWalkEnvCfg_PLAY(G1AmpStrutWalkEnvCfg):
    """Smaller-scene eval variant, matching Isaac Lab's `_PLAY` convention."""

    def __post_init__(self):
        self.scene.num_envs = 32
        self.scene.env_spacing = 3.0
