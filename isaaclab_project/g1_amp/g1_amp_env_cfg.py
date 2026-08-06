"""Project 2: G1 AMP environment config — velocity-tracking task reward layered on
an AMP style reward (see g1_amp_env.py for the reward/observation logic).

Adapted from linden713/humanoid_amp (BSD-3-Clause), which wires Isaac Lab's stock AMP
machinery to a direct-workflow G1 env but only for pure imitation (all task reward
scales are zero there).

Robot asset: Isaac Lab's own G1_29DOF_CFG (isaaclab_assets), NOT G1_MINIMAL_CFG (what
Project 1 / stock G1FlatEnvCfg actually uses). Confirmed via a smoke test: G1_MINIMAL_CFG
is a 37-joint variant (elbow_pitch+elbow_roll, full multi-finger hands, single torso_joint)
-- structurally incompatible with our GMR-retargeted motion data (which follows the
29-DOF convention: waist_yaw/roll/pitch split, single-DOF elbow, wrist_roll/pitch/yaw --
same as LAFAN1, G1 Moves, and linden713/humanoid_amp's own bundled robot). G1_29DOF_CFG
matches that convention exactly. Re-retargeting for G1_MINIMAL_CFG's exact joint layout
isn't something GMR supports out of the box (its two presets are 29-DOF no-hands and
43-DOF with-hands, neither with this elbow/wrist layout), so this repo's G1 doesn't have
pixel-identical PD gains/hand model to Project 1's -- a deliberate tradeoff, not an
oversight.
"""

from __future__ import annotations

import copy
import os
from dataclasses import MISSING

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg

from isaaclab_assets import G1_29DOF_CFG

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")

# G1_29DOF_CFG ships mocap-tracking gains for the upper body (arms stiffness 3000, waist 5000
# -- position-hold values for kinematic replay, not RL). With noisy policy position targets
# (std 1.0 at action_scale 0.5 -> ~±0.5 rad) and a 300 N*m effort limit, gains that stiff turn
# the arms into a destabilizing torque source on the torso every step. Soften arms/waist to
# Project 1's values (G1_MINIMAL_CFG resolved config: arms 40/10, torso 200/5) for parity;
# legs/feet keep this asset's own values, which are already in Project 1's ballpark.
# deepcopy rather than .replace(): configclass replace() copies shallowly for fields not being
# replaced, so mutating .actuators on a replace()'d copy would write through to the shared
# G1_29DOF_CFG instance.
_G1_29DOF_SOFT_ARMS_CFG = copy.deepcopy(G1_29DOF_CFG)
_G1_29DOF_SOFT_ARMS_CFG.prim_path = "/World/envs/env_.*/Robot"
_G1_29DOF_SOFT_ARMS_CFG.actuators["arms"].stiffness = 40.0
_G1_29DOF_SOFT_ARMS_CFG.actuators["arms"].damping = 10.0
_G1_29DOF_SOFT_ARMS_CFG.actuators["waist"].stiffness = 200.0
_G1_29DOF_SOFT_ARMS_CFG.actuators["waist"].damping = 5.0


@configclass
class G1AmpEnvCfg(DirectRLEnvCfg):
    """G1 AMP environment config: velocity-tracking task reward + AMP style reward."""

    # task reward (velocity tracking, mirrors Project 1's track_lin_vel_xy_exp/track_ang_vel_z_exp
    # scales/shape, reimplemented in g1_amp_env.py since this is a direct-workflow env, not a
    # manager-based one, so there's no reward-manager term to reuse directly)
    rew_lin_vel_xy = 1.0
    rew_ang_vel_z = 1.0  # matches stock G1FlatEnvCfg's override (G1Rewards' own default is 2.0)
    # this is std**2, not std itself — Isaac Lab's own track_*_exp mdp functions take `std` and
    # divide error by std**2 internally; 0.25 here == Project 1's std=math.sqrt(0.25), same weights
    rew_track_sigma = 0.25

    # regularization — matches Project 1's *stock* g1_flat run (the full reward set, not the
    # stripped-down G1FlatMinimalEnvCfg used for the blog's "what breaks without X" narrative;
    # see logs/rsl_rl/g1_flat/.../params/env.yaml in rl-wbc-g1-baseline for the exact resolved
    # weights this was transcribed from). An earlier pass here only restored the "safety net"
    # terms (termination/action_rate/acc/torque/orientation) on the theory that AMP's style
    # reward alone would cover "look natural" the way Project 1's joint_deviation_* terms did
    # by hand -- but that's belt-and-suspenders, not either/or, so this pass adds those back too.
    #
    # feet_air_time and feet_slide are NOT included: both need a ContactSensor on the ankle
    # bodies, which this direct-workflow scene doesn't set up yet. Deferred as a follow-up, not
    # dropped for a design reason.
    rew_termination = -200.0
    rew_action_rate_l2 = -0.005
    rew_joint_pos_limits = -1.0  # ankle pitch/roll only, matching Project 1's asset_cfg restriction
    rew_joint_acc_l2 = -1.0e-7  # hip + knee only, matching Project 1's asset_cfg restriction
    rew_dof_torques_l2 = -2.0e-6  # hip + knee only, matching Project 1's asset_cfg restriction
    rew_flat_orientation_l2 = -1.0
    rew_lin_vel_z_l2 = -0.2
    rew_ang_vel_xy_l2 = -0.05
    rew_joint_deviation_hip = -0.1  # hip_yaw + hip_roll only (hip_pitch drives the actual gait)
    rew_joint_deviation_arms = -0.1  # shoulder pitch/roll/yaw + elbow (single-DOF here, not pitch+roll)
    rew_joint_deviation_torso = -0.1  # all 3 waist joints (Project 1's G1_MINIMAL_CFG has 1; we have 3)
    # joint_deviation_fingers has no equivalent here: fingers aren't in action_dof_indexes, so
    # they never move from default_joint_pos regardless -- the term would always be exactly 0.

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

    # target = default_joint_pos + action_scale * action, matching Project 1's actual action
    # space exactly (JointPositionAction, scale=0.5, use_default_offset=true) -- NOT the original
    # AMP reference implementation's "joint_limit_midpoint + full_joint_range * action" convention
    # (see the note in g1_amp_env.py's __init__ for why that was likely the real root cause of
    # episodes reliably dying within ~6-8 steps regardless of reward/PPO-hyperparameter changes).
    action_scale = 0.5

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

    # robot — Isaac Lab's own G1_29DOF_CFG (matching our retargeted motion data's joint
    # convention; see module docstring for why this isn't Post 1's G1_MINIMAL_CFG), with
    # arm/waist actuator gains softened to Project 1's values (see _G1_29DOF_SOFT_ARMS_CFG).
    robot: ArticulationCfg = _G1_29DOF_SOFT_ARMS_CFG


@configclass
class G1AmpStrutWalkEnvCfg(G1AmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "G1_strut_walk.npz")


@configclass
class G1AmpStrutWalkEnvCfg_PLAY(G1AmpStrutWalkEnvCfg):
    """Smaller-scene eval variant, matching Isaac Lab's `_PLAY` convention."""

    def __post_init__(self):
        self.scene.num_envs = 32
        self.scene.env_spacing = 3.0


@configclass
class G1AmpLindenWalkEnvCfg(G1AmpEnvCfg):
    """Binary-search control condition (2026-07-31): linden713/humanoid_amp's own known-working
    G1_walk.npz (LAFAN1 walk, 60 fps, demo videos public) run through OUR env, paired with that
    repo's verbatim agent config (skrl_g1_amp_linden_cfg.yaml, pure imitation). If this walks,
    our env/action/observation pipeline is exonerated end-to-end and the strut failure narrows
    to {our clip, our PPO regime}. Only delta vs their setup on the env side: our action mapping
    (default_joint_pos + 0.5*action) and softened arm/waist gains -- both deliberate, both
    already validated by the 0.5/0.5 combined run.
    NOTE: G1_linden_walk.npz is their file with `*_rubber_hand` bodies renamed to
    `*_hand_palm_link` to match this env's key-body lookup; motion content untouched
    (feet ~1 cm off ground -- close enough to grounded that no realignment was applied)."""

    motion_file = os.path.join(MOTIONS_DIR, "G1_linden_walk.npz")


@configclass
class G1AmpLindenWalkEnvCfg_PLAY(G1AmpLindenWalkEnvCfg):
    """Smaller-scene eval variant, matching Isaac Lab's `_PLAY` convention.

    Viewer follows env 0's robot root (close-up follow-cam for videos). Set here rather than
    via hydra override because Isaac Lab's config updater rejects overrides of None-defaulted
    fields (viewer.asset_name) with a NoneType type-check error."""

    def __post_init__(self):
        self.scene.num_envs = 32
        self.scene.env_spacing = 3.0
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        # front-on, further back: camera sits ahead of the robot along its direction of
        # travel (+X, the clip's own heading), so it walks toward camera. lookat is deliberately
        # low (near ankle height, not chest height) -- aiming at the robot's center pushes the
        # feet to the bottom edge of frame since the robot has much more height above the hips
        # than below them; aiming low keeps feet clear with margin at the cost of some empty
        # headroom above the head, the safer direction to err in.
        self.viewer.eye = (4.5, 0.0, 0.85)
        self.viewer.lookat = (0.0, 0.0, 0.35)
