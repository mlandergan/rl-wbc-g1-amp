# Project 1 baseline override — see project_01_baseline_locomotion.md section 5 for rationale.
#
# This subclasses Isaac Lab's stock G1FlatEnvCfg (which already re-weights the shared
# LocomotionVelocityRoughEnvCfg terms for flat ground) and strips it down to the reward set
# described in the tutorial blueprint's "minimal, interpretable reward" section:
#   velocity tracking, upright posture, base height, energy/torque penalty,
#   joint (acceleration/action) regularization, foot slip penalty.
#
# Terms disabled here (feet_air_time, dof_pos_limits, joint_deviation_*) are exactly the terms
# the *stock* config already carries. The blog post's payoff is showing what breaks without them
# (arm flailing, finger jitter, no swing-phase shaping) and then pointing at the stock config as
# "here's what upstream already added, and why."

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

# PLACEHOLDER — do not trust this number. Measure your installed G1 asset's resting torso height
# with a zero-action rollout (log `robot.data.root_pos_w[:, 2]` for a few steps after reset) and
# replace this before training. Asset revisions change standing height.
TARGET_BASE_HEIGHT_M = 0.74


@configclass
class G1FlatMinimalEnvCfg(G1FlatEnvCfg):
    """Six-term minimal reward variant of the stock G1 flat velocity task."""

    def __post_init__(self):
        super().__post_init__()

        # --- add: explicit base-height term (stock G1 has none; it relies on orientation +
        # termination instead) ---
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-1.0,
            params={
                "target_height": TARGET_BASE_HEIGHT_M,
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            },
        )

        # --- remove: swing-timing shaping (introduce this later, not in the minimal pass) ---
        self.rewards.feet_air_time = None

        # --- remove: joint-limit and default-pose regularization for non-essential joints
        # (arms/fingers/torso/hip-yaw-roll). Expect visibly worse upper-body stillness without
        # these; that's the intended demonstration, not a bug. ---
        self.rewards.dof_pos_limits = None
        self.rewards.joint_deviation_hip = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_torso = None


@configclass
class G1FlatMinimalEnvCfg_PLAY(G1FlatMinimalEnvCfg):
    """Smaller-scene, no-randomization eval variant, matching Isaac Lab's `_PLAY` convention."""

    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
