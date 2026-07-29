"""Project 1 baseline: minimal-reward G1 flat-ground velocity task.

Registers `Isaac-Velocity-Flat-G1-Baseline-Minimal-v0` (and its `-Play` variant) as a thin
override of Isaac Lab's stock `Isaac-Velocity-Flat-G1-v0` task. Requires Isaac Lab to already be
installed/importable (see ../../docker/Dockerfile) — this package is an extension, not a fork.
"""

import gymnasium as gym

from . import flat_env_cfg
from .agents import rsl_rl_ppo_cfg

gym.register(
    id="Isaac-Velocity-Flat-G1-Baseline-Minimal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{flat_env_cfg.__name__}:G1FlatMinimalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:G1FlatMinimalPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-Baseline-Minimal-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{flat_env_cfg.__name__}:G1FlatMinimalEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:G1FlatMinimalPPORunnerCfg",
    },
)
