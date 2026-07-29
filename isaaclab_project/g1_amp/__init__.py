"""Project 2: G1 AMP locomotion — velocity-tracking task reward + AMP style reward
against a Mixamo "Strut Walking" clip retargeted to G1 via GMR (see ../../data/mixamo/
and ../../third_party/gmr/).

Registers `Isaac-G1-AMP-StrutWalk-Direct-v0` (and its `-Play` variant), a direct-workflow
env adapted from linden713/humanoid_amp's pure-imitation G1AmpEnv (BSD-3-Clause).
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-G1-AMP-StrutWalk-Direct-v0",
    entry_point=f"{__name__}.g1_amp_env:G1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpStrutWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_g1_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_g1_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-G1-AMP-StrutWalk-Direct-Play-v0",
    entry_point=f"{__name__}.g1_amp_env:G1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpStrutWalkEnvCfg_PLAY",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_g1_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_g1_amp_cfg.yaml",
    },
)
