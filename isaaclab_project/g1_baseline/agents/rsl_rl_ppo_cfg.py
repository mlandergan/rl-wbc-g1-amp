# Deliberately identical hyperparameters to Isaac Lab's stock `G1FlatPPORunnerCfg`.
#
# Project 1 compares reward design (minimal vs. stock), not algorithm tuning — so both runs must
# share the exact same PPO runner config. Only `experiment_name` differs, so TensorBoard logs
# land in separate run directories (see reward_debug.py).

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatPPORunnerCfg,
)


@configclass
class G1FlatMinimalPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_flat_baseline_minimal"
