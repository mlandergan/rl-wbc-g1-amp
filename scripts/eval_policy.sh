#!/usr/bin/env bash
# Roll out a trained checkpoint with a small, non-randomized env (the `_PLAY` config variant).
# CHECKPOINT is a path *inside the container*, i.e. under /workspace/isaaclab_root/logs/...
# -- run `ls logs/rsl_rl/<experiment_name>/` on the host first to find one (it's the same path,
# just relative to wherever the ./logs volume is mounted from).
set -euo pipefail

TASK="${TASK:-Isaac-Velocity-Flat-G1-Baseline-Minimal-Play-v0}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a checkpoint path, e.g. logs/rsl_rl/g1_flat_baseline_minimal/<run>/model_1500.pt}"
NUM_ENVS="${NUM_ENVS:-32}"

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y \
  -e TASK="${TASK}" -e CHECKPOINT="${CHECKPOINT}" -e NUM_ENVS="${NUM_ENVS}" \
  -v "$(pwd)/logs:/workspace/isaaclab_root/logs" \
  rl-wbc-g1-amp \
  bash -lc '
    set -euo pipefail
    cd /workspace/isaaclab_root
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
      --task "${TASK}" --num_envs "${NUM_ENVS}" --checkpoint "${CHECKPOINT}" --headless
  '
