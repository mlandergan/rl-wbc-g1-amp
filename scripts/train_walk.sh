#!/usr/bin/env bash
# Launch PPO training inside the Project 1 Docker image. Run on the GCP VM after
# scripts/remote_setup.sh has built `rl-wbc-g1-baseline`. Mounts ./logs from the current
# directory so checkpoints/TensorBoard logs survive container exit -- that's what
# scripts/sync_results.sh pulls back to your local machine.
#
# TASK options:
#   Isaac-Velocity-Flat-G1-Baseline-Minimal-v0   (this project's stripped-down reward, default)
#   Isaac-Velocity-Flat-G1-v0                    (stock Isaac Lab reward, for comparison)
set -euo pipefail

TASK="${TASK:-Isaac-Velocity-Flat-G1-Baseline-Minimal-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"  # empty = task default; see agents/rsl_rl_ppo_cfg.py

mkdir -p "$(pwd)/logs"

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y \
  -e TASK="${TASK}" -e NUM_ENVS="${NUM_ENVS}" -e MAX_ITERATIONS="${MAX_ITERATIONS}" \
  -v "$(pwd)/logs:/workspace/isaaclab_root/logs" \
  rl-wbc-g1-baseline \
  bash -lc '
    set -euo pipefail
    cd /workspace/isaaclab_root
    extra_args=()
    [[ -n "${MAX_ITERATIONS:-}" ]] && extra_args+=(--max_iterations "${MAX_ITERATIONS}")
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task "${TASK}" --headless --num_envs "${NUM_ENVS}" "${extra_args[@]}"
  '
