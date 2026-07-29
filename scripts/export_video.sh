#!/usr/bin/env bash
# Same as eval_policy.sh but with Isaac Lab's built-in video recording flags. Output mp4 lands
# under logs/rsl_rl/<experiment_name>/<run>/videos/play/ (relative to the mounted ./logs volume).
set -euo pipefail

TASK="${TASK:-Isaac-Velocity-Flat-G1-Baseline-Minimal-Play-v0}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT, e.g. logs/rsl_rl/g1_flat_baseline_minimal/<run>/model_1500.pt}"
NUM_ENVS="${NUM_ENVS:-32}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y \
  -e TASK="${TASK}" -e CHECKPOINT="${CHECKPOINT}" -e NUM_ENVS="${NUM_ENVS}" -e VIDEO_LENGTH="${VIDEO_LENGTH}" \
  -v "$(pwd)/logs:/workspace/isaaclab_root/logs" \
  rl-wbc-g1-amp \
  bash -lc '
    set -euo pipefail
    cd /workspace/isaaclab_root
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
      --task "${TASK}" --num_envs "${NUM_ENVS}" --checkpoint "${CHECKPOINT}" \
      --headless --video --video_length "${VIDEO_LENGTH}" --enable_cameras
  '
