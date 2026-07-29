#!/usr/bin/env bash
# Launch AMP training (skrl) inside the Project 2 Docker image. Run on the GCP VM after
# scripts/remote_setup.sh has built `rl-wbc-g1-amp`. Mounts ./logs from the current
# directory so checkpoints/TensorBoard logs survive container exit -- that's what
# scripts/sync_results.sh pulls back to your local machine.
#
# TASK options:
#   Isaac-G1-AMP-StrutWalk-Direct-v0   (this project's task: velocity tracking + AMP style
#                                       reward against the Mixamo Strut Walking clip)
set -euo pipefail

TASK="${TASK:-Isaac-G1-AMP-StrutWalk-Direct-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"

mkdir -p "$(pwd)/logs"

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y \
  -e TASK="${TASK}" -e NUM_ENVS="${NUM_ENVS}" \
  -v "$(pwd)/logs:/workspace/isaaclab_root/logs" \
  rl-wbc-g1-amp \
  bash -lc '
    set -euo pipefail
    cd /workspace/isaaclab_root
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
      --task "${TASK}" --headless --num_envs "${NUM_ENVS}"
  '
