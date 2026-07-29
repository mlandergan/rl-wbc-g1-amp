# Project 1 baseline image: Isaac Lab (official NGC build) + our thin task-registration
# extension. See project_01_baseline_locomotion.md section 6.3 for why we build on the prebuilt
# image instead of a native Isaac Sim install.
#
# Build (from repo root):
#   docker build -f docker/Dockerfile -t rl-wbc-g1-baseline .
# Requires `docker login nvcr.io` with an NGC account first.
FROM nvcr.io/nvidia/isaac-lab:2.3.2

WORKDIR /workspace/rl_wbc_g1

COPY isaaclab_project/ ./isaaclab_project/
COPY docker/patch_register_task.py ./docker/patch_register_task.py

# Real interpreter env var -- respected regardless of whether `python` in this image is a
# symlink, wrapper script, or shell alias, unlike PATH tricks.
ENV PYTHONPATH=/workspace/rl_wbc_g1:${PYTHONPATH}

# Hook our gym.register() calls into Isaac Lab's rsl_rl train.py/play.py. This image has no bare
# `python3` on PATH at all (confirmed against isaac-lab:2.3.2) -- the only interpreter is Isaac
# Sim's bundled one. patch_register_task.py only touches text files (stdlib only), so it's safe
# to run with that interpreter directly instead of Isaac Lab's `isaaclab.sh -p` wrapper.
RUN /isaac-sim/kit/python/bin/python3.11 docker/patch_register_task.py

# Isaac Lab's own repo root (containing isaaclab.sh) isn't at a documented fixed path in the
# prebuilt NGC image, and its rsl_rl scripts write logs/checkpoints/videos relative to that root.
# Resolve it once at build time and pin it behind a stable symlink so scripts/*.sh on the host
# can mount a volume there without knowing the underlying path.
RUN ISAACLAB_SH="$(find / -xdev -name isaaclab.sh 2>/dev/null | head -n1)" && \
    test -n "${ISAACLAB_SH}" || (echo "isaaclab.sh not found in this image -- base image layout changed?" >&2 && exit 1) && \
    ln -s "$(dirname "${ISAACLAB_SH}")" /workspace/isaaclab_root

# The base isaac-lab image sets ENTRYPOINT to /isaac-sim/runheadless.sh, which swallows any CMD
# or `docker run <image> <cmd>` args as launch parameters for Isaac Sim's own GUI/streaming app
# instead of executing them -- confirmed via `docker inspect` and NVIDIA's own docs, which
# document `--entrypoint bash` as the standard override. Baking the override into the image
# itself (rather than requiring every docker run call site to pass --entrypoint) means
# scripts/*.sh's plain `docker run rl-wbc-g1-baseline bash -lc '...'` calls actually run bash.
ENTRYPOINT []
CMD ["/bin/bash"]
