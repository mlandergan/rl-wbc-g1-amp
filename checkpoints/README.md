# Checkpoints

`g1_amp_strut_walk_best.pt` is the checkpoint behind the walking video in the blog post's
Results section: task/style reward at 0.5/0.5, warm-started from a pure-imitation checkpoint,
24,000 steps. See `docs/HANDOFF_AMP_DEBUGGING.md` and the blog post for how it got there.

To watch it walk (inside the Docker image, from the repo root, with a display or
`--enable_cameras --video` for headless rendering):

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task Isaac-G1-AMP-LindenWalk-Direct-Play-v0 --num_envs 1 \
  --checkpoint checkpoints/g1_amp_strut_walk_best.pt \
  env.motion_file=/workspace/rl_wbc_g1/isaaclab_project/g1_amp/motions/G1_strut_walk.npz
```

Trained with `isaaclab_project/g1_amp/agents/skrl_g1_amp_linden_cfg.yaml`'s network shapes
([1024, 512] policy/value/discriminator) — a checkpoint from `skrl_g1_amp_cfg.yaml`'s
[256, 128, 128] networks won't load against this one, and vice versa.
