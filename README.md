# rl-wbc-g1-amp

Project 2 in the `rl-wbc-g1-*` series (see [`rl-wbc-g1-baseline`](../rl-wbc-g1-baseline) for
Project 1). Adds Adversarial Motion Priors (AMP) on top of the PPO baseline so the G1 learns a
walking *style* from reference motion instead of hand-tuned regularization terms.

**Status: scaffolding only, not yet trained or verified.** Local-only until GPU access comes
through — do not push to GitHub yet.

## Open architecture question

Isaac Lab's own native AMP support is through **skrl**, not **rsl_rl** — the library the rest of
this series is built on. Two ways to reconcile that, not yet decided:

1. Use `skrl` for this project only (Isaac Lab's AMP support works out of the box via
   `--algorithm AMP`), breaking the "one training library across the series" thread.
2. Port the approach from
   [`escontra/AMP_for_hardware`](https://github.com/escontra/AMP_for_hardware) — the reference
   implementation behind "Adversarial Motion Priors Make Good Substitutes for Complex Reward
   Functions," which extends **rsl_rl itself** with an AMP discriminator running alongside PPO —
   from Isaac Gym onto Isaac Lab. More work, keeps the series on one library, consistent with how
   SAC was handled in Project 1 (`rsl_rl_sac`, same "extend rsl_rl" pattern).

## Reference motion data

[`lvhaidong/LAFAN1_Retargeting_Dataset`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
on Hugging Face — Ubisoft's LAFAN1 mocap, retargeted to Unitree G1 (and H1/H1_2), CSV per-frame
joint configs.

**License caveat, read before redistributing anything:** the underlying LAFAN1 mocap data is
`CC BY-NC-ND 4.0` (non-commercial, no-derivatives) — only the retargeting *code* is MIT. This
needs to be stated plainly in the blog post, not glossed over.

AMASS is the other major reference-motion source in this space (larger, SMPL-based, aggregates
many mocap datasets) — worth covering in the post as context even though LAFAN1's G1-retargeted
CSVs are the practical starting point here.
