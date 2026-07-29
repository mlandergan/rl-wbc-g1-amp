# rl-wbc-g1-amp

Project 2 in the `rl-wbc-g1-*` series (see [`rl-wbc-g1-baseline`](../rl-wbc-g1-baseline) for
Project 1). Adds Adversarial Motion Priors (AMP) on top of the PPO baseline so the G1 learns a
walking *style* from reference motion instead of hand-tuned regularization terms.

**Status: in progress, not yet trained or verified.**

`docker/`, `scripts/`, and `isaaclab_project/g1_baseline/` are copied from
[`rl-wbc-g1-baseline`](https://github.com/mlandergan/rl-wbc-g1-baseline) as the starting point —
same GCP VM lifecycle, same Docker base, same Post 1 task reward — to be adapted with an AMP
style reward on top.

## Training library: skrl

Isaac Lab's native AMP support is through **skrl**, not **rsl_rl** (the library Project 1 uses) —
decided in favor of `skrl` since Isaac Lab's AMP agent works out of the box through it, versus
porting [`escontra/AMP_for_hardware`](https://github.com/escontra/AMP_for_hardware)'s
rsl_rl-based AMP implementation from Isaac Gym onto Isaac Lab from scratch. This breaks the
"one training library across the series" thread from Project 1, in exchange for building on a
supported path instead of porting research code across sim frameworks.

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
