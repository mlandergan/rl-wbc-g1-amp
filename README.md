# rl-wbc-g1-amp

Part of the `rl-wbc-g1-*` series (see [`rl-wbc-g1-baseline`](https://github.com/mlandergan/rl-wbc-g1-baseline)
for the PPO velocity-tracking baseline this builds on). Adds Adversarial Motion Priors (AMP) on
top of that baseline so the G1 learns a walking *style* from a reference motion clip instead of
hand-tuned regularization terms.

**Status: trained and working.** See [`checkpoints/README.md`](checkpoints/README.md) to run the
trained policy yourself. Write-up: [Stylized Walking for the Unitree G1 with Adversarial Motion
Priors](https://mlandergan.github.io/blog/g1-amp-stylized-walking/).

`docker/`, `scripts/`, and `isaaclab_project/g1_baseline/` are copied from
[`rl-wbc-g1-baseline`](https://github.com/mlandergan/rl-wbc-g1-baseline) as the starting point —
same GCP VM lifecycle, same Docker base, same task reward — adapted with an AMP style reward on
top.

## Training library: skrl

Isaac Lab's native AMP support is through **skrl**, not **rsl_rl** (the library the baseline
project uses) — decided in favor of `skrl` since Isaac Lab's AMP agent works out of the box
through it, versus porting [`escontra/AMP_for_hardware`](https://github.com/escontra/AMP_for_hardware)'s
rsl_rl-based AMP implementation from Isaac Gym onto Isaac Lab from scratch. This breaks the
"one training library across the series" thread from the baseline project, in exchange for
building on a supported path instead of porting research code across sim frameworks.

## Reference motion data

The reference clip is Mixamo's "Strut Walking" animation, retargeted onto the G1's 29-DOF joint
convention with [GMR](https://github.com/YanjieZe/GMR) (General Motion Retargeting). Retargeting
happens once, offline (`isaaclab_project/g1_amp/motions/convert_gmr_to_npz.py`); the retargeted
result is checked into `isaaclab_project/g1_amp/motions/G1_strut_walk.npz`.

**License caveat, read before redistributing anything:** Mixamo restricts redistributing its raw
motion assets, so the source `.bvh`/`.fbx` files aren't in this repo (gitignored) — only the
retargeted derivative and this project's own rendered visualizations are. Treat the reference
clip as personal use and check Mixamo's current terms before reusing it yourself.

[`lvhaidong/LAFAN1_Retargeting_Dataset`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
(Ubisoft's LAFAN1 mocap, pre-retargeted to the G1) was investigated as an alternative, and used
as a control condition during debugging, but was never this project's actual training data — its
`CC BY-NC-ND 4.0` license is stricter than Mixamo's (no derivatives at all), so it's excluded
from this repo entirely rather than relied on under a personal-use framing.
