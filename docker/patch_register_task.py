#!/usr/bin/env python3
"""Hook `isaaclab_project.g1_baseline`'s gym registration into Isaac Lab's rsl_rl
train.py / play.py, at the exact point Isaac Lab registers its own built-in tasks.

Verified against Isaac Lab's actual scripts/reinforcement_learning/rsl_rl/train.py, which
imports `isaaclab_tasks` (side-effect only, for gym.register calls) right after AppLauncher
starts the simulation app and before the @hydra_task_config main(). This is the same insertion
point Isaac Lab's own external-extension template uses for third-party tasks. Idempotent (safe
to run more than once) and does not assume a specific install path -- it searches the image's
filesystem for the two target files instead of hardcoding one.
"""

import pathlib
import subprocess
import sys

HOOK_LINE = "import isaaclab_project.g1_baseline  # noqa: F401 (Project 1 baseline task registration)"
ANCHOR = "import isaaclab_tasks  # noqa: F401"


def find_targets() -> list[pathlib.Path]:
    # -xdev keeps this from wandering into /proc, /sys, or other mounted filesystems.
    out = subprocess.run(
        ["find", "/", "-xdev", "-path", "*/reinforcement_learning/rsl_rl/train.py",
         "-o", "-path", "*/reinforcement_learning/rsl_rl/play.py"],
        capture_output=True, text=True,
    )
    return [pathlib.Path(p) for p in out.stdout.splitlines() if p]


def main() -> None:
    targets = find_targets()
    if not targets:
        sys.exit(
            "Could not find Isaac Lab's rsl_rl train.py/play.py anywhere on this filesystem. "
            "Is Isaac Lab actually installed in this image/container? If Isaac Lab's directory "
            "layout has changed since this was written, patch train.py/play.py by hand: add\n"
            f"    {HOOK_LINE}\n"
            f"immediately after the line `{ANCHOR}`."
        )

    for path in targets:
        text = path.read_text()
        if HOOK_LINE in text:
            print(f"already patched: {path}")
            continue
        if ANCHOR not in text:
            print(f"WARNING: anchor line not found in {path} -- skipping (Isaac Lab version mismatch?)")
            continue
        path.write_text(text.replace(ANCHOR, f"{ANCHOR}\n{HOOK_LINE}", 1))
        print(f"patched: {path}")


if __name__ == "__main__":
    main()
