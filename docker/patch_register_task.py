#!/usr/bin/env python3
"""Hook our tasks' gym registration into Isaac Lab's train.py / play.py entrypoints, at
the exact point Isaac Lab registers its own built-in tasks: `isaaclab_project.g1_baseline`
(Project 1, PPO via rsl_rl) into the rsl_rl entrypoints, and `isaaclab_project.g1_amp`
(Project 2, AMP via skrl) into the skrl entrypoints.

Verified against Isaac Lab's actual scripts/reinforcement_learning/{rsl_rl,skrl}/train.py,
both of which import `isaaclab_tasks` (side-effect only, for gym.register calls) right after
AppLauncher starts the simulation app and before the @hydra_task_config main(). This is the
same insertion point Isaac Lab's own external-extension template uses for third-party tasks.
Idempotent (safe to run more than once) and does not assume a specific install path -- it
searches the image's filesystem for the target files instead of hardcoding one.
"""

import pathlib
import subprocess
import sys

ANCHOR = "import isaaclab_tasks  # noqa: F401"

HOOKS = {
    "rsl_rl": "import isaaclab_project.g1_baseline  # noqa: F401 (Project 1 baseline task registration)",
    "skrl": "import isaaclab_project.g1_amp  # noqa: F401 (Project 2 AMP task registration)",
}


def find_targets(library: str) -> list[pathlib.Path]:
    # -xdev keeps this from wandering into /proc, /sys, or other mounted filesystems.
    out = subprocess.run(
        ["find", "/", "-xdev", "-path", f"*/reinforcement_learning/{library}/train.py",
         "-o", "-path", f"*/reinforcement_learning/{library}/play.py"],
        capture_output=True, text=True,
    )
    return [pathlib.Path(p) for p in out.stdout.splitlines() if p]


def main() -> None:
    any_found = False
    for library, hook_line in HOOKS.items():
        targets = find_targets(library)
        if not targets:
            print(
                f"WARNING: could not find Isaac Lab's {library} train.py/play.py anywhere on "
                f"this filesystem -- skipping. If that library isn't installed in this image, "
                f"this is expected; otherwise patch by hand: add\n"
                f"    {hook_line}\n"
                f"immediately after the line `{ANCHOR}` in {library}/train.py and play.py."
            )
            continue
        any_found = True
        for path in targets:
            text = path.read_text()
            if hook_line in text:
                print(f"already patched: {path}")
                continue
            if ANCHOR not in text:
                print(f"WARNING: anchor line not found in {path} -- skipping (Isaac Lab version mismatch?)")
                continue
            path.write_text(text.replace(ANCHOR, f"{ANCHOR}\n{hook_line}", 1))
            print(f"patched: {path}")

    if not any_found:
        sys.exit("Neither rsl_rl nor skrl train.py/play.py were found. Is Isaac Lab actually installed?")


if __name__ == "__main__":
    main()
