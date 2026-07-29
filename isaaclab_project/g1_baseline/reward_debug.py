#!/usr/bin/env python3
"""Per-reward-term diagnostics for an rsl_rl training run.

Isaac Lab's rsl_rl wrapper writes every reward-manager term to TensorBoard, but ships no tool to
inspect them component-wise. This script reads the event file(s) under an experiment's log
directory and either lists the available scalar tags or plots/exports a chosen subset.

Does not require Isaac Sim / Isaac Lab to be importable — only tensorboard + matplotlib. Run it
locally against a synced-down log directory (see ../../scripts/sync_results.sh), or on the VM.

Usage:
    # first, see what's actually logged for your installed Isaac Lab version -- tag names have
    # changed across releases, so don't assume the --tag-regex default below is correct.
    python reward_debug.py --logdir logs/rsl_rl/g1_flat_baseline_minimal --list-tags

    # then plot the reward-term family you found:
    python reward_debug.py --logdir logs/rsl_rl/g1_flat_baseline_minimal \
        --tag-regex "Episode_Reward/.*" --out reward_terms.png --csv reward_terms.csv
"""

import argparse
import csv
import glob
import os
import re
import sys


def find_event_files(logdir: str) -> list[str]:
    pattern = os.path.join(logdir, "**", "events.out.tfevents.*")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event files found under {logdir!r}")
    return files


def load_scalars(event_files: list[str]) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    series: dict[str, list[tuple[int, float]]] = {}
    for path in event_files:
        run_dir = os.path.dirname(path)
        acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            series.setdefault(tag, [])
            for event in acc.Scalars(tag):
                series[tag].append((event.step, event.value))
    for tag in series:
        series[tag].sort(key=lambda p: p[0])
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logdir", required=True, help="Path to a single run's log directory")
    parser.add_argument("--list-tags", action="store_true", help="Print all available scalar tags and exit")
    parser.add_argument(
        "--tag-regex",
        default="Episode_Reward/.*",
        help="Regex filtering which scalar tags to plot/export (check --list-tags first)",
    )
    parser.add_argument("--out", default=None, help="Path to save a PNG plot (requires matplotlib)")
    parser.add_argument("--csv", default=None, help="Path to save raw (tag, step, value) rows as CSV")
    args = parser.parse_args()

    event_files = find_event_files(args.logdir)
    series = load_scalars(event_files)

    if args.list_tags:
        for tag in sorted(series):
            print(tag)
        return

    pattern = re.compile(args.tag_regex)
    matched = {tag: points for tag, points in series.items() if pattern.match(tag)}
    if not matched:
        print(f"No tags matched {args.tag_regex!r}. Run with --list-tags to see what's available.", file=sys.stderr)
        sys.exit(1)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tag", "step", "value"])
            for tag, points in matched.items():
                for step, value in points:
                    writer.writerow([tag, step, value])
        print(f"Wrote {args.csv}")

    if args.out:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        for tag, points in sorted(matched.items()):
            steps = [p[0] for p in points]
            values = [p[1] for p in points]
            ax.plot(steps, values, label=tag.split("/")[-1])
        ax.set_xlabel("iteration")
        ax.set_ylabel("mean episodic reward component")
        ax.legend(fontsize="small")
        ax.set_title(os.path.basename(os.path.normpath(args.logdir)))
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"Wrote {args.out}")

    if not args.csv and not args.out:
        for tag, points in sorted(matched.items()):
            last_step, last_value = points[-1]
            print(f"{tag:45s} last={last_value:+.4f} (step {last_step}, {len(points)} points)")


if __name__ == "__main__":
    main()
