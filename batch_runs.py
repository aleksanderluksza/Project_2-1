#!/usr/bin/env python3
"""
batch_runs.py  --  run several ML-Agents trainings back-to-back and log each one.

It sweeps a list of max_steps values (and optionally repeats each a few times),
writing a temporary config per run so your original 3DBall.yaml is never changed.
Every run appends one row to training_data.csv via run_and_log.py.

Usage (venv active, inside the ml-agents folder):
  python batch_runs.py --base-config config/ppo/3DBall.yaml ^
      --env builds/3DBall/UnityEnvironment.exe --game 3DBall ^
      --steps 100000 250000 500000 --repeats 2

That runs 3 step-values x 2 repeats = 6 trainings, ids 3DBall_batch_001 ... _006.
Repeats matter: the same settings won't give the exact same time twice (OS noise),
so 2-3 repeats per setting gives your model a truer picture.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime

import yaml


def make_temp_config(base_path, max_steps, out_path):
    """Copy the base config but override max_steps for the first behavior."""
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    name = list(cfg["behaviors"].keys())[0]
    cfg["behaviors"][name]["max_steps"] = int(max_steps)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Batch-run ML-Agents trainings and log each.")
    ap.add_argument("--base-config", required=True, help="Config to start from, e.g. config/ppo/3DBall.yaml")
    ap.add_argument("--env", required=True, help="Built Unity environment executable.")
    ap.add_argument("--game", required=True, help="Game/task name for the CSV.")
    ap.add_argument("--steps", nargs="+", type=int, required=True,
                    help="List of max_steps values to sweep, e.g. 100000 250000 500000")
    ap.add_argument("--repeats", type=int, default=1, help="How many times to repeat each steps value.")
    ap.add_argument("--csv", default="training_data.csv", help="CSV file to append to.")
    ap.add_argument("--id-prefix", default=None, help="Run-id prefix (default: <game>_batch).")
    args = ap.parse_args()

    prefix = args.id_prefix or f"{args.game}_batch"
    tmp_config = "config_batch_tmp.yaml"

    # Build the full task list up front so we can show progress.
    tasks = []
    for s in args.steps:
        for _ in range(args.repeats):
            tasks.append(s)

    print(f"=== batch_runs: {len(tasks)} trainings queued "
          f"({len(args.steps)} step-values x {args.repeats} repeats) ===")
    print(f"    steps: {args.steps}")
    print(f"    started: {datetime.now().isoformat(timespec='seconds')}\n")

    n = 0
    ok = 0
    for steps in tasks:
        n += 1
        run_id = f"{prefix}_{n:03d}"
        make_temp_config(args.base_config, steps, tmp_config)
        print(f"\n########## [{n}/{len(tasks)}] run_id={run_id}  max_steps={steps} ##########")

        cmd = [sys.executable, "run_and_log.py",
               "--config", tmp_config,
               "--run-id", run_id,
               "--env", args.env,
               "--game", args.game,
               "--csv", args.csv]
        result = subprocess.run(cmd)
        if result.returncode == 0:
            ok += 1
        else:
            print(f"(run {run_id} returned code {result.returncode} - see output above; continuing)")

        time.sleep(2)  # small breather so the Unity process fully releases

    print(f"\n=== batch done: {ok}/{len(tasks)} runs completed. "
          f"Data in {args.csv}. finished: {datetime.now().isoformat(timespec='seconds')} ===")


if __name__ == "__main__":
    main()
