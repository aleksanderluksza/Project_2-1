" mac version "
"""
batch_runs.py  --  run several ML-Agents trainings back-to-back and log each one.

Sweeps a list of max_steps values (optionally repeated), writing a temporary
config per run so the original yaml is never changed. Every run appends one row
to training_data.csv via run_and_log.py.
"""

import argparse
import socket
import subprocess
import sys
import time
from datetime import datetime

import yaml

PORT = 5005  # ML-Agents base port for built environments


def make_temp_config(base_path, max_steps, out_path):
    """Copy the base config but override max_steps for the first behavior."""
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    name = list(cfg["behaviors"].keys())[0]
    cfg["behaviors"][name]["max_steps"] = int(max_steps)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def free_port(port=PORT, timeout=40):
    """Kill leftover Unity/trainer processes and wait until the port can be bound."""
    subprocess.run(["pkill", "-f", "UnityEnvironment"])
    subprocess.run(["pkill", "-f", "mlagents-learn"])
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            try:
                s.bind(("localhost", port))
                return True
            except OSError:
                time.sleep(1)
    print(f"WARNING: port {port} still busy after {timeout}s")
    return False


def main():
    ap = argparse.ArgumentParser(description="Batch-run ML-Agents trainings and log each.")
    ap.add_argument("--base-config", required=True, help="Config to start from, e.g. config/ppo/3DBall.yaml")
    ap.add_argument("--env", required=True, help="Built Unity environment (.app on macOS).")
    ap.add_argument("--game", required=True, help="Game/task name for the CSV.")
    ap.add_argument("--steps", nargs="+", type=int, required=True,
                    help="List of max_steps values to sweep, e.g. 100000 250000 500000")
    ap.add_argument("--repeats", type=int, default=1, help="How many times to repeat each steps value.")
    ap.add_argument("--csv", default="training_data.csv", help="CSV file to append to.")
    ap.add_argument("--id-prefix", default=None, help="Run-id prefix (default: <game>_batch).")
    args = ap.parse_args()

    prefix = args.id_prefix or f"{args.game}_batch"
    tmp_config = "config_batch_tmp.yaml"

    tasks = [s for s in args.steps for _ in range(args.repeats)]

    print(f"=== batch_runs: {len(tasks)} trainings queued "
          f"({len(args.steps)} step-values x {args.repeats} repeats) ===")
    print(f"    steps: {args.steps}")
    print(f"    started: {datetime.now().isoformat(timespec='seconds')}\n")

    ok = 0
    for n, steps in enumerate(tasks, start=1):
        run_id = f"{prefix}_{n:03d}"
        make_temp_config(args.base_config, steps, tmp_config)
        print(f"\n########## [{n}/{len(tasks)}] run_id={run_id}  max_steps={steps} ##########")

        if not free_port():
            print(f"(skipping {run_id}: port {PORT} is still in use)")
            continue

        # run_and_log.py already adds --timeout-wait 120 and --no-graphics itself.
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

        free_port()  # clean up after the run so the next one starts clean

    print(f"\n=== batch done: {ok}/{len(tasks)} runs completed. "
          f"Data in {args.csv}. finished: {datetime.now().isoformat(timespec='seconds')} ===")


if __name__ == "__main__":
    main()