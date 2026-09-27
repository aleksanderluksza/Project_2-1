#!/usr/bin/env python3
"""
run_and_log.py  --  Team 12, Project 2.1 data-collection pipeline.

Runs one ML-Agents training run, measures how expensive it was, and appends a
single row to a CSV file. One row = one training run.

It records:
  INPUTS   : game/task, algorithm, key hyperparameters, number of steps, hardware
  OUTPUTS  : wall-clock time (seconds), peak RAM (MB), final mean reward

Example (Windows PowerShell, inside the ml-agents folder, venv active):
  python run_and_log.py --config config/ppo/3DBall.yaml --run-id 3dball_001 ^
      --env builds/3DBall/UnityEnvironment.exe --game 3DBall

  --env is a BUILT Unity environment executable. Build it once from the Unity
  editor (File > Build Settings). If you leave --env out, ML-Agents waits for you
  to press Play in the editor -- fine for a quick test, but for automated data
  collection use a build so no clicking is needed.

Requires: pip install pyyaml psutil   (mlagents is already in your venv)
"""

import argparse
import csv
import os
import platform
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

try:
    import yaml
except ImportError:
    sys.exit("Missing dependency. Run:  pip install pyyaml")
try:
    import psutil
except ImportError:
    sys.exit("Missing dependency. Run:  pip install psutil")


# ---------------------------------------------------------------- config parsing
def read_config_features(config_path):
    """Pull the algorithm + key hyperparameters out of the ML-Agents YAML config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    behaviors = cfg.get("behaviors", {})
    if not behaviors:
        return {"behavior_name": "", "trainer_type": "", "max_steps": "",
                "learning_rate": "", "batch_size": "", "buffer_size": "",
                "hidden_units": "", "num_layers": ""}
    # Use the first behavior in the file (these example configs have one).
    name = list(behaviors.keys())[0]
    b = behaviors[name] or {}
    hp = b.get("hyperparameters", {}) or {}
    net = b.get("network_settings", {}) or {}
    return {
        "behavior_name": name,
        "trainer_type": b.get("trainer_type", ""),
        "max_steps": b.get("max_steps", ""),
        "learning_rate": hp.get("learning_rate", ""),
        "batch_size": hp.get("batch_size", ""),
        "buffer_size": hp.get("buffer_size", ""),
        "hidden_units": net.get("hidden_units", ""),
        "num_layers": net.get("num_layers", ""),
    }


# ---------------------------------------------------------------- hardware info
def read_hardware():
    """Best-effort description of the machine (an input feature for the model)."""
    cpu_model = platform.processor()
    if not cpu_model and os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo"):
            if "model name" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    freq = psutil.cpu_freq()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu_model": cpu_model or "unknown",
        "cpu_cores_physical": psutil.cpu_count(logical=False) or "",
        "cpu_cores_logical": psutil.cpu_count(logical=True) or "",
        "cpu_freq_mhz": round(freq.max or freq.current, 0) if freq else "",
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
    }


# ---------------------------------------------------------------- memory sampling
class MemorySampler(threading.Thread):
    """Polls total RAM of the training process AND its children, keeps the peak."""
    def __init__(self, pid, interval=0.5):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.peak_mb = 0.0
        self._stop_event = threading.Event()

    def run(self):
        try:
            root = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return
        while not self._stop_event.is_set():
            try:
                procs = [root] + root.children(recursive=True)
                total = 0
                for p in procs:
                    try:
                        total += p.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.peak_mb = max(self.peak_mb, total / (1024 ** 2))
            except psutil.NoSuchProcess:
                break
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()


# ---------------------------------------------------------------- run training
REWARD_RE = re.compile(r"Mean Reward:\s*(-?\d+(?:\.\d+)?)")
STEP_RE = re.compile(r"Step:\s*(\d+)")


def run_training(cmd):
    """Launch mlagents-learn, stream its output, track peak RAM, last reward/step."""
    print(">>> Running:", " ".join(cmd), flush=True)
    start = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    sampler = MemorySampler(proc.pid)
    sampler.start()

    last_reward, last_step = "", ""
    for line in proc.stdout:            # stream and parse as it trains
        print(line, end="")
        m = REWARD_RE.search(line)
        if m:
            last_reward = m.group(1)
        s = STEP_RE.search(line)
        if s:
            last_step = s.group(1)
    proc.wait()
    sampler.stop()
    try:
        sampler.join(timeout=2)
    except Exception as e:
        print(f"(note: memory sampler join issue, ignored: {e})")

    elapsed = time.time() - start
    return {
        "wall_time_seconds": round(elapsed, 1),
        "peak_ram_mb": round(sampler.peak_mb, 1),
        "final_mean_reward": last_reward,
        "final_step": last_step,
        "return_code": proc.returncode,
    }


# ---------------------------------------------------------------- csv
COLUMNS = [
    "run_id", "timestamp", "game",
    "trainer_type", "max_steps", "learning_rate", "batch_size", "buffer_size",
    "hidden_units", "num_layers",
    "os", "cpu_model", "cpu_cores_physical", "cpu_cores_logical", "cpu_freq_mhz",
    "ram_total_gb",
    "wall_time_seconds", "peak_ram_mb", "final_mean_reward", "final_step",
    "return_code",
]


def append_row(csv_path, row):
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


# ---------------------------------------------------------------- main
SCRIPT_VERSION = "v2 (fixed)"


def main():
    print(f"=== run_and_log.py {SCRIPT_VERSION} ===", flush=True)
    ap = argparse.ArgumentParser(description="Run an ML-Agents training and log its cost to CSV.")
    ap.add_argument("--config", required=True, help="Path to the ML-Agents YAML config.")
    ap.add_argument("--run-id", required=True, help="Unique id for this run.")
    ap.add_argument("--env", default=None, help="Path to a built Unity environment executable.")
    ap.add_argument("--game", default=None, help="Game/task name (defaults to the config's behavior name).")
    ap.add_argument("--csv", default="training_data.csv", help="CSV file to append to.")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Anything after --extra is passed straight to mlagents-learn.")
    args = ap.parse_args()

    features = read_config_features(args.config)
    hardware = read_hardware()
    game = args.game or features["behavior_name"]

    cmd = ["mlagents-learn", args.config, "--run-id", args.run_id, "--force"]
    if args.env:
        cmd += ["--env", args.env]
    if args.extra:
        cmd += args.extra

    result = run_training(cmd)

    row = {"run_id": args.run_id,
           "timestamp": datetime.now().isoformat(timespec="seconds"),
           "game": game}
    row.update(features)
    row.update(hardware)
    row.update(result)
    # behavior_name is captured as 'game'; drop the duplicate key
    row.pop("behavior_name", None)

    append_row(args.csv, row)

    print("\n" + "=" * 60)
    if result["return_code"] == 0:
        print(f"Run '{args.run_id}' logged to {args.csv}")
    else:
        print(f"WARNING: training exited with code {result['return_code']}. Row still logged; check it.")
    print(f"  time: {result['wall_time_seconds']} s | peak RAM: {result['peak_ram_mb']} MB "
          f"| final reward: {result['final_mean_reward']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
