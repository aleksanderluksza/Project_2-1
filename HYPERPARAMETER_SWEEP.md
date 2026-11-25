# Hyperparameter Sweep Guide (Team 12)

Extra data collection for RQ1. Only the members assigned to it need to do this
(see "Who runs it" below). Everyone else keeps running the normal batches from
`DATA_COLLECTION.md`.

## Why we do this

RQ1 predicts training time and peak RAM from the game, the hyperparameters and the
hardware. In our data every game always uses its own default hyperparameters, so the
model cannot separate the effect of the game from the effect of its settings.
This sweep changes one setting at a time within each game, so the model can learn what
batch size, network width (`hidden_units`) and network depth (`num_layers`) do to time
and RAM.

- Used by: RQ1 only.
- RQ2 and RQ3 must NOT use these rows. Changing hyperparameters changes how well the
  agent learns. Every sweep row has `_sweep_` in its `run_id`, so filter them out with
  `run_id` not containing `_sweep_`.

## Who runs it

Two members, on different machines (write the names here once agreed):

- Member 1: ______
- Member 2: ______

## What each run is

Every sweep run uses 100,000 steps and 2 repeats. We change ONE setting per run and keep
the others at the game's default. The default-settings runs already exist from your
normal batch (the 100k rows), so do not repeat them.

| Game | Default (batch / hidden / layers) | batch_size | hidden_units | num_layers |
|---|---|---|---|---|
| 3DBall | 64 / 128 / 2 | 32, 128 | 64, 256 | 1, 3 |
| Pyramids | 128 / 512 / 2 | 64, 256 | 256, 1024 | 1, 3 |
| Sorter | 512 / 128 / 2 | 256, 1024 | 64, 256 | 1, 3 |
| FoodCollector | 1024 / 256 / 1 | 512, 2048 | 128, 512 | 2, 3 |

That is 6 changed settings x 2 repeats = 12 new runs per game, 48 in total.

Rough time per game, estimated from one laptop (your machine may be faster or slower):
3DBall ~15 min, Pyramids ~25 min, Sorter ~45 min, FoodCollector ~1 h.
Run the heavy games last, or overnight.

## Step 1: add the `--set` option to `batch_runs.py` (once)

Open your `batch_runs.py` and make these four edits. Nothing else changes, and the CSV
columns stay exactly the same.

1. Replace the whole `make_temp_config` function with:

```python
HP_KEYS = {"batch_size", "buffer_size", "learning_rate"}   # under hyperparameters
NET_KEYS = {"hidden_units", "num_layers"}                  # under network_settings

def make_temp_config(base_path, max_steps, out_path, overrides=None):
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    name = list(cfg["behaviors"].keys())[0]
    b = cfg["behaviors"][name]
    b["max_steps"] = int(max_steps)
    for key, val in (overrides or {}).items():
        if key in HP_KEYS:
            b["hyperparameters"][key] = val
        elif key in NET_KEYS:
            b["network_settings"][key] = val
        else:
            raise SystemExit(f"Unknown --set key: {key}")
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path
```

2. In `main()`, directly under the `--id-prefix` argument, add:

```python
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="Override settings, e.g. --set batch_size=128 hidden_units=256")
```

3. Directly under `args = ap.parse_args()`, add:

```python
    overrides = {}
    for item in args.set:
        k, v = item.split("=")
        try:
            overrides[k] = int(v)
        except ValueError:
            overrides[k] = float(v)
```

4. Inside the loop, change the call to:

```python
        make_temp_config(args.base_config, steps, tmp_config, overrides)
```

Do not change `run_and_log.py`. Use the same one you use for your normal runs, so the
sweep is recorded under the same conditions as your baseline.

## Step 2: test with one run

Open PowerShell in your `ml-agents` folder with the venv active (the PowerShell
commands below do not work in cmd):

```powershell
python batch_runs.py --base-config config/ppo/3DBall.yaml --env builds/3DBall/UnityEnvironment.exe --game 3DBall --steps 100000 --repeats 1 --id-prefix 3DBall_test --set batch_size=128
```

Check two things:
- The training log shows `batch_size: 128` under the hyperparameters.
- The last row of `training_data.csv` has 128 in the `batch_size` column and
  `return_code` 0.

Then delete the `3DBall_test` row from the CSV.

## Step 3: run the sweep

Paste this helper once per PowerShell window:

```powershell
function Sweep($game, $cfg, $envpath, $key, $values) {
  foreach ($v in $values) {
    python batch_runs.py --base-config $cfg --env $envpath --game $game --steps 100000 --repeats 2 --id-prefix "${game}_sweep_${key}${v}" --set "${key}=${v}"
  }
}
```

Then run the games one at a time:

```powershell
# 3DBall
$g="3DBall"; $c="config/ppo/3DBall.yaml"; $e="builds/3DBall/UnityEnvironment.exe"
Sweep $g $c $e "batch_size"   @(32,128)
Sweep $g $c $e "hidden_units" @(64,256)
Sweep $g $c $e "num_layers"   @(1,3)
```

```powershell
# Pyramids
$g="Pyramids"; $c="config/ppo/Pyramids.yaml"; $e="builds/Pyramids/UnityEnvironment.exe"
Sweep $g $c $e "batch_size"   @(64,256)
Sweep $g $c $e "hidden_units" @(256,1024)
Sweep $g $c $e "num_layers"   @(1,3)
```

```powershell
# Sorter
$g="Sorter"; $c="config/ppo/Sorter_curriculum.yaml"; $e="builds/Sorter/UnityEnvironment.exe"
Sweep $g $c $e "batch_size"   @(256,1024)
Sweep $g $c $e "hidden_units" @(64,256)
Sweep $g $c $e "num_layers"   @(1,3)
```

```powershell
# FoodCollector
$g="FoodCollector"; $c="config/ppo/FoodCollector.yaml"; $e="builds/FoodCollector/UnityEnvironment.exe"
Sweep $g $c $e "batch_size"   @(512,2048)
Sweep $g $c $e "hidden_units" @(128,512)
Sweep $g $c $e "num_layers"   @(2,3)
```

Run IDs look like `Pyramids_sweep_batch_size256_001`.

## Step 3b: the same on macOS (zsh)

The four edits in Step 1 are identical on macOS. Only the commands differ: the game
path is your `.app` build, the helper is a zsh function, and `caffeinate` keeps the Mac
awake. Do not paste comment lines into zsh.

Test run (use your own `.app` path; if you use a modified copy of `batch_runs.py`,
use its file name):

```zsh
python batch_runs.py --base-config config/ppo/3DBall.yaml --env <path to 3DBall.app> --game 3DBall --steps 100000 --repeats 1 --id-prefix 3DBall_test --set batch_size=128
```

Helper (paste once per terminal window):

```zsh
sweep() {
  game=$1; cfg=$2; envpath=$3; key=$4; shift 4
  for v in "$@"; do
    caffeinate -dimsu python batch_runs.py --base-config "$cfg" --env "$envpath" --game "$game" --steps 100000 --repeats 2 --id-prefix "${game}_sweep_${key}${v}" --set "${key}=${v}"
  done
}
```

Usage, one line per setting, with the values from the table above:

```zsh
sweep 3DBall config/ppo/3DBall.yaml <path to 3DBall.app> batch_size 32 128
sweep 3DBall config/ppo/3DBall.yaml <path to 3DBall.app> hidden_units 64 256
sweep 3DBall config/ppo/3DBall.yaml <path to 3DBall.app> num_layers 1 3
```

Repeat for Pyramids, Sorter (`config/ppo/Sorter_curriculum.yaml`) and FoodCollector with
their own values.

Check the results (12 rows per game):

```zsh
cut -d, -f1,7,9,10,17,21 training_data.csv | tail -12
```

If a run fails within seconds with "address already in use", run
`pkill -f UnityEnvironment` and rerun that setting.

## Rules while it runs

- Sleep off, laptop plugged in, no heavy work on the machine (same as normal runs).
- Do not run two sweeps at the same time; they share one port.
- Do not close the Unity window that opens; the script manages it.
- Keep `batch_size` at or below `buffer_size`. The values in the table already do.

## Check the results

After each game:

```powershell
Import-Csv training_data.csv | Select-Object -Last 12 run_id,game,batch_size,hidden_units,num_layers,wall_time_seconds,return_code | Format-Table
```

You should see 12 rows, each with the swept value in its column and `return_code` 0.
If a row has `return_code` 1, rerun just that setting.

## Troubleshooting

- **`Unknown --set key`**: a typo in the setting name. Valid names: `batch_size`,
  `buffer_size`, `learning_rate`, `hidden_units`, `num_layers`.
- **A run fails within a few seconds / "address already in use"**: a leftover Unity
  process is holding the port. Run `taskkill /F /IM UnityEnvironment.exe`, then rerun
  that setting.
- **PowerShell blocks the venv activation**: run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.

## Send your data back

Share your `training_data.csv` as before, renamed with your name (for example
`training_data_vito.csv`). The sweep rows are in the same file, marked by `_sweep_` in
the `run_id`. Do NOT overwrite someone else's file.

## Checklist

- [ ] `--set` edits made in `batch_runs.py`
- [ ] Test run worked (batch size in log and CSV), test row deleted
- [ ] Sweep run for all four games (12 new rows each)
- [ ] All new rows have `return_code` 0
- [ ] CSV renamed with your name and shared
