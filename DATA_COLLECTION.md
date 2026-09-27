# Data Collection Guide (Team 12)

How to run training runs on your own machine and add rows to our shared dataset.
Everyone uses the same pipeline; only the game changes. All rows go into one CSV so
Research Questions 1, 2 and 3 can all read from the same pool.

## What you need (one-time setup)

1. Clone the repo and set up the environment (see the main README): Python 3.10,
   a virtual environment, the recommended ml-agents fork/branch.
2. These package versions (important, newer ones break ml-agents):
   torch 2.2.2, numpy 1.23.5, onnx 1.12.0, protobuf 3.19.6.
   Check with:
   ```
   python -c "import torch, numpy, onnx; print(torch.__version__, numpy.__version__, onnx.__version__)"
   ```
   You should see: 2.2.2 ... 1.23.5 ... 1.12.0
3. Copy `run_and_log.py` and `batch_runs.py` into your ml-agents folder.
4. Install two helper packages: `pip install pyyaml psutil`
5. Turn OFF sleep on your machine (Settings > System > Power). If the PC sleeps
   mid-run, the recorded time is wrong and the row is useless.

## Build your game (one-time per game)

Each game needs a Unity build once:
1. Open the project in Unity 2022.3.4f1.
2. In the Project window open the game's scene, e.g.
   `Assets/ML-Agents/Examples/<Game>/Scenes/<Game>.unity`
3. File > Build Settings > Add Open Scenes. Untick any other scene so only this one is checked.
4. Build into `builds/<Game>/`. This creates `builds/<Game>/UnityEnvironment.exe`.

## Which games to run

Everyone runs ALL four games: 3DBall, FoodCollector, Pyramids, Sorter.

Running every game on every machine gives full coverage of game x hardware, which is
what makes the dataset strong: the model can tell apart the effect of the game from the
effect of the PC. Each question then reads its slice of the shared pool:

- RQ1 (cost, time + RAM): all rows, all games
- RQ2 (steps to beat baseline): Pyramids rows
- RQ3 (performance / reward): Sorter rows

All 6 members: Vito, Antonio, Aleksander, Eryk, Jaime, Dumitru.

If time is short, do the lighter games first (3DBall, then Pyramids/Sorter at 500k),
and leave FoodCollector (heavy, ~1-2h) for last or overnight.

Everyone running all four games on their own machine is what gives us the same games
across different hardware, which is exactly what the model needs.

## Collect data (the actual runs)

With the virtual environment active, in the ml-agents folder, run the batch script.
It runs several trainings back to back at different step counts and logs each one:

```
python batch_runs.py --base-config config/ppo/<Game>.yaml --env builds/<Game>/UnityEnvironment.exe --game <Game> --steps 100000 250000 500000 --repeats 2
```

Example for 3DBall:
```
python batch_runs.py --base-config config/ppo/3DBall.yaml --env builds/3DBall/UnityEnvironment.exe --game 3DBall --steps 100000 250000 500000 --repeats 2
```

- It appends rows to `training_data.csv` (it never overwrites; safe to run many times).
- Heavier games (FoodCollector, Pyramids) take longer, up to a couple of hours. Leave it running, keep the PC awake, don't do heavy work on it during a run (it inflates the time).
- Do not close the Unity window that opens; the script manages it.

### Sorter: use the batch script, do NOT run it raw

Sorter's config is `config/ppo/Sorter_curriculum.yaml`, and its built-in max_steps is
5,000,000 with a curriculum, so running it directly takes about 3 hours per run.

Always use the batch script for Sorter too, which overrides max_steps to our shared
values (100k / 250k / 500k) so its rows are comparable with the other games:

```
python batch_runs.py --base-config config/ppo/Sorter_curriculum.yaml --env builds/Sorter/UnityEnvironment.exe --game Sorter --steps 100000 250000 500000 --repeats 2
```

At these step counts Sorter will not fully learn (low reward), which is fine for the
cost question (RQ1). Only RQ3 (performance) may want a few longer Sorter runs on top.

## What gets recorded (one row per run)

Inputs: game, trainer, max_steps, hyperparameters, and your hardware (OS, CPU, cores,
clock speed, RAM). Outputs: training time (seconds), peak RAM (MB), final mean reward.

## Send your data back

After your runs, share your `training_data.csv` (rename it with your name, e.g.
`training_data_vito.csv`) so we can merge everyone's into one master dataset.
Do NOT overwrite someone else's file.

## Quick checklist

- [ ] Environment set up, versions correct (2.2.2 / 1.23.5 / 1.12.0)
- [ ] Scripts copied in, `pyyaml` and `psutil` installed
- [ ] Sleep disabled
- [ ] Game built into `builds/<Game>/`
- [ ] Ran the batch, `training_data.csv` has your rows
- [ ] Renamed and shared your CSV
