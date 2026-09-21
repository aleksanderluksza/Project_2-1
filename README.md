# BCS2720 Project 2.1: Machine Learning in the Unity Game Engine

Group repository for Project 2.1 (AI and Machine Learning module, 2026-2027). We collect data
from Unity ML-Agents training runs and study it with machine learning techniques.

This README is the setup documentation required by the project manual: it is written so that
any team member (or a new member joining mid-phase) can get a working environment from
scratch. Every one of the six group members installs this on their own machine, and everyone
uses the same fork, branch, and package versions.

---

## Required versions

| Tool | Version |
|---|---|
| Python | 3.10.x (3.10.12 required by the manual; 3.10.11 is the easy Windows option, and works) |
| ML-Agents | Dennis Soemers' fork, branch `fix-numpy-release-21-branch`, commit `c080ea212` |
| Unity | 2022.3.4f1 (this project's pinned editor, see `Project/ProjectSettings/ProjectVersion.txt`) |
| Git | any recent version |
| Visual Studio | Windows only, for C++ build tools |

Do not upgrade `setuptools` past 80 in the venv (see Troubleshooting).

---

## Setup

### 1. Git and this repository
Install Git and create a GitHub account. Everyone commits to this one public repository. Clone
the recommended fork and branch (it pins numpy to versions that work with Python 3.10):
```
git clone --branch fix-numpy-release-21-branch https://github.com/DennisSoemers/ml-agents.git
cd ml-agents
```

### 2. Install Python 3.10
- Windows: download the 3.10.11 installer from python.org and tick "Add Python to PATH".
- macOS: `brew install python@3.10` or the python.org installer.
- Linux (Ubuntu 24.04): system Python is 3.12, so use the deadsnakes PPA or a version manager
  (pyenv, uv) to get 3.10.x.

### 3. Create and activate a virtual environment
From inside the `ml-agents` folder:
```
# Windows
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1        # PowerShell
.\venv\Scripts\activate.bat        # cmd

# macOS / Linux
python3.10 -m venv venv
source venv/bin/activate
```
Your prompt must start with `(venv)`, and `python --version` must report 3.10.x, before you
install or run anything. If PowerShell blocks the activator, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in the same window, then
activate again.

### 4. Install the ML-Agents Python packages
With the venv active, install the environments package first, then the trainers package:
```
pip install --upgrade pip wheel
pip install -e ./ml-agents-envs
pip install -e ./ml-agents
```
On Windows this also installs pywin32 and pypiwin32 automatically. (Linux only: the default
torch wheel bundles several GB of CUDA libraries; if disk is tight, first run
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.)

### 5. Install Unity
Install Unity 2022.3.4f1 (Unity Personal or Student license). Open the `Project/` folder as a
Unity project with that exact editor version to avoid an unintended project upgrade.

### 6. Verify
With the venv active:
```
mlagents-learn --help
python --version
pip freeze
```
`mlagents-learn --help` should print its usage text. A deprecation warning (torch
`set_default_tensor_type`, or `pkg_resources`) is harmless. To verify Unity, open an example
scene (for example SoccerTwos) and press Play.

---

## Troubleshooting

- **PowerShell: "script execution is disabled".** Run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again; or use
  `activate.bat`.
- **`python` shows the wrong version (for example 3.14).** The venv is not active. Activate it
  and confirm the prompt shows `(venv)` and `python --version` reports 3.10.x.
- **`ModuleNotFoundError: No module named 'pkg_resources'` from mlagents-learn.** setuptools 81
  and above removed pkg_resources. Fix with `pip install "setuptools<81"`; do not over-upgrade
  setuptools.
- **`OSError: [Errno 28] No space left on device` installing torch (Linux).** Clear the pip
  cache and install a CPU/smaller torch (see step 4).

---

## Verified known-good environment (reference)

One machine confirmed end to end. Use it to check your own setup matches.

| Item | Value |
|---|---|
| OS | Windows 11 Home, 25H2 (build 26200.9457) |
| Python | 3.10.11 (in the project venv) |
| mlagents / mlagents-envs | editable from fork @ `c080ea212` |
| numpy | 1.23.5 (pinned <1.24) |
| torch | 2.14.0 |
| onnx | 1.12.0 |
| protobuf | 3.19.6 |
| grpcio | 1.48.2 |
| tensorboard | 2.20.0 |
| PettingZoo | 1.15.0 |
| gym | 0.26.2 |
| Unity | 2022.3.4f1 |
| mlagents-learn CLI | runs (harmless torch deprecation warning) |

---

## Team workflow

- All six members use the same repository, fork, branch, and versions.
- Tasks are tracked in the GitHub **Issues** tab, assigned per member, and reviewed in each
  project meeting with the tutor.
- Keep this README up to date so a new member can get running quickly.
