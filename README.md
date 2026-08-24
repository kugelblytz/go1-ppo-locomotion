# Lab 1 - Reinforcement Learning with a Quadruped

## Overview

In this lab you will complete and tune a code base for training a four-legged robot in MuJoCo.
This includes implementing a component of the PPO algorithm, tuning the cost function to instruct the robot to walk over an elevated step, and reason about possible extensions and steps towards real-world deployment.

The lab will have three different levels, 'E', 'C' and 'A'. The level naming corresponds to the grade that is obtained if the respective level is completed. Levels E and C are code completion exercises, while level A is a small research task.
The levels are incremental, so e.g. reaching grade C is only possible if levels E and C are successfully completed, similarly level E and C and A are required for reaching an A grade.

## Lab Structure

- **Level E** (`lab1_E.ipynb`): Implement PPO loss function, and show that the robot can follow simple velocity commands
- **Level C** (`lab1_C.ipynb`): Design how the robot perceives the terrain, tune its rewards, and train it to climb obstacles.
- **Level A**: Discussion deliverable - reason about extensions and real-world deployment (see below)

## Getting Started
We will provide instructions to use our GPU cluster in the lab.
### Clone the repository

First, clone the lab repository:

```git clone https://github.com/finnBsch/eai2026_lab1_rl.git```

Then navigate to the cloned directory:

```cd eai2026_lab1_rl```

### Create the Python environment with `uv`

This lab requires Python 3.12. The cluster may not provide that version as a
system module, so we use `uv` to install an isolated Python and virtual
environment without administrator access.

Install `uv` once for your user account:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

If `uv` was already installed, verify that it is available with `uv --version`.
From the repository directory, install Python 3.12 and create the environment:

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
python --version
```

The final command must report Python 3.12. Install the lab and Jupyter
dependencies into the activated environment:

```bash
uv pip install -r requirements.txt
uv pip install ipykernel jupyterlab
```

On the headless teaching cluster, install the repository-local MuJoCo software
renderer while the environment is activated and the current directory is the
lab repository:

```bash
chmod +x setup_mujoco_headless.sh
./setup_mujoco_headless.sh
```

### Make the environment available to Jupyter

Register the environment as a named Jupyter kernel:

```bash
python -m ipykernel install --user \
  --name eai-lab1 \
  --display-name "EAI Lab 1 (Python 3.12)"
jupyter kernelspec list
```

In VS Code, open a notebook, click the kernel selector in the upper-right,
choose **Select Another Kernel** (then **Jupyter Kernel**, if shown), and select
**EAI Lab 1 (Python 3.12)**.

Verify the installation before opening the notebooks:

```bash
python -c "import jax, mujoco; print(jax.devices()); print(mujoco.__version__)"
```

The output should contain a CUDA device. The first environment run downloads
the robot assets and compiles the simulation, so it can take several minutes.

## Checkpoints and optional keyboard control

Level C saves PPO parameters at every evaluation and once more at the end of training under `artifacts/notebook_checkpoints/<run timestamp>/`. The last path is also available in the notebook as `latest_checkpoint`.

The keyboard controller is optional and intended for a local computer with a
desktop display. Download the complete checkpoint directory—not only the
`params` file—because its adjacent `perception.json` records the terrain-sample
layout used by the policy. Use Jupyter's file browser and **Download** button to
download the complete `final` folder to your computer.

Clone the same repository locally and follow the `uv` environment and
requirements installation above. Then load the downloaded policy from a local
terminal:

```bash
python remote_control.py \
  --checkpoint artifacts/downloaded_checkpoints/<run>/final/params \
  --fullscreen
```

Use W/S for forward/backward velocity, A/D for left/right velocity, Q/E for yaw velocity, X or Space to stop, R to reset, and Escape to close. The window uses the same tracking-camera render, visible terrain groups, 3D samples, and ego terrain-observation panel as the notebook videos. It renders at 25 Hz while control runs at 50 Hz. The controller automatically selects CUDA, Apple Metal, or CPU in that order.

## Level A - Detailed Instructions
For Level A, you will need to write a **1-2 page discussion** about the policy you trained in Level C and potential extensions to improve it. Use the papers linked below as references and inspiration for your discussion.

### Required Discussion Points:
Your report must cover the following topics:

1. **Perception/Reward Function Analysis**
   - Examine the reward implementations in `perceptive_go1.py` and the reward weights in the Level C notebook.
   - Explain what information the policy uses for decision-making
   - Discuss how this perception influences obstacle climbing behavior, and what the robot must do to climb obstacles.

2. **System Limitations and Improvements**
   - Identify at least 2 specific problems with the current setup regarding:
     - Safety considerations for real-world deployment
     - Perception for real-world deployment
     - Applicability of the approach to more complex environments
   - Discuss what is needed to overcome those specific problems, for instance you could suggest additional sensors and explain how they would be integrated

3. **Real-World Deployment Considerations**
   - Discuss the sim-to-real transfer challenges
   - Address robustness, safety, and reliability requirements
   - Reference insights from the provided papers to support your arguments


## Related Resources

- PPO paper (OpenAI): https://arxiv.org/pdf/1707.06347
- Perceptive Quadruped RL (ETH Zurich): https://arxiv.org/pdf/2201.08117
- Non-perceptive Quadruped RL (ETH Zurich): https://arxiv.org/pdf/2010.11251
