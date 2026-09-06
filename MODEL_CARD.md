---
library_name: jax
tags:
  - reinforcement-learning
  - robotics
  - ppo
  - jax
  - mujoco
  - brax
  - unitree-go1
---

# Go1 PPO locomotion

Model information for PPO locomotion experiments with a simulated Unitree Go1,
from flat-ground velocity tracking to exploratory terrain-aware control.

## Checkpoint availability

| Policy | Training record | Availability |
| --- | --- | --- |
| Flat-ground PPO | 201,523,200 environment steps | Private Hugging Face archive: `checkpoints/level_e/final/params` |
| Rough terrain, body sensing | 201,523,200 environment steps | Private Hugging Face archive: `checkpoints/level_c/final/params` plus `perception.json` |
| Rough terrain, feet sensing | 504,627,200 environment steps | Private Hugging Face archive: `checkpoints/level_c_feet_sensing/final/params` plus `perception.json` |

All three uploaded parameter files were checked against the corresponding local
run artifacts by SHA-256 on 2026-09-06. The model archive is private, so this
availability does not imply anonymous public access. The source repository
deliberately excludes generated checkpoints.

## Demonstrations

The source repository includes notebook-extracted deterministic rollouts:

- `docs/assets/flat-ground-ppo.mp4`: 0.5 m/s forward command on flat terrain;
- `docs/assets/rough-terrain-body-sensing.mp4`: body-anchored terrain scan;
- `docs/assets/rough-terrain-feet-sensing.mp4`: foot-anchored terrain scans.

The rough-terrain videos are qualitative results. The two runs used different
training budgets and no shared held-out traversal success metric was recorded,
so they do not establish that one sensing design is superior.

## Model inputs

The flat-ground policy uses the standard Go1 joystick observation and does not
need terrain metadata.

A rough-terrain policy is architecture-dependent on the exact terrain scan
used during training. Its checkpoint directory must include `perception.json`
next to `params`. Version 2 metadata records:

- `coordinate_frame` (`robot_yaw`);
- each `body`, `FL`, `FR`, `RL`, or `RR` anchor and its ordered XY offsets;
- `height_scale` and `use_height_scan`;
- the training `terrain_type`.

At runtime, sample offsets are anchored to the torso or current foot position
and rotated by torso yaw. Heights are encoded relative to terrain beneath the
torso. Changing the sample count or ordering changes the policy input shape or
semantics and is not checkpoint-compatible.

## Method and provenance

The flat-ground assignment implements the PPO likelihood ratio, clipped
surrogate policy objective, value loss, entropy regularization, and combined
loss. The terrain assignment explores body-relative and foot-relative height
observations plus a tuned reward combining command tracking, stability,
smoothness, energy, and terrain-relative swing clearance.

The project builds on the [EAI 2026 Lab 1 starter
repository](https://github.com/finnBsch/eai2026_lab1_rl) and its Brax, JAX, and
MuJoCo Playground training and simulation stack. The repository author did not
write the complete framework.

## Intended use and limitations

These policies are intended for coursework and simulation experiments. They
have not been validated or deployed on physical Go1 hardware. The environment
uses simulator terrain heights directly; a real system would require a sensor
and estimation pipeline that reproduces the expected observation semantics,
along with sim-to-real validation and safety controls.
