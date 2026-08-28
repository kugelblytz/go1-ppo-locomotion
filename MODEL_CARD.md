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

<p align="center">
  <img src="assets/go1-ppo-banner.svg" alt="Go1 PPO Locomotion" width="100%">
</p>

# Go1 PPO Locomotion

This repository is the model archive for Proximal Policy Optimization
locomotion experiments with a simulated Unitree Go1 quadruped.

> **Status:** the custom Level E PPO loss and flat-ground experiment are
> complete. No checkpoint has been exported from the notebook yet, so this
> repository currently contains documentation only.

## Source code

The implementation, experiment notebooks, environment setup, and loading tools
are in the private GitHub repository:

[kugelblytz/go1-ppo-locomotion](https://github.com/kugelblytz/go1-ppo-locomotion)

The project is based on the
[EAI 2026 Lab 1 starter repository](https://github.com/finnBsch/eai2026_lab1_rl).
The original instructions are preserved in the project source repository.

## Method

The project trains a Go1 locomotion policy with PPO using JAX, Brax, and MuJoCo
Playground. Level E implements the clipped surrogate policy objective, value
loss, entropy regularization, and combined PPO loss for flat-ground training.
Terrain perception and obstacle traversal belong to the later Level C stage and
are not claimed as completed results here.

## Checkpoint layout

The final upload will preserve the complete checkpoint directory:

```text
final/
├── params
└── perception.json
```

`perception.json` must remain adjacent to `params`; it records the terrain
sampling geometry expected by the trained policy.

## Results

Training steps, evaluation reward, selected configuration, and demonstration
media will be documented here after final training and evaluation.

## Intended use

This policy is intended for coursework, reinforcement-learning experiments,
and simulation-only evaluation. It has not been validated on physical robot
hardware.
