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

# Go1 Perceptive Rough-Terrain PPO

This repository hosts a Proximal Policy Optimization policy for perceptive
rough-terrain locomotion with a simulated Unitree Go1 quadruped.

> **Status:** the model repository has been created, but the final Level C
> checkpoint is still training and has not yet been uploaded.

## Source code

The implementation, experiment notebooks, environment setup, and loading tools
are in the private GitHub repository:

[kugelblytz/go1-perceptive-locomotion-ppo](https://github.com/kugelblytz/go1-perceptive-locomotion-ppo)

The project is based on the
[EAI 2026 Lab 1 starter repository](https://github.com/finnBsch/eai2026_lab1_rl).
The original instructions are preserved in the project source repository.

## Method

The project trains a Go1 locomotion policy with PPO using JAX, Brax, and MuJoCo
Playground. The work includes a custom PPO loss implementation and a perceptive
policy that receives local terrain-height observations for obstacle traversal.

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
