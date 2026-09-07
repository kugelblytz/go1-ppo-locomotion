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

[Source repository](https://github.com/kugelblytz/go1-ppo-locomotion) ·
[Hugging Face model archive](https://huggingface.co/kugelblytz/go1-ppo-locomotion) ·
[PPO paper](https://arxiv.org/abs/1707.06347)

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
The [checkpoint inventory](https://github.com/kugelblytz/go1-ppo-locomotion/blob/main/docs/checkpoints.json) records the archive revision,
local run paths, parameter hashes, and complete rough-terrain sampling metadata.
The two uploaded perception specifications also match their local copies.
The archive additionally contains an older 50,790,400-step rough checkpoint at
`checkpoints/20260828_172535/step_000050790400/`; it is not a final result here.

## Demonstrations

The source repository includes notebook-extracted deterministic rollouts:

- [Flat-ground PPO](https://github.com/kugelblytz/go1-ppo-locomotion/blob/main/docs/assets/flat-ground-ppo.mp4): 0.5 m/s forward command on flat terrain;
- [Body sensing](https://github.com/kugelblytz/go1-ppo-locomotion/blob/main/docs/assets/rough-terrain-body-sensing.mp4): raised terrain with a torso grid;
- [Feet sensing](https://github.com/kugelblytz/go1-ppo-locomotion/blob/main/docs/assets/rough-terrain-feet-sensing.mp4): raised terrain with foot rings.

The rough-terrain videos are qualitative results. The two runs used different
training budgets and no shared held-out traversal success metric was recorded,
so they do not establish that one sensing design is superior.
The final logs show `eval/episode_reward=104.664` for body sensing and 108.363
for feet sensing. Reward settings also differ, so these totals do not measure
the isolated effect of perception. See the [experiment record](https://github.com/kugelblytz/go1-ppo-locomotion/blob/main/docs/experiments.md)
for exact sources and limitations. The flat-ground step count is retained from
the existing uploaded model card rather than a printed notebook step metric.

## Model inputs

The flat-ground policy uses the standard Go1 joystick observation and does not
need terrain metadata.

A rough-terrain policy is architecture-dependent on the exact terrain scan
used during training. Its checkpoint directory must include `perception.json`
next to `params`. Version 2 metadata records:

- `version` (2) and `coordinate_frame` (`robot_yaw`);
- each `body`, `FL`, `FR`, `RL`, or `RR` anchor and its ordered XY offsets;
- `height_scale` and `use_height_scan`;
- the training `terrain_type`.

At runtime, sample offsets are anchored to the torso or current foot position
and rotated by torso yaw. Heights are encoded relative to terrain beneath the
torso. Changing the sample count or ordering changes the policy input shape or
semantics and is not checkpoint-compatible.

The body policy uses 100 heights (actor input 148; critic input 223); the feet
policy uses 200 (actor 248; critic 323). Heights are offset by terrain under the
torso, divided by 0.5 m, and clipped to [−1, 1]. The flattening order is `body`,
`FR`, `FL`, `RR`, `RL`, omitting unused anchors. These conventions must match
even when two patterns have equal sample counts.

The metadata does not store the complete training/reward configuration, network
architecture, or noise settings. The saved notebooks use 0.01 m height noise
and an action scale of 0.8. Keep the matching notebook and environment code for
reproduction. The supplied keyboard controller reads sample geometry but uses
CLI/default settings for other fields; see [loading instructions](https://github.com/kugelblytz/go1-ppo-locomotion/blob/main/docs/running.md).

## Method and provenance

The flat-ground assignment implements the PPO likelihood ratio, clipped
surrogate policy objective, value loss, entropy regularization, and combined
loss. The terrain assignment explores body-relative and foot-relative height
observations plus a tuned reward combining command tracking, stability,
smoothness, energy, and terrain-relative swing clearance.
The rough notebooks train with Brax's supplied PPO loss; the custom loss is
used in the flat-ground assignment.

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
