# Experiment record and notebook guide

The completed project investigates how terrain observation design affects learned
Go1 locomotion. The saved experiments vary sampling geometry, reward weights,
and training duration; they are exploratory runs, not a controlled ablation.

## Main notebooks

| Notebook | Original name | Saved run | Evidence |
| --- | --- | --- | --- |
| [ppo_flat_ground.ipynb](../ppo_flat_ground.ipynb) | `lab1_E.ipynb` | Flat-ground PPO | Implemented loss in cell 7; completed training in cell 11; six evaluation videos in cell 13 |
| [terrain_body_sensing.ipynb](../terrain_body_sensing.ipynb) | `lab1_C.ipynb` | `20260830_175137` | 100 body samples; final training log in cell 13; six evaluation videos in cell 15 |
| [terrain_feet_sensing.ipynb](../terrain_feet_sensing.ipynb) | `lab1_C_feet_sensing.ipynb` | `20260830_184942` | 200 foot samples; final training log in cell 13; six evaluation videos in cell 15 |

Cell and output indices in this documentation are **zero-based**. The three
notebooks were renamed without changing their contents, execution counts, or
embedded outputs. Retained assignment prompts and TODO comments describe the
original exercise; the implementations and saved runs are complete. In the body
notebook, the stale `(36, 2)` comment describes a 100-point array in the active
code; the saved observation size and checkpoint metadata both confirm 100.

## Observation designs

- **Body grid:** 10 × 10 samples over local x/y offsets from −0.4 to +0.4 m.
  Samples translate with the torso and are oriented by the robot's yaw.
- **Foot rings:** 50 samples per foot, comprising four rings with 6, 10, 14,
  and 20 points. Radii span 0.025–0.20 m. Each ring set follows its foot's
  position while retaining the robot's yaw orientation.
- The fixed flattening order is `body`, `FR`, `FL`, `RR`, `RL`, omitting unused
  anchors. The final actor observation sizes are 148 (body) and 248 (feet);
  the corresponding critic inputs are 223 and 323.

The supplied environment samples a precomputed simulator elevation map. Local
sample offsets use +x forward and +y left. Heights are encoded relative to terrain
under the torso, divided by `height_scale=0.5`, and clipped to [−1, 1]; the saved
rough configurations also use 0.01 m height noise. This is simulated terrain
information, not a camera, lidar, or reconstructed sensor map.

## Recorded training evaluations

| Experiment | Actual environment steps | Final logged `eval/episode_reward` |
| --- | ---: | ---: |
| Flat ground | 201,523,200 | Not preserved as a text metric |
| Body grid, 100 samples | 201,523,200 | 104.664 |
| Foot rings, 200 samples | 504,627,200 | 108.363 |

The flat-ground step count comes from the existing model card associated with
the uploaded checkpoint. Its notebook requests 200 million steps and preserves
the completed run and videos; it does not print the final actual step count.
The rough-terrain numbers are printed directly in cell 13. `progress()` in cell
12 labels `metrics["eval/episode_reward"]` as `reward` in those logs. Actual step
counts exceed requested budgets because training advances in batches.

The foot-ring run also records 25.482 at 126,156,800 steps, 66.546 at
252,313,600 steps, and 107.604 at 378,470,400 steps. These are evaluations during
one training run, not independent replications.

The reward totals cannot establish which observation design is better: the final
feet run has 2.5× the requested training budget, twice the terrain inputs, and a
different stand-still penalty. No common held-out benchmark, multi-seed statistics,
success rate, or aggregate velocity error is retained. The rendered evaluations
do not export numerical comparisons: despite its original docstring,
`utils.evaluate_policy()` displays videos and does not return average reward.
It also runs for a fixed horizon without stopping on `done`, so video duration
alone is not an episode-survival metric.

## Reward settings

The final saved notebook configurations use these scales. They document choices,
not evidence that each term independently improved behavior.

| Term | Body grid | Foot rings | Purpose |
| --- | ---: | ---: | --- |
| `tracking_lin_vel` | 4.0 | 4.0 | Match commanded forward/lateral velocity |
| `tracking_ang_vel` | 0.5 | 0.5 | Match commanded yaw rate |
| `orientation` | −1.0 | −1.0 | Penalize body tilt |
| `lin_vel_z` | −0.25 | −0.25 | Penalize local vertical motion |
| `ang_vel_xy` | −0.05 | −0.05 | Penalize roll/pitch angular velocity |
| `pose` | 0.25 | 0.25 | Encourage nominal joint pose |
| `feet_clearance` | 1.5 | 1.5 | Reward terrain-relative swing peak on touchdown |
| `feet_slip` | −0.05 | −0.05 | Penalize foot sliding during contact |
| `feet_air_time` | 2.0 | 2.0 | Shape swing duration at touchdown |
| `foot_contact` | 1.5 | 1.5 | Reward support, saturating at two contacting feet |
| `stand_still` | −5.0 | −1.0 | Penalize joint deviation under a zero command |
| `action_rate` | −0.01 | −0.01 | Penalize consecutive action changes |
| `torques` | −0.0002 | −0.0002 | Penalize squared motor torques |
| `energy` | −0.001 | −0.001 | Penalize mechanical power use |
| `dof_pos_limits` | −1.0 | −1.0 | Penalize soft joint-limit violations |
| `termination` | −1.0 | −1.0 | Penalize falling/collapse |

`command_progress`, `base_height`, `dof_acc`, `stall`, and absolute `feet_height`
have zero weight in both final configurations. The environment explicitly
disables absolute foot-height reward. Reward terms are summed with their signed
weights and multiplied by the 0.02 s control timestep.

Terrain-relative clearance is `foot_z - terrain_height(foot_x, foot_y)`. A foot
resting on a 10 cm platform has approximately zero clearance even though its
world-space height is 10 cm. The completed-swing reward uses the peak clearance,
saturates at 0.10 m, and pays at touchdown after at least 0.08 s of air time when
a movement command is active. The separate air-time term uses a 0.5 s threshold,
so short swings can make a negative contribution even with a positive weight.

## Earlier snapshots: retain as experiment history

All three were inspected from commit
[`f4f4848`](https://github.com/kugelblytz/go1-ppo-locomotion/tree/f4f484867323b95effd49dae7d6df0f6ad26f3db).
They contain distinct configurations and videos; none is an exact redundant copy.
The names “best” and “new_best” are historical labels, not verified rankings.

| Historical notebook | Saved run | Samples | Steps / final logged reward | Recommended archive name |
| --- | --- | --- | --- | --- |
| [lab1_C best.ipynb](https://github.com/kugelblytz/go1-ppo-locomotion/blob/f4f484867323b95effd49dae7d6df0f6ad26f3db/lab1_C%20best.ipynb) | `20260829_201301` | 9 offsets per foot, 36 total | 201,523,200 / 82.825 | `terrain_feet_grid_36_no_airtime.ipynb` |
| [lab1_C new_best.ipynb](https://github.com/kugelblytz/go1-ppo-locomotion/blob/f4f484867323b95effd49dae7d6df0f6ad26f3db/lab1_C%20new_best.ipynb) | `20260829_212906` | Same 36 foot offsets | 201,523,200 / 83.108 | `terrain_feet_grid_36_airtime.ipynb` |
| [lab1_C copy.ipynb](https://github.com/kugelblytz/go1-ppo-locomotion/blob/f4f484867323b95effd49dae7d6df0f6ad26f3db/lab1_C%20copy.ipynb) | `20260830_010519` | One sample directly under each foot | 201,523,200 / 79.323 | `terrain_feet_points_4.ipynb` |

The 36-point pair changes `feet_air_time` from 0 to 0.5; the 4-point run also
uses `foot_contact=0.15`. Both differ from the later body and ring configurations
(which use clearance 1.5, air time 2.0, contact 1.5, and slip −0.05). These
snapshots are meaningful experiment history rather than exact duplicates. They
were removed from the portfolio-facing tree to keep the main entry points clear,
but remain available through the linked Git revision. The recommended archive
names above should be used if they are restored later. The three descriptively
named notebooks are the main project entry points.

## Evidence limits

The body notebook has unexecuted markers on its saved sampling/reward cells;
its 100-point layout is independently corroborated by checkpoint metadata and
the printed 148/223 observation dimensions. The reward table describes saved
source, which cannot prove the exact in-memory configuration of a past kernel.
The checkpoints contain parameters, normalization state, and terrain metadata,
but do not serialize the complete reward configuration or training provenance.

Selected forward videos show both rough policies stepping over raised box
sections and continuing forward. They support a qualitative demonstration,
not general obstacle-traversal reliability or sim-to-real capability. All work
is simulation-only; there has been no deployment on a physical Go1.
