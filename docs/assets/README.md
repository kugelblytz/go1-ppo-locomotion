# Rollout media provenance

These MP4s are decoded directly from saved notebook HTML outputs. No simulation
was rerun and no frames were generated, cut, or re-encoded in the MP4s.

| File | Original notebook → current name | Cell / output | Demonstration |
| --- | --- | --- | --- |
| [flat-ground-ppo.mp4](flat-ground-ppo.mp4) | `lab1_E.ipynb` → `ppo_flat_ground.ipynb` | 13 / 3 | Flat-ground forward velocity command |
| [rough-terrain-body-sensing.mp4](rough-terrain-body-sensing.mp4) | `lab1_C.ipynb` → `terrain_body_sensing.ipynb` | 15 / 4 | Raised box terrain with 100 body-relative height samples |
| [rough-terrain-feet-sensing.mp4](rough-terrain-feet-sensing.mp4) | `lab1_C_feet_sensing.ipynb` → `terrain_feet_sensing.ipynb` | 15 / 4 | Raised box terrain with 200 foot-relative height samples |

Indices are zero-based. All three clips are the **second video** in the final
evaluation cell, using a deterministic policy with command `[0.5, 0, 0]`
(forward m/s, lateral m/s, yaw rad/s). The command order is defined in
[`utils.evaluate_policy`](../../utils.py). The evaluation cells pass a zero
velocity-kick range; these clips are not evidence of push-recovery performance.
Each MP4 is H.264, 640 × 480, 25 fps, 500 frames, and 20 seconds long.

The corresponding GIFs preview the opening 8 seconds at 320 × 240 and 8 fps
with a reduced palette. They preserve playback speed and loop. Use the
original MP4s to inspect foot motion and the terrain overlay at full resolution.

The main body and final foot-ring runs were selected to show the two observation
designs using the same forward command. The earlier `lab1_C best.ipynb` and
`lab1_C new_best.ipynb` were also inspected: both actually use 36 **foot-relative**
samples, and would be misleading choices for the body-sensing demonstration.
The four-point `lab1_C copy.ipynb` remains documented as a separate snapshot in
the [experiment record](../experiments.md).

## Re-extract or verify

From the repository root, using Python's standard library:

```bash
python scripts/export_notebook_media.py
python scripts/export_notebook_media.py --check
```

To regenerate previews as well, install `ffmpeg` and run:

```bash
python scripts/export_notebook_media.py --previews
```

[`media-sources.json`](media-sources.json) records exact cell/output indices,
the original Git revision and notebook hashes, commands, MP4 hashes, and media
properties. The exporter checks the saved video hash before writing, so rerunning
a notebook cannot silently replace the documented demonstration.

The previous `level-e-forward-evaluation.gif` was replaced by the consistently
named `flat-ground-ppo.gif`. Hugging Face retains its earlier independently
exported Level E media; the files here come from the final saved Git notebook
and need not share that older export's byte hash.
