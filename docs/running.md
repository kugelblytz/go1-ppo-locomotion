# Running the experiments

The original setup targets Python 3.12, CUDA 12, and a GPU. Runtime package
versions are listed in [`requirements.txt`](../requirements.txt); some helper
dependencies are unpinned, so this is not a fully locked environment.

From the repository root, with `uv` installed:

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install ipykernel jupyterlab
./setup_mujoco_headless.sh
python -c "import jax, mujoco; print(jax.devices()); print(mujoco.__version__)"
jupyter lab
```

The headless setup script configures the repository-local software renderer.
The [original assignment setup](../ASSIGNMENT_README.md#getting-started) includes
kernel registration. Keep Jupyter's working directory at the repository root.

## Training

Open one of the renamed notebooks:

| Notebook | Requested steps | Saved actual steps |
| --- | ---: | ---: |
| [ppo_flat_ground.ipynb](../ppo_flat_ground.ipynb) | 200,000,000 | 201,523,200, recorded in the existing model card |
| [terrain_body_sensing.ipynb](../terrain_body_sensing.ipynb) | 200,000,000 | 201,523,200 |
| [terrain_feet_sensing.ipynb](../terrain_feet_sensing.ipynb) | 500,000,000 | 504,627,200 |

Reduce `ppo_training_params["num_timesteps"]` before calling `train_fn` for a
short run. New outputs are new experiments; they are not reproductions of the
saved metrics merely because they use the same notebook. The rough notebooks
save `params` and `perception.json` under
`artifacts/notebook_checkpoints/<timestamp>/final/`. The flat notebook retains
parameters in memory after training; its separate existing checkpoint export is
documented in the [model card](../MODEL_CARD.md).

## Existing checkpoints

The [Hugging Face archive](https://huggingface.co/kugelblytz/go1-ppo-locomotion)
contains all three final checkpoints, but was private when checked on
2026-09-06. Download access requires authorization to that repository. The
[checkpoint inventory](checkpoints.json) records paths, hashes, and the exact
rough-terrain metadata. Checkpoints are not included in a Git clone.

For a downloaded **rough-terrain** checkpoint, keep `perception.json` beside
`params` and run the supplied controller on a machine with a desktop display:

```bash
python remote_control.py \
  --checkpoint artifacts/downloaded_checkpoints/<run>/final/params \
  --terrain boxes --scan on --fullscreen
```

W/S commands forward/backward motion, A/D lateral motion, Q/E yaw, X or Space
stops, R resets, and Escape exits. This controller constructs the rough-terrain
environment; it is not the loader for the flat-ground policy.

The controller restores sample anchors and offsets from metadata, but takes
terrain and scan activation from CLI arguments and otherwise uses environment
defaults. The documented final policies both use `boxes`, scan enabled, and
`height_scale=0.5`, matching those defaults. For other checkpoints, also match
the height normalization, noise, network configuration, and action scale in
the environment. Merely keeping a metadata file next to parameters does not
make every recorded field automatically applied by the current controller.

For **flat ground**, use the flat notebook's setup and network configuration
with `Go1JoystickFlatTerrain`. Load `checkpoints/level_e/final/params` through
`brax.io.model.load_params` and construct its deterministic inference function
with `ppo_networks.make_inference_fn`; it has no terrain scan or
`perception.json`. A rough-terrain network's input dimensions are incompatible.

## View the saved results without training

Open the [standalone MP4s](assets/README.md) or existing notebook outputs.
Media extraction only needs Python's standard library; GIF generation
additionally needs `ffmpeg`:

```bash
python scripts/export_notebook_media.py --check
python scripts/export_notebook_media.py --previews
```

These commands never import the simulation stack or execute notebook code.
