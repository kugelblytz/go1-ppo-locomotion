"""Load a Level C PPO checkpoint and drive the rendered Go1 with the keyboard."""

from __future__ import annotations

import argparse
import functools
import os
from pathlib import Path
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax


def _select_compute_device() -> jax.Device:
  """Selects the fastest available JAX device for interactive control."""
  for platform, label in (
      ("cuda", "CUDA"),
      ("metal", "Apple Metal/MPS"),
      ("cpu", "CPU"),
  ):
    try:
      devices = jax.devices(platform)
    except RuntimeError:
      continue
    if devices:
      device = devices[0]
      jax.config.update("jax_default_device", device)
      print(f"Remote control backend: {label} ({device.device_kind})")
      return device
  raise RuntimeError("No usable JAX compute device was found.")


_COMPUTE_DEVICE = _select_compute_device()

from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
import jax.numpy as jnp
import mujoco
import cv2
import numpy as np
from mujoco_playground.config import locomotion_params

from perceptive_go1 import make_env
from terrain_samples import load_perception_spec
from utils import _draw_perception_grid, _overlay_scan_map


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--terrain", choices=("boxes", "heightfield", "combined"), default="boxes")
  parser.add_argument("--scan", choices=("on", "off"), default="on")
  parser.add_argument("--speed-step", type=float, default=0.2)
  parser.add_argument("--render-every", type=int, default=1)
  parser.add_argument("--width", type=int, default=960)
  parser.add_argument("--height", type=int, default=540)
  parser.add_argument("--fullscreen", action="store_true")
  parser.add_argument("--contacts", action="store_true")
  args = parser.parse_args()
  if args.render_every < 1:
    parser.error("--render-every must be at least 1")

  perception_spec = load_perception_spec(args.checkpoint)
  sample_offsets = None
  if perception_spec is not None:
    if "samples" in perception_spec:
      sample_offsets = {
          anchor: np.asarray(points, dtype=np.float32)
          for anchor, points in perception_spec["samples"].items()
      }
    else:  # Version-1 body-only metadata.
      sample_offsets = {"body": np.asarray(
          perception_spec["sample_offsets_xy"], dtype=np.float32
      )}
    point_count = sum(len(points) for points in sample_offsets.values())
    print(
        f"Loaded {point_count} terrain sample locations from "
        f"{args.checkpoint.parent / 'perception.json'}"
    )
  else:
    print("Legacy checkpoint: using the default terrain sample pattern.")
  env = make_env(
      args.scan == "on",
      terrain_type=args.terrain,
      terrain_sample_offsets=sample_offsets,
      impl="jax",
  )
  ppo_cfg = locomotion_params.brax_ppo_config("Go1JoystickRoughTerrain")
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks, **ppo_cfg.network_factory
  )
  networks = network_factory(
      env.observation_size,
      env.action_size,
      preprocess_observations_fn=running_statistics.normalize,
  )
  make_policy = ppo_networks.make_inference_fn(networks)
  params = brax_model.load_params(str(args.checkpoint))
  policy = jax.jit(make_policy(params, deterministic=True))
  reset = jax.jit(env.reset)
  scan_points_fn = jax.jit(env.scan_points)
  scan_ego_offsets_fn = jax.jit(env.scan_ego_offsets)

  @jax.jit
  def control_tick(state, action_rng, command):
    # Command is [forward, left, yaw-rate] in the robot's current body frame.
    state.info["command"] = command
    state.info["steps_until_next_cmd"] = jnp.asarray(
        1_000_000, dtype=jnp.int32
    )
    state = state.replace(obs=env._get_obs(state.data, state.info))
    action, _ = policy(state.obs, action_rng)
    return env.step(state, action)

  controls = {"vx": 0.0, "vy": 0.0, "yaw": 0.0, "reset": False}

  def handle_key(keycode: int) -> None:
    key = chr(keycode).upper() if 0 <= keycode < 256 else ""
    delta = args.speed_step
    if key == "W": controls["vx"] = min(2.0, controls["vx"] + delta)
    elif key == "S": controls["vx"] = max(-2.0, controls["vx"] - delta)
    elif key == "A": controls["vy"] = min(0.8, controls["vy"] + delta)
    elif key == "D": controls["vy"] = max(-0.8, controls["vy"] - delta)
    elif key == "Q": controls["yaw"] = min(1.0, controls["yaw"] + delta)
    elif key == "E": controls["yaw"] = max(-1.0, controls["yaw"] - delta)
    elif key in ("X", " "):
      controls.update(vx=0.0, vy=0.0, yaw=0.0)
    elif key == "R": controls["reset"] = True
    print(
        f"command vx={controls['vx']:+.1f} vy={controls['vy']:+.1f} "
        f"yaw={controls['yaw']:+.1f} rad/s"
    )

  rng = jax.random.key(0)
  state = reset(rng)

  # Compile before opening the window so the first key press is responsive.
  rng, warmup_rng = jax.random.split(rng)
  warmup_state = control_tick(state, warmup_rng, jnp.zeros(3))
  jax.block_until_ready(warmup_state.data.qpos)
  if args.scan == "on":
    warmup_points, _ = scan_points_fn(warmup_state.data)
    warmup_offsets = scan_ego_offsets_fn(warmup_state.data)
    jax.block_until_ready(warmup_points)
    jax.block_until_ready(warmup_offsets)
  state = reset(rng)

  scene_option = mujoco.MjvOption()
  scene_option.geomgroup[2] = True
  scene_option.geomgroup[3] = False
  scene_option.geomgroup[4] = True
  scene_option.geomgroup[5] = True
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = args.contacts
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
  renderer = mujoco.Renderer(
      env.mj_model, height=args.height, width=args.width
  )
  render_data = mujoco.MjData(env.mj_model)

  print("W/S forward/back | A/D left/right | Q/E yaw | X/Space stop | R reset")
  print("Click the Remote Control window before using the keys; Esc closes it.")
  cv2.namedWindow("Level C Remote Control", cv2.WINDOW_NORMAL)
  if args.fullscreen:
    cv2.setWindowProperty(
        "Level C Remote Control", cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )
  control_step = 0
  try:
    while True:
      started = time.perf_counter()
      if controls["reset"] or bool(state.done):
        rng, reset_rng = jax.random.split(rng)
        state = reset(reset_rng)
        controls["reset"] = False

      rng, action_rng = jax.random.split(rng)
      command = jnp.asarray(
          [controls["vx"], controls["vy"], controls["yaw"]],
          dtype=jnp.float32,
      )
      state = control_tick(state, action_rng, command)
      control_step += 1

      if control_step % args.render_every:
        time.sleep(max(0.0, env.dt - (time.perf_counter() - started)))
        continue

      modify_scene = None
      scan_values = None
      display_offsets = None
      if args.scan == "on":
        points, heights = scan_points_fn(state.data)
        display_offsets = np.asarray(scan_ego_offsets_fn(state.data))
        points = np.asarray(points).copy()
        points[:, 2] = np.asarray(heights)
        scan_values = np.asarray(state.obs["state"][-env.scan_size:])
        modify_scene = functools.partial(
            _draw_perception_grid, points=points, values=scan_values
        )

      # This is the notebook's env.render path, kept alive across frames to
      # avoid constructing a new OpenGL renderer on every control step.
      render_data.qpos[:] = np.asarray(state.data.qpos)
      render_data.qvel[:] = np.asarray(state.data.qvel)
      render_data.mocap_pos[:] = np.asarray(state.data.mocap_pos)
      render_data.mocap_quat[:] = np.asarray(state.data.mocap_quat)
      render_data.xfrc_applied[:] = np.asarray(state.data.xfrc_applied)
      mujoco.mj_forward(env.mj_model, render_data)
      renderer.update_scene(render_data, camera="track", scene_option=scene_option)
      if modify_scene:
        modify_scene(renderer.scene)
      frame = renderer.render()
      frame = _overlay_scan_map(frame, scan_values, display_offsets)
      cv2.putText(
          frame,
          f"vx {controls['vx']:+.1f}  vy {controls['vy']:+.1f}  yaw {controls['yaw']:+.1f}",
          (116, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1,
          cv2.LINE_AA,
      )
      cv2.imshow("Level C Remote Control", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
      remaining_ms = max(1, round(1000 * (env.dt - (time.perf_counter() - started))))
      keycode = cv2.waitKey(remaining_ms) & 0xFF
      if keycode == 27:
        break
      if keycode != 255:
        handle_key(keycode)
  finally:
    renderer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
