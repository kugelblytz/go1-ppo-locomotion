"""Student-facing terrain observation design and checkpoint metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


SamplePattern = Mapping[str, np.ndarray]
TerrainSampleConstructor = Callable[[], SamplePattern]
ANCHOR_ORDER = ("body", "FR", "FL", "RR", "RL")
MAX_TERRAIN_SAMPLES = 200


def terrain_sample_pattern() -> SamplePattern:
  """Returns body-anchored ego-frame locations sampled by the policy.

  Students may replace this implementation with any finite, non-empty set of
  points. Positive x is forward and positive y is left. The default exactly
  reproduces the current 17 x 11, 1.6 m x 1.0 m observation grid.
  """
  xs = np.linspace(-0.8, 0.8, 17, dtype=np.float32)
  ys = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
  return {
      "body": np.asarray([(x, y) for x in xs for y in ys], dtype=np.float32)
  }


def resolve_sample_pattern(
    constructor: TerrainSampleConstructor | None = None,
    offsets: SamplePattern | None = None,
) -> dict[str, np.ndarray]:
  """Runs and validates a terrain sample constructor once at environment init."""
  if constructor is not None and offsets is not None:
    raise ValueError("Pass either a constructor or resolved offsets, not both.")
  pattern = terrain_sample_pattern() if constructor is None and offsets is None else (
      constructor() if constructor is not None else offsets
  )
  if not isinstance(pattern, Mapping):
    raise TypeError(
        "Terrain sample pattern must be a dictionary keyed by 'body', 'FL', "
        "'FR', 'RL', or 'RR'."
    )
  unknown = set(pattern) - set(ANCHOR_ORDER)
  if unknown:
    raise ValueError(
        f"Unknown terrain sample anchors {sorted(unknown)}; use {ANCHOR_ORDER}."
    )
  resolved = {}
  for anchor in ANCHOR_ORDER:
    if anchor not in pattern:
      continue
    points = np.asarray(pattern[anchor], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
      raise ValueError(
          f"Samples for {anchor!r} must be shaped (N, 2) or (N, 3); "
          f"got {points.shape}."
      )
    if points.shape[0] and not np.all(np.isfinite(points)):
      raise ValueError(f"Samples for {anchor!r} must be finite.")
    resolved[anchor] = points[:, :2]
  sample_count = sum(len(points) for points in resolved.values())
  if not resolved or sample_count == 0:
    raise ValueError("Terrain sample pattern must contain at least one point.")
  if sample_count > MAX_TERRAIN_SAMPLES:
    raise ValueError(
        f"Terrain observation budget exceeded: {sample_count} samples, "
        f"maximum {MAX_TERRAIN_SAMPLES}."
    )
  return resolved


def perception_metadata_path(checkpoint: str | Path) -> Path:
  path = Path(checkpoint)
  return (path if path.is_dir() else path.parent) / "perception.json"


def save_perception_spec(checkpoint: str | Path, env) -> Path:
  """Saves resolved sensor geometry beside Brax policy parameters."""
  path = perception_metadata_path(checkpoint)
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = {
      "version": 2,
      "coordinate_frame": "robot_yaw",
      "samples": {
          anchor: np.asarray(points).tolist()
          for anchor, points in env.sample_pattern.items()
      },
      "height_scale": float(env._config.perception.height_scale),
      "use_height_scan": bool(env._config.perception.use_height_scan),
      "terrain_type": str(env._config.terrain_type),
  }
  path.write_text(json.dumps(payload, indent=2))
  return path


def load_perception_spec(checkpoint: str | Path) -> dict | None:
  """Loads sensor geometry, returning None for legacy checkpoints."""
  path = perception_metadata_path(checkpoint)
  return json.loads(path.read_text()) if path.exists() else None
