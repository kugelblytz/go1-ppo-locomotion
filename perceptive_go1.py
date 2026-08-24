"""Local perceptive Go1 task using upstream MuJoCo Playground and MJX-Warp."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import xml.etree.ElementTree as et

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import go1_constants as consts
from mujoco_playground._src.locomotion.go1 import joystick
from terrain_samples import (
    TerrainSampleConstructor,
    resolve_sample_pattern,
)


HEIGHT_SCALE = 0.5
MAP_MIN = -10.5
MAP_MAX = 10.5
MAP_RESOLUTION = 0.05
# Go1 uses group 2 for visual geoms and group 3 for collision geoms. Keep
# terrain in an exclusive group so offline ray rasterization cannot see robot.
TERRAIN_GEOM_GROUP = 4
_ROBOT_XML = "go1_mjx_feetonly.xml"


def feet_air_time_reward(
    air_time: jax.Array,
    first_contact: jax.Array,
    command: jax.Array,
    threshold: float = 0.5,
) -> jax.Array:
  """Legged Gym/Isaac Lab touchdown reward, independent of terrain type."""
  return jp.sum((air_time - threshold) * first_contact) * (
      jp.linalg.norm(command[:2]) > 0.1
  )


def completed_swing_height_reward(
    swing_peak: jax.Array,
    air_time: jax.Array,
    first_contact: jax.Array,
    command: jax.Array,
    target_height: float,
    min_air_time: float,
) -> jax.Array:
  """Rewards a useful terrain-relative foot lift once, on touchdown."""
  completed = first_contact & (air_time >= min_air_time)
  height_score = jp.clip(swing_peak / target_height, 0.0, 1.0)
  return jp.sum(height_score * completed) * (
      jp.linalg.norm(command[:2]) > 0.1
  )


def default_config() -> config_dict.ConfigDict:
  config = joystick.default_config()
  config.impl = "warp"
  config.naconmax = 8 * 8192
  config.njmax = 64
  config.terrain_type = "boxes"
  # Keep the policy-to-joint-target interface fixed across training and play.
  config.action_scale = 0.8
  config.perception = config_dict.create(
      use_height_scan=True,
      noise_std=0.01,
      height_scale=HEIGHT_SCALE,
      target_base_height=0.30,
  )
  # Preserve upstream Go1 reward defaults. The local task adds several
  # optional terms, which start disabled until explicitly tuned.
  scales = config.reward_config.scales
  # Local feet_clearance is a positive completed-swing reward, not a cost.
  scales.feet_clearance = 0.1
  # Absolute world-space foot height is invalid on raised terrain.
  scales.feet_height = 0.0
  # The local long-swing touchdown reward is optional and starts disabled.
  scales.feet_air_time = 0.0
  scales.command_progress = 0.0
  scales.dof_acc = 0.0
  scales.stall = 0.0
  scales.foot_contact = 0.0
  scales.base_height = 0.0
  config.reward_config.min_swing_air_time = 0.08
  return config


def _build_terrain(scene_xml: str, terrain_type: str) -> str:
  """Builds a box, heightfield, or combined static terrain scene."""
  if terrain_type not in ("boxes", "heightfield", "combined"):
    raise ValueError(
        f"Unknown terrain_type={terrain_type!r}; expected boxes, heightfield, "
        "or combined."
    )
  scene = et.fromstring(scene_xml)
  include = scene.find(f"./include[@file='{_ROBOT_XML}']")
  if include is None:
    raise ValueError("Could not find Go1 include in rough-terrain scene.")

  # Upstream provides a shallow static heightfield.  Static model geometry is
  # intentional: MJX-Warp can batch Data/worlds efficiently without batching
  # or rebuilding Model objects between resets.
  asset = scene.find("./asset")
  worldbody = scene.find("./worldbody")
  if asset is None or worldbody is None:
    raise ValueError("Unexpected upstream rough-terrain scene structure.")
  if terrain_type == "boxes":
    asset.clear()
    et.SubElement(
        asset, "texture", type="2d", name="groundplane", builtin="checker",
        mark="edge", rgb1="1 1 1", rgb2="1 1 1", markrgb="0 0 0",
        width="300", height="300",
    )
    et.SubElement(
        asset, "material", name="groundplane", texture="groundplane",
        texuniform="true", texrepeat="5 5", reflectance="0",
    )
    worldbody.clear()
    et.SubElement(
        worldbody, "geom", name="floor", size="0 0 0.01", type="plane",
        material="groundplane", contype="1", conaffinity="0", priority="1",
        friction="0.6", condim="3", group=str(TERRAIN_GEOM_GROUP),
    )
  else:
    floor = worldbody.find("./geom[@name='floor']")
    if floor is None:
      raise ValueError("Could not find upstream heightfield floor geom.")
    floor.set("group", str(TERRAIN_GEOM_GROUP))

  if terrain_type == "heightfield":
    return et.tostring(scene, encoding="unicode")

  walls = (
      # MuJoCo box z position and half-size are equal, so the final values
      # below produce obstacle heights of 10, 8, 12, 10, 14, and 12 cm.
      ("wall1", "1.5 0 0.05", "0.3 10.0 0.05"),
      ("wall2", "-1.5 0 0.04", "0.3 10.0 0.04"),
      ("wall3", "0 1.5 0.06", "10.0 0.3 0.06"),
      ("wall4", "0 -1.5 0.05", "10.0 0.3 0.05"),
      ("wall5", "3.5 0 0.07", "0.2 8.0 0.07"),
      ("wall6", "-3.5 0 0.06", "0.2 8.0 0.06"),
  )
  for name, pos, size in walls:
    body = et.SubElement(worldbody, "body", name=name, pos=pos)
    et.SubElement(
        body,
        "geom",
        contype="1",
        conaffinity="0",
        friction="1 0.005 0.0001",
        rgba="0.6 0.6 0.6 1",
        type="box",
        size=size,
        group=str(TERRAIN_GEOM_GROUP),
    )
  return et.tostring(scene, encoding="unicode")


def _portable_robot_xml(robot_xml: bytes) -> bytes:
  """Makes bundled Go1 assets independent of a Menagerie checkout path."""
  robot = et.fromstring(robot_xml)
  compiler = robot.find("./compiler")
  if compiler is not None:
    compiler.set("meshdir", "")
  # Playground's asset dictionary is keyed by basename (for example,
  # ``trunk.stl``), while the packaged XML retains source-tree-relative paths.
  # MuJoCo otherwise tries to open those paths from the local filesystem.
  for asset in robot.findall(".//*[@file]"):
    filename = asset.get("file")
    if filename:
      asset.set("file", filename.replace("\\", "/").rsplit("/", 1)[-1])
  return et.tostring(robot, encoding="utf-8")


def _rasterize_terrain(model: mujoco.MjModel) -> np.ndarray:
  """Rasterizes visible terrain-group geometry once with CPU MuJoCo rays."""
  size = int(round((MAP_MAX - MAP_MIN) / MAP_RESOLUTION)) + 1
  coordinates = MAP_MIN + np.arange(size) * MAP_RESOLUTION
  heights = np.zeros((size, size), dtype=np.float32)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  geomgroup = np.zeros(6, dtype=np.uint8)
  geomgroup[TERRAIN_GEOM_GROUP] = 1
  direction = np.array([0.0, 0.0, -1.0])
  origin = np.array([0.0, 0.0, 2.0])
  geomid = np.empty(1, dtype=np.int32)
  for row, y in enumerate(coordinates):
    origin[1] = y
    for col, x in enumerate(coordinates):
      origin[0] = x
      distance = mujoco.mj_ray(
          model, data, origin, direction, geomgroup, True, -1, geomid
      )
      heights[row, col] = 0.0 if distance < 0 else origin[2] - distance
  return heights


class PerceptiveJoystick(joystick.Joystick):
  """Upstream rough-terrain task with an optional actor height scan."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      terrain_sample_constructor: TerrainSampleConstructor | None = None,
      terrain_sample_offsets: Any = None,
  ):
    # This mirrors upstream Go1Env construction, replacing only the in-memory
    # MJCF. No installed package or external repository is modified.
    if config.terrain_type != "boxes":
      # Heightfield narrowphase produces more simultaneous contacts than the
      # small box course.  These values avoid silent contact/constraint loss
      # at the notebook's 8,192-world training batch.
      config.naconmax = max(config.naconmax, 8 * 8192)
      config.njmax = max(config.njmax, 80)
    mjx_env.MjxEnv.__init__(self, config, config_overrides)
    # Unlike registry-created environments, this local subclass constructs the
    # Go1 directly. Ensure Playground's version-pinned MuJoCo Menagerie checkout
    # exists before collecting its mesh bytes. This is a one-time download for
    # a clean virtual environment and requires no custom fork.
    mjx_env.ensure_menagerie_exists()
    self._model_assets = go1_base.get_assets()
    self._model_assets[_ROBOT_XML] = _portable_robot_xml(
        self._model_assets[_ROBOT_XML]
    )
    # Upstream's foot sensors explicitly match geom2="floor" and therefore
    # miss raised terrain. A one-geom MuJoCo contact sensor matches contact
    # with any compatible world geom and is supported by MJX-Warp.
    self._model_assets["sensor_feet.xml"] = self._model_assets[
        "sensor_feet.xml"
    ].replace(b' geom2="floor"', b"")
    source_path = consts.FEET_ONLY_ROUGH_TERRAIN_XML
    xml = _build_terrain(source_path.read_text(), self._config.terrain_type)
    self._mj_model = mujoco.MjModel.from_xml_string(
        xml, assets=self._model_assets
    )
    self._mj_model.opt.timestep = self._config.sim_dt
    self._mj_model.dof_damping[6:] = self._config.Kd
    self._mj_model.actuator_gainprm[:, 0] = self._config.Kp
    self._mj_model.actuator_biasprm[:, 1] = -self._config.Kp
    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160
    self._terrain_map = jp.asarray(_rasterize_terrain(self._mj_model))
    resolved_pattern = resolve_sample_pattern(
        constructor=terrain_sample_constructor,
        offsets=terrain_sample_offsets,
    )
    self._sample_pattern = {
        anchor: jp.asarray(points, dtype=jp.float32)
        for anchor, points in resolved_pattern.items()
    }
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = str(source_path)
    self._imu_site_id = self._mj_model.site("imu").id
    self._feet_floor_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in consts.FEET_GEOMS
    ]
    self._post_init()
    # legged_gym shrinks limits about the interval midpoint.  Multiplying
    # each endpoint (Playground's joystick implementation) differs for every
    # asymmetric joint range.
    joint_mid = 0.5 * (self._lowers + self._uppers)
    joint_range = self._uppers - self._lowers
    soft_factor = self._config.soft_joint_pos_limit_factor
    self._soft_lowers = joint_mid - 0.5 * joint_range * soft_factor
    self._soft_uppers = joint_mid + 0.5 * joint_range * soft_factor

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Uses the ordinary randomized reset with extra reward state."""
    state = super().reset(rng)
    state.info["last_dof_vel"] = state.data.qvel[6:]
    # Terrain-relative per-foot peaks, reset after each completed swing.
    state.info["swing_peak"] = jp.zeros(4)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Steps physics with legged_gym-equivalent reward/contact bookkeeping."""
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)

    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    contact = self._feet_terrain_contact(data)
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) & contact_filt
    state.info["feet_air_time"] += self.dt
    foot_clearance = (
        data.site_xpos[self._feet_site_id, -1]
        - self._terrain_height_below_feet(data)
    )
    state.info["swing_peak"] = jp.where(
        ~contact,
        jp.maximum(state.info["swing_peak"], foot_clearance),
        state.info["swing_peak"],
    )

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)
    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        name: value * self._config.reward_config.scales[name]
        for name, value in rewards.items()
    }
    # Isaac Lab sums signed reward terms and scales by control dt. Keeping
    # negative outcomes visible avoids a zero-gradient clipped region.
    reward = sum(rewards.values()) * self.dt

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["last_dof_vel"] = data.qvel[6:]
    state.info["steps_until_next_cmd"] = jp.asarray(1_000_000, dtype=jp.int32)
    state.info["feet_air_time"] *= ~contact_filt
    state.info["last_contact"] = contact
    # No per-step payout: touchdown pays once and clears only that foot.
    state.info["swing_peak"] *= ~first_contact
    for name, value in rewards.items():
      state.metrics[f"reward/{name}"] = value
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    return state.replace(
        data=data, obs=obs, reward=reward, done=done.astype(reward.dtype)
    )

  def _sample_terrain(self, xy: jax.Array) -> jax.Array:
    """Bilinearly samples world-space XY points from the elevation map."""
    grid = (xy - MAP_MIN) / MAP_RESOLUTION
    max_index = self._terrain_map.shape[0] - 1
    grid = jp.clip(grid, 0.0, float(max_index))
    lower = jp.floor(grid).astype(jp.int32)
    upper = jp.minimum(lower + 1, max_index)
    fraction = grid - lower

    x0, y0 = lower[..., 0], lower[..., 1]
    x1, y1 = upper[..., 0], upper[..., 1]
    fx, fy = fraction[..., 0], fraction[..., 1]
    h00 = self._terrain_map[y0, x0]
    h10 = self._terrain_map[y0, x1]
    h01 = self._terrain_map[y1, x0]
    h11 = self._terrain_map[y1, x1]
    return (
        h00 * (1.0 - fx) * (1.0 - fy)
        + h10 * fx * (1.0 - fy)
        + h01 * (1.0 - fx) * fy
        + h11 * fx * fy
    )

  def _feet_terrain_contact(self, data: mjx.Data) -> jax.Array:
    """Returns one contact bit per foot for any compatible world geom."""
    return jp.asarray([
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._feet_floor_found_sensor
    ])

  @property
  def terrain_map(self) -> jax.Array:
    """Elevation map used by policy observations and clearance rewards."""
    return self._terrain_map

  def scan_points(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
    """Returns world-space scan points and sampled terrain heights."""
    rotation = self._yaw_rotation(data)
    yaw_xy = rotation[:2, :2]
    foot_xy = data.site_xpos[self._feet_site_id, :2]
    anchored_points = []
    foot_index = {name: index for index, name in enumerate(consts.FEET_SITES)}
    for anchor, offsets in self._sample_pattern.items():
      origin_xy = (
          data.xpos[self._torso_body_id, :2]
          if anchor == "body"
          else foot_xy[foot_index[anchor]]
      )
      xy = origin_xy + offsets @ yaw_xy.T
      anchored_points.append(jp.column_stack((xy, jp.zeros(xy.shape[0]))))
    points = jp.concatenate(anchored_points, axis=0)
    heights = self._sample_terrain(points[:, :2])
    return points, heights

  @property
  def scan_size(self) -> int:
    return sum(points.shape[0] for points in self._sample_pattern.values())

  def scan_ego_offsets(self, data: mjx.Data) -> jax.Array:
    """Returns the samples' current positions in the torso-yaw frame."""
    points, _ = self.scan_points(data)
    world_delta = points[:, :2] - data.xpos[self._torso_body_id, :2]
    return world_delta @ self._yaw_rotation(data)[:2, :2]

  @property
  def sample_pattern(self) -> dict[str, jax.Array]:
    """Resolved anchor-to-offset mapping in stable policy observation order."""
    return self._sample_pattern

  def _yaw_rotation(self, data: mjx.Data) -> jax.Array:
    """Returns a level local-to-world rotation using torso yaw only."""
    w, x, y, z = data.xquat[self._torso_body_id]
    cosine = 1.0 - 2.0 * (y * y + z * z)
    sine = 2.0 * (w * z + x * y)
    return jp.asarray([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])

  def _terrain_height_below_feet(self, data: mjx.Data) -> jax.Array:
    """Returns terrain Z directly below each foot."""
    return self._sample_terrain(data.site_xpos[self._feet_site_id, :2])

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    rewards = super()._get_reward(
        data, action, info, metrics, done, first_contact, contact
    )
    # Reproduce legged_gym's active A1 rough-terrain reward formulas.  The
    # similarly named Playground terms are not all equivalent: its vertical
    # and roll/pitch velocity costs use world-frame velocities, and its torque
    # regularizer is sqrt(L2) + L1 rather than sum(torque**2).
    imu_global_linvel = self.get_global_linvel(data)
    global_angvel = self.get_global_angvel(data)
    imu_offset = (
        data.site_xpos[self._imu_site_id] - data.xpos[self._torso_body_id]
    )
    root_global_linvel = imu_global_linvel - jp.cross(
        global_angvel, imu_offset
    )
    local_linvel = (
        data.xmat[self._torso_body_id].T @ root_global_linvel
    )
    local_angvel = self.get_gyro(data)
    # Rewards matching the commanded forward and sideways velocity.
    rewards["tracking_lin_vel"] = jp.exp(
        -jp.sum(jp.square(info["command"][:2] - local_linvel[:2]))
        / self._config.reward_config.tracking_sigma
    )
    # Rewards matching the commanded yaw rate.
    rewards["tracking_ang_vel"] = jp.exp(
        -jp.square(info["command"][2] - local_angvel[2])
        / self._config.reward_config.tracking_sigma
    )
    # Penalizes bouncing up and down.
    rewards["lin_vel_z"] = jp.square(local_linvel[2])
    # Penalizes rolling and pitching motion.
    rewards["ang_vel_xy"] = jp.sum(jp.square(local_angvel[:2]))
    # Penalizes large motor torques.
    rewards["torques"] = jp.sum(jp.square(data.actuator_force))
    # Penalizes abrupt changes between consecutive actions.
    rewards["action_rate"] = jp.sum(jp.square(action - info["last_act"]))
    # Penalizes rapid changes in joint velocity.
    rewards["dof_acc"] = jp.sum(
        jp.square((data.qvel[6:] - info["last_dof_vel"]) / self.dt)
    )
    # Rewards reaching useful terrain-relative foot height during each swing.
    rewards["feet_clearance"] = completed_swing_height_reward(
        info["swing_peak"],
        info["feet_air_time"],
        first_contact,
        info["command"],
        self._config.reward_config.max_foot_height,
        self._config.reward_config.min_swing_air_time,
    )
    # Disables absolute foot-height rewards, which are invalid on raised terrain.
    rewards["feet_height"] = jp.zeros(())
    command_speed = jp.linalg.norm(info["command"][:2])
    # Rewards sufficiently long swing phases once when a foot touches down.
    rewards["feet_air_time"] = feet_air_time_reward(
        info["feet_air_time"], first_contact, info["command"]
    )
    actual_speed = jp.linalg.norm(self.get_local_linvel(data)[:2])
    # Penalizes failing to move while a movement command is active.
    rewards["stall"] = (
        (command_speed > 0.15) & (actual_speed < 0.3 * command_speed)
    ).astype(jp.float32)
    # Rewards maintaining contact with at least two feet.
    rewards["foot_contact"] = jp.minimum(
        jp.sum(contact.astype(jp.float32)) / 2.0, 1.0
    )
    command_xy = info["command"][:2]
    command_norm = jp.linalg.norm(command_xy)
    command_direction = command_xy / jp.maximum(command_norm, 0.2)
    # Rewards velocity in the commanded local direction.
    rewards["command_progress"] = jp.clip(
        jp.dot(command_direction, self.get_local_linvel(data)[:2]),
        -1.0,
        1.0,
    )
    terrain_z = self._sample_terrain(data.xpos[self._torso_body_id, :2])
    relative_base_height = data.xpos[self._torso_body_id, 2] - terrain_z
    # Penalizes deviation from the desired terrain-relative body height.
    rewards["base_height"] = jp.square(
        relative_base_height - self._config.perception.target_base_height
    )
    return rewards

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    """Rejects upside-down and collapsed gaits while allowing real climbing."""
    terrain_z = self._sample_terrain(data.xpos[self._torso_body_id, :2])
    relative_height = data.xpos[self._torso_body_id, 2] - terrain_z
    tipped = self.get_upvector(data)[-1] < 0.5
    collapsed = relative_height < 0.18
    return tipped | collapsed

  def _height_scan(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> jax.Array:
    _, height = self.scan_points(data)

    if self._config.perception.noise_std > 0:
      info["rng"], noise_rng = jax.random.split(info["rng"])
      height += jax.random.normal(noise_rng, height.shape) * (
          self._config.perception.noise_std
      )

    center_height = self._sample_terrain(
        data.xpos[self._torso_body_id, :2]
    )
    scale = self._config.perception.height_scale
    return jp.clip((height - center_height) / scale, -1.0, 1.0)

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
    obs = super()._get_obs(data, info)
    if not self._config.perception.use_height_scan:
      return obs
    scan = self._height_scan(data, info)
    return {
        "state": jp.hstack((obs["state"], scan)),
        "privileged_state": jp.hstack((obs["privileged_state"], scan)),
    }


def make_env(
    use_height_scan: bool,
    terrain_type: str = "boxes",
    terrain_sample_constructor: TerrainSampleConstructor | None = None,
    terrain_sample_offsets: Any = None,
    **config_overrides: Any,
):
  config = default_config()
  config.perception.use_height_scan = use_height_scan
  config.terrain_type = terrain_type
  return PerceptiveJoystick(
      config=config,
      config_overrides=config_overrides,
      terrain_sample_constructor=terrain_sample_constructor,
      terrain_sample_offsets=terrain_sample_offsets,
  )
