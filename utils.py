import matplotlib.pyplot as plt
import jax
import mujoco
import numpy as np
import jax.numpy as jp
import functools
import cv2
import mediapy as media
from mujoco_playground._src.gait import draw_joystick_command


def _draw_perception_grid(scene, points, values):
    """Draw the actor's exact ego-centric terrain observation."""
    max_abs = max(float(np.max(np.abs(values))), 0.1)
    for point, value in zip(points, values):
        if scene.ngeom >= scene.maxgeom:
            return
        # Blue is lower than the local reference; red is higher.
        # Normalize each frame's colors so small but real step differences
        # remain visible.  Geometry height still shows the unmodified sample.
        normalized = float(np.clip(0.5 * (value / max_abs + 1.0), 0.0, 1.0))
        rgba = np.array(
            [normalized, 0.2, 1.0 - normalized, 0.55], dtype=np.float32
        )
        marker_pos = np.asarray(point, dtype=np.float64).copy()
        marker_pos[2] += 0.018
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.006, 0.0, 0.0], dtype=np.float64),
            marker_pos,
            np.eye(3).reshape(-1),
            rgba,
        )
        scene.ngeom += 1


def _draw_training_overlay(scene, command_draw_fn, scan_points, scan_values):
    command_draw_fn(scene)
    if scan_points is not None:
        _draw_perception_grid(scene, scan_points, scan_values)


def _interpolate_scan_rectangle(values, offsets, size=90):
    """Visualizes arbitrary samples on a rectangle without changing policy input."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    offsets = np.asarray(offsets, dtype=np.float32)
    if offsets.shape != (values.size, 2):
        raise ValueError(
            f"Expected one (x, y) offset per value; got {offsets.shape} and "
            f"{values.shape}."
        )

    unique_x = np.unique(offsets[:, 0])
    unique_y = np.unique(offsets[:, 1])
    if unique_x.size * unique_y.size == values.size:
        # Preserve crisp cells for complete Cartesian grids, including the
        # provided 17 x 11 baseline.
        raster = np.empty((unique_y.size, unique_x.size), dtype=np.float32)
        x_index = {float(x): i for i, x in enumerate(unique_x)}
        y_index = {float(y): i for i, y in enumerate(unique_y[::-1])}
        for (x, y), value in zip(offsets, values):
            raster[y_index[float(y)], x_index[float(x)]] = value
        return cv2.resize(
            raster, (size, size), interpolation=cv2.INTER_NEAREST
        ), offsets

    # Four-neighbour inverse-distance interpolation for arbitrary patterns.
    # This affects only the display; the policy still receives `values`.
    padding = 0.05
    x_min, x_max = float(offsets[:, 0].min()), float(offsets[:, 0].max())
    y_min, y_max = float(offsets[:, 1].min()), float(offsets[:, 1].max())
    if x_min == x_max:
        x_min, x_max = x_min - padding, x_max + padding
    if y_min == y_max:
        y_min, y_max = y_min - padding, y_max + padding
    rows_y = np.linspace(y_max, y_min, size, dtype=np.float32)
    cols_x = np.linspace(x_min, x_max, size, dtype=np.float32)
    query_y, query_x = np.meshgrid(rows_y, cols_x, indexing="ij")
    query = np.stack((query_x, query_y), axis=-1)
    distance_sq = np.sum(
        (query[:, :, None, :] - offsets[None, None, :, :]) ** 2, axis=-1
    )
    k = min(4, values.size)
    nearest = np.argpartition(distance_sq, k - 1, axis=-1)[..., :k]
    nearest_dist = np.take_along_axis(distance_sq, nearest, axis=-1)
    nearest_values = values[nearest]
    weights = 1.0 / np.maximum(nearest_dist, 1e-8)
    raster = np.sum(weights * nearest_values, axis=-1) / np.sum(weights, axis=-1)
    return raster, offsets


def _overlay_scan_map(frame, values, offsets=None):
    """Add a compact 2D view of the exact actor terrain input."""
    if values is None:
        return frame
    values = np.asarray(values)
    # A 0.1 normalized value is 5 cm with the default observation scale;
    # using it as a floor prevents sensor noise on flat ground being painted
    # as a full-height red/blue feature.
    max_abs = max(float(np.max(np.abs(values))), 0.1)
    color_values = np.clip(values / max_abs, -1.0, 1.0)
    if offsets is not None:
        values_2d, marker_offsets = _interpolate_scan_rectangle(
            color_values, offsets
        )
        t = np.clip(0.5 * (values_2d + 1.0), 0.0, 1.0)[..., None]
        low = np.array([35, 90, 240], dtype=np.float32)
        high = np.array([240, 65, 35], dtype=np.float32)
        rgb = (low * (1.0 - t) + high * t).astype(np.uint8)
        grid = cv2.resize(rgb, (90, 90), interpolation=cv2.INTER_NEAREST)
        x_min, x_max = marker_offsets[:, 0].min(), marker_offsets[:, 0].max()
        y_min, y_max = marker_offsets[:, 1].min(), marker_offsets[:, 1].max()
        x_span = max(float(x_max - x_min), 1e-6)
        y_span = max(float(y_max - y_min), 1e-6)
        for x, y in marker_offsets:
            px = int(round(89 * (x - x_min) / x_span))
            py = int(round(89 * (y_max - y) / y_span))
            cv2.circle(grid, (px, py), 1, (25, 25, 25), -1)
    elif values.size == 187:
        values_2d = color_values.reshape(17, 11)[::-1]
        t = np.clip(0.5 * (values_2d + 1.0), 0.0, 1.0)[..., None]
        low = np.array([35, 90, 240], dtype=np.float32)
        high = np.array([240, 65, 35], dtype=np.float32)
        rgb = (low * (1.0 - t) + high * t).astype(np.uint8)
        grid = cv2.resize(rgb, (90, 90), interpolation=cv2.INTER_NEAREST)
    elif values.size != 25:
        # Four foot-centred radial patterns: FL, FR, RL, RR.  The centre,
        # inner cardinal ring, and outer 8-point ring retain spatial layout.
        foot_values = color_values[-52:].reshape(4, 13)
        canvas = np.full((90, 90, 3), 245, dtype=np.uint8)
        centers = ((65, 24), (65, 66), (25, 24), (25, 66))
        angles = np.arange(8) * np.pi / 4.0
        offsets = [(0, 0)]
        offsets += [(7 * np.cos(a), 7 * np.sin(a)) for a in angles[::2]]
        offsets += [(14 * np.cos(a), 14 * np.sin(a)) for a in angles]
        for center, samples in zip(centers, foot_values):
            for (dx, dy), value in zip(offsets, samples):
                t = float(np.clip(0.5 * (value + 1.0), 0.0, 1.0))
                color = tuple(int(x) for x in (
                    np.array([35, 90, 240]) * (1.0 - t)
                    + np.array([240, 65, 35]) * t
                ))
                cv2.circle(canvas, (int(center[0] + dx), int(center[1] + dy)), 2, color, -1)
        grid = canvas
    else:
        values = color_values.reshape(5, 5)
        t = np.clip(0.5 * (values + 1.0), 0.0, 1.0)[..., None]
        low = np.array([35, 90, 240], dtype=np.float32)
        high = np.array([240, 65, 35], dtype=np.float32)
        rgb = (low * (1.0 - t) + high * t).astype(np.uint8)
        grid = cv2.resize(rgb, (90, 90), interpolation=cv2.INTER_NEAREST)
        for coordinate in range(0, 91, 18):
            cv2.line(grid, (coordinate, 0), (coordinate, 89), (25, 25, 25), 1)
            cv2.line(grid, (0, coordinate), (89, coordinate), (25, 25, 25), 1)

    panel = np.full((118, 100, 3), 245, dtype=np.uint8)
    panel[24:114, 5:95] = grid
    cv2.putText(
        panel,
        "terrain obs",
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    result = frame.copy()
    result[8:126, 8:108] = panel
    return result

def render_video_during_training(current_policy, step_num, jit_step,jit_reset, env_cfg, eval_env_for_video):
    """Render a video using the current policy during training"""
    try:
        # The constructed environment owns the validated, effective config.
        # Do not depend on a mutable notebook-global copy during callbacks.
        env_cfg = eval_env_for_video._config
        jit_inference_fn = jax.jit(current_policy)
        from mujoco_playground._src.gait import draw_joystick_command
        
        # Use a simple forward command for training videos
        command = jp.array([1.0, 0.0, 0.0])  # Move forward
        
        rng = jax.random.PRNGKey(42)  # Fixed seed for consistency
        rollout = []
        modify_scene_fns = []
        scan_history = []
        scan_offset_history = []
        reward_history = []
        
        state = jit_reset(rng)
        state.info["command"] = command
        
        rollout_length = min(160, env_cfg.episode_length)
        
        for _ in range(rollout_length):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            state.info["command"] = command
            rollout.append(state)
            
            raw_rewards = {k[7:]: v for k, v in state.metrics.items() if k.startswith("reward/")}
            scaled_rewards = {}
            for k, v in raw_rewards.items():
                scaled_rewards[k] = v
            reward_history.append(scaled_rewards)
            
            # Add visualization elements
            xyz = np.array(state.data.xpos[eval_env_for_video._torso_body_id])
            xyz += np.array([0, 0, 0.2])
            x_axis = state.data.xmat[eval_env_for_video._torso_body_id, 0]
            yaw = -np.arctan2(x_axis[1], x_axis[0])
            command_draw_fn = functools.partial(
                draw_joystick_command,
                cmd=state.info["command"],
                xyz=xyz,
                theta=yaw,
                scl=abs(state.info["command"][0]) / env_cfg.command_config.a[0],
            )
            scan_points = None
            scan_values = None
            use_height_scan = (
                "perception" in env_cfg
                and env_cfg.perception.use_height_scan
                and hasattr(eval_env_for_video, "scan_points")
            )
            if use_height_scan:
                points, heights = eval_env_for_video.scan_points(state.data)
                scan_points = np.asarray(points).copy()
                scan_points[:, 2] = np.asarray(heights)
                # These are the exact noisy, normalized values seen by actor.
                scan_values = np.asarray(
                    state.obs["state"][-eval_env_for_video.scan_size:]
                )
                scan_offsets = np.asarray(
                    eval_env_for_video.scan_ego_offsets(state.data)
                )
            else:
                scan_offsets = None
            scan_history.append(scan_values)
            scan_offset_history.append(scan_offsets)
            modify_scene_fns.append(
                functools.partial(
                    _draw_training_overlay,
                    command_draw_fn=command_draw_fn,
                    scan_points=scan_points,
                    scan_values=scan_values,
                )
            )

        render_every = 2
        fps = 1.0 / eval_env_for_video.dt / render_every
        traj = rollout[::render_every]
        mod_fns = modify_scene_fns[::render_every]

        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[2] = True
        scene_option.geomgroup[3] = False
        scene_option.geomgroup[4] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

        frames = eval_env_for_video.render(
            traj,
            camera="track",
            scene_option=scene_option,
            width=640,
            height=480,
            modify_scene_fns=mod_fns,
        )
        
        # Get reward component keys from first frame
        reward_keys = set()
        if reward_history:
            reward_keys = set(reward_history[0].keys())
        
        # Calculate global min/max values for consistent y-axis scaling
        def calculate_global_limits(reward_history, reward_groups, reward_keys):
            """Calculate global min/max for each reward group from all data"""
            group_limits = {}
            
            for group_name, reward_names in reward_groups.items():
                all_values = []
                for reward_name in reward_names:
                    if reward_name in reward_keys:
                        values = [reward_history[i].get(reward_name, 0.0) for i in range(len(reward_history))]
                        all_values.extend(values)
                
                if all_values:
                    min_val = min(all_values)
                    max_val = max(all_values)
                    # Add some padding (10% of range)
                    range_val = max_val - min_val
                    padding = max(0.1 * range_val, 0.1)  # At least 0.1 padding
                    group_limits[group_name] = (min_val - padding, max_val + padding)
                else:
                    group_limits[group_name] = (-1, 1)  # Default range
            
            return group_limits
        
        # Define reward groups
        reward_groups = {
            'tracking': ['tracking_lin_vel', 'tracking_ang_vel'],
            'base': ['orientation', 'lin_vel_z', 'ang_vel_xy', 'pose', 'stand_still', 'torso_height'],
            'feet': ['feet_air_time', 'feet_clearance', 'feet_slip'],
            'energy': ['torques', 'action_rate', 'energy', 'termination', 'dof_pos_limits']
        }
        
        # Calculate global limits once for all frames
        global_limits = calculate_global_limits(reward_history, reward_groups, reward_keys)
        
        # Create reward plots for each frame
        def create_reward_plot(frame_idx, reward_history, reward_keys):
            """Create animated reward plot for a specific frame"""
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
            fig.suptitle(f'Reward Components - Step {step_num} - Frame {frame_idx}', fontsize=14)
            
            # Prepare data up to current frame
            current_step = min(frame_idx * render_every, len(reward_history) - 1)
            steps = list(range(0, current_step + 1))
            
            if current_step >= 0:
                # Plot 1: Tracking rewards
                for reward_name in reward_groups['tracking']:
                    if reward_name in reward_keys:
                        values = [reward_history[i].get(reward_name, 0.0) for i in range(current_step + 1)]
                        ax1.plot(steps, values, label=reward_name, linewidth=2)
                ax1.set_title('Tracking Rewards')
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(global_limits['tracking'])
                
                # Plot 2: Base stability rewards
                for reward_name in reward_groups['base']:
                    if reward_name in reward_keys:
                        values = [reward_history[i].get(reward_name, 0.0) for i in range(current_step + 1)]
                        ax2.plot(steps, values, label=reward_name, linewidth=2)
                ax2.set_title('Base Stability')
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(global_limits['base'])
                
                # Plot 3: Feet rewards
                for reward_name in reward_groups['feet']:
                    if reward_name in reward_keys:
                        values = [reward_history[i].get(reward_name, 0.0) for i in range(current_step + 1)]
                        ax3.plot(steps, values, label=reward_name, linewidth=2)
                ax3.set_title('Feet Control')
                ax3.legend(fontsize=8)
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim(global_limits['feet'])
                
                # Plot 4: Energy/Action rewards
                for reward_name in reward_groups['energy']:
                    if reward_name in reward_keys:
                        values = [reward_history[i].get(reward_name, 0.0) for i in range(current_step + 1)]
                        ax4.plot(steps, values, label=reward_name, linewidth=2)
                ax4.set_title('Energy & Actions')
                ax4.legend(fontsize=8)
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(global_limits['energy'])
            
            plt.tight_layout()
            
            # Convert plot to image
            fig.canvas.draw()
            # Use buffer_rgba() instead of deprecated tostring_rgb()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            plot_image = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # Convert RGBA to RGB
            plot_image = plot_image[:, :, :3]
            plt.close(fig)
            
            return plot_image
        
        # Create combined frames (video + reward plot side by side)
        combined_frames = []
        for frame_idx, frame in enumerate(frames):
            scan_index = min(frame_idx * render_every, len(scan_history) - 1)
            frame = _overlay_scan_map(
                frame, scan_history[scan_index], scan_offset_history[scan_index]
            )
            # Create reward plot for this frame
            reward_plot = create_reward_plot(frame_idx, reward_history, reward_keys)
            
            # Resize frames to match heights
            video_height = frame.shape[0]
            plot_height, plot_width = reward_plot.shape[:2]
            
            # Scale plot to match video height
            if plot_height != video_height:
                scale_factor = video_height / plot_height
                new_plot_width = int(plot_width * scale_factor)
                reward_plot = cv2.resize(reward_plot, (new_plot_width, video_height))
            
            # Combine frames horizontally
            combined_frame = np.hstack([frame, reward_plot])
            combined_frames.append(combined_frame)
        
        media.show_video(combined_frames, fps=fps)
        
    except Exception as e:
        print(f"Failed to render video at step {step_num}: {e}")


def evaluate_policy(
    env,
    jit_inference_fn,
    jit_step,
    jit_reset,
    env_cfg,
    eval_env,
    velocity_kick_range: tuple = (-1.0, 1.0),
    kick_duration_range: tuple = (0.2, 1.0),
):
    """Evaluate the policy over multiple episodes and return average reward."""
    env_cfg = eval_env._config
    x_vels = [0.0, 0.5, 1.0, 0.0, 0.0, 0.5]
    y_vels = [0.0, 0.0, 0.0, 0.5, 1.0, 0.5]
    yaw_vels = [1.0, 0.0, 0.0, 0.0, 0.0, 0.1]

    for run_id, (x_vel, y_vel, yaw_vel) in enumerate(zip(x_vels, y_vels, yaw_vels)):
        def sample_pert(rng):
            rng, key1, key2 = jax.random.split(rng, 3)
            pert_mag = jax.random.uniform(
                key1, minval=velocity_kick_range[0], maxval=velocity_kick_range[1]
            )
            duration_seconds = jax.random.uniform(
                key2, minval=kick_duration_range[0], maxval=kick_duration_range[1]
            )
            duration_steps = jp.round(duration_seconds / eval_env.dt).astype(jp.int32)
            state.info["pert_mag"] = pert_mag
            state.info["pert_duration"] = duration_steps
            state.info["pert_duration_seconds"] = duration_seconds
            return rng


        rng = jax.random.PRNGKey(0)
        rollout = []
        modify_scene_fns = []
        scan_history = []
        scan_offset_history = []

        swing_peak = []
        rewards = []
        linvel = []
        angvel = []
        track = []
        foot_vel = []
        rews = []
        contact = []
        command = jp.array([x_vel, y_vel, yaw_vel])

        state = jit_reset(rng)
        if state.info["steps_since_last_pert"] < state.info["steps_until_next_pert"]:
            rng = sample_pert(rng)
        state.info["command"] = command
        for _ in range(env_cfg.episode_length):
            if state.info["steps_since_last_pert"] < state.info["steps_until_next_pert"]:
                rng = sample_pert(rng)
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            state.info["command"] = command
            rews.append(
                {k: v for k, v in state.metrics.items() if k.startswith("reward/")}
            )
            rollout.append(state)
            swing_peak.append(state.info["swing_peak"])
            rewards.append(
                {k[7:]: v for k, v in state.metrics.items() if k.startswith("reward/")}
            )
            linvel.append(env.get_global_linvel(state.data))
            angvel.append(env.get_gyro(state.data))
            track.append(
                env._reward_tracking_lin_vel(
                    state.info["command"], env.get_local_linvel(state.data)
                )
            )

            feet_vel = state.data.sensordata[env._foot_linvel_sensor_adr]
            vel_xy = feet_vel[..., :2]
            vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
            foot_vel.append(vel_norm)

            contact.append(state.info["last_contact"])

            xyz = np.array(state.data.xpos[env._torso_body_id])
            xyz += np.array([0, 0, 0.2])
            x_axis = state.data.xmat[env._torso_body_id, 0]
            yaw = -np.arctan2(x_axis[1], x_axis[0])
            command_draw_fn = functools.partial(
                draw_joystick_command,
                cmd=state.info["command"],
                xyz=xyz,
                theta=yaw,
                scl=abs(state.info["command"][0])
                / env_cfg.command_config.a[0],
            )
            scan_points = None
            scan_values = None
            use_height_scan = (
                "perception" in env_cfg
                and env_cfg.perception.use_height_scan
                and hasattr(eval_env, "scan_points")
            )
            if use_height_scan:
                points, heights = eval_env.scan_points(state.data)
                scan_points = np.asarray(points).copy()
                scan_points[:, 2] = np.asarray(heights)
                scan_values = np.asarray(
                    state.obs["state"][-eval_env.scan_size:]
                )
                scan_offsets = np.asarray(eval_env.scan_ego_offsets(state.data))
            else:
                scan_offsets = None
            scan_history.append(scan_values)
            scan_offset_history.append(scan_offsets)
            modify_scene_fns.append(functools.partial(
                _draw_training_overlay,
                command_draw_fn=command_draw_fn,
                scan_points=scan_points,
                scan_values=scan_values,
            ))


        render_every = 2
        fps = 1.0 / eval_env.dt / render_every
        traj = rollout[::render_every]
        mod_fns = modify_scene_fns[::render_every]

        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[2] = True
        scene_option.geomgroup[3] = False
        scene_option.geomgroup[4] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

        frames = eval_env.render(
            traj,
            camera="track",
            scene_option=scene_option,
            width=640,
            height=480,
            modify_scene_fns=mod_fns,
        )
        frames = [
            _overlay_scan_map(
                frame, scan_history[index], scan_offset_history[index]
            )
            for index, frame in zip(range(0, len(scan_history), render_every), frames)
        ]
        media.show_video(frames, fps=fps)
