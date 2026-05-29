#!/usr/bin/env python3
"""Play back a motion NPZ file with MuJoCo native viewer.

Supports both 29-DOF and 23-DOF robot models so you can visually compare
before and after conversion.

Usage:
  # Play 23DOF motion
  python scripts/play_motion.py \
    --npz src/assets/motions/g1/23dof_amp/WalkandRun/jog_forward_loop_003__A022.npz

  # Play original 29DOF motion
  python scripts/play_motion.py \
    --npz src/assets/motions/g1/amp/WalkandRun/jog_forward_loop_003__A022.npz

  # Explicit model selection
  python scripts/play_motion.py \
    --npz src/assets/motions/g1/23dof_amp/WalkandRun/jog_forward_loop_003__A022.npz \
    --dof 23 

  # Offscreen mode (no display needed, saves video)
  python scripts/play_motion.py \
    --npz src/assets/motions/g1/amp/WalkandRun/jog_forward_loop_003__A022.npz \
    --headless --video-output /tmp/jog.mp4

"""

import argparse
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer as mj_viewer
import numpy as np

from src.assets.robots.unitree_g1.g1_23dof_constants import (
    G1_23DOF_XML,
    get_g1_23dof_robot_cfg,
)
from src.assets.robots.unitree_g1.g1_constants import (
    G1_XML,
    get_g1_robot_cfg,
)
from mjlab.entity import Entity
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


def load_npz(path: str) -> dict:
    """Load NPZ file and validate required keys."""
    data = np.load(path)
    required = ["fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                 "body_lin_vel_w", "body_ang_vel_w"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"Missing keys: {missing}")
    return data


def setup_entity(dof: int):
    """Create a 23DOF or 29DOF robot entity for simulation."""
    if dof == 23:
        return Entity(get_g1_23dof_robot_cfg())
    return Entity(get_g1_robot_cfg())


def play_motion(
    npz_path: str,
    dof: int = 0,
    headless: bool = False,
    video_output: str | None = None,
    realtime: bool = True,
    realtime_scale: float = 1.0,
    loop: bool = True,
):
    """Play back a motion NPZ file in MuJoCo viewer.

    Args:
        npz_path: Path to NPZ motion file.
        dof: 23 or 29 (auto-detected from body count if 0).
        headless: Run without display (offscreen rendering).
        video_output: Save video to this path (headless mode).
        realtime: Match wall-clock time.
        realtime_scale: Speed multiplier (1.0 = real-time).
        loop: Loop playback.
    """
    motion = load_npz(npz_path)

    fps = int(motion["fps"].flat[0])
    joint_pos = motion["joint_pos"]  # (T, n_joints)
    body_pos_w = motion["body_pos_w"]  # (T, n_bodies, 3)
    body_quat_w = motion["body_quat_w"]  # (T, n_bodies, 4)
    body_lin_vel_w = motion["body_lin_vel_w"]
    body_ang_vel_w = motion["body_ang_vel_w"]

    num_frames = joint_pos.shape[0]
    n_joints = joint_pos.shape[1]

    # Auto-detect DOF from joint count
    if dof == 0:
        dof = n_joints
        print(f"  Auto-detected DOF: {dof} (from {n_joints} joints)")

    if (dof == 23 and n_joints != 23) or (dof == 29 and n_joints != 29):
        print(f"[WARN] Joint count ({n_joints}) doesn't match --dof={dof}")

    dt = 1.0 / fps
    total_duration = num_frames * dt

    # Create robot entity
    robot = setup_entity(dof)
    model = robot.spec.compile()
    if model is None:
        raise RuntimeError("Failed to compile MuJoCo model")
    data = mujoco.MjData(model)

    print(f"Motion: {Path(npz_path).name}")
    print(f"  DOF:    {dof} ({n_joints} joints)")
    print(f"  Frames: {num_frames} @ {fps} fps ({total_duration:.1f}s)")
    print(f"  Bodies: {body_pos_w.shape[1]}")
    print(f"  Viewer: {'headless (offscreen)' if headless else 'window'}")

    # Setup offscreen renderer if needed
    renderer = None
    frames = []
    if headless and video_output:
        viewer_cfg = ViewerConfig(
            height=480, width=640,
            origin_type=ViewerConfig.OriginType.ASSET_ROOT,
            distance=2.0, elevation=-5.0, azimuth=90.0,
        )
        renderer = OffscreenRenderer(model=model, cfg=viewer_cfg)
        renderer.initialize()

    # Get joint and body names for verification
    joint_names = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        joint_names.append(name or f"actuator_{i}")

    body_names = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        body_names.append(name or f"body_{i}")

    print(f"  Joints: {len(joint_names)}")
    print(f"  Bodies: {len(body_names)}")

    # Playback
    start_wall = time.perf_counter()
    frame_count = 0

    if not headless:
        print("\nOpening MuJoCo viewer...")
        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                if frame_count >= num_frames:
                    if loop:
                        frame_count = 0
                        start_wall = time.perf_counter()
                    else:
                        print("Playback finished.")
                        break

                # Set root pose from NPZ body data
                # body_pos_w[frame, 0] = pelvis position (first robot body)
                # body_quat_w[frame, 0] = pelvis orientation
                # freejoint = position(3) + quat(4) = 7 entries
                data.qpos[0:3] = body_pos_w[frame_count, 0, :]
                data.qpos[3:7] = body_quat_w[frame_count, 0, :]
                # Set joint positions (remaining qpos entries)
                for j in range(min(n_joints, model.nq - 7)):
                    data.qpos[7 + j] = joint_pos[frame_count, j]

                mujoco.mj_forward(model, data)
                viewer.sync()

                if realtime:
                    sim_elapsed = frame_count * dt
                    wall_elapsed = time.perf_counter() - start_wall
                    target_wall = sim_elapsed / max(realtime_scale, 1e-6)
                    sleep_s = target_wall - wall_elapsed
                    if sleep_s > 0:
                        time.sleep(sleep_s)

                frame_count += 1
    else:
        # Headless playback
        print("\nRunning headless playback...")
        pbar_width = 40
        while frame_count < num_frames:
            data.qpos[0:3] = body_pos_w[frame_count, 0, :]
            data.qpos[3:7] = body_quat_w[frame_count, 0, :]
            for j in range(min(n_joints, model.nq - 7)):
                data.qpos[7 + j] = joint_pos[frame_count, j]

            mujoco.mj_forward(model, data)

            if renderer is not None:
                renderer.update(data)
                frames.append(renderer.render())

            if realtime:
                sim_elapsed = frame_count * dt
                wall_elapsed = time.perf_counter() - start_wall
                target_wall = sim_elapsed / max(realtime_scale, 1e-6)
                sleep_s = target_wall - wall_elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

            frame_count += 1
            # Progress bar
            progress = frame_count / num_frames
            bar_len = int(progress * pbar_width)
            print(f"\r  [{('#' * bar_len).ljust(pbar_width)}] "
                  f"{frame_count}/{num_frames} ({100*progress:.0f}%)", end="")

        print()

        # Save video
        if renderer is not None and frames and video_output:
            try:
                import imageio.v3 as iio
            except ImportError:
                print("[WARN] imageio not installed, skipping video save")
                return
            os.makedirs(os.path.dirname(video_output), exist_ok=True)
            iio.imwrite(video_output, np.stack(frames, axis=0), fps=fps)
            print(f"  Video saved: {video_output}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Play back motion NPZ file with MuJoCo viewer."
    )
    parser.add_argument("--npz", type=str, required=True,
                        help="Path to motion NPZ file")
    parser.add_argument("--dof", type=int, default=0, choices=[0, 23, 29],
                        help="DOF count (0 = auto-detect from joint count)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display")
    parser.add_argument("--video-output", type=str, default=None,
                        help="Save video to file (headless mode)")
    parser.add_argument("--no-realtime", action="store_true",
                        help="Play as fast as possible")
    parser.add_argument("--realtime-scale", type=float, default=1.0,
                        help="Playback speed (1.0 = real-time)")
    parser.add_argument("--no-loop", action="store_true",
                        help="Stop after one playback")
    args = parser.parse_args()

    if not os.path.exists(args.npz):
        print(f"[ERROR] File not found: {args.npz}")
        return

    play_motion(
        npz_path=args.npz,
        dof=args.dof,
        headless=args.headless,
        video_output=args.video_output,
        realtime=not args.no_realtime,
        realtime_scale=args.realtime_scale,
        loop=not args.no_loop,
    )


if __name__ == "__main__":
    main()
