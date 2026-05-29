#!/usr/bin/env python3
"""Convert 29-DOF motion NPZ files to 23-DOF format.

For each input NPZ:
  - joint_pos / joint_vel: remove 6 joints (waist_roll, waist_pitch,
    left_wrist_pitch, left_wrist_yaw, right_wrist_pitch, right_wrist_yaw).
  - body_* data: use body-name matching between the 29-DOF and 23-DOF MuJoCo
    models to remap body indices correctly.

Usage:
  # Convert all motion files (WalkandRun + Recovery)
  python scripts/convert_npz_23dof.py

  # Convert only WalkandRun
  python scripts/convert_npz_23dof.py --motion-dir WalkandRun

  # Custom paths
  python scripts/convert_npz_23dof.py \
    --input-base src/assets/motions/g1/amp \
    --output-base src/assets/motions/g1/23dof_amp
"""

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np

from src.assets.robots.unitree_g1.g1_23dof_constants import (
    G1_23DOF_XML,
)
from src.assets.robots.unitree_g1.g1_constants import (
    G1_XML,
)

# ---------------------------------------------------------------------------
# Joint mapping (29-DOF → 23-DOF)
# ---------------------------------------------------------------------------

_DOF_NAMES_29 = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

_DOF_NAMES_23 = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint",
]

# Indices in _DOF_NAMES_29 to REMOVE (6 joints → 23 remaining)
_JOINT_REMOVE_INDICES = sorted(
    i for i, name in enumerate(_DOF_NAMES_29) if name not in _DOF_NAMES_23
)

# Map from 29-DOF column → 23-DOF column
_JOINT_MAP_29_TO_23: dict[int, int] = {}
_new_idx = 0
for old_idx in range(len(_DOF_NAMES_29)):
    if old_idx not in _JOINT_REMOVE_INDICES:
        _JOINT_MAP_29_TO_23[old_idx] = _new_idx
        _new_idx += 1

# ---------------------------------------------------------------------------
# Body index mapping (built from MuJoCo model at runtime)
# ---------------------------------------------------------------------------

# Some bodies have different names between 29-DOF and 23-DOF models even though
# they represent the same physical link.  Handle these manually.
_BODY_NAME_ALIASES: dict[str, str] = {
    "left_wrist_roll_link": "left_wrist_roll_rubber_hand",
    "right_wrist_roll_link": "right_wrist_roll_rubber_hand",
}


def _get_body_names(xml_path: str) -> list[str]:
    """Return ordered list of body names from a MuJoCo XML model."""
    spec = mujoco.MjSpec.from_file(xml_path)
    spec.assets = {}
    model = spec.compile()
    if model is None:
        raise RuntimeError(f"Failed to compile model from {xml_path}")
    names = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        names.append(name)
    return names


def build_body_mapping() -> dict[int, int]:
    """Build mapping from 29-DOF NPZ body index → 23-DOF NPZ body index.

    NPZ files store one entry per **robot body** (worldbody is excluded).
    MuJoCo's body list includes worldbody at index 0, so we offset by -1.

    Only shared bodies (same name) are included.  Bodies that exist in the
    29-DOF model but not in the 23-DOF model are dropped.
    """
    names_29 = _get_body_names(str(G1_XML))
    names_23 = _get_body_names(str(G1_23DOF_XML))

    assert names_29[0] == "world", "Expected worldbody at index 0"
    assert names_23[0] == "world", "Expected worldbody at index 0"

    # Drop worldbody — NPZ indices start from the first robot body.
    robot_names_29 = names_29[1:]
    robot_names_23 = names_23[1:]

    # Build lookup: 23-DOF robot body name → NPZ index
    lookup_23: dict[str, int] = {}
    for idx, name in enumerate(robot_names_23):
        lookup_23[name] = idx
    # Add aliases for bodies with different names in the two models
    for alias, canonical in _BODY_NAME_ALIASES.items():
        if canonical in lookup_23 and alias not in lookup_23:
            lookup_23[alias] = lookup_23[canonical]

    # Build mapping
    mapping: dict[int, int] = {}
    for idx_29, name in enumerate(robot_names_29):
        if name in lookup_23:
            mapping[idx_29] = lookup_23[name]

    return mapping


# ---------------------------------------------------------------------------
# NPZ conversion helpers
# ---------------------------------------------------------------------------


def convert_joints(data_29: np.ndarray) -> np.ndarray:
    """Convert joint_pos or joint_vel from 29-DOF to 23-DOF.

    Args:
        data_29: shape (N, 29).

    Returns:
        shape (N, 23).
    """
    out = np.zeros((data_29.shape[0], 23), dtype=data_29.dtype)
    for old_idx, new_idx in _JOINT_MAP_29_TO_23.items():
        out[:, new_idx] = data_29[:, old_idx]
    return out


def convert_bodies(
    data_29: np.ndarray, body_mapping: dict[int, int]
) -> np.ndarray:
    """Remap body-indexed data from 29-DOF order to 23-DOF order.

    Args:
        data_29: shape (N, B_29, ...) where B_29 = 30 robot bodies.
        body_mapping: {29_idx → 23_idx}.

    Returns:
        shape (N, B_23, ...) where B_23 = 24 robot bodies.
    """
    b_23 = max(body_mapping.values()) + 1
    out = np.zeros((data_29.shape[0], b_23, *data_29.shape[2:]), dtype=data_29.dtype)
    for old_idx, new_idx in body_mapping.items():
        out[:, new_idx] = data_29[:, old_idx]
    return out


def convert_npz(
    input_path: str,
    output_path: str,
    body_mapping: dict[int, int],
    verbose: bool = True,
) -> None:
    """Convert a single NPZ file from 29-DOF to 23-DOF."""
    data = np.load(input_path)

    joint_pos = convert_joints(data["joint_pos"])
    joint_vel = convert_joints(data["joint_vel"])
    body_pos_w = convert_bodies(data["body_pos_w"], body_mapping)
    body_quat_w = convert_bodies(data["body_quat_w"], body_mapping)
    body_lin_vel_w = convert_bodies(data["body_lin_vel_w"], body_mapping)
    body_ang_vel_w = convert_bodies(data["body_ang_vel_w"], body_mapping)

    result = {
        "fps": data["fps"],
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, **result)

    if verbose:
        print(f"  ✓ {Path(input_path).name} → {Path(output_path).name}")
        print(f"    joint: {data['joint_pos'].shape} → {joint_pos.shape}")
        print(f"    body:  {data['body_pos_w'].shape} → {body_pos_w.shape}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert 29-DOF motion NPZ files to 23-DOF format."
    )
    parser.add_argument(
        "--input-base",
        default="src/assets/motions/g1/amp",
        help="Base directory containing 29-DOF motion data (default: src/assets/motions/g1/amp)",
    )
    parser.add_argument(
        "--output-base",
        default="src/assets/motions/g1/23dof_amp",
        help="Output directory for 23-DOF motion data (default: src/assets/motions/g1/23dof_amp)",
    )
    parser.add_argument(
        "--motion-dir",
        default=None,
        help='Subdirectory to convert, e.g. "WalkandRun" or "Recovery". '
        "If omitted, converts both.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be converted without actually converting.",
    )
    args = parser.parse_args()

    base_dir = Path(args.input_base)
    out_dir = Path(args.output_base)

    # Collect input directories
    if args.motion_dir:
        subdirs = [args.motion_dir]
    else:
        subdirs = ["WalkandRun", "Recovery"]

    all_input_files: list[tuple[Path, Path]] = []
    for sub in subdirs:
        src = base_dir / sub
        if not src.is_dir():
            print(f"[WARN] Directory not found: {src}")
            continue
        for npz_path in sorted(src.glob("*.npz")):
            rel = npz_path.relative_to(base_dir)
            dst = out_dir / rel
            all_input_files.append((npz_path, dst))

    if not all_input_files:
        print(f"[ERROR] No NPZ files found in {[str(base_dir / s) for s in subdirs]}")
        return

    print(f"Found {len(all_input_files)} NPZ file(s) to convert")
    print(f"  Input base:  {base_dir}")
    print(f"  Output base: {out_dir}")
    print()

    # Build body mapping from MuJoCo models
    print("Building body index mapping...")
    body_mapping = build_body_mapping()
    print(f"  Mapped {len(body_mapping)} bodies (29DOF → 23DOF)")
    print(f"  23DOF body count: {max(body_mapping.values()) + 1}")
    print(f"  Removed 29DOF bodies: "
          f"{set(range(30)) - set(body_mapping.keys())}")
    print()

    if args.dry_run:
        print("[DRY RUN] Files to convert:")
        for src, dst in all_input_files:
            print(f"  {src} → {dst}")
        return

    # Convert each file
    success = 0
    errors = 0
    for src_path, dst_path in all_input_files:
        try:
            convert_npz(str(src_path), str(dst_path), body_mapping)
            success += 1
        except Exception as e:
            print(f"  ✗ {src_path.name}: {e}")
            errors += 1

    print(f"\nDone: {success} converted, {errors} errors")
    if success > 0:
        print(f"Output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
