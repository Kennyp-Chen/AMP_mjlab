# AMP_mjlab

[中文 README](README_zh.md)

Deployment integration code is in [ccrpRepo/wbc_fsm](https://github.com/ccrpRepo/wbc_fsm), under `MJAmp State`.

G1 AMP motion control project built on top of mjlab + rsl_rl.

Key features of this repository:

- A single policy learns both locomotion (walk/run) and recovery (fall-and-get-up)
- AMP discriminator regularizes motion style and priors
- Training and deployment pipelines are consistent, with direct ONNX policy export support

## Core Idea

Instead of training separate policies for locomotion and recovery and switching between them, this project learns both capabilities in one unified policy.

Implementation highlights:

- Motion data split:
  - Walk/Run data: `src/assets/motions/g1/amp/WalkandRun`
  - Recovery data: `src/assets/motions/g1/amp/Recovery`
- Delayed termination/reset:
  - A subset of environments does not reset immediately after termination
  - These environments receive a recovery window and reset states sampled from recovery clips
- Unified AMP training:
  - One actor-critic + One AMP discriminator
  - Velocity tracking, perturbation robustness, and recovery are learned together

This reduces discontinuities caused by policy switching and yields more consistent behavior.

## Requirements

- Linux
- Python 3.11 (recommended)
- Working MuJoCo and GPU driver setup

## Quick Start

### 1. Install

```bash
conda activate mjlab
cd AMP_mjlab
python -m pip install -e .
```

### 2. Apply mjlab Patch (Optional)

If you do not apply this patch, remove `history_ordering` configuration from the code.

What this patch does:

- It adds an option for how observation history is flattened: by time (`time`) or by term (`term`).
- Default mjlab behavior supports only `term` ordering.

Patch file:

- `mjlab_patch/mjlab/managers/observation_manager.py`

Example command:

```bash
cp mjlab_patch/mjlab/managers/observation_manager.py \
  /home/crp/miniconda3/envs/mjlab/lib/python3.11/site-packages/mjlab/managers/observation_manager.py
```

### 3. List Available Tasks

```bash
python scripts/list_envs.py --keyword AMP
```

Main tasks (29-DOF):

- `Unitree-G1-AMP-Rough`
- `Unitree-G1-AMP-Flat`

New tasks (23-DOF):

- `Unitree-G1-23DOF-AMP-Rough`
- `Unitree-G1-23DOF-AMP-Flat`

## Training

**29-DOF (original):**

```bash
python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096
```

**23-DOF (reduced joint space):**

```bash
python scripts/train.py Unitree-G1-23DOF-AMP-Flat --env.scene.num-envs=4096
```

Logs are saved by default to:

- 29-DOF: `logs/rsl_rl/g1_amp_locomotion/<time_stamp_run>/`
- 23-DOF: `logs/rsl_rl/g1_23dof_amp_locomotion/<time_stamp_run>/`

## Training Curve Note (Important)

- Around `2w` iterations (about 20k), the policy often suddenly learns fall-recovery behavior.
- As a result, multiple metrics in `logs` may show abrupt jumps. This is expected and not necessarily a training failure.

![Training log transition example](logs.png)

## Evaluation and Visualization

Replay with a trained checkpoint:

**29-DOF:**

```bash
python scripts/play.py Unitree-G1-AMP-Rough \
  --checkpoint-file logs/rsl_rl/g1_amp_locomotion/<run_dir>/model_<iter>.pt
```

**23-DOF:**

```bash
python scripts/play.py Unitree-G1-23DOF-AMP-Rough \
  --checkpoint-file logs/rsl_rl/g1_23dof_amp_locomotion/<run_dir>/model_<iter>.pt
```

Note: ONNX export is enabled by default in both training and play workflows.

## Motion Data Preparation

### Standard (29-DOF) Conversion

CSV-to-NPZ conversion script:

```bash
python scripts/csv_to_npz.py --help
```

Recommended data layout:

- Raw CSV: `motion_data_csv/amp`
- Converted NPZ: `src/assets/motions/g1/amp/WalkandRun` and `src/assets/motions/g1/amp/Recovery`

If valid NPZ files exist in these folders, training config loads them automatically.

### 29-DOF to 23-DOF Conversion

The 23-DOF robot removes waist yaw/roll, wrist pitch/yaw joints (6 DOF removed).
A dedicated conversion script maps 29-DOF motion data to the 23-DOF joint and body space:

```bash
python scripts/convert_npz_23dof.py \
  --input src/assets/motions/g1/amp \
  --output src/assets/motions/g1/23dof_amp
```

Body mapping (29→23, worldbody excluded):
- Keep: body indices 0-12, 15-20, 23-27
- Remove: waist_yaw(13), waist_roll(14), left_wrist_pitch(21), left_wrist_yaw(22), right_wrist_pitch(28), right_wrist_yaw(29)

Joint mapping removes 6 DOF: left_waist_yaw, left_waist_roll, right_waist_yaw, right_waist_roll, left_wrist_pitch, left_wrist_yaw, right_wrist_pitch, right_wrist_yaw, and the corresponding joint velocity entries.

To preview motion data:

```bash
python scripts/play_motion.py --npz src/assets/motions/g1/23dof_amp/WalkandRun/amp_walk_forward.npz
```

## 23-DOF Architecture

The 23-DOF variant reduces the Unitree G1 from 29 actuated joints to 23 by removing:

| Removed Joints | Reason |
|---|---|
| left_waist_yaw, left_waist_roll | Waist DOF |
| right_waist_yaw, right_waist_roll | Waist DOF |
| left_wrist_pitch, left_wrist_yaw | End-effector DOF |
| right_wrist_pitch, right_wrist_yaw | End-effector DOF |

This reduction:
- Lowers action dimension from 29 to 23
- Simplifies the policy network
- Focuses learning on essential locomotion joints (hip, knee, ankle, shoulder, elbow)
- Uses the same AMP observation structure with updated body names

Task configs are located at `src/tasks/amp_loco/config/g1_23dof/`.

## Repository Structure

- `src/tasks/amp_loco`: AMP locomotion/recovery task implementation
- `src/tasks/amp_loco/config/g1`: G1 29-DOF task registration, env configs, RL configs
- `src/tasks/amp_loco/config/g1_23dof`: G1 23-DOF task registration, env configs, RL configs
- `src/tasks/amp_loco/mdp`: rewards, observations, events, termination logic
- `scripts/train.py`: training entry point
- `scripts/play.py`: playback entry point
- `scripts/csv_to_npz.py`: standard 29-DOF motion data conversion tool
- `scripts/convert_npz_23dof.py`: 29-DOF to 23-DOF motion data converter
- `scripts/play_motion.py`: NPZ motion playback visualizer
- `mjlab_patch`: required local patch for mjlab

## Highlights

- One policy unifies walk/run and recovery skills
- AMP + velocity objective jointly optimize style and task performance
- Delayed reset with recovery sampling explicitly improves recovery ability
- End-to-end pipeline supports ONNX export for deployment

## Acknowledgements

- Thanks to [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) for open-sourcing their work and inspiration.
- Thanks to [Open-X-Humanoid/TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab); the rsl_rl AMP part in this project references their implementation.
