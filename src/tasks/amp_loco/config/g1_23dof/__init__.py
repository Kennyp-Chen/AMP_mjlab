from mjlab.tasks.registry import register_mjlab_task
from src.tasks.amp_loco.rl import AMPOnPolicyRunner

from .env_cfgs import (
  g1_23dof_amp_flat_env_cfg,
  g1_23dof_amp_rough_env_cfg,
)
from .rl_cfg import g1_23dof_amp_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-23DOF-AMP-Rough",
  env_cfg=g1_23dof_amp_rough_env_cfg(),
  play_env_cfg=g1_23dof_amp_rough_env_cfg(play=True),
  rl_cfg=g1_23dof_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-23DOF-AMP-Flat",
  env_cfg=g1_23dof_amp_flat_env_cfg(),
  play_env_cfg=g1_23dof_amp_flat_env_cfg(play=True),
  rl_cfg=g1_23dof_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

# ------------------------------------------------------------------
# Experiment tasks (parameter tuning on Flat terrain)
# ------------------------------------------------------------------

# ExpX: Higher height tracking weight — strengthens the only positive
# reward signal available during the recovery window (23DOF variant).
register_mjlab_task(
  task_id="Unitree-G1-23DOF-AMP-Flat-HeightReward2x",
  env_cfg=g1_23dof_amp_flat_env_cfg(experiment="height_reward_2x"),
  play_env_cfg=g1_23dof_amp_flat_env_cfg(play=True, experiment="height_reward_2x"),
  rl_cfg=g1_23dof_amp_ppo_runner_cfg(experiment="height_reward_2x"),
  runner_cls=AMPOnPolicyRunner,
)
