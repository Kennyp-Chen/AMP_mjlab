from mjlab.tasks.registry import register_mjlab_task
from src.tasks.amp_loco.rl import AMPOnPolicyRunner

from .env_cfgs import (
  g1_amp_flat_env_cfg,
  g1_amp_rough_env_cfg,
)
from .rl_cfg import g1_amp_ppo_runner_cfg

# ------------------------------------------------------------------
# Baseline tasks (original parameters)
# ------------------------------------------------------------------

register_mjlab_task(
  task_id="Unitree-G1-AMP-Rough",
  env_cfg=g1_amp_rough_env_cfg(),
  play_env_cfg=g1_amp_rough_env_cfg(play=True),
  rl_cfg=g1_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-AMP-Flat",
  env_cfg=g1_amp_flat_env_cfg(),
  play_env_cfg=g1_amp_flat_env_cfg(play=True),
  rl_cfg=g1_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

# ------------------------------------------------------------------
# Experiment tasks (parameter tuning on Flat terrain)
# ------------------------------------------------------------------

# ExpA: Lower termination height — gives the robot more room to
# transition through low-height states (kneeling) during recovery.
register_mjlab_task(
  task_id="Unitree-G1-AMP-Flat-Height03",
  env_cfg=g1_amp_flat_env_cfg(experiment="height_0.3"),
  play_env_cfg=g1_amp_flat_env_cfg(play=True, experiment="height_0.3"),
  rl_cfg=g1_amp_ppo_runner_cfg(experiment="height_0.3"),
  runner_cls=AMPOnPolicyRunner,
)

# ExpB: Higher termination height — forces stricter fall prevention,
#  the policy must keep the root higher at all times.
register_mjlab_task(
  task_id="Unitree-G1-AMP-Flat-Height07",
  env_cfg=g1_amp_flat_env_cfg(experiment="height_0.7"),
  play_env_cfg=g1_amp_flat_env_cfg(play=True, experiment="height_0.7"),
  rl_cfg=g1_amp_ppo_runner_cfg(experiment="height_0.7"),
  runner_cls=AMPOnPolicyRunner,
)

# ExpC: Longer recovery window — gives more time (16s vs 5s)
#  for the policy to learn getting up from a fallen state.
register_mjlab_task(
  task_id="Unitree-G1-AMP-Flat-Recovery800",
  env_cfg=g1_amp_flat_env_cfg(experiment="recovery_800"),
  play_env_cfg=g1_amp_flat_env_cfg(play=True, experiment="recovery_800"),
  rl_cfg=g1_amp_ppo_runner_cfg(experiment="recovery_800"),
  runner_cls=AMPOnPolicyRunner,
)

# ExpD: Higher height tracking weight — strengthens the only positive
#  reward signal available during the recovery window.
register_mjlab_task(
  task_id="Unitree-G1-AMP-Flat-HeightReward3x",
  env_cfg=g1_amp_flat_env_cfg(experiment="height_reward_3x"),
  play_env_cfg=g1_amp_flat_env_cfg(play=True, experiment="height_reward_3x"),
  rl_cfg=g1_amp_ppo_runner_cfg(experiment="height_reward_3x"),
  runner_cls=AMPOnPolicyRunner,
)

# ExpE: Different random seed (default=42 → 100) — varies policy init
#  and env randomization to see if recovery emerges with different seed.
register_mjlab_task(
  task_id="Unitree-G1-AMP-Flat-Seed100",
  env_cfg=g1_amp_flat_env_cfg(experiment="seed_100"),
  play_env_cfg=g1_amp_flat_env_cfg(play=True, experiment="seed_100"),
  rl_cfg=g1_amp_ppo_runner_cfg(experiment="seed_100", seed=100),
  runner_cls=AMPOnPolicyRunner,
)

# ExpF: Higher height tracking weight — strengthens the only positive
#  reward signal available during the recovery window.
register_mjlab_task(
  task_id="Unitree-G1-AMP-Flat-HeightReward2x",
  env_cfg=g1_amp_flat_env_cfg(experiment="height_reward_2x"),
  play_env_cfg=g1_amp_flat_env_cfg(play=True, experiment="height_reward_2x"),
  rl_cfg=g1_amp_ppo_runner_cfg(experiment="height_reward_2x"),
  runner_cls=AMPOnPolicyRunner,
)

# ExpG: Higher termination height — forces stricter fall prevention,
#  the policy must keep the root higher at all times.
register_mjlab_task(
  task_id="Unitree-G1-AMP-Flat-Height06",
  env_cfg=g1_amp_flat_env_cfg(experiment="height_0.6"),
  play_env_cfg=g1_amp_flat_env_cfg(play=True, experiment="height_0.6"),
  rl_cfg=g1_amp_ppo_runner_cfg(experiment="height_0.6"),
  runner_cls=AMPOnPolicyRunner,
)
