# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""RL config for DM Control Suite."""

from ml_collections import config_dict

from mujoco_playground._src import dm_control_suite
from module.termination_fn import get_termination_fn


def brax_ppo_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax PPO config for the given environment."""
  env_config = dm_control_suite.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=60_000_000,
      num_evals=10,
      reward_scaling=10.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=30,
      num_minibatches=32,
      num_updates_per_batch=16,
      discounting=0.995,
      learning_rate=1e-3,
      entropy_cost=1e-2,
      num_envs=2048,
      batch_size=1024,
  )

  if env_name.startswith("AcrobotSwingup"):
    rl_config.num_timesteps = 100_000_000
  if env_name == "BallInCup":
    rl_config.discounting = 0.95
  elif env_name.startswith("Swimmer"):
    rl_config.num_timesteps = 100_000_000
  elif env_name == "WalkerRun":
    rl_config.num_timesteps = 100_000_000
  elif env_name == "FingerSpin":
    rl_config.discounting = 0.95
  elif env_name == "PendulumSwingUp":
    rl_config.action_repeat = 4
    rl_config.num_updates_per_batch = 4

  return rl_config


def brax_vision_ppo_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax Vision PPO config for the given environment."""
  env_config = dm_control_suite.get_default_config(env_name)

  rl_config = config_dict.create(
      madrona_backend=True,
      wrap_env=False,
      num_timesteps=1_000_000,
      num_evals=5,
      reward_scaling=0.1,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=10,
      num_minibatches=8,
      num_updates_per_batch=8,
      discounting=0.97,
      learning_rate=5e-4,
      entropy_cost=5e-3,
      num_envs=1024,
      num_eval_envs=1024,
      batch_size=256,
      max_grad_norm=1.0,
  )

  if env_name != "CartpoleBalance":
    raise NotImplementedError(f"Vision PPO params not tested for {env_name}")

  return rl_config


def brax_sac_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax SAC config for the given environment."""
  
  env_config = dm_control_suite.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=5_000_000,
      num_evals=10,
      reward_scaling=1.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      discounting=0.99,
      learning_rate=1e-3,
      num_envs=128,
      batch_size=512,
      grad_updates_per_step=8,
      max_replay_size=1048576 * 4,
      min_replay_size=8192,
      network_factory=config_dict.create(
          q_network_layer_norm=True,
      ),
  )

  if env_name == "PendulumSwingUp":
    rl_config.action_repeat = 4
  if env_name =="HopperHop":
    rl_config.num_timesteps = 10_000_000
  if (
      env_name.startswith("Acrobot")
      or env_name.startswith("Swimmer")
      or env_name.startswith("Finger")
      or env_name.startswith("Hopper")
      or env_name
      in ("CheetahRun", "HumanoidWalk", "PendulumSwingUp", "WalkerRun")
  ):
    rl_config.num_timesteps = 10_000_000

  return rl_config
def brax_sac_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax SAC config for the given environment."""
  
  env_config = dm_control_suite.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=5_000_000,
      num_evals=10,
      reward_scaling=1.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      discounting=0.99,
      learning_rate=1e-3,
      num_envs=128,
      batch_size=256,
      grad_updates_per_step=8,
      max_replay_size=1048576 * 4,
      min_replay_size=8192,
      network_factory=config_dict.create(
          q_network_layer_norm=True,
      ),
  )

  if env_name == "PendulumSwingUp":
    rl_config.action_repeat = 4
  if env_name =="HopperHop":
    rl_config.num_timesteps = 10_000_000
  if (
      env_name.startswith("Acrobot")
      or env_name.startswith("Swimmer")
      or env_name.startswith("Finger")
      or env_name.startswith("Hopper")
      or env_name
      in ("CheetahRun", "HumanoidWalk", "PendulumSwingUp", "WalkerRun")
  ):
    rl_config.num_timesteps = 20_000_000

  return rl_config

def brax_rambo_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax SAC config for the given environment."""
  
  env_config = dm_control_suite.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=5_000_000,
      num_evals=10,
      reward_scaling=1.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      discounting=0.99,  
      learning_rate=1e-3,
      num_envs=128,
      batch_size=512,
      grad_updates_per_step=8,
      max_replay_size=1048576 * 4,
      min_replay_size=8192,
      rollout_length =1,          #added
      real_ratio = 0.9,           #added
      adv_weight = 0.,          #added
      rollout_batch_size = 100000,   # added
      model_train_freq = 500,        # added
      termination_fn = get_termination_fn(env_name.lower()),     #added
      network_factory=config_dict.create(
          q_network_layer_norm=True,
      ),
  )

  if env_name == "PendulumSwingUp":
    rl_config.action_repeat = 4
  if env_name =="HopperHop":
    rl_config.num_timesteps = 20_000_000
  if (
      env_name.startswith("Acrobot")
      or env_name.startswith("Swimmer")
      or env_name.startswith("Finger")
      or env_name.startswith("Hopper")
      or env_name
      in ("CheetahRun", "HumanoidWalk", "PendulumSwingUp", "WalkerRun")
  ):
    rl_config.num_timesteps = 20_000_000

  return rl_config

def brax_wdsac_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax SAC config for the given environment."""
  
  env_config = dm_control_suite.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=5_000_000,
      num_evals=10,
      reward_scaling=1.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      discounting=0.99,  
      learning_rate=1e-3,
      num_envs=32,
      batch_size=256,
      grad_updates_per_step=8,
      max_replay_size=1048576 * 4,
      min_replay_size=8192,
      n_nominals = 10,# added
      delta = 0.01,    #added
      lambda_update_steps = 100,  #added: number of lambda optimization steps
      single_lambda = False, #added
      distance_type = "wass", #added
      lmbda_lr = 3e-4, #added
      init_lmbda = 0., #added
      network_factory=config_dict.create(
          q_network_layer_norm=True,
      ),
  )

  if env_name == "PendulumSwingUp":
    rl_config.action_repeat = 4
  if env_name =="HopperHop":
    rl_config.num_timesteps = 10_000_000
  if (
      env_name.startswith("Acrobot")
      or env_name.startswith("Swimmer")
      or env_name.startswith("Finger")
      or env_name.startswith("Hopper")
      or env_name
      in ("CheetahRun", "HumanoidWalk", "PendulumSwingUp", "WalkerRun")
  ):
    rl_config.num_timesteps = 20_000_000

  return rl_config


def brax_flowsac_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax SAC config for the given environment."""
  
  env_config = dm_control_suite.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=5_000_000,
      num_evals=10,
      reward_scaling=1.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      discounting=0.99,  
      learning_rate=1e-3,
      num_envs=32,
      batch_size=256,
      grad_updates_per_step=8,
      max_replay_size=1048576 * 4,
      min_replay_size=8192,
      delta = 0.01,    #added
      lambda_update_steps = 10,  #added: number of lambda optimization steps
      single_lambda = False, #added
      # distance_type = "wass", #added
      lmbda_lr = 3e-4, #added
      flow_lr = 3e-4,  #added
      init_lmbda = 0., #added
      network_factory=config_dict.create(
          q_network_layer_norm=True,
      ),
  )

  if env_name == "PendulumSwingUp":
    rl_config.action_repeat = 4
  if env_name =="HopperHop":
    rl_config.num_timesteps = 10_000_000
  if (
      env_name.startswith("Acrobot")
      or env_name.startswith("Swimmer")
      or env_name.startswith("Finger")
      or env_name.startswith("Hopper")
      or env_name
      in ("CheetahRun", "HumanoidWalk", "PendulumSwingUp", "WalkerRun")
  ):
    rl_config.num_timesteps = 20_000_000

  return rl_config

def brax_tdmpc_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax SAC config for the given environment."""
  env_config = dm_control_suite.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=5_000_000,
      batch_size = 256,
      reward_coef = 0.1,
      value_coef = 0.1,
      consistency_coef = 20,
      rho = 0.5,
      enc_lr_scale = 0.3,
      grad_clip_norm = 20,
      discount_denom = 5,
      discount_min = 0.95,
      discount_max = 0.995,
      buffer_size = 5_000_000,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      discounting=0.99,
      learning_rate=1e-3,
      grad_updates_per_step=8,
      tau = 0.01,
      mpc = True,
      iterations =6,
      num_samples = 512,
      num_elites = 64,
      num_pi_trajs =24,
      horion = 3,
      min_std = 0.05,
      max_std = 2,
      temperature = 0.5,
      actor_mode = "residual",
      log_std_min = -10,
      log_std_max = 2,
      prior_coef = 1.0,
      scale_threshold = 2.0,
      entropy_coef = 1e-4,
      awac_lambda = 0.3333,
      exp_adv_min = 0.1,
      exp_adv_max = 10.0,
      num_bins = 101,
      vmin = -10,
      vmax = 10,
      model_size = 1,
      enc_dim = 256,
      num_enc_layers = 2,
      num_channels = 32,
      latent_dim = 512,
      task_dim = 96,
      num_q = 5,
      dropout = 0.01,
      simnorm_dim = 8,
  )

  # if env_name == "PendulumSwingUp":
  #   rl_config.action_repeat = 4

  # if (
  #     env_name.startswith("Acrobot")
  #     or env_name.startswith("Swimmer")
  #     or env_name.startswith("Finger")
  #     or env_name.startswith("Hopper")
  #     or env_name
  #     in ("CheetahRun", "HumanoidWalk", "PendulumSwingUp", "WalkerRun")
  # ):
  #   rl_config.num_timesteps = 10_000_000

  return rl_config
