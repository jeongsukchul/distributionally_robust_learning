# Copyright 2025 The Brax Authors.
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

"""Soft Actor-Critic training.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

import functools
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import replay_buffers
from module.buffer import DynamicBatchQueue
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from agents.rambo import checkpoint 
from agents.rambo import losses as rambo_losses
from agents.rambo import networks as rambo_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import optax
Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
from learning.module.wrapper.evaluator import Evaluator

ReplayBufferState = Any

# _PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""

  dynamics_optimizer_state : optax.OptState
  dynamics_params: Params
  policy_optimizer_state: optax.OptState
  policy_params: Params
  q_optimizer_state: optax.OptState
  q_params: Params
  target_q_params: Params
  gradient_steps: types.UInt64
  env_steps: types.UInt64
  alpha_optimizer_state: optax.OptState
  alpha_params: Params
  normalizer_params: running_statistics.RunningStatisticsState


def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    # local_devices_to_use: int ,
    rambo_network: rambo_networks.RAMBONetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    dynamics_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
) -> TrainingState:
  """Inits the training state and replicates it over devices."""
  key_dynamics, key_policy, key_q = jax.random.split(key,3)
  log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
  alpha_optimizer_state = alpha_optimizer.init(log_alpha)

  dynamics_params = rambo_network.dynamics_network.init(key_dynamics)
  dynamics_optimizer_state = dynamics_optimizer.init(dynamics_params)
  policy_params = rambo_network.policy_network.init(key_policy)
  policy_optimizer_state = policy_optimizer.init(policy_params)
  q_params = rambo_network.q_network.init(key_q)
  q_optimizer_state = q_optimizer.init(q_params)

  normalizer_params = running_statistics.init_state(
      specs.Array((obs_size,), jnp.dtype('float32'))
  )

  training_state = TrainingState(
      dynamics_optimizer_state=dynamics_optimizer_state,
      dynamics_params=dynamics_params,
      policy_optimizer_state=policy_optimizer_state,
      policy_params=policy_params,
      q_optimizer_state=q_optimizer_state,
      q_params=q_params,
      target_q_params=q_params,
      gradient_steps=types.UInt64(hi=0, lo=0),
      env_steps=types.UInt64(hi=0, lo=0),
      alpha_optimizer_state=alpha_optimizer_state,
      alpha_params=log_alpha,
      normalizer_params=normalizer_params,
  )
#   return jax.device_put_replicated(
#       training_state, jax.local_devices()[:local_devices_to_use]
#   )
  return training_state


def train(
    environment: envs.Env,
    termination_fn : Any,            #added
    num_timesteps,
    episode_length: int,
    wrap_env: bool = True,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    discounting: float = 0.9,
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 1,
    normalize_observations: bool = False,
    max_devices_per_host: Optional[int] = 4,
    reward_scaling: float = 1.0,
    tau: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    grad_updates_per_step: int = 1,
    rollout_length: int = 1,          #added
    adv_weight : float = 3e-4,        #added
    real_ratio : float = 0.5,         #added
    rollout_batch_size : int = 10000, #added
    model_train_freq : int = 250,       #added
    n_elites : int = 5,                  #added
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[
        rambo_networks.RAMBONetworks
    ] = rambo_networks.make_rambo_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    checkpoint_logdir: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
):
  """RAMBO training."""
#   process_id = jax.process_index()
#   local_devices_to_use = jax.local_device_count()
#   if max_devices_per_host is not None:
#     local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
#   device_count = local_devices_to_use * jax.process_count()
#   logging.info(
#       'local_device_count: %s; total_device_count: %s',
#       local_devices_to_use,
#       device_count,
#   )
#   if min_replay_size >= num_timesteps:
#     raise ValueError(
#         'No training will happen because min_replay_size >= num_timesteps'
#     )
#   print("local devices to use", local_devices_to_use)
#   print("device count", device_count)
  st = time.time()
  if max_replay_size is None:
    max_replay_size = num_timesteps
  num_timesteps = num_timesteps//model_train_freq * model_train_freq
  # The number of environment steps executed for every `actor_step()` call.
  env_steps_per_actor_step = action_repeat * num_envs
  # equals to ceil(min_replay_size / env_steps_per_actor_step)
  num_prefill_actor_steps = -(-min_replay_size // num_envs)
  num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
  assert num_timesteps - num_prefill_env_steps >= 0
  num_evals_after_init = max(num_evals - 1, 1)
  # The number of run_one_sac_epoch calls per run_sac_training.
  # equals to
  # ceil(num_timesteps - num_prefill_env_steps /
  #      (num_evals_after_init * env_steps_per_actor_step))
  num_training_steps_per_epoch = -(
      -(num_timesteps - num_prefill_env_steps)
      // (num_evals_after_init * env_steps_per_actor_step *model_train_freq)
  ) * model_train_freq
#   num_training_steps_per_epoch = num_training_steps_per_epoch//model_train_freq * model_train_freq
  print("num evals after init", num_evals_after_init)
  print("num_traning steps per epoch", num_training_steps_per_epoch)
#   assert num_envs % device_count == 0
  env = environment
  if wrap_env:
    if wrap_env_fn is not None:
      wrap_for_training = wrap_env_fn
    elif isinstance(env, envs.Env):
      wrap_for_training = envs.training.wrap
    else:
      raise ValueError('Unsupported environment type: %s' % type(env))

    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)
    v_randomization_fn = None
    if randomization_fn is not None:
      v_randomization_fn = functools.partial(
          randomization_fn,
          rng=jax.random.split(
              key, num_envs  #// jax.process_count()  // local_devices_to_use
          ),
      )
    env = wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )  # pytype: disable=wrong-keyword-args

  obs_size = env.observation_size
  if isinstance(obs_size, Dict):
    raise NotImplementedError('Dictionary observations not implemented in SAC')
  action_size = env.action_size

  normalize_fn = lambda x, y: x
  denormalize_fn = lambda x, y: x
  if normalize_observations:
    normalize_fn = running_statistics.normalize
    denormalize_fn = running_statistics.denormalize
  rambo_network = network_factory(
      observation_size=obs_size,
      action_size=action_size,
      preprocess_observations_fn=normalize_fn,
      postprocess_observations_fn=denormalize_fn,
  )

  make_policy = rambo_networks.make_inference_fn(rambo_network)

  alpha_optimizer = optax.adam(learning_rate=3e-4)
  dynamics_optimizer = optax.adam(learning_rate=learning_rate)
  policy_optimizer = optax.adam(learning_rate=learning_rate)
  q_optimizer = optax.adam(learning_rate=learning_rate)

  dummy_obs = jnp.zeros((obs_size,))
  dummy_action = jnp.zeros((action_size,))
  dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=dummy_obs,
      action=dummy_action,
      reward=0.0,
      discount=0.0,
      next_observation=dummy_obs,
      extras={'state_extras': {'truncation': 0.0}, 'policy_extras': {}},
  )
  real_batch_size = int(batch_size * real_ratio * grad_updates_per_step ) #// device_count)
  fake_batch_size = batch_size  * grad_updates_per_step  - real_batch_size # // device_count - real_batch_size
  replay_buffer = DynamicBatchQueue(
      max_replay_size=max_replay_size , # // device_count,
      dummy_data_sample=dummy_transition,
      sample_batch_size=real_batch_size,
  )
  fake_buffer = DynamicBatchQueue(
      max_replay_size= int(rollout_batch_size * rollout_length), #//device_count),
      dummy_data_sample=dummy_transition,
      sample_batch_size=fake_batch_size,
  )
  rollout_fn = rambo_networks.make_rollout_fn(rambo_network=rambo_network, termination_fn=termination_fn,
                                                   replay_buffer=replay_buffer, fake_buffer=fake_buffer)
  dynamics_loss, alpha_loss, critic_loss, actor_loss = rambo_losses.make_losses(
      rambo_network=rambo_network,
      reward_scaling=reward_scaling,
      discounting=discounting,
      action_size=action_size,
      adv_weight=adv_weight
  )
  dynamics_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      dynamics_loss, dynamics_optimizer , pmap_axis_name=None, has_aux=True#, pmap_axis_name=_PMAP_AXIS_NAME
  )
  alpha_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      alpha_loss, alpha_optimizer , pmap_axis_name=None#, pmap_axis_name=_PMAP_AXIS_NAME
  )
  critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      critic_loss, q_optimizer , pmap_axis_name=None#, pmap_axis_name=_PMAP_AXIS_NAME
  )
  actor_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      actor_loss, policy_optimizer, pmap_axis_name=None #, pmap_axis_name=_PMAP_AXIS_NAME
  )

  def sgd_step(
      carry: Tuple[TrainingState, PRNGKey], transitions: Transition
  ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
    training_state, key = carry

    key, key_dynamics, key_alpha, key_critic, key_actor = jax.random.split(key, 5)

    # dynamics_loss, dynamics_params, dynamics_optimizer_state = dynamics_update(
    #   training_state.dynamics_params,
    #   training_state.policy_params,
    #   training_state.normalizer_params,
    #   training_state.q_params,
    #   transitions,
    #   termination_fn,
    #   key_dynamics,
    #   optimizer_state=training_state.dynamics_optimizer_state,
    # )

    alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
        training_state.alpha_params,
        training_state.policy_params,
        training_state.normalizer_params,
        transitions,
        key_alpha,
        optimizer_state=training_state.alpha_optimizer_state,
    )
    alpha = jnp.exp(training_state.alpha_params)
    critic_loss, q_params, q_optimizer_state = critic_update(
        training_state.q_params,
        training_state.policy_params,
        training_state.normalizer_params,
        training_state.target_q_params,
        alpha,
        transitions,
        key_critic,
        optimizer_state=training_state.q_optimizer_state,
    )
    actor_loss, policy_params, policy_optimizer_state = actor_update(
        training_state.policy_params,
        training_state.normalizer_params,
        training_state.q_params,
        alpha,
        transitions,
        key_actor,
        optimizer_state=training_state.policy_optimizer_state,
    )

    new_target_q_params = jax.tree_util.tree_map(
        lambda x, y: x * (1 - tau) + y * tau,
        training_state.target_q_params,
        q_params,
    )

    metrics = {
        # 'dynamics_loss' : dynamics_loss,
        'critic_loss': critic_loss,
        'actor_loss': actor_loss,
        'alpha_loss': alpha_loss,
        'alpha': jnp.exp(alpha_params),
    }

    new_training_state = training_state.replace(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=new_target_q_params,
        gradient_steps=training_state.gradient_steps + 1,
        env_steps=training_state.env_steps,
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=alpha_params,
        normalizer_params=training_state.normalizer_params,
    )
    return (new_training_state, key), metrics

  def get_experience(
      normalizer_params: running_statistics.RunningStatisticsState,
      policy_params: Params,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[
      running_statistics.RunningStatisticsState,
      envs.State,
      ReplayBufferState,
  ]:
    policy = make_policy((normalizer_params, policy_params))
    env_state, transitions = acting.actor_step(
        env, env_state, policy, key, extra_fields=('truncation',)
    )

    normalizer_params = running_statistics.update(
        normalizer_params,
        transitions.observation,
        #pmap_axis_name=_PMAP_AXIS_NAME,
    )
    simul_info ={
          "simul/reward_mean" : transitions.reward.mean(),
          "simul/reward_std" : transitions.reward.std(),
          "simul/reward_max" : transitions.reward.max(),
          "simul/reward_min" : transitions.reward.min(),
        }
    buffer_state = replay_buffer.insert(buffer_state, transitions)
    return normalizer_params, env_state, buffer_state, simul_info
  def training_step(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      fake_buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[
      TrainingState,
      envs.State,
      ReplayBufferState,
      Metrics,
  ]:
    experience_key, training_key, rollout_key = jax.random.split(key,3)
    normalizer_params, env_state, buffer_state, simul_info = get_experience(
        training_state.normalizer_params,
        training_state.policy_params,
        env_state,
        buffer_state,
        experience_key,
    )
    training_state = training_state.replace(
        normalizer_params=normalizer_params,
        env_steps=training_state.env_steps + env_steps_per_actor_step,
    )

    buffer_state, real_transitions = replay_buffer.sample(buffer_state)
    fake_buffer_state, fake_transitions = fake_buffer.sample(fake_buffer_state)

    transitions = jax.tree_util.tree_map(
      lambda x,y: jnp.concatenate([x,y],axis=0),
      real_transitions,
      fake_transitions
    )

    # Change the front dimension of transitions so 'update_step' is called
    # grad_updates_per_step times by the scan.
    transitions = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
        transitions,
    )

    (training_state, _), metrics = jax.lax.scan(
        sgd_step, (training_state, training_key), transitions
    )
    metrics.update(simul_info)
    metrics['buffer_current_size'] = replay_buffer.size(buffer_state)  # pytype: disable=unsupported-operands  # lax-types
    metrics['fake_buffer_current_size'] = fake_buffer.size(fake_buffer_state)  # pytype: disable=unsupported-operands  # lax-types
    # metrics.update(rollout_info)
    return training_state, env_state, buffer_state, fake_buffer_state, metrics
  def dynamics_step(
      carry: Tuple[TrainingState, PRNGKey], transitions: Transition
  ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
    training_state, key = carry
    (dynamics_loss, loss_info) , dynamics_params, dynamics_optimizer_state = dynamics_update(
      training_state.dynamics_params,
      training_state.policy_params,
      training_state.normalizer_params,
      training_state.q_params,
      transitions,
      termination_fn,
      key,
      optimizer_state=training_state.dynamics_optimizer_state,
    )
    loss_info.update({'dynamics_loss' : dynamics_loss})
    return (training_state.replace(
      dynamics_optimizer_state=dynamics_optimizer_state,
      dynamics_params = dynamics_params,), key),  loss_info
  def prefill_replay_buffer(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:

    def f(carry, unused):
      del unused
      training_state, env_state, buffer_state, key = carry
      key, new_key = jax.random.split(key)
      new_normalizer_params, env_state, buffer_state, simul_info = get_experience(
          training_state.normalizer_params,
          training_state.policy_params,
          env_state,
          buffer_state,
          key,
      )
      new_training_state = training_state.replace(
          normalizer_params=new_normalizer_params,
          env_steps=training_state.env_steps + env_steps_per_actor_step,
      )
      return (new_training_state, env_state, buffer_state, new_key), ()

    return jax.lax.scan(
        f,
        (training_state, env_state, buffer_state, key),
        (),
        length=num_prefill_actor_steps,
    )[0]

#   prefill_replay_buffer = jax.pmap(
#       prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME
#   )

  def training_model_interval(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      fake_buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:


    def f(carry, unused_t):
      ts, es, bs, fs, k = carry
      k, new_key = jax.random.split(k)
      ts, es, bs, fs, metrics = training_step(ts, es, bs, fs, k)
      return (ts, es, bs, fs, new_key), metrics

    #get samples from replay buffer
    buffer_state, transitions = replay_buffer.sample_batch(buffer_state, num_training_steps_per_epoch * batch_size )#//device_count)
    transitions = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (num_training_steps_per_epoch, -1) + x.shape[1:]),
        transitions,
    )

    # dynamics training
    (training_state, key), dynamics_metrics = jax.lax.scan(
        dynamics_step, (training_state, key), transitions
    )
    #validation
    (means, _), normal_fn, denormal_fn = \
        rambo_network.dynamics_network.apply(training_state.normalizer_params, training_state.dynamics_params, 
                                              transitions.observation[-1], transitions.action[-1])
    target = jnp.concatenate([normal_fn(transitions.next_observation[-1] - transitions.observation[-1], 
                                        training_state.normalizer_params), transitions.reward[-1].reshape(-1, 1)], axis=-1)    
    mse_loss = ((means-target)**2).mean(axis=(1,2))
    elite_idxs = jnp.argsort(mse_loss)[:n_elites]
    print("elite_idxs", elite_idxs)
    
    # rollout from the info of elite_idxs
    fake_buffer_state, rollout_info =  rollout_fn(
        rng=key, 
        policy=make_policy((training_state.normalizer_params, training_state.policy_params)),
        normalizer_params=training_state.normalizer_params, 
        dynamics_params=training_state.dynamics_params,
        buffer_state=buffer_state, 
        fake_buffer_state=fake_buffer_state,
        rollout_length=rollout_length,
        rollout_batch_size=rollout_batch_size, # //device_count,
        elite_idxs= elite_idxs
    )

    (training_state, env_state, buffer_state, fake_buffer_state, key), metrics = jax.lax.scan(
        f,
        (training_state, env_state, buffer_state, fake_buffer_state, key),
        (),
        length=num_training_steps_per_epoch,
    )
    metrics.update(dynamics_metrics)
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    metrics.update(rollout_info)
    metrics.update({"mse_loss" : mse_loss})
    return training_state, env_state, buffer_state, fake_buffer_state, metrics, elite_idxs

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      fake_buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
    nonlocal training_walltime
    t = time.time()
    training_state, env_state, buffer_state, fake_buffer_state, metrics, elite_idxs = training_model_interval(
        training_state, env_state, buffer_state, fake_buffer_state, key
    )
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (
        env_steps_per_actor_step * num_training_steps_per_epoch
    ) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()},
    }
    return training_state, env_state, buffer_state, fake_buffer_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade

  global_key, local_key = jax.random.split(rng)
#   local_key = jax.random.fold_in(local_key, process_id)

  # Training state init
  training_state = _init_training_state(
      key=global_key,
      obs_size=obs_size,
    #   local_devices_to_use=local_devices_to_use,
      rambo_network=rambo_network,
      alpha_optimizer=alpha_optimizer,
      policy_optimizer=policy_optimizer,
      dynamics_optimizer=dynamics_optimizer,
      q_optimizer=q_optimizer,
  )
  del global_key

  if restore_checkpoint_path is not None:
    params = checkpoint.load(restore_checkpoint_path)
    training_state = training_state.replace(
        normalizer_params=params[0],
        policy_params=params[1],
    )

  local_key, rb_key, fb_key, env_key, eval_key = jax.random.split(local_key, 5)

  # Env init
  env_keys = jax.random.split(env_key, num_envs)# // jax.process_count())
#   env_keys = jnp.reshape(
#       env_keys, (local_devices_to_use, -1) + env_keys.shape[1:]
#   )
#   env_state = jax.pmap(env.reset)(env_keys)
  env_state =env.reset(env_keys)

  # Replay buffer init
#   buffer_state = jax.pmap(replay_buffer.init)(
#       jax.random.split(rb_key, local_devices_to_use)
#   ) 
  buffer_state = replay_buffer.init(rb_key)
#   fake_buffer_state = jax.pmap(fake_buffer.init)(
#       jax.random.split(fb_key, local_devices_to_use)
#   )
  fake_buffer_state = fake_buffer.init(fb_key)
  if not eval_env:
    eval_env = environment
  if wrap_env:
    if randomization_fn is not None:
      v_randomization_fn = functools.partial(
          randomization_fn, rng=jax.random.split(eval_key, num_eval_envs)
      )
    eval_env = wrap_for_training(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )  # pytype: disable=wrong-keyword-args

  evaluator = Evaluator(
      eval_env,
      functools.partial(make_policy, deterministic=deterministic_eval),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key,
  )
  ed = time.time()
  print('setup time', ed-st)
  # Run initial eval
  metrics = {}
#   if process_id == 0 and num_evals > 1:
  if num_evals > 1:
    st = time.time()
    metrics = evaluator.run_evaluation(
        # _unpmap(
            (training_state.normalizer_params, training_state.policy_params),
        # ),
        training_metrics={},
    )
    logging.info(metrics)
    progress_fn(0, metrics)
    ed = time.time()
    print("first eval time", ed-st)
  # Create and initialize the replay buffer.
  t = time.time()
  prefill_key, local_key = jax.random.split(local_key)
#   prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
  training_state, env_state, buffer_state, _ = prefill_replay_buffer(
      training_state, env_state, buffer_state, prefill_key#s
  )

#   replay_size = (
    #   jnp.sum(jax.vmap(replay_buffer.size)(buffer_state))  # * jax.process_count()
#   )
  replay_size = replay_buffer.size(buffer_state)
  logging.info('replay size after prefill %s', replay_size)
  assert replay_size >= min_replay_size

  training_walltime = time.time() - t

  current_step = 0
  for _ in range(num_evals_after_init):
    logging.info('step %s', current_step)

    # Optimization
    epoch_key, local_key = jax.random.split(local_key)
    # epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
    (training_state, env_state, buffer_state, fake_buffer_state, training_metrics) = (
        training_epoch_with_timing(
            training_state, env_state, buffer_state, fake_buffer_state, epoch_key#s
        )
    )
    # current_step = int(_unpmap(training_state.env_steps))
    current_step = int(training_state.env_steps)
    # Eval and logging
    # if process_id == 0:
    if checkpoint_logdir:
        # params = _unpmap(
        params=    (training_state.normalizer_params, training_state.policy_params)
        # )
        ckpt_config = checkpoint.network_config(
            observation_size=obs_size,
            action_size=env.action_size,
            normalize_observations=normalize_observations,
            network_factory=network_factory,
        )
        checkpoint.save(checkpoint_logdir, current_step, params, ckpt_config)

    # Run evals.
    metrics = evaluator.run_evaluation(
    #   _unpmap(
            (training_state.normalizer_params, training_state.policy_params),
    #   ),
        training_metrics,
    )
    logging.info(metrics)
    progress_fn(current_step, metrics)

  total_steps = current_step
  if not total_steps >= num_timesteps:
    raise AssertionError(
        f'Total steps {total_steps} is less than `num_timesteps`='
        f' {num_timesteps}.'
    )

  #params = _unpmap(
  params=    (training_state.normalizer_params, training_state.policy_params)
  #)

  # If there was no mistakes the training_state should still be identical on all
  # devices.
#   pmap.assert_is_replicated(training_state)
  logging.info('total steps: %s', total_steps)
#   pmap.synchronize_hosts()
  return (make_policy, params, metrics)
