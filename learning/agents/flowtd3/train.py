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
import struct
from learning.module.wrapper.adv_wrapper import wrap_for_adv_training
from learning.module.wrapper.evaluator import Evaluator, AdvEvaluator
from learning.module.normalizing_flow.simple_flow import render_flow_pdf_1d_subplots
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union, NamedTuple, Sequence

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import replay_buffers
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from agents.flowtd3 import checkpoint
from agents.flowtd3 import losses as flowtd3_losses
from agents.flowtd3 import networks as flowtd3_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.envs.base import Wrapper
import flax
import jax
import jax.numpy as jnp
import optax
from brax.envs.base import Wrapper, Env, State
from brax.training.types import Policy, PolicyParams, PRNGKey, Metrics, Transition
from flax.core import FrozenDict
from mujoco_playground._src.wrapper import Wrapper, wrap_for_brax_training
Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any

_PMAP_AXIS_NAME = 'i'

class TransitionwithParams(NamedTuple):
  """Transition with additional dynamics parameters."""
  observation: jax.Array
  dynamics_params: jax.Array
  action: jax.Array
  reward: jax.Array
  discount: jax.Array
  next_observation: jax.Array
  extras: FrozenDict[str, Any]  # recommended
@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""

  policy_optimizer_state: optax.OptState
  policy_params: Params
  q_optimizer_state: optax.OptState
  q_params: Params
  target_q_params: Params
  gradient_steps: types.UInt64
  env_steps: types.UInt64
  normalizer_params: running_statistics.RunningStatisticsState
  flow_optimizer_state : optax.OptState
  flow_params : Params
  target_flow_params : Params
  noise_scales: jnp.ndarray

def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)


def _init_training_state(
    key: PRNGKey,
    obs_size: Union[int, Dict[str, specs.Array]],    
    local_devices_to_use: int,
    flowtd3_network: flowtd3_networks.FlowTd3Networks,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    flow_optimizer: optax.GradientTransformation,
    num_envs : int,
    std_max: float =0.4,
    std_min : float =0.05,
) -> TrainingState:
  """Inits the training state and replicates it over devices."""
  key_policy, key_q, key_noise, key_flow = jax.random.split(key,4 )

  policy_params = flowtd3_network.policy_network.init(key_policy)
  policy_optimizer_state = policy_optimizer.init(policy_params)
  q_params = flowtd3_network.q_network.init(key_q)
  q_optimizer_state = q_optimizer.init(q_params)

  flow_params = flowtd3_network.flow_network.init({"params": key_flow ,'constant':key_flow})
  target_flow_params = flowtd3_network.flow_network.init({"params": key_flow ,'constant':key_flow})
  flow_optimizer_state = flow_optimizer.init(flow_params)
  normalizer_params = running_statistics.init_state(
    #   specs.Array((obs_size,), jnp.dtype('float32'))
    obs_size if isinstance(obs_size, dict) else specs.Array((obs_size,), jnp.dtype('float32'))
  )
  training_state = TrainingState(
      policy_optimizer_state=policy_optimizer_state,
      policy_params=policy_params,
      q_optimizer_state=q_optimizer_state,
      q_params=q_params,
      target_q_params=q_params,
      gradient_steps=types.UInt64(hi=0, lo=0),
      env_steps=types.UInt64(hi=0, lo=0),
      normalizer_params=normalizer_params,
      flow_optimizer_state=flow_optimizer_state,
      flow_params=flow_params,
      target_flow_params=target_flow_params,
      noise_scales= jax.random.normal(key_noise, (num_envs, )) *(std_max - std_min) + std_min,
  )
  return jax.device_put_replicated(
      training_state, jax.local_devices()[:local_devices_to_use]
  )


def train(
    environment: envs.Env,
    num_timesteps,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    flow_lr : float = 1e-4,  # Learning rate for flow network
    init_lmbda: float = 0.1,   #added
    alpha : float = 0.5,
    discounting: float = 0.9, 
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 1,
    normalize_observations: bool = False,
    max_devices_per_host: Optional[int] = None,
    reward_scaling: float = 1.0,
    tau: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    grad_updates_per_step: int = 1,
    network_factory: types.NetworkFactory[
        flowtd3_networks.FlowTd3Networks
    ] = flowtd3_networks.make_flowtd3_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    eval_randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    checkpoint_logdir: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    dr_train_ratio = 1.0,
    std_max=0.4,
    std_min=0.05,
    eval_with_training_env=False,
    use_wandb=False,
):
  """flowtd3 training."""
  process_id = jax.process_index()
  local_devices_to_use = jax.local_device_count()
  if max_devices_per_host is not None:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  device_count = local_devices_to_use * jax.process_count()
  logging.info(
      'local_device_count: %s; total_device_count: %s',
      local_devices_to_use,
      device_count,
  )

  if min_replay_size >= num_timesteps:
    raise ValueError(
        'No training will happen because min_replay_size >= num_timesteps'
    )

  if max_replay_size is None:
    max_replay_size = num_timesteps
  st = time.time()
  # The number of environment steps executed for every `actor_step()` call.
  env_steps_per_actor_step = action_repeat * num_envs
  # equals to ceil(min_replay_size / env_steps_per_actor_step)
  num_prefill_actor_steps = -(-min_replay_size // num_envs)
  num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
  assert num_timesteps - num_prefill_env_steps >= 0
  num_evals_after_init = max(num_evals - 1, 1)
  # The number of run_one_td3_epoch calls per run_td3_training.
  # equals to
  # ceil(num_timesteps - num_prefill_env_steps /
  #      (num_evals_after_init * env_steps_per_actor_step))
  num_training_steps_per_epoch = -(
      -(num_timesteps - num_prefill_env_steps)
      // (num_evals_after_init * env_steps_per_actor_step)
  )

  assert num_envs % device_count == 0
  import copy
  env = copy.deepcopy(environment)

  rng = jax.random.PRNGKey(seed)
  rng, key = jax.random.split(rng)


  obs_shape = env.observation_size
  print("flowtd3 OBS SIZE", obs_shape)
  action_size = env.action_size
  if hasattr(env, 'dr_range'):
    dr_range_low, dr_range_high = env.dr_range
    dr_range = dr_range_high - dr_range_low
    dr_range_mid = (dr_range_high + dr_range_low) / 2.0
    dr_range_low, dr_range_high = dr_range_mid - dr_range/2 * dr_train_ratio, dr_range_mid + dr_range/2 * dr_train_ratio
    volume = jnp.prod(jnp.maximum(dr_range_high - dr_range_low, 0.0))
    print("volume : ", volume)
  else:
    # Fallback configuration if environment doesn't have dr_range
    raise ValueError("Environment does not have dr_range attribute. Please provide a valid environment with dr_range.")
  training_dr_range=  dr_range_low, dr_range_high
  env = wrap_for_adv_training(
      env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=functools.partial(randomization_fn,dr_range=training_dr_range),
  )  # pytype: disable=wrong-keyword-args
  normalize_fn = lambda x, y: x
  if normalize_observations:
    normalize_fn = running_statistics.normalize
  flowtd3_network = network_factory(
      observation_size=obs_shape,
      action_size=action_size,
      preprocess_observations_fn=normalize_fn,
      dynamics_param_size=len(dr_range_low),
  )
  make_policy = flowtd3_networks.make_inference_fn(flowtd3_network)


  policy_optimizer = optax.adam(learning_rate=learning_rate)
  q_optimizer = optax.adam(learning_rate=learning_rate)
  flow_optimizer = optax.adam(learning_rate=flow_lr)  # Flow network optimizer

  dummy_params = jnp.zeros((len(dr_range_low),))  # Dummy dynamics parameters
  dummy_obs = { key: jnp.zeros(obs_shape[key]) for key in obs_shape } if isinstance(obs_shape, dict) else jnp.zeros((obs_shape,))
  print("dummy_obs", dummy_obs)
  dummy_action = jnp.zeros((action_size,))
  dummy_transition = TransitionwithParams(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=dummy_obs,
      action=dummy_action,
      dynamics_params=dummy_params,
      reward=0.0,
      discount=0.0,
      next_observation=dummy_obs,
      extras={'state_extras': {'truncation': 0.0}, 'policy_extras': {}},
  )
  max_replay_size = max(max_replay_size // device_count, env.episode_length * num_envs // device_count)
  replay_buffer = replay_buffers.UniformSamplingQueue(
      max_replay_size=max_replay_size,
      dummy_data_sample=dummy_transition,
      sample_batch_size=batch_size * grad_updates_per_step // device_count,
  )

  critic_loss, actor_loss, flow_loss = flowtd3_losses.make_losses(
      flowtd3_network=flowtd3_network,
      reward_scaling=reward_scaling,
      discounting=discounting,
      action_size=action_size,
  )
  critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      critic_loss, q_optimizer, has_aux=True, pmap_axis_name=_PMAP_AXIS_NAME
  )
  actor_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
  )
  flow_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
    flow_loss, flow_optimizer, has_aux=True, pmap_axis_name=_PMAP_AXIS_NAME
  )
  def sgd_step(
      carry: Tuple[TrainingState, PRNGKey], transitions: Transition
  ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
    training_state, key = carry
    noise_clip=0.5
    policy_noise = 0.2
    key, key_critic, key_actor,key_noise,key_flow = jax.random.split(key, 5)
    
    (flow_loss, (next_q_adv, value_loss, kl_loss)), flow_params, flow_optimizer_state = flow_update(
        training_state.flow_params,
        training_state.target_flow_params,
        training_state.policy_params,
        training_state.normalizer_params,
        training_state.target_q_params,
        transitions,
        dr_range_high,
        dr_range_low,
        init_lmbda,
        alpha,
        key_flow,  # Reuse key_actor for flow update
        optimizer_state=training_state.flow_optimizer_state,
    )
    noise = jax.random.normal(key_noise, shape=transitions.action.shape) * policy_noise
    noise = jnp.clip(noise,-noise_clip, noise_clip)
    (critic_loss, (current_q, next_v)), q_params, q_optimizer_state = critic_update(
        training_state.q_params,
        training_state.policy_params,
        training_state.normalizer_params,
        training_state.target_q_params,
        transitions,
        noise,
        key_critic,
        optimizer_state=training_state.q_optimizer_state,
    )
    actor_loss, policy_params, policy_optimizer_state = actor_update(
        training_state.policy_params,
        training_state.normalizer_params,
        training_state.q_params,
        transitions,
        key_actor,
        optimizer_state=training_state.policy_optimizer_state,
    )

    new_target_q_params = jax.tree_util.tree_map(
        lambda x, y: x * (1 - tau) + y * tau,
        training_state.target_q_params,
        q_params,
    )
    new_target_flow_params = jax.tree_util.tree_map(
        lambda x, y: x * (1 - tau) + y * tau,
        training_state.target_q_params,
        flow_params,
    )
    metrics = {
        'critic_loss': critic_loss,
        'actor_loss': actor_loss,
        'flow_loss' : flow_loss,
        'flow_value_loss':value_loss,
        'flow_kl_loss' : kl_loss,
        'current_q_min' : current_q.min(),
        'current_q_max' : current_q.max(),
        'current_q_mean' : current_q.mean(),
        'next_v_min' : next_v.min(),
        'next_v_max' : next_v.max(),
        'next_v_mean' : next_v.mean(),
    }

    new_training_state = training_state.replace(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=new_target_q_params,
        flow_optimizer_state=flow_optimizer_state,
        flow_params=flow_params,
        target_flow_params = new_target_flow_params,
        gradient_steps=training_state.gradient_steps + 1,
        env_steps=training_state.env_steps,
        normalizer_params=training_state.normalizer_params,
        noise_scales=training_state.noise_scales,
    )
    return (new_training_state, key), metrics

  def adv_step(
    env: Env,
    env_state: State,
    dynamics_params: jnp.ndarray,
    policy: Policy,
    noise_scales : jnp.ndarray,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
  ):
    step_key, key = jax.random.split(key)
    actions, policy_extras = policy(env_state.obs, noise_scales, key)
    # dynamics_params = jax.random.uniform(key=step_key, shape=(num_envs,len(dr_range_low)), minval=dr_range_low, maxval=dr_range_high)
    params = env_state.info["dr_params"] * (1 - env_state.done[..., None]) + dynamics_params * env_state.done[..., None]
    nstate = env.step(env_state, actions, params)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, TransitionwithParams(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        dynamics_params=dynamics_params,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation= nstate.obs,
        extras={'policy_extras': policy_extras, 'state_extras': state_extras},
    )
  def get_experience(
      normalizer_params: running_statistics.RunningStatisticsState,
      policy_params: Params,
      noise_scales: jnp.ndarray,
      env_state: envs.State,
      dynamics_params: jnp.ndarray, #[num_envs(local), dynamics_param_size]
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[
      running_statistics.RunningStatisticsState,
      envs.State,
      ReplayBufferState,
  ]:
    noise_key, key = jax.random.split(key)
    policy = make_policy((normalizer_params, policy_params))
    env_state, transitions = adv_step(
        env, env_state, dynamics_params, policy, noise_scales, key, extra_fields=('truncation',)
    )

    normalizer_params = running_statistics.update(
        normalizer_params,
        transitions.observation,
        pmap_axis_name=_PMAP_AXIS_NAME,
    )
    noise_scales = (1-env_state.done)* noise_scales + \
          env_state.done* (jax.random.normal(noise_key, shape=noise_scales.shape) *(std_max - std_min) + std_min)
    

    simul_info ={
      "simul/reward_mean" : transitions.reward.mean(),
      "simul/reward_std" : transitions.reward.std(),
      "simul/reward_max" : transitions.reward.max(),
      "simul/reward_min" : transitions.reward.min(),
      "simul/dynamics_params_mean" : dynamics_params.mean(),
      "simul/dynamics_params_std" : dynamics_params.std(),

    }
    buffer_state = replay_buffer.insert(buffer_state, transitions)
    return normalizer_params, noise_scales, env_state, buffer_state, simul_info

  def training_step(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[
      TrainingState,
      envs.State,
      ReplayBufferState,
      Metrics,
  ]:
    experience_key, param_key, training_key = jax.random.split(key, 3)
    dynamics_params, logp = flowtd3_network.flow_network.apply(
        training_state.flow_params,
        low=dr_range_low,
        high=dr_range_high,
        mode='sample',
        rng=param_key,
        n_samples=num_envs // jax.process_count() // local_devices_to_use,
    )
    dynamics_params = jax.lax.stop_gradient(dynamics_params)
    normalizer_params, noise_scales, env_state, buffer_state, simul_info = get_experience(
        training_state.normalizer_params,
        training_state.policy_params,
        training_state.noise_scales,
        env_state,
        dynamics_params,
        buffer_state,
        experience_key,
    )
    training_state = training_state.replace(
        normalizer_params=normalizer_params,
        noise_scales = noise_scales,
        env_steps=training_state.env_steps + env_steps_per_actor_step,
    )

    buffer_state, transitions = replay_buffer.sample(buffer_state)
    # Change the front dimension of transitions so 'update_step' is called
    # grad_updates_per_step times by the scan.
    transitions = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
        transitions,
    )
    (training_state, _), metrics = jax.lax.scan(
        sgd_step, (training_state, training_key), transitions
    )

    metrics['buffer_current_size'] = replay_buffer.size(buffer_state)
    metrics.update(simul_info)
    return training_state, env_state, buffer_state, metrics

  def prefill_replay_buffer(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:

    def f(carry, params):
      training_state, env_state, buffer_state, key = carry
      key, new_key = jax.random.split(key)
      new_normalizer_params, new_noise_scales, env_state, buffer_state, simul_info = get_experience(
          training_state.normalizer_params,
          training_state.policy_params,
          training_state.noise_scales,
          env_state,
          params,
          buffer_state,
          key,
      )
      new_training_state = training_state.replace(
          normalizer_params=new_normalizer_params,
          noise_scales = new_noise_scales,
          env_steps=training_state.env_steps + env_steps_per_actor_step,
      )
      return (new_training_state, env_state, buffer_state, new_key), ()
    param_key, key = jax.random.split(key)
    dynamics_params, _= flowtd3_network.flow_network.apply(
        training_state.flow_params, low=dr_range_low, high=dr_range_high, mode='sample', rng=param_key, \
          n_samples=num_prefill_actor_steps * num_envs // jax.process_count() // local_devices_to_use
        )
    dynamics_params = jax.lax.stop_gradient(dynamics_params) 
    # dynamics_params = jax.random.uniform(param_key, shape=(num_prefill_actor_steps * num_envs // jax.process_count() // local_devices_to_use, len(dr_range_low)), minval=dr_range_low, maxval=dr_range_high)

    dynamics_params = jnp.reshape(
        dynamics_params, (num_prefill_actor_steps, num_envs // jax.process_count() // local_devices_to_use) + dynamics_params.shape[1:]
    )
    return jax.lax.scan(
        f,
        (training_state, env_state, buffer_state, key),
        dynamics_params,
        length=num_prefill_actor_steps,
    )[0]

  prefill_replay_buffer = jax.pmap(
      prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME
  )

  def training_epoch(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:

    def f(carry, unused_t):
      ts, es, bs, k = carry
      k, new_key = jax.random.split(k)
      ts, es, bs, metrics = training_step(ts, es, bs, k)
      return (ts, es, bs, new_key), metrics

    (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
        f,
        (training_state, env_state, buffer_state, key),
        (),
        length=num_training_steps_per_epoch,
    )
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return training_state, env_state, buffer_state, metrics

  training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
    nonlocal training_walltime
    t = time.time()
    (training_state, env_state, buffer_state, metrics) = training_epoch(
        training_state, env_state, buffer_state, key
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
    return training_state, env_state, buffer_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade

  global_key, local_key = jax.random.split(rng)
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, rb_key, env_key, param_key, eval_key = jax.random.split(local_key, 5)

    # Env init
  env_keys = jax.random.split(env_key, num_envs // jax.process_count())
  env_keys = jnp.reshape(
      env_keys, (local_devices_to_use, -1) + env_keys.shape[1:]
  )

  dynamics_params = jnp.zeros((local_devices_to_use, num_envs//jax.process_count(), len(dr_range_low)))
  # dynamics_params = jnp.reshape(dynamics_params, (local_devices_to_use, -1) + dynamics_params.shape[1:])
  env_state = jax.pmap(env.reset)(env_keys, dynamics_params)
  obs_shape = jax.tree_util.tree_map(
      lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs

  )

  print("flowtd3 OBS SHAPE2", obs_shape)
  # Training state init
  training_state = _init_training_state(
      key=global_key,
      obs_size=obs_shape,
      local_devices_to_use=local_devices_to_use,
      flowtd3_network=flowtd3_network,
      policy_optimizer=policy_optimizer,
      q_optimizer=q_optimizer,
      flow_optimizer=flow_optimizer,
      num_envs=num_envs,
      std_max=std_max,
      std_min=std_min,
  )
  del global_key
  dynamics_params, _ = flowtd3_network.flow_network.apply(
    _unpmap(training_state.flow_params),
    low=dr_range_low,
    high=dr_range_high,
    mode='sample',
    rng=param_key,
    n_samples=num_eval_envs,
  )
  dynamics_params = jax.lax.stop_gradient(dynamics_params)
  # dynamics_params = jnp.reshape(dynamics_params, (local_devices_to_use, -1) + dynamics_params.shape[1:])
  env_state = jax.pmap(env.reset)(env_keys, dynamics_params[None,...])
  if restore_checkpoint_path is not None:
    params = checkpoint.load(restore_checkpoint_path)
    training_state = training_state.replace(
        normalizer_params=params[0],
        policy_params=params[1],
        noise_scales=params[2],
    )


  # Replay buffer init
  buffer_state = jax.pmap(replay_buffer.init)(
      jax.random.split(rb_key, local_devices_to_use)
  )

  eval_env = copy.deepcopy(environment)
  if eval_with_training_env:
    eval_env = wrap_for_adv_training(
      eval_env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=functools.partial(randomization_fn,dr_range=env.dr_range),
    )
    evaluator = AdvEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=True),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )
  else:
    v_randomization_fn=functools.partial(randomization_fn, 
      rng=jax.random.split(key, num_eval_envs // jax.process_count()//local_devices_to_use), dr_range=env.dr_range
    )

    eval_env = wrap_for_brax_training(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )  # pytype: disable=wrong-keyword-args

    evaluator = Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=True),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )
  ed = time.time()
  print("setup time", ed-st)

  
  if process_id==0:
    fig1 = render_flow_pdf_1d_subplots(
            flowtd3_network.flow_network,
              _unpmap(training_state.flow_params),
              ndim=dr_range_low.shape[0],
              low=dr_range_low,
              high=dr_range_high,
              training_step=0,
              use_wandb=use_wandb,
        )

  # Create and initialize the replay buffer.
  t = time.time()
  prefill_key, local_key = jax.random.split(local_key)
  prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
  training_state, env_state, buffer_state, _ = prefill_replay_buffer(
      training_state, env_state, buffer_state, prefill_keys
  )

  replay_size = (
      jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
  )
  logging.info('replay size after prefill %s', replay_size)
  assert replay_size >= min_replay_size
  training_walltime = time.time() - t

  current_step = 0
  for _ in range(num_evals_after_init):
    logging.info('step %s', current_step)

    # Optimization
    epoch_key, local_key = jax.random.split(local_key)
    epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
    (training_state, env_state, buffer_state, training_metrics) = (
        training_epoch_with_timing(
            training_state, env_state, buffer_state, epoch_keys
        )
    )
    current_step = int(_unpmap(training_state.env_steps))

    # Eval and logging
    if process_id == 0:
      if checkpoint_logdir:
        params = _unpmap(
            (training_state.normalizer_params, training_state.policy_params, training_state.noise_scales)
        )
        ckpt_config = checkpoint.network_config(
            observation_size=obs_shape,
            action_size=env.action_size,
            normalize_observations=normalize_observations,
            network_factory=network_factory,
        )
        checkpoint.save(checkpoint_logdir, current_step, params, ckpt_config)
      fig = render_flow_pdf_1d_subplots(
        flowtd3_network.flow_network,
            _unpmap(training_state.flow_params),
          ndim=dr_range_low.shape[0],
          low=dr_range_low,
          high=dr_range_high,
          training_step=current_step,
          use_wandb=use_wandb,
      )
      # Run evals.
      if eval_with_training_env:
        param_key, local_key = jax.random.split(local_key)
        dynamics_params, _ = flowtd3_network.flow_network.apply(
          _unpmap(training_state.flow_params),
          low=dr_range_low,
          high=dr_range_high,
          mode='sample',
          rng=param_key,
          n_samples=num_eval_envs,
        )
        dynamics_params = jax.lax.stop_gradient(dynamics_params)
        local_key, test_key = jax.random.split(local_key)
        env_keys = jax.random.split(test_key, num_eval_envs)
        env_keys = jnp.reshape(
            env_keys, (local_devices_to_use, -1) + env_keys.shape[1:]
        )
        # jax.pmap(eval_env.reset)(env_keys)
        metrics = evaluator.run_evaluation(
          _unpmap(
                (training_state.normalizer_params, training_state.policy_params),
          ),
            dynamics_params,
            training_metrics,
        )
      else:
        metrics = evaluator.run_evaluation(
          _unpmap(
                (training_state.normalizer_params, training_state.policy_params)
          ),
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

  params = _unpmap(
      (training_state.normalizer_params, training_state.policy_params)
  )

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()
  return (functools.partial(make_policy, deterministic=True), params, metrics)
