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
from learning.module.wrapper.adv_wrapper import wrap_for_adv_training
import time
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union

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
from agents.flowsac import checkpoint 
from agents.flowsac import losses as flowsac_losses
from agents.flowsac import networks as flowsac_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.training.acme.types import NestedArray
from brax.envs.base import Wrapper, Env, State
from brax.base import System
import flax
import jax
import jax.numpy as jnp
import optax
from mujoco import mjx
from custom_envs import mjx_env
from brax.envs.wrappers import training as brax_training
from learning.module.wrapper.adv_wrapper import wrap_for_adv_training
from learning.module.wrapper.evaluator import Evaluator, AdvEvaluator
# from module.distribution import render_flow_all_dims_1d_linspace, check_mass
from learning.module.normalizing_flow.simple_flow import render_flow_pdf_1d_subplots
import distrax
Metrics = types.Metrics
Transition = types.Transition

_PMAP_AXIS_NAME = 'i'


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
ReplayBufferState = Any

class TransitionwithParams(NamedTuple):
  """Transition with additional dynamics parameters."""
  observation: jax.Array
  dynamics_params: jax.Array
  action: jax.Array
  reward: jax.Array
  discount: jax.Array
  next_observation: jax.Array
  extras: Dict[str, Any] = {}


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
  alpha_optimizer_state: optax.OptState
  alpha_params: Params
  lmbda_optimizer_state: optax.OptState
  lmbda_params: Params
  flow_optimizer_state: optax.OptState
  flow_params: Params
  target_flow_params: Params
  normalizer_params: running_statistics.RunningStatisticsState


def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)


def _init_training_state(
    key: PRNGKey,
    obs_size: Union[int, Dict[str, specs.Array]],
    local_devices_to_use: int ,
    flowsac_network: flowsac_networks.FLOWSACNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    lmbda_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    flow_optimizer: optax.GradientTransformation,
    single_lambda : bool,
    batch_size : int, 
    init_lmbda : float,
) -> TrainingState:
  """Inits the training state and replicates it over devices."""
  key_policy, key_q, key_lmbda, key_flow = jax.random.split(key,4)
  log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
  alpha_optimizer_state = alpha_optimizer.init(log_alpha)

  # lmbda_params = flowsac_network.lmbda_network.init(key_lmbda)
  if single_lambda:
    lmbda_params =jnp.asarray(init_lmbda, dtype=jnp.float32)
  else:
    lmbda_params = init_lmbda * jnp.ones(batch_size)

  lmbda_optimizer_state = lmbda_optimizer.init(lmbda_params) 

  policy_params = flowsac_network.policy_network.init(key_policy)
  policy_optimizer_state = policy_optimizer.init(policy_params)
  q_params = flowsac_network.q_network.init(key_q)
  q_optimizer_state = q_optimizer.init(q_params)
  
  # Initialize flow network parameters
  flow_params = flowsac_network.flow_network.init({"params": key_flow ,'constant':key_flow})
  flow_optimizer_state = flow_optimizer.init(flow_params)
  normalizer_params = running_statistics.init_state(
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
      alpha_optimizer_state=alpha_optimizer_state,
      alpha_params=log_alpha,
      lmbda_optimizer_state=lmbda_optimizer_state,
      lmbda_params=lmbda_params,
      flow_optimizer_state=flow_optimizer_state,
      flow_params=flow_params,
      target_flow_params=flow_params,
      normalizer_params=normalizer_params,
  )
  return jax.device_put_replicated(
      training_state, jax.local_devices()[:local_devices_to_use]
  )
  return training_state


def train(
    environment: envs.Env,
    num_timesteps,
    episode_length: int,
    wrap_env: bool = True,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    wrap_eval_env_fn: Optional[Callable[[Any], Any]] = None,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 1024,
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
    flow_lr : float = 1e-4,  # Learning rate for flow network
    delta : float = 0.1,                    #added
    lambda_update_steps: int = 100,         #added: number of lambda optimization steps
    lmbda_lr : float = 3e-4,                #added
    init_lmbda : float = 0.,                #added
    dr_flow : bool= False,                  #added
    dr_train_ratio : float= 0.9,          #added
    use_wandb : bool= False, #added
    eval_with_training_env : bool = False, #added
    deterministic_eval: bool = False, 
    network_factory: types.NetworkFactory[
        flowsac_networks.FLOWSACNetworks
    ] = flowsac_networks.make_flowsac_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    checkpoint_logdir: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    
):
  """flowsac training."""
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
  print("local devices to use", local_devices_to_use)
  print("device count", device_count)
  st = time.time()
  if max_replay_size is None:
    max_replay_size = num_timesteps
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
      // (num_evals_after_init * env_steps_per_actor_step))
  if num_envs % device_count != 0:
    raise ValueError(f'num_envs ({num_envs}) must be divisible by device_count ({device_count})')
  num_envs_per_device = num_envs // device_count

  import copy
  env = copy.deepcopy(environment)
    
  rng = jax.random.PRNGKey(seed)
  rng, key = jax.random.split(rng)
  # v_randomization_fn=functools.partial(randomization_fn, 
  #     rng=jax.random.split(key, num_envs // jax.process_count()//local_devices_to_use), 
  # )
  env = wrap_for_adv_training(
      env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=randomization_fn,
  )
  
  obs_size = env.observation_size
  action_size = env.action_size
  if hasattr(env, 'dr_range'):
    dr_range_low, dr_range_high = env.dr_range
    dr_train_ratio = 0.9
    dr_range = dr_range_high - dr_range_low
    dr_range_mid = (dr_range_high + dr_range_low) / 2.0
    dr_range_low, dr_range_high = dr_range_mid - dr_range/2 * dr_train_ratio, dr_range_mid + dr_range/2 * dr_train_ratio
    volume = jnp.prod(jnp.maximum(dr_range_high - dr_range_low, 0.0))
    print("volume : ", volume)
  else:
    # Fallback configuration if environment doesn't have dr_range
    ValueError("Environment does not have dr_range attribute. Please provide a valid environment with dr_range.")
  

  normalize_fn = lambda x, y: x
  if normalize_observations:
    normalize_fn = running_statistics.normalize
  flowsac_network = network_factory(
      observation_size=obs_size,
      action_size=action_size,
      preprocess_observations_fn=normalize_fn,
      dynamics_param_size=len(dr_range_low)
  )

  make_policy = flowsac_networks.make_inference_fn(flowsac_network)
  # Get dynamics configuration from the environment

  alpha_optimizer = optax.adam(learning_rate=3e-4)
  lmbda_optimizer = optax.adam(learning_rate=lmbda_lr)
  policy_optimizer = optax.adam(learning_rate=learning_rate)
  q_optimizer = optax.adam(learning_rate=learning_rate)
  flow_optimizer = optax.adam(learning_rate=flow_lr)  # Flow network optimizer
  dummy_params = jnp.zeros((len(dr_range_low),))  # Dummy dynamics parameters
  dummy_obs = {k: jnp.zeros(obs_size[k]) for k in obs_size} if isinstance(obs_size, dict) else jnp.zeros((obs_size,))
  print("dummy obs", dummy_obs)
  dummy_action = jnp.zeros((action_size,))
  dummy_transition = TransitionwithParams(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=dummy_obs,
      action=dummy_action,
      dynamics_params=dummy_params,
      reward=0.,
      discount=0.,
      next_observation=dummy_obs,
      extras={'state_extras': {'truncation': 0.}, 'policy_extras': {}},
  )
  print("max replay size", max_replay_size // device_count)
  print("N * num_envs", env.episode_length * num_envs // device_count)
  max_replay_size = max(max_replay_size // device_count, env.episode_length * num_envs // device_count)
  replay_buffer = DynamicBatchQueue(
      max_replay_size=max_replay_size,
      dummy_data_sample=dummy_transition,
      sample_batch_size=batch_size * grad_updates_per_step // device_count,
  )
  lmbda_loss, alpha_loss, critic_loss, actor_loss, flow_loss, flow_dr_loss = flowsac_losses.make_losses(
      flowsac_network=flowsac_network,
      reward_scaling=reward_scaling,
      discounting=discounting,
      action_size=action_size,
  )


  lmbda_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        lmbda_loss, lmbda_optimizer ,has_aux=True, pmap_axis_name=_PMAP_AXIS_NAME
    )
  alpha_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      alpha_loss, alpha_optimizer ,pmap_axis_name=_PMAP_AXIS_NAME
  )
  critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      critic_loss, q_optimizer , has_aux=True, pmap_axis_name=_PMAP_AXIS_NAME
  )
  actor_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
  )
  if dr_flow:
    flow_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      flow_dr_loss, flow_optimizer, has_aux=True, pmap_axis_name=_PMAP_AXIS_NAME
    )
  else:
    flow_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        flow_loss, flow_optimizer, has_aux=True, pmap_axis_name=_PMAP_AXIS_NAME
    )

  def sgd_step(
      carry: Tuple[TrainingState, PRNGKey], transitions : Tuple[TransitionwithParams, jnp.ndarray],  
  ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
    training_state, key = carry
    # transitions, gradient_steps = inputs
    key, key_lmbda, key_alpha, key_flow, key_critic, key_actor = jax.random.split(key, 6)

    alpha = jnp.exp(training_state.alpha_params)

    alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
        training_state.alpha_params,
        training_state.policy_params,
        training_state.normalizer_params,
        transitions,
        key_alpha,
        optimizer_state=training_state.alpha_optimizer_state,
    )
    
    # Update flow network to generate adversarial dynamics parameters
    (flow_loss, (next_v_adv, value_loss, kl_loss)), flow_params, flow_optimizer_state = flow_update(
        training_state.flow_params,
        training_state.target_flow_params,
        training_state.policy_params,
        training_state.normalizer_params,
        training_state.target_q_params,
        alpha,
        transitions,
        dr_range_high,
        dr_range_low,
        key_flow,  # Reuse key_actor for flow update
        training_state.lmbda_params,
        optimizer_state=training_state.flow_optimizer_state,
    )
    # def lambda_optimization_step(carry, unused):
    #     lmbda_params, lmbda_optimizer_state, prev_loss = carry
    #     (lmbda_loss, loss_info), new_lmbda_params, new_lmbda_optimizer_state = lmbda_update(
    #         lmbda_params,
    #         value_loss,
    #         kl_loss,
    #         delta,
    #         prev_loss,  # Pass previous loss
    #         optimizer_state=lmbda_optimizer_state,
    #     )
    #     return (new_lmbda_params, new_lmbda_optimizer_state, lmbda_loss), (lmbda_loss, loss_info)
        
    # (lmbda_params, lmbda_optimizer_state, _), (lmbda_loss, loss_info) = jax.lax.scan(
    #       lambda_optimization_step,
    #       (training_state.lmbda_params, training_state.lmbda_optimizer_state, jnp.array(0.0)),  # Initialize prev_loss as 0
    #       None,
    #       length=lambda_update_steps  # number of lambda optimization steps
    #   )
    # lmbda_loss = lmbda_loss[-1]
    # loss_info = loss_info[-1]  # Get the final loss reduction info
    (critic_loss, (current_q, next_v)), q_params, q_optimizer_state = critic_update(
        training_state.q_params,
        training_state.target_q_params,
        training_state.policy_params,
        training_state.normalizer_params,
        transitions,
        alpha,
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
    newt_target_flow_params = jax.tree_util.tree_map(
        lambda x, y: x * (1 - tau) + y * tau,
        training_state.target_flow_params,
        flow_params,
    )

    metrics = {
        # 'lmbda_loss' : lmbda_loss,
        # 'lmbda_loss_reduction': loss_info,  # Add loss reduction info
        # 'next_v_adv': next_v_adv.mean(),
        'flow_value_loss': value_loss,
        'flow_kl_loss': kl_loss,
        'critic_loss': critic_loss,
        'actor_loss': actor_loss,
        'flow_loss': flow_loss,  # Add flow network loss
        'alpha_loss': alpha_loss,
        'alpha': jnp.exp(alpha_params),
        # 'lmbda' : lmbda_params
        'current_q_min' : current_q.min(),
        'current_q_max' : current_q.max(),
        'current_q_mean' : current_q.mean(),
        'next_v_min' : next_v.min(),
        'next_v_max' : next_v.max(),
        'next_v_mean' : next_v.mean(),
    }

    new_training_state = training_state.replace(
        # lmbda_optimizer_state=lmbda_optimizer_state,
        # lmbda_params=lmbda_params,
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=new_target_q_params,
        flow_optimizer_state=flow_optimizer_state,
        flow_params=flow_params,
        target_flow_params=newt_target_flow_params,
        gradient_steps=training_state.gradient_steps + 1,
        env_steps=training_state.env_steps,
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=alpha_params,
        normalizer_params=training_state.normalizer_params,
    )
    return (new_training_state, key), metrics


  def adv_step(
    env: envs.Env,
    env_state: envs.State,
    dynamics_params: jnp.ndarray,
    policy: types.Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
  ):
    
    actions, policy_extras = policy(env_state.obs, key)

    nstate = env.step(env_state, actions, dynamics_params)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, TransitionwithParams(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        dynamics_params=dynamics_params,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation= nstate.obs,
        extras={'policy_extras': policy_extras, 'state_extras': state_extras},
      )

  def get_experience(
      normalizer_params: running_statistics.RunningStatisticsState,
      policy_params: Params,
      env_state: envs.State,
      dynamics_params: jnp.ndarray, #[num_envs(local), dynamics_param_size]
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[
      running_statistics.RunningStatisticsState,
      envs.State,
      ReplayBufferState,
  ]:
    policy = make_policy((normalizer_params, policy_params))
    print("dynamics_params in get_experience", dynamics_params)
    next_state, transitions = adv_step(
        env, env_state, dynamics_params, policy, key, extra_fields=('truncation',)
    )

    normalizer_params = running_statistics.update(
        normalizer_params,
        transitions.observation,
        pmap_axis_name=_PMAP_AXIS_NAME,
    )
    simul_info ={
          "simul/reward_mean" : transitions.reward.mean(),
          "simul/reward_std" : transitions.reward.std(),
          "simul/reward_max" : transitions.reward.max(),
          "simul/reward_min" : transitions.reward.min(),
          "simul/params_mean" : transitions.dynamics_params.mean(),
          "simul/params_std" : transitions.dynamics_params.std(),

        }
    buffer_state = replay_buffer.insert(buffer_state, transitions)
    return normalizer_params, next_state, buffer_state, simul_info
  def training_step(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      # gradient_steps: int,
      key: PRNGKey,
  ) -> Tuple[
      TrainingState,
      envs.State,
      ReplayBufferState,
      Metrics,
  ]:
    experience_key, training_key = jax.random.split(key)

    # Sample dynamics parameters from the flow network
    param_key, training_key = jax.random.split(training_key)
    dynamics_params, logp = flowsac_network.flow_network.apply(
        training_state.flow_params,
        low=dr_range_low,
        high=dr_range_high,
        mode='sample',
        rng=param_key,
        n_samples=num_envs // jax.process_count() // local_devices_to_use,
    )
    # u = jax.random.uniform(param_key, (num_envs // jax.process_count() // local_devices_to_use, dr_range_low.shape[0]))#dr_range_low.shape[0]))
    # dynamics_params = dr_range_low[None, ...] + (dr_range_high - dr_range_low)[None, ...] * u
    dynamics_params = jax.lax.stop_gradient(dynamics_params)  # Ensure gradients do not flow through dynamics parameters
    normalizer_params, env_state, buffer_state, simul_info = get_experience(
        training_state.normalizer_params,
        training_state.policy_params,
        env_state,
        dynamics_params,
        buffer_state,
        experience_key,
    )
    training_state = training_state.replace(
        normalizer_params=normalizer_params,
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
        sgd_step, init=(training_state, training_key), xs=transitions,
    )
    metrics.update(simul_info)
    metrics['buffer_current_size'] = replay_buffer.size(buffer_state)  # pytype: disable=unsupported-operands  # lax-types
    # metrics.update(rollout_info)

    return training_state, env_state, buffer_state, metrics
  def prefill_replay_buffer(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:

    def f(carry, params):
      # del unused
      training_state, env_state, buffer_state, key = carry
      key, new_key = jax.random.split(key)
      new_normalizer_params, env_state, buffer_state, simul_info = get_experience(
          training_state.normalizer_params,
          training_state.policy_params,
          env_state,
          params,
          buffer_state,
          key,
      )
      new_training_state = training_state.replace(
          normalizer_params=new_normalizer_params,
          env_steps=training_state.env_steps + env_steps_per_actor_step,
      )
      return (new_training_state, env_state, buffer_state, new_key), ()
    param_key, key = jax.random.split(key)
    dynamics_params, _= flowsac_network.flow_network.apply(
        training_state.flow_params, low=dr_range_low, high=dr_range_high, mode='sample', rng=param_key, \
          n_samples=num_prefill_actor_steps * num_envs // jax.process_count() // local_devices_to_use
        )
    dynamics_params = jax.lax.stop_gradient(dynamics_params)  # Ensure gradients do not flow through dynamics parameters
    # u = jax.random.uniform(param_key, (num_prefill_actor_steps * num_envs // jax.process_count() // local_devices_to_use, dr_range_low.shape[0]))#dr_range_low.shape[0]))
    # dynamics_params = dr_range_low[None, ...] + (dr_range_high - dr_range_low)[None, ...] * u
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
      # gradient_steps : int,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:


    def f(carry, unused):
      ts, es, bs, k = carry
      k, new_key = jax.random.split(k)
      ts, es, bs, metrics = training_step(ts, es, bs, k)
      return (ts, es, bs, new_key), metrics

    (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
        f,
        (training_state, env_state, buffer_state, key),
        (),
        #  gradient_steps + jnp.arange(num_training_steps_per_epoch),
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
    training_state, env_state, buffer_state, metrics = training_epoch(
        training_state, env_state, buffer_state, key#, jnp.ones(local_devices_to_use)* int(training_state.gradient_steps)
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
  local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)
  # env.reset per device
  env_keys = jax.random.split(env_key, num_envs// jax.process_count())
  env_keys = jnp.reshape(
      env_keys, (local_devices_to_use, -1) + env_keys.shape[1:]
  )
  env_state = jax.pmap(env.reset)(env_keys)
  # Build obs specs per device for normalizer init
  obs_specs = jax.tree_util.tree_map(
      lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
  )

  # Training state init
  training_state = _init_training_state(
      key=global_key,
      obs_size=obs_specs,
      local_devices_to_use=local_devices_to_use,
      flowsac_network=flowsac_network,
      alpha_optimizer=alpha_optimizer,
      policy_optimizer=policy_optimizer,
      lmbda_optimizer=lmbda_optimizer,
      q_optimizer=q_optimizer,
      flow_optimizer=flow_optimizer,
      single_lambda=True,
      batch_size=batch_size // grad_updates_per_step,
      init_lmbda= init_lmbda,
  )
  del global_key

  if restore_checkpoint_path is not None:
    params = checkpoint.load(restore_checkpoint_path)
    training_state = training_state.replace(
        normalizer_params=jax.device_put_replicated(params[0], jax.local_devices()[:local_devices_to_use]),
        policy_params=jax.device_put_replicated(params[1], jax.local_devices()[:local_devices_to_use]),
    )


  # Replay buffer init
  buffer_state = jax.pmap(replay_buffer.init)(
      jax.random.split(rb_key, local_devices_to_use)
  ) 
  # buffer_state = replay_buffer.init(rb_key)
  if not eval_env:
    eval_env = copy.deepcopy(environment)
  if eval_with_training_env:
    eval_env = wrap_for_adv_training(
      eval_env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=randomization_fn,
    )
    evaluator = AdvEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )
  else:
    v_randomization_fn=functools.partial(randomization_fn, 
      rng=jax.random.split(key, num_eval_envs // jax.process_count()//local_devices_to_use), params=env.dr_range if hasattr(env,'dr_range')  else None
    )
    eval_env = wrap_eval_env_fn(
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
  # if process_id == 0 and num_evals > 1:
    # if num_evals > 1:
      # st = time.time()
      # metrics = evaluator.run_evaluation(
      #     # _unpmap(
      #         (training_state.normalizer_params, training_state.policy_params),
      #     # ),
      #     training_metrics={},
      # )
      # logging.info(metrics)
      # progress_fn(0, metrics)
      # f
      # print("check mass : ", check_mass(flowsac_network.flow_network, _unpmap(training_state.flow_params), \
      #                                   dr_range_low, dr_range_high, training_step=0))
      # ed = time.time()
      # print("first eval time", ed-st)
  #Create and initialize the replay buffer.
  t = time.time()

  # init with uniform distribution
  flow_pretrain_steps = 3000

  def kl_loss(
      flow_params : Params,
      # beta: float, #annealing parameters
      key: jax.random.PRNGKey,
  ):
    param_key, key = jax.random.split(key)
    dynamics_params, logp = flowsac_network.flow_network.apply(
      flow_params,
      low=dr_range_low,
      high=dr_range_high,
      mode='sample',
      rng=param_key,
      n_samples=10000,)

    logp = jnp.clip(logp, -1e6, 1e6)
    return logp.mean()

    # u = jax.random.uniform(key, (10000, 2))#dr_range_low.shape[0]))
    # x = dr_range_low[:2] + (dr_range_high[:2] - dr_range_low[:2]) * u
    # logp = flowsac_network.flow_network.apply(flow_params, mode="log_prob", x=x, \
    #                                           low=dr_range_low[:2], high=dr_range_high[:2])
    # return -jnp.mean(logp)
  

  pretrain_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      kl_loss, flow_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
  )
  
  def flow_pretrain(
      training_state : TrainingState,
      key: PRNGKey,
  ):
    #---------------------------------------
    def flow_step(
      carry : Tuple[TrainingState, PRNGKey],
      inputs,
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
      training_state, key = carry 
      flow_key, key = jax.random.split(key)
      
      pretrain_loss, flow_params, flow_optimizer_state= pretrain_update(
        training_state.flow_params,
        flow_key,
        optimizer_state=training_state.flow_optimizer_state,
      )
      return (training_state.replace(
        flow_params=flow_params, 
        flow_optimizer_state=flow_optimizer_state), key), {'pretrain_kl_loss': pretrain_loss}
    #---------------------------------------
    (training_state, key ), metrics = jax.lax.scan(
      flow_step, (training_state, key,), (), flow_pretrain_steps
    )
    return training_state, metrics
  flow_pretrain = jax.pmap(
      flow_pretrain, axis_name=_PMAP_AXIS_NAME
  )
  if process_id==0:
    fig1 = render_flow_pdf_1d_subplots(
            flowsac_network.flow_network,
              _unpmap(training_state.flow_params),
              ndim=dr_range_low.shape[0],
              low=dr_range_low,
              high=dr_range_high,
              training_step=0,
              use_wandb=use_wandb,
        )
  if dr_flow:
    pretrain_key, local_key = jax.random.split(local_key)
    pretrain_keys = jax.random.split(pretrain_key, local_devices_to_use)
    training_state, pretrain_metric = flow_pretrain(
        training_state, pretrain_keys
    )

    if process_id==0:
      fig1 = render_flow_pdf_1d_subplots(
              flowsac_network.flow_network,
                _unpmap(training_state.flow_params),
                ndim=dr_range_low.shape[0],
                low=dr_range_low,
                high=dr_range_high,
                training_step=11,
                use_wandb=use_wandb,
          )
    pretrain_time = time.time() - t
    print("flow pretrain time", pretrain_time)
  t2 = time.time()
  prefill_key, local_key = jax.random.split(local_key)
  prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
  training_state, env_state, buffer_state, _ = prefill_replay_buffer(
      training_state, env_state, buffer_state, prefill_keys
  )

  replay_size = (
      jnp.sum(jax.vmap(replay_buffer.size)(buffer_state))  # * jax.process_count()
  )
  logging.info('replay size after prefill %s', replay_size)
  assert replay_size >= min_replay_size
  prefill_time = time.time() - t2
  print("prefill buffer time", prefill_time)
  training_walltime = time.time() - t

  current_step = 0
  for _ in range(num_evals_after_init):
    logging.info('step %s', current_step)


    # Optimization
    epoch_key, local_key = jax.random.split(local_key)
    epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
    training_metrics = {}
    (training_state, env_state, buffer_state, training_metrics) = (
        training_epoch_with_timing(
            training_state, env_state, buffer_state, epoch_keys
        )
    )
    
    current_step = int(_unpmap(training_state.env_steps))
    # current_step = int(training_state.env_steps)
    # Eval and logging
    
    if process_id == 0:
      if checkpoint_logdir:
          params = _unpmap(
              (training_state.normalizer_params, training_state.policy_params)
          )
          ckpt_config = checkpoint.network_config(
              observation_size=obs_size,
              action_size=env.action_size,
              normalize_observations=normalize_observations,
              network_factory=network_factory,
          )
          checkpoint.save(checkpoint_logdir, current_step, params, ckpt_config)
      #plot the normalizing flow pdf
      st = time.time()
      fig = render_flow_pdf_1d_subplots(
        flowsac_network.flow_network,
            _unpmap(training_state.flow_params),
          ndim=dr_range_low.shape[0],
          low=dr_range_low,
          high=dr_range_high,
          training_step=current_step,
          use_wandb=use_wandb,
      )
      ed = time.time()- st
      print("plot rendering time", ed-st)
      # print("check mass : ", check_mass(flowsac_network.flow_network, _unpmap(training_state.flow_params), \
      #                                   dr_range_low, dr_range_high, training_step=current_step))

      # Run evals.
      if eval_with_training_env:
        param_key, local_key = jax.random.split(local_key)
        dynamics_params, _ = flowsac_network.flow_network.apply(
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
  return (make_policy, params, metrics)
