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

"""TD3 losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

from typing import Any, Sequence, Tuple


from brax.training import types
from agents.flowtd3 import networks as flowtd3_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp
from brax import envs


_PMAP_AXIS_NAME = 'i'
from brax.training.acme import running_statistics
from brax.envs.base import Wrapper, Env, State


def make_losses(
    flowtd3_network: flowtd3_networks.FlowTd3Networks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
    dynamics_param_size: int,
    fab_online: bool = False,
    alpha:int = 2,
    use_advantage: bool = True,
    use_normalization: bool = True,
):
  """Creates the td3 losses."""

  policy_network = flowtd3_network.policy_network
  q_network = flowtd3_network.q_network
  flow_network = flowtd3_network.flow_network


  def critic_loss(
      q_params: Params,
      policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      transitions: Any,
      noise: jnp.ndarray,
      key: PRNGKey,
  ) -> jnp.ndarray:
    q_old_action = q_network.apply(
        normalizer_params, q_params, transitions.observation, transitions.action
    )
    next_action = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation
    )
    next_action = jnp.clip(next_action + noise, -1.0, 1.0)
    next_q = q_network.apply(
        normalizer_params,
        target_q_params,
        transitions.next_observation,
        next_action,
    )
    next_v = jnp.min(next_q, axis=-1) 
    target_q = jax.lax.stop_gradient(
        transitions.reward * reward_scaling
        + transitions.discount * discounting * next_v
    )
    q_error = q_old_action - jnp.expand_dims(target_q, -1)

    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    q_error *= jnp.expand_dims(1 - truncation, -1)

    q_loss = 0.5 * jnp.mean(jnp.square(q_error))
    return q_loss, (q_old_action, next_v)

  def actor_loss(
      policy_params: Params,
      normalizer_params: Any,
      q_params: Params,
      transitions: Any,
      key: PRNGKey,
  ) -> jnp.ndarray:
    action = policy_network.apply(
        normalizer_params, policy_params, transitions.observation
    )
    q_action = q_network.apply(
        normalizer_params, q_params, transitions.observation, action
    )
    min_q = jnp.min(q_action, axis=-1)
    return -jnp.mean(min_q)

  def fab_loss(
    flow_params: Params,
    normalizer_params: Any,
    policy_params: Params,
    current_q_params: Params,
    dynamics_params: jnp.ndarray,
    transitions: Any,
    dr_range_high,
    dr_range_low,
    lmbda_params:Params,
    key: jax.random.PRNGKey,
  ):
    def log_p_fn(x):
       return q_network.apply(normalizer_params, current_q_params, transitions.next_observation, \
                              policy_network.apply(normalizer_params, policy_params, transitions.next_observation))
    def log_q_fn(x):
       return flow_network.apply(flow_params, mode='log_prob', low=dr_range_low, high=dr_range_high, x=x)
    point, log_w, smc_state, smc_info = smc.step(x0, smc_state, log_q_fn, )

  def flow_loss(
      flow_params: Params,
      normalizer_params: Any,
      policy_params: Params,
      current_q_params: Params,
      dynamics_params: jnp.ndarray,
      transitions: Any,
    #   noise_scales : jnp.ndarray,
    #   env_state,
    #   buffer_state,
      dr_range_high,
      dr_range_low,
      lmbda_params:Params,
      key: jax.random.PRNGKey,
    #   num_envs,
    #   env,
    #  render_flow_pdf_2d_subplots  make_policy,
    # render_flow_pdf_2d_subplots  std_min,
    #   std_max,
  ):
    """Loss for training the flow network to generate adversarial dynamics parameters."""
    # If dr_low/dr_high are 1-D (shape: [D]) → scalar
    data_log_prob = flowtd3_network.flow_network.apply(
        flow_params,
        low=dr_range_low,
        high=dr_range_high,
        mode='log_prob',
        x=dynamics_params,
    )
    data_log_prob = jnp.clip(data_log_prob, -1e6, 1e6)
    # normalizer_params, noise_scales, env_state, buffer_state, simul_info, transitions = get_experience(
    #     normalizer_params,
    #     policy_params,
    #     noise_scales,
    #     env_state,
    #     dynamics_params,
    #     buffer_state,
    #     key2,
    #     env,
    #     replay_buffer,
    #     make_policy,
    #     std_min,
    #     std_max,
    # )

    _, current_logp = flow_network.apply(
        flow_params,
        mode='sample',
        low=dr_range_low,
        high=dr_range_high,
        rng=key,
        n_samples=10000,
    )
    kl_loss = current_logp.mean()
    # Get next action and log prob from policy for adversarial observation
    next_action = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation
    )
    # Get Q-value for adversarial next state-action pair
    next_q_adv = q_network.apply(normalizer_params, current_q_params, transitions.next_observation, next_action).min(-1)
    # next_q_adv = (jax.lax.stop_gradient(1/next_q_adv.mean())) * next_q_adv
    current_q = q_network.apply(normalizer_params, current_q_params, transitions.observation, transitions.action).min(-1)
    # normalized_current_q = (jax.lax.stop_gradient(1/current_q.mean())) * current_q
    # value_loss = volume*(jnp.exp(data_log_prob)*data_log_prob * normalized_next_v_adv).mean()
    truncation = transitions.extras['state_extras']['truncation']
    advantage = transitions.reward + transitions.discount* next_q_adv - current_q
    if use_normalization:
        advantage = (jax.lax.stop_gradient(1/advantage.mean())) * advantage
    advantage *= 1-truncation
    if use_advantage:
        value_loss = (data_log_prob * advantage).mean()
    else: 
        value_loss = (data_log_prob * next_q_adv).mean()
    # return lmbda_params* value_loss + kl_loss, (env_state, buffer_state, normalizer_params, noise_scales, simul_info, value_loss, kl_loss)
    return lmbda_params*value_loss + kl_loss, (value_loss, kl_loss)
  return critic_loss, actor_loss, flow_loss



# def adv_step(
#     env: Env,
#     env_state: State,
#     dynamics_params: jnp.ndarray,
#     policy: Any,
#     noise_scales : jnp.ndarray,
#     key: PRNGKey,
#     extra_fields: Sequence[str] = (),
#   ):
#     step_key, key = jax.random.split(key)
#     actions, policy_extras = policy(env_state.obs, noise_scales, key)
#     # dynamics_params = jax.random.uniform(key=step_key, shape=(num_envs,len(dr_range_low)), minval=dr_range_low, maxval=dr_range_high)
#     params = env_state.info["dr_params"] * (1 - env_state.done[..., None]) + dynamics_params * env_state.done[..., None]
#     nstate = env.step(env_state, actions, params)
#     state_extras = {x: nstate.info[x] for x in extra_fields}
#     return nstate, TransitionwithParams(  # pytype: disable=wrong-arg-types  # jax-ndarray
#         observation=env_state.obs,
#         action=actions,
#         dynamics_params=dynamics_params,
#         reward=nstate.reward,
#         discount=1 - nstate.done,
#         next_observation= nstate.obs,
#         extras={'policy_extras': policy_extras, 'state_extras': state_extras},
#     )
# def get_experience(
#       normalizer_params: running_statistics.RunningStatisticsState,
#       policy_params: Params,
#       noise_scales: jnp.ndarray,
#       env_state: envs.State,
#       dynamics_params: jnp.ndarray, #[num_envs(local), dynamics_param_size]
#       buffer_state: Any,
#       key: PRNGKey,
#       env,
#       replay_buffer,
#       make_policy,
#       std_min,
#       std_max,
#   ) -> Tuple[
#       running_statistics.RunningStatisticsState,
#       envs.State,
#       Any,
#   ]:
#     noise_key, key = jax.random.split(key)
#     policy = make_policy((normalizer_params, policy_params))
#     env_state, transitions = adv_step(
#         env, env_state, dynamics_params, policy, noise_scales, key, extra_fields=('truncation',)
#     )

#     normalizer_params = running_statistics.update(
#         normalizer_params,
#         transitions.observation,
#         pmap_axis_name=_PMAP_AXIS_NAME,
#     )
#     noise_scales = (1-env_state.done)* noise_scales + \
#           env_state.done* (jax.random.normal(noise_key, shape=noise_scales.shape) *(std_max - std_min) + std_min)
    

#     simul_info ={
#       "simul/reward_mean" : transitions.reward.mean(),
#       "simul/reward_std" : transitions.reward.std(),
#       "simul/reward_max" : transitions.reward.max(),
#       "simul/reward_min" : transitions.reward.min(),
#       "simul/dynamics_params_mean" : dynamics_params.mean(),
#       "simul/dynamics_params_std" : dynamics_params.std(),

#     }
#     buffer_state = replay_buffer.insert(buffer_state, transitions)
    # return normalizer_params, noise_scales, env_state, buffer_state, simul_info, transitions