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

from typing import Any

from brax.training import types
from agents.flowtd3 import networks as flowtd3_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp



def make_losses(
    flowtd3_network: flowtd3_networks.FlowTd3Networks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
):
  """Creates the td3 losses."""

  target_entropy = -0.5 * action_size
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

    
  def flow_loss(
      flow_params: Params,
      policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      transitions: Any,
      dr_range_high,
      dr_range_low,
      lmbda_params:Params,
      key: jax.random.PRNGKey,
  ):
    """Loss for training the flow network to generate adversarial dynamics parameters."""
    # If dr_low/dr_high are 1-D (shape: [D]) → scalar
    data_log_prob = flow_network.apply(
        flow_params,
        mode='log_prob',
        low=dr_range_low,
        high=dr_range_high,
        x=transitions.dynamics_params,      # <- pass the data here
    )
    data_log_prob = jnp.clip(data_log_prob, -1e6, 1e6)
    _, current_logp = flow_network.apply(
        flow_params,
        mode='sample',
        low=dr_range_low,
        high=dr_range_high,
        n_samples=1000,
        rng=key,
    )
    # kl_loss = volume*(jnp.exp(current_logp)*current_logp).mean()
    kl_loss = current_logp.mean()
    # Get next action and log prob from policy for adversarial observation
    next_action = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation
    )

    # Get Q-value for adversarial next state-action pair
    next_q_adv = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_action).min(-1)
    
    normalized_next_v_adv = (jax.lax.stop_gradient(1/next_q_adv.mean())) * next_q_adv
    # value_loss = volume*(jnp.exp(data_log_prob)*data_log_prob * normalized_next_v_adv).mean()
    value_loss = (data_log_prob * transitions.discount* normalized_next_v_adv).mean()
    return lmbda_params* value_loss + kl_loss, (next_q_adv, value_loss, kl_loss)
  return critic_loss, actor_loss, flow_loss
