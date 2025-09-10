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

"""Soft Actor-Critic losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

from typing import Any

from brax.training import types
from agents.flowsac import networks as flowsac_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.training.distribution import NormalDistribution
import jax
import jax.numpy as jnp
from mujoco_playground._src.wrapper import Wrapper

Transition = types.Transition


def make_losses(
    flowsac_network: flowsac_networks.FLOWSACNetworks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
):
  """Creates the FLOWSAC losses."""

  target_entropy = -0.5 * action_size
  policy_network = flowsac_network.policy_network
  q_network = flowsac_network.q_network
  flow_network = flowsac_network.flow_network
#   lmbda_network = flowsac_network.lmbda_network
  parametric_action_distribution = flowsac_network.parametric_action_distribution

  def alpha_loss(
      log_alpha: jnp.ndarray,
      policy_params: Params,
      normalizer_params: Any,
      transitions: Transition,
      key: PRNGKey,
  ) -> jnp.ndarray:
    """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
    dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.observation
    )
    action = parametric_action_distribution.sample_no_postprocessing(
        dist_params, key
    )
    log_prob = parametric_action_distribution.log_prob(dist_params, action)
    alpha = jnp.exp(log_alpha)
    alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
    return jnp.mean(alpha_loss)

  def lmbda_loss(
      lmbda_params: jnp.ndarray,
      value_loss,
      kl_loss,
      delta : float, 
      prev_loss: jnp.ndarray,  # Add prev_loss parameter
  ):
    lmbda = jnp.maximum(lmbda_params, 0.0)
    # with n_nominals next_observations
    loss =  value_loss + lmbda* (kl_loss - delta)
    loss = -loss.mean()
    # Compute loss reduction info
    loss_info = loss - prev_loss

    return loss, loss_info

  def critic_loss(
      q_params: Params,
      target_q_params : Params,
      policy_params : Params,
      normalizer_params: Any,
      transitions: Transition,
      alpha : jnp.ndarray,
    #   next_v : jnp.ndarray,
      key,
  ) -> jnp.ndarray:
    
    q_old_action = q_network.apply(
        normalizer_params, q_params, transitions.observation, transitions.action
    )
    next_dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation
    )
    next_action = parametric_action_distribution.sample_no_postprocessing(
        next_dist_params, key
    )
    next_log_prob = parametric_action_distribution.log_prob(
        next_dist_params, next_action
    )
    next_action = parametric_action_distribution.postprocess(next_action)
    next_q = q_network.apply(
        normalizer_params,
        target_q_params,
        transitions.next_observation,
        next_action,
    )
    next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
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
      alpha: jnp.ndarray,
      transitions: Transition,
      key: PRNGKey,
  ) -> jnp.ndarray:
    dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.observation
    )
    action = parametric_action_distribution.sample_no_postprocessing(
        dist_params, key
    )
    log_prob = parametric_action_distribution.log_prob(dist_params, action)
    action = parametric_action_distribution.postprocess(action)
    q_action = q_network.apply(
        normalizer_params, q_params, transitions.observation, action
    )
    min_q = jnp.min(q_action, axis=-1)
    actor_loss = alpha * log_prob - min_q
    return jnp.mean(actor_loss)


  def flow_loss(
      flow_params: Params,
      target_flow_params: Params,
      policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      alpha: jnp.ndarray,
      transitions: Transition,
      dr_range_high,
      dr_range_low,
      key: jax.random.PRNGKey,
      lmbda_params:Params,
  ):
    """Loss for training the flow network to generate adversarial dynamics parameters."""
    action_key, current_key, target_key = jax.random.split(key, 3)
    volume = jnp.prod(dr_range_high-dr_range_low)
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
        rng=current_key,
    )
    # kl_loss = volume*(jnp.exp(current_logp)*current_logp).mean()
    kl_loss = current_logp.mean()
    proximal_loss = 0
    # target_sample, target_logp = flow_network.apply(
    #     target_flow_params,
    #     mode='sample',
    #     low=dr_range_low,
    #     high=dr_range_high,
    #     n_samples=1000,
    #     rng=target_key,
    # )
    # target_logp = jnp.clip(target_logp, -1e6, 1e6)
    # current_logp_with_target = flow_network.apply(
    #     flow_params,
    #     mode='log_prob',
    #     low=dr_range_low,
    #     high=dr_range_high,
    #     x=target_sample
    # )
    # current_logp_with_target = jnp.clip(current_logp_with_target, -1e6, 1e6)
    # proximal_loss = 0*(target_logp - current_logp_with_target).mean()


    # Get next action and log prob from policy for adversarial observation
    next_dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation
    )
    next_action = parametric_action_distribution.sample_no_postprocessing(
        next_dist_params, action_key
    )
    next_log_prob = parametric_action_distribution.log_prob(
        next_dist_params, next_action
    )
    
    # Get Q-value for adversarial next state-action pair
    next_q_adv = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_action).min(-1)
    
    # Calculate adversarial next V value
    next_v_adv = next_q_adv - alpha * next_log_prob
    normalized_next_v_adv = (jax.lax.stop_gradient(1/next_v_adv.mean())) * next_v_adv
    # value_loss = volume*(jnp.exp(data_log_prob)*data_log_prob * normalized_next_v_adv).mean()
    value_loss = (data_log_prob * normalized_next_v_adv).mean()
    return lmbda_params* value_loss + 0*kl_loss + proximal_loss, (next_v_adv, value_loss, kl_loss)
  def flow_dr_loss(
      flow_params: Params,
      target_flow_params: Params,
      policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      alpha: jnp.ndarray,
      transitions: Transition,
      dr_range_high,
      dr_range_low,
      key: jax.random.PRNGKey,
      lmbda_params:Params,
  ):
    """Loss for training the flow network to generate adversarial dynamics parameters."""
    action_key, current_key, target_key = jax.random.split(key, 3)
    volume = jnp.prod(dr_range_high-dr_range_low)
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
        rng=current_key,
    )
    # kl_loss = volume*(jnp.exp(current_logp)*current_logp).mean()
    kl_loss = current_logp.mean()
    proximal_loss = 0
    # target_sample, target_logp = flow_network.apply(
    #     target_flow_params,
    #     mode='sample',
    #     low=dr_range_low,
    #     high=dr_range_high,
    #     n_samples=1000,
    #     rng=target_key,
    # )
    # target_logp = jnp.clip(target_logp, -1e6, 1e6)
    # current_logp_with_target = flow_network.apply(
    #     flow_params,
    #     mode='log_prob',
    #     low=dr_range_low,
    #     high=dr_range_high,
    #     x=target_sample
    # )
    # current_logp_with_target = jnp.clip(current_logp_with_target, -1e6, 1e6)
    # proximal_loss = 0*(target_logp - current_logp_with_target).mean()


    # Get next action and log prob from policy for adversarial observation
    next_dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation
    )
    next_action = parametric_action_distribution.sample_no_postprocessing(
        next_dist_params, action_key
    )
    next_log_prob = parametric_action_distribution.log_prob(
        next_dist_params, next_action
    )
    
    # Get Q-value for adversarial next state-action pair
    next_q_adv = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_action).min(-1)
    
    # Calculate adversarial next V value
    next_v_adv = next_q_adv - alpha * next_log_prob
    normalized_next_v_adv = (jax.lax.stop_gradient(1/next_v_adv.mean())) * next_v_adv
    value_loss = (data_log_prob * normalized_next_v_adv).mean()
    return lmbda_params* value_loss +  kl_loss + proximal_loss, (next_v_adv, value_loss, kl_loss)
#   def flow_dr_loss(
#       flow_params: Params,
#       target_flow_params: Params,
#       policy_params: Params,
#       normalizer_params: Any,
#       target_q_params: Params,
#       alpha: jnp.ndarray,
#       transitions: Transition,
#       dr_range_high,
#       dr_range_low,
#       key: jax.random.PRNGKey,
#       lmbda_params:Params,
#   ):
#     """Loss for training the flow network to generate adversarial dynamics parameters."""
#     action_key, current_key, target_key = jax.random.split(key, 3)
#     # If dr_low/dr_high are 1-D (shape: [D]) → scalar
#     data_log_prob = flow_network.apply(
#         flow_params,
#         mode='log_prob',
#         low=dr_range_low,
#         high=dr_range_high,
#         x=transitions.dynamics_params,      # <- pass the data here
#     )
#     data_log_prob = jnp.clip(data_log_prob, -1e6, 1e6)
#     _, current_logp = flow_network.apply(
#         flow_params,
#         mode='sample',
#         low=dr_range_low,
#         high=dr_range_high,
#         n_samples=1000,
#         rng=current_key,
#     )
#     kl_loss = current_logp.mean()
#     proximal_loss = 0
#     # target_sample, target_logp = flow_network.apply(
#     #     target_flow_params,
#     #     mode='sample',
#     #     low=dr_range_low,
#     #     high=dr_range_high,
#     #     n_samples=1000,
#     #     rng=target_key,
#     # )
#     # target_logp = jnp.clip(target_logp, -1e6, 1e6)
#     # current_logp_with_target = flow_network.apply(
#     #     flow_params,
#     #     mode='log_prob',
#     #     low=dr_range_low,
#     #     high=dr_range_high,
#     #     x=target_sample
#     # )
#     # current_logp_with_target = jnp.clip(current_logp_with_target, -1e6, 1e6)
#     # proximal_loss = 0*(target_logp - current_logp_with_target).mean()


#     # Get next action and log prob from policy for adversarial observation
#     next_dist_params = policy_network.apply(
#         normalizer_params, policy_params, transitions.next_observation
#     )
#     next_action = parametric_action_distribution.sample_no_postprocessing(
#         next_dist_params, action_key
#     )
#     next_log_prob = parametric_action_distribution.log_prob(
#         next_dist_params, next_action
#     )
    
#     # Get Q-value for adversarial next state-action pair
#     next_q_adv = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_action).min(-1)
    
#     # Calculate adversarial next V value
#     next_v_adv = next_q_adv - alpha * next_log_prob
#     # value_loss = (data_log_prob * next_v_adv).mean()
#     normalized_next_v_adv = 1/(jax.lax.stop_gradient(next_v_adv).mean()) * next_v_adv
#     volume = jnp.prod(dr_range_high - dr_range_low)
#     value_loss = volume*(jnp.exp(data_log_prob)*data_log_prob * normalized_next_v_adv).mean()
#     return -lmbda_params* value_loss + kl_loss + proximal_loss, (next_v_adv, value_loss, kl_loss)
  return lmbda_loss, alpha_loss, critic_loss, actor_loss, flow_loss, flow_dr_loss
