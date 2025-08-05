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
from agents.rambo import networks as rambo_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.training.distribution import NormalDistribution
import jax
import jax.numpy as jnp

Transition = types.Transition


def make_losses(
    rambo_network: rambo_networks.RAMBONetworks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
    adv_weight : float,
):
  """Creates the RAMBO losses."""

  target_entropy = -0.5 * action_size
  dynamics_network = rambo_network.dynamics_network
  policy_network = rambo_network.policy_network
  q_network = rambo_network.q_network
  parametric_action_distribution = rambo_network.parametric_action_distribution

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

  def critic_loss(
      q_params: Params,
      policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      alpha: jnp.ndarray,
      transitions: Transition,
      key: PRNGKey,
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
    return q_loss

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

  def dynamics_loss(
      dynamics_params :Params,
      policy_params: Params,
      normalizer_params: Any,
      q_params: Params,
      transitions: Transition,
      termination_fn: Any,
      key: PRNGKey,
  ):

    policy_key, dynamics_key = jax.random.split(key)
    dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.observation
    )
    action = parametric_action_distribution.sample(
        dist_params, policy_key
    )

    #supervised loss + validation loss
    (means, logvar), normal_fn, denormal_fn = dynamics_network.apply(normalizer_params, dynamics_params, transitions.observation, transitions.action)
    assert len(means.shape) == len(logvar.shape) == 3
    target = jnp.concatenate([normal_fn(transitions.next_observation - transitions.observation, normalizer_params), transitions.reward.reshape(-1, 1)], axis=-1)    
    mle_loss = (((means - target) ** 2) * jnp.exp(-logvar)).mean(axis=(1,2))
    var_loss = logvar.sum(0).mean()
    max_logvar = dynamics_params["params"]["max_logvar"]
    min_logvar = dynamics_params["params"]["min_logvar"]
    logvar_diff = (max_logvar - min_logvar).sum()
    supervised_loss = mle_loss.mean() + var_loss + 0.001 * logvar_diff

    #adversary loss 
    means = jnp.concatenate([means[..., :-1] + normal_fn(transitions.observation, normalizer_params) , means[..., -1:]], axis=-1)
    dist = NormalDistribution(loc=means, scale=jnp.exp(0.5*logvar))
    samples = dist.sample(dynamics_key)
    n_ensemble = len(means)
    next_observation = samples[..., :-1].mean(axis=0)
    next_observation = denormal_fn(next_observation, normalizer_params)
    rewards = samples[..., -1].mean(axis=0)
    terminals = termination_fn(transitions.observation, action, next_observation)
    #1) calculate log probs
    dynamics_log_prob = dist.log_prob(samples).sum(-1)
    max_log_prob = jnp.max(dynamics_log_prob, axis=0, keepdims=True)
    dynamics_log_prob = jax.nn.logsumexp(dynamics_log_prob - max_log_prob, axis=0) + max_log_prob.squeeze() - jnp.log(n_ensemble)
    #2) calculate advantages
    dist_params = policy_network.apply(
        normalizer_params, policy_params, next_observation
    )
    next_action = parametric_action_distribution.sample(
        dist_params, policy_key
    )
    min_qn = q_network.apply(
        normalizer_params, q_params, next_observation, next_action
    ).min(axis=-1)
    min_q = q_network.apply(
        normalizer_params, q_params, transitions.observation, transitions.action
    ).min(axis=-1)
    target_q = rewards * reward_scaling + (1 - terminals) * discounting * min_qn
    advantage = target_q - min_q
    advantage = (advantage - advantage.mean()) / jnp.maximum(advantage.std(), 1e-6)#normalizing advantages
    advantage = jax.lax.stop_gradient(advantage)
    adv_loss = (dynamics_log_prob * advantage).mean()
    
    loss_info = {
      "supervised_loss" : supervised_loss,
      "adv_loss" : adv_loss,
    }
    return supervised_loss + adv_weight * adv_loss, loss_info

  return dynamics_loss, alpha_loss, critic_loss, actor_loss
