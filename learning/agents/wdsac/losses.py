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
from agents.wdsac import networks as wdsac_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.training.distribution import NormalDistribution
import jax
import jax.numpy as jnp
from mujoco_playground._src.wrapper import Wrapper

Transition = types.Transition


def make_losses(
    wdsac_network: wdsac_networks.WDSACNetworks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
):
  """Creates the WDSAC losses."""

  target_entropy = -0.5 * action_size
  policy_network = wdsac_network.policy_network
  q_network = wdsac_network.q_network
#   lmbda_network = wdsac_network.lmbda_network
  parametric_action_distribution = wdsac_network.parametric_action_distribution

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
      policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      alpha: jnp.ndarray,
      transitions: Transition,
      n_nominals : int,
      key: PRNGKey,
      delta : float, 
      prev_loss: jnp.ndarray,  # Add prev_loss parameter
  ):
    batch_size = transitions.action.shape[0]
    # with n_nominals next_observations
    lmbda = jnp.maximum(lmbda_params, 0.0)
    next_dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation
    )
    next_action = parametric_action_distribution.sample_no_postprocessing(
        next_dist_params, key
    )
    next_log_prob = parametric_action_distribution.log_prob(
        next_dist_params, next_action
    )
    next_q = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_action).min(-1)
    next_v = next_q - alpha*next_log_prob
    def penalized_v(i):
        obs_i = jnp.expand_dims(transitions.next_observation[:, i, :], axis=1)  # shape: (batch_size, 1, obs_dim)
        diff = obs_i - transitions.next_observation  # shape: (batch_size, n_nominals, obs_dim)
        penalty = jnp.mean(jnp.square(diff), axis=(1, 2))  # shape: (batch_size,)
        return next_v[:, i] + lmbda * penalty  # shape: (batch_size,)

    next_vs = jax.vmap(penalized_v)(jnp.arange(n_nominals))  # shape: (n_nominals, batch_size)
    next_vs = next_vs.T  # shape: (batch_size, n_nominals)
    next_v = jnp.min(next_vs, axis=1)  #(batch_size,)
    min_indices = jnp.argmin(next_vs, axis=1)
    next_v = -lmbda * delta**2 + next_v
    loss = -next_v.mean()
    
    # Compute loss reduction info
    loss_info = loss - prev_loss

    return loss, (next_v, min_indices, loss_info)

  def kl_lmbda_loss(
      lmbda_params: jnp.ndarray,
      policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      alpha: jnp.ndarray,
      transitions: Transition,
      n_nominals : int,
      key: PRNGKey,
      delta : float, 
      prev_loss: jnp.ndarray,  # Add prev_loss parameter
  ):
    batch_size = transitions.action.shape[0]
    idx_key, key = jax.random.split(key)
    # with n_nominals next_observationsey
    lmbda = jnp.maximum(lmbda_params, 0.0)
    print("next obs", transitions.next_observation)
    next_dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation)
    next_action = parametric_action_distribution.sample_no_postprocessing(
        next_dist_params, key
    )
    next_log_prob = parametric_action_distribution.log_prob(
        next_dist_params, next_action
    )
    print("next action", next_action)
    print("next log prob", next_log_prob)
    next_q = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_action).min(-1)
    next_vs = next_q - alpha*next_log_prob          #(batch size, n_nominals)
    next_v = -lmbda * (jax.nn.logsumexp(-next_vs/jnp.expand_dims(lmbda,axis=-1), axis=-1) - jnp.log(n_nominals))  -lmbda * delta
    loss = -next_v.mean()
    min_indices = jax.random.randint(idx_key, (batch_size,), 0, n_nominals)
    
    # Compute loss reduction info
    loss_info = loss - prev_loss
    print("next_q", next_q)
    print("next_vs", next_vs)
    print("next_v", next_v)

    
    return loss, (next_vs, min_indices, loss_info)
  def tv_lmbda_loss(
      lmbda_params: jnp.ndarray,
      policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      alpha: jnp.ndarray,
      transitions: Transition,
      n_nominals : int,
      key: PRNGKey,
      delta : float, 
      prev_loss: jnp.ndarray,  # Add prev_loss parameter
  ):
    idx_key, key = jax.random.split(key)
    batch_size = transitions.action.shape[0]
    # with n_nominals next_observations
    lmbda = jnp.maximum(lmbda_params, 0.0)
    next_dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation)
    next_action = parametric_action_distribution.sample_no_postprocessing(
        next_dist_params, key
    )
    next_log_prob = parametric_action_distribution.log_prob(
        next_dist_params, next_action
    )
    next_q = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_action).min(-1)
    next_vs = next_q - alpha*next_log_prob          #(batch size, n_nominals)
    next_v = -lmbda * jnp.max(jnp.maximum(jnp.expand_dims(lmbda, axis=-1) - next_vs,0.))+(1-delta)*lmbda 
    # next_v = -lmbda * (jax.nn.logsumexp(-next_vs/jnp.expand_dims(lmbda,axis=-1), axis=-1) - jnp.log(n_nominals))  -lmbda * delta
    loss = -next_v.mean()
    min_indices = jax.random.randint(idx_key, (batch_size,), 0, n_nominals)

    # Compute loss reduction info
    loss_info = loss - prev_loss
    
    return loss, (next_vs, min_indices, loss_info)

  def critic_loss(
      q_params: Params,
      normalizer_params: Any,
      transitions: Transition,
      next_vs : jnp.ndarray,
      min_indices : jnp.ndarray,
  ) -> jnp.ndarray:
    
    batch_size = transitions.action.shape[0]
    q_old_action = q_network.apply(
        normalizer_params, q_params, transitions.observation, transitions.action
    )
    discounted_next_v = transitions.discount * next_vs
    
    reward = transitions.reward#[jnp.arange(batch_size), min_indices]
    target_q = jax.lax.stop_gradient(
        reward* reward_scaling
        + discounting * discounted_next_v
    )     
    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    target_q *= 1-truncation
    mask = 1- truncation
    counts = mask.sum(axis=-1, keepdims=True).clip(min=1) 
    print("counts", counts)
    q_error = q_old_action - target_q.sum(axis=-1, keepdims=True)/counts
    q_error *= mask.sum(axis=-1, keepdims=True).clip(max=1)
    print("q error",q_error)
    # q_error = jnp.expand_dims(q_old_action - jnp.expand_dims(target_q, -1)

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


  return lmbda_loss, kl_lmbda_loss, tv_lmbda_loss, alpha_loss, critic_loss, actor_loss
