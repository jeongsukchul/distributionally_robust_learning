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
      policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      alpha: jnp.ndarray,
      transitions: Transition,
      n_nominals : int,
      key: PRNGKey,
      lmbda: jnp.ndarray,
      delta : float, 
  ):
    # with n_nominals next_observations
    next_q = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, transitions.next_action)
    print("next_q shape 1", next_q)
    batch_size = transitions.observation.shape[0]
    next_q = next_q.reshape(n_nominals, batch_size)
    next_log_prob = next_log_prob.reshape(n_nominals, batch_size, -1)
    next_obs = transitions.next_observation.reshape(n_nominals, batch_size, -1)

    print("next_q shape 2", next_q)
    
    print("argmax shape", jnp.argmax(next_q, axis=0))
    print("nextobs ", next_obs)
    next_qs = jnp.array([next_q - lmbda * (jnp.expand_dims(next_obs[i],axis=0) \
                                           - next_obs).mean(axis=(0,2)) for i in range(n_nominals) ])  #(n_nominals, batch_size)
    print("next_vs", next_qs.shape)
    loss = lmbda * delta -  next_qs
    next_q = jnp.min(next_qs, axis=0)  #(batch_size,)

    loss = lmbda * delta - next_q

    print("argmin shape", jnp.argmin(next_qs, axis=0))
    print("next_ q shape", next_q)
    next_state = transitions.next_observation[jnp.argmin(next_qs,axis=0), :]
    print("next state shape", next_state)
    next_dist_params = policy_network.apply(
        normalizer_params, policy_params, next_state
    )
    next_action = parametric_action_distribution.sample_no_postprocessing(
        next_dist_params, key
    )
    next_log_prob = parametric_action_distribution.log_prob(
        next_dist_params, next_action
    )
    print('next log prob', next_log_prob)
    next_v = next_q - alpha*next_log_prob
    print("next_log_prob", next_log_prob)
    # next_action = parametric_action_distribution.postprocess(next_action)
    return loss, next_v

  def critic_loss(
      q_params: Params,
      normalizer_params: Any,
      transitions: Transition,
      next_v : jnp.ndarray,
  ) -> jnp.ndarray:
    
    q_old_action = q_network.apply(
        normalizer_params, q_params, transitions.observation, transitions.action
    )

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


  return lmbda_loss, alpha_loss, critic_loss, actor_loss
