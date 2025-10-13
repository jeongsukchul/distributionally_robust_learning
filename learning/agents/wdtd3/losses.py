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
from agents.wdtd3 import networks as wdtd3_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.training.distribution import NormalDistribution
import jax
import jax.numpy as jnp
from mujoco_playground._src.wrapper import Wrapper

Any = types.Any

def make_losses(
    wdtd3_network: wdtd3_networks.WDTD3Networks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
):
  """Creates the WDTD3 losses."""

  policy_network = wdtd3_network.policy_network
  q_network = wdtd3_network.q_network
#   lmbda_network = wdtd3_network.lmbda_network

  def lmbda_loss(
      lmbda_params: jnp.ndarray,
      policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      transitions: Any,
      n_nominals : int,
      noise: jnp.ndarray,
      delta : float, 
      prev_loss: jnp.ndarray,  # Add prev_loss parameter
      key: PRNGKey,
  ):
    batch_size = transitions.action.shape[0]
    # with n_nominals next_observations
    lmbda = jnp.maximum(lmbda_params, 0.0)
    next_action = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation
    )
    next_action = jnp.clip(next_action + noise, -1.0, 1.0)
    next_q = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_action).min(-1) * transitions.discount
    print("transitions.next_observation", transitions.next_observation)

    def penalized_v(i):
        obs_i = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x[:, i, :], axis=1), transitions.next_observation)  # shape: (batch_size, 1, obs_dim)
        diff = obs_i["state"] - transitions.next_observation["state"]  # shape: (batch_size, n_nominals, obs_dim)
        penalty = jnp.mean(jnp.square(diff), axis=(1, 2))  # shape: (batch_size,)
        return next_q[:, i] + lmbda * penalty  # shape: (batch_size,)

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
      transitions: Any,
      n_nominals : int,
      noise: jnp.ndarray,
      delta : float, 
      prev_loss: jnp.ndarray,  # Add prev_loss parameter
      key: PRNGKey,
  ):
    batch_size = transitions.action.shape[0]
    idx_key, key = jax.random.split(key)
    # with n_nominals next_observationsey
    lmbda = jnp.maximum(lmbda_params, 0.0)
    next_action = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation)
    next_action = jnp.clip(next_action+noise, -1.0, 1.0)
    next_q = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_action).min(-1) * transitions.discount
    next_v = -lmbda * (jax.nn.logsumexp(-next_q/jnp.expand_dims(lmbda,axis=-1), axis=-1) - jnp.log(n_nominals))  -lmbda * delta
    loss = -next_v.mean()
    min_indices = jax.random.randint(idx_key, (batch_size,), 0, n_nominals)
    
    # Compute loss reduction info
    loss_info = loss - prev_loss

    return loss, (next_v, min_indices, loss_info)
  def tv_lmbda_loss(
      lmbda_params: jnp.ndarray,
      policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      transitions: Any,
      n_nominals : int,
      noise: jnp.ndarray,
      delta : float, 
      prev_loss: jnp.ndarray,  # Add prev_loss parameter
      key: PRNGKey,
  ):
    idx_key, key = jax.random.split(key)
    batch_size = transitions.action.shape[0]
    # with n_nominals next_observations
    lmbda = jnp.maximum(lmbda_params, 0.0)
    next_action = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation)
    next_action = jnp.clip(next_action+noise, -1.0, 1.0)
    next_q = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_action).min(-1) * transitions.discount
    next_v = -lmbda * jnp.max(jnp.maximum(jnp.expand_dims(lmbda, axis=-1) - next_q,0.))+(1-delta)*lmbda 
    # next_v = -lmbda * (jax.nn.logsumexp(-next_vs/jnp.expand_dims(lmbda,axis=-1), axis=-1) - jnp.log(n_nominals))  -lmbda * delta
    loss = -next_v.mean()
    min_indices = jax.random.randint(idx_key, (batch_size,), 0, n_nominals)
    # Compute loss reduction info
    loss_info = loss - prev_loss
    
    return loss, (next_v, min_indices, loss_info)

  def critic_loss(
      q_params: Params,
      normalizer_params: Any,
      transitions: Any,
      next_v : jnp.ndarray,
      min_indices : jnp.ndarray,
  ) -> jnp.ndarray:
    print("next v shape",next_v)
    batch_size = transitions.action.shape[0]
    q_old_action = q_network.apply(
        normalizer_params, q_params, transitions.observation, transitions.action
    )
    print("q old action", q_old_action)
    reward = jnp.mean(transitions.reward, -1) #[jnp.arange(batch_size), min_indices]
    print("reward", reward)
    target_q = jax.lax.stop_gradient(
        reward* reward_scaling
        + discounting * next_v
    )     
    # Better bootstrapping for truncated episodes.
    # truncation = transitions.extras['state_extras']['truncation']
    # target_q *= 1-truncation
    # mask = 1- truncation
    # counts = mask.sum(axis=-1, keepdims=True).clip(min=1) 
    # print("counts", counts)
    # q_error = q_old_action - target_q.sum(axis=-1, keepdims=True)/counts
    # q_error *= mask.sum(axis=-1, keepdims=True).clip(max=1)
    # print("q error",q_error)
    q_error = q_old_action - jnp.expand_dims(target_q, -1)

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


  return lmbda_loss, kl_lmbda_loss, tv_lmbda_loss, critic_loss, actor_loss
