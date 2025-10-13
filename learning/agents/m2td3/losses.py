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

"""M2TD3 losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

from typing import Any

from brax.training import types
from agents.m2td3 import networks as m2td3_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp

Transition = types.Transition


def make_losses(
    m2td3_network: m2td3_networks.M2TD3Networks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
):
  """Creates the m2td3 losses."""

  policy_network = m2td3_network.policy_network
  q_network = m2td3_network.q_network

  def critic_loss(
      q_params: Params,
      policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      transitions: Transition,
      noise: jnp.ndarray,
      dr_low: jnp.ndarray,
      dr_high: jnp.ndarray,
      omega_noise_rate:float,
      omega_clip:float,
      key: PRNGKey,
  ) -> jnp.ndarray:
    q_old_action = q_network.apply(
        normalizer_params, q_params, transitions.observation, transitions.action, transitions.dynamics_params
    )
    omega_noise = 0.2* omega_noise_rate * (dr_high-dr_low)
    param_noise = jnp.clip(jax.random.normal(key, shape=transitions.dynamics_params.shape) * omega_noise[None, ...], -omega_clip, omega_clip)
    next_params = jnp.clip(transitions.dynamics_params + param_noise, dr_low, dr_high)
    next_action = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation, 
    )
    next_action = jnp.clip(next_action + noise, -1.0, 1.0)
    next_q = q_network.apply(
        normalizer_params,
        target_q_params,
        transitions.next_observation,
        next_action,
        next_params,
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
  def omega_loss(
      omega_params: Any,
      policy_params: Params,
      normalizer_params: Any,
      q_params: Params,
      transitions: Transition,
      key: PRNGKey,
  ) -> jnp.ndarray:
      
    action = policy_network.apply(
        normalizer_params, policy_params, transitions.observation
    )
    def f(i, unused_t):
        q_action = q_network.apply(
            normalizer_params, q_params, transitions.observation, action, omega_params[:,i]
        ).mean(axis=-1)
        return i+1, (q_action)
    idx = jnp.arange(omega_params.shape[1])
    _, qs = jax.lax.scan(
        f,
        0,
        (),
        length=omega_params.shape[1],
    )
    print("qs shape", qs)
    min_q = jnp.min(qs, axis=0)
    min_idx = jnp.argmin(qs,axis=0)
    print("min idx shape", min_idx)
    return jnp.mean(min_q), min_idx
  
  def actor_loss(
      policy_params: Params,
      normalizer_params: Any,
      q_params: Params,
      dynamics_params: Any,
      transitions: Transition,
      key: PRNGKey,
  ) -> jnp.ndarray:
    action = policy_network.apply(
        normalizer_params, policy_params, transitions.observation
    )
    print("dynamics_params in actor loss", dynamics_params)
    q_action = q_network.apply(
        normalizer_params, q_params, transitions.observation, action, dynamics_params,
    )
    min_q = jnp.min(q_action, axis=-1)
    return -jnp.mean(min_q)

  return critic_loss, actor_loss, omega_loss
