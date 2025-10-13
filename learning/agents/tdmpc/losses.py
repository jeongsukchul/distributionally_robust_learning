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

"""TDMPC losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

from typing import Any

from brax.training import types
from agents.tdmpc import networks as tdmpc_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp

Transition = types.Transition

def soft_crossentropy(pred_logits: jax.Array, target: jax.Array,
                      low: float, high: float, num_bins: int) -> jax.Array:
  pred = jax.nn.log_softmax(pred_logits, axis=-1)
  target = tdmpc_networks.two_hot(target, low, high, num_bins)
  return -(pred * target).sum(axis=-1)


def make_losses(
    tdmpc_network: tdmpc_networks.TDMPCNetworks,
    reward_scaling: float,
    discounting: float,
    rho: float,
    consistency_coef: float,
    reward_coef: float,
    value_coef: float,
    entropy_coef: float,
):
  encoder          = tdmpc_network.encoder
  dynamics_network = tdmpc_network.dynamics_network
  reward_network   = tdmpc_network.reward_network
  policy_network   = tdmpc_network.policy_network
  q_network        = tdmpc_network.q_network
  symlog_min       = tdmpc_network.symlog_min
  symlog_max       = tdmpc_network.symlog_max
  num_bins         = tdmpc_network.num_bins

  def _two_hot_ce(logits, targets):
    # logits: (..., num_bins), targets: (...) (continuous)
    pred = jax.nn.log_softmax(logits, axis=-1)
    tgt  = tdmpc_networks.two_hot(targets, symlog_min, symlog_max, num_bins)
    return -(pred * tgt).sum(axis=-1)  # (...,)

  def model_loss(
      encoder_params: Params,
      dynamics_params: Params,
      reward_params: Params,
      q_params: Params,
      normalizer_params: Any,
      policy_params: Params,
      target_q_params: Params,
      transitions: types.Transition,
      key: PRNGKey,
  ):
    # shapes
    B, H, O = transitions.observation.shape
    A       = transitions.action.shape[-1]

    # Precompute weights/masks (static over batch)
    rho_vec   = rho ** jnp.arange(H, dtype=transitions.observation.dtype)           # (H,)
    disc_vec  = discounting ** jnp.arange(H, dtype=transitions.observation.dtype)   # (H,)

    # alive_t = Π_{k< t} discount_{k}
    # discount == 1 - done ∈ {0,1} in your pipeline
    # exclusive cumprod: alive[:,0]=1, alive[:,t>0]=∏_{k=0}^{t-1} discount[:,k]
    alive = jnp.concatenate(
      [jnp.ones((B, 1), transitions.discount.dtype),
       jnp.cumprod(transitions.discount, axis=1)[:, :-1]],
      axis=1
    )  # (B, H)

    # Encode z_0 and z_{t+1} (teacher)
    z0      = encoder.apply(normalizer_params, encoder_params, transitions.observation[:, 0, :])     # (B, Z)
    next_z  = encoder.apply(normalizer_params, encoder_params, transitions.next_observation)         # (B, H, Z)
    next_z  = jax.lax.stop_gradient(next_z)

    # ---------- Consistency rollout: z_{t+1} = f(z_t, a_t) ----------
    # scan over time: carry = z_t, out = z_{t+1}
    def dyn_step(z_t, a_t):
      z_tp1 = dynamics_network.apply(dynamics_params, z_t, a_t)
      return z_tp1, (z_t, z_tp1)

    _, (z_t_all, z_hat_tp1_all) = jax.lax.scan(
        dyn_step, z0, jnp.moveaxis(transitions.action, 1, 0)
    )
    z_t_all        = jnp.moveaxis(z_t_all, 0, 1)         # (B, H, Z)
    z_hat_tp1_all  = jnp.moveaxis(z_hat_tp1_all, 0, 1)   # (B, H, Z)
    # def dyn_step(z_t, a_t):
    #   z_tp1 = dynamics_network.apply(dynamics_params, z_t, a_t)  # (B, Z)
    #   return z_tp1, z_tp1

    # z_seq = []  # will collect z_t for t=0..H (for reward/Q we need z_t)
    # z_t   = z0
    # for t in range(H):
    #   z_seq.append(z_t)
    #   z_t, z_hat_tp1 = dyn_step(z_t, transitions.action[:, t, :])
    # z_seq.append(z_t)
    # z_seq = jnp.stack(z_seq, axis=1)           # (B, H+1, Z)
    # z_hat_tp1_all = z_seq[:, 1:, :]            # (B, H, Z)   model-predicted z_{t+1}

    # consistency loss over t=0..H-1
    per_t_cons = jnp.mean((z_hat_tp1_all - next_z)**2, axis=-1)  # (B, H)
    per_t_cons = per_t_cons * alive                              # mask invalid steps
    cons_loss  = (rho_vec * per_t_cons).mean()                   # average over B,H

    # ---------- Reward loss: r_t(z_t, a_t) ----------
    # Reward MLP applied at (z_t, a_t), t=0..H-1
    z_t_all         = z_t_all                                         # (B, H, Z)
    reward_logits   = reward_network.apply(reward_params, z_t_all, transitions.action)[1]  # (B, H, num_bins)
    # CE target is raw reward_t
    reward_ce       = _two_hot_ce(reward_logits, transitions.reward)               # (B, H)
    reward_ce       = reward_ce * alive
    rew_loss        = (rho_vec * reward_ce).mean()

    # ---------- Value loss (distributional) ----------
    # Bootstrap from next_z (already encoded from data): a' = π(next_z)
    key, act_key = jax.random.split(key)
    next_action  = policy_network.apply(policy_params, next_z, act_key)[0]       # (B, H, A)
    next_q, _    = q_network.apply(target_q_params, next_z, next_action)         # (C, B, H, num_bins_inv path)
    # we want scalar V_{t+1}; your q_apply returns both Q (decoded) and logits; use decoded Q:
    # next_q shape in your code path is (C, B, H), since two_hot_inv is applied inside q_apply.
    # If your q_apply returns (Q, logits) where Q is (C, B, H), we already have it.
    # Take min over critics:
    next_v = jnp.min(next_q, axis=0)                                             # (B, H)

    target_q = jax.lax.stop_gradient(
      reward_scaling * transitions.reward + transitions.discount * discounting * next_v
    )  # (B, H)

    # Evaluate current Q logits at (z_t, a_t)
    _, q_old_logits = q_network.apply(q_params, z_t_all, transitions.action)      # (C, B, H, num_bins)
    # Average CE across critics
    q_ce = _two_hot_ce(q_old_logits, target_q[None, ...])                         # (C, B, H)
    q_ce = q_ce.mean(axis=0)                                                      # (B, H)
    q_ce = q_ce * alive
    val_loss = (rho_vec * q_ce).mean()

    total = consistency_coef * cons_loss + reward_coef * rew_loss + value_coef * val_loss
    return total, (cons_loss, rew_loss, val_loss, z_t_all)  # keep z_t_all for actor

  def actor_loss(
      policy_params: Params,
      q_params: Params,
      latent: jnp.ndarray,  # expect (B, H, Z)
      key: PRNGKey,
  ):
    # No printing in jitted paths; keep it pure & small
    B, H, Z = latent.shape
    key, act_key = jax.random.split(key)
    action, _, _, log_probs = policy_network.apply(policy_params, latent, act_key)   # (B, H, A), (B,H)
    Qs, _ = q_network.apply(q_params, latent, action)                                # (C, B, H)
    Q     = Qs.mean(axis=0)                                                          # (B, H)
    rho_vec = rho ** jnp.arange(H, dtype=latent.dtype)                               # (H,)
    # Entropy reg is per-timestep; weight with rho and mean over batch
    loss = (rho_vec * (entropy_coef * log_probs - Q).mean(axis=0)).mean()
    return loss

  return model_loss, actor_loss