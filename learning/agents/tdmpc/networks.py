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

"""tdmpc networks."""

import functools
from typing import Mapping, Sequence, Tuple

from brax.training import distribution
from module import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import flax.linen as nn
import jax.numpy as jnp
import jax
@flax.struct.dataclass
class TDMPCNetworks:
  
  encoder: networks.FeedForwardNetwork
  dynamics_network: networks.FeedForwardNetwork
  reward_network: networks.FeedForwardNetwork
  policy_network: networks.FeedForwardNetwork
  q_network: networks.FeedForwardNetwork
  symlog_min: float
  symlog_max: float
  num_bins :int 
def make_inference_fn(tdmpc_networks: TDMPCNetworks):
  """Creates params and inference function for the tdmpc agent."""

  def make_policy(
      params, deterministic: bool = False,
  ) -> types.Policy:
    normalizer_params, encoder_params, policy_params = params

    def policy(
        observations: types.Observation,
        prev_plan: jnp.ndarray,
        key: PRNGKey,
    ):  
        latent = tdmpc_networks.encoder.apply(normalizer_params,encoder_params, observations)
        action, mean, log_std, log_probs = tdmpc_networks.policy_network.apply(policy_params, latent, key)
        if deterministic:
            return mean, {}
        else:
            return action, {'log_probs' :log_probs}
    return policy
  return make_policy
def make_mppi_fn(tdmpc_networks: TDMPCNetworks):
    def make_plan(
        params,
        action_size, 
        deterministic: bool = False,
        discounting : float = 0.99,
        max_plan_std : float = 2.,
        min_plan_std : float = 0.05,
        mppi_iterations : int = 6,
        temperature : float = 1.0,
        num_elites : int = 64,
        num_samples: int = 512,
        policy_prior_samples : int = 24,
        horizon : int = 3,
    )-> types.Policy:
        normalizer_params, encoder_params, dynamics_params, reward_params, policy_params, q_params = params
        def estimate_value(latent, actions, key):
            # latent: (B, N, Z), actions: (B, N, H, A)
            def step(carry, a_t):
                lat = carry
                r_t, _ = tdmpc_networks.reward_network.apply(reward_params, lat, a_t)  # (B, N)
                lat = tdmpc_networks.dynamics_network.apply(dynamics_params, lat, a_t)  # (B, N, Z)
                return lat, r_t

            latent_T, rewards = jax.lax.scan(step, latent, jnp.moveaxis(actions, -2, 0))  # H major
            # rewards: (H, B, N)
            disc = discounting ** jnp.arange(rewards.shape[0], dtype=rewards.dtype)  # (H,)
            G = (rewards * disc[:, None, None]).sum(0)  # (B, N)

            # bootstrap
            action_key, Q_key = jax.random.split(key)
            next_action, *_ = tdmpc_networks.policy_network.apply(policy_params, latent_T, action_key)  # (B, N, A)
            Qs, _ = tdmpc_networks.q_network.apply(q_params, latent_T, next_action)  # (C, B, N)
            Q = Qs.mean(axis=0)
            return G + (discounting ** actions.shape[-2]) * Q
        def plan(
            observations: types.Observation,
            prev_plan: jnp.ndarray,
            key: PRNGKey,
        ):  
            batch_size = observations.shape[0]
            # action_key, key = jax.random.split(key)
            # action_keys = jax.random.split(action_key, horizon)
            # actions = jnp.zeros((batch_size, num_samples, horizon, action_size)) #B, N, H, A
            latent = tdmpc_networks.encoder.apply(normalizer_params,encoder_params, observations) #B, Z

            #prior action samples
            # latents = latent[...,None, :].repeat(policy_prior_samples, axis=-2) # B, pr, Z
            latents = jnp.broadcast_to(latent[:, None, :], (latent.shape[0], policy_prior_samples, latent.shape[-1]))
            # prior_actions = jnp.zeros((batch_size, policy_prior_samples, horizon, action_size)) #B, pr, H, A
            def prior_body(carry, _):
                latents, key = carry
                action_key, key = jax.random.split(key)
                action, _, _, _ = tdmpc_networks.policy_network.apply(policy_params, latents, action_key)
                # prior_actions = prior_actions.at[...,t,:].set(action)
                latents = tdmpc_networks.dynamics_network.apply(dynamics_params, latents, action)
                return (latents, key), action

            (latents, key), prior_actions = jax.lax.scan(prior_body, (latents, key), (), length=horizon)
            prior_actions = jnp.moveaxis(prior_actions, 0, -2)

            #mppi planning
            # latents = latent[...,None, :].repeat(num_samples, axis=-2) # B, N, Z
            latents = jnp.broadcast_to(latent[:, None, :], (latent.shape[0], num_samples, latent.shape[-1]))
            key, mppi_noise_key, *value_keys = jax.random.split(
                key, 2+ mppi_iterations
            )
            noise = jax.random.normal(
               mppi_noise_key,
               shape=(batch_size, num_samples-policy_prior_samples, mppi_iterations, horizon, action_size),
            )
            mean = jnp.zeros((batch_size, horizon, action_size)) # B, H, A
            mean = mean.at[..., :-1, :].set(prev_plan[0][..., 1:, :]) # prev plan used for mean
            std = jnp.full((batch_size, horizon, action_size), max_plan_std) #B, H, A

            def mppi_body(carry, _):
            # for i in range(mppi_iterations):
                mean, std, key, i = carry
                key, vkey = jax.random.split(key)
                rand_actions = (mean[:, None] + std[:, None] * noise[..., i, :, :]).clip(-1, 1)  # (B, N-P, H, A)
                actions = jnp.concatenate([prior_actions, rand_actions], axis=1)  # (B, N, H, A)
                values = estimate_value(latents, actions, vkey) # B, N
                elite_values, elite_inds = jax.lax.top_k(values, num_elites)
                # elite values shape -> (B, E), # elite _inds (B, E)
                elite_actions = jnp.take_along_axis(actions, elite_inds[..., None, None], axis=1) # B, E, H, A
                score = jax.nn.softmax(temperature * elite_values) # B, E
                w = score[..., None, None]
                new_mean = jnp.sum(w * elite_actions, axis=1) # B, H, A
                new_std = jnp.sqrt(jnp.sum(w*(elite_actions-mean[:, None])**2, axis=1)).clip(min_plan_std,max_plan_std)
                return (new_mean, new_std, key, i+1), (elite_values, elite_actions)
            (mean, std, key, _), (elite_values, elite_actions) = jax.lax.scan(mppi_body, (mean,std,key, 0), (), length=mppi_iterations)

            elite_values= elite_values[-1]
            elite_actions = elite_actions[-1]

            # set last action
            key, gumbel_key = jax.random.split(key)
            gumbels = jax.random.gumbel(gumbel_key, shape=elite_values.shape) #B, E
            gumbel_scores = temperature * elite_values + gumbels
            action_ind = jnp.argmax(gumbel_scores, axis=-1) #B
            action = jnp.take_along_axis( 
               elite_actions, action_ind[..., None, None, None], axis=1
            ).squeeze(1) # B, H, A
            if deterministic:
                final_action = action[...,0,:]
            else:
                key, final_noise_key = jax.random.split(key)
                final_action = action[...,0,:] + std[...,0,:] * jax.random.normal(final_noise_key, shape=(batch_size, action_size))
            return final_action, {'plan' : (mean, std)}
        return plan
    return make_plan
def symlog(x: jax.Array) -> jax.Array:
  return jnp.sign(x) * jnp.log(1 + jnp.abs(x))


def symexp(x):
  return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

from einops import rearrange

def simnorm(x: jax.Array, simplex_dim: int = 8) -> jax.Array:
  x = rearrange(x, '...(L V) -> ... L V', V=simplex_dim)
  x = jax.nn.softmax(x, axis=-1)
  return rearrange(x, '... L V -> ... (L V)')

def two_hot(x: jax.Array, low: float, high: float, num_bins: int) -> jax.Array:
  """
  Generate two-hot encoded tensor from input tensor.

  Parameters
  ----------
  x : jax.Array
      Input tensor of continuous values. Shape: (*batch_dim, num_values)
      Should **not** have a leading singleton dimension at the end.
  low : float
      Minimum value under consideration in log-space
  high : float
      Maximum value under consideration in log-space
  num_bins : int
      Number of encoding bins

  Returns
  -------
  jax.Array
      _description_
  """
  bin_size = (high - low) / (num_bins - 1)

  x = jnp.clip(symlog(x), low, high)
  bin_index = jnp.floor((x - low) / bin_size).astype(int)
  bin_offset = ((x - low) / bin_size - bin_index.astype(float))

  # Two-hot encode
  two_hot = jax.nn.one_hot(bin_index, num_bins) * (1 - bin_offset[..., None]) +\
      jax.nn.one_hot(bin_index + 1, num_bins) * bin_offset[..., None]

  return two_hot
def two_hot_inv(x: jax.Array,
                low: float, high: float, num_bins: int,
                apply_softmax: bool = True) -> jax.Array:

  bins = jnp.linspace(low, high, num_bins)

  if apply_softmax:
    x = jax.nn.softmax(x, axis=-1)

  x = jnp.sum(x * bins, axis=-1)
  return symexp(x)

def make_latent_network(
    obs_size: types.ObservationSize,
    latent_size : int,
    action_size : int,
    num_bins: int,
    encoder_hidden_layer_sizes: Sequence[int] = (256, 256),
    hidden_layer_sizes: Sequence[int] = (256, 256),
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    activation: networks.ActivationFn = jax.nn.mish,
    kernel_init: networks.Initializer = nn.initializers.truncated_normal(stddev=0.02),
    layer_norm: bool = False,
    simnorm_dim : int = 8,
    obs_key: str = 'state',
    symlog_min : float = -10,
    symlog_max : float = 10,
    min_log_std: float = -10,
    max_log_std: float = 2,
    n_critics : int = 5,
):
    class EncoderNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = networks.MLP(
                layer_sizes=list(encoder_hidden_layer_sizes) + [latent_size],
                activation=activation,
                kernel_init=kernel_init,
                layer_norm=layer_norm,
            ) (x)
            return simnorm(x, simplex_dim=simnorm_dim)
    encoder_module = EncoderNet()
    def encoder_apply(processor_params, encoder_params, obs):
        if isinstance(obs, Mapping):
            obs = preprocess_observations_fn(
                obs[obs_key], networks.normalizer_select(processor_params, obs_key)
            )
        else:
            obs = preprocess_observations_fn(obs, processor_params)
        return encoder_module.apply(encoder_params, obs)

    class DynamicsNet(nn.Module):
        @nn.compact
        def __call__(self, latent, action):
            x = jnp.concatenate([latent,action],axis=-1)
            x = networks.MLP(
                layer_sizes=list([latent_size] * 3) ,
                activation=activation,
                kernel_init=kernel_init,
                layer_norm=layer_norm,
            ) (x)
            return simnorm(x, simplex_dim=simnorm_dim)
    dynamics_module = DynamicsNet()

    class RewardNet(nn.Module):
        @nn.compact
        def __call__(self, latent, action):
            x = networks.MLP(
                layer_sizes=list([latent_size] * 2),
                activation=activation,
                kernel_init=kernel_init,
                layer_norm=layer_norm,
            ) (jnp.concatenate([latent,action],axis=-1))
            x = nn.Dense(num_bins, kernel_init=nn.initializers.zeros)(x)
            return x
    reward_module =  RewardNet()
    def reward_apply(reward_params, latent, action):
        logits = reward_module.apply(reward_params, latent, action)
        reward = two_hot_inv(
            logits, symlog_min, symlog_max, num_bins
        )
        return reward, logits

    class ActorNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = networks.MLP(
                layer_sizes=list(hidden_layer_sizes) + [2*action_size],
                activation=activation,
                kernel_init=kernel_init,
                layer_norm=layer_norm,
            ) (x)
            
            return jnp.tanh(x)
    policy_module = ActorNet()
    def policy_apply(policy_params, latent, key):
        mean, log_std = jnp.split(policy_module.apply(policy_params, latent), 2, axis=-1)
        log_std = min_log_std + 0.5 * \
            (max_log_std - min_log_std) * (jnp.tanh(log_std) + 1)
        eps = jax.random.normal(key, mean.shape)
        action = mean + eps * jnp.exp(log_std)
        residual = (-0.5 * eps**2 - log_std).sum(-1)
        log_probs = action.shape[-1] * (residual - 0.5 * jnp.log(2 * jnp.pi))

        # Squash tanh
        mean = jnp.tanh(mean)
        action = jnp.tanh(action)
        log_probs -= jnp.log(nn.relu(1 - action**2) + 1e-6).sum(-1)

        return action, mean, log_std, log_probs
    class QNet(nn.Module):
        @nn.compact
        def __call__(self, latent: jnp.ndarray, actions: jnp.ndarray):
            hidden = jnp.concatenate([latent, actions], axis=-1)
            res = []
            for _ in range(n_critics):
                q = networks.MLP(
                    layer_sizes=list(hidden_layer_sizes),
                    activation=activation,
                    kernel_init=jax.nn.initializers.zeros,
                    layer_norm=layer_norm,
                )(hidden)
                q = nn.Dense(num_bins, kernel_init=nn.initializers.zeros)(q)
                res.append(q)
            return jnp.stack(res, axis=0)
    q_module = QNet()
    def q_apply(q_params, latent, action):
        logits = q_module.apply(q_params, latent, action)
        Q = two_hot_inv(logits, symlog_min, symlog_max, num_bins)
        return Q, logits

    obs_size = networks._get_obs_state_size(obs_size, obs_key)
    print("obs size, latentsize actionsize", obs_size, latent_size, action_size)
    dummy_obs = jnp.zeros((1, obs_size))
    dummy_latent = jnp.zeros((1, latent_size))
    dummy_action = jnp.zeros((1, action_size))
    encoder = networks.FeedForwardNetwork(
       init=lambda key: encoder_module.init(key, dummy_obs), \
        apply=encoder_apply,
    )
    dynamics_network = networks.FeedForwardNetwork(
       init=lambda key: dynamics_module.init(key, dummy_latent, dummy_action), \
        apply=lambda params, latent, action: dynamics_module.apply(params, latent, action)
    )
    reward_network = networks.FeedForwardNetwork(
       init=lambda key: reward_module.init(key, dummy_latent, dummy_action), \
        apply=reward_apply,
    )
    policy_network = networks.FeedForwardNetwork(
       init=lambda key: policy_module.init(key, dummy_latent), \
        apply=policy_apply,
    )
    q_network = networks.FeedForwardNetwork(
       init=lambda key: q_module.init(key, dummy_latent, dummy_action), \
        apply=q_apply,
    )
    return encoder, dynamics_network, reward_network, policy_network, q_network

def make_tdmpc_networks(
    observation_size: int,
    action_size: int,
    latent_size : int,
    num_bins : int = 101,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    encoder_hidden_layer_sizes: Sequence[int] = (256, 256),
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = jax.nn.mish,
    n_critics:int = 5,
    symlog_min = -10,
    symlog_max = 10,
    simnorm_dim=8,
) -> TDMPCNetworks:
  """Make tdmpc networks."""
  print("encoder hidden", encoder_hidden_layer_sizes)
  encoder, dynamics_network, reward_network, policy_network, q_network = make_latent_network(
        observation_size,
        latent_size,
        action_size,
        num_bins,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        hidden_layer_sizes=hidden_layer_sizes,
        preprocess_observations_fn=preprocess_observations_fn,
        activation = activation,
        n_critics = n_critics,
        symlog_min=symlog_min,
        symlog_max=symlog_max,
        simnorm_dim=simnorm_dim,
    )
  return TDMPCNetworks(
      encoder=encoder,
      dynamics_network=dynamics_network,
      reward_network=reward_network,
      policy_network=policy_network,
      q_network=q_network,
      symlog_min  = -10,
      symlog_max  = 10,
      num_bins=num_bins,
  )