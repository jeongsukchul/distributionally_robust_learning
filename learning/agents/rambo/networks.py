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

"""RAMBO networks."""

from typing import Literal, Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from module.dynamics import make_dynamics_network
from brax.training import types
from brax.training.types import PRNGKey
from brax.training.types import Transition
import flax
from flax import linen
import jax 
import jax.numpy as jnp

@flax.struct.dataclass
class RAMBONetworks:
  dynamics_network: networks.FeedForwardNetwork
  policy_network: networks.FeedForwardNetwork
  q_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(rambo_networks: RAMBONetworks):
  """Creates params and inference function for the RAMBO agent."""

  def make_policy(
      params: types.PolicyParams, deterministic: bool = False
  ) -> types.Policy:

    def policy(
        observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      logits = rambo_networks.policy_network.apply(*params, observations)
      if deterministic:
        return rambo_networks.parametric_action_distribution.mode(logits), {}
      return (
          rambo_networks.parametric_action_distribution.sample(
              logits, key_sample
          ),
          {},
      )

    return policy

  return make_policy

def make_rollout_fn(rambo_network: RAMBONetworks, termination_fn, replay_buffer, fake_buffer):
  """Creates params and inference function for the RAMBO agent."""
  def _rollout_fn(
      rng, 
      policy,
      normalizer_params,
      dynamics_params, 
      buffer_state,
      fake_buffer_state,
      rollout_length,
      rollout_batch_size,
      elite_idxs,
      ):
      # Discard params of non-elite models
      def sample_transition(rng, obs):
        rng_action, rng_dynamics, rng_noise, rng_index = jax.random.split(rng, 4)
        actions, policy_extras = policy(obs, rng_action)
        (ensemble_mean, ensemble_logvar), normal_fn, denormal_fn = rambo_network.dynamics_network.apply(
            normalizer_params, dynamics_params, obs, actions
        )
        ensemble_std = jnp.exp(0.5 * ensemble_logvar)
        
        elite = jax.random.choice(rng_index, jnp.array(elite_idxs))

        sample_mean, sample_std = \
          ensemble_mean[elite], ensemble_std[elite]
        noise = jax.random.normal(key=rng_noise, shape=sample_mean.shape)
        samples = sample_mean + noise * sample_std
        delta_obs, reward = samples[:-1], samples[-1]
        next_obs = denormal_fn(normal_fn(obs, normalizer_params)+ delta_obs, normalizer_params)
        done = termination_fn(obs, actions, next_obs)
        return Transition(
            observation = obs,
            action = actions,
            reward = reward,
            discount = 1 - done,
            next_observation = next_obs,
            extras={'policy_extras' : policy_extras},
        )
      @jax.vmap
      def _sample_step(carry, _):
          obs, rng = carry
          rng_step, rng = jax.random.split(rng)
          transition = sample_transition(
              rng_step,
              obs,
          )

          return (transition.next_observation, rng), transition

      buffer_state, init_transition = replay_buffer.sample_batch(buffer_state, rollout_batch_size)
      rng, rng_rollout = jax.random.split(rng)

      init_obs = init_transition.observation
      rng_rollout = jax.random.split(rng_rollout, init_obs.shape[0])
      _, transitions = jax.lax.scan(
          _sample_step, (init_obs, rng_rollout), None, length=rollout_length
      )

      transitions = jax.tree_util.tree_map(
          lambda x: x.reshape(-1, *x.shape[2:]).squeeze(), transitions
      )
      rollout_info  = { 
          "rollout_info/reward_mean" : transitions.reward.mean(),
          "rollout_info/reward_std" : transitions.reward.std(),
          "rollout_info/reward_max" : transitions.reward.max(),
          "rollout_info/reward_min" : transitions.reward.min(),
                        }
      fake_buffer_state = fake_buffer.insert(fake_buffer_state, transitions)
      return fake_buffer_state, rollout_info

  return _rollout_fn

def make_rambo_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    postprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    dynamics_hidden_layer_sizes: Sequence[int] = (256, 256,256,256),
    n_ensemble : int = 7,
    activation: networks.ActivationFn = linen.relu,
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
) -> RAMBONetworks:
  """Make RAMBO networks."""
  parametric_action_distribution: distribution.ParametricDistribution
  if distribution_type == 'normal':
    parametric_action_distribution = distribution.NormalDistribution(
        event_size=action_size
    )
  elif distribution_type == 'tanh_normal':
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
  else:
    raise ValueError(
        f'Unsupported distribution type: {distribution_type}. Must be one'
        ' of "normal" or "tanh_normal".'
    )
  dynamics_network = make_dynamics_network(
    observation_size, action_size, 
    preprocess_observations_fn, 
    postprocess_observations_fn,
    n_ensemble, 
    dynamics_hidden_layer_sizes,
  )
  policy_network = networks.make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=policy_network_layer_norm,
  )
  q_network = networks.make_q_network(
      observation_size,
      action_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=q_network_layer_norm,
  )
  return RAMBONetworks(
      dynamics_network=dynamics_network,
      policy_network=policy_network,
      q_network=q_network,
      parametric_action_distribution=parametric_action_distribution,
  )
