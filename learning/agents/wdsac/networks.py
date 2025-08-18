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

"""WDSAC networks."""

from typing import Literal, Sequence, Tuple, Callable, Any

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
from brax.training.types import Transition
from brax.training.networks import MLP
import flax
from flax import linen
import jax 
import jax.numpy as jnp
import dataclasses
@flax.struct.dataclass
class WDSACNetworks:
  # lmbda_network: networks.FeedForwardNetwork
  policy_network: networks.FeedForwardNetwork
  q_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]

def make_inference_fn(wdsac_networks: WDSACNetworks):
  """Creates params and inference function for the WDSAC agent."""

  def make_policy(
      params: types.PolicyParams, deterministic: bool = False
  ) -> types.Policy:

    def policy(
        observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      logits = wdsac_networks.policy_network.apply(*params, observations)
      if deterministic:
        return wdsac_networks.parametric_action_distribution.mode(logits), {}
      return (
          wdsac_networks.parametric_action_distribution.sample(
              logits, key_sample
          ),
          {},
      )

    return policy

  return make_policy
# def make_lmbda_networks(
#     obs_size: types.ObservationSize,
#     action_size: int,
#     preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
#     hidden_layer_sizes: Sequence[int] = (32, 32),
#     activation: ActivationFn = linen.relu,
#     batch_size : int, 
# ):
#   class LmbdaModule(linen.Module):
#     @linen.compact
#     def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
#       hidden = jnp.concatenate([obs,actions], axis=-1)
#       return MLP(
#         layer_sizes=list(hidden_layer_sizes) + [1],
#         activation=activation,
#         kernel_init=jax.nn.initializers.lecun_uniform(),
#         )(hidden)
#   lmbda_module = LmbdaModule()
#   def apply(processor_params, lmbda_params, obs, actions):
#     obs = preprocess_observations_fn(obs, processor_params)
#     return jnp.squeeze(lmbda_module.apply(lmbda_params, obs, actions), axis=-1)
  
#   dummy_obs = jnp.zeros((1,obs_size))
#   dummy_action = jnp.zeros((1, action_size))
#   return networks.FeedForwardNetwork(
#       init=lambda key: lmbda_module.init(key, dummy_obs, dummy_action), apply=apply
#   )
def make_wdsac_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    policy_obs_key: str = 'state',
    value_obs_key: str = 'privileged_state',
) -> WDSACNetworks:
  """Make WDSAC networks."""
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

  # lmbda_network = make_lmbda_networks(
  #   observation_size,
  #   action_size,
  #   preprocess_observations_fn=preprocess_observations_fn,
  #   hidden_layer_sizes=hidden_layer_sizes,
  #   activation=activation,
  #   layer_norm=policy_network_layer_norm,
  # )
  policy_network = networks.make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=policy_network_layer_norm,
      obs_key = policy_obs_key,
  )
  q_network = networks.make_q_network(
      observation_size,
      action_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=q_network_layer_norm,
      obs_key=value_obs_key,
  )
  return WDSACNetworks(
      # lmbda_network = lmbda_network,
      policy_network=policy_network,
      q_network=q_network,
      parametric_action_distribution=parametric_action_distribution,
  )
