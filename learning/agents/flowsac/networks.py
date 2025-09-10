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

"""FLOWSAC networks."""

from typing import Literal, Sequence, Tuple, Callable, Any

from brax.training import distribution
from module import networks
from brax.training import types
from brax.training.types import PRNGKey
from brax.training.types import Transition
from brax.training.networks import MLP
import flax
from flax import linen
import jax 
import jax.numpy as jnp
import dataclasses
from module.distribution import make_flow_network, FlowRQSConfig
from module.simple_flow import make_realnvp_flow_networks
@flax.struct.dataclass
class FLOWSACNetworks:
  # lmbda_network: networks.FeedForwardNetwork
  policy_network: networks.FeedForwardNetwork
  q_network: networks.FeedForwardNetwork
  flow_network: networks.FeedForwardNetwork  # Normalizing flow for adversarial dynamics
  parametric_action_distribution: distribution.ParametricDistribution

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]

def make_inference_fn(flowsac_networks: FLOWSACNetworks):
  """Creates params and inference function for the FLOWSAC agent."""

  def make_policy(
      params: types.PolicyParams, deterministic: bool = False
  ) -> types.Policy:

    def policy(
        observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      logits = flowsac_networks.policy_network.apply(*params, observations)
      if deterministic:
        return flowsac_networks.parametric_action_distribution.mode(logits), {}
      return (
          flowsac_networks.parametric_action_distribution.sample(
              logits, key_sample
          ),
          {},
      )

    return policy

  return make_policy

def make_flowsac_networks(
    observation_size: int,
    action_size: int,
    dynamics_param_size : int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (256,),
    critic_hidden_layer_sizes: Sequence[int] = (512, 512),
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    policy_obs_key: str = 'state',
    value_obs_key: str = 'privileged_state',
    simba: bool = False,
) -> FLOWSACNetworks:
  """Make FLOWSAC networks."""
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
  # flow_cfg = Fl`owRQSConfig(
  #     n_dim = dynamics_param_size,
  # )`
  flow_network = make_realnvp_flow_networks(
    in_channels=dynamics_param_size)
  if simba:
    policy_network = networks.make_simba_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      layer_norm=policy_network_layer_norm,
      obs_key = policy_obs_key,
    )
  else:
    policy_network = networks.make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=policy_network_layer_norm,
        obs_key = policy_obs_key,
    )
  # Create the flow network for adversarial dynamics using JAX normalizing flows
  # The actual parameter size will be determined when the environment is available
  # We'll use a reasonable default size that can be updated later
  if simba:
    q_network = networks.make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=critic_hidden_layer_sizes,
        activation=activation,
        layer_norm=q_network_layer_norm,
        obs_key=value_obs_key,
    )
  else:
    q_network = networks.make_simba_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=q_network_layer_norm,
        obs_key=value_obs_key,
    )
  return FLOWSACNetworks(
      policy_network=policy_network,
      q_network=q_network,
      flow_network=flow_network,
      parametric_action_distribution=parametric_action_distribution,
  )
